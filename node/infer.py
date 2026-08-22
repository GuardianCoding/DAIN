"""Local inference for the node agent — what POST /infer runs.

The node does not implement a model runtime. It talks to a llama-server over
llama.cpp's OpenAI-compatible HTTP API and normalises the reply. There are
exactly two ways a node gets one, and they map onto the two topologies:

  FAN-OUT   set DAIN_INFER_MODEL to a GGUF on this machine. The node starts
            and supervises its own llama-server on 127.0.0.1, the same way it
            already supervises rpc-server. Every node holds a full copy of a
            small model; POST /api/jobs {"kind":"infer","fanout":5} then runs
            the same prompt on five machines at once.

  PIPELINE  set DAIN_LLAMA_ENDPOINT to an existing llama-server, typically the
            head on gpu-01 with the workers attached over --rpc. This node
            supervises nothing and forwards. One model spans the cluster.

Set neither and /infer answers 503 saying so, rather than timing out or
pretending. Set both and the explicit endpoint wins, because "use this exact
server" is the more specific instruction.

The bind is always loopback when we supervise: llama-server has no
authentication, so the reasoning behind rpc_worker_command refusing 0.0.0.0
applies here too.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

LOG = logging.getLogger("dain.node.infer")

ENDPOINT_ENV = "DAIN_LLAMA_ENDPOINT"
MODEL_ENV = "DAIN_INFER_MODEL"
PORT_ENV = "DAIN_LLAMA_PORT"
BIN_DIR_ENV = "DAIN_LLAMA_BIN"
CONTEXT_ENV = "DAIN_INFER_CONTEXT"
SLOTS_ENV = "DAIN_INFER_SLOTS"

DEFAULT_LLAMA_BIN_DIR = "/opt/dain/llama.cpp/build/bin"
SERVER_BINARY_NAME = "llama-server"
DEFAULT_LLAMA_PORT = 8080
DEFAULT_CONTEXT = 8192
DEFAULT_SLOTS = 1

LOOPBACK = "127.0.0.1"
DEFAULT_MAX_TOKENS = 256
DEFAULT_TEMPERATURE = 0.7
MAX_PROMPT_CHARS = 8_000  # §6.3 cap, mirrored by the dashboard

# Model load is minutes for a large GGUF, so readiness is polled per request
# rather than waited on at startup — the node must finish joining regardless.
READY_TIMEOUT_S = 180.0
READY_POLL_S = 0.5
HEALTH_TIMEOUT_S = 2.0
REQUEST_TIMEOUT_S = 300.0
STOP_TIMEOUT_S = 10.0


class InferenceUnavailableError(RuntimeError):
    """No llama-server is configured, reachable, or ready on this node."""


def _clean(value: str | None) -> str | None:
    text = (value or "").strip()
    return text or None


def _int_env(name: str, default: int) -> int:
    raw = _clean(os.getenv(name))
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        LOG.warning("%s=%r is not an integer; using %s", name, raw, default)
        return default


def resolve_server_binary(bin_dir: str) -> Path | None:
    """Locate llama-server, or None if this node has no build."""
    candidate = Path(bin_dir) / SERVER_BINARY_NAME
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return candidate

    on_path = shutil.which(SERVER_BINARY_NAME)
    return Path(on_path) if on_path else None


@dataclass
class LocalInference:
    """Resolves where inference happens and speaks to it.

    `endpoint` set   -> forward to it, supervise nothing (pipeline).
    `model_file` set -> supervise a local llama-server (fan-out).
    Neither          -> /infer is unavailable, and says why.
    """

    endpoint: str | None = None
    model_file: str | None = None
    port: int = DEFAULT_LLAMA_PORT
    context: int = DEFAULT_CONTEXT
    slots: int = DEFAULT_SLOTS
    bin_dir: str = DEFAULT_LLAMA_BIN_DIR
    ready_timeout_s: float = READY_TIMEOUT_S
    client: httpx.AsyncClient | None = None
    proc: subprocess.Popen[bytes] | None = field(default=None, init=False)
    _ready: bool = field(default=False, init=False)

    @classmethod
    def from_environment(cls) -> LocalInference:
        return cls(
            endpoint=_clean(os.getenv(ENDPOINT_ENV)),
            model_file=_clean(os.getenv(MODEL_ENV)),
            port=_int_env(PORT_ENV, DEFAULT_LLAMA_PORT),
            context=_int_env(CONTEXT_ENV, DEFAULT_CONTEXT),
            slots=_int_env(SLOTS_ENV, DEFAULT_SLOTS),
            bin_dir=os.getenv(BIN_DIR_ENV, DEFAULT_LLAMA_BIN_DIR),
        )

    @property
    def supervises(self) -> bool:
        """True when this node runs its own server. An explicit endpoint wins."""
        return self.endpoint is None and self.model_file is not None

    @property
    def configured(self) -> bool:
        return self.endpoint is not None or self.model_file is not None

    @property
    def base_url(self) -> str:
        if self.endpoint is not None:
            return self.endpoint.rstrip("/")
        return f"http://{LOOPBACK}:{self.port}"

    @property
    def running(self) -> bool:
        """Whether a server we supervise is still alive."""
        if not self.supervises:
            return self.configured
        return self.proc is not None and self.proc.poll() is None

    def server_command(self) -> list[str]:
        """argv for the supervised server. Loopback-bound: llama-server has no
        authentication, so anything reaching a wildcard bind can spend this
        machine's compute."""
        if self.model_file is None:
            raise InferenceUnavailableError(f"${MODEL_ENV} is not set")
        if not Path(self.model_file).is_file():
            raise InferenceUnavailableError(f"model file not found: {self.model_file}")

        binary = resolve_server_binary(self.bin_dir)
        if binary is None:
            raise InferenceUnavailableError(
                f"no {SERVER_BINARY_NAME} under {self.bin_dir} or on PATH"
            )

        return [
            str(binary),
            "-m", self.model_file,
            "-c", str(self.context),
            "-np", str(self.slots),
            "-fa", "on",
            "-ngl", "999",
            "--metrics",
            "--host", LOOPBACK,
            "--port", str(self.port),
        ]

    def start(self) -> None:
        """Launch the supervised server, if this node has one.

        Non-blocking: the model may take minutes to load and joining the pool
        must not wait on it. Readiness is checked per request instead.
        """
        if not self.supervises or self.running:
            return

        try:
            command = self.server_command()
        except InferenceUnavailableError as exc:
            LOG.warning("not starting %s: %s", SERVER_BINARY_NAME, exc)
            return

        LOG.info("starting %s on %s:%s", SERVER_BINARY_NAME, LOOPBACK, self.port)
        self._ready = False
        self.proc = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def stop(self) -> None:
        proc, self.proc = self.proc, None
        self._ready = False
        if proc is None or proc.poll() is not None:
            return

        LOG.info("stopping %s (pid %s)", SERVER_BINARY_NAME, proc.pid)
        proc.terminate()
        try:
            proc.wait(timeout=STOP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            proc.kill()

    def _ensure_client(self) -> httpx.AsyncClient:
        if self.client is None or self.client.is_closed:
            self.client = httpx.AsyncClient()
        return self.client

    async def aclose(self) -> None:
        client, self.client = self.client, None
        if client is not None and not client.is_closed:
            with contextlib.suppress(Exception):
                await client.aclose()

    async def await_ready(self) -> None:
        """Block until llama-server answers /health, or give up with a 503.

        A supervised server is started but not waited for, so the first /infer
        after boot pays the model-load cost. Without this it would instead fail
        on connection-refused, which reads as misconfiguration rather than
        "still loading".
        """
        if not self.configured:
            raise InferenceUnavailableError(
                f"no inference backend on this node: set ${MODEL_ENV} to a GGUF "
                f"to run one here, or ${ENDPOINT_ENV} to forward to an existing "
                f"{SERVER_BINARY_NAME}"
            )
        if self._ready:
            return

        client = self._ensure_client()
        deadline = time.monotonic() + self.ready_timeout_s
        last_error = "no response"

        while time.monotonic() < deadline:
            if self.supervises and self.proc is not None and self.proc.poll() is not None:
                raise InferenceUnavailableError(
                    f"{SERVER_BINARY_NAME} exited with code {self.proc.returncode}"
                )
            try:
                response = await client.get(
                    f"{self.base_url}/health", timeout=HEALTH_TIMEOUT_S
                )
                if response.status_code == 200:
                    self._ready = True
                    return
                last_error = f"HTTP {response.status_code}"
            except httpx.HTTPError as exc:
                last_error = f"{type(exc).__name__}: {exc}"

            await asyncio.sleep(READY_POLL_S)

        raise InferenceUnavailableError(
            f"{self.base_url} not ready after {self.ready_timeout_s:.0f}s "
            f"({last_error})"
        )

    async def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Run one prompt and normalise llama-server's reply.

        Returns the flattened shape the dashboard renders, not the raw OpenAI
        envelope — the jobs table should not have to know about
        choices[0].message.content.
        """
        prompt = payload.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("payload.prompt must not be empty")
        if len(prompt) > MAX_PROMPT_CHARS:
            raise ValueError(
                f"payload.prompt is {len(prompt)} chars; the cap is {MAX_PROMPT_CHARS}"
            )

        max_tokens = payload.get("max_tokens", DEFAULT_MAX_TOKENS)
        temperature = payload.get("temperature", DEFAULT_TEMPERATURE)
        # ValueError, not TypeError (ruff TRY004): the route maps ValueError ->
        # 422 and InferenceUnavailableError -> 503, and that split decides
        # whether the queue retries the shard on another node.
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
            raise ValueError("payload.max_tokens must be an integer")  # noqa: TRY004
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            raise ValueError("payload.temperature must be a number")  # noqa: TRY004

        await self.await_ready()
        client = self._ensure_client()

        body: dict[str, Any] = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }

        started = time.monotonic()
        try:
            response = await client.post(
                f"{self.base_url}/v1/chat/completions",
                json=body,
                timeout=REQUEST_TIMEOUT_S,
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as exc:
            raise InferenceUnavailableError(
                f"{self.base_url} request failed: {type(exc).__name__}: {exc}"
            ) from exc
        except ValueError as exc:
            raise InferenceUnavailableError(
                f"{self.base_url} returned a non-JSON body"
            ) from exc

        return _normalise(
            data,
            elapsed_s=time.monotonic() - started,
            endpoint=self.base_url,
        )


def _normalise(
    data: dict[str, Any], *, elapsed_s: float, endpoint: str
) -> dict[str, Any]:
    """Flatten llama-server's OpenAI-shaped reply, defensively.

    Every field below has been absent in some llama.cpp build, so nothing here
    indexes without checking. A missing `timings` block must not turn a
    successful generation into a failed job.
    """
    choices = data.get("choices")
    text = ""
    finish_reason = None
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        first = choices[0]
        message = first.get("message")
        if isinstance(message, dict):
            text = message.get("content") or ""
        finish_reason = first.get("finish_reason")

    raw_usage = data.get("usage")
    usage = raw_usage if isinstance(raw_usage, dict) else {}
    raw_timings = data.get("timings")
    timings = raw_timings if isinstance(raw_timings, dict) else {}

    completion_tokens = usage.get("completion_tokens")
    tok_s = timings.get("predicted_per_second")
    if tok_s is None and isinstance(completion_tokens, int) and elapsed_s > 0:
        tok_s = completion_tokens / elapsed_s

    return {
        "text": text,
        "model": data.get("model"),
        "endpoint": endpoint,
        "finish_reason": finish_reason,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": completion_tokens,
        "tok_s": round(tok_s, 2) if isinstance(tok_s, (int, float)) else None,
        "duration_s": round(elapsed_s, 3),
    }
