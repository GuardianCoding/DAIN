"""llama-bench on one node — what POST /bench runs.

Unlike /infer this needs no server: llama-bench is a standalone binary that
loads a model, times prefill and decode, and prints JSON.
`scripts/distribute_llama.sh` already pushes it to every worker.

Two callers want the same two numbers and must not drift apart:

  POST /bench   INF-6 measurement, on demand, results to benchmarks.csv
  SCH-1         the calibration probe at node start, filling tg_tok_s and
                pp_tok_s on the NodeProfile before it joins

Both go through `measure()`, so whoever wires SCH-1 does not re-implement the
parse. Every node currently reports 0.0 for both, which is why the real
scheduler still cannot be swapped in behind GET /api/plan:

    RuntimeError: no node has a measured tg_tok_s

WHAT THE NUMBERS MEAN. `llama-bench -p 512 -n 128` emits one row per phase:
prefill rows have n_prompt > 0 and n_gen == 0, decode rows the reverse.
`avg_ts` is tokens/second averaged over `-r` repetitions. Prefill is the pp
number, decode is tg. Mixing them up puts a ~10x error into the cost model,
because prefill is roughly an order of magnitude faster than decode.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOG = logging.getLogger("dain.node.bench")

BENCH_MODEL_ENV = "DAIN_BENCH_MODEL"
INFER_MODEL_ENV = "DAIN_INFER_MODEL"  # sensible fallback: bench what you serve
BIN_DIR_ENV = "DAIN_LLAMA_BIN"

DEFAULT_LLAMA_BIN_DIR = "/opt/dain/llama.cpp/build/bin"
BENCH_BINARY_NAME = "llama-bench"

# Matches infer.launch.llama_bench_command, deliberately: the calibration probe
# and the INF-6 measurement have to be the same benchmark or their numbers are
# not comparable.
DEFAULT_PROMPT_TOKENS = 512
DEFAULT_GEN_TOKENS = 128
DEFAULT_REPETITIONS = 3

# A cold load of a large GGUF on a slow node plus three repetitions is minutes,
# not seconds. Bounded anyway so a wedged run cannot pin a node forever.
BENCH_TIMEOUT_S = 900.0
MAX_REPETITIONS = 10
MAX_TOKENS = 4096


class BenchUnavailableError(RuntimeError):
    """No llama-bench, no model, or the run failed."""


@dataclass(frozen=True)
class BenchResult:
    """One node's measured throughput. Immutable — a measurement is a fact."""

    pp_tok_s: float | None
    tg_tok_s: float | None
    pp_stddev: float | None = None
    tg_stddev: float | None = None
    model: str | None = None
    build_commit: str | None = None
    n_gpu_layers: int | None = None
    n_threads: int | None = None
    repetitions: int = DEFAULT_REPETITIONS

    def as_dict(self) -> dict[str, Any]:
        return {
            "pp_tok_s": self.pp_tok_s,
            "tg_tok_s": self.tg_tok_s,
            "pp_stddev": self.pp_stddev,
            "tg_stddev": self.tg_stddev,
            "model": self.model,
            "build_commit": self.build_commit,
            "n_gpu_layers": self.n_gpu_layers,
            "n_threads": self.n_threads,
            "repetitions": self.repetitions,
        }


def _clean(value: str | None) -> str | None:
    text = (value or "").strip()
    return text or None


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def resolve_bench_binary(bin_dir: str) -> Path | None:
    candidate = Path(bin_dir) / BENCH_BINARY_NAME
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return candidate

    on_path = shutil.which(BENCH_BINARY_NAME)
    return Path(on_path) if on_path else None


def parse_bench_json(raw: str) -> BenchResult:
    """Turn llama-bench's `-o json` output into one BenchResult.

    Split by phase rather than by row order: llama-bench has changed row order
    between releases, and pinning to "first row is prefill" is exactly the kind
    of assumption that silently swaps a 300 tok/s prefill for a 30 tok/s decode.
    """
    try:
        rows = json.loads(raw)
    except ValueError as exc:
        raise BenchUnavailableError(
            f"{BENCH_BINARY_NAME} did not emit JSON (was -o json passed?)"
        ) from exc

    if not isinstance(rows, list) or not rows:
        raise BenchUnavailableError(f"{BENCH_BINARY_NAME} returned no measurements")

    pp_row: dict[str, Any] | None = None
    tg_row: dict[str, Any] | None = None

    for row in rows:
        if not isinstance(row, dict):
            continue
        n_prompt = _as_int(row.get("n_prompt")) or 0
        n_gen = _as_int(row.get("n_gen")) or 0
        if n_prompt > 0 and n_gen == 0:
            pp_row = row
        elif n_gen > 0 and n_prompt == 0:
            tg_row = row

    if pp_row is None and tg_row is None:
        raise BenchUnavailableError(
            f"{BENCH_BINARY_NAME} output had no prefill or decode row"
        )

    reference = tg_row or pp_row or {}
    return BenchResult(
        pp_tok_s=_as_float((pp_row or {}).get("avg_ts")),
        tg_tok_s=_as_float((tg_row or {}).get("avg_ts")),
        pp_stddev=_as_float((pp_row or {}).get("stddev_ts")),
        tg_stddev=_as_float((tg_row or {}).get("stddev_ts")),
        model=reference.get("model_type") or reference.get("model_filename"),
        build_commit=reference.get("build_commit"),
        n_gpu_layers=_as_int(reference.get("n_gpu_layers")),
        n_threads=_as_int(reference.get("n_threads")),
        repetitions=_as_int(reference.get("reps")) or DEFAULT_REPETITIONS,
    )


@dataclass
class LocalBench:
    """Runs llama-bench on this machine and parses the result."""

    model_file: str | None = None
    bin_dir: str = DEFAULT_LLAMA_BIN_DIR
    repetitions: int = DEFAULT_REPETITIONS
    prompt_tokens: int = DEFAULT_PROMPT_TOKENS
    gen_tokens: int = DEFAULT_GEN_TOKENS
    timeout_s: float = BENCH_TIMEOUT_S

    @classmethod
    def from_environment(cls) -> LocalBench:
        return cls(
            # Falls back to the served model so a fan-out node needs one
            # variable, not two, and benches what it actually runs.
            model_file=_clean(os.getenv(BENCH_MODEL_ENV))
            or _clean(os.getenv(INFER_MODEL_ENV)),
            bin_dir=os.getenv(BIN_DIR_ENV, DEFAULT_LLAMA_BIN_DIR),
        )

    @property
    def configured(self) -> bool:
        return self.model_file is not None

    @property
    def available(self) -> bool:
        return self.configured and resolve_bench_binary(self.bin_dir) is not None

    def command(
        self,
        model_file: str,
        *,
        repetitions: int,
        prompt_tokens: int,
        gen_tokens: int,
    ) -> list[str]:
        binary = resolve_bench_binary(self.bin_dir)
        if binary is None:
            raise BenchUnavailableError(
                f"no {BENCH_BINARY_NAME} under {self.bin_dir} or on PATH"
            )
        if not Path(model_file).is_file():
            raise BenchUnavailableError(f"model file not found: {model_file}")

        return [
            str(binary),
            "-m", model_file,
            "-p", str(prompt_tokens),
            "-n", str(gen_tokens),
            "-r", str(repetitions),
            "-o", "json",
        ]

    def _resolve(self, payload: dict[str, Any]) -> tuple[str, int, int, int]:
        """Validate payload overrides.

        ValueError for a caller mistake (-> 422); BenchUnavailableError for a
        node that simply cannot do this (-> 503). The queue retries the second
        on another node and not the first, so the distinction matters.
        """
        model_file = payload.get("model", self.model_file)
        if not isinstance(model_file, str) or not model_file.strip():
            raise BenchUnavailableError(
                f"no model to benchmark: set ${BENCH_MODEL_ENV} (or "
                f"${INFER_MODEL_ENV}) on this node, or pass payload.model"
            )

        repetitions = payload.get("repetitions", self.repetitions)
        prompt_tokens = payload.get("prompt_tokens", self.prompt_tokens)
        gen_tokens = payload.get("gen_tokens", self.gen_tokens)

        for name, value, cap in (
            ("repetitions", repetitions, MAX_REPETITIONS),
            ("prompt_tokens", prompt_tokens, MAX_TOKENS),
            ("gen_tokens", gen_tokens, MAX_TOKENS),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"payload.{name} must be an integer")
            if not 1 <= value <= cap:
                raise ValueError(f"payload.{name} must be between 1 and {cap}")

        return model_file.strip(), repetitions, prompt_tokens, gen_tokens

    async def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        model_file, repetitions, prompt_tokens, gen_tokens = self._resolve(payload)
        command = self.command(
            model_file,
            repetitions=repetitions,
            prompt_tokens=prompt_tokens,
            gen_tokens=gen_tokens,
        )

        LOG.info("running %s", " ".join(command))
        # to_thread, matching how /exec runs the sandbox: llama-bench is a
        # blocking CPU-bound subprocess and must not stall the event loop
        # serving heartbeats.
        stdout = await asyncio.to_thread(_run_subprocess, command, self.timeout_s)
        measured = parse_bench_json(stdout)

        return {
            **measured.as_dict(),
            "model_file": model_file,
            "prompt_tokens": prompt_tokens,
            "gen_tokens": gen_tokens,
            "repetitions": repetitions,
        }


def _run_subprocess(command: list[str], timeout_s: float) -> str:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise BenchUnavailableError(
            f"{BENCH_BINARY_NAME} exceeded {timeout_s:.0f}s"
        ) from exc
    except OSError as exc:
        raise BenchUnavailableError(f"could not run {BENCH_BINARY_NAME}: {exc}") from exc

    if completed.returncode != 0:
        # llama-bench puts the useful part (missing file, unsupported quant,
        # out of memory) on stderr, so surface that rather than the exit code.
        detail = (completed.stderr or completed.stdout or "").strip()
        raise BenchUnavailableError(
            f"{BENCH_BINARY_NAME} exited {completed.returncode}: {detail[-400:]}"
        )

    return completed.stdout


async def measure(bench: LocalBench) -> BenchResult:
    """SCH-1's entry point: this node's calibrated speed, at the defaults.

    Wire this into node start, before join, and fill tg_tok_s / pp_tok_s on the
    profile from the result. That is what unblocks the real scheduler behind
    GET /api/plan.
    """
    payload = await bench.run({})
    return BenchResult(
        pp_tok_s=payload["pp_tok_s"],
        tg_tok_s=payload["tg_tok_s"],
        pp_stddev=payload["pp_stddev"],
        tg_stddev=payload["tg_stddev"],
        model=payload["model"],
        build_commit=payload["build_commit"],
        n_gpu_layers=payload["n_gpu_layers"],
        n_threads=payload["n_threads"],
        repetitions=payload["repetitions"],
    )
