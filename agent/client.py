"""HTTP transport to the control plane — everything the agent's tools call.

Deliberately separate from agent/tools.py: the JSON schema a model is shown and
the httpx call made on its behalf are different concerns, and the second has to
be unit-testable without a model in the loop.

Nothing here formats for a model. Every method returns ctl's own JSON, and
every failure raises DainError carrying ctl's own `detail` string — those texts
were written to be read ("no node has a measured tg_tok_s — profile the cluster
before planning") and replacing them with a generic message throws away the one
thing that lets a model recover. Turning an error into a tool result is
agent/tools.py's job.

The agent's own thinking does NOT come through here. That goes straight to the
llama-server head on :8080. This is only the tool surface, and mixing the two
gives you a loop that queues a job to think about queueing a job.
"""

from __future__ import annotations

import asyncio
import os
import time
from types import TracebackType
from typing import Any, Self

import httpx

from ctl.queue import DEFAULT_TIMEOUTS_S

CTL_ENV = "DAIN_CTL"
DEFAULT_CTL = "127.0.0.1:8000"

DEFAULT_POLL_INTERVAL_S = 0.25
REQUEST_TIMEOUT_S = 10.0

# contracts.NodeState values the queue refuses to dispatch to. "degraded" is
# deliberately absent: a degraded node is slow, not gone, and excluding it
# would shrink the pool during exactly the recovery the demo shows off.
UNAVAILABLE_STATES = frozenset({"joining", "offline"})

# The third layer of the same invariant the queue's table encodes:
#
#     node ceiling  <  queue dispatch timeout  <  agent polling deadline
#
# The queue decides a job's fate; polling only observes it. Give up first and
# the agent reports a timeout for a job that was still going to succeed, which
# a model cannot tell apart from a broken cluster. Derived rather than written
# out so retuning the queue cannot silently invert the ordering.
POLL_MARGIN_S = 30.0

# Only reachable via a kind that is not in the queue's table, which today is
# none of them. Present so a future kind degrades to "waits a while" rather
# than "gives up immediately".
UNKNOWN_KIND_DEADLINE_S = 60.0


class DainError(RuntimeError):
    """The control plane refused, failed, or could not be reached."""


def job_deadline_s(kind: str) -> float:
    """How long to keep polling a job of this kind before giving up."""
    return DEFAULT_TIMEOUTS_S.get(kind, UNKNOWN_KIND_DEADLINE_S) + POLL_MARGIN_S


def normalise_api_base(ctl: str) -> str:
    """Accept `host:port`, a full URL, or either with `/api` already on it.

    ctl serves REST under /api but mounts the socket at bare /feed, so the
    trailing /api is easy to include once and forget. The dashboard hit the
    mirror image of this and now warns about it in lib/config.ts.
    """
    base = (ctl or "").strip().rstrip("/")
    if not base:
        raise ValueError("control-plane address must not be empty")
    if not base.startswith(("http://", "https://")):
        base = f"http://{base}"
    if base.endswith("/api"):
        return base
    return f"{base}/api"


class DainClient:
    def __init__(
        self,
        ctl: str = DEFAULT_CTL,
        *,
        client: httpx.AsyncClient | None = None,
        poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    ) -> None:
        if poll_interval_s <= 0:
            raise ValueError("poll_interval_s must be greater than zero")

        self.ctl = ctl
        self.api_base = normalise_api_base(ctl)
        self.poll_interval_s = poll_interval_s
        self.client = client or httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S)
        self.owns_client = client is None

    @classmethod
    def from_environment(cls) -> DainClient:
        return cls(os.getenv(CTL_ENV, DEFAULT_CTL).strip() or DEFAULT_CTL)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self.owns_client and not self.client.is_closed:
            await self.client.aclose()

    # -- reads ---------------------------------------------------------------

    async def nodes(self) -> list[dict[str, Any]]:
        body = await self._request("GET", "/nodes")
        return body if isinstance(body, list) else []

    async def available_nodes(self) -> list[str]:
        """Node ids the queue would actually dispatch to, in registry order.

        The same filter ctl.queue._ranked_nodes applies. Asking for a fan-out
        wider than this is a 503 from ctl ("fan-out N requested but only M
        node(s) are available"), so callers size their work against it.
        """
        return [
            node["id"]
            for node in await self.nodes()
            if isinstance(node, dict)
            and node.get("id")
            and node.get("state") not in UNAVAILABLE_STATES
        ]

    async def metrics(self) -> dict[str, Any]:
        body = await self._request("GET", "/metrics")
        return body if isinstance(body, dict) else {}

    async def plan(
        self, model: str, context: int = 8192, slots: int = 1
    ) -> dict[str, Any]:
        return await self._request(
            "GET",
            "/plan",
            params={"model": model, "context": context, "slots": slots},
        )

    async def job(self, job_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/jobs/{job_id}")

    # -- jobs ----------------------------------------------------------------

    async def submit(
        self,
        kind: str,
        payload: dict[str, Any],
        *,
        fanout: int = 1,
        node_id: str | None = None,
    ) -> dict[str, Any]:
        return await self._request(
            "POST",
            "/jobs",
            json={
                "kind": kind,
                "payload": payload,
                "fanout": fanout,
                "node_id": node_id,
            },
        )

    async def submit_and_wait(
        self,
        kind: str,
        payload: dict[str, Any],
        *,
        fanout: int = 1,
        node_id: str | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """Queue a job and poll until it settles.

        Returns the job record whether it succeeded or failed — a failed job is
        an outcome the model should be told about, not an exception. Only an
        unreachable ctl or a job still running past the deadline raises.
        """
        job = await self.submit(kind, payload, fanout=fanout, node_id=node_id)
        job_id = job.get("id")
        if not isinstance(job_id, str) or not job_id:
            raise DainError(f"ctl accepted the {kind} job but returned no id")

        deadline_s = job_deadline_s(kind) if timeout_s is None else timeout_s
        expires_at = time.monotonic() + deadline_s

        while True:
            job = await self.job(job_id)
            if job.get("status") in {"done", "failed"}:
                return job
            if time.monotonic() >= expires_at:
                raise DainError(
                    f"{kind} job {job_id} was still "
                    f"{job.get('status', 'running')} after {deadline_s:.0f}s"
                )
            await asyncio.sleep(self.poll_interval_s)

    # -- plumbing ------------------------------------------------------------

    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        url = f"{self.api_base}{path}"
        try:
            response = await self.client.request(method, url, **kwargs)
        except httpx.HTTPError as exc:
            raise DainError(
                f"control plane at {self.ctl} is unreachable: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        if response.status_code >= 400:
            raise DainError(
                f"{method} {path} returned {response.status_code}: {_detail(response)}"
            )

        try:
            return response.json()
        except ValueError as exc:
            raise DainError(f"{method} {path} returned a non-JSON body") from exc


def _detail(response: httpx.Response) -> str:
    """ctl's own explanation, which is the part worth keeping."""
    try:
        body = response.json()
    except ValueError:
        return response.text.strip() or "no detail"

    if isinstance(body, dict):
        detail = body.get("detail")
        if isinstance(detail, str) and detail:
            return detail
        if detail is not None:
            return str(detail)
    return response.text.strip() or "no detail"
