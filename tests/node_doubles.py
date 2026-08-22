"""Shared test doubles for the NODE-1 suites.

Imported by tests/test_dain_node.py (the control plane conversation) and
tests/test_node_fabric.py (the local machine).
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from collections import namedtuple
from typing import Any

import httpx

from contracts import NodeProfile

CTL = "192.168.50.20:8000"
NODE_ID = "office-01"
FABRIC_IP = "192.168.50.11"
POOL_SECRET = "test-pool-secret"
RPC_PATH = "/opt/dain/llama.cpp/build/bin/rpc-server"

FakeAddr = namedtuple("FakeAddr", "family address")
FakeMemory = namedtuple("FakeMemory", "total available")


def make_profile(node_id: str = NODE_ID, host: str = FABRIC_IP) -> NodeProfile:
    return NodeProfile(
        id=node_id,
        host=host,
        cpu="Intel Core i7-6700",
        cores=8,
        ram_total_mb=8192,
        ram_free_mb=6000,
        gpu=None,
        vram_total_mb=0,
        backend="cpu",
        mem_bandwidth_gbs=0.0,
        tg_tok_s=0.0,
        pp_tok_s=0.0,
        rtt_ms=0.0,
        state="joining",
    )


class StopLoop(Exception):
    """Ends heartbeat_loop deterministically once the script runs out."""


class FakeControlPlane:
    """Scripted control plane that records everything the agent sends.

    Each scripted entry is either a response to return or an exception to
    raise. Once the script is exhausted the next request raises StopLoop, so
    a test can run the real forever-loop and still terminate.
    """

    def __init__(self, *script: httpx.Response | Exception) -> None:
        self.script: list[httpx.Response | Exception] = list(script)
        self.requests: list[httpx.Request] = []

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if not self.script:
            raise StopLoop
        reply = self.script.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self._handle))

    def body(self, index: int = 0) -> dict[str, Any]:
        return json.loads(self.requests[index].content)

    def paths(self) -> list[str]:
        return [request.url.path for request in self.requests]


class FakeProcess:
    """Stands in for a live rpc-server child."""

    def __init__(self, *, stubborn: bool = False) -> None:
        self.pid = 4242
        self.stubborn = stubborn
        self.terminated = False
        self.killed = False
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        if not self.stubborn:
            self.returncode = 0

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is None:
            raise subprocess.TimeoutExpired(cmd="rpc-server", timeout=timeout or 0)
        return self.returncode

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


async def never_returns(agent, *args, **kwargs) -> None:
    """Stands in for heartbeat_loop, which never returns on its own."""
    await asyncio.Event().wait()
