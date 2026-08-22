"""DAIN node agent — NODE-1.

One file, one command:

    DAIN_POOL_SECRET=... python -m node.dain_node --ctl 192.168.50.20:8000

It profiles the machine, registers with the control plane, heartbeats every
two seconds, exposes /health /profile /metrics /index /search, and supervises
the local rpc-server. Set DAIN_INDEX_ROOT to the directory this node is allowed
to index; it defaults to /srv/dain/index.

Two contracts from the control plane (ctl/main.py) are load-bearing here:

    POST /api/nodes/join                  {"profile": ..., "pool_secret": ...} -> 201 | 403
    POST /api/nodes/{node_id}/heartbeat   {"metrics": ... | null}              -> 200 | 404

Neither takes a bare profile, and join is not a heartbeat: the registry
counts missed heartbeats and declares a node offline after three of them
(cluster.toml [membership]).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import httpx
import psutil
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from contracts import NodeMetrics, NodeProfile
from node.auth import verify_job_request
from node.index import IndexNotReadyError, LocalFileIndex

LOG = logging.getLogger("dain.node")

# cluster.toml [membership] — the control plane declares a node offline after
# three consecutive misses, so its interval and this one must agree.
HEARTBEAT_INTERVAL_S = 2.0

# cluster.toml [discovery] — ports are conventions, addresses are not.
DEFAULT_NODE_PORT = 9100
DEFAULT_RPC_PORT = 50052

# cluster.toml [paths].llama. Every node is Linux x86-64, so there is one
# binary name and one layout; DAIN_LLAMA_BIN overrides it on a dev box.
DEFAULT_LLAMA_BIN_DIR = "/opt/dain/llama.cpp/build/bin"
RPC_BINARY_NAME = "rpc-server"

# cluster.toml [discovery] names these; the values never land in the repo.
POOL_SECRET_ENV = "DAIN_POOL_SECRET"
FABRIC_IFACE_ENV = "DAIN_FABRIC_IFACE"
CTL_ENDPOINT_ENV = "DAIN_CTL"
LLAMA_BIN_DIR_ENV = "DAIN_LLAMA_BIN"

# Every node is Linux, so this is the CPU name; the old PROCESSOR_IDENTIFIER
# environment variable it replaces only ever existed on Windows.
CPUINFO_PATH = Path("/proc/cpuinfo")

LOOPBACK = "127.0.0.1"
WILDCARD_ADDRESSES = frozenset({"", "0.0.0.0", "::"})
BYTES_PER_MIB = 1024 * 1024
HTTP_TIMEOUT_S = 2.0
RPC_STOP_TIMEOUT_S = 3.0
DISCARD_PORT = 9  # UDP connect() to here fixes a route without sending a packet

HTTP_CREATED = 201
HTTP_FORBIDDEN = 403
HTTP_NOT_FOUND = 404

EXIT_OK = 0
EXIT_MISCONFIGURED = 2


# --------------------------------------------------------------------------
# Fabric address detection
# --------------------------------------------------------------------------


def interface_ipv4(iface: str) -> str | None:
    """Return the first IPv4 address bound to `iface`, if it has one."""
    for addr in psutil.net_if_addrs().get(iface, []):
        if addr.family == socket.AF_INET:
            return addr.address
    return None


def route_source_ip(ctl_host: str) -> str | None:
    """Ask the kernel which local address it would use to reach `ctl_host`.

    A UDP connect() only fixes the route; nothing is transmitted, so this is
    safe to call before the control plane is up.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect((ctl_host, DISCARD_PORT))
            return probe.getsockname()[0]
    except OSError:
        return None


def detect_fabric_ip(ctl_host: str) -> str:
    """Resolve the address this node should be reached on.

    No address is hardcoded: DAIN_FABRIC_IFACE wins when it is set, otherwise
    it is whichever interface has a route to the control plane (cluster.toml
    [discovery].bind_interface_env). Loopback is the last resort and means the
    fabric is down.
    """
    iface = os.environ.get(FABRIC_IFACE_ENV)
    if iface:
        address = interface_ipv4(iface)
        if address:
            return address
        LOG.warning(
            "%s=%s has no IPv4 address; falling back to the route to %s",
            FABRIC_IFACE_ENV,
            iface,
            ctl_host,
        )

    address = route_source_ip(ctl_host)
    if address and address != LOOPBACK:
        return address

    LOG.warning(
        "no route to %s; binding loopback only. Set %s to name the fabric NIC.",
        ctl_host,
        FABRIC_IFACE_ENV,
    )
    return LOOPBACK


# --------------------------------------------------------------------------
# Self profiling
# --------------------------------------------------------------------------


def read_cpu_model(cpuinfo: Path = CPUINFO_PATH) -> str:
    """The CPU model name as Linux reports it."""
    try:
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine() or "unknown"


def build_local_profile(node_id: str, fabric_ip: str) -> NodeProfile:
    """Probe this machine into the NodeProfile the scheduler consumes.

    The measured fields stay at zero until SCH-1's calibration probe fills
    them in — a spec-sheet guess here silently oversizes this node's slice of
    the model.
    """
    memory = psutil.virtual_memory()
    return NodeProfile(
        id=node_id,
        host=fabric_ip,
        cpu=read_cpu_model(),
        cores=psutil.cpu_count(logical=True) or 1,
        ram_total_mb=memory.total // BYTES_PER_MIB,
        ram_free_mb=memory.available // BYTES_PER_MIB,
        gpu=None,
        vram_total_mb=0,
        backend="cpu",
        mem_bandwidth_gbs=0.0,
        tg_tok_s=0.0,
        pp_tok_s=0.0,
        rtt_ms=0.0,
        state="joining",
    )


def sample_metrics(node_id: str) -> NodeMetrics:
    """One live sample for the heartbeat body and the /metrics endpoint."""
    memory = psutil.virtual_memory()
    return NodeMetrics(
        node_id=node_id,
        timestamp=time.time(),
        cpu_percent=psutil.cpu_percent(),
        ram_free_mb=memory.available // BYTES_PER_MIB,
        gpu_percent=None,
        vram_free_mb=None,
        jobs_running=0,
    )


# --------------------------------------------------------------------------
# Local rpc-server supervision
# --------------------------------------------------------------------------


def resolve_rpc_binary() -> Path | None:
    """Locate the Linux rpc-server binary, or None if this node has no build."""
    bin_dir = Path(os.environ.get(LLAMA_BIN_DIR_ENV, DEFAULT_LLAMA_BIN_DIR))
    candidate = bin_dir / RPC_BINARY_NAME
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return candidate

    on_path = shutil.which(RPC_BINARY_NAME)
    return Path(on_path) if on_path else None


def start_rpc_server(
    fabric_ip: str, port: int = DEFAULT_RPC_PORT
) -> subprocess.Popen[bytes] | None:
    """Start rpc-server bound to the fabric interface only.

    rpc-server has no authentication of any kind, so a wildcard bind hands
    arbitrary compute to anything that can reach this box. That is refused
    rather than warned about.
    """
    if fabric_ip in WILDCARD_ADDRESSES:
        raise ValueError(
            f"refusing to bind rpc-server to {fabric_ip!r}: it has no authentication"
        )

    binary = resolve_rpc_binary()
    if binary is None:
        LOG.warning(
            "no %s under %s or on PATH; this node joins but serves no inference",
            RPC_BINARY_NAME,
            os.environ.get(LLAMA_BIN_DIR_ENV, DEFAULT_LLAMA_BIN_DIR),
        )
        return None

    LOG.info("starting %s on %s:%s", binary, fabric_ip, port)
    return subprocess.Popen(
        [str(binary), "--host", fabric_ip, "-p", str(port), "-c"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def stop_rpc_server(proc: subprocess.Popen[bytes] | None) -> None:
    """Terminate a running rpc-server, killing it if it will not go quietly."""
    if proc is None or proc.poll() is not None:
        return

    LOG.info("stopping %s (pid %s)", RPC_BINARY_NAME, proc.pid)
    proc.terminate()
    try:
        proc.wait(timeout=RPC_STOP_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        proc.kill()


# --------------------------------------------------------------------------
# Agent state
# --------------------------------------------------------------------------


@dataclass
class NodeAgent:
    """Runtime state for one agent process.

    `profile` is replaced rather than mutated, so a reader that captured it
    keeps a consistent snapshot.
    """

    profile: NodeProfile
    ctl: str
    pool_secret: str
    rpc_port: int = DEFAULT_RPC_PORT
    rpc_proc: subprocess.Popen[bytes] | None = None
    search_index: LocalFileIndex = field(
        default_factory=LocalFileIndex.from_environment
    )

    @property
    def join_url(self) -> str:
        return f"http://{self.ctl}/api/nodes/join"

    @property
    def heartbeat_url(self) -> str:
        return f"http://{self.ctl}/api/nodes/{self.profile.id}/heartbeat"


# --------------------------------------------------------------------------
# Control plane client
# --------------------------------------------------------------------------


def decode_body(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return None


def adopt_reported_state(agent: NodeAgent, body: Any) -> None:
    """Take the control plane's word for this node's state, if it sent one."""
    if isinstance(body, dict) and body.get("state"):
        agent.profile = replace(agent.profile, state=body["state"])


async def join_pool(client: httpx.AsyncClient, agent: NodeAgent) -> bool:
    """Register with the control plane. True once it answers 201."""
    payload = {"profile": asdict(agent.profile), "pool_secret": agent.pool_secret}

    try:
        response = await client.post(
            agent.join_url, json=payload, timeout=HTTP_TIMEOUT_S
        )
    except httpx.HTTPError as exc:
        LOG.warning("join to %s failed: %s", agent.ctl, exc)
        return False

    if response.status_code == HTTP_FORBIDDEN:
        LOG.error(
            "join REFUSED for %s — the pool secret in $%s does not match the "
            "control plane",
            agent.profile.id,
            POOL_SECRET_ENV,
        )
        return False

    if response.status_code != HTTP_CREATED:
        LOG.warning("join returned %s: %s", response.status_code, response.text[:200])
        return False

    adopt_reported_state(agent, decode_body(response))
    LOG.info("joined the pool as %s on %s", agent.profile.id, agent.profile.host)
    return True


async def send_heartbeat(client: httpx.AsyncClient, agent: NodeAgent) -> bool:
    """Send one heartbeat. False means the control plane wants a fresh join.

    A transport error returns True: the control plane counts the miss and this
    node keeps beating rather than re-registering over a flapping link.
    """
    payload = {"metrics": asdict(sample_metrics(agent.profile.id))}

    try:
        response = await client.post(
            agent.heartbeat_url, json=payload, timeout=HTTP_TIMEOUT_S
        )
    except httpx.HTTPError as exc:
        LOG.warning("heartbeat to %s failed: %s", agent.ctl, exc)
        return True

    if response.status_code == HTTP_NOT_FOUND:
        LOG.info("control plane does not know %s; re-joining", agent.profile.id)
        return False

    if response.is_error:
        LOG.warning(
            "heartbeat returned %s: %s", response.status_code, response.text[:200]
        )
        return True

    adopt_reported_state(agent, decode_body(response))
    return True


async def heartbeat_loop(
    agent: NodeAgent,
    client: httpx.AsyncClient | None = None,
    interval_s: float = HEARTBEAT_INTERVAL_S,
) -> None:
    """Join once, then heartbeat forever, re-joining whenever ctl forgets us."""
    async with contextlib.AsyncExitStack() as stack:
        if client is None:
            client = await stack.enter_async_context(httpx.AsyncClient())

        registered = await join_pool(client, agent)
        while True:
            await asyncio.sleep(interval_s)
            registered = (
                await send_heartbeat(client, agent)
                if registered
                else await join_pool(client, agent)
            )


# --------------------------------------------------------------------------
# HTTP surface
# --------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(scope: FastAPI) -> AsyncIterator[None]:
    agent = current_agent(scope)
    agent.rpc_proc = start_rpc_server(agent.profile.host, agent.rpc_port)
    beat = asyncio.create_task(heartbeat_loop(agent))

    try:
        yield
    finally:
        beat.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await beat
        stop_rpc_server(agent.rpc_proc)
        agent.rpc_proc = None


app = FastAPI(title="DAIN Node Agent", lifespan=lifespan)


class LocalJobRequest(BaseModel):
    job_id: str = Field(min_length=1)
    kind: Literal["index", "search"]
    payload: dict[str, Any] = Field(default_factory=dict)
    shard_index: int = Field(ge=0)
    shard_count: int = Field(ge=1)
    issued_at: int = Field(ge=0)
    signature: str = Field(min_length=64, max_length=64)


def authenticated_agent(request: LocalJobRequest) -> NodeAgent:
    agent = current_agent()
    if not verify_job_request(
        agent.pool_secret,
        job_id=request.job_id,
        kind=request.kind,
        payload=request.payload,
        shard_index=request.shard_index,
        shard_count=request.shard_count,
        issued_at=request.issued_at,
        signature=request.signature,
    ):
        raise HTTPException(status_code=HTTP_FORBIDDEN, detail="invalid job signature")
    return agent


def configure(
    profile: NodeProfile,
    ctl: str,
    pool_secret: str,
    rpc_port: int = DEFAULT_RPC_PORT,
    search_index: LocalFileIndex | None = None,
) -> NodeAgent:
    """Install the agent state the routes and the lifespan read.

    Called before uvicorn starts, so /profile never serves a placeholder and
    rpc-server binds the address this node actually reported at join.
    """
    agent = NodeAgent(
        profile=profile,
        ctl=ctl,
        pool_secret=pool_secret,
        rpc_port=rpc_port,
        search_index=search_index or LocalFileIndex.from_environment(),
    )
    app.state.agent = agent
    return agent


def current_agent(scope: FastAPI = app) -> NodeAgent:
    agent = getattr(scope.state, "agent", None)
    if agent is None:
        raise RuntimeError("node agent is not configured; call configure() first")
    return agent


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/profile")
async def get_profile() -> dict[str, Any]:
    return asdict(current_agent().profile)


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics() -> str:
    agent = current_agent()
    sample = sample_metrics(agent.profile.id)
    label = f'{{node_id="{sample.node_id}"}}'
    rpc_up = int(agent.rpc_proc is not None and agent.rpc_proc.poll() is None)

    return (
        f"# HELP node_cpu_utilisation CPU usage percent\n"
        f"# TYPE node_cpu_utilisation gauge\n"
        f"node_cpu_utilisation{label} {sample.cpu_percent}\n"
        f"# HELP node_memory_free_mib Available RAM in MiB\n"
        f"# TYPE node_memory_free_mib gauge\n"
        f"node_memory_free_mib{label} {sample.ram_free_mb}\n"
        f"# HELP node_rpc_server_up 1 when the local rpc-server is running\n"
        f"# TYPE node_rpc_server_up gauge\n"
        f"node_rpc_server_up{label} {rpc_up}\n"
    )


@app.post("/index")
async def refresh_index(request: LocalJobRequest) -> dict[str, Any]:
    agent = authenticated_agent(request)
    if request.kind != "index":
        raise HTTPException(status_code=422, detail="kind must be index")

    stats = await asyncio.to_thread(agent.search_index.refresh)
    return {
        "ok": True,
        "result": {
            "node_id": agent.profile.id,
            "shard_index": request.shard_index,
            "shard_count": request.shard_count,
            **stats,
        },
    }


@app.post("/search")
async def search(request: LocalJobRequest) -> dict[str, Any]:
    agent = authenticated_agent(request)
    if request.kind != "search":
        raise HTTPException(status_code=422, detail="kind must be search")

    query = request.payload.get("query")
    if not isinstance(query, str) or not query.strip():
        raise HTTPException(status_code=422, detail="payload.query must not be empty")

    limit = request.payload.get("limit", 5)
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise HTTPException(status_code=422, detail="payload.limit must be an integer")

    try:
        hits = await asyncio.to_thread(agent.search_index.search, query, limit)
    except IndexNotReadyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {
        "ok": True,
        "result": {
            "node_id": agent.profile.id,
            "query": query,
            "shard_index": request.shard_index,
            "shard_count": request.shard_count,
            "hits": [
                {
                    **hit,
                    "node_id": agent.profile.id,
                    "source": f"{agent.profile.id}:{hit['path']}",
                }
                for hit in hits
            ],
        },
    }


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="dain-node", description="DAIN node agent (NODE-1)"
    )
    parser.add_argument(
        "--ctl",
        default=os.environ.get(CTL_ENDPOINT_ENV, ""),
        help=f"control plane host:port (or ${CTL_ENDPOINT_ENV})",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_NODE_PORT)
    parser.add_argument("--rpc-port", type=int, default=DEFAULT_RPC_PORT)
    parser.add_argument("--node-id", default=None, help="defaults to the hostname")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="[node] %(levelname)s %(message)s")
    args = parse_args(argv)

    if not args.ctl:
        LOG.error(
            "no control plane endpoint: pass --ctl host:port or set $%s "
            "(NODE-2's mDNS discovery removes this argument)",
            CTL_ENDPOINT_ENV,
        )
        return EXIT_MISCONFIGURED

    pool_secret = os.environ.get(POOL_SECRET_ENV)
    if not pool_secret:
        LOG.error("$%s is not set; the control plane answers 403", POOL_SECRET_ENV)
        return EXIT_MISCONFIGURED

    # Profile first: the join payload, /profile and the rpc-server bind address
    # all read it, and all three start with the server.
    ctl_host = args.ctl.rsplit(":", 1)[0]
    fabric_ip = detect_fabric_ip(ctl_host)
    profile = build_local_profile(args.node_id or socket.gethostname(), fabric_ip)
    configure(profile, ctl=args.ctl, pool_secret=pool_secret, rpc_port=args.rpc_port)

    LOG.info(
        "profiled %s: %s, %s cores, %s MiB RAM, fabric %s",
        profile.id,
        profile.cpu,
        profile.cores,
        profile.ram_total_mb,
        fabric_ip,
    )

    # Bind the agent to the fabric too. uvicorn's own signal handling drives
    # the lifespan shutdown that stops rpc-server, so no handler is installed
    # at import time.
    uvicorn.run(app, host=fabric_ip, port=args.port, log_level="info")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
