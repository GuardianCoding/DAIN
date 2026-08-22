import argparse
import asyncio
import os
import signal
import socket
import subprocess
import sys
from contextlib import asynccontextmanager

import httpx
import psutil
import uvicorn
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

from contracts import NodeProfile


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rpc_proc
    # Startup
    rpc_proc = start_rpc_server("192.168.50.101", 50052)
    yield
    # Shutdown cleanup child
    cleanup_rpc_server()


app = FastAPI(title="DAIN Node Agent", lifespan=lifespan)

# Placeholder hardcoded profile
CURRENT_PROFILE = NodeProfile(
    id="node-tmp",
    host="127.0.0.1",
    cpu="Intel Core",
    cores=8,
    ram_total_mb=16384,
    ram_free_mb=12000,
    gpu=None,
    vram_total_mb=0,
    backend="cpu",
    mem_bandwidth_gbs=25.0,
    tg_tok_s=15.0,
    pp_tok_s=60.0,
    rtt_ms=0.5,
    state="joining",
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/profile")
async def get_profile():
    return CURRENT_PROFILE


# For telemetry
@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    mem = psutil.virtual_memory()
    cpu = psutil.cpu_percent()
    return (
        f"# HELP node_cpu_utilisation CPU usage percent\n"
        f"node_cpu_utilisation {cpu}\n"
        f"# HELP node_memory_free_bytes Available RAM\n"
        f"node_memory_free_bytes {mem.available}\n"
    )


async def heartbeat_loop(ctl_host: str):
    async with httpx.AsyncClient() as client:
        while True:
            try:
                # Registration / Heartbeat to Control Plane
                await client.post(
                    f"http://{ctl_host}/api/nodes/join",
                    json=CURRENT_PROFILE.__dict__,
                    timeout=2.0,
                )
            except Exception:
                pass
            await asyncio.sleep(2.0)


def build_local_profile(node_id: str, fabric_ip: str) -> NodeProfile:
    mem = psutil.virtual_memory()
    return NodeProfile(
        id=node_id,
        host=fabric_ip,
        cpu=os.environ.get("PROCESSOR_IDENTIFIER", "CPU"),
        cores=psutil.cpu_count(logical=True) or 4,
        ram_total_mb=int(mem.total / (1024 * 1024)),
        ram_free_mb=int(mem.available / (1024 * 1024)),
        gpu=None,  # Populate via Vulkan/DX probe if available
        vram_total_mb=0,
        backend="cpu",  # "vulkan" if GPU active
        mem_bandwidth_gbs=0.0,  # Populated by profiler
        tg_tok_s=0.0,
        pp_tok_s=0.0,
        rtt_ms=0.5,
        state="joining",
    )


# rpc-server
rpc_proc: subprocess.Popen | None = None


# Need to check where the rpc binary will be and adjust accordingly
def start_rpc_server(
    fabric_ip: str, port: int = 50052, rpc_bin: str = "rpc-server.exe"
):
    if os.path.exists(rpc_bin):
        return subprocess.Popen(
            [rpc_bin, "--host", fabric_ip, "-p", str(port), "-c"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    return None


def cleanup_rpc_server():
    global rpc_proc
    if rpc_proc and rpc_proc.poll() is None:
        print("[NODE] Stopping rpc-server")
        rpc_proc.terminate()
        try:
            rpc_proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            rpc_proc.kill()
        rpc_proc = None


signal.signal(signal.SIGINT, lambda s, f: (cleanup_rpc_server(), sys.exit(0)))
signal.signal(signal.SIGTERM, lambda s, f: (cleanup_rpc_server(), sys.exit(0)))


def get_fabric_ip(preferred_subet: str = "192.168.50.") -> str:
    for iface, addrs in psutil.net_if_addrs().items():
        for addr in addrs:
            if addr.family == socket.AF_INET and addr.address.startswith(
                preferred_subet
            ):
                return addr.address
    return "127.0.0.1"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ctl", type=str, default="192.168.50.100:8000")
    parser.add_argument("--port", type=int, default=9100)
    args = parser.parse_args()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.create_task(heartbeat_loop(args.ctl))

    config = uvicorn.Config(app, host="0.0.0.0", port=args.port, loop="asyncio")
    server = uvicorn.Server(config)
    loop.run_until_complete(server.serve())

    CURRENT_PROFILE = build_local_profile(socket.gethostname(), get_fabric_ip())
