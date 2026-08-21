import argparse
import asyncio
import httpx
from fastapi import FastAPI
import uvicorn
from contracts import NodeProfile

app = FastAPI(title="DAIN Node Agent")

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
            except Exception as e:
                pass
            await asyncio.sleep(2.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ctl", type=str, default="192.168.0.100:8000")
    parser.add_argument("--port", type=int, default=9100)
    args = parser.parse_args()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.create_task(heartbeat_loop(args.ctl))

    config = uvicorn.Config(app, host="0.0.0.0", port=args.port, loop="asyncio")
    server = uvicorn.Server(config)
    loop.run_until_complete(server.serve())
