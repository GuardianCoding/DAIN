import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict, replace
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query, Response, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.websockets import WebSocketDisconnect

from contracts import NodeMetrics, NodeProfile
from ctl.mock import MOCK_POOL_SECRET, MOCK_STATE
from ctl.registry import NodeRegistry


class JoinRequest(BaseModel):
    profile: NodeProfile
    pool_secret: str


class JobRequest(BaseModel):
    kind: Literal["infer", "exec", "index", "search", "bench"]
    payload: dict[str, Any] = Field(default_factory=dict)
    fanout: int = Field(default=1, ge=1, le=16)
    node_id: str | None = None


class RaceRequest(BaseModel):
    task: str = Field(min_length=1)
    mode: Literal["serial", "fanout"]


class HeartbeatRequest(BaseModel):
    metrics: NodeMetrics | None = None


def request_replan(node_id: str, reason: str) -> None:
    print(f"Scheduler re-plan requested: node={node_id}, reason={reason}")


REGISTRY = NodeRegistry(
    heartbeat_interval_s=2.0,
    missed_heartbeats_offline=3,
    on_replan=request_replan,
)


def seed_registry() -> None:
    REGISTRY.reset()

    metrics_by_node = {metric.node_id: metric for metric in MOCK_STATE.metrics()}

    for profile in MOCK_STATE.list_nodes():
        profile_copy = replace(profile)

        REGISTRY.register(
            profile_copy,
            heartbeat_required=False,
        )
        REGISTRY.heartbeat(
            profile_copy.id,
            metrics_by_node[profile_copy.id],
        )


seed_registry()


async def monitor_heartbeats() -> None:
    while True:
        await asyncio.sleep(1.0)
        REGISTRY.sweep()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    monitor_task = asyncio.create_task(monitor_heartbeats())

    try:
        yield
    finally:
        monitor_task.cancel()

        try:
            await monitor_task
        except asyncio.CancelledError:
            pass


app = FastAPI(
    title="DAIN control plane",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root() -> dict[str, str]:
    return {"service": "DAIN mock control plane", "docs": "/docs"}


@app.get("/health")
def check_health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/nodes")
def get_nodes() -> list[dict[str, Any]]:
    return [asdict(profile) for profile in REGISTRY.list_profiles()]


@app.post("/api/nodes/join", status_code=201)
def join_node(request: JoinRequest) -> dict[str, Any]:
    if request.pool_secret != MOCK_POOL_SECRET:
        raise HTTPException(status_code=403, detail="invalid pool secret")

    profile = REGISTRY.register(request.profile)
    return asdict(profile)


@app.delete("/api/nodes/{node_id}", status_code=204)
def delete_node(node_id: str) -> Response:
    if not REGISTRY.remove(node_id):
        raise HTTPException(status_code=404, detail="node not found")

    return Response(status_code=204)


@app.get("/api/plan")
def get_plan(model: str = Query(min_length=1)) -> dict[str, Any]:
    try:
        return asdict(MOCK_STATE.plan(model))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/jobs", status_code=201)
def create_job(request: JobRequest) -> dict[str, Any]:
    try:
        job = MOCK_STATE.create_job(
            kind=request.kind,
            payload=request.payload,
            fanout=request.fanout,
            node_id=request.node_id,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="requested node not found") from exc

    return MOCK_STATE.job_response(job)


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    job = MOCK_STATE.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return MOCK_STATE.job_response(job)


@app.get("/api/metrics")
def get_metrics() -> dict[str, Any]:
    return {
        "type": "metrics",
        "nodes": [asdict(metric) for metric in REGISTRY.latest_metrics()],
    }


@app.post("/api/race")
def run_race(request: RaceRequest) -> dict[str, Any]:
    return MOCK_STATE.race(request.task, request.mode)


@app.websocket("/feed")
async def send_feed(websocket: WebSocket) -> None:
    await websocket.accept()
    await websocket.send_json(
        {
            "type": "topology",
            "nodes": get_nodes(),
        }
    )

    last_sequence = 0

    try:
        while True:
            await websocket.send_json(get_metrics())

            events = REGISTRY.events_after(last_sequence)

            for event in events:
                await websocket.send_json(
                    {
                        "type": "event",
                        **asdict(event),
                    }
                )
                last_sequence = event.sequence

            # Jobs and execution flow remain mocked during CP-2.
            await websocket.send_json(MOCK_STATE.flow_frame())
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        return


@app.post("/api/nodes/{node_id}/heartbeat")
def heartbeat(node_id: str, request: HeartbeatRequest) -> dict[str, Any]:
    try:
        record = REGISTRY.heartbeat(node_id, request.metrics)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="node not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {
        "node_id": record.profile.id,
        "state": record.profile.state,
        "missed_heartbeats": record.missed_heartbeats,
    }
