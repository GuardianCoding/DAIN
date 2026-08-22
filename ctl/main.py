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
from ctl.queue import JobQueue, NodeUnavailableError
from ctl.registry import NodeRegistry
from ctl.telemetry import TelemetryFanIn


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

JOB_QUEUE = JobQueue(REGISTRY)
TELEMETRY = TelemetryFanIn(REGISTRY)


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

    TELEMETRY.reset(REGISTRY.latest_metrics())


seed_registry()


async def monitor_heartbeats() -> None:
    while True:
        await asyncio.sleep(1.0)
        REGISTRY.sweep()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    monitor_task = asyncio.create_task(monitor_heartbeats())
    await TELEMETRY.start()

    try:
        yield
    finally:
        monitor_task.cancel()

        try:
            await monitor_task
        except asyncio.CancelledError:
            pass

        await TELEMETRY.close()
        await JOB_QUEUE.close()


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

    TELEMETRY.remove(node_id)
    return Response(status_code=204)


@app.get("/api/plan")
def get_plan(model: str = Query(min_length=1)) -> dict[str, Any]:
    try:
        return asdict(MOCK_STATE.plan(model))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/jobs", status_code=201)
async def create_job(request: JobRequest) -> dict[str, Any]:
    try:
        job = await JOB_QUEUE.submit(
            kind=request.kind,
            payload=request.payload,
            fanout=request.fanout,
            node_id=request.node_id,
        )
    except NodeUnavailableError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    response = JOB_QUEUE.response(job.id)
    if response is None:
        raise HTTPException(status_code=500, detail="job was not stored")

    return response


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    response = JOB_QUEUE.response(job_id)

    if response is None:
        raise HTTPException(status_code=404, detail="job not found")

    return response


@app.get("/api/metrics")
def get_metrics() -> dict[str, Any]:
    return TELEMETRY.frame()


@app.post("/api/race")
def run_race(request: RaceRequest) -> dict[str, Any]:
    return MOCK_STATE.race(request.task, request.mode)


@app.websocket("/feed")
async def send_feed(websocket: WebSocket) -> None:
    await websocket.accept()
    topology = get_nodes()
    await websocket.send_json({"type": "topology", "nodes": topology})
    last_registry_sequence = 0
    last_queue_sequence = 0
    last_topology = topology

    try:
        while True:
            topology = get_nodes()
            if topology != last_topology:
                await websocket.send_json({"type": "topology", "nodes": topology})
                last_topology = topology

            await websocket.send_json(get_metrics())

            registry_events = REGISTRY.events_after(last_registry_sequence)

            for event in registry_events:
                await websocket.send_json(
                    {
                        "type": "event",
                        "source": "registry",
                        **asdict(event),
                    }
                )
                last_registry_sequence = event.sequence

            queue_events = JOB_QUEUE.events_after(last_queue_sequence)

            for event in queue_events:
                event_data = asdict(event)

                if event.node_id is None:
                    # Queued, started, completed, failed and cancelled.
                    await websocket.send_json(
                        {
                            "type": "event",
                            "source": "queue",
                            **event_data,
                        }
                    )
                else:
                    # Dispatch, retry and reassignment between ctl and a node.
                    await websocket.send_json(
                        {
                            "type": "flow",
                            "source": "ctl",
                            "target": event.node_id,
                            "label": event.event,
                            **event_data,
                        }
                    )

                last_queue_sequence = event.sequence
                await asyncio.sleep(TELEMETRY.interval_s)

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

    if request.metrics is not None:
        TELEMETRY.record(request.metrics)

    return {
        "node_id": record.profile.id,
        "state": record.profile.state,
        "missed_heartbeats": record.missed_heartbeats,
    }
