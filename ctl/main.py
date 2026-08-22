import asyncio
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict, replace
from typing import Any, Literal

from fastapi import FastAPI, Header, HTTPException, Query, Response, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.websockets import WebSocketDisconnect

from contracts import NodeMetrics, NodeProfile
from ctl.auth import AuthenticationError, JoinAuthManager
from ctl.mock import MOCK_POOL_SECRET, MOCK_STATE
from ctl.queue import JobQueue, NodeUnavailableError
from ctl.registry import NodeRegistry
from ctl.telemetry import TelemetryFanIn
from node.discovery import advertise_control_plane


class JoinRequest(BaseModel):
    profile: NodeProfile
    nonce: str = Field(min_length=1)
    signature: str = Field(min_length=64, max_length=64)


class JoinChallengeRequest(BaseModel):
    node_id: str = Field(min_length=1)


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

JOB_QUEUE = JobQueue(REGISTRY, pool_secret=MOCK_POOL_SECRET)
TELEMETRY = TelemetryFanIn(REGISTRY)
AUTH = JoinAuthManager(MOCK_POOL_SECRET)


def seed_registry() -> None:
    REGISTRY.reset()
    AUTH.reset()

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


MOCK_NODE_ENV = "DAIN_MOCK_NODES"


def _mock_nodes_requested() -> bool:
    return os.getenv(MOCK_NODE_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


# Mock nodes are OFF by default. They used to be seeded unconditionally at
# import, so a live control plane always advertised gpu-01, office-01,
# office-02 and mac-01 whether or not those machines existed. On the fabric
# that is worse than useless: the dashboard shows four idle nodes, nothing is
# listening on :9100 for any of them, and a real node that joins is buried
# among the fakes. They register with heartbeat_required=False, so the offline
# sweep never retires them either — they sit there looking healthy forever.
#
# Set DAIN_MOCK_NODES=1 for a UI demo with no hardware. tests/test_main.py
# calls seed_registry() directly in its fixtures and is unaffected.
if _mock_nodes_requested():
    seed_registry()


async def monitor_heartbeats() -> None:
    while True:
        await asyncio.sleep(1.0)
        REGISTRY.sweep()


async def _wait_for_next_feed_cycle() -> None:
    await asyncio.sleep(TELEMETRY.interval_s)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    monitor_task = asyncio.create_task(monitor_heartbeats())
    await TELEMETRY.start()
    advertisement = None

    try:
        advertisement = await asyncio.to_thread(advertise_control_plane)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"mDNS control-plane advertisement unavailable: {exc}")

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
        if advertisement is not None:
            await asyncio.to_thread(advertisement.close)


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
    try:
        token = AUTH.complete_join(
            asdict(request.profile),
            request.nonce,
            request.signature,
        )
    except AuthenticationError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc

    profile = REGISTRY.register(request.profile)
    return {
        **asdict(profile),
        "access_token": token.access_token,
        "token_type": "bearer",
        "expires_at": token.expires_at,
    }


@app.post("/api/nodes/join/challenge")
def create_join_challenge(request: JoinChallengeRequest) -> dict[str, Any]:
    challenge = AUTH.issue_challenge(request.node_id)
    return {
        "node_id": challenge.node_id,
        "nonce": challenge.nonce,
        "expires_at": challenge.expires_at,
    }


@app.delete("/api/nodes/{node_id}", status_code=204)
def delete_node(node_id: str) -> Response:
    if not REGISTRY.remove(node_id):
        raise HTTPException(status_code=404, detail="node not found")

    TELEMETRY.remove(node_id)
    AUTH.revoke(node_id)
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

            await _wait_for_next_feed_cycle()

    except WebSocketDisconnect:
        return


@app.post("/api/nodes/{node_id}/heartbeat")
def heartbeat(
    node_id: str,
    request: HeartbeatRequest,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    token = _bearer_token(authorization)
    if token is None or not AUTH.validate_token(node_id, token):
        raise HTTPException(
            status_code=401,
            detail="invalid or expired bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

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


def _bearer_token(authorization: str | None) -> str | None:
    if authorization is None:
        return None
    scheme, separator, token = authorization.partition(" ")
    if not separator or scheme.casefold() != "bearer" or not token:
        return None
    return token
