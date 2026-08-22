import asyncio
import json
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from ctl.main import JOB_QUEUE, TELEMETRY, app, seed_registry
from ctl.mock import MOCK_POOL_SECRET, reset_mock_state

client: TestClient


@pytest.fixture(scope="module", autouse=True)
def run_control_plane():
    global client

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)

        return httpx.Response(
            200,
            json={
                "ok": True,
                "result": {
                    "node": request.url.host,
                    "kind": body["kind"],
                },
            },
        )

    original_client = JOB_QUEUE.client
    asyncio.run(original_client.aclose())

    mock_node_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    JOB_QUEUE.client = mock_node_client
    JOB_QUEUE.owns_client = False

    with TestClient(app) as running_client:
        client = running_client
        yield

    asyncio.run(mock_node_client.aclose())


@pytest.fixture(autouse=True)
def reset_state():
    reset_mock_state()
    seed_registry()

    with JOB_QUEUE.lock:
        JOB_QUEUE.jobs.clear()
        JOB_QUEUE.fanouts.clear()
        JOB_QUEUE.assigned_nodes.clear()
        JOB_QUEUE.tasks.clear()
        JOB_QUEUE.semaphores.clear()
        JOB_QUEUE.in_flight.clear()
        JOB_QUEUE.events.clear()
        JOB_QUEUE.next_sequence = 1

    yield

    reset_mock_state()
    seed_registry()


def joined_node_payload() -> dict:
    return {
        "profile": {
            "id": "guest-01",
            "host": "192.168.50.14",
            "cpu": "Mock guest CPU",
            "cores": 8,
            "ram_total_mb": 16384,
            "ram_free_mb": 12288,
            "gpu": None,
            "vram_total_mb": 0,
            "backend": "cpu",
            "mem_bandwidth_gbs": 40.0,
            "tg_tok_s": 8.0,
            "pp_tok_s": 55.0,
            "rtt_ms": 0.6,
            "state": "joining",
        },
        "pool_secret": MOCK_POOL_SECRET,
    }


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_mock_nodes():
    response = client.get("/api/nodes")
    nodes = response.json()

    assert response.status_code == 200
    assert len(nodes) == 4
    assert {node["id"] for node in nodes} == {
        "gpu-01",
        "office-01",
        "office-02",
        "mac-01",
    }


def test_join_rejects_wrong_secret():
    payload = joined_node_payload()
    payload["pool_secret"] = "wrong"

    response = client.post("/api/nodes/join", json=payload)

    assert response.status_code == 403


def test_join_and_delete_node():
    join_response = client.post("/api/nodes/join", json=joined_node_payload())

    assert join_response.status_code == 201
    assert join_response.json()["id"] == "guest-01"
    assert len(client.get("/api/nodes").json()) == 5

    delete_response = client.delete("/api/nodes/guest-01")

    assert delete_response.status_code == 204
    assert len(client.get("/api/nodes").json()) == 4
    assert client.delete("/api/nodes/missing").status_code == 404


def test_mock_plan_covers_every_node():
    response = client.get("/api/plan", params={"model": "gpt-oss-20b"})
    plan = response.json()

    assert response.status_code == 200
    assert plan["model_id"] == "gpt-oss-20b"
    assert set(plan["layers"]) == {
        "gpu-01",
        "office-01",
        "office-02",
        "mac-01",
    }
    assert len(plan["tensor_split"]) == 4
    assert plan["rationale"]


def test_create_and_retrieve_job():
    create_response = client.post(
        "/api/jobs",
        json={
            "kind": "infer",
            "payload": {"prompt": "Summarise the cluster"},
            "fanout": 2,
        },
    )
    created = create_response.json()

    assert create_response.status_code == 201
    assert created["status"] == "queued"
    assert created["fanout"] == 2
    assert len(created["assigned_nodes"]) == 2

    for _ in range(100):
        get_response = client.get(f"/api/jobs/{created['id']}")
        retrieved = get_response.json()

        if retrieved["status"] in {"done", "failed"}:
            break

        time.sleep(0.005)

    assert get_response.status_code == 200
    assert retrieved["id"] == created["id"]
    assert retrieved["status"] == "done"
    assert retrieved["fanout"] == 2
    assert retrieved["result"]["errors"] == []
    assert client.get("/api/jobs/missing").status_code == 404


def test_job_rejects_unknown_pinned_node():
    response = client.post(
        "/api/jobs",
        json={"kind": "bench", "payload": {}, "node_id": "missing"},
    )

    assert response.status_code == 404


def test_metrics_cover_every_node():
    response = client.get("/api/metrics")
    message = response.json()

    assert response.status_code == 200
    assert message["type"] == "metrics"
    assert {metric["node_id"] for metric in message["nodes"]} == {
        "gpu-01",
        "office-01",
        "office-02",
        "mac-01",
    }
    assert set(message["history"]) == {
        "gpu-01",
        "office-01",
        "office-02",
        "mac-01",
    }
    assert message["llama"] == {}
    assert message["llama_history"] == []
    assert all(metric["timestamp"] > 0 for metric in message["nodes"])


def test_race_models_serial_and_fanout():
    serial = client.post(
        "/api/race", json={"task": "Review twenty files", "mode": "serial"}
    ).json()
    fanout = client.post(
        "/api/race", json={"task": "Review twenty files", "mode": "fanout"}
    ).json()

    assert serial["nodes_used"] == 1
    assert serial["speedup"] == 1.0
    assert fanout["nodes_used"] == 4
    assert fanout["speedup"] == 3.3


def test_feed_exposes_all_four_frame_types():
    created_job = client.post(
        "/api/jobs",
        json={
            "kind": "search",
            "payload": {"query": "cluster"},
            "fanout": 2,
        },
    ).json()

    with client.websocket_connect("/feed") as websocket:
        topology = websocket.receive_json()
        frames_by_type: dict[str, dict] = {}

        for _ in range(30):
            frame = websocket.receive_json()
            frame_type = frame.get("type")

            if frame_type in {"metrics", "event"}:
                frames_by_type.setdefault(frame_type, frame)

            if frame_type == "flow" and frame.get("job_id") == created_job["id"]:
                frames_by_type["flow"] = frame

            if {"metrics", "event", "flow"} <= frames_by_type.keys():
                break
        else:
            pytest.fail(
                f"Feed did not provide all frame types: {frames_by_type.keys()}"
            )

    metrics = frames_by_type["metrics"]
    event = frames_by_type["event"]
    flow = frames_by_type["flow"]

    assert topology["type"] == "topology"
    assert len(topology["nodes"]) == 4

    assert len(metrics["nodes"]) == 4
    assert len(metrics["history"]) == 4

    assert event["message"]

    assert flow["source"] == "ctl"
    assert flow["job_id"] == created_job["id"]
    assert flow["target"] in created_job["assigned_nodes"]


def test_heartbeat_updates_joined_node():
    client.post("/api/nodes/join", json=joined_node_payload())

    response = client.post(
        "/api/nodes/guest-01/heartbeat",
        json={
            "metrics": {
                "node_id": "guest-01",
                "timestamp": 1000.0,
                "cpu_percent": 25.0,
                "ram_free_mb": 11000,
                "gpu_percent": None,
                "vram_free_mb": None,
                "jobs_running": 1,
            }
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "node_id": "guest-01",
        "state": "idle",
        "missed_heartbeats": 0,
    }
    frame = client.get("/api/metrics").json()
    guest = next(metric for metric in frame["nodes"] if metric["node_id"] == "guest-01")

    assert guest["ram_free_mb"] == 11000
    assert frame["history"]["guest-01"][-1]["timestamp"] == 1000.0


def test_heartbeat_rejects_unknown_node():
    response = client.post(
        "/api/nodes/missing/heartbeat",
        json={},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "node not found"


def test_heartbeat_rejects_mismatched_metrics():
    client.post("/api/nodes/join", json=joined_node_payload())

    response = client.post(
        "/api/nodes/guest-01/heartbeat",
        json={
            "metrics": {
                "node_id": "office-01",
                "timestamp": 1000.0,
                "cpu_percent": 25.0,
                "ram_free_mb": 6000,
                "gpu_percent": None,
                "vram_free_mb": None,
                "jobs_running": 0,
            }
        },
    )

    assert response.status_code == 422


def test_telemetry_background_task_is_running():
    assert TELEMETRY.task is not None
    assert not TELEMETRY.task.done()


def test_feed_broadcasts_topology_changes():
    with client.websocket_connect("/feed") as websocket:
        initial = websocket.receive_json()
        assert initial["type"] == "topology"

        response = client.post(
            "/api/nodes/join",
            json=joined_node_payload(),
        )
        assert response.status_code == 201

        for _ in range(20):
            frame = websocket.receive_json()

            if frame.get("type") != "topology":
                continue

            if any(node["id"] == "guest-01" for node in frame["nodes"]):
                break
        else:
            pytest.fail("feed did not broadcast the topology change")


def test_metrics_frame_matches_dashboard_contract():
    frame = client.get("/api/metrics").json()

    assert set(frame) == {
        "type",
        "nodes",
        "history",
        "llama",
        "llama_history",
        "errors",
    }
    assert frame["type"] == "metrics"

    node_fields = {
        "node_id",
        "timestamp",
        "cpu_percent",
        "ram_free_mb",
        "gpu_percent",
        "vram_free_mb",
        "jobs_running",
    }

    assert all(set(sample) == node_fields for sample in frame["nodes"])
    assert all(len(samples) <= 60 for samples in frame["history"].values())
