from fastapi.testclient import TestClient

from ctl.main import app

client = TestClient(app)


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


def test_feed():
    with client.websocket_connect("/feed") as websocket:
        message = websocket.receive_json()

        assert {message["type"]} == {"topology"}
        metrics_message = websocket.receive_json()

        assert metrics_message["type"] == "metrics"

        metrics = metrics_message["nodes"]
        assert {metric["node_id"] for metric in metrics} == {
            "gpu-01",
            "office-01",
            "office-02",
            "mac-01",
        }

        gpu_metrics = next(
            metric for metric in metrics if metric["node_id"] == "gpu-01"
        )
        assert gpu_metrics["gpu_percent"] is not None
        assert gpu_metrics["jobs_running"] == 1
