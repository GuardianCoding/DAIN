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
        nodes = message["nodes"]
        assert len(nodes) == 4
        assert {node["id"] for node in nodes} == {
            "gpu-01",
            "office-01",
            "office-02",
            "mac-01",
        }
