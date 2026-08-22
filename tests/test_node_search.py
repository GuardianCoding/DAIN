from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from node import dain_node
from node.index import LocalFileIndex
from tests.node_doubles import CTL, POOL_SECRET, make_profile


def job_request(kind: str, payload: dict | None = None) -> dict:
    return {
        "job_id": "job-01",
        "kind": kind,
        "payload": payload or {},
        "shard_index": 0,
        "shard_count": 1,
    }


@pytest.fixture(autouse=True)
def clean_agent_state():
    yield
    if getattr(dain_node.app.state, "agent", None) is not None:
        del dain_node.app.state.agent


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    search_index = LocalFileIndex(tmp_path)
    dain_node.configure(
        make_profile(),
        ctl=CTL,
        pool_secret=POOL_SECRET,
        search_index=search_index,
    )
    return TestClient(dain_node.app)


def test_index_endpoint_refreshes_the_configured_root(
    client: TestClient,
    tmp_path: Path,
) -> None:
    (tmp_path / "notes.md").write_text("DAIN telemetry and cluster notes")

    response = client.post("/index", json=job_request("index"))

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["files_indexed"] == 1
    assert result["node_id"] == "office-01"


def test_search_returns_ranked_scored_snippets(
    client: TestClient,
    tmp_path: Path,
) -> None:
    (tmp_path / "best.md").write_text(
        "Telemetry telemetry fan-in keeps a sixty sample history."
    )
    (tmp_path / "other.txt").write_text("Telemetry appears once here.")
    client.post("/index", json=job_request("index"))

    response = client.post(
        "/search",
        json=job_request("search", {"query": "telemetry", "limit": 2}),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert [hit["path"] for hit in result["hits"]] == ["best.md", "other.txt"]
    assert result["hits"][0]["score"] > result["hits"][1]["score"]
    assert "Telemetry" in result["hits"][0]["snippet"]


def test_search_auto_indexes_on_first_query(
    client: TestClient,
    tmp_path: Path,
) -> None:
    (tmp_path / "readme.md").write_text("distributed inference fabric")

    response = client.post(
        "/search",
        json=job_request("search", {"query": "inference"}),
    )

    assert response.status_code == 200
    assert response.json()["result"]["hits"][0]["path"] == "readme.md"


def test_search_rejects_an_empty_query(client: TestClient) -> None:
    response = client.post(
        "/search",
        json=job_request("search", {"query": " "}),
    )

    assert response.status_code == 422


def test_search_rejects_an_excessive_limit(client: TestClient) -> None:
    response = client.post(
        "/search",
        json=job_request("search", {"query": "dain", "limit": 21}),
    )

    assert response.status_code == 422


def test_index_skips_symlinks_outside_the_root(
    client: TestClient,
    tmp_path: Path,
) -> None:
    outside = tmp_path.parent / "private.txt"
    outside.write_text("secret material that must not be indexed")
    link = tmp_path / "escape.txt"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks are unavailable")

    client.post("/index", json=job_request("index"))
    response = client.post(
        "/search",
        json=job_request("search", {"query": "secret"}),
    )

    assert response.status_code == 200
    assert response.json()["result"]["hits"] == []
