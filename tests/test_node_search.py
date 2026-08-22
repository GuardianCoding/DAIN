import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import ClassVar

import pytest
from fastapi.testclient import TestClient

from node import dain_node
from node.auth import sign_job_request
from node.index import (
    EmbeddingUnavailableError,
    LocalEmbeddingModel,
    LocalFileIndex,
)
from tests.node_doubles import CTL, POOL_SECRET, make_profile


class FakeEmbeddingModel:
    model_id = "test/semantic-small"
    concepts: ClassVar[dict[str, int]] = {
        "automobile": 0,
        "car": 0,
        "maintenance": 1,
        "repair": 1,
        "monitoring": 2,
        "observability": 2,
        "telemetry": 2,
        "cluster": 3,
        "distributed": 3,
        "pool": 3,
    }

    def __init__(self) -> None:
        self.document_batches = 0
        self.query_calls = 0

    def embed_documents(self, texts: list[str]) -> list[tuple[float, ...]]:
        self.document_batches += 1
        return [self._embed(text) for text in texts]

    def embed_query(self, query: str) -> tuple[float, ...]:
        self.query_calls += 1
        return self._embed(query)

    def _embed(self, text: str) -> tuple[float, ...]:
        vector = [0.0] * 16
        for token in text.casefold().split():
            clean = token.strip(".,:;!?()[]{}")
            dimension = self.concepts.get(clean)
            if dimension is None:
                digest = hashlib.sha256(clean.encode()).digest()
                dimension = 4 + digest[0] % 12
            vector[dimension] += 1.0
        return tuple(vector)


class FailingEmbeddingModel(FakeEmbeddingModel):
    def embed_documents(self, texts: list[str]) -> list[tuple[float, ...]]:
        raise EmbeddingUnavailableError("test embedding model unavailable")


def job_request(
    kind: str,
    payload: dict | None = None,
    *,
    pool_secret: str = POOL_SECRET,
    shard_index: int = 0,
    shard_count: int = 1,
    issued_at: int | None = None,
) -> dict:
    body = {
        "job_id": "job-01",
        "kind": kind,
        "payload": payload or {},
        "shard_index": shard_index,
        "shard_count": shard_count,
        "issued_at": int(time.time()) if issued_at is None else issued_at,
    }
    body["signature"] = sign_job_request(
        pool_secret,
        job_id=body["job_id"],
        kind=body["kind"],
        payload=body["payload"],
        shard_index=body["shard_index"],
        shard_count=body["shard_count"],
        issued_at=body["issued_at"],
    )
    return body


@pytest.fixture(autouse=True)
def clean_agent_state():
    yield
    if getattr(dain_node.app.state, "agent", None) is not None:
        del dain_node.app.state.agent


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    search_index = LocalFileIndex(tmp_path, embedder=FakeEmbeddingModel())
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
    assert result["shard_count"] == 1
    assert result["embedding_model"] == "test/semantic-small"
    assert result["embedding_dimensions"] == 16


def test_index_endpoint_reports_an_unavailable_local_model(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("semantic search")
    dain_node.configure(
        make_profile(),
        ctl=CTL,
        pool_secret=POOL_SECRET,
        search_index=LocalFileIndex(tmp_path, embedder=FailingEmbeddingModel()),
    )
    client = TestClient(dain_node.app)

    response = client.post("/index", json=job_request("index"))

    assert response.status_code == 503
    assert "embedding model unavailable" in response.json()["detail"]


def test_search_returns_ranked_scored_snippets(
    client: TestClient,
    tmp_path: Path,
) -> None:
    (tmp_path / "best.md").write_text(
        "Telemetry observability keeps the distributed cluster visible."
    )
    (tmp_path / "other.txt").write_text("Telemetry appears in this local note.")
    client.post("/index", json=job_request("index"))

    response = client.post(
        "/search",
        json=job_request("search", {"query": "telemetry cluster", "limit": 2}),
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert [hit["path"] for hit in result["hits"]] == ["best.md", "other.txt"]
    assert result["hits"][0]["score"] > result["hits"][1]["score"]
    assert "Telemetry" in result["hits"][0]["snippet"]
    assert result["hits"][0]["node_id"] == "office-01"
    assert result["hits"][0]["source"] == "office-01:best.md"
    assert result["embedding_model"] == "test/semantic-small"


def test_search_uses_semantic_embeddings_instead_of_lexical_overlap(
    client: TestClient,
    tmp_path: Path,
) -> None:
    (tmp_path / "vehicle.md").write_text("A practical car repair manual.")
    (tmp_path / "unrelated.md").write_text("Telemetry from a compute cluster.")
    client.post("/index", json=job_request("index"))

    response = client.post(
        "/search",
        json=job_request("search", {"query": "automobile maintenance", "limit": 2}),
    )

    assert response.status_code == 200
    assert response.json()["result"]["hits"][0]["path"] == "vehicle.md"


def test_search_requires_an_explicit_index_job(
    client: TestClient,
    tmp_path: Path,
) -> None:
    (tmp_path / "readme.md").write_text("distributed inference fabric")

    response = client.post(
        "/search",
        json=job_request("search", {"query": "inference"}),
    )

    assert response.status_code == 409
    assert "index job first" in response.json()["detail"]


def test_search_rejects_a_wrong_pool_secret(client: TestClient) -> None:
    response = client.post(
        "/search",
        json=job_request(
            "search",
            {"query": "dain"},
            pool_secret="wrong-pool-secret",
        ),
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "invalid job signature"


def test_search_rejects_a_tampered_signed_payload(client: TestClient) -> None:
    request = job_request("search", {"query": "original"})
    request["payload"]["query"] = "tampered"

    response = client.post("/search", json=request)

    assert response.status_code == 403


def test_search_rejects_a_stale_signed_request(client: TestClient) -> None:
    response = client.post(
        "/search",
        json=job_request(
            "search",
            {"query": "dain"},
            issued_at=int(time.time()) - 31,
        ),
    )

    assert response.status_code == 403


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


def test_concurrent_refreshes_share_one_filesystem_scan(
    tmp_path: Path,
    monkeypatch,
) -> None:
    for index in range(50):
        (tmp_path / f"document-{index}.txt").write_text(f"document {index}")

    embedder = FakeEmbeddingModel()
    search_index = LocalFileIndex(tmp_path, embedder=embedder)
    original_read = search_index._read_document
    reads = 0
    reads_lock = threading.Lock()
    start = threading.Barrier(5)

    def counted_read(root: Path, path: Path):
        nonlocal reads
        with reads_lock:
            reads += 1
        time.sleep(0.001)
        return original_read(root, path)

    def refresh() -> dict:
        start.wait()
        return search_index.refresh()

    monkeypatch.setattr(search_index, "_read_document", counted_read)

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(refresh) for _ in range(4)]
        start.wait()
        results = [future.result(timeout=2.0) for future in futures]

    assert reads == 50
    assert {result["files_indexed"] for result in results} == {50}
    assert embedder.document_batches == 1


def test_scores_are_comparable_across_different_corpus_sizes(tmp_path: Path) -> None:
    small_root = tmp_path / "small"
    large_root = tmp_path / "large"
    small_root.mkdir()
    large_root.mkdir()
    text = "telemetry telemetry cluster"
    (small_root / "target.md").write_text(text)
    (large_root / "target.md").write_text(text)
    for index in range(20):
        (large_root / f"filler-{index}.txt").write_text("unrelated material")

    small = LocalFileIndex(small_root, embedder=FakeEmbeddingModel())
    large = LocalFileIndex(large_root, embedder=FakeEmbeddingModel())
    small.refresh()
    large.refresh()

    assert (
        small.search("telemetry")[0]["score"] == large.search("telemetry")[0]["score"]
    )


def test_index_enforces_file_and_total_byte_caps(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("123456")
    (tmp_path / "b.txt").write_text("123456")
    (tmp_path / "c.txt").write_text("123456")

    file_limited = LocalFileIndex(
        tmp_path,
        max_files=2,
        max_total_bytes=100,
        embedder=FakeEmbeddingModel(),
    )
    byte_limited = LocalFileIndex(
        tmp_path,
        max_files=10,
        max_total_bytes=10,
        embedder=FakeEmbeddingModel(),
    )

    assert file_limited.refresh()["files_indexed"] == 2
    assert file_limited.indexed_bytes == 12
    assert byte_limited.refresh()["files_indexed"] == 1
    assert byte_limited.indexed_bytes == 6


def test_fastembed_adapter_uses_passage_and_query_modes() -> None:
    calls: list[tuple[str, object]] = []

    class Backend:
        def passage_embed(self, texts):
            calls.append(("passages", texts))
            return iter([(1.0, 0.0), (0.0, 1.0)])

        def query_embed(self, query):
            calls.append(("query", query))
            return iter([(0.5, 0.5)])

    model = LocalEmbeddingModel("test/local-model")
    model._model = Backend()

    documents = model.embed_documents(["first", "second"])
    query = model.embed_query("question")

    assert documents == [(1.0, 0.0), (0.0, 1.0)]
    assert query == (0.5, 0.5)
    assert calls == [
        ("passages", ["first", "second"]),
        ("query", "question"),
    ]
