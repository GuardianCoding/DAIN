"""Small semantic document index used by the node agent's search endpoint.

The configured root is the trust boundary. Jobs can supply a query and result
limit, but never a path. Symlinks are skipped so an indexed tree cannot escape
that root accidentally. Every node uses the same local embedding model so its
cosine scores can be merged by the controller without corpus-dependent IDF.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from threading import Lock, RLock
from typing import Any, Protocol

INDEX_ROOT_ENV = "DAIN_INDEX_ROOT"
EMBED_MODEL_ENV = "DAIN_EMBED_MODEL"
EMBED_CACHE_ENV = "DAIN_EMBED_CACHE"
EMBED_ALLOW_DOWNLOAD_ENV = "DAIN_EMBED_ALLOW_DOWNLOAD"
DEFAULT_INDEX_ROOT = "/srv/dain/index"
DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_RESULT_LIMIT = 5
MAX_RESULT_LIMIT = 20
MAX_FILE_BYTES = 1_000_000
MAX_INDEX_BYTES = 256 * 1024 * 1024
MAX_INDEX_FILES = 10_000
MAX_SNIPPET_CHARS = 240
MAX_EMBED_CHARS = 4_000

TEXT_SUFFIXES = frozenset(
    {
        ".c",
        ".cpp",
        ".csv",
        ".h",
        ".hpp",
        ".java",
        ".js",
        ".json",
        ".log",
        ".md",
        ".py",
        ".rs",
        ".toml",
        ".ts",
        ".tsx",
        ".txt",
        ".yaml",
        ".yml",
    }
)

TOKEN_PATTERN = re.compile(r"[a-z0-9_]+")


@dataclass(frozen=True)
class IndexedDocument:
    path: str
    text: str
    embedding: tuple[float, ...]
    size_bytes: int


class IndexNotReadyError(RuntimeError):
    pass


class EmbeddingUnavailableError(RuntimeError):
    pass


class EmbeddingModel(Protocol):
    model_id: str

    def embed_documents(self, texts: list[str]) -> list[tuple[float, ...]]: ...

    def embed_query(self, query: str) -> tuple[float, ...]: ...


class LocalEmbeddingModel:
    """Lazy FastEmbed adapter; inference stays on this node with no API call."""

    def __init__(
        self,
        model_id: str = DEFAULT_EMBED_MODEL,
        *,
        cache_dir: str | None = None,
        local_files_only: bool | None = None,
    ) -> None:
        self.model_id = model_id
        self.cache_dir = cache_dir or os.getenv(EMBED_CACHE_ENV)
        self.local_files_only = (
            not _enabled(os.getenv(EMBED_ALLOW_DOWNLOAD_ENV))
            if local_files_only is None
            else local_files_only
        )
        self._model: Any | None = None
        self._lock = RLock()

    @classmethod
    def from_environment(cls) -> LocalEmbeddingModel:
        return cls(os.getenv(EMBED_MODEL_ENV, DEFAULT_EMBED_MODEL))

    def prewarm(self) -> None:
        self._load()

    def embed_documents(self, texts: list[str]) -> list[tuple[float, ...]]:
        if not texts:
            return []
        try:
            with self._lock:
                return [
                    tuple(float(value) for value in vector)
                    for vector in self._load().passage_embed(texts)
                ]
        except EmbeddingUnavailableError:
            raise
        except Exception as exc:
            raise EmbeddingUnavailableError(
                f"local embedding model {self.model_id!r} failed"
            ) from exc

    def embed_query(self, query: str) -> tuple[float, ...]:
        try:
            with self._lock:
                vector = next(iter(self._load().query_embed(query)))
            return tuple(float(value) for value in vector)
        except EmbeddingUnavailableError:
            raise
        except Exception as exc:
            raise EmbeddingUnavailableError(
                f"local embedding model {self.model_id!r} failed"
            ) from exc

    def _load(self) -> Any:
        with self._lock:
            if self._model is not None:
                return self._model
            try:
                from fastembed import TextEmbedding

                self._model = TextEmbedding(
                    model_name=self.model_id,
                    cache_dir=self.cache_dir,
                    lazy_load=False,
                    local_files_only=self.local_files_only,
                )
            except Exception as exc:
                raise EmbeddingUnavailableError(
                    f"local embedding model {self.model_id!r} is unavailable"
                ) from exc
            return self._model


class LocalFileIndex:
    def __init__(
        self,
        root: Path | str,
        *,
        max_files: int = MAX_INDEX_FILES,
        max_total_bytes: int = MAX_INDEX_BYTES,
        embedder: EmbeddingModel | None = None,
    ) -> None:
        if max_files <= 0:
            raise ValueError("max_files must be greater than zero")
        if max_total_bytes <= 0:
            raise ValueError("max_total_bytes must be greater than zero")

        self.root = Path(root).expanduser()
        self.max_files = max_files
        self.max_total_bytes = max_total_bytes
        self.embedder = embedder or LocalEmbeddingModel.from_environment()
        self.documents: tuple[IndexedDocument, ...] = ()
        self.indexed_at: float | None = None
        self.indexed_bytes = 0
        self.embedding_dimensions = 0
        self.generation = 0
        self.lock = RLock()
        self.refresh_lock = Lock()

    @classmethod
    def from_environment(cls) -> LocalFileIndex:
        return cls(os.getenv(INDEX_ROOT_ENV, DEFAULT_INDEX_ROOT))

    def refresh(self) -> dict[str, Any]:
        root = self.root.resolve()
        with self.lock:
            observed_generation = self.generation

        with self.refresh_lock:
            with self.lock:
                if self.generation != observed_generation:
                    return self._stats(root)

            documents: list[IndexedDocument] = []
            indexed_bytes = 0

            if root.is_dir():
                for path in sorted(root.rglob("*")):
                    if len(documents) >= self.max_files:
                        break

                    document = self._read_document(root, path)
                    if document is None:
                        continue
                    if indexed_bytes + document.size_bytes > self.max_total_bytes:
                        break

                    documents.append(document)
                    indexed_bytes += document.size_bytes

            if documents:
                vectors = self.embedder.embed_documents(
                    [embedding_text(document.text) for document in documents]
                )
                if len(vectors) != len(documents):
                    raise EmbeddingUnavailableError(
                        "embedding model returned the wrong number of vectors"
                    )
                dimensions = len(vectors[0])
                if dimensions == 0 or any(
                    len(vector) != dimensions for vector in vectors
                ):
                    raise EmbeddingUnavailableError(
                        "embedding model returned inconsistent vector dimensions"
                    )
                documents = [
                    IndexedDocument(
                        path=document.path,
                        text=document.text,
                        embedding=normalise(vector),
                        size_bytes=document.size_bytes,
                    )
                    for document, vector in zip(documents, vectors, strict=True)
                ]
            else:
                dimensions = 0

            indexed_at = time.time()
            with self.lock:
                self.documents = tuple(documents)
                self.indexed_at = indexed_at
                self.indexed_bytes = indexed_bytes
                self.embedding_dimensions = dimensions
                self.generation += 1
                return self._stats(root)

    def _stats(self, root: Path) -> dict[str, Any]:
        return {
            "root": str(root),
            "files_indexed": len(self.documents),
            "bytes_indexed": self.indexed_bytes,
            "indexed_at": self.indexed_at,
            "embedding_model": self.model_id,
            "embedding_dimensions": self.embedding_dimensions,
        }

    @property
    def model_id(self) -> str:
        return self.embedder.model_id

    def search(
        self,
        query: str,
        limit: int = DEFAULT_RESULT_LIMIT,
    ) -> list[dict[str, Any]]:
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")
        if not 1 <= limit <= MAX_RESULT_LIMIT:
            raise ValueError(f"limit must be between 1 and {MAX_RESULT_LIMIT}")

        with self.lock:
            if self.indexed_at is None:
                raise IndexNotReadyError(
                    "index is not ready; submit an index job first"
                )
            documents = self.documents

        if not documents:
            return []
        query_embedding = normalise(self.embedder.embed_query(query))
        if len(query_embedding) != len(documents[0].embedding):
            raise EmbeddingUnavailableError(
                "query and document embedding dimensions do not match"
            )
        query_terms = set(tokenize(query))

        ranked: list[dict[str, Any]] = []
        for document in documents:
            score = cosine_similarity(document.embedding, query_embedding)

            ranked.append(
                {
                    "path": document.path,
                    "score": round(score, 4),
                    "snippet": make_snippet(document.text, query_terms),
                }
            )

        ranked.sort(key=lambda hit: (-hit["score"], hit["path"]))
        return ranked[:limit]

    def _read_document(
        self,
        root: Path,
        path: Path,
    ) -> IndexedDocument | None:
        if path.is_symlink() or not path.is_file():
            return None
        if path.suffix.lower() not in TEXT_SUFFIXES:
            return None

        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(root)
            if resolved.stat().st_size > MAX_FILE_BYTES:
                return None
            text = resolved.read_text(encoding="utf-8", errors="ignore")
        except (OSError, ValueError):
            return None

        if not text.strip():
            return None

        return IndexedDocument(
            path=resolved.relative_to(root).as_posix(),
            text=text,
            embedding=(),
            size_bytes=resolved.stat().st_size,
        )


def tokenize(value: str) -> list[str]:
    return TOKEN_PATTERN.findall(value.casefold())


def make_snippet(
    text: str,
    query_terms: set[str],
) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    matching = next(
        (
            line
            for line in lines
            if any(term in line.casefold() for term in query_terms)
        ),
        lines[0] if lines else "",
    )
    collapsed = " ".join(matching.split())
    if len(collapsed) <= MAX_SNIPPET_CHARS:
        return collapsed
    return f"{collapsed[: MAX_SNIPPET_CHARS - 1].rstrip()}…"


def embedding_text(text: str) -> str:
    return text[:MAX_EMBED_CHARS]


def normalise(vector: tuple[float, ...]) -> tuple[float, ...]:
    magnitude = sqrt(sum(value * value for value in vector))
    if magnitude == 0:
        raise EmbeddingUnavailableError("embedding model returned a zero vector")
    return tuple(value / magnitude for value in vector)


def cosine_similarity(
    left: tuple[float, ...],
    right: tuple[float, ...],
) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _enabled(value: str | None) -> bool:
    return value is not None and value.strip().casefold() in {"1", "true", "yes", "on"}
