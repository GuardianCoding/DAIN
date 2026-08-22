"""Small local document index used by the node agent's search endpoint.

The configured root is the trust boundary. Jobs can supply a query and result
limit, but never a path. Symlinks are skipped so an indexed tree cannot escape
that root accidentally.
"""

from __future__ import annotations

import os
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from threading import Lock, RLock
from typing import Any

INDEX_ROOT_ENV = "DAIN_INDEX_ROOT"
DEFAULT_INDEX_ROOT = "/srv/dain/index"
DEFAULT_RESULT_LIMIT = 5
MAX_RESULT_LIMIT = 20
MAX_FILE_BYTES = 1_000_000
MAX_INDEX_BYTES = 256 * 1024 * 1024
MAX_INDEX_FILES = 10_000
MAX_SNIPPET_CHARS = 240

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
    terms: Counter[str]
    size_bytes: int


class IndexNotReadyError(RuntimeError):
    pass


class LocalFileIndex:
    def __init__(
        self,
        root: Path | str,
        *,
        max_files: int = MAX_INDEX_FILES,
        max_total_bytes: int = MAX_INDEX_BYTES,
    ) -> None:
        if max_files <= 0:
            raise ValueError("max_files must be greater than zero")
        if max_total_bytes <= 0:
            raise ValueError("max_total_bytes must be greater than zero")

        self.root = Path(root).expanduser()
        self.max_files = max_files
        self.max_total_bytes = max_total_bytes
        self.documents: tuple[IndexedDocument, ...] = ()
        self.indexed_at: float | None = None
        self.indexed_bytes = 0
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

            indexed_at = time.time()
            with self.lock:
                self.documents = tuple(documents)
                self.indexed_at = indexed_at
                self.indexed_bytes = indexed_bytes
                self.generation += 1
                return self._stats(root)

    def _stats(self, root: Path) -> dict[str, Any]:
        return {
            "root": str(root),
            "files_indexed": len(self.documents),
            "bytes_indexed": self.indexed_bytes,
            "indexed_at": self.indexed_at,
        }

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

        query_terms = Counter(tokenize(query))
        if not query_terms or not documents:
            return []

        ranked: list[dict[str, Any]] = []
        for document in documents:
            score = self._score(
                document,
                query,
                query_terms,
            )
            if score <= 0:
                continue

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
            terms=Counter(tokenize(text)),
            size_bytes=resolved.stat().st_size,
        )

    @staticmethod
    def _score(
        document: IndexedDocument,
        query: str,
        query_terms: Counter[str],
    ) -> float:
        query_weight = sum(query_terms.values())
        matched_weight = sum(
            query_frequency
            for term, query_frequency in query_terms.items()
            if document.terms.get(term, 0) > 0
        )
        frequency_weight = sum(
            min(document.terms.get(term, 0), 3) * query_frequency
            for term, query_frequency in query_terms.items()
        )

        coverage = matched_weight / query_weight
        frequency = frequency_weight / (3 * query_weight)
        phrase = 1.0 if query.casefold() in document.text.casefold() else 0.0

        # Every component is bounded, so scores remain comparable across nodes
        # regardless of each node's local corpus size and document frequency.
        return 0.65 * coverage + 0.25 * frequency + 0.10 * phrase


def tokenize(value: str) -> list[str]:
    return TOKEN_PATTERN.findall(value.casefold())


def make_snippet(
    text: str,
    query_terms: Counter[str],
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
