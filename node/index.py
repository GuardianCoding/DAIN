"""Small local document index used by the node agent's search endpoint.

The configured root is the trust boundary. Jobs can supply a query and result
limit, but never a path. Symlinks are skipped so an indexed tree cannot escape
that root accidentally.
"""

from __future__ import annotations

import math
import os
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any

INDEX_ROOT_ENV = "DAIN_INDEX_ROOT"
DEFAULT_INDEX_ROOT = "/srv/dain/index"
DEFAULT_RESULT_LIMIT = 5
MAX_RESULT_LIMIT = 20
MAX_FILE_BYTES = 1_000_000
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


class LocalFileIndex:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).expanduser()
        self.documents: tuple[IndexedDocument, ...] = ()
        self.indexed_at: float | None = None
        self.lock = RLock()

    @classmethod
    def from_environment(cls) -> LocalFileIndex:
        return cls(os.getenv(INDEX_ROOT_ENV, DEFAULT_INDEX_ROOT))

    def refresh(self) -> dict[str, Any]:
        root = self.root.resolve()
        documents: list[IndexedDocument] = []

        if root.is_dir():
            for path in sorted(root.rglob("*")):
                document = self._read_document(root, path)
                if document is not None:
                    documents.append(document)

        indexed_at = time.time()
        with self.lock:
            self.documents = tuple(documents)
            self.indexed_at = indexed_at

        return {
            "root": str(root),
            "files_indexed": len(documents),
            "indexed_at": indexed_at,
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
            needs_refresh = self.indexed_at is None
        if needs_refresh:
            self.refresh()

        with self.lock:
            documents = self.documents

        query_terms = Counter(tokenize(query))
        if not query_terms or not documents:
            return []

        document_frequency = Counter(
            term
            for document in documents
            for term in query_terms
            if term in document.terms
        )

        ranked: list[dict[str, Any]] = []
        for document in documents:
            score = self._score(
                document,
                query,
                query_terms,
                document_frequency,
                len(documents),
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
        )

    @staticmethod
    def _score(
        document: IndexedDocument,
        query: str,
        query_terms: Counter[str],
        document_frequency: Counter[str],
        document_count: int,
    ) -> float:
        score = 0.0
        for term, query_frequency in query_terms.items():
            term_frequency = document.terms.get(term, 0)
            if term_frequency == 0:
                continue

            inverse_document_frequency = (
                math.log((document_count + 1) / (document_frequency[term] + 1)) + 1.0
            )
            score += (
                (1.0 + math.log(term_frequency))
                * inverse_document_frequency
                * query_frequency
            )

        if query.casefold() in document.text.casefold():
            score += 2.0
        return score


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
