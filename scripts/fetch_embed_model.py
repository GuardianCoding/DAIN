#!/usr/bin/env python3
"""Pre-cache the NODE-5 embedding model so /index works without network.

    ./scripts/fetch_embed_model.py                  # default model + cache
    ./scripts/fetch_embed_model.py --check          # report, download nothing

WHY THIS EXISTS:

node/index.py refuses to download at request time unless
DAIN_EMBED_ALLOW_DOWNLOAD is set, so on a cold machine POST /index answers

    503  embedding model BAAI/bge-small-en-v1.5 is unavailable offline

and POST /search then answers 409, because no index was ever built. Both read
as bugs and neither is: nothing had fetched the model. Run this once per node
that will serve index/search, while you still have a network.

NOTE ON WHICH MODEL: this is NOT the `embed` entry in infer/models.toml. That
ladder is llama.cpp GGUF weights for the inference fabric and names
Qwen3-Embedding-0.6B; node/index.py independently defaults to FastEmbed's
BAAI/bge-small-en-v1.5. Two different stacks, two different caches, and only
the second one serves /index. If the team wants a single embedding model that
is a decision to make deliberately — this script follows node/index.py,
because that is what /index actually loads.

scripts/install_node.sh already does this during a normal install. Reach for
this when the cache is missing anyway: a node installed before that step
existed, a cache wiped by cleanup, or a hand-built node.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from node.index import (
    DEFAULT_EMBED_MODEL,
    EMBED_CACHE_ENV,
    EMBED_MODEL_ENV,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="fetch_embed_model",
        description="Download the FastEmbed model /index needs.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv(EMBED_MODEL_ENV, DEFAULT_EMBED_MODEL),
        help=f"model id (default: ${EMBED_MODEL_ENV} or {DEFAULT_EMBED_MODEL})",
    )
    parser.add_argument(
        "--cache",
        default=os.getenv(EMBED_CACHE_ENV),
        help=f"cache directory (default: ${EMBED_CACHE_ENV}, else FastEmbed's)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="report whether the model already loads offline; download nothing",
    )
    return parser.parse_args(argv)


def load(model_id: str, cache: str | None, *, local_only: bool):
    """Load exactly the way node/index.py does.

    FastEmbed, not sentence-transformers — different constructor, different
    cache layout. Loading it via the wrong library would report success while
    leaving /index's cache empty. lazy_load=False forces the weights to be
    fetched now rather than on first embed, which is the whole point.
    """
    from fastembed import TextEmbedding

    return TextEmbedding(
        model_name=model_id,
        cache_dir=cache,
        lazy_load=False,
        local_files_only=local_only,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        import fastembed  # noqa: F401
    except ImportError:
        print(
            "fastembed is not installed in this environment.\n"
            "Run this on the node that serves /index, inside its venv.",
            file=sys.stderr,
        )
        return 1

    print(f"model: {args.model}")
    print(f"cache: {args.cache or 'the FastEmbed default cache'}")

    # Always try offline first. If it loads there is nothing to do, and we
    # avoid touching the network on a machine that is already provisioned.
    try:
        load(args.model, args.cache, local_only=True)
    except Exception as exc:  # noqa: BLE001 - FastEmbed raises unrelated types
        if args.check:
            print(f"NOT CACHED — /index would return 503 ({type(exc).__name__})")
            return 1
        print("not cached locally; downloading...")
    else:
        print("already cached — /index can load it offline")
        return 0

    try:
        load(args.model, args.cache, local_only=False)
    except Exception as exc:  # noqa: BLE001
        print(f"download failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    # Prove the thing we actually care about: that it now loads with the
    # network unavailable, which is the condition /index runs under.
    try:
        load(args.model, args.cache, local_only=True)
    except Exception as exc:  # noqa: BLE001
        print(f"downloaded, but still will not load offline: {exc}", file=sys.stderr)
        return 1

    print("cached and verified offline — /index will work on this node")
    if args.cache:
        print("\nSet on the node so it finds the same cache:")
        print(f"  export {EMBED_CACHE_ENV}={args.cache}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
