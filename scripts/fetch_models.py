#!/usr/bin/env python3
"""Fetch the DAIN model ladder in priority order.

Python rather than shell because it reads the same models.toml the rest of
infer/ does, and a second parser for that file is a second thing to get wrong.
Needs Python 3.11+ (tomllib) and the `hf` CLI.

    python3 scripts/fetch_models.py --list
    python3 scripts/fetch_models.py --dest /srv/dain/models --max-priority 4
    python3 scripts/fetch_models.py --dest /srv/dain/models          # everything

Priority order is the whole point: if the venue link dies partway through,
everything already fetched is still a working demo. Never reorder it.
Re-running resumes — `hf download` picks up where it stopped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from infer.models import (  # noqa: E402  (path shim must precede the import)
    BYTES_PER_GB,
    LADDER_PATH,
    ModelSpec,
    format_ladder,
    load_ladder,
)

MANIFEST_NAME = "manifest.json"
DISK_SAFETY_MARGIN_GB = 10.0


def check_disk_space(dest: Path, needed_gb: float) -> None:
    """Refuse to start a multi-hour download that cannot possibly finish."""
    dest.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(dest).free / BYTES_PER_GB
    required = needed_gb + DISK_SAFETY_MARGIN_GB
    if free_gb < required:
        raise RuntimeError(
            f"{dest} has {free_gb:.1f} GB free; need {required:.1f} GB "
            f"({needed_gb:.1f} GB of models + {DISK_SAFETY_MARGIN_GB:.0f} GB margin). "
            f"Free space, or pass --dest on a larger volume."
        )
    print(f"  disk: {free_gb:.1f} GB free at {dest}, need {required:.1f} GB -- ok")


def directory_size_bytes(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def download(spec: ModelSpec, dest: Path, *, dry_run: bool) -> Path:
    """Fetch one model. Resumable, so a dropped link costs minutes, not the file."""
    target = dest / spec.model_id
    command = [
        "hf", "download", spec.repo,
        "--include", spec.include,
        "--local-dir", str(target),
    ]

    print(f"\n[{spec.priority}] {spec.model_id}  {spec.size_gb:.1f} GB  <- {spec.repo}")
    if spec.claim:
        print(f"      {spec.claim.splitlines()[0]}")
    print(f"      $ {' '.join(command)}")
    if dry_run:
        return target

    env = {**os.environ, "HF_HUB_ENABLE_HF_TRANSFER": "1"}
    started = time.monotonic()
    result = subprocess.run(command, env=env, check=False)
    elapsed_min = (time.monotonic() - started) / 60

    if result.returncode != 0:
        raise RuntimeError(
            f"{spec.model_id} failed (exit {result.returncode}) after {elapsed_min:.1f} min. "
            f"Re-run this script — the download resumes."
        )

    actual_gb = directory_size_bytes(target) / BYTES_PER_GB
    print(f"      done: {actual_gb:.1f} GB in {elapsed_min:.1f} min")
    return target


def sha256_of(path: Path, chunk_bytes: int = 1 << 20) -> str:
    """Streamed digest — these files do not fit in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _describe_files(model_dir: Path, *, checksum: bool) -> list[dict]:
    """List the GGUF files, optionally with digests for post-transport checking."""
    entries = []
    for item in sorted(model_dir.rglob("*.gguf")):
        record = {"name": item.name, "size_bytes": item.stat().st_size}
        if checksum:
            record["sha256"] = sha256_of(item)
        entries.append(record)
    return entries


def write_manifest(dest: Path, specs: tuple[ModelSpec, ...], *, checksum: bool = False) -> Path:
    """Record what landed where, so a copy can be verified after transport.

    Worth the wait when the models travel on an external drive: a truncated
    63 GB file looks fine until llama-server fails to load it at the venue.
    """
    payload = {
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dest": str(dest),
        "models": [
            {
                "model_id": spec.model_id,
                "role": spec.role,
                "repo": spec.repo,
                "expected_gb": spec.size_gb,
                "actual_bytes": directory_size_bytes(dest / spec.model_id),
                "files": _describe_files(dest / spec.model_id, checksum=checksum),
            }
            for spec in specs
        ],
    }
    manifest_path = dest / MANIFEST_NAME
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def select(specs: tuple[ModelSpec, ...], only: str | None, max_priority: int | None) -> tuple[ModelSpec, ...]:
    """Filter the ladder, preserving priority order."""
    chosen = specs
    if only:
        wanted = {name.strip() for name in only.split(",") if name.strip()}
        known = {spec.model_id for spec in specs}
        unknown = wanted - known
        if unknown:
            raise ValueError(
                f"unknown model id(s): {', '.join(sorted(unknown))}. "
                f"Known: {', '.join(sorted(known))}"
            )
        chosen = tuple(spec for spec in chosen if spec.model_id in wanted)
    if max_priority is not None:
        chosen = tuple(spec for spec in chosen if spec.priority <= max_priority)
    if not chosen:
        raise ValueError("selection matched no models")
    return chosen


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch the DAIN model ladder in priority order.",
    )
    parser.add_argument("--dest", type=Path, help="Directory to download into (required unless --list)")
    parser.add_argument("--ladder", type=Path, default=LADDER_PATH, help="Path to models.toml")
    parser.add_argument("--list", action="store_true", help="Print the ladder and exit")
    parser.add_argument("--max-priority", type=int, help="Stop after this priority level")
    parser.add_argument("--only", help="Comma-separated model ids to fetch")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without downloading")
    parser.add_argument(
        "--checksum",
        action="store_true",
        help="Record sha256 per file. Slow, but the only way to prove a drive copy survived.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        specs = load_ladder(args.ladder)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if args.list:
        print(format_ladder(specs))
        return 0

    if args.dest is None:
        print("error: --dest is required unless --list is given", file=sys.stderr)
        return 2

    try:
        chosen = select(specs, args.only, args.max_priority)
        total_gb = sum(spec.size_gb for spec in chosen)
        print(f"Fetching {len(chosen)} model(s), {total_gb:.1f} GB total, into {args.dest}")
        if not args.dry_run:
            check_disk_space(args.dest, total_gb)
        for spec in chosen:
            download(spec, args.dest, dry_run=args.dry_run)
        if not args.dry_run:
            if args.checksum:
                print("\nchecksumming (this takes a while on 60 GB files)...")
            print(f"\nmanifest: {write_manifest(args.dest, chosen, checksum=args.checksum)}")
    except (OSError, RuntimeError, ValueError) as error:
        print(f"\nerror: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
