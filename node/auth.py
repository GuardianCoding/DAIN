"""Authentication helpers shared by the controller and node job endpoints."""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from typing import Any

JOB_AUTH_MAX_AGE_S = 30


def sign_job_request(
    pool_secret: str,
    *,
    job_id: str,
    kind: str,
    payload: dict[str, Any],
    shard_index: int,
    shard_count: int,
    issued_at: int,
) -> str:
    """Return a stable HMAC for every security-relevant request field."""
    if not pool_secret:
        raise ValueError("pool_secret must not be empty")

    message = _canonical_job_request(
        job_id=job_id,
        kind=kind,
        payload=payload,
        shard_index=shard_index,
        shard_count=shard_count,
        issued_at=issued_at,
    )
    return hmac.new(
        pool_secret.encode("utf-8"),
        message,
        hashlib.sha256,
    ).hexdigest()


def verify_job_request(
    pool_secret: str,
    *,
    job_id: str,
    kind: str,
    payload: dict[str, Any],
    shard_index: int,
    shard_count: int,
    issued_at: int,
    signature: str,
    now: int | None = None,
) -> bool:
    """Accept a correctly signed request only within the replay window."""
    current_time = int(time.time()) if now is None else now
    if abs(current_time - issued_at) > JOB_AUTH_MAX_AGE_S:
        return False

    try:
        expected = sign_job_request(
            pool_secret,
            job_id=job_id,
            kind=kind,
            payload=payload,
            shard_index=shard_index,
            shard_count=shard_count,
            issued_at=issued_at,
        )
    except (TypeError, ValueError):
        return False

    return hmac.compare_digest(expected, signature)


def _canonical_job_request(
    *,
    job_id: str,
    kind: str,
    payload: dict[str, Any],
    shard_index: int,
    shard_count: int,
    issued_at: int,
) -> bytes:
    return json.dumps(
        {
            "issued_at": issued_at,
            "job_id": job_id,
            "kind": kind,
            "payload": payload,
            "shard_count": shard_count,
            "shard_index": shard_index,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
