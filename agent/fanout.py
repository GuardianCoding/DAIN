"""ask_pool — one prompt per machine, all at once.

WHY N JOBS AND NOT ONE FAN-OUT JOB:

ctl.queue._split_payload shards on a `tasks` or `items` list, handing each node
`{"tasks": [subset]}`. But node.infer.LocalInference.complete() reads
`payload["prompt"]` and raises without it, which the route maps to 422. So
`{"kind": "infer", "payload": {"tasks": [...]}, "fanout": 5}` fails on every
node today.

Sending N separate single-prompt jobs works right now with no changes anywhere
else, and it buys something the sharded version could not: each job is pinned
to a named node. Left unpinned the queue ranks by least-busy and can land every
prompt on the same machine, which is the exact opposite of the demonstration.

The trade is that the fan-out is ours rather than the queue's. Each job still
appears on /feed individually, so the dashboard draws all N regardless.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from agent.client import DainClient, DainError

MAX_TOKENS = 256


@dataclass(frozen=True)
class PoolAnswer:
    node_id: str
    prompt: str
    text: str = ""
    tok_s: float | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def _first_shard_result(job: dict[str, Any]) -> dict[str, Any]:
    result = job.get("result") or {}
    shards = result.get("shards") or []
    if shards and isinstance(shards[0], dict):
        inner = shards[0].get("result")
        if isinstance(inner, dict):
            return inner
    return {}


def _shard_error(job: dict[str, Any]) -> str:
    """The node's own words, which are the readable half of a failed job."""
    result = job.get("result") or {}
    errors = result.get("errors") or []
    for entry in errors:
        if isinstance(entry, dict) and entry.get("error"):
            return str(entry["error"])
    return f"job {job.get('id', '?')} {job.get('status', 'failed')} without a reason"


async def _one(
    client: DainClient,
    node_id: str,
    prompt: str,
    max_tokens: int,
) -> PoolAnswer:
    try:
        job = await client.submit_and_wait(
            "infer",
            {"prompt": prompt, "max_tokens": max_tokens},
            node_id=node_id,
        )
    except DainError as exc:
        return PoolAnswer(node_id=node_id, prompt=prompt, error=str(exc))

    if job.get("status") != "done":
        return PoolAnswer(node_id=node_id, prompt=prompt, error=_shard_error(job))

    payload = _first_shard_result(job)
    tok_s = payload.get("tok_s")
    return PoolAnswer(
        node_id=node_id,
        prompt=prompt,
        text=str(payload.get("text") or ""),
        tok_s=tok_s if isinstance(tok_s, (int, float)) else None,
    )


async def ask_pool(
    client: DainClient,
    prompts: list[str],
    *,
    max_tokens: int = MAX_TOKENS,
) -> tuple[PoolAnswer, ...]:
    """Run every prompt concurrently, one per node, wrapping when short.

    Answers come back in prompt order regardless of which finished first, and a
    node that fails costs its own answer only — a partly-configured pool (some
    nodes without $DAIN_INFER_MODEL) is the normal state, not a reason to throw
    away the answers that did arrive.
    """
    if not prompts:
        return ()

    try:
        nodes = await client.available_nodes()
    except DainError as exc:
        return tuple(
            PoolAnswer(node_id="", prompt=prompt, error=str(exc)) for prompt in prompts
        )

    if not nodes:
        reason = "no nodes are available to answer; the pool is empty or all offline"
        return tuple(
            PoolAnswer(node_id="", prompt=prompt, error=reason) for prompt in prompts
        )

    return tuple(
        await asyncio.gather(
            *(
                _one(client, nodes[index % len(nodes)], prompt, max_tokens)
                for index, prompt in enumerate(prompts)
            )
        )
    )
