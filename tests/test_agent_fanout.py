import json

import httpx
import pytest

from agent.client import DainClient
from agent.fanout import ask_pool


def profile(node_id: str, state: str = "idle") -> dict:
    return {
        "id": node_id,
        "host": f"{node_id}.local",
        "cpu": "test",
        "cores": 4,
        "ram_total_mb": 8192,
        "ram_free_mb": 6144,
        "gpu": None,
        "vram_total_mb": 0,
        "backend": "cpu",
        "mem_bandwidth_gbs": 20.0,
        "tg_tok_s": 11.0,
        "pp_tok_s": 40.0,
        "rtt_ms": 0.5,
        "state": state,
    }


def fake_ctl(nodes: list[dict], *, fail_on: set[str] | None = None) -> tuple:
    """A ctl that completes every infer job, failing the named nodes."""
    failing = fail_on or set()
    submitted: list[dict] = []
    jobs: dict[str, dict] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/nodes":
            return httpx.Response(200, json=nodes)

        if path == "/api/jobs" and request.method == "POST":
            body = json.loads(request.content)
            submitted.append(body)
            job_id = f"job-{len(submitted)}"
            node_id = body["node_id"]
            prompt = body["payload"]["prompt"]

            if node_id in failing:
                jobs[job_id] = {
                    "id": job_id,
                    "status": "failed",
                    "result": {
                        "shards": [],
                        "errors": [
                            {
                                "shard_index": 0,
                                "error": (
                                    f"shard 0 rejected by {node_id}: HTTP 503: "
                                    "no inference backend on this node"
                                ),
                            }
                        ],
                    },
                }
            else:
                jobs[job_id] = {
                    "id": job_id,
                    "status": "done",
                    "result": {
                        "shards": [
                            {
                                "shard_index": 0,
                                "node_id": node_id,
                                "result": {
                                    "node_id": node_id,
                                    "text": f"{node_id} says: {prompt}",
                                    "tok_s": 12.5,
                                },
                            }
                        ],
                        "errors": [],
                    },
                }
            return httpx.Response(201, json={"id": job_id, "status": "queued"})

        return httpx.Response(200, json=jobs[path.rsplit("/", 1)[1]])

    client = DainClient(
        "ctl.local:8000",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        poll_interval_s=0.001,
    )
    return client, submitted


@pytest.mark.asyncio
async def test_each_prompt_is_pinned_to_its_own_node():
    """The whole point: N subtasks on N machines at once. Left unpinned the
    queue's least-busy ranking can put every prompt on the same node."""
    client, submitted = fake_ctl([profile("a"), profile("b"), profile("c")])
    answers = await ask_pool(client, ["one", "two", "three"])
    await client.aclose()

    assert [a.node_id for a in answers] == ["a", "b", "c"]
    assert sorted(job["node_id"] for job in submitted) == ["a", "b", "c"]
    # A prompt is a whole job, never a shard: /infer reads payload.prompt and
    # 422s on the {"tasks": [...]} shape the queue's splitter would produce.
    assert all(job["fanout"] == 1 for job in submitted)
    assert all("prompt" in job["payload"] for job in submitted)


@pytest.mark.asyncio
async def test_answers_keep_the_prompt_they_belong_to():
    client, _ = fake_ctl([profile("a"), profile("b")])
    answers = await ask_pool(client, ["alpha", "beta"])
    await client.aclose()

    assert [a.prompt for a in answers] == ["alpha", "beta"]
    assert "alpha" in answers[0].text
    assert answers[0].tok_s == 12.5


@pytest.mark.asyncio
async def test_more_prompts_than_nodes_wrap_around():
    client, submitted = fake_ctl([profile("a"), profile("b")])
    answers = await ask_pool(client, ["1", "2", "3"])
    await client.aclose()

    assert [a.node_id for a in answers] == ["a", "b", "a"]
    assert len(submitted) == 3


@pytest.mark.asyncio
async def test_one_node_failing_does_not_lose_the_others():
    """Nodes without DAIN_INFER_MODEL 503. That is the normal state of a
    partly-configured pool, not a reason to lose three good answers."""
    client, _ = fake_ctl([profile("a"), profile("b"), profile("c")], fail_on={"b"})
    answers = await ask_pool(client, ["1", "2", "3"])
    await client.aclose()

    assert [a.ok for a in answers] == [True, False, True]
    assert "no inference backend" in answers[1].error


@pytest.mark.asyncio
async def test_offline_and_joining_nodes_are_not_given_work():
    client, submitted = fake_ctl(
        [profile("a"), profile("b", state="offline"), profile("c", state="joining")]
    )
    answers = await ask_pool(client, ["only one"])
    await client.aclose()

    assert [a.node_id for a in answers] == ["a"]
    assert len(submitted) == 1


@pytest.mark.asyncio
async def test_an_empty_pool_is_an_error_per_prompt_not_a_crash():
    client, _ = fake_ctl([profile("a", state="offline")])
    answers = await ask_pool(client, ["hello"])
    await client.aclose()

    assert len(answers) == 1
    assert not answers[0].ok
    assert "no nodes" in answers[0].error.lower()


@pytest.mark.asyncio
async def test_no_prompts_does_no_work():
    client, submitted = fake_ctl([profile("a")])
    assert await ask_pool(client, []) == ()
    await client.aclose()

    assert submitted == []
