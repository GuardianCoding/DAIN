import asyncio
import json

import httpx
import pytest

from contracts import NodeMetrics, NodeProfile
from ctl.queue import JobQueue
from ctl.registry import NodeRegistry


def make_registry() -> NodeRegistry:
    registry = NodeRegistry()
    profile = NodeProfile(
        id="node-01",
        host="node-01.local",
        cpu="Test CPU",
        cores=4,
        ram_total_mb=8192,
        ram_free_mb=6144,
        gpu=None,
        vram_total_mb=0,
        backend="cpu",
        mem_bandwidth_gbs=30.0,
        tg_tok_s=10.0,
        pp_tok_s=40.0,
        rtt_ms=0.2,
    )
    registry.register(profile, heartbeat_required=False)
    registry.heartbeat(
        "node-01",
        NodeMetrics(
            node_id="node-01",
            timestamp=1.0,
            cpu_percent=10.0,
            ram_free_mb=6144,
            gpu_percent=None,
            vram_free_mb=None,
            jobs_running=0,
        ),
    )
    return registry


@pytest.mark.asyncio
async def test_submit_returns_before_work_finishes():
    registry = make_registry()
    request_started = asyncio.Event()
    allow_completion = asyncio.Event()

    async def handler(_request: httpx.Request) -> httpx.Response:
        request_started.set()
        await allow_completion.wait()
        return httpx.Response(
            200,
            json={"ok": True, "result": {"completed": True}},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    queue = JobQueue(registry, client=client)
    job = await queue.submit(
        "search",
        {"tasks": [{"query": "alpha"}]},
        fanout=1,
    )

    assert job.status == "queued"
    await asyncio.wait_for(request_started.wait(), timeout=1.0)
    assert queue.get(job.id) is not None
    assert queue.get(job.id).status == "running"

    allow_completion.set()
    completed = await queue.wait(job.id, timeout=1.0)
    assert completed.status == "done"
    assert completed.result == {
        "shards": [
            {
                "shard_index": 0,
                "node_id": "node-01",
                "result": {"completed": True},
            }
        ],
        "errors": [],
    }

    await queue.close()
    await client.aclose()


def add_node(
    registry: NodeRegistry,
    node_id: str,
    *,
    jobs_running: int = 0,
    state: str = "idle",
) -> None:
    profile = NodeProfile(
        id=node_id,
        host=f"{node_id}.local",
        cpu="Test CPU",
        cores=4,
        ram_total_mb=8192,
        ram_free_mb=6144,
        gpu=None,
        vram_total_mb=0,
        backend="cpu",
        mem_bandwidth_gbs=30.0,
        tg_tok_s=10.0,
        pp_tok_s=40.0,
        rtt_ms=0.2,
    )
    registry.register(profile, heartbeat_required=False)
    registry.heartbeat(
        node_id,
        NodeMetrics(
            node_id=node_id,
            timestamp=1.0,
            cpu_percent=10.0,
            ram_free_mb=6144,
            gpu_percent=None,
            vram_free_mb=None,
            jobs_running=jobs_running,
        ),
    )
    if state != "idle":
        record = registry.get_record(node_id)
        assert record is not None
        record.profile.state = state


@pytest.mark.asyncio
async def test_fanout_splits_tasks_across_least_busy_nodes_concurrently():
    registry = make_registry()
    add_node(registry, "node-02", jobs_running=2)
    add_node(registry, "node-03", jobs_running=0)
    add_node(registry, "node-04", jobs_running=1)

    active = 0
    maximum_active = 0
    all_started = asyncio.Event()
    received: dict[str, list[int]] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        if active == 3:
            all_started.set()

        await asyncio.wait_for(all_started.wait(), timeout=1.0)
        body = json.loads(request.content)
        received[request.url.host] = body["payload"]["tasks"]
        active -= 1
        return httpx.Response(200, json={"ok": True, "result": request.url.host})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    queue = JobQueue(registry, client=client)

    job = await queue.submit(
        "search",
        {"tasks": [0, 1, 2, 3, 4, 5]},
        fanout=3,
    )
    completed = await queue.wait(job.id, timeout=1.0)
    response = queue.response(job.id)

    assert completed.status == "done"
    assert response is not None
    assert response["assigned_nodes"] == ["node-01", "node-03", "node-04"]
    assert maximum_active == 3
    assert received == {
        "node-01.local": [0, 3],
        "node-03.local": [1, 4],
        "node-04.local": [2, 5],
    }

    await queue.close()
    await client.aclose()


@pytest.mark.asyncio
async def test_pinned_job_uses_only_requested_node():
    registry = make_registry()
    add_node(registry, "node-02")
    requested_hosts: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requested_hosts.append(request.url.host)
        return httpx.Response(200, json={"ok": True, "result": "done"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    queue = JobQueue(registry, client=client)
    job = await queue.submit(
        "exec",
        {"tasks": ["a", "b", "c", "d"]},
        fanout=4,
        node_id="node-02",
    )
    await queue.wait(job.id, timeout=1.0)
    response = queue.response(job.id)

    assert response is not None
    assert response["fanout"] == 1
    assert response["assigned_nodes"] == ["node-02"]
    assert requested_hosts == ["node-02.local"]

    await queue.close()
    await client.aclose()


@pytest.mark.asyncio
async def test_offline_nodes_are_not_selected():
    registry = make_registry()
    record = registry.get_record("node-01")
    assert record is not None
    record.profile.state = "offline"
    add_node(registry, "node-02")

    requested_hosts: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requested_hosts.append(request.url.host)
        return httpx.Response(200, json={"ok": True, "result": "done"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    queue = JobQueue(registry, client=client)
    job = await queue.submit("search", {"query": "alpha"}, fanout=1)
    await queue.wait(job.id, timeout=1.0)

    assert requested_hosts == ["node-02.local"]

    await queue.close()
    await client.aclose()


@pytest.mark.asyncio
async def test_per_node_semaphore_prevents_overlapping_requests():
    registry = make_registry()
    active = 0
    maximum_active = 0

    async def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        try:
            await asyncio.sleep(0.01)
            return httpx.Response(200, json={"ok": True, "result": "done"})
        finally:
            active -= 1

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    queue = JobQueue(registry, client=client, per_node_limit=1)

    first = await queue.submit("exec", {"command": "first"}, node_id="node-01")
    second = await queue.submit("exec", {"command": "second"}, node_id="node-01")
    await asyncio.gather(
        queue.wait(first.id, timeout=1.0),
        queue.wait(second.id, timeout=1.0),
    )

    assert maximum_active == 1
    assert first.status == "done"
    assert second.status == "done"

    await queue.close()
    await client.aclose()


@pytest.mark.asyncio
async def test_failed_node_is_retried_once_then_shard_is_reassigned():
    registry = make_registry()
    add_node(registry, "node-02", jobs_running=1)
    attempts: dict[str, int] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        attempts[host] = attempts.get(host, 0) + 1
        if host == "node-01.local":
            return httpx.Response(503, json={"error": "node unavailable"})
        return httpx.Response(200, json={"ok": True, "result": "recovered"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    queue = JobQueue(registry, client=client)
    job = await queue.submit("search", {"query": "alpha"}, fanout=1)
    completed = await queue.wait(job.id, timeout=1.0)

    assert completed.status == "done"
    assert attempts == {
        "node-01.local": 2,
        "node-02.local": 1,
    }
    assert completed.result is not None
    assert completed.result["shards"][0]["node_id"] == "node-02"
    assert [event.event for event in queue.events if event.event == "reassigned"] == [
        "reassigned"
    ]

    await queue.close()
    await client.aclose()


@pytest.mark.asyncio
async def test_job_fails_when_no_replacement_node_is_available():
    registry = make_registry()
    attempts = 0

    async def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(503, json={"error": "node unavailable"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    queue = JobQueue(registry, client=client)
    job = await queue.submit("search", {"query": "alpha"}, fanout=1)
    completed = await queue.wait(job.id, timeout=1.0)

    assert attempts == 2
    assert completed.status == "failed"
    assert completed.result is not None
    assert completed.result["shards"] == []
    assert len(completed.result["errors"]) == 1

    await queue.close()
    await client.aclose()


@pytest.mark.asyncio
async def test_queue_events_have_increasing_sequences():
    registry = make_registry()

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True, "result": "done"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    queue = JobQueue(registry, client=client)
    job = await queue.submit("search", {"query": "alpha"})
    await queue.wait(job.id, timeout=1.0)

    assert [event.event for event in queue.events] == [
        "queued",
        "started",
        "dispatched",
        "completed",
    ]
    assert [event.sequence for event in queue.events] == [1, 2, 3, 4]
    assert [event.sequence for event in queue.events_after(2)] == [3, 4]

    await queue.close()
    await client.aclose()


@pytest.mark.asyncio
async def test_close_cancels_running_jobs_and_rejects_new_jobs():
    registry = make_registry()
    request_started = asyncio.Event()
    never_complete = asyncio.Event()

    async def handler(_request: httpx.Request) -> httpx.Response:
        request_started.set()
        await never_complete.wait()
        return httpx.Response(200, json={"ok": True, "result": "done"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    queue = JobQueue(registry, client=client)
    job = await queue.submit("search", {"query": "alpha"})
    await asyncio.wait_for(request_started.wait(), timeout=1.0)

    await queue.close()
    await asyncio.sleep(0)

    assert queue.closed is True
    assert job.status == "failed"
    assert job.result == {"shards": [], "errors": [{"error": "cancelled"}]}
    assert any(event.event == "cancelled" for event in queue.events)

    with pytest.raises(RuntimeError, match="closed"):
        await queue.submit("search", {"query": "beta"})

    await client.aclose()


@pytest.mark.asyncio
async def test_twenty_fanout_jobs_keep_four_nodes_busy():
    registry = make_registry()
    add_node(registry, "node-02")
    add_node(registry, "node-03")
    add_node(registry, "node-04")

    active = 0
    maximum_active = 0
    requests_by_host: dict[str, int] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal active, maximum_active
        host = request.url.host
        requests_by_host[host] = requests_by_host.get(host, 0) + 1
        active += 1
        maximum_active = max(maximum_active, active)
        try:
            await asyncio.sleep(0.002)
            return httpx.Response(200, json={"ok": True, "result": "done"})
        finally:
            active -= 1

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    queue = JobQueue(registry, client=client, per_node_limit=1)
    jobs = [
        await queue.submit(
            "search",
            {"tasks": [0, 1, 2, 3]},
            fanout=4,
        )
        for _ in range(20)
    ]
    await asyncio.gather(*(queue.wait(job.id, timeout=2.0) for job in jobs))

    assert all(job.status == "done" for job in jobs)
    assert maximum_active == 4
    assert requests_by_host == {
        "node-01.local": 20,
        "node-02.local": 20,
        "node-03.local": 20,
        "node-04.local": 20,
    }

    await queue.close()
    await client.aclose()
