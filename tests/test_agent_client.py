import httpx
import pytest

from agent.client import DainClient, DainError, job_deadline_s
from ctl.queue import DEFAULT_TIMEOUTS_S


def make_client(handler) -> DainClient:
    transport = httpx.MockTransport(handler)
    return DainClient("ctl.local:8000", client=httpx.AsyncClient(transport=transport))


@pytest.mark.parametrize("kind", sorted(DEFAULT_TIMEOUTS_S))
def test_polling_deadline_outlasts_the_queue_for_every_kind(kind: str):
    """The third layer of the same invariant.

    node ceiling < queue dispatch timeout < agent polling deadline. Undercut it
    here and the agent reports "timed out" for a job the queue was still going
    to finish, which is indistinguishable to the model from a broken cluster.
    """
    assert job_deadline_s(kind) > DEFAULT_TIMEOUTS_S[kind]


@pytest.mark.parametrize(
    "ctl",
    [
        "ctl.local:8000",
        "http://ctl.local:8000",
        "http://ctl.local:8000/",
        "ctl.local:8000/api",
    ],
)
@pytest.mark.asyncio
async def test_api_base_is_normalised_from_any_reasonable_spelling(ctl: str):
    seen: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json=[])

    client = DainClient(
        ctl, client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    await client.nodes()
    await client.aclose()

    assert seen == ["http://ctl.local:8000/api/nodes"]


@pytest.mark.asyncio
async def test_nodes_returns_the_registry_list():
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[{"id": "gpu-01"}])

    client = make_client(handler)
    assert await client.nodes() == [{"id": "gpu-01"}]
    await client.aclose()


@pytest.mark.asyncio
async def test_ctl_detail_survives_into_the_error_message():
    """The 503 texts were written to be read by a model, so they must not be
    replaced with a generic 'request failed'."""
    detail = "working has no entry in KV_GEOMETRY; read kv_heads off the GGUF header"

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"detail": detail})

    client = make_client(handler)
    with pytest.raises(DainError, match="read kv_heads off the GGUF header"):
        await client.plan("working")
    await client.aclose()


@pytest.mark.asyncio
async def test_a_transport_failure_names_the_control_plane():
    async def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    client = make_client(handler)
    with pytest.raises(DainError, match="ctl.local:8000"):
        await client.nodes()
    await client.aclose()


@pytest.mark.asyncio
async def test_submit_and_wait_polls_until_the_job_leaves_running():
    polls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal polls
        if request.method == "POST":
            return httpx.Response(201, json={"id": "job-1", "status": "queued"})
        polls += 1
        status = "done" if polls >= 3 else "running"
        return httpx.Response(200, json={"id": "job-1", "status": status, "result": {}})

    client = DainClient(
        "ctl.local:8000",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        poll_interval_s=0.001,
    )
    job = await client.submit_and_wait("search", {"query": "alpha"})
    await client.aclose()

    assert job["status"] == "done"
    assert polls == 3


@pytest.mark.asyncio
async def test_a_failed_job_is_returned_not_raised():
    """Tools phrase failures for the model themselves; the transport must not
    decide that a failed job is an exception."""

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(201, json={"id": "job-1", "status": "queued"})
        return httpx.Response(
            200,
            json={
                "id": "job-1",
                "status": "failed",
                "result": {"errors": [{"error": "boom"}]},
            },
        )

    client = DainClient(
        "ctl.local:8000",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        poll_interval_s=0.001,
    )
    job = await client.submit_and_wait("exec", {"argv": ["ls"]})
    await client.aclose()

    assert job["status"] == "failed"


@pytest.mark.asyncio
async def test_giving_up_on_a_slow_job_says_which_job_and_how_long():
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(201, json={"id": "job-1", "status": "queued"})
        return httpx.Response(200, json={"id": "job-1", "status": "running"})

    client = DainClient(
        "ctl.local:8000",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        poll_interval_s=0.001,
    )
    with pytest.raises(DainError, match="job-1"):
        await client.submit_and_wait("infer", {"prompt": "hi"}, timeout_s=0.01)
    await client.aclose()
