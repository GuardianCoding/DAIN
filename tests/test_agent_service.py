import httpx
import httpx2
import pytest
from fastapi.testclient import TestClient
from openai import AsyncOpenAI

from agent import service
from agent.client import DainClient
from agent.loop import Agent

NODES = [
    {
        "id": "gpu-01",
        "host": "10.0.0.1",
        "cpu": "Ryzen 5 9600X",
        "cores": 12,
        "ram_total_mb": 62976,
        "ram_free_mb": 51200,
        "gpu": "RTX 5070 Ti",
        "vram_total_mb": 16384,
        "backend": "cuda",
        "mem_bandwidth_gbs": 80.0,
        "tg_tok_s": 42.1,
        "pp_tok_s": 310.0,
        "rtt_ms": 0.2,
        "state": "idle",
    }
]
METRICS = {"type": "metrics", "nodes": [], "errors": {}}


def completion(message: dict, finish_reason: str = "stop") -> dict:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": "stub",
        "choices": [{"index": 0, "finish_reason": finish_reason, "message": message}],
    }


def client_for(responses: list[dict], *, head_fails: bool = False) -> TestClient:
    remaining = list(responses)

    async def ctl_handler(request: httpx.Request) -> httpx.Response:
        body = NODES if request.url.path == "/api/nodes" else METRICS
        return httpx.Response(200, json=body)

    async def head_handler(_request: httpx2.Request) -> httpx2.Response:
        if head_fails:
            raise httpx2.ConnectError("connection refused")
        return httpx2.Response(200, json=remaining.pop(0))

    dain = DainClient(
        "ctl.local:8000",
        client=httpx.AsyncClient(transport=httpx.MockTransport(ctl_handler)),
    )
    llm = AsyncOpenAI(
        base_url="http://head.local:8080/v1",
        api_key="none",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(head_handler)),
    )
    service.configure(Agent(dain, llm=llm, endpoint="head.local:8080"))
    return TestClient(service.app)


def test_health_reports_where_it_thinks_and_where_it_acts():
    """Both endpoints in one place: the commonest failure is pointing the page
    at a live service whose head or ctl is somewhere else entirely."""
    with client_for([]) as client:
        body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["endpoint"] == "head.local:8080"
    assert body["ctl"] == "ctl.local:8000"


def test_tools_lists_what_the_agent_can_do():
    with client_for([]) as client:
        body = client.get("/tools").json()

    names = [tool["name"] for tool in body["tools"]]
    assert "cluster_status" in names
    assert all(tool["description"] for tool in body["tools"])


def test_chat_returns_the_answer_and_the_tool_calls_behind_it():
    """The dashboard shows the work, not just the answer — a reply with no
    visible tool call is indistinguishable from the model making it up."""
    responses = [
        completion(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "cluster_status", "arguments": "{}"},
                    }
                ],
            },
            finish_reason="tool_calls",
        ),
        completion({"role": "assistant", "content": "gpu-01, 50.0GiB free."}),
    ]
    with client_for(responses) as client:
        response = client.post("/chat", json={"prompt": "which node?"})

    assert response.status_code == 200
    body = response.json()
    assert body["text"] == "gpu-01, 50.0GiB free."
    assert [call["name"] for call in body["tool_calls"]] == ["cluster_status"]
    assert "gpu-01" in body["tool_calls"][0]["result"]
    assert body["turns"] == 2
    assert body["hit_turn_cap"] is False


def test_the_returned_messages_can_be_sent_back_as_history():
    """Conversation state lives in the browser, not here. The service stays
    stateless so restarting it cannot end a conversation mid-demo."""
    with client_for([completion({"role": "assistant", "content": "first"})]) as client:
        first = client.post("/chat", json={"prompt": "hello"}).json()

    with client_for([completion({"role": "assistant", "content": "second"})]) as client:
        second = client.post(
            "/chat", json={"prompt": "and again", "history": first["messages"]}
        ).json()

    assert second["text"] == "second"
    # The prior exchange plus this one: history is carried, not dropped.
    assert len(second["messages"]) > len(first["messages"])


def test_an_unreachable_head_is_a_503_naming_the_endpoint():
    """503 rather than 500: the head being down is an expected operational
    state, and the message is what tells an operator how to fix it."""
    with client_for([], head_fails=True) as client:
        response = client.post("/chat", json={"prompt": "hello"})

    assert response.status_code == 503
    assert "head.local:8080" in response.json()["detail"]


def test_an_empty_prompt_is_rejected_before_the_model_is_called():
    with client_for([]) as client:
        assert client.post("/chat", json={"prompt": "   "}).status_code == 422


def test_history_is_capped_so_a_long_session_cannot_grow_without_bound():
    """A browser tab left open all afternoon would otherwise resend an
    ever-growing transcript until the head refuses the request outright."""
    flood = [{"role": "user", "content": f"turn {n}"} for n in range(500)]

    with client_for([completion({"role": "assistant", "content": "ok"})]) as client:
        body = client.post("/chat", json={"prompt": "next", "history": flood}).json()

    assert len(body["messages"]) <= service.MAX_HISTORY_MESSAGES + 2


def test_malformed_history_entries_are_dropped_not_fatal():
    """History is whatever the browser sent back; it is not trusted input."""
    with client_for([completion({"role": "assistant", "content": "ok"})]) as client:
        response = client.post(
            "/chat",
            json={
                "prompt": "hi",
                "history": ["not an object", {"no_role": True}, 42],
            },
        )

    assert response.status_code == 200


@pytest.mark.parametrize("origin", ["http://localhost:3000", "http://gpu-01:3000"])
def test_the_dashboard_origin_is_allowed(origin: str):
    """The page is served from :3000 and this from :8100, so without CORS every
    request fails in the browser while curl works perfectly."""
    with client_for([]) as client:
        response = client.get("/health", headers={"Origin": origin})

    assert response.headers["access-control-allow-origin"] in {"*", origin}
