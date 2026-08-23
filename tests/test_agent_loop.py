import httpx
import httpx2
import pytest
from openai import AsyncOpenAI

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


def dain_client() -> DainClient:
    async def handler(request: httpx.Request) -> httpx.Response:
        body = NODES if request.url.path == "/api/nodes" else METRICS
        return httpx.Response(200, json=body)

    return DainClient(
        "ctl.local:8000",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def completion(message: dict, finish_reason: str = "stop") -> dict:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": "local",
        "choices": [{"index": 0, "finish_reason": finish_reason, "message": message}],
    }


def tool_call(call_id: str, name: str, arguments: str) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def agent_for(responses: list[dict], **kwargs) -> tuple[Agent, list]:
    """An Agent whose head replies with `responses` in order."""
    sent: list = []
    remaining = list(responses)

    async def handler(request: httpx2.Request) -> httpx2.Response:
        sent.append(request)
        return httpx2.Response(200, json=remaining.pop(0))

    llm = AsyncOpenAI(
        base_url="http://head.local:8080/v1",
        api_key="none",
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    return Agent(dain_client(), llm=llm, **kwargs), sent


@pytest.mark.asyncio
async def test_a_question_needing_no_tool_returns_the_models_text():
    agent, sent = agent_for(
        [completion({"role": "assistant", "content": "Five machines."})]
    )
    reply = await agent.ask("how many machines?")
    await agent.aclose()

    assert reply.text == "Five machines."
    assert reply.tool_calls == ()
    assert len(sent) == 1


@pytest.mark.asyncio
async def test_a_tool_call_runs_and_its_result_comes_back_to_the_model():
    agent, _sent = agent_for(
        [
            completion(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tool_call("c1", "cluster_status", "{}")],
                },
                finish_reason="tool_calls",
            ),
            completion({"role": "assistant", "content": "gpu-01, with 50.0GiB free."}),
        ]
    )
    reply = await agent.ask("which machine has the most free RAM?")
    await agent.aclose()

    assert reply.text == "gpu-01, with 50.0GiB free."
    assert [call.name for call in reply.tool_calls] == ["cluster_status"]
    assert "gpu-01" in reply.tool_calls[0].result
    # The tool result must reach the model as a tool-role message, or the second
    # turn is the model answering from nothing.
    tool_messages = [m for m in reply.messages if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert "gpu-01" in tool_messages[0]["content"]


@pytest.mark.asyncio
async def test_several_tool_calls_in_one_turn_all_run():
    agent, _sent = agent_for(
        [
            completion(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        tool_call("c1", "cluster_status", "{}"),
                        tool_call("c2", "plan_placement", '{"model": "castoff"}'),
                    ],
                },
                finish_reason="tool_calls",
            ),
            completion({"role": "assistant", "content": "Done."}),
        ]
    )
    reply = await agent.ask("status and plan please")
    await agent.aclose()

    assert [call.name for call in reply.tool_calls] == [
        "cluster_status",
        "plan_placement",
    ]
    assert len([m for m in reply.messages if m.get("role") == "tool"]) == 2


@pytest.mark.asyncio
async def test_malformed_tool_arguments_are_handed_back_as_a_result():
    """Small models emit broken JSON. That has to be a correctable tool result,
    not a JSONDecodeError that ends the conversation."""
    agent, _sent = agent_for(
        [
            completion(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tool_call("c1", "plan_placement", "{model: cast")],
                },
                finish_reason="tool_calls",
            ),
            completion({"role": "assistant", "content": "Sorry, retrying."}),
        ]
    )
    reply = await agent.ask("plan castoff")
    await agent.aclose()

    assert "not valid JSON" in reply.tool_calls[0].result


@pytest.mark.asyncio
async def test_an_identical_repeated_call_is_answered_from_the_first_result():
    """A 20B model will call the same tool five times. Re-dispatching is a
    wasted round trip to the cluster and teaches it nothing new."""
    agent, _sent = agent_for(
        [
            completion(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tool_call("c1", "cluster_status", "{}")],
                },
                finish_reason="tool_calls",
            ),
            completion(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tool_call("c2", "cluster_status", "{}")],
                },
                finish_reason="tool_calls",
            ),
            completion({"role": "assistant", "content": "gpu-01."}),
        ]
    )
    reply = await agent.ask("which machine?")
    await agent.aclose()

    assert len(reply.tool_calls) == 2
    assert "already called" in reply.tool_calls[1].result
    assert "gpu-01" in reply.tool_calls[1].result


@pytest.mark.asyncio
async def test_the_turn_cap_ends_with_an_honest_admission():
    looping = completion(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [tool_call("c1", "cluster_status", "{}")],
        },
        finish_reason="tool_calls",
    )
    agent, _sent = agent_for([looping] * 3, max_turns=3)
    reply = await agent.ask("go in circles")
    await agent.aclose()

    assert reply.hit_turn_cap
    assert reply.turns == 3
    assert "could not" in reply.text.lower()


def test_an_explicit_endpoint_overrides_the_environment_rather_than_colliding(
    monkeypatch,
):
    """--endpoint and $DAIN_AGENT_ENDPOINT both feed the same parameter, so
    forwarding one alongside the other is a TypeError at startup."""
    monkeypatch.setenv("DAIN_AGENT_ENDPOINT", "from-env:8080")
    monkeypatch.setenv("DAIN_AGENT_MODEL", "from-env-model")

    agent = Agent.from_environment(dain_client(), endpoint="from-flag:8080")

    assert agent.endpoint == "from-flag:8080"
    assert agent.model == "from-env-model"


def test_environment_defaults_apply_when_no_flag_is_given(monkeypatch):
    monkeypatch.setenv("DAIN_AGENT_ENDPOINT", "from-env:8080")

    agent = Agent.from_environment(dain_client(), max_turns=2)

    assert agent.endpoint == "from-env:8080"
    assert agent.max_turns == 2


@pytest.mark.asyncio
async def test_an_unreachable_head_says_where_it_looked():
    async def handler(_request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("connection refused")

    llm = AsyncOpenAI(
        base_url="http://head.local:8080/v1",
        api_key="none",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    agent = Agent(dain_client(), llm=llm, endpoint="head.local:8080")

    with pytest.raises(RuntimeError, match="head.local:8080"):
        await agent.ask("hello")
    await agent.aclose()
