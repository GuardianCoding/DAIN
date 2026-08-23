"""The conversation loop: prompt -> tool calls -> answer.

Runs in its own process, like scripts/serve_head.py and for the same reason:
ctl has to stay restartable without killing a conversation.

TWO ENDPOINTS, AND THEY ARE NOT THE SAME ONE:

  the head, :8080   where this module thinks. A direct llama-server call, no
                    queue involved. Mixing this up with the one below gives you
                    a loop that queues a job to think about queueing a job.
  ctl, :8000        where the TOOLS act, through DainClient. Every tool call
                    becomes a job, so it appears on ctl's /feed, so the
                    dashboard already draws it. The agent working and the
                    picture of it working are the same event stream.

The head must have been started with --jinja or none of this functions: without
it llama.cpp substitutes a chat template carrying no tool-call grammar and the
model answers with prose describing the call it would like to make.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any

from openai import APIConnectionError, APIStatusError, AsyncOpenAI

from agent.client import DainClient
from agent.tools import call_tool, tool_schemas

ENDPOINT_ENV = "DAIN_AGENT_ENDPOINT"
MODEL_ENV = "DAIN_AGENT_MODEL"

DEFAULT_ENDPOINT = "127.0.0.1:8080"
# llama-server serves whatever GGUF it was launched with and ignores this
# field. It exists because the OpenAI schema requires it.
DEFAULT_MODEL = "local"

# Small models loop: a 20B will call the same tool five times given the chance.
# The cap is what turns that from a hung demo into an honest admission.
MAX_TURNS = 6

GAVE_UP = (
    "I could not determine that. I used all {turns} of my tool-calling turns "
    "without reaching an answer, so rather than guess: try asking for one "
    "specific thing at a time."
)

SYSTEM_PROMPT = """\
You are the operator interface to DAIN: a pool of ordinary computers that act \
as one machine. You have tools that read the live state of that pool.

Rules:
- Never guess a fact about the cluster. Call a tool. Your training data knows \
nothing about these specific machines, and the tools read what is true now.
- Report measured numbers as measured. If a tool says a value is uncalibrated \
or unavailable, say so plainly rather than filling the gap with an estimate.
- If a tool returns an error, tell the user what it said and what would fix it. \
Errors here are usually normal: nodes calibrate, models load, indexes go cold.
- Be brief. Two or three sentences unless the user asks for detail.\
"""


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]
    result: str


@dataclass(frozen=True)
class AgentReply:
    text: str
    tool_calls: tuple[ToolCall, ...] = ()
    turns: int = 0
    hit_turn_cap: bool = False
    # The conversation after this exchange, system prompt excluded, ready to
    # hand back to ask() as `history`.
    messages: tuple[dict[str, Any], ...] = ()


class Agent:
    def __init__(
        self,
        dain: DainClient,
        *,
        endpoint: str = DEFAULT_ENDPOINT,
        model: str = DEFAULT_MODEL,
        max_turns: int = MAX_TURNS,
        llm: AsyncOpenAI | None = None,
    ) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be at least one")

        self.dain = dain
        self.endpoint = endpoint
        self.model = model
        self.max_turns = max_turns
        self.llm = llm or AsyncOpenAI(
            base_url=f"http://{endpoint.rstrip('/')}/v1",
            # llama-server has no authentication. A value is required by the
            # client, never checked by the server.
            api_key="none",
        )
        self.owns_llm = llm is None

    @classmethod
    def from_environment(cls, dain: DainClient, **kwargs: Any) -> Agent:
        """Environment for the defaults, kwargs for the overrides.

        Built as one dict rather than passed positionally so that an explicit
        endpoint= from a --endpoint flag REPLACES the environment's rather than
        colliding with it.
        """
        settings: dict[str, Any] = {
            "endpoint": os.getenv(ENDPOINT_ENV, DEFAULT_ENDPOINT).strip()
            or DEFAULT_ENDPOINT,
            "model": os.getenv(MODEL_ENV, DEFAULT_MODEL).strip() or DEFAULT_MODEL,
        }
        settings.update(kwargs)
        return cls(dain, **settings)

    async def aclose(self) -> None:
        if self.owns_llm:
            await self.llm.close()

    async def ask(
        self,
        prompt: str,
        history: list[dict[str, Any]] | None = None,
    ) -> AgentReply:
        conversation: list[dict[str, Any]] = [
            *(history or []),
            {"role": "user", "content": prompt},
        ]
        performed: list[ToolCall] = []
        # Keyed by (name, canonical arguments) so a model that asks the same
        # question twice gets the same answer without a second trip to the
        # cluster. Scoped to one ask(), because between turns the cluster moves.
        seen: dict[tuple[str, str], str] = {}

        for turn in range(1, self.max_turns + 1):
            message = await self._complete(conversation)
            conversation.append(message)

            calls = message.get("tool_calls") or []
            if not calls:
                return AgentReply(
                    text=message.get("content") or "",
                    tool_calls=tuple(performed),
                    turns=turn,
                    messages=tuple(conversation),
                )

            results = await asyncio.gather(
                *(self._dispatch(call, seen) for call in calls)
            )
            for call, outcome in zip(calls, results, strict=True):
                performed.append(outcome)
                conversation.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id", ""),
                        "content": outcome.result,
                    }
                )

        return AgentReply(
            text=GAVE_UP.format(turns=self.max_turns),
            tool_calls=tuple(performed),
            turns=self.max_turns,
            hit_turn_cap=True,
            messages=tuple(conversation),
        )

    async def _complete(self, conversation: list[dict[str, Any]]) -> dict[str, Any]:
        try:
            response = await self.llm.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, *conversation],
                tools=tool_schemas(),
            )
        except APIConnectionError as exc:
            raise RuntimeError(
                f"no llama-server answering at {self.endpoint}. Start the head "
                f"with ./scripts/serve_head.py --model castoff, or point "
                f"${ENDPOINT_ENV} somewhere else."
            ) from exc
        except APIStatusError as exc:
            raise RuntimeError(
                f"llama-server at {self.endpoint} returned "
                f"{exc.status_code}: {exc.message}"
            ) from exc

        message = response.choices[0].message
        # exclude_none drops the nulls llama.cpp emits for absent fields, which
        # some builds then reject on the next request when handed back.
        return message.model_dump(exclude_none=True)

    async def _dispatch(
        self,
        call: dict[str, Any],
        seen: dict[tuple[str, str], str],
    ) -> ToolCall:
        function = call.get("function") or {}
        name = function.get("name") or "unknown"
        raw = function.get("arguments") or "{}"

        try:
            arguments = json.loads(raw) if raw.strip() else {}
        except (TypeError, ValueError):
            # Correctable by the model on its next turn, so it is a result and
            # not an exception. Quoting the input is what makes it correctable.
            return ToolCall(
                name=name,
                arguments={},
                result=(
                    f"The arguments you sent to {name} are not valid JSON: {raw!r}. "
                    f"Send a JSON object, for example {{}} for no arguments."
                ),
            )

        if not isinstance(arguments, dict):
            return ToolCall(
                name=name,
                arguments={},
                result=f"{name} takes a JSON object of arguments, not {raw!r}.",
            )

        key = (name, json.dumps(arguments, sort_keys=True))
        if key in seen:
            return ToolCall(
                name=name,
                arguments=arguments,
                result=(
                    f"You already called {name} with these arguments this turn. "
                    f"The result has not changed:\n\n{seen[key]}"
                ),
            )

        result = await call_tool(self.dain, name, arguments)
        seen[key] = result
        return ToolCall(name=name, arguments=arguments, result=result)
