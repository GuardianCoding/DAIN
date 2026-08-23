"""HTTP in front of the agent loop, so the dashboard can talk to it.

    uv run uvicorn agent.service:app --host 0.0.0.0 --port 8100

WHY ITS OWN PROCESS, AGAIN:

Same reasoning as scripts/serve_head.py and scripts/run_agent.py. ctl has to
stay restartable without ending a conversation, and the browser is a viewer —
putting the loop behind Next.js means the demo dies with the dev server. Three
processes, three failure domains: ctl :8000, the head :8080, this :8100.

STATELESS ON PURPOSE. The conversation lives in the browser: /chat returns the
full message list and the page sends it back as `history` next turn. Restarting
this service mid-demo therefore costs nothing — the next message carries the
whole thread with it. It also means two tabs are two independent conversations
without any session plumbing here.

The `history` a browser returns is not trusted input: it is capped and filtered
before it reaches the model.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agent.client import DainClient
from agent.loop import Agent
from agent.tools import TOOLS

# A browser tab left open all afternoon would otherwise resend an ever-growing
# transcript until the head refuses the request outright. Trimming the oldest
# turns is the honest failure: the conversation loses its beginning rather than
# stopping dead.
MAX_HISTORY_MESSAGES = 40
MAX_PROMPT_CHARS = 8000  # ~2k tokens, matching the dashboard's create-job cap

VALID_ROLES = frozenset({"system", "user", "assistant", "tool"})


class ChatRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=MAX_PROMPT_CHARS)
    history: list[Any] = Field(default_factory=list)


def sanitise_history(history: list[Any]) -> list[dict[str, Any]]:
    """Keep the well-formed tail of whatever the browser sent back.

    Anything that is not a dict with a known role is dropped rather than
    rejected: a single malformed entry should cost that entry, not the
    conversation. The tail is kept because recent turns carry the context the
    next answer depends on.
    """
    kept = [
        message
        for message in history
        if isinstance(message, dict) and message.get("role") in VALID_ROLES
    ]
    return kept[-MAX_HISTORY_MESSAGES:]


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    # Only build from the environment if nothing was injected — tests and
    # embedders call configure() first and must not have it overwritten.
    if getattr(app.state, "agent", None) is None:
        configure(Agent.from_environment(DainClient.from_environment()))
    try:
        yield
    finally:
        agent = getattr(app.state, "agent", None)
        if agent is not None:
            await agent.aclose()
            await agent.dain.aclose()


app = FastAPI(title="DAIN agent", version="0.1.0", lifespan=lifespan)
app.state.agent = None

# The page is served from :3000 and this from :8100, so without CORS every
# request fails in the browser while curl works perfectly. Same posture as
# ctl.main — this is a LAN demo tool, not a public service.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def configure(agent: Agent) -> Agent:
    app.state.agent = agent
    return agent


def current_agent() -> Agent:
    agent = getattr(app.state, "agent", None)
    if agent is None:  # pragma: no cover - lifespan always configures one
        raise HTTPException(status_code=503, detail="agent is not configured")
    return agent


@app.get("/health")
def health() -> dict[str, Any]:
    """Both endpoints, because pointing the page at a service whose head or ctl
    is somewhere else is the commonest way this looks broken."""
    agent = current_agent()
    return {
        "status": "ok",
        "endpoint": agent.endpoint,
        "ctl": agent.dain.ctl,
        "model": agent.model,
        "max_turns": agent.max_turns,
    }


@app.get("/tools")
def tools() -> dict[str, Any]:
    return {
        "tools": [
            {
                "name": tool.name,
                "description": tool.description,
                "required": tool.parameters.get("required", []),
            }
            for tool in TOOLS
        ]
    }


@app.post("/chat")
async def chat(request: ChatRequest) -> dict[str, Any]:
    if not request.prompt.strip():
        raise HTTPException(status_code=422, detail="prompt must not be blank")

    agent = current_agent()

    try:
        reply = await agent.ask(
            request.prompt.strip(),
            history=sanitise_history(request.history),
        )
    except RuntimeError as exc:
        # The head being down or erroring is an expected operational state, not
        # a bug in this service. 503 keeps that distinction, and the message
        # already names the endpoint and the command that starts it.
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "text": reply.text,
        "tool_calls": [asdict(call) for call in reply.tool_calls],
        "turns": reply.turns,
        "hit_turn_cap": reply.hit_turn_cap,
        "messages": list(reply.messages),
    }


def main() -> int:  # pragma: no cover - thin uvicorn wrapper
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("DAIN_AGENT_HOST", "0.0.0.0"),
        port=int(os.getenv("DAIN_AGENT_PORT", "8100")),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
