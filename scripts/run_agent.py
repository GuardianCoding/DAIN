#!/usr/bin/env python3
"""Talk to the pool.

    ./scripts/run_agent.py                                  # interactive
    ./scripts/run_agent.py --once "which machine is busiest?"
    ./scripts/run_agent.py --tools                          # what it can do

WHERE THIS BELONGS, AND WHY NOT ELSEWHERE:

  not ctl/          same reasoning that keeps serve_head.py out of it. ctl must
                    stay restartable, and restarting it should not end a
                    conversation half way through a tool call.
  not the dashboard the browser is a viewer. Putting the loop behind Next.js
                    means the demo dies if the laptop's dev server does.

It needs two things running: ctl on :8000 (for the tools) and a llama-server
head on :8080 started WITH --jinja (for the thinking). They are separate
endpoints on purpose — see the module docstring in agent/loop.py.

Every tool call goes through ctl as a job, so it shows up on /feed and the
dashboard draws it live. Run this next to the dashboard and the two halves of
the demo are the same event stream.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agent.client import CTL_ENV, DEFAULT_CTL, DainClient
from agent.loop import (
    DEFAULT_ENDPOINT,
    DEFAULT_MODEL,
    MAX_TURNS,
    Agent,
    AgentReply,
)
from agent.tools import TOOLS

PROMPT = "dain> "
SUMMARY_WIDTH = 100


def print_tools() -> None:
    for tool in TOOLS:
        required = tool.parameters.get("required") or []
        arguments = ", ".join(required) if required else "no arguments"
        print(f"{tool.name}({arguments})")
        print(f"    {tool.description}")
        print()


def render(reply: AgentReply, *, show_work: bool) -> None:
    """Tool calls to stderr, the answer to stdout.

    Split so `--once ... > answer.txt` captures the answer alone while an
    operator watching the terminal still sees which machines were touched.
    """
    if show_work:
        for call in reply.tool_calls:
            print(f"  [tool] {call.name} {call.arguments or ''}", file=sys.stderr)
            first_line = call.result.splitlines()[0] if call.result else ""
            print(f"         -> {first_line[:SUMMARY_WIDTH]}", file=sys.stderr)

    print(reply.text)

    if reply.hit_turn_cap:
        print(f"  (stopped after {reply.turns} turns)", file=sys.stderr)


async def interactive(agent: Agent, *, show_work: bool) -> int:
    print(
        f"Thinking on {agent.endpoint}; tools act through {agent.dain.ctl}. "
        "Ctrl-D to quit.",
        file=sys.stderr,
    )
    history: list[dict[str, Any]] = []

    while True:
        try:
            prompt = input(PROMPT).strip()
        except (EOFError, KeyboardInterrupt):
            print(file=sys.stderr)
            return 0

        if not prompt:
            continue
        if prompt in {"exit", "quit"}:
            return 0

        try:
            reply = await agent.ask(prompt, history=history)
        except RuntimeError as exc:
            # An unreachable head is worth recovering from: start it in another
            # terminal and carry on with the same conversation.
            print(f"error: {exc}", file=sys.stderr)
            continue

        render(reply, show_work=show_work)
        history = list(reply.messages)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_agent",
        description="Ask the DAIN pool questions, with the cluster as its tools.",
    )
    parser.add_argument(
        "--ctl",
        default=None,
        help=f"control plane host:port (default: ${CTL_ENV} or {DEFAULT_CTL})",
    )
    parser.add_argument(
        "--endpoint",
        default=None,
        help=f"llama-server head host:port (default: {DEFAULT_ENDPOINT})",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"model name; llama-server ignores it (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=MAX_TURNS,
        help=f"tool-calling turns before giving up (default: {MAX_TURNS})",
    )
    parser.add_argument("--once", default=None, help="ask one question and exit")
    parser.add_argument(
        "--quiet", action="store_true", help="hide tool calls; print only the answer"
    )
    parser.add_argument("--tools", action="store_true", help="list the tools and exit")
    return parser.parse_args(argv)


async def run(args: argparse.Namespace) -> int:
    dain = DainClient(args.ctl) if args.ctl else DainClient.from_environment()
    overrides: dict[str, Any] = {"max_turns": args.max_turns}
    if args.endpoint:
        overrides["endpoint"] = args.endpoint
    if args.model:
        overrides["model"] = args.model
    agent = Agent.from_environment(dain, **overrides)

    try:
        if args.once:
            try:
                reply = await agent.ask(args.once)
            except RuntimeError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 1
            render(reply, show_work=not args.quiet)
            return 0
        return await interactive(agent, show_work=not args.quiet)
    finally:
        await agent.aclose()
        await dain.aclose()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.tools:
        print_tools()
        return 0
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
