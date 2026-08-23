"""The tool surface the model is given — the cluster itself, as functions.

This is not a chatbot with a calculator bolted on. Every tool here is backed by
live cluster state, so "which machine has the most free memory right now?" is
answered from the registry rather than guessed.

Two rules shape the whole module.

**Tools return prose, not JSON.** A 20B model reading `ram_free=10.0GiB` is
markedly more reliable than the same model parsing a nested object and picking
the right key. The structured form is one layer down in agent/client.py if
anything ever needs it.

**call_tool never raises.** A tool that 503s is the normal case, not the
exceptional one: nodes calibrate, models load, indexes go cold. The reason
comes back as an ordinary tool result so the model can read it and adapt —
crash the loop instead and a recoverable state becomes a dead conversation.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from agent.client import DainClient, DainError
from agent.fanout import ask_pool as run_pool
from infer.spec import known_models
from node.sandbox import DEFAULT_ALLOWLIST

MIB_PER_GIB = 1024.0
DEFAULT_SEARCH_LIMIT = 5
MAX_FANOUT = 16  # ctl.main.JobRequest caps fanout here
ALLOWED_PROGRAMS = ", ".join(sorted(DEFAULT_ALLOWLIST))


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    run: Callable[[DainClient, dict[str, Any]], Awaitable[str]]

    def schema(self) -> dict[str, Any]:
        """The OpenAI-shaped definition llama-server expects with --jinja."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def _gib(mib: float | None) -> str:
    if not isinstance(mib, (int, float)) or isinstance(mib, bool):
        return "unknown"
    return f"{mib / MIB_PER_GIB:.1f}GiB"


def _node_line(profile: dict[str, Any], metric: dict[str, Any] | None) -> str:
    """One node, one line, every value labelled.

    Labelled key=value beats a column layout here: a small model asked to
    compare two nodes has to align them itself, and headers are one more thing
    to get wrong.
    """
    node_id = profile.get("id", "unknown")
    fields = [
        f"state={profile.get('state', 'unknown')}",
        f"backend={profile.get('backend', 'unknown')}",
        f"cores={profile.get('cores', 'unknown')}",
    ]

    # The profile's ram_free_mb is frozen at join; the metric's is current.
    # They are deliberately not merged upstream because they mean different
    # things, so the choice has to be made explicitly, here, every time.
    ram_free = metric.get("ram_free_mb") if metric else profile.get("ram_free_mb")
    fields.append(f"ram_free={_gib(ram_free)}/{_gib(profile.get('ram_total_mb'))}")

    vram_total = profile.get("vram_total_mb") or 0
    if vram_total:
        vram_free = metric.get("vram_free_mb") if metric else None
        free_text = _gib(vram_free) if vram_free is not None else "unknown"
        fields.append(f"vram_free={free_text}/{_gib(vram_total)}")
    else:
        fields.append("vram=none")

    if metric:
        fields.append(f"cpu={metric.get('cpu_percent', 'unknown')}%")
        gpu_percent = metric.get("gpu_percent")
        if gpu_percent is not None:
            fields.append(f"gpu={gpu_percent}%")

    # tg_tok_s is 0.0 until the node calibrates with llama-bench at start.
    # Printed literally, a model will report that the node generates zero
    # tokens per second, which is a measurement it never made.
    decode = profile.get("tg_tok_s")
    if isinstance(decode, (int, float)) and decode > 0:
        fields.append(f"decode={decode}tok/s")
    else:
        fields.append("decode=uncalibrated")

    if metric:
        fields.append(f"jobs_running={metric.get('jobs_running', 0)}")
    else:
        fields.append("(no live telemetry; figures are from join time)")

    return f"{node_id}: " + " ".join(fields)


async def cluster_status(client: DainClient, _arguments: dict[str, Any]) -> str:
    profiles, metrics = await asyncio.gather(client.nodes(), client.metrics())

    if not profiles:
        return (
            "The pool has no nodes registered. Either no node agent has joined "
            "yet, or ctl was restarted (its registry is in memory)."
        )

    by_node = {
        sample["node_id"]: sample
        for sample in metrics.get("nodes", [])
        if isinstance(sample, dict) and "node_id" in sample
    }

    states: dict[str, int] = {}
    for profile in profiles:
        state = str(profile.get("state", "unknown"))
        states[state] = states.get(state, 0) + 1
    summary = ", ".join(f"{count} {state}" for state, count in sorted(states.items()))

    lines = [
        f"{len(profiles)} node(s) registered: {summary}.",
        "",
        *(_node_line(profile, by_node.get(profile.get("id"))) for profile in profiles),
    ]

    errors = metrics.get("errors") or {}
    if isinstance(errors, dict) and errors:
        lines.append("")
        lines.append("Telemetry scrape errors:")
        lines.extend(f"  {node_id}: {reason}" for node_id, reason in errors.items())

    return "\n".join(lines)


async def plan_placement(client: DainClient, arguments: dict[str, Any]) -> str:
    model = arguments.get("model")
    if not isinstance(model, str) or not model.strip():
        return (
            "plan_placement needs a 'model' argument. Valid model keys are: "
            + ", ".join(known_models())
        )

    try:
        plan = await client.plan(
            model.strip(),
            context=int(arguments.get("context", 8192)),
            slots=int(arguments.get("slots", 1)),
        )
    except (TypeError, ValueError):
        return "plan_placement needs 'context' and 'slots' to be whole numbers."
    except DainError as exc:
        message = str(exc)
        if "404" in message:
            # models.toml keys and roles are two namespaces that disagree on
            # half the ladder, so naming the keys is most of the recovery.
            return f"{message}\nValid model keys are: {', '.join(known_models())}"
        return message

    layers = plan.get("layers") or {}
    placement = ", ".join(
        f"{node_id} layers {bounds[0]}-{bounds[1]}"
        for node_id, bounds in layers.items()
        if isinstance(bounds, (list, tuple)) and len(bounds) == 2
    )
    return (
        f"Plan for {plan.get('model_id', model)}: "
        f"{placement or 'no layers assigned'}.\n"
        f"Predicted throughput {plan.get('predicted_tok_s', 'unknown')} tok/s.\n"
        f"Reasoning: {plan.get('rationale', 'none given')}"
    )


def _job_errors(job: dict[str, Any]) -> list[str]:
    result = job.get("result") or {}
    return [
        str(entry["error"])
        for entry in (result.get("errors") or [])
        if isinstance(entry, dict) and entry.get("error")
    ]


async def search_files(client: DainClient, arguments: dict[str, Any]) -> str:
    query = arguments.get("query")
    if not isinstance(query, str) or not query.strip():
        return "search_files needs a 'query' argument: the text to search for."

    limit = arguments.get("limit", DEFAULT_SEARCH_LIMIT)
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        limit = DEFAULT_SEARCH_LIMIT

    # Fan out to the whole pool. Sized against live membership because ctl
    # refuses a fan-out wider than the nodes it has, and a hardcoded 5 would
    # start failing the moment a machine drops out mid-demo.
    nodes = await client.available_nodes()
    if not nodes:
        return "No nodes are available to search."

    job = await client.submit_and_wait(
        "search",
        {"query": query.strip(), "limit": limit},
        fanout=min(len(nodes), MAX_FANOUT),
    )

    result = job.get("result") or {}
    hits = result.get("hits") or []
    searched = result.get("nodes_searched") or []
    errors = _job_errors(job)

    lines: list[str] = []
    if hits:
        lines.append(f"{len(hits)} hit(s) across {len(searched)} node(s):")
        lines.extend(
            f"  {hit.get('source', 'unknown')} (score {hit.get('score', '?')}): "
            f"{(hit.get('snippet') or '').strip()[:200]}"
            for hit in hits
            if isinstance(hit, dict)
        )
    else:
        lines.append(f"No files matched {query.strip()!r}.")

    # Reported alongside the hits, never instead of them: one cold index must
    # not discard what the other machines actually found.
    if errors:
        lines.append("")
        lines.append("Some nodes could not search:")
        lines.extend(f"  {error}" for error in errors)

    return "\n".join(lines)


async def run_command(client: DainClient, arguments: dict[str, Any]) -> str:
    argv = arguments.get("argv")
    if (
        not isinstance(argv, list)
        or not argv
        or not all(isinstance(value, str) for value in argv)
    ):
        return (
            "run_command needs 'argv': a list of strings, the program first. "
            f"Allowed programs: {ALLOWED_PROGRAMS}."
        )

    # node.sandbox is the real boundary — it re-checks this, runs under
    # bubblewrap, and rejects escaping paths. Checking here as well is purely
    # so a refusal comes back as an answer the model can act on instead of
    # spending a dispatch to learn the same thing.
    program = argv[0]
    if program not in DEFAULT_ALLOWLIST:
        return (
            f"{program!r} is not on the sandbox allowlist, so it will not run. "
            f"Allowed programs: {ALLOWED_PROGRAMS}."
        )

    node = arguments.get("node")
    job = await client.submit_and_wait(
        "exec",
        {"argv": argv},
        node_id=node if isinstance(node, str) and node.strip() else None,
    )

    errors = _job_errors(job)
    if errors:
        return f"The command failed: {errors[0]}"

    shards = (job.get("result") or {}).get("shards") or []
    if not shards or not isinstance(shards[0], dict):
        return "The command returned no output."

    outcome = shards[0].get("result") or {}
    ran_on = shards[0].get("node_id", "an unknown node")
    stdout = (outcome.get("stdout") or "").strip()
    stderr = (outcome.get("stderr") or "").strip()

    lines = [f"Ran on {ran_on}, exit {outcome.get('exit_code', '?')}."]
    if stdout:
        lines.append(stdout)
    if stderr:
        lines.append(f"stderr: {stderr}")
    if outcome.get("timed_out"):
        lines.append("The command hit its timeout and was killed.")
    return "\n".join(lines)


async def ask_pool(client: DainClient, arguments: dict[str, Any]) -> str:
    prompts = arguments.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        return "ask_pool needs 'prompts': a list of strings, one task per entry."
    if not all(isinstance(prompt, str) and prompt.strip() for prompt in prompts):
        return "Every entry in 'prompts' must be a non-empty string."

    answers = await run_pool(client, [prompt.strip() for prompt in prompts])

    lines: list[str] = []
    for answer in answers:
        if answer.ok:
            speed = f" at {answer.tok_s} tok/s" if answer.tok_s else ""
            lines.append(f"[{answer.node_id}{speed}] {answer.text.strip()}")
        else:
            where = answer.node_id or "the pool"
            lines.append(f"[{where}] failed: {answer.error}")
    return "\n\n".join(lines)


TOOLS: tuple[Tool, ...] = (
    Tool(
        name="cluster_status",
        description=(
            "Live state of every machine in the pool: free RAM and VRAM, CPU "
            "and GPU load, measured decode speed, and how many jobs each is "
            "running. Use this for any question about what the cluster has, "
            "what it is doing, or which machine is best for something. The "
            "figures are measured now, not estimated."
        ),
        parameters={"type": "object", "properties": {}, "required": []},
        run=cluster_status,
    ),
    Tool(
        name="plan_placement",
        description=(
            "Ask the scheduler how it would split a model across the pool: "
            "which layers land on which machine, and the predicted throughput. "
            "Use this for 'could we run X' or 'how would X be split' questions. "
            "This plans only; it does not start anything."
        ),
        parameters={
            "type": "object",
            "properties": {
                "model": {
                    "type": "string",
                    "description": "Model key from models.toml.",
                    "enum": list(known_models()),
                },
                "context": {
                    "type": "integer",
                    "description": "Context length in tokens. Default 8192.",
                },
                "slots": {
                    "type": "integer",
                    "description": (
                        "Concurrent sessions to size the KV cache for. Default 1."
                    ),
                },
            },
            "required": ["model"],
        },
        run=plan_placement,
    ),
    Tool(
        name="search_files",
        description=(
            "Search the files on every machine in the pool at once, by meaning "
            "rather than exact wording. Each machine searches its own disk and "
            "the results are merged, so hits are labelled node:path. Use this "
            "to find documents, notes or code anywhere in the pool."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to look for, in plain language.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum hits to return. Default 5.",
                },
            },
            "required": ["query"],
        },
        run=search_files,
    ),
    Tool(
        name="run_command",
        description=(
            "Run a read-only shell command on one machine, sandboxed. Only "
            f"these programs are permitted: {ALLOWED_PROGRAMS}. Nothing that "
            "writes files, opens a network connection, or starts another "
            "program will run. Name a machine with 'node', or omit it and the "
            "least busy one is chosen."
        ),
        parameters={
            "type": "object",
            "properties": {
                "argv": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Command and arguments, program first.",
                },
                "node": {
                    "type": "string",
                    "description": "Node id to run on. Optional.",
                },
            },
            "required": ["argv"],
        },
        run=run_command,
    ),
    Tool(
        name="ask_pool",
        description=(
            "Send several independent prompts to several machines at once, one "
            "prompt per machine, and get every answer back. Use this to split a "
            "job into parts that do not depend on each other. Do not use it for "
            "a single question — answer that yourself."
        ),
        parameters={
            "type": "object",
            "properties": {
                "prompts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "One self-contained task per entry.",
                }
            },
            "required": ["prompts"],
        },
        run=ask_pool,
    ),
)

TOOLS_BY_NAME: dict[str, Tool] = {tool.name: tool for tool in TOOLS}


def tool_schemas() -> list[dict[str, Any]]:
    """What gets sent to llama-server as `tools`."""
    return [tool.schema() for tool in TOOLS]


async def call_tool(client: DainClient, name: str, arguments: dict[str, Any]) -> str:
    """Run one tool call and return something the model can read.

    Never raises. Every failure — unknown tool, bad arguments, a node that is
    still loading a model — comes back as text, because the model recovering
    from it is the whole point. The one thing that must not happen is the
    conversation ending because a machine was busy.
    """
    tool = TOOLS_BY_NAME.get(name)
    if tool is None:
        return f"There is no tool called {name!r}. Available tools: " + ", ".join(
            sorted(TOOLS_BY_NAME)
        )

    try:
        return await tool.run(client, arguments or {})
    except DainError as exc:
        return str(exc)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - a crashed tool must not end the turn
        return f"The {name} tool failed unexpectedly: {type(exc).__name__}: {exc}"
