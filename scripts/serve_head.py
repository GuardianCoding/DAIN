#!/usr/bin/env python3
"""Start the pipeline head — the one llama-server the whole cluster shares.

    ./scripts/serve_head.py --model castoff                 # start it
    ./scripts/serve_head.py --model castoff --dry-run       # print the argv
    ./scripts/serve_head.py --model castoff --watch         # restart on membership change

WHERE THIS BELONGS, AND WHY NOT ELSEWHERE:

  not infer/launch.py   that module is pure by contract — it returns argv and
                        runs nothing, which is what makes it testable and what
                        lets sched/ and infer/ stay unaware of each other.
  not ctl/              a control plane that supervises a multi-gigabyte model
                        process is no longer a control plane, and ctl has to
                        stay restartable without killing inference.
  not node/dain_node.py the node agent knows itself and ctl. It cannot build
                        `--rpc` because it does not know the other members.

That leaves an operator script on the head, which is also where the handover
already told people to paste this by hand. Doing it by hand is the risk: the
`--rpc` order defines what `--tensor-split` means positionally, so a
hand-edited command silently puts layers on the wrong machines.

WHAT IT DOES. Asks ctl who is alive, orders the head first and the workers
deterministically, and hands both to infer.launch.llama_server_command. The
entry point is then the head's :8080, llama-server's OpenAI-compatible API —
the job queue is NOT involved in this topology. Point the node agents at it
with DAIN_LLAMA_ENDPOINT if you want /infer jobs to reach it too.

Workers need nothing but rpc-server, which the node agent already starts on
join. Only the head needs the model file: it streams each worker its slice at
load, and `rpc-server -c` caches it there.
"""

from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import time
import tomllib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from infer.launch import (
    CLUSTER_PATH,
    Cluster,
    Member,
    llama_server_command,
    load_cluster,
)

DEFAULT_HEAD = "gpu-01"
DEFAULT_OS_CLASS = "linux-headless"
POLL_INTERVAL_S = 2.0
STOP_TIMEOUT_S = 20.0
HTTP_TIMEOUT_S = 5.0


class HeadError(RuntimeError):
    """Something the operator needs to fix before this can start."""


# --------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------


def os_class_map(path: Path = CLUSTER_PATH) -> dict[str, str]:
    """node_id -> os_class from [[planning.nodes]].

    Member requires os_class, but llama_server_command never reads it — it is
    carried for sched's budget_from_profile, which raises on an unknown value.
    Reading it from config is safe here precisely because it describes the
    machine, not where the machine is; addresses still come from the registry.
    """
    if not path.is_file():
        return {}

    with path.open("rb") as handle:
        raw = tomllib.load(handle)

    nodes = raw.get("planning", {}).get("nodes", [])
    return {
        str(node["id"]): str(node.get("os_class", DEFAULT_OS_CLASS))
        for node in nodes
        if isinstance(node, dict) and "id" in node
    }


def build_members(
    nodes: list[dict[str, Any]],
    head_id: str,
    os_classes: dict[str, str],
    excluded: frozenset[str] = frozenset(),
) -> tuple[Member, ...]:
    """Live membership, head first, workers in a deterministic order.

    Worker ORDER is load-bearing: rpc_endpoints() emits `--rpc` in this order
    and `--tensor-split` is positional over that list. Sorting by node_id means
    two runs a minute apart produce the same split for the same set of nodes,
    which is the only way an A/B comparison means anything.
    """
    if head_id in excluded:
        raise HeadError(f"head {head_id!r} cannot also be excluded")

    alive = [
        node
        for node in nodes
        if node.get("state") != "offline" and str(node.get("id")) not in excluded
    ]
    if not alive:
        raise HeadError("no live nodes — is any node agent running and joined?")

    by_id = {str(node["id"]): node for node in alive}
    if head_id not in by_id:
        raise HeadError(
            f"head {head_id!r} is not in live membership "
            f"({', '.join(sorted(by_id))}). Pass --head with one of those."
        )

    def to_member(node: dict[str, Any], *, is_head: bool) -> Member:
        node_id = str(node["id"])
        return Member(
            node_id=node_id,
            host=str(node["host"]),
            os_class=os_classes.get(node_id, DEFAULT_OS_CLASS),
            backend=str(node.get("backend", "cpu")),
            is_head=is_head,
        )

    workers = sorted(node_id for node_id in by_id if node_id != head_id)
    return (
        to_member(by_id[head_id], is_head=True),
        *(to_member(by_id[node_id], is_head=False) for node_id in workers),
    )


def membership_key(members: tuple[Member, ...]) -> tuple[tuple[str, str], ...]:
    """What --watch compares. Identity and address only: a node's free memory
    moving is not a reason to drop every KV cache in the cluster."""
    return tuple((member.node_id, member.host) for member in members)


def excluded_node_ids(value: str) -> frozenset[str]:
    """Parse the service-friendly comma-separated exclusion list."""
    return frozenset(node_id.strip() for node_id in value.split(",") if node_id.strip())


def resolve_model_file(cluster: Cluster, model_id: str, filename: str | None) -> str:
    """Locate the GGUF for a model KEY from infer/models.toml.

    The key, not the role: fetch_models.py downloads into <models>/<model_id>/
    and Cluster.model_file resolves <models>/<model_id>/<file>, so the
    identifier IS the directory name. `castoff`, not `castoff_capacity`.
    """
    if filename:
        return cluster.model_file(model_id, filename)

    models_root = Path(cluster.path_for("models"))
    directory = models_root / model_id
    if not directory.is_dir():
        available = (
            sorted(child.name for child in models_root.glob("*") if child.is_dir())
            if models_root.is_dir()
            else []
        )
        raise HeadError(
            f"no model directory {directory}. Pass the KEY from "
            f"infer/models.toml, not the role"
            + (f" — found: {', '.join(available)}" if available else "")
        )

    candidates = sorted(directory.glob("*.gguf"))
    if not candidates:
        raise HeadError(f"no .gguf under {directory} — has fetch_models.py run?")
    if len(candidates) > 1:
        names = ", ".join(path.name for path in candidates)
        raise HeadError(f"{directory} holds several GGUFs ({names}); pass --file")
    return str(candidates[0])


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------


def fetch_nodes(ctl: str) -> list[dict[str, Any]]:
    import httpx

    try:
        response = httpx.get(f"http://{ctl}/api/nodes", timeout=HTTP_TIMEOUT_S)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as exc:
        raise HeadError(f"cannot reach the control plane at {ctl}: {exc}") from exc


def fetch_placement(ctl: str, model_id: str):
    """Sean's split from GET /api/plan, or None with a reason printed.

    503 is the normal answer until SCH-1 lands, because every node still
    reports tg_tok_s = 0.0. Falling back to --fit is correct behaviour, not a
    failure: it is also the baseline half of the placement A/B.
    """
    import httpx

    try:
        response = httpx.get(
            f"http://{ctl}/api/plan",
            params={"model": model_id},
            timeout=HTTP_TIMEOUT_S,
        )
    except httpx.HTTPError as exc:
        print(f"note: /api/plan unreachable ({exc}); using --fit on", file=sys.stderr)
        return None

    if response.status_code != 200:
        detail = response.text.strip()[:200]
        print(
            f"note: /api/plan returned {response.status_code} ({detail}); "
            f"using --fit on",
            file=sys.stderr,
        )
        return None

    data = response.json()
    return SimpleNamespace(
        model_id=data["model_id"],
        layers={key: tuple(value) for key, value in data["layers"].items()},
        n_cpu_moe=data.get("n_cpu_moe", {}),
        tensor_split=data["tensor_split"],
    )


def run_head(command: list[str]) -> subprocess.Popen[bytes]:
    print(" ".join(command), file=sys.stderr)
    return subprocess.Popen(command)


def stop_head(proc: subprocess.Popen[bytes] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=STOP_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        proc.kill()


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="serve_head",
        description="Start the shared llama-server head across live membership.",
    )
    parser.add_argument(
        "--model", required=True, help="model KEY from infer/models.toml"
    )
    parser.add_argument(
        "--file", default=None, help="GGUF filename (default: the only one there)"
    )
    parser.add_argument(
        "--head", default=DEFAULT_HEAD, help=f"head node id (default: {DEFAULT_HEAD})"
    )
    parser.add_argument(
        "--ctl", default=None, help="ctl host:port (default: <head>:<ctl_port>)"
    )
    parser.add_argument(
        "--context", type=int, default=8192, help="context length (default: 8192)"
    )
    parser.add_argument(
        "--slots", type=int, default=1, help="concurrent sessions (default: 1)"
    )
    parser.add_argument(
        "--placement",
        action="store_true",
        help="use GET /api/plan instead of llama.cpp's --fit on",
    )
    parser.add_argument(
        "--exclude",
        default="",
        metavar="NODE_ID,...",
        help="omit these node ids from the RPC pool (comma-separated)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the command and exit"
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="restart when membership changes (llama.cpp fixes --rpc at start)",
    )
    return parser.parse_args(argv)


def plan_command(
    args: argparse.Namespace, cluster: Cluster
) -> tuple[list[str], tuple[Member, ...]]:
    ctl = args.ctl or f"{args.head}:{cluster.ctl_port}"
    excluded = excluded_node_ids(args.exclude)
    members = build_members(fetch_nodes(ctl), args.head, os_class_map(), excluded)
    model_file = resolve_model_file(cluster, args.model, args.file)
    placement = fetch_placement(ctl, args.model) if args.placement else None

    command = llama_server_command(
        cluster,
        model_file,
        members,
        placement,
        context=args.context,
        slots=args.slots,
    )
    return command, members


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        cluster = load_cluster()
        command, members = plan_command(args, cluster)
    except (HeadError, ValueError, FileNotFoundError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    head, workers = members[0], members[1:]
    print(
        f"head {head.node_id} ({head.host}:{cluster.llama_port}) "
        f"+ {len(workers)} worker(s): "
        f"{', '.join(m.node_id for m in workers) or 'none'}",
        file=sys.stderr,
    )

    if args.dry_run:
        print(" ".join(command))
        return 0

    if cluster.pinned_commit == "UNVERIFIED":
        # Not fatal, but llama.cpp's RPC protocol has no version negotiation:
        # nodes built from different commits connect happily and then hang or
        # return noise. Worth one line of warning before a demo.
        print(
            "warning: cluster.toml pinned_commit is UNVERIFIED — mismatched "
            "llama.cpp builds hang over RPC rather than erroring",
            file=sys.stderr,
        )

    proc = run_head(command)
    current = membership_key(members)
    stopping = False

    def handle_signal(_signum, _frame):
        nonlocal stopping
        stopping = True
        stop_head(proc)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    if not args.watch:
        return proc.wait()

    print(
        f"watching membership every {POLL_INTERVAL_S:.0f}s; "
        f"debounce {cluster.replan_debounce_s:.0f}s",
        file=sys.stderr,
    )
    changed_at: float | None = None

    while not stopping:
        if proc.poll() is not None:
            print(f"llama-server exited {proc.returncode}", file=sys.stderr)
            return proc.returncode

        time.sleep(POLL_INTERVAL_S)

        try:
            ctl = args.ctl or f"{args.head}:{cluster.ctl_port}"
            latest = build_members(
                fetch_nodes(ctl),
                args.head,
                os_class_map(),
                excluded_node_ids(args.exclude),
            )
        except HeadError as exc:
            print(f"membership check failed: {exc}", file=sys.stderr)
            continue

        if membership_key(latest) == current:
            changed_at = None
            continue

        # Debounce: a half-seated cable flaps, and restarting the head on every
        # flap thrashes a multi-gigabyte model load for the whole demo.
        if changed_at is None:
            changed_at = time.monotonic()
            print("membership changed; waiting for it to settle", file=sys.stderr)
            continue
        if time.monotonic() - changed_at < cluster.replan_debounce_s:
            continue

        print("membership settled; restarting head", file=sys.stderr)
        stop_head(proc)
        try:
            command, members = plan_command(args, cluster)
        except (HeadError, ValueError, KeyError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        proc = run_head(command)
        current = membership_key(members)
        changed_at = None

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
