"""Build llama.cpp command lines for the DAIN cluster.

Every function here is pure: it returns a list of argv tokens and runs nothing.
These commands get pasted into terminals and read aloud at 3am, so they have to
be inspectable before they are executable.

Every node is Linux x86-64 — gpu-02 through WSL2 — so there is one binary name,
one path layout and no OS branching anywhere in this module. If a Windows node
ever comes back, it does not come back here: it gets its own module.

Addressing is RUNTIME, never configuration. A `Member` is a node the control
plane currently believes is alive — it comes from GET /api/nodes, which is fed
by mDNS join and expired by missed heartbeats. Nothing here reads a configured
IP, because a configured IP survives the node that owned it.

The constraint that shapes this module: llama.cpp fixes its --rpc endpoint list
at llama-server start, and --tensor-split is positional against that list. So
expanding or contracting the pool means re-plan + restart, and a plan computed
against a membership that has since changed is a silently wrong split. That is
what `_assert_plan_matches_membership` exists to prevent.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

CLUSTER_PATH = Path(__file__).resolve().parent.parent / "cluster.toml"


class Placement(Protocol):
    """Structurally identical to contracts.Assignment, imported by neither.

    Sean's scheduler produces one; this module consumes it. Keeping it a
    Protocol means infer/ and sched/ never import each other.
    """

    model_id: str
    layers: dict[str, tuple[int, int]]
    n_cpu_moe: dict[str, int]
    tensor_split: list[float]


@dataclass(frozen=True)
class Member:
    """A node the registry currently believes is alive.

    Built from contracts.NodeProfile at plan time. Never from a config file —
    if it came from config it would outlive the node.
    """

    node_id: str
    host: str
    os_class: str
    backend: str
    is_head: bool = False


@dataclass(frozen=True)
class Cluster:
    """Settings shared by every node. Contains no node identity and no addresses."""

    paths: dict[str, str]
    rpc_port: int
    llama_port: int
    ctl_port: int
    mdns_service: str
    pinned_commit: str
    replan_debounce_s: float
    min_workers: int

    def path_for(self, key: str) -> str:
        """One path per key, because every node is Linux.

        gpu-02 runs under WSL2, which is Linux as far as these paths are
        concerned — provided nothing lives under /mnt/c. See [wsl] in
        cluster.toml.
        """
        try:
            return self.paths[key]
        except KeyError as error:
            raise KeyError(f"cluster.toml has no [paths].{key}") from error

    def binary(self, name: str) -> str:
        return f"{self.path_for('llama')}/{name}"

    def model_file(self, model_id: str, filename: str) -> str:
        return f"{self.path_for('models')}/{model_id}/{filename}"


def load_cluster(path: Path = CLUSTER_PATH) -> Cluster:
    """Parse and validate cluster.toml. Fails loudly rather than half-loading."""
    if not path.is_file():
        raise FileNotFoundError(f"cluster file not found: {path}")

    with path.open("rb") as handle:
        raw = tomllib.load(handle)

    for section in ("discovery", "paths", "llama", "membership"):
        if section not in raw:
            raise ValueError(f"{path} is missing the [{section}] section")

    discovery = raw["discovery"]
    membership = raw["membership"]
    return Cluster(
        paths={str(key): str(value) for key, value in raw["paths"].items()},
        rpc_port=int(discovery["rpc_port"]),
        llama_port=int(discovery["llama_port"]),
        ctl_port=int(discovery["ctl_port"]),
        mdns_service=str(discovery["mdns_service"]),
        pinned_commit=str(raw["llama"].get("pinned_commit", "UNVERIFIED")),
        replan_debounce_s=float(membership.get("replan_debounce_s", 5)),
        min_workers=int(membership.get("min_workers", 0)),
    )


def split_head(members: tuple[Member, ...]) -> tuple[Member, tuple[Member, ...]]:
    """Separate the head from its workers, preserving worker order."""
    heads = [member for member in members if member.is_head]
    if len(heads) != 1:
        raise ValueError(f"expected exactly one head in membership, found {len(heads)}")
    return heads[0], tuple(member for member in members if not member.is_head)


def rpc_endpoints(cluster: Cluster, workers: tuple[Member, ...]) -> str:
    """The --rpc list. Its ORDER defines what --tensor-split means."""
    return ",".join(f"{worker.host}:{cluster.rpc_port}" for worker in workers)


def verify_build_command(cluster: Cluster, member: Member) -> list[str]:
    """INF-1: assert the same COMMIT on every node.

    `member` is unused for the path now that every node is Linux, but stays in
    the signature because the caller iterates members and the RESULT is
    per-node: this command is run over ssh on each one and the outputs compared.
    """
    return [cluster.binary("llama-server"), "--version"]


def rpc_worker_command(cluster: Cluster, member: Member, bind_address: str) -> list[str]:
    """Start an RPC worker, bound to one address.

    `bind_address` is resolved on the node itself (from DAIN_FABRIC_IFACE, or
    the interface with a route to the control plane) — it is never read from
    config. rpc-server has NO authentication whatsoever, so binding this to
    0.0.0.0 or to venue WiFi hands arbitrary compute to anyone who can reach it.

    On gpu-02 that address is the LAN address WSL sees under mirrored
    networking. Under WSL's default NAT it would be a 172.x address on a virtual
    switch nothing else can route to, which is why [wsl] in cluster.toml
    requires mirrored mode.

    `member` no longer selects a path — every node is Linux — but stays in the
    signature so the call site records which node each argv belongs to.
    """
    if bind_address in ("0.0.0.0", "::", ""):
        raise ValueError(
            f"refusing to bind rpc-server to {bind_address!r}: it has no authentication. "
            f"Pass the fabric interface address for this node."
        )
    return [
        cluster.binary("rpc-server"),
        "--host", bind_address,
        "-p", str(cluster.rpc_port),
        "-c",                       # cache weights locally; makes re-plans fast
    ]


def _assert_plan_matches_membership(placement: Placement, members: tuple[Member, ...]) -> None:
    """A plan computed against different membership is a silently wrong split.

    --tensor-split is positional over [head device, *rpc devices]. If a node
    joined or died since the plan was made, every share shifts by one and the
    model loads with layers on the wrong machines. Fail instead.
    """
    expected = len(members)
    if len(placement.tensor_split) != expected:
        raise ValueError(
            f"placement is stale: tensor_split has {len(placement.tensor_split)} shares "
            f"but membership has {expected} node(s). Re-plan against current membership."
        )
    planned = set(placement.layers)
    live = {member.node_id for member in members}
    if planned != live:
        raise ValueError(
            f"placement is stale: planned for {sorted(planned)} but membership is "
            f"{sorted(live)}. Re-plan against current membership."
        )


def llama_server_command(
    cluster: Cluster,
    model_file: str,
    members: tuple[Member, ...],
    placement: Placement | None = None,
    *,
    context: int = 8192,
    slots: int = 1,
) -> list[str]:
    """Start the head server across the currently live members.

    With no placement, `--fit on` lets llama.cpp choose (its default splits in
    proportion to free memory, which hands the most work to the slowest node).
    With a placement, Sean's split applies. Those two commands are the A/B in
    INF-6 benchmark #6.
    """
    head, workers = split_head(members)
    if len(workers) < cluster.min_workers:
        raise ValueError(f"only {len(workers)} worker(s); cluster.min_workers is {cluster.min_workers}")

    command = [
        cluster.binary("llama-server"),
        "-m", model_file,
        "-c", str(context),
        "-np", str(slots),
        "-fa", "on",
        "-ngl", "999",
        # Use the GGUF's own chat template. Recent llama.cpp only parses tool
        # calls out of the model's template, and without --jinja it falls back
        # to a built-in that carries no tool-call grammar: the model then
        # answers with prose describing the call it would make. That failure
        # looks like "this model is too small for tools" rather than a missing
        # argument, which is an expensive afternoon. The agent layer's entire
        # tool surface hangs off this flag.
        "--jinja",
        "--metrics",
        "--host", head.host,
        "--port", str(cluster.llama_port),
    ]
    if workers:
        command += ["--rpc", rpc_endpoints(cluster, workers)]

    if placement is None:
        return command + ["--fit", "on"]

    _assert_plan_matches_membership(placement, members)
    ordered = (head, *workers)
    shares = [placement.tensor_split[list(placement.layers).index(m.node_id)] for m in ordered]
    command += ["--tensor-split", ",".join(f"{share:.4f}" for share in shares)]

    head_moe = placement.n_cpu_moe.get(head.node_id)
    if head_moe is not None:
        command += ["--n-cpu-moe", str(head_moe)]
    return command


def solo_probe_command(cluster: Cluster, member: Member, model_file: str, *, context: int = 8192) -> list[str]:
    """The 'fails on this node alone' half of the capacity proof.

    Run it on every node and let it fail on camera. A capacity claim with no
    demonstrated failure is an assertion; with one it is evidence.
    """
    return [
        cluster.binary("llama-server"),
        "-m", model_file,
        "-c", str(context),
        "-ngl", "999",
        "--fit", "off",            # off, so it fails loudly instead of silently shrinking
        "--host", member.host,
        "--port", str(cluster.llama_port),
    ]


def llama_bench_command(cluster: Cluster, member: Member, model_file: str, *, repetitions: int = 3) -> list[str]:
    """Calibration probe behind SCH-1. Measured throughput, never a spec sheet.

    `member` is kept for the same reason as in rpc_worker_command: the result is
    run on one specific node and the caller needs to keep track of which.
    """
    return [
        cluster.binary("llama-bench"),
        "-m", model_file,
        "-p", "512",
        "-n", "128",
        "-r", str(repetitions),
        "-o", "json",
    ]
