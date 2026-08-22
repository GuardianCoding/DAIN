"""Tests for infer/launch.py — command construction against LIVE membership.

The bug these exist to prevent: --tensor-split is positional over the --rpc
list, so a plan computed before a node joined or died puts layers on the wrong
machines and llama.cpp reports nothing wrong. In a cluster meant to expand and
contract mid-demo that is the default failure, not an edge case.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from infer.launch import (
    Member,
    llama_bench_command,
    llama_server_command,
    load_cluster,
    rpc_endpoints,
    rpc_worker_command,
    solo_probe_command,
    split_head,
)

CLUSTER_TOML = """
[discovery]
mdns_service = "_dain._tcp.local."
rpc_port = 50052
ctl_port = 8000
node_port = 9100
llama_port = 8080

[membership]
heartbeat_interval_s = 2
replan_debounce_s = 5
min_workers = 0

[paths.linux]
models = "/srv/dain/models"
llama = "/opt/dain/llama.cpp/build/bin"

[paths.windows]
models = "C:/dain/models"
llama = "C:/dain/llama.cpp/build/bin/Release"

[llama]
pinned_commit = "abc123"
"""


@dataclass
class FakePlacement:
    """Shaped like contracts.Assignment."""

    model_id: str = "working"
    layers: dict = field(default_factory=dict)
    n_cpu_moe: dict = field(default_factory=dict)
    tensor_split: list = field(default_factory=list)


@pytest.fixture
def cluster(tmp_path):
    path = tmp_path / "cluster.toml"
    path.write_text(CLUSTER_TOML, encoding="utf-8")
    return load_cluster(path)


# Hosts are supplied at runtime by the registry, never read from config.
HEAD = Member("gpu-01", "10.0.0.1", "linux-desktop", "cuda", is_head=True)
WIN_WORKER = Member("gpu-02", "10.0.0.2", "windows-desktop", "vulkan")
CPU_WORKER = Member("office-01", "10.0.0.3", "linux-headless", "cpu")


class TestMembership:
    def test_splits_head_from_workers(self):
        head, workers = split_head((HEAD, WIN_WORKER, CPU_WORKER))
        assert head.node_id == "gpu-01"
        assert [worker.node_id for worker in workers] == ["gpu-02", "office-01"]

    def test_rejects_membership_with_no_head(self):
        with pytest.raises(ValueError, match="exactly one head"):
            split_head((WIN_WORKER, CPU_WORKER))

    def test_rejects_membership_with_two_heads(self):
        second = Member("gpu-02", "10.0.0.2", "linux-desktop", "vulkan", is_head=True)
        with pytest.raises(ValueError, match="exactly one head"):
            split_head((HEAD, second))

    def test_endpoints_follow_member_order(self, cluster):
        # Arrange / Act — this ORDER is what --tensor-split is positional against
        endpoints = rpc_endpoints(cluster, (WIN_WORKER, CPU_WORKER))

        # Assert
        assert endpoints == "10.0.0.2:50052,10.0.0.3:50052"

    def test_a_head_with_no_workers_is_a_valid_cluster(self, cluster):
        # A single machine must still serve; the pool starts at one and grows.
        assert "--rpc" not in llama_server_command(cluster, "/m/model.gguf", (HEAD,))


class TestPathsAcrossOperatingSystems:
    def test_windows_member_gets_an_exe_and_windows_paths(self, cluster):
        command = llama_bench_command(cluster, WIN_WORKER, "C:/dain/models/w/m.gguf")
        assert command[0] == "C:/dain/llama.cpp/build/bin/Release/llama-bench.exe"

    def test_linux_member_gets_no_suffix_and_posix_paths(self, cluster):
        command = llama_bench_command(cluster, CPU_WORKER, "/srv/dain/models/w/m.gguf")
        assert command[0] == "/opt/dain/llama.cpp/build/bin/llama-bench"


class TestRpcWorkerBinding:
    @pytest.mark.parametrize("address", ["0.0.0.0", "::", ""])
    def test_refuses_to_bind_to_a_wildcard_address(self, cluster, address):
        # rpc-server has no authentication; a wildcard bind hands anyone who can
        # reach the box arbitrary compute on it.
        with pytest.raises(ValueError, match="no authentication"):
            rpc_worker_command(cluster, CPU_WORKER, address)

    def test_binds_to_the_address_resolved_on_the_node(self, cluster):
        command = rpc_worker_command(cluster, CPU_WORKER, "10.0.0.3")
        assert "--host" in command and "10.0.0.3" in command

    def test_enables_the_local_weight_cache(self, cluster):
        # Without -c every re-plan re-streams every layer over the wire.
        assert "-c" in rpc_worker_command(cluster, CPU_WORKER, "10.0.0.3")


class TestServerCommand:
    def test_uses_fit_on_when_no_placement_is_given(self, cluster):
        command = llama_server_command(cluster, "/m/model.gguf", (HEAD, CPU_WORKER))
        assert command[command.index("--fit") + 1] == "on"

    def test_applies_placement_shares_in_member_order(self, cluster):
        # Arrange — placement keyed by node id, deliberately in a different
        # order from membership, to prove ordering is derived and not assumed
        placement = FakePlacement(
            layers={"office-01": (0, 5), "gpu-01": (6, 35)},
            tensor_split=[0.2, 0.8],
            n_cpu_moe={"gpu-01": 24},
        )

        # Act
        command = llama_server_command(cluster, "/m/model.gguf", (HEAD, CPU_WORKER), placement)

        # Assert — head share first, then workers in --rpc order
        assert command[command.index("--tensor-split") + 1] == "0.8000,0.2000"

    def test_passes_n_cpu_moe_for_the_head(self, cluster):
        placement = FakePlacement(
            layers={"gpu-01": (0, 35)}, tensor_split=[1.0], n_cpu_moe={"gpu-01": 24}
        )
        command = llama_server_command(cluster, "/m/model.gguf", (HEAD,), placement)
        assert command[command.index("--n-cpu-moe") + 1] == "24"

    def test_rejects_a_plan_whose_share_count_does_not_match_membership(self, cluster):
        # Arrange — a node died between planning and launching
        placement = FakePlacement(
            layers={"gpu-01": (0, 20), "office-01": (21, 35)}, tensor_split=[0.5, 0.5]
        )

        # Act / Assert
        with pytest.raises(ValueError, match="stale"):
            llama_server_command(cluster, "/m/model.gguf", (HEAD,), placement)

    def test_rejects_a_plan_for_different_nodes(self, cluster):
        # Arrange — same node count, but a node was swapped mid-demo
        placement = FakePlacement(
            layers={"gpu-01": (0, 20), "nuc-01": (21, 35)}, tensor_split=[0.7, 0.3]
        )

        # Act / Assert
        with pytest.raises(ValueError, match="stale"):
            llama_server_command(cluster, "/m/model.gguf", (HEAD, CPU_WORKER), placement)

    def test_carries_context_and_slot_count(self, cluster):
        command = llama_server_command(cluster, "/m/model.gguf", (HEAD,), context=131072, slots=4)
        assert command[command.index("-c") + 1] == "131072"
        assert command[command.index("-np") + 1] == "4"


class TestSoloProbe:
    def test_disables_fit_so_it_fails_loudly(self, cluster):
        # --fit on silently shrinks the model to whatever fits, which destroys
        # the "it fails on this node alone" half of the capacity proof.
        command = solo_probe_command(cluster, CPU_WORKER, "/m/model.gguf")
        assert command[command.index("--fit") + 1] == "off"

    def test_never_contacts_another_node(self, cluster):
        assert "--rpc" not in solo_probe_command(cluster, CPU_WORKER, "/m/model.gguf")
