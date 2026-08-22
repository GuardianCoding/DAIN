"""The pipeline head launcher — membership ordering and model resolution.

The I/O (httpx, subprocess, signals) is deliberately untested here; what
matters and what silently corrupts a run is the pure part, because `--rpc`
order defines what `--tensor-split` means positionally.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

_spec = importlib.util.spec_from_file_location(
    "serve_head", REPO_ROOT / "scripts" / "serve_head.py"
)
serve_head = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(serve_head)

from infer.launch import Cluster, llama_server_command

HEAD = "gpu-01"
OS_CLASSES = {
    "gpu-01": "linux-desktop",
    "gpu-02": "linux-wsl",
    "office-01": "linux-headless",
    "nuc-01": "linux-headless",
}


def node(node_id: str, host: str, *, state: str = "idle", backend: str = "cpu") -> dict:
    return {"id": node_id, "host": host, "state": state, "backend": backend}


LIVE = [
    node("office-01", "192.168.50.11"),
    node("gpu-01", "192.168.50.10", backend="cuda"),
    node("nuc-01", "192.168.50.14"),
]


def make_cluster(tmp_path: Path) -> Cluster:
    return Cluster(
        paths={
            "models": str(tmp_path / "models"),
            "llama": "/opt/dain/llama.cpp/build/bin",
        },
        rpc_port=50052,
        llama_port=8080,
        ctl_port=8000,
        mdns_service="_dain._tcp.local.",
        pinned_commit="abc1234",
        replan_debounce_s=5.0,
        min_workers=0,
    )


class TestBuildMembers:
    def test_head_comes_first_regardless_of_registry_order(self):
        # ctl returns nodes in registration order; split_head() requires
        # exactly one head, and llama_server_command builds --rpc from the
        # workers that follow it.
        members = serve_head.build_members(LIVE, HEAD, OS_CLASSES)

        assert members[0].node_id == HEAD
        assert members[0].is_head is True
        assert all(not m.is_head for m in members[1:])

    def test_workers_are_ordered_deterministically(self):
        """Two runs over the same nodes must produce the same --rpc order.

        --tensor-split is positional over that list, so an unstable order
        silently moves layers between machines and makes the placement A/B
        meaningless.
        """
        forward = serve_head.build_members(LIVE, HEAD, OS_CLASSES)
        backward = serve_head.build_members(list(reversed(LIVE)), HEAD, OS_CLASSES)

        assert [m.node_id for m in forward] == [m.node_id for m in backward]
        assert [m.node_id for m in forward[1:]] == ["nuc-01", "office-01"]

    def test_offline_nodes_are_excluded(self):
        nodes = [*LIVE, node("gpu-02", "192.168.50.13", state="offline")]

        members = serve_head.build_members(nodes, HEAD, OS_CLASSES)

        assert "gpu-02" not in [m.node_id for m in members]

    def test_addresses_come_from_the_registry(self):
        # No static IPs anywhere: a node is addressed by what it reported at
        # join, never by config.
        members = serve_head.build_members(LIVE, HEAD, OS_CLASSES)

        assert members[0].host == "192.168.50.10"

    def test_os_class_is_carried_through(self):
        members = serve_head.build_members(LIVE, HEAD, OS_CLASSES)

        assert members[0].os_class == "linux-desktop"

    def test_unknown_node_gets_a_valid_os_class(self):
        # budget_from_profile raises on anything outside the three known
        # values, so defaulting to a *valid* one beats propagating None.
        members = serve_head.build_members(LIVE, HEAD, {})

        assert {m.os_class for m in members} == {"linux-headless"}

    def test_empty_membership_is_an_error_not_an_empty_command(self):
        with pytest.raises(serve_head.HeadError, match="no live nodes"):
            serve_head.build_members([], HEAD, OS_CLASSES)

    def test_all_offline_is_the_same_error(self):
        nodes = [node("gpu-01", "192.168.50.10", state="offline")]

        with pytest.raises(serve_head.HeadError, match="no live nodes"):
            serve_head.build_members(nodes, HEAD, OS_CLASSES)

    def test_missing_head_lists_what_is_actually_there(self):
        with pytest.raises(serve_head.HeadError, match="office-01"):
            serve_head.build_members(LIVE, "not-a-node", OS_CLASSES)


class TestMembershipKey:
    def test_same_nodes_and_addresses_compare_equal(self):
        first = serve_head.build_members(LIVE, HEAD, OS_CLASSES)
        second = serve_head.build_members(list(reversed(LIVE)), HEAD, OS_CLASSES)

        assert serve_head.membership_key(first) == serve_head.membership_key(second)

    def test_a_node_leaving_is_a_change(self):
        before = serve_head.build_members(LIVE, HEAD, OS_CLASSES)
        after = serve_head.build_members(LIVE[:2], HEAD, OS_CLASSES)

        assert serve_head.membership_key(before) != serve_head.membership_key(after)

    def test_a_node_changing_address_is_a_change(self):
        before = serve_head.build_members(LIVE, HEAD, OS_CLASSES)
        moved = [node("office-01", "192.168.50.99"), *LIVE[1:]]
        after = serve_head.build_members(moved, HEAD, OS_CLASSES)

        assert serve_head.membership_key(before) != serve_head.membership_key(after)


class TestResolveModelFile:
    def test_explicit_filename_resolves_under_the_model_key(self, tmp_path):
        resolved = serve_head.resolve_model_file(
            make_cluster(tmp_path), "castoff", "a.gguf"
        )

        assert resolved.endswith("/models/castoff/a.gguf")

    def test_a_single_gguf_is_found_without_naming_it(self, tmp_path):
        directory = tmp_path / "models" / "castoff"
        directory.mkdir(parents=True)
        (directory / "gpt-oss-20b-mxfp4.gguf").write_bytes(b"gguf")

        resolved = serve_head.resolve_model_file(make_cluster(tmp_path), "castoff", None)

        assert resolved.endswith("gpt-oss-20b-mxfp4.gguf")

    def test_several_ggufs_demand_an_explicit_choice(self, tmp_path):
        directory = tmp_path / "models" / "working"
        directory.mkdir(parents=True)
        (directory / "a.gguf").write_bytes(b"gguf")
        (directory / "b.gguf").write_bytes(b"gguf")

        with pytest.raises(serve_head.HeadError, match="pass --file"):
            serve_head.resolve_model_file(make_cluster(tmp_path), "working", None)

    def test_a_role_instead_of_a_key_names_the_valid_keys(self, tmp_path):
        """The trap the handover calls out.

        `role` is a second namespace that disagrees with the table key on half
        the ladder — castoff/castoff_capacity, embed/embedding. The identifier
        IS the directory name, so passing the role finds nothing.
        """
        (tmp_path / "models" / "castoff").mkdir(parents=True)

        with pytest.raises(serve_head.HeadError, match="castoff"):
            serve_head.resolve_model_file(
                make_cluster(tmp_path), "castoff_capacity", None
            )

    def test_empty_model_directory_points_at_the_downloader(self, tmp_path):
        (tmp_path / "models" / "castoff").mkdir(parents=True)

        with pytest.raises(serve_head.HeadError, match="fetch_models"):
            serve_head.resolve_model_file(make_cluster(tmp_path), "castoff", None)


class TestCommandIntegration:
    def test_members_feed_llama_server_command_in_rpc_order(self, tmp_path):
        members = serve_head.build_members(LIVE, HEAD, OS_CLASSES)

        command = llama_server_command(
            make_cluster(tmp_path), "/models/castoff/a.gguf", members
        )

        rpc = command[command.index("--rpc") + 1]
        # Worker order from build_members, with the head absent — the head
        # serves its own layers and is never in its own --rpc list.
        assert rpc == "192.168.50.14:50052,192.168.50.11:50052"
        assert "192.168.50.10" not in rpc

    def test_no_placement_means_fit_on(self, tmp_path):
        members = serve_head.build_members(LIVE, HEAD, OS_CLASSES)

        command = llama_server_command(
            make_cluster(tmp_path), "/models/castoff/a.gguf", members
        )

        assert command[command.index("--fit") + 1] == "on"

    def test_head_binds_the_address_it_reported(self, tmp_path):
        members = serve_head.build_members(LIVE, HEAD, OS_CLASSES)

        command = llama_server_command(
            make_cluster(tmp_path), "/models/castoff/a.gguf", members
        )

        assert command[command.index("--host") + 1] == "192.168.50.10"
        assert command[command.index("--port") + 1] == "8080"


class TestArgs:
    def test_model_is_required(self):
        with pytest.raises(SystemExit):
            serve_head.parse_args([])

    def test_defaults_match_the_documented_demo(self):
        args = serve_head.parse_args(["--model", "castoff"])

        assert args.head == "gpu-01"
        assert args.context == 8192
        assert args.slots == 1
        # --fit on unless placement is asked for: that is the baseline half of
        # the A/B, and the only thing that works before SCH-1 lands.
        assert args.placement is False
