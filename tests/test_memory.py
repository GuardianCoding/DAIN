"""Tests for infer/memory.py — the arithmetic behind the capacity claim.

The claim is the one thing on stage a judge can falsify from the audience, so
the maths behind it gets tested rather than eyeballed.

Two sources of node budgets, and the distinction matters:
  * budget_from_profile — RUNTIME. What a live node reported at join.
  * planning_budgets    — OFFLINE. The [[planning.nodes]] fixture, used only to
                          answer "which models are worth downloading?" before
                          any node exists. Carries no addresses.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from infer.memory import (
    CLUSTER_PATH,
    DEFAULT_OVERHEAD_MIB,
    GPT_OSS_120B_KV,
    OS_RESERVE_MIB,
    VRAM_RESERVE_MIB,
    Footprint,
    KVGeometry,
    NodeBudget,
    budget_from_profile,
    capacity_report,
    fits,
    fits_pooled,
    footprint,
    kv_cache_mib,
    planning_budgets,
)
from infer.models import LADDER_PATH, ModelSpec, by_role, load_ladder

# No addresses anywhere: nodes are discovered at runtime, and this fixture is
# only ever used for offline capacity planning.
CLUSTER_TOML = """
[discovery]
mdns_service = "_dain._tcp.local."
rpc_port = 50052
ctl_port = 8000
node_port = 9100
llama_port = 8080

[membership]
replan_debounce_s = 5
min_workers = 0

[paths.linux]
models = "/srv/dain/models"
llama = "/opt/dain/llama.cpp/build/bin"

[llama]
pinned_commit = "abc123"

[[planning.nodes]]
id = "big"
ram_mb = 65536
vram_mb = 16384
backend = "cuda"
os_class = "linux-desktop"
verified = true

[[planning.nodes]]
id = "small"
ram_mb = 4096
vram_mb = 0
backend = "cpu"
os_class = "linux-headless"
verified = true
"""


@dataclass
class FakeProfile:
    """Shaped like contracts.NodeProfile — what a live node reports at join."""

    id: str
    ram_total_mb: int
    ram_free_mb: int
    vram_total_mb: int
    backend: str = "cpu"


def write_cluster(tmp_path, body: str = CLUSTER_TOML):
    path = tmp_path / "cluster.toml"
    path.write_text(body, encoding="utf-8")
    return path


def make_spec(size_gb: float, model_id: str = "test") -> ModelSpec:
    return ModelSpec(
        model_id=model_id,
        role="headline",
        repo="org/repo",
        include="*.gguf",
        size_gb=size_gb,
        params_total_b=117.0,
        params_active_b=5.1,
        priority=1,
        claim="",
        notes="",
    )


def budget(node_id: str, ram_mib: int, vram_mib: int = 0, verified: bool = True) -> NodeBudget:
    return NodeBudget(
        node_id=node_id,
        ram_usable_mib=ram_mib,
        vram_usable_mib=vram_mib,
        os_class="linux-headless",
        backend="cpu",
        verified=verified,
    )


class TestLiveProfileBudgets:
    """The runtime path: budgets come from what a live node reported."""

    def test_prefers_measured_free_memory_over_the_reserve_table(self):
        # Arrange — a real reading beats a table of guesses
        profile = FakeProfile("nuc-01", ram_total_mb=4096, ram_free_mb=3100, vram_total_mb=0)

        # Act
        result = budget_from_profile(profile, "linux-headless")

        # Assert
        assert result.ram_usable_mib == 3100
        assert result.verified

    def test_falls_back_to_the_reserve_table_when_nothing_was_measured(self):
        # Arrange
        profile = FakeProfile("nuc-01", ram_total_mb=4096, ram_free_mb=0, vram_total_mb=0)

        # Act
        result = budget_from_profile(profile, "linux-headless")

        # Assert
        assert result.ram_usable_mib == 4096 - OS_RESERVE_MIB["linux-headless"]
        assert not result.verified

    def test_subtracts_the_vram_reserve_when_a_gpu_is_present(self):
        # Arrange
        profile = FakeProfile("gpu-01", ram_total_mb=65536, ram_free_mb=62000, vram_total_mb=16384)

        # Act
        result = budget_from_profile(profile, "linux-desktop")

        # Assert
        assert result.vram_usable_mib == 16384 - VRAM_RESERVE_MIB

    def test_leaves_vram_at_zero_for_a_cpu_node(self):
        # Arrange
        profile = FakeProfile("office-01", ram_total_mb=8192, ram_free_mb=7400, vram_total_mb=0)

        # Act / Assert
        assert budget_from_profile(profile, "linux-headless").vram_usable_mib == 0

    def test_carries_the_backend_through(self):
        profile = FakeProfile("office-01", 8192, 7400, 0, backend="cpu")
        assert budget_from_profile(profile, "linux-headless").backend == "cpu"

    def test_a_wsl_node_is_budgeted_from_the_vm_allocation_not_the_host(self):
        # Inside WSL, /proc/meminfo reports the VM's share, so the Windows
        # host's cut is already gone before OS_RESERVE_MIB is consulted. A
        # host-sized reserve here would double-count it and shrink gpu-02 twice.
        profile = FakeProfile("gpu-02", ram_total_mb=11264, ram_free_mb=0, vram_total_mb=0)

        result = budget_from_profile(profile, "linux-wsl")

        assert result.ram_usable_mib == 11264 - OS_RESERVE_MIB["linux-wsl"]

    def test_a_wsl_node_claims_no_vram(self):
        # WSL2 exposes the GPU as /dev/dxg, which llama.cpp's Vulkan backend
        # cannot use. If this ever reports VRAM, someone put a real driver in
        # front of it and the planning fixture needs revisiting.
        profile = FakeProfile("gpu-02", ram_total_mb=11264, ram_free_mb=10700, vram_total_mb=0)
        assert budget_from_profile(profile, "linux-wsl").vram_usable_mib == 0

    def test_rejects_a_windows_os_class(self):
        # Windows support was removed when gpu-02 moved to WSL2. A profile still
        # claiming it is a node nobody reconfigured — fail, do not guess.
        profile = FakeProfile("gpu-02", 16384, 11800, 8192)
        with pytest.raises(ValueError, match="os_class"):
            budget_from_profile(profile, "windows-desktop")

    def test_rejects_unknown_os_class(self):
        profile = FakeProfile("x", ram_total_mb=8192, ram_free_mb=7000, vram_total_mb=0)
        with pytest.raises(ValueError, match="os_class"):
            budget_from_profile(profile, "plan9")

    def test_rejects_a_node_reporting_no_memory(self):
        profile = FakeProfile("x", ram_total_mb=0, ram_free_mb=0, vram_total_mb=0)
        with pytest.raises(ValueError, match="ram_total_mb"):
            budget_from_profile(profile, "linux-headless")


class TestPlanningBudgets:
    """The offline path: 'what should I download?' before any node exists."""

    def test_subtracts_os_reserve_from_ram(self, tmp_path):
        # Arrange
        path = write_cluster(tmp_path)

        # Act
        budgets = {b.node_id: b for b in planning_budgets(path)}

        # Assert
        assert budgets["small"].ram_usable_mib == 4096 - OS_RESERVE_MIB["linux-headless"]

    def test_subtracts_vram_reserve_only_when_a_gpu_is_present(self, tmp_path):
        # Arrange
        path = write_cluster(tmp_path)

        # Act
        budgets = {b.node_id: b for b in planning_budgets(path)}

        # Assert
        assert budgets["big"].vram_usable_mib == 16384 - VRAM_RESERVE_MIB
        assert budgets["small"].vram_usable_mib == 0

    def test_reads_every_planning_entry(self, tmp_path):
        assert {b.node_id for b in planning_budgets(write_cluster(tmp_path))} == {"big", "small"}

    def test_rejects_an_entry_with_no_id(self, tmp_path):
        body = CLUSTER_TOML.replace('id = "small"\n', "")
        with pytest.raises(ValueError, match="needs an id"):
            planning_budgets(write_cluster(tmp_path, body))

    def test_rejects_unknown_os_class(self, tmp_path):
        body = CLUSTER_TOML.replace('os_class = "linux-headless"', 'os_class = "plan9"')
        with pytest.raises(ValueError, match="os_class"):
            planning_budgets(write_cluster(tmp_path, body))

    def test_rejects_missing_ram(self, tmp_path):
        body = CLUSTER_TOML.replace("ram_mb = 4096", "ram_mb = 0")
        with pytest.raises(ValueError, match="ram_mb"):
            planning_budgets(write_cluster(tmp_path, body))

    def test_rejects_a_file_with_no_planning_section(self, tmp_path):
        body = "[discovery]\nrpc_port = 1\n"
        with pytest.raises(ValueError, match=r"planning\.nodes"):
            planning_budgets(write_cluster(tmp_path, body))

    def test_raises_when_file_absent(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            planning_budgets(tmp_path / "nope.toml")


class TestKVCache:
    def test_scales_linearly_with_context(self):
        # Arrange
        geometry = KVGeometry(layers=36, kv_heads=8, head_dim=64)

        # Act
        small = kv_cache_mib(geometry, 1024)
        large = kv_cache_mib(geometry, 2048)

        # Assert
        assert large == pytest.approx(small * 2)

    def test_scales_linearly_with_slots(self):
        # Arrange
        geometry = KVGeometry(layers=36, kv_heads=8, head_dim=64)

        # Act
        one = kv_cache_mib(geometry, 4096, slots=1)
        four = kv_cache_mib(geometry, 4096, slots=4)

        # Assert — concurrency is what pushes the headline model off one node
        assert four == pytest.approx(one * 4)

    def test_sliding_window_layers_do_not_hold_a_scaled_cache(self):
        # Arrange
        full = KVGeometry(layers=36, kv_heads=8, head_dim=64)
        alternating = KVGeometry(layers=36, kv_heads=8, head_dim=64, full_attention_layers=18)

        # Act / Assert
        assert kv_cache_mib(alternating, 8192) == pytest.approx(kv_cache_mib(full, 8192) / 2)

    def test_matches_hand_computed_bytes(self):
        # Arrange — 2 (K+V) * 8 heads * 64 dim * 2 bytes = 2048 B per token per layer
        geometry = KVGeometry(layers=10, kv_heads=8, head_dim=64, dtype_bytes=2)

        # Act
        result = kv_cache_mib(geometry, context_tokens=1024)

        # Assert
        assert result == pytest.approx(2048 * 10 * 1024 / (1024**2))

    def test_quantised_cache_halves_the_size(self):
        # Arrange
        f16 = KVGeometry(layers=36, kv_heads=8, head_dim=64, dtype_bytes=2)
        q8 = KVGeometry(layers=36, kv_heads=8, head_dim=64, dtype_bytes=1)

        # Act / Assert
        assert kv_cache_mib(q8, 8192) == pytest.approx(kv_cache_mib(f16, 8192) / 2)

    @pytest.mark.parametrize("context,slots", [(0, 1), (-1, 1), (1024, 0), (1024, -2)])
    def test_rejects_non_positive_inputs(self, context, slots):
        geometry = KVGeometry(layers=36, kv_heads=8, head_dim=64)
        with pytest.raises(ValueError):
            kv_cache_mib(geometry, context, slots)

    def test_rejects_impossible_full_attention_layer_count(self):
        with pytest.raises(ValueError, match="full_attention_layers"):
            KVGeometry(layers=36, kv_heads=8, head_dim=64, full_attention_layers=40)

    def test_rejects_non_positive_geometry(self):
        with pytest.raises(ValueError, match="positive"):
            KVGeometry(layers=0, kv_heads=8, head_dim=64)

    def test_defaults_to_estimated_source(self):
        # Nothing may back an on-stage claim until this says "measured".
        assert KVGeometry(layers=36, kv_heads=8, head_dim=64).source == "estimated"


class TestFootprintAndFit:
    def test_footprint_sums_weights_kv_and_overhead(self):
        # Arrange
        spec = make_spec(size_gb=10.0)
        geometry = KVGeometry(layers=36, kv_heads=8, head_dim=64)

        # Act
        need = footprint(spec, geometry, context_tokens=4096)

        # Assert
        expected = spec.weights_mib + kv_cache_mib(geometry, 4096) + DEFAULT_OVERHEAD_MIB
        assert need.total_mib == pytest.approx(expected)

    def test_footprint_records_the_configuration_it_was_sized_for(self):
        spec = make_spec(size_gb=10.0)
        need = footprint(spec, GPT_OSS_120B_KV, context_tokens=131072, slots=4)
        assert (need.context_tokens, need.slots) == (131072, 4)

    def test_fits_when_budget_exceeds_requirement(self):
        # Arrange
        need = Footprint("m", weights_mib=1000, kv_mib=100, overhead_mib=100, context_tokens=1, slots=1)

        # Act
        result = fits(budget("n", ram_mib=2000), need)

        # Assert
        assert result.fits
        assert result.headroom_mib == pytest.approx(800)

    def test_does_not_fit_when_budget_is_short(self):
        # Arrange
        need = Footprint("m", weights_mib=1000, kv_mib=100, overhead_mib=100, context_tokens=1, slots=1)

        # Act
        result = fits(budget("n", ram_mib=1000), need)

        # Assert
        assert not result.fits
        assert result.headroom_mib < 0

    def test_pooling_sums_every_node(self):
        # Arrange
        need = Footprint("m", weights_mib=2500, kv_mib=0, overhead_mib=0, context_tokens=1, slots=1)
        budgets = (budget("a", 1000), budget("b", 1000), budget("c", 1000))

        # Act
        result = fits_pooled(budgets, need)

        # Assert
        assert result.fits
        assert result.available_mib == pytest.approx(3000)

    def test_vram_counts_toward_the_budget(self):
        assert budget("gpu", ram_mib=1000, vram_mib=500).total_usable_mib == 1500


class TestCapacityReport:
    GEOMETRY = KVGeometry(layers=36, kv_heads=8, head_dim=64, source="measured")

    def test_holds_when_no_node_fits_alone_but_pool_does(self):
        # Arrange — 2500 needed, no node has it, three together do
        need = Footprint("m", weights_mib=2500, kv_mib=0, overhead_mib=0, context_tokens=8192, slots=4)
        budgets = (budget("a", 1000), budget("b", 1000), budget("c", 1000))

        # Act
        report = capacity_report(need, budgets, self.GEOMETRY)

        # Assert
        assert "capacity claim HOLDS" in report

    def test_rejects_the_claim_when_one_node_holds_it_alone(self):
        # Arrange — this is the gpu-01 trap the whole module exists to catch
        need = Footprint("m", weights_mib=500, kv_mib=0, overhead_mib=0, context_tokens=8192, slots=1)
        budgets = (budget("big", 5000), budget("small", 100))

        # Act
        report = capacity_report(need, budgets, self.GEOMETRY)

        # Assert
        assert "NOT a capacity claim" in report
        assert "big" in report

    def test_reports_failure_when_the_pool_is_too_small(self):
        # Arrange
        need = Footprint("m", weights_mib=99_000, kv_mib=0, overhead_mib=0, context_tokens=8192, slots=1)
        budgets = (budget("a", 1000), budget("b", 1000))

        # Act
        report = capacity_report(need, budgets, self.GEOMETRY)

        # Assert
        assert "does not fit the cluster" in report

    def test_warns_about_unverified_nodes(self):
        # Arrange
        need = Footprint("m", weights_mib=2500, kv_mib=0, overhead_mib=0, context_tokens=8192, slots=1)
        budgets = (budget("a", 1000), budget("b", 1000, verified=False), budget("c", 1000))

        # Act
        report = capacity_report(need, budgets, self.GEOMETRY)

        # Assert
        assert "unverified node(s): b" in report

    def test_warns_when_kv_geometry_is_only_estimated(self):
        # Arrange
        need = Footprint("m", weights_mib=2500, kv_mib=0, overhead_mib=0, context_tokens=8192, slots=1)
        budgets = (budget("a", 1000), budget("b", 1000), budget("c", 1000))
        estimated = KVGeometry(layers=36, kv_heads=8, head_dim=64, source="estimated")

        # Act
        report = capacity_report(need, budgets, estimated)

        # Assert
        assert "KV geometry is estimated" in report

    def test_lists_every_node_in_the_evidence_table(self):
        need = Footprint("m", weights_mib=2500, kv_mib=0, overhead_mib=0, context_tokens=8192, slots=1)
        report = capacity_report(need, (budget("a", 1000), budget("b", 1000), budget("c", 1000)), self.GEOMETRY)
        assert all(node_id in report for node_id in ("a", "b", "c"))

    def test_rejects_an_empty_cluster(self):
        need = Footprint("m", weights_mib=1, kv_mib=0, overhead_mib=0, context_tokens=1, slots=1)
        with pytest.raises(ValueError, match="no compute nodes"):
            capacity_report(need, (), self.GEOMETRY)


class TestRealClusterDecisions:
    """The shipped planning fixture and ladder, not toys.

    These lock in the two findings the demo plan rests on. They are ESTIMATES
    until scripts/inventory.sh has run on all five nodes — if one of these
    starts failing after real measurements land, the demo plan changed, not the
    test.
    """

    @pytest.fixture(scope="class")
    def budgets(self):
        return planning_budgets(CLUSTER_PATH)

    @pytest.fixture(scope="class")
    def specs(self):
        return load_ladder(LADDER_PATH)

    def test_planning_fixture_carries_no_addresses(self, budgets):
        # Addressing is runtime-only. A host in this file would outlive its node.
        assert CLUSTER_PATH.read_text(encoding="utf-8").count("192.168.") == 0
        assert len(budgets) == 5

    def test_head_node_dominates_the_pool(self, budgets):
        # The fact that reshapes the whole plan: one machine is most of the
        # cluster, so pipeline placement is only interesting where memory binds.
        largest = max(budgets, key=lambda b: b.total_usable_mib)
        pooled = sum(b.total_usable_mib for b in budgets)
        assert largest.node_id == "gpu-01"
        assert largest.total_usable_mib / pooled > 0.6

    def test_headline_weights_alone_are_not_a_capacity_claim(self, specs, budgets):
        # 59 GiB of weights fit gpu-01 with room to spare. Claiming "no machine
        # here holds this" on weights alone is falsifiable from the audience.
        need = footprint(by_role(specs, "headline"), GPT_OSS_120B_KV, context_tokens=131072)
        gpu01 = next(b for b in budgets if b.node_id == "gpu-01")
        assert fits(gpu01, need).fits

    def test_headline_becomes_a_capacity_claim_at_four_concurrent_sessions(self, specs, budgets):
        # Four agent sessions at 128k each add ~18 GiB of KV, which is what
        # finally pushes it off gpu-01. Margin is thin (~1.8 GiB) — measure
        # before saying this out loud.
        need = footprint(by_role(specs, "headline"), GPT_OSS_120B_KV, context_tokens=131072, slots=4)
        gpu01 = next(b for b in budgets if b.node_id == "gpu-01")
        assert not fits(gpu01, need).fits
        assert fits_pooled(budgets, need).fits

    def test_castoff_claim_holds_on_weights_alone(self, specs, budgets):
        # The robust claim, and the reason it leads: no KV geometry, no
        # concurrency trick, ~5 GiB of margin. Just three machines nobody wants.
        castoff = {"office-01", "office-02", "nuc-01"}
        three = tuple(b for b in budgets if b.node_id in castoff)
        weights = by_role(specs, "castoff_capacity").weights_mib

        assert len(three) == 3
        assert max(b.total_usable_mib for b in three) < weights
        assert sum(b.total_usable_mib for b in three) > weights

    def test_replica_fits_the_smallest_node(self, specs, budgets):
        # If this fails, fan-out drops from five nodes to four.
        smallest = min(budgets, key=lambda b: b.total_usable_mib)
        assert by_role(specs, "replica").weights_mib < smallest.total_usable_mib

    def test_nothing_is_verified_yet(self, budgets):
        # Guard against this suite quietly becoming "proof". Delete this test
        # once the inventory has run and verified = true everywhere.
        assert not any(b.verified for b in budgets)
