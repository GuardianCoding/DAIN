"""scheduler_spec() — the bridge from the model ladder to sched.plan().

The KV numbers below are not invented for the test: they are the figures the
/api/plan audit published, recomputed here from the geometry. If a geometry is
edited carelessly these stop matching, which is the point.
"""

import pytest

from infer.models import load_ladder
from infer.spec import (
    KV_GEOMETRY,
    ModelSpecUnavailable,
    known_models,
    resolve_model_id,
    scheduler_spec,
    unverified_models,
)


class TestResolveModelId:
    def test_a_key_resolves_to_itself(self):
        assert resolve_model_id("castoff") == "castoff"

    def test_a_role_resolves_to_its_key(self):
        # role and key disagree on half the ladder; the key is the directory
        # name the weights sit in, so it is what must reach Assignment.
        assert resolve_model_id("castoff_capacity") == "castoff"
        assert resolve_model_id("embedding") == "embed"
        assert resolve_model_id("speculative") == "mtp"
        assert resolve_model_id("spare_quant") == "working_spare"

    def test_surrounding_whitespace_is_tolerated(self):
        assert resolve_model_id("  castoff ") == "castoff"

    def test_the_name_the_old_test_sent_is_not_valid(self):
        # tests/test_main.py used to send this and expect a 200.
        with pytest.raises(KeyError):
            resolve_model_id("gpt-oss-20b")

    def test_the_name_the_scheduler_mock_used_is_not_valid(self):
        with pytest.raises(KeyError):
            resolve_model_id("qwen3.6-35b-a3b")

    def test_the_error_lists_what_is_valid(self):
        with pytest.raises(KeyError, match="castoff"):
            resolve_model_id("nonsense")

    def test_every_ladder_key_resolves(self):
        for spec in load_ladder():
            assert resolve_model_id(spec.model_id) == spec.model_id


class TestSchedulerSpec:
    def test_returns_exactly_the_four_fields_sched_plan_reads(self):
        spec = scheduler_spec("castoff", 8192, 1)

        assert set(spec) == {
            "model_id",
            "total_layers",
            "file_size_mb",
            "kv_mb_per_layer",
        }

    def test_model_id_is_canonicalised_to_the_key(self):
        assert scheduler_spec("castoff_capacity", 8192, 1)["model_id"] == "castoff"

    def test_file_size_is_binary_mib_not_decimal_gb(self):
        """size_gb x 1000 would say 11500. The scheduler compares against
        ram_free_mb, which is binary MiB from `free -m`; the 4.9% gap is about
        the margin the capacity claim turns on."""
        spec = scheduler_spec("castoff", 8192, 1)

        assert spec["file_size_mb"] == pytest.approx(10967, abs=1)
        assert spec["file_size_mb"] != pytest.approx(11500, abs=1)

    @pytest.mark.parametrize(
        ("model", "context", "slots", "expected"),
        [
            ("castoff", 8_192, 1, 8.0),
            ("castoff", 32_768, 1, 32.0),
            ("headline", 8_192, 1, 8.0),
            ("headline", 131_072, 4, 512.0),
        ],
    )
    def test_kv_per_layer_matches_the_published_figures(
        self, model, context, slots, expected
    ):
        spec = scheduler_spec(model, context, slots)

        assert spec["kv_mb_per_layer"] == pytest.approx(expected, rel=1e-6)

    def test_the_placeholder_is_only_right_at_8k_by_one_session(self):
        """sched/mock.py's 8.0 was correct for 8k x 1 and wrong by 64x at the
        headline model's real settings. That is the whole reason context and
        slots are arguments."""
        small = scheduler_spec("headline", 8_192, 1)["kv_mb_per_layer"]
        real = scheduler_spec("headline", 131_072, 4)["kv_mb_per_layer"]

        assert small == pytest.approx(8.0)
        assert real / small == pytest.approx(64.0)

    def test_cache_scales_with_context_and_with_sessions(self):
        base = scheduler_spec("castoff", 8_192, 1)["kv_mb_per_layer"]

        assert scheduler_spec("castoff", 16_384, 1)["kv_mb_per_layer"] == pytest.approx(
            base * 2
        )
        assert scheduler_spec("castoff", 8_192, 2)["kv_mb_per_layer"] == pytest.approx(
            base * 2
        )

    def test_divides_by_total_layers_not_cached_layers(self):
        """gpt-oss alternates full and sliding-window attention, so only half
        the layers hold a context-scaled cache — but sched.cost multiplies this
        by a node's layer count, so it must be spread across all of them."""
        geometry = KV_GEOMETRY["castoff"]
        assert geometry.cached_layers == 12
        assert geometry.layers == 24

        spec = scheduler_spec("castoff", 8_192, 1)

        # 192 MiB over 24 layers, not over the 12 that actually cache.
        assert spec["kv_mb_per_layer"] == pytest.approx(192 / 24)

    def test_unknown_model_raises_keyerror_for_a_404(self):
        with pytest.raises(KeyError):
            scheduler_spec("not-a-model", 8192, 1)

    def test_a_model_with_no_geometry_refuses_rather_than_guesses(self):
        # Qwen3.6-35B-A3B's block count has not been read off the GGUF. A wrong
        # layer count stays invisible until the machine OOMs.
        with pytest.raises(ModelSpecUnavailable, match="working"):
            scheduler_spec("working", 8192, 1)

    def test_zero_context_or_slots_is_rejected(self):
        with pytest.raises(ValueError):
            scheduler_spec("castoff", 0, 1)
        with pytest.raises(ValueError):
            scheduler_spec("castoff", 8192, 0)


class TestProvenance:
    def test_layer_counts_agree_between_the_ladder_and_the_geometry(self):
        """Two sources for one fact. If they drift, one was hand-edited and
        every split built from it is wrong."""
        by_id = {spec.model_id: spec for spec in load_ladder()}

        for model_id, geometry in KV_GEOMETRY.items():
            assert by_id[model_id].total_layers == geometry.layers, model_id

    def test_headline_geometry_is_the_one_the_repo_already_carried(self):
        assert KV_GEOMETRY["headline"].layers == 36
        assert KV_GEOMETRY["headline"].full_attention_layers == 18

    def test_every_geometry_is_still_an_estimate(self):
        """Fails the day someone marks one measured — deliberately, so the
        list gets re-read. Until llama-server's loader log is captured, no
        derived capacity figure should go on a slide."""
        assert set(unverified_models()) == set(KV_GEOMETRY)

    def test_models_without_geometry_are_the_ones_with_no_layer_count(self):
        by_id = {spec.model_id: spec for spec in load_ladder()}
        missing_geometry = set(known_models()) - set(KV_GEOMETRY)

        assert missing_geometry == {"working", "mtp", "working_spare"}
        for model_id in missing_geometry:
            assert by_id[model_id].total_layers == 0
