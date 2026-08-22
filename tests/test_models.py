"""Tests for infer/models.py and the real ladder in infer/models.toml.

The ladder is data that five people act on at 3am, so its invariants are tests
rather than conventions: unique priorities, a replica that fits the smallest
node, and a download order where stopping early still leaves a working demo.
"""

from __future__ import annotations

import pytest

from infer.models import (
    LADDER_PATH,
    ModelSpec,
    by_role,
    estimate_hours,
    format_ladder,
    load_ladder,
)

MINIMAL_TOML = """
[models.alpha]
role = "calibration"
repo = "org/alpha"
include = "*.gguf"
size_gb = 1.0
priority = 1
total_layers = 24

[models.beta]
role = "working"
repo = "org/beta"
include = "*.gguf"
size_gb = 2.0
params_total_b = 35.0
params_active_b = 3.0
priority = 2
total_layers = 36
claim = "does a thing"
"""


def write_ladder(tmp_path, body: str = MINIMAL_TOML):
    path = tmp_path / "models.toml"
    path.write_text(body, encoding="utf-8")
    return path


class TestLoadLadder:
    def test_returns_specs_sorted_by_priority(self, tmp_path):
        # Arrange
        body = MINIMAL_TOML.replace("priority = 1", "priority = 9")
        path = write_ladder(tmp_path, body)

        # Act
        specs = load_ladder(path)

        # Assert
        assert [spec.model_id for spec in specs] == ["beta", "alpha"]

    def test_optional_fields_default_without_failing(self, tmp_path):
        # Arrange
        path = write_ladder(tmp_path)

        # Act
        alpha = next(spec for spec in load_ladder(path) if spec.model_id == "alpha")

        # Assert
        assert alpha.claim == ""
        assert alpha.params_active_b == 0.0

    def test_rejects_duplicate_priorities(self, tmp_path):
        # Arrange
        path = write_ladder(tmp_path, MINIMAL_TOML.replace("priority = 2", "priority = 1"))

        # Act / Assert — two people both downloading "next" is a lost hour
        with pytest.raises(ValueError, match="duplicate priority"):
            load_ladder(path)

    def test_rejects_missing_required_key(self, tmp_path):
        # Arrange
        path = write_ladder(tmp_path, MINIMAL_TOML.replace('repo = "org/alpha"\n', ""))

        # Act / Assert
        with pytest.raises(ValueError, match="missing required key"):
            load_ladder(path)

    @pytest.mark.parametrize("bad", ["size_gb = 0", "size_gb = -3"])
    def test_rejects_non_positive_size(self, tmp_path, bad):
        path = write_ladder(tmp_path, MINIMAL_TOML.replace("size_gb = 1.0", bad))
        with pytest.raises(ValueError, match="size_gb"):
            load_ladder(path)

    def test_rejects_non_positive_priority(self, tmp_path):
        path = write_ladder(tmp_path, MINIMAL_TOML.replace("priority = 1", "priority = 0"))
        with pytest.raises(ValueError, match="priority"):
            load_ladder(path)

    def test_rejects_a_file_with_no_models(self, tmp_path):
        path = write_ladder(tmp_path, "[meta]\nhost_node = 'gpu-01'\n")
        with pytest.raises(ValueError, match="no \\[models"):
            load_ladder(path)

    def test_raises_when_file_absent(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_ladder(tmp_path / "nope.toml")


class TestModelSpec:
    def test_converts_decimal_gb_to_binary_mib(self):
        # Arrange — 1 GB is NOT 1024 MiB, and confusing them oversizes a claim
        spec = ModelSpec("m", "r", "org/r", "*", 1.0, 1.0, 1.0, 1, 24, "", "")

        # Act / Assert
        assert spec.weights_mib == pytest.approx(1000**3 / 1024**2)

    def test_sparsity_is_total_over_active(self):
        # Arrange
        spec = ModelSpec("m", "r", "org/r", "*", 20.0, 35.0, 3.0, 1, 24, "", "")

        # Act / Assert
        assert spec.sparsity == pytest.approx(35.0 / 3.0)

    def test_sparsity_rejects_unknown_active_params(self):
        spec = ModelSpec("m", "r", "org/r", "*", 20.0, 35.0, 0.0, 1, 24, "", "")
        with pytest.raises(ValueError, match="params_active_b"):
            _ = spec.sparsity


class TestLookupAndFormatting:
    def test_by_role_finds_the_single_match(self, tmp_path):
        specs = load_ladder(write_ladder(tmp_path))
        assert by_role(specs, "working").model_id == "beta"

    def test_by_role_raises_on_unknown_role(self, tmp_path):
        specs = load_ladder(write_ladder(tmp_path))
        with pytest.raises(KeyError, match="no model with role"):
            by_role(specs, "headline")

    def test_by_role_raises_when_ambiguous(self, tmp_path):
        body = MINIMAL_TOML.replace('role = "working"', 'role = "calibration"')
        specs = load_ladder(write_ladder(tmp_path, body))
        with pytest.raises(KeyError, match="ambiguous"):
            by_role(specs, "calibration")

    def test_estimate_hours_scales_inversely_with_link_speed(self):
        assert estimate_hours(10.0, 20.0) == pytest.approx(estimate_hours(10.0, 40.0) * 2)

    def test_estimate_hours_rejects_a_dead_link(self):
        with pytest.raises(ValueError, match="positive"):
            estimate_hours(10.0, 0.0)

    def test_format_ladder_shows_every_model(self, tmp_path):
        specs = load_ladder(write_ladder(tmp_path))
        rendered = format_ladder(specs)
        assert "alpha" in rendered and "beta" in rendered and "TOTAL" in rendered


class TestRealLadderInvariants:
    """The shipped ladder, not a fixture. These guard decisions, not code."""

    @pytest.fixture(scope="class")
    def specs(self):
        return load_ladder(LADDER_PATH)

    def test_every_required_role_is_present(self, specs):
        roles = {spec.role for spec in specs}
        assert {"calibration", "replica", "working", "castoff_capacity", "headline"} <= roles

    def test_replica_fits_the_smallest_node(self, specs):
        # nuc-01 has ~3.2 GiB usable under headless Linux. A replica that does
        # not fit it drops fan-out from five nodes to four.
        assert by_role(specs, "replica").weights_gib < 3.0

    def test_calibration_model_is_small_enough_to_run_everywhere_fast(self, specs):
        assert by_role(specs, "calibration").size_gb < 1.0

    def test_calibration_downloads_first(self, specs):
        # It blocks SCH-1 and therefore the whole scheduler critical path.
        assert specs[0].role == "calibration"

    def test_headline_downloads_after_every_core_model(self, specs):
        # The 63 GB file must never sit in front of the models the demo needs.
        headline = by_role(specs, "headline")
        core = ("calibration", "replica", "working", "castoff_capacity")
        assert all(headline.priority > by_role(specs, role).priority for role in core)

    def test_demo_is_complete_without_the_headline_model(self, specs):
        # If the venue link dies, everything below the headline must still be a
        # working demo, and a sane overnight download.
        headline_priority = by_role(specs, "headline").priority
        without_headline = sum(
            spec.size_gb for spec in specs if spec.priority < headline_priority
        )
        assert without_headline < 40.0

    def test_moe_models_are_genuinely_sparse(self, specs):
        # Decode speed is bandwidth / active bytes. A dense model of the same
        # file size runs ~10x slower on these nodes.
        for role in ("working", "castoff_capacity", "headline"):
            assert by_role(specs, role).sparsity >= 4.0
