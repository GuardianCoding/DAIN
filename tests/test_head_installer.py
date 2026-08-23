from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[1]
INSTALLER = ROOT / "scripts" / "install_head.sh"


def installer_environment(tmp_path: Path) -> dict[str, str]:
    return {
        **os.environ,
        "DAIN_POOL_SECRET": "head-secret-must-not-appear",
        "DAIN_FABRIC_IFACE": "enp8s0",
        "DAIN_FABRIC_HOST": "192.0.2.1",
        "DAIN_HEAD_NODE_ID": "gpu-01",
        "DAIN_MODEL_ID": "castoff",
        "DAIN_MODEL_FILE": "model.gguf",
        "DAIN_BENCH_MODEL": "/models/calibration.gguf",
        "DAIN_HEAD_EXCLUDE": "mac-01,node-104",
        "DAIN_RUNTIME_USER": "dain-operator",
        "DAIN_RUNTIME_GROUP": "dain-operator",
        "DAIN_APP_DIR": str(tmp_path / "app"),
        "DAIN_CONFIG_DIR": str(tmp_path / "etc"),
        "DAIN_SYSTEMD_DIR": str(tmp_path / "systemd"),
    }


def test_head_installer_has_valid_bash_syntax():
    result = subprocess.run(
        ["bash", "-n", str(INSTALLER)], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr


def test_head_installer_dry_run_covers_all_services_without_exposing_secret(tmp_path):
    environment = installer_environment(tmp_path)
    secret = environment["DAIN_POOL_SECRET"]

    result = subprocess.run(
        ["bash", str(INSTALLER), "--dry-run"],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert secret not in result.stdout
    assert "protected environment file" in result.stdout
    assert "dain-ctl.service" in result.stdout
    assert "dain-node.service" in result.stdout
    assert "dain-head.service" in result.stdout
    assert "systemctl restart dain-ctl.service" in result.stdout
    assert "systemctl restart dain-node.service" in result.stdout
    assert "systemctl restart dain-head.service" in result.stdout


def test_head_installer_requires_the_model_selection(tmp_path):
    environment = installer_environment(tmp_path)
    environment.pop("DAIN_MODEL_FILE")

    result = subprocess.run(
        ["bash", str(INSTALLER), "--dry-run"],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )

    assert result.returncode == 2
    assert "DAIN_MODEL_FILE is required" in result.stderr


def test_units_use_restart_policies_watch_mode_and_exclusions():
    source = INSTALLER.read_text()

    assert "Restart=always" in source
    assert "--watch" in source
    assert "--exclude=\\${DAIN_HEAD_EXCLUDE}" in source
    assert "EnvironmentFile=${ENV_FILE}" in source
    assert "DAIN_POOL_SECRET=mock-only-secret" not in source
