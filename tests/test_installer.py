from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[1]
INSTALLER = ROOT / "scripts" / "install_node.sh"


def test_installer_has_valid_bash_syntax():
    result = subprocess.run(
        ["bash", "-n", str(INSTALLER)], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr


def test_dry_run_covers_the_complete_install_without_exposing_secret(tmp_path):
    secret = "this-secret-must-not-appear"
    environment = {
        **os.environ,
        "DAIN_POOL_SECRET": secret,
        "DAIN_NODE_ID": "office-01",
        "DAIN_INSTALL_ROOT": str(tmp_path / "opt"),
        "DAIN_CONFIG_DIR": str(tmp_path / "etc"),
        "DAIN_SYSTEMD_DIR": str(tmp_path / "systemd"),
        "DAIN_CACHE_DIR": str(tmp_path / "cache"),
        "DAIN_SCRATCH_ROOT": str(tmp_path / "scratch"),
        "DAIN_INDEX_ROOT": str(tmp_path / "index"),
    }

    result = subprocess.run(
        ["bash", str(INSTALLER), "--dry-run"],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert secret not in result.stdout
    assert "bubblewrap" in result.stdout
    assert "git clone" in result.stdout
    assert "uv sync" in result.stdout
    assert "install protected environment file" in result.stdout
    assert "systemctl enable --now dain-node" in result.stdout
    assert "Leaving network addressing unchanged" in result.stdout
    assert "Leaving firewall policy unchanged" in result.stdout


def test_explicit_network_configuration_is_opt_in_and_rendered(tmp_path):
    environment = {
        **os.environ,
        "DAIN_POOL_SECRET": "test-secret",
        "DAIN_INSTALL_ROOT": str(tmp_path / "opt"),
        "DAIN_CONFIG_DIR": str(tmp_path / "etc"),
        "DAIN_SYSTEMD_DIR": str(tmp_path / "systemd"),
        "DAIN_CACHE_DIR": str(tmp_path / "cache"),
        "DAIN_SCRATCH_ROOT": str(tmp_path / "scratch"),
        "DAIN_INDEX_ROOT": str(tmp_path / "index"),
        "DAIN_FABRIC_IFACE": "enp1s0",
        "DAIN_STATIC_IP_CIDR": "192.168.50.11/24",
        "DAIN_GATEWAY": "192.168.50.1",
        "DAIN_DNS": "192.168.50.1",
        "DAIN_FABRIC_CIDR": "192.168.50.0/24",
        "DAIN_MANAGE_FIREWALL": "1",
    }

    result = subprocess.run(
        ["bash", str(INSTALLER), "--dry-run"],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "nmcli connection" in result.stdout
    assert "192.168.50.11/24" in result.stdout
    assert "ufw allow from 192.168.50.0/24" in result.stdout
    assert "ufw --force enable" in result.stdout


def test_static_address_requires_an_interface(tmp_path):
    environment = {
        **os.environ,
        "DAIN_POOL_SECRET": "test-secret",
        "DAIN_INSTALL_ROOT": str(tmp_path / "opt"),
        "DAIN_STATIC_IP_CIDR": "192.168.50.11/24",
    }
    environment.pop("DAIN_FABRIC_IFACE", None)

    result = subprocess.run(
        ["bash", str(INSTALLER), "--dry-run"],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )

    assert result.returncode == 2
    assert "DAIN_FABRIC_IFACE is required" in result.stderr


def test_installer_does_not_embed_a_pool_secret_or_lan_host():
    source = INSTALLER.read_text()

    assert "MOCK_POOL_SECRET" not in source
    assert "192.168.50.10" not in source
    assert "192.168.50.11" not in source
    assert "DAIN_POOL_SECRET is required" in source
    assert "chmod 0600" in source
