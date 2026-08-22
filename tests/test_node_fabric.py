"""NODE-1: what the node knows about its own machine.

Self-profiling, the address it reports at join, and the Linux rpc-server it
supervises. The regressions guarded here are the Windows ones the agent was
first written against — `rpc-server.exe`, `%PROCESSOR_IDENTIFIER%`, and a
hardcoded fabric address — plus the rule that rpc-server, which has no
authentication at all, never binds a wildcard.
"""

from __future__ import annotations

import socket
from pathlib import Path

import pytest

from contracts import NodeMetrics
from node import dain_node
from tests.node_doubles import (
    FABRIC_IP,
    NODE_ID,
    RPC_PATH,
    FakeAddr,
    FakeMemory,
    FakeProcess,
)

CTL_HOST = "192.168.50.20"


# --------------------------------------------------------------------------
# Self profiling
# --------------------------------------------------------------------------


def test_build_local_profile_uses_the_detected_fabric_ip_as_host():
    # Act
    profile = dain_node.build_local_profile(NODE_ID, FABRIC_IP)

    # Assert
    assert profile.host == FABRIC_IP
    assert profile.id == NODE_ID
    assert profile.state == "joining"


def test_build_local_profile_reports_measured_memory_and_cores(monkeypatch):
    # Arrange
    monkeypatch.setattr(
        dain_node.psutil,
        "virtual_memory",
        lambda: FakeMemory(total=8192 * dain_node.BYTES_PER_MIB, available=0),
    )
    monkeypatch.setattr(dain_node.psutil, "cpu_count", lambda logical=True: 8)

    # Act
    profile = dain_node.build_local_profile(NODE_ID, FABRIC_IP)

    # Assert
    assert profile.ram_total_mb == 8192
    assert profile.cores == 8
    # The measured fields stay zero until SCH-1's calibration probe fills them.
    assert profile.mem_bandwidth_gbs == 0.0
    assert profile.tg_tok_s == 0.0
    assert profile.pp_tok_s == 0.0


def test_read_cpu_model_uses_the_linux_cpuinfo_model_name(tmp_path):
    # Arrange
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text(
        "processor\t: 0\n"
        "vendor_id\t: GenuineIntel\n"
        "model name\t: Intel(R) Core(TM) i7-6700 CPU @ 3.40GHz\n"
        "cpu MHz\t\t: 3408.000\n"
    )

    # Act
    model = dain_node.read_cpu_model(cpuinfo)

    # Assert
    assert model == "Intel(R) Core(TM) i7-6700 CPU @ 3.40GHz"


def test_read_cpu_model_falls_back_when_cpuinfo_is_unreadable(tmp_path):
    # Act — no PROCESSOR_IDENTIFIER lookup: that env var is Windows-only.
    model = dain_node.read_cpu_model(tmp_path / "absent")

    # Assert
    assert model
    assert model != "CPU"


def test_sample_metrics_matches_the_node_metrics_contract():
    # Act
    sample = dain_node.sample_metrics(NODE_ID)

    # Assert
    assert isinstance(sample, NodeMetrics)
    assert sample.node_id == NODE_ID
    assert sample.jobs_running == 0


# --------------------------------------------------------------------------
# Fabric address detection
# --------------------------------------------------------------------------


def test_detect_fabric_ip_prefers_the_named_interface(monkeypatch):
    # Arrange
    monkeypatch.setenv(dain_node.FABRIC_IFACE_ENV, "eth1")
    monkeypatch.setattr(
        dain_node.psutil,
        "net_if_addrs",
        lambda: {
            "wlan0": [FakeAddr(socket.AF_INET, "10.0.0.5")],
            "eth1": [FakeAddr(socket.AF_INET, FABRIC_IP)],
        },
    )
    monkeypatch.setattr(dain_node, "route_source_ip", lambda host: "10.0.0.5")

    # Act / Assert — never the WiFi address when the fabric NIC is named.
    assert dain_node.detect_fabric_ip(CTL_HOST) == FABRIC_IP


def test_detect_fabric_ip_uses_the_route_to_the_control_plane(monkeypatch):
    # Arrange
    monkeypatch.delenv(dain_node.FABRIC_IFACE_ENV, raising=False)
    monkeypatch.setattr(dain_node, "route_source_ip", lambda host: FABRIC_IP)

    # Act / Assert
    assert dain_node.detect_fabric_ip(CTL_HOST) == FABRIC_IP


def test_detect_fabric_ip_falls_back_to_the_route_when_the_interface_has_no_address(
    monkeypatch,
):
    # Arrange
    monkeypatch.setenv(dain_node.FABRIC_IFACE_ENV, "eth9")
    monkeypatch.setattr(dain_node.psutil, "net_if_addrs", dict)
    monkeypatch.setattr(dain_node, "route_source_ip", lambda host: FABRIC_IP)

    # Act / Assert
    assert dain_node.detect_fabric_ip(CTL_HOST) == FABRIC_IP


def test_detect_fabric_ip_returns_loopback_when_the_fabric_is_down(monkeypatch):
    # Arrange
    monkeypatch.delenv(dain_node.FABRIC_IFACE_ENV, raising=False)
    monkeypatch.setattr(dain_node, "route_source_ip", lambda host: None)

    # Act / Assert — loopback, never a wildcard.
    assert dain_node.detect_fabric_ip(CTL_HOST) == dain_node.LOOPBACK


def test_interface_ipv4_ignores_non_ipv4_addresses(monkeypatch):
    # Arrange
    monkeypatch.setattr(
        dain_node.psutil,
        "net_if_addrs",
        lambda: {
            "eth0": [
                FakeAddr(socket.AF_INET6, "fe80::1"),
                FakeAddr(socket.AF_INET, FABRIC_IP),
            ]
        },
    )

    # Act / Assert
    assert dain_node.interface_ipv4("eth0") == FABRIC_IP


def test_interface_ipv4_returns_none_for_an_unknown_interface(monkeypatch):
    # Arrange
    monkeypatch.setattr(dain_node.psutil, "net_if_addrs", dict)

    # Act / Assert
    assert dain_node.interface_ipv4("eth0") is None


def test_route_source_ip_returns_the_address_the_kernel_would_send_from():
    # Act — loopback needs no network, and no packet is sent either way.
    assert dain_node.route_source_ip("127.0.0.1") == "127.0.0.1"


def test_route_source_ip_returns_none_when_the_host_cannot_be_resolved(monkeypatch):
    # Arrange
    def unresolvable(*args, **kwargs):
        raise OSError("name or service not known")

    monkeypatch.setattr(dain_node.socket, "socket", unresolvable)

    # Act / Assert
    assert dain_node.route_source_ip("ctl.invalid") is None


# --------------------------------------------------------------------------
# Linux rpc-server path
# --------------------------------------------------------------------------


def test_the_rpc_binary_is_the_linux_one_from_cluster_toml():
    # Assert — the regression this guards is "rpc-server.exe".
    assert dain_node.RPC_BINARY_NAME == "rpc-server"
    assert dain_node.DEFAULT_LLAMA_BIN_DIR == "/opt/dain/llama.cpp/build/bin"


def test_resolve_rpc_binary_prefers_the_configured_llama_bin_dir(monkeypatch, tmp_path):
    # Arrange
    binary = tmp_path / dain_node.RPC_BINARY_NAME
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    monkeypatch.setenv(dain_node.LLAMA_BIN_DIR_ENV, str(tmp_path))

    # Act / Assert
    assert dain_node.resolve_rpc_binary() == binary


def test_resolve_rpc_binary_skips_a_file_that_is_not_executable(monkeypatch, tmp_path):
    # Arrange
    stub = tmp_path / dain_node.RPC_BINARY_NAME
    stub.write_text("not a build artefact")
    stub.chmod(0o644)
    monkeypatch.setenv(dain_node.LLAMA_BIN_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(dain_node.shutil, "which", lambda name: None)

    # Act / Assert
    assert dain_node.resolve_rpc_binary() is None


def test_resolve_rpc_binary_falls_back_to_path(monkeypatch, tmp_path):
    # Arrange
    monkeypatch.setenv(dain_node.LLAMA_BIN_DIR_ENV, str(tmp_path / "absent"))
    monkeypatch.setattr(
        dain_node.shutil, "which", lambda name: f"/usr/local/bin/{name}"
    )

    # Act / Assert
    assert dain_node.resolve_rpc_binary() == Path("/usr/local/bin/rpc-server")


def test_resolve_rpc_binary_returns_none_when_the_node_has_no_build(
    monkeypatch, tmp_path
):
    # Arrange
    monkeypatch.setenv(dain_node.LLAMA_BIN_DIR_ENV, str(tmp_path / "absent"))
    monkeypatch.setattr(dain_node.shutil, "which", lambda name: None)

    # Act / Assert
    assert dain_node.resolve_rpc_binary() is None


def test_start_rpc_server_binds_the_detected_fabric_ip(monkeypatch):
    # Arrange
    monkeypatch.setattr(dain_node, "resolve_rpc_binary", lambda: Path(RPC_PATH))
    started: list[list[str]] = []
    monkeypatch.setattr(
        dain_node.subprocess, "Popen", lambda argv, **kwargs: started.append(argv)
    )

    # Act
    dain_node.start_rpc_server(FABRIC_IP, port=50052)

    # Assert
    assert started[0] == [RPC_PATH, "--host", FABRIC_IP, "-p", "50052", "-c"]


@pytest.mark.parametrize("wildcard", ["0.0.0.0", "", "::"])
def test_start_rpc_server_refuses_a_wildcard_bind(wildcard):
    # Act / Assert — rpc-server has no authentication whatsoever.
    with pytest.raises(ValueError, match="no authentication"):
        dain_node.start_rpc_server(wildcard)


def test_start_rpc_server_returns_none_when_there_is_no_binary(monkeypatch):
    # Arrange
    monkeypatch.setattr(dain_node, "resolve_rpc_binary", lambda: None)

    # Act / Assert — a node with no build still joins; it just serves nothing.
    assert dain_node.start_rpc_server(FABRIC_IP) is None


def test_stop_rpc_server_terminates_a_running_child():
    # Arrange
    proc = FakeProcess()

    # Act
    dain_node.stop_rpc_server(proc)

    # Assert
    assert proc.terminated is True
    assert proc.killed is False


def test_stop_rpc_server_kills_a_child_that_ignores_terminate():
    # Arrange
    proc = FakeProcess(stubborn=True)

    # Act
    dain_node.stop_rpc_server(proc)

    # Assert
    assert proc.killed is True


def test_stop_rpc_server_ignores_a_child_that_already_exited():
    # Arrange
    proc = FakeProcess()
    proc.returncode = 0

    # Act
    dain_node.stop_rpc_server(proc)

    # Assert
    assert proc.terminated is False


def test_stop_rpc_server_ignores_a_node_that_never_started_one():
    # Act / Assert — must not raise.
    dain_node.stop_rpc_server(None)
