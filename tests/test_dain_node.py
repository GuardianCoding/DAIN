"""NODE-1: the node agent's conversation with the control plane.

Three of the four things that broke when the node agent and the control plane
were written on separate branches live here: the shape of the join payload,
heartbeating to the heartbeat endpoint rather than re-posting join, and
building the profile before the server starts. The fourth — the Linux
rpc-server path and the detected fabric address — is in test_node_fabric.py.
"""

from __future__ import annotations

from dataclasses import asdict, fields
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from contracts import NodeMetrics
from node import dain_node
from tests.node_doubles import (
    CTL,
    FABRIC_IP,
    NODE_ID,
    POOL_SECRET,
    FakeControlPlane,
    FakeProcess,
    StopLoop,
    make_profile,
    never_returns,
)


@pytest.fixture(autouse=True)
def clean_agent_state():
    """No test may leak a configured agent into the next one."""
    yield
    if getattr(dain_node.app.state, "agent", None) is not None:
        del dain_node.app.state.agent


@pytest.fixture
def agent() -> dain_node.NodeAgent:
    return dain_node.NodeAgent(profile=make_profile(), ctl=CTL, pool_secret=POOL_SECRET)


@pytest.fixture
def client(agent: dain_node.NodeAgent) -> TestClient:
    """A client over a configured app. Not a context manager, so no lifespan."""
    dain_node.app.state.agent = agent
    return TestClient(dain_node.app)


# --------------------------------------------------------------------------
# Join payload
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_join_wraps_the_profile_with_the_pool_secret(agent):
    # Arrange
    ctl = FakeControlPlane(httpx.Response(201, json=asdict(agent.profile)))

    # Act
    async with ctl.client() as http:
        joined = await dain_node.join_pool(http, agent)

    # Assert — a bare profile is what the control plane rejects.
    assert joined is True
    assert set(ctl.body().keys()) == {"profile", "pool_secret"}
    assert ctl.body()["pool_secret"] == POOL_SECRET
    assert ctl.body()["profile"] == asdict(make_profile())


@pytest.mark.asyncio
async def test_join_posts_to_the_control_plane_join_endpoint(agent):
    # Arrange
    ctl = FakeControlPlane(httpx.Response(201, json=asdict(agent.profile)))

    # Act
    async with ctl.client() as http:
        await dain_node.join_pool(http, agent)

    # Assert
    assert str(ctl.requests[0].url) == f"http://{CTL}/api/nodes/join"
    assert ctl.requests[0].method == "POST"


@pytest.mark.asyncio
async def test_join_adopts_the_state_the_control_plane_reports(agent):
    # Arrange
    accepted = asdict(make_profile()) | {"state": "idle"}
    ctl = FakeControlPlane(httpx.Response(201, json=accepted))

    # Act
    async with ctl.client() as http:
        await dain_node.join_pool(http, agent)

    # Assert
    assert agent.profile.state == "idle"


@pytest.mark.asyncio
async def test_join_is_refused_when_the_pool_secret_is_wrong(agent):
    # Arrange
    ctl = FakeControlPlane(httpx.Response(403, json={"detail": "invalid pool secret"}))

    # Act
    async with ctl.client() as http:
        joined = await dain_node.join_pool(http, agent)

    # Assert
    assert joined is False
    assert agent.profile.state == "joining"


@pytest.mark.asyncio
async def test_join_fails_on_any_status_other_than_201(agent):
    # Arrange
    ctl = FakeControlPlane(httpx.Response(200, json=asdict(agent.profile)))

    # Act
    async with ctl.client() as http:
        joined = await dain_node.join_pool(http, agent)

    # Assert
    assert joined is False


@pytest.mark.asyncio
async def test_join_survives_an_unreachable_control_plane(agent):
    # Arrange
    ctl = FakeControlPlane(httpx.ConnectError("no route to host"))

    # Act
    async with ctl.client() as http:
        joined = await dain_node.join_pool(http, agent)

    # Assert
    assert joined is False


# --------------------------------------------------------------------------
# Heartbeat
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_heartbeat_posts_to_the_node_specific_heartbeat_endpoint(agent):
    # Arrange
    ctl = FakeControlPlane(httpx.Response(200, json={"node_id": NODE_ID}))

    # Act
    async with ctl.client() as http:
        await dain_node.send_heartbeat(http, agent)

    # Assert — not /api/nodes/join, which is a 201 create.
    assert str(ctl.requests[0].url) == f"http://{CTL}/api/nodes/{NODE_ID}/heartbeat"


@pytest.mark.asyncio
async def test_heartbeat_body_carries_a_node_metrics_sample(agent):
    # Arrange
    ctl = FakeControlPlane(httpx.Response(200, json={"node_id": NODE_ID}))

    # Act
    async with ctl.client() as http:
        await dain_node.send_heartbeat(http, agent)

    # Assert
    metrics = ctl.body()["metrics"]
    assert set(metrics.keys()) == {field.name for field in fields(NodeMetrics)}
    assert metrics["node_id"] == NODE_ID


@pytest.mark.asyncio
async def test_heartbeat_adopts_the_state_the_control_plane_reports(agent):
    # Arrange
    ctl = FakeControlPlane(
        httpx.Response(200, json={"node_id": NODE_ID, "state": "computing"})
    )

    # Act
    async with ctl.client() as http:
        await dain_node.send_heartbeat(http, agent)

    # Assert
    assert agent.profile.state == "computing"


@pytest.mark.asyncio
async def test_heartbeat_asks_for_a_rejoin_when_the_node_is_unknown(agent):
    # Arrange
    ctl = FakeControlPlane(httpx.Response(404, json={"detail": "node not found"}))

    # Act
    async with ctl.client() as http:
        still_registered = await dain_node.send_heartbeat(http, agent)

    # Assert
    assert still_registered is False


@pytest.mark.asyncio
async def test_heartbeat_keeps_beating_through_a_transport_error(agent):
    # Arrange — a missed beat is the control plane's to count, not ours.
    ctl = FakeControlPlane(httpx.ConnectError("cable half out"))

    # Act
    async with ctl.client() as http:
        still_registered = await dain_node.send_heartbeat(http, agent)

    # Assert
    assert still_registered is True


@pytest.mark.asyncio
async def test_heartbeat_loop_joins_once_then_heartbeats(agent):
    # Arrange
    ctl = FakeControlPlane(
        httpx.Response(201, json=asdict(agent.profile)),
        httpx.Response(200, json={"node_id": NODE_ID}),
        httpx.Response(200, json={"node_id": NODE_ID}),
    )

    # Act
    async with ctl.client() as http:
        with pytest.raises(StopLoop):
            await dain_node.heartbeat_loop(agent, client=http, interval_s=0)

    # Assert
    assert ctl.paths()[:3] == [
        "/api/nodes/join",
        f"/api/nodes/{NODE_ID}/heartbeat",
        f"/api/nodes/{NODE_ID}/heartbeat",
    ]


@pytest.mark.asyncio
async def test_heartbeat_loop_rejoins_after_the_control_plane_forgets_the_node(agent):
    # Arrange
    ctl = FakeControlPlane(
        httpx.Response(201, json=asdict(agent.profile)),
        httpx.Response(404, json={"detail": "node not found"}),
        httpx.Response(201, json=asdict(agent.profile)),
    )

    # Act
    async with ctl.client() as http:
        with pytest.raises(StopLoop):
            await dain_node.heartbeat_loop(agent, client=http, interval_s=0)

    # Assert
    assert ctl.paths()[:4] == [
        "/api/nodes/join",
        f"/api/nodes/{NODE_ID}/heartbeat",
        "/api/nodes/join",
        f"/api/nodes/{NODE_ID}/heartbeat",
    ]


@pytest.mark.asyncio
async def test_heartbeat_loop_retries_the_join_when_it_is_refused(agent):
    # Arrange
    ctl = FakeControlPlane(
        httpx.Response(403, json={"detail": "invalid pool secret"}),
        httpx.Response(403, json={"detail": "invalid pool secret"}),
    )

    # Act
    async with ctl.client() as http:
        with pytest.raises(StopLoop):
            await dain_node.heartbeat_loop(agent, client=http, interval_s=0)

    # Assert — a refused node never heartbeats, it only retries the join.
    assert ctl.paths() == ["/api/nodes/join"] * 3


# --------------------------------------------------------------------------
# The profile is built before the server starts
# --------------------------------------------------------------------------


def test_profile_endpoint_serves_the_configured_profile(client):
    # Act
    response = client.get("/profile")

    # Assert
    assert response.status_code == 200
    assert response.json() == asdict(make_profile())


def test_profile_endpoint_refuses_before_the_agent_is_configured():
    # Arrange — no configure(), which is the placeholder-profile regression.
    unconfigured = TestClient(dain_node.app)

    # Act / Assert
    with pytest.raises(RuntimeError, match="not configured"):
        unconfigured.get("/profile")


def test_main_builds_the_profile_before_the_server_starts(monkeypatch):
    # Arrange
    monkeypatch.setenv(dain_node.POOL_SECRET_ENV, POOL_SECRET)
    monkeypatch.setattr(dain_node, "detect_fabric_ip", lambda ctl_host: FABRIC_IP)
    captured: dict[str, Any] = {}

    def fake_run(app, **kwargs):
        captured["agent"] = dain_node.current_agent()
        captured["host"] = kwargs["host"]

    monkeypatch.setattr(dain_node.uvicorn, "run", fake_run)

    # Act
    exit_code = dain_node.main(["--ctl", CTL, "--node-id", NODE_ID])

    # Assert — the real profile exists by the time uvicorn is handed the app.
    assert exit_code == dain_node.EXIT_OK
    assert captured["agent"].profile.id == NODE_ID
    assert captured["agent"].profile.host == FABRIC_IP
    assert captured["host"] == FABRIC_IP


def test_main_refuses_to_start_without_a_pool_secret(monkeypatch):
    # Arrange
    monkeypatch.delenv(dain_node.POOL_SECRET_ENV, raising=False)

    # Act
    exit_code = dain_node.main(["--ctl", CTL])

    # Assert
    assert exit_code == dain_node.EXIT_MISCONFIGURED


def test_main_refuses_to_start_without_a_control_plane_endpoint(monkeypatch):
    # Arrange
    monkeypatch.delenv(dain_node.CTL_ENDPOINT_ENV, raising=False)
    monkeypatch.setenv(dain_node.POOL_SECRET_ENV, POOL_SECRET)

    # Act
    exit_code = dain_node.main([])

    # Assert — no address is baked in; NODE-2's mDNS is what removes the flag.
    assert exit_code == dain_node.EXIT_MISCONFIGURED


# --------------------------------------------------------------------------
# Health and telemetry
# --------------------------------------------------------------------------


def test_health_reports_ok(client):
    # Act
    response = client.get("/health")

    # Assert
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_metrics_are_prometheus_text_labelled_with_the_node_id(client):
    # Act
    response = client.get("/metrics")

    # Assert
    assert response.headers["content-type"].startswith("text/plain")
    assert f'node_cpu_utilisation{{node_id="{NODE_ID}"}}' in response.text
    assert f'node_memory_free_mib{{node_id="{NODE_ID}"}}' in response.text


def test_metrics_report_the_rpc_server_down_when_it_is_not_running(client, agent):
    # Arrange — the lifespan never ran, so there is no child process.
    assert agent.rpc_proc is None

    # Act
    response = client.get("/metrics")

    # Assert
    assert f'node_rpc_server_up{{node_id="{NODE_ID}"}} 0' in response.text


def test_metrics_report_the_rpc_server_up_when_it_is_running(client, agent):
    # Arrange
    agent.rpc_proc = FakeProcess()

    # Act
    response = client.get("/metrics")

    # Assert
    assert f'node_rpc_server_up{{node_id="{NODE_ID}"}} 1' in response.text


# --------------------------------------------------------------------------
# Agent wiring
# --------------------------------------------------------------------------


def test_configure_installs_the_agent_the_routes_read():
    # Act
    agent = dain_node.configure(make_profile(), ctl=CTL, pool_secret=POOL_SECRET)

    # Assert
    assert dain_node.current_agent() is agent
    assert agent.join_url == f"http://{CTL}/api/nodes/join"
    assert agent.heartbeat_url == f"http://{CTL}/api/nodes/{NODE_ID}/heartbeat"


def test_current_agent_raises_before_configure():
    # Act / Assert
    with pytest.raises(RuntimeError, match="not configured"):
        dain_node.current_agent()


@pytest.mark.asyncio
async def test_lifespan_starts_and_stops_the_rpc_server(monkeypatch, agent):
    # Arrange
    dain_node.app.state.agent = agent
    proc = FakeProcess()
    monkeypatch.setattr(dain_node, "start_rpc_server", lambda host, port: proc)
    monkeypatch.setattr(dain_node, "heartbeat_loop", never_returns)

    # Act
    async with dain_node.lifespan(dain_node.app):
        assert agent.rpc_proc is proc

    # Assert
    assert proc.terminated is True
    assert agent.rpc_proc is None
