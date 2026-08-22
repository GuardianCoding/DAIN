from pathlib import Path

from zeroconf import ServiceInfo

from node.discovery import (
    CTL_ADVERTISE_HOST_ENV,
    SERVICE_TYPE,
    advertise_control_plane,
    advertise_node,
    discover_control_plane,
    local_ipv4,
)
from tests.node_doubles import FABRIC_IP, make_profile


class FakeZeroconf:
    def __init__(self, discovered: ServiceInfo | None = None) -> None:
        self.discovered = discovered
        self.registered: list[ServiceInfo] = []
        self.unregistered: list[ServiceInfo] = []
        self.closed = False

    def register_service(
        self,
        info: ServiceInfo,
        allow_name_change: bool = False,
    ) -> None:
        assert allow_name_change
        self.registered.append(info)

    def unregister_service(self, info: ServiceInfo) -> None:
        self.unregistered.append(info)

    def get_service_info(
        self,
        _type: str,
        _name: str,
        timeout: int = 0,
    ) -> ServiceInfo | None:
        assert timeout == 500
        return self.discovered

    def close(self) -> None:
        self.closed = True


class ImmediateBrowser:
    def __init__(self, zeroconf, service_type, listener) -> None:
        self.cancelled = False
        listener.add_service(
            zeroconf,
            service_type,
            f"found.{service_type}",
        )

    def cancel(self) -> None:
        self.cancelled = True


def test_control_plane_advertises_its_role_and_endpoint(monkeypatch) -> None:
    monkeypatch.delenv("DAIN_MDNS_DISABLED", raising=False)
    fake = FakeZeroconf()

    advertisement = advertise_control_plane(
        host="192.168.50.20",
        port=8000,
        zeroconf_factory=lambda: fake,
    )

    assert advertisement is not None
    info = fake.registered[0]
    assert info.type == SERVICE_TYPE
    assert info.parsed_addresses() == ["192.168.50.20"]
    assert info.port == 8000
    assert info.decoded_properties["role"] == "control-plane"

    advertisement.close()
    assert fake.unregistered == [info]
    assert fake.closed


def test_node_advertisement_carries_profile_metadata(monkeypatch) -> None:
    monkeypatch.delenv("DAIN_MDNS_DISABLED", raising=False)
    fake = FakeZeroconf()

    advertisement = advertise_node(
        make_profile(),
        zeroconf_factory=lambda: fake,
    )

    assert advertisement is not None
    properties = fake.registered[0].decoded_properties
    assert properties == {
        "role": "node",
        "node_id": "office-01",
        "ram_mb": "8192",
        "backend": "cpu",
        "version": "1",
    }


def test_node_discovers_the_control_plane_without_an_address(monkeypatch) -> None:
    monkeypatch.delenv("DAIN_MDNS_DISABLED", raising=False)
    info = ServiceInfo(
        SERVICE_TYPE,
        f"dain-control-plane.{SERVICE_TYPE}",
        port=8000,
        properties={"role": "control-plane"},
        parsed_addresses=["192.168.50.20"],
    )
    fake = FakeZeroconf(info)

    endpoint = discover_control_plane(
        timeout_s=0.1,
        zeroconf_factory=lambda: fake,
        browser_factory=ImmediateBrowser,
    )

    assert endpoint == "192.168.50.20:8000"
    assert fake.closed


def test_discovery_ignores_a_node_advertisement(monkeypatch) -> None:
    monkeypatch.delenv("DAIN_MDNS_DISABLED", raising=False)
    info = ServiceInfo(
        SERVICE_TYPE,
        f"office-01.{SERVICE_TYPE}",
        port=9100,
        properties={"role": "node"},
        parsed_addresses=[FABRIC_IP],
    )

    endpoint = discover_control_plane(
        timeout_s=0.001,
        zeroconf_factory=lambda: FakeZeroconf(info),
        browser_factory=ImmediateBrowser,
    )

    assert endpoint is None


def test_local_ipv4_honours_the_explicit_interface_address(monkeypatch) -> None:
    monkeypatch.setenv(CTL_ADVERTISE_HOST_ENV, FABRIC_IP)

    assert local_ipv4() == FABRIC_IP


def test_discovery_module_contains_no_committed_addresses() -> None:
    source = Path("node/discovery.py").read_text()

    assert "192.168.50." not in source
