"""mDNS advertisement and control-plane discovery for NODE-2."""

from __future__ import annotations

import os
import socket
from collections.abc import Callable
from dataclasses import dataclass
from threading import Event, Lock
from typing import Any, Protocol

from zeroconf import IPVersion, ServiceBrowser, ServiceInfo, Zeroconf

from contracts import NodeProfile

SERVICE_TYPE = "_dain._tcp.local."
MDNS_DISABLED_ENV = "DAIN_MDNS_DISABLED"
CTL_ADVERTISE_HOST_ENV = "DAIN_CTL_ADVERTISE_HOST"
DEFAULT_CTL_PORT = 8000
DEFAULT_NODE_PORT = 9100
DISCOVERY_TIMEOUT_S = 3.0
MDNS_MULTICAST_ADDRESS = "224.0.0.251"
MDNS_PORT = 5353


class Browser(Protocol):
    def cancel(self) -> None: ...


@dataclass
class ServiceAdvertisement:
    zeroconf: Zeroconf
    info: ServiceInfo
    closed: bool = False

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            self.zeroconf.unregister_service(self.info)
        finally:
            self.zeroconf.close()


def advertise_control_plane(
    *,
    host: str | None = None,
    port: int = DEFAULT_CTL_PORT,
    zeroconf_factory: Callable[[], Zeroconf] = Zeroconf,
) -> ServiceAdvertisement | None:
    if discovery_disabled():
        return None
    address = host or os.getenv(CTL_ADVERTISE_HOST_ENV) or local_ipv4()
    return _advertise(
        instance="dain-control-plane",
        host=address,
        port=port,
        properties={"role": "control-plane", "version": "1"},
        zeroconf_factory=zeroconf_factory,
    )


def advertise_node(
    profile: NodeProfile,
    *,
    port: int = DEFAULT_NODE_PORT,
    zeroconf_factory: Callable[[], Zeroconf] = Zeroconf,
) -> ServiceAdvertisement | None:
    if discovery_disabled():
        return None
    return _advertise(
        instance=profile.id,
        host=profile.host,
        port=port,
        properties={
            "role": "node",
            "node_id": profile.id,
            "ram_mb": str(profile.ram_total_mb),
            "backend": profile.backend,
            "version": "1",
        },
        zeroconf_factory=zeroconf_factory,
    )


def _advertise(
    *,
    instance: str,
    host: str,
    port: int,
    properties: dict[str, str],
    zeroconf_factory: Callable[[], Zeroconf],
) -> ServiceAdvertisement:
    if not host or host in {"0.0.0.0", "::"}:
        raise ValueError("mDNS advertisement requires one concrete host address")
    if not 1 <= port <= 65535:
        raise ValueError("mDNS advertisement port must be between 1 and 65535")

    info = ServiceInfo(
        SERVICE_TYPE,
        f"{instance}.{SERVICE_TYPE}",
        port=port,
        properties=properties,
        parsed_addresses=[host],
        server=f"{instance}.local.",
    )
    zeroconf = zeroconf_factory()
    try:
        zeroconf.register_service(info, allow_name_change=True)
    except BaseException:
        zeroconf.close()
        raise
    return ServiceAdvertisement(zeroconf, info)


class _ControlPlaneListener:
    def __init__(self, found: Event) -> None:
        self.found = found
        self.endpoint: str | None = None
        self.lock = Lock()

    def add_service(self, zeroconf: Zeroconf, type_: str, name: str) -> None:
        self._consider(zeroconf, type_, name)

    def update_service(self, zeroconf: Zeroconf, type_: str, name: str) -> None:
        self._consider(zeroconf, type_, name)

    def remove_service(self, _zeroconf: Zeroconf, _type: str, _name: str) -> None:
        return None

    def _consider(self, zeroconf: Zeroconf, type_: str, name: str) -> None:
        info = zeroconf.get_service_info(type_, name, timeout=500)
        if info is None or info.decoded_properties.get("role") != "control-plane":
            return

        addresses = info.parsed_addresses(IPVersion.V4Only)
        host = addresses[0] if addresses else (info.server or "").rstrip(".")
        if not host or info.port is None:
            return

        with self.lock:
            if self.endpoint is None:
                self.endpoint = f"{host}:{info.port}"
                self.found.set()


def discover_control_plane(
    *,
    timeout_s: float = DISCOVERY_TIMEOUT_S,
    zeroconf_factory: Callable[[], Zeroconf] = Zeroconf,
    browser_factory: Callable[[Zeroconf, str, Any], Browser] = ServiceBrowser,
) -> str | None:
    if discovery_disabled():
        return None
    if timeout_s <= 0:
        raise ValueError("timeout_s must be greater than zero")

    found = Event()
    listener = _ControlPlaneListener(found)
    zeroconf = zeroconf_factory()
    browser = browser_factory(zeroconf, SERVICE_TYPE, listener)
    try:
        found.wait(timeout_s)
        return listener.endpoint
    finally:
        browser.cancel()
        zeroconf.close()


def local_ipv4() -> str:
    override = os.getenv(CTL_ADVERTISE_HOST_ENV)
    if override:
        return override

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect((MDNS_MULTICAST_ADDRESS, MDNS_PORT))
            address = probe.getsockname()[0]
    except OSError:
        address = ""

    if not address or address == "0.0.0.0":
        raise RuntimeError(
            f"could not detect the mDNS address; set {CTL_ADVERTISE_HOST_ENV}"
        )
    return address


def discovery_disabled() -> bool:
    return os.getenv(MDNS_DISABLED_ENV, "").casefold() in {"1", "true", "yes"}
