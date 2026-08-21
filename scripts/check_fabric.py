#!/usr/bin/env python3
"""Prove the fabric before blaming the code.

Most "the cluster is broken" is "the network is broken", and this rules it out
in about ninety seconds. Python rather than PowerShell because four nodes are
Linux and one is Windows, and the checks have to agree across both.

    # on every node except one
    python3 scripts/check_fabric.py listen

    # on the remaining node
    python3 scripts/check_fabric.py probe

    # before committing to the overnight drive plan
    python3 scripts/check_fabric.py speedtest

WHAT CHANGED FROM THE BRIEF: it pings four fixed addresses. There are no fixed
addresses any more — nodes are discovered. So the thing to verify is the
transport discovery actually depends on: link-local MULTICAST across the
switch. If multicast is blocked, mDNS join fails and no amount of unicast ping
tells you why.

The probe uses 224.0.0.251 — the mDNS group, so the same scope and the same
treatment by IGMP snooping — on a different port, so it cannot interfere with
real responders.
"""

from __future__ import annotations

import argparse
import socket
import struct
import sys
import time
import tomllib
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from infer.models import estimate_hours, load_ladder  # noqa: E402

CLUSTER_PATH = REPO_ROOT / "cluster.toml"

MDNS_GROUP = "224.0.0.251"      # same group as mDNS: same scope, same IGMP treatment
PROBE_PORT = 45454              # NOT 5353 — must not interfere with real responders
PROBE_MAGIC = b"DAIN-FABRIC-PROBE"
PROBE_TIMEOUT_S = 3.0
TCP_TIMEOUT_S = 2.0

# Sub-millisecond is what a gigabit switch gives you. Above this and something
# is routing over Wi-Fi or through a device you did not intend.
RTT_WARN_MS = 2.0

SPEEDTEST_URL = "https://speed.cloudflare.com/__down?bytes=25000000"
SPEEDTEST_BYTES = 25_000_000
DRIVE_THRESHOLD_HOURS = 2.0


@dataclass(frozen=True)
class Peer:
    host: str
    name: str
    rtt_ms: float | None = None
    rpc_open: bool = False


def load_discovery(path: Path = CLUSTER_PATH) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"cluster file not found: {path}")
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    if "discovery" not in raw:
        raise ValueError(f"{path} is missing the [discovery] section")
    return raw["discovery"]


def local_addresses() -> list[str]:
    """Every IPv4 address this host answers on, loopback excluded."""
    try:
        infos = socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
    except socket.gaierror:
        infos = []
    found = sorted({info[4][0] for info in infos if not info[4][0].startswith("127.")})
    if found:
        return found

    # getaddrinfo is unreliable on a host with no resolvable hostname.
    probe_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe_socket.connect((MDNS_GROUP, PROBE_PORT))
        return [probe_socket.getsockname()[0]]
    except OSError:
        return []
    finally:
        probe_socket.close()


def _multicast_socket(bind_address: str) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("", PROBE_PORT))
    membership = struct.pack("4s4s", socket.inet_aton(MDNS_GROUP), socket.inet_aton(bind_address))
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, membership)
    return sock


def listen(bind_address: str) -> int:
    """Answer probes until interrupted. Run this on every node but one."""
    sock = _multicast_socket(bind_address)
    print(f"listening on {MDNS_GROUP}:{PROBE_PORT} via {bind_address}  (ctrl-c to stop)")
    try:
        while True:
            payload, sender = sock.recvfrom(1024)
            if payload.startswith(PROBE_MAGIC):
                sock.sendto(PROBE_MAGIC + b"|" + socket.gethostname().encode(), sender)
                print(f"  answered {sender[0]}")
    except KeyboardInterrupt:
        print("\nstopped")
        return 0
    finally:
        sock.close()


def probe(bind_address: str) -> list[Peer]:
    """Shout on the multicast group and collect whoever answers."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.inet_aton(bind_address))
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 1)   # never leave the segment
    sock.settimeout(0.4)

    peers: dict[str, str] = {}
    sock.sendto(PROBE_MAGIC, (MDNS_GROUP, PROBE_PORT))
    deadline = time.monotonic() + PROBE_TIMEOUT_S
    while time.monotonic() < deadline:
        try:
            payload, sender = sock.recvfrom(1024)
        except socket.timeout:
            continue
        if payload.startswith(PROBE_MAGIC) and sender[0] != bind_address:
            peers[sender[0]] = payload.split(b"|", 1)[-1].decode(errors="replace")
    sock.close()
    return [Peer(host=host, name=name) for host, name in sorted(peers.items())]


def measure_peer(peer: Peer, rpc_port: int) -> Peer:
    """TCP connect time stands in for RTT — no ICMP permissions, same answer."""
    started = time.monotonic()
    reachable = False
    try:
        with socket.create_connection((peer.host, rpc_port), timeout=TCP_TIMEOUT_S):
            reachable = True
    except OSError:
        pass
    elapsed_ms = (time.monotonic() - started) * 1000
    return Peer(peer.host, peer.name, rtt_ms=elapsed_ms if reachable else None, rpc_open=reachable)


def report(peers: list[Peer], rpc_port: int) -> int:
    if not peers:
        print("\nNO PEERS ANSWERED.")
        print("  Multicast is not crossing the switch, or nothing is in `listen` mode.")
        print("  This is exactly what breaks mDNS join. Check, in order:")
        print("    - a host firewall (Windows: the NIC must be Private, not Public)")
        print("    - a Wi-Fi adapter still up and stealing the route -- disable it")
        print("    - IGMP snooping on a managed switch with no querier")
        return 1

    print(f"\n{'PEER':<16} {'NAME':<20} {'RTT':>9}   RPC :{rpc_port}")
    closed = 0
    for peer in peers:
        rtt = f"{peer.rtt_ms:.2f} ms" if peer.rtt_ms is not None else "  --"
        print(f"{peer.host:<16} {peer.name:<20} {rtt:>9}   {'open' if peer.rpc_open else 'CLOSED'}")
        if not peer.rpc_open:
            closed += 1
        elif peer.rtt_ms and peer.rtt_ms > RTT_WARN_MS:
            print(f"    slow ({peer.rtt_ms:.2f} ms). Sub-millisecond is normal on gigabit —")
            print("    this smells like Wi-Fi or a second hop.")

    if closed:
        print(f"\n{closed} peer(s) discoverable but not accepting RPC on {rpc_port}.")
        print("  The node is reachable, so this is rpc-server not running, or a firewall rule.")
        print("  Multicast working while TCP fails is the good case: discovery is fine.")
        return 1

    print("\nFabric is healthy: multicast crosses the switch and every peer accepts RPC.")
    return 0


def speedtest() -> int:
    """Measure the link, then say what it means for the model ladder."""
    print(f"downloading {SPEEDTEST_BYTES / 1e6:.0f} MB from speed.cloudflare.com ...")
    started = time.monotonic()
    try:
        with urllib.request.urlopen(SPEEDTEST_URL, timeout=60) as response:
            downloaded = 0
            while chunk := response.read(1 << 16):
                downloaded += len(chunk)
    except OSError as error:
        print(f"speedtest failed: {error}", file=sys.stderr)
        return 1

    elapsed = time.monotonic() - started
    mbps = (downloaded * 8) / elapsed / 1e6
    print(f"  {downloaded / 1e6:.0f} MB in {elapsed:.1f}s = {mbps:.1f} Mbps\n")

    specs = load_ladder()
    demo_gb = sum(spec.size_gb for spec in specs if spec.priority <= 4)
    total_gb = sum(spec.size_gb for spec in specs)
    demo_hours = estimate_hours(demo_gb, mbps)
    print(f"  demo set (priority 1-4, {demo_gb:.1f} GB): {demo_hours:.1f} h")
    print(f"  full ladder ({total_gb:.1f} GB):          {estimate_hours(total_gb, mbps):.1f} h")

    if demo_hours > DRIVE_THRESHOLD_HOURS:
        print(f"\n  Over {DRIVE_THRESHOLD_HOURS:.0f} h for the demo set — use the external drive.")
    else:
        print("\n  Fast enough to pull the demo set here. Still take the drive for the headline.")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the DAIN fabric.")
    parser.add_argument("mode", choices=("probe", "listen", "speedtest"), nargs="?", default="probe")
    parser.add_argument("--bind", help="Fabric interface address (default: auto-detect)")
    parser.add_argument("--cluster", type=Path, default=CLUSTER_PATH)
    return parser.parse_args(argv)


def resolve_bind(explicit: str | None) -> str:
    if explicit:
        return explicit
    addresses = local_addresses()
    if not addresses:
        raise RuntimeError("no non-loopback IPv4 address found. Is the fabric NIC up?")
    if len(addresses) > 1:
        print(f"note: several addresses {addresses}; using {addresses[0]}. Override with --bind.")
    return addresses[0]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.mode == "speedtest":
            return speedtest()

        discovery = load_discovery(args.cluster)
        rpc_port = int(discovery["rpc_port"])
        bind_address = resolve_bind(args.bind)
        print(f"fabric interface: {bind_address}")

        if args.mode == "listen":
            return listen(bind_address)

        peers = [measure_peer(peer, rpc_port) for peer in probe(bind_address)]
        return report(peers, rpc_port)
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
