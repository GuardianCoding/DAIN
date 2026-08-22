import pytest

from contracts import NodeMetrics, NodeProfile
from ctl.registry import NodeRegistry


class FakeClock:
    def __init__(self, now: float = 0.0):
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class CountingLock:
    def __init__(self) -> None:
        self.entries = 0

    def __enter__(self):
        self.entries += 1
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        return None


def make_profile(node_id: str = "gpu-01") -> NodeProfile:
    return NodeProfile(
        id=node_id,
        host="192.168.50.10",
        cpu="Intel Core i7",
        cores=8,
        ram_total_mb=65536,
        ram_free_mb=49152,
        gpu="NVIDIA GeForce RTX 5070 Ti",
        vram_total_mb=16384,
        backend="cuda",
        mem_bandwidth_gbs=400.0,
        tg_tok_s=30.0,
        pp_tok_s=150.0,
        rtt_ms=0.5,
    )


def make_metrics(node_id: str = "gpu-01") -> NodeMetrics:
    return NodeMetrics(
        node_id=node_id,
        timestamp=1_700_000_000.0,
        cpu_percent=25.0,
        ram_free_mb=45000,
        gpu_percent=40.0,
        vram_free_mb=12000,
        jobs_running=1,
    )


def make_registry(
    clock: FakeClock,
    on_replan=None,
) -> NodeRegistry:
    return NodeRegistry(
        heartbeat_interval_s=2.0,
        missed_heartbeats_offline=3,
        clock=clock,
        wall_clock=lambda: 1_700_000_000.0,
        on_replan=on_replan,
    )


def test_register_stores_joining_node():
    clock = FakeClock()
    registry = make_registry(clock)
    profile = make_profile()

    result = registry.register(profile)
    record = registry.get_record("gpu-01")

    assert result is profile
    assert record is not None
    assert record.profile is profile
    assert record.profile.state == "joining"
    assert record.last_heartbeat == 0.0
    assert record.missed_heartbeats == 0
    assert registry.list_profiles() == [profile]

    assert len(registry.events) == 1
    assert registry.events[0].event == "joined"
    assert registry.events[0].node_id == "gpu-01"
    assert registry.events[0].sequence == 1


def test_register_is_idempotent():
    clock = FakeClock()
    registry = make_registry(clock)

    first_profile = make_profile()
    registry.register(first_profile)

    clock.advance(2.0)

    updated_profile = make_profile()
    updated_profile.host = "192.168.50.20"
    registry.register(updated_profile)

    record = registry.get_record("gpu-01")

    assert record is not None
    assert len(registry.nodes) == 1
    assert record.profile.host == "192.168.50.20"
    assert record.last_heartbeat == 2.0
    assert record.missed_heartbeats == 0

    # Re-registering must not create another joined event.
    assert len(registry.events) == 1
    assert registry.events[0].event == "joined"


def test_heartbeat_changes_joining_node_to_idle():
    clock = FakeClock()
    registry = make_registry(clock)
    registry.register(make_profile())

    clock.advance(1.0)
    record = registry.heartbeat("gpu-01")

    assert record.profile.state == "idle"
    assert record.last_heartbeat == 1.0
    assert record.missed_heartbeats == 0

    # A normal heartbeat should not create another event.
    assert [event.event for event in registry.events] == ["joined"]


def test_heartbeat_stores_metrics():
    clock = FakeClock()
    registry = make_registry(clock)
    registry.register(make_profile())

    metrics = make_metrics()
    record = registry.heartbeat("gpu-01", metrics)

    assert record.metrics is metrics
    assert registry.latest_metrics() == [metrics]


def test_heartbeat_rejects_unknown_node():
    clock = FakeClock()
    registry = make_registry(clock)

    with pytest.raises(KeyError):
        registry.heartbeat("missing-node")


def test_heartbeat_rejects_metrics_for_different_node():
    clock = FakeClock()
    registry = make_registry(clock)
    registry.register(make_profile("gpu-01"))

    incorrect_metrics = make_metrics("office-01")

    with pytest.raises(ValueError, match="office-01"):
        registry.heartbeat("gpu-01", incorrect_metrics)


def test_node_goes_offline_after_exactly_three_misses():
    clock = FakeClock()
    replans: list[tuple[str, str]] = []

    registry = make_registry(
        clock,
        on_replan=lambda node_id, reason: replans.append((node_id, reason)),
    )
    registry.register(make_profile())

    clock.now = 5.99
    events = registry.sweep()
    record = registry.get_record("gpu-01")

    assert record is not None
    assert events == []
    assert record.profile.state != "offline"
    assert record.missed_heartbeats == 2
    assert replans == []

    clock.now = 6.0
    events = registry.sweep()

    assert len(events) == 1
    assert events[0].event == "offline"
    assert events[0].node_id == "gpu-01"
    assert events[0].replan_required is True
    assert record.profile.state == "offline"
    assert record.missed_heartbeats == 3
    assert replans == [("gpu-01", "heartbeat_timeout")]


def test_repeated_sweeps_emit_only_one_offline_event():
    clock = FakeClock()
    replans: list[tuple[str, str]] = []

    registry = make_registry(
        clock,
        on_replan=lambda node_id, reason: replans.append((node_id, reason)),
    )
    registry.register(make_profile())

    clock.now = 6.0
    first_events = registry.sweep()

    clock.now = 20.0
    second_events = registry.sweep()
    third_events = registry.sweep()

    offline_events = [event for event in registry.events if event.event == "offline"]

    assert len(first_events) == 1
    assert second_events == []
    assert third_events == []
    assert len(offline_events) == 1
    assert replans == [("gpu-01", "heartbeat_timeout")]


def test_heartbeat_recovers_offline_node_once():
    clock = FakeClock()
    registry = make_registry(clock)
    registry.register(make_profile())

    clock.now = 6.0
    registry.sweep()

    clock.now = 7.0
    record = registry.heartbeat("gpu-01")

    assert record.profile.state == "idle"
    assert record.last_heartbeat == 7.0
    assert record.missed_heartbeats == 0

    recovered_events = [
        event for event in registry.events if event.event == "recovered"
    ]
    assert len(recovered_events) == 1

    clock.now = 8.0
    registry.heartbeat("gpu-01")

    recovered_events = [
        event for event in registry.events if event.event == "recovered"
    ]
    assert len(recovered_events) == 1


def test_node_without_required_heartbeat_stays_online():
    clock = FakeClock()
    registry = make_registry(clock)
    registry.register(make_profile(), heartbeat_required=False)

    clock.now = 100.0
    events = registry.sweep()
    record = registry.get_record("gpu-01")

    assert record is not None
    assert events == []
    assert record.profile.state == "joining"
    assert record.missed_heartbeats == 0


def test_remove_node():
    clock = FakeClock()
    registry = make_registry(clock)
    registry.register(make_profile())

    assert registry.remove("gpu-01") is True
    assert registry.get_record("gpu-01") is None
    assert registry.list_profiles() == []

    # Removing it again reports that nothing was removed.
    assert registry.remove("gpu-01") is False


def test_events_after_returns_only_newer_events():
    clock = FakeClock()
    registry = make_registry(clock)
    registry.register(make_profile())

    clock.now = 6.0
    registry.sweep()

    assert [event.sequence for event in registry.events_after(0)] == [1, 2]
    assert [event.sequence for event in registry.events_after(1)] == [2]
    assert registry.events_after(2) == []


def test_reset_clears_registry_state():
    clock = FakeClock()
    registry = make_registry(clock)
    registry.register(make_profile())
    registry.heartbeat("gpu-01", make_metrics())

    registry.reset()

    assert registry.list_profiles() == []
    assert registry.latest_metrics() == []
    assert registry.events_after(0) == []
    assert registry.next_sequence == 1


def test_public_registry_accessors_take_the_lock():
    clock = FakeClock()
    registry = make_registry(clock)
    registry.register(make_profile())
    registry.heartbeat("gpu-01", make_metrics())
    lock = CountingLock()
    registry.lock = lock

    registry.list_profiles()
    registry.get_record("gpu-01")
    registry.remove("missing")
    registry.events_after(0)
    registry.latest_metrics()
    registry.reset()

    assert lock.entries == 6
    assert registry.lock is lock
