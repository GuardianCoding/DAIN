from contracts import NodeProfile

gpu_01 = NodeProfile(
    id="gpu-01",
    host="192.168.50.10",
    cpu="Intel Core i7",
    cores=4,  # Replace with the actual core count
    ram_total_mb=64 * 1024,
    ram_free_mb=46 * 1024,
    gpu="NVIDIA GeForce RTX 5070 Ti",
    vram_total_mb=16 * 1024,
    backend="cuda",
    mem_bandwidth_gbs=0.0,
    tg_tok_s=0.0,
    pp_tok_s=0.0,
    rtt_ms=0.0,
    state="idle",
)


office_01 = NodeProfile(
    id="office-01",
    host="192.168.50.11",
    cpu="Intel Core i7 vPro",
    cores=4,  # Mock value; replace after profiling the machine.
    ram_total_mb=8 * 1024,
    ram_free_mb=6 * 1024,
    gpu="Intel integrated graphics",
    vram_total_mb=0,
    backend="cpu",
    mem_bandwidth_gbs=0.0,
    tg_tok_s=0.0,
    pp_tok_s=0.0,
    rtt_ms=0.4,
    state="idle",
)


office_02 = NodeProfile(
    id="office-02",
    host="192.168.50.12",
    cpu="Intel Core i7 vPro",
    cores=4,  # Mock value; replace after profiling the machine.
    ram_total_mb=8 * 1024,
    ram_free_mb=6 * 1024,
    gpu="Intel integrated graphics",
    vram_total_mb=0,
    backend="cpu",
    mem_bandwidth_gbs=0.0,
    tg_tok_s=0.0,
    pp_tok_s=0.0,
    rtt_ms=0.4,
    state="idle",
)


mac_01 = NodeProfile(
    id="mac-01",
    host="192.168.50.13",
    cpu="Apple M5 Pro",
    cores=12,  # Mock value; replace with the Mac's actual core count.
    ram_total_mb=24 * 1024,
    ram_free_mb=18 * 1024,
    gpu="Apple M5 Pro integrated GPU",
    vram_total_mb=24 * 1024,  # Unified memory; do not add this to RAM capacity.
    backend="metal",
    mem_bandwidth_gbs=0.0,
    tg_tok_s=0.0,
    pp_tok_s=0.0,
    rtt_ms=0.5,
    state="idle",
)


MOCK_NODES = [
    gpu_01,
    office_01,
    office_02,
    mac_01,
]
