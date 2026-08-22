"""The bridge between the model ladder and the scheduler — `scheduler_spec()`.

sched.plan() wants four fields and the repo carried none of them together:

    {"model_id", "total_layers", "file_size_mb", "kv_mb_per_layer"}

This lives in infer/, not ctl/main.py, because every input is a fact about the
model files, the cache geometry belongs next to the maths that uses it, and it
keeps file I/O out of sched/. The endpoint then stays five lines and a 404.

THE THREE NUMBERS, AND THE TRAP IN EACH:

  file_size_mb     ModelSpec.weights_mib. Do NOT use size_gb * 1000 — size_gb
                   is decimal GB from Hugging Face, while the scheduler
                   compares against ram_free_mb, which is binary MiB from
                   `free -m`. castoff is 10967, not 11500; the 4.9% gap is
                   roughly the margin the capacity claim turns on.

  total_layers     Now a required key in models.toml. Authoritative source is
                   the GGUF header (`block_count`), or llama-server's loader
                   log. Everything here is still `estimated` — see below.

  kv_mb_per_layer  NOT a model constant. It scales with context length and
                   concurrent sessions, so scheduler_spec() takes both. The
                   8.0 that used to sit in sched/mock.py is right for 8k x 1
                   and wrong by 64x at 128k x 4.

  Divide the cache by TOTAL layers, not cached ones. Both gpt-oss models
  alternate full attention with sliding-window layers, so only half hold a
  context-scaled cache — but sched.cost multiplies this figure by a node's
  layer count, so it has to be spread across all of them.

THE HARD PART: plan at 8k and launch at 128k and the split is silently wrong.
`context` and `slots` here must be the same values llama_server_command() is
launched with. Nothing enforces that across a process boundary; it is why both
are required arguments rather than defaulted constants.

PROVENANCE. Every geometry below is `source="estimated"`, derived from
published architecture. `unverified_models()` lists them, and the numbers stop
being estimates on the first successful llama-server load — its startup prints
`llama_kv_cache: KV self size = NNN MiB` and the block count. Until then, do
not put a derived capacity figure on a slide.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from infer.memory import GPT_OSS_120B_KV, KVGeometry, kv_cache_mib
from infer.models import ModelSpec, load_ladder

# Cache geometry per model KEY (never the role — the key is the directory name
# the weights sit in). Absent means we genuinely do not know: scheduler_spec()
# raises rather than guessing, because a wrong geometry silently mis-sizes
# every node's slice.
#
# Qwen3 dense models: GQA with 8 KV heads at 128 head dim, full attention on
# every layer. gpt-oss: 8 KV heads at 64 head dim, alternating full and
# sliding-window, so half the layers hold a context-scaled cache.
KV_GEOMETRY: dict[str, KVGeometry] = {
    "calibration": KVGeometry(layers=28, kv_heads=8, head_dim=128),
    "embed": KVGeometry(layers=28, kv_heads=8, head_dim=128),
    "replica": KVGeometry(layers=36, kv_heads=8, head_dim=128),
    "castoff": KVGeometry(
        layers=24, kv_heads=8, head_dim=64, full_attention_layers=12
    ),
    "headline": GPT_OSS_120B_KV,
    # working / mtp / working_spare (Qwen3.6-35B-A3B) are deliberately absent.
    # Its block count is not published anywhere we trust and nobody has read
    # the GGUF header yet. Guessing would be worse than a 503, because a wrong
    # layer count stays invisible until the machine OOMs.
}


class ModelSpecUnavailable(RuntimeError):
    """Known model, but we cannot build a scheduler spec for it yet."""


@lru_cache(maxsize=1)
def _ladder() -> tuple[ModelSpec, ...]:
    """load_ladder() re-parsed the TOML on every call; /api/plan is hot enough
    that it should not touch the disk per request."""
    return load_ladder()


def reset_cache() -> None:
    """Drop the cached ladder. For tests that edit models.toml."""
    _ladder.cache_clear()


def known_models() -> tuple[str, ...]:
    return tuple(spec.model_id for spec in _ladder())


def resolve_model_id(identifier: str) -> str:
    """Canonicalise to the models.toml KEY, accepting a role as an alias.

    `role` is a second namespace that disagrees with the key on half the
    ladder — castoff/castoff_capacity, embed/embedding, mtp/speculative,
    working_spare/spare_quant. Roles are accepted for kindness, but the key is
    what reaches Assignment.model_id, because fetch_models.py writes weights
    into <dest>/<model_id>/ and Cluster.model_file resolves
    <models>/<model_id>/<file>. The identifier IS the directory name.

    Raises KeyError for anything unknown — the caller maps that to 404, since
    a misspelled model is a bad request, not a capacity failure.
    """
    wanted = identifier.strip()
    specs = _ladder()

    for spec in specs:
        if spec.model_id == wanted:
            return spec.model_id
    for spec in specs:
        if spec.role == wanted:
            return spec.model_id

    raise KeyError(
        f"unknown model {identifier!r}; pass one of: "
        f"{', '.join(spec.model_id for spec in specs)}"
    )


def geometry_for(model_id: str) -> KVGeometry:
    try:
        return KV_GEOMETRY[model_id]
    except KeyError as exc:
        raise ModelSpecUnavailable(
            f"no KV cache geometry for {model_id!r}. Read block_count, "
            f"head_count_kv and key_length from its GGUF header (or "
            f"llama-server's loader log) and add it to infer/spec.KV_GEOMETRY"
        ) from exc


def unverified_models() -> tuple[str, ...]:
    """Models whose geometry is still derived rather than measured.

    Everything, today. Kept as a function so a test can assert the list only
    ever shrinks, and so nobody has to grep for source="estimated".
    """
    return tuple(
        model_id
        for model_id, geometry in sorted(KV_GEOMETRY.items())
        if geometry.source != "measured"
    )


def scheduler_spec(model_id: str, context: int, slots: int) -> dict[str, Any]:
    """The model_spec dict sched.plan() consumes.

    `context` and `slots` are required, not defaulted: they change the answer,
    and they must match what llama_server_command() is launched with.
    """
    if context <= 0 or slots <= 0:
        raise ValueError(f"context and slots must be positive, got {context}, {slots}")

    key = resolve_model_id(model_id)
    spec = next(item for item in _ladder() if item.model_id == key)
    geometry = geometry_for(key)

    total_layers = spec.total_layers
    if total_layers <= 0:
        raise ModelSpecUnavailable(f"{key} has no total_layers in models.toml")
    if geometry.layers != total_layers:
        # Two sources for one fact is a bug waiting to happen; if they
        # disagree, one was hand-edited and the resulting split is wrong.
        raise ModelSpecUnavailable(
            f"{key}: models.toml says {total_layers} layers but its KV "
            f"geometry says {geometry.layers}"
        )

    kv_total_mib = kv_cache_mib(geometry, context, slots)

    return {
        "model_id": key,
        "total_layers": total_layers,
        # weights_mib, never size_gb * 1000 — see the module docstring.
        "file_size_mb": spec.weights_mib,
        # Total layers, not cached_layers: sched.cost multiplies this by a
        # node's layer count.
        "kv_mb_per_layer": kv_total_mib / total_layers,
    }
