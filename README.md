# DAIN

DAIN turns a small group of computers into one observable pool for model
inference, jobs, memory and files. This branch contains the CP-1 mock control
plane: a stable API that lets the node, scheduler, inference and dashboard
workstreams develop without waiting for real hardware.

## Run the mock control plane

Requirements: Python 3.12+ and `uv`.

```bash
uv sync
uv run uvicorn ctl.main:app --reload --host 127.0.0.1 --port 8000
```

Open:

- API documentation: <http://127.0.0.1:8000/docs>
- Node list: <http://127.0.0.1:8000/api/nodes>
- Health check: <http://127.0.0.1:8000/health>
- WebSocket feed: `ws://127.0.0.1:8000/feed`

Run the checks:

```bash
uv run ruff format --check ctl tests
uv run ruff check ctl tests
uv run pytest
```

## Frozen CP-1 interface

| Method | Path | Mock behaviour |
| --- | --- | --- |
| `GET` | `/api/nodes` | Returns the current in-memory node registry. |
| `POST` | `/api/nodes/join` | Adds or replaces a node; returns `201` or `403`. |
| `DELETE` | `/api/nodes/{id}` | Removes a node; returns `204` or `404`. |
| `GET` | `/api/plan?model=...` | Returns a deterministic mock `Assignment`. |
| `POST` | `/api/jobs` | Creates a queued job and mock fan-out assignment. |
| `GET` | `/api/jobs/{id}` | Returns the stored job or `404`. |
| `GET` | `/api/metrics` | Returns one live `NodeMetrics` sample per node. |
| `POST` | `/api/race` | Returns a deterministic serial or fan-out result. |
| `WS` | `/feed` | Streams `topology`, `metrics`, `event` and `flow` frames. |

The interactive `/docs` page is the canonical source for request bodies.

The mock join secret defaults to `mock-only-secret`. Override it when needed:

```bash
DAIN_POOL_SECRET=local-development-only \
  uv run uvicorn ctl.main:app --reload --host 127.0.0.1 --port 8000
```

This is a development mock, not production authentication. Never place a real
pool secret in the repository.

## WebSocket frames

The first frame is always a topology snapshot. Each cycle then emits one frame
of each remaining type:

```text
topology -> metrics -> event -> flow -> metrics -> event -> flow -> ...
```

- `topology`: full `NodeProfile` objects.
- `metrics`: full `NodeMetrics` snapshots. Metal unified memory is reported as
  RAM and is not duplicated as separate free VRAM.
- `event`: a human-readable cluster event for the dashboard log.
- `flow`: a job movement from the controller to a target node.

## CP-1 boundary

Everything is deterministic and in memory. Restarting the server resets nodes,
jobs and metrics. Real heartbeats, retries, scheduling, inference, authentication
and execution belong to later workstreams and are deliberately not implemented
in this mock.
