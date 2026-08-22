# DAIN

DAIN turns a small group of computers into one observable pool for model
inference, jobs, memory and files. The control plane provides a stable API,
node registry, asynchronous job queue and live telemetry feed for the node,
scheduler, inference and dashboard workstreams.

## Run the control plane

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

## Control-plane interface

| Method | Path | behaviour |
| --- | --- | --- |
| `GET` | `/api/nodes` | Returns the current in-memory node registry. |
| `POST` | `/api/nodes/join` | Adds or replaces a node; returns `201` or `403`. |
| `DELETE` | `/api/nodes/{id}` | Removes a node; returns `204` or `404`. |
| `GET` | `/api/plan?model=...` | Returns a deterministic mock `Assignment`. |
| `POST` | `/api/jobs` | Creates a queued job and mock fan-out assignment. |
| `GET` | `/api/jobs/{id}` | Returns the stored job or `404`. |
| `GET` | `/api/metrics` | Returns latest node and llama-server metrics, 60-sample histories and polling errors. |
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

## Install a Linux node

On a Debian or Ubuntu node, export the same pool secret used by the controller
and run the installer. It installs Python 3.12, locked project dependencies,
bubblewrap isolation and a restarting `dain-node` systemd service. Re-running
the command safely updates the existing installation.

```bash
export DAIN_POOL_SECRET='replace-with-the-pool-secret'
curl -fsSL \
  https://raw.githubusercontent.com/GuardianCoding/DAIN/main/scripts/install_node.sh \
  | sudo --preserve-env=DAIN_POOL_SECRET bash
```

The node discovers the control plane over mDNS. Set `DAIN_CTL=host:8000` only
when multicast discovery is unavailable. Static addressing and host-firewall
changes are deliberately opt-in; see the variables documented at the top of
`scripts/install_node.sh`. The secret is stored in `/etc/dain/node.env` with
mode `0600`, not in the service unit or repository.

## Distributed file search

Each node indexes only the directory configured by `DAIN_INDEX_ROOT`. Run an
explicit `index` job before searching; a cold `/search` returns `409` instead
of starting an unbounded filesystem walk inside the two-second search timeout.

```bash
curl -X POST http://127.0.0.1:8000/api/jobs \
  -H 'content-type: application/json' \
  -d '{"kind":"index","payload":{},"fanout":4}'

curl -X POST http://127.0.0.1:8000/api/jobs \
  -H 'content-type: application/json' \
  -d '{"kind":"search","payload":{"query":"telemetry","limit":10},"fanout":4}'
```

The controller signs every node-job request with a short-lived HMAC covering
the complete request body. Nodes reject missing, stale, incorrectly signed or
tampered requests before touching the filesystem. The pool secret is read from
`DAIN_POOL_SECRET` and is never sent in a node-job request.

Index refreshes are single-flight and bounded to 10,000 files, 256 MiB total,
and 1 MiB per file. Search uses the 67 MB local
`BAAI/bge-small-en-v1.5` model through FastEmbed; the installer downloads it
once into `/var/cache/dain/fastembed`. All nodes report the model identifier,
and cosine scores are corpus-independent and comparable across nodes. Each
merged hit includes a unique `node_id:path` source, and `nodes_searched`
identifies every machine that contributed.

## WebSocket frames

The first frame is always a topology snapshot. Metrics frames follow at 2 Hz.
Topology, event and flow frames are emitted when their underlying state changes:

```text
topology -> metrics -> [topology | event | flow] -> metrics -> event -> flow -> ...
```
## Telemetry configuration

The fan-in polls heartbeat-managed nodes at `http://<node-host>:9100/metrics`.
Set the llama-server Prometheus endpoint before starting the control plane:

```bash
DAIN_LLAMA_METRICS_URL=http://gpu-01:8080/metrics \
  uv run uvicorn ctl.main:app --host 0.0.0.0 --port 8000
```

The dashboard should load its initial state from GET /api/nodes, then connect
to ws://<control-plane-host>:8000/feed. It can use history directly for
sparklines and display the last good sample while a source appears in errors.
Before handing the feed to the dashboard, verify:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/api/nodes
curl http://127.0.0.1:8000/api/metrics
uv run pytest tests/test_main.py tests/test_telemetry.py -q
```


- `topology`: full `NodeProfile` objects.
- `metrics`: full `NodeMetrics` snapshots. Metal unified memory is reported as
  RAM and is not duplicated as separate free VRAM.
- `event`: a human-readable cluster event for the dashboard log.
- `flow`: a job movement from the controller to a target node.

## Current boundaries

Control-plane state remains in memory, so restarting the server resets nodes,
jobs and telemetry history. The pool secret is development-only authentication;
production authentication and durable state are outside the current scope.
