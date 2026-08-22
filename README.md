# DAIN

DAIN turns a small group of ordinary computers into one addressable pool for
model inference, jobs, memory and files. A control plane holds the registry,
job queue and telemetry; node agents join over mDNS and serve work; a Next.js
dashboard watches the whole thing live.

Requirements: Python 3.12+, `uv`, and Node 20+ for the dashboard.

---

## Quick start (one machine)

Two terminals. This gets you a working dashboard with no cluster hardware.

```bash
# 1. control plane
uv sync
DAIN_MOCK_NODES=1 uv run uvicorn ctl.main:app --reload --port 8000

# 2. dashboard
cd agent/dain-dashboard
npm install
cp .env.example .env.local     # defaults already point at 127.0.0.1:8000
npm run dev                    # http://localhost:3000
```

`DAIN_MOCK_NODES=1` seeds four fake nodes so the UI has something to draw.
**Leave it unset for any real cluster** — otherwise ctl advertises `gpu-01`,
`office-01`, `office-02` and `mac-01` whether or not those machines exist, and
a node that genuinely joins is buried among fakes that never go offline.

---

## Operating the control plane

```bash
uv run uvicorn ctl.main:app --host 0.0.0.0 --port 8000
```

`--host 0.0.0.0`, not `127.0.0.1`, or nothing off that box can reach it.

- API docs: <http://127.0.0.1:8000/docs> — canonical for request bodies
- Health: `/health` · Nodes: `/api/nodes` · Metrics: `/api/metrics`
- Feed: `ws://127.0.0.1:8000/feed`

| Variable | Effect |
| --- | --- |
| `DAIN_MOCK_NODES` | `1` seeds four fake nodes. Off by default. |
| `DAIN_POOL_SECRET` | HMAC secret shared with node agents. Development default only. |
| `DAIN_LLAMA_METRICS_URL` | llama-server Prometheus endpoint to scrape, e.g. `http://gpu-01:8080/metrics`. |

State is in memory: restarting ctl resets nodes, jobs and telemetry history.

### Interface

| Method | Path | Behaviour |
| --- | --- | --- |
| `GET` | `/api/nodes` | Current in-memory node registry. |
| `POST` | `/api/nodes/join/challenge` | One-use, 30-second join nonce. |
| `POST` | `/api/nodes/join` | Verifies the nonce HMAC, registers, returns a short-lived bearer token. |
| `DELETE` | `/api/nodes/{id}` | Removes a node; `204` or `404`. |
| `GET` | `/api/plan?model=...` | **Still the deterministic mock** — see Known gaps. |
| `POST` | `/api/jobs` | Queues a job and fans it out. |
| `GET` | `/api/jobs/{id}` | Stored job or `404`. |
| `GET` | `/api/metrics` | Latest node and llama-server metrics, 60-sample histories, polling errors. |
| `POST` | `/api/race` | Deterministic serial or fan-out result. |
| `WS` | `/feed` | Streams `topology`, `metrics`, `event` and `flow` frames. |

---

## Operating the dashboard

```bash
cd agent/dain-dashboard
npm install
npm run dev          # http://localhost:3000
```

Endpoints come from `.env.local` (gitignored; copy `.env.example`):

```bash
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000/api
NEXT_PUBLIC_FEED_URL=ws://127.0.0.1:8000/feed
```

Three things to get right:

1. **The API base must end in `/api`.** ctl serves REST under `/api` but mounts
   the socket at bare `/feed`. Omit it and every REST call 404s while the feed
   keeps working — a half-broken state that reads like a backend fault. The
   Settings page warns when it looks wrong.
2. **`NEXT_PUBLIC_*` is inlined at build time.** After editing `.env.local`,
   **restart `npm run dev`**; reloading the page is not enough.
3. **Use the LAN IP, not `gpu-01`,** unless mDNS resolves from whichever laptop
   runs the browser.

If both are unset the app falls back to `127.0.0.1:8000` rather than failing
silently. A dropped feed retries every 2 s and says so in the sidebar.

| Page | What it shows |
| --- | --- |
| `/dashboard` | Node cards: static profile plus live CPU load, RAM, GPU%, VRAM and running jobs at 2 Hz. |
| `/create-job` | Submit any job kind. The form is kind-aware — see below. |
| `/jobs` | Job table with fan-out, per-shard results, generated text and bench throughput. |
| `/settings` | Resolved endpoints, connection state, known gaps. Start here when something looks wrong. |

Checks:

```bash
cd agent/dain-dashboard
npx tsc --noEmit     # not `npx tsc` — that installs an unrelated package
npm run lint
npm run build
```

---

## Job kinds

The queue dispatches each kind to a route on the node agent. **Each takes a
different payload**, and the node validates it — a wrong shape is a `422`, not
a silent no-op.

| Kind | Node route | Payload | Notes |
| --- | --- | --- | --- |
| `exec` | `/exec` | `{"argv": ["uname","-a"]}` | bubblewrap-sandboxed. Also `cwd`, `timeout_s`. |
| `index` | `/index` | `{}` | Re-scans `DAIN_INDEX_ROOT`. `503` if the embedding cache is missing. |
| `search` | `/search` | `{"query": "...", "limit": 5}` | `409` until `index` has run on that node. |
| `infer` | `/infer` | `{"prompt": "...", "max_tokens": 256}` | Needs an inference backend — see below. |
| `bench` | `/bench` | `{"repetitions": 3}` | Runs `llama-bench`. Needs a GGUF on the node. |

```bash
curl -X POST http://127.0.0.1:8000/api/jobs \
  -H 'content-type: application/json' \
  -d '{"kind":"exec","payload":{"argv":["uname","-a"]},"fanout":3}'
```

Fan-out with no `tasks`/`items` list in the payload sends the **same** payload
to every node — which is exactly the replica demo: one prompt, five machines,
compare tok/s.

The controller signs every node-job request with a short-lived HMAC covering
the whole body. Nodes reject missing, stale, or tampered requests before
touching the filesystem. The pool secret is never sent.

---

## Inference

Two topologies. They are genuinely different and use different code paths.

### Pipeline — one model spanning the cluster

One `llama-server` on the head, `--rpc` to the workers. Clients talk to **one
endpoint**, the head's `:8080`. **The job queue is not involved.**

```bash
./scripts/serve_head.py --model castoff              # start it
./scripts/serve_head.py --model castoff --dry-run    # print the argv first
./scripts/serve_head.py --model castoff --watch      # restart on membership change
```

It asks ctl who is alive, orders the head first and the workers
deterministically, and builds the command through
`infer.launch.llama_server_command`. Do not hand-write this: `--rpc` order
defines what `--tensor-split` means positionally, so an edited command silently
puts layers on the wrong machines.

Pass the **key** from `infer/models.toml` (`castoff`), never the role
(`castoff_capacity`) — the key is the directory name.

```bash
curl http://gpu-01:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"Write a haiku about idle office PCs."}]}'
```

Only the head needs the model file; it streams each worker its slice at load,
and `rpc-server -c` caches it there. Workers need only `rpc-server`, which the
node agent starts on join.

### Fan-out — every node runs its own copy

The queue *is* involved. Each node agent needs a backend; set **one** of:

```bash
DAIN_INFER_MODEL=/srv/dain/models/replica/Qwen3-4B-Instruct-2507-Q4_K_M.gguf
#   this node starts and supervises its own llama-server on loopback

DAIN_LLAMA_ENDPOINT=http://gpu-01:8080
#   this node forwards to an existing head instead
```

Set neither and `/infer` returns `503` naming both. Set both and the explicit
endpoint wins. Startup does not block on model load — the node still joins the
pool, and `/infer` polls readiness per request, so the first call after boot
pays the load cost.

### Benchmarking

```bash
DAIN_BENCH_MODEL=/srv/dain/models/calibration/Qwen3-0.6B-Q4_K_M.gguf
```

Falls back to `DAIN_INFER_MODEL`. `/bench` runs `llama-bench -p 512 -n 128 -r 3`
and returns measured prefill and decode tok/s. Those defaults match
`infer.launch.llama_bench_command` deliberately — the INF-6 measurement and the
SCH-1 calibration probe must be the same benchmark or their numbers are not
comparable. `node.bench.measure()` is the reusable entry point for SCH-1.

---

## Installing a node

```bash
export DAIN_POOL_SECRET='replace-with-the-pool-secret'
curl -fsSL \
  https://raw.githubusercontent.com/GuardianCoding/DAIN/main/scripts/install_node.sh \
  | sudo --preserve-env=DAIN_POOL_SECRET bash
```

Installs Python 3.12, locked dependencies, bubblewrap isolation and a
restarting `dain-node` systemd service. Re-running safely updates in place.

Nodes discover ctl over mDNS. Set `DAIN_CTL=host:8000` only when multicast is
unavailable. The secret lives in `/etc/dain/node.env` mode `0600`, never in the
unit file or repository.

| Variable | Effect |
| --- | --- |
| `DAIN_CTL` | Control plane `host:port`, when mDNS is unavailable. |
| `DAIN_FABRIC_IFACE` | Interface to report and bind `rpc-server` to. |
| `DAIN_LLAMA_BIN` | llama.cpp binary directory (default `/opt/dain/llama.cpp/build/bin`). |
| `DAIN_INDEX_ROOT` | Directory this node indexes. |
| `DAIN_INFER_MODEL` / `DAIN_LLAMA_ENDPOINT` | Inference backend, above. |
| `DAIN_BENCH_MODEL` | Model for `/bench`. |

Node routes: `/health` `/profile` `/metrics` `/index` `/search` `/exec`
`/infer` `/bench`.

---

## Distributed file search

Each node indexes only `DAIN_INDEX_ROOT`. Run an `index` job before searching;
a cold `/search` returns `409` rather than starting an unbounded filesystem
walk inside the two-second timeout.

Refreshes are single-flight and bounded to 10,000 files, 256 MiB total, 1 MiB
per file. Embedding is the 67 MB `BAAI/bge-small-en-v1.5` model through
**FastEmbed**; the installer caches it into `/var/cache/dain/fastembed` and
reloads it with network access disabled. Runtime downloads are off, so a
missing cache fails installation rather than the first demo search. If a node
somehow lacks it:

```bash
./scripts/fetch_embed_model.py --check    # report without downloading
./scripts/fetch_embed_model.py
```

This is a different stack from `infer/models.toml`'s `embed` entry, which is
GGUF weights for the inference fabric. Two models, two caches; only FastEmbed
serves `/index`.

---

## WebSocket frames

First frame is always a topology snapshot; metrics follow at 2 Hz. Topology,
event and flow frames are emitted when their state changes.

```text
topology -> metrics -> [topology | event | flow] -> metrics -> event -> flow -> ...
```

- `topology`: full `NodeProfile` objects — static, frozen at join.
- `metrics`: full `NodeMetrics` snapshots — live. Both carry `ram_free_mb` and
  they mean different things; the dashboard keeps them separate deliberately.
  Metal unified memory is reported as RAM, never duplicated as free VRAM.
- `event`: a human-readable cluster event for the dashboard log.
- `flow`: a job movement from the controller to a target node.

---

## Repository layout

```
contracts.py              frozen API surface — NodeProfile, Assignment, Job
cluster.toml              discovery, membership, paths, pinned commit, planning fixture

ctl/main.py               FastAPI control plane :8000
ctl/registry.py           node registry, heartbeats, offline sweep, replan events
ctl/queue.py              async job queue, fan-out, sharding, HMAC-signed dispatch
ctl/telemetry.py          telemetry fan-in; scrapes node :9100 and llama-server :8080
ctl/auth.py               request signing
ctl/mock.py               deterministic mock — still backs GET /api/plan

node/dain_node.py         node agent :9100; mDNS join, heartbeat, supervises rpc-server
node/infer.py             /infer backend — supervises or forwards to llama-server
node/bench.py             /bench — runs llama-bench, parses it (also SCH-1's parse)
node/index.py             file index + search
node/sandbox.py           bubblewrap-isolated /exec
node/discovery.py         mDNS
node/auth.py              verifies controller signatures

sched/plan.py             assign-by-speed then repair. Correct. Not wired in.
sched/cost.py             pure cost/memory maths
infer/models.toml         the model ladder — 8 models by role and download priority
infer/launch.py           pure llama.cpp command builders (returns argv, runs nothing)
infer/memory.py           usable memory, KV cache, capacity_report()
infer/bench.py            benchmark record schema (benchmarks.csv)

agent/dain-dashboard/     Next.js 16 + React 19 dashboard
  lib/config.ts           the only reader of NEXT_PUBLIC_*
  lib/feed/               one WebSocket for the whole app, accumulated state

scripts/serve_head.py         start the pipeline head across live membership
scripts/build_llama.sh        two builds on the head from one commit
scripts/distribute_llama.sh   push binaries to workers, verify all nodes
scripts/install_node.sh       one-command node install + systemd unit
scripts/inventory.sh          per-node memory topology (DIMM, XMP, WSL checks)
scripts/check_fabric.py       multicast/RPC fabric verification + speedtest
scripts/fetch_models.py       priority-ordered resumable model downloader
scripts/fetch_embed_model.py  pre-cache the FastEmbed model /index needs
scripts/loadtest_node.sh      known-level CPU load, to prove the telemetry path
```

---

## Checks

```bash
uv run ruff format --check ctl tests
uv run ruff check ctl tests
uv run pytest
```

---

## Known gaps

- **`GET /api/plan` still returns the mock.** `sched/plan.py` is finished but
  unwired: every node reports `tg_tok_s = 0.0`, so the real scheduler raises
  `no node has a measured tg_tok_s`. `node.bench.measure()` now produces
  exactly that number — wiring it at node start (SCH-1) is what unblocks this.
- **Nothing starts the pipeline head automatically.** `serve_head.py` is run by
  hand on the head node, by design.
- **Control-plane state is in memory.** Restarting resets nodes, jobs and
  telemetry.
- **The pool secret is development-only authentication.** Production auth and
  durable state are out of scope.
- **`cluster.toml` `pinned_commit`** may still be `UNVERIFIED`. llama.cpp's RPC
  protocol has no version negotiation, so nodes built from different commits
  connect happily and then hang or return noise. `serve_head.py` warns.
