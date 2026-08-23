# DAIN

DAIN turns a small group of ordinary computers into one addressable pool for
model inference, jobs, memory and files. A control plane holds the registry,
job queue and telemetry; node agents join over mDNS and serve work; a Next.js
dashboard watches the whole thing live; and an agent sits on top whose tools
*are* the cluster — it answers "which machine has the most free memory?" by
reading the live registry, not by guessing.

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

The fake nodes have no agent listening on `:9100`, so jobs dispatched to them
fail with connection errors. That is the correct result, and it exercises the
error paths; for work that has to actually run, join a real node.

The agent needs a `llama-server` as well as ctl, so it is not part of this
two-terminal start — see [The agent](#the-agent).

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
| `GET` | `/api/plan?model=&context=&slots=` | The real scheduler over live membership. `404` unknown model, `503` uncalibrated pool or missing KV geometry. |
| `POST` | `/api/jobs` | Queues a job and fans it out. |
| `GET` | `/api/jobs/{id}` | Stored job or `404`. |
| `GET` | `/api/metrics` | Latest node and llama-server metrics, 60-sample histories, polling errors. |
| `POST` | `/api/race` | Deterministic serial or fan-out result. |
| `WS` | `/feed` | Streams `topology`, `metrics`, `event` and `flow` frames. |

### Placement

`GET /api/plan` runs `sched.plan()` against whoever is actually alive.

```bash
curl 'http://127.0.0.1:8000/api/plan?model=castoff&context=8192&slots=1'
```

`context` and `slots` are parameters because the KV cache scales with both, and
**they must match what `llama_server_command()` is launched with.** Plan at 8k
and launch at 128k × 4 and the split is wrong by 64×, silently.

Pass the **key** from `infer/models.toml`. Roles are accepted as aliases and
canonicalised, but the key is what reaches `Assignment.model_id`. Three models
(`working`, `mtp`, `working_spare`) have no KV geometry yet and return `503`
naming what to read off the GGUF header — see Known gaps.

A `503` saying `no node has a measured tg_tok_s` means the pool has not
calibrated. Nodes measure themselves with `llama-bench` at start; one with no
`DAIN_BENCH_MODEL` or `DAIN_INFER_MODEL` joins uncalibrated and cannot be
placed, though it still serves `exec`, `index` and `search`.

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

| Kind | Node route | Payload | Dispatch timeout | Notes |
| --- | --- | --- | --- | --- |
| `exec` | `/exec` | `{"argv": ["uname","-a"]}` | 90 s | bubblewrap-sandboxed. Also `cwd`, `timeout_s`. |
| `index` | `/index` | `{}` | 30 s | Re-scans `DAIN_INDEX_ROOT`. `503` if the embedding cache is missing. |
| `search` | `/search` | `{"query": "...", "limit": 5}` | 30 s | `409` until `index` has run on that node. |
| `infer` | `/infer` | `{"prompt": "...", "max_tokens": 256}` | 540 s | Needs an inference backend — see below. |
| `bench` | `/bench` | `{"repetitions": 3}` | 960 s | Runs `llama-bench`. Needs a GGUF on the node. |

**The timeouts are derived, not chosen.** Each is the node's own ceiling for
that kind plus a margin (`ctl.queue.DEFAULT_TIMEOUTS_S`). Undercut one and a
`ReadTimeout` is indistinguishable from a dead node: the queue retries, then
reassigns the same payload to every remaining machine, so one slow generation
becomes the same prompt running on the whole pool and returning nothing.

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

The command includes **`--jinja`**, which makes llama.cpp use the GGUF's own
chat template. Without it llama.cpp substitutes a built-in template carrying no
tool-call grammar, and the model replies with prose *describing* the tool call
it would like to make. Everything in [The agent](#the-agent) depends on it.
`--placement` uses `GET /api/plan` instead of llama.cpp's `--fit on`.

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

## The agent

An agent whose tool surface is the pool itself. Ask it which machine has the
most free memory and it answers *correctly*, because the tool is the live
registry rather than a guess.

```bash
./scripts/run_agent.py                                    # interactive
./scripts/run_agent.py --once "which machine is busiest?"
./scripts/run_agent.py --tools                            # list the tools
```

It needs **two endpoints, and they are not the same one**:

| | Where | What for |
| --- | --- | --- |
| Thinking | the head, `:8080` | A direct `llama-server` call. No queue involved. |
| Acting | ctl, `:8000` | Every tool call becomes a job. |

Mixing those up gives you a loop that queues a job to think about queueing a
job. The payoff of the split: because tool calls are ordinary jobs, they emit
`flow` and `event` frames on `/feed`, so **the dashboard already draws the
agent working** without anyone writing visualisation code.

| Variable | Effect |
| --- | --- |
| `DAIN_CTL` | Control plane `host:port` (default `127.0.0.1:8000`). |
| `DAIN_AGENT_ENDPOINT` | llama-server head `host:port` (default `127.0.0.1:8080`). |
| `DAIN_AGENT_MODEL` | Model name sent to llama-server, which ignores it. |

### The tools

| Tool | Backed by | Notes |
| --- | --- | --- |
| `cluster_status()` | `/api/nodes` + `/api/metrics` | Merged. Prefers live telemetry over the join-time profile. |
| `plan_placement(model)` | `/api/plan` | How the scheduler would split a model, and why. Plans only. |
| `search_files(query)` | `search` job, fanned out | Every machine searches its own disk; hits come back `node:path`. |
| `run_command(argv, node)` | `exec` job | Sandboxed. Allowlist imported from `node.sandbox`, not copied. |
| `ask_pool(prompts)` | N pinned `infer` jobs | One prompt per machine, concurrently. |

Three behaviours worth knowing, each of them deliberate:

- **Tools return prose, not JSON.** A 20B model reading `ram_free=10.0GiB` is
  markedly more reliable than the same model parsing an object and picking the
  right key.
- **`call_tool` never raises.** A `503` is the normal case — nodes calibrate,
  models load, indexes go cold — so the reason comes back as an ordinary tool
  result for the model to read and adapt to. The `503` texts were written to be
  readable for exactly this.
- **Turns are capped at six.** Small models will call the same tool five times.
  Past the cap it says it could not determine the answer rather than guessing.
  An identical repeated call is answered from the first result.

`ask_pool` sends N separate single-prompt jobs rather than one fan-out job.
`ctl.queue._split_payload` shards on a `tasks` list, but `/infer` reads
`payload["prompt"]` and `422`s without it, so the sharded shape fails today.
Pinning each job to a named node also stops the least-busy ranking landing
every prompt on the same machine.

### The showdown page

```bash
python3 -m http.server 8123 --directory agent
# then open http://127.0.0.1:8123/showdown.html?ctl=gpu-01:8000&head=gpu-01:8080
```

The same prompt, at the same time, on one ordinary node running the 4B replica
versus the whole pool running a model no single machine here can hold. No build
step and no framework on purpose — the dashboard is the thing that breaks at
3am. The left pane goes through ctl as a job, so it also appears on `/feed`.

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

**Set `DAIN_NODE_ID`.** It defaults to `hostname -s`, and the pool's names are
not hostnames: `serve_head.py --head` defaults to `gpu-01`, `os_class_map()`
joins `cluster.toml`'s `[[planning.nodes]]` to live membership by id, and the
demo script says these names out loud. A node that joins as its hostname has no
entry in that table. Worse, hostnames are not guaranteed unique — two machines
here answer to `password3`, and the registry keys on id, so the second to join
would silently replace the first.

```bash
DAIN_NODE_ID=office-01 DAIN_FABRIC_IFACE=enp1s0 \
  DAIN_POOL_SECRET='...' sudo -E ./scripts/install_node.sh
```

| Variable | Effect |
| --- | --- |
| `DAIN_NODE_ID` | This node's id in the pool. **Set it** — see above. |
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
ctl/mock.py               deterministic fixtures for DAIN_MOCK_NODES and /api/race

node/dain_node.py         node agent :9100; mDNS join, heartbeat, supervises rpc-server
node/infer.py             /infer backend — supervises or forwards to llama-server
node/bench.py             /bench — runs llama-bench, parses it (also SCH-1's parse)
node/index.py             file index + search
node/sandbox.py           bubblewrap-isolated /exec
node/discovery.py         mDNS
node/auth.py              verifies controller signatures

sched/plan.py             assign-by-speed then repair; behind GET /api/plan
sched/cost.py             pure cost/memory maths (includes COMPUTE_OVERHEAD_MB)
infer/models.toml         the model ladder — 8 models by role and download priority
infer/launch.py           pure llama.cpp command builders (returns argv, runs nothing)
infer/spec.py             scheduler_spec() — bridges the ladder to sched.plan()
infer/memory.py           usable memory, KV cache, capacity_report()
infer/bench.py            benchmark record schema (benchmarks.csv)

agent/client.py           the only thing that speaks HTTP to ctl; submit-and-poll
agent/tools.py            tool definitions + the ctl calls behind them
agent/fanout.py           ask_pool — one prompt per machine, concurrently
agent/loop.py             the conversation loop: prompt -> tool calls -> answer
agent/showdown.html       one node vs the pool, side by side. No build step.
agent/dain-dashboard/     Next.js 16 + React 19 dashboard
  lib/config.ts           the only reader of NEXT_PUBLIC_*
  lib/feed/               one WebSocket for the whole app, accumulated state

scripts/run_agent.py          talk to the pool; the agent's operator entry point
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
uv run pytest                             # 457 passed
uv run ruff check agent ctl sched         # clean
uv run ruff format --check agent ctl      # clean
```

**Neither lint target is clean repo-wide, and the difference is historical, not
a regression.** `uv run ruff check .` reports 10 findings across `infer/`,
`node/`, `scripts/` and `tests/`; `ruff format --check .` would reformat 16
files in those same areas, which were written in a wider style than `ruff
format` produces. Both are worth clearing, but doing so is a large diff
touching files unrelated to whatever change is in flight, so the commands above
are scoped to the directories that currently pass. Widen them as areas are
cleaned up.

---

## Known gaps

- **Three models cannot be planned.** `working`, `mtp` and `working_spare` have
  `total_layers` but no entry in `infer.spec.KV_GEOMETRY`, so `GET /api/plan`
  returns `503` naming what to read off the GGUF header. `working` is most of
  the demo. `calibration`, `replica`, `castoff`, `headline` and `embed` plan
  fine.
- **Every KV geometry that exists is `source="estimated"`.**
  `infer.spec.unverified_models()` lists them. They come from spec sheets, not
  from a loader log, so any capacity number derived from them is an estimate.
- **The agent has not been run against a real model.** Its plumbing is tested
  end to end against a stub that emits well-formed tool calls. Whether
  gpt-oss-20b actually emits structured calls through llama.cpp's `--jinja`
  path is unverified — check `llama-server --help | grep jinja` on the pinned
  build, then watch for a `[tool]` line from `run_agent.py`.
- **Nothing starts the pipeline head automatically.** `serve_head.py` is run by
  hand on the head node, by design.
- **Control-plane state is in memory.** Restarting resets nodes, jobs and
  telemetry.
- **The pool secret is development-only authentication.** Production auth and
  durable state are out of scope.
- **`ruff format` fails on 16 pre-existing files.** See Checks.
