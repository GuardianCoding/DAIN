#!/usr/bin/env bash
# Drive one DAIN node's CPU to a known level, so you can prove the
# node -> :9100/metrics -> ctl telemetry -> dashboard path end to end.
#
#   ./scripts/loadtest_node.sh              # saturate every core for 30s
#   ./scripts/loadtest_node.sh 60           # ...for 60s
#   ./scripts/loadtest_node.sh 60 4         # ...4 workers only, for 60s
#
# WHY THIS EXISTS:
#
# The obvious test — one `python -c 'while True: pass'` — is a trap, and it
# already cost us an afternoon. A single busy loop pins exactly one core, so on
# lachlan-node's 12 logical cores the honest reading is 1/12 = 8.3%. We saw a
# 7.1% peak and read it as "telemetry is broken". It wasn't; the load was.
#
# So this defaults to one worker per logical core and PRINTS the number it
# expects, which turns "is telemetry working?" into a comparison rather than a
# guess. Confirm from any machine that can reach ctl:
#
#   curl -s http://<ctl-host>:8000/api/metrics | python3 -m json.tool
#
# cpu_percent for this node should climb toward that number within a couple of
# poll cycles (TelemetryFanIn samples at 2 Hz).
#
# Load only — it allocates nothing. To exercise the RAM bar, load a model.

set -euo pipefail

DURATION="${1:-30}"

detect_cores() {
  if command -v nproc >/dev/null 2>&1; then
    nproc                                   # Linux, incl. WSL
  elif command -v sysctl >/dev/null 2>&1; then
    sysctl -n hw.logicalcpu                 # macOS
  else
    echo 1
  fi
}

cores="$(detect_cores)"
WORKERS="${2:-$cores}"

for value in "$DURATION" "$WORKERS"; do
  case "$value" in
    ''|*[!0-9]*) echo "usage: $0 [duration_seconds] [workers]" >&2; exit 2 ;;
  esac
done

if [ "$WORKERS" -lt 1 ] || [ "$DURATION" -lt 1 ]; then
  echo "usage: $0 [duration_seconds] [workers]  (both must be >= 1)" >&2
  exit 2
fi

PYTHON="$(command -v python3 || command -v python || true)"
if [ -z "$PYTHON" ]; then
  echo "no python3 on PATH — cannot generate load" >&2
  exit 1
fi

pids=()

# Reap the workers on ANY exit path, Ctrl-C included. Left-behind busy loops
# poison every measurement taken on this node afterwards, which is a miserable
# thing to be debugging an hour later.
cleanup() {
  trap - EXIT INT TERM
  for pid in "${pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
  echo "stopped ${#pids[@]} worker(s)"
}
trap cleanup EXIT INT TERM

expected=$(( WORKERS * 100 / cores ))
[ "$expected" -gt 100 ] && expected=100

echo "node has ${cores} logical core(s); starting ${WORKERS} worker(s) for ${DURATION}s"
echo "expect cpu_percent to approach ~${expected}% in /api/metrics"

for _ in $(seq 1 "$WORKERS"); do
  "$PYTHON" -c 'while True: pass' &
  pids+=($!)
done

sleep "$DURATION"
