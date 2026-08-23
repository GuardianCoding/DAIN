#!/usr/bin/env bash
# Install persistent controller, head-node, and llama-server services on Linux.
# Safe to run repeatedly. Run from the DAIN checkout with sudo -E.
#
# Required:
#   DAIN_POOL_SECRET=... DAIN_FABRIC_IFACE=enp8s0 \
#   DAIN_FABRIC_HOST=192.168.50.1 DAIN_HEAD_NODE_ID=fedora-test \
#   DAIN_MODEL_ID=castoff DAIN_MODEL_FILE=gpt-oss-20b-MXFP4.gguf \
#   DAIN_BENCH_MODEL=/srv/dain/models/calibration/Qwen3-0.6B-Q4_K_M.gguf \
#   sudo -E ./scripts/install_head.sh
#
# Optional:
#   DAIN_HEAD_EXCLUDE=mac-01,node-104  # comma-separated RPC exclusions

set -Eeuo pipefail

readonly RUNTIME_USER="${DAIN_RUNTIME_USER:-${SUDO_USER:-$(id -un)}}"
readonly RUNTIME_GROUP="${DAIN_RUNTIME_GROUP:-$RUNTIME_USER}"
runtime_home_default="$HOME"
if command -v getent >/dev/null 2>&1; then
  passwd_home="$(getent passwd "$RUNTIME_USER" | cut -d: -f6 || true)"
  if [[ -n "$passwd_home" ]]; then
    runtime_home_default="$passwd_home"
  fi
fi
readonly RUNTIME_HOME="${DAIN_RUNTIME_HOME:-$runtime_home_default}"
readonly APP_DIR="${DAIN_APP_DIR:-$PWD}"
readonly CONFIG_DIR="${DAIN_CONFIG_DIR:-/etc/dain}"
readonly ENV_FILE="${CONFIG_DIR}/head.env"
readonly SYSTEMD_DIR="${DAIN_SYSTEMD_DIR:-/etc/systemd/system}"
readonly CTL_UNIT="${SYSTEMD_DIR}/dain-ctl.service"
readonly NODE_UNIT="${SYSTEMD_DIR}/dain-node.service"
readonly HEAD_UNIT="${SYSTEMD_DIR}/dain-head.service"
readonly CTL_PORT="${DAIN_CTL_PORT:-8000}"
readonly NODE_PORT="${DAIN_NODE_PORT:-9100}"
readonly LLAMA_PORT="${DAIN_LLAMA_PORT:-8080}"
readonly LLAMA_BIN="${DAIN_LLAMA_BIN:-/opt/dain/llama.cpp/build/bin}"
readonly INDEX_ROOT="${DAIN_INDEX_ROOT:-/tmp/dain-index}"
readonly EMBED_CACHE="${DAIN_EMBED_CACHE:-${RUNTIME_HOME}/.cache/dain/fastembed}"
readonly READY_TIMEOUT="${DAIN_HEAD_READY_TIMEOUT:-900}"

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
elif [[ $# -gt 0 ]]; then
  echo "usage: install_head.sh [--dry-run]" >&2
  exit 2
fi

log() {
  printf '==> %s\n' "$*"
}

run() {
  if (( DRY_RUN )); then
    printf '+ '
    printf '%q ' "$@"
    printf '\n'
    return 0
  fi
  "$@"
}

require_value() {
  local name="$1"
  local value="${!name:-}"
  if [[ -z "$value" ]]; then
    echo "${name} is required" >&2
    exit 2
  fi
  if [[ "$value" == *$'\n'* || "$value" == *$'\r'* ]]; then
    echo "${name} must not contain a newline" >&2
    exit 2
  fi
}

environment_value() {
  local value="$1"
  value="${value//\\/\\\\}"
  value="${value//\"/\\\"}"
  printf '"%s"' "$value"
}

require_configuration() {
  local required
  for required in \
    DAIN_POOL_SECRET \
    DAIN_FABRIC_IFACE \
    DAIN_FABRIC_HOST \
    DAIN_HEAD_NODE_ID \
    DAIN_MODEL_ID \
    DAIN_MODEL_FILE \
    DAIN_BENCH_MODEL
  do
    require_value "$required"
  done

  if [[ "$RUNTIME_USER" == "root" ]]; then
    echo "Set DAIN_RUNTIME_USER when running directly as root" >&2
    exit 2
  fi
}

check_platform() {
  if (( DRY_RUN )); then
    return
  fi
  if [[ "$(uname -s)" != "Linux" || $EUID -ne 0 ]]; then
    echo "Run this Linux installer with sudo -E" >&2
    exit 2
  fi
  if [[ ! -x "${APP_DIR}/.venv/bin/python" ]]; then
    echo "${APP_DIR}/.venv/bin/python is missing; run uv sync first" >&2
    exit 2
  fi
  if [[ ! -x "${LLAMA_BIN}/llama-server" ]]; then
    echo "${LLAMA_BIN}/llama-server is missing" >&2
    exit 2
  fi
  if [[ ! -f "$DAIN_BENCH_MODEL" ]]; then
    echo "DAIN_BENCH_MODEL does not exist: ${DAIN_BENCH_MODEL}" >&2
    exit 2
  fi
  if [[ ! -f "/srv/dain/models/${DAIN_MODEL_ID}/${DAIN_MODEL_FILE}" ]]; then
    echo "head model does not exist under /srv/dain/models" >&2
    exit 2
  fi
}

write_environment_file() {
  log "Writing protected head configuration"
  if (( DRY_RUN )); then
    printf '+ install protected environment file %q (secret redacted)\n' "$ENV_FILE"
    return
  fi

  install -d -m 0755 "$CONFIG_DIR"
  local temporary
  temporary="$(mktemp "${CONFIG_DIR}/head.env.XXXXXX")"
  chown root:"$RUNTIME_GROUP" "$temporary"
  chmod 0640 "$temporary"
  {
    printf 'DAIN_POOL_SECRET=%s\n' "$(environment_value "$DAIN_POOL_SECRET")"
    printf 'DAIN_CTL=%s\n' "$(environment_value "${DAIN_FABRIC_HOST}:${CTL_PORT}")"
    printf 'DAIN_CTL_ADVERTISE_HOST=%s\n' "$(environment_value "$DAIN_FABRIC_HOST")"
    printf 'DAIN_FABRIC_IFACE=%s\n' "$(environment_value "$DAIN_FABRIC_IFACE")"
    printf 'DAIN_INDEX_ROOT=%s\n' "$(environment_value "$INDEX_ROOT")"
    printf 'DAIN_EMBED_CACHE=%s\n' "$(environment_value "$EMBED_CACHE")"
    printf 'HF_HUB_OFFLINE="1"\n'
    printf 'DAIN_LLAMA_BIN=%s\n' "$(environment_value "$LLAMA_BIN")"
    printf 'DAIN_LLAMA_ENDPOINT=%s\n' \
      "$(environment_value "http://${DAIN_FABRIC_HOST}:${LLAMA_PORT}")"
    printf 'DAIN_LLAMA_METRICS_URL=%s\n' \
      "$(environment_value "http://${DAIN_FABRIC_HOST}:${LLAMA_PORT}/metrics")"
    printf 'DAIN_BENCH_MODEL=%s\n' "$(environment_value "$DAIN_BENCH_MODEL")"
    printf 'DAIN_HEAD_NODE_ID=%s\n' "$(environment_value "$DAIN_HEAD_NODE_ID")"
    printf 'DAIN_MODEL_ID=%s\n' "$(environment_value "$DAIN_MODEL_ID")"
    printf 'DAIN_MODEL_FILE=%s\n' "$(environment_value "$DAIN_MODEL_FILE")"
    printf 'DAIN_HEAD_EXCLUDE=%s\n' \
      "$(environment_value "${DAIN_HEAD_EXCLUDE:-}")"
    printf 'DAIN_CTL_PORT=%s\n' "$(environment_value "$CTL_PORT")"
    printf 'DAIN_NODE_PORT=%s\n' "$(environment_value "$NODE_PORT")"
  } >"$temporary"
  mv -f "$temporary" "$ENV_FILE"
}

write_units() {
  log "Installing systemd services"
  if (( DRY_RUN )); then
    printf '+ install systemd units %q %q %q\n' "$CTL_UNIT" "$NODE_UNIT" "$HEAD_UNIT"
    return
  fi

  cat >"$CTL_UNIT" <<EOF
[Unit]
Description=DAIN control plane
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${RUNTIME_USER}
Group=${RUNTIME_GROUP}
WorkingDirectory=${APP_DIR}
EnvironmentFile=${ENV_FILE}
ExecStart=${APP_DIR}/.venv/bin/uvicorn ctl.main:app --host 0.0.0.0 --port \${DAIN_CTL_PORT}
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

  cat >"$NODE_UNIT" <<EOF
[Unit]
Description=DAIN head compute node
After=network-online.target dain-ctl.service
Wants=network-online.target dain-ctl.service

[Service]
Type=simple
User=${RUNTIME_USER}
Group=${RUNTIME_GROUP}
WorkingDirectory=${APP_DIR}
EnvironmentFile=${ENV_FILE}
ExecStart=${APP_DIR}/.venv/bin/python -m node.dain_node --node-id \${DAIN_HEAD_NODE_ID} --port \${DAIN_NODE_PORT}
Restart=always
RestartSec=3
KillMode=control-group
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
EOF

  cat >"$HEAD_UNIT" <<EOF
[Unit]
Description=DAIN shared inference head
After=network-online.target dain-ctl.service dain-node.service
Wants=network-online.target dain-ctl.service dain-node.service
StartLimitIntervalSec=0

[Service]
Type=simple
User=${RUNTIME_USER}
Group=${RUNTIME_GROUP}
WorkingDirectory=${APP_DIR}
EnvironmentFile=${ENV_FILE}
ExecStart=${APP_DIR}/.venv/bin/python scripts/serve_head.py --model=\${DAIN_MODEL_ID} --file=\${DAIN_MODEL_FILE} --head=\${DAIN_HEAD_NODE_ID} --ctl=\${DAIN_CTL} --exclude=\${DAIN_HEAD_EXCLUDE} --watch
Restart=always
RestartSec=5
KillMode=control-group
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
EOF

  chmod 0644 "$CTL_UNIT" "$NODE_UNIT" "$HEAD_UNIT"
}

wait_for_url() {
  local label="$1"
  local url="$2"
  local timeout="$3"
  local deadline=$(( SECONDS + timeout ))
  until curl -fsS --connect-timeout 2 "$url" >/dev/null 2>&1; do
    if (( SECONDS >= deadline )); then
      echo "${label} did not become ready within ${timeout}s" >&2
      return 1
    fi
    sleep 2
  done
}

start_and_verify() {
  log "Starting controller and head node"
  run systemctl daemon-reload
  run systemctl enable dain-ctl.service
  run systemctl restart dain-ctl.service
  if (( ! DRY_RUN )); then
    wait_for_url controller "http://127.0.0.1:${CTL_PORT}/health" 60
  fi
  run systemctl enable dain-node.service
  run systemctl restart dain-node.service
  if (( ! DRY_RUN )); then
    wait_for_url node "http://${DAIN_FABRIC_HOST}:${NODE_PORT}/health" 240
  fi

  log "Starting shared inference head"
  run systemctl enable dain-head.service
  run systemctl restart dain-head.service
  if (( ! DRY_RUN )); then
    if ! wait_for_url inference-head "http://${DAIN_FABRIC_HOST}:${LLAMA_PORT}/health" "$READY_TIMEOUT"; then
      systemctl status dain-head.service --no-pager >&2 || true
      journalctl -u dain-head.service -n 60 --no-pager >&2 || true
      exit 1
    fi
  fi

  run systemctl is-active --quiet dain-ctl.service
  run systemctl is-active --quiet dain-node.service
  run systemctl is-active --quiet dain-head.service
}

main() {
  require_configuration
  check_platform
  write_environment_file
  write_units
  start_and_verify
  log "DAIN head services are enabled and healthy"
}

main
