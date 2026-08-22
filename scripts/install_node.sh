#!/usr/bin/env bash
# Install or update one Linux DAIN node. Safe to run repeatedly.
#
# Minimal invocation:
#   curl -fsSL https://raw.githubusercontent.com/GuardianCoding/DAIN/main/scripts/install_node.sh \
#     | sudo --preserve-env=DAIN_POOL_SECRET bash
#
# Optional explicit network setup:
#   DAIN_NODE_ID=office-01 DAIN_FABRIC_IFACE=enp1s0 \
#   DAIN_STATIC_IP_CIDR=10.20.30.11/24 DAIN_GATEWAY=10.20.30.1 \
#   DAIN_DNS=10.20.30.1 DAIN_FABRIC_CIDR=10.20.30.0/24 \
#   DAIN_MANAGE_FIREWALL=1 DAIN_POOL_SECRET='...' sudo -E ./scripts/install_node.sh

set -Eeuo pipefail

readonly SERVICE_NAME="dain-node"
readonly DAIN_USER="dain"
readonly INSTALL_ROOT="${DAIN_INSTALL_ROOT:-/opt/dain}"
readonly APP_DIR="${DAIN_APP_DIR:-${INSTALL_ROOT}/app}"
readonly UV_HOME="${INSTALL_ROOT}/uv"
readonly PYTHON_HOME="${INSTALL_ROOT}/python"
readonly CACHE_DIR="${DAIN_CACHE_DIR:-/var/cache/dain}"
readonly CONFIG_DIR="${DAIN_CONFIG_DIR:-/etc/dain}"
readonly ENV_FILE="${CONFIG_DIR}/node.env"
readonly UNIT_FILE="${DAIN_SYSTEMD_DIR:-/etc/systemd/system}/${SERVICE_NAME}.service"
readonly SCRATCH_ROOT="${DAIN_SCRATCH_ROOT:-/var/tmp/dain}"
readonly INDEX_ROOT="${DAIN_INDEX_ROOT:-/srv/dain/index}"
readonly EMBED_CACHE="${DAIN_EMBED_CACHE:-${CACHE_DIR}/fastembed}"
readonly EMBED_MODEL="${DAIN_EMBED_MODEL:-BAAI/bge-small-en-v1.5}"
readonly REPO_URL="${DAIN_REPO_URL:-https://github.com/GuardianCoding/DAIN.git}"
readonly REPO_REF="${DAIN_REF:-main}"
readonly UV_VERSION="${DAIN_UV_VERSION:-0.12.5}"

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
elif [[ $# -gt 0 ]]; then
  echo "usage: install_node.sh [--dry-run]" >&2
  exit 2
fi

started_at="${SECONDS}"

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

require_safe_value() {
  local name="$1"
  local value="$2"
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
  if [[ -z "${DAIN_POOL_SECRET:-}" ]]; then
    echo "DAIN_POOL_SECRET is required and is never written to the repository" >&2
    exit 2
  fi
  require_safe_value DAIN_POOL_SECRET "$DAIN_POOL_SECRET"
  require_safe_value DAIN_NODE_ID "${DAIN_NODE_ID:-$(hostname -s)}"

  if [[ -n "${DAIN_STATIC_IP_CIDR:-}" && -z "${DAIN_FABRIC_IFACE:-}" ]]; then
    echo "DAIN_FABRIC_IFACE is required with DAIN_STATIC_IP_CIDR" >&2
    exit 2
  fi
  if [[ "${DAIN_MANAGE_FIREWALL:-0}" == "1" && -z "${DAIN_FABRIC_CIDR:-}" ]]; then
    echo "DAIN_FABRIC_CIDR is required when DAIN_MANAGE_FIREWALL=1" >&2
    exit 2
  fi
}

check_platform() {
  if (( DRY_RUN )); then
    return
  fi
  if [[ "$(uname -s)" != "Linux" ]]; then
    echo "This installer supports Linux nodes only" >&2
    exit 2
  fi
  if (( EUID != 0 )); then
    echo "Run this installer as root (for example with sudo -E)" >&2
    exit 2
  fi
  if ! command -v apt-get >/dev/null 2>&1; then
    echo "This installer currently supports Debian/Ubuntu apt-based nodes" >&2
    exit 2
  fi
}

install_system_dependencies() {
  log "Installing Linux prerequisites"
  run apt-get update -qq
  run env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    bubblewrap ca-certificates curl git iproute2 python3 python3-venv ripgrep ufw util-linux
}

ensure_service_user() {
  if (( DRY_RUN )) || ! id "$DAIN_USER" >/dev/null 2>&1; then
    run useradd --system --home-dir "$INSTALL_ROOT" --shell /usr/sbin/nologin "$DAIN_USER"
  fi
  run install -d -m 0755 "$INSTALL_ROOT"
  run install -d -o "$DAIN_USER" -g "$DAIN_USER" -m 0750 "$CACHE_DIR" "$EMBED_CACHE"
  run install -d -o "$DAIN_USER" -g "$DAIN_USER" -m 0700 "$SCRATCH_ROOT"
  run install -d -o "$DAIN_USER" -g "$DAIN_USER" -m 0750 "$INDEX_ROOT"
}

fetch_agent() {
  log "Fetching DAIN ${REPO_REF}"
  if (( DRY_RUN )); then
    run git clone --depth 1 --branch "$REPO_REF" "$REPO_URL" "$APP_DIR"
    return
  fi

  if [[ -d "${APP_DIR}/.git" ]]; then
    git -C "$APP_DIR" fetch --depth 1 origin "$REPO_REF"
    git -C "$APP_DIR" checkout --detach --force FETCH_HEAD
  elif [[ -e "$APP_DIR" ]]; then
    echo "${APP_DIR} exists but is not a git checkout; refusing to overwrite it" >&2
    exit 2
  else
    git clone --depth 1 --branch "$REPO_REF" "$REPO_URL" "$APP_DIR"
  fi
}

install_python_environment() {
  log "Installing Python 3.12 environment from the locked dependencies"
  if [[ ! -x "${UV_HOME}/bin/uv" ]]; then
    run python3 -m venv "$UV_HOME"
    run "${UV_HOME}/bin/pip" install --disable-pip-version-check "uv==${UV_VERSION}"
  fi
  run env UV_CACHE_DIR="$CACHE_DIR" UV_PYTHON_INSTALL_DIR="$PYTHON_HOME" \
    "${UV_HOME}/bin/uv" python install 3.12
  run env UV_CACHE_DIR="$CACHE_DIR" UV_PYTHON_INSTALL_DIR="$PYTHON_HOME" \
    "${UV_HOME}/bin/uv" sync --project "$APP_DIR" --frozen --no-dev --python 3.12
}

write_environment_file() {
  log "Writing root-only node configuration"
  if (( DRY_RUN )); then
    printf '+ install protected environment file %q (secret redacted)\n' "$ENV_FILE"
    return
  fi

  install -d -m 0755 "$CONFIG_DIR"
  local temporary
  temporary="$(mktemp "${CONFIG_DIR}/node.env.XXXXXX")"
  chmod 0600 "$temporary"
  {
    printf 'DAIN_POOL_SECRET=%s\n' "$(environment_value "$DAIN_POOL_SECRET")"
    printf 'DAIN_NODE_ID=%s\n' "$(environment_value "${DAIN_NODE_ID:-$(hostname -s)}")"
    printf 'DAIN_SCRATCH_ROOT=%s\n' "$(environment_value "$SCRATCH_ROOT")"
    printf 'DAIN_INDEX_ROOT=%s\n' "$(environment_value "$INDEX_ROOT")"
    printf 'DAIN_EMBED_CACHE=%s\n' "$(environment_value "$EMBED_CACHE")"
    printf 'DAIN_EMBED_MODEL=%s\n' "$(environment_value "$EMBED_MODEL")"
    if [[ -n "${DAIN_CTL:-}" ]]; then
      printf 'DAIN_CTL=%s\n' "$(environment_value "$DAIN_CTL")"
    fi
    if [[ -n "${DAIN_FABRIC_IFACE:-}" ]]; then
      printf 'DAIN_FABRIC_IFACE=%s\n' "$(environment_value "$DAIN_FABRIC_IFACE")"
    fi
  } >"$temporary"
  mv -f "$temporary" "$ENV_FILE"
}

prewarm_embedding_model() {
  log "Caching local embedding model ${EMBED_MODEL}"
  run runuser -u "$DAIN_USER" -- env \
    PYTHONPATH="$APP_DIR" DAIN_EMBED_CACHE="$EMBED_CACHE" DAIN_EMBED_MODEL="$EMBED_MODEL" \
    "${APP_DIR}/.venv/bin/python" -c \
    'from node.index import LocalEmbeddingModel; LocalEmbeddingModel.from_environment().prewarm()'
}

write_systemd_unit() {
  log "Installing systemd service"
  if (( DRY_RUN )); then
    printf '+ install systemd unit %q\n' "$UNIT_FILE"
    return
  fi

  cat >"$UNIT_FILE" <<EOF
[Unit]
Description=DAIN compute node
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${DAIN_USER}
Group=${DAIN_USER}
WorkingDirectory=${APP_DIR}
EnvironmentFile=${ENV_FILE}
ExecStart=${APP_DIR}/.venv/bin/python -m node.dain_node --node-id \${DAIN_NODE_ID}
Restart=on-failure
RestartSec=2
NoNewPrivileges=true
PrivateTmp=true
ProtectHome=true
ProtectSystem=strict
ReadWritePaths=${SCRATCH_ROOT} ${INDEX_ROOT} ${EMBED_CACHE}
RestrictSUIDSGID=true

[Install]
WantedBy=multi-user.target
EOF
  chmod 0644 "$UNIT_FILE"
}

configure_static_ip() {
  if [[ -z "${DAIN_STATIC_IP_CIDR:-}" ]]; then
    log "Leaving network addressing unchanged (no DAIN_STATIC_IP_CIDR)"
    return
  fi
  if ! (( DRY_RUN )) && ! command -v nmcli >/dev/null 2>&1; then
    echo "NetworkManager/nmcli is required for explicit static-IP setup" >&2
    exit 2
  fi

  log "Applying explicit static address to ${DAIN_FABRIC_IFACE}"
  local connection="dain-${DAIN_FABRIC_IFACE}"
  if (( DRY_RUN )) || ! nmcli -g NAME connection show | grep -Fxq "$connection"; then
    run nmcli connection add type ethernet ifname "$DAIN_FABRIC_IFACE" con-name "$connection"
  fi
  run nmcli connection modify "$connection" ipv4.method manual \
    ipv4.addresses "$DAIN_STATIC_IP_CIDR"
  if [[ -n "${DAIN_GATEWAY:-}" ]]; then
    run nmcli connection modify "$connection" ipv4.gateway "$DAIN_GATEWAY"
  fi
  if [[ -n "${DAIN_DNS:-}" ]]; then
    run nmcli connection modify "$connection" ipv4.dns "$DAIN_DNS"
  fi
  run nmcli connection up "$connection"
}

configure_firewall() {
  if [[ "${DAIN_MANAGE_FIREWALL:-0}" != "1" ]]; then
    log "Leaving firewall policy unchanged (no DAIN_MANAGE_FIREWALL=1)"
    return
  fi

  log "Restricting DAIN ports to ${DAIN_FABRIC_CIDR}"
  run ufw allow OpenSSH
  run ufw allow from "$DAIN_FABRIC_CIDR" to any port 9100 proto tcp comment DAIN-agent
  run ufw allow from "$DAIN_FABRIC_CIDR" to any port 50052 proto tcp comment DAIN-rpc
  run ufw allow from "$DAIN_FABRIC_CIDR" to any port 5353 proto udp comment DAIN-mdns
  run ufw --force enable
}

start_and_verify() {
  log "Starting DAIN node"
  run systemctl daemon-reload
  run systemctl enable --now "$SERVICE_NAME"
  if (( DRY_RUN )); then
    run systemctl is-active --quiet "$SERVICE_NAME"
    return
  fi

  local deadline=$(( SECONDS + 45 ))
  until systemctl is-active --quiet "$SERVICE_NAME" && \
    ss -H -ltn "sport = :9100" | grep -q .; do
    if (( SECONDS >= deadline )); then
      systemctl status "$SERVICE_NAME" --no-pager >&2 || true
      journalctl -u "$SERVICE_NAME" -n 40 --no-pager >&2 || true
      echo "DAIN node did not become ready within 45 seconds" >&2
      exit 1
    fi
    sleep 1
  done
}

main() {
  require_configuration
  check_platform
  install_system_dependencies
  ensure_service_user
  fetch_agent
  install_python_environment
  write_environment_file
  prewarm_embedding_model
  write_systemd_unit
  configure_static_ip
  configure_firewall
  start_and_verify
  log "DAIN node ready in $(( SECONDS - started_at )) seconds"
}

main
