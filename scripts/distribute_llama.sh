#!/usr/bin/env bash
# INF-1, second half: push the worker build to the other four nodes and prove
# all five agree. Run on gpu-01, after scripts/build_llama.sh.
#
#   ./scripts/distribute_llama.sh                       # push, then verify
#   ./scripts/distribute_llama.sh --verify-only         # just check agreement
#   WORKER_NODES="office-01 nuc-01" ./scripts/distribute_llama.sh
#
# Node names are ssh targets — put them in ~/.ssh/config with their LAN
# addresses. They are deliberately NOT in cluster.toml: that file carries no
# addresses, because a configured address outlives the node that owned it. This
# script is build-time plumbing over ssh, not the runtime addressing path.
#
# THE ACCEPTANCE TEST, and why it is two tests:
#
#   1. Every node reports the SAME COMMIT from `llama-server --version`.
#      This is the one that matters. llama.cpp's RPC protocol has no version
#      negotiation, so mismatched nodes connect happily and then hang or return
#      noise — the worst failure mode in the project, because it looks like a
#      network problem and no amount of network debugging fixes it.
#
#   2. The four worker nodes hold a BYTE-IDENTICAL rpc-server.
#      Now possible, and only because every node is Linux x86-64 sharing one
#      build. It catches what the commit check cannot: a node someone rebuilt
#      locally, or a copy that was truncated.
#
# gpu-01 is exempt from (2) — it runs the CUDA build, which is a different
# binary from the same commit, on purpose.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLUSTER_TOML="${REPO_ROOT}/cluster.toml"

LLAMA_SRC="${LLAMA_SRC:-/opt/dain/llama.cpp}"
WORKER_BUILD_DIR="${LLAMA_SRC}/build-worker"
WORKER_NODES="${WORKER_NODES:-password password1 password3 password4 abdallah}"
HEAD_NODE="${HEAD_NODE:-main}"

read_toml_string() {
  local key="$1"
  grep -E "^[[:space:]]*${key}[[:space:]]*=" "${CLUSTER_TOML}" | head -1 | cut -d'"' -f2
}

LLAMA_BIN_DIR="$(read_toml_string llama)"
PINNED_COMMIT="$(read_toml_string pinned_commit)"

if [[ -z "${LLAMA_BIN_DIR}" ]]; then
  echo "ERROR: cluster.toml has no [paths].llama" >&2
  exit 1
fi

verify_only=0
[[ "${1:-}" == "--verify-only" ]] && verify_only=1

# Workers need rpc-server to do their job, llama-server to answer --version for
# the acceptance test, and llama-bench for Sean's SCH-1 calibration probe. The
# worker build is static, so there are no .so files to chase.
BINARIES=(ggml-rpc-server llama-server llama-bench llama-cli)

# --- Push ---------------------------------------------------------------------
if [[ "${verify_only}" == "0" ]]; then
  if [[ ! -x "${WORKER_BUILD_DIR}/bin/ggml-rpc-server" ]]; then
    echo "ERROR: no worker build at ${WORKER_BUILD_DIR}/bin." >&2
    echo "       Run ./scripts/build_llama.sh first." >&2
    exit 1
  fi

  staging="$(mktemp -d)"
  trap 'rm -rf "${staging}"' EXIT
  for binary in "${BINARIES[@]}"; do
    if [[ -x "${WORKER_BUILD_DIR}/bin/${binary}" ]]; then
      cp -a "${WORKER_BUILD_DIR}/bin/${binary}" "${staging}/"
    else
      echo "WARNING: ${binary} not in the worker build; skipping it."
    fi
  done

  for node in ${WORKER_NODES}; do
    echo "==> ${node}"
    if ! ssh "${node}" "mkdir -p '${LLAMA_BIN_DIR}'"; then
      echo "    UNREACHABLE over ssh. Skipping."
      continue
    fi
    rsync -ah --info=progress2 "${staging}/" "${node}:${LLAMA_BIN_DIR}/"
  done
  echo
fi

# --- Verify -------------------------------------------------------------------
# llama-server writes --version to STDERR. Without 2>&1 this reads as empty
# output and looks like a missing binary.
version_of() {
  local node="$1"
  if [[ "${node}" == "${HEAD_NODE}" ]]; then
    "${LLAMA_BIN_DIR}/llama-server" --version 2>&1 | head -1
  else
    ssh "${node}" "'${LLAMA_BIN_DIR}/llama-server' --version 2>&1 | head -1"
  fi
}

checksum_of() {
  ssh "$1" "sha256sum '${LLAMA_BIN_DIR}/ggml-rpc-server' 2>/dev/null | cut -d' ' -f1"
}

echo "--- commit agreement (the test that matters) -------------------"
declare -A versions=()
failed=0
for node in ${HEAD_NODE} ${WORKER_NODES}; do
  if ! version="$(version_of "${node}" 2>/dev/null)" || [[ -z "${version}" ]]; then
    printf '%-12s UNREACHABLE or no binary\n' "${node}"
    failed=1
    continue
  fi
  versions["${node}"]="${version}"
  printf '%-12s %s\n' "${node}" "${version}"
done

echo
if [[ "${#versions[@]}" -eq 0 ]]; then
  echo "INF-1 INCOMPLETE: no node answered at all."
else
  distinct="$(printf '%s\n' "${versions[@]}" | sort -u | wc -l | tr -d ' ')"
  if [[ "${failed}" == "1" ]]; then
    echo "INF-1 INCOMPLETE: at least one node did not answer."
  elif [[ "${distinct}" == "1" ]]; then
    echo "PASS: all ${#versions[@]} nodes report the same build."
  else
    echo "FAIL: ${distinct} different builds across the pool. The RPC handshake"
    echo "      will connect and then hang or return noise. Rebuild the odd ones"
    echo "      out from the pinned commit before doing anything else."
    failed=1
  fi
fi

echo
echo "--- worker binary checksums (bonus, now that every node is Linux) ---"
declare -A sums=()
for node in ${WORKER_NODES}; do
  sum="$(checksum_of "${node}" 2>/dev/null || true)"
  printf '%-12s %s\n' "${node}" "${sum:-unreachable}"
  [[ -n "${sum}" ]] && sums["${node}"]="${sum}"
done

if ((${#sums[@]} > 1)); then
  distinct_sums="$(printf '%s\n' "${sums[@]}" | sort -u | wc -l | tr -d ' ')"
  echo
  if [[ "${distinct_sums}" == "1" ]]; then
    echo "PASS: every reachable worker holds an identical rpc-server."
  else
    echo "FAIL: workers hold ${distinct_sums} different rpc-server binaries."
    echo "      Someone rebuilt locally, or a copy was truncated. Re-run without"
    echo "      --verify-only to overwrite them all from this build."
    failed=1
  fi
fi

# --- The pin ------------------------------------------------------------------
echo
if [[ "${PINNED_COMMIT}" == "UNVERIFIED" || -z "${PINNED_COMMIT}" ]]; then
  echo "cluster.toml still says pinned_commit = \"UNVERIFIED\". INF-1 does not"
  echo "pass until it names the commit above. Set it now, while you are looking"
  echo "at the output that proves it."
  failed=1
else
  echo "cluster.toml pins: ${PINNED_COMMIT}"
  if [[ "${#versions[@]}" -gt 0 ]] &&
    ! printf '%s\n' "${versions[@]}" | grep -qF "${PINNED_COMMIT}"; then
    echo "FAIL: the pinned commit does not appear in any node's --version."
    echo "      Either the pin is stale or the pool is running something else."
    failed=1
  fi
fi

exit "${failed}"
