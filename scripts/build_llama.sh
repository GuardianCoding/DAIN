#!/usr/bin/env bash
# INF-1, first half: build llama.cpp on gpu-01, twice, from ONE commit.
#
#   ./scripts/build_llama.sh              # build both, at whatever HEAD is
#   ./scripts/build_llama.sh <commit>     # build both, pinned to <commit>
#   ./scripts/build_llama.sh --head-only  # rebuild just gpu-01's CUDA build
#
# Then run scripts/distribute_llama.sh to push the worker build to the other
# four nodes and verify all five agree.
#
# WHY TWO BUILDS AND NOT ONE:
#
# GGML_NATIVE defaults to ON and adds -march=native. gpu-01 is Zen 5 and has
# AVX-512. office-01/02 are Skylake and nuc-01 is Kaby Lake, which have AVX2
# and no AVX-512. Build once on gpu-01 with the defaults, copy the binary, and
# it dies with SIGILL on the first tensor op — a crash that reads like a
# corrupt model file and costs an evening to trace. So the worker build turns
# GGML_NATIVE off and names the ISA explicitly.
#
# gpu-02 runs the WORKER build too, not a Vulkan one. Under WSL2 the GPU is
# /dev/dxg (D3D12); Vulkan there means Mesa's dzn layer, which llama.cpp's
# Vulkan backend does not run reliably on, and ROCm-in-WSL is gfx1100+ only
# while the RX 5700 XT is RDNA1. gpu-02 is a CPU node now.
#
# WHY THE COMMIT AND NOT A CHECKSUM:
#
# llama.cpp's RPC wire protocol has no version negotiation. Two nodes built
# from different commits connect happily, then hang or return noise. The commit
# is the invariant; the checksum is a bonus the worker nodes now also get,
# because they finally share one build.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLUSTER_TOML="${REPO_ROOT}/cluster.toml"

LLAMA_SRC="${LLAMA_SRC:-/opt/dain/llama.cpp}"
LLAMA_UPSTREAM="${LLAMA_UPSTREAM:-https://github.com/ggml-org/llama.cpp}"
HEAD_BUILD_DIR="${LLAMA_SRC}/build"            # matches [paths].llama minus /bin
WORKER_BUILD_DIR="${LLAMA_SRC}/build-worker"

# Read the flag strings out of cluster.toml rather than duplicating them here.
# Deliberately a narrow grep and not a TOML parser: this runs on a bare node
# before any Python environment exists.
read_flag() {
  local key="$1"
  local value
  value="$(grep -E "^[[:space:]]*${key}[[:space:]]*=" "${CLUSTER_TOML}" | head -1 | cut -d'"' -f2)"
  if [[ -z "${value}" ]]; then
    echo "ERROR: cluster.toml has no [llama].${key}" >&2
    exit 1
  fi
  printf '%s' "${value}"
}

head_only=0
pin=""
for arg in "$@"; do
  case "${arg}" in
    --head-only) head_only=1 ;;
    -h|--help)   sed -n '2,10p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *)           pin="${arg}" ;;
  esac
done

# --- Toolchain checks, before a 20-minute build fails at the last link -------
missing=()
for tool in git cmake; do
  command -v "${tool}" >/dev/null 2>&1 || missing+=("${tool}")
done
if ((${#missing[@]})); then
  echo "ERROR: missing build tools: ${missing[*]}" >&2
  echo "  sudo apt install -y git cmake build-essential ccache libcurl4-openssl-dev" >&2
  exit 1
fi

# --- Source tree at a known commit ------------------------------------------
if [[ ! -d "${LLAMA_SRC}/.git" ]]; then
  echo "==> cloning llama.cpp into ${LLAMA_SRC}"
  mkdir -p "$(dirname "${LLAMA_SRC}")"
  git clone "${LLAMA_UPSTREAM}" "${LLAMA_SRC}"
fi

cd "${LLAMA_SRC}"
if [[ -n "${pin}" ]]; then
  echo "==> fetching and pinning to ${pin}"
  git fetch --all --tags
  git checkout --detach "${pin}"
fi

COMMIT="$(git rev-parse --short HEAD)"
echo "==> building at commit ${COMMIT}"

if ! git diff --quiet HEAD 2>/dev/null; then
  echo "WARNING: the llama.cpp tree has local modifications. Two nodes built"
  echo "         from this will report the same commit and still differ."
fi

# --- Build A: gpu-01's own, CUDA ---------------------------------------------
# CMAKE_CUDA_ARCHITECTURES=120 is the RTX 5070 Ti (GB203, compute 12.0). It
# needs CUDA toolkit 12.8+; an older toolkit does not know sm_120.
echo
echo "==> [1/2] head build (CUDA) -> ${HEAD_BUILD_DIR}"
if ! command -v nvcc >/dev/null 2>&1; then
  echo "WARNING: nvcc is not on PATH. If this is gpu-01, install the CUDA"
  echo "         toolkit (12.8+) first — Blackwell needs it — because this"
  echo "         build is about to fall back to CPU and look fine."
fi
# shellcheck disable=SC2046  # flag string is intentionally word-split
cmake -B "${HEAD_BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release $(read_flag build_flags_head)
cmake --build "${HEAD_BUILD_DIR}" --config Release -j "$(nproc)"

if [[ "${head_only}" == "1" ]]; then
  echo
  echo "head build done at ${COMMIT}. Skipping the worker build (--head-only)."
  exit 0
fi

# --- Build B: the other four, portable ---------------------------------------
echo
echo "==> [2/2] worker build (portable AVX2, static) -> ${WORKER_BUILD_DIR}"
# shellcheck disable=SC2046  # flag string is intentionally word-split
cmake -B "${WORKER_BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release $(read_flag build_flags_worker)
cmake --build "${WORKER_BUILD_DIR}" --config Release -j "$(nproc)"

# --- Prove the portable build is actually portable ---------------------------
# Catching an AVX-512 instruction here costs a second. Catching it on the NUC
# costs an evening, and it presents as a corrupt-model error.
echo
echo "==> checking the worker build for instructions the older nodes lack"
if command -v objdump >/dev/null 2>&1; then
  bad="$(objdump -d "${WORKER_BUILD_DIR}/bin/rpc-server" 2>/dev/null \
          | grep -cE '%[zk]mm[0-9]|\{%k[1-7]\}' || true)"
  if [[ "${bad}" != "0" ]]; then
    echo "FAIL: rpc-server contains ${bad} AVX-512-looking instruction(s)."
    echo "      GGML_NATIVE did not get turned off. This binary will SIGILL on"
    echo "      office-01/02 and nuc-01. Delete ${WORKER_BUILD_DIR} and rebuild."
    exit 1
  fi
  echo "OK: no AVX-512 instructions in rpc-server."
else
  echo "SKIPPED: objdump not installed (binutils). Cannot verify portability"
  echo "         here — but the first run on a worker node will tell you loudly."
fi

cat <<EOF

--- built, both configurations, at ${COMMIT} -------------------
head   (gpu-01, CUDA)      ${HEAD_BUILD_DIR}/bin
worker (the other four)    ${WORKER_BUILD_DIR}/bin

Next:
  1. Record the commit in cluster.toml:
       pinned_commit = "${COMMIT}"
  2. Push the worker build and verify all five agree:
       ./scripts/distribute_llama.sh
EOF
