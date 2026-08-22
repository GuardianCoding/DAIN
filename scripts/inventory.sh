#!/usr/bin/env bash
# Memory topology for one DAIN Linux node. Run once per machine, before the
# cluster exists. gpu-02 is Windows — use scripts/inventory.ps1 there.
#
#   sudo ./scripts/inventory.sh nuc-01
#
# WHY THIS EXISTS, when nodes self-report at join:
#
# NodeProfile tells the scheduler that a node is slow. It cannot tell you WHY,
# and it cannot tell you until the cluster is already running. Only dmidecode
# separates "one 8 GB stick at 2133 in a dual-channel board" — a BIOS change or
# a $40 purchase, both with lead time — from a node that is simply small.
#
# It also catches what the control group structurally cannot: office-01 and
# office-02 validate the profiler by agreeing within 10%. If BOTH sit at JEDEC
# defaults they agree perfectly and are both running at half speed.
#
# Everything NodeProfile already reports (cpu, cores, threads, measured
# bandwidth) is deliberately NOT printed. The TOML fragment carries only the
# fields infer/memory.py:_parse_budget actually reads.

set -euo pipefail

NODE_ID="${1:-$(hostname)}"

# Smallest replica in the ladder: Qwen3-4B Q4_K_M, 2.5 GB -> 2384 MiB.
# A node below this cannot hold a replica and sits out the fan-out demo.
REPLICA_MIB=2384

mem_total_mib=$(( $(awk '/^MemTotal:/    {print $2}' /proc/meminfo) / 1024 ))
mem_avail_mib=$(( $(awk '/^MemAvailable:/{print $2}' /proc/meminfo) / 1024 ))

os_class="linux-headless"
[[ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]] && os_class="linux-desktop"

# --- GPU: only a DEDICATED card is usable memory we can plan against ---------
vram_mib=0
backend="cpu"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  vram_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
  backend="cuda"
else
  for card in /sys/class/drm/card*/device/mem_info_vram_total; do
    [[ -r "$card" ]] || continue
    vram_mib=$(( $(cat "$card") / 1024 / 1024 ))
    backend="vulkan"
    break
  done
fi

echo "=== ${NODE_ID} — memory topology ==============================="
echo "MemTotal      ${mem_total_mib} MiB"
echo "MemAvailable  ${mem_avail_mib} MiB     <- what a model can actually have"
echo "GPU           ${backend}, ${vram_mib} MiB VRAM"
echo

# --- DIMM topology: the part nothing else can tell you ----------------------
populated=0
xmp_off=0

if [[ $EUID -ne 0 ]] || ! command -v dmidecode >/dev/null 2>&1; then
  echo "DIMMs         UNKNOWN — needs 'sudo' and dmidecode(8)."
  echo "              This is the whole reason the script exists. Re-run with sudo."
else
  printf "%-12s %-10s %-10s %-10s\n" "LOCATOR" "SIZE" "RATED" "CONFIGURED"
  while IFS='|' read -r locator size rated configured; do
    [[ -z "${size}" ]] && continue
    populated=$(( populated + 1 ))
    printf "%-12s %-10s %-10s %-10s\n" "${locator}" "${size}" "${rated:-?}" "${configured:-?}"
    rated_n=${rated%% *}
    conf_n=${configured%% *}
    if [[ "${rated_n}" =~ ^[0-9]+$ && "${conf_n}" =~ ^[0-9]+$ && "${conf_n}" -lt "${rated_n}" ]]; then
      xmp_off=1
    fi
  done < <(dmidecode -t memory | awk -F': ' '
      /Memory Device/                        { size=""; rated=""; conf=""; loc="" }
      $1 ~ /[ \t]*Size$/                     { size=$2 }
      $1 ~ /[ \t]*Locator$/                  { loc=$2 }
      $1 ~ /[ \t]*Speed$/                    { rated=$2 }
      $1 ~ /[ \t]*Configured Memory Speed$/  { conf=$2
          if (size ~ /^[0-9]/) printf "%s|%s|%s|%s\n", loc, size, rated, conf }
  ')
fi

echo
echo "--- actions ---------------------------------------------------"
if [[ "${populated}" == "1" ]]; then
  echo "ONE DIMM. Half the memory bandwidth, therefore roughly half the decode"
  echo "speed, and no spec sheet says so. A matched second stick is the cheapest"
  echo "performance in this project."
fi
if [[ "${xmp_off}" == "1" ]]; then
  echo "MEMORY BELOW ITS RATED SPEED. Enable XMP/EXPO in BIOS and re-run."
  echo "One documented case went 2000 -> 6000 MT/s for ~3x token generation."
fi
if [[ "${mem_avail_mib}" -lt "${REPLICA_MIB}" ]]; then
  echo "BELOW REPLICA THRESHOLD (${mem_avail_mib} < ${REPLICA_MIB} MiB). This node"
  echo "cannot hold the 4B fan-out replica. Free memory or drop it to worker-only."
fi
if [[ "${os_class}" == "linux-desktop" ]]; then
  echo "GRAPHICAL SESSION RUNNING. 'systemctl isolate multi-user.target' frees ~1.7 GiB."
fi
[[ "${populated}" != "1" && "${xmp_off}" != "1" && "${mem_avail_mib}" -ge "${REPLICA_MIB}" ]] \
  && echo "Memory topology is healthy."

cat <<EOF

--- paste into cluster.toml under [[planning.nodes]] ------------
[[planning.nodes]]
id       = "${NODE_ID}"
ram_mb   = ${mem_total_mib}
vram_mb  = ${vram_mib}
backend  = "${backend}"
os_class = "${os_class}"
verified = true
EOF
