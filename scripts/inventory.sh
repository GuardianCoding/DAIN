#!/usr/bin/env bash
# Memory topology for one DAIN node. Run once per machine, before the cluster
# exists. Every node is Linux, including gpu-02 — run it inside WSL there, not
# in PowerShell.
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

# cluster.toml [wsl].memory_gb. gpu-02's .wslconfig must grant this much, or the
# planning fixture and the live profile disagree. Tolerance covers the ~200 MiB
# the VM's own structures take off the top before /proc/meminfo sees it.
WSL_EXPECTED_MIB=11264
WSL_TOLERANCE_MIB=768

mem_total_mib=$(( $(awk '/^MemTotal:/    {print $2}' /proc/meminfo) / 1024 ))
mem_avail_mib=$(( $(awk '/^MemAvailable:/{print $2}' /proc/meminfo) / 1024 ))

# --- Which kind of Linux is this? -------------------------------------------
# WSL first: it also reports DISPLAY when WSLg is present, so checking for a
# graphical session before checking for WSL misclassifies gpu-02 as a desktop
# and charges it a 2500 MiB reserve it does not pay.
is_wsl=0
if [[ -r /proc/sys/kernel/osrelease ]] && grep -qiE 'microsoft|wsl' /proc/sys/kernel/osrelease; then
  is_wsl=1
  os_class="linux-wsl"
elif [[ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]]; then
  os_class="linux-desktop"
else
  os_class="linux-headless"
fi

# --- GPU: only a DEDICATED card is usable memory we can plan against ---------
#
# Under WSL2 the only GPU path that reaches llama.cpp is CUDA via /dev/dxg,
# which NVIDIA supports properly. AMD does not: Vulkan there means Mesa's dzn
# layer, which llama.cpp's Vulkan backend does not run reliably on, and
# ROCm-in-WSL covers gfx1100+ only. So on WSL we probe for CUDA and stop —
# reporting AMD VRAM we cannot allocate would oversize gpu-02's slice and the
# model would fail to load with the planner insisting it fits.
vram_mib=0
backend="cpu"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  vram_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
  backend="cuda"
elif [[ "${is_wsl}" == "0" ]]; then
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

if [[ "${is_wsl}" == "1" ]]; then
  # WSL2 is a VM with no SMBIOS, so dmidecode has nothing to read. The DIMM
  # question still matters on this box — it just has to be asked from the
  # Windows side, where the answer is one command.
  echo "DIMMs         NOT VISIBLE FROM WSL — no SMBIOS inside the VM."
  echo "              Ask the host instead, in PowerShell on gpu-02:"
  echo "                Get-CimInstance Win32_PhysicalMemory |"
  echo "                  Select-Object DeviceLocator, Capacity, Speed, ConfiguredClockSpeed"
  echo "              One stick, or ConfiguredClockSpeed below Speed, costs this"
  echo "              node roughly half its decode rate. Same finding, same fix."
elif [[ $EUID -ne 0 ]] || ! command -v dmidecode >/dev/null 2>&1; then
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

wsl_action=0
if [[ "${is_wsl}" == "1" ]]; then
  # 1. Did .wslconfig actually take? WSL2 defaults to half of host RAM, and the
  #    symptom of an unset .wslconfig is not an error — it is a node that is
  #    quietly 3 GiB smaller than the planner believes.
  if [[ "${mem_total_mib}" -lt $(( WSL_EXPECTED_MIB - WSL_TOLERANCE_MIB )) ]]; then
    wsl_action=1
    echo "WSL MEMORY TOO LOW (${mem_total_mib} MiB, expected ~${WSL_EXPECTED_MIB} MiB)."
    echo "The .wslconfig memory setting did not take — WSL2 defaults to half of"
    echo "host RAM. Put this in %USERPROFILE%\\.wslconfig on the Windows side:"
    echo "    [wsl2]"
    echo "    networkingMode=mirrored"
    echo "    memory=11GB"
    echo "then 'wsl --shutdown' from PowerShell and re-run this script."
  fi

  # 2. Is networking mirrored? NAT mode is the one that silently breaks mDNS:
  #    multicast does not cross the virtual switch, so the node never joins and
  #    it reads as a firewall problem no firewall change fixes.
  if ip -4 addr show 2>/dev/null | grep -qE 'inet 172\.(1[6-9]|2[0-9]|3[01])\.'; then
    wsl_action=1
    echo "WSL IS ON A NAT ADDRESS (172.16-31.x). Multicast does not cross the WSL"
    echo "virtual switch, so mDNS join will never happen and this node stays"
    echo "invisible to discovery. Set networkingMode=mirrored in .wslconfig,"
    echo "'wsl --shutdown', then allow inbound in an ADMIN PowerShell:"
    echo "    Set-NetFirewallHyperVVMSetting -Name '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}' -DefaultInboundAction Allow"
  fi

  # 3. Weights must not live on the Windows filesystem.
  if [[ -e /srv/dain/models ]] && df -P /srv/dain/models 2>/dev/null | tail -1 | grep -qE '^(drvfs|[A-Za-z]:)'; then
    wsl_action=1
    echo "MODELS ARE ON THE WINDOWS FILESYSTEM (9p/drvfs). Every read crosses the"
    echo "bridge and mmap of a GGUF there is worse still. Move them onto the WSL"
    echo "filesystem — though note only gpu-01 needs model files at all."
  fi
fi

[[ "${populated}" != "1" && "${xmp_off}" != "1" && "${wsl_action}" == "0" \
   && "${mem_avail_mib}" -ge "${REPLICA_MIB}" ]] \
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
