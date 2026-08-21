# Memory topology for the DAIN Windows node (gpu-02). Run once, before the
# cluster exists. The Linux nodes use scripts/inventory.sh.
#
#   .\scripts\inventory.ps1 -NodeId gpu-02
#
# WHY THIS EXISTS, when nodes self-report at join:
#
# NodeProfile tells the scheduler a node is slow. It cannot tell you WHY, and
# not until the cluster is already running. Only the DIMM table separates "one
# stick parked at JEDEC defaults" — a BIOS change with lead time — from a node
# that is simply small.
#
# ConfiguredClockSpeed is the number that matters, not Speed: a 3200 stick
# running at 2133 reports both, and only the configured value is real.
#
# Everything NodeProfile already reports (cpu, cores, threads, measured
# bandwidth) is deliberately NOT printed. The TOML fragment carries only the
# fields infer/memory.py:_parse_budget actually reads.

param([string]$NodeId = "gpu-02")

$ErrorActionPreference = "Stop"

# Smallest replica in the ladder: Qwen3-4B Q4_K_M, 2.5 GB -> 2384 MiB.
$ReplicaMib = 2384

$system = Get-CimInstance Win32_ComputerSystem
$os     = Get-CimInstance Win32_OperatingSystem
$sticks = @(Get-CimInstance Win32_PhysicalMemory)
$video  = Get-CimInstance Win32_VideoController | Select-Object -First 1

$ramMib   = [math]::Round($system.TotalPhysicalMemory / 1MB)
$availMib = [math]::Round($os.FreePhysicalMemory / 1KB)

# AdapterRAM is a 32-bit field and silently caps at 4 GB, so an 8 GB card reads
# as ~4095. The registry holds the real value.
$vramMib = 0
$gpuKey = "HKLM:\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}\0000"
$qwMemory = (Get-ItemProperty -Path $gpuKey -Name "HardwareInformation.qwMemorySize" -ErrorAction SilentlyContinue)."HardwareInformation.qwMemorySize"
if ($qwMemory) {
  $vramMib = [math]::Round($qwMemory / 1MB)
} elseif ($video.AdapterRAM -gt 0) {
  $vramMib = [math]::Round($video.AdapterRAM / 1MB)
}

# RDNA1 has no usable ROCm, so an AMD card means Vulkan and only Vulkan.
$backend = "cpu"
if ($video.Name -match "NVIDIA|GeForce|RTX")  { $backend = "cuda" }
elseif ($video.Name -match "AMD|Radeon|RX")   { $backend = "vulkan" }

Write-Output "=== $NodeId - memory topology ==============================="
Write-Output "TotalPhysicalMemory  $ramMib MiB"
Write-Output "FreePhysicalMemory   $availMib MiB     <- what a model can actually have"
Write-Output "GPU                  $backend, $vramMib MiB VRAM  ($($video.Name))"
Write-Output ""

$xmpOff = $false
Write-Output ("{0,-14} {1,-10} {2,-10} {3,-10}" -f "LOCATOR", "SIZE", "RATED", "CONFIGURED")
foreach ($stick in $sticks) {
  $sizeMib = [math]::Round($stick.Capacity / 1MB)
  $configured = $stick.ConfiguredClockSpeed
  Write-Output ("{0,-14} {1,-10} {2,-10} {3,-10}" -f `
    $stick.DeviceLocator, "$sizeMib MiB", $stick.Speed, $(if ($configured) { $configured } else { "?" }))
  if ($configured -and $stick.Speed -and $configured -lt $stick.Speed) { $xmpOff = $true }
}

Write-Output ""
Write-Output "--- actions ---------------------------------------------------"
$healthy = $true
if ($sticks.Count -eq 1) {
  $healthy = $false
  Write-Output "ONE DIMM. Half the memory bandwidth, therefore roughly half the decode"
  Write-Output "speed, and no spec sheet says so. A matched second stick is the cheapest"
  Write-Output "performance in this project."
}
if ($xmpOff) {
  $healthy = $false
  Write-Output "MEMORY BELOW ITS RATED SPEED. Enable XMP/EXPO in BIOS and re-run."
  Write-Output "One documented case went 2000 -> 6000 MT/s for ~3x token generation."
}
if ($availMib -lt $ReplicaMib) {
  $healthy = $false
  Write-Output "BELOW REPLICA THRESHOLD ($availMib < $ReplicaMib MiB). This node cannot"
  Write-Output "hold the 4B fan-out replica. Free memory or drop it to worker-only."
}
if ($healthy) { Write-Output "Memory topology is healthy." }

# TODO: these three belong in scripts/check-fabric.ps1 once that exists. They
# live here only because this is currently the sole script that runs on gpu-02
# before it can join, and they are the three settings that make RPC fail with
# no error message at all.
Write-Output ""
Write-Output "--- pre-join blockers (moving to check-fabric.ps1) -------------"
$netProfile = Get-NetConnectionProfile -InterfaceAlias Ethernet -ErrorAction SilentlyContinue
if ($netProfile -and $netProfile.NetworkCategory -ne "Private") {
  Write-Output "Ethernet is '$($netProfile.NetworkCategory)'. Windows Firewall blocks RPC silently. Set it to Private."
}
if (Get-NetAdapter -Physical | Where-Object { $_.Status -eq "Up" -and $_.Name -match "Wi-?Fi" }) {
  Write-Output "Wi-Fi adapter is up. Windows may route cluster traffic over it. Disable it."
}
if ((powercfg /getactivescheme) -notmatch "High performance|Ultimate") {
  Write-Output "Power scheme is not High Performance. Run: powercfg /setactive SCHEME_MIN"
}

Write-Output ""
Write-Output "--- paste into cluster.toml under [[planning.nodes]] ------------"
Write-Output "[[planning.nodes]]"
Write-Output "id       = `"$NodeId`""
Write-Output "ram_mb   = $ramMib"
Write-Output "vram_mb  = $vramMib"
Write-Output "backend  = `"$backend`""
Write-Output "os_class = `"windows-desktop`""
Write-Output "verified = true"
