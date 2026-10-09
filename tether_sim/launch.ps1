param([string]$Distro='Ubuntu-Laya', [string]$Python='/home/n33du/ai/tether-sim-venv/bin/python')
$repoRoot = Split-Path -Parent $PSScriptRoot
& wsl.exe -d $Distro --cd $repoRoot -- $Python -m tether_sim.viewer
