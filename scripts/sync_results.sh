#!/bin/bash
# =============================================================================
# Drone RL Lab — Sync Results to GitHub
# =============================================================================
# Run after training completes to push results back to the repo.
# Then `git pull` on your local machine to get the results.
#
# Usage: bash scripts/sync_results.sh "exp_011 racing GPU baseline"
# =============================================================================

cd /root/drone-rl-lab

MSG="${1:-gpu training results}"

git add results/ outbox/ configs/
git commit -m "gpu: $MSG"
git push

echo ""
echo "================================================"
echo "  RESULTS PUSHED TO GITHUB"
echo ""
echo "  On your local machine, run:"
echo "    cd /media/drone-rl-lab && git pull"
echo ""
echo "  !! STOP YOUR POD TO STOP BILLING !!"
echo "  runpodctl stop pod \$RUNPOD_POD_ID"
echo "  Or: RunPod dashboard → Stop Pod"
echo "================================================"
echo ""
