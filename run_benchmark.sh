#!/bin/bash
# Full benchmark: 4 controllers x 3 levels x 5 runs = 60 sims
source /home/jeff/drones-venv/bin/activate
cd /media/lsy_drone_racing

echo "========================================="
echo "BENCHMARK START: $(date)"
echo "========================================="

# Level 0
echo ""
echo "--- LEVEL 0: state_controller (state mode) ---"
python scripts/sim.py --config level0.toml --controller state_controller.py --n_runs 5 2>&1 | grep -E "Flight|Finished|Gates"

echo ""
echo "--- LEVEL 0: attitude_controller (attitude mode) ---"
python scripts/sim.py --config level0_attitude.toml --controller attitude_controller.py --n_runs 5 2>&1 | grep -E "Flight|Finished|Gates"

echo ""
echo "--- LEVEL 0: attitude_rl / theirs (attitude mode) ---"
python scripts/sim.py --config level0_attitude.toml --controller attitude_rl.py --n_runs 5 2>&1 | grep -E "Flight|Finished|Gates"

echo ""
echo "--- LEVEL 0: attitude_rl_exp013 / ours (attitude mode) ---"
python scripts/sim.py --config level0_attitude.toml --controller attitude_rl_exp013.py --n_runs 5 2>&1 | grep -E "Flight|Finished|Gates"

# Level 1
echo ""
echo "--- LEVEL 1: state_controller (state mode) ---"
python scripts/sim.py --config level1.toml --controller state_controller.py --n_runs 5 2>&1 | grep -E "Flight|Finished|Gates"

echo ""
echo "--- LEVEL 1: attitude_controller (attitude mode) ---"
python scripts/sim.py --config level1_attitude.toml --controller attitude_controller.py --n_runs 5 2>&1 | grep -E "Flight|Finished|Gates"

echo ""
echo "--- LEVEL 1: attitude_rl / theirs (attitude mode) ---"
python scripts/sim.py --config level1_attitude.toml --controller attitude_rl.py --n_runs 5 2>&1 | grep -E "Flight|Finished|Gates"

echo ""
echo "--- LEVEL 1: attitude_rl_exp013 / ours (attitude mode) ---"
python scripts/sim.py --config level1_attitude.toml --controller attitude_rl_exp013.py --n_runs 5 2>&1 | grep -E "Flight|Finished|Gates"

# Level 2
echo ""
echo "--- LEVEL 2: state_controller (state mode) ---"
python scripts/sim.py --config level2.toml --controller state_controller.py --n_runs 5 2>&1 | grep -E "Flight|Finished|Gates"

echo ""
echo "--- LEVEL 2: attitude_controller (attitude mode) ---"
python scripts/sim.py --config level2_attitude.toml --controller attitude_controller.py --n_runs 5 2>&1 | grep -E "Flight|Finished|Gates"

echo ""
echo "--- LEVEL 2: attitude_rl / theirs (attitude mode) ---"
python scripts/sim.py --config level2_attitude.toml --controller attitude_rl.py --n_runs 5 2>&1 | grep -E "Flight|Finished|Gates"

echo ""
echo "--- LEVEL 2: attitude_rl_exp013 / ours (attitude mode) ---"
python scripts/sim.py --config level2_attitude.toml --controller attitude_rl_exp013.py --n_runs 5 2>&1 | grep -E "Flight|Finished|Gates"

echo ""
echo "========================================="
echo "BENCHMARK DONE: $(date)"
echo "========================================="
