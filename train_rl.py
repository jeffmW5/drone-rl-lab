"""
Drone RL Lab — Experiment Runner
=================================
This is the main training script for the drone-rl-lab agentic loop.

Linux Claude: modify ONLY the section marked "EXPERIMENT CONFIGURATION".
The reward function lives in CustomHoverAviary._computeReward().
Everything below the "DO NOT MODIFY" line is infrastructure — leave it alone.

How to run:
    source ~/repos/drones-venv/bin/activate
    python train_rl.py
"""

import os
import sys
import json
import time
from datetime import datetime
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

# =============================================================================
# EXPERIMENT CONFIGURATION — Linux Claude modifies this section
# =============================================================================

EXPERIMENT_NAME = "exp_003_quadratic_reward"
EXPERIMENT_HYPOTHESIS = (
    "Quadratic reward provides stronger gradient near target. Does the drone achieve tighter hover or trigger success termination?"
)

# Wall-clock training budget in seconds (3 minutes default)
TRAINING_BUDGET_SECONDS = 180

# PPO hyperparameters — stable-baselines3 defaults shown explicitly
PPO_KWARGS = dict(
    learning_rate=3e-4,   # step size for gradient descent
    n_steps=2048,         # timesteps collected before each update
    batch_size=64,        # mini-batch size for each gradient step
    gamma=0.99,           # discount factor (how much future rewards matter)
    gae_lambda=0.95,      # smoothing for advantage estimation
)


class CustomHoverAviary(HoverAviary):
    """
    Our custom drone environment. Inherits everything from HoverAviary
    (physics, observation space, action space, termination conditions)
    but lets us override the reward function in one place.

    Linux Claude: modify _computeReward() to test new reward shapes.
    """

    # Hover target: [x, y, z] in meters
    TARGET_POS_CUSTOM = np.array([0.0, 0.0, 1.0])

    def _computeReward(self):
        """
        BASELINE REWARD: max(0, 2 - distance^4)

        RL concept: this is a "dense" reward — the agent gets feedback
        at every timestep, not just when it reaches the goal. The quartic
        (distance^4) term means rewards drop off steeply with distance,
        giving strong signal near the target but weak signal far away.
        """
        state = self._getDroneStateVector(0)
        dist = np.linalg.norm(self.TARGET_POS_CUSTOM - state[0:3])
        return max(0, 2 - dist**2)


# =============================================================================
# DO NOT MODIFY BELOW THIS LINE — training infrastructure
# =============================================================================

class TimeBudgetCallback(BaseCallback):
    """Stops training when the wall-clock budget is exceeded."""

    def __init__(self, budget_seconds: int, verbose: int = 0):
        super().__init__(verbose)
        self.budget_seconds = budget_seconds
        self.start_time: float = 0.0

    def _on_training_start(self) -> None:
        self.start_time = time.time()

    def _on_step(self) -> bool:
        elapsed = time.time() - self.start_time
        if elapsed > self.budget_seconds:
            print(f"\n[TimeBudget] {self.budget_seconds}s elapsed — stopping training.")
            return False
        return True


def run():
    # ── Paths ─────────────────────────────────────────────────────────────────
    lab_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(lab_dir, "results", EXPERIMENT_NAME)
    os.makedirs(results_dir, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  EXPERIMENT: {EXPERIMENT_NAME}")
    print(f"  HYPOTHESIS: {EXPERIMENT_HYPOTHESIS}")
    print(f"  BUDGET:     {TRAINING_BUDGET_SECONDS}s")
    print(f"  OUTPUT:     {results_dir}")
    print(f"{'='*65}\n")

    # ── Environments ──────────────────────────────────────────────────────────
    env_kwargs = dict(obs=ObservationType.KIN, act=ActionType.ONE_D_RPM)
    train_env = make_vec_env(CustomHoverAviary, env_kwargs=env_kwargs,
                             n_envs=1, seed=0)
    eval_env = CustomHoverAviary(**env_kwargs)

    print(f"[INFO] Action space:      {train_env.action_space}")
    print(f"[INFO] Observation space: {train_env.observation_space}\n")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = PPO("MlpPolicy", train_env, verbose=1, **PPO_KWARGS)

    # ── Callbacks ─────────────────────────────────────────────────────────────
    time_cb = TimeBudgetCallback(budget_seconds=TRAINING_BUDGET_SECONDS)
    eval_cb = EvalCallback(
        eval_env,
        verbose=1,
        best_model_save_path=results_dir,
        log_path=results_dir,
        eval_freq=1000,
        deterministic=True,
        render=False,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    wall_start = time.time()
    model.learn(
        total_timesteps=int(1e8),   # large cap — TimeBudgetCallback will stop us
        callback=[time_cb, eval_cb],
        log_interval=100,
    )
    elapsed = time.time() - wall_start

    # ── Evaluate best model ───────────────────────────────────────────────────
    best_model_path = os.path.join(results_dir, "best_model.zip")
    if os.path.exists(best_model_path):
        best_model = PPO.load(best_model_path)
        mean_reward, std_reward = evaluate_policy(
            best_model, eval_env, n_eval_episodes=10
        )
    else:
        print("[WARN] No best_model.zip found — using final model for eval.")
        mean_reward, std_reward = evaluate_policy(
            model, eval_env, n_eval_episodes=10
        )

    train_env.close()
    eval_env.close()

    # ── Save metrics ──────────────────────────────────────────────────────────
    metrics = {
        "experiment": EXPERIMENT_NAME,
        "hypothesis": EXPERIMENT_HYPOTHESIS,
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "mean_reward": round(float(mean_reward), 3),
        "std_reward": round(float(std_reward), 3),
        "timesteps_trained": int(model.num_timesteps),
        "ppo_kwargs": PPO_KWARGS,
    }

    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[Saved] {metrics_path}")

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  RESULTS")
    print(f"  mean_reward  = {mean_reward:.3f} ± {std_reward:.3f}")
    print(f"  timesteps    = {model.num_timesteps:,}")
    print(f"  wall time    = {elapsed:.1f}s")
    print(f"{'='*65}\n")

    # ── Update OUTBOX.md ──────────────────────────────────────────────────────
    outbox_path = os.path.join(lab_dir, "OUTBOX.md")
    with open(outbox_path, "w") as f:
        f.write("# OUTBOX — Results from Linux Claude\n\n")
        f.write("> This file is written by Linux Claude after each experiment. "
                "Windows Claude reads this to plan the next one.\n\n---\n\n")
        f.write(f"## {EXPERIMENT_NAME} — Latest Results\n\n")
        f.write(f"**Hypothesis:** {EXPERIMENT_HYPOTHESIS}\n\n")
        f.write(f"| Metric | Value |\n|--------|-------|\n")
        f.write(f"| mean_reward | {mean_reward:.3f} ± {std_reward:.3f} |\n")
        f.write(f"| timesteps_trained | {model.num_timesteps:,} |\n")
        f.write(f"| wall_time | {elapsed:.1f}s |\n\n")
        f.write(f"*(Linux Claude: write your full EXPERIMENT.md to "
                f"`results/{EXPERIMENT_NAME}/EXPERIMENT.md` and then update this "
                f"file with your observations and suggested next steps.)*\n")

    print(f"[Updated] OUTBOX.md")
    print(f"\nNext steps for Linux Claude:")
    print(f"  1. Write results/{EXPERIMENT_NAME}/EXPERIMENT.md")
    print(f"  2. git add -A && git commit -m '{EXPERIMENT_NAME}: <summary>'")
    print(f"  3. Update OUTBOX.md with your analysis\n")

    return metrics


if __name__ == "__main__":
    run()
