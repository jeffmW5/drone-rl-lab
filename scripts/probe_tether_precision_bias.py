"""Single-seed PD diagnostic; interventions are not benchmark results."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd()))
import numpy as np
from tether_sim.generalize import make_task

config = json.loads(Path('configs/tether_research_recovery_v001.json').read_text())
results = []
for label in ('randomized', 'wind_off', 'wind_off_matched_motors'):
    task = make_task(config)
    task.reset(seed=51001)
    if label != 'randomized':
        task.amplitude = 0.
    if label == 'wind_off_matched_motors':
        task.env.model.actuator_gainprm[:,0] = 1.
    rows = []
    for _ in range(4500):
        _, _, done, truncated, info = task.step(np.zeros(4))
        if task.env.data.time >= 80.:
            payload_goal = np.r_[task.goal, task.altitude-task.env.config.length-.02]
            rows.append(np.r_[task.env.data.qpos[:3]-task.env.target,
                              task.env.data.xpos[task.env.payload]-payload_goal])
        if done or truncated:
            break
    results.append(dict(case=label,seed=51001,crashed=bool(done),
                        final_error_mm=info['payload_goal_error_mm'],
                        route_remaining_m=info['route_remaining_m'],
                        tail_samples=len(rows),
                        tail_mean_drone_xyz_error_mm=(1000*np.mean(rows,axis=0)[:3]).tolist() if rows else None,
                        tail_mean_payload_xyz_error_mm=(1000*np.mean(rows,axis=0)[3:]).tolist() if rows else None))
    task.close()
    print(json.dumps(results[-1]),flush=True)
output = Path('tmp/tether-research/recovery_bias_probe.json')
output.write_text(json.dumps(dict(scope='single-seed PD interventions; no hardware validation or causal generalization',
                                config='configs/tether_research_recovery_v001.json',results=results),indent=2))
