# Local 1070 Ti tether research loop

Status: launching tether_controller_research_001

Validation selects provisional candidates; training reward does not select winners.
No hardware accuracy claim. Fresh audit seeds are reserved for a final selected policy.

| Trial | State | Crashes | 1 mm successes | Mean final error (mm) | Decision |
|---|---|---:|---:|---:|---|
| tether_controller_research_001 | running |  |  |  |  |

Provisional incumbent: {"checkpoint": "best", "evidence": "tether_sim/evidence/policy_controller_probe_v002.json:results.0.012.PPO_residual", "run": "/home/n33du/ai/drone-rl-lab/tmp/tether-runs/tether_puffer_precision_v001", "validation": {"completed_routes": 3, "crashes": 0, "errors_mm": [156.87507675921213, 104.19294403752377, 145.3659365827118], "mean_final_error_mm": 135.47798579314926, "seeds": [51001, 51002, 51003], "successes": 0}}
