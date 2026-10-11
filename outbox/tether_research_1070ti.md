# Local 1070 Ti tether research loop

Status: needs review: failed trial; no automatic continuation

Validation selects provisional candidates; training reward does not select winners.
No hardware accuracy claim. Fresh audit seeds are reserved for a final selected policy.

| Trial | State | Crashes | 1 mm successes | Mean final error (mm) | Decision |
|---|---|---:|---:|---:|---|
| tether_research_001_lower_lr | complete | 0 | 0 | 114.284 | retain incumbent |
| tether_research_002_more_steps | failed |  |  |  | invalid run; no promotion |
| tether_research_003_long_horizon | queued |  |  |  |  |

Provisional incumbent: {"checkpoint": "best", "evidence": "/home/n33du/ai/drone-rl-lab/tmp/tether-runs/tether_puffer_precision_v001/eval_163840.json", "run": "/home/n33du/ai/drone-rl-lab/tmp/tether-runs/tether_puffer_precision_v001", "validation": {"completed_routes": 3, "crashes": 0, "errors_mm": [102.85655853599674, 103.06168299639855, 144.75183378120317], "mean_final_error_mm": 116.89002510453281, "seeds": [51001, 51002, 51003], "successes": 0}}

Error: MuJoCo numerical instability killed Process-3; parent waited on dead worker
