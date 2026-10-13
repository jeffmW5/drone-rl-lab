# Continuous CUDA research

{
  "status": "ready for next bounded trial",
  "phase": "takeoff",
  "paused": false,
  "incumbents": {
    "takeoff": {
      "checkpoint": "/home/n33du/ai/drone-rl-lab/tmp/cuda-research/cuda_auto_0001_takeoff/best.pt",
      "score": {
        "runs": [
          {
            "seed": 51011,
            "mean_final_error_mm": 1977.3721611267238,
            "errors_mm": [
              1924.522959712785,
              1881.2214817656104,
              1903.3449096449433,
              1931.3087069026215,
              1904.4310825369669,
              2210.6153308928388,
              2017.72645448807,
              2045.8063630699558
            ],
            "crashes": 0,
            "takeoffs": 8,
            "hover_successes": 0,
            "peak_hover_hold_s": [
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            "peak_drone_height_m": [
              2.555306911468506,
              2.5455758571624756,
              2.564427137374878,
              2.541923999786377,
              2.570624828338623,
              2.7555534839630127,
              2.5293262004852295,
              2.617532253265381
            ],
            "scope": "Prototype five-second destination episode; not original navigation or sustained1mm benchmark"
          },
          {
            "seed": 51012,
            "mean_final_error_mm": 1955.3084432855235,
            "errors_mm": [
              2019.610139672674,
              2001.8925240803424,
              1954.9593753439028,
              1921.4920853173512,
              1956.1191649348605,
              1956.8923681667945,
              1874.2919396195996,
              1957.2099491486617
            ],
            "crashes": 0,
            "takeoffs": 8,
            "hover_successes": 0,
            "peak_hover_hold_s": [
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            "peak_drone_height_m": [
              2.541623115539551,
              2.618020534515381,
              2.6554195880889893,
              2.5390758514404297,
              2.634330987930298,
              2.653881072998047,
              2.652323007583618,
              2.6328861713409424
            ],
            "scope": "Prototype five-second destination episode; not original navigation or sustained1mm benchmark"
          },
          {
            "seed": 51013,
            "mean_final_error_mm": 2025.0984221001752,
            "errors_mm": [
              2184.5026208193003,
              1956.456812989296,
              1895.8025592756512,
              2159.9990806631863,
              1911.5312486668995,
              2164.951246110548,
              2066.0467157027115,
              1861.4970925738082
            ],
            "crashes": 0,
            "takeoffs": 8,
            "hover_successes": 0,
            "peak_hover_hold_s": [
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            "peak_drone_height_m": [
              2.7299206256866455,
              2.647240161895752,
              2.543764591217041,
              2.7272350788116455,
              2.545780897140503,
              2.743102550506592,
              2.7554852962493896,
              2.5429530143737793
            ],
            "scope": "Prototype five-second destination episode; not original navigation or sustained1mm benchmark"
          }
        ],
        "crashes": 0,
        "takeoffs": 24,
        "hover_successes": 0,
        "mean_final_error_mm": 1985.9263421708074
      }
    }
  },
  "error": null
}

Full rejected trials: state/cuda_research/loop.json. No hardware accuracy claim.

## 2026-10-12 12:28 UTC methodology update
Review0030 closes all three rollout-length pairs: no sustained hovers and all six final checkpoints worse than initial. The selected trial242 periodic step32768 reaches822.218mm (initial857.312), but its final regresses1295.322mm; provisional reused validation only. Pure learned takeoff remains unsolved for stable hover.
Queued v019: three seed pairs of raw versus fixed physical input conditioning, identical initial functions via weight-column migration. CPU actual-checkpoint equivalence, shared/separate critic roundtrip and finite-gradient checks passed. GPU outcome pending; healthy serial daemon69922 retained, no compilation/restart. Scripts synced explicitly to native WSL; native checkout retains unrelated edits and is not a clean copy of Windows commit.

## 2026-10-12 13:28 UTC braking campaign review
Queuev020 completed: reward2/6 final errors1757.800/1454.838,956.556/938.053,3327.988/3197.482mm. Added drone velocity cost helps relative errors3/3 pairs, but all finals regress from822.218mm initial, all best_step0 and zero sustained hovers. No promotion; no new coefficient sweep. Full frozen evidence and sources in review0034. Healthy serial daemon69922 continues; no original1mm or hardware success.

## 2026-10-12 14:48 UTC channel exploration review
Queuev022 complete; wider roll/pitch noise failed paired consistency and all final policies regressed. Ground takeoff evaluation still zero airborne acquisition and sustained hover. Periodic improvements789.848mm baseline307 and807.159mm treatment310 regressed at final; not convergence. No method promotion or new width sweep. Evidence/sources review0038, daemon69922 healthy/unpaused, audit seeds reserved.

## 2026-10-13 completed tenfold campaign and next reward test

Review0112 closes the completed six long runs885–890 and catches up the usage-limit review gap through936, using a dated931–936 addendum. All six completed655360 transitions each. None of240 periodic checks or six finals achieved sustained1s ground zone hover; maximum temporary hold0.54s. Fine reward16 improved five-second escape-free checkpoint counts in each training seed (11/26/36 versus6/10/7;73/120 versus23/120). This is a provisional correlated checkpoint signal, not independent validation or reliable hover. Final890 avoids escape but is still moving about1.25m/s; low altitude endpoints are not millimeter precision. Regressions, true failures and missing default terminal traces remain preserved.

One falsifiable reward17 adaptation is frozen: score horizontal velocity against an inward reference proportional to center displacement, capped at0.5m/s, tending to zero at the center. It changes reward only; no action helper, physics, reset, exploration or credit-horizon change. Three655360-transition treatments at70601–70603 reuse immutable long reward16 controls886/888/890, fixed-source759 and identical learner/simulator/action settings. Ten CPU groups passed, including old14/15/16 parity, direction/sign/cap/boundaries, unchanged motor actions and reset/failure logic; existing fine-reward checks passed12 groups. Leased new-run physics checks and actual initial/source comparisons are separately recorded; full learned outcomes remain pending.

Actual frozen trial937 is velocity_reference_reward17_seed70601. Daemon554137 remains active/unpaused with one trainer; no restart, healthy interruption, duplicate GPU work or concurrent compilation occurred. Its leased physics checks passed; backend/physics/host hashes and all10 initial policy tensors match the immutable886 control. Actual initial physical metrics remain pending in the final result, not inferred from weights alone. Sources, versioned review and queue are in state/cuda_research/methodology/review_0112 and configs/cuda_research/methodology_queue_v000_user_abca_00_velocity_reference.json. Fresh official HF retrieval and original-method fallbacks motivate the hypothesis but are not simulator results. Hardware/audit seeds remain reserved; original strict1mm obstacle navigation and hardware accuracy remain unachieved.

