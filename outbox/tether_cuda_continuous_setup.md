# Continuous learned-motor research setup

User requested maximum practical serial research, no daily training cap.
scripts/cuda_autoresearch.py is a persistent daemon in Ubuntu-Laya. It starts
the next trial immediately after completion, with no10minute playback idle time.
Persistent viewer proxy remains on localhost8767; each child uses8768.

65536-transition trials,90s elapsed learner-loop budget plus finite initial/final
validation. Periodic validation time is included in learner-loop elapsed time.
128 parallel environments, shared GPU lease, daemon OS lock. Duplicate daemon
launch rejected in verification; six research harness checks passed.
Each new trial has immutable config/run/source snapshots. Counterevidence and
initial/periodic/final checkpoints remain available. Invalid CUDA numerics raise,
not successful resets. Three consecutive process failures pause for review.
pause/resume/status via python3 scripts/cuda_autoresearch.py; pause drains healthy
training and prevents the next launch. Existing per-trial settings remain frozen.

Rewardv2 explicitly encourages upright takeoff/hover, penalizes ground sitting,
payload speed, angular velocity and crashes. Pure motor RL; no PD. Fresh policies
have a documented equal-throttle initialization prior. Small learning-rate sweep
1e-5/3e-6/3e-5 and throttle priors.61/.60/.62, varying training seeds. Resume the
best same-phase checkpoint. Reused validation seeds51011-51013, eight domains
each, provisional only. No original audit seeds used.

Rank: crashes, sustained prototype hover, takeoffs, then destination error.
Prototype hover means payload within10cm and below0.2m/s for1s. Phase gates require
zero crashes,>=18/24 hover successes, mean error<100mm. Takeoff->hover->destination.
Not original90s obstacle route/sustained1mm contract, nor verified hardware accuracy.
Agent heartbeat reviews batches and concrete failures; daemon performs continuous
bounded parameter trials. C contact/attachment fidelity still needs work.

First trial cuda_auto_0001_takeoff completed65536 transitions in67.219s including
periodic validation. Initial:0 crashes,24/24 takeoffs,0 hover successes,1985.926mm
mean final error. Trained final:8 crashes,24/24 takeoffs,0 hover,5745.132mm error.
Rejected as improvement; initial checkpoint retained. Second trial launched
automatically. No successful learned takeoff/hover policy claimed.

Daemon was gracefully restarted after its healthy child drained to install source
snapshots; no healthy trainer interrupted. Live state and retained ledger:
state/cuda_research/loop.json, outbox/tether_cuda_loop.md, tmp/cuda-research/.
Heartbeat now supervises this daemon, not the old residual/MuJoCo campaign.
