# Status -- Last Updated 2026-03-26

## Active Thread

**exp_057 -- Body-Frame Gate Observations** is currently training on RunPod.

- Started 2026-03-26 around 01:56 UTC with a 3600s budget
- Config: `configs/exp_057_body_frame_obs.yaml`
- Structural change: `body_frame_obs=true` (gate relative positions + normals in drone body frame)
- Companion tuning: `progress_coef=20` to reduce the over-aggression seen in exp_056
- Pod: `cw7hef3jd7kijr` at `213.144.200.243:10118`

## Latest Completed Result

**exp_056 -- Bilateral Progress Reward** completed on 2026-03-25.

- Final training reward: **28.92 mean** at 3.71M steps
- Peak training reward: **40.96** at iter 480
- Benchmark: **0 gates, 0 finishes, 0.64s average dive-crash**
- Takeaway: bilateral progress fixed directionality, but `progress_coef=50`
  overpowered altitude / stability and made the deterministic mean dive too aggressively

## Ready Next

- **exp_058 -- Soft-Collision Curriculum**
  - `soft_collision=true`
  - Phase 1 suppresses crash termination with penalty + respawn, then restores hard termination later
- **exp_059 -- Asymmetric Actor-Critic**
  - actor uses policy observations, critic also sees privileged gate state

Recommended order after `exp_057`: benchmark `exp_057`, then run `exp_058`,
then `exp_059` unless `exp_057` reveals a more urgent blocker.

## Reference Milestones

- **Legacy trajectory-following best lap:** `exp_016` -- 13.49s average lap, 2/10 Level 2 finishes
- **Early direct RaceCoreEnv best gate count:** `exp_023` -- 0.8 average gates on Level 2 benchmark
- **Current direct-racing benchmark reference:** `exp_046` -- 1.37s average flight, 0 gates, consistent flights toward gate 0

## Current Reading Of The Lab

- The trajectory-following line proved RL can finish but hit a structural ceiling.
- The active direct-racing line now has better directional reward shaping, but
  benchmark stability from the real start state is still the main bottleneck.
- Benchmark outcomes remain primary. Raw training reward is useful inside one
  experiment family, but not a global leaderboard across reward redesigns.
