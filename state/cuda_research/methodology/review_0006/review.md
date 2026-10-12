# Methodology review 0006: exploration partial results

Last reviewed trial index: 57. Pending: trials58/59, training seed15003. Do not
start a competing campaign or interrupt this healthy final pair. This review is
partial, not the final three-seed result. Preserve it when writing a later review.

|Seed|log_std=-4 final mm/crashes|log_std=-2 final mm/crashes|
|---|---:|---:|
|15001|3350.845 /0|9899.313 /0|
|15002|8485.475 /0|4472.607 /18|

All four completed trials reached262144 transitions and24 takeoffs with0 sustained
hover. Selected checkpoints remain initial best_step0. Smaller error in the second
wide-exploration trial cannot compensate for18 crash events. Crash counts may
include repeat events after resets; do not interpret18 as18 distinct crashed domains.
Initial reference is0 crashes24 takeoffs0 hover1080.487mm, not learned progress.

Source: native tmp/cuda-research/trials54-57/result.json and periodic evaluations;
compact evidence tether_sim/evidence/cuda_exploration_partial_v001.json. Trial57
crash events by validation seed:51011=4,51012=5,51013=9. First-domain saved
trajectory at step225 shows angular velocities [0.527,-0.545,0.010] for51011 and
[0.549,-0.589,0.010] for51012; those individual saved frames are not terminal.
These samples do not identify the exact contact or numerical cause of every crash.
Confidence high for recorded results; causal explanation remains uncertain.

Literature: HF full markdown2311.13081v2 retrieved here. Learning to Fly in Seconds
sectionIV discusses decaying exploration alongside a reward curriculum,
https://arxiv.org/html/2311.13081v2 . Our experiment raises initial Gaussian PPO
variance without reproducing its TD3 curriculum; the paper does not support a
claim that higher fixed initial variance alone should succeed. Prior original-source
fallback2501.18490 IV-B/V https://arxiv.org/html/2501.18490 discusses bounded
initialization/truncation, with different task/physics/sample budget. Neither tests
our tether backend. Do not claim method replication or verified hardware accuracy.

Next action: finish the third pair and aggregate common checkpoints plus final
paired outcomes. Unless strong contradictory evidence emerges, reject the wide
variance candidate for crash regression. Inspect contact/attachment and action
response fidelity before another optimizer sweep. No new hypothesis is launched
from this incomplete batch; reserved audit seeds remain untouched.
