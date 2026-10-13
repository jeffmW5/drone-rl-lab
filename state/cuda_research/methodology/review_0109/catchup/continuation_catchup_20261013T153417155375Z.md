# Native continuation catch-up evidence

Recorded UTC: 2026-10-13T15:34:19.218619+00:00
Completed indices audited:856–874; last completed reviewed by this collection:874.

CPU/systemPython only, no GPU/build/train or source/state/config/review edits. Exact artifacts and first validation initial/final trajectories retained. Reused three-seed still-air validation is provisional/nonindependent. No hardware accuracy or strict1mm navigation result.

Three groups remain separate: matchedfactorial856–859 (earlier848–855 outside this catch-up); fixedsource4096 diagnostic instrumentation860/861; default weightcontinuation862+. No reward-only ranking or promotion claim.

| Trial | Group | Steps | Reward / obs / mask | Initial crashes / hover / altitude mm | Best step / crashes / hover / altitude mm | Final crashes / hover / altitude mm | Max final hold s |
|---|---|---:|---|---|---|---|---:|
|856|matched_factorial_seed70403|65536/65536|15/2/1|0/0/6.660|0/0/0/6.660|0/0/34.687|0.280|
|857|matched_factorial_seed70403|65536/65536|15/2/1|0/0/6.660|49152/0/0/5.586|0/0/29.365|0.300|
|858|matched_factorial_seed70403|262144/262144|15/2/1|0/0/6.660|0/0/0/6.660|24/0/743.544|0.000|
|859|matched_factorial_seed70403|262144/262144|15/2/1|0/0/6.660|49152/0/0/5.586|0/0/115.750|0.000|
|860|fixedsource_instrumentation_plumbing|4096/4096|15/2/1|24/0/522.991|4096/24/0/373.469|24/0/373.469|0.080|
|861|fixedsource_instrumentation_plumbing|4096/4096|15/2/1|24/0/534.253|4096/24/0/385.056|24/0/385.056|0.000|
|862|default_weight_continuation|65536/65536|14/1/1|0/0/6.357|0/0/0/6.357|24/0/708.204|0.000|
|863|default_weight_continuation|65536/65536|14/1/1|0/0/6.357|0/0/0/6.357|24/0/40.505|0.000|
|864|default_weight_continuation|65536/65536|14/1/1|0/0/6.357|0/0/0/6.357|24/0/198.800|0.000|
|865|default_weight_continuation|65536/65536|14/1/1|0/0/6.357|0/0/0/6.357|24/0/187.017|0.000|
|866|default_weight_continuation|65536/65536|14/1/1|0/0/6.357|0/0/0/6.357|24/0/299.580|0.000|
|867|default_weight_continuation|65536/65536|14/1/1|0/0/6.357|0/0/0/6.357|24/0/118.783|0.000|
|868|default_weight_continuation|65536/65536|14/1/1|0/0/6.357|0/0/0/6.357|24/0/93.782|0.000|
|869|default_weight_continuation|65536/65536|14/1/1|0/0/6.357|0/0/0/6.357|24/0/128.968|0.000|
|870|default_weight_continuation|65536/65536|14/1/1|0/0/6.357|0/0/0/6.357|24/0/212.362|0.000|
|871|default_weight_continuation|65536/65536|14/1/1|0/0/6.357|0/0/0/6.357|24/0/117.079|0.000|
|872|default_weight_continuation|65536/65536|14/1/1|0/0/6.357|0/0/0/6.357|0/0/905.548|0.000|
|873|default_weight_continuation|65536/65536|14/1/1|0/0/905.548|0/0/0/905.548|24/0/252.162|0.000|
|874|default_weight_continuation|65536/65536|14/1/1|0/0/905.548|0/0/0/905.548|24/0/3.145|0.000|

No sustained-zone-hover final successes in this snapshot: True. Budget-censored completed runs: []. Unexplained step shortfalls:[]

All source file paths/SHA256, periodic checkpoint evaluation metrics, exact first-run initial/final trajectories and their canonical hashes, and4.52-second carrier speed/extent samples are in the paired JSON. Auto-reset can make failed endpoints appear close to target: read crashes/escapes and sustained holds together.

Observation: default continuation repeatedly completes bounded runs without sustained hover. It should not be conflated with a method improvement or independent replications. Explanation remains unproved; preserve rejected longer/mask results and avoid another unmotivated reward sweep. New targeted reset curriculum is a separate future test, not an outcome from these artifacts.
