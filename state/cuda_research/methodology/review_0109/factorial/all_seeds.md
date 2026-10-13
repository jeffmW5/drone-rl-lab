# Three-seed independent factorialtrajectory assessment
Three pairedtrainingseeds70401-70403, frozen2x2lossmask/duration, reused identical stillair evaluation domains. Savedtrajectories only; noGPU execution or newbenchmarks.
Retrieved 2026-10-13T14:51:15.872198+00:00.

|Seed|Trial|Mask|Steps|Finalescapes|Finalhover|Finalpeakhold(s)|Maxperiodichold(s)|Altitudeerr(mm)|4.52s speed(m/s)|4.52s extent(m)|Beststep|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|70401|848|1|65536|0|0|0.360|0.360|59.834|0.501|0.589|0|
|70401|849|2|65536|0|0|0.300|0.300|56.991|0.646|0.850|0|
|70401|850|1|262144|24|0|0.240|0.360|522.991|0.162|0.004|0|
|70401|851|2|262144|24|0|0.060|0.320|534.253|0.137|0.003|0|
|70402|852|1|65536|0|0|0.060|0.160|9.411|1.037|0.780|32768|
|70402|853|2|65536|0|0|0.140|0.140|35.795|1.303|0.978|0|
|70402|854|1|262144|0|0|0.360|0.360|65.113|0.800|0.838|32768|
|70402|855|2|262144|24|0|0.160|0.180|535.497|0.137|0.003|0|
|70403|856|1|65536|0|0|0.280|0.340|34.687|0.683|1.018|0|
|70403|857|2|65536|0|0|0.300|0.300|29.365|0.382|0.776|49152|
|70403|858|1|262144|24|0|0.000|0.340|743.544|1.162|1.697|0|
|70403|859|2|262144|0|0|0.000|0.560|115.750|0.863|1.206|49152|

The final sample columnsare usually4.52s; whenescapes occurred,theymaydescribe resetratherthanactualstability. Exactfirstfailureevents absentfromlegacy artifacts. Allfinal/periodiczonehovercountszero;fullcounters determineholds. Uniformaltitudeimprovement isnothoveror1mmtracking. Nohardwarecalibration claim.

## Interpretation

Both masks finish escape-free on all3short runs and escape on2of3long finals. No1s hover at any saved checkpoint; maximum periodic hold0.56s occurs in corrected70403 and regresses to0s at final. Which mask survives differs by seed: original70402, corrected70403. Corrected short altitude error improves2pairs and regresses1; stopping remains unsolved. Longer learning after identical65k prefixes produces final failure2of3underbothmasks, while retaining the original70402escape-free counterexample. Mathematical loss-mask correctness and reliable learned-flight improvement are separate questions.

Do not average small post-reset positions/speeds as recovery. All validation-seed trajectories are identical still-air domains. Three training seeds are paired replications of the optimizer experiment, not evidence of randomized environmental robustness. No causal mechanism established solely from these saved trajectories; original strict1mm navigation and hardware accuracy remain unachieved.
