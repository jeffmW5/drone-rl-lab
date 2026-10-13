# Factorial70401 independent savedtrajectory review
Retrieved 2026-10-13T14:04:23.354105+00:00. No GPU evaluation or production writes.

Saved deterministic5s evaluation artifacts, full14node physics5 stillair zone1, reusedseeds51011..13. Sparse saved frames do not measure continuous holds; full-evaluation hold counters used. No new simulator/GPU run, production edit or trial interruption.

Initial checkpoint weights exactly equal: True.

|Trial|Step|Kind|Escapes|Hover|Full-eval max hold(s)|Carrier altitude error(mm)|Sample max XY extent(m)|Last sample height(m)|Last sample speed(m/s)|Lastsample tilt(deg)|
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|850|0|initial|0|0|0.100|6.660|0.410|0.911|0.882|4.732|
|850|16384|periodic|0|0|0.320|9.403|0.561|0.889|0.276|1.556|
|850|32768|periodic|0|0|0.220|9.898|0.718|0.889|0.415|0.447|
|850|49152|periodic|0|0|0.120|19.833|0.699|0.887|0.745|3.396|
|850|65536|periodic|0|0|0.360|59.834|0.589|0.846|0.501|1.603|
|850|81920|periodic|0|0|0.240|103.835|1.437|0.813|0.923|3.431|
|850|98304|periodic|24|0|0.000|547.418|1.527|0.054|0.110|0.004|
|850|114688|periodic|24|0|0.000|432.127|1.711|0.088|0.327|0.028|
|850|131072|periodic|24|0|0.000|267.735|1.413|0.188|0.583|0.134|
|850|147456|periodic|24|0|0.000|356.674|1.849|0.127|0.449|0.076|
|850|163840|periodic|24|0|0.080|493.454|1.623|0.067|0.218|0.018|
|850|180224|periodic|24|0|0.260|644.997|1.978|0.726|1.439|8.298|
|850|196608|periodic|24|0|0.300|749.104|1.622|0.736|1.296|7.451|
|850|212992|periodic|24|0|0.260|644.618|1.998|0.726|1.468|9.560|
|850|229376|periodic|24|0|0.000|418.603|1.729|0.095|0.350|0.044|
|850|245760|periodic|24|0|0.080|634.405|1.348|0.050|0.000|0.000|
|850|262144|final|24|0|0.240|522.991|1.523|0.059|0.162|0.014|
|850|262144|periodic|24|0|0.240|522.991|1.523|0.059|0.162|0.014|
|851|0|initial|0|0|0.100|6.660|0.410|0.911|0.882|4.732|
|851|16384|periodic|0|0|0.220|25.265|0.725|0.874|0.431|0.639|
|851|32768|periodic|0|0|0.000|36.140|1.009|0.867|0.712|1.013|
|851|49152|periodic|0|0|0.000|52.122|1.274|0.857|0.959|3.355|
|851|65536|periodic|0|0|0.300|56.991|0.850|0.851|0.646|1.559|
|851|81920|periodic|0|0|0.280|46.084|0.850|0.859|0.620|1.671|
|851|98304|periodic|0|0|0.300|77.466|1.231|0.836|0.735|3.806|
|851|114688|periodic|24|0|0.180|711.600|1.786|0.804|1.269|7.030|
|851|131072|periodic|24|0|0.100|689.716|1.868|0.812|1.313|8.140|
|851|147456|periodic|24|0|0.220|717.753|1.755|0.807|1.247|7.792|
|851|163840|periodic|0|0|0.300|65.184|1.151|0.853|0.713|4.819|
|851|180224|periodic|0|0|0.320|22.407|0.700|0.883|0.418|2.479|
|851|196608|periodic|0|0|0.280|71.344|1.345|0.854|0.964|6.318|
|851|212992|periodic|0|0|0.320|40.518|0.908|0.873|0.600|4.369|
|851|229376|periodic|0|0|0.320|104.709|1.233|0.826|0.874|6.135|
|851|245760|periodic|24|0|0.240|672.877|1.889|0.790|1.432|9.634|
|851|262144|final|24|0|0.060|534.253|1.516|0.056|0.137|0.012|
|851|262144|periodic|24|0|0.060|534.253|1.516|0.056|0.137|0.012|

Both masks evaluated on deterministic stillair trajectories;24episode counts do not mean24 independentdomains. Unique saved trajectories percheckpoint are listed inJSON. Full-evaluation counters determine sustainedhover; sparseframes alone cannot. Lastsaved periodicframe normally4.52s, not final5s endpoint. Altitudeerror is5s scalar and can differ sharply fromlastsampleheight. Escaped then reset endpoints are never evidence ofsuccess.

851 stateatread: {"status": "complete", "steps": 262144}; finalavailable=True. If incomplete, do notattribute a finaloutcome.

## Independent interpretation

851 completed while files were being inspected; final262144result and periodic262144evaluation both included. Both arms start from exactly identical policyweights and identical savedinitialtrajectory. At65536, mask1/2 had zeroescapes and zerohover, full-eval peakhold0.36/0.30s. Altitudeerrors59.834/56.991mm are similar, while saved4.52s mask2speed0.646m/s and extent0.850m are worse than mask1speed0.501m/s and extent0.589m. Both violate0.2m/s criterion despite near-targetheight0.846/0.851m and smalltilt1.603/1.559degrees.

Beyond65536 the masked arm has transientbehavioral differences: at180224 mask2 haszeroescapes,22.407mm altitudeerror,0.32s peakhold,4.52s speed0.418m/s andextent0.700m; mask1 has24escapes,644.997mm altitudeerror,0.26s hold,4.52s speed1.439m/s andextent1.978m. Mask2 also haszeroescapes163840/196608/212992/229376, whereas mask1 has24escapes atall thosecheckpoints. This is a singlematchedseed temporarydifference, not reliableimprovement or sustainedhover. Mask2 nevertheless escapes at114688/131072/147456/245760 and final; behavior is nonmonotonic.

Final262144 botharms have24escapes andzerohover. Full-eval maxhold0.24s(mask1) and0.06s(mask2); altitudeerrors522.991/534.253mm. Saved4.52s states are again nearinitialground andXYcenter (heights0.059/0.056m, speeds0.162/0.137m/s, almostidentityattitude), consistent with post-failure reset. Small lastsampleextent orspeed therefore doesnot demonstratebraking/recovery. Exactterminaltimes/velocities are absent fromthese sparse snapshots: allsavedframes labelterminalfalse even though fullcountersrecord24escapes. Do not derivefirstescapetime fromthisartifact. A later read-only first-terminal diagnostic would be needed for that.

Eachcheckpoint's three savedseedtrajectories are exactly identical (uniquetrajectorydomains=1). Each8-agentrun likewiseusesidenticalstillairdomain;24failurecount is repeateddeterministicbehavior, not24 independentfailures. Initialaltitudeerror6.660mm is lower thanallperiodic/finalvalues, and bothselectedbest_step0/promotedfalse. The requested1s stablehover remainsunmet throughout. Longertrainingdidnot solve thisseed; maskedlearnerhas transient escape reduction but regresses atfinal. Replicate frozen remainingseedarms before causal/promotionclaims. No evidence here alone provesphysicswrong or establisheswhichloss termcauseddrift.
