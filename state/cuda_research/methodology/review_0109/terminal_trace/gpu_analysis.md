# GPU saved terminaltrace independent analysis
Read-only savedGPU terminaltrace artifacts860/861; initialfixedsource replay reproduces850/851finalpolicy. New pre-reset statedecodedfromFP32obs, exactabsolute/episode steps; no newGPUexecution, production orbenchmark change by auditor.
Retrieved 2026-10-13T14:54:00.497047+00:00.

|Trial|Stage|Sourcelegacyexact|Terminals|Failedcode|Firsttime(s)|Extent(m)|Height(m)|Horizspeed(m/s)|TotalSpeed(m/s)|Tilt(deg)|Pitch(deg)|PitchRate(rad/s)|
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|860|initial|True|24|3|4.340|2.003145|0.724237|1.652249|1.694190|10.303843|10.143099|0.129599|
|860|final|notclaimed|24|3|4.140|2.018275|0.744158|1.789817|1.798273|10.895020|10.770551|0.143164|
|861|initial|True|24|3|4.360|2.015104|0.843425|1.624652|1.660198|10.365783|9.497604|0.107164|
|861|final|notclaimed|24|3|4.160|2.020055|0.858211|1.771224|1.777899|10.724877|10.038965|0.117208|

Flag3 meansX/Yzoneboundary failure, not numericfailure orfloorcontact. Initial stages are fixed850/851finalpolicy reproductions; final stages follow4096additionaltrainingtransitions and cannotbeclaimed identicaltolegacyreference. Traceexactlegacycomparison preservesoldmetricsandtrajectory, showing instrumentation doesnot improveflight. Repeatedlanes/seedsareidenticalstillairdomains. Timebeyondfirstterminal thenauto-reset isnotrecovery.

The initialsource850terminalisnearhoverheight butmovesoutward acrosspositiveX at1.64m/s withpositivepitchabout10degreesandincreasingpitchrate. Terminal motorpitchtorqueisalsopositive, so thisobservedstate isnotbrakingX. Thisfactsupportsfailed attitude/velocity recovery butdoesnotproveauniversalcommandsignerror: neuralresponseandconstraintmodesdependonstate. No fulltrajectorycredit orcausaloptimizerdefect inferredfromoneendstate.


Independent verification additionally compares860/861 initial dictionaries recursively against850/851 final dictionaries after removing only evaluation_terminal_trace: both exactlyequal in structureandvalues. ReferenceSHA256matchesmanifest. All24events perarm arefinite firstzonefailures(code3), untruncated, absolute/episodestepsagree, decodedcarrierextent>=2m; uniquefirsttimes4.34/4.36s. This resolves resetambiguityfor850/851 specifically, without inferringunsavedterminalstateforotherfactorialruns.
