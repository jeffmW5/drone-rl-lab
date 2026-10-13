# Fine-reward GPU artifact addendum: corrected dispatch and partial first pair

Retrieved UTC:2026-10-13T16:13:54.436916+00:00. Formalreviewprefix remains879. Actual finepair is881(R15)/882(R16), seed70601. Treatmentstatus:complete.

The earlier trial-number inference was wrong: frozen880intent is the second recovery reset5 arm, seed70502,262144transitions. Newfinequeue was created after880launched, so its first pair movedto881/882. This addendum preserves the correction; no frozenintent was relabeled.

Both finepairplans usegroundreset3/collective observation2/physics5/mask2,65536transitions and600s budget. Initialweights match759best exactly, all initialtensors finite, and frozen core-source hashes match. R16helper snapshot exists andmatches productionSHA 4e468a9713dca4b512b43bad6ea714e0a0379316239ab60ed0b1cec547e8afe9.

881 finished65536 transitions in107.819s without budgetcensoring. Initial/final:0failures,24takeoffs,0groundhover; altitudeerror6.660→57.834mm andpeakhold0.10→0.16s. Initialcheckpoint remainsselected(beststep0), not learnedprogress.

882 initialphysical metrics/trajectories are available for equality comparison.

Trial881 step16384:failures0,takeoffs24,hover0,hold0.100s,altitudeerror27.932mm;scanned30points,maxspeed1.289m/s,maxtilt6.229deg,minimumcounterfactual R16core0.050000,negativecorepoints0.

Trial881 step32768:failures0,takeoffs24,hover0,hold0.140s,altitudeerror34.929mm;scanned30points,maxspeed1.337m/s,maxtilt6.944deg,minimumcounterfactual R16core0.050000,negativecorepoints0.

Trial881 step49152:failures0,takeoffs24,hover0,hold0.120s,altitudeerror25.561mm;scanned30points,maxspeed1.184m/s,maxtilt6.278deg,minimumcounterfactual R16core0.050000,negativecorepoints0.

Trial881 step65536:failures0,takeoffs24,hover0,hold0.160s,altitudeerror57.834mm;scanned30points,maxspeed1.474m/s,maxtilt7.527deg,minimumcounterfactual R16core0.050000,negativecorepoints0.

Trial882 step16384:failures0,takeoffs24,hover0,hold0.060s,altitudeerror27.572mm;scanned30points,maxspeed1.449m/s,maxtilt6.958deg,minimumcounterfactual R16core0.050000,negativecorepoints0.

Trial882 step32768:failures0,takeoffs24,hover0,hold0.160s,altitudeerror64.220mm;scanned30points,maxspeed1.462m/s,maxtilt7.439deg,minimumcounterfactual R16core0.050000,negativecorepoints0.

Trial882 step49152:failures0,takeoffs24,hover0,hold0.160s,altitudeerror51.791mm;scanned30points,maxspeed1.417m/s,maxtilt7.123deg,minimumcounterfactual R16core0.050000,negativecorepoints0.

Trial882 step65536:failures24,takeoffs24,hover0,hold0.180s,altitudeerror733.094mm;scanned54points,maxspeed2.135m/s,maxtilt8.560deg,minimumcounterfactual R16core0.050000,negativecorepoints0.

These are sparsepost-step samples and exactterminal observations where available, not fulltrainingbuffers. Absenceof negativecore inthis sample doesnot remove the synthetic highspeed/invertedcounterexample. Postreset endpoints and initializedholds donot establishhover. Numericalfinite checks, terminal flags/times/tilts, sourcehashes and partial treatmentstatus are preserved inJSON.

NoR16benefit orpromotion isclaimed while thepairis incomplete. Healthyserial training was preserved; noGPUcalls,source/queue/metricchanges orreview.md edit occurred.
