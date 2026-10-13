# Follow-up protocols

V002 (trial580): compare production geometry to a sloping, exactly rest-length ground geometry. Both cases initialize motors at balancing thrust; this isolates geometry, not the actual motors-off training startup. V001 results are preserved separately; no overwrite.

V003 (trial582): retain warm-motor comparisons and add both geometries with motors initially zero. Equal commands still request total system weight; existing25ms motor lag and1ms physical integration retained. Still air, no obstacles; actual randomized benchmark and learned controller not changed. Output v003 is pending when this protocol is written. Distinguish numerical diagnostic from physical calibration.

Training reset3 is opt-in corrected ground geometry. CPU production helper confirms each link exactly .6/13, drone.05m/payload.025m floor heights, legacy1 and mixed2 preserved, input untouched. C kernel/source/solver unchanged. No claim that geometry repair solves learned hover. Any learning comparison must retain frozen legacy evaluation for paired comparability and separately identify which start is used. No automatic default switch or benchmark rewrite.
