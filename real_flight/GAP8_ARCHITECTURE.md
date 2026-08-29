# GAP8 — how the cores are actually arranged

Written 2026-08-29 to settle a specific question: is one of the cores an
orchestrator, and does that explain the 5.64× multi-core speedup?

**Short answer: yes there is an orchestrator core, but it is not one of the 8,
so it does not explain the 5.64×.**

## The layout

GAP8 has **nine** RISC-V cores in two separate power and clock domains:

| | Cores | Role |
| --- | --- | --- |
| **SoC / Fabric Controller (FC)** | 1 | The orchestrator. Runs the OS, owns the peripherals (camera, SPI, UART, hyperbus), and dispatches work to the cluster. Has its own clock and voltage domain. |
| **Cluster** | 8 | The compute array. Shares a 64 KB L1 TCDM scratchpad and a shared instruction cache. Can be powered down entirely when idle. |

512 KB of L2 SRAM sits with the SoC. Off-chip hyperflash/hyperram is L3 — that
is where DroNet's 1.29 MB of weights live, paged in per layer by DORY's tiler.

## Does a cluster core orchestrate the other seven?

No. In the PULP runtime, the FC hands a task to the cluster, and the receiving
cluster core calls `pi_cl_team_fork(...)`. **The calling core participates in
the parallel work** rather than sitting out — it runs the forked function
alongside the others and there is an implicit barrier at the end, so it returns
only once every core has finished.

So `CORE=8` really means 8 cores computing. The orchestration cost lives on the
FC, in a different clock domain, not inside the parallel region.

## What this means for our 5.64×

If a cluster core were lost to coordination, the ceiling would be 7× and 5.64×
would be 81% of it. That is not the situation. Against a true 8× ceiling,
**5.64× is ~70% parallel efficiency**, and the gap is ordinary Amdahl overhead:
per-layer setup that runs serially, DMA and L3 paging waits while weights are
fetched, shared-instruction-cache contention, and the fork/barrier itself once
per parallel region.

Practical consequence: **there is no lost core to reclaim.** Getting past
63.82 ms/inference means less serial work or less L3 traffic — a smaller model
that fits without paging, or fewer layers — not a scheduling fix. That is
consistent with the frame-budget conclusion that model size is the lever.

## Clock domains

From `gap_sdk`'s own `gap8/rtos/pulp/pulp-os/kernel/gap/pmu_driver.c`:

```c
#define SOC_MIN_FREQ       150000000   /* FC / SoC domain */
#define SOC_MAX_FREQ       250000000
#define CLUSTER_MIN_FREQ    87000000   /* where the layers execute */
#define CLUSTER_MAX_FREQ   175000000
```

Frequency is voltage-coupled (`SOC_FV_SLOPE` / `CLUSTER_FV_SLOPE`), and
DroNet's generated `main.c` boots at `PMU_set_voltage(1000, 0)` — the low
voltage rail.

**On our board these headers are aspirational.** 150 MHz on the cluster hangs
reproducibly at the same layer across three attempts that varied FC clock and
voltage independently (FACT-025). 100 MHz is the verified-correct ceiling.
Root cause unknown, and unresolved — do not plan against 175 MHz.

## Sources

- GAP8 Hardware Reference Manual — https://gwt-website-files.s3.amazonaws.com/gap8_datasheet.pdf
- GreenWaves GAP8 docs — https://github.com/GreenWaves-Technologies/gap8_docs
- PULP cluster fork semantics — https://github.com/pulp-platform/pulp-rt-examples/blob/master/cluster/fork/cluster.c
- Clock limits — `gap_sdk` `pmu_driver.c`, quoted in `GAP8_PERF_RESULT.md`

Local measurements: FACT-023 (1 core), FACT-024 (8 cores), FACT-025 (clock).
