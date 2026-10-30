# CompilerPilot measured results

**The E1 noise gate failed. Performance experiments E2–E5 were not performed.** The recorded compiler/PGO builds below are functional qualifications only; no speedup or winning build is claimed.

Recorded 0 trials; the E3 inventory contains 0 experiments. 0 correctness-passing, measurement-valid trials belong to the selected single runs. 0 correctness/build failures; 0 correctness-passing trials excluded by the measurement gate.

E3 comparison: **incomplete**. Target: four strategies × 3 seeds × 40 candidate builds.

| Strategy | Observed builds per seed | Coverage |
|---|---|---|
| random | no trials | 0/3 seeds; incomplete |
| genetic | no trials | 0/3 seeds; incomplete |
| surrogate | no trials | 0/3 seeds; incomplete |
| agent | no trials | 0/3 seeds; incomplete |

## Experiment status

| Experiment | Recorded status |
|---|---|
| E1: machine noise | failed noise gate; 10 baseline observations |
| E2: one-knob sweep | not performed |
| E3: fixed-budget search comparison | not performed |
| E4: held-out predictor quality | not performed; insufficient valid search trials |
| E5: performance-winner confirmation | not performed; no performance winner |

## Experiment inventory

Experiments are never combined within a strategy/seed. The fixed selection rule uses the earliest complete predeclared protocol run; if none qualifies, it shows the longest single partial run, breaking ties by start time and run ID. Throughput does not affect this selection. Every attempt remains listed below, including replaced or failed runs.

| Run | Strategy | Seed | Builds | Valid | Failed | Noise-excluded | Selected | Protocol status |
|---|---|---:|---:|---:|---:|---:|---|---|

No search experiments are recorded.

## E1: baseline noise

Baseline files are reported separately by experiment. The qualification measurement is a smoke check; E1 requires ten baseline observations. Noise gate: conservative coefficient of variation below 2%; candidate significance threshold: three baseline standard deviations.

| Baseline run | Observations | Throughput (M cmd/s) | p99 (ns) | Spread (M cmd/s) | CV | Compile (s) | .text (KiB) | Correct | Noise gate / E1 |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| [qualification](qualification-baseline.json) | 1 | 15.063 | 280.0 | 2.075 | 12.88% | 20.14 | 94.89 | pass | FAILED; E1 incomplete |
| [E1-quiet](runs/E1-quiet/baseline.json) | 10 | 18.043 | 271.0 | 2.455 | 13.61% | 18.63 | 94.89 | pass | FAILED; E1 measurement complete |

![E1 raw baseline replay throughput and CV](figures/e1-E1-quiet.png)

Every raw replay is plotted. Diamond markers show each observation median; the lower panel shows its CV. The gate uses the larger within-observation or across-observation noise estimate. Figure sources and hashes are recorded in [e1-plot-data.json](e1-plot-data.json).

## Functional build qualification

These raw measurements establish that each compiler/PGO path builds, replays, and passes the per-build differential oracle. They ran on a noisy shared host and are not a controlled performance comparison. No speedup ratio is calculated.

| Qualification build | Raw replays | Median (M cmd/s) | p99 (ns) | CV | Compile (s) | .text (KiB) | Correctness |
|---|---:|---:|---:|---:|---:|---:|---|
| [GCC baseline](qualification-baseline.json) | 5 | 15.063 | 280.0 | 12.88% | 20.14 | 94.89 | pass |
| [GCC PGO](qualification-pgo.json) | 5 | 16.422 | 271.0 | 6.20% | 32.92 | 69.36 | pass |
| [Clang ThinLTO + PGO](clang-pgo-qualification.json) | 5 | 15.412 | 220.0 | 13.90% | 37.81 | 61.41 | pass |

Correctness-gate mutation check: **pass**. The intentionally modified matching logic built successfully, and the harness rejected it. Pinned Ledger source unchanged: yes. Evidence: [gate-check.json](gate-check.json).

Baseline functional correctness qualification: **pass** across 100,000,000 differential commands (100 seeds × 1,000,000 orders), with 15/15 CTest targets completed. This validates the baseline's behavior; it is not E5 performance-winner confirmation. Evidence: [full-validation-baseline.json](full-validation-baseline.json).

## Recorded search builds

Each strategy row is its best observed candidate, ranked by median raw speedup; repeated measurements of that candidate are summarized with medians. Values include only trials passing both gates. A raw ratio is not a confirmed improvement.

| Build | Throughput (M cmd/s) | p99 (ns) | Compile (s) | .text (KiB) | Raw speedup | Baseline spread | Correctness | Noise significance |
|---|---:|---:|---:|---:|---:|---:|---|---|

No trials currently pass both correctness and measurement gates. No performance winner is reported.

## Predictor validation

Not enough distinct valid candidates for held-out validation (0 available).

## Interpretation and limits

Only recorded measurements appear here. Search results do not establish a final winner until interleaved confirmation, the full 100-seed differential test, and ctest pass. These observations describe one benchmark on its recorded machine; CPU-specific flags and synthetic PGO training may not transfer. Model and provider versions can change agent behavior. Unavailable hardware counters must be reported as unavailable rather than zero.
