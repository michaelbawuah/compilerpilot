# Experiment protocol

The research protocol follows five questions. Functional qualification is recorded separately from performance experiments.

| Experiment | Method | Acceptance evidence |
|---|---|---|
| E1: noise | Ten baseline observations; five raw replay runs each; fixed core, input and warmup | Median, raw runs, within-observation and across-observation spread. Conservative CV <2%. |
| E2: knobs | Vary one knob at a time from GCC O3/native, keeping all others fixed | Throughput, p50/p99, compile seconds, text bytes and correctness for each value. |
| E3: search | Random/genetic/surrogate/agent × seeds 0,1,2 × 40 unique builds | Best-so-far curves, per-seed results and complete attempt inventory. |
| E4: predictor | Deduplicate candidates, fit transforms/models inside each fold | Out-of-fold MAE in speedup percentage points and Spearman rank correlation; scatter. |
| E5: confirmation | Interleave top three and baseline for ten rounds | Noise-aware medians, 100-seed × 1M differential, complete CTest pass. |

Input generation is deterministic. Both inputs contain one million commands by default. Evaluation seed is 20261011; PGO training seed is 20261012. Exact input hashes and the Ledger commit are stored with every run. A manifest prevents silent reuse under a different data protocol.

Build time covers clean `cmake --build` wall time, adding both phases for PGO. Configure time is recorded separately. Text size is the `.text` section measured by `size -A` in bytes, displayed in KiB. Throughput selects Ledger's exact `fast-bitmap` engine key. Spread uses sample standard deviation. CV is a fraction, displayed as a percentage. The initial gate uses the larger within-observation or across-observation noise estimate.

The baseline is remeasured after each twenty completed unique builds. More than 2% drift or a failing noise/correctness gate discards that batch. Search stops; a new explicit run is required. Reports retain all failed attempts and never combine independent runs into a better-looking curve. Compile failures consume search budget, and every successful build must complete its correctness measurement before finish.

The deliberate mutation check copies Ledger to a temporary directory, changes buy crossing from `>=` to `>`, compiles it and requires the specific reference-checksum mismatch. A compile failure, timeout or missing executable cannot qualify the gate. The original submodule must remain unchanged.

The model receives typed capabilities and compact feedback. Compiler flags are selected from a finite allow-list. No provider output reaches a shell. Requests, hypotheses, tools and results are traced; credentials are redacted and excluded from artifacts. Exact model ID, temperature and provider seed support are recorded. Missing credentials or unsupported API settings yield an incomplete experiment.

PGO uses independent input bytes and seed. GCC generation/use stays in the same object directory. Clang raw profiles are merged with matching `llvm-profdata`; ThinLTO links with `lld`. Untrained helper executables can emit missing-profile warnings; missing profiles for the benchmark hot path invalidate PGO qualification. Unsafe math is not in the search space.
