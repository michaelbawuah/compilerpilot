# Qualification status — October 30, 2026

The implementation is ready for a controlled research run. The performance study is blocked at the required noise gate.

| Requirement | Evidence / status |
|---|---|
| Separate project, pinned Ledger | [Public CompilerPilot repository](https://github.com/michaelbawuah/compilerpilot); Ledger commit c0dbffd6a036feb27e225362a92df9792d8e844a; source unchanged |
| Flag allow-list, isolated builds, timeouts | Implemented and tested; no free-form shell flags or unsafe math |
| Correctness gates | Real buy-crossing mutation compiled and was rejected by the reference-checksum gate |
| GCC baseline | Replay checksums and 5 × 100K differential pass |
| GCC PGO | Instrument, separate-seed training and profile use pass; core profile-consumption evidence verified |
| Clang ThinLTO + PGO | Instrument, merge, lld link, profile use and correctness pass; core profile counts verified |
| Baseline full validation | 100M commands, zero mismatches; all 15 CTest suites complete and pass, checked via JUnit |
| E1: ten quiet baseline observations | Completed; conservative CV 13.61%, above 2% limit |
| E2/E3: knob sweep and strategy comparison | Not run because E1 failed; no speedup or strategy ranking claimed |
| E4: predictor accuracy | Model and out-of-fold evaluation implemented; no measured training table available |
| E5: winning-build performance confirmation | Interleaved confirmer implemented/tested; no winner exists yet |
| Hardware counters | perf runs, but VM reports requested hardware events unsupported |
| Live LLM tool-use experiment | API adapters and protocol tests implemented; requires provider credential and pinned model |
| Public repository | Created at https://github.com/michaelbawuah/compilerpilot; source and qualification evidence published |

`qualification-clang-pgo.json` retains an earlier failed attempt during toolchain setup. The later `clang-pgo-qualification.json` records the successful clean attempt. Both remain visible. Qualification timings are raw observations from a noisy host; they are not comparable speedup results.

To continue: use a quiet Linux host, configure one provider credential as an environment variable, rerun E1, and only proceed if the gate passes. `scripts/run_study.py` preflights credentials and stops on noise. Ledger remains an unchanged pinned submodule in the separate public repository.
