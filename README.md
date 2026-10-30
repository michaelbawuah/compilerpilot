# CompilerPilot has moved into Ledger

CompilerPilot is now maintained in **[Ledger under `tools/compilerpilot`](https://github.com/michaelbawuah/Ledger/tree/main/tools/compilerpilot)**. It uses the live Ledger source, with no submodule or separate checkout.

- [Current setup and commands](https://github.com/michaelbawuah/Ledger/blob/main/tools/compilerpilot/README.md)
- [Measured results and limitations](https://github.com/michaelbawuah/Ledger/blob/main/tools/compilerpilot/results/summary.md)
- [Dedicated CompilerPilot CI in Ledger](https://github.com/michaelbawuah/Ledger/actions/workflows/compilerpilot.yml)

This repository retains the initial standalone implementation and qualification evidence for reproducibility. Development and CI have moved to Ledger; the standalone CI workflow is retired. The [original qualified snapshot](https://github.com/michaelbawuah/compilerpilot/tree/7459c23b6e07e5e5065a3d0fcdc7f4b7519c3c3c) remains available in history.

The historical E1 measurement recorded 13.61% baseline variation against a 2% limit. The performance sweep and strategy comparison have not run, and the LLM agent has not been exercised against a live model. No compiler speedup or agent-over-baselines advantage is claimed.
