# Supported entrypoints

- `medicalharness bootstrap`, `doctor`, `smoke`, `run-one`: portable launcher in `medharness/release.py`.
- `run_langgraph_harness.py`: MH-Lab agent loop and module switches.
- `run_claude_code.py`: generic CLI bridge for an installed Claude Code binary.

Run each through the launcher so workspace, proxy, budget and scoring are created consistently. See [running.md](../../docs/running.md).
