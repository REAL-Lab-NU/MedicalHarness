# Data and task preparation

The current runner consumes harness-bench task directories: `task.yaml`, `prompt.txt`, a public `fixtures/` directory and a separate `oracle_grade.py`. Only fixtures are copied into the model workspace. Keep task definitions, gold files and graders outside the workspace.

| Benchmark | Distributed here | External requirement |
|---|---|---|
| Calc | 32 task records, scorer and SQLite fixture builder | None for export; a model endpoint for inference |
| Memory | Selection/provenance manifest, loader and scorer | User-provided chart/question/answer data |
| HealthAgent | Task metadata, frozen subset and Compose templates | Pinned upstream checkout, Docker images and licensed/public assets |
| HealthAdmin | Synthetic task metadata, frozen subset and deterministic scorer | Pinned portal checkout, running portal and CDP browser |

Exporters accept `--output DIRECTORY`, optional task IDs, and `--overwrite` only when replacement is intended. They refuse to silently replace existing task directories. The full frozen subsets use 32 Calc, 23 Memory, 28 HealthAgent and 24 HealthAdmin tasks (107 total); supplying custom data changes that benchmark.

```bash
python scripts/export_calc_to_harnessbench.py --output tasks
python scripts/export_healthadmin_to_harnessbench.py --output tasks
```

The HealthAdmin export itself does not launch the portal/browser. Set `HEALTHADMIN_PORTAL_BASE_URL` and `HB_CDP_URL` before export/run to use non-default local endpoints. The scorer reports deterministic portal checks; no remote semantic judge is enabled by the release launcher.

## Memory stays private

Chart text, questions and answers are not bundled or downloaded by the installer. Set `MEDMEMORY_DATA_ROOT` to your private directory with `samples.jsonl` and `gold.jsonl`, then run:

```bash
export MEDMEMORY_DATA_ROOT=/absolute/path/to/your/private/memory-package
python scripts/export_memory_to_harnessbench.py --output tasks
```

The original package format has one sample per line with `id`, `input` and `metadata`. `input` contains the rendered chart divided by `SESSION n` lines, followed by an `END OF CHART` delimiter and a `QUESTION: ...` line. The gold sidecar has matching IDs and accepted aliases. A **synthetic format example**, not a released benchmark item:

```json
{"id":"synthetic-1","input":"SESSION 1\nA synthetic value is 42.\nEND OF CHART\nQUESTION: What was the synthetic value?","metadata":{"descriptors":{"context_tokens":25}}}
```

```json
{"id":"synthetic-1","gold":{"accepted_aliases":["42"]}}
```

The first line belongs in `samples.jsonl`, the second in `gold.jsonl`. This lets you test your own authorized data without the original restricted text. To reproduce the original subset, you need the matching private package; its hashes and upstream source commit `8d708070d0dfe017676947b33d65c43302c79034` are retained in `medharness/data/memory/manifest.json`. The source selection workflow is not rebuilt by these exporters.

Exported Memory task fixtures contain chart text. Do not commit or distribute your generated `tasks/`, `runs/`, raw traces or workspaces without checking the relevant data terms. Questions and gold do not belong in package or wheel artifacts.

## HealthAgent

Obtain [microsoft/HealthAgentBench](https://github.com/microsoft/HealthAgentBench) at commit `6caf4395603991e66562e0c6a5e89e32296a8484` and acquire the upstream assets under their original terms. The source's `medharness/data/healthagent/manifest.json` documents family-specific licenses and requirements.

```bash
export HEALTHAGENTBENCH_ROOT=/absolute/path/to/HealthAgentBench
export HEALTHAGENT_EXTERNAL_ROOT=/absolute/path/to/public-slide-inputs
export HEALTHAGENT_VERIFIER_CACHE=/absolute/path/to/verifier-mask-cache
python scripts/export_healthagent_to_harnessbench.py --output tasks medagent-017
```

This exporter **does run Docker Compose** to materialize only the agent-visible workspace. It does not start a model or GPU service. The task oracle later invokes the unmodified upstream verifier inside the task image, with network disabled. Complete any required asset downloads before running.

Large inputs mounted outside the task workspace remain in `HEALTHAGENT_EXTERNAL_ROOT`. Add `run-one --allow-path "$HEALTHAGENT_EXTERNAL_ROOT"` when that directory is inside an otherwise hidden tree. This allowance must contain only public task inputs, not verifier labels or answer keys. The documented original full set includes asset-gated families that are not prerequisites for the synthetic smoke.

## Package boundaries

`medharness.data.registry.load_records` and `load_gold` are framework-independent. The retained manifests describe provenance; the release does not ship medical chart text, container caches, browser builds, checkpoint weights or collected model trajectories.
