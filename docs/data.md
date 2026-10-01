# Data and task preparation

Task packages contain `task.yaml`, `prompt.txt`, an agent-visible `fixtures/` directory, and a separate `oracle_grade.py`. The runner copies fixtures into the model workspace. Task definitions, gold files, and graders stay outside that workspace.

| Benchmark | Included materials | External requirements |
|---|---|---|
| Calc | 32 task records, scorer, and SQLite fixture builder | Model endpoint for inference |
| Memory | Task manifest, loader, and scorer | Chart, question, and answer package |
| HealthAgent | Task metadata, selected task IDs, and Compose templates | Upstream checkout, Docker images, and task assets |
| HealthAdmin | Synthetic task metadata, selected task IDs, and deterministic scorer | Portal checkout, running portal, and CDP browser |

Exporters accept `--output DIRECTORY` and optional task IDs. Use `--overwrite` to replace existing exported tasks. The benchmark contains 32 Calc, 23 Memory, 28 HealthAgent, and 24 HealthAdmin tasks.

```bash
python scripts/export_calc_to_harnessbench.py --output tasks
python scripts/export_healthadmin_to_harnessbench.py --output tasks
```

Prepare the HealthAdmin portal and browser before running an episode. Set `HEALTHADMIN_PORTAL_BASE_URL` and `HB_CDP_URL` before export and execution to select their endpoints. The scorer evaluates deterministic portal checks.

## Prepare MedMemory data

Provide `samples.jsonl` and `gold.jsonl` in a private data directory and set `MEDMEMORY_DATA_ROOT`:

```bash
export MEDMEMORY_DATA_ROOT=/absolute/path/to/your/private/memory-package
python scripts/export_memory_to_harnessbench.py --output tasks
```

Each sample has `id`, `input`, and `metadata` fields. `input` contains the chart divided by `SESSION n` lines, followed by an `END OF CHART` delimiter and a `QUESTION: ...` line. The gold sidecar pairs each sample ID with accepted answer aliases.

Synthetic example for `samples.jsonl`:

```json
{"id":"synthetic-1","input":"SESSION 1\nA synthetic value is 42.\nEND OF CHART\nQUESTION: What was the synthetic value?","metadata":{"descriptors":{"context_tokens":25}}}
```

Matching entry in `gold.jsonl`:

```json
{"id":"synthetic-1","gold":{"accepted_aliases":["42"]}}
```

The exporter converts the prepared data package into runner tasks. `medharness/data/memory/manifest.json` records the benchmark's content hashes and upstream commit `8d708070d0dfe017676947b33d65c43302c79034`. Use the matching data package to run the benchmark task set.

Exported fixtures and execution traces contain task text. Keep these files with the private data package and follow the source data's distribution terms.

## Prepare HealthAgent assets

Obtain [microsoft/HealthAgentBench](https://github.com/microsoft/HealthAgentBench) at commit `6caf4395603991e66562e0c6a5e89e32296a8484`. Prepare its task images and assets using the upstream instructions. Dataset licenses and requirements are recorded in `medharness/data/healthagent/manifest.json`.

```bash
export HEALTHAGENTBENCH_ROOT=/absolute/path/to/HealthAgentBench
export HEALTHAGENT_EXTERNAL_ROOT=/absolute/path/to/public-slide-inputs
export HEALTHAGENT_VERIFIER_CACHE=/absolute/path/to/verifier-mask-cache
python scripts/export_healthagent_to_harnessbench.py --output tasks medagent-017
```

The exporter uses Docker Compose to prepare the agent-visible workspace. The task oracle runs the upstream verifier inside the task image with network access disabled. Download the required assets before export.

`HEALTHAGENT_EXTERNAL_ROOT` holds public task inputs mounted outside the workspace. Add `run-one --allow-path "$HEALTHAGENT_EXTERNAL_ROOT"` when that directory is inside an otherwise hidden tree. Store verifier labels and answer keys separately from agent-visible inputs.
