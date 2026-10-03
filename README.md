<div align="center">

# MedicalHarness

**A Controlled Evaluation of LLMs and Agent Harnesses on Medical Tasks**

[⚙️ Installation](docs/environment.md) · [🚀 Run a task](docs/running.md) · [📚 Data setup](docs/data.md)

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Linux-555555?logo=linux&logoColor=white)

</div>

MedicalHarness studies how the system around an LLM changes its performance on medical tasks. This repository provides **MedicalHarnessBench**, task exporters and scoring, and **MH-Lab**, a configurable agent loop for studying context management, planning, and tool exposure. It also supports five product harnesses through a shared execution backend.

<p align="center">
  <img src="https://real-lab-nu.github.io/MedicalHarness/assets/overview.png" alt="MedicalHarness benchmark and study overview" width="100%">
</p>

## 🩺 Four medical task environments

| Environment | Tasks | Capability |
| --- | ---: | --- |
| 🧮 **MedTool** | 32 | Read patient data, choose clinical calculators, and compute numeric results. |
| 🗂️ **MedMemory** | 23 | Retrieve evidence from longitudinal records and answer factual questions. |
| 📋 **MedPlanning** | 28 | Complete clinical data workflows, including trial screening and EHR processing. |
| 🌐 **MedWeb** | 24 | Carry out workflows across an EMR, a payer portal, and a fax portal. |

Run tasks with **MH-Lab**, **OpenClaw**, **Hermes**, **ZeroClaw**, **Codex**, or **Claude Code**. Product CLIs, model services, and any required external assets are installed separately. The [environment guide](docs/environment.md) lists dependencies and supported interfaces.

## 🚀 Quick start

Use Python 3.11 or later on Linux:

```bash
git clone https://github.com/REAL-Lab-NU/MedicalHarness.git
cd MedicalHarness
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Prepare the pinned backend and check the execution path:

```bash
medicalharness bootstrap
medicalharness smoke
```

The smoke test runs on CPU with a synthetic task and scripted model responses. It exercises MH-Lab, the request proxy, file tools, and deterministic scoring.

### Run a benchmark task

MedTool task definitions and the fixture builder are included. Export a task, start your model service separately, and supply its model name and endpoint:

```bash
python scripts/export_calc_to_harnessbench.py --output tasks medcalc-001

medicalharness run-one \
  --tasks tasks \
  --task medcalc-001 \
  --harness mhlab \
  --model YOUR_SERVED_MODEL_NAME \
  --upstream http://127.0.0.1:8000/v1 \
  --output runs/medtool-example
```

Use `medicalharness doctor` to check local dependencies. Add `--dry-run` to `run-one` to inspect the generated configuration before launching an episode. See [running tasks](docs/running.md) for product harnesses, browser setup, time limits, and output files.

## 🧩 MH-Lab controls

| Mechanism | Launcher argument | What changes |
| --- | --- | --- |
| Context management | `--context-policy` | How accumulated history is retained, shortened, or summarized. |
| Planning | `--planning on/off` | The plan tool, planning instructions, and reminders. |
| Tool exposure | `--tool-bridge on/off` | Direct tool access or a discovery-and-dispatch interface. |

The [running guide](docs/running.md#mh-lab-switches) documents the available values and additional controls. The implementation is [run_langgraph_harness.py](scripts/harness/run_langgraph_harness.py).

## 📖 Documentation

| Guide | Contents |
| --- | --- |
| [Installation and environment](docs/environment.md) | Python dependencies, model protocols, product CLIs, and browser requirements. |
| [Data and task preparation](docs/data.md) | Task export, benchmark assets, and private MedMemory data. |
| [Running episodes](docs/running.md) | Commands, configuration switches, isolation, and scoring outputs. |
| [Benchmark provenance](PROVENANCE.md) | Upstream sources and the selected task sets. |

```text
medharness/       data loading, deterministic scoring, and the launcher
scripts/          task exporters and harness entrypoints
configs/          portable configuration templates
patches/          patches for the pinned execution backend
tests/            execution and scoring checks
docs/             installation, data, and usage guides
```

## 📚 Data and attribution

MedicalHarnessBench adapts tasks from MedMCP-Calc, MedMemoryBench, HealthAgentBench, and HealthAdminBench. Upstream code and data retain their original terms. MedMemory chart text is supplied separately. See [data setup](docs/data.md) and [third-party notices](THIRD_PARTY_NOTICES.md).
