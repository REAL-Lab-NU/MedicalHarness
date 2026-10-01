# Third-party code and data

Code and data obtained from upstream projects retain their original licenses and attribution. Copies of the available upstream license texts are in [third_party_licenses](third_party_licenses/). A project-level code license does not replace the terms attached to benchmark materials or model weights.

| Component | Source | Terms recorded by the benchmark package |
| --- | --- | --- |
| MedMCP-Calc task definitions | [SPIRAL-MED/MedMCP-Calc](https://github.com/SPIRAL-MED/MedMCP-Calc) | MIT for task definitions and code. The upstream EHR database has separate data-use terms and is not included. |
| MedMemoryBench | [AQ-MedAI/MedMemoryBench](https://github.com/AQ-MedAI/MedMemoryBench) | Upstream license statements differ between sources. This release distributes identifiers, hashes, and loader/scorer code, without record or question/answer text. |
| HealthAgentBench task definitions | [microsoft/HealthAgentBench](https://github.com/microsoft/HealthAgentBench) | MIT for task code. Individual clinical datasets retain their own terms. |
| HealthAdminBench | [ethan-kinetic/HealthAdminBench](https://github.com/ethan-kinetic/HealthAdminBench) | Apache-2.0 for the synthetic task definitions and portal code. |
| harness-bench | [Qihoo360/harness-bench](https://github.com/Qihoo360/harness-bench) | Installed as a separate pinned checkout. Retain its license when redistributing it. Local modifications are supplied as patches. |

HealthAgentBench's selected tasks use clinical-trial materials, MIMIC-IV demo data, and CAMELYON16 slides. These assets are prepared separately. MIMIC-IV demo carries ODbL terms and CAMELYON16 is recorded as CC0 in the task manifest. No credentialed MIMIC-IV/MIMIC-CXR, EHRSHOT, or CT-RATE corpus is distributed in this repository.

The full provenance and per-material notes are in `medharness/data/*/manifest.json`. Read the upstream terms when obtaining assets or model checkpoints. Product harnesses and models are installed separately and keep their own licenses.
