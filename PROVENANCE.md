# Benchmark provenance

MedicalHarnessBench combines four task environments. The checked-in package manifests record their upstream revisions, item identifiers, content hashes, and task selection. The experiment-facing names and source package names map as follows:

| Environment | Source package | Selected tasks | Pinned upstream revision |
| --- | --- | ---: | --- |
| MedTool | MedMCP-Calc (`calc`) | 32 | `81f3ff2bd603de10ccef55d73c0c2e91181514d9` |
| MedMemory | MedMemoryBench (`memory`) | 23 | `8d708070d0dfe017676947b33d65c43302c79034` |
| MedPlanning | HealthAgentBench (`healthagent`) | 28 | `6caf4395603991e66562e0c6a5e89e32296a8484` |
| MedWeb | HealthAdminBench (`healthadmin`) | 24 | `e71a8f4d6923037805b7f51fbbf608d12ea56cf5` |

HealthAgentBench and HealthAdminBench packages describe upstream pools of 54 and 135 tasks. Their `subset-v1.json` files select the 28 and 24 tasks used in MedicalHarnessBench.

MedMemory's manifest records task identifiers and hashes. Supply its questions, answers, and longitudinal charts in a private data directory. Task exporters keep model-visible inputs separate from oracle files. See [the data guide](docs/data.md) for preparation and [third-party notices](THIRD_PARTY_NOTICES.md) for upstream terms.
