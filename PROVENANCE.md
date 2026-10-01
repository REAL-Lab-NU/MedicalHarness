# Benchmark provenance

MedicalHarnessBench combines four task environments. The checked-in package manifests record their upstream revisions, item identifiers, content hashes, and task selection. The experiment-facing names and source package names map as follows:

| Environment | Source package | Selected tasks | Pinned upstream revision |
| --- | --- | ---: | --- |
| MedTool | MedMCP-Calc (`calc`) | 32 | `81f3ff2bd603de10ccef55d73c0c2e91181514d9` |
| MedMemory | MedMemoryBench (`memory`) | 23 | `8d708070d0dfe017676947b33d65c43302c79034` |
| MedPlanning | HealthAgentBench (`healthagent`) | 28 | `6caf4395603991e66562e0c6a5e89e32296a8484` |
| MedWeb | HealthAdminBench (`healthadmin`) | 24 | `e71a8f4d6923037805b7f51fbbf608d12ea56cf5` |

HealthAgentBench and HealthAdminBench packages also describe the larger upstream pools. The selected task IDs are in their `subset-v1.json` files. Loading the paper subset should produce 28 and 24 tasks respectively, rather than the complete pools of 54 and 135.

MedMemory's manifest is included, but its questions, answer text, and longitudinal charts must be supplied separately. Task exporters keep model-visible inputs separate from oracle files. See [the data guide](docs/data.md) for preparation and [third-party notices](THIRD_PARTY_NOTICES.md) for upstream terms.
