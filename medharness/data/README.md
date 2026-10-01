# Benchmark package metadata

Run exported harness-bench tasks with `medicalharness run-one`. See [data preparation](../../docs/data.md) and [running](../../docs/running.md).

`registry.load_records()` reads task inputs and metadata. `load_gold()` loads scoring references separately, outside the model workspace. Both use only the Python standard library.

For MedMemory, set `MEDMEMORY_DATA_ROOT` to a private directory containing `samples.jsonl` and `gold.jsonl`. The tracked manifest records task identifiers and hashes. The data guide specifies the expected file format.
