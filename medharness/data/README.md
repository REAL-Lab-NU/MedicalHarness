# Benchmark package metadata

The current entrypoint is `medicalharness run-one`, backed by exported harness-bench tasks. See [data preparation](../../docs/data.md) and [running](../../docs/running.md).

`registry.load_records()` reads task inputs and metadata. `load_gold()` loads scoring references separately, outside the model workspace. Both use only the Python standard library.

Memory chart text, questions and answers are not distributed. Set `MEDMEMORY_DATA_ROOT` to a private directory containing `samples.jsonl` and `gold.jsonl`. The tracked manifest records the original selection and hashes; see the data guide for the expected format.
