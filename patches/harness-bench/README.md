# Pinned evaluation runner patches

Run `medicalharness bootstrap`. It clones the commit in `UPSTREAM.txt`, applies every numbered patch in lexical order, and writes `medicalharness-bootstrap.json` with SHA-256 hashes. Existing destinations are never modified by bootstrap.

Patches 0001–0013 preserve the study's runner corrections: provider routing and accounting, correct task workspaces, timeout results, output ceilings, Anthropic parsing, product config compatibility, isolated agent processes, and optional submission-check experiments. The submission-check feature is not enabled by the release launcher.

Patch 0014 is the publication portability layer: evaluator roots come from environment/configuration instead of one machine's directories; runtime paths are explicit; optional upstream authentication is injected without putting credentials in profiles; auth headers are redacted in traces. The historical patches retain their original context, so machine paths in an old diff are not the final runtime configuration.

`moltis_acp.py` is copied because the historical adapter registry imports it. Moltis is not among the five supported products and is not used by the public launcher.

No task assets are included in the upstream clone. Export or supply task packages separately; see [data.md](../../docs/data.md). The raw proxy traces contain task data and are not public release artifacts by default.
