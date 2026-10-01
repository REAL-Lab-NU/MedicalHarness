# Pinned evaluation runner patches

Run `medicalharness bootstrap`. It clones the commit in `UPSTREAM.txt`, applies every numbered patch in lexical order, and writes `medicalharness-bootstrap.json` with SHA-256 hashes. Choose a new directory for the checkout.

Patches 0001–0013 configure provider routing and accounting, task workspaces, timeout results, output caps, Anthropic parsing, product configuration, process isolation, and submission checks. Submission checks are disabled by default in the launcher.

Patch 0014 reads evaluator roots and runtime paths from environment variables and configuration. It adds upstream authentication from environment variables and redacts authentication headers in traces.

`moltis_acp.py` supplies a module imported by the upstream adapter registry.

Prepare task packages following [data.md](../../docs/data.md).
