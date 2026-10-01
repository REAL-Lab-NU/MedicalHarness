# Runtime profiles

`medicalharness run-one` renders a fresh profile for the selected harness using the templates here. Model, endpoint, context window, output cap and browser endpoint come from CLI arguments. Credentials are read from environment variables.

Pass `--registry /absolute/path/harness.json` to use a custom registry. It uses harness-bench's JSON `{"models": {"hermes": {"adapter": "hermes_agent", ...}}}` schema. The selected key must match `--harness`. Use absolute paths for referenced configuration files.

Versions and prerequisites are in [environment.md](../../docs/environment.md).
