# Runtime profiles

`medicalharness run-one` renders a fresh profile for the selected harness using the templates here. Model, endpoint, context window, output cap and browser endpoint come from CLI arguments. No user-global credentials or memory directory is copied.

Advanced users can pass `--registry /absolute/path/harness.json`. It uses harness-bench's JSON `{"models": {"hermes": {"adapter": "hermes_agent", ...}}}` schema; the selected key must match `--harness`, and referenced config files should use absolute paths. Keep credentials in environment variables.

Generated profiles make the products runnable; they do not claim to reproduce every historical paper setting. Versions and prerequisites are in [environment.md](../../docs/environment.md).
