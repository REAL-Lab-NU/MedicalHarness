# Environment

Use Python 3.11 or later and a Linux source checkout. The single-episode launcher and MH-Lab are the supported entrypoints.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
medicalharness bootstrap
medicalharness doctor
medicalharness smoke
```

`bootstrap` clones [Qihoo360/harness-bench](https://github.com/Qihoo360/harness-bench) at `1025086a446653702b80cfb48babbeec35db6b2c` into `vendor/harness-bench`, then applies the ordered patches. It refuses to modify an existing destination. Use `--destination PATH`, `HARNESSBENCH_ROOT`, or a `--hb PATH` argument for another location. `--source /path/to/local/git/checkout` supports offline bootstrapping from an existing object database. The bootstrap manifest records the commit and patch hashes. No services or model downloads start automatically.

The smoke uses a canned localhost OpenAI endpoint, the actual MH-Lab graph, the usage proxy, native filesystem tools and a deterministic oracle. It requires no GPU, product CLI, medical data or browser. Its score proves that the plumbing works, not model performance. It deliberately uses trusted synthetic inputs without namespace isolation. The full default test suite is also model-free:

```bash
python -m pytest -q
```

The four direct graph dependencies are pinned to versions observed in the study runtime: LangGraph 1.2.11, langchain-core 1.6.2, langchain-openai 1.6.2 and langchain-mcp-adapters 0.3.2. `requirements-observed.txt` is provenance, not a transitive lockfile. Product CLIs and vLLM are separate installations; they are not installed by the Python package.

`requirements-runtime.lock.txt` records the full dependency set from a fresh Python 3.11 installation that passed `pip check` and the complete synthetic smoke. To use those exact package versions in a new environment, install the lock before the editable checkout:

```bash
python -m pip install -r requirements-runtime.lock.txt
python -m pip install --no-deps -e .
```

The lock includes the test dependencies. It does not pin external CLIs, model servers, browser binaries, Docker images or npm packages. For another Python version or platform, the normal editable install can resolve a compatible environment instead.

## Product CLIs

| Harness | Version reported in the paper | Required provider protocol |
|---|---|---|
| OpenClaw | 2026.9.2 | OpenAI Chat Completions |
| Hermes | 0.21.0 | OpenAI Chat Completions |
| ZeroClaw | 0.8.5 | OpenAI Chat Completions |
| Codex CLI | 0.156.1 at study end | OpenAI Responses |
| Claude Code | 2.1.267–2.1.270 across model blocks | Anthropic Messages |

Install the chosen CLI following its upstream instructions and put it on `PATH`, or supply `run-one --command /absolute/path/to/cli`. Claude Code is a separately licensed product. Updated CLI releases may change config schemas; the generated profiles target the interfaces listed above. There is no automatic installation or product upgrade.

Serve your chosen checkpoint separately. `--upstream` is an OpenAI base URL such as `http://127.0.0.1:8000/v1`. For Claude Code, the service must additionally accept `/v1/messages`; Codex requires `/v1/responses`. Merely having Chat Completions available is insufficient for those two. Enable the checkpoint's appropriate tool-call parser and sufficient context length in your model server. This release does not choose GPU placement or launch vLLM.

For an authenticated endpoint, export `MEDHARNESS_UPSTREAM_API_KEY` (preferred), `VLLM_API_KEY`, or `OPENAI_API_KEY`. The launcher preserves explicit values and the proxy injects the selected credential upstream; generated profiles use placeholders. Authorization headers are redacted in proxy trace files. Traces still contain task inputs and outputs and should remain private when the data requires it.

## Isolation and browser dependencies

Real `run-one` episodes require `bubblewrap` (`bwrap`) and a host that permits user namespaces. The adapter hides the checkout, task/oracle tree, other run artifacts and global agent state, exposing the current sandbox and required runtime paths. This is filesystem/process isolation, **not a network firewall**. Use a dedicated environment and your own network policy for stronger restrictions. `--no-isolation` is intended for trusted synthetic smoke inputs.

HealthAdmin needs these separately provisioned resources:

- `pip install -e '.[browser]'` and a compatible Chromium installed with Playwright or your system package manager.
- The HealthAdminBench portal, built from commit `e71a8f4d6923037805b7f51fbbf608d12ea56cf5`, running locally (default `http://127.0.0.1:3002`).
- A browser with CDP enabled (default `http://127.0.0.1:9222`). Assign one browser/context to one episode until grading finishes.
- Node.js and Playwright MCP. Install `@playwright/mcp` separately and set `PLAYWRIGHT_MCP_CLI=/absolute/path/to/node_modules/@playwright/mcp/cli.js`, or install it under the checkout's `node_modules`. Record the version you use; no verified study-wide npm lock was recovered for publication.

The shared MCP shim removes JavaScript evaluation tools. OpenClaw uses its native browser attachment. Shell-capable products still require inspection of browser-channel traces; this shim does not by itself prevent shell code from reaching CDP. HealthAgent export and grading require Docker/Compose and the upstream task images/assets. See [data.md](data.md).
