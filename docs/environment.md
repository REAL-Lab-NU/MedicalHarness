# Environment

Use Python 3.11 or later and a Linux source checkout. The launcher provides task execution through MH-Lab and the product harnesses.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
medicalharness bootstrap
medicalharness doctor
medicalharness smoke
```

`bootstrap` clones [Qihoo360/harness-bench](https://github.com/Qihoo360/harness-bench) at `1025086a446653702b80cfb48babbeec35db6b2c` into a new `vendor/harness-bench` directory and applies the numbered patches. The bootstrap manifest records the commit and patch hashes. Use `--destination PATH`, `HARNESSBENCH_ROOT`, or `--hb PATH` to select another location. `--source /path/to/local/git/checkout` uses an existing Git checkout for offline installation.

The CPU smoke test runs a synthetic task with scripted model responses through MH-Lab, the request proxy, file tools, and deterministic scoring. It uses trusted synthetic inputs with namespace isolation disabled. Run the test suite with:

```bash
python -m pytest -q
```

## Python dependencies

The graph uses LangGraph 1.2.11, langchain-core 1.6.2, langchain-openai 1.6.2, and langchain-mcp-adapters 0.3.2. `requirements-observed.txt` lists the direct dependency versions. `requirements-runtime.lock.txt` pins the Python 3.11 runtime and test dependencies:

```bash
python -m pip install -r requirements-runtime.lock.txt
python -m pip install --no-deps -e .
```

Install the selected product CLI, model server, browser, and task containers separately using the requirements below.

## Product CLIs

| Harness | Version | Provider protocol |
|---|---|---|
| OpenClaw | 2026.9.2 | OpenAI Chat Completions |
| Hermes | 0.21.0 | OpenAI Chat Completions |
| ZeroClaw | 0.8.5 | OpenAI Chat Completions |
| Codex CLI | 0.156.1 | OpenAI Responses |
| Claude Code | 2.1.267–2.1.270 | Anthropic Messages |

Install the chosen CLI using its upstream instructions and put it on `PATH`, or supply `run-one --command /absolute/path/to/cli`. The configuration templates use the interfaces listed above.

Serve the checkpoint with its tool-call parser and required context length enabled. Set `--upstream` to the service base URL, such as `http://127.0.0.1:8000/v1`. Claude Code requires `/v1/messages`, and Codex requires `/v1/responses`.

For authenticated endpoints, export `MEDHARNESS_UPSTREAM_API_KEY` (preferred), `VLLM_API_KEY`, or `OPENAI_API_KEY`. The proxy reads the credential from the environment and redacts authorization headers in trace files. Keep traces from private tasks with their source data.

## Isolation

`run-one` uses `bubblewrap` (`bwrap`) and requires host support for user namespaces. Each agent sees its task sandbox and required runtime paths. Task definitions, graders, other run directories, and global agent state remain outside the agent's filesystem view. Network access follows the host's network policy. `--no-isolation` disables namespace isolation for trusted synthetic tests.

## Browser and task services

HealthAdmin uses these resources:

- `pip install -e '.[browser]'` and a compatible Chromium installation.
- The HealthAdminBench portal at commit `e71a8f4d6923037805b7f51fbbf608d12ea56cf5`, running locally (default `http://127.0.0.1:3002`).
- A browser with CDP enabled (default `http://127.0.0.1:9222`). Assign one browser/context to each episode until grading finishes.
- Node.js and `@playwright/mcp`. Set `PLAYWRIGHT_MCP_CLI=/absolute/path/to/node_modules/@playwright/mcp/cli.js`, or install it under the checkout's `node_modules`. Pin and record the package version used for the run.

The MCP shim exposes browser tools with JavaScript evaluation removed. OpenClaw uses its native browser attachment. Shell access to CDP follows the host's network policy.

HealthAgent export and grading use Docker Compose and the upstream task images and assets. See [data preparation](data.md).
