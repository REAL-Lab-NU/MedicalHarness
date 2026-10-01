# Run one episode

Install and bootstrap as described in [environment.md](environment.md). Check the execution path with:

```bash
medicalharness smoke
```

This CPU test uses scripted model responses, calls file tools through MH-Lab, and grades a synthetic submission. The command prints the result path.

## First benchmark task

Calc data and its deterministic scorer are included. Export a task and connect the launcher to your model service:

```bash
python scripts/export_calc_to_harnessbench.py --output tasks medcalc-001
medicalharness run-one \
  --tasks tasks --task medcalc-001 --harness mhlab \
  --model YOUR_SERVED_MODEL_NAME --upstream http://127.0.0.1:8000/v1 \
  --output runs/calc-example
```

Choose a new output directory for each episode. `--dry-run` writes the profile and execution command for inspection. `medicalharness doctor` checks local dependencies.

| Setting | Default | Argument |
|---|---:|---|
| Agent execution time | 3,600 seconds | `--timeout` |
| Output token cap | 16,384 tokens | `--max-tokens` |
| Total context window | 81,920 tokens | `--context-window` |

Set the context window to a value supported by the model service. MH-Lab uses a 5% context-counting reserve. `--grading-timeout` sets the outer runner's scoring and cleanup allowance. The agent execution deadline uses `--timeout`.

## Product harnesses

Replace `--harness mhlab` with `openclaw`, `hermes`, `zeroclaw`, `codex`, or `claudecode`. The launcher generates a profile for the task, request proxy, and oracle. Install the selected CLI first. Use `--command PATH` to select its executable, or `--registry PATH` to supply a harness-bench registry. The selected key under `models` must match `--harness`. Provider protocols are listed in [environment.md](environment.md).

## MH-Lab switches

MH-Lab is implemented in `scripts/harness/run_langgraph_harness.py`. The launcher exposes these controls:

| Argument | Values | Effect |
|---|---|---|
| `--planning` | `on`, `off` | Plan tool, planning instructions, and per-turn reminders |
| `--context-policy` | `none`, `elide`, `elide_recall`, `summarize`, `elide_then_summarize` | History retention, elision, recall, and summarization |
| `--tool-bridge` | `on`, `off` | Tool discovery and dispatch interface |
| `--verification` | `on`, `off` | One reminder when required submission files are absent or empty |

The verification switch checks that required files exist and contain content. The proxy fills missing sampling fields from `--seed` and `--temperature` and retains values explicitly supplied by the product. A custom registry can pass additional MH-Lab arguments, including action-space settings and summary thresholds.

## Browser episode

Prepare the portal, MCP server, and CDP browser, then run:

```bash
export HEALTHADMIN_PORTAL_BASE_URL=http://127.0.0.1:3002
export HB_CDP_URL=http://127.0.0.1:9222
python scripts/export_healthadmin_to_harnessbench.py --output tasks medadmin-016
medicalharness run-one \
  --tasks tasks --task medadmin-016 --harness mhlab --tool-mode browser \
  --model YOUR_SERVED_MODEL_NAME --upstream http://127.0.0.1:8000/v1 \
  --cdp-url "$HB_CDP_URL" --output runs/browser-example
```

The hooks reset portal state before execution. The oracle grades the resulting application state. Run one episode at a time per CDP browser. `--mcp-config` accepts the format `{"mcpServers": {"playwright": {"command": "...", "args": [...]}}}`.

## Outputs and scoring

| Path | Contents |
|---|---|
| `OUTPUT/run.json` | Run settings |
| `OUTPUT/config/` | Generated harness profiles |
| `OUTPUT/runner.log` | Execution log |
| `OUTPUT/results/` | Task scores and execution metadata |
| Sandbox `usage-proxy/` | Requests, responses, and token accounting |

MH-Lab also writes `events.jsonl`, `context-transforms.jsonl`, and `langgraph-manifest.json` in the sandbox.

`adapter_results[*].ok` records execution status. Timeout metadata records deadline events, and `oracle_result.outcome_score` contains the task score. Deterministic oracles grade submitted files for file-based tasks and final portal state for browser tasks.
