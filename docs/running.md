# Run one episode

Install and bootstrap as described in [environment.md](environment.md). A complete GPU-free check is:

```bash
medicalharness smoke
```

This actually runs the graph, calls a canned localhost provider through the usage proxy, writes a file and grades it. It prints the result path. No model inference occurs.

## First benchmark task

Calc data and its deterministic scorer are included. Export a single task, then point the launcher at your separately running model service:

```bash
python scripts/export_calc_to_harnessbench.py --output tasks medcalc-001
medicalharness run-one \
  --tasks tasks --task medcalc-001 --harness mhlab \
  --model YOUR_SERVED_MODEL_NAME --upstream http://127.0.0.1:8000/v1 \
  --output runs/calc-example
```

Each output directory must be new; the launcher never resumes or overwrites a scored episode. Add `--dry-run` to render the profile and command without launching the agent. `doctor` only checks local prerequisites, not endpoint availability.

The defaults are 3,600 seconds for agent execution, 16,384 output tokens and an 81,920-token total context window. MH-Lab additionally keeps its original 5% counting reserve. Change these with `--timeout`, `--max-tokens` and `--context-window`; the model service must support the chosen window. These portable defaults are not an assertion that every paper model used identical settings. The extra `--grading-timeout` bounds the outer runner's scoring/cleanup allowance and does not extend the agent deadline.

## Five products

Replace `--harness mhlab` with `openclaw`, `hermes`, `zeroclaw`, `codex`, or `claudecode`. The launcher renders a fresh profile using the same task, proxy and oracle. The CLI must already be installed. Use `--command PATH` for a non-default binary, or `--registry PATH` for a validated custom harness-bench registry whose `models` key matches `--harness`. See the provider protocol requirements in [environment.md](environment.md).

## MH-Lab switches

The implementation is `scripts/harness/run_langgraph_harness.py`; it retains the paper's graph and trace instrumentation. Arguments on `run-one` include:

| Argument | Values | Effect |
|---|---|---|
| `--planning` | `on`, `off` | Plan tool, protocol and per-turn plan reminders |
| `--context-policy` | `none`, `elide`, `elide_recall`, `summarize`, `elide_then_summarize` | Context tiers T0–T4 |
| `--tool-bridge` | `on`, `off` | Expose task tools indirectly through discovery and dispatch |
| `--verification` | `on`, `off` | One reminder when required submission files are absent/empty |

The verification switch checks artifact presence, not answer correctness. `--seed` and `--temperature` populate missing sampling fields at the proxy, retaining explicit product values. The native MH-Lab CLI exposes additional research controls such as action-space and summary thresholds; use a custom registry to forward them.

## Browser episode

After provisioning the portal, MCP and CDP browser:

```bash
export HEALTHADMIN_PORTAL_BASE_URL=http://127.0.0.1:3002
export HB_CDP_URL=http://127.0.0.1:9222
python scripts/export_healthadmin_to_harnessbench.py --output tasks medadmin-016
medicalharness run-one \
  --tasks tasks --task medadmin-016 --harness mhlab --tool-mode browser \
  --model YOUR_SERVED_MODEL_NAME --upstream http://127.0.0.1:8000/v1 \
  --cdp-url "$HB_CDP_URL" --output runs/browser-example
```

The hooks reset portal state before the episode and the oracle reads the resulting state. Browser submission is application state, not a final chat answer. Run one episode at a time per CDP browser. An optional `--mcp-config` accepts the usual `{"mcpServers": {"playwright": {"command": "...", "args": [...]}}}` format.

## Results and limits

`OUTPUT/run.json` records the setup; `OUTPUT/config/` contains rendered profiles; `OUTPUT/runner.log` contains process diagnostics; `OUTPUT/results/` contains the original task result. Each sandbox retains `usage-proxy/requests.jsonl`, raw model requests/responses and the submitted files. MH-Lab also writes `events.jsonl`, `context-transforms.jsonl` and `langgraph-manifest.json`.

Inspect `adapter_results[*].ok`, timeout metadata, `oracle_result.outcome_score` and submission files separately. A process starting successfully does not mean it answered correctly. The release skips the generic optional LLM process judge and preserves deterministic task oracles. Its raw file-based scores are not a replacement for the paper's later answer-recovery and analysis pipeline.

These commands have been checked with a synthetic local provider. Running medical tasks against a real model, or running a third-party product/browser, additionally depends on your supplied model service, CLI versions and external assets; the installer does not provision them.
