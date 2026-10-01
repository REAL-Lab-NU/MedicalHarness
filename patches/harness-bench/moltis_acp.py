"""Drive Moltis over ACP (JSON-RPC 2.0 on stdio), because its CLI agent entry point is a stub.

WHY NOT `moltis agent`. The sibling `moltis.py` adapter shells out to `moltis agent --message ...
--session-id ... --timeout ...`. In the build we compiled (0.1.0, 20260902.03) that path is dead:
`crates/cli/src/main.rs` routes `Commands::Agent` to `moltis_agents::runner::run_agent`, which is a
stub that unconditionally bails with "run_agent requires a configured provider and tool registry; use
run_agent_loop instead" -- no feature gate, no working variant. The flags it passes do not exist
either; `moltis agent --help` offers only `--message` and `--thinking`. The only headless path linked
into the binary is `moltis acp`, so that is what this adapter speaks.

THE PROTOCOL, as established by probing the server (see moltis-acp-probe.py):
    -> initialize   {protocolVersion: 1, clientCapabilities: {...}}
    <- {protocolVersion, agentCapabilities, agentInfo: {name: "moltis", version}}
    -> session/new  {cwd, mcpServers: []}          <- cwd is how the agent learns the workspace
    <- {sessionId}
    -> session/prompt {sessionId, prompt: [{type: "text", text}]}
    <~ session/update notifications, streaming agent_message_chunk / agent_thought_chunk / tool calls
    <- {stopReason: "end_turn"}                     <- the turn is over when this reply lands

TOKEN ACCOUNTING. `[providers.openai].base_url` is rewritten to the usage proxy, the same way every
other harness here is routed, so token counts and the HB_PIN_* sampling pins apply to this row too.

RUNTIME DEFAULTS. The config this adapter copies must already carry the MedicalHarness overrides --
Moltis ships `agent_timeout_secs = 600`, `agent_max_iterations = 25`, an empty exec allowlist with
`approval_mode = "on-miss"`, and `sandbox.mode = "all"` with a read-only workspace mount. Any one of
those turns every episode into a zero for reasons that have nothing to do with the scaffold.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from harnessbench.adapters.base import BaseAdapter
from harnessbench.models import AdapterRunContext, AdapterRunResult
from harnessbench.usage_proxy import register_routes


def _resolve(raw: str | Path) -> Path:
    p = Path(os.path.expanduser(str(raw)))
    if not p.is_absolute():
        p = Path(__file__).resolve().parents[3] / p
    return p


def _route_provider_through_proxy(config_path: Path, proxy_base_url: str,
                                  proxy_routes_file: Path) -> None:
    """Point every `[providers.<id>].base_url` at the usage proxy."""
    if not config_path.is_file() or not proxy_base_url:
        return
    text = config_path.read_text(encoding="utf-8")
    routes: dict[str, dict[str, str]] = {}
    for match in re.finditer(r"(?ms)^\[providers\.([^\]]+)\]\n(.*?)(?=^\[|\Z)", text):
        provider, block = match.group(1), match.group(0)
        url_match = re.search(r'(?m)^base_url\s*=\s*"([^"]+)"', block)
        if not url_match:
            continue
        upstream = url_match.group(1).rstrip("/")
        if upstream.startswith(proxy_base_url.rstrip("/")):
            continue
        prefix = f"/moltis/{re.sub(r'[^A-Za-z0-9_.-]+', '-', provider)}"
        routes[prefix] = {"framework": "moltis", "provider": provider, "upstream": upstream}
        text = text.replace(block, block.replace(url_match.group(0),
                                                 f'base_url = "{proxy_base_url}{prefix}"'))
    if routes:
        register_routes(proxy_routes_file, routes)
        config_path.write_text(text, encoding="utf-8")


class MoltisAcpAdapter(BaseAdapter):
    """Run one Moltis episode as a single ACP session."""

    name = "moltis_acp"

    def run(self, ctx: AdapterRunContext) -> AdapterRunResult:
        command = str(ctx.model_config.get("command") or "moltis")
        source_config = ctx.model_config.get("user_config")
        if not source_config:
            return AdapterRunResult(ok=False, stderr="moltis_acp needs user_config (a moltis.toml)")
        source_config = _resolve(source_config)
        if not source_config.is_file():
            return AdapterRunResult(ok=False, stderr=f"missing Moltis config: {source_config}")

        # Per-episode config/data so concurrent slots cannot share session state or a sqlite file.
        home = ctx.sandbox / ".moltis"
        cfg_dir, data_dir = home / "config", home / "data"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_config, cfg_dir / "moltis.toml")
        defaults = source_config.parent / "defaults.toml"
        if defaults.is_file():
            shutil.copy2(defaults, cfg_dir / "defaults.toml")

        proxy_url = str(ctx.env.get("HARNESSBENCH_LLM_PROXY_URL") or "")
        routes_raw = ctx.env.get("HARNESSBENCH_LLM_PROXY_ROUTES")
        if proxy_url and routes_raw:
            _route_provider_through_proxy(cfg_dir / "moltis.toml", proxy_url, Path(routes_raw))

        env = os.environ.copy()
        env.update(ctx.env)
        env["MOLTIS_CONFIG_DIR"] = str(cfg_dir)
        env["MOLTIS_DATA_DIR"] = str(data_dir)
        # the binary links libzvec_c_api.so out of the cargo build tree
        extra_lib = str(ctx.model_config.get("ld_library_path") or "")
        if extra_lib:
            env["LD_LIBRARY_PATH"] = extra_lib + ":" + env.get("LD_LIBRARY_PATH", "")

        cmd = [command, "acp", "--config-dir", str(cfg_dir)]
        log = ctx.sandbox / "moltis-acp.log"
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, bufsize=1,
                                errors="replace",  # a stray non-UTF-8 byte must not kill the reader
                                cwd=str(ctx.workspace), env=env)

        replies: dict[Any, dict] = {}
        chunks: list[str] = []
        notes: list[str] = []
        stderr_lines: list[str] = []

        def pump_stderr() -> None:
            # Streamed, not buffered until the end: when an episode hangs it is killed from outside,
            # finish() never runs, and a buffered log is lost exactly when it is the only evidence.
            err_log = ctx.sandbox / "moltis-stderr.log"
            with err_log.open("a", encoding="utf-8") as fh:
                for line in proc.stderr:  # type: ignore[union-attr]
                    stderr_lines.append(line)
                    fh.write(line)
                    fh.flush()

        pump_error: list[str] = []

        def pump_stdout() -> None:
            try:
                _pump_stdout_inner()
            except Exception as exc:  # noqa: BLE001 - the thread dies silently otherwise
                pump_error.append(f"{type(exc).__name__}: {exc}")

        def _pump_stdout_inner() -> None:
            for line in proc.stdout:  # type: ignore[union-attr]
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "id" in msg and ("result" in msg or "error" in msg):
                    replies[msg["id"]] = msg
                    continue
                if msg.get("method") == "session/update":
                    upd = ((msg.get("params") or {}).get("update") or {})
                    kind = upd.get("sessionUpdate")
                    # `content` is a dict for message/thought chunks but a LIST of content blocks for
                    # tool_call updates. Assuming the dict shape kills this reader on the agent's very
                    # first tool call, and the turn then looks like a silent hang.
                    content = upd.get("content")
                    if isinstance(content, dict):
                        text = content.get("text") or ""
                    elif isinstance(content, list):
                        text = "".join(c.get("text") or "" for c in content if isinstance(c, dict))
                    else:
                        text = ""
                    if kind == "agent_message_chunk":
                        chunks.append(text)
                    elif kind:
                        notes.append(kind)

        threading.Thread(target=pump_stderr, daemon=True).start()
        threading.Thread(target=pump_stdout, daemon=True).start()

        def send(obj: dict) -> None:
            proc.stdin.write(json.dumps(obj) + "\n")  # type: ignore[union-attr]
            proc.stdin.flush()  # type: ignore[union-attr]

        # Stage markers land on disk as they happen. Writing the log only at the end means a hang
        # is indistinguishable from a slow model: there is nothing to look at while it is stuck.
        def mark(stage: str) -> None:
            with log.open("a", encoding="utf-8") as fh:
                fh.write(f"{time.strftime('%H:%M:%S')} {stage}\n")

        mark(f"spawned pid={proc.pid} cwd={ctx.workspace}")
        deadline = time.time() + max(60, int(ctx.timeout_sec or 3600))

        def wait_for(rid: Any) -> dict | None:
            while time.time() < deadline:
                if rid in replies:
                    return replies[rid]
                if pump_error:
                    mark(f"!! stdout reader died: {pump_error[0]}")
                    return None
                if proc.poll() is not None and rid not in replies:
                    return None
                time.sleep(0.25)
            return None

        def finish(ok: bool, err: str, meta: dict) -> AdapterRunResult:
            try:
                proc.kill()
            except OSError:
                pass
            out = "".join(chunks)
            # append: the stage markers written during the run are the only record of where a hang
            # happened, and write_text would erase them at exactly the moment they matter
            with log.open("a", encoding="utf-8") as fh:
                fh.write(f"\n--- result ok={ok} err={err[:300]} ---\n"
                         f"--- agent message ({len(out)} chars) ---\n{out[:4000]}\n"
                         f"--- updates ---\n{json.dumps(notes[:400])}\n")
            return AdapterRunResult(ok=ok, command=cmd, stdout=out,
                                    stderr=err or "".join(stderr_lines[-50:]), metadata=meta)

        try:
            mark("-> initialize")
            send({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                  "params": {"protocolVersion": 1,
                             "clientCapabilities": {"fs": {"readTextFile": False,
                                                           "writeTextFile": False}}}})
            init = wait_for(1)
            mark(f"<- initialize {'ok' if init else 'TIMEOUT'}")
            if init is None or "error" in init:
                return finish(False, f"initialize failed: {json.dumps(init)[:400]}", {"stage": "initialize"})

            mark("-> session/new")
            send({"jsonrpc": "2.0", "id": 2, "method": "session/new",
                  "params": {"cwd": str(ctx.workspace), "mcpServers": []}})
            new = wait_for(2)
            mark(f"<- session/new {'ok' if new else 'TIMEOUT'}")
            if new is None or "error" in new:
                return finish(False, f"session/new failed: {json.dumps(new)[:400]}", {"stage": "session/new"})
            sid = (new.get("result") or {}).get("sessionId")
            if not sid:
                return finish(False, f"no sessionId: {json.dumps(new)[:400]}", {"stage": "session/new"})

            mark("-> session/prompt")
            send({"jsonrpc": "2.0", "id": 3, "method": "session/prompt",
                  "params": {"sessionId": sid, "prompt": [{"type": "text", "text": ctx.prompt}]}})
            done = wait_for(3)
            mark(f"<- session/prompt {'ok' if done else 'TIMEOUT'}")
            if done is None:
                return finish(False, "session/prompt did not return before timeout",
                              {"stage": "session/prompt", "timed_out": True,
                               "timeout_sec": ctx.timeout_sec})
            if "error" in done:
                return finish(False, f"session/prompt error: {json.dumps(done)[:400]}",
                              {"stage": "session/prompt"})
            stop = (done.get("result") or {}).get("stopReason")
            return finish(True, "", {"stage": "done", "stop_reason": stop,
                                     "session_id": sid, "updates": len(notes)})
        except (BrokenPipeError, OSError) as exc:
            return finish(False, f"ACP transport died: {exc}", {"stage": "transport"})
