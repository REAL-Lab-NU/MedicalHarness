#!/usr/bin/env python
"""MCP proxy that enforces the MedWeb browser-action interface.

The proxy removes JavaScript execution tools from ``tools/list`` and rejects
their invocation through ``tools/call``. Agents interact with the portal UI,
and the oracle grades the resulting portal state. Other requests and responses
pass through to the Playwright MCP server.

Usage as the MCP server command in a harness configuration:

    python scripts/mcp_browser_no_js.py --cdp-endpoint http://127.0.0.1:9232

All arguments are passed to ``@playwright/mcp``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import threading

# JavaScript execution tools excluded from the browser-action interface.
BLOCKED = {"browser_evaluate", "browser_run_code_unsafe"}

MCP_CLI = os.environ.get(
    "PLAYWRIGHT_MCP_CLI", str(Path(__file__).resolve().parents[1] / "node_modules/@playwright/mcp/cli.js")
)


def _pump_stderr(proc: subprocess.Popen) -> None:
    for line in proc.stderr:  # type: ignore[union-attr]
        sys.stderr.write(line)
        sys.stderr.flush()


def main() -> int:
    proc = subprocess.Popen(
        ["node", MCP_CLI, *sys.argv[1:]],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1,
    )
    threading.Thread(target=_pump_stderr, args=(proc,), daemon=True).start()

    # Rejected tool-call request IDs receive local error responses.
    refused: dict[object, str] = {}
    lock = threading.Lock()

    def client_to_server() -> None:
        for line in sys.stdin:
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                proc.stdin.write(line)  # type: ignore[union-attr]
                proc.stdin.flush()  # type: ignore[union-attr]
                continue
            name = (msg.get("params") or {}).get("name")
            if msg.get("method") == "tools/call" and name in BLOCKED:
                with lock:
                    refused[msg.get("id")] = name
                sys.stdout.write(json.dumps({
                    "jsonrpc": "2.0", "id": msg.get("id"),
                    "result": {
                        "isError": True,
                        "content": [{"type": "text", "text": (
                            f"{name} is disabled for this task. Executing JavaScript does not count as "
                            "work done in the application -- navigate and act through the UI tools "
                            "(browser_navigate, browser_click, browser_fill_form, browser_type, ...)."
                        )}],
                    },
                }) + "\n")
                sys.stdout.flush()
                continue
            proc.stdin.write(line)  # type: ignore[union-attr]
            proc.stdin.flush()  # type: ignore[union-attr]
        try:
            proc.stdin.close()  # type: ignore[union-attr]
        except OSError:
            pass

    threading.Thread(target=client_to_server, daemon=True).start()

    for line in proc.stdout:  # type: ignore[union-attr]
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            sys.stdout.write(line)
            sys.stdout.flush()
            continue
        tools = ((msg.get("result") or {}) if isinstance(msg.get("result"), dict) else {}).get("tools")
        if isinstance(tools, list):
            msg["result"]["tools"] = [t for t in tools if t.get("name") not in BLOCKED]
            line = json.dumps(msg) + "\n"
        sys.stdout.write(line)
        sys.stdout.flush()

    return proc.wait()


if __name__ == "__main__":
    raise SystemExit(main())
