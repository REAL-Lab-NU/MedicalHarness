#!/usr/bin/env python
"""Export MedWeb tasks into harness-bench format.

The local EMR, payer and fax portals supply the browser workspace. Hooks reset
portal state before an episode. The oracle reads the final state over CDP and
applies deterministic task checks. A running portal and a dedicated CDP browser
are required. See docs/environment.md for browser setup.
"""
from __future__ import annotations

import argparse
import os
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from medharness.data import registry as R  # noqa: E402

OUT_ROOT = Path(os.environ.get("MEDHARNESS_TASKS_ROOT", str(REPO / "tasks")))
PREFIX = "medadmin"
PORTAL_ORIGIN = os.environ.get("HEALTHADMIN_PORTAL_BASE_URL", "http://127.0.0.1:3002")
CDP_URL = os.environ.get("HB_CDP_URL", "http://127.0.0.1:9222")

HOOKS = '''"""Per-episode setup: clear portal state and open the task's start page.

Mirrors upstream `HealthAdminEnvironment.reset()`, which generates a run id, clears the single
`portals_state` localStorage key and navigates to the task start URL. There is no server-side state,
so this is the whole of the initialisation.
"""
from __future__ import annotations

from typing import Any

import os

# One Chrome per concurrent slot: state is per-browser localStorage, so the CDP endpoint is
# the thing that has to differ between slots. HB_CDP_URL is set by the queue runner.
CDP_URL = os.environ.get("HB_CDP_URL", "{cdp_url}")
START_URL = "{start_url}"


def _portal_page(browser):
    for ctx in browser.contexts:
        for page in ctx.pages:
            if "{origin}" in (page.url or ""):
                return page
    ctx = browser.contexts[0] if browser.contexts else browser.new_context()
    return ctx.pages[0] if ctx.pages else ctx.new_page()


def prepare_runtime(ctx: dict[str, Any]) -> dict[str, Any]:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp(CDP_URL)
        try:
            page = _portal_page(browser)
            page.goto(START_URL, wait_until="networkidle", timeout=60000)
            page.evaluate("() => localStorage.removeItem('portals_state')")
            page.reload(wait_until="networkidle", timeout=60000)
            page.wait_for_timeout(1500)
            return {{"portal_ready": True, "start_url": START_URL}}
        finally:
            browser.close()
'''

ORACLE = '''"""Score one healthadmin episode from the portal state the agent left behind.

`_final_state` is a port of upstream `HealthAdminEnvironment.get_final_state()` (environment.py:925)
and `_build_payer_state` (environment.py:~888): the jmespath checks address `full_state.*`,
`payer_a_state.full_state.*` and friends, so the shape has to match upstream's exactly or every check
silently reads null. `score_task_deterministic` grades the final state using the
frozen process/outcome rules and deterministic checks.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

REPO = Path(os.environ.get("MEDHARNESS_REPO", "{repo}"))

# One Chrome per concurrent slot: state is per-browser localStorage, so the CDP endpoint is
# the thing that has to differ between slots. HB_CDP_URL is set by the queue runner.
CDP_URL = os.environ.get("HB_CDP_URL", "{cdp_url}")
ORIGIN = "{origin}"
TASK_ID = "{task_id}"
UPSTREAM_ID = "{upstream_id}"


def _read_portal_state() -> dict[str, Any]:
    from playwright.sync_api import sync_playwright

    empty = {{"emr": {{}}, "payerA": {{}}, "payerB": {{}}, "fax": {{}}}}
    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp(CDP_URL)
        try:
            page = None
            for c in browser.contexts:
                for pg in c.pages:
                    if ORIGIN in (pg.url or ""):
                        page = pg
                        break
                if page:
                    break
            if page is None:
                return empty
            raw = page.evaluate("() => localStorage.getItem('portals_state')")
        finally:
            browser.close()
    try:
        cur = json.loads(raw) if raw else {{}}
    except ValueError:
        cur = {{}}
    if not isinstance(cur, dict):
        cur = {{}}
    return {{k: (cur.get(k) if isinstance(cur.get(k), dict) else {{}})
            for k in ("emr", "payerA", "payerB", "fax")}}


def _build_payer_state(state: dict[str, Any]) -> dict[str, Any]:
    """Port of upstream _build_payer_state."""
    submissions = state.get("submissions")
    submissions = submissions if isinstance(submissions, list) else []
    auth_searches = state.get("authSearches")
    auth_searches = auth_searches if isinstance(auth_searches, list) else []
    eligibility_checks = state.get("eligibilityChecks")
    eligibility_checks = eligibility_checks if isinstance(eligibility_checks, list) else []
    appeal_actions = state.get("appealActions")
    if not isinstance(appeal_actions, dict):
        fallback = state.get("agentActions")
        appeal_actions = fallback if isinstance(fallback, dict) else {{}}
    added = {{"priorAuth": submissions[-1]}} if submissions else {{}}
    return {{
        "config": state.get("initialState", {{}}),
        "full_state": {{"appealActions": appeal_actions, "agentActions": appeal_actions}},
        "agentActions": appeal_actions,
        "initialfinaldiff": {{"added": added, "updated": {{}}, "removed": {{}}}},
        "differences": {{
            "priorAuth": {{"added": submissions}},
            "authSearches": auth_searches,
            "eligibilityChecks": eligibility_checks,
        }},
    }}


def _final_state(portal_state: dict[str, Any]) -> dict[str, Any]:
    """Port of upstream get_final_state, minus the fields only a stepping harness can fill."""
    full_state = portal_state.get("emr") or {{}}
    fax_state = portal_state.get("fax") or {{}}
    if fax_state:
        full_state["faxPortal"] = fax_state
    actions = full_state.get("agentActions")
    actions = actions if isinstance(actions, dict) else {{}}
    visited = actions.get("visitedPages")
    viewed = actions.get("viewedDocuments")
    payer_a = _build_payer_state(portal_state.get("payerA") or {{}})
    payer_b = _build_payer_state(portal_state.get("payerB") or {{}})
    return {{
        "success": True,
        "task_id": UPSTREAM_ID,
        "environment": "emr",
        "episode_completed": True,
        "actions": {{
            "history": [],
            "visited_pages": visited if isinstance(visited, list) else [],
            "viewed_documents": viewed if isinstance(viewed, list) else [],
        }},
        "actions_history": [],
        "full_state": full_state,
        "payer_a_state": payer_a,
        "payer_b_state": payer_b,
        "aetna_state": payer_a,
        "anthem_state": payer_b,
    }}


def score_workspace(workspace: Path) -> dict[str, Any]:
    sys.path.insert(0, str(REPO))
    from medharness.data import registry as R
    from medharness.scorers.healthadmin import score_task_deterministic

    gold = R.load_gold("healthadmin")[UPSTREAM_ID]["gold"]
    try:
        portal_state = _read_portal_state()
    except Exception as exc:  # noqa: BLE001 - a dead browser must not look like a wrong answer
        return _fail(f"could not read portal state: {{type(exc).__name__}}: {{exc}}")

    state = _final_state(portal_state)
    score = score_task_deterministic(gold, state)
    outcome = score["outcome"]
    process = score["process"]
    return {{
        "task": TASK_ID,
        "workspace": str(workspace),
        "checks": [
            {{"id": "outcome", "label": "every outcome check passes",
             "pass": bool(outcome["strict_pass"]), "weight": 1.0,
             "detail": f"{{outcome['passed_check_count']}}/{{outcome['check_count']}}"}},
            {{"id": "process", "label": "process checks",
             "pass": bool(process["strict_pass"]), "weight": 0.0,
             "detail": f"{{process['passed_check_count']}}/{{process['check_count']}}"}},
        ],
        "outcome_score": 1.0 if outcome["strict_pass"] else 0.0,
        "metrics": {{
            "strict_outcome_pass": int(bool(outcome["strict_pass"])),
            "outcome_check_pass_rate": outcome["check_pass_rate"],
            "process_check_pass_rate": process["check_pass_rate"],
            "all_deterministic_pass_rate": score["all_deterministic"]["check_pass_rate"],
            "outcome_checks": outcome["check_count"],
            "process_checks": process["check_count"],
            "visited_pages": len(state["actions"]["visited_pages"]),
            "submission_failure": None,
        }},
        "outcome_llm_weight": 0.0,
    }}


def _fail(detail: str) -> dict[str, Any]:
    return {{
        "task": TASK_ID,
        "workspace": "",
        "checks": [{{"id": "outcome", "label": "every outcome check passes",
                    "pass": False, "weight": 1.0, "detail": detail}}],
        "outcome_score": 0.0,
        "metrics": {{"strict_outcome_pass": 0, "submission_failure": detail}},
        "outcome_llm_weight": 0.0,
    }}
'''

PROMPT = """You are working inside a hospital's web applications. Everything you need is in the browser;
there are no files to read and no data on disk.

## The task

{instruction}

## The environment

Use the browser tool. Three applications are served from `{origin}`:

- the EMR, at `{origin}/emr` -- worklist, denials, patient records, documents, triage notes;
- the Payer A provider portal, at `{origin}/payer-a` -- eligibility, prior authorisations, appeals;
- the fax portal, at `{origin}/fax-portal`.

Start at `{start_url}`. Navigate with the browser as a person would: click links and buttons, fill and
submit forms, open documents. Do not fetch these pages with shell commands and do not execute
JavaScript -- work done outside the browser UI does not count, because the applications record what
you actually did in them.

Take as many steps as the work needs; there is no turn budget. You have no internet access; the only
reachable host is `{origin}`.

## What counts as done

Your work is judged on the state you leave behind in the applications: which records you opened, which
documents you viewed, what you submitted, and what you wrote. Finish the actions the task asks for
rather than describing what you would do -- a summary in your reply scores nothing on its own.
"""


def main() -> int:
    global OUT_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task_ids', nargs='*', help='Optional exported/upstream task IDs')
    parser.add_argument('--output', type=Path, default=OUT_ROOT)
    parser.add_argument('--overwrite', action='store_true', help='Replace only selected existing task directories')
    args = parser.parse_args()
    OUT_ROOT = args.output.resolve()
    only = args.task_ids or None
    samples = R.load_records("healthadmin", check=False, subset=True)
    gold = R.load_gold("healthadmin")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    written = []

    for i, sample in enumerate(samples, 1):
        upstream = str(sample.id)
        task_id = f"{PREFIX}-{i:03d}"
        if only and upstream not in only and task_id not in only:
            continue
        env_private = gold[upstream].get("env_private") or {}
        origin = PORTAL_ORIGIN
        start_path = env_private.get("start_path") or "/emr/worklist"
        start_url = f"{origin}{start_path}"

        if only and str(sample.id) not in only and task_id not in only:
            continue
        task_dir = OUT_ROOT / task_id
        if task_dir.exists():
            if not args.overwrite:
                raise FileExistsError(f"Refusing to overwrite {task_dir}; use --overwrite")
            shutil.rmtree(task_dir)
        task_dir.mkdir(parents=True)
        (task_dir / "fixtures").mkdir()

        (task_dir / "prompt.txt").write_text(
            PROMPT.format(instruction=sample.input.strip(), origin=origin, start_url=start_url),
            encoding="utf-8")
        (task_dir / "hooks.py").write_text(
            HOOKS.format(cdp_url=CDP_URL, start_url=start_url, origin=origin), encoding="utf-8")
        (task_dir / "oracle_grade.py").write_text(
            ORACLE.format(repo=str(REPO), cdp_url=CDP_URL, origin=origin,
                          task_id=task_id, upstream_id=upstream), encoding="utf-8")
        (task_dir / "task.yaml").write_text(json.dumps({
            "task_id": task_id,
            "title": f"HealthAdminBench {upstream}",
            "class": "Browser work across EMR and payer portals, graded from portal state",
            "prompt_file": "prompt.txt",
            "fixtures_dir": "fixtures",
            "oracle_module": "oracle_grade.py",
            "hooks_module": "hooks.py",
            "timeout_sec": 3600,
            "tags": ["medical", "healthadmin", "browser", "medharness"],
        }, indent=2), encoding="utf-8")
        written.append((task_id, upstream, start_path))
        print(f"{task_id} ({upstream}) start={start_path}")

    print(f"\nexported {len(written)} tasks to {OUT_ROOT}")
    if written:
        print(f"  ids: {written[0][0]} .. {written[-1][0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
