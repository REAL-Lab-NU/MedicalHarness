#!/usr/bin/env python3
"""MH-Lab: a fixed LangGraph ReAct loop with switchable components.

The harness-bench runner supplies the workspace, isolation, usage proxy and
scoring. This process registers a proxy route and executes the graph until the
model returns a final answer or reaches an execution limit. Task outputs stay
in the workspace. Gold references remain outside the process, and all model
requests use the proxy.

Components:
  P  Planning exposes update_plan, planning instructions, a first-turn reminder
     and the current plan on each request. Disabling P removes all four.
  C  Context policy selects none, elide, elide_recall, summarize or
     elide_then_summarize. Elision replaces large middle-history observations
     with stubs. Recall exposes their stored originals through recall_event.
     Summarization replaces the oldest complete turn blocks with a running
     summary from the same model. Single policies use --summary-trigger.
     The combined policy uses --elide-trigger followed by --summary-trigger.
     The none policy retains full history until the input budget is exceeded.
  A  Action space selects file tools plus bash, or bash alone. Enabled planning
     and recall tools remain available in both modes.
  B  Tool bridge replaces direct task-tool exposure with tool_search,
     tool_describe and tool_call. Planning and recall tools remain visible.
     The underlying task tools and executors stay the same.
  V  Verification checks that output files named in the task prompt exist and
     are non-empty. Missing files trigger one reminder per episode before the
     model continues.

Defaults are planning on, context summarization, file tools, tool bridge off
and verification off. Proxy request bodies record the model-visible context.
The harness writes events.jsonl, context-transforms.jsonl and langgraph-manifest.json.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import AsyncExitStack
import fnmatch
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
from typing import Any, TypedDict
from urllib.parse import urlsplit

# ----------------------------------------------------------------------------------------------
# Fixed execution constants recorded in the manifest
# ----------------------------------------------------------------------------------------------
TOOL_RESULT_MAX_CHARS = 24_000        # Tool results are truncated to 24k characters.
READ_DEFAULT_LIMIT = 2000
READ_MAX_BYTES = 4_000_000
STUCK_REMIND_AT = 5                    # identical calls (any status) or identical failing calls
STUCK_STOP_AT = 8                      # identical failing calls
STUCK_REPEAT_STOP_AT = 15              # identical calls of any status after the reminder was ignored
MAX_STEPS = 500                        # Graph recursion limit. The wall-clock deadline sets the episode budget.
BASH_DEFAULT_TIMEOUT = 120
BASH_MAX_TIMEOUT = 600
DENIED_COMMANDS = ('rm -rf', 'sudo ', 'git push', 'git reset --hard', 'chmod -R', 'chown -R', 'mkfs', 'shutdown', 'reboot')
IGNORE_PATTERNS = ('.git/**', '__pycache__/**', '.venv/**', 'node_modules/**')
SUBMISSION_PATH_RE = re.compile(r"(?:/[\w./\-]+)?/((?:out|submission)/[\w.\-]+(?:/[\w.\-]+)*)")
VERIFICATION_REMINDER = """<system-reminder>
Verification: the task requires the following file(s), which do not exist or are empty:
{FILES}
Write your final answer to each of them now, in exactly the format the task specified, then finish. Do not start the task over.
</system-reminder>"""
PRIVATE_LOG_NAMES = ('events.jsonl', 'usage-proxy', 'langgraph-manifest', 'context-transforms', 'langgraph-home', 'elided/')
COUNT_MARGIN = 0.05                    # counting reserve subtracted from the input budget
ELIDE_MIN_CHARS = 1500                 # a tool observation this long in the middle region is 'bulky'
DEFAULT_CHARS_PER_TOKEN = 3.5          # until the first real prompt_tokens calibrates the ratio

BASE_SYSTEM_PROMPT = """You are an autonomous agent completing one task inside a local workspace.
Your job is to complete the task with high reliability and deliver exactly what the task asks for.

Core workflow:
1. Read the task carefully; note the required output files and their exact format.
2. Explore the workspace before acting (list_files, glob_files, grep_text, read_file).
3. Before modifying an existing file you MUST read it first with read_file.
4. Prefer edit_file (targeted old_text/new_text replacement) over write_file (overwrite) for existing files.
5. Use bash for computation, scripts, databases and verification.
6. Write the required output files before you finish; the final reply alone is not a submission.

Rules:
- Never fabricate file contents, data or results. Only claim a result you actually produced.
- Never operate outside the workspace; paths are relative to the workspace root (absolute workspace paths are accepted).
- Destructive shell commands are denied by the permission layer. Don't expect approval.
- Read-only tool calls can be issued in parallel; the harness will run them in order.
- After each tool call, read the result. If it errored, FIX THE INPUT rather than retrying the same call."""

BROWSER_SYSTEM_PROMPT = """You are an autonomous agent completing one task through a web browser.
Your job is to complete the task with high reliability using the browser tools provided.

Core workflow:
1. Read the task carefully; note every action and record the task asks for.
2. Take a snapshot before acting on a page; act on element references from the latest snapshot.
3. Verify the state of the page after every consequential action.

Rules:
- Never fabricate page contents or results. Only claim an action you actually performed.
- After each tool call, read the result. If it errored, FIX THE INPUT rather than retrying the same call."""

BASH_ONLY_SYSTEM_PROMPT = """You are an autonomous agent completing one task inside a local workspace.
Your job is to complete the task with high reliability and deliver exactly what the task asks for.

Core workflow:
1. Read the task carefully; note the required output files and their exact format.
2. Explore the workspace before acting.
3. Before modifying an existing file you MUST read it first.
4. Prefer surgical edits over rewriting a whole file.
5. Use the shell for computation, scripts, databases and verification.
6. Write the required output files before you finish; the final reply alone is not a submission.

Rules:
- Never fabricate file contents, data or results. Only claim a result you actually produced.
- Never operate outside the workspace; paths are relative to the workspace root (absolute workspace paths are accepted).
- Destructive shell commands are denied by the permission layer. Don't expect approval.
- After each tool call, read the result. If it errored, FIX THE INPUT rather than retrying the same call.

# Working with only a shell
Every interaction with the workspace goes through `bash` — you write the commands yourself (cat, sed, grep, python, sqlite3, ...)."""

ELIDED_STUB_RECALL = "[tool output elided: {N_LINES} lines / {N_CHARS} chars. Use recall_event({EVENT_ID}) for the full output, or re-read/re-run.]"
ELIDED_STUB = "[tool output elided: {N_LINES} lines / {N_CHARS} chars. Re-read or re-run to get it again.]"

SUMMARY_WRAPPER_RECALL = """<context-summary>
{SUMMARY}
(Older events up to event {EVENT_ID} were summarized. Tool outputs that were elided before the summary remain recallable via recall_event(id) with the id shown in their stub; everything else in the summarized region is gone from your context; re-read files or re-run commands if you need it.)
</context-summary>"""

PLANNING_SYSTEM_BLOCK = """# Task planning
You have an `update_plan` tool that holds a todo list. The harness shows the current plan back to you
before every turn, so use it to stay on track.
- For any non-trivial task (about 3+ steps), your FIRST action MUST be to call `update_plan` to break
  the task into concrete steps.
- Keep it current: mark exactly ONE task `in_progress` while you work on it, and mark a task
  `completed` the moment it is actually done (don't batch completions).
- Add new tasks as you discover them; re-send the COMPLETE list each time.
- Skip planning only for a single trivial step or a purely informational request."""

PLANNING_FIRST_TURN_REMINDER = """<system-reminder>
You have not created a plan yet. Before taking any other action, call update_plan to break this task into
concrete steps. (Skip planning only for a single trivial step or a purely informational request.)
</system-reminder>"""

PLANNING_TURN_REMINDER = """<system-reminder>
Current plan (update it via update_plan as you progress; keep exactly one task in_progress, and mark a
task completed the moment it is actually done):
{PLAN}
</system-reminder>"""

SUMMARY_PROMPT = """You are compacting an agent's conversation to save context. You are given the OLDEST part of the
transcript (and possibly a previous summary to update).
Respond with TEXT ONLY — do NOT call any tools. Produce a concise structured summary under these
exact headings (omit a heading only if truly empty):
## Goal
The task / intent (preserve it precisely, including the required output files and format).
## Evidence gathered
Facts found so far, each with its source (file path, record, page), keeping exact values, units, dates, names and identifiers.
## Done
What has been accomplished, with concrete results.
## Pending
What still needs doing.
## Errors & fixes
Errors hit and how they were resolved (or are still open).
## Current state
Anything needed to continue (paths, intermediate results, files written).
## Next step
The immediate next action, in line with the most recent work.
Be specific (keep exact paths, identifiers, numbers, error messages). If a previous summary is given,
UPDATE it: keep still-true facts, drop stale ones, merge in the new events. Summarize — don't transcribe."""

SUMMARY_WRAPPER = """<context-summary>
{SUMMARY}
(Older events were summarized. The raw content is no longer in your context; re-read files or re-run commands if you need details that are missing.)
</context-summary>"""

STUCK_FAILING_REMINDER = """<system-reminder>
You have called `{TOOL}` with the SAME arguments {N} times and it keeps failing the same way.
Repeating it will NOT work. STOP — read the actual error, then try a DIFFERENT command, inspect
more context, or step back and reconsider your plan. Do NOT issue the same call again.
</system-reminder>"""

STUCK_REPEAT_REMINDER = """<system-reminder>
You have called `{TOOL}` with the SAME arguments {N} times. You're not making progress — you
already have this result. Move on to the next concrete step instead of repeating it.
</system-reminder>"""

# Allowlist of Playwright MCP actions in browser mode.
BROWSER_TOOLS = frozenset({
    'browser_navigate', 'browser_navigate_back', 'browser_snapshot', 'browser_click',
    'browser_type', 'browser_fill_form', 'browser_select_option', 'browser_press_key',
    'browser_hover', 'browser_drag', 'browser_wait_for', 'browser_tabs',
    'browser_handle_dialog', 'browser_resize',
})   # Keep the externally managed CDP browser open across tool calls.


# ----------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--prompt-file', type=Path, required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--upstream', required=True)
    p.add_argument('--max-tokens', type=int, default=16384)
    p.add_argument('--context-window', type=int, default=262144,
                   help='Research window W: the total prompt+output capacity this harness enforces')
    p.add_argument('--timeout-seconds', type=int, default=1800, help='Shared wall-clock deadline for the whole episode')
    p.add_argument('--tool-mode', choices=('files', 'browser'), default='files')
    p.add_argument('--mcp-config', type=Path)
    p.add_argument('--planning', choices=('on', 'off'), default='on')
    p.add_argument('--context-policy', choices=('none', 'elide', 'elide_recall', 'summarize', 'elide_then_summarize'), default='summarize',
                   help='T0 none | T1 elide | T2 elide_recall | T3 summarize | T4 elide_then_summarize')
    p.add_argument('--elide-trigger', type=float, default=0.60, help='T4 only: fraction of the input budget that triggers elision (soft threshold)')
    p.add_argument('--action-space', choices=('tools', 'bash'), default='tools', help='predefined workspace tools, or bash only')
    p.add_argument('--tool-bridge', choices=('on', 'off'), default='off',
                   help='B: hide the workspace/browser tools behind tool_search / tool_describe / tool_call (Hermes-style lazy discovery)')
    p.add_argument('--verification', choices=('on', 'off'), default='off',
                   help='V: after the final reply, check that the files the task names exist and are non-empty; if not, remind once and continue')
    p.add_argument('--summary-trigger', type=float, default=0.80, help='Fraction of the input budget that triggers summarization')
    p.add_argument('--recent-fraction', type=float, default=0.30, help='Fraction of the input budget kept verbatim as the recent window')
    p.add_argument('--seed', type=int, default=None, help='Sampling seed sent explicitly; the proxy pins one when absent')
    p.add_argument('--condition-id', default=None, help='Free label recorded in the manifest')
    args = p.parse_args(argv)
    if not 0 < args.max_tokens < args.context_window or args.timeout_seconds <= 0:
        p.error('require 0 < max-tokens < context-window and positive timeout-seconds')
    if not 0 < args.recent_fraction < args.summary_trigger < 1:
        p.error('require 0 < recent-fraction < summary-trigger < 1')
    if not args.recent_fraction < args.elide_trigger <= args.summary_trigger:
        p.error('require recent-fraction < elide-trigger <= summary-trigger')
    if args.tool_mode == 'browser' and args.action_space == 'bash':
        p.error('bash-only action space applies to files mode')
    if args.tool_mode == 'browser' and not args.mcp_config:
        p.error('browser mode requires --mcp-config')
    if args.tool_mode == 'files' and args.mcp_config:
        p.error('--mcp-config is only used in browser mode')
    for key in ('workspace', 'prompt_file', 'mcp_config'):
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, value.resolve(strict=True))
    return args


def register_proxy(upstream: str) -> str:
    base = os.environ.get('HARNESSBENCH_LLM_PROXY_URL', '').rstrip('/')
    routes_path = os.environ.get('HARNESSBENCH_LLM_PROXY_ROUTES')
    if not base or not routes_path:
        raise ValueError('HARNESSBENCH_LLM_PROXY_URL and HARNESSBENCH_LLM_PROXY_ROUTES are required; no direct upstream')
    for url in (base, upstream):
        parsed = urlsplit(url)
        if parsed.scheme not in ('http', 'https') or not parsed.netloc or parsed.username or parsed.query or parsed.fragment:
            raise ValueError('proxy and upstream must be HTTP base URLs without credentials/query/fragment')
    path = Path(routes_path)
    routes = json.loads(path.read_text()) if path.exists() else {}
    if not isinstance(routes, dict):
        raise ValueError('usage proxy routes must be an object')
    routes['/langgraph/vllm'] = {'framework': 'langgraph', 'provider': 'vllm', 'upstream': upstream.rstrip('/')}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(routes, indent=2) + '\n')
    return base + '/langgraph/vllm'


def serializable(value):
    if hasattr(value, 'model_dump'):
        return value.model_dump(mode='json')
    if isinstance(value, Path):
        return str(value)
    return str(value)


class EventLog:
    """Append-only episode log. A model request is logged before it is sent."""

    def __init__(self, sandbox: Path):
        self.path = sandbox / 'events.jsonl'
        self.transforms = sandbox / 'context-transforms.jsonl'
        self.lock = threading.Lock()
        self.started = time.monotonic()
        self.n = 0

    def emit(self, event: str, **data) -> int:
        with self.lock:
            self.n += 1
            row = {'event_id': self.n, 'event': event, 'wall_time': time.time(),
                   'elapsed_seconds': round(time.monotonic() - self.started, 3), **data}
            with self.path.open('a') as f:
                f.write(json.dumps(row, default=serializable, ensure_ascii=False) + '\n')
            return self.n

    def transform(self, **data):
        with self.lock, self.transforms.open('a') as f:
            f.write(json.dumps({'wall_time': time.time(), **data}, default=serializable, ensure_ascii=False) + '\n')


# ----------------------------------------------------------------------------------------------
# Workspace tools (files mode)
# ----------------------------------------------------------------------------------------------
class ToolError(Exception):
    """A recoverable tool error, returned to the model as the observation."""


class Workspace:
    def __init__(self, root: Path, deadline: 'Deadline'):
        self.root = root
        self.deadline = deadline
        self.full_reads: dict[str, str] = {}   # path -> content hash recorded by a full read or a write

    def resolve(self, path: str, must_exist=False) -> Path:
        if not path or not isinstance(path, str):
            raise ToolError('path must be a non-empty string')
        raw = Path(path)
        candidate = raw if raw.is_absolute() else self.root / raw
        try:
            resolved = candidate.resolve()
        except OSError as exc:
            raise ToolError(f'cannot resolve path: {exc}')
        if resolved != self.root and self.root not in resolved.parents:
            raise ToolError(f'path is outside the workspace: {path}')
        if must_exist and not resolved.exists():
            raise ToolError(f'path does not exist: {path}')
        return resolved

    @staticmethod
    def ignored(rel: str) -> bool:
        return any(fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(rel + '/', pat) for pat in IGNORE_PATTERNS)

    @staticmethod
    def digest(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    # -- read --------------------------------------------------------------------------------
    def read_file(self, path: str, offset: int = 1, limit: int = READ_DEFAULT_LIMIT) -> str:
        target = self.resolve(path, must_exist=True)
        if target.is_dir():
            raise ToolError(f'{path} is a directory; use list_files')
        size = target.stat().st_size
        if size > READ_MAX_BYTES:
            raise ToolError(f'file is {size} bytes, above the read cap; use bash with head/sed instead')
        data = target.read_bytes()
        if b'\x00' in data[:8192]:
            raise ToolError('binary file (NUL byte in the first 8KB)')
        text = data.decode('utf-8', errors='replace')
        lines = text.splitlines()
        offset = max(int(offset or 1), 1)
        limit = max(int(limit or READ_DEFAULT_LIMIT), 1)
        if offset > max(len(lines), 1):
            raise ToolError(f'offset {offset} is past the end of the file ({len(lines)} lines)')
        chunk = lines[offset - 1: offset - 1 + limit]
        full = offset == 1 and offset - 1 + limit >= len(lines)
        if full:
            self.full_reads[str(target)] = self.digest(data)
        header = f'[{target.relative_to(self.root)}: lines {offset}-{offset + len(chunk) - 1} of {len(lines)}; full_read={full}]'
        body = '\n'.join(f'{offset + i:6d}\t{line}' for i, line in enumerate(chunk))
        return header + '\n' + body

    def list_files(self, path: str = '.', recursive: bool = False, max_entries: int = 200) -> str:
        target = self.resolve(path, must_exist=True)
        if not target.is_dir():
            raise ToolError(f'{path} is not a directory')
        max_entries = min(max(int(max_entries or 200), 1), 2000)
        out = []
        for base, dirs, files in os.walk(target):
            basep = Path(base)
            rel_base = basep.relative_to(self.root)
            dirs[:] = sorted(d for d in dirs if not self.ignored(str(rel_base / d)))
            depth = len(basep.relative_to(target).parts)
            indent = '  ' * depth
            for d in dirs:
                out.append(f'{indent}{d}/')
            for f in sorted(files):
                rel = str(rel_base / f)
                if self.ignored(rel):
                    continue
                try:
                    out.append(f'{indent}{f}  ({(basep / f).stat().st_size} bytes)')
                except OSError:
                    out.append(f'{indent}{f}')
            if len(out) >= max_entries:
                out = out[:max_entries] + [f'... truncated at {max_entries} entries']
                break
            if not recursive:
                break
        return '\n'.join(out) or '(empty directory)'

    def glob_files(self, pattern: str, path: str = '.', max_matches: int = 200) -> str:
        target = self.resolve(path, must_exist=True)
        max_matches = min(max(int(max_matches or 200), 1), 1000)
        hits = []
        for candidate in sorted(target.glob(pattern)):
            if candidate.is_file():
                rel = str(candidate.relative_to(self.root))
                if not self.ignored(rel):
                    hits.append(rel)
            if len(hits) >= max_matches:
                break
        return '\n'.join(hits) or '(no matches)'

    def grep_text(self, query: str, path: str = '.', include: str | None = None,
                  case_sensitive: bool = True, max_matches: int = 100) -> str:
        target = self.resolve(path, must_exist=True)
        max_matches = min(max(int(max_matches or 100), 1), 1000)
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            rx = re.compile(query, flags)
        except re.error as exc:
            raise ToolError(f'invalid regex: {exc}')
        hits = []
        files = [target] if target.is_file() else sorted(p for p in target.rglob('*') if p.is_file())
        for f in files:
            rel = str(f.relative_to(self.root))
            if self.ignored(rel) or (include and not fnmatch.fnmatch(f.name, include)):
                continue
            try:
                if f.stat().st_size > 10_000_000:
                    continue
                data = f.read_bytes()
            except OSError:
                continue
            if b'\x00' in data[:8192]:
                continue
            for i, line in enumerate(data.decode('utf-8', errors='replace').splitlines(), 1):
                if rx.search(line):
                    hits.append(f'{rel}:{i}:{line}')
                    if len(hits) >= max_matches:
                        return '\n'.join(hits) + f'\n... truncated at {max_matches} matches'
        return '\n'.join(hits) or '(no matches)'

    # -- write -------------------------------------------------------------------------------
    def _gate(self, target: Path, path: str):
        key = str(target)
        if key not in self.full_reads:
            raise ToolError(f'{path} exists but was never fully read in this session; call read_file first')
        if self.digest(target.read_bytes()) != self.full_reads[key]:
            raise ToolError(f'{path} was modified since it was read; call read_file again')

    def write_file(self, path: str, content: str, overwrite: bool = False) -> str:
        target = self.resolve(path)
        if target.is_dir():
            raise ToolError(f'{path} is a directory')
        if not isinstance(content, str):
            raise ToolError('content must be a string')
        if target.exists():
            if not overwrite:
                raise ToolError(f'{path} exists; use edit_file or pass overwrite=true')
            self._gate(target, path)
            before = target.read_text(encoding='utf-8', errors='replace')
        else:
            before = ''
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding='utf-8')
        self.full_reads[str(target)] = self.digest(content.encode('utf-8'))
        return unified_diff(before, content, str(target.relative_to(self.root)))

    def edit_file(self, path: str, old_text: str, new_text: str, replace_all: bool = False) -> str:
        target = self.resolve(path)
        if not target.exists():
            raise ToolError(f'{path} does not exist; use write_file')
        if target.is_dir():
            raise ToolError(f'{path} is a directory')
        if not isinstance(old_text, str) or not isinstance(new_text, str) or not old_text:
            raise ToolError('old_text and new_text must be strings and old_text must be non-empty')
        if old_text == new_text:
            raise ToolError('old_text equals new_text (no-op)')
        self._gate(target, path)
        before = target.read_text(encoding='utf-8', errors='replace')
        count = before.count(old_text)
        if count == 0:
            raise ToolError('old_text not found in the file')
        if count > 1 and not replace_all:
            raise ToolError(f'old_text matches {count} times; add context to make it unique or pass replace_all=true')
        after = before.replace(old_text, new_text) if replace_all else before.replace(old_text, new_text, 1)
        target.write_text(after, encoding='utf-8')
        self.full_reads[str(target)] = self.digest(after.encode('utf-8'))
        return unified_diff(before, after, str(target.relative_to(self.root)))

    # -- shell -------------------------------------------------------------------------------
    def bash(self, command: str, timeout_seconds: int = BASH_DEFAULT_TIMEOUT, cwd: str | None = None) -> tuple[str, bool]:
        if not isinstance(command, str) or not command.strip():
            raise ToolError('command must be a non-empty string')
        lowered = ' ' + command.strip() + ' '
        for denied in DENIED_COMMANDS:
            if denied in lowered:
                raise ToolError(f'command denied by the permission layer: contains `{denied.strip()}`')
        workdir = self.resolve(cwd or '.', must_exist=True)
        if not workdir.is_dir():
            raise ToolError('cwd is not a directory')
        for token in PRIVATE_LOG_NAMES:
            if token in command:
                raise ToolError(f'command denied: `{token}` is the episode\'s own record, not task material')
        timeout = min(max(int(timeout_seconds or BASH_DEFAULT_TIMEOUT), 1), BASH_MAX_TIMEOUT)
        timeout = max(1, min(timeout, int(self.deadline.remaining())))
        # A separate process session lets a timeout terminate the shell and its child processes.
        proc = subprocess.Popen(['/bin/sh', '-c', command], cwd=workdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, errors='replace', start_new_session=True,
                                env={**os.environ, 'HOME': os.environ.get('HOME', str(self.root))})
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = proc.communicate()
            return f'timed_out=True after {timeout}s\n--- stdout ---\n{stdout or ""}', True
        out = f'exit_code={proc.returncode}\n'
        if stdout:
            out += '--- stdout ---\n' + stdout
        if stderr:
            out += ('\n' if stdout else '') + '--- stderr ---\n' + stderr
        return out, proc.returncode != 0


def unified_diff(before: str, after: str, name: str) -> str:
    import difflib
    lines = list(difflib.unified_diff(before.splitlines(), after.splitlines(), fromfile=f'a/{name}', tofile=f'b/{name}', lineterm=''))
    return '\n'.join(lines) if lines else f'(no change to {name})'


class Deadline:
    def __init__(self, seconds: float):
        self.end = time.monotonic() + seconds

    def remaining(self) -> float:
        return max(0.0, self.end - time.monotonic())

    def expired(self) -> bool:
        return self.remaining() <= 0


# ----------------------------------------------------------------------------------------------
# Tool schemas
# ----------------------------------------------------------------------------------------------
def files_tool_schemas() -> list[dict]:
    def t(name, desc, props, required):
        return {'type': 'function', 'function': {'name': name, 'description': desc,
                'parameters': {'type': 'object', 'properties': props, 'required': required}}}
    return [
        t('read_file', 'Read a text file from the workspace, returned with 1-based line numbers and a header. Call this BEFORE edit_file or write_file(overwrite=true) on any existing file; the harness enforces read-before-write and only a full read (offset=1, limit covering the whole file) satisfies it. Default limit is 2000 lines. Errors: missing path, path outside the workspace, directory, binary file, file above the read cap (use bash with head/sed), offset past EOF.',
          {'path': {'type': 'string'}, 'offset': {'type': 'integer', 'description': '1-based first line (default 1)'}, 'limit': {'type': 'integer', 'description': 'lines to return (default 2000)'}}, ['path']),
        t('write_file', 'Create a new file or overwrite an existing one; returns a unified diff. New files need no overwrite flag and parent directories are created. Overwriting an existing file requires overwrite=true AND a prior full read_file of that path in this session; the harness rejects writes to unread or externally modified files.',
          {'path': {'type': 'string'}, 'content': {'type': 'string'}, 'overwrite': {'type': 'boolean'}}, ['path', 'content']),
        t('edit_file', 'Replace an exact string in an existing file (byte-exact match including whitespace); returns a unified diff. old_text must occur at least once; if it occurs more than once, add context or pass replace_all=true. The file must have been fully read in this session first.',
          {'path': {'type': 'string'}, 'old_text': {'type': 'string'}, 'new_text': {'type': 'string'}, 'replace_all': {'type': 'boolean'}}, ['path', 'old_text', 'new_text']),
        t('list_files', 'List files and directories in the workspace as an indented tree with byte sizes (default 200 entries, max 2000). .git, __pycache__, .venv and node_modules are hidden.',
          {'path': {'type': 'string', 'description': 'directory (default workspace root)'}, 'recursive': {'type': 'boolean'}, 'max_entries': {'type': 'integer'}}, []),
        t('glob_files', 'Find files matching a glob pattern (** recursive, * within a segment) under a directory; returns workspace-relative paths (default 200, max 1000). Empty result is not an error.',
          {'pattern': {'type': 'string'}, 'path': {'type': 'string'}, 'max_matches': {'type': 'integer'}}, ['pattern']),
        t('grep_text', 'Search file contents by regular expression; returns path:line:text (default 100 matches). Use include to restrict to a filename glob such as *.csv. Binary files and files over 10MB are skipped.',
          {'query': {'type': 'string'}, 'path': {'type': 'string'}, 'include': {'type': 'string'}, 'case_sensitive': {'type': 'boolean'}, 'max_matches': {'type': 'integer'}}, ['query']),
        t('bash', 'Execute a shell command via /bin/sh -c in the workspace and return exit code, stdout and stderr; a non-zero exit code is reported as a tool error. cwd defaults to the workspace root and must stay inside it. timeout_seconds defaults to 120 (max 600); on expiry the process is killed and partial output returned. Destructive commands (rm -rf, sudo, git push, git reset --hard, chmod -R, chown -R) are denied. Files changed through bash are not tracked: read_file them again before edit_file.',
          {'command': {'type': 'string'}, 'timeout_seconds': {'type': 'integer'}, 'cwd': {'type': 'string'}}, ['command']),
    ]


def planning_tool_schema() -> dict:
    return {'type': 'function', 'function': {
        'name': 'update_plan',
        'description': 'Create or update your task plan (a todo list). The harness shows this plan back to you before every turn so you stay on track and don\'t lose steps. For any non-trivial task (about 3+ steps), call this FIRST to lay out the steps, then keep it updated as you work. Pass the COMPLETE updated list every time — it fully replaces the previous list. Each item has `content` (imperative), `status` (pending | in_progress | completed) and `activeForm` (present-continuous). Keep EXACTLY ONE task in_progress at a time; mark a task completed the moment it is actually done. Updates the in-memory plan only; it touches no files.',
        'parameters': {'type': 'object', 'properties': {'plan': {'type': 'array', 'items': {'type': 'object', 'properties': {
            'content': {'type': 'string'}, 'status': {'type': 'string', 'enum': ['pending', 'in_progress', 'completed']},
            'activeForm': {'type': 'string'}}, 'required': ['content', 'status']}}}, 'required': ['plan']}}}


BRIDGE_SYSTEM_BLOCK = """# Tool discovery
Most tools are not listed directly. Use `tool_search` with a few keywords to find tools (it returns names and one-line descriptions), `tool_describe` to read a tool's full argument schema, and `tool_call` with the tool name and a JSON object of arguments to run it. Search before calling; a tool you have not discovered cannot be called."""


def bridge_tool_schemas() -> list[dict]:
    def t(name, desc, props, required):
        return {'type': 'function', 'function': {'name': name, 'description': desc,
                'parameters': {'type': 'object', 'properties': props, 'required': required}}}
    return [
        t('tool_search', 'Find available tools by keyword. Returns up to `limit` matching tool names with a one-line description each (default 5). Matches against tool names and descriptions.',
          {'query': {'type': 'string'}, 'limit': {'type': 'integer'}}, ['query']),
        t('tool_describe', 'Return the full description and JSON argument schema of one tool found with tool_search.',
          {'name': {'type': 'string'}}, ['name']),
        t('tool_call', 'Invoke a tool by name with its arguments as a JSON object, exactly as tool_describe specified them. Returns the tool result.',
          {'name': {'type': 'string'}, 'arguments': {'type': 'object'}}, ['name', 'arguments']),
    ]


def recall_tool_schema() -> dict:
    return {'type': 'function', 'function': {
        'name': 'recall_event',
        'description': 'Fetch the full original content of a past tool output by its event id. When the conversation has been compacted, old tool outputs are replaced by stubs like "[tool output elided ... Use recall_event(41) ...]"; call this with that id to get the verbatim output back. Often you can instead just re-read the file or re-run the command; use recall_event when the output is not easily reproducible. Read-only; returns an error for an unknown id.',
        'parameters': {'type': 'object', 'properties': {'id': {'type': 'integer'}}, 'required': ['id']}}}


def validate_plan(plan: Any) -> list[dict]:
    if not isinstance(plan, list) or not plan:
        raise ToolError('plan must be a non-empty list of items')
    out, active = [], 0
    for i, item in enumerate(plan, 1):
        if not isinstance(item, dict) or not isinstance(item.get('content'), str) or not item['content'].strip():
            raise ToolError(f'item {i}: content must be a non-empty string')
        status = item.get('status')
        if status not in ('pending', 'in_progress', 'completed'):
            raise ToolError(f'item {i}: status must be pending, in_progress or completed')
        active += status == 'in_progress'
        out.append({'id': i, 'content': item['content'].strip(), 'status': status, 'activeForm': str(item.get('activeForm') or '').strip()})
    if active > 1:
        raise ToolError(f'{active} items are in_progress; keep exactly one')
    return out


def render_plan(plan: list[dict]) -> str:
    marks = {'pending': '[ ]', 'in_progress': '[>]', 'completed': '[x]'}
    return '\n'.join(f'{marks[i["status"]]} {i["id"]}. {i["content"]}' for i in plan)


# ----------------------------------------------------------------------------------------------
# The graph
# ----------------------------------------------------------------------------------------------
class State(TypedDict, total=False):
    turn: int
    termination: str | None
    final_text: str | None
    plan: list[dict] | None
    plan_updates: int


class Harness:
    """Store history, context policy, tools and the model client.

    LangGraph nodes share this object and pass compact serializable state.
    """

    def __init__(self, args, base_url, sandbox, log, manifest):
        from langchain_core.messages import SystemMessage, HumanMessage
        from langchain_openai import ChatOpenAI
        self.args, self.sandbox, self.log, self.manifest = args, sandbox, log, manifest
        self.deadline = Deadline(args.timeout_seconds)
        self.workspace = Workspace(args.workspace, self.deadline)
        self.input_budget = int((args.context_window - args.max_tokens) * (1 - COUNT_MARGIN))
        self.trigger_tokens = int(self.input_budget * args.summary_trigger)
        self.recent_tokens = int(self.input_budget * args.recent_fraction)
        kwargs = dict(model=args.model, base_url=base_url, api_key='not-needed', max_tokens=args.max_tokens,
                      timeout=args.timeout_seconds, max_retries=0, streaming=False, use_responses_api=False)
        if args.seed is not None:
            kwargs['seed'] = args.seed
        self.model = ChatOpenAI(**kwargs)
        self.tool_schemas: list[dict] = []
        self.mcp_tools: dict[str, Any] = {}
        # Raw history stores LangChain messages and bookkeeping in an append-only trajectory.
        # Plan and loop reminders are added separately to each request.
        self.history: list[dict] = []
        self.summary: dict | None = None       # {'text', 'covered_through', 'version'}
        self.chars_per_token = DEFAULT_CHARS_PER_TOKEN
        self.last_view_chars = 0
        self.streak: dict = {'key': None, 'n': 0, 'failing': 0, 'reminded': False}
        self.pending_reminders: list[str] = []
        self.requests = 0
        self.usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'summary_requests': 0, 'summary_prompt_tokens': 0, 'summary_completion_tokens': 0}
        self.elides = args.context_policy in ('elide', 'elide_recall', 'elide_then_summarize')
        self.recalls = args.context_policy in ('elide_recall', 'elide_then_summarize')
        self.summarizes = args.context_policy in ('summarize', 'elide_then_summarize')
        self.elide_tokens = int(self.input_budget * (args.elide_trigger if args.context_policy == 'elide_then_summarize' else args.summary_trigger))
        self.elided: dict[int, str] = {}       # event_id -> original observation (the external store)
        self.recall_calls = 0
        self.elide_dir = sandbox / 'elided'
        if args.tool_mode == 'browser':
            system = BROWSER_SYSTEM_PROMPT
        else:
            system = BASH_ONLY_SYSTEM_PROMPT if args.action_space == 'bash' else BASE_SYSTEM_PROMPT
        if args.tool_bridge == 'on':
            system += '\n\n' + BRIDGE_SYSTEM_BLOCK
        if args.planning == 'on':
            system += '\n\n' + PLANNING_SYSTEM_BLOCK
        self.preamble = [SystemMessage(content=system), HumanMessage(content=args.prompt_file.read_text())]
        self.expected_files = []
        if args.verification == 'on' and args.tool_mode == 'files':
            prompt_text = args.prompt_file.read_text()
            for rel in SUBMISSION_PATH_RE.findall(prompt_text):
                if str(args.workspace / rel) in prompt_text and rel not in self.expected_files:
                    self.expected_files.append(rel)
        self.verification_reminders = 0
        self.verification_missing_at_end: list[str] | None = None
        self.preamble_chars = sum(len(m.content) for m in self.preamble)

    # -- tools -------------------------------------------------------------------------------
    async def setup_tools(self, stack):
        if self.args.tool_mode == 'files':
            self.tool_schemas = files_tool_schemas()
            if self.args.action_space == 'bash':
                self.tool_schemas = [t for t in self.tool_schemas if t['function']['name'] == 'bash']
        else:
            from langchain_mcp_adapters.client import MultiServerMCPClient
            from langchain_mcp_adapters.tools import load_mcp_tools
            from langchain_core.utils.function_calling import convert_to_openai_tool
            config = json.loads(self.args.mcp_config.read_text())
            connections = config.get('mcpServers', config)
            if not isinstance(connections, dict) or not connections:
                raise ValueError('MCP config must contain server connections')
            connections = {name: {**value, 'transport': value.get('transport', 'stdio')} for name, value in connections.items()}
            client = MultiServerMCPClient(connections)
            tools = []
            for name in connections:
                session = await stack.enter_async_context(client.session(name))
                tools.extend(await load_mcp_tools(session))
            allowed = [t for t in tools if t.name in BROWSER_TOOLS]
            if not any(t.name == 'browser_snapshot' for t in allowed):
                raise ValueError('browser MCP must provide browser_snapshot')
            self.mcp_tools = {t.name: t for t in allowed}
            self.tool_schemas = [convert_to_openai_tool(t) for t in allowed]
            self.log.emit('browser_tools', allowed=sorted(self.mcp_tools), excluded=[t.name for t in tools if t.name not in BROWSER_TOOLS])
        self.hidden_schemas: dict[str, dict] = {}
        if self.args.tool_bridge == 'on':
            # The request exposes bridge tools for discovering and invoking task tools.
            self.hidden_schemas = {t['function']['name']: t for t in self.tool_schemas}
            self.tool_schemas = bridge_tool_schemas()
            self.bridge_discovered: set[str] = set()
            self.bridge_calls = {'tool_search': 0, 'tool_describe': 0, 'tool_call': 0, 'undiscovered': 0}
        if self.args.planning == 'on':
            self.tool_schemas.append(planning_tool_schema())
        if self.recalls:
            self.tool_schemas.append(recall_tool_schema())
        self.manifest['tool_names'] = [s['function']['name'] for s in self.tool_schemas]
        self.manifest['hidden_tool_names'] = sorted(self.hidden_schemas)

    async def run_tool(self, name: str, arguments: dict, via_bridge: bool = False) -> tuple[str, bool]:
        """Return (observation, is_error), including input errors as tool observations."""
        try:
            if self.args.tool_bridge == 'on' and name in ('tool_search', 'tool_describe', 'tool_call'):
                return await self.run_bridge(name, arguments)
            if self.args.tool_bridge == 'on' and name in self.hidden_schemas and not via_bridge:
                # A hidden tool named directly is unknown to the model-facing array.
                self.bridge_calls['undiscovered'] += 1
                raise ToolError(f'unknown tool: {name}. Use tool_search to find tools and tool_call to invoke them.')
            if name == 'recall_event' and self.recalls:
                event_id = arguments.get('id')
                if not isinstance(event_id, int) or event_id not in self.elided:
                    raise ToolError(f'no elided event with id {event_id!r}')
                self.recall_calls += 1
                self.log.emit('recall_event', recalled_event_id=event_id, chars=len(self.elided[event_id]))
                return self.elided[event_id], False
            if self.args.tool_mode == 'files' and self.args.action_space == 'bash' and name not in ('bash', 'update_plan', 'recall_event'):
                raise ToolError(f'unknown tool: {name} (only bash is available)')
            if name == 'update_plan':
                if self.args.planning != 'on':
                    raise ToolError('unknown tool: update_plan')
                plan = validate_plan(arguments.get('plan'))
                self.plan = plan
                self.plan_updates += 1
                return f'Plan updated ({len(plan)} items).\n' + render_plan(plan), False
            if self.args.tool_mode == 'browser':
                tool = self.mcp_tools.get(name)
                if tool is None:
                    raise ToolError(f'unknown tool: {name}')
                result = await asyncio.wait_for(tool.ainvoke(arguments), timeout=max(1.0, self.deadline.remaining()))
                return (result if isinstance(result, str) else json.dumps(result, default=serializable, ensure_ascii=False)), False
            ws = self.workspace
            if name == 'read_file':
                return ws.read_file(arguments.get('path'), arguments.get('offset', 1) or 1, arguments.get('limit', READ_DEFAULT_LIMIT) or READ_DEFAULT_LIMIT), False
            if name == 'write_file':
                return ws.write_file(arguments.get('path'), arguments.get('content'), bool(arguments.get('overwrite', False))), False
            if name == 'edit_file':
                return ws.edit_file(arguments.get('path'), arguments.get('old_text'), arguments.get('new_text'), bool(arguments.get('replace_all', False))), False
            if name == 'list_files':
                return ws.list_files(arguments.get('path') or '.', bool(arguments.get('recursive', False)), arguments.get('max_entries') or 200), False
            if name == 'glob_files':
                return ws.glob_files(arguments.get('pattern'), arguments.get('path') or '.', arguments.get('max_matches') or 200), False
            if name == 'grep_text':
                return ws.grep_text(arguments.get('query'), arguments.get('path') or '.', arguments.get('include'),
                                    bool(arguments.get('case_sensitive', True)), arguments.get('max_matches') or 100), False
            if name == 'bash':
                return await asyncio.get_running_loop().run_in_executor(
                    None, lambda: ws.bash(arguments.get('command'), arguments.get('timeout_seconds') or BASH_DEFAULT_TIMEOUT, arguments.get('cwd')))
            raise ToolError(f'unknown tool: {name}')
        except ToolError as exc:
            return f'error: {exc}', True
        except asyncio.TimeoutError:
            return 'error: tool call exceeded the episode deadline', True
        except Exception as exc:
            # Return malformed arguments as tool observations for the next model turn.
            self.log.emit('tool_exception', tool=name, error=f'{type(exc).__name__}: {exc}')
            return f'error: {type(exc).__name__}: {exc}', True

    # -- context view ------------------------------------------------------------------------
    plan: list[dict] | None = None
    plan_updates = 0

    def estimate_tokens(self, chars: int) -> int:
        return int(chars / self.chars_per_token)

    def message_chars(self, msg) -> int:
        content = msg.content if isinstance(msg.content, str) else json.dumps(msg.content, ensure_ascii=False)
        extra = json.dumps(getattr(msg, 'tool_calls', None) or [], ensure_ascii=False) if getattr(msg, 'tool_calls', None) else ''
        return len(content) + len(extra)

    def visible_history(self) -> list[dict]:
        """History entries not covered by the running summary."""
        if self.summary is None:
            return self.history
        return [h for h in self.history if h['event_id'] > self.summary['covered_through']]

    def build_view(self):
        from langchain_core.messages import HumanMessage
        from langchain_core.messages import ToolMessage
        view = list(self.preamble)
        if self.summary is not None:
            wrapper = SUMMARY_WRAPPER_RECALL.format(SUMMARY=self.summary['text'], EVENT_ID=self.summary['covered_through']) if self.recalls else SUMMARY_WRAPPER.format(SUMMARY=self.summary['text'])
            view.append(HumanMessage(content=wrapper))
        for h in self.visible_history():
            m = h['message']
            if h['event_id'] in self.elided:
                original = self.elided[h['event_id']]
                stub = (ELIDED_STUB_RECALL if self.recalls else ELIDED_STUB).format(N_LINES=original.count('\n') + 1, N_CHARS=len(original), EVENT_ID=h['event_id'])
                m = ToolMessage(content=stub, tool_call_id=m.tool_call_id, name=m.name, status=m.status)
            view.append(m)
        injections = list(self.pending_reminders)
        if self.args.planning == 'on':
            injections.append(PLANNING_TURN_REMINDER.format(PLAN=render_plan(self.plan)) if self.plan else PLANNING_FIRST_TURN_REMINDER)
        if injections:
            view.append(HumanMessage(content='\n\n'.join(injections)))
        return view

    def view_chars(self, view) -> int:
        return sum(self.message_chars(m) for m in view) + 400 * len(self.tool_schemas)

    def turn_blocks(self, entries: list[dict]) -> list[list[dict]]:
        """Group history entries into complete blocks: one AI message plus the tool results it caused."""
        blocks, current = [], []
        for h in entries:
            if h['role'] == 'ai':
                if current:
                    blocks.append(current)
                current = [h]
            else:
                current.append(h)
        if current:
            blocks.append(current)
        return blocks

    async def maybe_summarize(self):
        """Apply the context policy before a request. Returns None or a termination reason."""
        view = self.build_view()
        est = self.estimate_tokens(self.view_chars(view))
        if self.args.context_policy == 'none':
            if est > self.input_budget:
                self.log.emit('context_overflow_predicted', estimated_tokens=est, input_budget=self.input_budget)
                return 'context_overflow'
            return None
        if self.elides and est >= self.elide_tokens:
            est = self.elide_middle(est)
        if not self.summarizes:
            if est > self.input_budget:
                self.log.emit('context_overflow_predicted', estimated_tokens=est, input_budget=self.input_budget,
                              view=[(type(m).__name__, self.message_chars(m)) for m in self.build_view()])
                return 'context_overflow'
            return None
        if est < self.trigger_tokens:
            return None
        blocks = self.turn_blocks(self.visible_history())
        # Protect the recent window: at least two blocks, and enough blocks to fill recent_tokens.
        keep, kept_chars = 0, 0
        for block in reversed(blocks):
            block_chars = sum(self.message_chars(h['message']) for h in block)
            if keep >= 2 and self.estimate_tokens(kept_chars + block_chars) > self.recent_tokens:
                break
            keep += 1
            kept_chars += block_chars
        old = blocks[:len(blocks) - keep]
        if not old:
            self.log.emit('summary_skipped', reason='nothing outside the protected window', estimated_tokens=est)
            return 'context_overflow' if est > self.input_budget else None
        # The summary call itself must fit: feed the oldest blocks in chunks that fit the budget.
        selected, chars = [], 0
        chunk_limit = self.input_budget * self.chars_per_token * 0.6
        for block in old:
            block_chars = sum(self.message_chars(h['message']) for h in block)
            if selected and chars + block_chars > chunk_limit:
                break
            selected.append(block)
            chars += block_chars
        covered_through = selected[-1][-1]['event_id']
        source_ids = [h['event_id'] for b in selected for h in b]
        self.log.emit('summary_triggered', estimated_tokens=est, trigger_tokens=self.trigger_tokens, blocks_selected=len(selected),
                      blocks_kept=keep, source_event_ids=source_ids, summary_version=(self.summary or {}).get('version', 0) + 1)
        text = await self.summarize(selected)
        if text is None:
            return 'provider_failure'
        version = (self.summary or {}).get('version', 0) + 1
        self.summary = {'text': text, 'covered_through': covered_through, 'version': version}
        after = self.estimate_tokens(self.view_chars(self.build_view()))
        self.log.emit('summary_applied', summary_version=version, covered_through=covered_through, estimated_tokens_before=est, estimated_tokens_after=after)
        self.log.transform(kind='summary', version=version, covered_event_ids=source_ids, covered_through=covered_through,
                           estimated_tokens_before=est, estimated_tokens_after=after, summary_chars=len(text), summary=text)
        if after > self.input_budget:
            # Summarize remaining old blocks if the input still exceeds the budget.
            return await self.maybe_summarize() if len(old) > len(selected) else 'context_overflow'
        return None

    def elide_middle(self, est: int) -> int:
        """Replace large middle-history observations with stubs and store their originals.

        The recall tool exposes stored observations when enabled. Return the new token estimate.
        """
        blocks = self.turn_blocks(self.visible_history())
        keep, kept_chars = 0, 0
        for block in reversed(blocks):
            block_chars = sum(self.message_chars(h['message']) for h in block)
            if keep >= 2 and self.estimate_tokens(kept_chars + block_chars) > self.recent_tokens:
                break
            keep += 1
            kept_chars += block_chars
        newly = []
        for block in blocks[:len(blocks) - keep]:
            for h in block:
                if h['role'] == 'tool' and h['event_id'] not in self.elided:
                    content = h['message'].content if isinstance(h['message'].content, str) else json.dumps(h['message'].content)
                    if len(content) >= ELIDE_MIN_CHARS:
                        self.elided[h['event_id']] = content
                        newly.append(h['event_id'])
        if not newly:
            return est
        if self.recalls:
            self.elide_dir.mkdir(exist_ok=True)
            for event_id in newly:
                (self.elide_dir / f'{event_id}.txt').write_text(self.elided[event_id], encoding='utf-8')
        after = self.estimate_tokens(self.view_chars(self.build_view()))
        self.log.emit('history_elided', event_ids=newly, estimated_tokens_before=est, estimated_tokens_after=after, recallable=self.recalls)
        self.log.transform(kind='elision', event_ids=newly, estimated_tokens_before=est, estimated_tokens_after=after, recallable=self.recalls)
        return after

    async def summarize(self, blocks: list[list[dict]]) -> str | None:
        from langchain_core.messages import SystemMessage, HumanMessage
        transcript = []
        for block in blocks:
            for h in block:
                m = h['message']
                if h['role'] == 'ai':
                    calls = getattr(m, 'tool_calls', None) or []
                    transcript.append('ASSISTANT: ' + (m.content if isinstance(m.content, str) else json.dumps(m.content)) +
                                      ''.join(f'\n  -> call {c["name"]}({json.dumps(c["args"], ensure_ascii=False)[:2000]})' for c in calls))
                else:
                    content = m.content if isinstance(m.content, str) else json.dumps(m.content)
                    if h['event_id'] in self.elided:
                        content = ELIDED_STUB.format(N_LINES=content.count('\n') + 1, N_CHARS=len(content), EVENT_ID=h['event_id'])
                    transcript.append(f'TOOL RESULT ({h.get("tool_name")}): ' + content)
        user = ''
        if self.summary is not None:
            user += '## Previous summary\n' + self.summary['text'] + '\n\n'
        user += '## Oldest transcript events to fold in\n' + '\n\n'.join(transcript)
        messages = [SystemMessage(content=SUMMARY_PROMPT), HumanMessage(content=user)]
        self.usage['summary_requests'] += 1
        self.log.emit('model_call_started', kind='summary', chars=len(user))
        try:
            response = await asyncio.wait_for(self.model.ainvoke(messages), timeout=max(1.0, self.deadline.remaining()))
        except Exception as exc:
            self.log.emit('model_call_failed', kind='summary', error=f'{type(exc).__name__}: {exc}')
            return None
        usage = (response.usage_metadata or {}) if hasattr(response, 'usage_metadata') else {}
        self.usage['summary_prompt_tokens'] += usage.get('input_tokens', 0) or 0
        self.usage['summary_completion_tokens'] += usage.get('output_tokens', 0) or 0
        text = response.content if isinstance(response.content, str) else json.dumps(response.content)
        self.log.emit('model_call_completed', kind='summary', usage=usage, chars=len(text))
        if not text.strip():
            self.log.emit('summary_empty')
            return None
        return text.strip()

    # -- graph nodes -------------------------------------------------------------------------
    async def node_prepare(self, state: State) -> State:
        if self.deadline.expired():
            return {'termination': 'timeout'}
        if state.get('turn', 0) >= MAX_STEPS:
            return {'termination': 'max_steps'}
        term = await self.maybe_summarize()
        return {'termination': term}

    async def node_call_model(self, state: State) -> State:
        view = self.build_view()
        self.pending_reminders = []
        chars = self.view_chars(view)
        turn = state.get('turn', 0) + 1
        self.requests += 1
        self.log.emit('model_call_started', kind='main', turn=turn, view_messages=len(view), estimated_tokens=self.estimate_tokens(chars))
        bound = self.model.bind_tools(self.tool_schemas) if self.tool_schemas else self.model
        try:
            response = await asyncio.wait_for(bound.ainvoke(view), timeout=max(1.0, self.deadline.remaining()))
        except asyncio.TimeoutError:
            self.log.emit('model_call_cancelled', kind='main', turn=turn, reason='deadline')
            return {'turn': turn, 'termination': 'timeout'}
        except Exception as exc:
            text = f'{type(exc).__name__}: {exc}'
            self.log.emit('model_call_failed', kind='main', turn=turn, error=text)
            overflow = 'input_tokens' in text or 'maximum context length' in text or 'context length' in text
            return {'turn': turn, 'termination': 'context_overflow' if overflow else 'provider_failure'}
        usage = (response.usage_metadata or {}) if hasattr(response, 'usage_metadata') else {}
        prompt_tokens = usage.get('input_tokens') or 0
        if prompt_tokens and chars:
            self.chars_per_token = max(1.5, min(6.0, chars / prompt_tokens))
        self.usage['prompt_tokens'] += prompt_tokens
        self.usage['completion_tokens'] += usage.get('output_tokens', 0) or 0
        finish = (response.response_metadata or {}).get('finish_reason')
        calls = list(getattr(response, 'tool_calls', None) or [])
        invalid = list(getattr(response, 'invalid_tool_calls', None) or [])
        content = response.content if isinstance(response.content, str) else json.dumps(response.content)
        event_id = self.log.emit('model_call_completed', kind='main', turn=turn, usage=usage, finish_reason=finish,
                                 tool_calls=[c['name'] for c in calls], invalid_tool_calls=len(invalid), content_chars=len(content))
        self.persist_counts(turn)
        # The response object itself is stored: it carries tool_calls and invalid_tool_calls as the
        # provider returned them, so the next request replays exactly what the model emitted.
        self.history.append({'event_id': event_id, 'role': 'ai', 'message': response})
        if calls or invalid:
            return {'turn': turn, 'termination': None, 'final_text': None}
        if finish == 'length':
            return {'turn': turn, 'termination': 'output_limit', 'final_text': content}
        if not content.strip():
            return {'turn': turn, 'termination': 'empty_output', 'final_text': content}
        missing = self.missing_files()
        if missing and self.verification_reminders == 0:
            # Verification adds one reminder when a final answer leaves required output files missing.
            self.verification_reminders += 1
            self.pending_reminders.append(VERIFICATION_REMINDER.format(FILES='\n'.join(f'  {self.args.workspace / rel}' for rel in missing)))
            self.log.emit('verification_reminder', missing=missing, turn=turn)
            return {'turn': turn, 'termination': None, 'final_text': None}
        self.verification_missing_at_end = missing
        return {'turn': turn, 'termination': 'normal', 'final_text': content}

    async def node_dispatch(self, state: State) -> State:
        from langchain_core.messages import ToolMessage
        ai = self.history[-1]['message']
        calls = list(getattr(ai, 'tool_calls', None) or [])
        invalid = list(getattr(ai, 'invalid_tool_calls', None) or [])
        for bad in invalid:
            calls.append({'name': bad.get('name') or 'unknown', 'args': {}, 'id': bad.get('id') or f'invalid-{self.log.n}', '_invalid': bad.get('error')})
        for call in calls:
            if self.deadline.expired():
                return {'termination': 'timeout'}
            name, arguments, call_id = call['name'], call.get('args') or {}, call.get('id') or f'call-{self.log.n}'
            started = self.log.emit('tool_call_started', tool=name, args=arguments, call_id=call_id)
            if call.get('_invalid'):
                observation, is_error = f'error: malformed tool call ({call["_invalid"]})', True
            else:
                observation, is_error = await self.run_tool(name, arguments)
            if len(observation) > TOOL_RESULT_MAX_CHARS:
                observation = observation[:TOOL_RESULT_MAX_CHARS] + f'\n[truncated: {len(observation) - TOOL_RESULT_MAX_CHARS} more chars; re-read with an offset or narrow the command]'
            event_id = self.log.emit('tool_call_failed' if is_error else 'tool_call_completed', tool=name, call_id=call_id,
                                     started_event_id=started, chars=len(observation))
            self.history.append({'event_id': event_id, 'role': 'tool', 'tool_name': name,
                                 'message': ToolMessage(content=observation, tool_call_id=call_id, name=name, status='error' if is_error else 'success')})
            if name == 'update_plan' and not is_error:
                self.log.emit('plan_updated', items=len(self.plan or []), updates=self.plan_updates)
            term = self.track_streak(name, arguments, is_error)
            if term:
                return {'termination': term}
        return {'termination': None}

    def persist_counts(self, turn: int):
        """Persist current usage, summary and planning counters in the episode manifest."""
        self.manifest.update(turns=turn, requests=self.requests, usage=dict(self.usage),
                             plan_created=self.plan is not None, plan_updates=self.plan_updates,
                             summary_versions=(self.summary or {}).get('version', 0),
                             elided_events=len(self.elided), recall_calls=self.recall_calls,
                             chars_per_token=round(self.chars_per_token, 3), counts_partial=True)
        save_manifest(self.sandbox, self.manifest)

    async def run_bridge(self, name: str, arguments: dict) -> tuple[str, bool]:
        self.bridge_calls[name] += 1
        if name == 'tool_search':
            query = str(arguments.get('query') or '').lower()
            words = [w for w in re.split(r'[^a-z0-9_]+', query) if w]
            limit = min(max(int(arguments.get('limit') or 5), 1), 25)
            scored = []
            for tname, schema in self.hidden_schemas.items():
                text = (tname + ' ' + schema['function']['description']).lower()
                score = sum(text.count(w) for w in words) + (3 if any(w in tname for w in words) else 0)
                if score > 0 or not words:
                    scored.append((score, tname))
            scored.sort(key=lambda x: (-x[0], x[1]))
            hits = [t for _, t in scored[:limit]]
            self.bridge_discovered.update(hits)
            if not hits:
                return 'no tools match; try other keywords (e.g. read, write, list, search, run)', False
            return '\n'.join(f"{t}: {self.hidden_schemas[t]['function']['description'].split('.')[0]}." for t in hits), False
        if name == 'tool_describe':
            tname = str(arguments.get('name') or '')
            if tname not in self.hidden_schemas:
                raise ToolError(f'unknown tool: {tname}')
            self.bridge_discovered.add(tname)
            return json.dumps(self.hidden_schemas[tname]['function'], ensure_ascii=False), False
        # tool_call
        tname = str(arguments.get('name') or '')
        inner = arguments.get('arguments')
        if tname not in self.hidden_schemas:
            raise ToolError(f'unknown tool: {tname}; use tool_search first')
        if tname not in self.bridge_discovered:
            self.bridge_calls['undiscovered'] += 1
            raise ToolError(f'{tname} has not been discovered in this session; call tool_search or tool_describe first')
        if isinstance(inner, str):
            try:
                inner = json.loads(inner)
            except json.JSONDecodeError:
                raise ToolError('arguments must be a JSON object')
        if not isinstance(inner, dict):
            raise ToolError('arguments must be a JSON object')
        return await self.run_tool(tname, inner, via_bridge=True)

    def missing_files(self) -> list[str]:
        out = []
        for rel in self.expected_files:
            path = self.args.workspace / rel
            if not path.is_file() or path.stat().st_size == 0:
                out.append(rel)
        return out

    def track_streak(self, name: str, arguments: dict, is_error: bool) -> str | None:
        key = name + ':' + json.dumps(arguments, sort_keys=True, ensure_ascii=False)
        s = self.streak
        if key != s['key']:
            self.streak = {'key': key, 'n': 1, 'failing': 1 if is_error else 0, 'reminded': False}
            return None
        s['n'] += 1
        s['failing'] = s['failing'] + 1 if is_error else 0
        if s['failing'] >= STUCK_STOP_AT:
            self.log.emit('stuck_stop', tool=name, streak=s['n'], failing=s['failing'])
            return 'stuck'
        if s['n'] >= STUCK_REPEAT_STOP_AT:
            # Stop after repeated identical calls reach the configured limit.
            self.log.emit('stuck_stop', tool=name, streak=s['n'], failing=s['failing'], kind='repeat')
            return 'stuck_repeat'
        if not s['reminded'] and (s['failing'] >= STUCK_REMIND_AT or s['n'] >= STUCK_REMIND_AT):
            s['reminded'] = True
            template = STUCK_FAILING_REMINDER if s['failing'] >= STUCK_REMIND_AT else STUCK_REPEAT_REMINDER
            self.pending_reminders.append(template.format(TOOL=name, N=s['n']))
            self.log.emit('stuck_reminder', tool=name, streak=s['n'], failing=s['failing'])
        return None

    def build_graph(self):
        from langgraph.graph import StateGraph, END
        g = StateGraph(State)
        g.add_node('prepare_input', self.node_prepare)
        g.add_node('call_model', self.node_call_model)
        g.add_node('dispatch_tools', self.node_dispatch)
        g.set_entry_point('prepare_input')
        g.add_conditional_edges('prepare_input', lambda s: END if s.get('termination') else 'call_model')
        g.add_conditional_edges('call_model', lambda s: END if s.get('termination') else 'dispatch_tools')
        g.add_conditional_edges('dispatch_tools', lambda s: END if s.get('termination') else 'prepare_input')
        return g.compile()


async def run(args, base_url, sandbox, log, manifest):
    harness = Harness(args, base_url, sandbox, log, manifest)
    manifest.update(input_budget=harness.input_budget, trigger_tokens=harness.trigger_tokens, recent_tokens=harness.recent_tokens)
    async with AsyncExitStack() as stack:
        await harness.setup_tools(stack)
        graph = harness.build_graph()
        manifest['graph_nodes'] = list(graph.get_graph().nodes)
        manifest['status'] = 'running'
        save_manifest(sandbox, manifest)
        final = await graph.ainvoke({'turn': 0, 'termination': None, 'final_text': None},
                                    config={'recursion_limit': MAX_STEPS * 4 + 10})
    manifest['counts_partial'] = False
    manifest.update(bridge_calls=getattr(harness, 'bridge_calls', None),
                    expected_files=harness.expected_files, verification_reminders=harness.verification_reminders,
                    verification_missing_at_end=harness.verification_missing_at_end)
    manifest.update(termination_reason=final.get('termination'), turns=final.get('turn', 0), requests=harness.requests,
                    usage=harness.usage, plan_created=harness.plan is not None, plan_updates=harness.plan_updates,
                    elided_events=len(harness.elided), recall_calls=harness.recall_calls,
                    summary_versions=(harness.summary or {}).get('version', 0), chars_per_token=round(harness.chars_per_token, 3))
    log.emit('episode_terminated', reason=final.get('termination'), turns=final.get('turn', 0))
    # stdout carries execution diagnostics. Scoring reads task output files.
    print(json.dumps({'status': 'completed', 'termination_reason': final.get('termination'), 'final_message': final.get('final_text')}, ensure_ascii=False))
    return final


def save_manifest(sandbox, manifest):
    target = sandbox / 'langgraph-manifest.json'
    temp = target.with_suffix('.tmp')
    temp.write_text(json.dumps(manifest, indent=2, default=serializable) + '\n')
    temp.replace(target)


def main(argv=None):
    args = parse_args(argv)
    sandbox = Path(os.environ.get('HARNESSBENCH_SANDBOX', str(args.workspace.parent))).resolve()
    sandbox.mkdir(parents=True, exist_ok=True)
    log = EventLog(sandbox)
    manifest = {'version': 1, 'harness': 'medicalharness_langgraph', 'condition_id': args.condition_id,
                'model': args.model, 'upstream': args.upstream, 'max_tokens': args.max_tokens,
                'context_window': args.context_window, 'timeout_seconds': args.timeout_seconds,
                'tool_mode': args.tool_mode, 'workspace': args.workspace, 'seed': args.seed,
                'components': {'planning': args.planning == 'on', 'context_policy': args.context_policy,
                               'summary_trigger': args.summary_trigger, 'recent_fraction': args.recent_fraction,
                               'elide_trigger': args.elide_trigger, 'action_space': args.action_space,
                               'verification': args.verification == 'on', 'tool_bridge': args.tool_bridge == 'on',
                               'skills': False, 'delegation': False},
                'substrate': {'tool_result_max_chars': TOOL_RESULT_MAX_CHARS, 'stuck_remind_at': STUCK_REMIND_AT,
                              'stuck_stop_at': STUCK_STOP_AT, 'stuck_repeat_stop_at': STUCK_REPEAT_STOP_AT, 'max_steps': MAX_STEPS, 'count_margin': COUNT_MARGIN,
                              'read_before_write': True, 'denied_commands': list(DENIED_COMMANDS)},
                'system_prompt_sha256': hashlib.sha256((BASE_SYSTEM_PROMPT + PLANNING_SYSTEM_BLOCK + SUMMARY_PROMPT).encode()).hexdigest(),
                'versions': {}, 'graph_nodes': [], 'tool_names': [], 'status': 'starting'}
    started = time.monotonic()
    try:
        base = register_proxy(args.upstream)
        os.chdir(args.workspace)
        private_home = sandbox / 'langgraph-home'
        private_home.mkdir(exist_ok=True)
        os.environ['HOME'] = str(private_home)
        os.environ['LANGSMITH_TRACING'] = 'false'
        os.environ['LANGCHAIN_TRACING_V2'] = 'false'
        for variable in ('LANGSMITH_API_KEY', 'LANGCHAIN_API_KEY'):
            os.environ.pop(variable, None)
        for package in ('langgraph', 'langchain-core', 'langchain-openai', 'langchain-mcp-adapters'):
            try:
                manifest['versions'][package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                manifest['versions'][package] = None
        save_manifest(sandbox, manifest)

        def deadline(signum, frame):
            raise TimeoutError('episode deadline reached')

        signal.signal(signal.SIGALRM, deadline)
        signal.setitimer(signal.ITIMER_REAL, max(1.0, args.timeout_seconds - (time.monotonic() - started)))
        asyncio.run(run(args, base, sandbox, log, manifest))
        manifest['status'] = 'completed'
        return_code = 0
    except (TimeoutError, asyncio.TimeoutError) as exc:
        manifest.update(status='timed_out', termination_reason='timeout', error=str(exc))
        log.emit('episode_terminated', reason='timeout', error=str(exc))
        return_code = 124
    except Exception as exc:
        manifest.update(status='error', termination_reason='runtime_failure', error=f'{type(exc).__name__}: {exc}')
        log.emit('episode_terminated', reason='runtime_failure', error=manifest['error'])
        print(manifest['error'], file=sys.stderr)
        return_code = 1
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        manifest['wall_seconds'] = round(time.monotonic() - started, 3)
        save_manifest(sandbox, manifest)
    return return_code


if __name__ == '__main__':
    raise SystemExit(main())
