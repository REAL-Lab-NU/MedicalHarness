"""Mechanism tests for the MH-Lab LangGraph harness using scripted responses.

A local OpenAI-compatible server supplies responses, and a local proxy records
request bodies for assertions on the model-visible context.

Run with:
    python -m pytest -q tests/test_langgraph_harness.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / 'scripts/harness/run_langgraph_harness.py'
PYTHON = sys.executable


class FakeModel:
    """Serves /langgraph/vllm/chat/completions. `script` is a list of responses in order; each is
    either ('text', str) or ('calls', [(name, args), ...]). Once the script runs out, replies text.
    Records every request body (the model-visible input) in `requests`."""

    def __init__(self, script, prompt_tokens=None):
        self.script = list(script)
        self.requests = []
        self.prompt_tokens = prompt_tokens or (lambda body: sum(len(json.dumps(m)) for m in body['messages']) // 4)
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                length = int(self.headers.get('Content-Length', 0))
                body = json.loads(self.rfile.read(length))
                outer.requests.append({'path': self.path, 'body': body})
                if body.get('tools'):
                    step = outer.script.pop(0) if outer.script else ('text', 'Done.')
                else:
                    step = ('text', '## Goal\nsummary of the old events\n## Next step\ncontinue')
                if step[0] == 'text':
                    message = {'role': 'assistant', 'content': step[1]}
                    finish = step[2] if len(step) > 2 else 'stop'
                else:
                    message = {'role': 'assistant', 'content': '', 'tool_calls': [
                        {'id': f'call-{len(outer.requests)}-{i}', 'type': 'function',
                         'function': {'name': name, 'arguments': json.dumps(args)}} for i, (name, args) in enumerate(step[1])]}
                    finish = 'tool_calls'
                ptok = outer.prompt_tokens(body)
                payload = {'id': 'x', 'object': 'chat.completion', 'model': body.get('model'),
                           'choices': [{'index': 0, 'message': message, 'finish_reason': finish}],
                           'usage': {'prompt_tokens': ptok, 'completion_tokens': 20, 'total_tokens': ptok + 20}}
                data = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f'http://127.0.0.1:{self.server.server_address[1]}'

    def close(self):
        self.server.shutdown()


def run_harness(tmp_path, fake, extra_args=(), prompt='Write the answer 42 to `$WORKSPACE/out/answer.txt`.', window=4000, max_tokens=500):
    workspace = tmp_path / 'workspace'
    (workspace / 'in').mkdir(parents=True)
    (workspace / 'in/chart.txt').write_text('line one\nline two\nline three\n')
    prompt_file = tmp_path / 'prompt.txt'
    prompt_file.write_text(prompt.replace('$WORKSPACE', str(workspace)))
    sandbox = tmp_path / 'sandbox'
    sandbox.mkdir()
    routes = sandbox / 'routes.json'
    env = {**os.environ, 'HARNESSBENCH_LLM_PROXY_URL': fake.url, 'HARNESSBENCH_LLM_PROXY_ROUTES': str(routes),
           'HARNESSBENCH_SANDBOX': str(sandbox)}
    cmd = [PYTHON, str(SCRIPT), '--workspace', str(workspace), '--prompt-file', str(prompt_file), '--model', 'fake',
           '--upstream', 'http://127.0.0.1:9/v1', '--max-tokens', str(max_tokens), '--context-window', str(window),
           '--timeout-seconds', '60', *extra_args]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=120)
    manifest = json.loads((sandbox / 'langgraph-manifest.json').read_text())
    events = [json.loads(l) for l in (sandbox / 'events.jsonl').read_text().splitlines()]
    return proc, manifest, events, workspace, sandbox


def tool_names(request):
    return sorted(t['function']['name'] for t in request['body'].get('tools') or [])


def texts(request):
    return [m.get('content') if isinstance(m.get('content'), str) else json.dumps(m.get('content')) for m in request['body']['messages']]


def test_loop_writes_real_artefact_and_terminates_normally(tmp_path):
    fake = FakeModel([
        ('calls', [('list_files', {'path': '.', 'recursive': True})]),
        ('calls', [('read_file', {'path': 'in/chart.txt'})]),
        ('calls', [('write_file', {'path': 'out/answer.txt', 'content': 'Final answer: 42\n'})]),
        ('text', 'Wrote the answer.'),
    ])
    try:
        proc, manifest, events, workspace, sandbox = run_harness(tmp_path, fake, ['--planning', 'off', '--context-policy', 'none'])
    finally:
        fake.close()
    assert proc.returncode == 0, proc.stderr
    assert (workspace / 'out/answer.txt').read_text() == 'Final answer: 42\n'
    assert manifest['termination_reason'] == 'normal' and manifest['turns'] == 4
    assert manifest['components'] == {'planning': False, 'context_policy': 'none', 'summary_trigger': 0.8, 'recent_fraction': 0.3,
                                      'elide_trigger': 0.6, 'action_space': 'tools', 'verification': False, 'tool_bridge': False, 'skills': False, 'delegation': False}
    assert 'update_plan' not in tool_names(fake.requests[0])
    assert not any('update_plan' in t or 'Task planning' in t for t in texts(fake.requests[0]))
    # A model request is logged before it is sent, and tool results are paired to their calls.
    kinds = [e['event'] for e in events]
    assert kinds.index('model_call_started') < kinds.index('model_call_completed')
    assert kinds.count('tool_call_completed') == 3 and 'episode_terminated' in kinds
    assert json.loads(proc.stdout.strip().splitlines()[-1])['termination_reason'] == 'normal'


def test_planning_on_exposes_tool_reminders_and_injects_plan_without_storing_it(tmp_path):
    plan = [{'content': 'Read the chart', 'status': 'in_progress'}, {'content': 'Write the answer', 'status': 'pending'}]
    fake = FakeModel([
        ('calls', [('update_plan', {'plan': plan})]),
        ('calls', [('read_file', {'path': 'in/chart.txt'})]),
        ('calls', [('update_plan', {'plan': [{'content': 'Read the chart', 'status': 'completed'}, {'content': 'Write the answer', 'status': 'in_progress'}]})]),
        ('text', 'Done.'),
    ])
    try:
        proc, manifest, events, workspace, sandbox = run_harness(tmp_path, fake, ['--planning', 'on', '--context-policy', 'none'])
    finally:
        fake.close()
    assert proc.returncode == 0, proc.stderr
    first, second, third, fourth = fake.requests[:4]
    assert 'update_plan' in tool_names(first)
    assert any('# Task planning' in t for t in texts(first))
    assert 'You have not created a plan yet' in texts(first)[-1]
    assert '[>] 1. Read the chart' in texts(second)[-1] and 'Current plan' in texts(second)[-1]
    assert '[x] 1. Read the chart' in texts(fourth)[-1] and '[>] 2. Write the answer' in texts(fourth)[-1]
    # Exactly one reminder per request: previous copies never accumulate in the history.
    assert sum('Current plan' in t for t in texts(fourth)) == 1
    assert manifest['plan_created'] is True and manifest['plan_updates'] == 2
    assert [e for e in events if e['event'] == 'plan_updated']


def test_invalid_plan_is_a_recoverable_tool_error(tmp_path):
    fake = FakeModel([
        ('calls', [('update_plan', {'plan': [{'content': 'a', 'status': 'in_progress'}, {'content': 'b', 'status': 'in_progress'}]})]),
        ('text', 'giving up'),
    ])
    try:
        proc, manifest, events, workspace, sandbox = run_harness(tmp_path, fake, ['--planning', 'on', '--context-policy', 'none'])
    finally:
        fake.close()
    tool_msg = [m for m in fake.requests[1]['body']['messages'] if m['role'] == 'tool'][0]
    assert 'keep exactly one' in tool_msg['content']
    assert manifest['plan_created'] is False and 'You have not created a plan yet' in texts(fake.requests[1])[-1]


def test_summarization_fires_at_trigger_and_replaces_old_blocks(tmp_path):
    # Each read returns ~700 chars; the fake counts one token per 4 chars; the window is 4000 with a
    # 500-token output cap, so the input budget is ~3325 tokens and the trigger ~2660.
    big = 'x' * 600
    script = [('calls', [('bash', {'command': f'echo {big}'})]) for _ in range(12)] + [('text', 'Done.')]
    fake = FakeModel(script)
    try:
        proc, manifest, events, workspace, sandbox = run_harness(tmp_path, fake, ['--planning', 'off', '--context-policy', 'summarize'])
    finally:
        fake.close()
    assert proc.returncode == 0, proc.stderr
    kinds = [e['event'] for e in events]
    assert 'summary_triggered' in kinds and 'summary_applied' in kinds
    assert manifest['summary_versions'] >= 1 and manifest['usage']['summary_requests'] >= 1
    summary_calls = [r for r in fake.requests if not r['body'].get('tools')]
    assert summary_calls and 'Respond with TEXT ONLY' in summary_calls[0]['body']['messages'][0]['content']
    # After the summary is applied, a main request carries the wrapper right after the preamble and
    # fewer raw tool results than before; the preamble (system + task) stays verbatim.
    after = [r for r in fake.requests if r['body'].get('tools') and any('<context-summary>' in (m.get('content') or '') for m in r['body']['messages'] if isinstance(m.get('content'), str))]
    assert after, 'no main request carried the summary'
    msgs = after[0]['body']['messages']
    assert msgs[0]['role'] == 'system' and msgs[1]['role'] == 'user' and '<context-summary>' in msgs[2]['content']
    # Tool messages never appear without their assistant call in the same view.
    ids_called = {c['id'] for m in msgs if m.get('tool_calls') for c in m['tool_calls']}
    assert all(m['tool_call_id'] in ids_called for m in msgs if m['role'] == 'tool')
    transforms = [json.loads(l) for l in (sandbox / 'context-transforms.jsonl').read_text().splitlines()]
    assert transforms[0]['kind'] == 'summary' and transforms[0]['covered_event_ids']
    assert manifest['termination_reason'] == 'normal'


def test_policy_none_ends_with_context_overflow_instead_of_summarizing(tmp_path):
    big = 'x' * 600
    script = [('calls', [('bash', {'command': f'echo {big}'})]) for _ in range(12)] + [('text', 'Done.')]
    fake = FakeModel(script)
    try:
        proc, manifest, events, workspace, sandbox = run_harness(tmp_path, fake, ['--planning', 'off', '--context-policy', 'none'])
    finally:
        fake.close()
    assert manifest['termination_reason'] == 'context_overflow'
    assert manifest['usage']['summary_requests'] == 0
    assert all(r['body'].get('tools') for r in fake.requests)


def test_stuck_detection_reminds_then_stops(tmp_path):
    script = [('calls', [('read_file', {'path': 'does/not/exist'})]) for _ in range(12)] + [('text', 'Done.')]
    fake = FakeModel(script)
    try:
        proc, manifest, events, workspace, sandbox = run_harness(tmp_path, fake, ['--planning', 'off', '--context-policy', 'none'])
    finally:
        fake.close()
    kinds = [e['event'] for e in events]
    assert 'stuck_reminder' in kinds and 'stuck_stop' in kinds
    assert manifest['termination_reason'] == 'stuck'
    reminded = [r for r in fake.requests if 'keeps failing the same way' in texts(r)[-1]]
    assert len(reminded) == 1, 'one reminder per streak'
    assert kinds.count('tool_call_failed') == 8


def test_workspace_guard_read_before_write_and_denied_commands(tmp_path):
    fake = FakeModel([
        ('calls', [('read_file', {'path': '../../etc/passwd'})]),
        ('calls', [('edit_file', {'path': 'in/chart.txt', 'old_text': 'one', 'new_text': 'uno'})]),
        ('calls', [('read_file', {'path': 'in/chart.txt'}), ('edit_file', {'path': 'in/chart.txt', 'old_text': 'one', 'new_text': 'uno'})]),
        ('calls', [('bash', {'command': 'rm -rf /'})]),
        ('calls', [('bash', {'command': 'printf hello; exit 3'})]),
        ('text', 'Done.'),
    ])
    try:
        proc, manifest, events, workspace, sandbox = run_harness(tmp_path, fake, ['--planning', 'off', '--context-policy', 'none'])
    finally:
        fake.close()
    tool_msgs = [m['content'] for r in fake.requests for m in r['body']['messages'] if m['role'] == 'tool']
    assert any('outside the workspace' in t for t in tool_msgs)
    assert any('never fully read' in t for t in tool_msgs)
    assert (workspace / 'in/chart.txt').read_text().startswith('line uno')
    assert any('denied by the permission layer' in t for t in tool_msgs)
    assert any('exit_code=3' in t and 'hello' in t for t in tool_msgs)


def test_tool_result_cap_and_output_limit_termination(tmp_path):
    fake = FakeModel([
        ('calls', [('bash', {'command': 'head -c 60000 /dev/zero | tr "\\0" a'})]),
        ('text', 'partial answer', 'length'),
    ])
    try:
        proc, manifest, events, workspace, sandbox = run_harness(tmp_path, fake, ['--planning', 'off', '--context-policy', 'none'], window=60000)
    finally:
        fake.close()
    tool_msg = [m['content'] for m in fake.requests[1]['body']['messages'] if m['role'] == 'tool'][0]
    assert len(tool_msg) < 24_500 and '[truncated:' in tool_msg
    assert manifest['termination_reason'] == 'output_limit'


def test_refuses_to_run_without_the_proxy(tmp_path):
    workspace = tmp_path / 'ws'
    workspace.mkdir()
    prompt = tmp_path / 'p.txt'
    prompt.write_text('hi')
    env = {k: v for k, v in os.environ.items() if not k.startswith('HARNESSBENCH_')}
    proc = subprocess.run([PYTHON, str(SCRIPT), '--workspace', str(workspace), '--prompt-file', str(prompt), '--model', 'm',
                           '--upstream', 'http://127.0.0.1:8010/v1'], capture_output=True, text=True, env=env)
    assert proc.returncode == 1 and 'no direct upstream' in proc.stderr


def test_bash_only_action_space_exposes_only_bash_and_the_component_tools(tmp_path):
    fake = FakeModel([
        ('calls', [('read_file', {'path': 'in/chart.txt'})]),
        ('calls', [('bash', {'command': 'mkdir -p out && printf "Final answer: 42\\n" > out/answer.txt'})]),
        ('text', 'Done.'),
    ])
    try:
        proc, manifest, events, workspace, sandbox = run_harness(tmp_path, fake, ['--planning', 'on', '--context-policy', 'none', '--action-space', 'bash'])
    finally:
        fake.close()
    assert proc.returncode == 0, proc.stderr
    assert tool_names(fake.requests[0]) == ['bash', 'update_plan']
    assert any('Working with only a shell' in t for t in texts(fake.requests[0]))
    tool_msg = [m['content'] for m in fake.requests[1]['body']['messages'] if m['role'] == 'tool'][0]
    assert 'only bash is available' in tool_msg
    assert (workspace / 'out/answer.txt').read_text() == 'Final answer: 42\n'
    assert manifest['components']['action_space'] == 'bash'


def elision_script(n=12):
    # A bulky observation (above ELIDE_MIN_CHARS) from a short command: elision targets observations,
    # not tool-call arguments, so the payload must not sit in the arguments.
    # Distinct arguments per call so the stuck detector (identical calls) does not end the run first.
    return [('calls', [('bash', {'command': f"head -c 2000 /dev/zero | tr '\\0' x; echo {i}"})]) for i in range(n)]


def test_tier1_elides_bulky_middle_outputs_without_recall(tmp_path):
    fake = FakeModel(elision_script() + [('text', 'Done.')])
    try:
        proc, manifest, events, workspace, sandbox = run_harness(tmp_path, fake, ['--planning', 'off', '--context-policy', 'elide'])
    finally:
        fake.close()
    kinds = [e['event'] for e in events]
    assert 'history_elided' in kinds and 'summary_triggered' not in kinds
    assert manifest['elided_events'] >= 1 and manifest['usage']['summary_requests'] == 0
    assert 'recall_event' not in tool_names(fake.requests[0])
    stubbed = [m for r in fake.requests for m in r['body']['messages'] if m['role'] == 'tool' and 'tool output elided' in m['content']]
    assert stubbed and all('Re-read or re-run' in m['content'] and 'recall_event' not in m['content'] for m in stubbed)
    # The most recent two blocks stay verbatim in every request.
    last = fake.requests[-1]['body']['messages']
    recent_tools = [m for m in last if m['role'] == 'tool'][-2:]
    assert all('tool output elided' not in m['content'] for m in recent_tools)
    assert manifest['termination_reason'] == 'normal'


def test_tier2_recall_returns_the_original(tmp_path):
    script = elision_script(10) + [('calls', [('recall_event', {'id': 4})]), ('text', 'Done.')]
    fake = FakeModel(script)
    try:
        proc, manifest, events, workspace, sandbox = run_harness(tmp_path, fake, ['--planning', 'off', '--context-policy', 'elide_recall'])
    finally:
        fake.close()
    assert 'recall_event' in tool_names(fake.requests[0])
    stubbed = [m for r in fake.requests for m in r['body']['messages'] if m['role'] == 'tool' and 'tool output elided' in m['content']]
    assert stubbed and all('recall_event(' in m['content'] for m in stubbed)
    recalled = [e for e in events if e['event'] == 'recall_event']
    assert recalled and recalled[0]['recalled_event_id'] == 4
    last_tool = [m['content'] for m in fake.requests[-1]['body']['messages'] if m['role'] == 'tool'][-1]
    assert 'x' * 2000 in last_tool and manifest['recall_calls'] == 1
    assert (sandbox / 'elided' / '4.txt').exists()


def test_tier4_elides_before_it_summarizes(tmp_path):
    # Elision runs first. Summarization follows when retained turns and stubs reach its threshold.
    fake = FakeModel(elision_script(24) + [('text', 'Done.')])
    try:
        proc, manifest, events, workspace, sandbox = run_harness(tmp_path, fake, ['--planning', 'off', '--context-policy', 'elide_then_summarize'])
    finally:
        fake.close()
    kinds = [e['event'] for e in events]
    assert 'history_elided' in kinds and 'summary_applied' in kinds
    assert kinds.index('history_elided') < kinds.index('summary_triggered')
    assert 'recall_event' in tool_names(fake.requests[0])
    summary_calls = [r for r in fake.requests if not r['body'].get('tools')]
    assert 'tool output elided' in summary_calls[0]['body']['messages'][1]['content']
    after = [r for r in fake.requests if r['body'].get('tools') and any('<context-summary>' in (m.get('content') or '') for m in r['body']['messages'] if isinstance(m.get('content'), str))]
    assert after and 'recallable via recall_event' in after[0]['body']['messages'][2]['content']


def test_malformed_tool_arguments_are_observations_not_crashes(tmp_path):
    fake = FakeModel([
        ('calls', [('grep_text', {'query': ['not', 'a', 'string']})]),      # re.compile TypeError
        ('calls', [('read_file', {'path': 'in/chart.txt', 'offset': 'one'})]),  # int() ValueError
        ('text', 'Done.'),
    ])
    try:
        proc, manifest, events, workspace, sandbox = run_harness(tmp_path, fake, ['--planning', 'off', '--context-policy', 'none'])
    finally:
        fake.close()
    assert proc.returncode == 0 and manifest['termination_reason'] == 'normal'
    tool_msgs = [m['content'] for r in fake.requests for m in r['body']['messages'] if m['role'] == 'tool']
    assert any(t.startswith('error: TypeError') for t in tool_msgs) and any(t.startswith('error: ValueError') for t in tool_msgs)
    assert [e for e in events if e['event'] == 'tool_exception']


def test_planning_off_rejects_a_hallucinated_update_plan(tmp_path):
    fake = FakeModel([('calls', [('update_plan', {'plan': [{'content': 'x', 'status': 'in_progress'}]})]), ('text', 'Done.')])
    try:
        proc, manifest, events, workspace, sandbox = run_harness(tmp_path, fake, ['--planning', 'off', '--context-policy', 'none'])
    finally:
        fake.close()
    tool_msg = [m['content'] for m in fake.requests[1]['body']['messages'] if m['role'] == 'tool'][0]
    assert 'unknown tool: update_plan' in tool_msg and manifest['plan_created'] is False and manifest['plan_updates'] == 0


def test_bash_timeout_kills_the_process_tree_and_logs_are_off_limits(tmp_path):
    fake = FakeModel([
        ('calls', [('bash', {'command': 'sleep 31.7 & echo started; wait', 'timeout_seconds': 1})]),
        ('calls', [('bash', {'command': 'cat ../events.jsonl'})]),
        ('text', 'Done.'),
    ])
    try:
        proc, manifest, events, workspace, sandbox = run_harness(tmp_path, fake, ['--planning', 'off', '--context-policy', 'none'])
    finally:
        fake.close()
    tool_msgs = [m['content'] for r in fake.requests for m in r['body']['messages'] if m['role'] == 'tool']
    assert any('timed_out=True' in t for t in tool_msgs)
    import subprocess as sp
    assert 'sleep 31.7' not in sp.run(['ps', '-eo', 'args'], capture_output=True, text=True).stdout
    assert any("episode's own record" in t for t in tool_msgs)


def test_counts_survive_a_deadline_kill(tmp_path):
    # A small context window ends the scripted tool loop with context_overflow.
    # The manifest retains the counters recorded before termination.
    script = [('calls', [('bash', {'command': "head -c 3000 /dev/zero | tr '\\0' x"})]) for _ in range(6)]
    fake = FakeModel(script)
    try:
        proc, manifest, events, workspace, sandbox = run_harness(tmp_path, fake, ['--planning', 'on', '--context-policy', 'none'])
    finally:
        fake.close()
    assert manifest['termination_reason'] == 'context_overflow'
    assert manifest['usage']['prompt_tokens'] > 0 and manifest['requests'] >= 1 and manifest['counts_partial'] is False
    # and the partial write exists in the event stream order: counts were persisted after the first call
    assert [e for e in events if e['event'] == 'model_call_completed']


def test_identical_successful_calls_stop_after_the_reminder_is_ignored(tmp_path):
    script = [('calls', [('list_files', {'path': '.'})]) for _ in range(30)] + [('text', 'Done.')]
    fake = FakeModel(script)
    try:
        proc, manifest, events, workspace, sandbox = run_harness(tmp_path, fake, ['--planning', 'off', '--context-policy', 'none'])
    finally:
        fake.close()
    kinds = [e['event'] for e in events]
    assert kinds.count('stuck_reminder') == 1 and manifest['termination_reason'] == 'stuck_repeat'
    assert kinds.count('tool_call_completed') == 15


def test_verification_reminds_once_when_the_named_file_is_missing(tmp_path):
    fake = FakeModel([
        ('text', 'The answer is 42.'),                                            # stops without writing
        ('calls', [('write_file', {'path': 'out/answer.txt', 'content': 'Final answer: 42\n'})]),
        ('text', 'Written.'),
    ])
    try:
        proc, manifest, events, workspace, sandbox = run_harness(tmp_path, fake, ['--planning', 'off', '--context-policy', 'none', '--verification', 'on'])
    finally:
        fake.close()
    assert proc.returncode == 0 and manifest['termination_reason'] == 'normal'
    assert manifest['expected_files'] == ['out/answer.txt'] and manifest['verification_reminders'] == 1
    assert manifest['verification_missing_at_end'] == []
    assert 'Verification: the task requires' in texts(fake.requests[1])[-1]
    assert (workspace / 'out/answer.txt').read_text() == 'Final answer: 42\n'


def test_verification_off_and_second_stop_is_final(tmp_path):
    fake = FakeModel([('text', 'The answer is 42.'), ('text', 'still not writing')])
    try:
        proc, manifest, events, workspace, sandbox = run_harness(tmp_path, fake, ['--planning', 'off', '--context-policy', 'none'])
    finally:
        fake.close()
    assert len(fake.requests) == 1 and manifest['verification_reminders'] == 0 and manifest['expected_files'] == []
    fake = FakeModel([('text', 'The answer is 42.'), ('text', 'still not writing')])
    try:
        proc, manifest, events, workspace, sandbox = run_harness(tmp_path / 'b', fake, ['--planning', 'off', '--context-policy', 'none', '--verification', 'on'])
    finally:
        fake.close()
    assert len(fake.requests) == 2 and manifest['verification_missing_at_end'] == ['out/answer.txt'] and manifest['termination_reason'] == 'normal'


def test_tool_bridge_hides_tools_and_requires_discovery(tmp_path):
    fake = FakeModel([
        ('calls', [('read_file', {'path': 'in/chart.txt'})]),                     # direct call: unknown
        ('calls', [('tool_call', {'name': 'read_file', 'arguments': {'path': 'in/chart.txt'}})]),  # undiscovered
        ('calls', [('tool_search', {'query': 'read file'})]),
        ('calls', [('tool_call', {'name': 'read_file', 'arguments': {'path': 'in/chart.txt'}})]),
        ('calls', [('tool_describe', {'name': 'write_file'})]),
        ('calls', [('tool_call', {'name': 'write_file', 'arguments': '{"path": "out/answer.txt", "content": "Final answer: 42\\n"}'})]),
        ('text', 'Done.'),
    ])
    try:
        proc, manifest, events, workspace, sandbox = run_harness(tmp_path, fake, ['--planning', 'on', '--context-policy', 'none', '--tool-bridge', 'on'])
    finally:
        fake.close()
    assert proc.returncode == 0, proc.stderr
    assert tool_names(fake.requests[0]) == ['tool_call', 'tool_describe', 'tool_search', 'update_plan']
    assert any('# Tool discovery' in t for t in texts(fake.requests[0]))
    msgs = [m['content'] for m in fake.requests[-1]['body']['messages'] if m['role'] == 'tool']
    assert 'unknown tool: read_file' in msgs[0] and 'has not been discovered' in msgs[1]
    assert 'read_file:' in msgs[2] and 'line one' in msgs[3] and '"name": "write_file"' in msgs[4]
    assert (workspace / 'out/answer.txt').read_text() == 'Final answer: 42\n'
    assert manifest['hidden_tool_names'] == ['bash', 'edit_file', 'glob_files', 'grep_text', 'list_files', 'read_file', 'write_file']
    assert manifest['bridge_calls'] == {'tool_search': 1, 'tool_describe': 1, 'tool_call': 3, 'undiscovered': 2}
