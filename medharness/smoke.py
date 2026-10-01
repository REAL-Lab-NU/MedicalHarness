"""Deterministic end-to-end wiring smoke: no real model or medical data."""
from __future__ import annotations
import json
from pathlib import Path
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from medharness.release import dump, parser, run_one


def smoke(options):
    root = options.output.resolve() if options.output else Path(tempfile.mkdtemp(prefix='medicalharness-smoke-'))
    root.mkdir(parents=True, exist_ok=True)
    task = root / 'tasks' / 'smoke-001'
    if task.exists() or (root / 'run').exists():
        raise FileExistsError(f'Smoke output already contains a task or run: {root}')
    (task / 'fixtures/in').mkdir(parents=True)
    (task / 'fixtures/in/fact.txt').write_text('Synthetic smoke fact: 42.\n')
    (task / 'prompt.txt').write_text('Read `$WORKSPACE/in/fact.txt`, then write Final answer: 42 to `$WORKSPACE/out/answer.txt`.')
    (task / 'oracle_grade.py').write_text('''from pathlib import Path

def score_workspace(workspace):
    answer = Path(workspace) / 'out/answer.txt'
    ok = answer.is_file() and answer.read_text().strip() == 'Final answer: 42'
    return {'outcome_score': float(ok), 'checks': [{'id': 'answer', 'pass': ok, 'weight': 1.0}], 'metrics': {'exact': int(ok)}}
''')
    dump(task / 'task.yaml', {'task_id': 'smoke-001', 'title': 'Synthetic wiring smoke', 'timeout_sec': 60,
        'prompt_file': 'prompt.txt', 'fixtures_dir': 'fixtures', 'oracle_module': 'oracle_grade.py'})
    requests = []
    class Canned(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            requests.append(body)
            step = len([m for m in body['messages'] if m['role'] == 'tool'])
            message = {'role': 'assistant', 'content': 'Synthetic task completed.'}
            finish = 'stop'
            actions = [('read_file', {'path': 'in/fact.txt'}),
                       ('write_file', {'path': 'out/answer.txt', 'content': 'Final answer: 42\n'})]
            if step < len(actions):
                name, args = actions[step]
                message['tool_calls'] = [{'id': f'smoke-{step}', 'type': 'function',
                    'function': {'name': name, 'arguments': json.dumps(args)}}]
                finish = 'tool_calls'
            payload = {'id': f'canned-{len(requests)}', 'object': 'chat.completion', 'created': 1,
                'model': body['model'], 'choices': [{'index': 0, 'message': message, 'finish_reason': finish}],
                'usage': {'prompt_tokens': 50, 'completion_tokens': 10, 'total_tokens': 60}}
            raw = json.dumps(payload).encode(); self.send_response(200)
            self.send_header('Content-Type', 'application/json'); self.send_header('Content-Length', str(len(raw)))
            self.end_headers(); self.wfile.write(raw)
    server = ThreadingHTTPServer(('127.0.0.1', 0), Canned)
    worker = threading.Thread(target=server.serve_forever, daemon=True); worker.start()
    try:
        args = parser().parse_args(['run-one', '--hb', str(options.hb), '--tasks', str(root / 'tasks'),
            '--task', 'smoke-001', '--harness', 'mhlab', '--model', 'canned-smoke',
            '--upstream', f'http://127.0.0.1:{server.server_port}/v1', '--output', str(root / 'run'),
            '--timeout', '60', '--max-tokens', '512', '--context-window', '8192',
            '--planning', 'off', '--context-policy', 'none', '--no-isolation'])
        code = run_one(args)
    finally:
        server.shutdown(); server.server_close(); worker.join(timeout=5)
    results = list((root / 'run/results').glob('*/*/smoke-001.json'))
    if code or len(results) != 1:
        raise RuntimeError(f'Smoke runner failed; inspect {root / "run/runner.log"}')
    result = json.loads(results[0].read_text())
    if not result['adapter_result']['ok'] or result['oracle_result']['outcome_score'] != 1.0:
        raise RuntimeError(f'Smoke did not submit the expected artifact; inspect {results[0]}')
    if len(requests) != 3 or any(r.get('seed') != 42 or r.get('max_completion_tokens', r.get('max_tokens')) != 512 for r in requests):
        raise RuntimeError('Model request count/seed/output budget differs from the smoke contract')
    if result['usage_summary'].get('request_count') != 3:
        raise RuntimeError('Usage proxy did not account for all smoke requests')
    print(json.dumps({'smoke': 'passed', 'model_inference': False, 'oracle_score': 1.0,
        'model_requests': len(requests), 'output': str(root), 'result': str(results[0]),
        'isolation_tested': False, 'note': 'Synthetic transport/tool/oracle test; not a benchmark result.'}, indent=2))
    return 0
