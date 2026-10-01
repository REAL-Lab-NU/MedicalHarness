"""Portable CLI, artifact export and deterministic oracle checks (no model/GPU)."""
from __future__ import annotations
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tomllib

import pytest

from medharness.release import HARNESSES, REPO, make_registry, parser


def oracle_module(path):
    spec = importlib.util.spec_from_file_location('release_test_oracle', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def export(bench, destination, *ids, env=None):
    result = subprocess.run([sys.executable, str(REPO / f'scripts/export_{bench}_to_harnessbench.py'),
                             '--output', str(destination), *ids], env=env,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr + result.stdout


def test_calc_export_fixture_and_oracle_without_inspect(tmp_path):
    tasks = tmp_path / 'tasks'
    export('calc', tasks, 'medcalc-001')
    task = tasks / 'medcalc-001'
    with sqlite3.connect(task / 'fixtures/in/patient_facts.sqlite') as database:
        assert database.execute('SELECT COUNT(*) FROM patient_facts').fetchone()[0] > 0
    assert not (task / 'fixtures/gold.jsonl').exists()
    workspace = tmp_path / 'workspace'; (workspace / 'out').mkdir(parents=True)
    oracle = oracle_module(task / 'oracle_grade.py')
    assert oracle.score_workspace(workspace)['outcome_score'] == 0
    (workspace / 'out/answer.json').write_text('{"calculators": []}')
    assert oracle.score_workspace(workspace)['outcome_score'] == 0
    overwrite = subprocess.run([sys.executable, str(REPO / 'scripts/export_calc_to_harnessbench.py'),
                                 '--output', str(tasks), 'medcalc-001'], capture_output=True, text=True)
    assert overwrite.returncode != 0 and 'overwrite' in overwrite.stderr.lower()


def test_private_memory_import_splits_chart_and_keeps_gold_out(tmp_path, monkeypatch):
    private = tmp_path / 'private'; private.mkdir()
    sample = {'id': 'synthetic-only', 'input': 'SESSION 1\nA synthetic value is 42.\nEND OF CHART\nQUESTION: What value?\n',
              'metadata': {'descriptors': {'context_tokens': 25}}}
    (private / 'samples.jsonl').write_text(json.dumps(sample) + '\n')
    (private / 'gold.jsonl').write_text(json.dumps({'id': sample['id'], 'gold': {'accepted_aliases': ['42']}}) + '\n')
    monkeypatch.setenv('MEDMEMORY_DATA_ROOT', str(private))
    tasks = tmp_path / 'tasks'; export('memory', tasks, env=os.environ.copy())
    task = tasks / 'medmem-001'
    contents = list((task / 'fixtures').rglob('*'))
    assert [p.name for p in contents if p.is_file()] == ['session_001.txt']
    assert 'QUESTION:' not in (task / 'fixtures/in/sessions/session_001.txt').read_text()
    workspace = tmp_path / 'workspace'; (workspace / 'out').mkdir(parents=True)
    (workspace / 'out/answer.txt').write_text('Final answer: 42\n')
    assert oracle_module(task / 'oracle_grade.py').score_workspace(workspace)['outcome_score'] == 1.0


def test_healthadmin_export_uses_requested_endpoints(tmp_path, monkeypatch):
    monkeypatch.setenv('HEALTHADMIN_PORTAL_BASE_URL', 'http://127.0.0.1:13002')
    monkeypatch.setenv('HB_CDP_URL', 'http://127.0.0.1:19222')
    tasks = tmp_path / 'tasks'; export('healthadmin', tasks, 'medadmin-016', env=os.environ.copy())
    task = tasks / 'medadmin-016'
    assert 'http://127.0.0.1:13002' in (task / 'prompt.txt').read_text()
    assert 'http://127.0.0.1:19222' in (task / 'hooks.py').read_text()
    for name in ('hooks.py', 'oracle_grade.py'):
        compile((task / name).read_text(), str(task / name), 'exec')


@pytest.mark.parametrize('harness', HARNESSES)
@pytest.mark.parametrize('tool_mode', ['files', 'browser'])
def test_rendered_product_profiles_are_parseable_and_use_requested_model(tmp_path, harness, tool_mode):
    args = parser().parse_args(['run-one', '--task', 'synthetic-1', '--model', 'local-test-model',
        '--upstream', 'http://127.0.0.1:18000/v1', '--output', str(tmp_path / 'out'),
        '--harness', harness, '--tool-mode', tool_mode])
    registry = make_registry(args, tmp_path / 'config')
    entry = registry['models'][harness]
    assert entry['timeout_sec'] == 3600 and entry['model'] == 'local-test-model'
    if 'user_config' in entry:
        path = Path(entry['user_config']); text = path.read_text()
        assert 'local-test-model' in text and '127.0.0.1:18000/v1' in text
        assert '__VLLM_PORT__' not in text
        if path.suffix == '.toml':
            tomllib.loads(text)
        elif path.suffix == '.json':
            json.loads(text)
        else:
            import yaml
            yaml.safe_load(text)
    else:
        # generic_cli applies str.format to each argument before launching.
        for value in entry['args']:
            value.format(workspace='/tmp/ws with spaces', prompt_file='/tmp/prompt.txt')
        assert 'http://127.0.0.1:18000/v1' in entry['args']
        if tool_mode == 'browser':
            assert '--mcp-config' in entry['args']
