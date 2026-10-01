"""Portable entrypoints for a single product-harness or MH-Lab episode."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import urlsplit

REPO = Path(__file__).resolve().parents[1]
HB_COMMIT = '1025086a446653702b80cfb48babbeec35db6b2c'
HB_REMOTE = 'https://github.com/Qihoo360/harness-bench.git'
HARNESSES = ('mhlab', 'openclaw', 'hermes', 'zeroclaw', 'codex', 'claudecode')


def hb_path():
    return Path(os.environ.get('HARNESSBENCH_ROOT', str(REPO / 'vendor/harness-bench'))).resolve()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def bootstrap(destination, source=HB_REMOTE):
    """Create a fresh pinned checkout; never modify an existing upstream tree."""
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f'{destination} exists; choose an empty --destination')
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['git', 'clone', '--no-checkout', source, str(destination)], check=True)
    subprocess.run(['git', '-C', str(destination), 'checkout', '--detach', HB_COMMIT], check=True)
    patches = sorted((REPO / 'patches/harness-bench').glob('*.patch'))
    for patch in patches:
        subprocess.run(['git', '-C', str(destination), 'apply', '--check', str(patch)], check=True)
        subprocess.run(['git', '-C', str(destination), 'apply', str(patch)], check=True)
    shutil.copy2(REPO / 'patches/harness-bench/moltis_acp.py', destination / 'src/harnessbench/adapters/moltis_acp.py')
    dump(destination / 'medicalharness-bootstrap.json', {
        'upstream': HB_REMOTE, 'commit': HB_COMMIT,
        'patches': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in patches},
        'note': 'Moltis adapter is retained only because the historical patch registry imports it; not a supported release harness.',
    })
    return destination


def doctor(args):
    packages = {}
    for name in ('medharness', 'langgraph', 'langchain-core', 'langchain-openai', 'langchain-mcp-adapters', 'PyYAML'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    report = {'python': sys.version.split()[0], 'packages': packages,
              'harnessbench': {'path': str(args.hb), 'ready': (args.hb / 'medicalharness-bootstrap.json').is_file()},
              'executables': {name: shutil.which(name) for name in ('git', 'bwrap', 'node', 'docker', 'openclaw', 'hermes', 'zeroclaw', 'codex', 'claude')},
              'browser': {'mcp_cli': os.environ.get('PLAYWRIGHT_MCP_CLI', str(REPO / 'node_modules/@playwright/mcp/cli.js')),
                          'cdp': os.environ.get('HB_CDP_URL'), 'portal': os.environ.get('HEALTHADMIN_PORTAL_BASE_URL')},
              'data': {'tasks': str(args.tasks), 'task_count': len(list(args.tasks.glob('*/task.yaml'))),
                       'restricted_memory_root': os.environ.get('MEDMEMORY_DATA_ROOT')},
              'network_checked': False,
              'note': 'Paths and installed packages only. This command does not start services or query remote endpoints.'}
    print(json.dumps(report, indent=2))
    return 0 if all(packages.values()) else 1


def browser_connection(args):
    if args.mcp_config:
        return json.loads(args.mcp_config.read_text())
    return {'mcpServers': {'playwright': {'command': sys.executable,
        'args': [str(REPO / 'scripts/mcp_browser_no_js.py'), '--cdp-endpoint', args.cdp_url]}}}


def make_registry(args, directory):
    """Render only the selected harness's isolated, credential-free profile."""
    directory.mkdir(parents=True, exist_ok=True)
    model = args.model
    entry = {'model': model, 'timeout_sec': args.timeout, 'use_usage_proxy': True,
             'session_prefix': 'mh-' + args.harness}
    browser = args.tool_mode == 'browser'
    mcp = browser_connection(args) if browser else None
    if args.harness in ('mhlab', 'claudecode'):
        script = 'run_langgraph_harness.py' if args.harness == 'mhlab' else 'run_claude_code.py'
        cli = [str(REPO / 'scripts/harness' / script), '--workspace', '{workspace}', '--prompt-file', '{prompt_file}',
               '--model', model, '--upstream', args.upstream, '--max-tokens', str(args.max_tokens),
               '--context-window', str(args.context_window), '--timeout-seconds', str(args.timeout)]
        if args.harness == 'mhlab':
            cli += ['--planning', args.planning, '--context-policy', args.context_policy,
                    '--verification', args.verification, '--tool-bridge', args.tool_bridge,
                    '--tool-mode', args.tool_mode, '--seed', str(args.seed)]
        elif args.command:
            cli += ['--command', args.command]
        if browser:
            config = directory / 'mcp.json'
            dump(config, mcp)
            cli += ['--mcp-config', str(config)]
        entry.update(adapter='generic_cli', command=sys.executable, args=cli)
    elif args.harness == 'hermes':
        import yaml
        config = yaml.safe_load((REPO / 'configs/harness/hermes/hermes.yaml').read_text())
        config['model'].update(base_url=args.upstream, default=model, max_tokens=args.max_tokens)
        config['providers']['vllm'].update(base_url=args.upstream, default_model=model, models=[model],
                                          context_length=args.context_window, max_output_tokens=args.max_tokens)
        if browser:
            config['mcp_servers'] = mcp['mcpServers']
            config['agent']['disabled_toolsets'] = ['browser']
        path = directory / 'hermes.yaml'
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        entry.update(adapter='hermes_agent', command=args.command or 'hermes', args=['--yolo', '--in', '{workspace}', '-z'], user_config=str(path))
    elif args.harness == 'openclaw':
        config = json.loads((REPO / 'configs/harness/openclaw/openclaw.json').read_text())
        config.pop('_comment', None)
        provider = config['models']['providers']['vllm']
        provider['baseUrl'] = args.upstream
        provider['models'] = [{'id': model, 'name': model, 'contextWindow': args.context_window, 'maxTokens': args.max_tokens}]
        config['browser'].update(enabled=browser, cdpUrl=args.cdp_url)
        config.setdefault('agents', {}).setdefault('defaults', {})['timeoutSeconds'] = args.timeout
        path = directory / 'openclaw.json'
        dump(path, config)
        entry.update(adapter='openclaw', command=args.command or 'openclaw', user_config=str(path),
                     args=['agent', '--local', '--model', 'vllm/' + model])
    elif args.harness == 'codex':
        text = '\n'.join([f'model = {json.dumps(model)}', 'model_provider = "vllm"', 'approval_policy = "never"',
            f'model_context_window = {args.context_window}', '[model_providers.vllm]', 'name = "Local OpenAI-compatible service"',
            f'base_url = {json.dumps(args.upstream)}', 'wire_api = "responses"', 'env_key = "VLLM_API_KEY"',
            'stream_idle_timeout_ms = 1800000', 'request_max_retries = 2']) + '\n'
        if browser:
            for name, server in mcp['mcpServers'].items():
                text += f'\n[mcp_servers.{name}]\ncommand = {json.dumps(server["command"])}\nargs = {json.dumps(server["args"])}\n'
                text += 'startup_timeout_sec = 60\ntool_timeout_sec = 300\ndefault_tools_approval_mode = "approve"\n'
        path = directory / 'codex.toml'; path.write_text(text)
        entry.update(adapter='codex', command=args.command or 'codex', user_config=str(path), sandbox='workspace-write')
    else:
        text = (REPO / 'configs/harness/zeroclaw/zeroclaw.toml').read_text()
        text = text.replace('http://127.0.0.1:__VLLM_PORT__/v1', args.upstream).replace('qwen36-35b-a3b-260k', model)
        text = text.replace('context_window = 262144', f'context_window = {args.context_window}')
        if browser:
            text = text.replace('[agents.main]\n', '[agents.main]\nmcp_bundles = ["browser"]\n')
            text = text.replace('require_approval_for_medium_risk = true', 'require_approval_for_medium_risk = false')
            text = text.replace('auto_approve = [', 'auto_approve = ["mcp_tool", ')
            text += '\n[mcp]\nenabled = true\ndeferred_loading = false\n'
            names = []
            for name, server in mcp['mcpServers'].items():
                names.append(name)
                text += '\n[[mcp.servers]]\nname = ' + json.dumps(name) + '\ntransport = "stdio"\n'
                text += 'command = ' + json.dumps(server['command']) + '\nargs = ' + json.dumps(server['args']) + '\n'
                text += 'tool_timeout_secs = 300\n'
            text += '\n[mcp_bundles.browser]\nservers = ' + json.dumps(names) + '\n'
        path = directory / 'zeroclaw.toml'; path.write_text(text)
        entry.update(adapter='zeroclaw', command=args.command or 'zeroclaw', user_config=str(path), extra_args=['-a', 'main'])
    return {'models': {args.harness: entry}}


def run_one(args):
    args.hb = args.hb.resolve(); args.tasks = args.tasks.resolve(); args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError(f'Refusing to reuse {args.output}; each episode needs a fresh --output')
    if not (args.hb / 'medicalharness-bootstrap.json').is_file():
        raise FileNotFoundError(f'Run medicalharness bootstrap first, or point --hb at a bootstrapped checkout: {args.hb}')
    task_file = args.tasks / args.task / 'task.yaml'
    if not task_file.is_file():
        raise FileNotFoundError(f'Missing {task_file}; export or supply the task package first')
    if not 0 < args.max_tokens < args.context_window:
        raise ValueError('Require 0 < --max-tokens < --context-window')
    if args.timeout <= 0:
        raise ValueError('--timeout must be positive')
    if args.tool_mode == 'browser' and not args.cdp_url:
        raise ValueError('Browser mode requires --cdp-url or HB_CDP_URL')
    config_dir = args.output / 'config'
    registry = json.loads(args.registry.read_text()) if args.registry else make_registry(args, config_dir)
    if args.harness not in registry.get('models', {}):
        raise ValueError(f'--registry has no models.{args.harness}')
    registry['models'][args.harness]['timeout_sec'] = args.timeout
    dump(config_dir / 'harness.json', registry)
    dump(config_dir / 'app.json', {'data_dir': str(args.output / 'data'), 'tasks_dir': str(args.tasks),
        'results_dir': str(args.output / 'results'), 'work_root': str(args.output / 'sandbox'),
        'default_timeout_sec': args.timeout, 'default_rounds': 1})
    env = dict(os.environ, MEDHARNESS_REPO=str(REPO), HARNESSBENCH_APP_CONFIG=str(config_dir / 'app.json'),
               HARNESSBENCH_HARNESS_CONFIG=str(config_dir / 'harness.json'),
               PYTHONPATH=str(args.hb / 'src') + os.pathsep + str(REPO),
               HB_PIN_SEED=str(args.seed), HB_PIN_TEMPERATURE=str(args.temperature), HB_PIN_TOP_P='0.95',
               HB_MAX_OUTPUT_TOKENS=str(args.max_tokens), HB_PROXY_UPSTREAM_TIMEOUT_SECONDS=str(args.timeout),
               HB_AGENT_ISOLATION='0' if args.no_isolation else '1', HARNESSBENCH_SKIP_PROCESS_GRADE='1')
    env.setdefault('VLLM_API_KEY', 'not-needed')
    env.setdefault('OPENAI_API_KEY', 'not-needed')
    credential = os.environ.get('MEDHARNESS_UPSTREAM_API_KEY') or os.environ.get('VLLM_API_KEY') or os.environ.get('OPENAI_API_KEY')
    if credential:
        env['MEDHARNESS_UPSTREAM_API_KEY'] = credential
    env['HB_AGENT_ISOLATION_HIDE_PATHS'] = json.dumps([str(args.output), str(args.tasks), str(REPO)])
    allow = [str(config_dir), str(REPO / 'scripts/harness'), str(REPO / 'scripts/mcp_browser_no_js.py')]
    allow.extend(str(path.resolve(strict=True)) for path in args.allow_path)
    if Path(sys.prefix).is_relative_to(REPO):
        allow.append(sys.prefix)
    env['HB_AGENT_ISOLATION_ALLOW_PATHS'] = json.dumps(allow)
    if args.cdp_url:
        env['HB_CDP_URL'] = args.cdp_url
    command = [sys.executable, '-m', 'harnessbench.cli', 'run-task', '--task', args.task, '--harness', args.harness]
    manifest = {'harness': args.harness, 'task': args.task, 'model': args.model, 'seed': args.seed,
        'timeout_seconds': args.timeout, 'max_tokens': args.max_tokens, 'context_window': args.context_window,
        'isolation': not args.no_isolation, 'command': command, 'dry_run': args.dry_run,
        'harnessbench_commit': HB_COMMIT, 'process_llm_judge': 'disabled; deterministic task oracle only',
        'note': 'Portable single-episode entrypoint; generated configs are not a claim of exact historical paper settings.'}
    dump(args.output / 'run.json', manifest)
    if args.dry_run:
        print(json.dumps(manifest, indent=2)); return 0
    if not args.no_isolation and not shutil.which('bwrap'):
        raise RuntimeError('bubblewrap (bwrap) required. --no-isolation is for trusted synthetic smoke only.')
    with (args.output / 'runner.log').open('w') as log:
        completed = subprocess.run(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   timeout=args.timeout + args.grading_timeout, check=False)
    paths = sorted((args.output / 'results').glob('*/*/*.json'))
    print(json.dumps({'returncode': completed.returncode, 'results': [str(p) for p in paths],
                      'runner_log': str(args.output / 'runner.log')}, indent=2))
    return completed.returncode


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    b = sub.add_parser('bootstrap', help='Clone one pinned harness-bench checkout and apply local patches')
    b.add_argument('--destination', type=Path, default=hb_path()); b.add_argument('--source', default=HB_REMOTE)
    d = sub.add_parser('doctor', help='Read-only local prerequisites report; does not start services')
    d.add_argument('--hb', type=Path, default=hb_path()); d.add_argument('--tasks', type=Path, default=REPO / 'tasks')
    r = sub.add_parser('run-one', help='Run one exported task; use --dry-run to prepare only')
    r.add_argument('--hb', type=Path, default=hb_path()); r.add_argument('--tasks', type=Path, default=REPO / 'tasks')
    r.add_argument('--task', required=True); r.add_argument('--harness', choices=HARNESSES, default='mhlab')
    r.add_argument('--model', required=True); r.add_argument('--upstream', required=True)
    r.add_argument('--output', type=Path, required=True); r.add_argument('--registry', type=Path)
    r.add_argument('--command', help='Installed product CLI path (optional)')
    r.add_argument('--allow-path', action='append', type=Path, default=[], help='Explicit read-only public input/runtime directory inside an otherwise hidden tree')
    r.add_argument('--timeout', type=int, default=3600); r.add_argument('--grading-timeout', type=int, default=1800)
    r.add_argument('--max-tokens', type=int, default=16384); r.add_argument('--context-window', type=int, default=81920)
    r.add_argument('--seed', type=int, default=42); r.add_argument('--temperature', type=float, default=0.8)
    r.add_argument('--tool-mode', choices=('files', 'browser'), default='files')
    r.add_argument('--mcp-config', type=Path); r.add_argument('--cdp-url', default=os.environ.get('HB_CDP_URL', 'http://127.0.0.1:9222'))
    r.add_argument('--planning', choices=('on', 'off'), default='on')
    r.add_argument('--context-policy', choices=('none', 'elide', 'elide_recall', 'summarize', 'elide_then_summarize'), default='summarize')
    r.add_argument('--verification', choices=('on', 'off'), default='off'); r.add_argument('--tool-bridge', choices=('on', 'off'), default='off')
    r.add_argument('--no-isolation', action='store_true', help='Trusted synthetic fixtures only; exposes host files to tools')
    r.add_argument('--dry-run', action='store_true')
    s = sub.add_parser('smoke', help='Canned localhost provider -> MH-Lab -> proxy -> file -> deterministic oracle (no GPU)')
    s.add_argument('--hb', type=Path, default=hb_path()); s.add_argument('--output', type=Path)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.action == 'bootstrap':
            print(bootstrap(args.destination, args.source)); return 0
        if args.action == 'doctor':
            return doctor(args)
        if args.action == 'smoke':
            from medharness.smoke import smoke
            return smoke(args)
        return run_one(args)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr); return 1


if __name__ == '__main__':
    raise SystemExit(main())
