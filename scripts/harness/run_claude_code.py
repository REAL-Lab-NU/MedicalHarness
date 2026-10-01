#!/usr/bin/env python3
"""Claude Code generic_cli bridge to an Anthropic-compatible local model service."""
import argparse
import json
import os
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, required=True); p.add_argument('--prompt-file', type=Path, required=True)
    p.add_argument('--model', required=True); p.add_argument('--upstream', required=True)
    p.add_argument('--command', default='claude'); p.add_argument('--mcp-config', type=Path)
    p.add_argument('--max-tokens', type=int, default=16384); p.add_argument('--context-window', type=int, default=81920)
    p.add_argument('--timeout-seconds', type=int, default=3600)
    a = p.parse_args()
    proxy = os.environ['HARNESSBENCH_LLM_PROXY_URL'].rstrip('/')
    routes = Path(os.environ['HARNESSBENCH_LLM_PROXY_ROUTES'])
    mapping = json.loads(routes.read_text()) if routes.exists() else {}
    upstream = a.upstream.rstrip('/')
    if upstream.endswith('/v1'):
        upstream = upstream[:-3]
    mapping['/claudecode/vllm'] = dict(framework='claude-code', provider='vllm', upstream=upstream)
    routes.parent.mkdir(parents=True, exist_ok=True); routes.write_text(json.dumps(mapping))
    home = Path(os.environ['HARNESSBENCH_SANDBOX']) / 'claude-home'; home.mkdir(exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith(('ANTHROPIC_', 'CLAUDE_', 'OPENAI_'))}
    env.update(HOME=str(home), ANTHROPIC_BASE_URL=proxy + '/claudecode/vllm', ANTHROPIC_API_KEY='not-needed',
        ANTHROPIC_MODEL=a.model, CLAUDE_CODE_MAX_CONTEXT_TOKENS=str(a.context_window),
        CLAUDE_CODE_MAX_OUTPUT_TOKENS=str(a.max_tokens), API_TIMEOUT_MS=str(a.timeout_seconds * 1000),
        CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC='1')
    cmd = [a.command, '-p', a.prompt_file.read_text(), '--permission-mode', 'bypassPermissions', '--max-turns', '200']
    if a.mcp_config:
        cmd += ['--mcp-config', str(a.mcp_config)]
    os.chdir(a.workspace)
    os.execvpe(a.command, cmd, env)


if __name__ == '__main__':
    main()
