#!/usr/bin/env python
"""Export MedPlanning tasks into harness-bench format.

Docker Compose materializes the upstream agent workspace as public fixtures.
Gold references and verifier files are kept outside that workspace. After the
episode, the oracle runs the upstream verifier inside its task image against
the submitted files. Required images and data assets must be prepared first.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from medharness.data import registry as R  # noqa: E402

HAB = Path(os.environ.get("HEALTHAGENTBENCH_ROOT", str(REPO / "vendor/HealthAgentBench"))).resolve()
COMPOSE_DIR = REPO / "medharness" / "data" / "healthagent" / "fixtures" / "compose"
OUT_ROOT = Path(os.environ.get("MEDHARNESS_TASKS_ROOT", str(REPO / "tasks")))
PREFIX = "medagent"
# Extract inputs mounted outside /workspace once to a shared read-only root.
EXTERNAL_ROOT = Path(os.environ.get("HEALTHAGENT_EXTERNAL_ROOT", str(REPO / "assets/healthagent-external"))).resolve()
# Cache CAMELYON16 verifier masks once and mount them at the upstream cache path.
VERIFIER_CACHES = {
    "tumor_area_selection_pathology": (
        Path(os.environ.get("HEALTHAGENT_VERIFIER_CACHE", str(REPO / "assets/tumor-verifier-cache"))).resolve(),
        "/tmp/tumor_area_selection_pathology_verifier",
    ),
}

ORACLE = '''from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

IMAGE = "{image}"
TESTS_DIR = Path("{tests_dir}")
# Inputs staged outside the workspace, bound back at the container paths upstream's verifier expects.
EXTERNAL_MOUNTS = {external_mounts}
TASK_ID = "{task_id}"
UPSTREAM_ID = "{upstream_id}"


def score_workspace(workspace: Path) -> dict[str, Any]:
    """Run upstream's own tests/test.sh inside the task image against the finished workspace."""
    if not shutil.which("docker"):
        return _fail("docker unavailable")

    # The verifier writes into the mounted /logs and may touch /workspace. Run it as the calling user
    # so nothing it leaves behind is root-owned and undeletable by the host afterwards.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        logs = Path(tmp) / "logs"
        logs.mkdir()
        cmd = [
            "docker", "run", "--rm", "--network", "none",
            "--user", f"{{os.getuid()}}:{{os.getgid()}}",
            "-v", f"{{workspace}}:/workspace:rw",
            "-v", f"{{TESTS_DIR}}:/tests:ro",
            "-v", f"{{logs}}:/logs:rw",
        ]
        for container_path, host_path in EXTERNAL_MOUNTS.items():
            cmd += ["-v", f"{{host_path}}:{{container_path}}:ro"]
        cmd += [
            "-w", "/workspace", IMAGE,
            "bash", "/tests/test.sh",
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        except subprocess.TimeoutExpired:
            return _fail("verifier timed out")

        reward_file = logs / "verifier" / "reward.txt"
        if not reward_file.is_file():
            tail = (proc.stderr or proc.stdout or "")[-400:]
            return _fail(f"verifier wrote no reward (exit {{proc.returncode}}): {{tail}}")
        try:
            reward = float(reward_file.read_text().strip())
        except ValueError:
            return _fail("reward.txt is not a number")

        metrics: dict[str, Any] = {{"reward": reward, "submission_failure": None}}
        mfile = logs / "verifier" / "metrics.json"
        if mfile.is_file():
            try:
                extra = json.loads(mfile.read_text())
                if isinstance(extra, dict):
                    metrics.update({{k: v for k, v in extra.items() if not isinstance(v, (dict, list))}})
            except ValueError:
                pass

        return {{
            "task": TASK_ID,
            "workspace": str(workspace),
            "checks": [{{"id": "reward", "label": "upstream verifier reward",
                        "pass": reward >= 1.0, "weight": 1.0, "detail": f"reward={{reward}}"}}],
            "outcome_score": reward,
            "metrics": metrics,
            "outcome_llm_weight": 0.0,
        }}


def _fail(detail: str) -> dict[str, Any]:
    return {{
        "task": TASK_ID,
        "workspace": "",
        "checks": [{{"id": "reward", "label": "upstream verifier reward",
                    "pass": False, "weight": 1.0, "detail": detail}}],
        "outcome_score": 0.0,
        "metrics": {{"reward": 0.0, "submission_failure": detail}},
        "outcome_llm_weight": 0.0,
    }}
'''

PROMPT_TAIL = """

---

Your working directory is `$WORKSPACE`. Every path above that begins with `$WORKSPACE` is a real path
there; read and write them directly.{external}

Take as many steps as the work needs; there is no turn budget. You have no network access -- everything
you need is already on disk.
"""

EXTERNAL_NOTE = """ The absolute paths above that do not begin with `$WORKSPACE` are read-only inputs
staged outside your working directory; read them in place and do not try to copy or modify them."""


def _image_for(task: str) -> str:
    compose = (COMPOSE_DIR / f"{task}.yaml").read_text()
    for line in compose.splitlines():
        stripped = line.strip()
        if stripped.startswith("image:"):
            return stripped.split(":", 1)[1].strip()
    raise ValueError(f"{task}: no image in compose")


def _external_mounts(compose: Path) -> list[str]:
    """Return named-volume mounts outside ``/workspace`` for the main service.

    Large shared inputs are extracted once and referenced by their host paths.
    """
    out: list[str] = []
    in_main = in_volumes = False
    for raw in compose.read_text().splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip())
        if stripped.startswith("- "):
            if in_main and in_volumes:
                parts = stripped[2:].split(":")
                # named volume (not a host path or a variable), mounted somewhere outside /workspace
                if len(parts) >= 2 and not parts[0].startswith(("/", "$", ".")):
                    if not parts[1].startswith("/workspace"):
                        out.append(parts[1])
            continue
        if indent == 2 and stripped.endswith(":"):
            in_main = stripped == "main:"
            in_volumes = False
            continue
        if in_main and indent == 4:
            in_volumes = stripped == "volumes:"
    return out


def _materialise(task: str, dest: Path, external_dest: Path) -> dict[str, str]:
    """Bring the upstream environment up and copy what the agent container sees.

    Returns {container_path: host_path} for inputs mounted outside /workspace, so the prompt can be
    rewritten to point at where they actually live on this machine.
    """
    compose = COMPOSE_DIR / f"{task}.yaml"
    project = f"hab-export-{task}".replace("_", "-")[:60]
    env = {
        **os.environ,
        "SAMPLE_METADATA_TASK_DIR": str(HAB / "tasks" / task),
        "SAMPLE_METADATA_HAB_ROOT": str(HAB),
    }
    base = ["docker", "compose", "-f", str(compose), "-p", project]
    subprocess.run([*base, "down", "-v"], env=env, capture_output=True, timeout=300)
    up = subprocess.run([*base, "up", "--wait"], env=env, capture_output=True, text=True, timeout=1800)
    try:
        if up.returncode != 0:
            raise RuntimeError(f"compose up failed: {(up.stderr or '')[-300:]}")
        ps = subprocess.run([*base, "ps", "--format", "{{.Name}}"], env=env,
                            capture_output=True, text=True, timeout=120)
        container = next((n for n in ps.stdout.split() if n.endswith("-main-1")), None)
        if container is None:
            raise RuntimeError(f"no main container among {ps.stdout.split()}")
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(["docker", "cp", f"{container}:/workspace", tmp],
                           check=True, capture_output=True, timeout=900)
            src = Path(tmp) / "workspace"
            shutil.rmtree(src / ".harbor", ignore_errors=True)
            dest.mkdir(parents=True, exist_ok=True)
            for child in src.iterdir():
                target = dest / child.name
                if child.is_dir():
                    shutil.copytree(child, target, dirs_exist_ok=True)
                else:
                    shutil.copy2(child, target)

        mapping: dict[str, str] = {}
        for cpath in _external_mounts(compose):
            host = external_dest / cpath.lstrip("/")
            if not host.is_dir():
                host.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory() as tmp:
                    subprocess.run(["docker", "cp", f"{container}:{cpath}", tmp],
                                   check=True, capture_output=True, timeout=3600)
                    got = next(Path(tmp).iterdir())
                    shutil.move(str(got), str(host))
            mapping[cpath] = str(host)
        return mapping
    finally:
        subprocess.run([*base, "down", "-v"], env=env, capture_output=True, timeout=300)


def main() -> int:
    global OUT_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task_ids', nargs='*', help='Optional exported/upstream task IDs')
    parser.add_argument('--output', type=Path, default=OUT_ROOT)
    parser.add_argument('--overwrite', action='store_true', help='Replace only selected existing task directories')
    args = parser.parse_args()
    OUT_ROOT = args.output.resolve()
    only = args.task_ids or None
    samples = R.load_records("healthagent", check=False, subset=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    written, failed = [], []

    for i, sample in enumerate(samples, 1):
        upstream = str(sample.id)
        task_id = f"{PREFIX}-{i:03d}"
        if only and upstream not in only and task_id not in only:
            continue
        if only and str(sample.id) not in only and task_id not in only:
            continue
        task_dir = OUT_ROOT / task_id
        if task_dir.exists():
            if not args.overwrite:
                raise FileExistsError(f"Refusing to overwrite {task_dir}; use --overwrite")
            shutil.rmtree(task_dir)
        fixtures = task_dir / "fixtures"
        try:
            mapping = _materialise(upstream, fixtures, EXTERNAL_ROOT / upstream)
        except Exception as exc:  # noqa: BLE001 - report the task error and continue exporting the remaining tasks
            shutil.rmtree(task_dir, ignore_errors=True)
            failed.append((task_id, upstream, f"{type(exc).__name__}: {exc}"))
            print(f"{task_id} ({upstream}): FAILED {exc}")
            continue

        instruction = (HAB / "tasks" / upstream / "instruction.md").read_text(encoding="utf-8")
        body = instruction.replace("/workspace", "$WORKSPACE")
        for cpath, hpath in sorted(mapping.items(), key=lambda kv: -len(kv[0])):
            body = body.replace(cpath, hpath)
        (task_dir / "prompt.txt").write_text(
            body + PROMPT_TAIL.format(external=EXTERNAL_NOTE if mapping else ""), encoding="utf-8")
        mounts = dict(mapping)
        for family, (host, container) in VERIFIER_CACHES.items():
            if upstream.startswith(family) and host.is_dir():
                mounts[container] = str(host)
        (task_dir / "oracle_grade.py").write_text(ORACLE.format(
            image=_image_for(upstream), tests_dir=HAB / "tasks" / upstream / "tests",
            task_id=task_id, upstream_id=upstream, external_mounts=repr(mounts)), encoding="utf-8")
        (task_dir / "task.yaml").write_text(json.dumps({
            "task_id": task_id,
            "title": f"HealthAgentBench {upstream}",
            "class": "File-artifact task in a prepared workspace, graded by upstream's verifier",
            "prompt_file": "prompt.txt",
            "fixtures_dir": "fixtures",
            "oracle_module": "oracle_grade.py",
            "timeout_sec": 3600,
            "tags": ["medical", "healthagent", "file-artifact", "medharness"],
        }, indent=2), encoding="utf-8")
        size = sum(p.stat().st_size for p in fixtures.rglob("*") if p.is_file())
        written.append((task_id, upstream, size))
        print(f"{task_id} ({upstream}): {size/1e6:.1f} MB")

    print(f"\nexported {len(written)} tasks to {OUT_ROOT}")
    if written:
        print(f"  fixtures: {min(w[2] for w in written)/1e6:.1f} to {max(w[2] for w in written)/1e6:.1f} MB"
              f"  (total {sum(w[2] for w in written)/1e6:.0f} MB)")
    if failed:
        print(f"  FAILED {len(failed)}:")
        for t, u, e in failed:
            print(f"    {t} ({u}): {e}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
