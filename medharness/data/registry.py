"""Shared schema and loaders for the four benchmark task packages.

``samples.jsonl`` holds model-visible task records. ``gold.jsonl`` holds
scorer and environment references. Exporters keep gold outside the model
workspace, and scorers retrieve it by sample ID.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
RESERVED = ("bench", "axis", "version", "upstream", "target_kind", "limits", "env", "floor", "descriptors")
AXES = {"healthadmin": "interaction_recovery", "calc": "tool_use", "healthagent": "file_artifact_handling", "memory": "long_term_memory"}
TARGET_KINDS = {"healthadmin": "outcome_checks_descriptive", "calc": "multi_output_json", "healthagent": "reward_scalar", "memory": "answer_aliases"}
#: External task packages use an environment-variable override or the default data directory.
#: Tracked manifests record the hashes of the separately supplied samples and gold.
DATA_ROOT_ENV = {"memory": ("MEDMEMORY_DATA_ROOT", ROOT / "memory")}


@dataclass(frozen=True)
class BenchSpec:
    name: str
    axis: str
    package_dir: Path
    samples_path: Path
    gold_path: Path
    manifest_path: Path
    committed: bool


def spec(bench: str) -> BenchSpec:
    if bench not in AXES:
        raise KeyError(f"unknown bench {bench!r}; known: {sorted(AXES)}")
    pkg = ROOT / bench
    if bench in DATA_ROOT_ENV:
        env_var, default = DATA_ROOT_ENV[bench]
        data = Path(os.environ.get(env_var, default))
        return BenchSpec(bench, AXES[bench], pkg, data / "samples.jsonl", data / "gold.jsonl", pkg / "manifest.json", False)
    return BenchSpec(bench, AXES[bench], pkg, pkg / "samples.jsonl", pkg / "gold.jsonl", pkg / "manifest.json", True)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest(bench: str) -> dict[str, Any]:
    return json.loads(spec(bench).manifest_path.read_text(encoding="utf-8"))


def verify(bench: str) -> dict[str, Any]:
    """Verify that package files exist and match the manifest hashes."""
    s = spec(bench)
    man = load_manifest(bench)
    missing = [str(p) for p in (s.samples_path, s.gold_path) if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"{bench}: missing {missing}" + (f" (set {DATA_ROOT_ENV[bench][0]})" if bench in DATA_ROOT_ENV else ""))
    got = {"samples": _sha256(s.samples_path), "gold": _sha256(s.gold_path)}
    want = {"samples": man["content_sha256_samples"], "gold": man["content_sha256_gold"]}
    if got != want:
        raise ValueError(f"{bench}: content hash mismatch {got} != {want}")
    return {"bench": bench, "item_count": man["item_count"], **got}


#: Benchmarks with a frozen selection of task IDs. Use subset=False to load the full package.
SUBSETS = {"healthadmin": "subset-v1.json", "healthagent": "subset-v1.json"}


def subset_ids(bench: str) -> list[str] | None:
    """The frozen task ids for this benchmark, or None if it runs on the full package."""
    name = SUBSETS.get(bench)
    if name is None:
        return None
    path = spec(bench).package_dir / name
    if not path.exists():
        raise FileNotFoundError(f"{bench}: required task subset is missing: {path}")
    return list(json.loads(path.read_text())["task_ids"])


def load_gold(bench: str) -> dict[str, dict[str, Any]]:
    """Load gold references for scorer and environment use outside the model workspace."""
    s = spec(bench)
    out: dict[str, dict[str, Any]] = {}
    with s.gold_path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            out[rec["id"]] = rec
    return out


def load_records(bench: str, *, check: bool = True, subset: bool = True):
    """Framework-independent records for the product/MH-Lab task exporters."""
    if check:
        verify(bench)
    records = [json.loads(line) for line in spec(bench).samples_path.read_text().splitlines() if line.strip()]
    ids = subset_ids(bench) if subset else None
    if ids is not None:
        keep = set(ids)
        records = [record for record in records if str(record['id']) in keep]
        missing = keep - {str(record['id']) for record in records}
        if missing:
            raise ValueError(f'{bench}: subset records missing: {sorted(missing)}')
    return [SimpleNamespace(**record) for record in records]
