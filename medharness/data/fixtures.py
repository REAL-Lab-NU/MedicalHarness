"""SQLite fixture construction for MedTool task export."""
from __future__ import annotations
import re
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping

def _unit(value: Any) -> str:
    m = re.search(r"\d\s*([A-Za-zµμ%/²^][A-Za-z0-9µμ%/.²^-]*)", str(value))
    return m.group(1) if m else ""


def _numeric(value: Any) -> float | None:
    m = re.search(r"-?\d+(?:\.\d+)?", str(value))
    return float(m.group(0)) if m else None


def build_fixture(fixture: Mapping[str, Any], patient_id: str, path: str | Path) -> Path:
    """Create the SQLite file for one sample from env_private.fixture (root_inputs + distractors)."""
    path = Path(path)
    if path.exists():
        path.unlink()
    con = sqlite3.connect(path)
    con.execute("""CREATE TABLE patient_facts (patient_id TEXT, field_name TEXT, value_text TEXT, observed_at TEXT,
                   episode_id TEXT, unit TEXT, sample_type TEXT, source TEXT)""")
    base = datetime(2025, 1, 1, 9, 0, 0)
    rows: list[tuple] = []
    order: dict[str, int] = {}
    for item in fixture.get("root_inputs") or []:
        field, value, ep = str(item["field"]), item.get("value"), str(item["episode_id"])
        idx = order.setdefault(ep, len(order))
        sample = "Serum" if any(t in field.casefold() for t in ("creatinine", "sodium", "albumin", "glucose", "bun")) else "Chart"
        rows.append((patient_id, field, str(value), (base + timedelta(days=idx)).isoformat(), ep, _unit(value), sample, "official_gold_input"))
    distractors = set(fixture.get("distractors") or [])
    if rows and "wrong_patient" in distractors:
        f = rows[0]; n = _numeric(f[2])
        rows.append(("P-WRONG", f[1], str(round(n * 0.5, 4)) if n is not None else "unrelated", "2025-12-31T09:00:00", "episode_wrong_patient", f[5], f[6], "wrong_patient_distractor"))
    if "wrong_sample_creatinine_urine" in distractors:
        cr = next((r for r in rows if "creatinine" in r[1].casefold() and r[0] == patient_id), None)
        if cr:
            rows.append((patient_id, cr[1], str(round((_numeric(cr[2]) or 1.0) * 40, 2)), "2025-12-30T09:00:00", "episode_wrong_sample", "mg/dL", "Urine", "wrong_sample_distractor"))
    if "unit_variant_weight_lb" in distractors:
        w = next((r for r in rows if "weight" in r[1].casefold() and r[0] == patient_id), None)
        if w and (n := _numeric(w[2])) is not None:
            rows.append((patient_id, w[1], f"{n * 2.20462:.1f} lb", "2025-12-29T09:00:00", "episode_unit_variant", "lb", "Chart", "unit_variant_distractor"))
    con.executemany("INSERT INTO patient_facts VALUES (?,?,?,?,?,?,?,?)", rows)
    con.commit(); con.close()
    return path
