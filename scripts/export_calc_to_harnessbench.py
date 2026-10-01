#!/usr/bin/env python
"""Export MedTool tasks into harness-bench format.

Each task provides a patient-data SQLite fixture and requests structured output
in out/answer.json. The oracle grades calculator selection and values after the
episode. Gold references stay in the source task package, outside fixtures/.
"""
from __future__ import annotations

import argparse
import sys
import os
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from medharness.data import registry as R
from medharness.data.fixtures import build_fixture

OUT_ROOT = Path(os.environ.get("MEDHARNESS_TASKS_ROOT", str(REPO / "tasks")))
PREFIX = "medcalc"

ORACLE = '''from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

REPO = Path(os.environ.get("MEDHARNESS_REPO", "{repo}"))
TASK_ID = "{sample_id}"


def _score(prediction, task):
    sys.path.insert(0, str(REPO))
    from medharness.scorers.calc import score_prediction
    return score_prediction(prediction, task)


def score_workspace(workspace: Path) -> dict[str, Any]:
    sys.path.insert(0, str(REPO))
    from medharness.data import registry as R
    from medharness.scorers.calc_submission import recover_submission

    target = workspace / "out" / "answer.json"
    raw = target.read_text(encoding="utf-8", errors="replace") if target.is_file() else ""
    prediction, failure = recover_submission(raw)

    gold = R.load_gold("calc")[TASK_ID]["gold"]["gold"]
    n_gold = len(gold.get("calculator_answers") or [])
    if prediction is None:
        return {{
            "task": "{task_id}",
            "workspace": str(workspace),
            "checks": [{{"id": "submission", "label": "out/answer.json parses as a calculator list",
                        "pass": False, "weight": 1.0, "detail": failure}}],
            "outcome_score": 0.0,
            "metrics": {{"submission_failure": failure, "n_gold_outputs": n_gold}},
        }}

    result = _score(prediction, gold)
    sel = result["selection"]
    return {{
        "task": "{task_id}",
        "workspace": str(workspace),
        "checks": [
            {{"id": "submission", "label": "out/answer.json parses", "pass": True, "weight": 0.0, "detail": None}},
            {{"id": "selection", "label": "named the required calculators",
             "pass": sel["f1"] >= 0.999, "weight": 0.0, "detail": f"F1={{sel['f1']:.3f}}"}},
            {{"id": "values", "label": "every gold output correct",
             "pass": result["n_correct"] == n_gold, "weight": 1.0,
             "detail": f"{{result['n_correct']}}/{{n_gold}} correct"}},
        ],
        # Strict utility combines calculator-selection F1 with value accuracy.
        "outcome_score": float(result["utility"]),
        "metrics": {{
            "utility": float(result["utility"]),
            "selection_f1": sel["f1"],
            "selection_precision": sel["precision"],
            "selection_recall": sel["recall"],
            "n_correct": result["n_correct"],
            "n_gold_outputs": n_gold,
            "value_accuracy": result["value_accuracy"],
            "submission_failure": None,
        }},
    }}
'''

PROMPT = """You are solving one clinical-calculation task. Your working directory is `$WORKSPACE`.

{scenario}

## The data

The patient's facts are in the SQLite database `$WORKSPACE/in/patient_facts.sqlite`, table
`patient_facts`, with columns: patient_id, field_name, value_text, observed_at, episode_id, unit,
sample_type, source. `episode_id` groups facts into encounters / time points; `observed_at` orders
them.

The table is deliberately contaminated, in the same three ways every time:

- rows belonging to other patients, sitting alongside this one's — filter on the patient id above;
- a creatinine drawn from urine next to the serum one — check `sample_type`;
- a body weight recorded in lb next to the kg one — check `unit`.

## What to produce

Work out which clinical calculators, scores and classifications this scenario actually requires,
compute each one from this patient's facts, and write them out. Nothing tells you which to use;
deciding that from the clinical narrative is the substance of the task.

Some calculators consume the output, or an intermediate component, of another. Carry those values
forward rather than recomputing them from scratch or inventing them.

**Name** each calculator by its full standard published name as it appears in the clinical
literature, including its usual acronym in parentheses when it has one — for example
`"Alvarado Score for Acute Appendicitis"`, not a bare acronym and not a label you coined. Names are
compared on their significant words, so the published wording scores where an ad-hoc abbreviation
does not.

**Value** is that calculator's final reported result, in the form the calculator conventionally
reports it:

- a number, when the instrument yields a score, ratio or quantity;
- the class label, when it yields a category, stage, grade or risk class — not the count of criteria
  that were met, and not a sentence of explanation.

Numbers are accepted within 2% or 0.1 of the reference, whichever is larger, so do not agonise over
rounding. Class labels are compared after normalisation, so `"Grade 2"`, `"G2"` and `"2"` are
equivalent.

If this patient's data genuinely does not support a calculator you believe is required, still name it
and give `"value": null`. A null is recorded as an abstention; a guessed number is recorded as wrong.

Selection is judged over *distinct* calculators: applying one at several time points counts once.
Return one row per episode, repeating the same name, and use the `episode_id` value as it appears in
the database — never collapse repeated results into a trend summary. Naming a calculator the scenario
does not require costs precision; omitting one it does require costs recall.

Take as many steps as the work needs; there is no turn budget.

## Output

Write exactly one JSON object to `$WORKSPACE/out/answer.json`, and nothing else to that file:

{{"calculators": [{{"name": "...", "episode_id": "...", "value": number | "class label" | null}}]}}
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
    samples = R.load_records("calc")
    gold = R.load_gold("calc")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    written = []

    for i, sample in enumerate(samples, 1):
        sid = str(sample.id)
        task_id = f"{PREFIX}-{i:03d}"
        if only and str(sample.id) not in only and task_id not in only:
            continue
        task_dir = OUT_ROOT / task_id
        fixtures = task_dir / "fixtures" / "in"
        if task_dir.exists():
            if not args.overwrite:
                raise FileExistsError(f"Refusing to overwrite {task_dir}; use --overwrite")
            shutil.rmtree(task_dir)
        fixtures.mkdir(parents=True)

        # Build the SQLite fixture from the task package environment data.
        build_fixture(gold[sid].get("env_private") or {},
                      str((sample.metadata.get("descriptors") or {}).get("patient_id") or "PATIENT"),
                      fixtures / "patient_facts.sqlite")

        (task_dir / "prompt.txt").write_text(
            PROMPT.format(scenario=sample.input.strip()), encoding="utf-8")
        (task_dir / "oracle_grade.py").write_text(
            ORACLE.format(sample_id=sid, task_id=task_id, repo=str(REPO)), encoding="utf-8")
        (task_dir / "task.yaml").write_text(json.dumps({
            "task_id": task_id,
            "title": f"MedMCP-Calc {sid}",
            "class": "Clinical calculation over a patient record",
            "prompt_file": "prompt.txt",
            "fixtures_dir": "fixtures",
            "oracle_module": "oracle_grade.py",
            "timeout_sec": 3600,
            "tags": ["medical", "calculator", "sqlite", "medharness"],
        }, indent=2), encoding="utf-8")
        written.append(task_id)

    if not written:
        raise ValueError("No tasks matched the requested IDs")
    print(f"exported {len(written)} tasks to {OUT_ROOT}")
    print(f"  ids: {written[0]} .. {written[-1]}")
    sizes = [(OUT_ROOT / t / "fixtures" / "in" / "patient_facts.sqlite").stat().st_size for t in written]
    print(f"  fixture sizes: {min(sizes):,} to {max(sizes):,} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
