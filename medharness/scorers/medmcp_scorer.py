"""MedMCP-Calc deterministic scorer — replaces the official LLM judge. Pure, deterministic, no model calls.

Scores a model's structured answer against the task gold on three axes:
  - selection: did it pick the right calculators?  (set-match precision/recall/F1, name-normalized)
  - numeric value: |pred - gold| within tolerance (relative for big numbers, absolute for small)
  - categorical value: canonical-normalized exact match (Grade 2 CRS == G2 == grade 2 == 2)
  - abstain: a calculator the model abstained on (value=null) is NOT scored wrong if a needed variable is
    genuinely absent — it is counted separately (abstentions), never as a fabricated wrong value.
"""
from __future__ import annotations

import re

_ROMAN = {"i": "1", "ii": "2", "iii": "3", "iv": "4", "v": "5", "vi": "6"}
_STOPWORDS = ("equation", "equations", "score", "scoring", "grading", "grade", "stage", "index", "tool",
              "calculator", "criteria", "classification", "formula", "the", "for", "of", "and", "calculation",
              "estimated", "estimate", "rate")


# Calculator names are matched through normalized token overlap and common aliases.
_ALIASES = {
    "kegfr": "kinetic egfr", "ckdepi": "ckd epi egfr gfr", "bsa": "body surface area",
    "cargtt": "carg cancer aging research toxicity", "ecog": "ecog performance status",
    "cockcroftgault": "cockcroft gault creatinine clearance crcl", "calvert": "calvert carboplatin auc",
    "ibw": "ideal body weight", "abw": "adjusted body weight",
    "corrected": "correction", "osmolarity": "osmolality",
}


def _toks(s: str):
    s = re.sub(r"\(([^)]*)\)", r" \1 ", s or "").lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    out = set()
    for t in s.split():
        if not t or t in _STOPWORDS:
            continue
        out.add(_ALIASES.get(t, t))
    # expand any alias phrases into their tokens
    flat = set()
    for t in out:
        flat.update(t.split())
    return flat


def norm_name(s: str) -> str:
    return " ".join(sorted(_toks(s)))


def names_match(a: str, b: str) -> bool:
    """Two calculator names refer to the same calculator if their significant token sets substantially
    overlap: one is (almost) a subset of the other, or Jaccard >= 0.45, or they share a distinctive acronym."""
    A, B = _toks(a), _toks(b)
    if not A or not B:
        return False
    inter = A & B
    if not inter:
        return False
    jacc = len(inter) / len(A | B)
    subset = inter == A or inter == B or len(inter) >= min(len(A), len(B))
    distinctive = any(len(t) >= 4 and t in B for t in A if t in _ALIASES.values() or t in ("ecog", "calvert", "cockcroft", "tash", "crs", "icans"))
    return jacc >= 0.45 or subset or distinctive


def _num(x):
    if isinstance(x, (int, float)):
        return float(x)
    m = re.search(r"-?\d+\.?\d*", str(x))
    return float(m.group(0)) if m else None


def norm_cat(x) -> str:
    """Canonical categorical token: 'Grade 2 CRS' / 'G2' / 'grade 2' / '2' -> '2'; 'Stage IA' -> 'a1' (sorted);
    'Positive' -> 'positive'. Strips grade/stage/g prefixes, maps roman numerals, sorts residual tokens."""
    s = str(x).lower().strip()
    s = re.sub(r"\bg(?=\d)", "grade ", s)                  # 'g2' -> 'grade 2'
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    toks = []
    for t in s.split():
        if t in _STOPWORDS or t in ("crs", "icans", "irae", "ice"):
            continue
        toks.append(_ROMAN.get(t, t))
    # split alnum like 'ia' -> letters+digits stay; sort for order-independence (2a == a2)
    flat = "".join(toks)
    return "".join(sorted(re.findall(r"[a-z0-9]", flat)))


def _is_num_gold(g) -> bool:
    """Gold is NUMERIC only if it is a number, or a string that is purely a number + optional unit
    ('29 mL/min', '120.6', '5.3'). A graded/staged/worded gold ('Grade 2 CRS', '1000 mg every 12 hours')
    is CATEGORICAL even though it contains digits."""
    if isinstance(g, (int, float)):
        return True
    return bool(re.match(r"^\s*-?\d+\.?\d*\s*[a-zA-Z%/.²]*\s*$", str(g)))


def score_value(pred, gold, *, num_rel=0.02, num_abs=0.1):
    """One calculator's value. Returns (correct: bool, kind, detail). Numeric gold -> tolerance; categorical
    gold -> canonical match. pred=None/'' -> abstained (scored separately, never a fabricated wrong value)."""
    if pred is None or (isinstance(pred, str) and not pred.strip()):
        return (False, "abstain", "no value")
    if _is_num_gold(gold):
        p, g = _num(pred), _num(gold)
        if p is None:
            return (False, "numeric", f"non-numeric pred {pred!r}")
        tol = max(num_abs, abs(g) * num_rel)
        return (abs(p - g) <= tol, "numeric", f"|{p}-{g}|<= {tol:.3g}")
    return (norm_cat(pred) == norm_cat(gold), "categorical", f"{norm_cat(pred)!r} vs {norm_cat(gold)!r}")


def score_selection(pred_names, gold_required):
    """precision / recall / F1 via greedy bipartite name-matching (names_match), not exact string equality."""
    # selection is over DISTINCT calculator TYPES — a calculator applied at 6 time points is still ONE
    # selection. required_calculators is the distinct gold; dedupe predicted names by normalized key.
    seen, P = set(), []
    for n in (pred_names or []):
        k = norm_name(n)
        if n and k not in seen:
            seen.add(k); P.append(n)
    G = [n for n in (gold_required or []) if n]
    if not G:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "matched": 0, "gold": 0}
    used_g, used_p, matched = set(), set(), 0
    # PASS 1: exact normalized-name equality (keeps gold-vs-gold a perfect 1:1, no cross-matching of similar
    # calculators). PASS 2: fuzzy names_match for model wording variants.
    for exact in (True, False):
        for pi, p in enumerate(P):
            if pi in used_p:
                continue
            pn = norm_name(p)
            for j, g in enumerate(G):
                if j in used_g:
                    continue
                if (norm_name(g) == pn) if exact else names_match(p, g):
                    used_g.add(j); used_p.add(pi); matched += 1; break
    prec = matched / len(P) if P else 0.0
    rec = matched / len(G)
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return {"precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3),
            "matched": matched, "gold": len(G), "predicted": len(P)}


def score_task(pred, gold):
    """pred: {'calculators': [{'name','value'}, ...]}. gold: the task dict (required_calculators +
    calculator_answers). Returns selection + per-calculator value scoring + a utility scalar."""
    pred_calcs = pred.get("calculators", []) if isinstance(pred, dict) else []
    sel = score_selection([c.get("name") for c in pred_calcs], gold["required_calculators"])
    # Match each gold answer to a predicted calculator (exact-first, then fuzzy), consuming predicted
    # instances. For recurring calculators, an explicit episode_id takes precedence over list order.
    remaining = list(pred_calcs)
    matched_pv = [None] * len(gold["calculator_answers"])
    for exact in (True, False):
        for gi, ca in enumerate(gold["calculator_answers"]):
            if matched_pv[gi] is not None:
                continue
            gn = norm_name(ca["name"])
            candidates = []
            for k, c in enumerate(remaining):
                name_matches = (
                    norm_name(c.get("name", "")) == gn
                    if exact
                    else names_match(c.get("name", ""), ca["name"])
                )
                if name_matches:
                    candidates.append(k)
            if not candidates:
                continue
            gold_episode = str(ca.get("episode_id") or "")
            episode_matches = [
                k for k in candidates
                if gold_episode and str(remaining[k].get("episode_id") or "") == gold_episode
            ]
            no_episode = [k for k in candidates if not remaining[k].get("episode_id")]
            if episode_matches:
                selected = episode_matches[0]
            elif no_episode:
                selected = no_episode[0]
            elif not gold_episode:
                selected = candidates[0]
            else:
                continue
            matched_pv[gi] = (remaining.pop(selected).get("value"),)
    rows, correct, scored, abstained = [], 0, 0, 0
    for gi, ca in enumerate(gold["calculator_answers"]):
        pv = matched_pv[gi][0] if matched_pv[gi] is not None else None
        ok, kind, detail = score_value(pv, ca["final_answer"])
        if kind == "abstain":
            abstained += 1
        else:
            scored += 1
            correct += 1 if ok else 0
        rows.append({"calc": ca["name"][:40], "gold": ca["final_answer"], "pred": pv, "ok": ok, "kind": kind})
    value_acc = correct / scored if scored else 0.0
    # utility = selection F1 * value accuracy (both must be right); abstentions excluded from value_acc
    utility = round(sel["f1"] * value_acc, 3)
    return {"selection": sel, "value_accuracy": round(value_acc, 3), "n_correct": correct,
            "n_scored": scored, "n_abstained": abstained, "utility": utility, "rows": rows}


if __name__ == "__main__":
    # self-test: normalization + a tiny scored example
    assert norm_cat("Grade 2 CRS") == norm_cat("G2") == norm_cat("grade 2") == norm_cat("2"), norm_cat("G2")
    assert norm_cat("Positive") == norm_cat("positive") and norm_cat("2A") == norm_cat("a2")
    assert score_value(29.1, 29)[0] and not score_value(35, 29)[0]
    assert score_value("Grade 2 CRS", "G2")[0] and not score_value("Grade 3", "G2")[0]
    g = {"required_calculators": ["Cockcroft-Gault (CrCl)", "CRS Grade"], "calculator_answers": [
        {"name": "Cockcroft-Gault (CrCl)", "final_answer": 29}, {"name": "CRS Grade", "final_answer": "Grade 2"}]}
    p = {"calculators": [{"name": "Cockcroft-Gault (CrCl)", "value": 29.2}, {"name": "CRS Grade", "value": "G2"}]}
    r = score_task(p, g)
    assert r["utility"] == 1.0, r
    print("medmcp_scorer self-test OK:", {k: r[k] for k in ("selection", "value_accuracy", "utility")})
