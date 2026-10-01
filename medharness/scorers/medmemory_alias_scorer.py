"""Deterministic, alias-aware scorer for the MedMemoryBench long-context final set.

Two matching modes are always computed and both are reported:

* ``exact``       normalized exact match of the extracted answer against any accepted alias.
                  Headline metric.  Robust to "shotgun" outputs that list many candidates,
                  including candidates hidden inside parentheses.
* ``containment`` upstream/v1-compatible: alphanumeric-only gold contained anywhere in the
                  alphanumeric-only output.  Reported for comparability only.  It is lenient by
                  construction (drops decimal points, signs and range dashes; a shotgun output
                  satisfies it), so it is never the headline.

``normalize_presentation`` removes presentation variance only: case, whitespace, markdown /
quote wrappers, list bullets, terminal punctuation, leading hedges, dash and hyphen glyphs,
digit-unit spacing, ``per`` -> ``/``, and parentheticals that carry no alternative or novel
value.  It never removes or reorders content words, never converts number words or units, and
never collapses superscript exponents into plain digits.  Aliases are extended only through a
recorded adjudication (``alias_adjudications.json`` next to the final set), never here.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Mapping

_SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")
_SUP_RE = re.compile(r"(?<=\d)[⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺]+")
_DASHES = re.compile("[‐‑‒–—―⁃−]")
_WRAP = re.compile(r"^(?:\*\*|\*|__|_|`|\"|'|“|”|‘|’|«|»)+|(?:\*\*|\*|__|_|`|\"|'|“|”|‘|’|«|»)+$")
_BULLET = re.compile(r"^[-•·]\s+(?=\S)")
_TERMINAL = re.compile(r"[.;,:!?。！？；，：]+$")
_HEDGE = re.compile(r"^(?:(?:approximately|approx\.?|about|around|roughly|estimated|est\.?)\s+|[~≈]\s*)")
_PAREN = re.compile(r"\s*\(([^()]*)\)")
_WHOLE_PAREN = re.compile(r"^\s*\((?:[^()]|\([^()]*\))*\)\s*$")
_ALT_PAREN = re.compile(r"^\s*(?:or|not|no|vs\.?|versus|instead|rather|except|possibly|maybe|alternatively|alt\.?)\b", re.I)
_NUM = re.compile(r"\d+(?:\.\d+)?")
_NUM_UNIT = re.compile(r"\d+(?:\.\d+)?\s*[a-zA-Zμ%/]")
_FINAL = re.compile(r"^\s*[*_#>\-\s]*(?:final answer|answer)\s*[*_]*\s*[:：]\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_ABSTAIN = re.compile(
    r"^\s*(?:unknown|i (?:do not|don't|cannot|can't) (?:know|determine|find)|not (?:enough|sufficient) information|"
    r"cannot be determined|no (?:such )?(?:record|information)|insufficient data)\b",
    re.IGNORECASE,
)


def _strip_wrappers(text: str) -> str:
    prev = None
    while prev != text:
        prev = text
        text = _WRAP.sub("", text).strip()
        text = _BULLET.sub("", text).strip()
        text = _TERMINAL.sub("", text).strip()
    return text


def _strip_hedges(text: str) -> str:
    prev = None
    while prev != text:
        prev = text
        text = _HEDGE.sub("", text).strip()
    return text


def _strip_parentheticals(text: str) -> str:
    """Remove parentheticals that carry no alternative marker and no novel valued number."""
    while _WHOLE_PAREN.fullmatch(text):
        text = text.strip()[1:-1].strip()
    prev = None
    while prev != text:
        prev = text
        out, pos = [], 0
        for m in _PAREN.finditer(text):
            inner = m.group(1)
            outside = text[: m.start()] + text[m.end():]
            keep = bool(_ALT_PAREN.match(inner)) or bool(
                (set(_NUM.findall(inner)) - set(_NUM.findall(outside))) and _NUM_UNIT.search(inner)
            )
            out.append(text[pos: m.start()])
            out.append(m.group(0) if keep else "")
            pos = m.end()
        out.append(text[pos:])
        text = "".join(out).strip()
        if text == prev:
            break
        # a kept parenthetical must not be re-examined as if it were new
        if all(_ALT_PAREN.match(m.group(1)) or (set(_NUM.findall(m.group(1))) - set(_NUM.findall(text[: m.start()] + text[m.end():]))) for m in _PAREN.finditer(text)):
            break
    return text


def normalize_presentation(value: Any) -> str:
    text = str(value or "")
    text = _SUP_RE.sub(lambda m: "^" + m.group(0).translate(_SUP), text)
    text = unicodedata.normalize("NFKC", text).replace("­", "")
    text = _DASHES.sub("-", text).replace("⁄", "/")
    text = " ".join(text.split()).casefold().strip()
    text = _strip_wrappers(text)
    text = _strip_hedges(text)
    text = _strip_parentheticals(text)
    text = _strip_wrappers(text)
    text = _strip_hedges(text)
    text = re.sub(r"(\d)([a-zμ%/])", r"\1 \2", text)
    text = re.sub(r"\s+per\s+", "/", text)
    text = re.sub(r"\s*/\s*", "/", text)
    text = re.sub(r"\s*-\s*", "-", text)
    return " ".join(text.split())


def normalize_alnum(value: Any) -> str:
    """v1 direct-eval compatible: NFKD, lowercase, alphanumerics only.  Deliberately unchanged."""
    text = unicodedata.normalize("NFKD", str(value or "")).lower()
    return "".join(ch for ch in text if ch.isalnum())


def extract_final_answer(raw: Any) -> str:
    """Take the last ``Answer:``/``Final answer:`` line (markdown-decorated or not) if present."""
    text = str(raw or "").strip()
    found = _FINAL.findall(text)
    return found[-1].strip() if found else text


def is_abstention(text: str) -> bool:
    return bool(_ABSTAIN.match(_strip_wrappers(text.strip())))


@dataclass(frozen=True)
class AnswerScore:
    exact: bool
    containment: bool
    abstained: bool
    shotgun_flag: bool
    extracted: str
    normalized: str
    matched_alias: str | None
    aliases: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {
            "exact": self.exact,
            "containment": self.containment,
            "abstained": self.abstained,
            "shotgun_flag": self.shotgun_flag,
            "extracted": self.extracted,
            "normalized": self.normalized,
            "matched_alias": self.matched_alias,
        }


def score_answer(item: Mapping[str, Any], raw_output: Any, *, shotgun_ratio: float = 3.0) -> AnswerScore:
    aliases = tuple(str(a) for a in item.get("accepted_aliases") or item.get("official_answers") or [])
    if not aliases:
        raise ValueError(f"item {item.get('item_id')} has no accepted aliases")
    extracted = extract_final_answer(raw_output)
    abstained = is_abstention(extracted)
    norm = normalize_presentation(extracted)
    norm_aliases = {normalize_presentation(a): a for a in aliases}
    matched = norm_aliases.get(norm)
    exact = matched is not None and not abstained
    out_alnum = normalize_alnum(extracted)
    contained = any(normalize_alnum(a) and normalize_alnum(a) in out_alnum for a in aliases) and not abstained
    longest_alias = max((len(normalize_alnum(a)) for a in aliases), default=1) or 1
    shotgun = contained and not exact and len(out_alnum) > shotgun_ratio * longest_alias
    return AnswerScore(exact, contained, abstained, shotgun, extracted, norm, matched, aliases)
