"""Deterministic MedTool scoring, including repeated calculator episodes."""
from __future__ import annotations

import re
from typing import Any

from medharness.scorers.medmcp_scorer import score_task

_CONCEPT_STOPWORDS = {
    "a", "and", "assessment", "based", "calculator", "calculation", "classification",
    "clinical", "dosing", "equation", "estimated", "for", "in", "index", "level", "of", "or",
    "pain", "patient", "result", "risk", "scale", "score", "serum", "status", "test", "the",
    "tool", "total", "using", "value",
}


def _concept_tokens(value: Any) -> set[str]:
    aliases = {
        "bsa": {"body", "surface", "area"},
        "crcl": {"creatinine", "clearance"},
        "egfr": {"glomerular", "filtration", "rate"},
        "gfr": {"glomerular", "filtration", "rate"},
        "ich": {"intracerebral", "hemorrhage"},
    }
    tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", str(value or "").casefold())
        if token not in _CONCEPT_STOPWORDS and not token.isdigit()
    }
    expanded = set(tokens)
    for token in tokens:
        expanded.update(aliases.get(token, set()))
    return expanded


def _values_match(input_value: Any, final_value: Any) -> tuple[bool, float]:
    left, right = _numeric(input_value), _numeric(final_value)
    if left is not None and right is not None:
        difference = abs(left - right)
        tolerance = max(0.05, abs(right) * 0.05)
        return difference <= tolerance, difference / max(abs(right), 1.0)
    a = re.sub(r"\s+", " ", str(input_value or "").strip().casefold())
    b = re.sub(r"\s+", " ", str(final_value or "").strip().casefold())
    return bool(a and a == b), 0.0 if a == b else 1.0


def _word_bigrams(value: Any) -> set[tuple[str, str]]:
    words = [
        token
        for token in re.findall(r"[a-z0-9]+", str(value or "").casefold())
        if token not in {"a", "an", "and", "for", "of", "or", "the", "with"}
    ]
    return set(zip(words, words[1:]))


def _acronyms(value: Any) -> set[str]:
    return set(re.findall(r"\b[A-Z][A-Z0-9-]{1,}\b", str(value or "")))


def _transformed_value_reference(value: Any, producer_name: Any) -> bool:
    producer_tokens = _concept_tokens(producer_name)
    for annotation in re.findall(r"\(([^)]*)\)", str(value or "")):
        annotation_tokens = _concept_tokens(annotation)
        has_transform_marker = bool(
            annotation_tokens & {"adjusted", "calculated", "corrected", "derived", "estimated", "step"}
        )
        if has_transform_marker and len(annotation_tokens & producer_tokens) >= 2:
            return True
    return False


def derived_input_edges(task: dict[str, Any]) -> list[dict[str, int]]:
    answers = task.get("calculator_answers", [])
    edges = []
    for consumer_index, answer in enumerate(answers):
        for input_index, item in enumerate(answer.get("inputs", [])):
            field = item.get("field")
            field_concepts = _concept_tokens(field)
            candidates = []
            for producer_index, producer in enumerate(answers):
                if producer_index == consumer_index:
                    continue
                producer_name = producer.get("name")
                consumer_name = answer.get("name")
                producer_concepts = _concept_tokens(producer_name)
                overlap = field_concepts & producer_concepts
                value_match, relative_difference = _values_match(
                    item.get("value"), producer.get("final_answer")
                )
                transformed_value = _transformed_value_reference(item.get("value"), producer_name)
                shared_phrase = bool(_word_bigrams(field) & _word_bigrams(producer_name))
                producer_acronyms = {value.casefold() for value in _acronyms(producer_name)}
                shared_acronym = bool(
                    {_value.casefold() for _value in _acronyms(field)} & producer_acronyms
                    or field_concepts & producer_acronyms
                )
                producer_before_adjacent = producer_index + 1 == consumer_index
                different_calculator = str(producer_name).casefold() != str(consumer_name).casefold()
                strong_semantic_reference = shared_phrase or len(overlap) >= 2
                is_dependency = transformed_value or (
                    value_match
                    and different_calculator
                    and (
                        strong_semantic_reference
                        or (shared_acronym and relative_difference <= 0.02)
                        or (producer_before_adjacent and bool(overlap) and relative_difference <= 0.02)
                    )
                )
                if not is_dependency:
                    continue
                candidates.append(
                    (
                        0 if strong_semantic_reference else 1,
                        0 if value_match else 1,
                        relative_difference,
                        abs(consumer_index - producer_index),
                        0 if producer_index < consumer_index else 1,
                        producer_index,
                    )
                )
            if candidates:
                producer_index = min(candidates)[-1]
                edges.append(
                    {
                        "consumer_index": consumer_index,
                        "input_index": input_index,
                        "producer_index": producer_index,
                    }
                )
    return edges


def _episode_roots(n_answers: int, edges: list[dict[str, int]]) -> dict[int, int]:
    parents = list(range(n_answers))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for edge in edges:
        left, right = find(edge["consumer_index"]), find(edge["producer_index"])
        if left != right:
            parents[max(left, right)] = min(left, right)
    return {index: find(index) for index in range(n_answers)}


def answer_episode_ids(task: dict[str, Any]) -> list[str]:
    answers = task.get("calculator_answers", [])
    roots = _episode_roots(len(answers), derived_input_edges(task))
    return [f"episode_{roots.get(index, index):03d}" for index in range(len(answers))]


def score_prediction(prediction: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    episodes = answer_episode_ids(task)
    scored_task = {
        **task,
        "calculator_answers": [
            {**answer, "episode_id": episodes[index]}
            for index, answer in enumerate(task.get("calculator_answers", []))
        ],
    }
    score = score_task(prediction, scored_task)
    legacy_value_accuracy = score["value_accuracy"]
    legacy_utility = score["utility"]
    total_answers = len(task.get("calculator_answers", []))
    strict_value_accuracy = score["n_correct"] / total_answers if total_answers else 0.0
    strict_utility = score["selection"]["f1"] * strict_value_accuracy
    return {
        **score,
        "legacy_value_accuracy": legacy_value_accuracy,
        "legacy_utility": legacy_utility,
        "strict_value_accuracy": round(strict_value_accuracy, 3),
        "value_accuracy": round(strict_value_accuracy, 3),
        "utility": round(strict_utility, 3),
    }


def _numeric(value: Any) -> float | None:
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    return float(match.group(0)) if match else None
