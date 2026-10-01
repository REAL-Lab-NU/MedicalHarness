"""Deterministic MedWeb checks on final portal state.

Process/outcome rules follow HealthAdminBench commit
 e71a8f4d6923037805b7f51fbbf608d12ea56cf5. No model judge is called.
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

CheckClass = Literal["process", "outcome"]


class HealthAdminScoringError(ValueError):
    """Base class for malformed inputs or unsupported upstream changes."""


class CheckClassificationError(HealthAdminScoringError):
    """Raised when a check matches neither or both frozen classifier arms."""


class RestrictedJMESPathError(HealthAdminScoringError):
    """Base class for restricted-JMESPath failures."""


class RestrictedJMESPathSyntaxError(RestrictedJMESPathError):
    """Raised for expressions outside the audited upstream syntax subset."""


class RestrictedJMESPathRuntimeError(RestrictedJMESPathError):
    """Raised when a supported expression receives incompatible state values."""


OUTCOME_QUERY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "prior_auth_submission_diff",
        re.compile(r"\b(?:aetna_state|anthem_state)\.differences\.priorAuth\.added\b"),
    ),
    (
        "appeal_submission",
        re.compile(
            r"\bpayer_[ab]_state\.full_state\.appealActions\."
            r"(?:submittedAppeal|submittedRationale|submittedAttachment(?:Names|Count)?)\b"
        ),
    ),
    (
        "fax_final_state",
        re.compile(
            r"\bfull_state\.faxPortal\."
            r"(?:faxesSent|attachmentNames|faxRecipient|faxNumber|useCertifiedDelivery|coverNotes)\b"
        ),
    ),
    (
        "worklist_final_state",
        re.compile(r"\bfull_state\.cleared(?:Referrals|Denials)\b"),
    ),
    (
        "agent_recorded_final_state",
        re.compile(
            r"\bfull_state\.agentActions\."
            r"(?:selectedDisposition|documentedAppealInEpic|addedAuthNote|"
            r"addedProgressNote|addedFollowUpTask)\b"
        ),
    ),
    (
        "documentation_content",
        re.compile(r"\bfull_state\.(?:triageNotes|communications)\b"),
    ),
)


PROCESS_QUERY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("emr_navigation_signals", re.compile(r"\bsignals\.")),
    (
        "agent_navigation_and_review_actions",
        re.compile(
            r"\bfull_state\.agentActions\."
            r"(?:accessedPayerPortalForDenial|downloadedSupportingDoc|readClinicalNote|"
            r"viewedAuthLetter|viewedDenialDetails|viewedDocuments|viewedPatientInquiry|"
            r"viewedPaymentPosting|viewedRemittanceImage)\b"
        ),
    ),
    (
        "payer_appeal_navigation_actions",
        re.compile(
            r"\bpayer_[ab]_state\.full_state\.appealActions\."
            r"(?:checkedEligibility|openedDisputeForm|searchedAuthInquiry|"
            r"searchedClaims|viewedClaimDetail)\b"
        ),
    ),
    (
        "payer_status_searches",
        re.compile(
            r"\b(?:aetna_state|anthem_state)\.differences\."
            r"(?:authSearches|eligibilityChecks)\b"
        ),
    ),
    (
        "fax_phonebook_lookup",
        re.compile(r"\bfull_state\.faxPortal\.lookedUpFaxNumber\b"),
    ),
)


def _check_source(check: Mapping[str, Any]) -> str:
    if check.get("type") == "llm_judge":
        return str(check.get("student_answer") or "")
    return str(check.get("query") or "")


def _pattern_names(
    source: str, patterns: Sequence[tuple[str, re.Pattern[str]]]
) -> list[str]:
    return [name for name, pattern in patterns if pattern.search(source)]


def classify_check(check: Mapping[str, Any]) -> CheckClass:
    """Classify one pinned-v2 check, failing closed on classifier drift."""

    source = _check_source(check)
    outcome = _pattern_names(source, OUTCOME_QUERY_PATTERNS)
    process = _pattern_names(source, PROCESS_QUERY_PATTERNS)
    if outcome and process:
        raise CheckClassificationError(
            "check matches both outcome and process rules: "
            f"source={source!r}, outcome={outcome}, process={process}"
        )
    if process:
        return "process"
    if outcome:
        return "outcome"
    raise CheckClassificationError(
        f"check matches neither frozen classifier arm: source={source!r}"
    )


@dataclass(frozen=True)
class _Token:
    kind: str
    value: Any
    offset: int


def _tokenize(expression: str) -> list[_Token]:
    tokens: list[_Token] = []
    i = 0
    two_character = {"||", "&&", "==", "!=", ">=", "<="}
    one_character = set(".[](),!><*@|-")
    while i < len(expression):
        char = expression[i]
        if char.isspace():
            i += 1
            continue
        pair = expression[i : i + 2]
        if pair in two_character:
            tokens.append(_Token(pair, pair, i))
            i += 2
            continue
        if char in one_character:
            tokens.append(_Token(char, char, i))
            i += 1
            continue
        if char == "'":
            start = i
            i += 1
            chars: list[str] = []
            while i < len(expression):
                if expression[i] == "'":
                    i += 1
                    break
                if expression[i] == "\\" and i + 1 < len(expression):
                    next_char = expression[i + 1]
                    if next_char in {"\\", "'"}:
                        chars.append(next_char)
                        i += 2
                        continue
                chars.append(expression[i])
                i += 1
            else:
                raise RestrictedJMESPathSyntaxError(
                    f"unterminated raw string at offset {start}: {expression!r}"
                )
            tokens.append(_Token("literal", "".join(chars), start))
            continue
        if char == "`":
            start = i
            i += 1
            escaped = False
            chars = []
            while i < len(expression):
                current = expression[i]
                if current == "`" and not escaped:
                    i += 1
                    break
                chars.append(current)
                escaped = current == "\\" and not escaped
                if current != "\\":
                    escaped = False
                i += 1
            else:
                raise RestrictedJMESPathSyntaxError(
                    f"unterminated JSON literal at offset {start}: {expression!r}"
                )
            raw = "".join(chars)
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise RestrictedJMESPathSyntaxError(
                    f"invalid JSON literal at offset {start}: {raw!r}"
                ) from exc
            tokens.append(_Token("literal", value, start))
            continue
        if char.isalpha() or char == "_":
            start = i
            i += 1
            while i < len(expression) and (
                expression[i].isalnum() or expression[i] == "_"
            ):
                i += 1
            tokens.append(_Token("identifier", expression[start:i], start))
            continue
        if char.isdigit():
            start = i
            i += 1
            while i < len(expression) and expression[i].isdigit():
                i += 1
            tokens.append(_Token("integer", int(expression[start:i]), start))
            continue
        raise RestrictedJMESPathSyntaxError(
            f"unsupported character {char!r} at offset {i}: {expression!r}"
        )
    tokens.append(_Token("eof", None, len(expression)))
    return tokens


class _Parser:
    def __init__(self, expression: str):
        self.expression = expression
        self.tokens = _tokenize(expression)
        self.index = 0

    @property
    def current(self) -> _Token:
        return self.tokens[self.index]

    def _accept(self, kind: str) -> _Token | None:
        if self.current.kind == kind:
            token = self.current
            self.index += 1
            return token
        return None

    def _expect(self, kind: str) -> _Token:
        token = self._accept(kind)
        if token is None:
            raise RestrictedJMESPathSyntaxError(
                f"expected {kind!r} at offset {self.current.offset}: "
                f"{self.expression!r}"
            )
        return token

    def parse(self) -> Any:
        node = self._parse_pipe()
        self._expect("eof")
        return node

    def _parse_pipe(self) -> Any:
        node = self._parse_or()
        while self._accept("|"):
            node = ("pipe", node, self._parse_or())
        return node

    def _parse_or(self) -> Any:
        node = self._parse_and()
        while self._accept("||"):
            node = ("or", node, self._parse_and())
        return node

    def _parse_and(self) -> Any:
        node = self._parse_comparison()
        while self._accept("&&"):
            node = ("and", node, self._parse_comparison())
        return node

    def _parse_comparison(self) -> Any:
        node = self._parse_unary()
        if self.current.kind in {"==", "!=", ">", "<", ">=", "<="}:
            operator = self.current.kind
            self.index += 1
            node = ("compare", operator, node, self._parse_unary())
        return node

    def _parse_unary(self) -> Any:
        if self._accept("!"):
            return ("not", self._parse_unary())
        return self._parse_postfix()

    def _parse_postfix(self) -> Any:
        node = self._parse_primary()
        while True:
            if self._accept("."):
                name = self._expect("identifier").value
                node = ("field", node, name)
                continue
            if self._accept("["):
                if self._accept("*"):
                    self._expect("]")
                    node = ("project", node)
                    continue
                if self._accept("]"):
                    node = ("flatten", node)
                    continue
                sign = -1 if self._accept("-") else 1
                token = self._accept("integer")
                if token is None:
                    raise RestrictedJMESPathSyntaxError(
                        f"only integer indices and [*] are supported: {self.expression!r}"
                    )
                self._expect("]")
                node = ("index", node, sign * token.value)
                continue
            break
        return node

    def _parse_primary(self) -> Any:
        literal = self._accept("literal")
        if literal is not None:
            return ("literal", literal.value)
        if self._accept("@"):
            return ("current",)
        if self._accept("["):
            self._expect("]")
            # In JMESPath, bare [] is the flatten operator applied to the
            # current value, not an empty-list literal.  Upstream uses both
            # this form and the JSON literal `[]`; preserving the distinction
            # matters for type-error behavior in join().
            return ("flatten", ("current",))
        if self._accept("("):
            node = self._parse_pipe()
            self._expect(")")
            return node
        identifier = self._accept("identifier")
        if identifier is not None:
            if self._accept("("):
                args = []
                if self.current.kind != ")":
                    args.append(self._parse_pipe())
                    while self._accept(","):
                        args.append(self._parse_pipe())
                self._expect(")")
                return ("function", identifier.value, tuple(args))
            return ("field", ("current",), identifier.value)
        raise RestrictedJMESPathSyntaxError(
            f"expected expression at offset {self.current.offset}: {self.expression!r}"
        )


@dataclass
class _Projection:
    values: list[Any]


def _jmes_truthy(value: Any) -> bool:
    # JMESPath deliberately treats numeric zero as truthy.  Avoid Python's
    # ``0 == False`` coercion here.
    if value is None or value is False:
        return False
    if isinstance(value, (str, list, dict)) and len(value) == 0:
        return False
    return True


def _jmes_equal(left: Any, right: Any) -> bool:
    """JMESPath equality without Python's bool/int type coercion."""

    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left == right
    numeric = (int, float)
    if isinstance(left, numeric) or isinstance(right, numeric):
        return (
            isinstance(left, numeric)
            and not isinstance(left, bool)
            and isinstance(right, numeric)
            and not isinstance(right, bool)
            and left == right
        )
    if type(left) is not type(right):
        return False
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _jmes_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, dict):
        return set(left) == set(right) and all(
            _jmes_equal(left[key], right[key]) for key in left
        )
    return left == right


def _unwrap(value: Any) -> Any:
    return value.values if isinstance(value, _Projection) else value


def _field(value: Any, name: str) -> Any:
    if isinstance(value, _Projection):
        projected = []
        for item in value.values:
            found = item.get(name) if isinstance(item, Mapping) else None
            if found is not None:
                projected.append(found)
        return _Projection(projected)
    if isinstance(value, Mapping):
        return value.get(name)
    return None


def _evaluate_ast(node: Any, root: Any, current: Any) -> Any:
    kind = node[0]
    if kind == "literal":
        return node[1]
    if kind == "current":
        return current
    if kind == "field":
        return _field(_evaluate_ast(node[1], root, current), node[2])
    if kind == "index":
        value = _unwrap(_evaluate_ast(node[1], root, current))
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            return None
        try:
            return value[node[2]]
        except IndexError:
            return None
    if kind == "project":
        value = _unwrap(_evaluate_ast(node[1], root, current))
        return _Projection(list(value) if isinstance(value, list) else [])
    if kind == "flatten":
        value = _unwrap(_evaluate_ast(node[1], root, current))
        if not isinstance(value, list):
            return None
        flattened = []
        for item in value:
            flattened.extend(item if isinstance(item, list) else [item])
        return _Projection(flattened)
    if kind == "or":
        left = _unwrap(_evaluate_ast(node[1], root, current))
        return left if _jmes_truthy(left) else _evaluate_ast(node[2], root, current)
    if kind == "and":
        left = _unwrap(_evaluate_ast(node[1], root, current))
        return _evaluate_ast(node[2], root, current) if _jmes_truthy(left) else left
    if kind == "not":
        return not _jmes_truthy(_unwrap(_evaluate_ast(node[1], root, current)))
    if kind == "compare":
        left = _unwrap(_evaluate_ast(node[2], root, current))
        right = _unwrap(_evaluate_ast(node[3], root, current))
        operator = node[1]
        if operator == "==":
            return _jmes_equal(left, right)
        if operator == "!=":
            return not _jmes_equal(left, right)
        # jmespath.py returns null for ordering against null or booleans.
        if left is None or right is None or isinstance(left, bool) or isinstance(right, bool):
            return None
        try:
            return {
                ">": lambda: left > right,
                "<": lambda: left < right,
                ">=": lambda: left >= right,
                "<=": lambda: left <= right,
            }[operator]()
        except TypeError as exc:
            raise RestrictedJMESPathRuntimeError(
                f"incomparable values: {left!r} {operator} {right!r}"
            ) from exc
    if kind == "pipe":
        piped = _unwrap(_evaluate_ast(node[1], root, current))
        return _evaluate_ast(node[2], piped, piped)
    if kind == "function":
        name = node[1]
        args = [_unwrap(_evaluate_ast(arg, root, current)) for arg in node[2]]
        if name == "contains" and len(args) == 2:
            subject, needle = args
            if not isinstance(subject, (str, list)):
                raise RestrictedJMESPathRuntimeError(
                    "contains() subject is not a string or array"
                )
            return needle in subject
        if name == "join" and len(args) == 2:
            separator, values = args
            if not isinstance(separator, str) or not isinstance(values, list):
                raise RestrictedJMESPathRuntimeError(
                    "join() arguments have invalid types"
                )
            if any(not isinstance(value, str) for value in values):
                raise RestrictedJMESPathRuntimeError(
                    "join() array contains a non-string"
                )
            return separator.join(values)
        if name == "length" and len(args) == 1:
            value = args[0]
            if not isinstance(value, (str, list, dict)):
                raise RestrictedJMESPathRuntimeError(
                    "length() argument has invalid type"
                )
            return len(value)
        raise RestrictedJMESPathSyntaxError(
            f"unsupported function call: {name}/{len(args)}"
        )
    raise RestrictedJMESPathSyntaxError(f"unsupported AST node: {kind}")


def restricted_jmespath_search(expression: str, state: Mapping[str, Any]) -> Any:
    """Evaluate the restricted expression language used by pinned v2 tasks."""

    tree = _Parser(expression).parse()
    return _unwrap(_evaluate_ast(tree, state, state))


def _values_match(actual: Any, expected: Any) -> bool:
    """Match values with the pinned upstream evaluator's coercion rules."""

    if actual is None and expected is None:
        return True
    if actual is None or expected is None:
        return False
    if isinstance(expected, bool):
        if isinstance(actual, bool):
            return actual == expected
        if isinstance(actual, str) and actual.lower() in {"true", "false"}:
            return actual.lower() == str(expected).lower()
    if (
        isinstance(expected, (int, float))
        and not isinstance(expected, bool)
        and isinstance(actual, (int, float))
        and not isinstance(actual, bool)
    ):
        return abs(actual - expected) < 1e-9
    if isinstance(expected, str) and isinstance(actual, str):
        return actual == expected
    if isinstance(expected, list) and isinstance(actual, list):
        return len(actual) == len(expected) and all(
            _values_match(actual_value, expected_value)
            for actual_value, expected_value in zip(actual, expected, strict=True)
        )
    if isinstance(expected, dict) and isinstance(actual, dict):
        return set(actual) == set(expected) and all(
            _values_match(actual[key], expected_value)
            for key, expected_value in expected.items()
        )
    return actual == expected


def evaluate_deterministic_check(
    check: Mapping[str, Any], state: Mapping[str, Any]
) -> dict[str, Any]:
    """Evaluate one upstream JMESPath check and return JSON-safe detail."""

    if check.get("type") != "jmespath":
        raise HealthAdminScoringError("deterministic evaluator only accepts jmespath checks")
    query = check.get("query")
    if not isinstance(query, str) or not query:
        return {"passed": False, "actual": None, "error": "missing query"}
    try:
        actual = restricted_jmespath_search(query, state)
    except RestrictedJMESPathRuntimeError as exc:
        return {"passed": False, "actual": None, "error": str(exc)}

    expected = check.get("expected_value")
    contains = check.get("contains_value")
    if expected is not None:
        passed = _values_match(actual, expected)
    elif contains is not None:
        try:
            passed = contains in actual
        except TypeError:
            passed = False
    else:
        return {
            "passed": False,
            "actual": actual,
            "error": "check has neither expected_value nor contains_value",
        }
    return {"passed": bool(passed), "actual": actual, "error": None}


def score_task_deterministic(
    task: Mapping[str, Any],
    final_state: Mapping[str, Any],
) -> dict[str, Any]:
    """Score deterministic checks without serializing gold or patient values."""

    deterministic = {"process": [0, 0, 0.0, 0.0], "outcome": [0, 0, 0.0, 0.0]}
    judge_count = 0
    for check in task["evals"]:
        check_class = classify_check(check)
        if check.get("type") == "llm_judge":
            judge_count += 1
            continue
        if check.get("type") != "jmespath":
            raise HealthAdminScoringError(
                f"unsupported pinned check type: {check.get('type')!r}"
            )
        result = evaluate_deterministic_check(check, final_state)
        points = float(check.get("points", 0.0))
        bucket = deterministic[check_class]
        bucket[0] += 1
        bucket[1] += int(result["passed"])
        bucket[2] += points
        bucket[3] += points if result["passed"] else 0.0

    def summarize(values: list[float]) -> dict[str, Any]:
        count, passed, points, passed_points = values
        return {
            "check_count": int(count),
            "passed_check_count": int(passed),
            "check_pass_rate": passed / count if count else None,
            "points": points,
            "passed_points": passed_points,
            "point_pass_rate": passed_points / points if points else None,
            "strict_pass": bool(count) and passed == count,
        }

    process = summarize(deterministic["process"])
    outcome = summarize(deterministic["outcome"])
    total_count = process["check_count"] + outcome["check_count"]
    total_passed = process["passed_check_count"] + outcome["passed_check_count"]
    return {
        "primary_metric": "strict_deterministic_outcome_pass",
        "outcome": outcome,
        "process": process,
        "all_deterministic": {
            "check_count": total_count,
            "passed_check_count": total_passed,
            "check_pass_rate": total_passed / total_count if total_count else None,
            "strict_pass": bool(total_count) and total_passed == total_count,
        },
        "llm_judge": {
            "check_count": judge_count,
            "invoked": False,
            "status": "unresolved",
        },
        "official_all_check_strict_pass": None,
    }
