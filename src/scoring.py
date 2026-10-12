"""Deterministic answer checks: every expected string must appear in the answer (numbers match as whole numbers)."""
import re


def _norm(text: str) -> str:
    return text.lower().replace(",", "").replace("$", "")


def contains(answer: str, expected: str) -> bool:
    a, e = _norm(answer), _norm(expected)
    if re.fullmatch(r"-?\d+(\.\d+)?", e):
        # whole-number match: "15" must not match inside "150", "15.5" or "2026-10-15", but "15875.5" matches "15,875.50"
        zeros = "0*" if "." in e else r"(?:\.0+)?"
        return re.search(rf"(?<![\d.-]){re.escape(e)}{zeros}(?!\d|\.\d)", a) is not None
    return e in a


def answer_ok(task: dict, answer: str | None) -> bool:
    if not answer:
        return False
    if not all(contains(answer, e) for e in task["expect"]):
        return False
    return not task["expect_any"] or any(contains(answer, e) for e in task["expect_any"])
