"""Question validation and answer formatting (stdlib only).

This module has no third-party imports so the unit tests run on plain Linux
without NPU hardware, models, or the Ryzen AI environment.
"""
from __future__ import annotations

from typing import Any

QUESTION_TYPES = ("noul", "choice", "score")

_ALLOWED_FIELDS = {"type", "instructions", "criteria", "threshold", "invert"}

DEFAULT_NOUL_THRESHOLD = 0.5
DEFAULT_NOUL_INVERT = False


def validate_questions(questions: Any) -> dict[str, dict[str, Any]]:
    """Validate the official JSON question form and fill noul defaults.

    Returns a normalized copy: each noul entry carries ``threshold`` (float)
    and ``invert`` (bool). Raises ValueError on any invalid input.
    """
    if not isinstance(questions, dict) or not questions:
        raise ValueError("questions must be a non-empty object")
    out: dict[str, dict[str, Any]] = {}
    for key, raw in questions.items():
        if not isinstance(key, str) or not key:
            raise ValueError("question keys must be non-empty strings")
        if not isinstance(raw, dict):
            raise ValueError(f"question '{key}' must be an object")
        unknown = set(raw) - _ALLOWED_FIELDS
        if unknown:
            raise ValueError(
                f"question '{key}' has unknown field(s): {sorted(unknown)}"
            )
        qtype = raw.get("type")
        if qtype not in QUESTION_TYPES:
            raise ValueError(
                f"question '{key}' has invalid type {qtype!r} "
                "(expected 'noul', 'choice' or 'score')"
            )
        instructions = raw.get("instructions")
        if not isinstance(instructions, str) or not instructions.strip():
            raise ValueError(f"question '{key}' needs non-empty 'instructions'")
        criteria = raw.get("criteria")
        if qtype == "choice":
            if not isinstance(criteria, dict) or len(criteria) < 2:
                raise ValueError(
                    f"question '{key}' needs 'criteria' as an object "
                    "with at least 2 options"
                )
            for label, desc in criteria.items():
                if not isinstance(label, str) or not label.strip():
                    raise ValueError(
                        f"question '{key}' has an invalid option label"
                    )
                if desc is not None and not isinstance(desc, str):
                    raise ValueError(
                        f"question '{key}' option '{label}' description "
                        "must be a string or null"
                    )
        elif qtype == "score":
            if (
                not isinstance(criteria, list)
                or not 2 <= len(criteria) <= 10
            ):
                raise ValueError(
                    f"question '{key}' needs 'criteria' as a list "
                    "of 2 to 10 level names"
                )
            for level in criteria:
                if not isinstance(level, str) or not level.strip():
                    raise ValueError(
                        f"question '{key}' has an invalid score level name"
                    )
        else:  # noul
            if criteria is not None and not isinstance(criteria, dict):
                raise ValueError(
                    f"question '{key}' criteria must be an object or omitted"
                )
        threshold = raw.get("threshold", DEFAULT_NOUL_THRESHOLD)
        invert = raw.get("invert", DEFAULT_NOUL_INVERT)
        if qtype != "noul" and ("threshold" in raw or "invert" in raw):
            raise ValueError(
                f"question '{key}': 'threshold' and 'invert' "
                "are only allowed for noul questions"
            )
        if qtype == "noul":
            if (
                not isinstance(threshold, (int, float))
                or isinstance(threshold, bool)
                or not 0.0 <= float(threshold) <= 1.0
            ):
                raise ValueError(
                    f"question '{key}' threshold must be a number in [0, 1]"
                )
            if not isinstance(invert, bool):
                raise ValueError(f"question '{key}' invert must be true/false")
            threshold = float(threshold)
        entry: dict[str, Any] = {
            "type": qtype,
            "instructions": instructions,
        }
        if criteria is not None:
            entry["criteria"] = criteria
        if qtype == "noul":
            entry["threshold"] = threshold
            entry["invert"] = invert
        out[key] = entry
    return out


def validate_state(state: Any) -> str | list[str]:
    """Accept a text state (str or list of str), reject anything else."""
    if isinstance(state, str):
        return state
    if isinstance(state, list) and all(isinstance(s, str) for s in state):
        return list(state)
    raise ValueError("state must be a string or a list of strings")


def check_seq_len(past_len: int, ids_len: int, max_seq_len: int) -> None:
    """Raise ValueError when a prompt exceeds the configured NPU maximum."""
    if past_len + ids_len > max_seq_len:
        raise ValueError(
            f"prompt exceeds configured NPU maximum {max_seq_len} "
            f"({past_len + ids_len} tokens)"
        )


def format_answer(
    qtype: str,
    labels: list[str],
    probs: list[float],
    threshold: float = DEFAULT_NOUL_THRESHOLD,
    invert: bool = False,
    legend: list[str] | None = None,
) -> dict[str, Any]:
    """Format one answer as the checkpoint's ``api.answer`` does; noul adds ``decision``.

    ``legend`` is a score question's level names, lowest first.
    """
    if len(labels) != len(probs):
        raise ValueError("labels and probabilities length mismatch")
    if not probs:
        raise ValueError("empty probability list")
    probs = [float(p) for p in probs]
    if qtype == "noul":
        if len(probs) != 2:
            raise ValueError("noul needs exactly 2 probabilities (yes, no)")
        p_yes = probs[0]
        if invert:
            p_yes = 1.0 - p_yes
        return {
            "type": "noul",
            "noul": p_yes,
            "decision": bool(p_yes >= threshold),
        }
    best = max(range(len(probs)), key=probs.__getitem__)
    if qtype == "choice":
        return {
            "type": "choice",
            "choice": labels[best],
            "confidence": probs[best],
            "probabilities": dict(zip(labels, probs)),
        }
    if qtype == "score":
        level = float(sum(i * p for i, p in enumerate(probs)))
        return {
            "type": "score",
            "score": level,
            "confidence": probs[best],
            "probabilities": {str(i): p for i, p in enumerate(probs)},
            "legend": {str(i): text for i, text in enumerate(legend or labels)},
        }
    raise ValueError(f"unknown question type {qtype!r}")
