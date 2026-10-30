"""Typed capabilities exposed to the model; arbitrary commands are never accepted."""
from __future__ import annotations

from copy import deepcopy
import json
import math
import re
from typing import Any

from .space import Candidate, KNOBS


def _object(properties: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


def _knob_schema(values: list[Any]) -> dict[str, Any]:
    kinds = {type(value) for value in values}
    if len(kinds) > 1:
        return {"anyOf": [{"type": "string", "enum": [v for v in values if type(v) is str]},
                           {"type": "integer", "enum": [v for v in values if type(v) is int]}]}
    kind = {str: "string", int: "integer", bool: "boolean"}[type(values[0])]
    return {"type": kind, "enum": values}


CANDIDATE_SCHEMA = _object({key: _knob_schema(values) for key, values in KNOBS.items()})
_BUILD_ID = {"type": "string", "minLength": 1, "maxLength": 128,
             "pattern": r"^[A-Za-z0-9_.-]+$"}
_TOOL_SPECS = {
    "list_knobs": ("List all allowed compiler choices and flag values.", _object({})),
    "build": ("Compile one candidate. This consumes one unique build from the budget, even if compilation fails.",
              _object({"candidate": CANDIDATE_SCHEMA})),
    "run_benchmark": ("Measure a compiled build and enforce reference-checksum and differential correctness gates.",
                      _object({"build_id": _BUILD_ID})),
    "get_compiler_feedback": ("Read compact optimization remarks for Ledger's own source files.",
                              _object({"build_id": _BUILD_ID})),
    "get_counters": ("Read perf hardware counters, or an explicit unavailable reason.",
                     _object({"build_id": _BUILD_ID})),
    "query_history": ("Get measured trial history; failed candidates are never eligible recommendations.",
                      _object({"top_k": {"type": "integer", "minimum": 1, "maximum": 100},
                               "sort_by": {"type": "string", "enum": ["throughput", "compile_s", "p99_ns", "speedup"]}})),
    "predict": ("Rank candidates with the learned surrogate; predictions are not measurements.",
                _object({"candidates": {"type": "array", "items": CANDIDATE_SCHEMA,
                                         "minItems": 1, "maxItems": 64}})),
    "finish": ("Recommend a build that passed correctness, with evidence and limitations. The harness makes the eligibility decision.",
               _object({"recommendation": _BUILD_ID,
                        "rationale": {"type": "string", "minLength": 1, "maxLength": 8000}})),
}


def _validate(value: Any, schema: dict[str, Any], path: str = "arguments") -> None:
    """Validate the finite schema subset we publish, including strict scalar types."""
    if "anyOf" in schema:
        for choice in schema["anyOf"]:
            try:
                _validate(value, choice, path)
                return
            except ValueError:
                pass
        raise ValueError(f"{path}: value does not match an allowed type/value")
    kind = schema.get("type")
    expected = {"object": dict, "array": list, "string": str, "integer": int,
                "number": (int, float), "boolean": bool}.get(kind)
    if expected is not None and (type(value) not in expected if isinstance(expected, tuple) else type(value) is not expected):
        raise ValueError(f"{path}: expected {kind}")
    if "enum" in schema and not any(type(value) is type(item) and value == item for item in schema["enum"]):
        raise ValueError(f"{path}: value is outside the allow-list")
    if kind == "object":
        extra = set(value) - set(schema["properties"])
        missing = set(schema["required"]) - set(value)
        if extra or missing:
            raise ValueError(f"{path}: unknown fields {sorted(extra)}; missing fields {sorted(missing)}")
        for key, item in value.items():
            _validate(item, schema["properties"][key], f"{path}.{key}")
    elif kind == "array":
        if not schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", math.inf):
            raise ValueError(f"{path}: array length is outside bounds")
        for index, item in enumerate(value):
            _validate(item, schema["items"], f"{path}[{index}]")
    elif kind == "string":
        if not schema.get("minLength", 0) <= len(value) <= schema.get("maxLength", math.inf):
            raise ValueError(f"{path}: string length is outside bounds")
        if "pattern" in schema and re.fullmatch(schema["pattern"], value) is None:
            raise ValueError(f"{path}: invalid identifier")
    elif kind in ("integer", "number"):
        if not math.isfinite(value) or not schema.get("minimum", -math.inf) <= value <= schema.get("maximum", math.inf):
            raise ValueError(f"{path}: number is outside bounds")


class ToolDispatcher:
    """Execute only named harness methods after local validation and budget checks."""

    def __init__(self, harness: Any, budget: int | None = None):
        if budget is not None and (type(budget) is not int or budget < 1):
            raise ValueError("budget must be a positive integer")
        self.harness = harness
        self.budget = budget
        used = getattr(harness, 'builds_used', 0)
        self.builds_used = used if type(used) is int else 0
        self.seen: set[str] = set(getattr(harness, 'seen', set()))
        self.finished = False
        self.recommendation: Any = None

    def definitions(self) -> list[dict[str, Any]]:
        return [{"type": "function", "function": {"name": name, "description": description,
                 "parameters": deepcopy(schema), "strict": True}}
                for name, (description, schema) in _TOOL_SPECS.items()]

    def dispatch(self, name: str, arguments: dict[str, Any]) -> Any:
        if self.finished:
            raise ValueError("The agent run has already finished")
        if name not in _TOOL_SPECS:
            raise ValueError(f"Unknown tool: {name}")
        _validate(arguments, _TOOL_SPECS[name][1])
        if name == "build":
            candidate = Candidate.from_dict(arguments["candidate"])
            fingerprint = candidate.hash()
            if fingerprint in self.seen:
                raise ValueError("Duplicate candidate: choose a different build")
            if self.budget is not None and self.builds_used >= self.budget:
                raise ValueError("Build budget exhausted; benchmark existing builds and finish")
        if name == 'finish' and self.budget is not None and self.builds_used < self.budget:
            raise ValueError('Use the full requested build budget before finish')
        # Harness implementations also enforce budget, source identity and recommendation eligibility.
        try:
            result = getattr(self.harness, name)(**arguments)
        except Exception:
            actual = getattr(self.harness, 'builds_used', None)
            if name == 'build' and type(actual) is int:
                self.builds_used = actual
                self.seen.update(getattr(self.harness, 'seen', set()))
            raise
        if name == 'build':
            self.seen.add(fingerprint)
            actual = getattr(self.harness, 'builds_used', None)
            self.builds_used = actual if type(actual) is int else self.builds_used + 1
        if name == "finish":
            self.finished = True
            self.recommendation = result
        return result
