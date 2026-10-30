"""Compact, source-filtered compiler and perf feedback; no LLM parsing needed."""
from __future__ import annotations

import math
from pathlib import Path
import re
from typing import Any

import yaml


class _RemarkLoader(yaml.SafeLoader):
    pass


def _remark(loader: _RemarkLoader, tag: str, node: yaml.Node) -> dict[str, Any]:
    value = loader.construct_mapping(node, deep=True)
    value["remark_type"] = tag
    return value


_RemarkLoader.add_multi_constructor("!", _remark)


def _own_source(value: str) -> bool:
    path = value.replace("\\", "/")
    return bool(re.search(r"(?:^|/)src/[^/]+|(?:^|/)include/ledger/", path))


def parse_size(text: str) -> int:
    """Sum executable text sections from GNU/LLVM `size -A` output, in bytes."""
    total = 0
    found = False
    for line in text.splitlines():
        columns = line.split()
        if len(columns) >= 2 and (columns[0] == ".text" or columns[0].startswith(".text.")):
            try:
                total += int(columns[1], 0)
            except ValueError as error:
                raise ValueError(f"Malformed text section size: {line}") from error
            found = True
    if not found:
        raise ValueError("No .text section in size output")
    return total


_COUNTERS = {
    "cycles": "cycles", "instructions": "instructions", "branch-misses": "branch_misses",
    "branches": "branches", "L1-icache-load-misses": "icache_misses",
}


def parse_perf(text: str) -> dict[str, Any]:
    """Read perf stat default, `-x ;`, or `-x ,` output, retaining missing values."""
    values: dict[str, Any] = {name: None for name in _COUNTERS.values()}
    unavailable: list[str] = []
    for line in text.splitlines():
        event = next((name for name in _COUNTERS if re.search(rf"(?<![\w-]){re.escape(name)}(?:[:\w]*)?(?![\w-])", line)), None)
        if event is None:
            continue
        key = _COUNTERS[event]
        if "<not " in line or "not supported" in line or "not counted" in line:
            unavailable.append(event)
            continue
        # perf CSV has value, unit, event; default text starts with value.
        first = line.strip().split(";", 1)[0] if ";" in line else line.strip().split()[0]
        if ";" not in line and ",," in line:
            first = line.split(",", 1)[0]
        first = first.replace(",", "")
        try:
            value = float(first)
        except ValueError:
            unavailable.append(event)
            continue
        if not math.isfinite(value) or value < 0:
            unavailable.append(event)
            continue
        values[key] = int(value) if value.is_integer() else value
    values["ipc"] = (values["instructions"] / values["cycles"]
                     if values["instructions"] is not None and values["cycles"] else None)
    values["branch_miss_pct"] = (100 * values["branch_misses"] / values["branches"]
                                 if values["branch_misses"] is not None and values["branches"] else None)
    values["available"] = any(value is not None for name, value in values.items() if name in _COUNTERS.values())
    values["unavailable_events"] = sorted(set(unavailable))
    if not values["available"]:
        values["reason"] = text.strip()[:500] or "perf output unavailable"
    return values


def parse_optimization_records(text: str) -> dict[str, Any]:
    """Parse Clang's tagged YAML stream; include Ledger-owned missed remarks only."""
    missed_inline: list[dict[str, Any]] = []
    missed_vectorize: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        for record in yaml.load_all(text, Loader=_RemarkLoader):
            if not isinstance(record, dict) or record.get("remark_type") != "Missed":
                continue
            location = record.get("DebugLoc") or {}
            filename = str(location.get("File", ""))
            if not _own_source(filename):
                continue
            pass_name = str(record.get("Pass", ""))
            args = record.get("Args") or []
            reason = "".join(str(value) for arg in args if isinstance(arg, dict)
                             for key, value in arg.items() if key != "DebugLoc")[:500]
            item = {"fn": str(record.get("Function", "")), "reason": reason,
                    "file": filename, "line": location.get("Line"), "pass": pass_name,
                    "hotness": record.get("Hotness", 0) or 0}
            if "inline" in pass_name:
                missed_inline.append(item)
            if "vector" in pass_name:
                missed_vectorize.append(item)
    except (yaml.YAMLError, TypeError, ValueError) as error:
        errors.append(str(error)[:300])
    def hottest(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(items, key=lambda item: float(item["hotness"]), reverse=True)[:10]
    return {"missed_inline": hottest(missed_inline), "missed_inline_count": len(missed_inline),
            "missed_vectorize": len(missed_vectorize), "vectorization_remarks": hottest(missed_vectorize),
            "parse_errors": errors}


def parse_gcc_feedback(text: str) -> dict[str, Any]:
    inline: list[dict[str, Any]] = []
    vectorize: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not _own_source(line) or "missed:" not in line:
            continue
        reason = line.split("missed:", 1)[1].strip()
        item = {"fn": "", "reason": reason[:500], "location": line.split("missed:", 1)[0].strip()}
        if "inlin" in reason.lower():
            inline.append(item)
        if "vector" in reason.lower():
            vectorize.append(item)
    return {"missed_inline": inline[:10], "missed_inline_count": len(inline),
            "missed_vectorize": len(vectorize), "vectorization_remarks": vectorize[:10]}


def parse_hot_functions(text: str) -> list[list[Any]]:
    """Extract top Ledger symbols from `perf report --stdio --no-children` text."""
    result: list[list[Any]] = []
    for line in text.splitlines():
        match = re.match(r"\s*([\d.]+)%\s+.*?\[\.\]\s+(.+)$", line)
        if match and ("ledger::" in match[2] or "_ZN6ledger" in match[2]):
            result.append([match[2].strip(), float(match[1])])
    return sorted(result, key=lambda item: item[1], reverse=True)[:10]


def summarize_feedback(build_dir: Path) -> dict[str, Any]:
    directory = Path(build_dir)
    result: dict[str, Any] = {"build": directory.name, "missed_inline": [], "missed_inline_count": 0,
                              "missed_vectorize": 0, "vectorization_remarks": [], "parse_errors": [],
                              "top_fns": [], "remarks_available": False}
    for path in sorted(directory.rglob("*.opt.yaml")):
        parsed = parse_optimization_records(path.read_text(errors="replace"))
        result["remarks_available"] = True
        for name in ("missed_inline", "vectorization_remarks", "parse_errors"):
            result[name].extend(parsed[name])
        for name in ("missed_inline_count", "missed_vectorize"):
            result[name] += parsed[name]
    gcc_paths = sorted(set(directory.rglob("*missed*.txt")) | set(directory.rglob("*opt-info*.txt")))
    if not gcc_paths and (directory / "build.log").is_file():
        gcc_paths = [directory / "build.log"]
    for path in gcc_paths:
        raw = path.read_text(errors="replace")
        result["remarks_available"] = result["remarks_available"] or "missed:" in raw
        parsed = parse_gcc_feedback(raw)
        for name in ("missed_inline", "vectorization_remarks"):
            result[name].extend(parsed[name])
        for name in ("missed_inline_count", "missed_vectorize"):
            result[name] += parsed[name]
    for name in ("missed_inline", "vectorization_remarks"):
        result[name] = sorted(result[name], key=lambda item: float(item.get("hotness", 0)), reverse=True)[:10]
    perf_paths = sorted(directory.glob("*perf*.txt"))
    if (directory / "counters" / "perf.log").is_file():
        perf_paths.insert(0, directory / "counters" / "perf.log")
    result["counters"] = parse_perf(perf_paths[0].read_text(errors="replace") if perf_paths else "")
    for path in directory.glob("*report*.txt"):
        result["top_fns"] = parse_hot_functions(path.read_text(errors="replace"))
        if result["top_fns"]:
            break
    result["ipc"] = result["counters"]["ipc"]
    result["branch_miss_pct"] = result["counters"]["branch_miss_pct"]
    if not result["remarks_available"]:
        result["remarks_reason"] = "No optimization records found"
    return result
