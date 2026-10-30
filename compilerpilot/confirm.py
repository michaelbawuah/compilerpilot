"""Interleaved confirmation, independent of the strategy that proposed a build."""
from __future__ import annotations

import math
import statistics
from typing import Any, Callable

from .build import BuildResult


def _valid_measurement(result: dict[str, Any]) -> bool:
    if result.get("correctness") is not True:
        return False
    for name in ("throughput", "p50_ns", "p99_ns", "cv"):
        value = result.get(name)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            return False
    return (result["throughput"] > 0 and 0 <= result["p50_ns"] <= result["p99_ns"]
            and result["cv"] >= 0)


def _median_observations(observations: list[dict[str, Any]]) -> dict[str, Any]:
    throughput = [result["throughput"] for result in observations]
    median = statistics.median(throughput)
    stdev = statistics.stdev(throughput) if len(throughput) > 1 else 0.0
    run_sigma = statistics.median(result["throughput"] * result["cv"] for result in observations)
    # Taking the larger estimate avoids counting dependent runs as independent evidence.
    sigma = max(stdev, run_sigma)
    return {"throughput": median,
            "p50_ns": statistics.median(result["p50_ns"] for result in observations),
            "p99_ns": statistics.median(result["p99_ns"] for result in observations),
            "cv": stdev / statistics.mean(throughput),
            "median_run_cv": statistics.median(result["cv"] for result in observations),
            "max_run_cv": max(result["cv"] for result in observations),
            "stdev": stdev, "noise_stdev": sigma,
            "spread": max(throughput) - min(throughput), "observations": len(observations)}


def _validation_passed(value: Any) -> bool:
    if not isinstance(value, dict) or value.get("correctness") is not True:
        return False
    # Full validation is owned by the runner. Explicit failures always override its summary.
    if any(value.get(key) != 0 for key in ("differential_exit_code", "ctest_exit_code")):
        return False
    for key in ("ctest", "differential", "ctest_passed", "differential_passed"):
        if key in value:
            status = value[key]
            if status is False:
                return False
            if isinstance(status, dict) and (status.get("ok") is False or status.get("returncode", 0) != 0):
                return False
    return True


def confirm_candidates(runner: Any, baseline: BuildResult, candidates: list[BuildResult],
                       rounds: int = 10, full_validation: bool = True,
                       callback: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    """Rerun A/B/C/baseline each round, then validate a significant measured winner.

    Coefficients of variation are fractions (0.02 means 2%). Observation callbacks
    permit an append-only trace before confirmation finishes. A recommendation is
    returned only after the final 100-seed differential and ctest gate passes.
    """
    if type(rounds) is not int or rounds < 2:
        raise ValueError("Confirmation requires at least two interleaved rounds")
    if not baseline.success:
        raise ValueError("Baseline build failed")
    selected: list[BuildResult] = []
    excluded: dict[str, str] = {}
    seen = {baseline.build_id}
    baseline_source = getattr(baseline, "provenance", {})
    for build in candidates:
        if build.build_id in seen:
            continue
        seen.add(build.build_id)
        if not build.success:
            excluded[build.build_id] = "build failed"
        elif any(baseline_source.get(key) and
                 getattr(build, "provenance", {}).get(key) != baseline_source[key]
                 for key in ("source_sha256", "ledger_commit")):
            excluded[build.build_id] = "source provenance differs from the baseline"
        elif len(selected) < 3:
            selected.append(build)
        else:
            excluded[build.build_id] = "only the top three supplied candidates are confirmed"
    output: dict[str, Any] = {"status": "measuring", "baseline": baseline.build_id,
                              "rounds": rounds, "observations": [], "medians": {},
                              "excluded": excluded, "recommendation": None,
                              "recommended_build": None, "winner": None,
                              "full_validation": None}
    by_build: dict[str, list[dict[str, Any]]] = {build.build_id: [] for build in [*selected, baseline]}
    failed: set[str] = set()
    baseline_input_hash: str | None = None
    for round_index in range(rounds):
        for build in [*selected, baseline]:
            if build.build_id in failed:
                continue
            label = f"confirm-{round_index:02d}-{build.build_id}"
            try:
                measured = runner.run(build, label=label)
            except Exception as error:
                measured = {"correctness": False, "error": str(error)[:1000]}
            observation = {**measured, "build_id": build.build_id, "round": round_index,
                           "label": label}
            output["observations"].append(observation)
            if callback is not None:
                callback(observation)
            if not _valid_measurement(measured):
                excluded[build.build_id] = "correctness or measurement failed during confirmation"
                failed.add(build.build_id)
                if build.build_id == baseline.build_id:
                    output.update(status="invalid_baseline", reason=excluded[build.build_id])
                    return output
                continue
            by_build[build.build_id].append(measured)
            if build.build_id == baseline.build_id:
                input_hash = measured.get("provenance", {}).get("input_sha256")
                if baseline_input_hash and input_hash != baseline_input_hash:
                    excluded[build.build_id] = "baseline evaluation input changed during confirmation"
                    output.update(status="invalid_baseline", reason=excluded[build.build_id])
                    return output
                baseline_input_hash = input_hash or baseline_input_hash
                if baseline_input_hash:
                    for candidate in selected:
                        records = by_build[candidate.build_id]
                        if any(record.get("provenance", {}).get("input_sha256") != baseline_input_hash
                               for record in records):
                            excluded[candidate.build_id] = "evaluation input provenance differs from the baseline"
                            failed.add(candidate.build_id)
    for build_id, observations in by_build.items():
        if build_id not in failed and observations:
            output["medians"][build_id] = _median_observations(observations)
    baseline_stats = output["medians"][baseline.build_id]
    baseline_results = by_build[baseline.build_id]
    split = rounds // 2
    first = statistics.median(item["throughput"] for item in baseline_results[:split])
    last = statistics.median(item["throughput"] for item in baseline_results[split:])
    drift = abs(last / first - 1.0)
    baseline_stats["drift"] = drift
    threshold = 3.0 * baseline_stats["noise_stdev"]
    output["threshold"] = {"absolute_throughput": threshold,
                            "relative": threshold / baseline_stats["throughput"],
                            "noise_multiplier": 3.0, "max_baseline_cv": 0.02,
                            "max_baseline_drift": 0.02, "max_candidate_cv": 0.02}
    for stats in output["medians"].values():
        stats["speedup"] = stats["throughput"] / baseline_stats["throughput"]
        stats["speedup_pct"] = 100 * (stats["speedup"] - 1.0)
        stats["significant"] = stats["throughput"] - baseline_stats["throughput"] > threshold
    for build in selected:
        stats = output["medians"].get(build.build_id)
        if stats and max(stats["cv"], stats["max_run_cv"]) >= 0.02:
            excluded[build.build_id] = "candidate variability exceeds the 2% gate"
            stats["significant"] = False
            failed.add(build.build_id)
    if max(baseline_stats["cv"], baseline_stats["max_run_cv"]) >= 0.02 or drift > 0.02:
        for stats in output["medians"].values():
            stats["significant"] = False
        output.update(status="noisy", reason="Baseline variability or drift exceeds the 2% gate")
        return output
    eligible = [build for build in selected if build.build_id not in failed
                and output["medians"][build.build_id]["significant"]]
    if not eligible:
        output.update(status="no_significant_improvement", reason="No correct candidate cleared 3x baseline noise")
        return output
    winner = max(eligible, key=lambda build: output["medians"][build.build_id]["throughput"])
    output["winner"] = winner.build_id
    if not full_validation:
        output.update(status="measured", reason="Full correctness validation has not run")
        return output
    try:
        validation = runner.full_validate(winner, seeds=100, orders=1000000)
    except Exception as error:
        validation = {"correctness": False, "error": str(error)[:1000]}
    output["full_validation"] = validation
    if not _validation_passed(validation):
        excluded[winner.build_id] = "final differential or ctest validation failed"
        output.update(status="validation_failed", reason=excluded[winner.build_id])
        return output
    output.update(status="confirmed", recommendation=winner.build_id,
                  recommended_build=winner.build_id)
    return output
