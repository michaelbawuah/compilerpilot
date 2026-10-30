"""Measured-build performance prediction and surrogate-guided proposals.

Only compiler choices are features. Optimization remarks and counters are not
used to rank unbuilt candidates: those values are unavailable before building.
"""
from __future__ import annotations

import math
import statistics
from typing import Any, Iterable

from compilerpilot.space import Candidate, sample_candidate
from compilerpilot.strategies.random_search import RandomSearch


def _positive(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if value > 0 and math.isfinite(value) else None


def _measurements(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One sample per candidate; aggregate repeats to avoid validation leakage."""
    groups: dict[str, dict[str, Any]] = {}
    for row in history:
        if row.get("correctness") is not True or row.get("measurement_valid", True) is not True:
            continue
        throughput = _positive(row.get("throughput"))
        baseline = _positive(row.get("baseline_throughput"))
        speedup = _positive(row.get("speedup"))
        if throughput is None:
            continue
        if baseline is not None:
            speedup = throughput / baseline
        if speedup is None:
            continue
        try:
            candidate = Candidate.from_dict(row["candidate"])
        except (KeyError, TypeError, ValueError):
            continue
        group = groups.setdefault(candidate.hash(), {"candidate": candidate, "speedups": [], "compile_s": []})
        group["speedups"].append(speedup)
        compile_s = _positive(row.get("compile_s"))
        if compile_s is not None:
            group["compile_s"].append(compile_s)
    return [{"candidate": g["candidate"], "speedup": statistics.median(g["speedups"]),
             "compile_s": statistics.median(g["compile_s"]) if g["compile_s"] else None}
            for _, g in sorted(groups.items())]


def _features(candidates: list[Candidate]) -> list[dict[str, str]]:
    # Treat every knob as categorical, including booleans and numeric thresholds.
    return [{key: str(value) for key, value in c.to_dict().items()} for c in candidates]


def _pipeline(seed: int = 0):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.pipeline import make_pipeline

    return make_pipeline(DictVectorizer(sparse=False), RandomForestRegressor(
        n_estimators=128, min_samples_leaf=1, max_features=1.0,
        random_state=seed, n_jobs=1))


def predict(candidates: list[Candidate], history: list[dict[str, Any]], *, seed: int = 0) -> list[dict[str, Any]]:
    """Predict speedup ratios (1.05 means +5%) from correct measured builds.

    With fewer than two unique measurements, return explicit unavailable values;
    callers must explore rather than mistake an untrained model for evidence.
    """
    measured = _measurements(history)
    predictions: list[dict[str, Any]] = [{"candidate": c.to_dict(), "predicted_speedup": None,
                                        "predicted_compile_s": None, "training_samples": len(measured)}
                                       for c in candidates]
    if len(measured) < 2 or not candidates:
        return predictions
    import numpy as np

    x = _features([row["candidate"] for row in measured])
    model = _pipeline(seed)
    model.fit(x, np.log([row["speedup"] for row in measured]))
    estimates = np.exp(model.predict(_features(candidates)))
    for row, estimate in zip(predictions, estimates):
        row["predicted_speedup"] = float(estimate)
    timed = [row for row in measured if row["compile_s"] is not None]
    if len(timed) >= 2:
        compile_model = _pipeline(seed)
        compile_model.fit(_features([row["candidate"] for row in timed]), np.log([row["compile_s"] for row in timed]))
        for row, estimate in zip(predictions, np.exp(compile_model.predict(_features(candidates)))):
            row["predicted_compile_s"] = float(estimate)
    return predictions


def cross_validate(history: list[dict[str, Any]], *, seed: int = 0) -> dict[str, Any]:
    """Out-of-fold validation, with duplicates merged before splitting.

    MAE is in percentage points of baseline speedup, not relative error against
    an arbitrary training score. Every prediction is from an unseen candidate.
    """
    measured = _measurements(history)
    result: dict[str, Any] = {"samples": len(measured), "folds": 0,
                              "mae_speedup_pct": None, "spearman": None,
                              "predictions": [], "method": "deduplicated out-of-fold"}
    if len(measured) < 3:
        result["reason"] = "at least three distinct correct measured candidates required"
        return result
    import numpy as np
    from scipy.stats import spearmanr
    from sklearn.model_selection import KFold, LeaveOneOut

    x = _features([row["candidate"] for row in measured])
    actual = np.asarray([row["speedup"] for row in measured])
    estimates = np.zeros(len(measured))
    cv = LeaveOneOut() if len(measured) < 5 else KFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (train, test) in enumerate(cv.split(x), 1):
        model = _pipeline(seed + fold)
        model.fit([x[i] for i in train], np.log(actual[train]))
        estimates[test] = np.exp(model.predict([x[i] for i in test]))
        result["folds"] = fold
    result["mae_speedup_pct"] = float(np.mean(np.abs(estimates - actual)) * 100)
    if np.ptp(actual) > 0 and np.ptp(estimates) > 0:
        correlation = float(spearmanr(actual, estimates).statistic)
        result["spearman"] = correlation if math.isfinite(correlation) else None
    result["predictions"] = [{"candidate": row["candidate"].to_dict(), "measured_speedup": float(a),
                               "predicted_speedup": float(p)}
                              for row, a, p in zip(measured, actual, estimates)]
    return result


class SurrogateSearch(RandomSearch):
    """Explore eight builds, then rank a fresh pool with a random forest."""

    def __init__(self, seed: int = 0, compilers: Iterable[str] = ("gcc", "clang"),
                 initial_random: int = 8, pool_size: int = 256):
        super().__init__(seed, compilers)
        if initial_random < 8 or pool_size < 1:
            raise ValueError("initial_random must be >=8 and pool_size must be positive")
        self.seed = seed
        self.initial_random = initial_random
        self.pool_size = pool_size

    def propose(self, history: list[dict[str, Any]], seen: set[str]) -> Candidate | None:
        measured = _measurements(history)
        if len(measured) < self.initial_random:
            return super().propose(history, seen)
        excluded = set(seen) | self._proposed
        for row in history:
            try:
                excluded.add(Candidate.from_dict(row["candidate"]).hash())
            except (KeyError, TypeError, ValueError):
                continue
        pool: dict[str, Candidate] = {}
        for _ in range(self.pool_size * 20):
            c = sample_candidate(self.rng, self.compilers)
            if c.hash() not in excluded:
                pool[c.hash()] = c
            if len(pool) >= self.pool_size:
                break
        if not pool:
            return None
        predictions = predict(list(pool.values()), history, seed=self.seed)
        best = max(predictions, key=lambda row: (row["predicted_speedup"], Candidate.from_dict(row["candidate"]).hash()))
        candidate = Candidate.from_dict(best["candidate"])
        self._proposed.add(candidate.hash())
        return candidate

    def predict(self, candidates: list[Candidate], history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return predict(candidates, history, seed=self.seed)
