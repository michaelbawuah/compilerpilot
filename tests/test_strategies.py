"""Search reproducibility and safeguards around learned fitness."""
from dataclasses import replace
import math
import random

import pytest

from compilerpilot.space import Candidate, sample_candidate
from compilerpilot.strategies.random_search import RandomSearch
from compilerpilot.strategies.genetic import GeneticSearch, _fitness
from compilerpilot.strategies.surrogate import SurrogateSearch, _measurements, predict, cross_validate


def measurement(candidate, throughput=105, **changes):
    return {"candidate": candidate.to_dict(), "correctness": True,
            "throughput": throughput, "baseline_throughput": 100,
            "speedup": throughput / 100 if isinstance(throughput, (int, float)) else 1.0,
            "compile_s": 2.0, **changes}


def measured_history(n=8):
    rng = random.Random(20)
    unique = {}
    while len(unique) < n:
        candidate = sample_candidate(rng, ("gcc",))
        unique[candidate.hash()] = candidate
    return [measurement(candidate, 100 + i) for i, candidate in enumerate(unique.values())]


@pytest.mark.parametrize("strategy", [RandomSearch, GeneticSearch, SurrogateSearch])
def test_same_seed_same_proposals_without_duplicates(strategy):
    one, two = strategy(seed=9, compilers=("gcc",)), strategy(seed=9, compilers=("gcc",))
    seen = set()
    history = []
    for _ in range(12):
        a, b = one.propose(history, seen), two.propose(history, seen)
        assert a == b
        assert a.hash() not in seen
        assert a.compiler == "gcc"
        seen.add(a.hash())
        history.append(measurement(a))


def test_failed_history_is_not_reproposed_but_never_fitness():
    candidate = Candidate()
    row = measurement(candidate, throughput=1e15, correctness=False)
    assert _fitness(row) is None
    assert _measurements([row]) == []
    proposed = RandomSearch(seed=2).propose([row], set())
    assert proposed.hash() != candidate.hash()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -3, 0, True, None])
def test_invalid_throughput_is_not_fitness(value):
    row = measurement(Candidate(), throughput=value)
    assert _fitness(row) is None
    assert _measurements([row]) == []


def test_model_deduplicates_candidates_before_folds():
    rows = measured_history(6)
    rows.append({**rows[0], "throughput": 110, "speedup": 1.1})
    rows.append(measurement(replace(Candidate(), opt="O2"), throughput=1e15, correctness=False))
    measured = _measurements(rows)
    assert len(measured) == 6
    result = cross_validate(rows)
    assert result["samples"] == 6
    assert result["folds"] == 5
    assert len(result["predictions"]) == 6
    assert math.isfinite(result["mae_speedup_pct"])


def test_tiny_history_is_explicitly_untrained():
    candidate = Candidate()
    result = predict([candidate], [measurement(candidate)])
    assert result[0]["predicted_speedup"] is None
    assert result[0]["training_samples"] == 1
    assert cross_validate([measurement(candidate)])["mae_speedup_pct"] is None


def test_failed_outlier_cannot_change_prediction():
    history = measured_history(8)
    candidate = Candidate()
    expected = predict([candidate], history)
    failed = measurement(replace(candidate, opt="O2"), throughput=1e15, correctness=False)
    assert predict([candidate], history + [failed]) == expected


def test_constant_targets_have_no_rank_correlation():
    history = [{**row, "throughput": 100, "speedup": 1.0} for row in measured_history(5)]
    result = cross_validate(history)
    assert result["spearman"] is None
    assert result["mae_speedup_pct"] == 0


def test_surrogate_and_genetic_only_return_valid_unseen_candidates():
    history = measured_history(10)
    seen = {Candidate.from_dict(row["candidate"]).hash() for row in history}
    for strategy in (GeneticSearch(seed=5, compilers=("gcc",)), SurrogateSearch(seed=5, compilers=("gcc",), pool_size=24)):
        candidate = strategy.propose(history, seen)
        assert candidate.hash() not in seen
        assert Candidate.from_dict(candidate.to_dict()) == candidate


def test_genetic_failed_outlier_does_not_affect_parent_choice():
    history = measured_history(8)
    failed = measurement(Candidate(), throughput=1e15, correctness=False)
    # Explicit seen is held equal so only parent choice could affect the result.
    seen = {Candidate().hash()}
    a = GeneticSearch(seed=13, compilers=("gcc",)).propose(history, seen)
    b = GeneticSearch(seed=13, compilers=("gcc",)).propose(history + [failed], seen)
    assert a == b


def test_noisy_outlier_is_not_used_for_fitness_or_training():
    noisy = measurement(Candidate(), throughput=1e15, measurement_valid=False)
    assert _fitness(noisy) is None
    assert _measurements([noisy]) == []


def test_speedup_is_recomputed_from_measured_throughput_and_baseline():
    row = measurement(Candidate(), throughput=105, speedup=99999)
    assert _measurements([row])[0]["speedup"] == 1.05
