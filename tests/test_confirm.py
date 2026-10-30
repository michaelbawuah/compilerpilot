from dataclasses import dataclass, field

import pytest

from compilerpilot.confirm import confirm_candidates


@dataclass
class Build:
    build_id: str
    success: bool = True
    provenance: dict = field(default_factory=dict)


class Runner:
    def __init__(self, throughputs=None, cvs=None, failures=None, validation=None):
        self.throughputs = throughputs or {"base": 100.0, "A": 120.0, "B": 110.0, "C": 90.0}
        self.cvs = cvs or {}
        self.failures = failures or set()
        self.validation = validation if validation is not None else {
            "correctness": True, "ctest_exit_code": 0, "differential_exit_code": 0}
        self.calls = []
        self.validation_calls = []
        self.counts = {}

    def run(self, build, label):
        self.calls.append((build.build_id, label))
        self.counts[build.build_id] = self.counts.get(build.build_id, 0) + 1
        if build.build_id in self.failures:
            return {"correctness": False}
        throughput = self.throughputs[build.build_id]
        if isinstance(throughput, list):
            throughput = throughput[self.counts[build.build_id] - 1]
        return {"correctness": True, "throughput": throughput, "p50_ns": 50.0,
                "p99_ns": 200.0, "cv": self.cvs.get(build.build_id, .001),
                "spread": throughput * .001, "runs": [throughput] * 5}

    def full_validate(self, build, seeds, orders):
        self.validation_calls.append((build.build_id, seeds, orders))
        return self.validation


def test_interleaves_top_three_then_base_and_validates_only_winner():
    runner = Runner()
    callbacks = []
    result = confirm_candidates(runner, Build("base"), [Build("A"), Build("B"), Build("C")],
                                rounds=3, callback=callbacks.append)
    assert [build for build, _ in runner.calls] == ["A", "B", "C", "base"] * 3
    assert len({label for _, label in runner.calls}) == 12
    assert result["status"] == "confirmed" and result["recommendation"] == "A"
    assert result["medians"]["A"]["speedup"] == 1.2
    assert runner.validation_calls == [("A", 100, 1000000)]
    assert callbacks == result["observations"]


def test_failed_candidate_cannot_win_even_with_high_prior_throughput():
    runner = Runner(failures={"A"})
    result = confirm_candidates(runner, Build("base"), [Build("A"), Build("B")], rounds=3)
    assert result["recommendation"] == "B"
    assert "A" in result["excluded"] and "A" not in result["medians"]
    assert sum(build == "A" for build, _ in runner.calls) == 1


def test_failed_build_excluded_and_duplicate_candidates_not_repeated():
    runner = Runner()
    result = confirm_candidates(runner, Build("base"), [Build("A", False), Build("B"), Build("B"), Build("base")], rounds=2)
    assert result["recommendation"] == "B"
    assert [build for build, _ in runner.calls] == ["B", "base"] * 2


def test_baseline_correctness_failure_invalidates_confirmation():
    runner = Runner(failures={"base"})
    result = confirm_candidates(runner, Build("base"), [Build("A")], rounds=3)
    assert result["status"] == "invalid_baseline" and result["recommendation"] is None
    assert runner.validation_calls == []


@pytest.mark.parametrize("cv", [.02, .05])
def test_high_run_noise_prevents_recommendation(cv):
    runner = Runner(cvs={"base": cv})
    result = confirm_candidates(runner, Build("base"), [Build("A")], rounds=3)
    assert result["status"] == "noisy" and result["recommendation"] is None
    assert runner.validation_calls == []
    assert not result["medians"]["A"]["significant"]


def test_baseline_drift_prevents_claiming_speedup():
    runner = Runner(throughputs={"base": [100, 100, 103, 103], "A": 125})
    result = confirm_candidates(runner, Build("base"), [Build("A")], rounds=4)
    assert result["status"] == "noisy"
    assert result["medians"]["base"]["drift"] > .02
    assert result["recommendation"] is None


def test_candidate_must_clear_three_times_baseline_noise():
    runner = Runner(throughputs={"base": 100, "A": 102}, cvs={"base": .01})
    result = confirm_candidates(runner, Build("base"), [Build("A")], rounds=3)
    assert result["status"] == "no_significant_improvement"
    assert result["threshold"]["absolute_throughput"] == 3
    assert runner.validation_calls == []


@pytest.mark.parametrize("validation", [
    {"correctness": False},
    {"correctness": True},
    True,
    {"correctness": True, "ctest_exit_code": 1, "differential_exit_code": 0},
    {"correctness": True, "ctest_exit_code": 0, "differential_exit_code": 1},
])
def test_final_correctness_failure_blocks_recommendation(validation):
    runner = Runner(validation=validation)
    result = confirm_candidates(runner, Build("base"), [Build("A")], rounds=2)
    assert result["status"] == "validation_failed"
    assert result["recommendation"] is None
    assert result["winner"] == "A"


def test_provisional_measurement_requires_full_validation_to_recommend():
    runner = Runner()
    result = confirm_candidates(runner, Build("base"), [Build("A")], rounds=2, full_validation=False)
    assert result["status"] == "measured" and result["recommendation"] is None
    assert result["winner"] == "A"
    assert runner.validation_calls == []


def test_invalid_rounds_and_baseline_build_rejected():
    with pytest.raises(ValueError):
        confirm_candidates(Runner(), Build("base"), [], rounds=1)
    with pytest.raises(ValueError):
        confirm_candidates(Runner(), Build("base", False), [])


def test_nonfinite_measurement_cannot_be_recommended():
    runner = Runner(throughputs={"base": 100, "A": float("nan")})
    result = confirm_candidates(runner, Build("base"), [Build("A")], rounds=2)
    assert result["status"] == "no_significant_improvement"
    assert "A" in result["excluded"]


@pytest.mark.parametrize("cv", [.02, .05])
def test_noisy_candidate_cannot_be_recommended_on_quiet_baseline(cv):
    runner = Runner(cvs={"A": cv})
    result = confirm_candidates(runner, Build("base"), [Build("A"), Build("B")], rounds=3)
    assert result["recommendation"] == "B"
    assert "A" in result["excluded"]
    assert not result["medians"]["A"]["significant"]
    assert len([item for item in result["observations"] if item["build_id"] == "A"]) == 3


def test_cross_round_candidate_noise_is_excluded():
    runner = Runner(throughputs={"base": 100, "A": [110, 130, 110, 130], "B": 108})
    result = confirm_candidates(runner, Build("base"), [Build("A"), Build("B")], rounds=4)
    assert result["recommendation"] == "B"
    assert result["medians"]["A"]["cv"] > .02
    assert "A" in result["excluded"]
    assert len([item for item in result["observations"] if item["build_id"] == "A"]) == 4


@pytest.mark.parametrize("candidate_provenance", [
    {"source_sha256": "changed", "ledger_commit": "commit"},
    {"source_sha256": "source", "ledger_commit": "other"},
    {},
])
def test_source_provenance_mismatch_excluded_before_benchmark(candidate_provenance):
    source = {"source_sha256": "source", "ledger_commit": "commit"}
    runner = Runner()
    result = confirm_candidates(runner, Build("base", provenance=source),
                                [Build("A", provenance=candidate_provenance), Build("B", provenance=source)], rounds=2)
    assert result["recommendation"] == "B"
    assert all(build != "A" for build, _ in runner.calls)
    assert "source provenance" in result["excluded"]["A"]


class ProvenanceRunner(Runner):
    def __init__(self, hashes):
        super().__init__()
        self.hashes = hashes

    def run(self, build, label):
        result = super().run(build, label)
        hash_value = self.hashes.get(build.build_id)
        if isinstance(hash_value, list):
            hash_value = hash_value[self.counts[build.build_id] - 1]
        result["provenance"] = {"input_sha256": hash_value}
        return result


def test_input_provenance_mismatch_prevents_recommendation():
    runner = ProvenanceRunner({"base": "fixed", "A": "different", "B": "fixed"})
    result = confirm_candidates(runner, Build("base"), [Build("A"), Build("B")], rounds=3)
    assert result["recommendation"] == "B"
    assert "input provenance" in result["excluded"]["A"]
    assert sum(build == "A" for build, _ in runner.calls) == 1


def test_input_provenance_can_not_change_later_in_confirmation():
    runner = ProvenanceRunner({"base": "fixed", "A": ["fixed", "different"]})
    result = confirm_candidates(runner, Build("base"), [Build("A")], rounds=2)
    assert result["recommendation"] is None
    assert "input provenance" in result["excluded"]["A"]
    assert len(result["observations"]) == 4


def test_changing_baseline_input_invalidates_all_comparisons():
    runner = ProvenanceRunner({"base": ["first", "second"], "A": "first"})
    result = confirm_candidates(runner, Build("base"), [Build("A")], rounds=2)
    assert result["status"] == "invalid_baseline"
    assert result["recommendation"] is None
    assert runner.validation_calls == []
