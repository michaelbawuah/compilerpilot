import copy
import csv
import json
import os
from pathlib import Path
import sys

import pytest

from compilerpilot.bench import BenchmarkConfig, BenchmarkRunner, parse_benchmark
from compilerpilot.build import BuildManager, BuildResult, CommandResult, file_sha256, run_command
from compilerpilot.space import Candidate


def evidence():
    checks = {"event_checksum": 123, "trade_checksum": 456, "events": 10,
              "trades": 2, "rejected_commands": 0, "live_orders_final": 1}
    metrics = {"median_commands_per_second": 110.0, "median_p50_ns": 20,
               "median_p99_ns": 30, **checks}
    names = ["reference", "fast-bitmap", "fast-linear"]
    summary = {"schema_version": 1, "runs": 3, "commands": 20,
               "engines": {name: dict(metrics) for name in names},
               "compiler": {"flags": "-O3 -DNDEBUG standard=C++23", "ndebug": True},
               "machine": {"affinity_cpus": "0", "cpu_model": "test fixture"}}
    rows = [{"engine": name, "run": str(run), "commands": "20",
             "commands_per_second": str(100 + run * 10 - 10), "p50_ns": "20",
             "p99_ns": "30", **{key: str(value) for key, value in checks.items() if key != "live_orders_final"}}
            for name in names for run in range(1, 4)]
    return summary, rows


def test_real_schema_noise_and_checksums():
    summary, rows = evidence()
    parsed = parse_benchmark(summary, rows, expected_runs=3, expected_flags=["-O3"])
    assert parsed["throughput"] == 110
    assert parsed["spread"] == 10
    assert parsed["cv"] == pytest.approx(10 / 110)
    assert parsed["runs"] == [100, 110, 120]


@pytest.mark.parametrize("tamper", ["summary_checksum", "raw_checksum", "median", "duplicate_run",
                                   "extra_flags", "nan"])
def test_reject_tampered_measurement(tamper):
    summary, rows = evidence()
    if tamper == "summary_checksum":
        summary["engines"]["fast-bitmap"]["event_checksum"] += 1
    elif tamper == "raw_checksum":
        rows[3]["trade_checksum"] = "999"
    elif tamper == "median":
        summary["engines"]["fast-bitmap"]["median_commands_per_second"] = 10000
    elif tamper == "duplicate_run":
        rows[4]["run"] = "1"
    elif tamper == "extra_flags":
        summary["compiler"]["flags"] = "-O3 -O2 -DNDEBUG standard=C++23"
    else:
        rows[4]["commands_per_second"] = "nan"
    with pytest.raises(ValueError):
        parse_benchmark(summary, rows, expected_runs=3, expected_flags=["-O3"])


def test_wrong_input_or_affinity_rejected(tmp_path):
    summary, rows = evidence()
    summary["input"] = str(tmp_path / "other.lob")
    with pytest.raises(ValueError, match="input provenance"):
        parse_benchmark(summary, rows, expected_runs=3, expected_input=tmp_path / "eval.lob")
    with pytest.raises(ValueError, match="single CPU"):
        parse_benchmark(summary, rows, expected_runs=3, expected_cpu=999)


def test_subprocess_timeout_and_output_limit(tmp_path):
    completed = run_command([sys.executable, "-c", "print('x' * 20000)"], timeout=5,
                            log=tmp_path / "large.log")
    assert completed.ok
    assert len(completed.output) <= 12288
    assert (tmp_path / "large.log").stat().st_size > 12288
    timed = run_command([sys.executable, "-c", "import time; time.sleep(5)"], timeout=.05,
                        log=tmp_path / "timeout.log")
    assert not timed.ok and timed.timed_out
    assert timed.elapsed_s < 2


def test_failed_build_is_recorded(tmp_path):
    result = BuildManager(tmp_path / "missing", tmp_path / "builds").build(Candidate())
    assert not result.success
    assert "source is missing" in result.error
    persisted = json.loads((Path(result.build_dir) / "build.json").read_text())
    assert persisted["success"] is False
    assert persisted["candidate"] == Candidate().to_dict()


def test_symlink_build_directory_is_rejected(tmp_path):
    builds = tmp_path / "builds"
    builds.mkdir()
    untouched = tmp_path / "untouched"
    untouched.mkdir()
    (builds / Candidate().hash()).symlink_to(untouched, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic link"):
        BuildManager(tmp_path / "ledger", builds).build(Candidate())
    assert untouched.is_dir()


def test_differential_failure_never_exposes_recommendable_metrics(tmp_path, monkeypatch):
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    binary = build_dir / "ledger_bench"
    binary.write_bytes(b"fixture-binary")
    input_path = tmp_path / "evaluation.lob"
    input_path.write_bytes(b"fixture-input")
    config = BenchmarkConfig(input_path, runs=3)
    result = BuildResult("fixture", Candidate().to_dict(), str(build_dir), success=True,
                         flags=["-O3"], provenance={"binary_sha256": file_sha256(binary)})

    def fake_run(argv, *, timeout, log, cwd=None):
        if "ledger_differential" in str(argv):
            return CommandResult([str(x) for x in argv], 1, .01, "reference mismatch")
        summary, rows = evidence()
        summary["input"] = str(config.input)
        summary["machine"]["affinity_cpus"] = str(config.cpu)
        output = Path(log).parent
        (output / "summary.json").write_text(json.dumps(summary))
        with (output / "runs.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return CommandResult([str(x) for x in argv], 0, .01, "")

    monkeypatch.setattr("compilerpilot.bench.run_command", fake_run)
    measurement = BenchmarkRunner(config).run(result)
    assert measurement["correctness"] is False
    assert measurement["throughput"] is None
    assert measurement["runs"] == []
    assert "Differential correctness gate failed" in measurement["error"]


def test_pgo_rejects_identical_training_and_evaluation_bytes(tmp_path):
    stream = tmp_path / "eval.lob"
    stream.write_bytes(b"same bytes, even under a different filename")
    result = BuildResult("fixture", {**Candidate().to_dict(), "pgo": True}, str(tmp_path / "build"),
                         success=True, provenance={"training_input_sha256": file_sha256(stream)})
    measurement = BenchmarkRunner(BenchmarkConfig(stream)).run(result)
    assert not measurement["correctness"]
    assert "must differ" in measurement["error"]


@pytest.mark.parametrize("kwargs", [{"runs": 2}, {"runs": 0}, {"warmup": -1},
                                   {"cpu": -1}, {"differential_seeds": 0},
                                   {"timeout": float("nan")}, {"differential_orders": 5.5},
                                   {"differential_seeds": True}])
def test_invalid_benchmark_configuration(kwargs, tmp_path):
    with pytest.raises(ValueError):
        BenchmarkConfig(tmp_path / "eval.lob", **kwargs)


@pytest.mark.parametrize("completion", ["partial", "failure", "complete"])
def test_full_gate_requires_complete_passing_ctest_evidence(completion, tmp_path, monkeypatch):
    (tmp_path / "build").mkdir()
    built = BuildResult("fixture", Candidate().to_dict(), str(tmp_path / "build"), success=True)

    def fake_run(argv, *, timeout, log, cwd=None):
        args = [str(item) for item in argv]
        output = Path(log).parent
        if "--show-only=json-v1" in args:
            Path(log).write_text(json.dumps({"tests": [{"name": "a"}, {"name": "b"}]}))
        elif "ctest" in args:
            second = "" if completion == "partial" else '<testcase name="b"/>'
            first = '<testcase name="a"><failure>broken</failure></testcase>' if completion == "failure" else '<testcase name="a"/>'
            (output / "ctest.xml").write_text("<testsuite>" + first + second + "</testsuite>")
        else:
            (output / "differential.json").write_text(json.dumps({"status": "pass", "seeds": 100,
                                                                "commands_per_seed": 1000000}))
        return CommandResult(args, 0, .01, "all processes exit zero")

    monkeypatch.setattr("compilerpilot.bench.run_command", fake_run)
    result = BenchmarkRunner(BenchmarkConfig(tmp_path / "eval.lob")).full_validate(built)
    assert result["correctness"] is (completion == "complete")
    assert result["ctest_planned_tests"] == 2
    assert result["ctest_completed_tests"] == (1 if completion == "partial" else 2)
