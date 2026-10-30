"""Reports never promote failed measurements or imply missing experiments ran."""
from dataclasses import replace
import json
from pathlib import Path

import pandas as pd

from compilerpilot.plot import generate_report
from compilerpilot.space import Candidate


def row(candidate=Candidate(), **changes):
    return {"candidate": json.dumps(candidate.to_dict()), "strategy": "random", "seed": 0,
            "build_index": 1, "throughput": 1000000., "speedup": 1., "baseline_throughput": 1000000.,
            "p99_ns": 200., "compile_s": 3., "text_bytes": 2048,
            "correctness": True, "measurement_valid": True, "significant": False, **changes}


def test_empty_report_has_no_invented_performance(tmp_path):
    result = generate_report(tmp_path)
    report = (tmp_path / "summary.md").read_text()
    assert result["trials"] == result["eligible"] == 0
    assert result["figures"] == []
    assert "No performance winner is reported" in report
    assert "incomplete" in report
    assert "1.0000×" not in report


def test_invalid_and_failed_trials_count_but_never_win(tmp_path):
    rows = [row(), row(build_index=2, throughput=9e12, speedup=9e6, correctness=False),
            row(build_index=3, throughput=8e12, speedup=8e6, measurement_valid=False)]
    pd.DataFrame(rows).to_parquet(tmp_path / "trials.parquet")
    result = generate_report(tmp_path)
    report = (tmp_path / "summary.md").read_text()
    assert result["trials"] == 3
    assert result["eligible"] == 1
    assert result["failed"] == result["measurement_excluded"] == 1
    assert "9,000,000" not in report
    assert "8,000,000" not in report
    assert "1.0000×" in report
    assert "not confirmed" in report
    assert "seed 0: 3/40" in report
    assert "1/3 seeds" in report
    assert Path(result["figures"][0]).is_file()


def protocol_run(root, run_id, strategy="random", seed=0, budget=2, *, noise_gate=True, complete=True, started="2026-10-30T10:00:00Z"):
    path = root / "runs" / run_id
    path.mkdir(parents=True, exist_ok=True)
    metric = {"correctness": True, "throughput": 1000000., "p99_ns": 200., "cv": .01, "spread": 10000.}
    (path / "protocol.json").write_text(json.dumps({"strategy": strategy, "seed": seed, "budget": budget}))
    (path / "baseline.json").write_text(json.dumps({"metric": metric, "observations": [metric] * 10,
                                                   "noise_gate": noise_gate, "repetitions": 10,
                                                   "build": {"candidate": Candidate().to_dict(), "compile_s": 3., "text_bytes": 2048}}))
    (path / "outcome.json").write_text(json.dumps({"status": "provisional_pending_confirmation" if complete else "incomplete", "builds_used": budget}))
    (path / "trace.jsonl").write_text(json.dumps({"time": started, "event": "baseline_checkpoint", "data": {"valid": noise_gate}}) + "\n")
    return [row(run_id=run_id, strategy=strategy, seed=seed, build_index=index) for index in range(1, budget + 1)]


def test_completion_requires_all_strategies_seeds_and_builds(tmp_path):
    rows = []
    for strategy in ("random", "genetic", "surrogate", "agent"):
        for seed in (0, 1, 2):
            rows.extend(protocol_run(tmp_path, f"{strategy}-{seed}", strategy, seed))
    pd.DataFrame(rows).to_parquet(tmp_path / "trials.parquet")
    result = generate_report(tmp_path, budget=2)
    assert result["complete"] is True
    pd.DataFrame(rows[:-1]).to_parquet(tmp_path / "trials.parquet")
    assert generate_report(tmp_path, budget=2)["complete"] is False


def test_predictor_validation_and_scatter_from_distinct_valid_rows(tmp_path):
    candidates = [replace(Candidate(), inline=100), replace(Candidate(), inline=500),
                  replace(Candidate(), unroll=True), replace(Candidate(), opt="O2")]
    rows = [row(candidate, build_index=i+1, throughput=1e6 * (1 + i / 100), speedup=1 + i / 100)
            for i, candidate in enumerate(candidates)]
    pd.DataFrame(rows).to_parquet(tmp_path / "trials.parquet")
    result = generate_report(tmp_path)
    assert result["predictor"]["samples"] == 4
    assert result["predictor"]["folds"] == 4
    assert len(result["figures"]) == 2
    assert all(Path(path).is_file() for path in result["figures"])
    report = (tmp_path / "summary.md").read_text()
    assert "MAE" in report and "4 distinct candidates" in report


def test_string_boolean_never_passes_measurement_gate(tmp_path):
    pd.DataFrame([row(correctness="true", measurement_valid="true")]).to_parquet(tmp_path / "trials.parquet")
    assert generate_report(tmp_path)["eligible"] == 0


def test_stale_derived_figures_are_removed_when_measurements_disappear(tmp_path):
    figures = tmp_path / "figures"
    figures.mkdir()
    (figures / "best-so-far.png").write_bytes(b"stale")
    (figures / "predictor-validation.png").write_bytes(b"stale")
    assert generate_report(tmp_path)["figures"] == []
    assert not list(figures.iterdir())


def test_independent_runs_are_not_merged_and_earliest_complete_wins(tmp_path):
    early = protocol_run(tmp_path, "early", started="2026-10-30T09:00:00Z")
    later = protocol_run(tmp_path, "later", started="2026-10-30T10:00:00Z")
    for trial in later:
        trial.update(throughput=9000000., speedup=9.)
    pd.DataFrame(early + later).to_parquet(tmp_path / "trials.parquet")
    result = generate_report(tmp_path, budget=2, expected_seeds=1)
    selected = [item for item in result["runs"] if item["selected"]]
    assert [item["run_id"] for item in selected] == ["early"]
    assert result["eligible"] == 2
    report = (tmp_path / "summary.md").read_text()
    assert "9.0000×" not in report
    assert "later" in report
    assert len(result["baselines"]) == 2


def test_longest_partial_is_single_run_and_missing_metadata_never_complete(tmp_path):
    first = [row(run_id="one", build_index=1, throughput=9000000., speedup=9.)]
    second = [row(run_id="two", build_index=i) for i in (1, 2)]
    pd.DataFrame(first + second).to_parquet(tmp_path / "trials.parquet")
    result = generate_report(tmp_path, budget=2, expected_seeds=1)
    assert result["eligible"] == 2
    assert result["coverage"]["random"]["complete"] is False
    assert [item["run_id"] for item in result["runs"] if item["selected"]] == ["two"]
    assert "9.0000×" not in (tmp_path / "summary.md").read_text()


def test_discarded_run_remains_visible_when_replaced_by_valid_run(tmp_path):
    discarded = protocol_run(tmp_path, "discarded", noise_gate=False)
    for trial in discarded:
        trial["measurement_valid"] = False
    good = protocol_run(tmp_path, "good")
    pd.DataFrame(discarded + good).to_parquet(tmp_path / "trials.parquet")
    result = generate_report(tmp_path, budget=2, expected_seeds=1)
    assert [item["run_id"] for item in result["runs"] if item["selected"]] == ["good"]
    rejected = next(item for item in result["runs"] if item["run_id"] == "discarded")
    assert rejected["noise_excluded"] == 2
    assert rejected["protocol_complete"] is False
    assert result["measurement_excluded"] == 2
    report = (tmp_path / "summary.md").read_text()
    assert "discarded noise/drift batch" in report
    assert "FAILED; E1 measurement complete" in report


def test_baseline_gate_failed_before_candidates_is_in_inventory(tmp_path):
    protocol_run(tmp_path, "baseline-stopped", noise_gate=False, complete=False)
    result = generate_report(tmp_path, budget=2)
    assert result["trials"] == 0
    assert result["runs"][0]["run_id"] == "baseline-stopped"
    assert result["runs"][0]["builds"] == 0
    assert result["baselines"][0]["noise_gate"] is False


def test_qualification_baseline_is_not_ten_observation_e1(tmp_path):
    metric = {"correctness": True, "throughput": 15000000., "p99_ns": 280., "cv": .13, "spread": 1950000.}
    (tmp_path / "qualification-baseline.json").write_text(json.dumps({"metric": metric,
        "build": {"candidate": Candidate().to_dict(), "compile_s": 20., "text_bytes": 97166}}))
    result = generate_report(tmp_path)
    assert result["baselines"][0]["repetitions"] == 1
    assert result["baselines"][0]["noise_gate"] is False
    assert result["baselines"][0]["e1_complete"] is False
    report = (tmp_path / "summary.md").read_text()
    assert "15.000" in report and "13.00%" in report
    assert "FAILED; E1 incomplete" in report


def test_report_cli_preserves_run_separation(tmp_path, monkeypatch, capsys):
    import runpy
    import sys
    early = protocol_run(tmp_path, "cli-early", started="2026-10-30T09:00:00Z")
    later = protocol_run(tmp_path, "cli-later", started="2026-10-30T10:00:00Z")
    later[0].update(speedup=9., throughput=9e6)
    pd.DataFrame(early + later).to_parquet(tmp_path / "trials.parquet")
    monkeypatch.setattr(sys, "argv", ["plot.py", "--results", str(tmp_path), "--budget", "2", "--expected-seeds", "1"])
    runpy.run_path(str(Path(__file__).parents[1] / "scripts" / "plot.py"), run_name="__main__")
    result = json.loads(capsys.readouterr().out)
    assert [item["run_id"] for item in result["runs"] if item["selected"]] == ["cli-early"]
    assert result["eligible"] == 2


def test_e1_figure_uses_all_raw_replays_and_preserves_source(tmp_path):
    import hashlib
    protocol_run(tmp_path, "E1-quiet", strategy="baseline", budget=1, noise_gate=False, complete=False)
    path = tmp_path / "runs" / "E1-quiet" / "baseline.json"
    source = json.loads(path.read_text())
    for index, observation in enumerate(source["observations"]):
        observation["runs"] = [1e6 + index * 10000 + j * 1000 for j in range(5)]
        observation["throughput"] = observation["runs"][2]
        observation["cv"] = .1361
    source["metric"]["cv"] = .1361
    source["metric"]["throughput"] = 1047000.
    path.write_text(json.dumps(source))
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    result = generate_report(tmp_path)
    assert len(result["e1"]) == 1
    assert result["e1"][0]["raw_runs"] == 50
    assert len(result["e1"][0]["observation_medians"]) == 10
    assert len(result["e1"][0]["observation_cvs"]) == 10
    assert result["e1"][0]["source_sha256"] == before
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert Path(result["figures"][0]).is_file()
    report = (tmp_path / "summary.md").read_text()
    assert "E1 noise gate failed" in report
    assert "Performance experiments E2–E5 were not performed" in report
    assert "13.61%" in report


def test_qualification_table_uses_raw_metrics_and_does_not_calculate_ratios(tmp_path):
    for name, key, throughput in [("qualification-baseline.json", "metric", 15e6),
                                  ("qualification-pgo.json", "metric", 16e6),
                                  ("clang-pgo-qualification.json", "measurement", 15.5e6)]:
        source = {key: {"correctness": True, "throughput": throughput, "p99_ns": 270., "cv": .13, "runs": [throughput] * 5},
                  "build": {"candidate": Candidate().to_dict(), "compile_s": 20., "text_bytes": 102400}}
        (tmp_path / name).write_text(json.dumps(source))
    result = generate_report(tmp_path)
    assert len(result["qualifications"]) == 3
    report = (tmp_path / "summary.md").read_text()
    section = report.split("## Functional build qualification")[1].split("## Recorded search builds")[0]
    assert "15.000" in section and "16.000" in section and "15.500" in section
    assert "13.00%" in section
    assert "×" not in section
    assert "No speedup ratio is calculated" in section


def test_zero_search_trials_produce_empty_typed_parquet(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    generate_report(tmp_path)
    table = pq.read_table(tmp_path / "trials.parquet")
    assert table.num_rows == 0
    assert table.schema.field("correctness").type == pa.bool_()
    assert table.schema.field("throughput").type == pa.float64()
    assert table.schema.field("candidate").type == pa.string()
