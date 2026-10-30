"""Generate figures and a factual report from the recorded Parquet trials."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from compilerpilot.strategies.surrogate import cross_validate

STRATEGIES = ("random", "genetic", "surrogate", "agent")
COLORS = {"random": "#6b7280", "genetic": "#d18b17", "surrogate": "#078a82", "agent": "#6c4fd3"}


def _true(value: Any) -> bool:
    # numpy.bool_ is returned by some Parquet engines; strings never qualify.
    return value is True or type(value).__name__ == "bool_" and bool(value)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _ensure_trial_table(results_dir: Path) -> None:
    """An empty typed index means zero search trials, not synthetic evidence."""
    path = results_dir / "trials.parquet"
    if path.exists():
        return
    import pyarrow as pa
    import pyarrow.parquet as pq
    schema = pa.schema([
        ("run_id", pa.string()), ("strategy", pa.string()), ("seed", pa.int64()),
        ("build_index", pa.int64()), ("candidate", pa.string()),
        ("throughput", pa.float64()), ("speedup", pa.float64()),
        ("p99_ns", pa.float64()), ("compile_s", pa.float64()), ("text_bytes", pa.int64()),
        ("baseline_throughput", pa.float64()), ("baseline_spread", pa.float64()),
        ("correctness", pa.bool_()), ("measurement_valid", pa.bool_()), ("significant", pa.bool_())])
    pq.write_table(pa.Table.from_pylist([], schema=schema), path)


def _load(results_dir: Path) -> list[dict[str, Any]]:
    import pandas as pd
    table = results_dir / "trials.parquet"
    if not table.exists():
        return []
    rows = pd.read_parquet(table).to_dict("records")
    for row in rows:
        if isinstance(row.get("candidate"), str):
            try:
                row["candidate"] = json.loads(row["candidate"])
            except (json.JSONDecodeError, TypeError):
                row["candidate"] = {}
    return rows


def _eligible(row: dict[str, Any]) -> bool:
    throughput, speedup = _number(row.get("throughput")), _number(row.get("speedup"))
    return (_true(row.get("correctness")) and _true(row.get("measurement_valid"))
            and throughput is not None and throughput > 0 and speedup is not None and speedup > 0)


def _median(rows: list[dict[str, Any]], key: str) -> float | None:
    from statistics import median
    values = [value for row in rows if (value := _number(row.get(key))) is not None]
    return float(median(values)) if values else None


def _fmt(value: float | None, digits: int = 2) -> str:
    return "unavailable" if value is None else f"{value:,.{digits}f}"


def _candidate_key(row: dict[str, Any]) -> str:
    return json.dumps(row.get("candidate", {}), sort_keys=True)


def _best(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
    from statistics import median
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(_candidate_key(row), []).append(row)
    if not groups:
        return None
    chosen = max(groups.values(), key=lambda group: median(float(r["speedup"]) for r in group))
    return chosen, chosen[0].get("candidate", {})


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _safe_run_dir(results_dir: Path, run_id: str) -> Path | None:
    if not run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in run_id):
        return None
    return results_dir / "runs" / run_id


def _trace_state(run_dir: Path | None) -> tuple[str, bool]:
    started, bad_checkpoint = "", False
    if run_dir is not None and (run_dir / "trace.jsonl").is_file():
        for line in (run_dir / "trace.jsonl").read_text().splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not started and isinstance(entry.get("time"), str):
                started = entry["time"]
            if entry.get("event") == "baseline_checkpoint" and not _true(entry.get("data", {}).get("valid")):
                bad_checkpoint = True
    return started, bad_checkpoint


def _select_runs(rows: list[dict[str, Any]], results_dir: Path, budget: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Choose whole experiments by a fixed outcome-independent rule.

    Prefer the earliest complete protocol run; otherwise show the longest single
    partial run. Performance never participates in this choice. All attempts stay
    in the inventory, including noise failures and experiments not selected.
    """
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        strategy, seed = row.get("strategy"), _number(row.get("seed"))
        if strategy not in STRATEGIES or seed is None or not seed.is_integer():
            continue
        raw_id = row.get("run_id")
        run_id = raw_id if isinstance(raw_id, str) and raw_id else f"legacy-{strategy}-{int(seed)}"
        groups.setdefault((strategy, int(seed), run_id), []).append(row)
    # A run stopped by its baseline gate has no candidate rows but is still evidence.
    for file in sorted((results_dir / "runs").glob("*/protocol.json")):
        protocol = _read_json(file)
        seed = _number(protocol.get("seed"))
        if protocol.get("strategy") in STRATEGIES and seed is not None and seed.is_integer():
            groups.setdefault((protocol["strategy"], int(seed), file.parent.name), [])
    inventory = []
    for (strategy, seed, run_id), attempts in sorted(groups.items()):
        run_dir = _safe_run_dir(results_dir, run_id)
        protocol = _read_json(run_dir / "protocol.json") if run_dir else {}
        baseline = _read_json(run_dir / "baseline.json") if run_dir else {}
        outcome = _read_json(run_dir / "outcome.json") if run_dir else {}
        if not outcome and run_dir:
            outcome = _read_json(run_dir / "recommendation.json")
        started, bad_checkpoint = _trace_state(run_dir)
        indices = [int(value) for row in attempts if (value := _number(row.get("build_index"))) is not None and value.is_integer()]
        length = max(indices, default=0)
        noise_excluded = sum(_true(row.get("correctness")) and not _true(row.get("measurement_valid")) for row in attempts)
        failures = sum(not _true(row.get("correctness")) for row in attempts)
        declared = protocol.get("strategy") == strategy and protocol.get("seed") == seed and protocol.get("budget") == budget
        baseline_cv = _number(baseline.get("metric", {}).get("cv"))
        baseline_valid = (_true(baseline.get("noise_gate")) and _true(baseline.get("metric", {}).get("correctness"))
                          and baseline_cv is not None and baseline_cv < .02
                          and baseline.get("repetitions", 0) >= 10 and len(baseline.get("observations", [])) >= 10)
        full_indices = sorted(indices) == list(range(1, budget + 1))
        completed_outcome = outcome.get("status") == "provisional_pending_confirmation" and outcome.get("builds_used") == budget
        complete = declared and baseline_valid and full_indices and completed_outcome and not noise_excluded and not bad_checkpoint
        reasons = []
        if not declared: reasons.append("missing/mismatched predeclared protocol")
        if not baseline_valid: reasons.append("baseline noise/E1 gate missing or failed")
        if not full_indices: reasons.append("candidate budget incomplete or duplicate indices")
        if not completed_outcome: reasons.append("completed outcome missing")
        if noise_excluded or bad_checkpoint: reasons.append("discarded noise/drift batch")
        inventory.append({"run_id": run_id, "strategy": strategy, "seed": seed, "builds": length,
                          "valid": sum(_eligible(row) for row in attempts), "failed": failures,
                          "noise_excluded": noise_excluded, "protocol_complete": complete,
                          "status": "complete" if complete else "; ".join(reasons), "selected": False,
                          "started": started, "_rows": attempts})
    selected = []
    for strategy, seed in sorted({(item["strategy"], item["seed"]) for item in inventory}):
        choices = [item for item in inventory if item["strategy"] == strategy and item["seed"] == seed]
        full = [item for item in choices if item["protocol_complete"]]
        if full:
            chosen = min(full, key=lambda item: (item["started"] or "9999", item["run_id"]))
            chosen["selection"] = "earliest complete predeclared protocol run"
        else:
            chosen = min(choices, key=lambda item: (-item["builds"], item["started"] or "9999", item["run_id"]))
            chosen["selection"] = "longest single partial run (incomplete)"
        chosen["selected"] = True
        selected.extend(chosen["_rows"])
    for item in inventory:
        del item["_rows"]
    return selected, inventory


def _baselines(results_dir: Path) -> list[dict[str, Any]]:
    """Read baseline observations without merging different experiments."""
    paths = sorted((results_dir / "runs").glob("*/baseline.json"))
    qualification = results_dir / "qualification-baseline.json"
    if qualification.is_file():
        paths.insert(0, qualification)
    summaries = []
    for path in paths:
        source = _read_json(path)
        metric, build = source.get("metric", {}), source.get("build", {})
        observations = source.get("observations", [])
        if not observations and metric:
            observations = [metric]
        throughput = _number(metric.get("throughput"))
        if throughput is None:
            continue
        cv = _number(metric.get("cv"))
        correct = _true(metric.get("correctness")) and all(_true(row.get("correctness")) for row in observations)
        repeated = source.get("repetitions", len(observations))
        noise_gate = bool(cv is not None and cv < .02 and ("noise_gate" not in source or _true(source["noise_gate"])))
        label = "qualification" if path == qualification else path.parent.name
        summaries.append({"run_id": label, "path": str(path.relative_to(results_dir)), "throughput": throughput,
                          "p99_ns": _number(metric.get("p99_ns")), "spread": _number(metric.get("spread")), "cv": cv,
                          "compile_s": _number(build.get("compile_s")), "text_bytes": _number(build.get("text_bytes")),
                          "correctness": correct, "noise_gate": noise_gate, "repetitions": repeated,
                          "e1_complete": repeated >= 10 and len(observations) >= 10 and correct and noise_gate,
                          "measurement_complete": repeated >= 10 and len(observations) >= 10 and correct,
                          "qualification": path == qualification, "candidate": build.get("candidate", {}),
                          "cpu_model": metric.get("provenance", {}).get("cpu_model"),
                          "input_sha256": metric.get("provenance", {}).get("input_sha256")})
    return summaries


def _qualifications(results_dir: Path) -> list[dict[str, Any]]:
    sources = [("GCC baseline", "qualification-baseline.json"),
               ("GCC PGO", "qualification-pgo.json"),
               ("Clang ThinLTO + PGO", "clang-pgo-qualification.json")]
    measurements = []
    for label, name in sources:
        source = _read_json(results_dir / name)
        build = source.get("build", {})
        metric = source.get("metric", source.get("measurement", {}))
        if not source or _number(metric.get("throughput")) is None:
            continue
        measurements.append({"label": label, "source": name, "throughput": metric["throughput"],
                             "p99_ns": _number(metric.get("p99_ns")), "cv": _number(metric.get("cv")),
                             "compile_s": _number(build.get("compile_s")), "text_bytes": _number(build.get("text_bytes")),
                             "correctness": _true(metric.get("correctness")), "raw_runs": len(metric.get("runs", []))})
    return measurements


def _e1_figures(results_dir: Path, figures_dir: Path) -> tuple[list[Path], list[dict[str, Any]]]:
    """Plot every raw baseline replay and the recorded observation CVs."""
    import hashlib
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths, evidence = [], []
    for source_path in sorted((results_dir / "runs").glob("*/baseline.json")):
        source = _read_json(source_path)
        observations = source.get("observations", [])
        if len(observations) < 10:
            continue
        raw_points, medians, cvs = [], [], []
        for index, observation in enumerate(observations, 1):
            raw = [_number(value) for value in observation.get("runs", [])]
            raw = [value for value in raw if value is not None]
            median = _number(observation.get("throughput"))
            cv = _number(observation.get("cv"))
            if median is not None:
                medians.append([index, median])
            if cv is not None:
                cvs.append([index, cv])
            for run, value in enumerate(raw):
                raw_points.append([index + (run - (len(raw) - 1) / 2) * .11, value])
        if not raw_points:
            continue
        overall_cv = _number(source.get("metric", {}).get("cv"))
        gate = _true(source.get("noise_gate")) and overall_cv is not None and overall_cv < .02
        status, color = ("passed", "#078a82") if gate else ("FAILED", "#b84135")
        fig, (ax, cv_ax) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                       gridspec_kw={"height_ratios": [2.0, 1.0]})
        ax.scatter([point[0] for point in raw_points], [point[1] / 1e6 for point in raw_points],
                   s=30, color="#64748b", alpha=.75, label=f"All {len(raw_points)} raw replays")
        ax.plot([point[0] for point in medians], [point[1] / 1e6 for point in medians],
                color="#6c4fd3", marker="D", markersize=5, linewidth=1.5, label="Five-replay observation median")
        ax.set(ylabel="Throughput (M commands/s)", title=f"E1: baseline stability — noise gate {status}")
        ax.legend(frameon=False, loc="best", fontsize=9)
        cv_ax.bar([point[0] for point in cvs], [point[1] * 100 for point in cvs], color="#94a3b8", width=.65,
                  label="Within-observation CV")
        cv_ax.axhline(2, color="#b84135", linewidth=1.5, linestyle="--", label="Quiet-host gate: CV < 2%")
        cv_ax.set(xlabel="Baseline observation (five raw replays each)", ylabel="CV (%)", xticks=range(1, len(observations)+1))
        cv_ax.legend(frameon=False, loc="best", fontsize=9)
        for panel in (ax, cv_ax):
            panel.grid(axis="y", alpha=.18)
            panel.spines[["top", "right"]].set_visible(False)
        fig.text(.015, .015, f"Conservative baseline CV: {_fmt(overall_cv * 100 if overall_cv is not None else None)}%  |  "
                 f"Gate {status}: required <2%  |  Source: runs/{source_path.parent.name}/baseline.json", fontsize=9, color=color)
        fig.tight_layout(rect=(0, .045, 1, 1))
        figures_dir.mkdir(parents=True, exist_ok=True)
        path = figures_dir / f"e1-{source_path.parent.name}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
        evidence.append({"figure": str(path.relative_to(results_dir)), "source": str(source_path.relative_to(results_dir)),
                         "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                         "raw_runs": len(raw_points), "observation_medians": medians, "observation_cvs": cvs,
                         "conservative_cv": overall_cv, "noise_gate": gate})
    active = {path.name for path in paths}
    for stale in figures_dir.glob("e1-*.png"):
        if stale.name not in active:
            stale.unlink()
    if evidence:
        (results_dir / "e1-plot-data.json").write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    else:
        (results_dir / "e1-plot-data.json").unlink(missing_ok=True)
    return paths, evidence


def _curves(rows: list[dict[str, Any]], figures_dir: Path, budget: int, expected_seeds: int, inventory: list[dict[str, Any]]) -> tuple[Path | None, dict[str, Any]]:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coverage: dict[str, Any] = {}
    fig, ax = plt.subplots(figsize=(9, 5.2))
    plotted = False
    for strategy in STRATEGIES:
        attempted = [row for row in rows if row.get("strategy") == strategy]
        seeds = sorted({int(row["seed"]) for row in attempted if _number(row.get("seed")) is not None})
        lengths: dict[int, int] = {}
        runs: list[list[float]] = []
        for seed in seeds:
            attempts = [row for row in attempted if int(row.get("seed", -1)) == seed]
            length = max((int(row.get("build_index", 0)) for row in attempts), default=0)
            lengths[seed] = length
            best = math.nan
            curve = []
            for index in range(1, length + 1):
                values = [float(row["speedup"]) for row in attempts
                          if int(row.get("build_index", 0)) == index and _eligible(row)]
                if values:
                    best = max(values + ([best] if math.isfinite(best) else []))
                curve.append(best)
            runs.append(curve)
        selected_runs = [item for item in inventory if item["strategy"] == strategy and item["selected"]]
        complete = (len(selected_runs) == expected_seeds and {item["seed"] for item in selected_runs} == set(range(expected_seeds))
                    and all(item["protocol_complete"] for item in selected_runs))
        coverage[strategy] = {"seeds": seeds, "builds_per_seed": lengths,
                              "expected_seeds": expected_seeds, "budget": budget, "complete": complete,
                              "selected_runs": [{k: v for k, v in item.items() if k != "started"} for item in selected_runs]}
        # Compare the same seed set at each point, without extending unfinished runs.
        horizon = min((len(run) for run in runs), default=0)
        if not horizon:
            continue
        matrix = np.asarray([run[:horizon] for run in runs])
        usable = np.all(np.isfinite(matrix), axis=0)
        if not np.any(usable):
            continue
        x = np.arange(1, horizon + 1)[usable]
        matrix = matrix[:, usable]
        means, low, high = matrix.mean(axis=0), matrix.min(axis=0), matrix.max(axis=0)
        label = f"{strategy} ({len(seeds)}/{expected_seeds} seeds; {'complete' if complete else 'incomplete'})"
        ax.plot(x, (means - 1) * 100, label=label, color=COLORS[strategy], linewidth=2, marker="o" if horizon < 10 else None)
        if len(seeds) > 1:
            ax.fill_between(x, (low - 1) * 100, (high - 1) * 100, color=COLORS[strategy], alpha=.14)
        plotted = True
    ax.axhline(0, color="#a0a5ae", linewidth=1)
    ax.set(title="Best valid measured build found by search", xlabel="Candidate builds used (failed builds count)",
           ylabel="Throughput change from baseline (%)", xlim=(.7, max(budget, 2) + .3))
    ax.grid(axis="y", alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)
    if plotted:
        ax.legend(frameon=False, fontsize=8)
        fig.text(.01, .01, "Line: mean; band: min–max across observed seeds. No extrapolation beyond the shortest run.", fontsize=8, color="#4b5563")
        fig.tight_layout(rect=(0, .035, 1, 1))
        figures_dir.mkdir(parents=True, exist_ok=True)
        path = figures_dir / "best-so-far.png"
        fig.savefig(path, dpi=170)
    else:
        path = None
        (figures_dir / "best-so-far.png").unlink(missing_ok=True)
    plt.close(fig)
    return path, coverage


def _scatter(validation: dict[str, Any], figures_dir: Path) -> Path | None:
    points = validation.get("predictions", [])
    if not points:
        (figures_dir / "predictor-validation.png").unlink(missing_ok=True)
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    measured = [(point["measured_speedup"] - 1) * 100 for point in points]
    predicted = [(point["predicted_speedup"] - 1) * 100 for point in points]
    fig, ax = plt.subplots(figsize=(6.2, 5.3))
    ax.scatter(measured, predicted, color="#078a82", alpha=.8, s=35)
    lower, upper = min(measured + predicted), max(measured + predicted)
    padding = max((upper - lower) * .08, .1)
    ax.plot([lower - padding, upper + padding], [lower - padding, upper + padding], color="#9ca3af", linestyle="--", linewidth=1)
    ax.set(title="Predictor: held-out candidates", xlabel="Measured throughput change (%)", ylabel="Out-of-fold predicted change (%)")
    ax.grid(alpha=.15)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / "predictor-validation.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def generate_report(results_dir: Path, *, budget: int = 40, expected_seeds: int = 3) -> dict[str, Any]:
    """Write summary.md and figures from real stored trials; never invent rows.

    A raw observed gain remains unconfirmed unless its recorded noise gate and
    final validation establish it. Failed and noisy trials remain in the counts.
    """
    if budget < 1 or expected_seeds < 1:
        raise ValueError("budget and expected_seeds must be positive")
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    _ensure_trial_table(results_dir)
    rows = _load(results_dir)
    selected_rows, inventory = _select_runs(rows, results_dir, budget)
    valid = [row for row in selected_rows if _eligible(row)]
    baselines = _baselines(results_dir)
    qualifications = _qualifications(results_dir)
    gate_check = _read_json(results_dir / "gate-check.json")
    full_validation = _read_json(results_dir / "full-validation-baseline.json")
    failed = sum(not _true(row.get("correctness")) for row in rows)
    invalid = sum(_true(row.get("correctness")) and not _true(row.get("measurement_valid")) for row in rows)
    figures_dir = results_dir / "figures"
    e1_figures, e1_evidence = _e1_figures(results_dir, figures_dir)
    curve, coverage = _curves(selected_rows, figures_dir, budget, expected_seeds, inventory)
    # Include only candidates admitted to the measured-results table.
    validation = cross_validate(valid)
    (results_dir / "predictor-validation.json").write_text(json.dumps(validation, indent=2, allow_nan=False) + "\n")
    scatter = _scatter(validation, figures_dir)
    complete = all(item["complete"] for item in coverage.values())
    e1_attempts = [base for base in baselines if not base["qualification"] and base["repetitions"] >= 10]
    failed_e1 = bool(e1_attempts) and not any(base["e1_complete"] for base in e1_attempts)
    lines = ["# CompilerPilot measured results", ""]
    if failed_e1 and not rows:
        lines.extend(["**The E1 noise gate failed. Performance experiments E2–E5 were not performed.** "
                      "The recorded compiler/PGO builds below are functional qualifications only; no speedup or winning build is claimed.", ""])
    lines.extend([
             f"Recorded {len(rows)} trials; the E3 inventory contains {len(inventory)} experiments. {len(valid)} correctness-passing, measurement-valid trials belong to the selected single runs. "
             f"{failed} correctness/build failures; {invalid} correctness-passing trials excluded by the measurement gate.", "",
             f"E3 comparison: **{'complete' if complete else 'incomplete'}**. Target: four strategies × {expected_seeds} seeds × {budget} candidate builds.", "",
             "| Strategy | Observed builds per seed | Coverage |",
             "|---|---|---|"])
    for strategy in STRATEGIES:
        item = coverage[strategy]
        counts = ", ".join(f"seed {seed}: {count}/{budget}" for seed, count in item["builds_per_seed"].items()) or "no trials"
        lines.append(f"| {strategy} | {counts} | {len(item['seeds'])}/{expected_seeds} seeds; {'complete' if item['complete'] else 'incomplete'} |")
    sweep_count = sum(row.get("strategy") == "sweep" for row in rows)
    confirmation = _read_json(results_dir / "confirmation.json")
    lines.extend(["", "## Experiment status", "", "| Experiment | Recorded status |", "|---|---|"])
    e1_status = "failed noise gate" if failed_e1 else "passing noise gate recorded" if any(base["e1_complete"] for base in e1_attempts) else "not completed"
    lines.append(f"| E1: machine noise | {e1_status}; {sum(base['repetitions'] for base in e1_attempts)} baseline observations |")
    lines.append(f"| E2: one-knob sweep | {str(sweep_count) + ' recorded trials; completion not inferred' if sweep_count else 'not performed'} |")
    lines.append(f"| E3: fixed-budget search comparison | {'complete' if complete else 'incomplete' if selected_rows else 'not performed'} |")
    lines.append(f"| E4: held-out predictor quality | {'recorded below' if validation['mae_speedup_pct'] is not None else 'not performed; insufficient valid search trials'} |")
    lines.append(f"| E5: performance-winner confirmation | {confirmation.get('status', 'not performed; no performance winner')} |")
    lines.extend(["", "## Experiment inventory", "",
                  "Experiments are never combined within a strategy/seed. The fixed selection rule uses the earliest complete predeclared protocol run; if none qualifies, it shows the longest single partial run, breaking ties by start time and run ID. Throughput does not affect this selection. Every attempt remains listed below, including replaced or failed runs.", "",
                  "| Run | Strategy | Seed | Builds | Valid | Failed | Noise-excluded | Selected | Protocol status |",
                  "|---|---|---:|---:|---:|---:|---:|---|---|"])
    for item in inventory:
        lines.append(f"| {item['run_id']} | {item['strategy']} | {item['seed']} | {item['builds']} | {item['valid']} | {item['failed']} | {item['noise_excluded']} | {'yes' if item['selected'] else 'no'} | {item['status']} |")
    if not inventory:
        lines.append("\nNo search experiments are recorded.")
    lines.extend(["", "## E1: baseline noise", "",
                  "Baseline files are reported separately by experiment. The qualification measurement is a smoke check; E1 requires ten baseline observations. Noise gate: conservative coefficient of variation below 2%; candidate significance threshold: three baseline standard deviations.", "",
                  "| Baseline run | Observations | Throughput (M cmd/s) | p99 (ns) | Spread (M cmd/s) | CV | Compile (s) | .text (KiB) | Correct | Noise gate / E1 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---|---|"])
    for base in baselines:
        label = f"[{base['run_id']}]({base['path']})"
        noise = 'pass' if base['noise_gate'] else 'FAILED'
        e1 = 'measurement complete' if base['measurement_complete'] and not base['qualification'] else 'incomplete'
        lines.append(f"| {label} | {base['repetitions']} | {_fmt(base['throughput'] / 1e6, 3)} | {_fmt(base['p99_ns'], 1)} | "
                     f"{_fmt(base['spread'] / 1e6 if base['spread'] is not None else None, 3)} | {_fmt(base['cv'] * 100 if base['cv'] is not None else None)}% | "
                     f"{_fmt(base['compile_s'])} | {_fmt(base['text_bytes'] / 1024 if base['text_bytes'] is not None else None)} | "
                     f"{'pass' if base['correctness'] else 'FAIL'} | {noise}; E1 {e1} |")
    if not baselines:
        lines.append("\nNo baseline measurements are recorded.")
    for figure in e1_figures:
        lines.extend(["", f"![E1 raw baseline replay throughput and CV]({figure.relative_to(results_dir)})"])
    if e1_figures:
        lines.extend(["", "Every raw replay is plotted. Diamond markers show each observation median; the lower panel shows its CV. "
                      "The gate uses the larger within-observation or across-observation noise estimate. Figure sources and hashes are recorded in [e1-plot-data.json](e1-plot-data.json)."])
    lines.extend(["", "## Functional build qualification", "",
                  "These raw measurements establish that each compiler/PGO path builds, replays, and passes the per-build differential oracle. "
                  "They ran on a noisy shared host and are not a controlled performance comparison. No speedup ratio is calculated.", "",
                  "| Qualification build | Raw replays | Median (M cmd/s) | p99 (ns) | CV | Compile (s) | .text (KiB) | Correctness |",
                  "|---|---:|---:|---:|---:|---:|---:|---|"])
    for qualification in qualifications:
        lines.append(f"| [{qualification['label']}]({qualification['source']}) | {qualification['raw_runs']} | "
                     f"{_fmt(qualification['throughput'] / 1e6, 3)} | {_fmt(qualification['p99_ns'], 1)} | "
                     f"{_fmt(qualification['cv'] * 100 if qualification['cv'] is not None else None)}% | "
                     f"{_fmt(qualification['compile_s'])} | {_fmt(qualification['text_bytes'] / 1024 if qualification['text_bytes'] is not None else None)} | "
                     f"{'pass' if qualification['correctness'] else 'FAIL'} |")
    if not qualifications:
        lines.append("\nNo functional qualification measurements are recorded.")
    if gate_check:
        qualified = _true(gate_check.get('qualified')) and gate_check.get('correctness') is False
        lines.extend(["", f"Correctness-gate mutation check: **{'pass' if qualified else 'not qualified'}**. "
                      f"The intentionally modified matching logic {'built successfully' if gate_check.get('build_success') else 'did not build'}, "
                      f"and the harness {'rejected it' if gate_check.get('correctness') is False else 'did not record rejection'}. "
                      f"Pinned Ledger source unchanged: {'yes' if gate_check.get('pinned_source_unchanged') else 'not verified'}. "
                      "Evidence: [gate-check.json](gate-check.json)."])
    if full_validation:
        passed = _true(full_validation.get('correctness')) and _true(full_validation.get('ctest_evidence_verified'))
        commands = (full_validation.get('seeds', 0) * full_validation.get('orders', 0))
        lines.extend(["", f"Baseline functional correctness qualification: **{'pass' if passed else 'not verified'}** across "
                      f"{commands:,} differential commands ({full_validation.get('seeds', 0)} seeds × {full_validation.get('orders', 0):,} orders), "
                      f"with {full_validation.get('ctest_completed_tests', 0)}/{full_validation.get('ctest_planned_tests', 0)} CTest targets completed. "
                      "This validates the baseline's behavior; it is not E5 performance-winner confirmation. "
                      "Evidence: [full-validation-baseline.json](full-validation-baseline.json)."])
    lines.extend(["", "## Recorded search builds", "",
                  "Each strategy row is its best observed candidate, ranked by median raw speedup; repeated measurements of that candidate are summarized with medians. "
                  "Values include only trials passing both gates. A raw ratio is not a confirmed improvement.", "",
                  "| Build | Throughput (M cmd/s) | p99 (ns) | Compile (s) | .text (KiB) | Raw speedup | Baseline spread | Correctness | Noise significance |",
                  "|---|---:|---:|---:|---:|---:|---:|---|---|"])
    table_rows = 0
    for strategy in STRATEGIES:
        strategy_rows = [row for row in valid if row.get("strategy") == strategy]
        chosen = _best(strategy_rows)
        if chosen is None:
            continue
        measured, candidate = chosen
        throughput, text_bytes = _median(measured, "throughput"), _median(measured, "text_bytes")
        label = "Baseline" if strategy == "baseline" else f"Best {strategy}"
        compiler = candidate.get("compiler", "unknown compiler")
        opt = candidate.get("opt", "unknown optimization")
        label += f" ({compiler} -{opt})"
        relative_spreads = [spread / baseline for row in measured
                            if (spread := _number(row.get("baseline_spread"))) is not None
                            and (baseline := _number(row.get("baseline_throughput"))) is not None and baseline > 0]
        spread_label = f"{_fmt(max(relative_spreads) * 100)}%" if relative_spreads else "unavailable"
        significant = all(_true(row.get("significant")) for row in measured)
        significance = "exceeds recorded noise threshold" if significant else "not confirmed"
        lines.append(f"| {label} | {_fmt(throughput / 1e6 if throughput is not None else None, 3)} | {_fmt(_median(measured, 'p99_ns'), 1)} | "
                     f"{_fmt(_median(measured, 'compile_s'))} | {_fmt(text_bytes / 1024 if text_bytes is not None else None)} | "
                     f"{_fmt(_median(measured, 'speedup'), 4)}× | {spread_label} | pass | {significance} |")
        table_rows += 1
    if not table_rows:
        lines.extend(["", "No trials currently pass both correctness and measurement gates. No performance winner is reported."])
    if curve:
        lines.extend(["", "![Best-so-far strategy comparison](figures/best-so-far.png)", "",
                      "Bands show the observed seed range. Incomplete runs are labeled and curves stop at the shortest observed run; missing seeds are not imputed."])
    lines.extend(["", "## Predictor validation", ""])
    if validation["mae_speedup_pct"] is None:
        lines.append(f"Not enough distinct valid candidates for held-out validation ({validation['samples']} available).")
    else:
        lines.append(f"{validation['samples']} distinct candidates, {validation['folds']} folds: MAE {_fmt(validation['mae_speedup_pct'], 3)} speedup percentage points; "
                     f"Spearman {_fmt(validation['spearman'], 3)}. Duplicate candidates are aggregated before splitting; categorical encoding and model fitting happen within each fold.")
    if scatter:
        lines.extend(["", "![Held-out predictor validation](figures/predictor-validation.png)"])
    lines.extend(["", "## Interpretation and limits", "",
                  "Only recorded measurements appear here. Search results do not establish a final winner until interleaved confirmation, the full 100-seed differential test, and ctest pass. "
                  "These observations describe one benchmark on its recorded machine; CPU-specific flags and synthetic PGO training may not transfer. "
                  "Model and provider versions can change agent behavior. Unavailable hardware counters must be reported as unavailable rather than zero.", ""])
    summary_path = results_dir / "summary.md"
    summary_path.write_text("\n".join(lines))
    return {"summary": str(summary_path), "trials": len(rows), "eligible": len(valid), "failed": failed,
            "measurement_excluded": invalid, "complete": complete, "coverage": coverage, "runs": inventory, "baselines": baselines,
            "figures": [str(path) for path in [*e1_figures, curve, scatter] if path], "predictor": validation,
            "e1": e1_evidence, "qualifications": qualifications, "gate_check": bool(gate_check), "baseline_validation": bool(full_validation)}
