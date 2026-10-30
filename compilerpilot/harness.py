"""All policies use the same budget, correctness oracle and noise rules."""
from __future__ import annotations
import hashlib
import json
import math
import statistics
from pathlib import Path
from .space import Candidate, list_knobs
from .build import BuildManager, BuildResult
from .bench import BenchmarkRunner
from .feedback import summarize_feedback
from .store import Store


class Harness:
    def __init__(self, manager: BuildManager, runner: BenchmarkRunner, store: Store,
                 budget: int = 40, strategy: str = 'random', seed: int = 0,
                 training_input: Path | None = None):
        if not 1 <= budget <= 10000:
            raise ValueError('budget must be in 1..10000')
        self.manager, self.runner, self.store = manager, runner, store
        self.budget, self.strategy, self.seed = budget, strategy, seed
        self.training_input = training_input
        self.builds_used = 0
        self.builds: dict[str, BuildResult] = {}
        self.build_indices: dict[str, int] = {}
        self.counter_cache: dict[str, dict] = {}
        self.seen: set[str] = set()
        self.history: list[dict] = []
        self.baseline: BuildResult | None = None
        self.baseline_metric: dict | None = None
        self.noise_ok = False
        self.finished = False
        self.last_checkpoint = 0

    def trace(self, event, data):
        self.store.trace(event, data)

    def list_knobs(self):
        return list_knobs()

    def establish_baseline(self, repetitions: int = 10) -> dict:
        if self.baseline is not None:
            raise RuntimeError('Baseline is already established')
        if repetitions < 2:
            raise ValueError('At least two baseline observations required')
        self.baseline = self.manager.build(Candidate())
        if not self.baseline.success:
            raise RuntimeError(f'Baseline build failed: {self.baseline.error}')
        self.builds[self.baseline.build_id] = self.baseline
        self.seen.add(Candidate().hash())
        observations = []
        for i in range(repetitions):
            result = self.runner.run(self.baseline, label=f'baseline-{i}')
            observations.append(result)
            self.trace('baseline_observation', result)
            if not result.get('correctness'):
                raise RuntimeError(f'Baseline failed correctness: {result.get("error")}')
        input_hashes = {o.get('provenance', {}).get('input_sha256') for o in observations}
        if len(input_hashes) > 1:
            raise RuntimeError('Evaluation input changed during baseline observations')
        values = [o['throughput'] for o in observations]
        median = statistics.median(values)
        # Use the larger within-replay or across-observation noise estimate.
        std = max(statistics.stdev(values), max(o.get('cv', 0) * o['throughput'] for o in observations))
        cv = std / median
        self.noise_ok = cv < .02
        self.baseline_metric = dict(observations[0], throughput=median, cv=cv, spread=std,
                                    p50_ns=statistics.median(o['p50_ns'] for o in observations),
                                    p99_ns=statistics.median(o['p99_ns'] for o in observations))
        result = {'build': self.baseline.to_dict(), 'metric': self.baseline_metric,
                  'observations': observations, 'noise_gate': self.noise_ok,
                  'repetitions': repetitions, 'threshold_throughput': median + 3 * std}
        self.store.write('baseline.json', result)
        self.trace('baseline', result)
        return result

    def _require_ready(self):
        if self.baseline_metric is None:
            raise RuntimeError('Establish baseline first')
        if not self.noise_ok:
            raise RuntimeError('Noise gate failed: baseline variation must be below 2%')
        if self.finished:
            raise RuntimeError('Run already finished')

    def build(self, candidate: dict) -> dict:
        self._require_ready()
        if self.builds_used and self.builds_used % 20 == 0 and self.last_checkpoint < self.builds_used:
            if self._pending_builds(self.builds_used):
                raise RuntimeError('Benchmark pending builds before starting the next 20-build batch')
            self.checkpoint()
            self._require_ready()
        parsed = Candidate.from_dict(candidate)
        if self.builds_used >= self.budget:
            raise RuntimeError('Build budget exhausted')
        if parsed.hash() in self.seen:
            raise ValueError('Duplicate candidate')
        if parsed.pgo and self.training_input is None:
            raise ValueError('PGO requires a separate training input')
        self.seen.add(parsed.hash())
        self.builds_used += 1
        built = self.manager.build(parsed, training_input=self.training_input)
        self.builds[built.build_id] = built
        self.build_indices[built.build_id] = self.builds_used
        baseline_source = self.baseline.provenance.get('source_sha256')
        candidate_source = built.provenance.get('source_sha256')
        baseline_commit = self.baseline.provenance.get('ledger_commit')
        candidate_commit = built.provenance.get('ledger_commit')
        if ((baseline_source is not None and candidate_source != baseline_source)
                or (baseline_commit is not None and candidate_commit != baseline_commit)):
            built.success = False
            built.error = 'Ledger source identity changed after baseline; only compiler choices may vary'
        self.trace('build', built.to_dict())
        if not built.success:
            self._record(built, {'correctness': False, 'error': built.error})
        return built.to_dict()

    def _record(self, built, metric):
        baseline = self.baseline_metric['throughput']
        good = metric.get('correctness') is True
        cv = metric.get('cv')
        valid = good and type(cv) in (int, float) and math.isfinite(cv) and 0 <= cv < .02
        baseline_input = self.baseline_metric.get('provenance', {}).get('input_sha256')
        input_matches = (not good or baseline_input is None or
                         metric.get('provenance', {}).get('input_sha256') == baseline_input)
        valid = valid and input_matches
        if not input_matches:
            metric = dict(metric, error='Evaluation input SHA256 changed after baseline')
        throughput = metric.get('throughput') if good else None
        std = self.baseline_metric['spread']
        trial = {
            'run_id': self.store.run_dir.name, 'strategy': self.strategy, 'seed': self.seed,
            'build_index': self.build_indices[built.build_id], 'build_id': built.build_id,
            'candidate': built.candidate, 'compile_s': built.compile_s,
            'text_bytes': built.text_bytes, 'compiler_version': built.compiler_version,
            'flags': built.flags, 'build_provenance': built.provenance,
            **metric, 'correctness': good, 'measurement_valid': valid,
            'baseline_throughput': baseline, 'baseline_spread': std,
            'speedup': throughput / baseline if throughput else None,
            'significant': bool(valid and throughput and throughput > baseline + 3 * std),
            'status': ('measured' if valid else ('discarded_input_change' if not input_matches
                       else 'discarded_candidate_noise')) if good else 'failed',
        }
        self.history.append(trial)
        self.store.append(trial)
        self.store.write('history.json', self.history)
        self.store.export()
        boundary = self.builds_used - self.builds_used % 20
        if boundary > self.last_checkpoint and not self._pending_builds(boundary):
            self.checkpoint(boundary)
        return trial

    def run_benchmark(self, build_id: str) -> dict:
        if build_id not in self.builds:
            raise ValueError('Unknown build_id')
        if any(t['build_id'] == build_id for t in self.history):
            raise ValueError('Candidate already evaluated')
        built = self.builds[build_id]
        if built is self.baseline:
            raise ValueError('Baseline is measured separately')
        if not built.success:
            raise ValueError('Cannot benchmark failed build')
        metric = self.runner.run(built)
        trial = self._record(built, metric)
        return trial

    def evaluate(self, candidate: Candidate) -> dict:
        built = self.build(candidate.to_dict())
        if not built['success']:
            return self.history[-1]
        return self.run_benchmark(built['build_id'])

    def _pending_builds(self, up_to: int | None = None) -> list[str]:
        measured = {row['build_id'] for row in self.history}
        return [build_id for build_id, built in self.builds.items()
                if built is not self.baseline and built.success and build_id not in measured
                and (up_to is None or self.build_indices[build_id] <= up_to)]

    def checkpoint(self, boundary: int | None = None):
        boundary = self.builds_used if boundary is None else boundary
        if boundary <= self.last_checkpoint:
            return {'valid': self.noise_ok, 'already_checked': True}
        if self._pending_builds(boundary):
            raise ValueError('Complete pending benchmarks before checking baseline drift')
        check = self.runner.run(self.baseline, label=f'checkpoint-{boundary}')
        drift = abs(check.get('throughput', 0) / self.baseline_metric['throughput'] - 1)
        baseline_input = self.baseline_metric.get('provenance', {}).get('input_sha256')
        input_matches = (baseline_input is None or
                         check.get('provenance', {}).get('input_sha256') == baseline_input)
        valid = (check.get('correctness') and drift <= .02 and check.get('cv', 1) < .02
                 and input_matches)
        self.trace('baseline_checkpoint', {'result': check, 'drift': drift,
                   'boundary': boundary, 'valid': bool(valid)})
        if not valid:
            for row in self.history:
                if self.last_checkpoint < row['build_index'] <= boundary:
                    row['measurement_valid'] = False
                    row['significant'] = False
                    row['status'] = 'discarded_drift'
            self.store.write('history.json', self.history)
            self.store.export()
            self.noise_ok = False
        self.last_checkpoint = boundary
        return {'valid': bool(valid), 'drift': drift}

    def get_compiler_feedback(self, build_id: str):
        if build_id not in self.builds:
            raise ValueError('Unknown build_id')
        feedback = summarize_feedback(Path(self.builds[build_id].build_dir))
        feedback['build_id'] = build_id
        feedback['text_bytes'] = self.builds[build_id].text_bytes
        self.trace('compiler_feedback', feedback)
        return feedback

    def get_counters(self, build_id: str):
        if build_id not in self.builds:
            raise ValueError('Unknown build_id')
        if build_id not in self.counter_cache:
            self.counter_cache[build_id] = self.runner.counters(self.builds[build_id])
        value = dict(self.counter_cache[build_id])
        baseline_id = self.baseline.build_id if self.baseline else None
        if baseline_id and baseline_id not in self.counter_cache:
            self.counter_cache[baseline_id] = self.runner.counters(self.baseline)
        baseline = self.counter_cache.get(baseline_id, {})
        value['baseline'] = baseline
        value['vs_baseline'] = {key: value[key] / baseline[key] - 1
            for key in ('ipc', 'branch_miss_pct', 'icache_misses', 'cycles', 'instructions')
            if isinstance(value.get(key), (int, float)) and isinstance(baseline.get(key), (int, float))
            and baseline[key] != 0}
        self.trace('counters', {'build_id': build_id, 'result': value})
        return value

    def query_history(self, top_k: int = 10, sort_by: str = 'throughput'):
        if type(top_k) is not int or not 1 <= top_k <= 100:
            raise ValueError('top_k must be in 1..100')
        if sort_by not in ('throughput', 'speedup', 'compile_s', 'p99_ns'):
            raise ValueError('Unsupported sort key')
        rows = [r for r in self.history if r.get('correctness') and r.get('measurement_valid')]
        rows.sort(key=lambda r: r.get(sort_by) or 0, reverse=sort_by in ('throughput', 'speedup'))
        def compact(row):
            keys = ('build_id', 'candidate', 'build_index', 'throughput', 'p99_ns', 'p50_ns',
                    'cv', 'baseline_spread', 'speedup', 'significant', 'correctness',
                    'measurement_valid', 'compile_s', 'text_bytes', 'status')
            value = {key: row[key] for key in keys if key in row}
            if isinstance(value.get('candidate'), dict):
                value['candidate'] = dict(value['candidate'])
            if row.get('error'):
                value['error'] = str(row['error'])[:1000]
            return value
        baseline = ({key: self.baseline_metric[key]
                     for key in ('throughput', 'p50_ns', 'p99_ns', 'cv', 'spread', 'correctness')
                     if key in self.baseline_metric} if self.baseline_metric else None)
        failed = [r for r in self.history
                  if not r.get('correctness') or not r.get('measurement_valid')]
        return {'best': [compact(row) for row in rows[:top_k]],
                'worst': [compact(row) for row in rows[-top_k:]],
                'failed': [compact(row) for row in failed[-top_k:]],
                'builds_used': self.builds_used, 'budget_left': self.budget-self.builds_used,
                'noise_gate': self.noise_ok, 'baseline': baseline}

    def predict(self, candidates):
        from .strategies.surrogate import predict
        return predict([Candidate.from_dict(c) for c in candidates], self.history, seed=self.seed)

    def finish(self, recommendation: str, rationale: str):
        if self.builds_used < self.budget:
            raise ValueError('Use the full requested build budget before finish')
        pending = self._pending_builds()
        if pending:
            raise ValueError('Benchmark every successful build before finish')
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError('Rationale required')
        if self.last_checkpoint < len(self.history):
            self.checkpoint()
        valid = [r for r in self.history if r['build_id'] == recommendation and r.get('correctness') and r.get('measurement_valid')]
        if recommendation != self.baseline.build_id and not valid:
            raise ValueError('Recommendation must have passed correctness and measurement gates')
        if recommendation != self.baseline.build_id and not valid[0]['significant']:
            raise ValueError('Improvement must exceed three baseline standard deviations')
        if not self.noise_ok:
            raise ValueError('Cannot recommend after failed noise/drift gate')
        self.finished = True
        result = {'recommendation': recommendation, 'rationale': rationale,
                  'status': 'provisional_pending_confirmation', 'builds_used': self.builds_used}
        self.store.write('recommendation.json', result)
        self.trace('finish', result)
        return result
