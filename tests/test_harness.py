import json
from pathlib import Path

import pandas as pd
import pytest

from compilerpilot.build import BuildResult
from compilerpilot.harness import Harness
from compilerpilot.space import Candidate
from compilerpilot.store import Store


def metric(throughput=100.0, correctness=True, cv=.001):
    return {'throughput': throughput, 'correctness': correctness,
            'cv': cv, 'spread': throughput * cv,
            'p50_ns': 10.0, 'p99_ns': 20.0, 'runs': [throughput] * 5}


class Manager:
    def __init__(self, root, failed=()):
        self.root = root
        self.failed = set(failed)
        self.calls = []

    def build(self, candidate, training_input=None):
        self.calls.append(candidate)
        path = self.root / candidate.hash()
        path.mkdir(exist_ok=True)
        return BuildResult(candidate.hash(), candidate.to_dict(), str(path),
                           success=candidate.hash() not in self.failed,
                           compile_s=1.0, text_bytes=1024,
                           error='broken compilation' if candidate.hash() in self.failed else None)


class Runner:
    def __init__(self, candidate_metrics=None, baseline_values=None, checkpoint=100.0):
        self.candidate_metrics = candidate_metrics or {}
        self.baseline_values = iter(baseline_values or [100.0] * 10)
        self.checkpoint = checkpoint
        self.counter_calls = []

    def run(self, build, label='bench'):
        if label.startswith('baseline-'):
            return metric(next(self.baseline_values))
        if label.startswith('checkpoint-'):
            return metric(self.checkpoint)
        return self.candidate_metrics.get(build.build_id, metric(110))

    def counters(self, build):
        self.counter_calls.append(build.build_id)
        return {'available': True, 'cycles': 100 if build.candidate['opt'] == 'O3' else 80,
                'instructions': 200, 'ipc': 2.0, 'branch_miss_pct': None}


def ready(tmp_path, budget=1, failed=(), runner=None):
    store = Store(tmp_path / 'results', 'test')
    manager = Manager(tmp_path, failed=failed)
    harness = Harness(manager, runner or Runner(), store, budget=budget)
    harness.establish_baseline(repetitions=2)
    return harness


def test_failed_build_counts_budget_and_can_never_be_recommended(tmp_path):
    candidate = Candidate(opt='O2')
    harness = ready(tmp_path, failed=[candidate.hash()])
    result = harness.evaluate(candidate)
    assert harness.builds_used == 1
    assert result['correctness'] is False
    with pytest.raises(ValueError, match='correctness'):
        harness.finish(candidate.hash(), 'fast')
    assert not harness.finished
    assert harness.finish(harness.baseline.build_id, 'All alternatives failed')['recommendation'] == harness.baseline.build_id
    frame = pd.read_parquet(tmp_path / 'results/trials.parquet')
    assert len(frame) == 1 and not bool(frame.iloc[0]['correctness'])


def test_correctness_failure_never_eligible_even_if_fast(tmp_path):
    candidate = Candidate(opt='O2')
    harness = ready(tmp_path, runner=Runner(candidate_metrics={candidate.hash(): metric(999, False)}))
    result = harness.evaluate(candidate)
    assert result['speedup'] is None and not result['significant']
    with pytest.raises(ValueError, match='correctness'):
        harness.finish(candidate.hash(), 'fast')


def test_duplicates_and_baseline_rebuild_do_not_consume_budget(tmp_path):
    harness = ready(tmp_path, budget=2)
    with pytest.raises(ValueError, match='Duplicate'):
        harness.build(Candidate().to_dict())
    assert harness.builds_used == 0 and len(harness.manager.calls) == 1
    harness.evaluate(Candidate(opt='O2'))
    with pytest.raises(ValueError, match='Duplicate'):
        harness.build(Candidate(opt='O2').to_dict())
    assert harness.builds_used == 1
    with pytest.raises(ValueError, match='full requested'):
        harness.finish(harness.baseline.build_id, 'Too soon')


def test_pending_measurement_and_build_indices(tmp_path):
    harness = ready(tmp_path, budget=2)
    first, second = Candidate(opt='O2'), Candidate(unroll=True)
    harness.build(first.to_dict())
    harness.build(second.to_dict())
    with pytest.raises(ValueError, match='every successful'):
        harness.finish(harness.baseline.build_id, 'Missing data')
    harness.run_benchmark(first.hash())
    harness.run_benchmark(second.hash())
    assert [row['build_index'] for row in harness.history] == [1, 2]
    assert len(pd.read_parquet(tmp_path / 'results/trials.parquet')) == 2


def test_noise_gate_prevents_search(tmp_path):
    harness = Harness(Manager(tmp_path), Runner(baseline_values=[80, 120]),
                      Store(tmp_path / 'results', 'noisy'), budget=1)
    assert not harness.establish_baseline(repetitions=2)['noise_gate']
    with pytest.raises(RuntimeError, match='Noise gate'):
        harness.build(Candidate(opt='O2').to_dict())
    assert harness.builds_used == 0


def test_drift_invalidates_json_and_parquet_snapshot(tmp_path):
    harness = ready(tmp_path, runner=Runner(checkpoint=110.0))
    candidate = Candidate(opt='O2')
    harness.evaluate(candidate)
    assert harness.history[0]['significant']
    with pytest.raises(ValueError):
        harness.finish(candidate.hash(), 'Appears fast')
    assert not harness.noise_ok
    row = json.loads((harness.store.run_dir / 'history.json').read_text())[0]
    frame = pd.read_parquet(tmp_path / 'results/trials.parquet')
    assert row['status'] == 'discarded_drift' and not row['measurement_valid']
    assert frame.iloc[0]['status'] == 'discarded_drift'
    assert not bool(frame.iloc[0]['measurement_valid'])
    with pytest.raises(RuntimeError, match='Noise gate'):
        harness.build(Candidate(unroll=True).to_dict())


def test_three_spreads_threshold_and_stable_finish(tmp_path):
    candidate = Candidate(opt='O2')
    harness = ready(tmp_path, runner=Runner(candidate_metrics={candidate.hash(): metric(100.2)}))
    assert not harness.evaluate(candidate)['significant']
    with pytest.raises(ValueError, match='three baseline'):
        harness.finish(candidate.hash(), 'Within noise')
    assert harness.finish(harness.baseline.build_id, 'No defensible improvement')['status'] == 'provisional_pending_confirmation'


def test_counter_feedback_includes_cached_baseline_comparison(tmp_path):
    harness = ready(tmp_path)
    candidate = Candidate(opt='O2')
    harness.build(candidate.to_dict())
    result = harness.get_counters(candidate.hash())
    assert result['baseline']['cycles'] == 100
    assert result['vs_baseline']['cycles'] == pytest.approx(-.2)
    harness.get_counters(candidate.hash())
    assert len(harness.runner.counter_calls) == 2


def test_store_append_does_not_lose_trial_under_stale_history(tmp_path):
    store = Store(tmp_path, 'append')
    first = {'build_id': 'a', 'measurement_valid': True, 'candidate': {'opt': 'O3'}}
    second = {'build_id': 'b', 'measurement_valid': True, 'candidate': {'opt': 'O2'}}
    store.append(first)
    store.write('history.json', [first])
    store.append(second)
    assert list(pd.read_parquet(tmp_path / 'trials.parquet')['build_id']) == ['a', 'b']
    store.write('history.json', [{**first, 'measurement_valid': False}, second])
    store.export()
    frame = pd.read_parquet(tmp_path / 'trials.parquet')
    assert not bool(frame.iloc[0]['measurement_valid'])
    assert len((store.run_dir / 'trials.jsonl').read_text().splitlines()) == 2


def test_twentieth_failed_build_also_triggers_drift_checkpoint(tmp_path):
    candidates = [Candidate(compiler='clang', opt='O2' if i & 1 else 'O3', isa='none' if i & 2 else 'native',
                            lto=bool(i & 4), unroll=bool(i & 8), vectorize=bool(i & 16))
                  for i in range(1, 21)]
    harness = ready(tmp_path, budget=20, failed=[candidate.hash() for candidate in candidates],
                    runner=Runner(checkpoint=110.0))
    for candidate in candidates:
        harness.evaluate(candidate)
    assert harness.builds_used == 20 and harness.last_checkpoint == 20
    assert not harness.noise_ok
    assert all(row['status'] == 'discarded_drift' for row in harness.history)


def test_checkpoint_waits_for_pending_twenty_then_blocks_new_batch(tmp_path):
    candidates = [Candidate(compiler='clang', opt='O2' if i & 1 else 'O3',
                            isa='none' if i & 2 else 'native', lto=bool(i & 4),
                            unroll=bool(i & 8), vectorize=bool(i & 16))
                  for i in range(1, 22)]
    harness = ready(tmp_path, budget=21)
    for candidate in candidates[:18]:
        harness.evaluate(candidate)
    harness.build(candidates[18].to_dict())
    harness.build(candidates[19].to_dict())
    harness.run_benchmark(candidates[19].hash())
    assert harness.last_checkpoint == 0
    with pytest.raises(RuntimeError, match='pending builds'):
        harness.build(candidates[20].to_dict())
    assert harness.builds_used == 20
    harness.run_benchmark(candidates[18].hash())
    assert harness.last_checkpoint == 20
    harness.evaluate(candidates[20])
    assert harness.builds_used == 21
    events = [json.loads(line) for line in (harness.store.run_dir / 'trace.jsonl').read_text().splitlines()]
    checkpoints = [event for event in events if event['event'] == 'baseline_checkpoint']
    assert len(checkpoints) == 1


def test_candidate_noise_and_input_change_are_not_recommendable(tmp_path):
    candidate = Candidate(opt='O2')
    noisy = metric(150, cv=.03)
    harness = ready(tmp_path, runner=Runner(candidate_metrics={candidate.hash(): noisy}))
    row = harness.evaluate(candidate)
    assert row['correctness'] and not row['measurement_valid'] and not row['significant']
    assert row['status'] == 'discarded_candidate_noise'
    with pytest.raises(ValueError, match='measurement gates'):
        harness.finish(candidate.hash(), 'Noisy speedup')


def test_source_identity_change_is_failed_build(tmp_path):
    harness = ready(tmp_path)
    harness.baseline.provenance['source_sha256'] = 'original'
    row = harness.evaluate(Candidate(opt='O2'))
    assert not row['correctness'] and row['status'] == 'failed'
    assert 'source identity' in row['error']


def test_evaluation_input_sha_change_invalidates_candidate(tmp_path):
    harness = ready(tmp_path)
    harness.baseline_metric['provenance'] = {'input_sha256': 'original'}
    row = harness.evaluate(Candidate(opt='O2'))
    assert row['correctness'] and not row['measurement_valid']
    assert row['status'] == 'discarded_input_change'



def test_history_is_compact_and_does_not_mutate_full_evidence(tmp_path):
    harness = ready(tmp_path)
    candidate = Candidate(opt='O2')
    harness.evaluate(candidate)
    harness.history[0]['build_provenance'] = {'commands': ['very long command'] * 50}
    harness.history[0]['error'] = 'x' * 2000
    harness.baseline_metric['provenance'] = {'commands': ['private full evidence']}
    summary = harness.query_history()
    assert 'build_provenance' not in summary['best'][0]
    assert 'provenance' not in summary['baseline']
    assert len(summary['best'][0]['error']) == 1000
    summary['best'][0]['candidate']['opt'] = 'changed'
    assert harness.history[0]['candidate']['opt'] == 'O2'
    assert len(harness.history[0]['error']) == 2000



def test_compile_failure_keeps_diagnostic_without_benchmark_input_provenance(tmp_path):
    candidate = Candidate(opt='O2')
    harness = ready(tmp_path, failed=[candidate.hash()])
    harness.baseline_metric['provenance'] = {'input_sha256': 'fixed-evaluation-input'}
    row = harness.evaluate(candidate)
    assert row['error'] == 'broken compilation'
    assert not row['correctness']
