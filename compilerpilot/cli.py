"""Reproducible local experiments; credentials are read only from the environment."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
from .space import Candidate, KNOBS
from .build import BuildManager, BuildResult, file_sha256, run_command, source_provenance
from .bench import BenchmarkConfig, BenchmarkRunner
from .harness import Harness
from .store import Store

PROJECT = Path(__file__).resolve().parents[1]


def prepare_data(ledger: Path, data: Path, commands: int, evaluation_seed: int, training_seed: int,
                 jobs: int = 2) -> dict:
    if evaluation_seed == training_seed:
        raise ValueError('Training and evaluation seeds must differ')
    if not 1 <= commands <= 5_000_000:
        raise ValueError('commands must be in 1..5000000')
    data.mkdir(parents=True, exist_ok=True)
    manifest_path = data / 'manifest.json'
    requested = {'commands': commands, 'evaluation_seed': evaluation_seed, 'training_seed': training_seed}
    source = source_provenance(ledger)
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if all(previous.get(k) == v for k, v in requested.items()) and previous.get('source') == source:
            if all((data/name).is_file() and file_sha256(data/name)==previous['files'][name]['sha256'] for name in ('eval.lob','train.lob')):
                return previous
        raise ValueError('Existing data differs from requested protocol; choose another --data directory')
    bootstrap = data / 'generator-build'
    configure = run_command(['cmake','-S',ledger,'-B',bootstrap,'-DCMAKE_BUILD_TYPE=Release',
                             '-DBUILD_TESTING=OFF','-DLEDGER_NATIVE=OFF'], timeout=120,log=data/'configure.log')
    if not configure.ok:
        raise RuntimeError(configure.output)
    built = run_command(['cmake','--build',bootstrap,'--target','ledger_generate','--parallel',str(jobs)],
                        timeout=300,log=data/'build.log')
    if not built.ok:
        raise RuntimeError(built.output)
    files = {}
    for name, seed in [('eval.lob',evaluation_seed),('train.lob',training_seed)]:
        generated = run_command([bootstrap/'ledger_generate','--commands',str(commands),'--seed',str(seed),
                                 '--out',data/name],timeout=120,log=data/(name+'.log'))
        if not generated.ok:
            raise RuntimeError(generated.output)
        files[name] = {'sha256':file_sha256(data/name),'seed':seed,'generator':json.loads(generated.output)}
    manifest = {**requested,'source':source,'files':files}
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def _common(parser):
    parser.add_argument('--ledger',type=Path,default=PROJECT/'ledger')
    parser.add_argument('--data',type=Path,default=PROJECT/'data')
    parser.add_argument('--results',type=Path,default=PROJECT/'results')
    parser.add_argument('--commands',type=int,default=1_000_000)
    parser.add_argument('--evaluation-seed',type=int,default=20261011)
    parser.add_argument('--training-seed',type=int,default=20261012)
    parser.add_argument('--cpu',type=int,default=None)
    parser.add_argument('--jobs',type=int,default=2)
    parser.add_argument('--baseline-repetitions',type=int,default=10)
    parser.add_argument('--run-id',default=None)


def _harness(args, strategy, budget):
    run_id = args.run_id or f'{strategy}-{args.seed if hasattr(args,"seed") else 0}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%f}'
    store = Store(args.results.resolve(),run_id)
    manifest = prepare_data(args.ledger.resolve(),args.data.resolve(),args.commands,args.evaluation_seed,args.training_seed,args.jobs)
    store.write('inputs.json',manifest)
    manager = BuildManager(args.ledger,PROJECT/'builds'/run_id,jobs=args.jobs)
    runner = BenchmarkRunner(BenchmarkConfig(args.data/'eval.lob',cpu=args.cpu))
    harness = Harness(manager,runner,store,budget=budget,strategy=strategy,
                      seed=getattr(args,'seed',0),training_input=args.data/'train.lob')
    store.write('protocol.json',{'strategy':strategy,'seed':getattr(args,'seed',0),'budget':budget,
                                'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                                'python':sys.version,'temperature':0 if strategy=='agent' else None,
                                'model':getattr(args,'model',None),'provider':getattr(args,'provider',None)})
    baseline = harness.establish_baseline(args.baseline_repetitions)
    if not baseline['noise_gate']:
        raise RuntimeError(f'Noise gate failed: CV {baseline["metric"]["cv"]:.2%}; evidence saved in {store.run_dir}')
    return harness


def experiment(args):
    if args.strategy == 'agent':
        if not args.model:
            raise ValueError('--model must pin the exact provider model for an agent run')
        key = 'OPENAI_API_KEY' if args.provider=='openai' else 'ANTHROPIC_API_KEY'
        if not os.getenv(key):
            raise RuntimeError(f'{key} is required for a real agent run')
    h = _harness(args,args.strategy,args.budget)
    try:
        if args.strategy == 'agent':
            from .strategies.agent import AgentRunner
            outcome = AgentRunner(h,provider=args.provider,model=args.model,seed=args.seed,max_calls=args.budget*12).run(args.budget)
            if outcome.get('status') != 'finished':
                raise RuntimeError(f'Agent experiment incomplete: {outcome.get("status")}')
        else:
            if args.strategy=='random':
                from .strategies.random_search import RandomSearch as Policy
            elif args.strategy=='genetic':
                from .strategies.genetic import GeneticSearch as Policy
            else:
                from .strategies.surrogate import SurrogateSearch as Policy
            policy = Policy(seed=args.seed,compilers=tuple(args.compilers.split(',')))
            for _ in range(args.budget):
                candidate=policy.propose(h.history,h.seen)
                if candidate is None:
                    raise RuntimeError('Search space exhausted before the fixed budget')
                trial=h.evaluate(candidate)
                print(json.dumps({k:trial.get(k) for k in ('build_index','build_id','throughput','correctness','speedup','status')}),flush=True)
                if not h.noise_ok:
                    raise RuntimeError('Baseline drift: batch discarded. Rerun on a quiet machine.')
            eligible=[r for r in h.history if r.get('measurement_valid') and r.get('significant')]
            best=max(eligible,key=lambda r:r['throughput'])['build_id'] if eligible else h.baseline.build_id
            outcome=h.finish(best,'Highest correctness-gated throughput above the noise threshold; interleaved confirmation is required.')
        h.store.write('outcome.json',outcome)
        h.store.export()
        print(json.dumps({'run':str(h.store.run_dir),'outcome':outcome},indent=2))
    except Exception as exc:
        h.trace('stopped',{'error':str(exc),'builds_used':h.builds_used,'complete':False})
        h.store.write('outcome.json',{'status':'incomplete','error':str(exc),'builds_used':h.builds_used})
        h.store.export()
        raise


def one_at_a_time(args):
    h=_harness(args,'sweep',100)
    baseline=Candidate().to_dict()
    for knob,values in KNOBS.items():
        for value in values:
            if value==baseline[knob] and type(value) is type(baseline[knob]):
                continue
            candidate=Candidate.from_dict({**baseline,knob:value})
            row=h.evaluate(candidate)
            h.trace('one_at_a_time',{'knob':knob,'value':value,'trial':row})
            if not h.noise_ok:
                raise RuntimeError('Sweep stopped after baseline drift; discarded batch must be rerun')
    h.checkpoint()
    h.store.export()
    print(h.store.run_dir)


def confirmation(args):
    from .confirm import confirm_candidates
    source=json.loads(args.baseline.read_text())
    baseline=BuildResult.from_dict(source.get('build',source))
    candidates=[]
    for path in args.candidates:
        loaded=json.loads(path.read_text())
        candidates.append(BuildResult.from_dict(loaded.get('build',loaded)))
    runner=BenchmarkRunner(BenchmarkConfig(args.input,cpu=args.cpu))
    result=confirm_candidates(runner,baseline,candidates,rounds=args.rounds,full_validation=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'recommendation':result.get('recommendation'),'output':str(args.output)}))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('data');_common(p)
    p=sub.add_parser('baseline');_common(p)
    p=sub.add_parser('sweep');_common(p)
    p=sub.add_parser('experiment');_common(p)
    p.add_argument('--strategy',choices=['random','genetic','surrogate','agent'],required=True)
    p.add_argument('--budget',type=int,default=40);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--compilers',default='gcc,clang')
    p.add_argument('--provider',choices=['openai','anthropic'],default='openai');p.add_argument('--model')
    p=sub.add_parser('confirm');p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--candidates',type=Path,nargs='+',required=True);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--rounds',type=int,default=10);p.add_argument('--cpu',type=int,default=None)
    p.add_argument('--output',type=Path,default=PROJECT/'results'/'confirmation.json')
    p=sub.add_parser('report');p.add_argument('--results',type=Path,default=PROJECT/'results')
    args=parser.parse_args(argv)
    try:
        if args.command=='data':
            print(json.dumps(prepare_data(args.ledger.resolve(),args.data.resolve(),args.commands,args.evaluation_seed,args.training_seed,args.jobs),indent=2))
        elif args.command=='baseline':
            h=_harness(args,'baseline',1)
            h.trace('counters',h.get_counters(h.baseline.build_id))
            print(json.dumps({'run':str(h.store.run_dir),'noise_gate':h.noise_ok,'baseline':h.baseline_metric},indent=2))
        elif args.command=='sweep':one_at_a_time(args)
        elif args.command=='experiment':experiment(args)
        elif args.command=='confirm':confirmation(args)
        else:
            from .plot import generate_report
            print(json.dumps(generate_report(args.results),indent=2))
        return 0
    except (ValueError,RuntimeError,OSError) as exc:
        print(str(exc),file=sys.stderr)
        return 1

if __name__=='__main__':
    raise SystemExit(main())
