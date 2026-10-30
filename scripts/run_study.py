"""Run E1-E3 sequentially; a failed noise gate stops the study."""
import argparse
import os
from compilerpilot.cli import main


def run(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--provider',choices=['openai','anthropic'],default='openai')
    p.add_argument('--model',required=True)
    p.add_argument('--cpu',type=int,default=0)
    p.add_argument('--jobs',type=int,default=2)
    args=p.parse_args(argv)
    variable='OPENAI_API_KEY' if args.provider=='openai' else 'ANTHROPIC_API_KEY'
    if not os.getenv(variable):
        p.error(f'{variable} is required before starting the full study')
    common=['--cpu',str(args.cpu),'--jobs',str(args.jobs)]
    if main(['baseline','--run-id','E1',*common]):return 1
    if main(['sweep','--run-id','E2',*common]):return 1
    for seed in range(3):
        for strategy in ('random','genetic','surrogate','agent'):
            command=['experiment','--strategy',strategy,'--budget','40','--seed',str(seed),
                     '--run-id',f'E3-{strategy}-{seed}',*common]
            if strategy=='agent':command+=['--provider',args.provider,'--model',args.model]
            if main(command):return 1
    return main(['report'])

if __name__=='__main__':
    raise SystemExit(run())
