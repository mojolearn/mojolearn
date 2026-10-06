#!/usr/bin/env python3
"""Real byte-LM retained sessions versus stateless calls, output-consuming time."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'apple_fast'))
from support import capture_main
from lm_task import train_case

def exercise(args):
    identity, case = train_case(args, resident=args.arm=='B')
    # Runtime lifecycle is the experiment switch; same frozen binary per arm.
    case['contract']['retention_cap']='native bounded session allocation'
    _, alternate = train_case(args,resident=args.arm=='B',vocabulary=131)
    alternate['contract']['retention_cap']='native bounded session allocation'
    return dict(binding=identity,cases={'cold-and-repeated-session':case,'alternate-vocabulary-shape':alternate})
if __name__=='__main__': capture_main(exercise)
