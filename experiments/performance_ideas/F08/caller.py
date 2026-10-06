#!/usr/bin/env python3
"""Backward-only A/B measured on training, including swapped views and recovery."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'apple_fast'))
from support import capture_main
from lm_task import train_case

def exercise(args):
    identity,case=train_case(args,resident=True,vocabulary=513)
    return dict(binding=identity,cases={'train-checkpoint-refusal':case})
if __name__=='__main__':capture_main(exercise)
