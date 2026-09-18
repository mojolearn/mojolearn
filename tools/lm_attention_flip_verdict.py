#!/usr/bin/env python3
"""Mechanical two-corpus NVIDIA verdict, with an observed losing-time control."""
import copy
import json
import math
import statistics
import sys
from pathlib import Path
from lm_attention_repair_compare import compare, default_capacity


def verdict(pairs):
    ratios=[]
    for a,b in pairs:
        compare(a,b,False)
        default_capacity(b)
        before=statistics.median(x['seconds'] for x in a['steps'][-50:])
        after=statistics.median(x['seconds'] for x in b['steps'][-50:])
        assert math.isfinite(before) and math.isfinite(after) and before>0 and after>0, 'timing'
        ratios.append(after/before)
    assert len(ratios)==2, 'two corpora'
    gm=math.sqrt(math.prod(ratios))
    return gm<1,ratios,gm


def main():
    root=Path(sys.argv[1])
    pairs=[tuple(json.loads((root/corpus/arm/'result.json').read_text()) for arm in ('legacy','repaired')) for corpus in ('enwik8','pile_github')]
    broken=copy.deepcopy(pairs)
    for a,b in broken:
        for x,y in zip(a['steps'],b['steps']):y['seconds']=2*x['seconds']
    bad,_,gm=verdict(broken)
    assert bad is False and gm==2, 'BLIND losing-time verdict'
    print('EXPECTED NO FLIP deliberately doubled candidate times; geomean after/before=2')
    flip,ratios,gm=verdict(pairs)
    print(('FLIP' if flip else 'NO FLIP')+' NVIDIA: enwik8 after/before='+str(ratios[0])+'; Pile GitHub after/before='+str(ratios[1])+'; geomean='+str(gm)+'; every loss/state witness equal on both corpora')
    if not flip:raise SystemExit(1)


if __name__=='__main__':main()
