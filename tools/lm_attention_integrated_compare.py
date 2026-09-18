#!/usr/bin/env python3
"""Merged-source qualification against the saved independent legacy run."""
import copy
import json
import statistics
import sys
from pathlib import Path
from lm_attention_repair_compare import compare, default_capacity, HASHES


def reports(old, new, verbose=False):
    for x,y in zip(old['steps'],new['steps']):
        for key in ('forward_status','backward_status','backward_repair_sites'):
            assert x['attention'][key] == y['attention'][key], key
            if verbose: print('MATCH step',x['step'],key,y['attention'][key])
        assert y['attention']['stage_lists'] == [12,12], 'stage_lists'


def main():
    root = Path(__file__).resolve().parents[1]
    baseline = root/'bench/results/lm_attention_fallback_2026-09-18/default/enwik8'
    a = json.loads((baseline/'legacy/result.json').read_text())
    old = json.loads((baseline/'repaired/result.json').read_text())
    b = json.loads(Path(sys.argv[1]).read_text())
    for key in HASHES:
        broken = copy.deepcopy(b); broken['steps'][-1][key] = 'corrupted'
        try: compare(a, broken, False)
        except AssertionError as e:
            assert str(e) == key, str(e)
            print('EXPECTED FAIL', key)
        else: raise AssertionError('BLIND '+key)
    compare(a,b)
    default_capacity(b)
    for key in ('forward_status','backward_status','backward_repair_sites','stage_lists'):
        broken=copy.deepcopy(b); broken['steps'][-1]['attention'][key]=[]
        try: reports(old,broken)
        except AssertionError as e:
            assert str(e)==key, str(e)
            print('EXPECTED FAIL',key)
        else: raise AssertionError('BLIND '+key)
    reports(old,b,True)
    rows=b['steps'][-50:]
    print('MERGED tail seconds',statistics.median(x['seconds'] for x in rows),
          'device MiB',[x['device_used_mb'] for x in rows if x['device_used_mb'] is not None])


if __name__ == '__main__': main()
