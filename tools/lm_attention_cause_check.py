#!/usr/bin/env python3
"""Isolate per-layer refusals across legacy, guard-only, and exact replay arms."""
import copy
import json
import sys
from pathlib import Path
from lm_attention_repair_compare import HASHES


def check(a,b,c,verbose=False):
    assert a['shape']==b['shape']==c['shape']==[1,2048,768,12,12,64,2048,12,50257] and a['seed']==b['seed']==c['seed'], 'inputs'
    assert a['corpus']['sha256']==b['corpus']['sha256']==c['corpus']['sha256'], 'corpus'
    assert len(a['steps'])==len(b['steps'])==len(c['steps'])==700, 'coverage'
    guard_only=dq=0
    for i,(x,y,z) in enumerate(zip(a['steps'],b['steps'],c['steps'])):
        assert x['step']==y['step']==z['step']==i, 'sequence'
        assert x['loss']==y['loss']==z['loss'], 'loss'
        if verbose: print('MATCH three-arm step',i,'loss',x['loss'])
        if i in (0,699):
            for key in HASHES:
                assert x[key]==y[key]==z[key], key
                if verbose: print('MATCH three-arm step',i,key,x[key])
        ax,ay,az=x['attention'],y['attention'],z['attention']
        assert ax['exact_tail_guard'] is False and ay['exact_tail_guard'] is az['exact_tail_guard'] is True, 'arms'
        assert all(len(a[k])==12 for a in (ax,ay,az) for k in ('forward_status','backward_status')) and len(az['backward_repair_sites'])==12, 'status coverage'
        assert all(v==0 for a in (ax,ay,az) for v in a['forward_status']), 'forward'
        for layer,(old,guard,repaired,sites) in enumerate(zip(ax['backward_status'],ay['backward_status'],az['backward_status'],az['backward_repair_sites'])):
            assert old in (0,2) and guard in (0,2) and repaired==0, 'status'
            if old==2 and guard==0:
                assert sites==0, 'guard-only cause'
                guard_only+=1
                if verbose: print('MATCH step',i,'layer',layer,'legacy=CORNER guard=RAN replay=RAN sites=0; unnecessary dk/dv guard')
            elif guard==2:
                assert old==2 and sites==2, 'dQ cause'
                dq+=1
                if verbose: print('MATCH step',i,'layer',layer,'legacy=CORNER guard=CORNER replay=RAN sites=2; dQ masked-tail replay')
            else:
                assert old==0 and sites==0, 'unaffected site'
    if verbose: print('ISOLATED guard-only',guard_only,'dQ',dq,'zdot training INERT')


def main():
    root=Path(sys.argv[1])
    a=json.loads((root/'tail/legacy/result.json').read_text())
    b=json.loads((root/'tail/guarded/result.json').read_text())
    c=json.loads((root/'repair/repaired/result.json').read_text())
    active=next((i,j) for i,r in enumerate(b['steps']) for j,v in enumerate(r['attention']['backward_status']) if v==2)
    for name,mask in (('missing dQ activity',0),('invented zdot attribution',1)):
        broken=copy.deepcopy(c); i,j=active
        broken['steps'][i]['attention']['backward_repair_sites'][j]=mask
        try: check(a,b,broken)
        except AssertionError as e:
            assert str(e)=='dQ cause',str(e)
            print('EXPECTED FAIL attribution',name)
        else: raise AssertionError('BLIND attribution '+name)
    for key in HASHES:
        broken=copy.deepcopy(c);broken['steps'][-1][key]='corrupted'
        try: check(a,b,broken)
        except AssertionError as e:
            assert str(e)==key,str(e)
            print('EXPECTED FAIL three-arm',key)
        else: raise AssertionError('BLIND three-arm '+key)
    broken=copy.deepcopy(c);broken['steps'][-1]['attention']['backward_status'].pop()
    try: check(a,b,broken)
    except AssertionError as e:
        assert str(e)=='status coverage',str(e)
        print('EXPECTED FAIL attribution missing layer')
    else: raise AssertionError('BLIND missing layer')
    tiny=[copy.deepcopy(x) for x in (a,b,c)]
    for x in tiny:x['shape'][5]=8
    try: check(*tiny)
    except AssertionError as e:
        assert str(e)=='inputs',str(e)
        print('EXPECTED FAIL blind head_dim=8 shape')
    else: raise AssertionError('BLIND head_dim=8')
    check(a,b,c,True)


if __name__=='__main__':main()
