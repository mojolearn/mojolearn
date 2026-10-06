#!/usr/bin/env python3
"""Individual Mamba fusion arms at actual full forward callers and references."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'apple_fast'))
from support import ROOT,capture_main,binding_check,consumed

def exercise(args):
    import numpy as np
    from mojolearn import Mamba1Block,Mamba2Block,Mamba3Block
    from mojolearn import _mojolearn_mamba as binding
    sys.path.insert(0,str(ROOT/'python/mojolearn/tests'))
    import test_mamba_surface as fixture
    cases={}
    families=[('mamba2',Mamba2Block,fixture.m2_weights,'residual.out')]
    if args.variant in ('mamba1-input','chunk-scan'):
        families=[('mamba1',Mamba1Block,fixture.m1_weights,'block.out')]
    elif args.variant=='mamba3-elementwise':
        families=[('mamba3',Mamba3Block,fixture.m3_weights_corpus,'residual.out')]
    for family,cls,weights,stage in families:
        directory=ROOT/'mamba/corpus'/family if family!='mamba1' else ROOT/'mamba/corpus'
        selected=[p for p in sorted(directory.iterdir()) if (p/'manifest.json').is_file() and 'init_states' not in p.name]
        count=0
        for path in selected:
            manifest=json.loads((path/'manifest.json').read_text())
            reference=manifest.get('stages',{}).get(stage)
            if not reference:continue
            b,length,dm=manifest['B'],manifest['L'],manifest['d_model']
            if length not in (1,4,8,64,256,257):continue  # test fixture coverage, never product dispatch
            x=fixture.f32(str(path/'x.f32'),(b,length,dm))
            opts={'dt_limit':tuple(float(v['value']) for v in manifest['dt_limit'])} if 'dt_limit' in manifest else {}
            model=cls(weights(str(path),dm),**opts)
            output,elapsed=consumed(lambda:model.forward(x,model.allocate_state(b)))
            ref=np.fromfile(path/reference['ref64'],dtype='<f8')
            error=float(np.max(np.abs(np.asarray(output,float).reshape(-1)-ref))/max(np.max(np.abs(ref)),1e-9))
            cases[path.name]=dict(contract=dict(family=family,shape=[b,length,dm],reference=reference['ref64']),
                metrics=dict(full_forward_error=dict(value=error,rtol=1,atol=1e-6)),forward_ms=elapsed)
            count+=1
        assert count>=2,'missing meaningful independent Mamba corpus coverage'
    return dict(binding=binding_check(binding,'mamba'),cases=cases)
if __name__=='__main__':capture_main(exercise)
