#!/usr/bin/env python3
"""Cloud-only whole-tensor pooled clipping and original two-level norm bits."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cloud', action='store_true', required=True)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--faults', action='store_true')
    args = p.parse_args()
    if not os.environ.get('RUNPOD_POD_ID'):
        raise SystemExit('RunPod required; no local execution')
    import numpy as np
    from mojolearn._training_impl import _load
    binding = _load('identical')
    assert binding.clip_parallel_available() == 1
    rng = np.random.default_rng(737261)
    checks = []

    def invoke(g, offsets, info, devices, maximum):
        os.environ['MOJOLEARN_OPTIMIZER_DEVICE_COUNT'] = str(devices)
        return binding.clip_grad_norm(g.ctypes.data,offsets.ctypes.data,info.ctypes.data,[len(offsets)-1,maximum])

    for sizes in ([1],[3,11,7,80],[129,3,513,17,65],[65537,3,1025,257,31], [3]*129):
        offsets = np.asarray([0]+list(np.cumsum(sizes)),dtype='<i4')
        for distribution in ('normal','zeros','scaled'):
            initial = rng.normal(0,.1,int(offsets[-1])).astype('<f4')
            if distribution == 'zeros':
                initial.fill(0)
            elif distribution == 'scaled':
                initial[::2] *= np.float32(65536)
                initial[1::2] *= np.float32(1/65536)
            initial.view('<u4')[0] = 0x80000000
            if len(initial)>1:
                initial.view('<u4')[1] = 1
            for maximum in (.0001, .1, 1e8):
                left,right = initial.copy(),initial.copy()
                a,b = np.full(2,-17,dtype='<f4'),np.full(2,-17,dtype='<f4')
                assert invoke(left,offsets,a,1,maximum) == len(left)
                assert invoke(right,offsets,b,2,maximum) == len(right)
                assert left.tobytes() == right.tobytes(), (sizes,distribution,maximum,'gradients')
                assert a.tobytes() == b.tobytes(), (sizes,distribution,maximum,'norm/coefficient')
                checks.append(dict(sizes=sizes,distribution=distribution,maximum=maximum,
                    sha256=hashlib.sha256(right.tobytes()+b.tobytes()).hexdigest()))
    offsets = np.asarray([0,3,14,21,101],dtype='<i4')
    original = rng.normal(0,.1,101).astype('<f4')
    refused = []
    for value in (np.nan,np.inf,-np.inf,np.finfo(np.float32).max):
        g = original.copy()
        g[-1] = value
        info = np.full(2,-17,dtype='<f4')
        before = g.tobytes(),info.tobytes()
        try:
            invoke(g,offsets,info,2,.1)
        except Exception:
            pass
        else:
            raise AssertionError('nonfinite norm accepted')
        assert (g.tobytes(),info.tobytes()) == before
        refused.append(str(value))
    for devices,maximum in ((0,.1),(65,.1),(3,.1),(2,0),(2,-1),(2,float('nan'))):
        g,info = original.copy(),np.full(2,-17,dtype='<f4')
        before = g.tobytes(),info.tobytes()
        try:
            invoke(g,offsets,info,devices,maximum)
        except Exception:
            pass
        else:
            raise AssertionError('invalid admission accepted')
        assert (g.tobytes(),info.tobytes()) == before
    if args.faults:
        assert binding.clip_pool_fault_available() == 1
        for owner in (0,1):
            g,info = original.copy(),np.full(2,-17,dtype='<f4')
            before = g.tobytes(),info.tobytes()
            os.environ['MOJOLEARN_CLIP_FAIL_OWNER'] = str(owner)
            try:
                try:
                    invoke(g,offsets,info,2,.1)
                except Exception as error:
                    assert 'output unchanged' in str(error)
                else:
                    raise AssertionError('post-scale fault did not refuse')
            finally:
                os.environ.pop('MOJOLEARN_CLIP_FAIL_OWNER',None)
            assert (g.tobytes(),info.tobytes()) == before
        a,b = original.copy(),original.copy()
        ia,ib = np.empty(2,dtype='<f4'),np.empty(2,dtype='<f4')
        invoke(a,offsets,ia,1,.1)
        invoke(b,offsets,ib,2,.1)
        assert a.tobytes()+ia.tobytes() == b.tobytes()+ib.tobytes()
    os.environ.pop('MOJOLEARN_OPTIMIZER_DEVICE_COUNT',None)
    args.report.write_text(json.dumps(dict(status='PASS',checks=checks,nonfinite_refusals=refused,
        native_faults=args.faults,scope='Whole-tensor gradient storage across two GPUs; original global norm and atomic publication. No throughput or full-model capacity claim.'),indent=2)+'\n')
    print('PASS',len(checks),'exact clips and atomic refusals; faults',args.faults)


if __name__ == '__main__':
    main()
