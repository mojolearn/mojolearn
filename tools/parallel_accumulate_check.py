#!/usr/bin/env python3
"""Cloud-only exact tree, column ownership and atomic accumulation checks."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cloud', action='store_true', required=True)
    p.add_argument('--report', type=Path, required=True)
    args = p.parse_args()
    if not os.environ.get('RUNPOD_POD_ID'):
        raise SystemExit('RunPod required; no local execution')
    import numpy as np
    from mojolearn._training_impl import _load
    binding = _load('identical')
    assert binding.accumulate_parallel_available() == 1
    rng = np.random.default_rng(78192)
    checks = []

    def invoke(parts, out, devices, n, steps, tokens=-1):
        os.environ['MOJOLEARN_OPTIMIZER_DEVICE_COUNT'] = str(devices)
        return binding.accumulate([int(out.ctypes.data), int(parts.ctypes.data)], [n, steps, tokens])

    for n in (1, 3, 127, 257, 1025, 65537):
        for steps in (1, 2, 4, 8, 16):
            parts = rng.normal(0, .1, (steps,n)).astype('<f4')
            parts.view('<u4')[0,0] = 0x80000000
            if n > 1:
                parts.view('<u4')[0,1] = 1
            if steps == 4:
                parts[:,0] = [2**24, 1, -2**24, 1]
            before = parts.tobytes()
            left = np.full(n, -17, dtype='<f4')
            right = left.copy()
            assert invoke(parts,left,1,n,steps) == n
            assert invoke(parts,right,2,n,steps) == n
            assert left.tobytes() == right.tobytes(), (n,steps)
            assert parts.tobytes() == before
            if steps == 4:
                assert right[0] == 0, 'tree was replaced by a sequential fold'
            checks.append(dict(n=n,steps=steps,sha256=hashlib.sha256(right.tobytes()).hexdigest()))
    # Preserve the original token-alignment admission independently of n.
    aligned = refused_alignment = 0
    parts = rng.normal(0,.1,(4,257)).astype('<f4')
    for tokens in (1, 33, 128, 512, 1024, 4096):
        legal = binding.accumulation_is_aligned([tokens,4]) == 1
        left = np.full(257,-17,dtype='<f4')
        right = left.copy()
        if legal:
            invoke(parts,left,1,257,4,tokens)
            invoke(parts,right,2,257,4,tokens)
            assert left.tobytes() == right.tobytes()
            aligned += 1
        else:
            before = right.tobytes()
            try:
                invoke(parts,right,2,257,4,tokens)
            except Exception:
                pass
            else:
                raise AssertionError('misaligned split accepted')
            assert right.tobytes() == before
            refused_alignment += 1
    assert aligned and refused_alignment
    refusals = []
    for devices,n,steps,tokens in ((0,257,4,-1),(65,257,4,-1),(3,257,4,-1),
                                   (2,0,4,-1),(2,257,0,-1),(2,257,3,-1),(2,257,4,0)):
        out = np.full(257,-17,dtype='<f4')
        before = out.tobytes()
        try:
            invoke(parts,out,devices,n,steps,tokens)
        except Exception:
            pass
        else:
            raise AssertionError('invalid admission accepted')
        assert out.tobytes() == before
        refusals.append([devices,n,steps,tokens])
    for bits in (0x7fc12345,0x7f800000,0xff800000):
        bad = parts.copy()
        bad.view('<u4')[-1,-1] = bits
        out = np.full(257,-17,dtype='<f4')
        before = out.tobytes()
        try:
            invoke(bad,out,2,257,4)
        except Exception:
            pass
        else:
            raise AssertionError('nonfinite last shard accepted')
        assert out.tobytes() == before
    os.environ.pop('MOJOLEARN_OPTIMIZER_DEVICE_COUNT',None)
    args.report.write_text(json.dumps(dict(status='PASS',checks=checks,refusals=refusals,
        aligned=aligned,refused_alignment=refused_alignment,nonfinite_refusals=3,
        scope='Two GPUs; disjoint gradient columns, original per-cell tree and atomic publication. No full-model capacity or throughput claim.'),indent=2)+'\n')
    print('PASS',len(checks),'exact cases and atomic refusals')


if __name__ == '__main__':
    main()
