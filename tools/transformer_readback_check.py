#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""One bounded Transformer backward case, with complete bytewise artifacts."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import time

import numpy as np
from bench_neural_decode import weights

CASES = {'small': (1, 5, 32, 0), 'ring': (2, 9, 128, 5), 'wide': (1, 7, 256, 0)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--binding', type=Path)
    ap.add_argument('--backend', choices=('cpu', 'metal', 'cuda', 'hip'))
    ap.add_argument('--case', choices=CASES, default='small')
    ap.add_argument('--out', type=Path)
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--compare', nargs=2, type=Path)
    args = ap.parse_args()
    if not __debug__:
        ap.error('verification requires Python assertions enabled (no -O)')
    if args.compare:
        with np.load(args.compare[0]) as a, np.load(args.compare[1]) as b:
            assert set(a.files) == set(b.files)
            for key in a.files:
                assert a[key].shape == b[key].shape and a[key].dtype == b[key].dtype
                assert a[key].tobytes() == b[key].tobytes(), key
            print('BYTE_EQUAL', len(a.files), 'complete arrays')
        return
    if not args.binding or not args.backend or not args.out or args.reps < 1:
        ap.error('--binding, --backend, --out and positive --reps required')
    spec = importlib.util.spec_from_file_location(args.binding.stem, args.binding.resolve())
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.transformer_numeric_mode() == 1
    assert mod.transformer_vendor() == args.backend
    b, length, dm, window = CASES[args.case]
    rng = np.random.default_rng(392)
    x = rng.uniform(-.5, .5, (b, length, dm)).astype(np.float32)
    dy = rng.uniform(-.25, .25, x.shape).astype(np.float32)
    w = weights('transformer', dm)
    outputs = [np.empty_like(x)] + [np.empty_like(a) for a in w]
    arrays = [x] + w + [dy] + outputs
    pointers = [a.ctypes.data for a in arrays]
    params = [b, length, dm, 2, 1, dm // 2, 2 * dm, window]
    artifacts, samples = {}, []
    for iteration in range(args.reps + 1):
        for a in outputs:
            a.fill(np.nan)  # prove every advertised output is written
        start = time.perf_counter()
        mod.transformer_backward(pointers, params)
        elapsed = (time.perf_counter() - start) * 1000
        for i, a in enumerate(outputs):
            assert np.isfinite(a).all(), ('unwritten/nonfinite gradient', i)
            if iteration:
                assert a.tobytes() == artifacts[f'gradient_{i}'].tobytes(), ('repeat', i)
            else:
                artifacts[f'gradient_{i}'] = a.copy()
        if iteration:
            samples.append(elapsed)
    # Existing callers may edit their weights: a second case uses the same
    # addresses after a real mutation, and is included in the cross-build check.
    w[2].flat[0] += np.float32(.125)
    for a in outputs:
        a.fill(np.nan)
    mod.transformer_backward(pointers, params)
    for i, a in enumerate(outputs):
        assert np.isfinite(a).all()
        artifacts[f'mutated_gradient_{i}'] = a.copy()
    assert any(artifacts[f'gradient_{i}'].tobytes() !=
               artifacts[f'mutated_gradient_{i}'].tobytes()
               for i in range(len(outputs))), 'weight mutation was not observed'
    np.savez(args.out, **artifacts)
    result = dict(backend=args.backend, case=args.case, samples_ms=samples,
                  median_ms=statistics.median(samples), arrays=len(artifacts),
                  binding_sha256=hashlib.sha256(args.binding.read_bytes()).hexdigest())
    args.out.with_suffix('.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
