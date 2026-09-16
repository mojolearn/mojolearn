#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Bounded native GPU decode regression probe, independent of CPU routing.

Run each build in a fresh process with --bindings pointing to its immutable
DSO directory. Compare --out NPZ files bytewise with --compare. This measures
small-call overhead, not model throughput or cross-vendor qualification.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import time

import numpy as np


def weights(kind, dm):
    rng = np.random.default_rng(21)
    if kind == 'transformer':
        shapes = [(dm,), (dm,), (dm, dm), (dm // 2, dm), (dm // 2, dm),
                  (dm, dm), (2 * dm, dm), (2 * dm, dm), (dm, 2 * dm)]
    else:
        di, r = 2 * dm, (dm + 15) // 16
        shapes = [(dm,), (2 * di, dm), (di, 1, 4), (di,), (r + 32, di),
                  (di, r), (di,), (di, 16), (di,), (dm, di)]
    return [(rng.standard_normal(s) * .1).astype(np.float32) for s in shapes]


def load(root, kind):
    name = '_mojolearn_' + ('mamba' if kind == 'mamba1' else kind)
    path = root / (name + '.so')
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    prefix = 'mamba' if kind == 'mamba1' else kind
    assert getattr(mod, prefix + '_numeric_mode')() == 1, 'requires IDENTICAL'
    vendor = getattr(mod, prefix + '_vendor')()
    assert vendor in ('metal', 'cuda', 'hip'), f'not a GPU binary: {vendor}'
    print(json.dumps({'binding': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                      'vendor': vendor}), flush=True)
    return mod


def run(mod, kind, b, length, dm, window, step):
    w = weights(kind, dm)
    x = np.random.default_rng(3).standard_normal((b, length, dm)).astype(np.float32)
    if kind == 'transformer':
        state = [np.zeros(b * (window or length) * (dm // 2), np.float32) for _ in range(2)]
    else:
        state = [np.zeros(b * 2 * dm * n, np.float32) for n in (4, 16)]
    result = []
    start = time.perf_counter()
    for pos in range(length) if step else [0]:
        xx = np.ascontiguousarray(x[:, pos:pos + 1]) if step else x
        yy = np.empty_like(xx)
        arrays = [xx] + w + state + [yy]
        addrs = [a.ctypes.data for a in arrays]
        if kind == 'transformer':
            params = [b, dm, 2, 1, dm // 2, 2 * dm, length, pos, window]
            if not step:
                params.insert(1, length)
        else:
            params = [b, dm] if step else [b, length, dm]
        fn = getattr(mod, kind + ('_decode_step' if step else '_forward'))
        fn(addrs, params)
        result.append(yy)
    elapsed = time.perf_counter() - start
    return elapsed, [np.concatenate(result, axis=1)] + state


def compare(a, b):
    assert set(a) == set(b), 'different cases'
    for key in a:
        assert a[key].shape == b[key].shape and a[key].dtype == b[key].dtype, key
        assert a[key].tobytes() == b[key].tobytes(), f'byte mismatch: {key}'
        print('BYTE_EQUAL', key, hashlib.sha256(a[key].tobytes()).hexdigest())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bindings', type=Path)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--compare', nargs=2, type=Path)
    args = ap.parse_args()
    if args.compare:
        with np.load(args.compare[0]) as a, np.load(args.compare[1]) as b:
            compare(a, b)
        return
    if args.bindings is None or args.out is None or args.reps < 1:
        ap.error('--bindings, --out and positive --reps required')
    artifacts, timing = {}, []
    for kind in ('transformer', 'mamba1'):
        mod = load(args.bindings.resolve(), kind)
        cases = [(1, 16, 32, 0), (2, 5, 64, 0)]
        if kind == 'transformer':
            cases.append((1, 5, 32, 3))
        for b, length, dm, window in cases:
            label = f'{kind}_b{b}_l{length}_d{dm}_w{window}'
            full_t, full = run(mod, kind, b, length, dm, window, False)
            for i, a in enumerate(full):
                assert np.isfinite(a).all(), label
                artifacts[f'{label}_full_{i}'] = a
            samples = []
            for _ in range(args.reps + 1):
                elapsed, step = run(mod, kind, b, length, dm, window, True)
                # Every output and both carried-state arrays, including warm-up.
                compare({str(i): a for i, a in enumerate(full)},
                        {str(i): a for i, a in enumerate(step)})
                samples.append(elapsed * 1000 / length)
            for i, a in enumerate(step):
                artifacts[f'{label}_step_{i}'] = a
            row = dict(case=label, ms_per_step=samples[1:], median_ms=statistics.median(samples[1:]),
                       prefill_ms=full_t * 1000)
            timing.append(row)
            print(json.dumps(row), flush=True)
    np.savez(args.out, **artifacts)
    args.out.with_suffix('.json').write_text(json.dumps(timing, indent=2) + '\n')


if __name__ == '__main__':
    main()
