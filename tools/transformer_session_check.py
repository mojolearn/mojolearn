#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Bounded native context/workspace reuse check against the existing entry point."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import time

import numpy as np
from bench_neural_decode import weights


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--binding', type=Path, required=True)
    ap.add_argument('--backend', choices=('metal', 'cuda', 'hip'), required=True)
    ap.add_argument('--group', choices=('reuse', 'refusals', 'lifetime', 'budget'), default='reuse')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--reverse-order', action='store_true')
    args = ap.parse_args()
    if not __debug__:
        ap.error('verification requires assertions (no -O)')
    spec = importlib.util.spec_from_file_location(args.binding.stem, args.binding.resolve())
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.transformer_vendor() == args.backend
    assert mod.transformer_numeric_mode() == 1
    owner = mod.transformer_session_create()
    assert list(mod.transformer_session_info(owner)) == [0, 0, False, 0]
    artifacts, timing, messages = {}, [], {}
    rng = np.random.default_rng(710)

    def make(b, length, dm, cap, window):
        x = rng.uniform(-.5, .5, (b, length, dm)).astype(np.float32)
        w = weights('transformer', dm)
        state = [np.full(b * (window or cap) * dm // 2, v, np.float32)
                 for v in (.125, -.25)]
        return x, w, state

    def call(session, x, w, state, pos, cap, window):
        b, length, dm = x.shape
        yy = np.full_like(x, np.nan)
        arrays = [x] + w + state + [yy]
        addr = [a.ctypes.data for a in arrays]
        params = [b, length, dm, 2, 1, dm // 2, 2 * dm, cap, pos, window]
        start = time.perf_counter()
        if session is None:
            n = mod.transformer_forward(addr, params)
        else:
            n = mod.transformer_session_forward(session, addr, params)
        elapsed = 1000 * (time.perf_counter() - start)
        assert n == pos + length
        assert np.isfinite(yy).all()
        return yy, elapsed

    def compare(case, x, w, state, pos, cap, window, reverse=False):
        left, right = [v.copy() for v in state], [v.copy() for v in state]
        reverse = not reverse if args.reverse_order else reverse
        if reverse:
            got, new_ms = call(owner, x, w, right, pos, cap, window)
            want, old_ms = call(None, x, w, left, pos, cap, window)
        else:
            want, old_ms = call(None, x, w, left, pos, cap, window)
            got, new_ms = call(owner, x, w, right, pos, cap, window)
        for label, a, b in zip(('y', 'k', 'v'), [want] + left, [got] + right):
            assert a.shape == b.shape and a.tobytes() == b.tobytes(), (case, label)
            # The budget case still compares complete caches in memory;
            # avoid writing tens of MiB of unused capacity into the artifact.
            if args.group != 'budget' or label == 'y':
                artifacts[case + '.' + label] = b.copy()
        timing.append(dict(case=case, baseline_ms=old_ms, retained_ms=new_ms))
        return right

    if args.group == 'reuse':
        for shape, (batch, length, dm, cap, window) in enumerate(
                ((1, 1, 32, 16, 0), (2, 1, 64, 8, 3), (1, 1, 128, 8, 0),
                 (1, 2, 32, 16, 0), (1, 1, 32, 16, 0))):
            x, w, state = make(batch, length, dm, cap, window)
            original = w[2].copy()
            for iteration in range(4):
                pos = iteration * length
                if iteration == 2:
                    w[2].flat[0] += np.float32(.125)
                    state[0].flat[0] += np.float32(.0625)
                x *= np.float32(-.75)
                state = compare(f'shape{shape}.pos{pos}', x, w, state, pos, cap,
                                window, reverse=iteration % 2 == 1)
                assert list(mod.transformer_session_info(owner))[:3] == [1, shape + 1, False]
            # Same-address state reset and restored weights must be observed.
            w[2][:] = original
            for a in state:
                a.fill(0)
            compare(f'shape{shape}.reset', x, w, state, 0, cap, window)
            assert list(mod.transformer_session_info(owner))[:3] == [1, shape + 1, False]
    elif args.group == 'refusals':
        x, w, state = make(1, 1, 32, 8, 0)
        compare('admit', x, w, state, 0, 8, 0)
        for kind in ('x', 'weight', 'cache', 'capacity'):
            xx, ww, ss = x.copy(), [a.copy() for a in w], [a.copy() for a in state]
            pos = 1
            if kind == 'x': xx.flat[0] = np.nan
            if kind == 'weight': ww[2].flat[0] = np.inf
            if kind == 'cache': ss[0].flat[0] = np.nan
            if kind == 'capacity': pos = 8
            errors = []
            for session in (None, owner):
                cache = [a.copy() for a in ss]
                try:
                    call(session, xx, ww, cache, pos, 8, 0)
                except Exception as e:
                    errors.append(str(e))
                else:
                    raise AssertionError('invalid input accepted: ' + kind)
                assert [a.tobytes() for a in cache] == [a.tobytes() for a in ss]
            assert errors[0] == errors[1], (kind, errors)
            messages[kind] = errors[0]
            compare('recovery.' + kind, x, w, state, 0, 8, 0)
            assert mod.transformer_session_info(owner)[0] == 1
    elif args.group == 'budget':
        # About 72 MiB of device buffers: above the 64 MiB retention cap.
        # Only one token is computed; this checks eviction, not throughput.
        x, w, state = make(256, 1, 32, 1024, 0)
        for iteration in range(2):
            compare('budget.' + str(iteration), x, w, state, 0, 1024, 0)
            assert list(mod.transformer_session_info(owner)) == [1, iteration + 1, False, 0]
    else:
        for iteration in range(3):
            x, w, state = make(1, 1, 32, 8, 0)
            compare('lifetime.' + str(iteration), x, w, state, 0, 8, 0)
            mod.transformer_session_close(owner)
            mod.transformer_session_close(owner)
            assert list(mod.transformer_session_info(owner)) == [1, 1, True, 0]
            try:
                call(owner, x, w, state, 0, 8, 0)
            except Exception as e:
                assert 'closed' in str(e)
            else:
                raise AssertionError('closed owner accepted')
            owner = mod.transformer_session_create()
    info = list(mod.transformer_session_info(owner))
    mod.transformer_session_close(owner)
    np.savez(args.out, **artifacts)
    warm = [v for v in timing if '.pos' in v['case'] and not v['case'].endswith('.pos0')]
    result = dict(backend=args.backend, group=args.group, reverse_order=args.reverse_order, arrays=len(artifacts),
                  ownership=info, samples=timing, refusals=messages,
                  warm_baseline_median_ms=statistics.median(v['baseline_ms'] for v in warm) if warm else None,
                  warm_retained_median_ms=statistics.median(v['retained_ms'] for v in warm) if warm else None,
                  binding_sha256=hashlib.sha256(args.binding.read_bytes()).hexdigest())
    args.out.with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('samples', 'refusals')}))


if __name__ == '__main__':
    main()
