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


def _large_block(kind, dm, rng):
    """A block at a realistic width through the PUBLIC classes: Transformer
    n_heads = dm/64, n_kv_heads = n_heads/4, head_dim 64, intermediate
    11/4 dm rounded to 64; Mamba-1 at its own dims rule."""
    import mojolearn as ml
    if kind == 'transformer':
        nh, hd = dm // 64, 64
        nkv = max(1, nh // 4)
        it = ((11 * dm // 4) + 63) // 64 * 64
        shapes = [(dm,), (dm,), (nh * hd, dm), (nkv * hd, dm), (nkv * hd, dm), (dm, nh * hd),
                  (it, dm), (it, dm), (dm, it)]
        names = ml.TransformerBlock._W_NAMES
        w = {n: (rng.standard_normal(s) * (0.02 if len(s) > 1 else 0.01)).astype(np.float32)
             for n, s in zip(names, shapes)}
        w['input_layernorm.weight'] += np.float32(1)
        w['post_attention_layernorm.weight'] += np.float32(1)
        return ml.TransformerBlock(w, n_heads=nh, n_kv_heads=nkv), dict(nh=nh, nkv=nkv, hd=hd, it=it)
    di, r = 2 * dm, (dm + 15) // 16
    shapes = {'norm.weight': (dm,), 'in_proj.weight': (2 * di, dm), 'conv1d.weight': (di, 1, 4),
              'conv1d.bias': (di,), 'x_proj.weight': (r + 32, di), 'dt_proj.weight': (di, r),
              'dt_proj.bias': (di,), 'A_log': (di, 16), 'D': (di,), 'out_proj.weight': (dm, di)}
    w = {n: (rng.standard_normal(s) * (0.02 if len(s) > 1 else 0.01)).astype(np.float32)
         for n, s in shapes.items()}
    w['norm.weight'] += np.float32(1)
    return ml.Mamba1Block(w), dict(di=di, r=r)


def _decode_arm(kind, blk, x, prefill, tokens, arm):
    """One arm over the same inputs: a fresh state prefilled to `prefill`
    positions through the PER-CALL forward, then `tokens` decode steps
    through the per-call `step` (arm 'percall') or a resident session
    (arm 'resident': `decode_session`, the binding's
    `transformer_decode_session_*` / `mamba1_session_*` entry points; the
    per-call arm goes through the block's own retained per-model context
    where the binding has one). Returns per-token seconds, the decoded
    outputs and the final state pieces."""
    b = x.shape[0]
    state = blk.allocate_state(b, prefill + tokens) if kind == 'transformer' else blk.allocate_state(b)
    blk.forward(np.ascontiguousarray(x[:, :prefill]), state)
    outs, samples = [], []
    if arm == 'resident':
        session = blk.decode_session(state)
        try:
            for t in range(prefill, prefill + tokens):
                xt = np.ascontiguousarray(x[:, t:t + 1])
                t0 = time.perf_counter()
                y = session.step(xt)
                samples.append(time.perf_counter() - t0)
                outs.append(np.asarray(y))
        finally:
            session.close()
    else:
        for t in range(prefill, prefill + tokens):
            xt = np.ascontiguousarray(x[:, t:t + 1])
            t0 = time.perf_counter()
            y = blk.step(xt, state)
            samples.append(time.perf_counter() - t0)
            outs.append(np.asarray(y))
    pieces = (('k_cache', 'v_cache') if kind == 'transformer' else ('conv_window', 'h'))
    return samples, np.concatenate(outs, axis=1), {p: np.asarray(getattr(state, p)).copy() for p in pieces}


def resident_ab(args):
    """Interleaved per-call vs resident decode through the public classes,
    at one large shape: `--rounds` rounds after one warmup, arm order
    reversed every round (A B, B A, ...), every decoded output and both
    final state pieces bytewise equal across arms and rounds and equal to
    the fresh full forward over prefill + tokens positions."""
    import mojolearn as ml
    rng = np.random.default_rng(2940)
    blk, dims = _large_block(args.kind, args.dm, rng)
    b, p, t = args.batch, args.prefill, args.tokens
    x = rng.standard_normal((b, p + t, args.dm)).astype(np.float32)
    full = np.asarray(blk.forward(x))[:, p:]
    record = dict(kind=args.kind, vendor=ml.vendor(), mode=ml.numeric_mode(), d_model=args.dm,
                  dims=dims, batch=b, prefill=p, tokens=t, rounds=args.rounds,
                  package_dir=str(Path(ml.__file__).parent), rounds_detail=[])
    ref = None
    per_arm = {'percall': [], 'resident': []}
    for rnd in range(args.rounds + 1):
        order = ('percall', 'resident') if rnd % 2 == 0 else ('resident', 'percall')
        row = dict(round=rnd, warmup=rnd == 0, order=list(order))
        for arm in order:
            samples, y, pieces = _decode_arm(args.kind, blk, x, p, t, arm)
            assert y.tobytes() == full.tobytes(), f'{arm} round {rnd}: decoded output != full forward'
            digest = {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in pieces.items()}
            digest['y'] = hashlib.sha256(y.tobytes()).hexdigest()
            if ref is None:
                ref = digest
            assert digest == ref, f'{arm} round {rnd}: state/output digests moved: {digest} vs {ref}'
            med = statistics.median(samples)
            row[arm] = dict(median_ms=med * 1000, min_ms=min(samples) * 1000, max_ms=max(samples) * 1000)
            if rnd > 0:
                per_arm[arm].append(med)
            print(json.dumps(dict(round=rnd, arm=arm, warmup=rnd == 0, median_ms=med * 1000)), flush=True)
        record['rounds_detail'].append(row)
    summary = {}
    for arm, meds in per_arm.items():
        summary[arm] = dict(median_ms=statistics.median(meds) * 1000, spread=max(meds) / min(meds),
                            round_medians_ms=[m * 1000 for m in meds])
    summary['paired_ratio_percall_over_resident'] = statistics.median(
        [a / c for a, c in zip(per_arm['percall'], per_arm['resident'])])
    summary['digests'] = ref
    summary['bytewise'] = 'every arm and round equal to the fresh full forward and to each other'
    record['summary'] = summary
    print(json.dumps(summary, indent=1), flush=True)
    if args.out:
        args.out.write_text(json.dumps(record, indent=1) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bindings', type=Path)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--compare', nargs=2, type=Path)
    ap.add_argument('--resident-ab', action='store_true',
                    help='interleaved per-call vs resident decode through the public classes '
                         '(DEVIATION 2940/2941); --out is then a JSON record')
    ap.add_argument('--kind', choices=('transformer', 'mamba1'), default='transformer')
    ap.add_argument('--dm', type=int, default=1024)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--prefill', type=int, default=1024)
    ap.add_argument('--tokens', type=int, default=64)
    ap.add_argument('--rounds', type=int, default=5)
    args = ap.parse_args()
    if args.compare:
        with np.load(args.compare[0]) as a, np.load(args.compare[1]) as b:
            compare(a, b)
        return
    if args.resident_ab:
        resident_ab(args)
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
