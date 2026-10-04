#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""MOJOLEARN_X_PREP_PINNED_OUT quality only (no opponents).

dump OUT.json    run the fixture on the installed FAST Apple x_prep binding,
                 check LabelBinarizer / MultiLabelBinarizer / MaxAbsScaler
                 outputs against an independent numpy definition, check the
                 pinned outputs' lifetime (a held output is never overwritten
                 by later calls, views outlive their estimator and program,
                 every buffer is released), and write a sha256 digest of
                 EVERY public output array (dtype, shape, bytes).
compare A B      exact A/B: every digest equal (0 ulp, 0 words).

Tolerance, fixed before any result: exact. The candidate changes only where
the output region's words land (a DMA into a pinned host buffer that the
result views, instead of a staged copy into a fresh mapping) and lowers the
output-region threshold from 2^27 to 2^22 words; no arithmetic changes, so
any differing word is a defect (a stale/reused/freed buffer).

Also prints, informational only (one run per arm, not a gate):
XPREP-PINNED-READ <case> call_ms=... read_ms=... : the call, then one full
CPU read of its output (sha256 over every byte). If the pinned memory reads
slower than ordinary memory, read_ms shows it, so a faster call that only
moves cost to the caller's first read is visible.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import time

import numpy as np
import mojolearn as ml

FIXTURE = 'xprep-pinned-v1-seed911'


def _digest(a):
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.data)
    return h.hexdigest()


def _check_lb(actual, values, classes, neg, pos, tag):
    """LabelBinarizer's definition, by row chunks (no full expected matrix)."""
    assert actual.dtype == np.int32 and actual.shape == (values.size, classes.size), tag
    for lo in range(0, values.size, 50_000):
        v = values[lo:lo + 50_000]
        expected = np.where(v[:, None] == classes[None, :], pos, neg).astype(np.int32)
        np.testing.assert_array_equal(actual[lo:lo + 50_000], expected, err_msg=tag)
    return actual.size


def _check_mlb(actual, rows, classes, tag):
    assert actual.dtype == np.int32 and actual.shape == (len(rows), classes.size), tag
    for lo in range(0, len(rows), 30_000):
        part = rows[lo:lo + 30_000]
        expected = np.zeros((len(part), classes.size), dtype=np.int32)
        r = np.repeat(np.arange(len(part)), [len(s) for s in part])
        lab = np.fromiter((v for s in part for v in s), dtype=np.int64, count=int(r.size))
        expected[r, np.searchsorted(classes, lab)] = 1
        np.testing.assert_array_equal(actual[lo:lo + 30_000], expected, err_msg=tag)
    return actual.size


def _timed(tag, fn):
    """(result as numpy, call_ms): the call, then one full read, printed."""
    t0 = time.perf_counter()
    r = fn()
    t1 = time.perf_counter()
    a = np.asarray(r)
    h = hashlib.sha256(np.ascontiguousarray(a).data).hexdigest()
    t2 = time.perf_counter()
    print(f'XPREP-PINNED-READ {tag} call_ms={(t1 - t0) * 1e3:.1f} read_ms={(t2 - t1) * 1e3:.1f} '
          f'mb={a.nbytes / 2 ** 20:.0f} sha={h[:12]}', flush=True)
    return a


def dump(path):
    assert os.environ.get('MOJOLEARN_NUMERIC_MODE') == 'fast'
    from mojolearn._expansion_prep import _prep_binding, _optional_prep_entry
    binding = _prep_binding('fast')
    assert str(binding.x_prep_vendor()) == 'metal'
    live = _optional_prep_entry(binding, 'x_prep_pinned_out_live')
    meta = dict(binding_sha256=hashlib.sha256(Path(binding.__file__).read_bytes()).hexdigest(),
                pinned_enabled=_optional_prep_entry(binding, 'x_prep_run_ranges_pinned') is not None,
                fixture=FIXTURE)
    rng = np.random.default_rng(911)
    out = {}
    checked = 0

    # LabelBinarizer at the board's taxi shape: 1M x 259 int32 (an output region either way)
    n, K = 1_000_000, 259
    y1 = rng.integers(0, K, n, dtype=np.int64)
    y2 = rng.integers(0, K, n, dtype=np.int64)
    classes = np.arange(K)
    for rep in range(2):   # warm-up shape first, then the board-like second call
        a = _timed(f'lb_taxi_shape_r{rep}', lambda: ml.LabelBinarizer().fit_transform(y1))
        if rep == 0:
            checked += _check_lb(a, y1, classes, 0, 1, 'lb_fit1')
            out['lb_fit1'] = _digest(a)
        else:
            assert _digest(a) == out['lb_fit1'], 'lb repeat'
        del a
    gc.collect()

    # lifetime: a HELD output must survive later same-shape calls (pool reuse)
    m = ml.LabelBinarizer()
    held = m.fit_transform(y1)
    held_np = np.asarray(held)
    other = np.asarray(ml.LabelBinarizer().fit_transform(y2))
    third = np.asarray(ml.LabelBinarizer().fit_transform(y2))
    checked += _check_lb(other, y2, classes, 0, 1, 'lb_fit2')
    out['lb_fit2'] = _digest(other)
    assert _digest(third) == out['lb_fit2'], 'lb second y2 call'
    del m, held                               # estimator and Array gone; numpy view remains
    gc.collect()
    _ = np.asarray(ml.LabelBinarizer().fit_transform(y2))   # would reuse a wrongly freed buffer
    gc.collect()
    checked += _check_lb(held_np, y1, classes, 0, 1, 'lb_held_after_reuse')
    assert _digest(held_np) == out['lb_fit1'], 'held output changed'
    held_rt = pickle.loads(pickle.dumps(ml.LabelBinarizer().fit_transform(y1)))
    assert _digest(np.asarray(held_rt)) == out['lb_fit1'], 'pickle round trip'
    del held_np, other, third, held_rt, _
    gc.collect()

    # mid-size outputs: 2^22 .. 2^27 words now take the region when pinned
    q = y1[:100_000]                         # 25.9M words
    m2 = ml.LabelBinarizer().fit(y1)
    a = np.asarray(m2.transform(q))
    checked += _check_lb(a, q, classes, 0, 1, 'lb_query_mid')
    out['lb_query_mid'] = _digest(a)
    del a
    m3 = ml.LabelBinarizer(neg_label=-3, pos_label=4)   # the general (non-scatter) kernel
    a = np.asarray(m3.fit_transform(y2[:300_000]))
    checked += _check_lb(a, y2[:300_000], classes, -3, 4, 'lb_neg')
    out['lb_neg'] = _digest(a)
    del a
    out['lb_classes'] = _digest(np.asarray(m2.classes_))

    # MultiLabelBinarizer: board-like 200k rows; classes ~ 1000 (an output region)
    def sets(count):
        return [set(map(int, r)) for r in rng.integers(0, 1000, (count, 6))]
    r1, r2 = sets(200_000), sets(40_000)
    mm = ml.MultiLabelBinarizer()
    a = _timed('mlb_200k', lambda: mm.fit_transform(r1))
    mcls = np.asarray(mm.classes_).astype(np.int64)
    np.testing.assert_array_equal(mcls, np.unique(np.fromiter((v for s in r1 for v in s), dtype=np.int64)))
    checked += _check_mlb(a, r1, mcls, 'mlb_fit1')
    out['mlb_fit1'] = _digest(a)
    del a
    a = np.asarray(mm.transform(r2))        # 40k x ~1000: 40M words, mid-size region
    checked += _check_mlb(a, r2, mcls, 'mlb_query_mid')
    out['mlb_query_mid'] = _digest(a)
    del a
    out['mlb_classes'] = _digest(mcls)

    # MaxAbsScaler: istella-like width, 600k x 220 (132M words: region either way) and a mid one
    Xm = rng.normal(size=(600_000, 220)).astype(np.float32)
    ma = ml.MaxAbsScaler().fit(Xm)
    t = _timed('maxabs_600k_220', lambda: ma.transform(Xm))
    np.testing.assert_array_equal(np.asarray(ma.max_abs_), np.abs(Xm).max(axis=0))
    np.testing.assert_allclose(t, Xm / np.abs(Xm).max(axis=0), rtol=1e-6, atol=0)
    checked += t.size
    out['maxabs_transform'] = _digest(t)
    out['maxabs_fit_transform'] = _digest(np.asarray(ml.MaxAbsScaler().fit_transform(Xm)))
    del t
    t = np.asarray(ma.transform(Xm[:40_000]))  # 8.8M words: mid-size region
    np.testing.assert_allclose(t, Xm[:40_000] / np.abs(Xm).max(axis=0), rtol=1e-6, atol=0)
    out['maxabs_mid'] = _digest(t)
    del t, Xm
    gc.collect()

    # Other output-region users (exact A/B only): scalers, TargetEncoder, OneHotEncoder, PolynomialFeatures
    X = rng.normal(size=(500_000, 20)).astype(np.float32)   # 10M words: mid-size
    for name, est in (('standard', ml.StandardScaler()), ('minmax', ml.MinMaxScaler()),
                      ('robust', ml.RobustScaler())):
        out[name + '_fit_transform'] = _digest(np.asarray(est.fit_transform(X)))
    pf = ml.PolynomialFeatures(degree=2)
    out['poly_fit_transform'] = _digest(np.asarray(pf.fit_transform(X[:100_000])))
    pff = ml.PolynomialFeatures(degree=2, order='F')
    out['poly_F_fit_transform'] = _digest(np.asarray(pff.fit_transform(X[:100_000])))
    Xc = rng.integers(0, 50, (1_000_000, 8)).astype(np.float32)
    yc = rng.integers(0, 2, 1_000_000, dtype=np.int32)
    te = ml.TargetEncoder(target_type='binary', smooth=0.0, cv=4, shuffle=True, random_state=42)
    out['te_fit_transform'] = _digest(_timed('te_1m_8', lambda: te.fit_transform(Xc, yc)))
    out['te_transform'] = _digest(np.asarray(te.transform(Xc)))
    for j, e in enumerate(te.encodings_):
        out[f'te_enc{j}'] = _digest(np.asarray(e))
    oh = ml.OneHotEncoder(sparse_output=False)
    out['onehot_fit_transform'] = _digest(np.asarray(oh.fit_transform(Xc[:200_000])))
    del X, Xc, yc
    gc.collect()

    if live is not None:
        gc.collect()
        n_live = int(live())
        assert n_live == 0, f'{n_live} pinned outputs still live after every view died'
        meta['pinned_live_end'] = n_live
    else:
        meta['pinned_live_end'] = None
    meta['arrays'] = len(out)
    meta['oracle_cells'] = checked
    Path(path).write_text(json.dumps(dict(meta=meta, digests=out), sort_keys=True, indent=1))
    print('XPREP-PINNED-CAPTURE ' + json.dumps(meta, sort_keys=True), flush=True)
    print(f'XPREP-PINNED-QUALITY status=PASS arrays={len(out)} oracle_cells={checked} path={path}', flush=True)


def compare(first, second):
    a = json.loads(Path(first).read_text())
    b = json.loads(Path(second).read_text())
    da, db = a['digests'], b['digests']
    assert a['meta']['fixture'] == b['meta']['fixture'] == FIXTURE
    assert sorted(da) == sorted(db) and len(da) >= 25, (sorted(da), sorted(db))
    bad = [k for k in sorted(da) if da[k] != db[k]]
    assert not bad, 'differing arrays: ' + ','.join(bad)
    print(f'XPREP-PINNED-AB status=PASS exact_arrays={len(da)}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('action', choices=['dump', 'compare'])
    p.add_argument('first')
    p.add_argument('second', nargs='?')
    args = p.parse_args()
    if args.action == 'dump':
        dump(args.first)
    else:
        compare(args.first, args.second)


if __name__ == '__main__':
    main()
