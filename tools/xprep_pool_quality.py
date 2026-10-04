#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""MOJOLEARN_X_PREP_POOL_ARENA quality only (no timing, no opponents).

dump OUT.json    run the fixture on the installed FAST Apple x_prep binding,
                 check LabelBinarizer / MultiLabelBinarizer outputs against an
                 independent numpy definition, and write a sha256 digest of
                 EVERY public output array (dtype, shape, bytes).
compare A B      exact A/B: every digest equal (0 ulp, 0 words).

Tolerance, fixed before any result: exact. The candidate changes only where
the program's device buffer comes from (a pooled buffer instead of a fresh
one) and clears the scratch region; no arithmetic changes, so any differing
word is a defect (a stale word read from a reused buffer).

Every program here is at least 2^24 device words (the pool's minimum), and
each estimator runs again on a buffer the previous call left dirty: the
same-shape repeats must reproduce the first call's words exactly, and the
LabelBinarizer runs cross the output-region threshold (2^27 words) as the
board's taxi row does (1M x 259 there, 600k x 259 here).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import mojolearn as ml

FIXTURE = 'xprep-pool-v1-seed907'


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


def dump(path):
    assert os.environ.get('MOJOLEARN_NUMERIC_MODE') == 'fast'
    from mojolearn._expansion_prep import _prep_binding, _optional_prep_entry
    binding = _prep_binding('fast')
    assert str(binding.x_prep_vendor()) == 'metal'
    meta = dict(binding_sha256=hashlib.sha256(Path(binding.__file__).read_bytes()).hexdigest(),
                pool_enabled=_optional_prep_entry(binding, 'x_prep_pool_arena') is not None,
                fixture=FIXTURE)
    rng = np.random.default_rng(907)
    out = {}
    checked = 0

    # LabelBinarizer: 600k x 259 int32 = 155.4M words (an output region)
    n, K = 600_000, 259
    y1 = rng.integers(0, K, n, dtype=np.int64)
    y2 = rng.integers(0, K, n, dtype=np.int64)
    classes = np.arange(K)
    m = ml.LabelBinarizer()
    a = np.asarray(m.fit_transform(y1))
    checked += _check_lb(a, y1, classes, 0, 1, 'lb_fit1')
    out['lb_fit1'] = _digest(a)
    del a
    m2 = ml.LabelBinarizer()
    a = np.asarray(m2.fit_transform(y2))   # same shape: the dirty pooled buffer
    checked += _check_lb(a, y2, classes, 0, 1, 'lb_fit2')
    out['lb_fit2'] = _digest(a)
    del a
    q = y1[:250_000]                        # 64.75M words: arena output, still pooled
    a = np.asarray(m2.transform(q))
    checked += _check_lb(a, q, classes, 0, 1, 'lb_query')
    out['lb_query'] = _digest(a)
    del a
    m3 = ml.LabelBinarizer(neg_label=-3, pos_label=4)   # the general (non-scatter) kernel
    a = np.asarray(m3.fit_transform(y2))
    checked += _check_lb(a, y2, classes, -3, 4, 'lb_neg')
    out['lb_neg'] = _digest(a)
    del a
    a = np.asarray(ml.LabelBinarizer().fit_transform(y1))
    out['lb_fit1_again'] = _digest(a)
    assert out['lb_fit1_again'] == out['lb_fit1'], 'lb repeat after dirty reuse'
    del a
    out['lb_classes'] = _digest(np.asarray(m.classes_))

    # MultiLabelBinarizer: 150k rows x ~1000 classes (an output region)
    def sets(count):
        return [set(map(int, r)) for r in rng.integers(0, 1000, (count, 5))]
    r1, r2 = sets(150_000), sets(150_000)
    mm = ml.MultiLabelBinarizer()
    a = np.asarray(mm.fit_transform(r1))
    mcls = np.asarray(mm.classes_).astype(np.int64)
    np.testing.assert_array_equal(mcls, np.unique(np.fromiter((v for s in r1 for v in s), dtype=np.int64)))
    checked += _check_mlb(a, r1, mcls, 'mlb_fit1')
    out['mlb_fit1'] = _digest(a)
    del a
    mm2 = ml.MultiLabelBinarizer()
    a = np.asarray(mm2.fit_transform(r2))
    mcls2 = np.asarray(mm2.classes_).astype(np.int64)
    checked += _check_mlb(a, r2, mcls2, 'mlb_fit2')
    out['mlb_fit2'] = _digest(a)
    del a
    a = np.asarray(mm2.transform(r2[:60_000]))
    checked += _check_mlb(a, r2[:60_000], mcls2, 'mlb_query')
    out['mlb_query'] = _digest(a)
    del a
    out['mlb_classes'] = _digest(mcls) + _digest(mcls2)

    # RobustScaler (sort scratch + work regions), MaxAbsScaler, TargetEncoder: A/B exact
    X = rng.normal(size=(1_000_000, 20)).astype(np.float32)
    rs = ml.RobustScaler()
    out['robust_fit_transform'] = _digest(np.asarray(rs.fit_transform(X)))
    out['robust_center'] = _digest(np.asarray(rs.center_))
    out['robust_scale'] = _digest(np.asarray(rs.scale_))
    again = _digest(np.asarray(ml.RobustScaler().fit_transform(X)))
    assert again == out['robust_fit_transform'], 'robust repeat after dirty reuse'
    Xm = rng.normal(size=(600_000, 64)).astype(np.float32)
    ma = ml.MaxAbsScaler().fit(Xm)
    t = np.asarray(ma.transform(Xm))
    np.testing.assert_array_equal(np.asarray(ma.max_abs_), np.abs(Xm).max(axis=0))
    np.testing.assert_allclose(t, Xm / np.abs(Xm).max(axis=0), rtol=1e-6, atol=0)
    out['maxabs_transform'] = _digest(t)
    out['maxabs_transform_again'] = _digest(np.asarray(ma.transform(Xm)))
    assert out['maxabs_transform_again'] == out['maxabs_transform'], 'maxabs repeat'
    del t, Xm
    Xc = rng.integers(0, 50, (1_000_000, 8)).astype(np.float32)
    yc = rng.integers(0, 2, 1_000_000, dtype=np.int32)
    te = ml.TargetEncoder(target_type='binary', smooth=0.0, cv=4, shuffle=True, random_state=42)
    out['te_fit_transform'] = _digest(np.asarray(te.fit_transform(Xc, yc)))
    out['te_transform'] = _digest(np.asarray(te.transform(Xc[:200_000])))
    for j, e in enumerate(te.encodings_):
        out[f'te_enc{j}'] = _digest(np.asarray(e))
    te2 = ml.TargetEncoder(target_type='binary', smooth=0.0, cv=4, shuffle=True, random_state=42)
    again = _digest(np.asarray(te2.fit_transform(Xc, yc)))
    assert again == out['te_fit_transform'], 'te repeat after dirty reuse'

    meta['arrays'] = len(out)
    meta['oracle_cells'] = checked
    Path(path).write_text(json.dumps(dict(meta=meta, digests=out), sort_keys=True, indent=1))
    print('XPREP-POOL-CAPTURE ' + json.dumps(meta, sort_keys=True), flush=True)
    print(f'XPREP-POOL-QUALITY status=PASS arrays={len(out)} oracle_cells={checked} path={path}', flush=True)


def compare(first, second):
    a = json.loads(Path(first).read_text())
    b = json.loads(Path(second).read_text())
    da, db = a['digests'], b['digests']
    assert a['meta']['fixture'] == b['meta']['fixture'] == FIXTURE
    assert sorted(da) == sorted(db) and len(da) >= 20, (sorted(da), sorted(db))
    bad = [k for k in sorted(da) if da[k] != db[k]]
    assert not bad, 'differing arrays: ' + ','.join(bad)
    print(f'XPREP-POOL-AB status=PASS exact_arrays={len(da)}', flush=True)


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
