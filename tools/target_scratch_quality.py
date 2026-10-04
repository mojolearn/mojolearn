#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Public TargetEncoder quality-only dump and exact main/candidate comparison."""
import argparse
import os
import numpy as np
import mojolearn as ml


def dump(path):
    assert os.environ.get('MOJOLEARN_NUMERIC_MODE') == 'fast'
    rng = np.random.default_rng(553)
    X = rng.integers(0, 23, (5003, 5)).astype(np.float32)
    Xq = X[:137].copy()
    Xq[::7, 2] = 999  # unknown categories must retain global-mean behavior
    output = {}
    for kind in ('continuous', 'binary', 'multiclass'):
        y = (rng.normal(size=X.shape[0]).astype(np.float32) if kind == 'continuous'
             else rng.integers(0, 2 if kind == 'binary' else 3, X.shape[0], dtype=np.int32))
        for smooth in (3.0, 'auto'):
            for given in (False, True):
                tag = f'{kind}_{smooth}_{given}'
                # Missing one training category exercises unfilled bucket tails.
                cats = [np.arange(22, dtype=np.float32)] * 5 if given else 'auto'
                model = ml.TargetEncoder(categories=cats, target_type=kind, smooth=smooth,
                                         cv=5, shuffle=False)
                fit_transform = np.asarray(model.fit_transform(X, y))
                pred = np.asarray(model.transform(Xq))
                model.fit(X, y)
                fit_pred = np.asarray(model.transform(Xq))
                for suffix, a in [('crossfit', fit_transform), ('transform', pred), ('fit_transform', fit_pred)]:
                    assert np.isfinite(a).all(), tag + suffix
                    output[tag + suffix] = a
                # Independent closed-form target mean and smoothed category means.
                if smooth == 3.0:
                    targets = y.astype(np.float64)[:, None] if kind != 'multiclass' else (y[:, None] == np.arange(3)[None, :]).astype(float)
                    mean = targets.mean(axis=0)
                    expected = np.empty_like(fit_pred, dtype=np.float64)
                    T = targets.shape[1]
                    for j in range(X.shape[1]):
                        allowed = np.arange(22 if given else 23)
                        for i, v in enumerate(Xq[:, j]):
                            mask = X[:, j] == v
                            value = (targets[mask].sum(axis=0) + 3 * mean) / (mask.sum() + 3) if v in allowed else mean
                            expected[i, j*T:(j+1)*T] = value
                    np.testing.assert_allclose(fit_pred, expected, atol=3e-5, rtol=3e-5)
                for j, a in enumerate(model.encodings_):
                    output[tag + f'enc{j}'] = np.asarray(a)
                output[tag + 'mean'] = np.asarray(model.target_mean_)
    np.savez(path, **output)
    print(f'TARGET-SCRATCH-QUALITY status=PASS arrays={len(output)} path={path}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['dump', 'compare'])
    p.add_argument('first')
    p.add_argument('second', nargs='?')
    args = p.parse_args()
    if args.action == 'dump':
        dump(args.first)
    else:
        a, b = np.load(args.first), np.load(args.second)
        assert sorted(a.files) == sorted(b.files)
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key], err_msg=key)
        print(f'TARGET-SCRATCH-AB status=PASS exact_arrays={len(a.files)}')


if __name__ == '__main__':
    main()
