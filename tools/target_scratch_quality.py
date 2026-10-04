#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Public TargetEncoder quality-only dump and exact main/candidate comparison."""
import argparse
import os
import hashlib
import json
from pathlib import Path
import numpy as np
import mojolearn as ml


def dump(path):
    assert os.environ.get('MOJOLEARN_NUMERIC_MODE') == 'fast'
    from mojolearn._expansion_prep import _prep_binding, _target_scratch
    binding = _prep_binding("fast")
    assert str(binding.x_prep_vendor()) == "metal"
    assert int(binding.x_prep_numeric_mode()) == 0
    metadata = dict(binding_sha256=hashlib.sha256(Path(binding.__file__).read_bytes()).hexdigest(),
                    scratch_enabled=_target_scratch("fast"), fixture="target-scratch-v1-seed553")
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
    metadata['arrays'] = len(output)
    print('TARGET-SCRATCH-CAPTURE ' + json.dumps(metadata, sort_keys=True))
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
        assert len(a.files) == 108 and sorted(a.files) == sorted(b.files)
        for key in a.files:
            assert a[key].dtype == b[key].dtype and a[key].shape == b[key].shape, key
            assert a[key].tobytes() == b[key].tobytes(), key
        print(f'TARGET-SCRATCH-AB status=PASS exact_arrays={len(a.files)}')


if __name__ == '__main__':
    main()
