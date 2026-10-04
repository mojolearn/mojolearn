#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Public-path output checks, quality only; no timing or opponent fit.

Run with the built FAST Apple binding and PYTHONPATH=python. Independently
construct every expected indicator/code from its mathematical definition,
including unseen labels, sparse class ranges, and general-sort fallbacks.
Optional --dump captures fitted classes and ALL predictions for exact A/B.
"""
import argparse
import os

import numpy as np
import mojolearn as ml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dump')
    args = parser.parse_args()
    assert os.environ.get('MOJOLEARN_NUMERIC_MODE') == 'fast'
    rng = np.random.default_rng(461)
    cases = {
        'board_width': rng.integers(0, 257, 50003, dtype=np.int64),
        'small_sparse': np.tile(np.array([0, 3, 4095], dtype=np.int32), 701),
        'binary': np.tile(np.array([0, 4095], dtype=np.uint32), 601),
        'single': np.full(1031, 7, dtype=np.int64),
        'negative_fallback': np.tile(np.array([-5, 0, 3], dtype=np.int64), 701),
        'range_fallback': np.tile(np.array([0, 4096, 9999], dtype=np.int64), 701),
        'float_fallback': np.tile(np.array([-0.5, 0.25, 2.5], dtype=np.float64), 701),
        'signed_zero': np.tile(np.array([-0.0, 0.0, 2.0], dtype=np.float32), 701),
    }
    captures = {}
    checked = 0
    for name, y in cases.items():
        classes = np.unique(y)
        query = np.concatenate([y[:37], np.array([8191], dtype=y.dtype)])
        for neg, pos in [(0, 1), (0, 7), (-3, 4)]:
            tag = f'{name}_{neg}_{pos}'
            model = ml.LabelBinarizer(neg_label=neg, pos_label=pos)
            fitted = np.asarray(model.fit_transform(y))
            np.testing.assert_array_equal(np.asarray(model.classes_), classes)
            for suffix, values, actual in [('train', y, fitted), ('query', query, np.asarray(model.transform(query)))]:
                if classes.size <= 2:
                    hit = values[:, None] == classes[-1] if classes.size == 2 else np.zeros((values.size, 1), dtype=bool)
                else:
                    hit = values[:, None] == classes[None, :]
                expected = np.where(hit, pos, neg)
                np.testing.assert_array_equal(actual, expected, err_msg=tag + suffix)
                captures[tag + suffix] = actual
                checked += actual.size
            np.testing.assert_array_equal(np.asarray(model.inverse_transform(fitted)), y)
            captures[tag + '_classes'] = np.asarray(model.classes_)
        enc = ml.LabelEncoder()
        codes = np.asarray(enc.fit_transform(y))
        np.testing.assert_array_equal(codes, np.searchsorted(classes, y))
        captures[name + '_codes'] = codes
    rows = [set(map(int, row)) for row in rng.integers(0, 300, (5003, 5))]
    rows += [set(), {4095, 0}, {0, 0}]
    for name, train in [('present', rows), ('fallback', rows + [{-7, 8191}])]:
        model = ml.MultiLabelBinarizer()
        actual = np.asarray(model.fit_transform(train))
        classes = sorted(set().union(*train))
        np.testing.assert_array_equal(np.asarray(model.classes_), classes)
        expected = np.array([[int(c in row) for c in classes] for row in train], dtype=np.int32)
        np.testing.assert_array_equal(actual, expected)
        query = [set(), {0, 7, 10001}, {4095}]
        pred = np.asarray(model.transform(query))
        np.testing.assert_array_equal(pred, [[int(c in row) for c in classes] for row in query])
        captures['mlb_' + name + '_train'] = actual
        captures['mlb_' + name + '_query'] = pred
        checked += actual.size + pred.size
    if args.dump:
        np.savez(args.dump, **captures)
    print(f'LABEL-FAST-QUALITY status=PASS checked_cells={checked} captures={len(captures)}')


if __name__ == '__main__':
    main()
