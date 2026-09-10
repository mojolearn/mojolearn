#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""WP2a same-fit native export gate; small correctness data, no timing claim."""
import argparse
import hashlib
import os
import random
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['fast', 'deterministic', 'identical'], required=True)
    parser.add_argument('--vendor', choices=['metal', 'cuda', 'hip'], required=True)
    args = parser.parse_args()
    os.environ['MOJOLEARN_NUMERIC_MODE'] = args.mode
    os.environ['MOJOLEARN_FOREST_EXPORT'] = 'verify'
    from mojolearn import (Array, RandomForestClassifier, RandomForestRegressor,
                           ExtraTreesClassifier, ExtraTreesRegressor)
    from mojolearn import _forest_protocol
    rng = random.Random(2482)
    X = Array.from_list([[rng.uniform(-4, 4) for _ in range(5)] for _ in range(257)], '<f4')
    cases = [(RandomForestClassifier, {}), (RandomForestRegressor, {}),
             (ExtraTreesClassifier, {}), (ExtraTreesRegressor, {}),
             (RandomForestClassifier, {"class_weight": {0: 0.5, 1: 1.25, 2: 2.0}})]
    original = _forest_protocol._export_fit_result
    captured = []
    def checked(native, descriptor, **kwargs):
        captured.append(native.forest_export_legacy(descriptor[0]))
        return original(native, descriptor, **kwargs)
    _forest_protocol._export_fit_result = checked
    try:
        for cls, extra in cases:
            classifier = cls._estimator_type == 'classifier'
            y = Array.from_list([i % 3 if classifier else rng.uniform(-2, 2) for i in range(257)],
                                '<i4' if classifier else '<f4')
            model = cls(n_estimators=7, max_depth=4, random_state=19, numeric_mode=args.mode, **extra)
            native = model._bind()
            prefix = 'rf' if cls.__name__.startswith('RandomForest') else 'trees'
            assert str(getattr(native, prefix + '_vendor')()) == args.vendor
            assert getattr(native, prefix + '_numeric_mode')() == {'fast':0, 'identical':1, 'deterministic':2}[args.mode]
            model.fit(X, y)
            expected = _forest_protocol._forest_fit_arrays(captured.pop())
            for name, value in zip(_forest_protocol._FOREST_ARRAYS, expected[:5]):
                assert getattr(model, name).tobytes() == value.tobytes(), name
            with tempfile.TemporaryDirectory() as directory:
                direct, legacy = Path(directory)/'direct.npz', Path(directory)/'legacy.npz'
                model.save(direct)
                for name, value in zip(_forest_protocol._FOREST_ARRAYS, expected[:5]):
                    setattr(model, name, value)
                model.save(legacy)
                assert direct.read_bytes() == legacy.read_bytes()
                digest = hashlib.sha256(direct.read_bytes()).hexdigest()
            print('PASS', cls.__name__, 'weighted' if extra else 'unweighted', args.mode, args.vendor, 'same-fit-five-arrays-and-save', digest, flush=True)
    finally:
        _forest_protocol._export_fit_result = original


if __name__ == '__main__':
    main()
