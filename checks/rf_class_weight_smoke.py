#!/usr/bin/env python3
"""Small real CUDA RF class-weight gate; no timing or competitor parity claim."""
import hashlib
import json
import os
import numpy as np
from mojolearn import RandomForestClassifier


def fingerprint(model):
    h = hashlib.sha256()
    for name in ('_offsets', '_colid', '_quesval', '_left_child', '_leaves'):
        a = np.ascontiguousarray(getattr(model, name))
        h.update(a.tobytes())
    return h.hexdigest()


def main():
    mode = os.environ.get('MOJOLEARN_NUMERIC_MODE')
    assert mode in ('fast', 'identical', 'deterministic'), 'explicit numeric mode required'
    rng = np.random.default_rng(17)
    x = rng.normal(size=(257, 5)).astype(np.float32)
    y = ((x[:, 0] + rng.normal(size=257) > 1.0)).astype(np.int32)
    results = []
    for bootstrap in (True, False):
        def fit(weight):
            model = RandomForestClassifier(n_estimators=3, max_depth=3,
                max_features=1.0, n_bins=16, bootstrap=bootstrap,
                random_state=7, n_streams=1, class_weight=weight, numeric_mode=mode)
            binding = model._bind()
            assert model.vendor_used() == 'cuda'
            assert binding.rf_numeric_mode() == {'fast': 0, 'identical': 1, 'deterministic': 2}[mode]
            model.fit(x, y)
            p = model.predict_proba(x)
            assert np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
            np.testing.assert_allclose(p.sum(axis=1), 1, atol=2e-6)
            return model, p
        plain, p0 = fit(None)
        unit, pu = fit({0: 1, 1: 1})
        assert fingerprint(plain) == fingerprint(unit)
        np.testing.assert_array_equal(p0, pu)
        for weights in ({0: 1, 1: 5}, 'balanced', {0: 0, 1: 1}):
            model, p = fit(weights)
            repeated, pr = fit(weights)
            assert fingerprint(model) == fingerprint(repeated)
            np.testing.assert_array_equal(p, pr)
            print('WEIGHT_CASE', bootstrap, weights, 'plain=', fingerprint(plain),
                  'weighted=', fingerprint(model), 'max_probability_change=', float(np.max(np.abs(p - p0))), flush=True)
            assert fingerprint(model) != fingerprint(plain), (bootstrap, weights, 'weighted path did not affect model')
            clipped = np.clip(p[np.arange(len(y)), y].astype(np.float64), 1e-15, 1)
            loss = float(-np.log(clipped).mean())  # Independent benchmark oracle only.
            results.append(dict(bootstrap=bootstrap, weights=weights,
                model_sha256=fingerprint(model), logloss=loss,
                accuracy=float(np.mean(np.argmax(p, axis=1) == y))))
    # Independent weighted stump oracle: a unit-scale integer bin would
    # erase the 0.25 minority weight; an unweighted bin gives 0.5 instead.
    stump_x = np.arange(8, dtype=np.float32).reshape(-1, 1)
    stump_y = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32)
    stump = RandomForestClassifier(n_estimators=1, max_depth=1,
        min_samples_split=100, bootstrap=False, class_weight={0: 0.25, 1: 1.5},
        random_state=7, n_streams=1, numeric_mode=mode).fit(stump_x, stump_y)
    np.testing.assert_allclose(stump.predict_proba(stump_x),
        np.tile(np.array([1/7, 6/7]), (8, 1)), rtol=2e-6, atol=2e-7)
    print(json.dumps(dict(mode=mode, vendor='cuda', cases=results), indent=2))
    print('RF_CLASS_WEIGHT_PASS')


if __name__ == '__main__':
    main()
