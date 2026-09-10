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
            assert fingerprint(model) != fingerprint(plain), 'weighted path did not affect model'
            clipped = np.clip(p[np.arange(len(y)), y].astype(np.float64), 1e-15, 1)
            loss = float(-np.log(clipped).mean())  # Independent benchmark oracle only.
            results.append(dict(bootstrap=bootstrap, weights=weights,
                model_sha256=fingerprint(model), logloss=loss,
                accuracy=float(np.mean(np.argmax(p, axis=1) == y))))
    print(json.dumps(dict(mode=mode, vendor='cuda', cases=results), indent=2))
    print('RF_CLASS_WEIGHT_PASS')


if __name__ == '__main__':
    main()
