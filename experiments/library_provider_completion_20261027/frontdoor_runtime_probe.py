#!/usr/bin/env python3
"""Bounded routing probe, not full native-runtime or performance qualification."""
import argparse
import json
from pathlib import Path
import traceback

def run():
    import numpy as np
    from mojolearn import SVC, SVR, LinearSVC, LinearSVR, KMeans
    from mojolearn import _backend, _native_family_bridge
    from mojolearn._mode import NumericModeMixin
    rng = np.random.default_rng(77)
    x = rng.normal(size=(16, 3)).astype('f4')
    yc = (x[:, 0] > 0).astype('i4')
    yr = (x[:, 0] - x[:, 1]).astype('f4')
    common = dict(numeric_mode='identical')
    cases = [
        ('LinearSVC', LinearSVC(max_iter=8, **common), yc, ('predict', 'decision_function', 'score')),
        ('LinearSVR', LinearSVR(max_iter=8, **common), yr, ('predict', 'score')),
        ('SVC', SVC(kernel='linear', max_iter=16, **common), yc, ('predict', 'decision_function', 'score')),
        ('SVR', SVR(kernel='linear', max_iter=16, **common), yr, ('predict', 'score')),
        ('KMeans', KMeans(n_clusters=2, n_init=1, max_iter=4, random_state=7, **common), None,
         ('predict', 'transform')),
    ]
    old_binding, old_bind = _backend.binding, NumericModeMixin._bind
    calls = []
    def refuse(*args, **kwargs):
        calls.append('Python binding loader entered during public operation')
        raise AssertionError(calls[-1])
    records = []
    try:
        _backend.binding = refuse
        NumericModeMixin._bind = refuse
        for label, model, y, methods in cases:
            try:
                model.fit(x) if y is None else model.fit(x, y)
                for method in methods:
                    value = getattr(model, method)(x, y) if method == 'score' else getattr(model, method)(x)
                    assert np.isfinite(np.asarray(value)).all()
                records.append(dict(case=label, status='passed'))
            except Exception:
                records.append(dict(case=label, status='failed', error=traceback.format_exc()))
    finally:
        _backend.binding, NumericModeMixin._bind = old_binding, old_bind
    # Runtime mode readback must reject a Python replacement before invoking
    # its hook, then recover after restoring the live native module entry.
    module = _native_family_bridge.shared_context()['family_binding_connections']['_mojolearn']['identical'][0]
    reader = _backend._vendor_fn('_mojolearn').removesuffix('_vendor') + '_numeric_mode'
    saved = getattr(module, reader)
    getter_calls = []
    def wrong_mode():
        getter_calls.append(True)
        return 0
    try:
        setattr(module, reader, wrong_mode)
        try:
            cases[-1][1].predict(x)
        except Exception as error:
            assert 'compiled entry' in str(error)
            assert not getter_calls
        else:
            raise AssertionError('Python mode getter replacement accepted')
    finally:
        setattr(module, reader, saved)
    cases[-1][1].predict(x)
    records.append(dict(case='native_mode_getter_replacement',status='passed'))
    return dict(scope='shared binding admission only', records=records, forbidden_loader_calls=calls,
                full_native_runtime_qualified=False,
                pending=['transitive operation dispatch', 'availability/property descriptors',
                         'all metadata read counts', 'cross-vendor and performance evidence'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run()
    except Exception:
        result = dict(scope='shared binding admission only', fatal=traceback.format_exc())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    raise SystemExit(1 if 'fatal' in result or any(r['status'] != 'passed' for r in result['records']) else 0)
