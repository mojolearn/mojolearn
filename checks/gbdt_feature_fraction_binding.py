"""Public GPU feature-fraction gate and pre-rebuild default fingerprint capture."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
from mojolearn import GradientBoosting
from mojolearn._gbdt_adapters import GradientBoostingClassifier, GradientBoostingRegressor


def fixture():
    rows = np.arange(256, dtype=np.int32)
    x = np.column_stack([((rows * (2*j+1) + j*13) % 127).astype(np.float32) / 32
                         for j in range(8)]).astype(np.float32)
    y = (x[:, 0] * .5 + x[:, 2] - x[:, 5] * .25).astype(np.float32)
    return x, y


def options(mode, policy, loss):
    return dict(loss=loss, n_estimators=3, max_depth=2, learning_rate=.25,
                border_count=8, random_state=17, numeric_mode=mode,
                grow_policy=policy, max_leaves=4 if policy == 'Lossguide' else None,
                bootstrap_type='Bayesian', bagging_temperature=0., random_strength=0.)


def fingerprint(model, x):
    return dict(model=hashlib.sha256(pickle.dumps(model.model_, protocol=5)).hexdigest(),
                prediction=hashlib.sha256(model.predict(x).tobytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture-default', type=Path)
    parser.add_argument('--compare-default', type=Path)
    args = parser.parse_args()
    x, y = fixture()
    baseline = {}
    checks = 0
    for mode in ('fast', 'deterministic', 'identical'):
        for policy in ('SymmetricTree', 'Depthwise', 'Lossguide'):
            for loss in ('RMSE', 'Logloss'):
                target = (y > np.median(y)).astype(np.float32) if loss == 'Logloss' else y
                opts = options(mode, policy, loss)
                default = GradientBoosting(**opts).fit(x, target)
                readback = default._bind('_mojolearn_gbdt').gbdt_numeric_mode()
                assert readback == {'fast': 0, 'identical': 1, 'deterministic': 2}[mode]
                key = '/'.join((mode, policy, loss))
                baseline[key] = fingerprint(default, x)
                checks += 1
                if args.capture_default:
                    continue
                explicit = GradientBoosting(**opts, feature_fraction=1.).fit(x, target)
                assert fingerprint(explicit, x) == baseline[key], (key, 'explicit default')
                for fraction in (.5, np.nextafter(0., 1.)):
                    first = GradientBoosting(**opts, feature_fraction=fraction).fit(x, target)
                    second = GradientBoosting(**opts, feature_fraction=fraction).fit(x, target)
                    assert fingerprint(first, x) == fingerprint(second, x), (key, fraction, 'repeat')
                    assert fingerprint(first, x)['model'] != baseline[key]['model'], (key, fraction, 'feature_fraction had no model effect')
                    assert np.isfinite(first.predict(x)).all()
                    cls = GradientBoostingClassifier if loss == 'Logloss' else GradientBoostingRegressor
                    adapter_opts = {k: v for k, v in opts.items() if k != 'loss'}
                    adapter = cls(**adapter_opts, feature_fraction=fraction).fit(x, target.astype(np.int32) if loss == 'Logloss' else target)
                    raw = adapter.decision_function(x) if loss == 'Logloss' else adapter.predict(x)
                    assert adapter.model_ == first.model_
                    np.testing.assert_array_equal(raw.view(np.uint32), first.predict(x).view(np.uint32))
                    checks += 3
                checks += 1
                print('PASS', key, 'feature fractions/default/adapters', flush=True)
        if not args.capture_default:
            # Exercise counted class weights plus all three optional tails,
            # evaluation forwarding, and weighted training in one small fit.
            opts = options(mode, 'Lossguide', 'Logloss')
            target = (y > np.median(y)).astype(np.float32)
            weighted = GradientBoosting(**opts, score_function='NewtonL2',
                class_weights=[1., 2.], min_split_gain=0., min_child_hessian=0.,
                feature_fraction=.5).fit(x, target,
                    sample_weight=np.linspace(.5, 1.5, len(x), dtype=np.float32),
                    eval_set=(x[:32], target[:32]))
            assert np.isfinite(weighted.predict(x)).all()
            assert weighted.test_loss_curve_ is not None
            checks += 1
    if args.capture_default:
        args.capture_default.write_text(json.dumps(baseline, indent=2, sort_keys=True) + '\n')
    if args.compare_default:
        assert json.loads(args.compare_default.read_text()) == baseline, 'pre-rebuild defaults changed'
    print(json.dumps({'status': 'PASS', 'checks': checks, 'default_fingerprints': baseline}, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
