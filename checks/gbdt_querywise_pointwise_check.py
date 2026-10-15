"""Post-build grouped-target checks; run explicitly on each candidate GPU.

This is a test controller, not product execution or a performance measurement.
Retain the complete command log. Copy the same --data NPZ to each vendor and
compare outputs from identical-mode runs; one vendor's repeat is insufficient.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import traceback

import numpy as np


def prepare(path):
    rng = np.random.default_rng(901)
    sizes = np.tile([1, 2, 7, 3, 16, 1, 5, 40, 2, 9], 4)
    group = np.repeat(np.arange(len(sizes)), sizes)
    x = rng.normal(size=(len(group), 7)).astype(np.float32)
    y = np.digitize(x[:, 0] + .3 * x[:, 1], [-.7, 0., .7]).astype(np.float32)
    np.savez(path, X=x, y=y, group=group)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--vendor', required=True)
    ap.add_argument('--prepare-only', action='store_true')
    ap.add_argument('--numeric-mode', choices=('identical', 'fast'), default='identical')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.prepare_only:
        if args.data.exists():
            raise FileExistsError(args.data)
        prepare(args.data)
        print(json.dumps({'data': str(args.data), 'sha256': sha(args.data)}))
        return 0

    # Import only after preparing fixtures, so fixture preparation needs no GPU.
    from mojolearn import GradientBoosting

    with np.load(args.data, allow_pickle=False) as data:
        x, y, groups = data['X'], data['y'], data['group']
    report = dict(complete=False, exit_status=None, vendor=args.vendor,
                  numeric_mode=args.numeric_mode, data=str(args.data),
                  data_sha256=sha(args.data), shape=list(x.shape),
                  version=importlib.metadata.version('mojolearn'),
                  scope='targeted numerical regression; no performance qualification',
                  cases=[], failures=[])
    summary = args.out / 'summary.json'

    def save():
        summary.write_text(json.dumps(report, indent=2) + '\n')

    def fit(loss, tag, *, group=groups, pairs=None, pairs_weight=None, **kw):
        params = dict(loss=loss, n_estimators=5, max_depth=3, learning_rate=.1,
                      random_state=19, use_pointwise_searcher=True,
                      boosting_type='Plain', bootstrap_type='No',
                      random_strength=0., numeric_mode=args.numeric_mode)
        params.update(kw)
        model = GradientBoosting(**params).fit(
            x, y, group_id=group, pairs=pairs, pairs_weight=pairs_weight)
        pred = np.asarray(model.predict(x))
        curve = np.asarray(model.loss_curve_, dtype=np.float64)
        if not np.isfinite(pred).all() or not np.isfinite(curve).all():
            raise AssertionError(f'{tag}: nonfinite output')
        if np.ptp(pred) == 0:
            raise AssertionError(f'{tag}: constant predictions')
        outputs = {}
        for name, value in (('predict', pred), ('loss_curve', curve)):
            path = args.out / f'{loss}-{tag}-{name}.npy'
            np.save(path, value, allow_pickle=False)
            outputs[name] = dict(path=str(path), sha256=sha(path),
                                 dtype=str(value.dtype), shape=list(value.shape))
        model_path = args.out / f'{loss}-{tag}-model.npz'
        model.save(model_path)
        outputs['model'] = dict(path=str(model_path), sha256=sha(model_path))
        # Independent test metric, retained beside every model, never a
        # cross-library admission gate. Ties preserve the input row order.
        ndcgs = []
        for q in np.unique(group):
            ix = np.flatnonzero(group == q)
            gains = np.expm1(y[ix].astype(np.float64) * np.log(2.))
            discount = 1. / np.log2(np.arange(len(ix)) + 2.)
            ideal = np.dot(np.sort(gains)[::-1], discount)
            order = np.argsort(-pred[ix], kind='stable')
            ndcgs.append(float(np.dot(gains[order], discount) / ideal) if ideal else 1.)
        report['cases'].append(dict(loss=loss, tag=tag, params=params,
                                    explicit_pairs=0 if pairs is None else len(pairs),
                                    mean_ndcg=float(np.mean(ndcgs)),
                                    outputs=outputs))
        save()
        return pred, curve

    save()
    try:
        for loss in ('QueryRMSE', 'PairLogit', 'YetiRank'):
            baseline, curve = fit(loss, 'plain')
            repeated, repeated_curve = fit(loss, 'repeat')
            if args.numeric_mode == 'identical':
                assert baseline.tobytes() == repeated.tobytes(), loss + ': repeat predictions'
                assert curve.tobytes() == repeated_curve.tobytes(), loss + ': repeat curve'
            if loss != 'YetiRank':
                assert curve[-1] < curve[0], loss + ': loss did not improve'
            zero, _ = fit(loss, 'bayesian-zero', bootstrap_type='Bayesian', bagging_temperature=0.)
            if args.numeric_mode == 'identical':
                assert zero.tobytes() == baseline.tobytes(), loss + ': zero bootstrap changed model'
            sampled, _ = fit(loss, 'bayesian', bootstrap_type='Bayesian', bagging_temperature=.7)
            assert sampled.tobytes() != baseline.tobytes(), loss + ': bootstrap ignored'
            noisy, _ = fit(loss, 'score-noise', random_strength=1.)
            assert noisy.tobytes() != baseline.tobytes(), loss + ': pointwise score noise ignored'
            fit(loss, 'newton-score', score_function='NewtonCosine')
            # Retain order, change boundaries: a pointwise-loss fallback would
            # ignore this. Every alternative group is still mixed-grade.
            alternate, _ = fit(loss, 'different-groups', group=np.arange(len(groups)) // 5)
            assert alternate.tobytes() != baseline.tobytes(), loss + ': grouping ignored'

        pairs = []
        for q in np.unique(groups):
            rows = np.flatnonzero(groups == q)
            for i, a in enumerate(rows):
                for b in rows[i + 1:]:
                    if y[a] != y[b]:
                        pairs.append((int(a), int(b)) if y[a] > y[b] else (int(b), int(a)))
        unit, _ = fit('PairLogit', 'explicit-unit', pairs=pairs)
        weighted, _ = fit('PairLogit', 'explicit-weighted', pairs=pairs,
                          pairs_weight=np.linspace(.5, 2., len(pairs)))
        assert unit.tobytes() != weighted.tobytes(), 'explicit pair weights ignored'
        # Simple uses the searcher's bootstrapped leaves in CatBoost. The new
        # grouped-oracle route must refuse it instead of silently substituting.
        try:
            fit('PairLogit', 'unsupported-simple', leaf_estimation_method='Simple')
        except Exception as exc:
            if 'requires Newton or Gradient' not in str(exc):
                raise
            report['cases'].append(dict(loss='PairLogit', tag='unsupported-simple',
                                        expected_refusal=str(exc)))
        else:
            raise AssertionError('querywise pointwise Simple was silently accepted')
        report['complete'] = True
        report['exit_status'] = 0
    except Exception:
        report['failures'].append(traceback.format_exc())
        report['exit_status'] = 1
        traceback.print_exc()
    finally:
        save()
    print(json.dumps(dict(complete=report['complete'], exit_status=report['exit_status'],
                          cases=len(report['cases']), failures=len(report['failures']),
                          evidence=str(summary))))
    return report['exit_status']


if __name__ == '__main__':
    sys.exit(main())
