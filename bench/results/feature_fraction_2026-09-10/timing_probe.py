"""Exploratory interleaved public-fit cost; sampling changes the fitted learner."""
import json, time
from pathlib import Path
import numpy as np
from mojolearn import GradientBoosting

rng = np.random.default_rng(20260910)
x = rng.standard_normal((9216, 32)).astype(np.float32)
y = (x[:, 0] + x[:, 3] * .5 + (x[:, 5] > 0).astype(np.float32) + x[:, 9] * .2).astype(np.float32)
train_x, test_x = x[:8192], x[8192:]
train_y, test_y = y[:8192], y[8192:]
fractions = [1., .5, .25]
records = []
for mode in ('fast', 'identical'):
    for policy in ('SymmetricTree', 'Depthwise', 'Lossguide'):
        opts = dict(loss='RMSE', n_estimators=8, max_depth=6, learning_rate=.15,
                    random_state=13, bootstrap_type='No', score_function='L2',
                    border_count=32, numeric_mode=mode, grow_policy=policy,
                    max_leaves=64 if policy == 'Lossguide' else None)
        for f in fractions:
            GradientBoosting(**opts, feature_fraction=f).fit(train_x, train_y)
        times = {f: [] for f in fractions}
        quality = {}
        for repeat in range(3):
            order = fractions[repeat:] + fractions[:repeat]
            for f in order:
                model = GradientBoosting(**opts, feature_fraction=f)
                start = time.perf_counter()
                model.fit(train_x, train_y)
                elapsed = time.perf_counter() - start
                times[f].append(elapsed)
                # Independent host quality oracle, not a product CPU scorer.
                residual = model.predict(test_x).astype(np.float64) - test_y
                quality[f] = float(np.sqrt(np.mean(residual * residual)))
        reference = times[1.]
        record = dict(mode=mode, policy=policy, rows=8192, features=32,
                      trees=8, depth=6, raw_seconds=times, heldout_rmse=quality,
                      reference_spread=max(reference)/min(reference),
                      median_ratio_vs_full={f: float(np.median(times[f])/np.median(reference)) for f in fractions})
        record['stable_reference'] = record['reference_spread'] <= 1.1
        records.append(record)
        print(json.dumps(record), flush=True)
Path('bench/results/feature_fraction_2026-09-10/timing.json').write_text(json.dumps(records, indent=2)+'\n')
