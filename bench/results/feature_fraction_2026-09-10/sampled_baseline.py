"""Same sampled learner fingerprints before/after allocation reuse changes."""
import hashlib, json, sys
import numpy as np
from mojolearn import GradientBoosting
rng = np.random.default_rng(20260910)
x = rng.standard_normal((4096, 32)).astype(np.float32)
x[:, 7] = 1
x[:, 14] = (x[:, 14] > 0)
x[:, 21] = np.floor(x[:, 21])
y = (x[:, 0] + x[:, 3] * .5 + (x[:, 5] > 0).astype(np.float32) + x[:, 9] * .2).astype(np.float32)
records = {}
for mode in ('fast', 'deterministic', 'identical'):
    for policy in ('SymmetricTree', 'Depthwise', 'Lossguide'):
        for fraction in (.5, .25):
            model = GradientBoosting(loss='RMSE', n_estimators=8, max_depth=6, learning_rate=.15,
                random_state=13, bootstrap_type='No', score_function='L2', border_count=32,
                numeric_mode=mode, grow_policy=policy,
                max_leaves=64 if policy == 'Lossguide' else None,
                feature_fraction=fraction).fit(x, y)
            records[f'{mode}/{policy}/{fraction}'] = {
                'model': hashlib.sha256(model.model_.encode()).hexdigest(),
                'prediction': hashlib.sha256(model.predict(x).tobytes()).hexdigest()}
with open(sys.argv[1], 'w') as out:
    json.dump(records, out, indent=2, sort_keys=True)
print('PASS sampled fingerprints', len(records), flush=True)
