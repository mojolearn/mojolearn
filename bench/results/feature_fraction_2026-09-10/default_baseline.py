"""Capture exact legacy-default model/prediction hashes before/after rebuilding."""
import hashlib, json, sys
import numpy as np
from mojolearn import GradientBoosting
X = np.asarray([[float(i % 8), float((i * 5) % 11), float((i * 3) % 7), 1.] for i in range(48)], np.float32)
y = ((X[:, 0] >= 4) ^ (X[:, 1] >= 5)).astype(np.float32)
records = {}
for mode in ('fast', 'deterministic', 'identical'):
    for policy in ('SymmetricTree', 'Depthwise', 'Lossguide'):
        for loss in ('RMSE', 'Logloss'):
            model = GradientBoosting(loss=loss, n_estimators=3, max_depth=3,
                random_state=13, bootstrap_type='No', score_function='L2',
                numeric_mode=mode, grow_policy=policy,
                max_leaves=8 if policy == 'Lossguide' else None).fit(X, y)
            value = model.model_
            if isinstance(value, str): value = value.encode()
            records[f'{mode}/{policy}/{loss}'] = {
                'model': hashlib.sha256(value).hexdigest(),
                'prediction': hashlib.sha256(model.predict(X).tobytes()).hexdigest()}
with open(sys.argv[1], 'w') as out:
    json.dump(records, out, indent=2, sort_keys=True)
print('PASS default fingerprints', len(records), flush=True)
