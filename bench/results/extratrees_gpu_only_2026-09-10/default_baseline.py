"""GPU-default forest arrays/predictions before and after removing CPU dispatch."""
import hashlib, json, sys
import numpy as np
from mojolearn import ExtraTreesClassifier, ExtraTreesRegressor
X = np.asarray([[i % 8, (i * 3) % 11] for i in range(32)], np.float32)
y = (X[:, 0] >= 4).astype(np.float32)
records = {}
for mode in ('fast', 'deterministic', 'identical'):
    for cls in (ExtraTreesClassifier, ExtraTreesRegressor):
        model = cls(n_estimators=2, max_depth=3, random_state=13, numeric_mode=mode).fit(X, y)
        digest = hashlib.sha256()
        for name in ('_offsets', '_colid', '_quesval', '_left_child', '_leaves'):
            digest.update(getattr(model, name).tobytes())
        digest.update(model.predict(X).tobytes())
        records[f'{mode}/{cls.__name__}'] = digest.hexdigest()
with open(sys.argv[1], 'w') as out:
    json.dump(records, out, indent=2, sort_keys=True)
print('PASS GPU-default forest fingerprints', len(records), flush=True)
