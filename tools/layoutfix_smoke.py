"""Import mojolearn and fit the estimators whose probes moved to
_backend.entry_or_none, on tiny data (lane/apple-fast-layoutfix)."""
import numpy as np
import mojolearn
from mojolearn import _backend
from mojolearn import RandomForestClassifier, ExtraTreesClassifier, IsolationForest, Ridge
from mojolearn._expansion_neighbors import AdditiveChi2Sampler

assert issubclass(_backend.NoCpuEntry, ImportError)
rng = np.random.RandomState(0)
X = rng.rand(64, 4).astype(np.float32)
y = (X[:, 0] > 0.5).astype(np.int64)
for est in (RandomForestClassifier(n_estimators=4, random_state=0),
            ExtraTreesClassifier(n_estimators=4, random_state=0)):
    est.fit(X, y)
    print("LAYOUTFIX", type(est).__name__, "ok", float((np.asarray(est.predict(X)) == y).mean()))
IsolationForest(n_estimators=4, random_state=0).fit(X)
print("LAYOUTFIX IsolationForest ok")
Ridge().fit(X, X[:, 0].astype(np.float64))
print("LAYOUTFIX Ridge ok")
Z = AdditiveChi2Sampler().fit_transform(X)
print("LAYOUTFIX AdditiveChi2Sampler ok", np.asarray(Z).shape)
print("LAYOUTFIX ALL OK", mojolearn.__file__)
