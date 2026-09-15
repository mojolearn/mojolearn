import sys
import numpy as np
from mojolearn import SpectralClustering
def say(*a):
    print(*a, flush=True)
rng = np.random.default_rng(1)
x = np.ascontiguousarray(rng.normal(size=(60, 2)).astype(np.float32))
say("fit plain")
SpectralClustering(n_clusters=3, n_neighbors=6, random_state=3).fit(x)
say("fit prediction_data")
m = SpectralClustering(n_clusters=3, n_neighbors=6, random_state=3, prediction_data=True).fit(x)
say("fit ok", np.asarray(m._pd_eigenvalues).tolist())
say("predict one row")
say("labels", np.asarray(m.predict(x[:1])).tolist())
say("predict all")
say("labels", np.asarray(m.predict(x)).tolist()[:10])
