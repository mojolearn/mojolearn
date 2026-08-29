import faulthandler, time
import numpy as np
faulthandler.dump_traceback_later(100, exit=True)
import mojolearn as ml
X = np.random.default_rng(0).standard_normal((20000, 16)).astype(np.float32)
y = (X[:, 3] > 0).astype(np.int32)
t = time.time(); ml.ExtraTreesClassifier(n_estimators=16, max_depth=8, random_state=7).fit(X, y); print(f"ET fit ok {time.time()-t:.2f}s", flush=True)
