import faulthandler, time
import numpy as np
faulthandler.dump_traceback_later(600, exit=True)
import mojolearn as ml
X = np.random.default_rng(0).standard_normal((2000, 8)).astype(np.float32)
t = time.time()
print("IF fit begin", flush=True)
ml.IsolationForest(n_estimators=4, random_state=5).fit(X)
print(f"IF fit returned {time.time()-t:.2f}s", flush=True)
