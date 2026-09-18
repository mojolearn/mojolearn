"""Two resident forests in one process: is the hang the second live DeviceContext?"""
import gc, os, sys, time
import numpy as np
sys.path.insert(0, "python")
import mojolearn as ml
from mojolearn._serialize import read_npz, scalar_str
mode = sys.argv[1]; paths = sys.argv[2:]
def load(p):
    arrays = read_npz(p, ("mojolearn-randomforest-1-parallel-groves-1", "mojolearn-extratrees-1-parallel-groves-1", "mojolearn-randomforest-1", "mojolearn-extratrees-1"))
    return getattr(ml, scalar_str(arrays, "estimator")).load(p)
x = np.load(os.path.join(os.path.dirname(paths[0]), "x_higgs.npy"))[:2000]
first = None
for i, p in enumerate(paths):
    t = time.perf_counter(); m = load(p); xi = x[:, :m.n_features_in_] if x.shape[1] >= m.n_features_in_ else np.ascontiguousarray(np.tile(x, (1, (m.n_features_in_ + x.shape[1] - 1) // x.shape[1]))[:, :m.n_features_in_])
    out = m.predict(np.ascontiguousarray(xi, dtype=np.float32))
    print(f"model {i} {os.path.basename(p)} predicted {out.shape} in {time.perf_counter() - t:.2f}s resident={getattr(m, '_resident', None) is not None}", flush=True)
    if mode == "release":
        del m; gc.collect(); print("released", flush=True)
    else:
        first = m if first is None else first
print("DONE", mode, flush=True)
