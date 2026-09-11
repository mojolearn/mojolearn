# Lane linear-cluster-istella (2026-09-11): write the stage probe's raw blocks
# from a prepped classical block, through the SHIPPED Python layer's own
# centering, so the probe's OLS input is the bytes the public fit uploads.
#
#   python3 probe_bins.py <python tree> <dataset> <data dir> <out prefix>
#
# <prefix>_X.bin (float32 rows x cols), _Xc.bin and _yc.bin (the centered
# design and target LinearRegression.fit hands to ols_fit), _init.bin (the
# k-means init, 64 x cols). Prints each file's sha256 prefix.
import hashlib
import os
import sys

os.environ["MOJOLEARN_NUMERIC_MODE"] = "identical"
sys.path.insert(0, sys.argv[1])
import numpy as np  # noqa: E402
from mojolearn import linear_model as lm  # noqa: E402
from mojolearn._buffer import as_f32_c  # noqa: E402

ds, data, prefix = sys.argv[2], sys.argv[3], sys.argv[4]
os.makedirs(os.path.dirname(prefix), exist_ok=True)
with np.load(os.path.join(data, "big-%s.npz" % ds)) as z:
    X = np.ascontiguousarray(z["X"], dtype=np.float32)
    y = np.ascontiguousarray(z["y"], dtype=np.float32)
    init = np.ascontiguousarray(z["init"], dtype=np.float32)


def dump(name, arr):
    arr = np.ascontiguousarray(np.asarray(arr), dtype=np.float32)
    arr.tofile(prefix + name)
    print(name, arr.shape, hashlib.sha256(arr.tobytes()).hexdigest()[:16], flush=True)


dump("_X.bin", X)
dump("_init.bin", init)
x, _ = as_f32_c(X, ndim=2, name="X")
mu32 = lm._column_means(x, None)
xc = lm._center(x, mu32)
t = lm._target_1d(y, X.shape[0], "one target", "lengths differ")
ymean = lm._vector_mean(t, None)
yc = lm._shift(t, lm._round_f32(ymean))
dump("_Xc.bin", np.frombuffer(memoryview(xc), dtype=np.float32).reshape(X.shape))
dump("_yc.bin", np.frombuffer(memoryview(yc), dtype=np.float32))
print("shape", X.shape[0], X.shape[1], flush=True)
