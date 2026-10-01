"""sha256 of the inputs tools/host_threads_ab_check.py builds for the transformer
at a length (default 2048), so two hosts can tell input drift from arithmetic:
    python3 tools/transformer_ab_input_sha.py [length]"""
import hashlib
import sys

import numpy as np

sys.path.insert(0, "tools")
from host_threads_ab_check import SHAPES, _transformer_weights  # noqa: E402

L = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
s = dict(SHAPES["transformer"])
s["length"] = L
rng = np.random.default_rng(11)
x = (rng.standard_normal((s["batch"], s["length"], s["d_model"]), dtype=np.float32) * np.float32(0.5)).astype("<f4")
w = _transformer_weights(np, rng, s)
h = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]
print("numpy", np.__version__, "length", L, "x", h(x))
for k in sorted(w):
    print("  ", k, w[k].shape, h(w[k]))
