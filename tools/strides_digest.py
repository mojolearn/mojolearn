#!/usr/bin/env python3
"""Mamba3Block forward + backward at one seeded shape; SHA-256 of y and every gradient (strides-pass A/B)."""
import hashlib, json, sys
import numpy as np
from mojolearn import Mamba3Block
B, L, dm = (int(v) for v in sys.argv[1:4])
di = 2 * dm; H = di // 64; N = 128; dip = 2 * di + 256 + 3 * H + 32
rng = np.random.default_rng(7)
f = lambda s, sc=0.02: (rng.standard_normal(s) * sc).astype(np.float32)
w = {"block_norm.weight": np.ones((dm,), np.float32), "in_proj.weight": f((dip, dm)), "dt_bias": f((H,), 0.1),
     "B_norm.weight": np.ones((N,), np.float32), "C_norm.weight": np.ones((N,), np.float32),
     "B_bias": np.ones((H, N), np.float32), "C_bias": np.ones((H, N), np.float32), "D": np.ones((H,), np.float32),
     "out_proj.weight": f((dm, di))}
blk = Mamba3Block(w); x = f((B, L, dm), 1.0); dy = f((B, L, dm), 1.0)
out = {}
for call in range(2):
    y = blk.forward(x); g = blk.backward(x, dy)
    items = [("y", y)] + (sorted(g.items()) if isinstance(g, dict) else [("g%d" % i, v) for i, v in enumerate(g if isinstance(g, (list, tuple)) else [g])])
    out["call%d" % call] = {k: hashlib.sha256(np.ascontiguousarray(np.asarray(v)).tobytes()).hexdigest()[:16] for k, v in items}
print(json.dumps(out, indent=1))
