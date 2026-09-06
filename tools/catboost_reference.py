# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CatBoost at the shape the port is measured on. CPU: its GPU cannot run here.

Matched to bench_realistic as closely as the two APIs allow: 800k x 100
numeric columns, border_count 254 (both defaults), depth 6, symmetric
(CatBoost's only GPU policy and our port's only policy), single thread of
boosting so ms/tree is the whole tree.
"""
import time, numpy as np, catboost, platform, os

rows, feats, depth, trees = 800_000, 100, 6, 20
rng = np.random.default_rng(0)
X = rng.integers(0, 255, size=(rows, feats)).astype(np.float32)
y = np.where(np.arange(rows) % 4 == 0, 3.0, -0.1).astype(np.float32)

pool = catboost.Pool(X, y)
# Quantize OUTSIDE the timed region. Our port consumes a compressed index
# and never builds borders, so leaving quantization inside fit charges
# CatBoost for work we do not do.
pool.quantize(border_count=254)
print(f"catboost {catboost.__version__} on {platform.machine()}, "
      f"{os.cpu_count()} cores, {rows} x {feats}, depth {depth}")

# quantization is not in our port, so it must not be in the timed region
best = None
for rep in range(3):
    m = catboost.CatBoostRegressor(
        iterations=trees, depth=depth, border_count=254,
        grow_policy="SymmetricTree", boosting_type="Plain",
        bootstrap_type="No", rsm=1.0, logging_level="Silent",
        allow_writing_files=False,
    )
    t0 = time.perf_counter()
    m.fit(pool)
    dt = time.perf_counter() - t0
    per_tree = dt * 1000.0 / trees
    print(f"  rep {rep}: {dt:.3f}s total, {per_tree:.2f} ms/tree")
    best = per_tree if best is None else min(best, per_tree)
print(f"CatBoost CPU: {best:.2f} ms/tree")
