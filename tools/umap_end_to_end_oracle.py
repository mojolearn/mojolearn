#!/usr/bin/env python3
"""Independent umap-learn structural oracle for the estimator fixture."""

import numpy as np

try:
    import umap
except ImportError as exc:
    raise SystemExit("install umap-learn to run this optional oracle") from exc

x = np.array([0, 1, 2.2, 4, 6.5, 10, 14.5, 20], dtype=np.float32)[:, None]
layout = umap.UMAP(
    n_neighbors=3,
    n_components=2,
    n_epochs=4,
    init="spectral",
    random_state=19,
    transform_seed=19,
).fit_transform(x)
if layout.shape != (8, 2) or not np.isfinite(layout).all():
    raise SystemExit("umap-learn returned an invalid fixture layout")
if np.max(np.linalg.norm(layout - layout.mean(axis=0), axis=1)) == 0:
    raise SystemExit("umap-learn returned a collapsed fixture layout")
print(layout)
