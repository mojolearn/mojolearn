#!/usr/bin/env python3
"""NumPy oracle for the normalized-path spectral initialization fixture."""

import numpy as np

n = 10
a = np.zeros((n, n), dtype=np.float64)
for i in range(n - 1):
    a[i, i + 1] = a[i + 1, i] = 1.0
d = a.sum(axis=1)
lap = np.eye(n) - (a / np.sqrt(d[:, None] * d[None, :]))
_, vectors = np.linalg.eigh(lap)
out = vectors[:, 1:4].copy()
out /= np.sqrt(d[:, None])
for c in range(3):
    pivot = np.argmax(np.abs(out[:, c]))
    out[:, c] *= (10.0 / abs(out[pivot, c])) * np.sign(out[pivot, c])
np.set_printoptions(precision=9, suppress=True)
print(out.astype(np.float32))
