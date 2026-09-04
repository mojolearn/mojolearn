#!/usr/bin/env python3
"""Independent mathematical oracle for checks in umap/checks/graph_check.mojo."""

import math

indices = [[0, 1, 2], [1, 0, 2], [2, 1, 0]]
distances = [[0.0, 1.0, 3.0], [0.0, 1.0, 2.0], [0.0, 2.0, 3.0]]
target = math.log2(3)

for row in distances:
    rho = next(d for d in row if d > 0.0)
    lo, hi = 0.0, 1.0
    score = lambda s: sum(1.0 if d <= rho else math.exp(-(d-rho)/s)
                          for d in row[1:])
    while score(hi) < target:
        hi *= 2.0
    for _ in range(64):
        sigma = (lo + hi) / 2.0
        if score(sigma) > target:
            hi = sigma
        else:
            lo = sigma
    print(f"rho={rho:.9g} sigma={sigma:.9g}")
