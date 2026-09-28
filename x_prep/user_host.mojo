# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""IterativeImputer with a user `estimator`: the host plumbing around the
estimator's own fit / predict (lane py-misc-prep, 2026-09-28). Before this
file `_fit_host` held the working block as nested Python lists, deep-copied
it every round, scanned a nested mask per feature, built each fit and
predict input with `Array.from_list` and ran the round's convergence test in
Python float64. These entries do the same work on the caller's float32
block in place:

- `ii_rows`: the rows whose mask word for feature j is (or is not) zero,
  ascending (the comprehensions `[i for i in range(n) if mask[i][j]]`).
- `ii_gather`: X[rows][:, cols] and, when asked, X[rows, j]: word copies in
  the comprehensions' order.
- `ii_scatter`: X[rows[r], j] = float32(clip(v[r])) with Python's
  `min(max(v, lo), hi)` (max keeps v unless lo > v; min keeps it unless
  hi < it, so a NaN passes through), then the double -> float32 cast
  `array('f')` makes (round to nearest even).
- `ii_conv`: the round's `max(sum(abs(a - b) for a, b in zip(ra, rb)) for
  ra, rb in zip(Xt, prev))` in binary64: each term one exact widening of two
  float32 words, one subtract, one fabs; the row sum in CPython's own
  order, Neumaier-compensated as `sum()` of floats is since 3.12
  (`compensated`), or plain left-to-right as before 3.12; the max keeps the
  first row's sum and replaces it only by a strictly greater one. No
  product, no library call, so nothing here can contract or differ by host.

Integer and copy work is exact; the only float arithmetic is `ii_conv`'s,
restated operation for operation, so no bit of any output moves. The
Python spelling stays the reference (`MOJOLEARN_HOTPATH=python` and a
binding without these entries run it)."""

comptime F32P = MutPointer[Float32, MutAnyOrigin]
comptime I32P = MutPointer[Int32, MutAnyOrigin]
comptime F64P = MutPointer[Float64, MutAnyOrigin]


def ii_rows(mask: F32P, n: Int, dk: Int, j: Int, missing: Bool, rows: I32P) -> Int:
    """rows[0:m] = the ascending rows whose mask[i, j] != 0 (`missing`) or == 0; returns m."""
    var m = 0
    for i in range(n):
        var v = mask[i * dk + j]
        if (v != Float32(0)) == missing:
            rows[m] = Int32(i)
            m += 1
    return m


def ii_gather(x: F32P, dk: Int, rows: I32P, m: Int, cols: I32P, nc: Int, j: Int, dst: F32P, y: F32P):
    """dst[r, c] = x[rows[r], cols[c]]; y[r] = x[rows[r], j] when j >= 0."""
    for r in range(m):
        var base = Int(rows[r]) * dk
        for c in range(nc):
            dst[r * nc + c] = x[base + Int(cols[c])]
        if j >= 0:
            y[r] = x[base + j]


def ii_scatter(x: F32P, dk: Int, rows: I32P, m: Int, j: Int, v: F64P, lo: Float64, hi: Float64, clip: Bool):
    """x[rows[r], j] = float32(min(max(v[r], lo), hi)) (Python's rule), or float32(v[r])."""
    for r in range(m):
        var t = v[r]
        if clip:
            if lo > t:
                t = lo
            if hi < t:
                t = hi
        x[Int(rows[r]) * dk + j] = Float32(t)


def ii_conv(a: F32P, b: F32P, n: Int, dk: Int, compensated: Bool) -> Float64:
    """The largest row sum of |a - b| (binary64), in CPython `sum` / `max` order."""
    var best = Float64(0)
    for i in range(n):
        var s = Float64(0)
        var c = Float64(0)
        for k in range(dk):
            var d = Float64(a[i * dk + k]) - Float64(b[i * dk + k])
            var x = abs(d)
            if compensated:
                var t = s + x
                if abs(s) >= abs(x):
                    c += (s - t) + x
                else:
                    c += (x - t) + s
                s = t
            else:
                s = s + x
        if compensated and c != Float64(0) and (c - c) == Float64(0):
            s = s + c
        if i == 0 or s > best:
            best = s
    return best
