# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""t-SNE's start y0 in Mojo (lane apple-fast-py2mojo-cluster, 2026-10-03).
`_expansion_ann.TSNE._init` built it in Python: init='pca' scaled the PCA
embedding by the float64 population standard deviation of its first column
(two `math.fsum`s, then `(emb / float32(std)) * float32(1e-4)` in numpy
float32); init='random' drew NumPy's PCG64 `(random((n, 2)) - 0.5) * 1e-4`.
Both bindings call `tsne_init_y0` before the fit, so the two columns read the
same start:

  mode 1 (random): a SplitMix64 stream keyed by random_state, one 53-bit
         uniform per value, `(u - 0.5) * 1e-4` in float64 rounded once to
         float32. NOT NumPy's stream (a new draw, the same distribution);
  mode 2 (pca):    the fsum port below (CPython `math.fsum`, exact) gives
         Python's mean and standard deviation bit for bit, then the two
         float32 operations numpy did.

Switch: `X_ANN_PY2MOJO` (default on; `-D MOJOLEARN_PY2MOJO_cluster_OFF` makes
the bindings report 0 through `x_ann_py2mojo` and Python builds y0 as
before)."""
from std.math import sqrt
from std.sys.compile import is_defined

from checks.numerics import identical_mul64

#: M3 A/B (lane/apple-fast-py2mojo-cluster 2b6f3bfd4, 2026-10-04, 1 run per
#: arm): ivf-refine taxi 1196.0 -> 1198.9 ms with -D MOJOLEARN_PY2MOJO_cluster_OFF,
#: NEUTRAL; default unchanged (on).
comptime X_ANN_PY2MOJO = not is_defined["MOJOLEARN_PY2MOJO_cluster_OFF"]()

comptime TSNE_INIT_GIVEN = 0
comptime TSNE_INIT_RANDOM = 1
comptime TSNE_INIT_PCA = 2


def fsum(values: List[Float64]) -> Float64:
    """CPython `math.fsum` (Shewchuk's exact partials, then the round-half-even
    correction of the top two), for finite inputs."""
    var partials = List[Float64]()
    for v in values:
        var x = v
        var i = 0
        for j in range(len(partials)):
            var y = partials[j]
            if abs(x) < abs(y):
                var t = x
                x = y
                y = t
            var hi = x + y
            var lo = y - (hi - x)
            if lo != Float64(0.0):
                partials[i] = lo
                i += 1
            x = hi
        while len(partials) > i:
            _ = partials.pop()
        partials.append(x)
    var n = len(partials)
    var hi = Float64(0.0)
    if n > 0:
        n -= 1
        hi = partials[n]
        var lo = Float64(0.0)
        while n > 0:
            var x = hi
            n -= 1
            var y = partials[n]
            hi = x + y
            var yr = hi - x
            lo = y - yr
            if lo != Float64(0.0):
                break
        if n > 0 and ((lo < Float64(0.0) and partials[n - 1] < Float64(0.0)) or (
            lo > Float64(0.0) and partials[n - 1] > Float64(0.0)
        )):
            var y = lo * Float64(2.0)
            var x = hi + y
            var yr = x - hi
            if y == yr:
                hi = x
    return hi


@always_inline
def _splitmix(state: UInt64) -> UInt64:
    var z = state
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


def tsne_init_y0(
    mode: Int, y0: List[Float32], n: Int, seed: UInt64, n_components: Int = 2
) raises -> List[Float32]:
    """The start (n x 2 float32): mode 0 `y0` as given, 1 random, 2 the PCA
    embedding `y0` scaled (module docstring).

    `n_components == 1` (lane sk-kernels, 2026-10-15): `y0` holds n values
    (the given column, or the one-component PCA embedding) and the start is
    still n x 2 with the second column +0. The fit then runs the
    two-dimensional kernels on it: every term of that column is
    q * (0 - 0) = +0 (attraction, repulsion, gradient, update), so it stays
    +0 for the whole run, every q is the one-dimensional Student-t kernel
    (degrees of freedom max(n_components - 1, 1) = 1 either way) and
    the first column is exactly the one-dimensional t-SNE.
    `tsne_first_components` takes it back out."""
    if n_components != 1 and n_components != 2:
        raise Error(
            "mojolearn TSNE: n_components must be 1 or 2, got " + String(n_components)
        )
    var k = n_components
    if mode == TSNE_INIT_GIVEN:
        if k == 2:
            return y0.copy()
        var given = List[Float32](capacity=2 * n)
        for i in range(n):
            given.append(y0[i])
            given.append(Float32(0.0))
        return given^
    var out = List[Float32](capacity=2 * n)
    if mode == TSNE_INIT_RANDOM:
        for t in range(k * n):
            var z = _splitmix(seed + UInt64(t + 1) * UInt64(0x9E3779B97F4A7C15))
            var u = Float64(z >> 11) * Float64(1.1102230246251565e-16)  # 2^-53
            out.append(Float32(identical_mul64(u - Float64(0.5), Float64(1e-4))))
            if k == 1:
                out.append(Float32(0.0))
        return out^
    if mode != TSNE_INIT_PCA:
        raise Error("mojolearn TSNE: unknown init mode " + String(mode))
    var col = List[Float64](capacity=n)
    for i in range(n):
        col.append(Float64(y0[k * i]))
    var mean = fsum(col) / Float64(n)
    var sq = List[Float64](capacity=n)
    for i in range(n):
        var dv = col[i] - mean
        sq.append(identical_mul64(dv, dv))
    var std = sqrt(fsum(sq) / Float64(n))
    if not std > Float64(0.0):
        raise Error("mojolearn TSNE: init='pca' gave a constant first component; pass init='random'")
    var s32 = Float32(std)
    var c = Float32(1e-4)
    for t in range(k * n):
        out.append((y0[t] / s32) * c)
        if k == 1:
            out.append(Float32(0.0))
    return out^


def tsne_first_components(y: List[Float32], n: Int, n_components: Int) -> List[Float32]:
    """The fitted n x 2 embedding as the caller's n x n_components: the
    whole list at 2, its first column at 1 (see `tsne_init_y0`)."""
    if n_components == 2:
        return y.copy()
    var out = List[Float32](capacity=n)
    for i in range(n):
        out.append(y[2 * i])
    return out^
