# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`normalize_y`'s un-normalization in Mojo (lane apple-fast-py2mojo-cluster,
2026-10-03). `_gp_impl.py` applied it in Python to every predicted mean,
standard deviation, covariance cell and sample_y draw: sklearn `_gpr.py:450`
and `:494`, `y_mean = std * y_mean + mean`, `y_var = y_var * std**2`, then the
square root. Each step is ONE binary32 rounding flushed like `ftz` (the
Python product or sum of two binary32 values was exact in a double before its
one rounding, and the binary32 root is correctly rounded), so `identical_mul`,
the add and `identical_sqrt` (`portable_sqrtf`, correctly rounded) give the
words Python gave. The device kernels (`gaussian_process/estimator.mojo`) and
the host column (`bindings/gp_host_predict.mojo`, `_mojolearn_gp_host.mojo`)
call these bodies.

Switch: `GP_PY2MOJO` (default on; `-D MOJOLEARN_PY2MOJO_cluster_OFF` makes the
bindings report 0 through `gp_py2mojo` and Python un-normalizes as before)."""
from std.sys.compile import is_defined

from checks.numerics import ftz, identical_mul, identical_sqrt

comptime GP_PY2MOJO = not is_defined["MOJOLEARN_PY2MOJO_cluster_OFF"]()

#: modes of `gp_unnorm_cell`
comptime GP_UNNORM_MEAN = 0
comptime GP_UNNORM_COV = 1
comptime GP_UNNORM_STD = 2


@always_inline
def gp_unnorm_cell(v: Float32, s: Float32, mu: Float32, mode: Int) -> Float32:
    """mode 0 (a mean or a draw): ftz(ftz(s * v) + mu); 1 (a covariance
    cell): ftz(v * ftz(s * s)); 2 (a standard deviation from the variance
    v): ftz(sqrt(ftz(v * ftz(s * s))))."""
    if mode == GP_UNNORM_MEAN:
        return ftz(ftz(identical_mul(s, v)) + mu)
    var s2 = ftz(identical_mul(s, s))
    var c = ftz(identical_mul(v, s2))
    if mode == GP_UNNORM_COV:
        return c
    return ftz(identical_sqrt(c))


def gp_unnorm_host(src: List[Float32], n: Int, s: Float32, mu: Float32, mode: Int) -> List[Float32]:
    """The host column's loop over `gp_unnorm_cell`."""
    var out = List[Float32](capacity=n)
    for i in range(n):
        out.append(gp_unnorm_cell(src[i], s, mu, mode))
    return out^


def gpc_ovr_targets(codes: List[Int32], n: Int, k: Int) -> List[Float32]:
    """The binary targets of one-vs-rest class k: 1 where the class code is
    k, 0 elsewhere (lane apple-fast-py2mojo-cluster: `_gpc_impl.py` built
    them in Python, n values per class). The binding reads the int32 codes
    where it read the float32 targets."""
    var y = List[Float32](capacity=n)
    for i in range(n):
        y.append(Float32(1.0) if Int(codes[i]) == k else Float32(0.0))
    return y^


#: `gpc_binary_out` kinds: the binary predict's class codes, predict_proba's pairs
comptime GPC_OUT_CODES = 1
comptime GPC_OUT_PAIRS = 2


def gpc_binary_out(mean_addr: Int, proba_addr: Int, n: Int, kind: Int, out_addr: Int) raises:
    """What `_gpc_impl.py` computed in Python from one binary fit's outputs
    (lane apple-fast-py2mojo-cluster): kind 1, the int64 class codes of
    `predict` (1 where the float32 latent mean is > 0); kind 2, the float64
    `predict_proba` rows `[1 - p, p]` of the class-1 probability p (one
    float64 subtraction, Python's)."""
    if kind == GPC_OUT_CODES:
        var mp = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=mean_addr)
        var op = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=out_addr)
        for t in range(n):
            op.unsafe_store(t, Int64(1) if mp.unsafe_load(t) > Float32(0.0) else Int64(0))
    elif kind == GPC_OUT_PAIRS:
        var pp = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=proba_addr)
        var op = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=out_addr)
        for t in range(n):
            var p = pp.unsafe_load(t)
            op.unsafe_store(2 * t, Float64(1.0) - p)
            op.unsafe_store(2 * t + 1, p)
    else:
        raise Error("gpc_predict: unknown output kind " + String(kind))
