# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""OneClassSVM's start (libsvm solve_one_class) as item functions: no host
reduction and no host per-sample fill before the device solver
(lane/fam2-neighbors, cpu-gpu audit 2026-10-04 case 2).

libsvm: nu_l = nu * sum(C_i) (C_i = 1 unweighted), then in sample order
alpha_i = min(C_i, nu_l) while nu_l > 0, nu_l reduced by each. In closed
form alpha_i = clamp(nu_l - P_i, 0, C_i) with P_i the sum of the C before i,
which is a prefix sum, so every sample is its own item:

  stage 0, one item per chunk c of OCI_CHUNK samples: the chunk's sum,
           ascending, float-float;
  stage 1, one item per chunk c (and one more, c == nc): the sum of the
           chunk sums before c, ascending, float-float; item nc stores
           nu_l = total * nu instead;
  stage 2, one item per sample i: the chunk offset plus the C of its chunk
           before i, ascending, float-float; alpha_i from nu_l - P_i.

Float-float (x_linear/ff.mojo, two float32, about 48 bits) stands in for the
binary64 the host helper used: there is no float64 on Metal. The device
kernels (x_neighbors/ocsvm_dev.mojo) and the host column
(x_neighbors/ocsvm_host.mojo) call these same items.

Nothing here imports a GPU module, so the CPU-only host binding compiles it.
"""
from std.sys.compile import is_defined
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from x_neighbors.items import FP
from x_linear.ff import FF, ff_of, ff_add, ff_add_f, ff_sub, ff_mul, ff_div, ff_f32
from x_neighbors.items import IP

#: lane/fam2-neighbors (2026-10-04): the alpha start by the three stages
#: above on the device (and the same items on the host column), not by the
#: base binding's host loop `ocsvm_alpha_init_f32`. Bits: alpha's start can
#: move in its last float32 bit at the one fractional sample (float-float vs
#: binary64), on all four columns together.
#: lane cpu2-l9-neighbors (2026-10-04): every mode (FAST too) and no _OFF
#: arm: the host loop is not a GPU route (owner rule, fixes not
#: optimizations), so `MOJOLEARN_IDN_OCSVM_DEV_INIT_OFF` is retired and the
#: Python glue has no host fallback.
comptime XN_OCSVM_DEV_INIT = True

#: samples per chunk (stage 2 walks at most OCI_CHUNK - 1 of them per item)
comptime OCI_CHUNK = 256


@always_inline
def oci_chunks(n: Int) -> Int:
    return (n + OCI_CHUNK - 1) // OCI_CHUNK if n > 0 else 0


@always_inline
def oci_part_item(c: Int, cv: FP, ph: FP, pl: FP, n: Int):
    """Stage 0: chunk c's sum of C, ascending, to (ph[c], pl[c])."""
    var lo = c * OCI_CHUNK
    var hi = min(lo + OCI_CHUNK, n)
    var s = ff_of(Float32(0))
    for i in range(lo, hi):
        s = ff_add_f(s, cv.unsafe_load(i))
    ph.unsafe_store(c, s.hi)
    pl.unsafe_store(c, s.lo)


@always_inline
def oci_scan_item(c: Int, ph: FP, pl: FP, oh: FP, ol: FP, n: Int, nu_hi: Float32, nu_lo: Float32):
    """Stage 1: the sum of the chunk sums before c, ascending, to
    (oh[c], ol[c]); item c == oci_chunks(n) stores nu_l = total * nu."""
    var nc = oci_chunks(n)
    var s = ff_of(Float32(0))
    for b in range(c):
        s = ff_add(s, FF(ph.unsafe_load(b), pl.unsafe_load(b)))
    if c == nc:
        s = ff_mul(s, FF(nu_hi, nu_lo))
    oh.unsafe_store(c, s.hi)
    ol.unsafe_store(c, s.lo)


@always_inline
def oci_alpha_item(i: Int, cv: FP, oh: FP, ol: FP, alpha: FP, n: Int):
    """Stage 2: alpha_i = clamp(nu_l - P_i, 0, C_i)."""
    var nc = oci_chunks(n)
    var c = i // OCI_CHUNK
    var p = FF(oh.unsafe_load(c), ol.unsafe_load(c))
    for j in range(c * OCI_CHUNK, i):
        p = ff_add_f(p, cv.unsafe_load(j))
    var rem = ff_sub(FF(oh.unsafe_load(nc), ol.unsafe_load(nc)), p)
    var w = cv.unsafe_load(i)
    var a = Float32(0)
    if rem.hi > Float32(0):
        a = rem.hi
        if a > w:
            a = w
    alpha.unsafe_store(i, a)


@always_inline
def oci_nu_hi(nu: Float64) -> Float32:
    """The float-float split of the binary64 hyperparameter nu (a scalar of
    the call, not data): the high word."""
    return Float32(nu)


@always_inline
def oci_nu_lo(nu: Float64) -> Float32:
    return Float32(nu - Float64(Float32(nu)))


#: lane/fam2-neighbors (2026-10-04), IDENTICAL, default ON: PageRank's
#: caller vectors (personalization, nstart, dangling) divided by their sum on
#: the device: the sum is stages 0 and 1 above (nu = 1), then one item per
#: element. Before, python/mojolearn/_expansion_neighbors.py `PageRank._unit`
#: walked the n values in Python (fsum, the sign test, n divisions).
#: Bits: an element can move in its last float32 bit (float-float quotient
#: vs the binary64 one), device and host column together.
#: Lane py-runtime-b: FAST registers it too (the Python walk is deleted:
#: no Python compute in the runtime). In IDENTICAL, -D
#: MOJOLEARN_IDN_XN_UNIT_DEV_OFF (or MOJOLEARN_IDN_ALL_OFF) leaves the binding
#: function unregistered and PageRank refuses a caller vector.
comptime XN_UNIT_DEV = GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL or not (
    is_defined["MOJOLEARN_IDN_XN_UNIT_DEV_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)


@always_inline
def unit_ff_item(i: Int, v: FP, oh: FP, ol: FP, res: FP, info: IP, n: Int):
    """res[i] = v[i] / total (the total is the stage-1 slot oci_chunks(n)).
    info[0] is set when any v is negative (every writer stores the same 1);
    item 0 sets info[1] when the total is zero. The caller zeroes info."""
    var nc = oci_chunks(n)
    var tot = FF(oh.unsafe_load(nc), ol.unsafe_load(nc))
    var w = v.unsafe_load(i)
    if w < Float32(0):
        info.unsafe_store(0, Int32(1))
    if i == 0 and tot.hi == Float32(0):
        info.unsafe_store(1, Int32(1))
    res.unsafe_store(i, ff_f32(ff_div(ff_of(w), tot)))
