# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE CLUSTER LANE'S n-SIZED POST-PROCESSING BODIES (lane cgr2-cluster,
2026-10-03): one source for the device kernels (`x_cluster/device_ops.mojo`)
and the host column's loops (`x_cluster/host/host_ops.mojo`).

THE FLOAT-FLOAT FOLD. A sum over n values that used to be one ascending
Float64 chain on the host (the MeanShift bandwidth mean, the mixture lower
bound's entropy and mean log-likelihood, the k-means++ potentials and their
cumulative table) is a FIXED-ORDER BLOCKED-THEN-TREE fold in float-float
(`x_linear/ff.mojo`, about 48 bits, every operation the IDENTICAL float32
arithmetic): chunks of FOLD_CHUNK values; in a chunk, lane q of FOLD_LANES
sums the values q, q + FOLD_LANES, ... ascending, then the lanes fold as a
binary tree (lane q takes lane q + off, off = FOLD_LANES / 2 .. 1); the chunk
totals fold the same way again until one value is left. The device runs a
chunk as one block; the host column walks the same lanes and the same tree,
so the words are the same on every vendor and on the host."""
from std.memory import bitcast

from checks.numerics import ftz, identical_sqrt
from x_cluster.bodies import FPtr, IPtr, splitmix_at
from x_linear.ff import FF, ff_add, ff_div, ff_f32, two_prod, two_sum

comptime FOLD_CHUNK = 1024
comptime FOLD_LANES = 256

#: the element of a fold: a[t]
comptime FM_VAL = 0
#: sqrt(a[t])
comptime FM_SQRT = 1
#: a[t] * b[t], exact (two-product)
comptime FM_PROD = 2
#: min(a[t], b[t]) (`a if a < b else b`)
comptime FM_MIN = 3
#: c[t] * min(a[t], b[t]), exact
comptime FM_WMIN = 4
#: the float-float value (a[t], b[t]) (a fold of chunk totals)
comptime FM_FF = 5


@always_inline
def _min_sel(v: Float32, c: Float32) -> Float32:
    return v if v < c else c


@always_inline
def ff_elem(mode: Int, a: FPtr, b: FPtr, c: FPtr, t: Int) -> FF:
    if mode == FM_VAL:
        return FF(ftz(a[t]), Float32(0))
    if mode == FM_SQRT:
        return FF(identical_sqrt(a[t]), Float32(0))
    if mode == FM_PROD:
        return two_prod(a[t], b[t])
    if mode == FM_MIN:
        return FF(ftz(_min_sel(a[t], b[t])), Float32(0))
    if mode == FM_WMIN:
        return two_prod(c[t], _min_sel(a[t], b[t]))
    return FF(a[t], b[t])


@always_inline
def ff_lane(mode: Int, a: FPtr, b: FPtr, c: FPtr, base: Int, end: Int, lane: Int) -> FF:
    """Lane `lane` of the chunk [base, end): its values ascending."""
    var acc = FF(Float32(0), Float32(0))
    var t = base + lane
    while t < end:
        acc = ff_add(acc, ff_elem(mode, a, b, c, t))
        t += FOLD_LANES
    return acc


def ff_chunk_host(mode: Int, a: FPtr, b: FPtr, c: FPtr, base: Int, end: Int) -> FF:
    """The device block's chunk total, walked on the host: the same lanes,
    the same tree."""
    var hi = List[Float32](length=FOLD_LANES, fill=Float32(0))
    var lo = List[Float32](length=FOLD_LANES, fill=Float32(0))
    for q in range(FOLD_LANES):
        var v = ff_lane(mode, a, b, c, base, end, q)
        hi[q] = v.hi
        lo[q] = v.lo
    var off = FOLD_LANES // 2
    while off > 0:
        for q in range(off):
            var v = ff_add(FF(hi[q], lo[q]), FF(hi[q + off], lo[q + off]))
            hi[q] = v.hi
            lo[q] = v.lo
        off //= 2
    return FF(hi[0], lo[0])


def ff_fold_host(mode: Int, a: FPtr, b: FPtr, c: FPtr, n: Int) -> FF:
    """The whole fold of n elements on the host (chunks, then the chunk
    totals again until one is left)."""
    if n <= 0:
        return FF(Float32(0), Float32(0))
    var nch = (n + FOLD_CHUNK - 1) // FOLD_CHUNK
    var th = List[Float32](length=nch, fill=Float32(0))
    var tl = List[Float32](length=nch, fill=Float32(0))
    for q in range(nch):
        var v = ff_chunk_host(mode, a, b, c, q * FOLD_CHUNK, min(n, (q + 1) * FOLD_CHUNK))
        th[q] = v.hi
        tl[q] = v.lo
    while nch > 1:
        var m = (nch + FOLD_CHUNK - 1) // FOLD_CHUNK
        var nh = List[Float32](length=m, fill=Float32(0))
        var nl = List[Float32](length=m, fill=Float32(0))
        var ph = th.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var pl = tl.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        for q in range(m):
            var v = ff_chunk_host(FM_FF, ph, pl, ph, q * FOLD_CHUNK, min(nch, (q + 1) * FOLD_CHUNK))
            nh[q] = v.hi
            nl[q] = v.lo
        th = nh^
        tl = nl^
        nch = m
    return FF(th[0], tl[0])


@always_inline
def ff_ge(x: FF, y: FF) -> Bool:
    """x >= y, (hi, lo) lexicographic: the one comparison both columns use."""
    if x.hi != y.hi:
        return x.hi > y.hi
    return x.lo >= y.lo


def ff_to_f64(x: FF) -> Float64:
    """hi + lo, exact in a double (48 significant bits)."""
    return Float64(x.hi) + Float64(x.lo)


def ff_of_f64(v: Float64) -> FF:
    """A double as float-float (host scalar): the nearest float, then the
    nearest float of the rest."""
    var hi = Float32(v)
    return FF(hi, Float32(v - Float64(hi)))


@always_inline
def kpp_search_cell(
    mode: Int, a: FPtr, b: FPtr, th: FPtr, tl: FPtr, n: Int, vh: Float32, vl: Float32
) -> Int:
    """searchsorted(cum, v, side='left') clipped to n - 1, where cum is the
    fold's running table: the first chunk whose running total (the chunk
    totals summed ascending) reaches v, then that chunk's values summed
    ascending from the total before it; the chunk's last index when the walk
    falls short by rounding. `mode` FM_VAL (a = closest) or FM_PROD
    (a = closest, b = weights)."""
    var v = FF(vh, vl)
    var nch = (n + FOLD_CHUNK - 1) // FOLD_CHUNK
    var acc = FF(Float32(0), Float32(0))
    var ch = -1
    for q in range(nch):
        var nxt = ff_add(acc, FF(th[q], tl[q]))
        if ff_ge(nxt, v):
            ch = q
            break
        acc = nxt
    if ch < 0:
        return n - 1
    var end = min(n, (ch + 1) * FOLD_CHUNK)
    for t in range(ch * FOLD_CHUNK, end):
        acc = ff_add(acc, ff_elem(mode, a, b, a, t))
        if ff_ge(acc, v):
            return t
    return end - 1


@always_inline
def order_key(v: Float32, k: Int) -> UInt64:
    """Ascending in v (-0.0 folded onto +0.0, which `<` treats as equal),
    then ascending in k: the lowest value, the lowest index on a tie."""
    var bits = bitcast[DType.uint32](v)
    if bits == UInt32(0x80000000):
        bits = UInt32(0)
    var o = (bits ^ UInt32(0x80000000)) if (bits & UInt32(0x80000000)) == UInt32(0) else ~bits
    return (UInt64(o) << 32) | UInt64(UInt32(k))


comptime KEY_NONE = UInt64(0xFFFFFFFFFFFFFFFF)


# ---------------------------------------------------------------- OPTICS
@always_inline
def optics_relax_cell(
    dist: FPtr, n: Int, point: Int, cp: Float32, max_eps: Float32, processed: IPtr, reach: FPtr,
    pred: IPtr, o: Int,
):
    """sklearn `_set_reach_dist` for row o against the point just ordered:
    an unprocessed row within `max_eps` takes `max(dist, core)` when that is
    strictly lower."""
    if o == point or processed[o] != 0:
        return
    var dd = dist[point * n + o]
    if not (dd <= max_eps):
        return
    var rd = dd if dd > cp else cp
    if rd < reach[o]:
        reach[o] = rd
        pred[o] = Int32(point)


# ---------------------------------------------------------------- MeanShift
@always_inline
def rows_equal(a: FPtr, i: Int, j: Int, d: Int) -> Bool:
    """Every coordinate equal under `!=` (a -0.0 equals a +0.0)."""
    for f in range(d):
        if a[i * d + f] != a[j * d + f]:
            return False
    return True


@always_inline
def first_equal_cell(a: FPtr, mask: IPtr, use_mask: Bool, d: Int, s: Int) -> Int32:
    """The lowest s' <= s (masked in) whose row equals row s; -1 when s is
    masked out. The dict's first-insertion key of row s."""
    if use_mask and mask[s] == 0:
        return Int32(-1)
    for q in range(s):
        if use_mask and mask[q] == 0:
            continue
        if rows_equal(a, q, s, d):
            return Int32(q)
    return Int32(s)


@always_inline
def center_greater(c: FPtr, rep_val: IPtr, u: Int, v: Int, d: Int) -> Bool:
    """(intensity, coordinates) of distinct center u > that of v,
    lexicographic (`rep_val` holds each representative's last intensity)."""
    var iu = rep_val[u]
    var iv = rep_val[v]
    if iu != iv:
        return iu > iv
    for f in range(d):
        var x = c[u * d + f]
        var y = c[v * d + f]
        if x != y:
            return x > y
    return False


# ---------------------------------------------------------------- mixtures
@always_inline
def unit_ff(z: UInt64) -> FF:
    """`SplitMix64.unit()` of the draw z as float-float without a Float64:
    the 53-bit integer z >> 11 split into its top 24 bits (exact) and the
    29 below (rounded to nearest), each scaled by an exact power of two."""
    var m = z >> 11
    var h = Float32(UInt32(m >> 29)) * Float32(5.9604644775390625e-08)  # 2^-24
    var l = Float32(UInt32(m & UInt64(0x1FFFFFFF))) * Float32(1.1102230246251565e-16)  # 2^-53
    return two_sum(h, l)


@always_inline
def rand_resp_row(state: UInt64, kc: Int, dst: FPtr, i: Int):
    """Row i of the 'random' start: kc unit draws (draw i * kc + k + 1 of
    the stream whose state is `state`), each divided by the row's sum, the
    sum and quotient in float-float."""
    var s = FF(Float32(0), Float32(0))
    for k in range(kc):
        s = ff_add(s, unit_ff(splitmix_at(state, UInt64(i * kc + k + 1))))
    for k in range(kc):
        var u = unit_ff(splitmix_at(state, UInt64(i * kc + k + 1)))
        dst[i * kc + k] = ff_f32(ff_div(u, s))
