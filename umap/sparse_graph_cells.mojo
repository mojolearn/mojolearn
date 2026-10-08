# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The CPU-safe half of `umap/sparse_graph.mojo` (lane rehearsal-suite-green,
2026-10-08): the CSR graph struct and the per-row and per-cell scalar
arithmetic the device kernels and the CPU column's builder
(`umap/host/sparse_graph_host.mojo`) share, moved out verbatim so the host
oracle imports no `std.gpu` / `max.gpu` module. `umap/sparse_graph.mojo`
imports every name back, so its importers are unchanged.
"""
from std.memory import bitcast

from checks.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    ftz,
    identical_div,
    identical_exp64,
    identical_log2_64,
    identical_mul,
    identical_mul_add,
    identical_pow64,
)
from checks.soft_f64 import (
    SF64_NAN,
    SF64_ONE,
    SF64_ZERO,
    sf64_add,
    sf64_div,
    sf64_exp,
    sf64_fma,
    sf64_from_f32,
    sf64_from_int,
    sf64_gt,
    sf64_is_nan,
    sf64_lt,
    sf64_mul,
    sf64_neg,
    sf64_pow,
    sf64_sub,
    sf64_to_f32,
    sf64_to_int,
)


comptime UG_F32P = MutPointer[Float32, MutAnyOrigin]
comptime UG_U32P = MutPointer[UInt32, MutAnyOrigin]
comptime UG_I32P = MutPointer[Int32, MutAnyOrigin]
comptime UG_TPB = 256
comptime UG_TWO = UInt64(0x4000000000000000)
comptime UG_HALF = UInt64(0x3FE0000000000000)
comptime UG_NO_ERROR = Int32(0x7FFFFFFF)


struct SparseFuzzySimplicialGraph(Copyable, Movable):
    var n_samples: Int
    var n_neighbors: Int
    var rhos: List[Float32]
    var sigmas: List[Float32]
    var directed_offsets: List[Int]
    var directed_indices: List[UInt32]
    var directed_values: List[Float32]
    var offsets: List[Int]
    var indices: List[UInt32]
    var values: List[Float32]

    def __init__(
        out self, n_samples: Int, n_neighbors: Int,
        var rhos: List[Float32], var sigmas: List[Float32],
        var directed_offsets: List[Int], var directed_indices: List[UInt32],
        var directed_values: List[Float32], var offsets: List[Int],
        var indices: List[UInt32], var values: List[Float32],
    ):
        self.n_samples = n_samples
        self.n_neighbors = n_neighbors
        self.rhos = rhos^
        self.sigmas = sigmas^
        self.directed_offsets = directed_offsets^
        self.directed_indices = directed_indices^
        self.directed_values = directed_values^
        self.offsets = offsets^
        self.indices = indices^
        self.values = values^

    def logical_payload_bytes(self) -> Int:
        """Occupied scalar bytes on the supported 64-bit hosts.

        Excludes caller inputs, temporary transpose/cursors, List spare
        capacity and allocator overhead; not a resident-memory measurement.
        """
        return (
            4 * (len(self.rhos) + len(self.sigmas))
            + 8 * (len(self.directed_offsets) + len(self.offsets))
            + 4 * (len(self.directed_indices) + len(self.directed_values))
            + 4 * (len(self.indices) + len(self.values))
        )



# ---------------------------------------------------------------------------
# The per-row statements both columns run.
# ---------------------------------------------------------------------------


def ug_constants(n_neighbors: Int) -> Tuple[UInt64, UInt64, UInt64]:
    """HOST: the sigma search's target `log2(k)` (the host seam, one scalar),
    umap-learn's SMOOTH_K_TOLERANCE 1e-5 and the bracket cap 1e20, as
    binary64 words for the soft arithmetic."""
    return (
        bitcast[DType.uint64](identical_log2_64(Float64(n_neighbors))),
        bitcast[DType.uint64](Float64(1.0e-5)),
        bitcast[DType.uint64](Float64(1.0e20)),
    )


@always_inline
def _ug_nz_kern(dp: UG_F32P, base: Int, k: Int, which: Int) -> Float32:
    """The `which`-th (0-based) positive distance of the row, in rank order."""
    var seen = 0
    for j in range(k):
        var d = dp[base + j]
        if d > Float32(0.0):
            if seen == which:
                return d
            seen += 1
    return Float32(0.0)


def ug_row_rho_kern(dp: UG_F32P, row: Int, k: Int, lc: Float32, tol: UInt64) -> Float32:
    """rho: the first positive distance; at a local_connectivity other than
    1, DEVIATION 5323 (PIN): the positive distances in rank order, index =
    floor(lc), rho = nz[index - 1] + interp (nz[index] - nz[index - 1]) as
    ONE binary64 fma rounded once to Float32 when interp > 1e-5, interp *
    nz[0] when index is 0, the largest positive distance when the row has
    fewer than lc of them, 0 when it has none."""
    var base = row * k
    var cnt = 0
    var first = Float32(0.0)
    for j in range(k):
        var d = dp[base + j]
        if d > Float32(0.0):
            if cnt == 0:
                first = d
            cnt += 1
    if lc == Float32(1.0):
        return first
    var lc64 = sf64_from_f32(lc)
    if not sf64_lt(sf64_from_int(cnt), lc64):
        var index = sf64_to_int(lc64)  # floor: lc >= 0
        var interp = sf64_sub(lc64, sf64_from_int(index))
        if index > 0:
            var rho = _ug_nz_kern(dp, base, k, index - 1)
            if sf64_gt(interp, tol):
                var diff = _ug_nz_kern(dp, base, k, index) - rho
                rho = sf64_to_f32(
                    sf64_fma(interp, sf64_from_f32(diff), sf64_from_f32(rho))
                )
            return rho
        if cnt > 0:
            return sf64_to_f32(sf64_mul(interp, sf64_from_f32(first)))
        return Float32(0.0)
    if cnt > 0:
        return _ug_nz_kern(dp, base, k, cnt - 1)
    return Float32(0.0)


@always_inline
def _ug_msum_kern(dp: UG_F32P, base: Int, k: Int, rho: UInt64, sigma: UInt64) -> UInt64:
    """`sum_{j >= 1} (1 if d_j - rho <= 0 else exp(-(d_j - rho) / sigma))`,
    binary64, ascending j."""
    var total = SF64_ZERO
    for j in range(1, k):
        var d = sf64_sub(sf64_from_f32(dp[base + j]), rho)
        if sf64_gt(d, SF64_ZERO):
            total = sf64_add(total, sf64_exp(sf64_div(sf64_neg(d), sigma)))
        else:
            total = sf64_add(total, SF64_ONE)
    return total


def ug_row_sigma(
    dp: UG_F32P, row: Int, k: Int, rho_f32: Float32,
    target: UInt64, tol: UInt64, big: UInt64,
) -> UInt64:
    """The row's sigma as a binary64 word, or `SF64_NAN` when doubling from
    1 passes 1e20 without reaching `target` (the host's bracket refusal).
    IDENTICAL: 64 bisection steps, fixed. FAST: the same steps with the
    `|value - target| <= 1e-5` early exit."""
    var base = row * k
    var rho = sf64_from_f32(rho_f32)
    var hi = SF64_ONE
    while sf64_lt(_ug_msum_kern(dp, base, k, rho, hi), target):
        hi = sf64_mul(hi, UG_TWO)
        if sf64_gt(hi, big):
            return SF64_NAN
    var lo = SF64_ZERO
    var sigma = hi
    var step = 0
    while step < 64:
        sigma = sf64_mul(sf64_add(lo, hi), UG_HALF)
        var value = _ug_msum_kern(dp, base, k, rho, sigma)
        comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
            var err = sf64_sub(value, target)
            if sf64_lt(err, SF64_ZERO):
                err = sf64_neg(err)
            if not sf64_gt(err, tol):
                return sigma
        if sf64_gt(value, target):
            hi = sigma
        else:
            lo = sigma
        step += 1
    return sigma


@always_inline
def ug_member(delta: Float32, sigma: Float32) -> Float32:
    """One directed membership: 1 at or below rho, else
    `Float32(exp(-delta / sigma))` in binary64, FLUSHED: a membership that
    rounds to a subnormal is 0 on every column. Unflushed, the Apple GPU's
    merge (`a + b`) and the adapter's `weight > 0` read it as zero where every
    other column kept the edge, which shifted every later edge's ordinal and
    so its negative samples (umap / x-decomp-umap-options / par-graph-umap on
    `ties`, the 0.8.36 reference recording: tied distances drive sigma to
    its floor, and the far memberships underflow)."""
    var d = ftz(delta)
    if d > Float32(0.0):
        return ftz(sf64_to_f32(
            sf64_exp(sf64_div(sf64_neg(sf64_from_f32(d)), sf64_from_f32(sigma)))
        ))
    return Float32(1.0)


@always_inline
def ug_merge_weight(a: Float32, b: Float32, mix: Float32) -> Float32:
    """`mix * (a + b - a b) + (1 - mix) * a b` with every product pinned and
    ONE rounding on the intersection's product (the fma). Operands, the
    product and the result are flushed, so no subnormal reaches an Apple
    add or compare (`ug_member`'s note)."""
    var fa = ftz(a)
    var fb = ftz(b)
    var intersection = ftz(identical_mul(fa, fb))
    var union = ftz((fa + fb) - intersection)
    return ftz(identical_mul_add(
        Float32(1.0) - mix, intersection, ftz(identical_mul(mix, union))
    ))


# ---------------------------------------------------------------------------
# The device build.


comptime UG_FLT_MAX_BITS = Int32(0x7F7FFFFF)
comptime UG_NO_NEG = Int32(-2147483648)


@always_inline
def ug_reset_scale(v: Float32, mx: Float32) -> Float32:
    """sklearn `normalize(norm='max')`'s cell: one float32 division by the
    row's largest stored value (the value itself in a row with none above 0)."""
    if mx > Float32(0.0):
        return ftz(identical_div(v, mx))
    return ftz(v)


@always_inline
def ug_reset_weight(a: Float32, b: Float32) -> Float32:
    """`S + S^T - S o S^T` per cell as `(a + b) - a b`, the product pinned."""
    var fa = ftz(a)
    var fb = ftz(b)
    return ftz(ftz(fa + fb) - ftz(identical_mul(fa, fb)))


@always_inline
def ug_categorical_weight(w: Float32, ti: Float32, tj: Float32, unknown: UInt64, far: UInt64) -> Float32:
    """An edge whose ends carry different labels scaled by exp(-far_dist),
    one with an unknown label (-1) by exp(-unknown_dist), each as
    `Float32(Float64(w) * exp)`; flushed."""
    if ti == Float32(-1.0) or tj == Float32(-1.0):
        return ftz(sf64_to_f32(sf64_mul(sf64_from_f32(w), unknown)))
    if ti != tj:
        return ftz(sf64_to_f32(sf64_mul(sf64_from_f32(w), far)))
    return ftz(w)


@always_inline
def ug_intersect_cell(
    lv_in: Float32, rv_in: Float32, left_min: UInt64, right_min: UInt64, low: Bool, expo: UInt64
) -> Float32:
    """One union cell of `general_sset_intersection`: `left + right`, or,
    when either side beats its floor (an absent or zero side reads as its
    floor), `left * right^expo` (`low`, weight < 0.5) or `left^expo * right`
    in binary64 with the portable pow, rounded once; flushed."""
    var lv = ftz(lv_in)
    var rv = ftz(rv_in)
    var out = ftz(lv + rv)
    var left_val = sf64_from_f32(lv) if lv != Float32(0.0) else left_min
    var right_val = sf64_from_f32(rv) if rv != Float32(0.0) else right_min
    if sf64_gt(left_val, left_min) or sf64_gt(right_val, right_min):
        if low:
            out = ftz(sf64_to_f32(sf64_mul(left_val, sf64_pow(right_val, expo))))
        else:
            out = ftz(sf64_to_f32(sf64_mul(sf64_pow(left_val, expo), right_val)))
    return out


def ug_categorical_constants(far_dist: Float64, unknown_dist: Float64) -> Tuple[UInt64, UInt64]:
    """HOST scalars both columns call: exp(-unknown_dist), exp(-far_dist)."""
    return (
        bitcast[DType.uint64](identical_exp64(-unknown_dist)),
        bitcast[DType.uint64](identical_exp64(-far_dist)),
    )


def ug_intersect_floor(min_stored: Float32) -> UInt64:
    """HOST scalar: half a graph's smallest stored value, at least 1e-8."""
    return bitcast[DType.uint64](max(Float64(min_stored) / 2.0, Float64(1.0e-8)))


def ug_intersect_expo(weight: Float32) -> Tuple[Bool, UInt64]:
    """HOST scalar: (weight < 0.5, w / (1 - w) or (1 - w) / w) in Float64."""
    var w64 = Float64(weight)
    if w64 < 0.5:
        return (True, bitcast[DType.uint64](w64 / (1.0 - w64)))
    return (False, bitcast[DType.uint64]((1.0 - w64) / w64))


def ug_min_stored(pos_bits: Int32, neg_bits: Int32) -> Float32:
    """HOST: a graph's smallest nonzero stored value from the device's two
    words (the most negative value's bits by `Atomic.max`, else the smallest
    positive one's by `Atomic.min` from FLT_MAX), as the host scan's
    `v != 0 and v < m` from m = FLT_MAX finds it."""
    if neg_bits != UG_NO_NEG:
        return bitcast[DType.float32](neg_bits)
    return bitcast[DType.float32](pos_bits)
