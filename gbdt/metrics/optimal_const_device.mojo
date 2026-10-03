# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`boost_from_average`'s starting constant ON THE DEVICE (cpu-gpu-cleanup
t-gbdt, 2026-10-02).

The doc-parallel and ordered fits downloaded the targets and weights,
looped over every row on the host (`calc_one_dimensional_optimum_const_
approx` over host lists) and went back to the device. They now call
`optimum_const_approx_device`: the targets and weights stay where they are,
every sum over the rows is a grid fold, and one small state word list comes
back at the end. The host finishes with scalar steps only (one division,
the float narrowing, `portable_log64` for the Logloss logit).

THE ORDER IS THE HOST TWIN'S (`gbdt/metrics/sample_quantile.mojo`,
`optimal_const_for_loss.mojo`), statement for statement:

  * every row sum is `bfa_tree_sum`: level 0 folds consecutive chunks of
    `BFA_FOLD_TPB` rows (padded with +0.0) by a halving tree in shared
    memory, the chunk sums are the next level, until one value remains;
  * the arithmetic is `checks/soft_f64.mojo` (IEEE binary64, round to
    nearest even; the Apple GPU has no float64), so each add is the host's
    native double add;
  * the quantile's binary search keeps its range by value, (has_lower, l_q,
    r_q, collected) in a device state list, one fold per step and a
    one-thread scalar step (no loop over rows) between them; min and max
    are `bfa_f32_before`'s total order (-0.0 before +0.0), which a tree
    finds whatever its shape;
  * below `SQ_LINEAR_SEARCH_MAX` rows (a constant bound, under one block)
    the linear search ranks the rows in one block (a stable sort by
    construction), and each thread sums its sorted prefix serially, the
    host's serial accumulation in sorted order.
"""

from std.gpu import block_idx, thread_idx
from std.math import ceildiv, inf
from std.memory import bitcast, stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from checks.numerics import portable_log64
from checks.soft_f64 import (
    SF64_ONE,
    SF64_ZERO,
    sf64_add,
    sf64_div,
    sf64_fma,
    sf64_from_f32,
    sf64_gt,
    sf64_lt,
    sf64_mul,
    sf64_neg,
    sf64_sub,
    sf64_to_f32,
)
from gbdt.metrics.sample_quantile import (
    BFA_FOLD_TPB,
    SAMPLE_QUANTILE_SABOTAGE,
    SQ_BINARY_SEARCH_ITERATIONS,
    SQ_DBL_EPSILON,
    SQ_LINEAR_SEARCH_MAX,
)
from gbdt.metrics.optimal_const_for_loss import QUANTILE_CONST_DELTA
from gbdt.targets.kernel.pointwise_targets import (
    OBJECTIVE_CROSSENTROPY,
    OBJECTIVE_LOGLOSS,
    OBJECTIVE_MAE,
    OBJECTIVE_MAPE,
    OBJECTIVE_QUANTILE,
    OBJECTIVE_RMSE,
)

comptime _TPB = BFA_FOLD_TPB
comptime _U64P = MutPointer[UInt64, MutAnyOrigin]
comptime _F32P = MutPointer[Float32, MutAnyOrigin]
comptime _TWO = UInt64(0x4000000000000000)

# ---- the leaf of a row sum ----------------------------------------------
comptime LEAF_T = 0  # Float64(t)
comptime LEAF_W = 1  # Float64(w) (1.0 unweighted)
comptime LEAF_TW = 2  # Float64(t) * Float64(w), exact
comptime LEAF_LEFT = 3  # w for a row inside the range with t <= q
comptime LEAF_LESS = 4  # w for t < q_result
comptime LEAF_EQUAL = 5  # w for t == q_result

# ---- the state list (binary64 words) ------------------------------------
comptime S_LQ = 0
comptime S_RQ = 1
comptime S_COLLECTED = 2
comptime S_NEED = 3
comptime S_HAS_LOWER = 4
comptime S_TOTAL = 5
comptime S_Q = 6
comptime S_SUM = 7
comptime S_TSUM = 8
comptime S_LESS = 9
comptime S_EQUAL = 10
comptime S_LEN = 11


@always_inline
def _f32_before(a: Float32, b: Float32) -> Bool:
    """`sample_quantile.bfa_f32_before` (the host twin), word for word."""
    if a < b:
        return True
    if b < a:
        return False
    return bitcast[DType.uint32](a) >> 31 == 1 and bitcast[DType.uint32](b) >> 31 == 0


@always_inline
def _le(a: UInt64, b: UInt64) -> Bool:
    """`a <= b` for non-NaN words (+0 == -0): the host's `not (a > b)`."""
    return not sf64_gt(a, b)


def bfa_leaf_kernel[MODE: Int](
    dst: MutPointer[UInt64, MutAnyOrigin],
    t: MutPointer[Float32, MutAnyOrigin],
    w: MutPointer[Float32, MutAnyOrigin],
    has_w: Int32,
    n_in: Int32,
    state: MutPointer[UInt64, MutAnyOrigin],
):
    """Level 0 of a `bfa_tree_sum`: block `b` folds the leaves of rows
    `[b * TPB, (b + 1) * TPB)` into `dst[b]` (rows past the end are +0.0)."""
    var tid = Int(thread_idx.x)
    var blk = Int(block_idx.x)
    var i = blk * _TPB + tid
    var s = stack_allocation[_TPB, Scalar[DType.uint64], address_space = AddressSpace.SHARED]()
    var v = SF64_ZERO
    if i < Int(n_in):
        var wi = SF64_ONE
        if has_w != 0:
            wi = sf64_from_f32(w.unsafe_load(i))
        var x = sf64_from_f32(t.unsafe_load(i))
        comptime if MODE == LEAF_T:
            v = x
        elif MODE == LEAF_W:
            v = wi
        elif MODE == LEAF_TW:
            v = sf64_mul(x, wi)  # two widened float32s: exact
        elif MODE == LEAF_LEFT:
            var lq = state.unsafe_load(S_LQ)
            var rq = state.unsafe_load(S_RQ)
            var q = sf64_div(sf64_add(lq, rq), _TWO)
            var inside = (state.unsafe_load(S_HAS_LOWER) == 0 or sf64_gt(x, lq)) and _le(x, rq)
            if inside and _le(x, q):
                v = wi
        elif MODE == LEAF_LESS:
            if sf64_lt(x, state.unsafe_load(S_Q)):
                v = wi
        else:
            var q = state.unsafe_load(S_Q)
            if not sf64_lt(x, q) and not sf64_gt(x, q):
                v = wi
    s[tid] = v
    barrier()
    var step = _TPB // 2
    while step > 0:
        if tid < step:
            s[tid] = sf64_add(s[tid], s[tid + step])
        barrier()
        step //= 2
    if tid == 0:
        dst.unsafe_store(blk, s[0])


def bfa_fold_kernel(
    dst: MutPointer[UInt64, MutAnyOrigin],
    src: MutPointer[UInt64, MutAnyOrigin],
    m_in: Int32,
):
    """One later level of a `bfa_tree_sum`: block `b` folds
    `src[b * TPB, (b + 1) * TPB)` into `dst[b]`."""
    var tid = Int(thread_idx.x)
    var blk = Int(block_idx.x)
    var i = blk * _TPB + tid
    var s = stack_allocation[_TPB, Scalar[DType.uint64], address_space = AddressSpace.SHARED]()
    var v = SF64_ZERO
    if i < Int(m_in):
        v = src.unsafe_load(i)
    s[tid] = v
    barrier()
    var step = _TPB // 2
    while step > 0:
        if tid < step:
            s[tid] = sf64_add(s[tid], s[tid + step])
        barrier()
        step //= 2
    if tid == 0:
        dst.unsafe_store(blk, s[0])


def bfa_minmax_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    src: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    first: Int32,
):
    """One level of the (min, max) fold in `_f32_before`'s order. Level 0
    (`first`) reads the targets, later levels the previous level's pairs;
    block `b` writes `dst[2b] = min`, `dst[2b + 1] = max`."""
    var tid = Int(thread_idx.x)
    var blk = Int(block_idx.x)
    var i = blk * _TPB + tid
    var smin = stack_allocation[_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var smax = stack_allocation[_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var lo = inf[DType.float32]()
    var hi = -lo
    if i < Int(m_in):
        if first != 0:
            lo = src.unsafe_load(i)
            hi = lo
        else:
            lo = src.unsafe_load(2 * i)
            hi = src.unsafe_load(2 * i + 1)
    smin[tid] = lo
    smax[tid] = hi
    barrier()
    var step = _TPB // 2
    while step > 0:
        if tid < step:
            if _f32_before(smin[tid + step], smin[tid]):
                smin[tid] = smin[tid + step]
            if _f32_before(smax[tid], smax[tid + step]):
                smax[tid] = smax[tid + step]
        barrier()
        step //= 2
    if tid == 0:
        dst.unsafe_store(2 * blk, smin[0])
        dst.unsafe_store(2 * blk + 1, smax[0])


def bfa_init_kernel(
    state: MutPointer[UInt64, MutAnyOrigin],
    minmax: MutPointer[Float32, MutAnyOrigin],
    alpha: UInt64,
    take_min: Int32,
    binary: Int32,
):
    """The scalar set-up of `calc_sample_quantile` (one thread, no rows):
    `need_floor = fma(total, alpha, -eps)`, and the binary search's range
    `l_q = mn - eps`, `r_q = mx`; at `alpha <= 0` (`take_min`) the answer
    is the minimum."""
    if thread_idx.x != 0 or block_idx.x != 0:
        return
    var eps = bitcast[DType.uint64](Float64(SQ_DBL_EPSILON))
    if take_min != 0:
        state.unsafe_store(S_Q, sf64_from_f32(minmax.unsafe_load(0)))
        return
    state.unsafe_store(S_NEED, sf64_fma(state.unsafe_load(S_TOTAL), alpha, sf64_neg(eps)))
    state.unsafe_store(S_COLLECTED, SF64_ZERO)
    state.unsafe_store(S_HAS_LOWER, SF64_ZERO)
    if binary != 0:
        var rq = sf64_from_f32(minmax.unsafe_load(1))
        state.unsafe_store(S_LQ, sf64_sub(sf64_from_f32(minmax.unsafe_load(0)), eps))
        state.unsafe_store(S_RQ, rq)
        state.unsafe_store(S_Q, rq)


def bfa_search_step_kernel(state: MutPointer[UInt64, MutAnyOrigin]):
    """One step of `CalcSampleQuantileBinarySearch` after its left weight
    is folded into `state[S_SUM]` (one thread, scalars only)."""
    if thread_idx.x != 0 or block_idx.x != 0:
        return
    var lq = state.unsafe_load(S_LQ)
    var rq = state.unsafe_load(S_RQ)
    var q = sf64_div(sf64_add(lq, rq), _TWO)
    var collected = state.unsafe_load(S_COLLECTED)
    var lw = state.unsafe_load(S_SUM)
    if sf64_lt(sf64_add(collected, lw), state.unsafe_load(S_NEED)):
        state.unsafe_store(S_LQ, q)
        state.unsafe_store(S_HAS_LOWER, SF64_ONE)
        state.unsafe_store(S_COLLECTED, sf64_add(collected, lw))
    else:
        state.unsafe_store(S_RQ, q)
        state.unsafe_store(S_Q, q)


def bfa_linear_kernel(
    state: MutPointer[UInt64, MutAnyOrigin],
    t: MutPointer[Float32, MutAnyOrigin],
    w: MutPointer[Float32, MutAnyOrigin],
    has_w: Int32,
    n_in: Int32,
):
    """`CalcSampleQuantileLinearSearch` for fewer than
    `SQ_LINEAR_SEARCH_MAX` (< TPB) rows, in one block: thread `i` ranks row
    `i` by the stable `<` order and scatters it; thread `p` sums the sorted
    weights `[0, p]` in order (the host's running `acc`) and offers `p` when
    `acc >= need_floor`; the smallest offer wins (none: the last row)."""
    var tid = Int(thread_idx.x)
    var n = Int(n_in)
    var sv = stack_allocation[_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var sw = stack_allocation[_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var key = stack_allocation[_TPB, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    if tid < n:
        var vi = t.unsafe_load(tid)
        var rank = 0
        for j in range(SQ_LINEAR_SEARCH_MAX):
            if j < n:
                var vj = t.unsafe_load(j)
                if vj < vi or (j < tid and not (vi < vj)):
                    rank += 1
        sv[rank] = vi
        sw[rank] = w.unsafe_load(tid) if has_w != 0 else Float32(1.0)
    barrier()
    var k = Int32(_TPB)
    if tid < n:
        var need = state.unsafe_load(S_NEED)
        var acc = SF64_ZERO
        for p in range(SQ_LINEAR_SEARCH_MAX):
            if p <= tid:
                acc = sf64_add(acc, sf64_from_f32(sw[p]))
        if not sf64_lt(acc, need):
            k = Int32(tid)
    key[tid] = k
    barrier()
    var step = _TPB // 2
    while step > 0:
        if tid < step and key[tid + step] < key[tid]:
            key[tid] = key[tid + step]
        barrier()
        step //= 2
    if tid == 0:
        var p = Int(key[0])
        var r = sv[p] if p < n else sv[n - 1]
        state.unsafe_store(S_Q, sf64_from_f32(r))


def bfa_delta_kernel(
    state: MutPointer[UInt64, MutAnyOrigin], alpha: UInt64, delta: UInt64,
):
    """`CalculateWeightedTargetQuantile`'s delta adjust (one thread,
    scalars): `fma(equal, alpha, less) >= fma(total, alpha, -eps)` moves
    `q` down by `delta`, else up."""
    if thread_idx.x != 0 or block_idx.x != 0:
        return
    var eps = bitcast[DType.uint64](Float64(SQ_DBL_EPSILON))
    var lhs = sf64_fma(state.unsafe_load(S_EQUAL), alpha, state.unsafe_load(S_LESS))
    var rhs = sf64_fma(state.unsafe_load(S_TOTAL), alpha, sf64_neg(eps))
    var q = state.unsafe_load(S_Q)
    if not sf64_lt(lhs, rhs):
        state.unsafe_store(S_Q, sf64_sub(q, delta))
    else:
        state.unsafe_store(S_Q, sf64_add(q, delta))


def mape_weights_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    t: MutPointer[Float32, MutAnyOrigin],
    w: MutPointer[Float32, MutAnyOrigin],
    has_w: Int32,
    n_in: Int32,
):
    """`CalculateOptimalConstApproxForMAPE`'s weights, one thread per row:
    `w / max(1, |t|)` in float32. The quotient is the binary64 quotient
    narrowed once, which IS the correctly rounded float32 quotient (53 >=
    2 * 24 + 2), the host's float `/` on every vendor."""
    var i = Int(block_idx.x) * _TPB + Int(thread_idx.x)
    if i >= Int(n_in):
        return
    var wi = w.unsafe_load(i) if has_w != 0 else Float32(1.0)
    var d = max(Float32(1.0), abs(t.unsafe_load(i)))
    dst.unsafe_store(i, sf64_to_f32(sf64_div(sf64_from_f32(wi), sf64_from_f32(d))))


struct BfaScratch(Movable):
    """The device lists one starting-constant computation needs."""

    var state: DeviceBuffer[DType.uint64]
    var a: DeviceBuffer[DType.uint64]
    var b: DeviceBuffer[DType.uint64]
    var mm_a: DeviceBuffer[DType.float32]
    var mm_b: DeviceBuffer[DType.float32]

    def __init__(out self, ctx: DeviceContext, n_rows: Int) raises:
        var nb = max(1, ceildiv(n_rows, _TPB))
        self.state = ctx.enqueue_create_buffer[DType.uint64](S_LEN)
        self.state.enqueue_fill(SF64_ZERO)
        self.a = ctx.enqueue_create_buffer[DType.uint64](nb)
        self.b = ctx.enqueue_create_buffer[DType.uint64](nb)
        self.mm_a = ctx.enqueue_create_buffer[DType.float32](2 * nb)
        self.mm_b = ctx.enqueue_create_buffer[DType.float32](2 * nb)


def _enqueue_tree_sum[MODE: Int](
    ctx: DeviceContext,
    mut sc: BfaScratch,
    slot: Int,
    t: MutPointer[Float32, MutAnyOrigin],
    w: MutPointer[Float32, MutAnyOrigin],
    has_w: Bool,
    n_rows: Int,
) raises:
    """`bfa_tree_sum` of MODE's leaves into `state[slot]`: level 0, then
    later levels until one chunk remains; the level that produces the one
    value writes the slot."""
    var out = rebind[_U64P](sc.state.unsafe_ptr()) + slot
    var nb = ceildiv(n_rows, _TPB)
    ctx.enqueue_function[bfa_leaf_kernel[MODE]](
        out if nb == 1 else rebind[_U64P](sc.a.unsafe_ptr()), t, w, Int32(1 if has_w else 0),
        Int32(n_rows), rebind[_U64P](sc.state.unsafe_ptr()),
        grid_dim=nb, block_dim=_TPB,
    )
    var m = nb
    var src_is_a = True
    while m > 1:
        var nb2 = ceildiv(m, _TPB)
        var src = rebind[_U64P](sc.a.unsafe_ptr()) if src_is_a else rebind[_U64P](sc.b.unsafe_ptr())
        var nxt = rebind[_U64P](sc.b.unsafe_ptr()) if src_is_a else rebind[_U64P](sc.a.unsafe_ptr())
        ctx.enqueue_function[bfa_fold_kernel](
            out if nb2 == 1 else nxt, src, Int32(m),
            grid_dim=nb2, block_dim=_TPB,
        )
        src_is_a = not src_is_a
        m = nb2


def _enqueue_minmax(
    ctx: DeviceContext, mut sc: BfaScratch,
    t: MutPointer[Float32, MutAnyOrigin], n_rows: Int,
) raises -> MutPointer[Float32, MutAnyOrigin]:
    """The (min, max) fold; returns the list whose first pair is the answer."""
    var nb = ceildiv(n_rows, _TPB)
    ctx.enqueue_function[bfa_minmax_kernel](
        rebind[_F32P](sc.mm_a.unsafe_ptr()), t, Int32(n_rows), Int32(1),
        grid_dim=nb, block_dim=_TPB,
    )
    var m = nb
    var src_is_a = True
    while m > 1:
        var nb2 = ceildiv(m, _TPB)
        var src = rebind[_F32P](sc.mm_a.unsafe_ptr()) if src_is_a else rebind[_F32P](sc.mm_b.unsafe_ptr())
        var nxt = rebind[_F32P](sc.mm_b.unsafe_ptr()) if src_is_a else rebind[_F32P](sc.mm_a.unsafe_ptr())
        ctx.enqueue_function[bfa_minmax_kernel](
            nxt, src, Int32(m), Int32(0), grid_dim=nb2, block_dim=_TPB,
        )
        src_is_a = not src_is_a
        m = nb2
    return rebind[_F32P](sc.mm_a.unsafe_ptr()) if src_is_a else rebind[_F32P](sc.mm_b.unsafe_ptr())


def _enqueue_sample_quantile(
    ctx: DeviceContext,
    mut sc: BfaScratch,
    t: MutPointer[Float32, MutAnyOrigin],
    w: MutPointer[Float32, MutAnyOrigin],
    has_w: Bool,
    n_rows: Int,
    alpha: Float64,
) raises:
    """`calc_sample_quantile` into `state[S_Q]` (and its total into
    `state[S_TOTAL]`), `n_rows >= 1`."""
    var alpha_w = bitcast[DType.uint64](alpha)
    var hw = Int32(1 if has_w else 0)
    if has_w:
        _enqueue_tree_sum[LEAF_W](ctx, sc, S_TOTAL, t, w, has_w, n_rows)
    else:
        # the total is the exact row count
        _store_word(ctx, sc, S_TOTAL, bitcast[DType.uint64](Float64(n_rows)))
    if alpha <= 0:
        var mm = _enqueue_minmax(ctx, sc, t, n_rows)
        ctx.enqueue_function[bfa_init_kernel](
            rebind[_U64P](sc.state.unsafe_ptr()), mm, alpha_w, Int32(1), Int32(0),
            grid_dim=1, block_dim=1,
        )
        return
    if n_rows < SQ_LINEAR_SEARCH_MAX:
        ctx.enqueue_function[bfa_init_kernel](
            rebind[_U64P](sc.state.unsafe_ptr()), rebind[_F32P](sc.mm_a.unsafe_ptr()), alpha_w, Int32(0), Int32(0),
            grid_dim=1, block_dim=1,
        )
        ctx.enqueue_function[bfa_linear_kernel](
            rebind[_U64P](sc.state.unsafe_ptr()), t, w, hw, Int32(n_rows),
            grid_dim=ceildiv(n_rows, _TPB), block_dim=_TPB,
        )
        return
    var mm = _enqueue_minmax(ctx, sc, t, n_rows)
    ctx.enqueue_function[bfa_init_kernel](
        rebind[_U64P](sc.state.unsafe_ptr()), mm, alpha_w, Int32(0), Int32(1),
        grid_dim=1, block_dim=1,
    )
    for _ in range(SQ_BINARY_SEARCH_ITERATIONS):
        _enqueue_tree_sum[LEAF_LEFT](ctx, sc, S_SUM, t, w, has_w, n_rows)
        ctx.enqueue_function[bfa_search_step_kernel](
            rebind[_U64P](sc.state.unsafe_ptr()), grid_dim=1, block_dim=1,
        )


def bfa_store_word_kernel(
    state: MutPointer[UInt64, MutAnyOrigin], slot: Int32, word: UInt64,
):
    """Writes one state word (one thread; a scalar, no rows)."""
    if thread_idx.x == 0 and block_idx.x == 0:
        state.unsafe_store(Int(slot), word)


def _store_word(ctx: DeviceContext, mut sc: BfaScratch, slot: Int, word: UInt64) raises:
    ctx.enqueue_function[bfa_store_word_kernel](
        rebind[_U64P](sc.state.unsafe_ptr()), Int32(slot), word, grid_dim=1, block_dim=1,
    )


def _enqueue_weighted_target_quantile(
    ctx: DeviceContext,
    mut sc: BfaScratch,
    t: MutPointer[Float32, MutAnyOrigin],
    w: MutPointer[Float32, MutAnyOrigin],
    has_w: Bool,
    n_rows: Int,
    alpha: Float64,
    delta: Float64,
) raises:
    """`calculate_weighted_target_quantile` before its float narrowing:
    the sample quantile, then (unless sabotaged) the delta adjust."""
    _enqueue_sample_quantile(ctx, sc, t, w, has_w, n_rows, alpha)
    comptime if SAMPLE_QUANTILE_SABOTAGE:
        return
    if delta > 0:
        _enqueue_tree_sum[LEAF_LESS](ctx, sc, S_LESS, t, w, has_w, n_rows)
        _enqueue_tree_sum[LEAF_EQUAL](ctx, sc, S_EQUAL, t, w, has_w, n_rows)
        ctx.enqueue_function[bfa_delta_kernel](
            rebind[_U64P](sc.state.unsafe_ptr()), bitcast[DType.uint64](alpha),
            bitcast[DType.uint64](delta), grid_dim=1, block_dim=1,
        )


def _read_state(ctx: DeviceContext, sc: BfaScratch) raises -> List[Float64]:
    """The state words, once, at the end (S_LEN scalars, not rows)."""
    var h = ctx.enqueue_create_host_buffer[DType.uint64](S_LEN)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=sc.state)
    ctx.synchronize()
    var out = List[Float64](capacity=S_LEN)
    for k in range(S_LEN):
        out.append(bitcast[DType.float64](h.unsafe_ptr().unsafe_load(k)))
    _ = h^
    return out^


def optimum_const_approx_device(
    ctx: DeviceContext,
    objective: Int,
    targets: DeviceBuffer[DType.float32],
    weights: DeviceBuffer[DType.float32],
    has_weights: Bool,
    n_rows: Int,
    alpha: Float64 = 0.5,
) raises -> Float64:
    """`calc_one_dimensional_optimum_const_approx` over device targets and
    weights: the same arms, refusals and bits (see the module docstring).
    `weights` is read only when `has_weights`."""
    if n_rows <= 0:
        raise Error("optimal const approx: empty target")
    if (
        objective != OBJECTIVE_RMSE and objective != OBJECTIVE_LOGLOSS
        and objective != OBJECTIVE_CROSSENTROPY and objective != OBJECTIVE_QUANTILE
        and objective != OBJECTIVE_MAE and objective != OBJECTIVE_MAPE
    ):
        raise Error(
            "boost_from_average is not implemented for this loss yet: RMSE,"
            " Logloss, CrossEntropy, Quantile, MAE and MAPE have"
            " CalcOptimumConstApprox arms here; the rest are refused by name"
            " rather than approximated."
        )
    var sc = BfaScratch(ctx, n_rows)
    var t = rebind[_F32P](targets.unsafe_ptr())
    var w = rebind[_F32P](weights.unsafe_ptr())
    if (
        objective == OBJECTIVE_RMSE or objective == OBJECTIVE_LOGLOSS
        or objective == OBJECTIVE_CROSSENTROPY
    ):
        # `calculate_weighted_target_average`
        if has_weights:
            _enqueue_tree_sum[LEAF_W](ctx, sc, S_TOTAL, t, w, True, n_rows)
            _enqueue_tree_sum[LEAF_TW](ctx, sc, S_TSUM, t, w, True, n_rows)
        else:
            _enqueue_tree_sum[LEAF_T](ctx, sc, S_TSUM, t, w, False, n_rows)
        var st = _read_state(ctx, sc)
        var summary_weight = st[S_TOTAL] if has_weights else Float64(n_rows)
        # their `return targetSum / summaryWeight;` through `inline float`
        var avg = Float64(Float32(st[S_TSUM] / summary_weight))
        if objective == OBJECTIVE_RMSE:
            return avg
        if avg <= 0.0 or avg >= 1.0:
            raise Error(
                "boost_from_average: the weighted mean target is "
                + String(avg)
                + ", outside (0, 1); a one-class pool has no finite"
                " log-odds"
            )
        # their `Logit`, `-log(1 / x - 1)` (`optimal_const_for_loss.mojo`)
        return -portable_log64(1.0 / avg - 1.0)
    if objective == OBJECTIVE_MAPE:
        # `calculate_optimal_const_approx_for_mape`: the sample median over
        # `w / max(1, |t|)`, no delta adjust
        var wm = ctx.enqueue_create_buffer[DType.float32](n_rows)
        ctx.enqueue_function[mape_weights_kernel](
            rebind[_F32P](wm.unsafe_ptr()), t, w, Int32(1 if has_weights else 0), Int32(n_rows),
            grid_dim=ceildiv(n_rows, _TPB), block_dim=_TPB,
        )
        comptime if SAMPLE_QUANTILE_SABOTAGE:
            _enqueue_sample_quantile(ctx, sc, t, rebind[_F32P](wm.unsafe_ptr()), True, n_rows, 0.75)
        else:
            _enqueue_sample_quantile(ctx, sc, t, rebind[_F32P](wm.unsafe_ptr()), True, n_rows, 0.5)
        var st = _read_state(ctx, sc)
        _ = wm^
        return Float64(Float32(st[S_Q]))
    # Quantile / MAE: their `inline float` return, widened back
    _enqueue_weighted_target_quantile(
        ctx, sc, t, w, has_weights, n_rows,
        0.5 if objective == OBJECTIVE_MAE else alpha, QUANTILE_CONST_DELTA,
    )
    var st = _read_state(ctx, sc)
    return Float64(Float32(st[S_Q]))
