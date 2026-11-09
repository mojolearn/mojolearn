# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The overfitting detector on the device (lane t1-tree-builder-ops,
2026-11-08; docs/plans/EXECUTOR_T1_TREE_OPS.md, tree op TREE_OP_OD_STEP).

`OverfittingDetector.add_error` + `is_need_stop`
(gbdt/overfitting_detector/overfitting_detector.mojo) as the boosting loop's
criterion. The state is binary64 bit patterns (the Apple GPU has no float64;
checks/soft_f64 is correctly rounded, so the words are the host detector's).
The host twin is `OverfittingDetector` itself: the same operations in the same
order (its `exp` is `sf64_exp`).
"""
from std.gpu import block_idx, thread_idx
from std.memory import bitcast, stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from checks.soft_f64 import (
    SF64_ONE, SF64_ZERO, sf64_div, sf64_exp, sf64_is_nan, sf64_lt, sf64_mul, sf64_neg, sf64_sub,
)
from core.step_gate import STEP_DONE, STEP_ITER, STEP_STOP_ITER, StepI32
from gbdt.overfitting_detector.overfitting_detector import (
    OD_EPS,
    OD_ITERATION_FORGET,
    OD_LAMBDA_FORGET,
    OD_LAMBDA_SCALE,
    OverfittingDetector,
)

comptime OD_W_THRESHOLD = 0
comptime OD_W_LOCAL_MAX = 1
comptime OD_W_EXPECTED_INC = 2
comptime OD_W_LAST_ERROR = 3
comptime OD_W_P_VALUE = 4
comptime OD_W_BEST_ERROR = 5
comptime OD_W_IS_EMPTY = 6
comptime OD_W_FROM_LOCAL_MAX = 7
comptime OD_W_WAIT = 8
comptime OD_W_BEST_ITERATION = 9
comptime OD_W_SEEN = 10
comptime OD_W_MAX_IS_OPTIMAL = 11
comptime OD_W_N_ERRORS = 12
comptime OD_W_HEAD = 13
comptime OD_W_LAMBDA_FORGET = 14
comptime OD_W_EPS = 15
comptime OD_W_LAMBDA_SCALE = 16
comptime OD_W_NEED_STOP = 17
comptime OD_DEVICE_WORDS = 18
#: Threads of `od_step_kernel`'s one block.
comptime OD_TPB = 256

comptime _U64 = MutPointer[UInt64, MutAnyOrigin]


@always_inline
def _od_lt(a: UInt64, b: UInt64) -> Bool:
    """IEEE `a < b` (False when either is NaN)."""
    return not sf64_is_nan(a) and not sf64_is_nan(b) and sf64_lt(a, b)


@always_inline
def _od_gt(a: UInt64, b: UInt64) -> Bool:
    return _od_lt(b, a)


struct OdDeviceState(Movable):
    """The detector's state on the device: `OD_DEVICE_WORDS` state words,
    the `Errors` ring (newest first, at most OD_ITERATION_FORGET) and the
    forgetting multipliers `0.99^i` by repeated multiplication (the host
    loop's `cur_mult`)."""

    var words: DeviceBuffer[DType.uint64]
    var ring: DeviceBuffer[DType.uint64]
    var mult: DeviceBuffer[DType.uint64]

    def __init__(out self, ctx: DeviceContext) raises:
        self.words = ctx.enqueue_create_buffer[DType.uint64](OD_DEVICE_WORDS)
        self.ring = ctx.enqueue_create_buffer[DType.uint64](OD_ITERATION_FORGET)
        self.mult = ctx.enqueue_create_buffer[DType.uint64](OD_ITERATION_FORGET)


def od_init_kernel(
    st: _U64, mult: _U64, threshold: UInt64, wait_in: Int32, max_is_optimal_in: Int32,
    lambda_forget: UInt64, eps: UInt64, lambda_scale: UInt64,
):
    """One thread: `make_overfitting_detector`'s initial state, and the
    multiplier table. The table is a product recurrence (serial by nature,
    OD_ITERATION_FORGET steps, once per fit)."""
    if Int(block_idx.x) != 0 or Int(thread_idx.x) != 0:
        return
    st.unsafe_store(OD_W_THRESHOLD, threshold)
    st.unsafe_store(OD_W_LOCAL_MAX, SF64_ZERO)
    st.unsafe_store(OD_W_EXPECTED_INC, SF64_ZERO)
    st.unsafe_store(OD_W_LAST_ERROR, SF64_ZERO)
    st.unsafe_store(OD_W_P_VALUE, SF64_ONE)
    st.unsafe_store(OD_W_BEST_ERROR, SF64_ZERO)
    st.unsafe_store(OD_W_IS_EMPTY, UInt64(1))
    st.unsafe_store(OD_W_FROM_LOCAL_MAX, UInt64(0))
    st.unsafe_store(OD_W_WAIT, UInt64(Int(wait_in)))
    st.unsafe_store(OD_W_BEST_ITERATION, UInt64(0))
    st.unsafe_store(OD_W_SEEN, UInt64(0))
    st.unsafe_store(OD_W_MAX_IS_OPTIMAL, UInt64(Int(max_is_optimal_in)))
    st.unsafe_store(OD_W_N_ERRORS, UInt64(0))
    st.unsafe_store(OD_W_HEAD, UInt64(OD_ITERATION_FORGET - 1))
    st.unsafe_store(OD_W_LAMBDA_FORGET, lambda_forget)
    st.unsafe_store(OD_W_EPS, eps)
    st.unsafe_store(OD_W_LAMBDA_SCALE, lambda_scale)
    st.unsafe_store(OD_W_NEED_STOP, UInt64(0))
    var m = SF64_ONE
    for i in range(OD_ITERATION_FORGET):
        mult.unsafe_store(i, m)
        m = sf64_mul(m, lambda_forget)


def od_step_kernel(
    words: StepI32, st: _U64, ring: _U64, mult: _U64, loss: _U64, it: Int32,
):
    """One block of OD_TPB threads, while the loop runs: `add_error(loss[0])`
    then `is_need_stop`, which sets the loop's done word at round `it`.

    The decaying maximum `ExpectedInc = max(0.99 E, max_i mult[i] (err -
    Errors[i]))` is the only loop over the history: each thread scans a
    strided slice in ascending order keeping the first strictly greater
    candidate (NaN never wins), and the block folds the pairs keeping the
    larger, the lower index on a tie. Max is exact, so the result is the
    host loop's word (its first occurrence of the maximum), whatever the
    thread count."""
    var tid = Int(thread_idx.x)
    if words.unsafe_load(STEP_DONE) != Int32(0):
        return
    var sv = stack_allocation[OD_TPB, Scalar[DType.uint64], address_space = AddressSpace.SHARED]()
    var si = stack_allocation[OD_TPB, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var sc = stack_allocation[8, Scalar[DType.uint64], address_space = AddressSpace.SHARED]()
    if tid == 0:
        var raw = loss.unsafe_load(0)
        var seen = Int(st.unsafe_load(OD_W_SEEN))
        # their best-iteration bookkeeping, which is ours
        if seen == 0 or _od_lt(raw, st.unsafe_load(OD_W_BEST_ERROR)):
            st.unsafe_store(OD_W_BEST_ERROR, raw)
            st.unsafe_store(OD_W_BEST_ITERATION, UInt64(seen))
        st.unsafe_store(OD_W_SEEN, UInt64(seen + 1))
        sc[4] = UInt64(0)
        # `if self.threshold <= 0.0: return`
        if _od_lt(SF64_ZERO, st.unsafe_load(OD_W_THRESHOLD)):
            var err = raw
            if st.unsafe_load(OD_W_MAX_IS_OPTIMAL) == UInt64(0):
                err = sf64_neg(raw)
            var empty = st.unsafe_load(OD_W_IS_EMPTY) != UInt64(0)
            if empty or _od_gt(err, st.unsafe_load(OD_W_LOCAL_MAX)):
                if empty:
                    st.unsafe_store(OD_W_IS_EMPTY, UInt64(0))
                    st.unsafe_store(OD_W_EXPECTED_INC, SF64_ZERO)
                st.unsafe_store(OD_W_LOCAL_MAX, err)
                st.unsafe_store(OD_W_FROM_LOCAL_MAX, UInt64(0))
            else:
                st.unsafe_store(OD_W_FROM_LOCAL_MAX, st.unsafe_load(OD_W_FROM_LOCAL_MAX) + UInt64(1))
            # `Errors.push_front(err)`, capped at ITERATION_FORGET
            var head = (Int(st.unsafe_load(OD_W_HEAD)) + 1) % OD_ITERATION_FORGET
            ring.unsafe_store(head, err)
            st.unsafe_store(OD_W_HEAD, UInt64(head))
            var n = Int(st.unsafe_load(OD_W_N_ERRORS)) + 1
            if n > OD_ITERATION_FORGET:
                n = OD_ITERATION_FORGET
            st.unsafe_store(OD_W_N_ERRORS, UInt64(n))
            sc[0] = err
            sc[1] = sf64_mul(st.unsafe_load(OD_W_EXPECTED_INC), st.unsafe_load(OD_W_LAMBDA_FORGET))
            sc[2] = UInt64(n)
            sc[3] = UInt64(head)
            sc[4] = UInt64(1)
    barrier()
    var active = sc[4] != UInt64(0)
    var best_v = SF64_ZERO
    var best_i = Int32(-1)
    if active:
        var err = sc[0]
        var n = Int(sc[2])
        var head = Int(sc[3])
        var i = tid
        while i < n:
            var e_i = ring.unsafe_load((head - i + OD_ITERATION_FORGET) % OD_ITERATION_FORGET)
            var cand = sf64_mul(mult.unsafe_load(i), sf64_sub(err, e_i))
            if not sf64_is_nan(cand) and (best_i < Int32(0) or _od_gt(cand, best_v)):
                best_v = cand
                best_i = Int32(i)
            i += OD_TPB
    sv[tid] = best_v
    si[tid] = best_i
    barrier()
    var step = OD_TPB // 2
    while step > 0:
        if tid < step:
            var ia = si[tid]
            var ib = si[tid + step]
            if ib >= Int32(0):
                var take_b = False
                if ia < Int32(0):
                    take_b = True
                elif _od_gt(sv[tid + step], sv[tid]):
                    take_b = True
                elif not _od_gt(sv[tid], sv[tid + step]) and ib < ia:
                    take_b = True
                if take_b:
                    sv[tid] = sv[tid + step]
                    si[tid] = ib
        barrier()
        step //= 2
    if tid != 0:
        return
    words.unsafe_store(STEP_ITER, it)
    if not active:
        return
    var e = sc[1]
    if not sf64_is_nan(e) and si[0] >= Int32(0) and _od_gt(sv[0], e):
        e = sv[0]
    st.unsafe_store(OD_W_EXPECTED_INC, e)
    var err0 = sc[0]
    st.unsafe_store(OD_W_LAST_ERROR, err0)
    # `UpdatePValue`
    var p_value = SF64_ONE
    if st.unsafe_load(OD_W_FROM_LOCAL_MAX) >= st.unsafe_load(OD_W_WAIT):
        var eps = st.unsafe_load(OD_W_EPS)
        var denom = sf64_sub(st.unsafe_load(OD_W_LOCAL_MAX), err0)
        if _od_lt(denom, eps):
            denom = eps
        var p = sf64_div(e, denom)
        if _od_lt(p, eps):
            p = eps
        p_value = sf64_exp(sf64_div(sf64_neg(st.unsafe_load(OD_W_LAMBDA_SCALE)), p))
    st.unsafe_store(OD_W_P_VALUE, p_value)
    # `IsNeedStop`: not empty and p < threshold (strict)
    if st.unsafe_load(OD_W_IS_EMPTY) == UInt64(0) and _od_lt(p_value, st.unsafe_load(OD_W_THRESHOLD)):
        st.unsafe_store(OD_W_NEED_STOP, UInt64(1))
        words.unsafe_store(STEP_STOP_ITER, it)
        words.unsafe_store(STEP_DONE, Int32(1))


def od_op_init(ctx: DeviceContext, mut state: OdDeviceState, detector: OverfittingDetector) raises:
    """Load a fresh detector (`make_overfitting_detector`'s result: its
    threshold, wait and sign convention) into the device state. Enqueue
    only."""
    if not detector.is_empty or detector.seen != 0:
        raise Error("od_op_init: the detector has already seen errors; pass a fresh one")
    if detector.iterations_wait < 0:
        raise Error("od_op_init: iterations_wait must be >= 0")
    ctx.enqueue_function[od_init_kernel](  # small-launch(one thread): the detector's state and multiplier table, once per fit
        state.words.unsafe_ptr(), state.mult.unsafe_ptr(),
        bitcast[DType.uint64](detector.threshold),
        Int32(detector.iterations_wait),
        Int32(1) if detector.max_is_optimal else Int32(0),
        bitcast[DType.uint64](OD_LAMBDA_FORGET),
        bitcast[DType.uint64](OD_EPS),
        bitcast[DType.uint64](OD_LAMBDA_SCALE),
        grid_dim=(1, 1, 1), block_dim=(1, 1, 1),
    )


def od_op_step(
    ctx: DeviceContext,
    mut words: DeviceBuffer[DType.int32],
    mut state: OdDeviceState,
    mut losses: DeviceBuffer[DType.uint64],
    it: Int,
) raises:
    """Tree op TREE_OP_OD_STEP, the boosting loop's criterion: the detector
    reads round `it`'s held-out loss (`losses[it - 1]`, written by
    `tree_op_heldout_loss`) and, when it would stop the host loop, sets
    the loop's done word with `it` as the stopping round (the round's tree
    is kept, as the host loop's `break` keeps it). `words` is the loop's
    register block (`LoopRegisters.words`). Enqueue last in the round."""
    if it < 1 or it > len(losses):
        raise Error("od_op_step: round " + String(it) + " outside the loss history")
    ctx.enqueue_function[od_step_kernel](
        words.unsafe_ptr(), state.words.unsafe_ptr(), state.ring.unsafe_ptr(),
        state.mult.unsafe_ptr(), losses.unsafe_ptr() + (it - 1), Int32(it),
        grid_dim=(1, 1, 1), block_dim=(OD_TPB, 1, 1),
    )
