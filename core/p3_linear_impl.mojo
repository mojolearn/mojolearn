# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Device work behind the linear-model prediction entry (lane p3-linear;
one-drain form lane ols-regression, 2026-11-14).

`ols_predict_device`: `X . coef + intercept` for Ridge and LinearRegression,
float32 words (package callers) or their exact binary64 widening (the public
float64 prediction). Per call: X and coef go up, the measured gemv
(`gemv_n_ols_predict`), ONE epilogue launch (intercept add, widening, and
the status word's zero), the finite test of the uploaded X when the door
deferred it, the result and the status word come down, ONE synchronize.

WHY (ols-regression, L40S paper board 80a7f8aca3 vs 0.8.55): ols taxi infer
Xq 5.45 -> 9.30 ms, istella 39.4 -> 60.8 ms. Against 0.8.55's `ols_predict`
the call had added a host finite scan of every X word at admission, a
separate widening launch and its buffer, and a synchronize between the
upload and the gemv. The scan moved here as one device status word read at
the call's one drain (core/admitted_finite_status.mojo); the add and the
widening are one launch; the mid-call synchronize is gone. The narrow form
also lost its pinned staging buffer and the serial host loop that copied
it out row by row: the result is copied straight into the caller's output
as the wide form always was.

NUMERICS: none. The epilogue is `_add_scalar_kernel`'s `ftz(ftz(v) +
ftz(intercept))` (skipped when the intercept is exactly 0.0, DEVIATION 527)
followed by `sf64_from_f32` of the stored float32 word, the same two steps
the two launches ran. The status word is a predicate on bits.
"""
from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, thread_idx
from checks.numerics import ftz
from checks.soft_f64 import sf64_from_f32
from core.admitted_finite_status import enqueue_finite_status
from core.gemm import gemv_n_ols_predict

comptime _EPI_TPB = 256


def ols_predict_epilogue_kernel(
    z: MutPointer[Float32, MutAnyOrigin],
    wide: MutPointer[UInt64, MutAnyOrigin],
    status: MutPointer[Int32, MutAnyOrigin],
    n_in: Int32,
    intercept: Float32,
    add: Int32,
    widen: Int32,
):
    """One thread per prediction, plus thread `n` for the status word.

    `add` != 0: `z[i] = ftz(ftz(z[i]) + ftz(intercept))`, the intercept
    epilogue of IDENTITY_PATHS row 10 / DEVIATION 527 (a float seam: the
    operand comes from the gemv and the result leaves the device; CUDA keeps
    a denormal Metal has flushed, so both sides are flushed). There is no
    multiply, so no contraction to pin. `widen` != 0: `wide[i]` = the exact
    binary64 bit pattern of that float32 word (`sf64_from_f32`; the Apple GPU
    has no float64) and `z` is not written back. Thread `n` stores 0 into
    `status[0]` before the finite test (launched after this kernel on the
    same stream) can flag it: no fill launch."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var n = Int(n_in)
    if i < n:
        var v = z.unsafe_load(i)
        if add != 0:
            v = ftz(ftz(v) + ftz(intercept))
        if widen != 0:
            wide.unsafe_store(i, sf64_from_f32(v))
        elif add != 0:
            z.unsafe_store(i, v)
    elif i == n:
        status.unsafe_store(0, Int32(0))


def ols_predict_device(
    ctx: DeviceContext,
    x_ptr: MutPointer[Float32, MutUntrackedOrigin],
    coef_ptr: MutPointer[Float32, MutUntrackedOrigin],
    out_addr: Int,
    n_rows: Int,
    n_features: Int,
    intercept: Float32,
    wide: Bool,
    test_x: Bool,
) raises -> Int:
    """`out[n_rows]` = X . coef + intercept: float32 words at `out_addr`, or
    with `wide` their exact binary64 widening (float64 words). `test_x`: the
    door deferred X's finite test (a borrowed X it did not scan); the words
    uploaded here are tested on the device and the call returns 1 when any is
    NaN or infinite (the predictions are then not meaningful and the door
    refuses), else 0. One synchronize per call."""
    var rows = max(n_rows, 1)
    # An empty X has no word to test (and no epilogue to zero the word).
    var test = test_x and n_rows > 0 and n_features > 0
    var x = ctx.enqueue_create_buffer[DType.float32](max(n_rows * n_features, 1))
    var coef = ctx.enqueue_create_buffer[DType.float32](max(n_features, 1))
    var z = ctx.enqueue_create_buffer[DType.float32](rows)
    var status = ctx.enqueue_create_buffer[DType.int32](1)
    # The widened words; a one-word placeholder on the narrow path (the
    # epilogue does not write it there).
    var w = ctx.enqueue_create_buffer[DType.uint64](rows if wide else 1)
    if n_rows * n_features > 0:
        ctx.enqueue_copy(dst_buf=x, src_ptr=x_ptr)
    if n_features > 0:
        ctx.enqueue_copy(dst_buf=coef, src_ptr=coef_ptr)
    # No synchronize here: the gemv is ordered after the copies on this
    # stream (0.8.55's ols_predict_host waited for the upload first).
    gemv_n_ols_predict(ctx, z, x, coef, n_rows, n_features)
    # The same host compare of a caller-supplied constant as before
    # (DEVIATION 527): `x + 0.0` is skipped exactly as there.
    var add = intercept != Float32(0.0)
    if n_rows > 0 and (add or wide or test):
        ctx.enqueue_function[ols_predict_epilogue_kernel](
            z.unsafe_ptr(), w.unsafe_ptr(), status.unsafe_ptr(), Int32(n_rows), intercept,
            Int32(1 if add else 0), Int32(1 if wide else 0),
            grid_dim=((n_rows + 1 + _EPI_TPB - 1) // _EPI_TPB, 1, 1), block_dim=(_EPI_TPB, 1, 1),
        )
    if test:
        enqueue_finite_status(ctx, x.unsafe_ptr(), n_rows * n_features, status.unsafe_ptr(), 0)
    var word = List[Int32](length=1, fill=0)
    if n_rows > 0:
        if wide:
            ctx.enqueue_copy(dst_ptr=MutPointer[UInt64, MutUntrackedOrigin](unsafe_from_address=out_addr), src_buf=w)
        else:
            ctx.enqueue_copy(dst_ptr=MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=out_addr), src_buf=z)
    if test:
        ctx.enqueue_copy(dst_ptr=MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(word.unsafe_ptr())),
                         src_buf=status)
    ctx.synchronize()
    var bad = 1 if test and word[0] != Int32(0) else 0
    _ = x^
    _ = coef^
    _ = z^
    _ = status^
    _ = w^
    _ = word^
    return bad
