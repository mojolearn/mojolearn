# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Device work behind the linear-model prediction entry (lane p3-linear;
one-drain form lane ols-regression, 2026-11-14).

`ols_predict_device`: `X . coef + intercept` for Ridge and LinearRegression,
float32 words for every caller (the binary64 widening of the public predict
was removed with its caller test, lane predict-dtype 2026-11-22). Per call:
X and coef go up, the measured gemv (`gemv_n_ols_predict`), ONE epilogue
launch (intercept add and the status word's zero), the finite test of the uploaded X when the door
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
ftz(intercept))` (skipped when the intercept is exactly 0.0, DEVIATION 527).
The status word is a predicate on bits.
"""
from max.gpu.host import DeviceBuffer, DeviceContext
from core.fold_x_device import fold_x_fill
from max.gpu import block_dim, block_idx, thread_idx
from checks.numerics import ftz
from core.admitted_finite_status import enqueue_finite_status
from core.gemm import gemv_n_ols_predict
from core.step_phase import (
    step_count_launch, step_count_sync, step_count_h2d, step_count_d2h, step_count_device_alloc,
)

comptime _EPI_TPB = 256


def ols_predict_epilogue_kernel(
    z: MutPointer[Float32, MutAnyOrigin],
    status: MutPointer[Int32, MutAnyOrigin],
    n_in: Int32,
    intercept: Float32,
    add: Int32,
):
    """One thread per prediction, plus thread `n` for the status word.

    `add` != 0: `z[i] = ftz(ftz(z[i]) + ftz(intercept))`, the intercept
    epilogue of IDENTITY_PATHS row 10 / DEVIATION 527 (a float seam: the
    operand comes from the gemv and the result leaves the device; CUDA keeps
    a denormal Metal has flushed, so both sides are flushed). There is no
    multiply, so no contraction to pin. Thread `n` stores 0 into
    `status[0]` before the finite test (launched after this kernel on the
    same stream) can flag it: no fill launch."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var n = Int(n_in)
    if i < n:
        var v = z.unsafe_load(i)
        if add != 0:
            v = ftz(ftz(v) + ftz(intercept))
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
    test_x: Bool,
) raises -> Int:
    """`out[n_rows]` = X . coef + intercept: float32 words at `out_addr`. `test_x`: the
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
    # The step_count_* lines count this call's own sites under
    # -D MOJOLEARN_STEP_PHASE_TIMERS (core/step_phase.mojo; nothing otherwise):
    # checks/ols_call_cost_check.mojo asserts them.
    comptime for _ in range(4):  # the four buffers created above
        step_count_device_alloc()
    if n_rows * n_features > 0:
        # THE FOLD X WORD (core/fold_x_device.mojo; lane vdf-scope-linear): a member's
        # fold slot is copied device to device; a host X is the old upload.
        fold_x_fill(ctx, x, Int(x_ptr), n_rows * n_features)
        step_count_h2d()
    if n_features > 0:
        ctx.enqueue_copy(dst_buf=coef, src_ptr=coef_ptr)
        step_count_h2d()
    # No synchronize here: the gemv is ordered after the copies on this
    # stream (0.8.55's ols_predict_host waited for the upload first).
    gemv_n_ols_predict(ctx, z, x, coef, n_rows, n_features)
    step_count_launch()
    # The same host compare of a caller-supplied constant as before
    # (DEVIATION 527): `x + 0.0` is skipped exactly as there.
    var add = intercept != Float32(0.0)
    if n_rows > 0 and (add or test):
        ctx.enqueue_function[ols_predict_epilogue_kernel](
            z.unsafe_ptr(), status.unsafe_ptr(), Int32(n_rows), intercept,
            Int32(1 if add else 0),
            grid_dim=((n_rows + 1 + _EPI_TPB - 1) // _EPI_TPB, 1, 1), block_dim=(_EPI_TPB, 1, 1),
        )
        step_count_launch()
    if test:
        enqueue_finite_status(ctx, x.unsafe_ptr(), n_rows * n_features, status.unsafe_ptr(), 0)
        step_count_launch()
    var word = List[Int32](length=1, fill=0)
    if n_rows > 0:
        ctx.enqueue_copy(dst_ptr=MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=out_addr), src_buf=z)
        step_count_d2h()
    if test:
        ctx.enqueue_copy(dst_ptr=MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(word.unsafe_ptr())),
                         src_buf=status)
        step_count_d2h()
    ctx.synchronize()
    step_count_sync()
    var bad = 1 if test and word[0] != Int32(0) else 0
    _ = x^
    _ = coef^
    _ = z^
    _ = status^
    _ = word^
    return bad


def ols_predict_lay_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    z: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    targets_in: Int32,
    target_in: Int32,
):
    """`dst[i, t] = z[i]` (rows x targets, row-major): one thread per row;
    an exact copy (lane w3f3-k)."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        dst.unsafe_store(i * Int(targets_in) + Int(target_in), z.unsafe_load(i))


def ols_predict_device_targets(
    ctx: DeviceContext,
    x_ptr: MutPointer[Float32, MutUntrackedOrigin],
    coef_ptr: MutPointer[Float32, MutUntrackedOrigin],
    intercepts: MutPointer[Float64, MutUntrackedOrigin],
    out_addr: Int,
    n_rows: Int,
    n_features: Int,
    targets: Int,
    test_x: Bool,
) raises -> Int:
    """A multi-target Ridge's `out[n_rows x targets]` = X . coef^T +
    intercepts (lane w3f3-k), float32 words at `out_addr`, row-major: X and
    every coefficient row go up ONCE, each target is the measured gemv
    (`gemv_n_ols_predict`, the one-target body, so each column is the
    one-target prediction's words) and the same epilogue into its own
    column, laid into one rows x targets block on the device, which comes
    down in one copy at the call's one synchronize. `test_x` as
    `ols_predict_device`."""
    var rows = max(n_rows, 1)
    var test = test_x and n_rows > 0 and n_features > 0
    var x = ctx.enqueue_create_buffer[DType.float32](max(n_rows * n_features, 1))
    var block = ctx.enqueue_create_buffer[DType.float32](max(n_rows * targets, 1))
    var status = ctx.enqueue_create_buffer[DType.int32](1)
    var coefs = List[DeviceBuffer[DType.float32]]()
    var zs = List[DeviceBuffer[DType.float32]]()
    for _ in range(targets):  # driver-loop: over the targets (one coefficient row and one result each)
        coefs.append(ctx.enqueue_create_buffer[DType.float32](max(n_features, 1)))
        zs.append(ctx.enqueue_create_buffer[DType.float32](rows))
    if n_rows * n_features > 0:
        ctx.enqueue_copy(dst_buf=x, src_ptr=x_ptr)
    if n_features > 0:
        for t in range(targets):  # driver-loop: over the targets (their coefficient rows, uploaded once)
            ctx.enqueue_copy(dst_buf=coefs[t], src_ptr=MutPointer[Float32, MutUntrackedOrigin](
                unsafe_from_address=Int(coef_ptr) + 4 * t * n_features))
    var grid = (n_rows + 1 + _EPI_TPB - 1) // _EPI_TPB
    for t in range(targets):  # driver-loop: over the targets (one gemv, epilogue and lay each)
        gemv_n_ols_predict(ctx, zs[t], x, coefs[t], n_rows, n_features)
        var intercept = Float32(intercepts[t])
        # The one-target rule per target: `x + 0.0` skipped (DEVIATION
        # 527); the first target's epilogue also zeroes the status word.
        var add = intercept != Float32(0.0)
        if n_rows > 0 and (add or (test and t == 0)):
            ctx.enqueue_function[ols_predict_epilogue_kernel](
                zs[t].unsafe_ptr(), status.unsafe_ptr(), Int32(n_rows), intercept, Int32(1 if add else 0),
                grid_dim=(grid, 1, 1), block_dim=(_EPI_TPB, 1, 1),
            )
        if n_rows > 0:
            ctx.enqueue_function[ols_predict_lay_kernel](
                block.unsafe_ptr(), zs[t].unsafe_ptr(), Int32(n_rows), Int32(targets), Int32(t),
                grid_dim=((n_rows + _EPI_TPB - 1) // _EPI_TPB, 1, 1), block_dim=(_EPI_TPB, 1, 1),
            )
    if test:
        enqueue_finite_status(ctx, x.unsafe_ptr(), n_rows * n_features, status.unsafe_ptr(), 0)
    var word = List[Int32](length=1, fill=0)
    if n_rows > 0:
        ctx.enqueue_copy(dst_ptr=MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=out_addr), src_buf=block)
    if test:
        ctx.enqueue_copy(dst_ptr=MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(word.unsafe_ptr())),
                         src_buf=status)
    ctx.synchronize()
    var bad = 1 if test and word[0] != Int32(0) else 0
    _ = x^
    _ = block^
    _ = status^
    _ = coefs^
    _ = zs^
    _ = word^
    return bad
