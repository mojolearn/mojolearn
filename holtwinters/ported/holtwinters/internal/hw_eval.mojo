"""cuML `cpp/src/holtwinters/internal/hw_eval.cuh` (v26.08.00).

`holtwinters_eval_device` (the level/trend/season recurrence, ONE THREAD
PER SERIES, serial in `i`), the global-scratch kernel and
`holtwinters_eval_gpu`.

THE IDENTITY CONTENT. Their recurrence is already order-fixed: one thread
walks `i = 0 .. n - shift - 1` and the SSE is `error_ += diff * diff` in
that same walk. There is no block fold, no warp primitive and no atomic, so
there is no summation order to pin; what CAN move between vendors is (a)
the CONTRACTION of every `a * x + b * y` (row 9), (b) the denormal policy
of every stored intermediate (row 10), and (c) the `pts / stmp_eps` and
`pts / clevel` divisions under MULTIPLICATIVE (row 10: correctly rounded on
every column measured). The spelling below is the pin, and the host oracle
(`holtwinters/mojo_only/hw_oracle.mojo`) is written to the SAME spelling:

    leveltrend = ftz(plevel + ptrend)
    xhat       = ftz(leveltrend + stmp)            | ftz(leveltrend * stmp)
    diff       = ftz(pts - xhat);  error = ftz(fma(diff, diff, error))
    clevel     = ftz(fma(alpha, ftz(pts - stmp),  ftz(ftz(1 - alpha) * leveltrend)))
               | ftz(fma(alpha, ftz(pts / stmp_eps), ...))
    ctrend     = ftz(fma(beta,  ftz(clevel - plevel), ftz(ftz(1 - beta) * ptrend)))
    cseason    = ftz(fma(gamma, ftz(pts - clevel), ftz(ftz(1 - gamma) * stmp)))
               | ftz(fma(gamma, ftz(pts / clevel), ...))

i.e. in every `a * x + b * y` the FIRST product (the parameter times the
observation term) is the fused one and the second is a stored product.
`fma` is `identical_mul_add` (a real fma under IDENTICAL, the naive chain
under FAST); `ftz` compiles away under FAST. `SAB_SWAP_FMA` fuses the other
product in the level update and must fail device-vs-oracle.

============ DEVIATION 698 (2026-08-23): THE LANE'S ONE FLUSH-AND-FUSE
============ RULE, NAMED, BECAUSE IT IS NOT THEIR SPELLING ================
The recurrence above had been written down but never NUMBERED, which left
the single most identity-load-bearing decision in the lane as a docstring.
It is a deviation and it is stated once, here, for every file in
`holtwinters/ported/`:

  FLUSH. Every STORED intermediate goes through `ftz` (`mojo_only/
  numerics.mojo`, IDENTITY_PATHS row 10). Theirs flushes nothing
  explicitly and inherits whatever the vendor's default denormal mode is
  -- which is exactly the disagreement row 10 exists to close (Apple
  flushes, NVIDIA and AMD keep). Any recurrence term can be subnormal:
  `pts - xhat_` on a nearly-recovered series, `1 - alpha_` for alpha at
  the clamp, `clevel - plevel` at convergence. Under FAST `ftz` compiles
  away, so FAST is the vendor-default arm and IDENTICAL is the pinned one.

  FUSE. At every `a * x + b * y` seam the FIRST product is FUSED into the
  add and the SECOND is stored: `fma(a, x, ftz(b * y))`. C++ says nothing
  about which of the two nvcc contracts (`-fmad=true` is the default and
  the choice is the compiler's), so THEIR spelling does not determine one;
  a pin was required and this is it. It is applied in `_mix` (level, trend
  and season), in the SSE accumulation `fma(diff, diff, error_)`, in
  `conv1d`, in `batched_ls_solver`, in the forecast's `fma(trend, i+1,
  level)`, in `_dot3`, in `fma(step, p, x)`, in `fma(step, cauchy,
  loss_ref)` and in the three off-diagonal Hessian terms. ONE rule, no
  exceptions, so a reader never has to ask which way a seam went.

  WHERE IT IS NOT APPLIED, deliberately: the `2 * rho * s * Hy` chains and
  the `k * s * s` chains keep every product stored, because theirs
  parenthesizes them left to right with no add to fuse INTO.

WHY a pin at all rather than "whatever the compiler does": the lane's goal
is Metal == CUDA == HIP, and three compilers contracting the same
expression by their own rules is precisely how that fails. The cost is
that our numbers are not bit-for-bit cuML's on NVIDIA; that was never the
goal and the README says so.
MEASURED: `SAB_SWAP_FMA` fuses the OTHER product in the level update and
`SAB_NO_FTZ` drops the flush; both FAIL device-vs-oracle under IDENTICAL
(README carries the lines). The oracle in `holtwinters/mojo_only/
hw_oracle.mojo` is written to the same rule, term for term.
============================================================================

`pseason` is the per-thread scratch of the last `frequency` season values:
their shared-memory kernel indexes it `pseason[s * blockDim.x]` from
`pseason + threadIdx.x`, the global one `pseason[s * batch_size]` from
`pseason + tid`; both are PLACEMENT (no arithmetic) and the global one is
the one ported (UNPORTED.tsv). Its contents are a pure function of the
series, never of the block.
"""

from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from holtwinters.ported.holtwinters.internal.hw_utils import (
    SAB_NO_FTZ,
    SAB_SWAP_FMA,
    STMP_EPS,
    abs_device,
    bound_device,
)
from mojo_only.numerics import ftz, identical_mul_add

#: `level`/`trend`/`season`/`xhat` output selection (theirs: pointer nullness)
comptime HW_WRITE_LEVEL = 1
comptime HW_WRITE_TREND = 2
comptime HW_WRITE_SEASON = 4
comptime HW_WRITE_XHAT = 8
comptime HW_WRITE_ALL = 15


@always_inline
def _f(x: Float32) -> Float32:
    """Row 10's stored-intermediate flush (a no-op under FAST and under the
    NO_FTZ sabotage)."""
    comptime if SAB_NO_FTZ:
        return x
    return ftz(x)


@always_inline
def _mix(a: Float32, x: Float32, one_minus_a: Float32, y: Float32) -> Float32:
    """`a * x + (1 - a) * y` in the pinned spelling: the first product fused,
    the second stored. `one_minus_a` is the caller's `_f(1 - a)`."""
    comptime if SAB_SWAP_FMA:
        return _f(identical_mul_add(one_minus_a, y, _f(a * x)))
    return _f(identical_mul_add(a, x, _f(one_minus_a * y)))


@always_inline
def holtwinters_eval_device(
    tid: Int,
    ts: MutPointer[Float32, MutAnyOrigin],
    n: Int,
    batch_size: Int,
    frequency: Int,
    shift: Int,
    plevel_in: Float32,
    ptrend_in: Float32,
    pseason: MutPointer[Float32, MutAnyOrigin],
    pseason_width: Int,
    start_season: MutPointer[Float32, MutAnyOrigin],
    use_beta: Bool,
    use_gamma: Bool,
    alpha_in: Float32,
    beta_in: Float32,
    gamma_in: Float32,
    level: MutPointer[Float32, MutAnyOrigin],
    trend: MutPointer[Float32, MutAnyOrigin],
    season: MutPointer[Float32, MutAnyOrigin],
    xhat: MutPointer[Float32, MutAnyOrigin],
    write_mask: Int,
    additive_seasonal: Bool,
) -> Float32:
    """`holtwinters_eval_device` (`hw_eval.cuh:13-94`) line for line; the
    pointer-nullness tests become `use_beta`/`use_gamma`/`write_mask`.
    `pseason` is already offset to this thread's scratch (`+ tid`), stride
    `pseason_width`. Returns the SSE."""
    var alpha_ = bound_device(alpha_in)
    var beta_ = bound_device(beta_in)
    var gamma_ = bound_device(gamma_in)
    var one_minus_alpha = _f(Float32(1.0) - alpha_)
    var one_minus_beta = _f(Float32(1.0) - beta_)
    var one_minus_gamma = _f(Float32(1.0) - gamma_)

    var plevel = plevel_in
    var ptrend = ptrend_in
    var error_ = Float32(0.0)
    var clevel = Float32(0.0)
    var ctrend = Float32(0.0)
    var cseason = Float32(0.0)
    var stmp_default: Float32 = Float32(0.0) if additive_seasonal else Float32(1.0)
    for i in range(n - shift):
        var s = i % frequency
        # IDX(tid, i + shift, batch_size) = tid + (i + shift) * batch_size
        var pts = ts.unsafe_load(tid + (i + shift) * batch_size)
        var leveltrend = _f(plevel + ptrend)

        # xhat
        var stmp: Float32
        if use_gamma:
            if i < frequency:
                stmp = start_season.unsafe_load(tid + i * batch_size)
            else:
                stmp = pseason.unsafe_load(s * pseason_width)
        else:
            stmp = stmp_default
        var xhat_: Float32
        if additive_seasonal:
            xhat_ = _f(leveltrend + stmp)
        else:
            xhat_ = _f(leveltrend * stmp)

        # Error
        var diff = _f(pts - xhat_)
        error_ = _f(identical_mul_add(diff, diff, error_))

        # Level
        if additive_seasonal:
            clevel = _mix(alpha_, _f(pts - stmp), one_minus_alpha, leveltrend)
        else:
            var stmp_eps: Float32 = stmp if abs_device(stmp) > STMP_EPS else STMP_EPS
            clevel = _mix(alpha_, _f(pts / stmp_eps), one_minus_alpha, leveltrend)

        # Trend
        if use_beta:
            ctrend = _mix(beta_, _f(clevel - plevel), one_minus_beta, ptrend)
            ptrend = ctrend

        # Seasonal
        if use_gamma:
            if additive_seasonal:
                cseason = _mix(gamma_, _f(pts - clevel), one_minus_gamma, stmp)
            else:
                cseason = _mix(gamma_, _f(pts / clevel), one_minus_gamma, stmp)
            pseason.unsafe_store(s * pseason_width, cseason)

        plevel = clevel

        if (write_mask & HW_WRITE_LEVEL) != 0:
            level.unsafe_store(tid + i * batch_size, clevel)
        if (write_mask & HW_WRITE_TREND) != 0:
            trend.unsafe_store(tid + i * batch_size, ctrend)
        if (write_mask & HW_WRITE_SEASON) != 0:
            season.unsafe_store(tid + i * batch_size, cseason)
        if (write_mask & HW_WRITE_XHAT) != 0:
            xhat.unsafe_store(tid + i * batch_size, xhat_)
    return error_


def holtwinters_eval_gpu_global_kernel(
    ts: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    batch_size_in: Int32,
    frequency_in: Int32,
    start_level: MutPointer[Float32, MutAnyOrigin],
    start_trend: MutPointer[Float32, MutAnyOrigin],
    start_season: MutPointer[Float32, MutAnyOrigin],
    pseason: MutPointer[Float32, MutAnyOrigin],
    alpha: MutPointer[Float32, MutAnyOrigin],
    beta: MutPointer[Float32, MutAnyOrigin],
    gamma: MutPointer[Float32, MutAnyOrigin],
    level: MutPointer[Float32, MutAnyOrigin],
    trend: MutPointer[Float32, MutAnyOrigin],
    season: MutPointer[Float32, MutAnyOrigin],
    xhat: MutPointer[Float32, MutAnyOrigin],
    error: MutPointer[Float32, MutAnyOrigin],
    write_mask_in: Int32,
    write_error_in: Int32,
    use_beta_in: Int32,
    use_gamma_in: Int32,
    additive_in: Int32,
):
    """`holtwinters_eval_gpu_global_kernel` (`hw_eval.cuh:158-217`)."""
    var tid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var n = Int(n_in)
    var batch_size = Int(batch_size_in)
    var frequency = Int(frequency_in)
    var use_beta = use_beta_in != 0
    var use_gamma = use_gamma_in != 0
    if tid < batch_size:
        var shift = 1
        var plevel = start_level.unsafe_load(tid)
        var ptrend = Float32(0.0)
        var alpha_ = alpha.unsafe_load(tid)
        var beta_: Float32 = beta.unsafe_load(tid) if use_beta else Float32(0.0)
        var gamma_: Float32 = gamma.unsafe_load(tid) if use_gamma else Float32(0.0)
        if use_gamma:
            shift = frequency
            ptrend = start_trend.unsafe_load(tid) if use_beta else Float32(0.0)
        elif use_beta:
            shift = 2
            ptrend = start_trend.unsafe_load(tid)
        var error_ = holtwinters_eval_device(
            tid, ts, n, batch_size, frequency, shift, plevel, ptrend,
            pseason.unsafe_offset(tid), batch_size, start_season, use_beta, use_gamma,
            alpha_, beta_, gamma_, level, trend, season, xhat,
            Int(write_mask_in), additive_seasonal=additive_in != 0,
        )
        if write_error_in != 0:
            error.unsafe_store(tid, error_)


def holtwinters_eval_gpu(
    ctx: DeviceContext,
    mut ts: DeviceBuffer[DType.float32],
    n: Int,
    batch_size: Int,
    frequency: Int,
    mut start_level: DeviceBuffer[DType.float32],
    mut start_trend: DeviceBuffer[DType.float32],
    mut start_season: DeviceBuffer[DType.float32],
    mut alpha: DeviceBuffer[DType.float32],
    mut beta: DeviceBuffer[DType.float32],
    mut gamma: DeviceBuffer[DType.float32],
    mut level: DeviceBuffer[DType.float32],
    mut trend: DeviceBuffer[DType.float32],
    mut season: DeviceBuffer[DType.float32],
    mut xhat: DeviceBuffer[DType.float32],
    mut error: DeviceBuffer[DType.float32],
    write_mask: Int,
    write_error: Bool,
    use_beta: Bool,
    use_gamma: Bool,
    additive: Bool,
    mut pseason: DeviceBuffer[DType.float32],
    tpb: Int,
) raises:
    """`holtwinters_eval_gpu` (`hw_eval.cuh:219-288`): the global-scratch
    arm (their shared arm is placement only). `pseason` must hold
    `batch_size * frequency` floats; `tpb` is their
    `GET_THREADS_PER_BLOCK(batch_size)` (scheduling; the gates vary it)."""
    if tpb <= 0:
        raise Error("holtwinters_eval_gpu: tpb must be positive")
    var total_blocks = (batch_size + tpb - 1) // tpb
    ctx.enqueue_function[holtwinters_eval_gpu_global_kernel](
        ts.unsafe_ptr(), Int32(n), Int32(batch_size), Int32(frequency),
        start_level.unsafe_ptr(), start_trend.unsafe_ptr(), start_season.unsafe_ptr(),
        pseason.unsafe_ptr(),
        alpha.unsafe_ptr(), beta.unsafe_ptr(), gamma.unsafe_ptr(),
        level.unsafe_ptr(), trend.unsafe_ptr(), season.unsafe_ptr(), xhat.unsafe_ptr(),
        error.unsafe_ptr(),
        Int32(write_mask), Int32(1 if write_error else 0),
        Int32(1 if use_beta else 0), Int32(1 if use_gamma else 0),
        Int32(1 if additive else 0),
        grid_dim=(total_blocks, 1, 1),
        block_dim=(tpb, 1, 1),
    )
    ctx.synchronize()
