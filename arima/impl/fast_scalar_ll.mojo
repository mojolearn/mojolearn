# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""K3 exact-observation scalar likelihood, Apple FAST opt-in only.

For rd=1, n_diff=0, Z=1, observing y fixes the posterior state to y.
Thus t>0 predicts T*y[t-1]+mu with variance Q; t=0 uses supplied alpha/P.
This is exact in real arithmetic, NOT bit-equivalent to main's recurrence.
No measured quality/speed claim. Two parallel reduction levels replace the
serial likelihood sum in the source-only K3 Metal experiment. Scratch is
the existing LL_ONLY-unused pred/vs/Fs buffers, with one partial per block.
"""
from std.gpu import block_dim, block_idx, thread_idx
from std.math import isfinite
from std.memory import stack_allocation
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST, ftz, identical_log, identical_mul_add

comptime ARIMA_FAST_SCALAR_LL = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
    and is_defined["MOJOLEARN_ARIMA_FAST_SCALAR_LL"]()
    and not is_defined["MOJOLEARN_ARIMA_FAST_SCALAR_LL_OFF"]()
)
comptime SCALAR_LL_TPB = 256
comptime SCALAR_LL_MAX_OBS = SCALAR_LL_TPB * SCALAR_LL_TPB
comptime SCALAR_LL_NO_ERROR = Int32(2147483647)
comptime SCALAR_LL_LOG_2PI = Float32(1.8378770664093453)


def scalar_ll_parts_kernel[CAPTURE: Bool = False](
    y: Pointer[Float32, ImmutAnyOrigin],
    T: Pointer[Float32, ImmutAnyOrigin],
    Q: Pointer[Float32, ImmutAnyOrigin],
    P0: Pointer[Float32, ImmutAnyOrigin],
    alpha0: Pointer[Float32, ImmutAnyOrigin],
    mu: Pointer[Float32, ImmutAnyOrigin],
    log_parts: MutPointer[Float32, MutAnyOrigin],
    quad_parts: MutPointer[Float32, MutAnyOrigin],
    error_parts: MutPointer[Float32, MutAnyOrigin],
    capture: MutPointer[Float32, MutAnyOrigin],
    nobs_in: Int32, batch_in: Int32, intercept_in: Int32,
):
    """One observation per lane, one group per time tile and series.

    CAPTURE writes pred/residual/F for the private numerical probe only.
    Production writes only block partials. Every lane reaches every barrier.
    """
    var nobs = Int(nobs_in)
    var chunks = (nobs + SCALAR_LL_TPB - 1) // SCALAR_LL_TPB
    var group = Int(block_idx.x)
    var bid = group // chunks
    var chunk = group % chunks
    var tid = Int(thread_idx.x)
    var t = chunk * SCALAR_LL_TPB + tid
    var ls = stack_allocation[SCALAR_LL_TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var qs = stack_allocation[SCALAR_LL_TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var es = stack_allocation[SCALAR_LL_TPB, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    var log_term = Float32(0.0)
    var quad_term = Float32(0.0)
    var bad = SCALAR_LL_NO_ERROR
    if bid < Int(batch_in) and t < nobs:
        var pos = bid * nobs + t
        var pred = ftz(alpha0[bid])
        var variance = ftz(P0[bid])
        if t > 0:
            var drift = ftz(mu[bid]) if intercept_in != 0 else Float32(0.0)
            # Keep the multiply and the intercept addition distinct, as in
            # the source recurrence; eliminating gain roundoff still moves bits.
            pred = ftz(ftz(T[bid] * y[pos - 1]) + drift)
            variance = ftz(Q[bid])
        var residual = ftz(y[pos] - pred)
        comptime if CAPTURE:
            var total = Int(batch_in) * nobs
            capture[pos] = pred
            capture[total + pos] = residual
            capture[2 * total + pos] = variance
        if variance <= Float32(0.0) or not isfinite(variance) or not isfinite(residual):
            bad = Int32(t + 1)
        else:
            log_term = ftz(identical_log(variance))
            quad_term = ftz(ftz(residual * residual) / variance)
            if not isfinite(log_term) or not isfinite(quad_term):
                bad = Int32(t + 1)
                log_term = Float32(0.0)
                quad_term = Float32(0.0)
    ls[tid], qs[tid], es[tid] = log_term, quad_term, bad
    barrier()
    var width = SCALAR_LL_TPB // 2
    while width > 0:
        if tid < width:
            ls[tid] = ftz(ls[tid] + ls[tid + width])
            qs[tid] = ftz(qs[tid] + qs[tid + width])
            es[tid] = min(es[tid], es[tid + width])
        barrier()
        width //= 2
    if tid == 0 and bid < Int(batch_in):
        log_parts[group], quad_parts[group] = ls[0], qs[0]
        # Codes are <=65536 and exact in f32; zero means no error.
        error_parts[group] = Float32(0.0) if es[0] == SCALAR_LL_NO_ERROR else Float32(es[0])


def scalar_ll_finish_kernel(
    log_parts: Pointer[Float32, ImmutAnyOrigin],
    quad_parts: Pointer[Float32, ImmutAnyOrigin],
    error_parts: Pointer[Float32, ImmutAnyOrigin],
    Q: Pointer[Float32, ImmutAnyOrigin],
    P: MutPointer[Float32, MutAnyOrigin],
    loglike: MutPointer[Float32, MutAnyOrigin],
    info: MutPointer[Int32, MutAnyOrigin],
    nobs_in: Int32, chunks_in: Int32,
):
    """One cooperative group per series; at most 256 block partials.

    Only the final scalar store uses lane zero. No lane loops over time.
    The existing main LL_ONLY kernel writes final P but leaves alpha intact;
    keep that workspace contract, rather than silently updating alpha here.
    """
    var bid = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var chunks = Int(chunks_in)
    var ls = stack_allocation[SCALAR_LL_TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var qs = stack_allocation[SCALAR_LL_TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var es = stack_allocation[SCALAR_LL_TPB, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    var l = Float32(0.0)
    var q = Float32(0.0)
    var e = SCALAR_LL_NO_ERROR
    if tid < chunks:
        var pos = bid * chunks + tid
        l, q = log_parts[pos], quad_parts[pos]
        var code = error_parts[pos]
        if code > Float32(0.0):
            e = Int32(code)
    ls[tid], qs[tid], es[tid] = l, q, e
    barrier()
    var width = SCALAR_LL_TPB // 2
    while width > 0:
        if tid < width:
            ls[tid] = ftz(ls[tid] + ls[tid + width])
            qs[tid] = ftz(qs[tid] + qs[tid + width])
            es[tid] = min(es[tid], es[tid + width])
        barrier()
        width //= 2
    if tid == 0:
        var n = Float32(nobs_in)
        var average = ftz(qs[0] / n)
        var inner = ftz(average + SCALAR_LL_LOG_2PI)
        var ll = ftz(Float32(-0.5) * ftz(identical_mul_add(n, inner, ls[0])))
        var err = es[0]
        if not isfinite(ll) and err == SCALAR_LL_NO_ERROR:
            err = Int32(nobs_in)
        info[bid] = Int32(0) if err == SCALAR_LL_NO_ERROR else err
        loglike[bid] = ll if err == SCALAR_LL_NO_ERROR else Float32(0.0)
        P[bid] = Q[bid]


def launch_scalar_ll[CAPTURE: Bool = False](
    ctx: DeviceContext,
    y: DeviceBuffer[DType.float32], T: DeviceBuffer[DType.float32],
    Q: DeviceBuffer[DType.float32], mut P: DeviceBuffer[DType.float32],
    alpha: DeviceBuffer[DType.float32], mu: DeviceBuffer[DType.float32],
    mut log_parts: DeviceBuffer[DType.float32], mut quad_parts: DeviceBuffer[DType.float32],
    mut error_parts: DeviceBuffer[DType.float32], mut loglike: DeviceBuffer[DType.float32],
    mut info: DeviceBuffer[DType.int32], mut capture: DeviceBuffer[DType.float32],
    nobs: Int, batch: Int, intercept: Int,
) raises:
    if nobs < 1 or nobs > SCALAR_LL_MAX_OBS or batch < 1:
        raise Error("K3 scalar likelihood: unsupported dimensions")
    var chunks = (nobs + SCALAR_LL_TPB - 1) // SCALAR_LL_TPB
    ctx.enqueue_function[scalar_ll_parts_kernel[CAPTURE]](
        y.unsafe_ptr(), T.unsafe_ptr(), Q.unsafe_ptr(), P.unsafe_ptr(),
        alpha.unsafe_ptr(), mu.unsafe_ptr(), log_parts.unsafe_ptr(),
        quad_parts.unsafe_ptr(), error_parts.unsafe_ptr(), capture.unsafe_ptr(),
        Int32(nobs), Int32(batch), Int32(intercept),
        grid_dim=(batch * chunks, 1, 1), block_dim=(SCALAR_LL_TPB, 1, 1),
    )
    ctx.enqueue_function[scalar_ll_finish_kernel](
        log_parts.unsafe_ptr(), quad_parts.unsafe_ptr(), error_parts.unsafe_ptr(),
        Q.unsafe_ptr(), P.unsafe_ptr(), loglike.unsafe_ptr(), info.unsafe_ptr(),
        Int32(nobs), Int32(chunks), grid_dim=(batch, 1, 1), block_dim=(SCALAR_LL_TPB, 1, 1),
    )
