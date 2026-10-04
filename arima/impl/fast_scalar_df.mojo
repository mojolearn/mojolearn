# SPDX-License-Identifier: Apache-2.0
"""K3 compensated arithmetic experiment; private probe only, never dispatched
by product fit/predict. Default OFF, Apple FAST only. Private-kernel quality
PASS; production integration, actual fit/forecast and timing remain owed.

The held scalar candidate remains separate. This implements the fixed-step
reference9723 proposal: preserve low words through residuals, log, two-level
256-lane reductions and the LL difference. No changed initializer or model.
"""
from std.gpu import block_idx, thread_idx, block_dim
from std.math import isfinite
from std.memory import stack_allocation, bitcast
from std.sys import llvm_intrinsic
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST, pinned_mul_f32

# PRIVATE-KERNEL-QUALITY PASS / default OFF, 2026-10-04, source fe5df7ab0:
# arima-k3-df-gpu-q-v1.json:13/13groups PASS,4 controls PASS,
# degradation_allowance=0; fixed step0.0009765625, supplied-state hashes checked.
# Actual compensated GPU stages/retained-low-word gradients pass versus saved
# actual-main GPU errors; zero scored timings, promotion_authorized=false.
# product_dispatch=false: device Jones/initializer, optimizer, public fit and
# forecast quality, production integration and board timing are still owed.
# Earlier scalar K3 actual-tail source102e0d70a remains HOLD-quality:
# corrected product reduction still leaves12/13groups HOLD,49 worse gradient
# components of648 (345 changed components improve). No timing admission.
# The former r2 harness gradient mismatch was repaired; do not cite that
# harness defect to dismiss the corrected actual-gradient failures.
# Private stage/gradient gates now pass as above; this does not discharge
# the separate actual fit/forecast checks or permit default promotion.
# Reference-only K1/K3 studies do not authorize product defaults.
# See docs/apple-fast/ARIMA_K3_DF_REFERENCE_PLAN.txt and
# docs/apple-fast/ARIMA_K3_DF_GPU_PLAN.txt; original scalar hold stays separate.
comptime ARIMA_K3_DF_ON = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
    and is_defined["MOJOLEARN_ARIMA_FAST_K3_DF_PROBE"]()
)
comptime DF_TPB = 256
comptime DF_MAX_OBS = DF_TPB * DF_TPB
comptime DF_NO_ERROR = Int32(2147483647)


@always_inline
def rn_add(a: Float32, b: Float32) -> Float32:
    # No fast-math flags: retain the two-sum graph, not reassociated FAST sums.
    return llvm_intrinsic["llvm.fma.f32", Float32, has_side_effect=False](a, Float32(1), b)


@always_inline
def rn_fma(a: Float32, b: Float32, c: Float32) -> Float32:
    return llvm_intrinsic["llvm.fma.f32", Float32, has_side_effect=False](a, b, c)


struct DF(Copyable, Movable):
    var hi: Float32
    var lo: Float32

    def __init__(out self, hi: Float32, lo: Float32 = Float32(0)):
        self.hi = hi
        self.lo = lo


@always_inline
def two_sum(a: Float32, b: Float32) -> DF:
    var s = rn_add(a, b)
    var v = rn_add(s, -a)
    return DF(s, rn_add(rn_add(a, -rn_add(s, -v)), rn_add(b, -v)))


@always_inline
def quick(a: Float32, b: Float32) -> DF:
    var s = rn_add(a, b)
    return DF(s, rn_add(b, -rn_add(s, -a)))


@always_inline
def df_add(a: DF, b: DF) -> DF:
    var s = two_sum(a.hi, b.hi)
    var t = two_sum(a.lo, b.lo)
    var u = quick(s.hi, rn_add(s.lo, t.hi))
    return quick(u.hi, rn_add(t.lo, u.lo))


@always_inline
def df_neg(a: DF) -> DF:
    return DF(-a.hi, -a.lo)


@always_inline
def df_sub(a: DF, b: DF) -> DF:
    return df_add(a, df_neg(b))


@always_inline
def df_mul(a: DF, b: DF) -> DF:
    var p = pinned_mul_f32(a.hi, b.hi)
    var e = rn_fma(a.hi, b.hi, -p)
    e = rn_fma(a.hi, b.lo, e)
    e = rn_fma(a.lo, b.hi, e)
    return quick(p, e)


@always_inline
def df_div(a: DF, b: DF) -> DF:
    var q1 = a.hi / b.hi
    var r = df_sub(a, df_mul(b, DF(q1)))
    var q2 = r.hi / b.hi
    r = df_sub(r, df_mul(b, DF(q2)))
    var q3 = r.hi / b.hi
    return df_add(quick(q1, q2), DF(q3))


@always_inline
def df_round(a: DF) -> Float32:
    return rn_add(a.hi, a.lo)


@always_inline
def df_log_positive(x: Float32) -> DF:
    # Exact exponent/mantissa extraction (frexp equivalent). Caller checks
    # positive finite variance first. Scale subnormals by 2^24; Metal FTZ
    # behavior is an owed gate, not an assumption of reference equivalence.
    var v = x
    var offset = 0
    var bits = bitcast[DType.uint32](v)
    if (bits & UInt32(0x7f800000)) == UInt32(0):
        v = pinned_mul_f32(v, Float32(16777216))
        offset = -24
        bits = bitcast[DType.uint32](v)
    var exponent = Int((bits >> 23) & UInt32(255)) - 126 + offset
    var mantissa = bitcast[DType.float32]((bits & UInt32(0x007fffff)) | UInt32(0x3f000000))
    if mantissa < Float32(0.7071067811865476):
        mantissa = pinned_mul_f32(mantissa, Float32(2))
        exponent -= 1
    var z = df_div(df_sub(DF(mantissa), DF(Float32(1))), df_add(DF(mantissa), DF(Float32(1))))
    var z2 = df_mul(z, z)
    var term = z.copy()
    var total = z.copy()
    for denominator in range(3, 26, 2):
        term = df_mul(term, z2)
        total = df_add(total, df_div(term, DF(Float32(denominator))))
    var ln2 = DF(Float32(0.6931471805599453), Float32(-1.904654299957768e-9))
    return df_add(df_mul(total, DF(Float32(2))), df_mul(DF(Float32(exponent)), ln2))


def k3_df_parts_kernel(
    y: Pointer[Float32, ImmutAnyOrigin], state: Pointer[Float32, ImmutAnyOrigin],
    parts: MutPointer[Float32, MutAnyOrigin], capture: MutPointer[Float32, MutAnyOrigin],
    nobs_in: Int32, batch_in: Int32, intercept_in: Int32,
):
    var nobs = Int(nobs_in)
    var batch = Int(batch_in)
    var chunks = (nobs + DF_TPB - 1) // DF_TPB
    var group = Int(block_idx.x)
    var bid = group // chunks
    var tid = Int(thread_idx.x)
    var t = (group % chunks) * DF_TPB + tid
    var count = batch * chunks
    var lh = stack_allocation[DF_TPB, Float32, address_space=AddressSpace.SHARED]()
    var ll = stack_allocation[DF_TPB, Float32, address_space=AddressSpace.SHARED]()
    var qh = stack_allocation[DF_TPB, Float32, address_space=AddressSpace.SHARED]()
    var ql = stack_allocation[DF_TPB, Float32, address_space=AddressSpace.SHARED]()
    var es = stack_allocation[DF_TPB, Int32, address_space=AddressSpace.SHARED]()
    var logs = DF(Float32(0))
    var square = DF(Float32(0))
    var error = DF_NO_ERROR
    if t < nobs:
        var pos = bid * nobs + t
        var prediction = DF(state[3 * batch + bid])
        var variance = state[2 * batch + bid]
        if t > 0:
            prediction = df_mul(DF(state[bid]), DF(y[pos - 1]))
            if intercept_in != 0:
                prediction = df_add(prediction, DF(state[4 * batch + bid]))
            variance = state[batch + bid]
        var residual = df_sub(DF(y[pos]), prediction)
        var total = batch * nobs
        capture[pos] = df_round(prediction)
        capture[total + pos] = df_round(residual)
        capture[2 * total + pos] = variance
        if variance <= Float32(0) or not isfinite(variance) or not isfinite(df_round(residual)):
            error = Int32(t + 1)
        else:
            logs = df_log_positive(variance)
            square = df_div(df_mul(residual, residual), DF(variance))
            if not isfinite(df_round(logs)) or not isfinite(df_round(square)):
                error = Int32(t + 1)
                logs = DF(Float32(0))
                square = DF(Float32(0))
    lh[tid], ll[tid], qh[tid], ql[tid], es[tid] = logs.hi, logs.lo, square.hi, square.lo, error
    barrier()
    var width = DF_TPB // 2
    while width > 0:
        if tid < width:
            var l = df_add(DF(lh[tid], ll[tid]), DF(lh[tid + width], ll[tid + width]))
            var q = df_add(DF(qh[tid], ql[tid]), DF(qh[tid + width], ql[tid + width]))
            lh[tid], ll[tid], qh[tid], ql[tid] = l.hi, l.lo, q.hi, q.lo
            es[tid] = min(es[tid], es[tid + width])
        barrier()
        width //= 2
    if tid == 0:
        parts[group] = lh[0]
        parts[count + group] = ll[0]
        parts[2 * count + group] = qh[0]
        parts[3 * count + group] = ql[0]
        parts[4 * count + group] = Float32(0) if es[0] == DF_NO_ERROR else Float32(es[0])


def k3_df_finish_kernel(
    parts: Pointer[Float32, ImmutAnyOrigin], state: Pointer[Float32, ImmutAnyOrigin],
    stats: MutPointer[Float32, MutAnyOrigin], info: MutPointer[Int32, MutAnyOrigin],
    nobs_in: Int32, batch_in: Int32,
):
    var bid = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var batch = Int(batch_in)
    var nobs = Int(nobs_in)
    var chunks = (nobs + DF_TPB - 1) // DF_TPB
    var count = batch * chunks
    var lh = stack_allocation[DF_TPB, Float32, address_space=AddressSpace.SHARED]()
    var ll = stack_allocation[DF_TPB, Float32, address_space=AddressSpace.SHARED]()
    var qh = stack_allocation[DF_TPB, Float32, address_space=AddressSpace.SHARED]()
    var ql = stack_allocation[DF_TPB, Float32, address_space=AddressSpace.SHARED]()
    var es = stack_allocation[DF_TPB, Int32, address_space=AddressSpace.SHARED]()
    lh[tid], ll[tid], qh[tid], ql[tid], es[tid] = Float32(0), Float32(0), Float32(0), Float32(0), DF_NO_ERROR
    if tid < chunks:
        var p = bid * chunks + tid
        lh[tid], ll[tid], qh[tid], ql[tid] = parts[p], parts[count + p], parts[2 * count + p], parts[3 * count + p]
        if parts[4 * count + p] > Float32(0):
            es[tid] = Int32(parts[4 * count + p])
    barrier()
    var width = DF_TPB // 2
    while width > 0:
        if tid < width:
            var l = df_add(DF(lh[tid], ll[tid]), DF(lh[tid + width], ll[tid + width]))
            var q = df_add(DF(qh[tid], ql[tid]), DF(qh[tid + width], ql[tid + width]))
            lh[tid], ll[tid], qh[tid], ql[tid] = l.hi, l.lo, q.hi, q.lo
            es[tid] = min(es[tid], es[tid + width])
        barrier()
        width //= 2
    if tid == 0:
        var n = DF(Float32(nobs))
        var log2pi = DF(Float32(1.8378770664093453), Float32(3.1268354230284965e-8))
        var value = df_mul(DF(Float32(-0.5)), df_add(df_mul(n, df_add(df_div(DF(qh[0], ql[0]), n), log2pi)), DF(lh[0], ll[0])))
        var error = es[0]
        if not isfinite(df_round(value)) and error == DF_NO_ERROR:
            error = Int32(nobs)
        info[bid] = Int32(0) if error == DF_NO_ERROR else error
        if error != DF_NO_ERROR:
            value = DF(Float32(0))
        stats[bid] = df_round(value)
        stats[batch + bid] = state[batch + bid]
        stats[2 * batch + bid] = value.hi
        stats[3 * batch + bid] = value.lo


def k3_df_gradient_kernel(
    stats: Pointer[Float32, ImmutAnyOrigin], info: Pointer[Int32, ImmutAnyOrigin],
    gradients: MutPointer[Float32, MutAnyOrigin], nobs_in: Int32, batch_in: Int32,
):
    # Probe input is model-major [base, raw0+h, raw1+h, raw2+h]. Low LL
    # words stay resident. Same h=2^-10, subtraction then /h then /(nobs-1).
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var batch = Int(batch_in)
    if i >= (batch // 4) * 3:
        return
    var base = (i // 3) * 4
    var member = base + 1 + i % 3
    var valid = nobs_in > 1
    for j in range(4):
        valid = valid and info[base + j] == 0
    var result = Float32(0)
    if valid:
        var a = DF(stats[2 * batch + member], stats[3 * batch + member])
        var b = DF(stats[2 * batch + base], stats[3 * batch + base])
        result = df_round(df_div(df_neg(df_div(df_sub(a, b), DF(Float32(0.0009765625)))), DF(Float32(nobs_in - 1))))
    gradients[i] = result
