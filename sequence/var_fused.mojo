# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane/apple-fast-w4-small (-D MOJOLEARN_SEQ_FAST_VAR_FUSED; FAST + Apple
only, device binding only): VAR's fit as ONE launch and its forecast with
one upload.

At the board's shape (n = 1,392, K = 16, p = 2: m = 33, R = 1,390) the
TSA2_VAR fit (sequence/pyapi.mojo::_var_fit_queued) is all fixed cost:
one upload, one workspace fill, eight launches (design, column scale, two
GEMMs, the threadgroup Cholesky, residuals, sigma_u, row scale), one
download and one wait; the M3 binding call measured 2.74 ms of the
2.79 ms fit. Every Metal command is a fixed cost at this size, so here the
whole fit is one threadgroup of VFUSED_TPB threads in one kernel:

  1. each design column's largest magnitude (partial maxima, then the
     ordered max of the partials: the same value as op_colscale's one
     loop, a max does not depend on the order) and its power-of-two scale;
  2. the scaled design Z and the target Ys (op_var_design then op_colscale:
     Z = st(mul(v, s)), the same words);
  3. Z'Z and Z'Ys (op_gemm's chain through `gemm_dot`), straight into
     threadgroup memory as the words `st` would have stored;
  4. the Cholesky factor and the K solves (var_chol_block_kernel's code and
     chains, sequence/vecar_block.mojo);
  5. the residuals (op_var_resid's chain) and the coefficients scaled back
     (op_rowscale), then a device-memory barrier;
  6. sigma_u (op_var_sigma's chain).

Device-memory hand-offs use `team_barrier` (on Apple
`threadgroup_barrier(mem_device | mem_threadgroup)`, x_linear/team.mojo).
No workspace fill (every word read is written first), one upload (y), one
download (params | resid | sigma_u | status, as SEQ_FAST_VAR_ONECOPY), one
wait. EVERY OUTPUT WORD'S CHAIN IS THE QUEUED FIT'S, so the outputs are
byte-identical to main's FAST + Apple fit.

The forecast: y's last rows and params go up in ONE staged copy (main: two
uploads), the output is not zero-filled first, then main's
var_forecast_block_kernel and one download: three commands instead of five,
the same words.

Shapes outside the gate (R >= 32768, where FAST's GEMM splits K, or
m*m + m*K > VAR_SMEM) return -1 and take main's path.
"""
from std.memory import bitcast, memcpy, stack_allocation
from std.gpu import thread_idx
from std.python import PythonObject
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from checks.numerics import ftz, identical_div, identical_sqrt
from sequence.ops import FP, Args, OP_VAR_FORECAST, fma3, gemm_dot, ld, mul, st, sub
from sequence.vecar import pow2_scale
from sequence.vecar_block import VAR_SMEM
from sequence.exec_device import DeviceExec
from sequence.pyapi import fptr, ival
from x_linear.team import team_barrier

#: threads of the one group
comptime VFUSED_TPB = 256


@always_inline
def _div(a: Float32, b: Float32) -> Float32:
    return ftz(identical_div(a, b))


def var_fit_fused_kernel(
    y: FP, Z: FP, Ys: FP, Bm: FP, Rs: FP, S: FP, status: FP,
    K_in: Int32, p_in: Int32, kt_in: Int32, m_in: Int32, R_in: Int32, inv_bits: Int32,
):
    """The whole TSA2_VAR fit on one threadgroup (module docstring). y
    [n, K]; Z [R, m], Ys [R, K] scratch; Bm [m, K], Rs [R, K], S [K, K],
    status [1] the outputs (Bm | Rs | S | status contiguous)."""
    var tid = Int(thread_idx.x)
    var nt = VFUSED_TPB
    var K = Int(K_in)
    var p = Int(p_in)
    var kt = Int(kt_in)
    var m = Int(m_in)
    var R = Int(R_in)
    var inv = bitcast[DType.float32](inv_bits)
    var mm = m * m
    var sh = stack_allocation[VAR_SMEM, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var pm = stack_allocation[VFUSED_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var sc = stack_allocation[64, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var flag = stack_allocation[1, Scalar[DType.int32], address_space = AddressSpace.SHARED]()

    # 1. column maxima: m columns x nch row chunks (m <= 64 <= nt)
    var nch = nt // m
    if nch < 1:
        nch = 1
    var rc = (R + nch - 1) // nch
    if tid < m * nch:
        var c = tid // nch
        var ch = tid - c * nch
        var r0 = ch * rc
        var r1 = min(R, r0 + rc)
        var mx = Float32(0.0)
        if c < kt:
            if r1 > r0:
                mx = Float32(1.0)
        else:
            var j = c - kt
            var lag = j // K + 1
            var vv = j - (lag - 1) * K
            for r in range(r0, r1):
                var v = abs(ld(y, (p + r - lag) * K + vv))
                if v > mx:
                    mx = v
        pm[tid] = mx
    if tid == 0:
        flag[0] = Int32(0)
    barrier()
    if tid < m:
        var mx = Float32(0.0)
        for ch in range(nch):
            var v = pm[tid * nch + ch]
            if v > mx:
                mx = v
        sc[tid] = pow2_scale(mx)
    barrier()

    # 2. the scaled design and the target
    var t = tid
    while t < R * m:
        var r = t // m
        var c = t - r * m
        var v: Float32
        if c < kt:
            v = Float32(1.0)
        else:
            var j = c - kt
            var lag = j // K + 1
            var vv = j - (lag - 1) * K
            v = ld(y, (p + r - lag) * K + vv)
        st(Z, t, mul(v, sc[c]))
        if c < K:
            st(Ys, r * K + c, ld(y, (p + r) * K + c))
        t += nt
    team_barrier()

    # 3. G = Z'Z into sh[0, mm), Z'Ys into sh[mm, mm + m K)
    t = tid
    while t < mm + m * K:
        if t < mm:
            var i = t // m
            var j = t - i * m
            sh[t] = ftz(gemm_dot(Z, i, m, Z, j, m, R, Float32(0.0)))
        else:
            var u = t - mm
            var i = u // K
            var c = u - i * K
            sh[t] = ftz(gemm_dot(Z, i, m, Ys, c, K, R, Float32(0.0)))
        t += nt
    barrier()

    # 4. var_chol_block_kernel's factor and solves
    for j in range(m):
        if tid == 0:
            var d = ftz(sh[j * m + j])
            for k in range(j):
                var l = ftz(sh[j * m + k])
                d = sub(d, mul(l, l))
            if not (d > Float32(0.0)):
                flag[0] = Int32(1 + j)
            else:
                sh[j * m + j] = ftz(ftz(identical_sqrt(d)))
        barrier()
        if flag[0] != Int32(0):
            if tid == 0:
                st(status, 0, Float32(Int(flag[0])))
            return
        var ljj = ftz(sh[j * m + j])
        var i = j + 1 + tid
        while i < m:
            var v = ftz(sh[i * m + j])
            for k in range(j):
                v = sub(v, mul(ftz(sh[i * m + k]), ftz(sh[j * m + k])))
            sh[i * m + j] = ftz(_div(v, ljj))
            i += nt
        barrier()
    var c = tid
    while c < K:
        for i in range(m):
            var v = ftz(sh[mm + i * K + c])
            for k in range(i):
                v = sub(v, mul(ftz(sh[i * m + k]), ftz(sh[mm + k * K + c])))
            sh[mm + i * K + c] = ftz(_div(v, ftz(sh[i * m + i])))
        var i = m - 1
        while i >= 0:
            var v = ftz(sh[mm + i * K + c])
            for k in range(i + 1, m):
                v = sub(v, mul(ftz(sh[k * m + i]), ftz(sh[mm + k * K + c])))
            sh[mm + i * K + c] = ftz(_div(v, ftz(sh[i * m + i])))
            i -= 1
        c += nt
    barrier()

    # 5. residuals (op_var_resid: gemm_dot's ascending fma chain, then one
    #    sub) and the coefficients scaled back (op_rowscale)
    t = tid
    while t < R * K:
        var r = t // K
        var cc = t - r * K
        var acc = Float32(0.0)
        for k in range(m):
            acc = fma3(ld(Z, r * m + k), ftz(sh[mm + k * K + cc]), acc)
        st(Rs, t, sub(ld(Ys, t), ftz(acc)))
        t += nt
    t = tid
    while t < m * K:
        var r = t // K
        st(Bm, t, mul(ftz(sh[mm + t]), ftz(sc[r])))
        t += nt
    if tid == 0:
        st(status, 0, Float32(0.0))
    team_barrier()

    # 6. sigma_u (op_var_sigma)
    t = tid
    while t < K * K:
        var i = t // K
        var j = t - i * K
        var s = gemm_dot(Rs, i, K, Rs, j, K, R, Float32(0.0))
        st(S, t, mul(ftz(s), inv))
        t += nt


def var_fit_fused_py(mut ex: DeviceExec, addrs: PythonObject, ip: PythonObject) raises -> Int:
    """`var_fit_py`'s contract (addrs = [y, params, sigma_u, resid],
    ip = [n, K, p, k_trend]; 0 or 1 + the failed pivot column) on one
    launch; -1 when this entry does not serve the call (invalid arguments
    included: main's entry then raises its own message)."""
    if len(addrs) != 4 or len(ip) != 4:
        return -1
    var n = ival(ip, 0)
    var K = ival(ip, 1)
    var p = ival(ip, 2)
    var kt = ival(ip, 3)
    if K < 1 or p < 1 or kt < 0 or kt > 1:
        return -1
    var R = n - p
    var m = kt + K * p
    if R - m < 1 or R >= 32768 or m * m + m * K > VAR_SMEM or m > 64:
        return -1
    var span = m * K + R * K + K * K + 1
    var ws = ex._alloc(n * K + R * m + R * K + span, False)
    ex.upload(ws, fptr(addrs[0], "y"), n * K)
    var Z = ws + n * K
    var Ys = Z + R * m
    var Bm = Ys + R * K
    var Rs = Bm + m * K
    var S = Rs + R * K
    var status = S + K * K
    var inv = Float32(1.0) / Float32(R - m)
    ex.ctx.enqueue_function[var_fit_fused_kernel](  # small-launch(m: design columns, at most 64): 256 threads split every cell, rows R below 32768 by the gate; one launch replaces eight fixed-cost ones
        ws, Z, Ys, Bm, Rs, S, status,
        Int32(K), Int32(p), Int32(kt), Int32(m), Int32(R), Int32(bitcast[DType.int32](inv)),
        grid_dim=(1, 1, 1), block_dim=(VFUSED_TPB, 1, 1),
    )
    var tmp = List[Float32](length=span, fill=Float32(0.0))
    var tp = FP(unsafe_from_address=Int(tmp.unsafe_ptr()))
    ex.download_async(tp, Bm, span)
    ex.sync()
    memcpy(dest=fptr(addrs[1], "params"), src=tp, count=m * K)
    memcpy(dest=fptr(addrs[3], "resid"), src=tp + m * K, count=R * K)
    memcpy(dest=fptr(addrs[2], "sigma_u"), src=tp + m * K + R * K, count=K * K)
    var code = Int(tp.unsafe_load(m * K + R * K + K * K))
    _ = tmp^
    return code


def var_forecast_fused_py(mut ex: DeviceExec, addrs: PythonObject, ip: PythonObject) raises -> Int:
    """`var_forecast_py`'s contract (addrs = [y_last (p, K), params (m, K),
    out (h, K)], ip = [K, p, k_trend, h]) with ONE upload of y | params
    and no fill of the output; -1 when not served."""
    if len(addrs) != 3 or len(ip) != 4:
        return -1
    var K = ival(ip, 0)
    var p = ival(ip, 1)
    var kt = ival(ip, 2)
    var h = ival(ip, 3)
    if K < 1 or p < 1 or kt < 0 or kt > 1 or h < 1:
        return -1
    var m = kt + K * p
    if (p + h) * K > VAR_SMEM:
        return -1
    var nin = p * K + m * K
    var tmp = List[Float32](length=nin, fill=Float32(0.0))
    var tp = FP(unsafe_from_address=Int(tmp.unsafe_ptr()))
    memcpy(dest=tp, src=fptr(addrs[0], "y"), count=p * K)
    memcpy(dest=tp + p * K, src=fptr(addrs[1], "params"), count=m * K)
    var buf = ex._alloc(nin + h * K, False)
    ex.upload(buf, tp, nin)   # staged at once: tmp may go after this
    _ = tmp^
    var a = Args()
    a.p0 = buf
    a.p1 = buf + p * K
    a.p2 = buf + nin
    a.i0 = K
    a.i1 = p
    a.i2 = kt
    a.i3 = h
    ex.launch[OP_VAR_FORECAST](a, 1)   # var_forecast_block_kernel under TSA2_VAR
    ex.download(fptr(addrs[2], "out"), buf + nin, h * K)
    return h * K
