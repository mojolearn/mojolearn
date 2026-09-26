"""FAST-only SMO block solve with several working-set elements per thread
(Apple).

`smo_block_solve_kernel` gives every working-set element its own thread,
so a 1024-element working set is 32 simdgroups and every inner iteration's
barriers and cross-warp folds span all 32. Here `THREADS` threads each
hold `EPT` elements in registers (element `p = e * THREADS + tid`, so the
kernel-row loads stay coalesced); the extremum folds first inside the
thread over its elements, then through `fast_smo_reduce` over
`THREADS / 32` warps.

Every selection is the same total order as the one-thread-per-element
kernel -- value, then the smaller key (keys are distinct training indices
for real elements), then the element position -- and every per-element
update is the same expression in the same precision, so the solve returns
the same alpha, delta_alpha and counters bit for bit.
"""

from max.gpu import thread_idx
from std.math import inf
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from checks.numerics import ftz, identical_mul_add
from svm.impl.fast_smo_reduce import fast_argext, fast_argmin_argmax
from svm.impl.smo_sets import in_lower, in_upper

comptime _ETA_EPS = Float32(1.0e-12)


@always_inline
def _beats[
    MAX: Bool
](ov: Float32, ok: Int32, op: Int32, mv: Float32, mk: Int32, mp: Int32) -> Bool:
    var tie = ov == mv and (ok < mk or (ok == mk and op < mp))
    comptime if MAX:
        return ov > mv or tie
    else:
        return ov < mv or tie


def smo_block_solve_ept_kernel[
    THREADS: Int, EPT: Int
](
    y_array: MutPointer[Float32, MutAnyOrigin],
    n_train_in: Int32,
    alpha: MutPointer[Float32, MutAnyOrigin],
    n_ws_in: Int32,
    delta_alpha: MutPointer[Float32, MutAnyOrigin],
    f_array: MutPointer[Float32, MutAnyOrigin],
    kernel: MutPointer[Float32, MutAnyOrigin],
    ws_idx: MutPointer[Int32, MutAnyOrigin],
    C_vec: MutPointer[Float32, MutAnyOrigin],
    eps: Float32,
    return_buff: MutPointer[Float32, MutAnyOrigin],
    max_iter_in: Int32,
):
    """Launch ONE block of `THREADS` threads; `n_ws <= THREADS * EPT`."""
    comptime WSIZE = THREADS * EPT
    comptime FW = 2 * (THREADS // 32)
    var n_ws = Int(n_ws_in)
    var max_iter = Int(max_iter_in)
    var tid = Int(thread_idx.x)

    var Kd = stack_allocation[
        WSIZE, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    var sh_tmp = stack_allocation[
        2, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    var fa_v = stack_allocation[FW, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var fa_k = stack_allocation[FW, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var fa_t = stack_allocation[FW, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var fb_v = stack_allocation[FW, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var fb_k = stack_allocation[FW, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var fb_t = stack_allocation[FW, Scalar[DType.int32], address_space = AddressSpace.SHARED]()

    var idx = Array[Int, EPT](fill=0)
    var y = Array[Float32, EPT](fill=0.0)
    var f = Array[Float32, EPT](fill=0.0)
    var a = Array[Float32, EPT](fill=0.0)
    var C = Array[Float32, EPT](fill=0.0)
    var key = Array[Int32, EPT](fill=Int32(2147483647))
    var a_save = Array[Float32, EPT](fill=0.0)

    comptime for e in range(EPT):
        var p = e * THREADS + tid
        if p < n_ws:
            idx[e] = Int(ws_idx.unsafe_load(p))
            y[e] = y_array.unsafe_load(idx[e])
            f[e] = f_array.unsafe_load(idx[e])
            a[e] = alpha.unsafe_load(idx[e])
            C[e] = C_vec.unsafe_load(idx[e])
            Kd[p] = kernel.unsafe_load(p + p * n_ws)
            key[e] = Int32(idx[e])
        a_save[e] = a[e]
    if tid == 0:
        sh_tmp[0] = Float32(0.0)
        sh_tmp[1] = Float32(0.0)
    barrier()

    var n_iter = 0
    var diff_end = Float32(0.0)
    var pos_inf = inf[DType.float32]()
    var neg_inf = -inf[DType.float32]()

    while n_iter < max_iter:
        # the thread's own argmin over X_upper and argmax over X_lower
        var mv = pos_inf
        var mk = Int32(2147483647)
        var mp = Int32(2147483647)
        var xv = neg_inf
        var xk = Int32(2147483647)
        var xp = Int32(2147483647)
        comptime for e in range(EPT):
            var p = Int32(e * THREADS + tid)
            var live = Int(p) < n_ws
            var ft = pos_inf
            if live and in_upper(a[e], y[e], C[e]):
                ft = f[e]
            var fl = neg_inf
            if live and in_lower(a[e], y[e], C[e]):
                fl = f[e]
            if _beats[False](ft, key[e], p, mv, mk, mp):
                mv = ft
                mk = key[e]
                mp = p
            if _beats[True](fl, key[e], p, xv, xk, xp):
                xv = fl
                xk = key[e]
                xp = p
        var rf = fast_argmin_argmax[THREADS](
            mv, mk, mp, xv, xk, xp, fa_v, fa_k, fa_t
        )
        var f_u = rf[0]
        var u = Int(rf[1])
        var f_max = rf[2]

        var diff = ftz(f_max - f_u)
        if n_iter == 0:
            if tid == 0:
                return_buff.unsafe_store(0, diff)
            var d10 = ftz(Float32(0.1) * diff)
            diff_end = eps if eps > d10 else d10
        if diff < diff_end:
            break

        var Kui = Array[Float32, EPT](fill=0.0)
        var lv = neg_inf
        var lk = Int32(2147483647)
        var lp = Int32(2147483647)
        var Kdu = Kd[u]
        comptime for e in range(EPT):
            var p = e * THREADS + tid
            var ft = neg_inf
            if p < n_ws:
                Kui[e] = kernel.unsafe_load(u * n_ws + p)
                if f_u < f[e] and in_lower(a[e], y[e], C[e]):
                    var eta_ui = ftz(ftz(Kd[p] + Kdu) - ftz(Float32(2.0) * Kui[e]))
                    if eta_ui < _ETA_EPS:
                        eta_ui = _ETA_EPS
                    var d = ftz(f_u - f[e])
                    ft = ftz(ftz(d * d) / eta_ui)
            if _beats[True](ft, key[e], Int32(p), lv, lk, lp):
                lv = ft
                lk = key[e]
                lp = Int32(p)
        var res2 = fast_argext[THREADS, True](lv, lk, lp, fb_v, fb_k, fb_t)
        var l = Int(res2[1])

        # Update alpha: the owners of u and l publish their bounds
        comptime for e in range(EPT):
            var p = e * THREADS + tid
            if p == u:
                sh_tmp[0] = C[e] - a[e] if y[e] > Float32(0.0) else a[e]
            if p == l:
                var tmp_l = a[e] if y[e] > Float32(0.0) else C[e] - a[e]
                # Kui == Kul for this element
                var eta_ul = ftz(ftz(Kdu + Kd[l]) - ftz(Float32(2.0) * Kui[e]))
                if eta_ul < _ETA_EPS:
                    eta_ul = _ETA_EPS
                var q_l = ftz(ftz(f[e] - f_u) / eta_ul)
                sh_tmp[1] = tmp_l if tmp_l < q_l else q_l
        barrier()
        var tmp_u = sh_tmp[0]
        var tmp_l2 = sh_tmp[1]
        var q = tmp_u if tmp_u < tmp_l2 else tmp_l2
        comptime for e in range(EPT):
            var p = e * THREADS + tid
            if p < n_ws:
                if p == u:
                    a[e] = ftz(a[e] + q * y[e])
                if p == l:
                    a[e] = ftz(a[e] - q * y[e])
                var Kli = kernel.unsafe_load(l * n_ws + p)
                f[e] = ftz(identical_mul_add(q, ftz(Kui[e] - Kli), f[e]))
        if q == Float32(0.0):
            break
        n_iter += 1

    comptime for e in range(EPT):
        var p = e * THREADS + tid
        if p < n_ws:
            alpha.unsafe_store(idx[e], a[e])
            delta_alpha.unsafe_store(p, ftz(ftz(a[e] - a_save[e]) * y[e]))
    if tid == 0:
        return_buff.unsafe_store(1, Float32(n_iter))
