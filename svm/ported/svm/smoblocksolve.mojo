"""`SmoBlockSolve`: one block solves the working-set QP by SMO.

PORT OF `cuml/cpp/src/svm/smoblocksolve.cuh` at cuML v26.08.00, the kernel
body transcribed branch for branch (`:154-271`). The math is documented in
their header and is not repeated here; what follows is what changed.

    typedef cub::BlockReduce<Pair, WSIZE>    -> pinned_block_argmin/argmax
    typedef cub::BlockReduce<math_t, WSIZE>  -> pinned_block_max_all
    __shared__ f_u, u, l, tmp_u, tmp_l, diff, diff_end  -> threadgroup slots
    __shared__ Kd[WSIZE]                     -> threadgroup slab

THE THREE PINS (svm/README.md, identity content section 3):

# =========================================================================
# DEVIATION 633: the two arg-reductions tie-break on the TRAINING INDEX
# (`ws_idx[tid]`, smaller wins). Theirs compares `KVPair::val` only
# (`kselection.cuh:66-79`, "@todo ... consider the key when values are the
# same?"), so the winner among equal f is whatever CUB's fold keeps. Equal
# f is ordinary (duplicate rows; every alpha at 0 on the first iteration
# gives f = -y, two values over the whole set), so this is reached on every
# fit's first inner iteration. Gated by the duplicated-rows fixture against
# the host oracle, which selects the same way in serial.
# =========================================================================

The contraction `f += q * (Kui - Kli)` is `identical_mul_add(q, ftz(Kui -
Kli), f)` under IDENTICAL (their CUDA build contracts it to an fma; Metal
through MAX contracts too; the pin is for the backend that does not); the
quotient `(f_u - f)^2 / eta` is `ftz(ftz(d * d) / eta)`; `eta` is
`max(ftz(ftz(Kd_t + Kd_u) - ftz(2 * Kui)), ETA_EPS)`. Both helpers compile
away under FAST, so the FAST kernel is the plain expression and ONE body
serves both modes (the association is the same either way).

WSIZE is a comptime parameter because the threadgroup slabs are; theirs is
`SMO_WS_SIZE = 1024` at the one call site. `n_ws <= WSIZE` threads carry
data; the rest pass the identity to every reduction and do nothing else.
A selection over a total order cannot see WSIZE, and the launch-invariance
gate runs two values of it on the same problem.
"""

from std.gpu import thread_idx
from std.memory import stack_allocation
from std.math import inf
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from mojo_only.numerics import ftz, identical_mul_add
from svm.mojo_only.pinned_argreduce import (
    pinned_block_argmax,
    pinned_block_argmin,
    pinned_block_max_all,
)
from svm.ported.svm.smo_sets import in_lower, in_upper


#: `constexpr const int SMO_WS_SIZE = 1024` (`smosolver.cuh:116`).
comptime SMO_WS_SIZE = 1024

#: `constexpr math_t ETA_EPS = 1.0e-12` (`smoblocksolve.cuh:168`).
comptime ETA_EPS = Float32(1.0e-12)


def smo_block_solve_kernel[
    WSIZE: Int
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
    """`SmoBlockSolve<math_t, WSIZE><<<1, n_ws>>>(...)` for `svmType =
    C_SVC`. Launch ONE block of `WSIZE` threads (theirs launches `n_ws`
    threads of a `WSIZE`-sized reduce; here the padding threads are real
    and inert)."""
    var n_ws = Int(n_ws_in)
    var max_iter = Int(max_iter_in)
    var tid = Int(thread_idx.x)
    var active = tid < n_ws

    var Kd = stack_allocation[
        WSIZE, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    var sh_tmp = stack_allocation[
        2, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()

    var idx = 0
    var y = Float32(0.0)
    var f = Float32(0.0)
    var a = Float32(0.0)
    var C = Float32(0.0)
    # Padding threads carry the largest key, so an active thread always
    # wins a tie against them and `u`, `l` are real threads (theirs: the
    # pair's key IS a real tid by construction).
    var key = Int32(2147483647)
    if active:
        idx = Int(ws_idx.unsafe_load(tid))
        # store values in registers
        y = y_array.unsafe_load(idx)
        f = f_array.unsafe_load(idx)
        a = alpha.unsafe_load(idx)
        C = C_vec.unsafe_load(idx)
        Kd[tid] = kernel.unsafe_load(tid + tid * n_ws)
        key = Int32(idx)
    var a_save = a
    if tid == 0:
        sh_tmp[0] = Float32(0.0)
        sh_tmp[1] = Float32(0.0)
    barrier()

    var n_iter = 0
    var diff_end = Float32(0.0)
    var pos_inf = inf[DType.float32]()
    var neg_inf = -inf[DType.float32]()

    while n_iter < max_iter:
        # mask values outside of X_upper
        var f_tmp = pos_inf
        if active and in_upper(a, y, C):
            f_tmp = f
        var res = pinned_block_argmin[WSIZE](f_tmp, key)
        var f_u = res[0]
        var u_key = res[1]
        # `u` is the THREAD holding the winning (value, key); theirs keeps
        # the thread id as the pair's key. One ballot through threadgroup
        # memory recovers it from the training index (keys are unique).
        if active and key == u_key:
            sh_tmp[0] = Float32(tid)
        barrier()
        var u = Int(sh_tmp[0])
        barrier()

        # select f_max to check stopping condition
        f_tmp = neg_inf
        if active and in_lower(a, y, C):
            f_tmp = f
        var Kui = Float32(0.0)
        if active:
            Kui = kernel.unsafe_load(u * n_ws + tid)
        var f_max = pinned_block_max_all[WSIZE](f_tmp)

        # f_max - f_u is used to check stopping condition.
        var diff = ftz(f_max - f_u)
        if n_iter == 0:
            if tid == 0:
                return_buff.unsafe_store(0, diff)
            var d10 = ftz(Float32(0.1) * diff)
            diff_end = eps if eps > d10 else d10
        if diff < diff_end:
            break

        if active and f_u < f and in_lower(a, y, C):
            var eta_ui = ftz(ftz(Kd[tid] + Kd[u]) - ftz(Float32(2.0) * Kui))
            if eta_ui < ETA_EPS:
                eta_ui = ETA_EPS
            var d = ftz(f_u - f)
            f_tmp = ftz(ftz(d * d) / eta_ui)
        else:
            f_tmp = neg_inf
        var res2 = pinned_block_argmax[WSIZE](f_tmp, key)
        var l_key = res2[1]
        if active and key == l_key:
            sh_tmp[0] = Float32(tid)
        barrier()
        var l = Int(sh_tmp[0])
        barrier()
        var Kli = Float32(0.0)
        if active:
            Kli = kernel.unsafe_load(l * n_ws + tid)

        # Update alpha (the clipping argument is in their comment block)
        if tid == u:
            sh_tmp[0] = C - a if y > Float32(0.0) else a
        if tid == l:
            var tmp_l = a if y > Float32(0.0) else C - a
            # note: Kui == Kul for this thread
            var eta_ul = ftz(ftz(Kd[u] + Kd[l]) - ftz(Float32(2.0) * Kui))
            if eta_ul < ETA_EPS:
                eta_ul = ETA_EPS
            var q_l = ftz(ftz(f - f_u) / eta_ul)
            sh_tmp[1] = tmp_l if tmp_l < q_l else q_l
        barrier()
        var tmp_u = sh_tmp[0]
        var tmp_l2 = sh_tmp[1]
        var q = tmp_u if tmp_u < tmp_l2 else tmp_l2
        barrier()
        if tid == u:
            a = ftz(a + q * y)
        if tid == l:
            a = ftz(a - q * y)
        f = ftz(identical_mul_add(q, ftz(Kui - Kli), f))
        if q == Float32(0.0):
            # Probably fp underflow
            break
        n_iter += 1

    # save results to global memory before exit
    if active:
        alpha.unsafe_store(idx, a)
        # it is actually y * \Delta \alpha
        delta_alpha.unsafe_store(tid, ftz(ftz(a - a_save) * y))
    # f is recalculated in f_update, therefore we do not need to save that
    if tid == 0:
        return_buff.unsafe_store(1, Float32(n_iter))
