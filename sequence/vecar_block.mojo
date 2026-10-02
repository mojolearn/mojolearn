# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""VAR's two one-thread operations as ONE THREADGROUP each (lane gap-prep2,
2026-10-02). `op_cholsolve` (sequence/vecar.mojo) factored Z'Z and solved
every equation on one thread, and `op_var_forecast` ran the whole recursion
on one thread: at the board's shape (K = 16, p = 2, m = 33, h = 48) about
23k and 25k dependent steps, each reading device memory, the M3's 19.5 ms
VAR cell against 2.5 ms on the L40S.

Here the operands live in threadgroup memory and the threads split the
independent cells:
  * Cholesky, column j: thread 0 forms the pivot (its chain of k < j), a
    barrier, then the rows i > j split over the threads (each row's own
    chain of k < j, then the quotient by the pivot), a barrier. A
    non-positive pivot sets the status word and every thread leaves.
  * the solve: the K equations (columns of B) split over the threads, each
    its own forward and backward substitution.
  * the forecast, step s: the K outputs split over the threads, each its
    own sum (trend, then lag 1's K terms, then lag 2's, ...), a barrier.
EVERY CELL'S CHAIN IS THE ONE-THREAD OP'S: the same flushed operands (`ld`
reads a word through `ftz`, here the threadgroup copy of that word is read
through `ftz`), the same order, the same `sub`/`mul`/`div`/`fma3`; the
words written are the same, so the device and the host column agree.
Shapes past VAR_SMEM words take the one-thread ops (DeviceExec.launch).
"""
from std.gpu import block_dim, thread_idx
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from checks.numerics import ftz, identical_div, identical_sqrt
from sequence.ops import FP, fma3, ld, mul, st, sub

#: threadgroup words either kernel stages (16 KB)
comptime VAR_SMEM = 4096
#: threads of the one group
comptime VAR_TPB = 128


@always_inline
def _div(a: Float32, b: Float32) -> Float32:
    return ftz(identical_div(a, b))


def var_chol_block_kernel(G: FP, Bm: FP, status: FP, m_in: Int32, K_in: Int32):
    """`op_cholsolve` on one threadgroup: G [m, m] (lower factor, in place),
    Bm [m, K] (the solutions, in place), status[0] = 0 or 1 + the column of
    the first non-positive pivot. m*m + m*K <= VAR_SMEM."""
    var tid = Int(thread_idx.x)
    var nt = Int(block_dim.x)
    var m = Int(m_in)
    var K = Int(K_in)
    var mm = m * m
    var sh = stack_allocation[VAR_SMEM, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var flag = stack_allocation[1, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var q = tid
    while q < mm:
        sh[q] = G.unsafe_load(q)
        q += nt
    q = tid
    while q < m * K:
        sh[mm + q] = Bm.unsafe_load(q)
        q += nt
    if tid == 0:
        flag[0] = Int32(0)
    barrier()
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
        # forward: L z = b
        for i in range(m):
            var v = ftz(sh[mm + i * K + c])
            for k in range(i):
                v = sub(v, mul(ftz(sh[i * m + k]), ftz(sh[mm + k * K + c])))
            sh[mm + i * K + c] = ftz(_div(v, ftz(sh[i * m + i])))
        # backward: L' x = z
        var i = m - 1
        while i >= 0:
            var v = ftz(sh[mm + i * K + c])
            for k in range(i + 1, m):
                v = sub(v, mul(ftz(sh[k * m + i]), ftz(sh[mm + k * K + c])))
            sh[mm + i * K + c] = ftz(_div(v, ftz(sh[i * m + i])))
            i -= 1
        c += nt
    barrier()
    q = tid
    while q < mm:
        G.unsafe_store(q, sh[q])
        q += nt
    q = tid
    while q < m * K:
        Bm.unsafe_store(q, sh[mm + q])
        q += nt
    if tid == 0:
        st(status, 0, Float32(0.0))


def var_forecast_block_kernel(Y: FP, P: FP, OUT: FP, K_in: Int32, p_in: Int32, kt_in: Int32, h_in: Int32):
    """`op_var_forecast` on one threadgroup: OUT [h, K] from the last p rows
    Y [p, K] under params P [m, K]. Threadgroup memory holds Y's rows then
    each step's output, (p + h) * K <= VAR_SMEM words."""
    var tid = Int(thread_idx.x)
    var nt = Int(block_dim.x)
    var K = Int(K_in)
    var p = Int(p_in)
    var kt = Int(kt_in)
    var h = Int(h_in)
    var sh = stack_allocation[VAR_SMEM, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var q = tid
    while q < p * K:
        sh[q] = Y.unsafe_load(q)
        q += nt
    barrier()
    for s in range(h):
        var j = tid
        while j < K:
            var acc = Float32(0.0)
            if kt == 1:
                acc = ld(P, j)
            for lag in range(1, p + 1):
                var row = (p + s - lag) * K
                for v in range(K):
                    acc = fma3(ld(P, (kt + (lag - 1) * K + v) * K + j), ftz(sh[row + v]), acc)
            sh[(p + s) * K + j] = ftz(acc)
            st(OUT, s * K + j, acc)
            j += nt
        barrier()
