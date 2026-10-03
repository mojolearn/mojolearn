# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Row-parallel Lasso CD with a 32-thread group per row, FAST on Apple
(lane/apple-fast-gap-clus3, 2026-10-03). Switch: `DECOMP_FAST_LASSO_GRP`,
`-D MOJOLEARN_DECOMP_FAST_LASSO_GRP` (default off until the M3 A/B); taken by
x_decomp/device.mojo `DevExec.lasso_rows`. IDENTICAL compiles none of this.

Cause: `lasso_rows_kernel` is a thread per row (`cells.lasso_row`): at
MiniBatchDictionaryLearning's batch (256 rows) the whole launch is 2 blocks
of the M3 Ultra's 80 cores, and every coordinate update of every sweep is a
k-long read-modify-write of the row's H strip in DEVICE memory (rows 4k
bytes apart), serial in one thread; the fit runs it once a step, up to
3,910 steps on Istella.

Here a row is a 32-thread block: G (k x k), the row's w, q and H = G w live
in threadgroup memory, every thread computes the same coordinate update
(uniform control flow, so the barriers are uniform) and thread l applies
it to H[l], H[l + 32], ... `cells.lasso_row`'s sequence otherwise
(coordinates ascending, the -w_j G[:, j] and + w_new G[:, j] steps, the
same stop rule), so the iterates are its words. k <= LG_MAXK, else the old
kernel.
"""
from std.gpu import block_idx, thread_idx
from std.memory import stack_allocation
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST, ftz, identical_mul_add
from x_decomp.cells import F32Ptr, div0, mul, sub

comptime DECOMP_FAST_LASSO_GRP = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
    and is_defined["MOJOLEARN_DECOMP_FAST_LASSO_GRP"]()
)
comptime LG_TPB = 32
comptime LG_MAXK = 64


def lasso_grp_kernel(
    g: F32Ptr, q: F32Ptr, w: F32Ptr, its: F32Ptr, n: Int32, k: Int32, alpha: Float32,
    max_iter: Int32, tol: Float32, positive: Int32,
):
    var i = Int(block_idx.x)
    var l = Int(thread_idx.x)
    var K = Int(k)
    var sG = stack_allocation[LG_MAXK * LG_MAXK, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var sH = stack_allocation[LG_MAXK, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var sW = stack_allocation[LG_MAXK, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var sQ = stack_allocation[LG_MAXK, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    if i >= Int(n):
        return
    var base = i * K
    for t in range(l, K * K, LG_TPB):
        sG[t] = g[t]
    for t in range(l, K, LG_TPB):
        sW[t] = w[base + t]
        sQ[t] = q[base + t]
    barrier()
    for j in range(l, K, LG_TPB):
        var acc = Float32(0)
        for c in range(K):
            acc = ftz(identical_mul_add(ftz(sG[j * K + c]), ftz(sW[c]), acc))
        sH[j] = acc
    barrier()
    var pos = positive != 0
    var it = 0
    for _ in range(Int(max_iter)):
        it += 1
        var w_max = Float32(0)
        var d_w_max = Float32(0)
        for j in range(K):
            var gjj = ftz(sG[j * K + j])
            if gjj == Float32(0):
                continue
            var w_j = ftz(sW[j])
            if w_j != Float32(0):
                for t in range(l, K, LG_TPB):
                    sH[t] = ftz(identical_mul_add(-w_j, ftz(sG[t * K + j]), ftz(sH[t])))
                barrier()
            var tmp = sub(sQ[j], sH[j])
            var nw = Float32(0)
            if pos:
                if tmp > Float32(0):
                    var m = sub(tmp, alpha)
                    nw = div0(m, gjj) if m > Float32(0) else Float32(0)
            else:
                var m = sub(abs(tmp), alpha)
                if m > Float32(0):
                    nw = div0(m if tmp > Float32(0) else -m, gjj)
            barrier()   # every thread has read sH[j] before it changes
            if l == 0:
                sW[j] = nw
            if nw != Float32(0):
                for t in range(l, K, LG_TPB):
                    sH[t] = ftz(identical_mul_add(nw, ftz(sG[t * K + j]), ftz(sH[t])))
            barrier()
            var dw = abs(sub(nw, w_j))
            if dw > d_w_max:
                d_w_max = dw
            if abs(nw) > w_max:
                w_max = abs(nw)
        if w_max == Float32(0) or d_w_max <= mul(tol, w_max):
            break
    for t in range(l, K, LG_TPB):
        w[base + t] = sW[t]
    if l == 0:
        its[i] = Float32(it)
