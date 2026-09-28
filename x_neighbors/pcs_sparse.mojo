# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""FAST on Apple: PolynomialCountSketch's row convolution over the nonzero
components of the running product (lane neighbors-apple3, 2026-09-28). Its
own module, imported only where a build selects it
(x_neighbors/iter_device.mojo, `-D MOJOLEARN_XN_PCS_SPARSE`). Launched at
PCS_ROW_TPB threads a block (PCS_SPARSE_TPB is the same 256)."""
from std.gpu import block_idx, thread_idx
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from checks.numerics import ftz, identical_mul_add
from x_neighbors.items import FP

comptime PCS_SPARSE_TPB = 256
comptime PCS_SPARSE_MAX_NC = 1024


def pcs_conv_row_sparse_kernel(acc: FP, sk: FP, res: FP, n_: Int64, nc_: Int64, degree_: Int64, p_: Int64):
    """`pcs_conv_row_kernel` over the nonzero components of the row's
    running product, a ascending. nc <= PCS_SPARSE_MAX_NC."""
    var nc = Int(nc_)
    var degree = Int(degree_)
    var p = Int(p_)
    var r = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var ar = stack_allocation[PCS_SPARSE_MAX_NC, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var sr = stack_allocation[PCS_SPARSE_MAX_NC, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var nz = stack_allocation[PCS_SPARSE_MAX_NC, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    var cnt_slot = stack_allocation[1, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    var q = tid
    while q < nc:
        ar[q] = acc.unsafe_load(r * nc + q)
        sr[q] = sk.unsafe_load((r * degree + p) * nc + q)
        q += PCS_SPARSE_TPB
    barrier()
    if tid == 0:
        var found = 0
        for a in range(nc):
            if ar[a] != Float32(0):
                nz[found] = Int32(a)
                found += 1
        cnt_slot[0] = Int32(found)
    barrier()
    var cnt = Int(cnt_slot[0])
    var h0 = tid * 4
    while h0 < nc:
        var s0 = Float32(0)
        var s1 = Float32(0)
        var s2 = Float32(0)
        var s3 = Float32(0)
        for e in range(cnt):
            var a = Int(nz[e])
            var av = ar[a]
            var b = h0 - a
            if b < 0:
                b += nc
            s0 = ftz(identical_mul_add(av, sr[b], s0))
            b += 1
            if b == nc:
                b = 0
            s1 = ftz(identical_mul_add(av, sr[b], s1))
            b += 1
            if b == nc:
                b = 0
            s2 = ftz(identical_mul_add(av, sr[b], s2))
            b += 1
            if b == nc:
                b = 0
            s3 = ftz(identical_mul_add(av, sr[b], s3))
        res.unsafe_store(r * nc + h0, s0)
        if h0 + 1 < nc:
            res.unsafe_store(r * nc + h0 + 1, s1)
        if h0 + 2 < nc:
            res.unsafe_store(r * nc + h0 + 2, s2)
        if h0 + 3 < nc:
            res.unsafe_store(r * nc + h0 + 3, s3)
        h0 += PCS_SPARSE_TPB * 4
