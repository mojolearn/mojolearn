# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""MANY small symmetric eigenproblems at once (cgr-decomp, 2026-10-03): the
round-robin Jacobi of x_decomp/rr.mojo, ONE BLOCK PER PROBLEM, the pairs of
a round, the 2 x 2 blocks of J^T A J and the rows of V J spread over the
block's threads. LLE's local spectra (LTSA, Hessian, modified: one n_neighbors
x n_neighbors Gram per sample) used to run one kit eigh per sample.

Every cell is the single solver's (`rr_cs`, `rr_block`, `rr_vrow`,
`rr_row_off`), the convergence test folds in `rr_off_fold`'s order (block
trees of RR_OFF_TPB rows, then slot t adds block sums t, t + RR_OFF_TPB, ...
and the tree again) and decides by `rr_converged` / `rr_fro_kept`, so every
problem gets the words `host_eigh_rr` gets for it, and the words the
single-matrix driver (`DevExec._eigh_par_on`) gets. The tail is
`host_sign_flip` (comparisons only) and `eigh_ascending`'s order
(`spectrum_rank_desc`, comparisons only).

Launch: grid = batch, block = RR_OFF_TPB. Scratch per problem: cs 2 h
(h = ceil(n / 2)), part 2 ceil(n / RR_OFF_TPB), diag n, v n x n.
info[2 b] = 1 converged / 0 not, info[2 b + 1] = sweeps run."""
from std.gpu import block_idx, thread_idx
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace

from checks.numerics import ftz
from decomposition.spectrum_order_device import spectrum_rank_desc
from x_decomp.cells import F32Ptr
from x_decomp.jacobi2 import dev_barrier
from x_decomp.rr import RR_OFF_TPB, rr_block, rr_converged, rr_cs, rr_fro_kept, rr_row_off, rr_vrow


@always_inline
def rrb_cs_len(n: Int) -> Int:
    return 2 * ((n + 1) // 2)


@always_inline
def rrb_part_len(n: Int) -> Int:
    return 2 * max((n + RR_OFF_TPB - 1) // RR_OFF_TPB, 1)


def rr_batch_kernel(
    a_all: F32Ptr, v_all: F32Ptr, cs_all: F32Ptr, part_all: F32Ptr, dg_all: F32Ptr, info: F32Ptr,
    w_out: F32Ptr, v_out: F32Ptr, n_in: Int32, sweeps_in: Int32, tol: Float32,
):
    var n = Int(n_in)
    var sweeps = Int(sweeps_in)
    var bid = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var m = n + (n % 2)
    var h = m // 2
    var nb = (n + RR_OFF_TPB - 1) // RR_OFF_TPB
    var a = a_all + bid * n * n
    var v = v_all + bid * n * n
    var cs = cs_all + bid * rrb_cs_len(n)
    var part = part_all + bid * rrb_part_len(n)
    var dg = dg_all + bid * n
    var so = stack_allocation[RR_OFF_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var sd = stack_allocation[RR_OFF_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var t = tid
    while t < n * n:
        var i = t // n
        v.unsafe_store(t, Float32(1.0) if i * n + i == t else Float32(0.0))
        t += RR_OFF_TPB
    dev_barrier()
    var converged = False
    var executed = 0
    var fro_in = Float32(-1.0)
    var fro_now = Float32(0.0)
    for sweep in range(sweeps + 1):
        # the test: rr_off_fold's block trees, then its fold past the blocks
        for b in range(nb):
            var k = b * RR_OFF_TPB + tid
            var o = SIMD[DType.float32, 2](0.0, 0.0)
            if k < n:
                o = rr_row_off(a, n, k)
            so[tid] = o[0]
            sd[tid] = o[1]
            dev_barrier()
            var w = RR_OFF_TPB // 2
            while w > 0:
                if tid < w:
                    so[tid] = ftz(so[tid] + so[tid + w])
                    sd[tid] = ftz(sd[tid] + sd[tid + w])
                dev_barrier()
                w = w // 2
            if tid == 0:
                part.unsafe_store(2 * b, so[0])
                part.unsafe_store(2 * b + 1, sd[0])
            dev_barrier()
        var ao = Float32(0.0)
        var ad = Float32(0.0)
        var bb = tid
        while bb < nb:
            ao = ftz(ao + part.unsafe_load(2 * bb))
            ad = ftz(ad + part.unsafe_load(2 * bb + 1))
            bb += RR_OFF_TPB
        so[tid] = ao
        sd[tid] = ad
        dev_barrier()
        var w2 = RR_OFF_TPB // 2
        while w2 > 0:
            if tid < w2:
                so[tid] = ftz(so[tid] + so[tid + w2])
                sd[tid] = ftz(sd[tid] + sd[tid + w2])
            dev_barrier()
            w2 = w2 // 2
        var off = so[0]
        var dd = sd[0]
        dev_barrier()
        fro_now = ftz(off + dd)
        if fro_in < Float32(0.0):
            fro_in = fro_now
        if rr_converged(off, dd, tol):
            converged = True
            break
        if sweep == sweeps:
            break
        executed += 1
        for rd in range(m - 1):
            var p = tid
            while p < h:
                var got = rr_cs(a, n, m, rd, p)
                cs.unsafe_store(2 * p, got[0])
                cs.unsafe_store(2 * p + 1, got[1])
                p += RR_OFF_TPB
            dev_barrier()
            var u = tid
            while u < h * h + n * h:
                if u < h * h:
                    var i = u // h
                    var j = u - i * h
                    if i <= j:
                        rr_block(a, cs, n, m, rd, i, j)
                else:
                    var x = u - h * h
                    var k = x // h
                    rr_vrow(v, cs, n, m, rd, k, x - k * h)
                u += RR_OFF_TPB
            dev_barrier()
    if converged and not rr_fro_kept(fro_in, fro_now):
        converged = False
    if tid == 0:
        info.unsafe_store(2 * bid, Float32(1.0) if converged else Float32(0.0))
        info.unsafe_store(2 * bid + 1, Float32(executed))
    # host_sign_flip, one thread a column (comparisons only)
    var col = tid
    while col < n:
        var biggest = Float32(0.0)
        for f in range(n):
            var mg = abs(v.unsafe_load(f * n + col))
            if mg > biggest:
                biggest = mg
        var first = n
        for f in range(n):
            if first == n and abs(v.unsafe_load(f * n + col)) == biggest:
                first = f
        if first < n and v.unsafe_load(first * n + col) < Float32(0.0):
            for f in range(n):
                v.unsafe_store(f * n + col, -v.unsafe_load(f * n + col))
        dg.unsafe_store(col, a.unsafe_load(col * n + col))
        col += RR_OFF_TPB
    dev_barrier()
    # eigh_ascending: value i to position n - 1 - rank_desc(i), its column with it
    var wo = w_out + bid * n
    var vo = v_out + bid * n * n
    var i2 = tid
    while i2 < n:
        var c = n - 1 - spectrum_rank_desc(dg, n, i2)
        wo.unsafe_store(c, dg.unsafe_load(i2))
        for r in range(n):
            vo.unsafe_store(r * n + c, v.unsafe_load(r * n + i2))
        i2 += RR_OFF_TPB

