# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""IVF-PQ: every subspace codebook trained by ONE batched Lloyd loop on the
device (lane/apple-fast-ann, 2026-10-02; FAST on Apple, behind
`-D MOJOLEARN_IVFPQ_FAST_DEVICE_CODEBOOKS=1`, x_ann/fast_env.mojo).

Cause (`x_ann/ivf_pq_device.mojo` `_codebooks`, `ivf_pq_build_device`): the
n x rot_dim residual matrix was downloaded to the host (352 MB on Istella),
each subspace's sample columns gathered on the host, and cluster/'s
`kmeans_fit` run once PER SUBSPACE (55 on Istella, 11 on taxi), each a
host-driven fit with its own seeding rounds, launches and `synchronize`s.

Here the residuals stay on the device. The training sample is a stride
sample of the rows (sample row s is dataset row (s n) // n_train); the
codebooks start from seeded sample rows (FAISS trains its PQ with
random-row seeds); then pq_iters times: `pqk_assign_kernel` (one thread
per (sample row, subspace), the subspace codebook staged, the lower code on
a tie), `pqk_partial_kernel` (one threadgroup per (block of PQK_ROWS rows,
subspace), thread t sums the block's rows labelled t IN ROW ORDER: no
atomics), `pqk_update_kernel` (one thread per (subspace, code) adds the
block partials in block order and divides by the count; an empty code
keeps its centroid). Three launches per iteration, no host sync, every
subspace in every launch. The order of every sum is fixed, so the result
is the same on every run. FAST bits move against the old codebooks
(another seeding, another sum order): paired recall check."""

from std.gpu import block_idx, block_dim, thread_idx
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import ftz, identical_mul_add
from x_ann.ivf_pq_core import F32P, I32P

#: threads per threadgroup (one per code in the partial sums), rows per
#: partial block (= the threads, so one thread stages one label)
comptime PQK_T = 256
comptime PQK_ROWS = 256
#: the widest subspace and the most codes this path takes
comptime PQK_LEN_MAX = 16
comptime PQK_CODES_MAX = 256


def _tid() -> Int:
    return Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)


def pqk_init_kernel(
    count: Int32, n: Int32, n_train: Int32, rot_dim: Int32, pq_len: Int32, n_codes: Int32, seed: Int32,
    r: F32P, cb: F32P,
):
    """cb[j, c, u] = the residual of sample row s(j, c), subspace j,
    coordinate u, with s(j, c) = c (n_train // n_codes) + (h(seed, j) mod
    (n_train // n_codes)): distinct rows per subspace, a seeded offset per
    subspace (splitmix64 of the seed and j)."""
    var e = _tid()
    if e < Int(count):
        var pl = Int(pq_len)
        var nc = Int(n_codes)
        var nt = Int(n_train)
        var u = e % pl
        var c = (e // pl) % nc
        var j = e // (pl * nc)
        var stride = nt // nc
        if stride < 1:
            stride = 1
        var h = UInt64(Int(seed)) ^ (UInt64(j + 1) * UInt64(0x9E3779B97F4A7C15))
        h = (h ^ (h >> 30)) * UInt64(0xBF58476D1CE4E5B9)
        h = (h ^ (h >> 27)) * UInt64(0x94D049BB133111EB)
        h = h ^ (h >> 31)
        var s = c * stride + Int(h % UInt64(stride))
        if s >= nt:
            s = nt - 1
        var row = (s * Int(n)) // nt
        cb.unsafe_store(e, ftz(r.unsafe_load(row * Int(rot_dim) + j * pl + u)))


def pqk_assign_kernel(
    n_train: Int32, n: Int32, r: F32P, cb: F32P, pq_dim: Int32, rot_dim: Int32, pq_len: Int32, n_codes: Int32,
    labels: I32P,
):
    """Threadgroup (x, j): subspace j's codebook (n_codes x pq_len words)
    staged once; thread t encodes sample row s = x PQK_T + t (dataset row
    (s n) // n_train) by the ascending fused square sum over the codes, the
    lower code on an exact tie (`pq_assign_cell`'s rule); labels[s pq_dim + j]."""
    var t = Int(thread_idx.x)
    var j = Int(block_idx.y)
    var s = Int(block_idx.x) * PQK_T + t
    var pl = Int(pq_len)
    var nc = Int(n_codes)
    var per = nc * pl
    var tile = stack_allocation[
        PQK_CODES_MAX * PQK_LEN_MAX, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    for e in range(t, per, PQK_T):
        tile[e] = cb.unsafe_load(j * per + e)
    barrier()
    if s >= Int(n_train):
        return
    var row = (s * Int(n)) // Int(n_train)
    var rv = InlineArray[Float32, PQK_LEN_MAX](fill=Float32(0.0))
    var off = row * Int(rot_dim) + j * pl
    for u in range(pl):
        rv[u] = ftz(r.unsafe_load(off + u))
    var best = 0
    var bd = Float32(0.0)
    for c in range(nc):
        var acc = Float32(0.0)
        for u in range(pl):
            var diff = ftz(rv[u] - tile[c * pl + u])
            acc = ftz(identical_mul_add(diff, diff, acc))
        if c == 0:
            bd = acc
        elif acc < bd:
            bd = acc
            best = c
    labels.unsafe_store(s * Int(pq_dim) + j, Int32(best))


def pqk_partial_kernel(
    n_train: Int32, n: Int32, r: F32P, labels: I32P, pq_dim: Int32, rot_dim: Int32, pq_len: Int32,
    n_codes: Int32, nb: Int32, psum: F32P, pcnt: I32P,
):
    """Threadgroup (b, j): the per-code sums and counts of sample rows
    b PQK_ROWS .. b PQK_ROWS + PQK_ROWS - 1 in subspace j. The rows' labels
    and residual words are staged; thread t (< n_codes) walks the rows in
    order adding those labelled t, and stores sum[u] at
    psum[((j nb + b) n_codes + t) pq_len + u] and the count at
    pcnt[(j nb + b) n_codes + t]. A fixed order per (block, code): no atomics."""
    var t = Int(thread_idx.x)
    var b = Int(block_idx.x)
    var j = Int(block_idx.y)
    var pl = Int(pq_len)
    var nc = Int(n_codes)
    var pd = Int(pq_dim)
    var nt = Int(n_train)
    var s0 = b * PQK_ROWS
    var sn = PQK_ROWS if nt - s0 > PQK_ROWS else nt - s0
    var lab = stack_allocation[PQK_ROWS, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    var vals = stack_allocation[
        PQK_ROWS * PQK_LEN_MAX, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    for e in range(t, PQK_ROWS * pl, PQK_T):
        var rr = e // pl
        var u = e % pl
        var v = Float32(0.0)
        if rr < sn:
            var row = ((s0 + rr) * Int(n)) // nt
            v = ftz(r.unsafe_load(row * Int(rot_dim) + j * pl + u))
        vals[rr * pl + u] = v
    if t < sn:
        lab[t] = labels.unsafe_load((s0 + t) * pd + j)
    barrier()
    if t >= nc:
        return
    var sum = InlineArray[Float32, PQK_LEN_MAX](fill=Float32(0.0))
    var cnt = 0
    for rr in range(sn):
        if Int(lab[rr]) == t:
            for u in range(pl):
                sum[u] = sum[u] + vals[rr * pl + u]
            cnt += 1
    var o = (j * Int(nb) + b) * nc + t
    for u in range(pl):
        psum.unsafe_store(o * pl + u, sum[u])
    pcnt.unsafe_store(o, Int32(cnt))


def pqk_update_kernel(count: Int32, pq_len: Int32, n_codes: Int32, nb: Int32, psum: F32P, pcnt: I32P, cb: F32P):
    """Thread e = j n_codes + c: the code's centroid = its block partials
    added in block order, over its count; a code with no rows keeps its
    centroid."""
    var e = _tid()
    if e < Int(count):
        var pl = Int(pq_len)
        var nc = Int(n_codes)
        var c = e % nc
        var j = e // nc
        var sum = InlineArray[Float32, PQK_LEN_MAX](fill=Float32(0.0))
        var cnt = 0
        for b in range(Int(nb)):
            var o = (j * Int(nb) + b) * nc + c
            cnt += Int(pcnt.unsafe_load(o))
            for u in range(pl):
                sum[u] = sum[u] + psum.unsafe_load(o * pl + u)
        if cnt > 0:
            var inv = Float32(1.0) / Float32(cnt)
            for u in range(pl):
                cb.unsafe_store(e * pl + u, sum[u] * inv)


def pq_codebooks_device(
    ctx: DeviceContext, mut dr: DeviceBuffer[DType.float32], n: Int, n_train: Int, rot_dim: Int, pq_dim: Int,
    pq_len: Int, n_codes: Int, pq_iters: Int, seed: Int,
) raises -> DeviceBuffer[DType.float32]:
    """The pq_dim x n_codes x pq_len codebooks on the device, from the
    residuals in `dr` (n x rot_dim), trained on the stride sample of
    n_train rows for pq_iters Lloyd iterations. Waits for its launches.
    The caller checks pq_len <= PQK_LEN_MAX and n_codes <= PQK_CODES_MAX."""
    var nb = (n_train + PQK_ROWS - 1) // PQK_ROWS
    var cells = pq_dim * n_codes * pq_len
    var dcb = ctx.enqueue_create_buffer[DType.float32](cells)
    var dlab = ctx.enqueue_create_buffer[DType.int32](n_train * pq_dim)
    var dps = ctx.enqueue_create_buffer[DType.float32](pq_dim * nb * n_codes * pq_len)
    var dpc = ctx.enqueue_create_buffer[DType.int32](pq_dim * nb * n_codes)
    ctx.enqueue_function[pqk_init_kernel](
        Int32(cells), Int32(n), Int32(n_train), Int32(rot_dim), Int32(pq_len), Int32(n_codes), Int32(seed),
        dr.unsafe_ptr(), dcb.unsafe_ptr(), grid_dim=(cells + PQK_T - 1) // PQK_T, block_dim=PQK_T,
    )
    for _ in range(pq_iters):
        ctx.enqueue_function[pqk_assign_kernel](
            Int32(n_train), Int32(n), dr.unsafe_ptr(), dcb.unsafe_ptr(), Int32(pq_dim), Int32(rot_dim),
            Int32(pq_len), Int32(n_codes), dlab.unsafe_ptr(),
            grid_dim=((n_train + PQK_T - 1) // PQK_T, pq_dim), block_dim=PQK_T,
        )
        ctx.enqueue_function[pqk_partial_kernel](
            Int32(n_train), Int32(n), dr.unsafe_ptr(), dlab.unsafe_ptr(), Int32(pq_dim), Int32(rot_dim),
            Int32(pq_len), Int32(n_codes), Int32(nb), dps.unsafe_ptr(), dpc.unsafe_ptr(),
            grid_dim=(nb, pq_dim), block_dim=PQK_T,
        )
        ctx.enqueue_function[pqk_update_kernel](
            Int32(pq_dim * n_codes), Int32(pq_len), Int32(n_codes), Int32(nb), dps.unsafe_ptr(), dpc.unsafe_ptr(),
            dcb.unsafe_ptr(), grid_dim=(pq_dim * n_codes + PQK_T - 1) // PQK_T, block_dim=PQK_T,
        )
    ctx.synchronize()
    _ = dpc^
    _ = dps^
    _ = dlab^
    return dcb^
