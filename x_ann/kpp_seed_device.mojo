# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The k-means++ seeding of `x_ann/kpp_seed.mojo` on the device (lane
apple-fast-fastonly2, `-D MOJOLEARN_IVF_FAST_SEED_DEVICE`, FAST on Apple, under
`IVF_FAST_SEED`).

Same rows (a stride sample of the training rows, `rows_per_seed` per seed),
same draws (cluster/'s `HostRng` stream, one `next_u64` per seed, drawn up front
on the host: O(k) words), same rule (first seed uniform, every next one with
probability proportional to its squared distance to the nearest seed so far).
Per seed two launches:

  `kpp_dist_kernel`    one thread per sampled row: the distance to the newest
                       seed folds into `nearest`; each block writes the sum of
                       its `nearest` values to `partials` (grid of blocks)
  `kpp_select_kernel`  the fold over the block partials: total, the block
                       whose prefix crosses `unit * total` (block scan over the
                       partials), then the row inside that block (block scan
                       over its KPP_TPB `nearest` values); copies the row into
                       the seed slot

FAST only: Float32 sums in a blocked order instead of the host's Float64 running
sum, so a pick can differ from the host's where the two sums round apart. The
paired quality check is the recall at k (bench/speed/ann_fast_quality.py)."""
from std.gpu import block_idx, thread_idx
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from max.gpu.host import DeviceBuffer, DeviceContext
from cluster.impl.detail.kmeans import HostRng
from x_ann.kpp_seed import kpp_seed_rows

comptime KPP_F32P = MutPointer[Float32, MutAnyOrigin]
comptime KPP_I32P = MutPointer[Int32, MutAnyOrigin]

#: threads per block of both kernels (the select kernel scans one distance
#: block in one pass, so the two must match)
comptime KPP_TPB = 256
comptime KPP_LOG2_TPB = 8


def kpp_dist_kernel(
    x: KPP_F32P, seeds: KPP_F32P, nearest: KPP_F32P, partials: KPP_F32P,
    ns: Int32, step: Int32, d: Int32, c: Int32,
):
    """Row i (< ns) of the sample: nearest[i] = min(nearest[i], |x[i*step] -
    seeds[c]|^2); partials[block] = the block's sum of nearest."""
    var tid = Int(thread_idx.x)
    var i = Int(block_idx.x) * KPP_TPB + tid
    var dd = Int(d)
    var cur = Float32(0)
    if i < Int(ns):
        var row_at = i * Int(step) * dd
        var seed_at = Int(c) * dd
        var dist = Float32(0)
        for t in range(dd):
            var diff = x[unsafe_offset=row_at + t] - seeds[unsafe_offset=seed_at + t]
            dist += diff * diff
        cur = nearest[unsafe_offset=i]
        if dist < cur:
            cur = dist
            nearest[unsafe_offset=i] = cur
    var sh = stack_allocation[KPP_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    sh[tid] = cur
    barrier()
    var h = KPP_TPB // 2
    while h > 0:
        if tid < h:
            sh[tid] = sh[tid] + sh[tid + h]
        barrier()
        h //= 2
    if tid == 0:
        partials[unsafe_offset=Int(block_idx.x)] = sh[0]


def kpp_select_kernel(
    x: KPP_F32P, seeds: KPP_F32P, nearest: KPP_F32P, partials: KPP_F32P,
    units: KPP_F32P, idxs: KPP_I32P,
    ns: Int32, nb: Int32, step: Int32, d: Int32, cn: Int32,
):
    """One block of KPP_TPB threads (a fold over nb partials and one block of
    nearest values, not over the n rows): picks seed cn and copies its row."""
    var tid = Int(thread_idx.x)
    var nbi = Int(nb)
    var sh = stack_allocation[KPP_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var shi = stack_allocation[KPP_TPB, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var res = stack_allocation[2, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var resf = stack_allocation[1, Scalar[DType.float32], address_space = AddressSpace.SHARED]()

    # 1. total of the partials, and the last block holding any weight
    var acc = Float32(0)
    var last = Int32(-1)
    var j = tid
    while j < nbi:
        var v = partials[unsafe_offset=j]
        acc += v
        if v > 0:
            last = Int32(j)
        j += KPP_TPB
    sh[tid] = acc
    shi[tid] = last
    barrier()
    var h = KPP_TPB // 2
    while h > 0:
        if tid < h:
            sh[tid] = sh[tid] + sh[tid + h]
            if shi[tid + h] > shi[tid]:
                shi[tid] = shi[tid + h]
        barrier()
        h //= 2
    var total = sh[0]
    var last_block = Int(shi[0])
    if tid == 0:
        res[0] = -1
        res[1] = -1
        resf[0] = 0
    barrier()

    if total > 0 and last_block >= 0:
        var target = units[unsafe_offset=Int(cn)] * total
        # 2. the block whose inclusive prefix first exceeds target
        var base = Float32(0)
        var start = 0
        while start < nbi:
            var v = partials[unsafe_offset=start + tid] if start + tid < nbi else Float32(0)
            sh[tid] = v
            barrier()
            comptime for s in range(KPP_LOG2_TPB):
                comptime off = 1 << s
                var w = sh[tid]
                if tid >= off:
                    w += sh[tid - off]
                barrier()
                sh[tid] = w
                barrier()
            var incl = base + sh[tid]
            var excl = base + (sh[tid - 1] if tid > 0 else Float32(0))
            if start + tid < nbi and incl > target and excl <= target and v > 0:
                res[0] = Int32(start + tid)
                resf[0] = excl
            var chunk = sh[KPP_TPB - 1]
            barrier()
            if res[0] >= 0:
                break
            base += chunk
            start += KPP_TPB
        var b = Int(res[0])
        var rem = target - resf[0]
        if b < 0:
            # Float32 rounding left target past the scanned sum: the last
            # block with weight, its last row with weight
            b = last_block
            rem = Float32.MAX
        barrier()
        # 3. the row inside block b
        var i = b * KPP_TPB + tid
        var v = nearest[unsafe_offset=i] if i < Int(ns) else Float32(0)
        sh[tid] = v
        shi[tid] = Int32(i) if v > 0 else Int32(-1)
        barrier()
        h = KPP_TPB // 2
        while h > 0:
            if tid < h and shi[tid + h] > shi[tid]:
                shi[tid] = shi[tid + h]
            barrier()
            h //= 2
        var last_row = shi[0]
        barrier()
        comptime for s in range(KPP_LOG2_TPB):
            comptime off = 1 << s
            var w = sh[tid]
            if tid >= off:
                w += sh[tid - off]
            barrier()
            sh[tid] = w
            barrier()
        var incl = sh[tid]
        var excl = sh[tid - 1] if tid > 0 else Float32(0)
        if i < Int(ns) and incl > rem and excl <= rem and v > 0:
            res[1] = Int32(i)
        barrier()
        if tid == 0 and res[1] < 0:
            res[1] = last_row
        barrier()
    if tid == 0 and res[1] < 0:
        res[1] = idxs[unsafe_offset=Int(cn)]
    barrier()

    # 4. the seed row
    var pick = Int(res[1])
    var dd = Int(d)
    var t = tid
    while t < dd:
        seeds[unsafe_offset=Int(cn) * dd + t] = x[unsafe_offset=pick * Int(step) * dd + t]
        t += KPP_TPB


def kpp_first_kernel(
    x: KPP_F32P, seeds: KPP_F32P, idxs: KPP_I32P, step: Int32, d: Int32,
):
    """Seed 0: the uniform draw's row."""
    var pick = Int(idxs[unsafe_offset=0])
    var dd = Int(d)
    var t = Int(thread_idx.x)
    while t < dd:
        seeds[unsafe_offset=t] = x[unsafe_offset=pick * Int(step) * dd + t]
        t += KPP_TPB


def kpp_seed_device(
    ctx: DeviceContext, mut x: DeviceBuffer[DType.float32], n: Int, d: Int, k: Int, seed: UInt64,
    rows_per_seed: Int, mut seeds: DeviceBuffer[DType.float32],
) raises:
    """`seeds` (k x d on the device) filled with k rows of `x` (n x d,
    row-major, on the device), the device form of `kpp_seed`. Needs n >= k >=
    1 and d >= 1. Synchronizes before it returns."""
    var ns = kpp_seed_rows(n, k, rows_per_seed)
    var step = n // ns
    var nb = (ns + KPP_TPB - 1) // KPP_TPB
    # the draws, one splitmix64 word per seed (the host loop's stream)
    var rng = HostRng(seed)
    var units = List[Float32](length=k, fill=Float32(0))
    var idxs = List[Int32](length=k, fill=Int32(0))
    for c in range(k):
        var u = rng.next_u64()
        idxs[c] = Int32(Int(u % UInt64(ns)))
        var f = Float32(Float64(u >> 11) * (1.0 / 9007199254740992.0))
        if f >= Float32(1.0):
            f = Float32(0.99999994)
        units[c] = f
    var d_units = ctx.enqueue_create_buffer[DType.float32](k)
    var d_idxs = ctx.enqueue_create_buffer[DType.int32](k)
    ctx.enqueue_copy(dst_buf=d_units, src_ptr=units.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_idxs, src_ptr=idxs.unsafe_ptr())
    var nearest = ctx.enqueue_create_buffer[DType.float32](ns)
    nearest.enqueue_fill(Float32.MAX)
    var partials = ctx.enqueue_create_buffer[DType.float32](nb)
    var xp = x.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var sp = seeds.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var np = nearest.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var pp = partials.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var up = d_units.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var ip = d_idxs.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    ctx.enqueue_function[kpp_first_kernel](
        xp, sp, ip, Int32(step), Int32(d), grid_dim=1, block_dim=KPP_TPB,
    )
    for c in range(k - 1):
        ctx.enqueue_function[kpp_dist_kernel](
            xp, sp, np, pp, Int32(ns), Int32(step), Int32(d), Int32(c),
            grid_dim=nb, block_dim=KPP_TPB,
        )
        ctx.enqueue_function[kpp_select_kernel](
            xp, sp, np, pp, up, ip, Int32(ns), Int32(nb), Int32(step), Int32(d), Int32(c + 1),
            grid_dim=1, block_dim=KPP_TPB,
        )
    ctx.synchronize()
    _ = units^
    _ = idxs^
