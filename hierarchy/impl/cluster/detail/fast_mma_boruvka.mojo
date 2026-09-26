# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""FAST, Apple: Boruvka's nearest-other-component search with the pair
products on the simdgroup matrix unit, and the SAME answer as
`fb_nearest_other_kernel`.

WHY. `fb_nearest_other_kernel` sums `(x - y)^2` in scalar FMAs out of shared
memory for every listed point against every point. The matrix unit computes
`q.x` for 8x8 blocks of pairs (`neighbors/impl/detail/fast_mma_knn.mojo`'s
intrinsics and layout).

EXACTNESS. The expanded value `|q'|^2 + |x'|^2 - 2 q'.x'` over data centered
on its mean is within `ME_MARGIN * (|q'|^2 + |x'|^2)` of the unexpanded
squared distance (see `dbscan/impl/neighbors/fast_mma_eps.mojo` for the
bound), so `lb = expanded - band` is a LOWER bound on it. It is only a
filter: a candidate in another component is recomputed with the scalar
kernel's own arithmetic (`d += (xi - xj)^2` in feature order, from the
original rows; the mutual-reachability arm then
`max(core_j, max(core_i, inv_alpha * sqrt(d)))`) whenever its lower bound
could beat the lane's best, and the best moves on a strict `<`. Each lane
walks its columns in ascending index order, so every lane holds the lowest
index among its exactly equal minima; the four lanes of a point and the
`grid.y` slices fold by (value, index), so the answer is the scalar
kernel's: the same value bits and the same lowest index on ties.

On the mutual-reachability arm the filter is `core_j < best` and
`lb < (best / inv_alpha)^2 * (1 + 2^-16)` (a superset of the pairs whose
rounded `inv_alpha * sqrt(d)` is below `best`).

The caller declines (runs the scalar kernel) when a centered row norm is not
finite or above `MB_NORM_CAP`, when a core distance is not finite, or when
`inv_alpha` is outside [1e-3, 1e3]: there the band does not bound the error,
or the scalar kernel's non-finite refusal must run.
"""

from std.bit import count_trailing_zeros
from std.gpu import block_idx, thread_idx
from std.gpu.primitives.warp import shuffle_xor
from std.math import sqrt
from std.memory import bitcast, stack_allocation
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.atomic import Atomic

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST
from neighbors.impl.detail.fast_mma_knn import _sg_load_t, _sg_mma

comptime FAST_MMA_BORUVKA_ENABLED = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and not is_defined["MOJOLEARN_BORUVKA_FAST_MMA_OFF"]()
)

comptime MB_SG = 8
comptime MB_TPB = MB_SG * 32
comptime MB_T = 128
comptime MB_MAX_D = 32
comptime MB_A = 2
"""Listed-point 8-blocks per simdgroup (4: 2.08 s against 1.72-1.80 s at
100k x 16; 1 with B = 8: 2.3-2.5 s)."""
comptime MB_B = 4
"""Point 8-blocks per step (8: 2.0-2.2 s)."""
comptime MB_MARGIN = Float32(3.0517578125e-05)
"""2^-15, the filter's relative band."""
comptime MB_NORM_CAP = Float32(1.0e30)
comptime MB_PAD = Float32(1.0e30)
comptime MB_BIG = Float32(3.0e38)
comptime MB_SLACK = Float32(1.0000152587890625)
"""1 + 2^-16: the reachability arm's threshold slack."""

comptime _M64 = SIMD[DType.float32, 64]


def fast_mma_boruvka_applies(n_features: Int) -> Bool:
    comptime if not FAST_MMA_BORUVKA_ENABLED:
        return False
    return n_features >= 1 and n_features <= MB_MAX_D


def mb_center_kernel(
    x: MutPointer[Float32, MutAnyOrigin],
    mean: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    k_in: Int32,
):
    """One block of 256: column means (any order; the center never reaches
    the answer)."""
    var n = Int(n_in)
    var k = Int(k_in)
    var tid = Int(thread_idx.x)
    var f = tid % 32
    var part = tid // 32
    var part_sum = stack_allocation[256, Float32, address_space=AddressSpace.SHARED]()
    var s = Float32(0)
    if f < k:
        var r = part
        while r < n:
            s += x[r * k + f]
            r += 8
    part_sum[tid] = s
    barrier()
    if tid < k:
        var t = Float32(0)
        for p in range(8):
            t += part_sum[p * 32 + tid]
        mean[tid] = t / Float32(n)


def mb_shift_kernel[MR: Bool](
    x: MutPointer[Float32, MutAnyOrigin],
    mean: MutPointer[Float32, MutAnyOrigin],
    core: MutPointer[Float32, MutAnyOrigin],
    xc: MutPointer[Float32, MutAnyOrigin],
    nc: MutPointer[Float32, MutAnyOrigin],
    flag: MutPointer[Int32, MutAnyOrigin],
    n_in: Int32,
    k_in: Int32,
):
    """`xc = x - mean`, `nc = |xc|^2`; raises `flag` for a row the filter
    cannot serve (exponent bits: FAST may fold a non-finite compare)."""
    var n = Int(n_in)
    var k = Int(k_in)
    var r = Int(block_idx.x) * 256 + Int(thread_idx.x)
    if r >= n:
        return
    var s = Float32(0)
    for c in range(k):
        var v = x[r * k + c] - mean[c]
        xc[r * k + c] = v
        s += v * v
    nc[r] = s
    var bad = (bitcast[DType.uint32](s) & 0x7F800000) == 0x7F800000 or s > MB_NORM_CAP
    comptime if MR:
        var cr = core[r]
        if (bitcast[DType.uint32](cr) & 0x7F800000) == 0x7F800000:
            bad = True
    if bad:
        _ = Atomic.fetch_add(flag, Int32(1))


@always_inline
def _exact_d2(
    x: MutPointer[Float32, MutAnyOrigin], i: Int, j: Int, dim: Int
) -> Float32:
    """`fb_nearest_other_kernel`'s per-pair sum: `d += df * df` in feature
    order."""
    var d = Float32(0)
    for t in range(dim):
        var df = x[i * dim + t] - x[j * dim + t]
        d += df * df
    return d


@always_inline
def _mr_thr(bd: Float32, inv_alpha: Float32) -> Float32:
    """The squared-distance filter bound for a reachability best `bd`."""
    if bd >= Float32(1.0e18):
        return MB_BIG
    var q = bd / inv_alpha
    return q * q * MB_SLACK


@always_inline
def _before(da: Float32, ja: Int32, db: Float32, jb: Int32) -> Bool:
    """(da, ja) is the better pair: smaller value, ties by lower index; an
    index of -1 (nothing found) is never before a found one."""
    if ja < 0:
        return False
    if jb < 0:
        return True
    return da < db or (da == db and ja < jb)


def fb_mma_nearest_kernel[D: Int, MR: Bool, A: Int, B: Int](
    xc: MutPointer[Float32, MutAnyOrigin],
    nc: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    core: MutPointer[Float32, MutAnyOrigin],
    inv_alpha: Float32,
    comp: MutPointer[Int32, MutAnyOrigin],
    part_d: MutPointer[Float32, MutAnyOrigin],
    part_j: MutPointer[Int32, MutAnyOrigin],
    m_in: Int32,
    d_in: Int32,
    todo: MutPointer[Int32, MutAnyOrigin],
    n_todo_in: Int32,
    slice_rows_in: Int32,
):
    """Listed points `[q0, q0 + 8A)` against the point slice `block_idx.y`;
    the slice's best (value, index) per listed point lands in
    `part_*[slice * n_todo + t]`."""
    comptime KS = D // 8
    comptime RB = 8 * B
    var m = Int(m_in)
    var dim = Int(d_in)
    var n_todo = Int(n_todo_in)
    var tid = Int(thread_idx.x)
    var sg = tid // 32
    var lane = tid % 32
    var qd = lane // 4
    var frow = (qd & 4) + ((lane // 2) % 4)
    var fcol = (qd & 2) * 2 + (lane % 2) * 2
    var q0 = (Int(block_idx.x) * MB_SG + sg) * 8 * A

    var qf = InlineArray[_M64, A * KS](fill=_M64(0))
    var qn = SIMD[DType.float32, A](0)
    var qi = SIMD[DType.int32, A](0)
    var ci = SIMD[DType.int32, A](-1)
    var cri = SIMD[DType.float32, A](0)
    var live = SIMD[DType.bool, A](fill=False)
    comptime for a in range(A):
        var t = q0 + 8 * a + frow
        if t < n_todo:
            var i = Int(todo[t])
            live[a] = True
            qi[a] = Int32(i)
            ci[a] = comp[i]
            qn[a] = nc[i]
            comptime if MR:
                cri[a] = core[i]
            comptime for kk in range(KS):
                var v = _M64(0)
                comptime for e in range(2):
                    var c = 8 * kk + fcol + e
                    if c < dim:
                        v[e] = Float32(-2) * xc[i * dim + c]
                qf[a * KS + kk] = v
    var bd = SIMD[DType.float32, A](Float32.MAX)
    var bj = SIMD[DType.int32, A](-1)
    var thr = SIMD[DType.float32, A](MB_BIG)
    comptime for a in range(A):
        if not live[a]:
            thr[a] = Float32(-1.0e38)

    var s = Int(block_idx.y)
    var lo = s * Int(slice_rows_in)
    var hi = min(m, lo + Int(slice_rows_in))
    var tile = stack_allocation[
        MB_T * D, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    var tnorm = stack_allocation[
        MB_T, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    var tcomp = stack_allocation[
        MB_T, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    var tcore = stack_allocation[
        MB_T, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    comptime PER = (MB_T * D) // MB_TPB
    var pf = SIMD[DType.float32, PER](0)
    comptime for u in range(PER):
        var e = tid + u * MB_TPB
        var r = e // D
        var c = e - r * D
        if lo + r < hi and c < dim:
            pf[u] = xc[(lo + r) * dim + c]
    var base = lo
    while base < hi:
        var rows = min(MB_T, hi - base)
        comptime for u in range(PER):
            tile[tid + u * MB_TPB] = pf[u]
        if tid < MB_T:
            if tid < rows:
                tnorm[tid] = nc[base + tid]
                tcomp[tid] = comp[base + tid]
                comptime if MR:
                    tcore[tid] = core[base + tid]
            else:
                tnorm[tid] = MB_PAD
                tcomp[tid] = Int32(-7)
                tcore[tid] = MB_BIG
        barrier()
        var nb = base + rows
        comptime for u in range(PER):
            var e = tid + u * MB_TPB
            var r = e // D
            var c = e - r * D
            var v = Float32(0)
            if nb + r < hi and c < dim:
                v = xc[(nb + r) * dim + c]
            pf[u] = v
        for cb in range(MB_T // RB):
            var xf = InlineArray[_M64, B * KS](fill=_M64(0))
            comptime for b in range(B):
                comptime for kk in range(KS):
                    xf[b * KS + kk] = _sg_load_t(
                        tile + (cb * RB + 8 * b) * D + 8 * kk, D
                    )
            var cv = SIMD[DType.float32, A * 2 * B](0)
            var nv = SIMD[DType.float32, 2 * B](0)
            var cc = SIMD[DType.int32, 2 * B](0)
            var cr = SIMD[DType.float32, 2 * B](0)
            comptime for b in range(B):
                var o = cb * RB + 8 * b + fcol
                var n2 = (tnorm + o).load[width=2]()
                var c2 = (tcomp + o).load[width=2]()
                nv[2 * b] = n2[0]
                nv[2 * b + 1] = n2[1]
                cc[2 * b] = c2[0]
                cc[2 * b + 1] = c2[1]
                comptime if MR:
                    var r2 = (tcore + o).load[width=2]()
                    cr[2 * b] = r2[0]
                    cr[2 * b + 1] = r2[1]
                comptime for a in range(A):
                    var acc = _M64(0)
                    comptime for kk in range(KS):
                        acc = _sg_mma(qf[a * KS + kk], xf[b * KS + kk], acc)
                    cv[a * 2 * B + 2 * b] = n2[0] + acc[0]
                    cv[a * 2 * B + 2 * b + 1] = n2[1] + acc[1]
            var col0 = base + cb * RB + fcol
            comptime for a in range(A):
                var msk = UInt32(0)
                comptime for j in range(2 * B):
                    var lb = cv[a * 2 * B + j] + qn[a] - (
                        MB_MARGIN * (qn[a] + nv[j]) + Float32(1.0e-30)
                    )
                    var ok = cc[j] != ci[a] and lb < thr[a]
                    comptime if MR:
                        ok = ok and cr[j] < bd[a]
                    if ok:
                        msk |= UInt32(1 << j)
                # The rare path, written once per listed point: exact
                # values, lowest column first.
                while msk != 0:
                    var j = Int(count_trailing_zeros(msk))
                    msk &= msk - 1
                    var cj = col0 + 8 * (j >> 1) + (j & 1)
                    if cj < hi:
                        var v = _exact_d2(x, Int(qi[a]), cj, dim)
                        comptime if MR:
                            var crj = core[cj]
                            v = max(crj, max(cri[a], inv_alpha * sqrt(v)))
                        if v < bd[a]:
                            bd[a] = v
                            bj[a] = Int32(cj)
                            comptime if MR:
                                thr[a] = _mr_thr(v, inv_alpha)
                                # Nothing beats core_i: every value is
                                # at least that.
                                if v <= cri[a]:
                                    thr[a] = Float32(-1.0e38)
                            else:
                                thr[a] = v
        barrier()
        base += rows

    # Fold the four lanes of each listed point (lane bits 0 and 3).
    comptime for a in range(A):
        comptime for step in range(2):
            comptime mask = UInt32(1) if step == 0 else UInt32(8)
            var od = shuffle_xor(bd[a], mask)
            var oj = shuffle_xor(bj[a], mask)
            if _before(od, oj, bd[a], bj[a]):
                bd[a] = od
                bj[a] = oj
        var t = q0 + 8 * a + frow
        if live[a] and (lane & 9) == 0:
            part_d[s * n_todo + t] = bd[a]
            part_j[s * n_todo + t] = bj[a]


def fb_mma_merge_kernel(
    part_d: MutPointer[Float32, MutAnyOrigin],
    part_j: MutPointer[Int32, MutAnyOrigin],
    todo: MutPointer[Int32, MutAnyOrigin],
    best_d: MutPointer[Float32, MutAnyOrigin],
    best_j: MutPointer[Int32, MutAnyOrigin],
    n_todo_in: Int32,
    slices_in: Int32,
):
    var n_todo = Int(n_todo_in)
    var t = Int(block_idx.x) * 256 + Int(thread_idx.x)
    if t >= n_todo:
        return
    var d = part_d[t]
    var j = part_j[t]
    for s in range(1, Int(slices_in)):
        var d2 = part_d[s * n_todo + t]
        var j2 = part_j[s * n_todo + t]
        if _before(d2, j2, d, j):
            d = d2
            j = j2
    var i = Int(todo[t])
    best_d[i] = d
    best_j[i] = j


struct MmaBoruvka(Movable):
    """The centered copy, its norms and the partial buffers, built once per
    MST."""
    var xc: DeviceBuffer[DType.float32]
    var nc: DeviceBuffer[DType.float32]
    var part_d: DeviceBuffer[DType.float32]
    var part_j: DeviceBuffer[DType.int32]
    var ok: Bool

    def __init__(
        out self,
        ctx: DeviceContext,
        mut x: DeviceBuffer[DType.float32],
        m: Int,
        n: Int,
        mutual_reach: Bool,
        core_ptr: MutPointer[Float32, MutAnyOrigin],
        inv_alpha: Float32,
    ) raises:
        self.ok = fast_mma_boruvka_applies(n)
        if mutual_reach and not (
            inv_alpha >= Float32(1.0e-3) and inv_alpha <= Float32(1.0e3)
        ):
            self.ok = False
        var cap = max(m, 960 * MB_SG * 8 * MB_A) if self.ok else 1
        self.xc = ctx.enqueue_create_buffer[DType.float32](m * n if self.ok else 1)
        self.nc = ctx.enqueue_create_buffer[DType.float32](m if self.ok else 1)
        self.part_d = ctx.enqueue_create_buffer[DType.float32](cap)
        self.part_j = ctx.enqueue_create_buffer[DType.int32](cap)
        comptime if FAST_MMA_BORUVKA_ENABLED:
            if self.ok:
                self._prepare(ctx, x, m, n, mutual_reach, core_ptr)

    def _prepare(
        mut self,
        ctx: DeviceContext,
        mut x: DeviceBuffer[DType.float32],
        m: Int,
        n: Int,
        mutual_reach: Bool,
        core_ptr: MutPointer[Float32, MutAnyOrigin],
    ) raises:
        var mean = ctx.enqueue_create_buffer[DType.float32](MB_MAX_D)
        var flag = ctx.enqueue_create_buffer[DType.int32](1)
        var flag_h = ctx.enqueue_create_host_buffer[DType.int32](1)
        ctx.enqueue_memset(flag, Int32(0))
        ctx.enqueue_function[mb_center_kernel](
            x.unsafe_ptr(), mean.unsafe_ptr(), Int32(m), Int32(n),
            grid_dim=(1, 1, 1), block_dim=(256, 1, 1),
        )
        if mutual_reach:
            ctx.enqueue_function[mb_shift_kernel[True]](
                x.unsafe_ptr(), mean.unsafe_ptr(), core_ptr,
                self.xc.unsafe_ptr(), self.nc.unsafe_ptr(), flag.unsafe_ptr(),
                Int32(m), Int32(n),
                grid_dim=((m + 255) // 256, 1, 1), block_dim=(256, 1, 1),
            )
        else:
            ctx.enqueue_function[mb_shift_kernel[False]](
                x.unsafe_ptr(), mean.unsafe_ptr(), core_ptr,
                self.xc.unsafe_ptr(), self.nc.unsafe_ptr(), flag.unsafe_ptr(),
                Int32(m), Int32(n),
                grid_dim=((m + 255) // 256, 1, 1), block_dim=(256, 1, 1),
            )
        ctx.enqueue_copy(dst_ptr=flag_h.unsafe_ptr(), src_buf=flag)
        ctx.synchronize()
        if flag_h.unsafe_ptr().unsafe_load(0) != 0:
            self.ok = False
        _ = mean^
        _ = flag^
        _ = flag_h^

    def enqueue(
        mut self,
        ctx: DeviceContext,
        mut x: DeviceBuffer[DType.float32],
        m: Int,
        n: Int,
        mutual_reach: Bool,
        core_ptr: MutPointer[Float32, MutAnyOrigin],
        inv_alpha: Float32,
        mut comp: DeviceBuffer[DType.int32],
        mut todo: DeviceBuffer[DType.int32],
        n_todo: Int,
        mut best_d: DeviceBuffer[DType.float32],
        mut best_j: DeviceBuffer[DType.int32],
    ) raises:
        """One round's search for the `n_todo` listed points."""
        comptime if FAST_MMA_BORUVKA_ENABLED:
            self._enqueue(
                ctx, x, m, n, mutual_reach, core_ptr, inv_alpha, comp, todo,
                n_todo, best_d, best_j,
            )

    def _enqueue(
        mut self,
        ctx: DeviceContext,
        mut x: DeviceBuffer[DType.float32],
        m: Int,
        n: Int,
        mutual_reach: Bool,
        core_ptr: MutPointer[Float32, MutAnyOrigin],
        inv_alpha: Float32,
        mut comp: DeviceBuffer[DType.int32],
        mut todo: DeviceBuffer[DType.int32],
        n_todo: Int,
        mut best_d: DeviceBuffer[DType.float32],
        mut best_j: DeviceBuffer[DType.int32],
    ) raises:
        var QPB = MB_SG * 8 * MB_A
        var qblocks = (n_todo + QPB - 1) // QPB
        var slices = 960 // qblocks
        if slices < 1:
            slices = 1
        if slices > 64:
            slices = 64
        var slice_rows = (m + slices - 1) // slices
        slice_rows = ((slice_rows + MB_T - 1) // MB_T) * MB_T
        slices = (m + slice_rows - 1) // slice_rows
        if n <= 8:
            self._launch[8](ctx, x, mutual_reach, core_ptr, inv_alpha, comp, todo, n_todo, m, n, slice_rows, qblocks, slices)
        elif n <= 16:
            self._launch[16](ctx, x, mutual_reach, core_ptr, inv_alpha, comp, todo, n_todo, m, n, slice_rows, qblocks, slices)
        else:
            self._launch[32](ctx, x, mutual_reach, core_ptr, inv_alpha, comp, todo, n_todo, m, n, slice_rows, qblocks, slices)
        ctx.enqueue_function[fb_mma_merge_kernel](
            self.part_d.unsafe_ptr(), self.part_j.unsafe_ptr(),
            todo.unsafe_ptr(), best_d.unsafe_ptr(), best_j.unsafe_ptr(),
            Int32(n_todo), Int32(slices),
            grid_dim=((n_todo + 255) // 256, 1, 1), block_dim=(256, 1, 1),
        )

    def _launch[D: Int](
        mut self,
        ctx: DeviceContext,
        mut x: DeviceBuffer[DType.float32],
        mutual_reach: Bool,
        core_ptr: MutPointer[Float32, MutAnyOrigin],
        inv_alpha: Float32,
        mut comp: DeviceBuffer[DType.int32],
        mut todo: DeviceBuffer[DType.int32],
        n_todo: Int,
        m: Int,
        n: Int,
        slice_rows: Int,
        qblocks: Int,
        slices: Int,
    ) raises:
        if mutual_reach:
            ctx.enqueue_function[fb_mma_nearest_kernel[D, True, MB_A, MB_B]](
                self.xc.unsafe_ptr(), self.nc.unsafe_ptr(), x.unsafe_ptr(),
                core_ptr, inv_alpha, comp.unsafe_ptr(),
                self.part_d.unsafe_ptr(), self.part_j.unsafe_ptr(),
                Int32(m), Int32(n), todo.unsafe_ptr(), Int32(n_todo),
                Int32(slice_rows),
                grid_dim=(qblocks, slices, 1), block_dim=(MB_TPB, 1, 1),
            )
        else:
            ctx.enqueue_function[fb_mma_nearest_kernel[D, False, MB_A, MB_B]](
                self.xc.unsafe_ptr(), self.nc.unsafe_ptr(), x.unsafe_ptr(),
                core_ptr, inv_alpha, comp.unsafe_ptr(),
                self.part_d.unsafe_ptr(), self.part_j.unsafe_ptr(),
                Int32(m), Int32(n), todo.unsafe_ptr(), Int32(n_todo),
                Int32(slice_rows),
                grid_dim=(qblocks, slices, 1), block_dim=(MB_TPB, 1, 1),
            )
