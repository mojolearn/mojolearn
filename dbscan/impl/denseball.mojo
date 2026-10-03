# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""FAST + Apple DBSCAN without an edge list: dense balls, early-exit cores,
union-find over landmark pairs. `-D MOJOLEARN_DBSCAN_FAST_DENSEBALL=1`.

WHY (docs/apple-fast/notes/dbscan-taxi.md). The ball-cover route
(`runner.mojo`) materializes every eps-edge as CSR and min-propagates labels
over it. On 1,000,000 standardized taxi rows at eps 3 nearly every pair is an
edge (~1e12), so the count and fill passes and every `weak_cc` pass walk
~1e12 entries, the batches split on `edge_cap`, and the race times out.

WHAT THIS DOES INSTEAD (all device work, no host loop over rows):

1. DENSE BALLS. The ball cover's index puts every row in the slice of its
   nearest landmark with that distance `d1` (slices sorted by `d1`). Two rows
   of one slice with `d1 <= h = eps/2 * (1 - 1e-3)` are within eps of each
   other (triangle inequality; the 1e-3 margin covers the float error of
   `d1` and of the squared distance). So such a dense prefix of `D >= min_pts`
   rows is a clique: every member is core and all of them are one component,
   rooted at the prefix's smallest row id. No distance is computed for them.
2. CORES of every other row: an eps count with ball-cover pruning (landmark
   bound, `d1` window by binary search) that STOPS at `min_pts`.
3. COMPONENTS: lock-free union-find (`parent`, CAS hook of the larger root
   under the smaller, so a root is always its component's smallest row id).
   One threadgroup per landmark pair (A <= B) not pruned by
   `d(A,B) > rad A + rad B + eps` looks for a core-core eps edge between
   different roots and hooks it, then stops the pair (it is retried next
   round). A pair is DONE when a full scan finds no such edge, or when every
   core row of A and B already shares one root (components only merge, so
   both are permanent). Rounds repeat until a round hooks nothing.
4. LABELS: a core row gets `root + 1`, which is the reference's `weak_cc`
   fixed point (`i + 1` min-propagated: the component's smallest core id
   + 1). A non-core row takes the smallest such label among its core eps
   neighbours, `MAX_LABEL` (noise) when it has none -- the border rule of
   DEVIATION 5130 (`runner.mojo` `border_pull_kernel`). The caller then runs
   the same `make_monotonic` + `relabel_for_skl` tail, so the output is the
   reference's labelling, not just a permutation of it.

The edge predicate is `eps_dist_sq(a, b) <= eps * eps` in Float32, the
ball-cover kernels' own (`common.mojo`), on the same rows (`x_reordered`).
Every pruning bound carries slack (`eps * (1 + 1e-3) + 1e-6`), so pruning
never drops a true edge. Unweighted, Euclidean only; the caller routes
anything else to the reference arm.
"""

from std.atomic import Atomic, Ordering
from std.gpu import block_dim, block_idx, thread_idx
from std.math import sqrt
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.primitives.block import max as block_max

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST
from core.device_zero import enqueue_fill
from neighbors.impl.ball_cover.common import eps_dist_sq


comptime DBSCAN_FAST_DENSEBALL = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and is_defined["MOJOLEARN_DBSCAN_FAST_DENSEBALL"]()
)

comptime DB_TPB = 256
comptime DB_PAIR_TPB = 64
comptime DB_MAX_LABEL = Int32(2147483647)


@always_inline
def _db_find(parent: MutPointer[Int32, MutAnyOrigin], x0: Int32) -> Int32:
    var r = x0
    while True:
        var p = Atomic.load[ordering = Ordering.RELAXED](parent + Int(r))
        if p == r:
            return r
        r = p


@always_inline
def _db_union(parent: MutPointer[Int32, MutAnyOrigin], a: Int32, b: Int32):
    var x = a
    var y = b
    while True:
        x = _db_find(parent, x)
        y = _db_find(parent, y)
        if x == y:
            return
        if x < y:
            var t = x
            x = y
            y = t
        var expected = x
        if Atomic.compare_exchange[
            success_ordering = Ordering.RELAXED,
            failure_ordering = Ordering.RELAXED,
            weak=True,
        ](parent + Int(x), expected, y):
            return


@always_inline
def _db_lower(
    d1: MutPointer[Float32, MutAnyOrigin], lo0: Int, hi0: Int, v: Float32
) -> Int:
    """First slot in [lo0, hi0) with d1 >= v (d1 ascending in a slice)."""
    var lo = lo0
    var hi = hi0
    while lo < hi:
        var mid = (lo + hi) // 2
        if d1.unsafe_load(mid) < v:
            lo = mid + 1
        else:
            hi = mid
    return lo


def db_dense_count_kernel(
    d1: MutPointer[Float32, MutAnyOrigin],
    cols: MutPointer[Int32, MutAnyOrigin],
    nearest: MutPointer[Int32, MutAnyOrigin],
    dcount: MutPointer[Int32, MutAnyOrigin],
    dmin: MutPointer[Int32, MutAnyOrigin],
    h: Float32,
    n_in: Int32,
):
    var s = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if s >= Int(n_in):
        return
    if d1.unsafe_load(s) <= h:
        var c = cols.unsafe_load(s)
        var k = Int(nearest.unsafe_load(Int(c)))
        _ = Atomic.fetch_add(dcount + k, Int32(1))
        _ = Atomic[DType.int32].min(dmin + k, c)


def db_dense_init_kernel(
    d1: MutPointer[Float32, MutAnyOrigin],
    cols: MutPointer[Int32, MutAnyOrigin],
    nearest: MutPointer[Int32, MutAnyOrigin],
    dcount: MutPointer[Int32, MutAnyOrigin],
    dmin: MutPointer[Int32, MutAnyOrigin],
    core: MutPointer[UInt8, MutAnyOrigin],
    parent: MutPointer[Int32, MutAnyOrigin],
    h: Float32,
    min_pts: Int32,
    n_in: Int32,
):
    """Dense-prefix rows of a big enough ball: core, parent = the prefix's
    smallest id. Every other row: undecided (core 0), its own root."""
    var s = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if s >= Int(n_in):
        return
    var c = cols.unsafe_load(s)
    var k = Int(nearest.unsafe_load(Int(c)))
    if d1.unsafe_load(s) <= h and dcount.unsafe_load(k) >= min_pts:
        core.unsafe_store(Int(c), UInt8(1))
        parent.unsafe_store(Int(c), dmin.unsafe_load(k))
    else:
        core.unsafe_store(Int(c), UInt8(0))
        parent.unsafe_store(Int(c), c)


def db_core_count_kernel(
    xr: MutPointer[Float32, MutAnyOrigin],
    r: MutPointer[Float32, MutAnyOrigin],
    ip: MutPointer[Int32, MutAnyOrigin],
    d1: MutPointer[Float32, MutAnyOrigin],
    cols: MutPointer[Int32, MutAnyOrigin],
    rad: MutPointer[Float32, MutAnyOrigin],
    nearest: MutPointer[Int32, MutAnyOrigin],
    dcount: MutPointer[Int32, MutAnyOrigin],
    core: MutPointer[UInt8, MutAnyOrigin],
    h: Float32,
    eps2: Float32,
    eps_s: Float32,
    min_pts: Int32,
    n_in: Int32,
    d_in: Int32,
    n_landmarks_in: Int32,
):
    """Rows outside a dense clique: count eps neighbours (self included, as
    the reference counts) and stop at `min_pts`."""
    var s = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if s >= Int(n_in):
        return
    var c = Int(cols.unsafe_load(s))
    var kq = Int(nearest.unsafe_load(c))
    if d1.unsafe_load(s) <= h and dcount.unsafe_load(kq) >= min_pts:
        return
    var d = Int(d_in)
    var need = Int(min_pts)
    var cnt = 0
    for k in range(Int(n_landmarks_in)):
        var dq = sqrt(eps_dist_sq(xr, s * d, r, k * d, d))
        if dq > rad.unsafe_load(k) + eps_s:
            continue
        var hi = Int(ip.unsafe_load(k + 1))
        var t = _db_lower(d1, Int(ip.unsafe_load(k)), hi, dq - eps_s)
        while t < hi:
            if d1.unsafe_load(t) > dq + eps_s:
                break
            if eps_dist_sq(xr, s * d, xr, t * d, d) <= eps2:
                cnt += 1
                if cnt >= need:
                    core.unsafe_store(c, UInt8(1))
                    return
            t += 1
    core.unsafe_store(c, UInt8(0))


def db_compress_kernel(parent: MutPointer[Int32, MutAnyOrigin], n_in: Int32):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_in):
        return
    parent.unsafe_store(i, _db_find(parent, Int32(i)))


def db_uniform_kernel(
    cols: MutPointer[Int32, MutAnyOrigin],
    nearest: MutPointer[Int32, MutAnyOrigin],
    core: MutPointer[UInt8, MutAnyOrigin],
    parent: MutPointer[Int32, MutAnyOrigin],
    umin: MutPointer[Int32, MutAnyOrigin],
    umax: MutPointer[Int32, MutAnyOrigin],
    n_in: Int32,
):
    """Per landmark: smallest and largest root over its core rows (after a
    compress, `parent` IS the root)."""
    var s = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if s >= Int(n_in):
        return
    var c = Int(cols.unsafe_load(s))
    if core.unsafe_load(c) == 0:
        return
    var k = Int(nearest.unsafe_load(c))
    var root = parent.unsafe_load(c)
    _ = Atomic[DType.int32].min(umin + k, root)
    _ = Atomic[DType.int32].max(umax + k, root)


def db_pair_kernel(
    xr: MutPointer[Float32, MutAnyOrigin],
    r: MutPointer[Float32, MutAnyOrigin],
    ip: MutPointer[Int32, MutAnyOrigin],
    d1: MutPointer[Float32, MutAnyOrigin],
    cols: MutPointer[Int32, MutAnyOrigin],
    rad: MutPointer[Float32, MutAnyOrigin],
    core: MutPointer[UInt8, MutAnyOrigin],
    parent: MutPointer[Int32, MutAnyOrigin],
    umin: MutPointer[Int32, MutAnyOrigin],
    umax: MutPointer[Int32, MutAnyOrigin],
    state: MutPointer[Int32, MutAnyOrigin],
    hooked: MutPointer[Int32, MutAnyOrigin],
    eps2: Float32,
    eps_s: Float32,
    d_in: Int32,
    n_landmarks_in: Int32,
):
    """One threadgroup per landmark pair (A, B), A <= B. `state`: 0 pending,
    1 done, 2 hooked this round (stops the group; retried next round)."""
    var nl = Int(n_landmarks_in)
    var p = Int(block_idx.x)
    var a = p // nl
    var b = p - a * nl
    if b < a:
        return
    var tid = Int(thread_idx.x)
    if state.unsafe_load(p) == 1:
        return
    var amax = umax.unsafe_load(a)
    var bmax = umax.unsafe_load(b)
    var amin = umin.unsafe_load(a)
    var bmin = umin.unsafe_load(b)
    if amax < 0 or bmax < 0:
        if tid == 0:
            state.unsafe_store(p, Int32(1))
        return
    if amin == amax and bmin == bmax and amin == bmin:
        if tid == 0:
            state.unsafe_store(p, Int32(1))
        return
    var d = Int(d_in)
    if a != b:
        var dab = sqrt(eps_dist_sq(r, a * d, r, b * d, d))
        if dab > rad.unsafe_load(a) + rad.unsafe_load(b) + eps_s:
            if tid == 0:
                state.unsafe_store(p, Int32(1))
            return

    var a0 = Int(ip.unsafe_load(a))
    var a1 = Int(ip.unsafe_load(a + 1))
    var b0 = Int(ip.unsafe_load(b))
    var b1 = Int(ip.unsafe_load(b + 1))
    var radb = rad.unsafe_load(b)
    var did = Int32(0)
    var si = a0 + tid
    while si < a1:
        if Atomic.load[ordering = Ordering.RELAXED](state + p) == 2:
            break
        var ci = cols.unsafe_load(si)
        if core.unsafe_load(Int(ci)) != 0:
            var dq = sqrt(eps_dist_sq(xr, si * d, r, b * d, d))
            if dq <= radb + eps_s:
                var ri = _db_find(parent, ci)
                var lo = b0
                if a == b:
                    lo = si + 1
                var t = _db_lower(d1, lo, b1, dq - eps_s)
                while t < b1:
                    if d1.unsafe_load(t) > dq + eps_s:
                        break
                    var cj = cols.unsafe_load(t)
                    if core.unsafe_load(Int(cj)) != 0:
                        var rj = _db_find(parent, cj)
                        if rj != ri:
                            if eps_dist_sq(xr, si * d, xr, t * d, d) <= eps2:
                                _db_union(parent, ci, cj)
                                did = 1
                                Atomic.store[ordering = Ordering.RELAXED](
                                    state + p, Int32(2)
                                )
                                break
                    t += 1
        if did != 0:
            break
        si += DB_PAIR_TPB
    var hit = block_max[block_size=DB_PAIR_TPB](did)
    if tid == 0:
        if hit == 0:
            state.unsafe_store(p, Int32(1))
        else:
            state.unsafe_store(p, Int32(0))
            _ = Atomic[DType.int32].max(hooked, Int32(1))


def db_label_kernel(
    xr: MutPointer[Float32, MutAnyOrigin],
    r: MutPointer[Float32, MutAnyOrigin],
    ip: MutPointer[Int32, MutAnyOrigin],
    d1: MutPointer[Float32, MutAnyOrigin],
    cols: MutPointer[Int32, MutAnyOrigin],
    rad: MutPointer[Float32, MutAnyOrigin],
    core: MutPointer[UInt8, MutAnyOrigin],
    parent: MutPointer[Int32, MutAnyOrigin],
    labels: MutPointer[Int32, MutAnyOrigin],
    eps2: Float32,
    eps_s: Float32,
    n_in: Int32,
    d_in: Int32,
    n_landmarks_in: Int32,
):
    """Core: root + 1. Non-core: the smallest core neighbour's label, or
    `MAX_LABEL` (noise). `parent` is compressed, so it holds the root."""
    var s = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if s >= Int(n_in):
        return
    var c = Int(cols.unsafe_load(s))
    if core.unsafe_load(c) != 0:
        labels.unsafe_store(c, parent.unsafe_load(c) + 1)
        return
    var d = Int(d_in)
    var best = DB_MAX_LABEL
    for k in range(Int(n_landmarks_in)):
        var dq = sqrt(eps_dist_sq(xr, s * d, r, k * d, d))
        if dq > rad.unsafe_load(k) + eps_s:
            continue
        var hi = Int(ip.unsafe_load(k + 1))
        var t = _db_lower(d1, Int(ip.unsafe_load(k)), hi, dq - eps_s)
        while t < hi:
            if d1.unsafe_load(t) > dq + eps_s:
                break
            var cj = Int(cols.unsafe_load(t))
            if core.unsafe_load(cj) != 0:
                var lj = parent.unsafe_load(cj) + 1
                if lj < best:
                    if eps_dist_sq(xr, s * d, xr, t * d, d) <= eps2:
                        best = lj
            t += 1
    labels.unsafe_store(c, best)


def dbscan_denseball_fit(
    ctx: DeviceContext,
    mut x_reordered: DeviceBuffer[DType.float32],
    mut r: DeviceBuffer[DType.float32],
    mut r_indptr: DeviceBuffer[DType.int32],
    mut r_1nn_cols: DeviceBuffer[DType.int32],
    mut r_1nn_dists: DeviceBuffer[DType.float32],
    mut r_radius: DeviceBuffer[DType.float32],
    mut nearest: DeviceBuffer[DType.int32],
    mut core: DeviceBuffer[DType.uint8],
    mut labels: DeviceBuffer[DType.int32],
    n_rows: Int,
    n_features: Int,
    n_landmarks: Int,
    eps: Float64,
    min_pts: Int,
) raises -> Int:
    """Fill `core` and `labels` (pre-relabel: `i + 1` / `MAX_LABEL`
    convention) from a built ball-cover index. Returns the union rounds."""
    var eps_f = Float32(eps)
    var eps2 = eps_f * eps_f
    var eps_s = eps_f * Float32(1.001) + Float32(1.0e-6)
    var h = Float32(0.5) * eps_f * Float32(0.999)
    var nl = n_landmarks
    var grid_n = (n_rows + DB_TPB - 1) // DB_TPB

    var parent = ctx.enqueue_create_buffer[DType.int32](n_rows)
    var dcount = ctx.enqueue_create_buffer[DType.int32](nl)
    var dmin = ctx.enqueue_create_buffer[DType.int32](nl)
    var umin = ctx.enqueue_create_buffer[DType.int32](nl)
    var umax = ctx.enqueue_create_buffer[DType.int32](nl)
    var state = ctx.enqueue_create_buffer[DType.int32](nl * nl)
    var hooked = ctx.enqueue_create_buffer[DType.int32](1)
    var h_hooked = ctx.enqueue_create_host_buffer[DType.int32](1)
    enqueue_fill(ctx, dcount, Int32(0))
    enqueue_fill(ctx, dmin, DB_MAX_LABEL)
    enqueue_fill(ctx, state, Int32(0))

    ctx.enqueue_function[db_dense_count_kernel](
        r_1nn_dists.unsafe_ptr(), r_1nn_cols.unsafe_ptr(),
        nearest.unsafe_ptr(), dcount.unsafe_ptr(), dmin.unsafe_ptr(), h,
        Int32(n_rows),
        grid_dim=(grid_n, 1, 1), block_dim=(DB_TPB, 1, 1),
    )
    ctx.enqueue_function[db_dense_init_kernel](
        r_1nn_dists.unsafe_ptr(), r_1nn_cols.unsafe_ptr(),
        nearest.unsafe_ptr(), dcount.unsafe_ptr(), dmin.unsafe_ptr(),
        core.unsafe_ptr(), parent.unsafe_ptr(), h, Int32(min_pts),
        Int32(n_rows),
        grid_dim=(grid_n, 1, 1), block_dim=(DB_TPB, 1, 1),
    )
    ctx.enqueue_function[db_core_count_kernel](
        x_reordered.unsafe_ptr(), r.unsafe_ptr(), r_indptr.unsafe_ptr(),
        r_1nn_dists.unsafe_ptr(), r_1nn_cols.unsafe_ptr(),
        r_radius.unsafe_ptr(), nearest.unsafe_ptr(), dcount.unsafe_ptr(),
        core.unsafe_ptr(), h, eps2, eps_s, Int32(min_pts), Int32(n_rows),
        Int32(n_features), Int32(nl),
        grid_dim=(grid_n, 1, 1), block_dim=(DB_TPB, 1, 1),
    )

    var rounds = 0
    while True:
        ctx.enqueue_function[db_compress_kernel](
            parent.unsafe_ptr(), Int32(n_rows),
            grid_dim=(grid_n, 1, 1), block_dim=(DB_TPB, 1, 1),
        )
        enqueue_fill(ctx, umin, DB_MAX_LABEL)
        enqueue_fill(ctx, umax, Int32(-1))
        enqueue_fill(ctx, hooked, Int32(0))
        ctx.enqueue_function[db_uniform_kernel](
            r_1nn_cols.unsafe_ptr(), nearest.unsafe_ptr(), core.unsafe_ptr(),
            parent.unsafe_ptr(), umin.unsafe_ptr(), umax.unsafe_ptr(),
            Int32(n_rows),
            grid_dim=(grid_n, 1, 1), block_dim=(DB_TPB, 1, 1),
        )
        ctx.enqueue_function[db_pair_kernel](
            x_reordered.unsafe_ptr(), r.unsafe_ptr(), r_indptr.unsafe_ptr(),
            r_1nn_dists.unsafe_ptr(), r_1nn_cols.unsafe_ptr(),
            r_radius.unsafe_ptr(), core.unsafe_ptr(), parent.unsafe_ptr(),
            umin.unsafe_ptr(), umax.unsafe_ptr(), state.unsafe_ptr(),
            hooked.unsafe_ptr(), eps2, eps_s, Int32(n_features), Int32(nl),
            grid_dim=(nl * nl, 1, 1), block_dim=(DB_PAIR_TPB, 1, 1),
        )
        ctx.enqueue_copy(dst_ptr=h_hooked.unsafe_ptr(), src_buf=hooked)
        ctx.synchronize()
        rounds += 1
        if h_hooked.unsafe_ptr().unsafe_load(0) == 0:
            break
        if rounds > n_rows + 1:
            raise Error(
                "dbscan denseball: union rounds passed n_rows + 1 ("
                + String(rounds) + "); every hooking round merges at least"
                " two components, so this is a bug"
            )

    ctx.enqueue_function[db_compress_kernel](
        parent.unsafe_ptr(), Int32(n_rows),
        grid_dim=(grid_n, 1, 1), block_dim=(DB_TPB, 1, 1),
    )
    ctx.enqueue_function[db_label_kernel](
        x_reordered.unsafe_ptr(), r.unsafe_ptr(), r_indptr.unsafe_ptr(),
        r_1nn_dists.unsafe_ptr(), r_1nn_cols.unsafe_ptr(),
        r_radius.unsafe_ptr(), core.unsafe_ptr(), parent.unsafe_ptr(),
        labels.unsafe_ptr(), eps2, eps_s, Int32(n_rows), Int32(n_features),
        Int32(nl),
        grid_dim=(grid_n, 1, 1), block_dim=(DB_TPB, 1, 1),
    )
    ctx.synchronize()
    _ = parent^
    _ = dcount^
    _ = dmin^
    _ = umin^
    _ = umax^
    _ = state^
    _ = hooked^
    _ = h_hooked^
    return rounds
