# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The CSR fuzzy graph, built on the device (lane c-cluster, 2026-10-02).

`sparse_fuzzy_simplicial_graph_device` runs every stage of umap-learn's
`fuzzy_simplicial_set` as grid-wide launches: the self-first adapter, the
input checks and rho (one thread per row), the sigma bisection (one thread
per row), the directed rows (one thread per row: duplicate max in rank
order, then an insertion sort by column), their compaction (an exclusive
scan of the row counts), the transpose (a STABLE radix sort of the entries
by column, so each transpose row keeps ascending source rows, the host
scatter's order), and the union merge (a count pass, a scan, a write pass).
Integer scans and counts have no order to pin. No host step sits inside the
build; the finished CSR is read back once into the struct the spectral
init and the optimizer take.

THE SAME WORDS ON EVERY COLUMN. The per-row arithmetic is shared with the
CPU column's builder (`umap/host/sparse_graph_host.mojo`): `ug_row_rho_kern`,
`ug_row_sigma`, `ug_member` and `ug_merge_weight` below. Their binary64
steps are `checks/soft_f64.mojo` (integer instructions, correctly rounded,
`sf64_exp` is `portable_exp64` statement for statement), since the Apple GPU
has no float64; their float32 products go through the pinned `identical_*`
seams. FAST and IDENTICAL differ only in the sigma search's early exit
(`_sigma_fast`'s `|value - target| <= 1e-5`).

Errors keep the host builder's first-error order: one Int32 key per
failure, folded by `Atomic.min` (an order-free integer min), decoded once.
"""
from std.atomic import Atomic
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import bitcast, memcpy
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    ftz,
    identical_div,
    identical_exp64,
    identical_log2_64,
    identical_mul,
    identical_mul_add,
    identical_pow64,
)
from checks.soft_f64 import (
    SF64_NAN,
    SF64_ONE,
    SF64_ZERO,
    sf64_add,
    sf64_div,
    sf64_exp,
    sf64_fma,
    sf64_from_f32,
    sf64_from_int,
    sf64_gt,
    sf64_is_nan,
    sf64_lt,
    sf64_mul,
    sf64_neg,
    sf64_pow,
    sf64_sub,
    sf64_to_f32,
    sf64_to_int,
)
from core.fast_radix_sort import fast_radix_sort_pairs_u32, frs_counts_len
from dbscan.impl.adjgraph.algo import exclusive_scan, scan_blocks_needed
from umap.graph import _finite


comptime UG_F32P = MutPointer[Float32, MutAnyOrigin]
comptime UG_U32P = MutPointer[UInt32, MutAnyOrigin]
comptime UG_I32P = MutPointer[Int32, MutAnyOrigin]
comptime UG_TPB = 256
comptime UG_TWO = UInt64(0x4000000000000000)
comptime UG_HALF = UInt64(0x3FE0000000000000)
comptime UG_NO_ERROR = Int32(0x7FFFFFFF)


struct SparseFuzzySimplicialGraph(Copyable, Movable):
    var n_samples: Int
    var n_neighbors: Int
    var rhos: List[Float32]
    var sigmas: List[Float32]
    var directed_offsets: List[Int]
    var directed_indices: List[UInt32]
    var directed_values: List[Float32]
    var offsets: List[Int]
    var indices: List[UInt32]
    var values: List[Float32]

    def __init__(
        out self, n_samples: Int, n_neighbors: Int,
        var rhos: List[Float32], var sigmas: List[Float32],
        var directed_offsets: List[Int], var directed_indices: List[UInt32],
        var directed_values: List[Float32], var offsets: List[Int],
        var indices: List[UInt32], var values: List[Float32],
    ):
        self.n_samples = n_samples
        self.n_neighbors = n_neighbors
        self.rhos = rhos^
        self.sigmas = sigmas^
        self.directed_offsets = directed_offsets^
        self.directed_indices = directed_indices^
        self.directed_values = directed_values^
        self.offsets = offsets^
        self.indices = indices^
        self.values = values^

    def logical_payload_bytes(self) -> Int:
        """Occupied scalar bytes on the supported 64-bit hosts.

        Excludes caller inputs, temporary transpose/cursors, List spare
        capacity and allocator overhead; not a resident-memory measurement.
        """
        return (
            4 * (len(self.rhos) + len(self.sigmas))
            + 8 * (len(self.directed_offsets) + len(self.offsets))
            + 4 * (len(self.directed_indices) + len(self.directed_values))
            + 4 * (len(self.indices) + len(self.values))
        )



# ---------------------------------------------------------------------------
# The per-row statements both columns run.
# ---------------------------------------------------------------------------


def ug_constants(n_neighbors: Int) -> Tuple[UInt64, UInt64, UInt64]:
    """HOST: the sigma search's target `log2(k)` (the host seam, one scalar),
    umap-learn's SMOOTH_K_TOLERANCE 1e-5 and the bracket cap 1e20, as
    binary64 words for the soft arithmetic."""
    return (
        bitcast[DType.uint64](identical_log2_64(Float64(n_neighbors))),
        bitcast[DType.uint64](Float64(1.0e-5)),
        bitcast[DType.uint64](Float64(1.0e20)),
    )


@always_inline
def _ug_nz_kern(dp: UG_F32P, base: Int, k: Int, which: Int) -> Float32:
    """The `which`-th (0-based) positive distance of the row, in rank order."""
    var seen = 0
    for j in range(k):
        var d = dp[base + j]
        if d > Float32(0.0):
            if seen == which:
                return d
            seen += 1
    return Float32(0.0)


def ug_row_rho_kern(dp: UG_F32P, row: Int, k: Int, lc: Float32, tol: UInt64) -> Float32:
    """rho: the first positive distance; at a local_connectivity other than
    1, DEVIATION 5323 (PIN): the positive distances in rank order, index =
    floor(lc), rho = nz[index - 1] + interp (nz[index] - nz[index - 1]) as
    ONE binary64 fma rounded once to Float32 when interp > 1e-5, interp *
    nz[0] when index is 0, the largest positive distance when the row has
    fewer than lc of them, 0 when it has none."""
    var base = row * k
    var cnt = 0
    var first = Float32(0.0)
    for j in range(k):
        var d = dp[base + j]
        if d > Float32(0.0):
            if cnt == 0:
                first = d
            cnt += 1
    if lc == Float32(1.0):
        return first
    var lc64 = sf64_from_f32(lc)
    if not sf64_lt(sf64_from_int(cnt), lc64):
        var index = sf64_to_int(lc64)  # floor: lc >= 0
        var interp = sf64_sub(lc64, sf64_from_int(index))
        if index > 0:
            var rho = _ug_nz_kern(dp, base, k, index - 1)
            if sf64_gt(interp, tol):
                var diff = _ug_nz_kern(dp, base, k, index) - rho
                rho = sf64_to_f32(
                    sf64_fma(interp, sf64_from_f32(diff), sf64_from_f32(rho))
                )
            return rho
        if cnt > 0:
            return sf64_to_f32(sf64_mul(interp, sf64_from_f32(first)))
        return Float32(0.0)
    if cnt > 0:
        return _ug_nz_kern(dp, base, k, cnt - 1)
    return Float32(0.0)


@always_inline
def _ug_msum_kern(dp: UG_F32P, base: Int, k: Int, rho: UInt64, sigma: UInt64) -> UInt64:
    """`sum_{j >= 1} (1 if d_j - rho <= 0 else exp(-(d_j - rho) / sigma))`,
    binary64, ascending j."""
    var total = SF64_ZERO
    for j in range(1, k):
        var d = sf64_sub(sf64_from_f32(dp[base + j]), rho)
        if sf64_gt(d, SF64_ZERO):
            total = sf64_add(total, sf64_exp(sf64_div(sf64_neg(d), sigma)))
        else:
            total = sf64_add(total, SF64_ONE)
    return total


def ug_row_sigma(
    dp: UG_F32P, row: Int, k: Int, rho_f32: Float32,
    target: UInt64, tol: UInt64, big: UInt64,
) -> UInt64:
    """The row's sigma as a binary64 word, or `SF64_NAN` when doubling from
    1 passes 1e20 without reaching `target` (the host's bracket refusal).
    IDENTICAL: 64 bisection steps, fixed. FAST: the same steps with the
    `|value - target| <= 1e-5` early exit."""
    var base = row * k
    var rho = sf64_from_f32(rho_f32)
    var hi = SF64_ONE
    while sf64_lt(_ug_msum_kern(dp, base, k, rho, hi), target):
        hi = sf64_mul(hi, UG_TWO)
        if sf64_gt(hi, big):
            return SF64_NAN
    var lo = SF64_ZERO
    var sigma = hi
    var step = 0
    while step < 64:
        sigma = sf64_mul(sf64_add(lo, hi), UG_HALF)
        var value = _ug_msum_kern(dp, base, k, rho, sigma)
        comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
            var err = sf64_sub(value, target)
            if sf64_lt(err, SF64_ZERO):
                err = sf64_neg(err)
            if not sf64_gt(err, tol):
                return sigma
        if sf64_gt(value, target):
            hi = sigma
        else:
            lo = sigma
        step += 1
    return sigma


@always_inline
def ug_member(delta: Float32, sigma: Float32) -> Float32:
    """One directed membership: 1 at or below rho, else
    `Float32(exp(-delta / sigma))` in binary64, FLUSHED: a membership that
    rounds to a subnormal is 0 on every column. Unflushed, the Apple GPU's
    merge (`a + b`) and the adapter's `weight > 0` read it as zero where every
    other column kept the edge, which shifted every later edge's ordinal and
    so its negative samples (umap / x-decomp-umap-options / par-graph-umap on
    `ties`, the 0.8.36 reference recording: tied distances drive sigma to
    its floor, and the far memberships underflow)."""
    var d = ftz(delta)
    if d > Float32(0.0):
        return ftz(sf64_to_f32(
            sf64_exp(sf64_div(sf64_neg(sf64_from_f32(d)), sf64_from_f32(sigma)))
        ))
    return Float32(1.0)


@always_inline
def ug_merge_weight(a: Float32, b: Float32, mix: Float32) -> Float32:
    """`mix * (a + b - a b) + (1 - mix) * a b` with every product pinned and
    ONE rounding on the intersection's product (the fma). Operands, the
    product and the result are flushed, so no subnormal reaches an Apple
    add or compare (`ug_member`'s note)."""
    var fa = ftz(a)
    var fb = ftz(b)
    var intersection = ftz(identical_mul(fa, fb))
    var union = ftz((fa + fb) - intersection)
    return ftz(identical_mul_add(
        Float32(1.0) - mix, intersection, ftz(identical_mul(mix, union))
    ))


# ---------------------------------------------------------------------------
# The device build.
# ---------------------------------------------------------------------------


@always_inline
def _ug_gid() -> Int:
    return Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)


@always_inline
def _ug_blocks(n: Int) -> Int:
    return max(1, (n + UG_TPB - 1) // UG_TPB)


def ug_canon_kernel(idx: UG_U32P, dist: UG_F32P, err: UG_I32P, n_in: Int32, k_in: Int32):
    """`umap/graph.mojo::canonicalize_self_neighbors` per row: self moves to
    slot 0 with distance 0, the first k-1 other candidates keep their order.
    A duplicate self records key `row`."""
    var row = _ug_gid()
    var k = Int(k_in)
    if row >= Int(n_in):
        return
    var base = row * k
    var self_slot = -1
    for col in range(k):
        if Int(idx[base + col]) == row:
            if self_slot >= 0:
                _ = Atomic.min(err, Int32(row))
                return
            self_slot = col
    if self_slot < 0:
        self_slot = k - 1
    var col = self_slot
    while col > 0:
        idx[base + col] = idx[base + col - 1]
        dist[base + col] = dist[base + col - 1]
        col -= 1
    idx[base] = UInt32(row)
    dist[base] = Float32(0.0)


def ug_rho_kernel(
    idx: UG_U32P, dist: UG_F32P, rhos: UG_F32P, err: UG_I32P,
    n_in: Int32, k_in: Int32, lc: Float32, tol: UInt64,
):
    """The host's caller-order checks (self in slot 0: key n + 2 row;
    finite, non-negative, sorted: key n + 2 row + 1), then rho."""
    var row = _ug_gid()
    var n = Int(n_in)
    var k = Int(k_in)
    if row >= n:
        return
    var base = row * k
    if Int(idx[base]) != row:
        _ = Atomic.min(err, Int32(n + 2 * row))
        return
    var previous = Float32(-1.0)
    for j in range(k):
        var d = dist[base + j]
        if not _finite(d) or d < Float32(0.0) or (j > 0 and d < previous):
            _ = Atomic.min(err, Int32(n + 2 * row + 1))
            return
        previous = d
    rhos[row] = ug_row_rho_kern(dist, row, k, lc, tol)


def ug_sigma_kernel(
    dist: UG_F32P, rhos: UG_F32P, sigmas: UG_F32P, err: UG_I32P,
    n_in: Int32, k_in: Int32, target: UInt64, tol: UInt64, big: UInt64,
):
    """One thread per row; a failed bracket records key 3n."""
    var row = _ug_gid()
    var n = Int(n_in)
    if row >= n:
        return
    var sigma = ug_row_sigma(dist, row, Int(k_in), rhos[row], target, tol, big)
    if sf64_is_nan(sigma):
        _ = Atomic.min(err, Int32(3 * n))
        sigmas[row] = Float32(0.0)
        return
    sigmas[row] = sf64_to_f32(sigma)


def ug_directed_kernel(
    idx: UG_U32P, dist: UG_F32P, rhos: UG_F32P, sigmas: UG_F32P,
    scol: UG_U32P, sval: UG_F32P, counts: UG_I32P, err: UG_I32P,
    n_in: Int32, k_in: Int32,
):
    """One directed row into its `k - 1` slots: memberships in rank order,
    a repeated column keeps the max, then an insertion sort by column. An
    invalid index records key 3n + 1 + row."""
    var row = _ug_gid()
    var n = Int(n_in)
    var k = Int(k_in)
    if row >= n:
        return
    var base = row * k
    var sb = row * (k - 1)
    var m = 0
    for j in range(1, k):
        var dst = Int(idx[base + j])
        if dst >= n or dst == row:
            _ = Atomic.min(err, Int32(3 * n + 1 + row))
            counts[row] = Int32(0)
            return
        var value = ug_member(dist[base + j] - rhos[row], sigmas[row])
        var existing = -1
        for at in range(m):
            if Int(scol[sb + at]) == dst:
                existing = at
                break
        if existing >= 0:
            if value > sval[sb + existing]:
                sval[sb + existing] = value
        else:
            scol[sb + m] = UInt32(dst)
            sval[sb + m] = value
            m += 1
    for at in range(1, m):
        var col = scol[sb + at]
        var val = sval[sb + at]
        var pos = at
        while pos > 0:
            if scol[sb + pos - 1] < col:
                break
            scol[sb + pos] = scol[sb + pos - 1]
            sval[sb + pos] = sval[sb + pos - 1]
            pos -= 1
        scol[sb + pos] = col
        sval[sb + pos] = val
    counts[row] = Int32(m)


def ug_compact_kernel(
    scol: UG_U32P, sval: UG_F32P, counts: UG_I32P, doff: UG_I32P,
    dcol: UG_U32P, dval: UG_F32P, erow: UG_U32P, n_in: Int32, k_in: Int32,
):
    """Row `row`'s slots to `[doff[row], doff[row + 1])`, with its row id."""
    var row = _ug_gid()
    if row >= Int(n_in):
        return
    var sb = row * (Int(k_in) - 1)
    var at = Int(doff[row])
    for t in range(Int(counts[row])):
        dcol[at + t] = scol[sb + t]
        dval[at + t] = sval[sb + t]
        erow[at + t] = UInt32(row)


def ug_tcount_kernel(dcol: UG_U32P, keys: UG_U32P, order: UG_U32P, tcount: UG_I32P, nnz_in: Int32):
    """Per entry: its column's count (an integer add, order-free), and the
    (column, entry) pair the stable sort takes."""
    var e = _ug_gid()
    if e >= Int(nnz_in):
        return
    var col = dcol[e]
    _ = Atomic.fetch_add(tcount.unsafe_offset(Int(col)), Int32(1))
    keys[e] = col
    order[e] = UInt32(e)


def ug_tgather_kernel(
    order: UG_U32P, erow: UG_U32P, dval: UG_F32P, tcol: UG_U32P, tval: UG_F32P, nnz_in: Int32
):
    """Sorted position p holds entry `order[p]`: its source row and value."""
    var p = _ug_gid()
    if p >= Int(nnz_in):
        return
    var e = Int(order[p])
    tcol[p] = erow[e]
    tval[p] = dval[e]


def ug_merge_kernel[WRITE: Bool](
    doff: UG_I32P, dcol: UG_U32P, dval: UG_F32P,
    toff: UG_I32P, tcol: UG_U32P, tval: UG_F32P,
    moff: UG_I32P, mcount: UG_I32P, ocol: UG_U32P, oval: UG_F32P,
    n_in: Int32, mix: Float32,
):
    """Row `row` of `S + S^T` merged in ascending column order: the count
    pass (`WRITE` false) and the write pass, the same walk."""
    var row = _ug_gid()
    var n = Int(n_in)
    if row >= n:
        return
    var left = Int(doff[row])
    var lend = Int(doff[row + 1])
    var right = Int(toff[row])
    var rend = Int(toff[row + 1])
    var out = 0
    var at = 0
    comptime if WRITE:
        at = Int(moff[row])
    while left < lend or right < rend:
        var lc = n
        var rc = n
        if left < lend:
            lc = Int(dcol[left])
        if right < rend:
            rc = Int(tcol[right])
        var col = min(lc, rc)
        var a = Float32(0.0)
        var b = Float32(0.0)
        if lc == col:
            a = dval[left]
            left += 1
        if rc == col:
            b = tval[right]
            right += 1
        comptime if WRITE:
            ocol[at + out] = UInt32(col)
            oval[at + out] = ug_merge_weight(a, b, mix)
        out += 1
    comptime if not WRITE:
        mcount[row] = Int32(out)


def ug_nonfinite_kernel(v: UG_F32P, flag: UG_I32P, n_in: Int32):
    """`flag[0] = 1` on a NaN or infinity (every writer stores the same 1)."""
    var i = _ug_gid()
    if i >= Int(n_in):
        return
    if not _finite(v[i]):
        flag[0] = Int32(1)


def ug_device_all_finite(ctx: DeviceContext, mut buf: DeviceBuffer[DType.float32], n: Int) raises -> Bool:
    """True when the first `n` values of `buf` are finite: one launch, one
    word back."""
    if n <= 0:
        return True
    var flag = ctx.enqueue_create_buffer[DType.int32](1)
    ctx.enqueue_memset(flag, Int32(0))
    ctx.enqueue_function[ug_nonfinite_kernel](
        buf.unsafe_ptr(), flag.unsafe_ptr(), Int32(n),
        grid_dim=_ug_blocks(n), block_dim=UG_TPB,
    )
    var ok = _ug_get_i32(ctx, flag, 0, 1)[0] == Int32(0)
    _ = flag^
    return ok


def _ug_scan(
    ctx: DeviceContext, mut counts: DeviceBuffer[DType.int32], n: Int
) raises -> DeviceBuffer[DType.int32]:
    """Exclusive scan of `n` counts into `n + 1` offsets (total last)."""
    var off = ctx.enqueue_create_buffer[DType.int32](n + 1)
    var bs = ctx.enqueue_create_buffer[DType.int32](scan_blocks_needed(n) + 1)
    exclusive_scan(ctx, off, counts, bs, n)
    _ = bs^
    return off^


def _ug_get_f32(ctx: DeviceContext, buf: DeviceBuffer[DType.float32], n: Int) raises -> List[Float32]:
    var out = List[Float32](length=n, fill=Float32(0.0))
    if n > 0:
        var view = buf.create_sub_buffer[DType.float32](0, n)
        ctx.enqueue_copy(dst_ptr=out.unsafe_ptr(), src_buf=view)
        ctx.synchronize()
        _ = view^
    return out^


def _ug_get_u32(ctx: DeviceContext, buf: DeviceBuffer[DType.uint32], n: Int) raises -> List[UInt32]:
    var out = List[UInt32](length=n, fill=UInt32(0))
    if n > 0:
        var view = buf.create_sub_buffer[DType.uint32](0, n)
        ctx.enqueue_copy(dst_ptr=out.unsafe_ptr(), src_buf=view)
        ctx.synchronize()
        _ = view^
    return out^


def _ug_get_i32(ctx: DeviceContext, buf: DeviceBuffer[DType.int32], at: Int, n: Int) raises -> List[Int32]:
    var out = List[Int32](length=n, fill=Int32(0))
    if n > 0:
        var view = buf.create_sub_buffer[DType.int32](at, n)
        ctx.enqueue_copy(dst_ptr=out.unsafe_ptr(), src_buf=view)
        ctx.synchronize()
        _ = view^
    return out^


def ug_widen_kernel(src: UG_I32P, dst: MutPointer[Int64, MutAnyOrigin], n_in: Int32):
    """One offset per thread, Int32 to the 64-bit `Int` the CSR lists hold."""
    var i = _ug_gid()
    if i >= Int(n_in):
        return
    dst[i] = Int64(src[i])


def _ug_get_offsets(ctx: DeviceContext, mut buf: DeviceBuffer[DType.int32], n: Int) raises -> List[Int]:
    """The offsets widened on the device (`ug_widen_kernel`) and copied
    straight into the list's storage (`Int` is 64-bit on every supported
    host): no per-element host pass (lane cpu3-neighbors, 2026-10-04)."""
    var out = List[Int](length=n, fill=0)
    if n > 0:
        var wide = ctx.enqueue_create_buffer[DType.int64](n)
        ctx.enqueue_function[ug_widen_kernel](
            buf.unsafe_ptr(), wide.unsafe_ptr(), Int32(n),
            grid_dim=_ug_blocks(n), block_dim=UG_TPB,
        )
        ctx.enqueue_copy(dst_ptr=out.unsafe_ptr().bitcast[Int64](), src_buf=wide)
        ctx.synchronize()
        _ = wide^
    return out^


def sparse_fuzzy_simplicial_graph_device(
    ctx: DeviceContext,
    mut idx: DeviceBuffer[DType.uint32],
    mut dist: DeviceBuffer[DType.float32],
    n_samples: Int, n_neighbors: Int,
    set_op_mix_ratio: Float32 = Float32(1.0),
    local_connectivity: Float32 = Float32(1.0),
    canonicalize: Bool = True,
) raises -> SparseFuzzySimplicialGraph:
    """The CSR fuzzy graph of a same-data k-NN held on the device (this
    file's header). `idx` and `dist` are `n x k` row-major, sorted by
    distance per row; `canonicalize` applies the self-first adapter to them
    in place first. Same refusals, in the same order, as the host builder."""
    var n = n_samples
    var k = n_neighbors
    if n < 2 or k < 2 or k > n:
        raise Error("invalid UMAP k-NN graph shape")
    if len(idx) < n * k or len(dist) < n * k:
        raise Error("UMAP k-NN arrays do not match their shape")
    if not _finite(set_op_mix_ratio):
        raise Error("UMAP set operation mix ratio must be finite")
    if set_op_mix_ratio < Float32(0.0) or set_op_mix_ratio > Float32(1.0):
        raise Error("UMAP set operation mix ratio must be in [0, 1]")
    if 4 * n + 1 >= Int(UG_NO_ERROR):
        raise Error("UMAP sparse graph: n_samples is past the Int32 error keys")
    var c = ug_constants(k)
    var rg = _ug_blocks(n)
    var err = ctx.enqueue_create_buffer[DType.int32](1)
    ctx.enqueue_memset(err, UG_NO_ERROR)
    var rhos = ctx.enqueue_create_buffer[DType.float32](n)
    var sigmas = ctx.enqueue_create_buffer[DType.float32](n)
    var scol = ctx.enqueue_create_buffer[DType.uint32](n * (k - 1))
    var sval = ctx.enqueue_create_buffer[DType.float32](n * (k - 1))
    var counts = ctx.enqueue_create_buffer[DType.int32](n)
    if canonicalize:
        ctx.enqueue_function[ug_canon_kernel](
            idx.unsafe_ptr(), dist.unsafe_ptr(), err.unsafe_ptr(), Int32(n), Int32(k),
            grid_dim=rg, block_dim=UG_TPB,
        )
    ctx.enqueue_function[ug_rho_kernel](
        idx.unsafe_ptr(), dist.unsafe_ptr(), rhos.unsafe_ptr(), err.unsafe_ptr(),
        Int32(n), Int32(k), local_connectivity, c[1],
        grid_dim=rg, block_dim=UG_TPB,
    )
    ctx.enqueue_function[ug_sigma_kernel](
        dist.unsafe_ptr(), rhos.unsafe_ptr(), sigmas.unsafe_ptr(), err.unsafe_ptr(),
        Int32(n), Int32(k), c[0], c[1], c[2],
        grid_dim=rg, block_dim=UG_TPB,
    )
    ctx.enqueue_function[ug_directed_kernel](
        idx.unsafe_ptr(), dist.unsafe_ptr(), rhos.unsafe_ptr(), sigmas.unsafe_ptr(),
        scol.unsafe_ptr(), sval.unsafe_ptr(), counts.unsafe_ptr(), err.unsafe_ptr(),
        Int32(n), Int32(k),
        grid_dim=rg, block_dim=UG_TPB,
    )
    var doff = _ug_scan(ctx, counts, n)
    var key = Int(_ug_get_i32(ctx, err, 0, 1)[0])
    if key != Int(UG_NO_ERROR):
        if key < n:
            raise Error("duplicate self index in UMAP neighbors")
        if key < 3 * n:
            if (key - n) % 2 == 0:
                raise Error("UMAP expects self in k-NN slot zero")
            raise Error("UMAP k-NN distances must be finite and sorted")
        if key == 3 * n:
            raise Error("UMAP sigma search did not bracket its target")
        raise Error("UMAP k-NN index is invalid or repeats self")
    var nnz = Int(_ug_get_i32(ctx, doff, n, 1)[0])
    var cap = max(nnz, 1)
    var dcol = ctx.enqueue_create_buffer[DType.uint32](cap)
    var dval = ctx.enqueue_create_buffer[DType.float32](cap)
    var erow = ctx.enqueue_create_buffer[DType.uint32](cap)
    ctx.enqueue_function[ug_compact_kernel](
        scol.unsafe_ptr(), sval.unsafe_ptr(), counts.unsafe_ptr(), doff.unsafe_ptr(),
        dcol.unsafe_ptr(), dval.unsafe_ptr(), erow.unsafe_ptr(), Int32(n), Int32(k),
        grid_dim=rg, block_dim=UG_TPB,
    )
    # The transpose: a stable sort of (column, entry) keeps every column's
    # entries in ascending source row, the host scatter's order.
    var tcount = ctx.enqueue_create_buffer[DType.int32](n)
    ctx.enqueue_memset(tcount, Int32(0))
    var keys = ctx.enqueue_create_buffer[DType.uint32](cap)
    var order = ctx.enqueue_create_buffer[DType.uint32](cap)
    var tkeys = ctx.enqueue_create_buffer[DType.uint32](cap)
    var torder = ctx.enqueue_create_buffer[DType.uint32](cap)
    var sort_counts = ctx.enqueue_create_buffer[DType.int32](max(frs_counts_len(cap), 1))
    var eg = _ug_blocks(nnz)
    if nnz > 0:
        ctx.enqueue_function[ug_tcount_kernel](
            dcol.unsafe_ptr(), keys.unsafe_ptr(), order.unsafe_ptr(), tcount.unsafe_ptr(),
            Int32(nnz), grid_dim=eg, block_dim=UG_TPB,
        )
        fast_radix_sort_pairs_u32(ctx, nnz, keys, order, tkeys, torder, sort_counts)
    var toff = _ug_scan(ctx, tcount, n)
    var tcol = ctx.enqueue_create_buffer[DType.uint32](cap)
    var tval = ctx.enqueue_create_buffer[DType.float32](cap)
    if nnz > 0:
        ctx.enqueue_function[ug_tgather_kernel](
            order.unsafe_ptr(), erow.unsafe_ptr(), dval.unsafe_ptr(),
            tcol.unsafe_ptr(), tval.unsafe_ptr(), Int32(nnz),
            grid_dim=eg, block_dim=UG_TPB,
        )
    # The union merge: count, scan, write.
    var mcount = ctx.enqueue_create_buffer[DType.int32](n)
    # the count pass reads neither `moff` nor the outputs: one-cell dummies
    var no_off = ctx.enqueue_create_buffer[DType.int32](1)
    var no_col = ctx.enqueue_create_buffer[DType.uint32](1)
    var no_val = ctx.enqueue_create_buffer[DType.float32](1)
    ctx.enqueue_function[ug_merge_kernel[False]](
        doff.unsafe_ptr(), dcol.unsafe_ptr(), dval.unsafe_ptr(),
        toff.unsafe_ptr(), tcol.unsafe_ptr(), tval.unsafe_ptr(),
        no_off.unsafe_ptr(), mcount.unsafe_ptr(), no_col.unsafe_ptr(), no_val.unsafe_ptr(),
        Int32(n), set_op_mix_ratio,
        grid_dim=rg, block_dim=UG_TPB,
    )
    var moff = _ug_scan(ctx, mcount, n)
    var mnz = Int(_ug_get_i32(ctx, moff, n, 1)[0])
    _ = no_off^
    _ = no_col^
    _ = no_val^
    var ocol = ctx.enqueue_create_buffer[DType.uint32](max(mnz, 1))
    var oval = ctx.enqueue_create_buffer[DType.float32](max(mnz, 1))
    ctx.enqueue_function[ug_merge_kernel[True]](
        doff.unsafe_ptr(), dcol.unsafe_ptr(), dval.unsafe_ptr(),
        toff.unsafe_ptr(), tcol.unsafe_ptr(), tval.unsafe_ptr(),
        moff.unsafe_ptr(), mcount.unsafe_ptr(), ocol.unsafe_ptr(), oval.unsafe_ptr(),
        Int32(n), set_op_mix_ratio,
        grid_dim=rg, block_dim=UG_TPB,
    )
    var graph = SparseFuzzySimplicialGraph(
        n, k,
        _ug_get_f32(ctx, rhos, n), _ug_get_f32(ctx, sigmas, n),
        _ug_get_offsets(ctx, doff, n + 1), _ug_get_u32(ctx, dcol, nnz),
        _ug_get_f32(ctx, dval, nnz), _ug_get_offsets(ctx, moff, n + 1),
        _ug_get_u32(ctx, ocol, mnz), _ug_get_f32(ctx, oval, mnz),
    )
    _ = err^
    _ = rhos^
    _ = sigmas^
    _ = scol^
    _ = sval^
    _ = counts^
    _ = doff^
    _ = dcol^
    _ = dval^
    _ = erow^
    _ = tcount^
    _ = keys^
    _ = order^
    _ = tkeys^
    _ = torder^
    _ = sort_counts^
    _ = toff^
    _ = tcol^
    _ = tval^
    _ = mcount^
    _ = moff^
    _ = ocol^
    _ = oval^
    return graph^


# ---------------------------------------------------------------------------
# Supervised UMAP (lane/algos-decomp, 2026-09-27; DEVIATION 5324, PIN): the
# target's graph set operations of umap-learn's `umap_.py`, every CSR row in
# ascending column order.
#
# ON THE DEVICE (lane cpu4-umap, 2026-10-04). These were host walks over
# every stored edge inside the supervised GPU fit. Each now runs as grid-wide
# launches: the graph CSR goes up once, a count pass, a scan and a write
# pass build each result (one thread per row), the transpose of
# `reset_local_connectivity` is the build's stable radix sort by column, and
# the finished CSR comes back once into the struct the spectral init and the
# optimizer take. The per-cell statements below (`ug_reset_scale`,
# `ug_reset_weight`, `ug_categorical_weight`, `ug_intersect_cell`) are shared
# with the host column (`umap/host/sparse_graph_host.mojo::host_*`): binary64
# steps in the soft arithmetic (`sf64_pow` is `portable_pow64` statement for
# statement), float32 steps pinned and flushed, so both columns return the
# same words. Bits moved against the old host walks only where a subnormal
# was involved (the flush model), and in FAST where the pow was the stdlib's.
# ---------------------------------------------------------------------------


comptime UG_FLT_MAX_BITS = Int32(0x7F7FFFFF)
comptime UG_NO_NEG = Int32(-2147483648)


@always_inline
def ug_reset_scale(v: Float32, mx: Float32) -> Float32:
    """sklearn `normalize(norm='max')`'s cell: one float32 division by the
    row's largest stored value (the value itself in a row with none above 0)."""
    if mx > Float32(0.0):
        return ftz(identical_div(v, mx))
    return ftz(v)


@always_inline
def ug_reset_weight(a: Float32, b: Float32) -> Float32:
    """`S + S^T - S o S^T` per cell as `(a + b) - a b`, the product pinned."""
    var fa = ftz(a)
    var fb = ftz(b)
    return ftz(ftz(fa + fb) - ftz(identical_mul(fa, fb)))


@always_inline
def ug_categorical_weight(w: Float32, ti: Float32, tj: Float32, unknown: UInt64, far: UInt64) -> Float32:
    """An edge whose ends carry different labels scaled by exp(-far_dist),
    one with an unknown label (-1) by exp(-unknown_dist), each as
    `Float32(Float64(w) * exp)`; flushed."""
    if ti == Float32(-1.0) or tj == Float32(-1.0):
        return ftz(sf64_to_f32(sf64_mul(sf64_from_f32(w), unknown)))
    if ti != tj:
        return ftz(sf64_to_f32(sf64_mul(sf64_from_f32(w), far)))
    return ftz(w)


@always_inline
def ug_intersect_cell(
    lv_in: Float32, rv_in: Float32, left_min: UInt64, right_min: UInt64, low: Bool, expo: UInt64
) -> Float32:
    """One union cell of `general_sset_intersection`: `left + right`, or,
    when either side beats its floor (an absent or zero side reads as its
    floor), `left * right^expo` (`low`, weight < 0.5) or `left^expo * right`
    in binary64 with the portable pow, rounded once; flushed."""
    var lv = ftz(lv_in)
    var rv = ftz(rv_in)
    var out = ftz(lv + rv)
    var left_val = sf64_from_f32(lv) if lv != Float32(0.0) else left_min
    var right_val = sf64_from_f32(rv) if rv != Float32(0.0) else right_min
    if sf64_gt(left_val, left_min) or sf64_gt(right_val, right_min):
        if low:
            out = ftz(sf64_to_f32(sf64_mul(left_val, sf64_pow(right_val, expo))))
        else:
            out = ftz(sf64_to_f32(sf64_mul(sf64_pow(left_val, expo), right_val)))
    return out


def ug_categorical_constants(far_dist: Float64, unknown_dist: Float64) -> Tuple[UInt64, UInt64]:
    """HOST scalars both columns call: exp(-unknown_dist), exp(-far_dist)."""
    return (
        bitcast[DType.uint64](identical_exp64(-unknown_dist)),
        bitcast[DType.uint64](identical_exp64(-far_dist)),
    )


def ug_intersect_floor(min_stored: Float32) -> UInt64:
    """HOST scalar: half a graph's smallest stored value, at least 1e-8."""
    return bitcast[DType.uint64](max(Float64(min_stored) / 2.0, Float64(1.0e-8)))


def ug_intersect_expo(weight: Float32) -> Tuple[Bool, UInt64]:
    """HOST scalar: (weight < 0.5, w / (1 - w) or (1 - w) / w) in Float64."""
    var w64 = Float64(weight)
    if w64 < 0.5:
        return (True, bitcast[DType.uint64](w64 / (1.0 - w64)))
    return (False, bitcast[DType.uint64]((1.0 - w64) / w64))


def ug_min_stored(pos_bits: Int32, neg_bits: Int32) -> Float32:
    """HOST: a graph's smallest nonzero stored value from the device's two
    words (the most negative value's bits by `Atomic.max`, else the smallest
    positive one's by `Atomic.min` from FLT_MAX), as the host scan's
    `v != 0 and v < m` from m = FLT_MAX finds it."""
    if neg_bits != UG_NO_NEG:
        return bitcast[DType.float32](neg_bits)
    return bitcast[DType.float32](pos_bits)


def ug_min_stored_kernel(v: UG_F32P, pos: UG_I32P, neg: UG_I32P, n_in: Int32):
    """`pos[0]` the smallest positive value's bits (Atomic.min, from
    FLT_MAX: infinities and NaNs never win), `neg[0]` the most negative
    value's bits (Atomic.max over the signed words, from INT32_MIN)."""
    var i = _ug_gid()
    if i >= Int(n_in):
        return
    var x = ftz(v[i])
    if x > Float32(0.0):
        _ = Atomic.min(pos, bitcast[DType.int32](x))
    elif x < Float32(0.0):
        _ = Atomic.max(neg, bitcast[DType.int32](x))


def ug_narrow_kernel(src: MutPointer[Int64, MutAnyOrigin], dst: UG_I32P, n_in: Int32):
    """One offset per thread, the CSR list's 64-bit `Int` to Int32."""
    var i = _ug_gid()
    if i >= Int(n_in):
        return
    dst[i] = Int32(src[i])


def ug_row_ids_kernel(off: UG_I32P, erow: UG_U32P, n_in: Int32):
    """Each stored entry's row id (one thread per row)."""
    var row = _ug_gid()
    if row >= Int(n_in):
        return
    for e in range(Int(off[row]), Int(off[row + 1])):
        erow[e] = UInt32(row)


def ug_reset_scale_kernel(off: UG_I32P, val: UG_F32P, nv: UG_F32P, n_in: Int32):
    """Row `row` divided by its largest stored value (strict >, from 0)."""
    var row = _ug_gid()
    if row >= Int(n_in):
        return
    var b = Int(off[row])
    var end = Int(off[row + 1])
    var mx = Float32(0.0)
    for e in range(b, end):
        var v = ftz(val[e])
        if v > mx:
            mx = v
    for e in range(b, end):
        nv[e] = ug_reset_scale(val[e], mx)


def ug_reset_merge_kernel[WRITE: Bool](
    off: UG_I32P, col: UG_U32P, val: UG_F32P,
    toff: UG_I32P, tcol: UG_U32P, tval: UG_F32P,
    moff: UG_I32P, mcount: UG_I32P, ocol: UG_U32P, oval: UG_F32P, n_in: Int32,
):
    """Row `row` of `S + S^T - S o S^T` in ascending column order, exact
    zeros eliminated: the count pass (`WRITE` false) and the write pass."""
    var row = _ug_gid()
    var n = Int(n_in)
    if row >= n:
        return
    var left = Int(off[row])
    var lend = Int(off[row + 1])
    var right = Int(toff[row])
    var rend = Int(toff[row + 1])
    var out = 0
    var at = 0
    comptime if WRITE:
        at = Int(moff[row])
    while left < lend or right < rend:
        var lc = n
        var rc = n
        if left < lend:
            lc = Int(col[left])
        if right < rend:
            rc = Int(tcol[right])
        var c = min(lc, rc)
        var a = Float32(0.0)
        var b = Float32(0.0)
        if lc == c:
            a = val[left]
            left += 1
        if rc == c:
            b = tval[right]
            right += 1
        var w = ug_reset_weight(a, b)
        if w != Float32(0.0):
            comptime if WRITE:
                ocol[at + out] = UInt32(c)
                oval[at + out] = w
            out += 1
    comptime if not WRITE:
        mcount[row] = Int32(out)


def ug_categorical_kernel[WRITE: Bool](
    off: UG_I32P, col: UG_U32P, val: UG_F32P, target: UG_F32P,
    moff: UG_I32P, mcount: UG_I32P, ocol: UG_U32P, oval: UG_F32P,
    n_in: Int32, unknown: UInt64, far: UInt64,
):
    """Row `row` of the label-scaled graph, exact zeros eliminated: the
    count pass (`WRITE` false) and the write pass."""
    var row = _ug_gid()
    if row >= Int(n_in):
        return
    var ti = target[row]
    var out = 0
    var at = 0
    comptime if WRITE:
        at = Int(moff[row])
    for e in range(Int(off[row]), Int(off[row + 1])):
        var j = Int(col[e])
        var w = ug_categorical_weight(val[e], ti, target[j], unknown, far)
        if w != Float32(0.0):
            comptime if WRITE:
                ocol[at + out] = UInt32(j)
                oval[at + out] = w
            out += 1
    comptime if not WRITE:
        mcount[row] = Int32(out)


def ug_intersect_kernel[WRITE: Bool](
    loff: UG_I32P, lcol: UG_U32P, lval: UG_F32P,
    roff: UG_I32P, rcol: UG_U32P, rval: UG_F32P,
    moff: UG_I32P, mcount: UG_I32P, ocol: UG_U32P, oval: UG_F32P,
    n_in: Int32, left_min: UInt64, right_min: UInt64, low: Int32, expo: UInt64,
):
    """Row `row` of the union pattern of the two graphs, every cell kept:
    the count pass (`WRITE` false) and the write pass."""
    var row = _ug_gid()
    var n = Int(n_in)
    if row >= n:
        return
    var a = Int(loff[row])
    var aend = Int(loff[row + 1])
    var b = Int(roff[row])
    var bend = Int(roff[row + 1])
    var out = 0
    var at = 0
    comptime if WRITE:
        at = Int(moff[row])
    while a < aend or b < bend:
        var ac = n
        var bc = n
        if a < aend:
            ac = Int(lcol[a])
        if b < bend:
            bc = Int(rcol[b])
        var c = min(ac, bc)
        var lv = Float32(0.0)
        var rv = Float32(0.0)
        if ac == c:
            lv = lval[a]
            a += 1
        if bc == c:
            rv = rval[b]
            b += 1
        comptime if WRITE:
            ocol[at + out] = UInt32(c)
            oval[at + out] = ug_intersect_cell(lv, rv, left_min, right_min, low != Int32(0), expo)
        out += 1
    comptime if not WRITE:
        mcount[row] = Int32(out)


struct UgDeviceCsr(Movable):
    """A CSR on the device: Int32 offsets (n + 1), columns and values."""

    var off: DeviceBuffer[DType.int32]
    var col: DeviceBuffer[DType.uint32]
    var val: DeviceBuffer[DType.float32]
    var nnz: Int

    def __init__(
        out self,
        var off: DeviceBuffer[DType.int32],
        var col: DeviceBuffer[DType.uint32],
        var val: DeviceBuffer[DType.float32],
        nnz: Int,
    ):
        self.off = off^
        self.col = col^
        self.val = val^
        self.nnz = nnz


def _ug_csr_up(ctx: DeviceContext, graph: SparseFuzzySimplicialGraph) raises -> UgDeviceCsr:
    """The graph's union CSR up once (bulk copies through pinned staging);
    offsets narrowed to Int32 on the device."""
    var n = graph.n_samples
    var nnz = len(graph.indices)
    if len(graph.offsets) != n + 1 or len(graph.values) != nnz:
        raise Error("UMAP sparse graph shape mismatch")
    if n > 2147483646 or nnz > 2147483647:
        raise Error("UMAP sparse graph exceeds the kernel Int32 range")
    var cap = max(nnz, 1)
    var h_off = ctx.enqueue_create_host_buffer[DType.int64](n + 1)
    var h_col = ctx.enqueue_create_host_buffer[DType.uint32](cap)
    var h_val = ctx.enqueue_create_host_buffer[DType.float32](cap)
    ctx.synchronize()
    memcpy(dest=h_off.unsafe_ptr(), src=graph.offsets.unsafe_ptr().bitcast[Int64](), count=n + 1)
    if nnz > 0:
        memcpy(dest=h_col.unsafe_ptr(), src=graph.indices.unsafe_ptr(), count=nnz)
        memcpy(dest=h_val.unsafe_ptr(), src=graph.values.unsafe_ptr(), count=nnz)
    var off64 = ctx.enqueue_create_buffer[DType.int64](n + 1)
    var off = ctx.enqueue_create_buffer[DType.int32](n + 1)
    var col = ctx.enqueue_create_buffer[DType.uint32](cap)
    var val = ctx.enqueue_create_buffer[DType.float32](cap)
    ctx.enqueue_copy(dst_buf=off64, src_ptr=h_off.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=col, src_ptr=h_col.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=val, src_ptr=h_val.unsafe_ptr())
    ctx.enqueue_function[ug_narrow_kernel](
        off64.unsafe_ptr(), off.unsafe_ptr(), Int32(n + 1),
        grid_dim=_ug_blocks(n + 1), block_dim=UG_TPB,
    )
    ctx.synchronize()
    _ = h_off^
    _ = h_col^
    _ = h_val^
    _ = off64^
    return UgDeviceCsr(off^, col^, val^, nnz)


def _ug_reset_device(
    ctx: DeviceContext, graph: SparseFuzzySimplicialGraph, mut csr: UgDeviceCsr
) raises -> SparseFuzzySimplicialGraph:
    """`reset_local_connectivity` of a device CSR: the row max scale, the
    transpose (a STABLE sort of (column, entry) keeps each column's entries
    in ascending source row, the host scatter's order), the zero-eliminating
    union merge; the result back once into `graph`'s struct."""
    var n = graph.n_samples
    var nnz = csr.nnz
    var cap = max(nnz, 1)
    var rg = _ug_blocks(n)
    var nv = ctx.enqueue_create_buffer[DType.float32](cap)
    var erow = ctx.enqueue_create_buffer[DType.uint32](cap)
    ctx.enqueue_function[ug_reset_scale_kernel](
        csr.off.unsafe_ptr(), csr.val.unsafe_ptr(), nv.unsafe_ptr(), Int32(n),
        grid_dim=rg, block_dim=UG_TPB,
    )
    ctx.enqueue_function[ug_row_ids_kernel](
        csr.off.unsafe_ptr(), erow.unsafe_ptr(), Int32(n),
        grid_dim=rg, block_dim=UG_TPB,
    )
    var tcount = ctx.enqueue_create_buffer[DType.int32](n)
    ctx.enqueue_memset(tcount, Int32(0))
    var keys = ctx.enqueue_create_buffer[DType.uint32](cap)
    var order = ctx.enqueue_create_buffer[DType.uint32](cap)
    var tkeys = ctx.enqueue_create_buffer[DType.uint32](cap)
    var torder = ctx.enqueue_create_buffer[DType.uint32](cap)
    var sort_counts = ctx.enqueue_create_buffer[DType.int32](max(frs_counts_len(cap), 1))
    var eg = _ug_blocks(nnz)
    if nnz > 0:
        ctx.enqueue_function[ug_tcount_kernel](
            csr.col.unsafe_ptr(), keys.unsafe_ptr(), order.unsafe_ptr(), tcount.unsafe_ptr(),
            Int32(nnz), grid_dim=eg, block_dim=UG_TPB,
        )
        fast_radix_sort_pairs_u32(ctx, nnz, keys, order, tkeys, torder, sort_counts)
    var toff = _ug_scan(ctx, tcount, n)
    var tcol = ctx.enqueue_create_buffer[DType.uint32](cap)
    var tval = ctx.enqueue_create_buffer[DType.float32](cap)
    if nnz > 0:
        ctx.enqueue_function[ug_tgather_kernel](
            order.unsafe_ptr(), erow.unsafe_ptr(), nv.unsafe_ptr(),
            tcol.unsafe_ptr(), tval.unsafe_ptr(), Int32(nnz),
            grid_dim=eg, block_dim=UG_TPB,
        )
    var mcount = ctx.enqueue_create_buffer[DType.int32](n)
    var no_off = ctx.enqueue_create_buffer[DType.int32](1)
    var no_col = ctx.enqueue_create_buffer[DType.uint32](1)
    var no_val = ctx.enqueue_create_buffer[DType.float32](1)
    ctx.enqueue_function[ug_reset_merge_kernel[False]](
        csr.off.unsafe_ptr(), csr.col.unsafe_ptr(), nv.unsafe_ptr(),
        toff.unsafe_ptr(), tcol.unsafe_ptr(), tval.unsafe_ptr(),
        no_off.unsafe_ptr(), mcount.unsafe_ptr(), no_col.unsafe_ptr(), no_val.unsafe_ptr(),
        Int32(n), grid_dim=rg, block_dim=UG_TPB,
    )
    var moff = _ug_scan(ctx, mcount, n)
    var mnz = Int(_ug_get_i32(ctx, moff, n, 1)[0])
    var ocol = ctx.enqueue_create_buffer[DType.uint32](max(mnz, 1))
    var oval = ctx.enqueue_create_buffer[DType.float32](max(mnz, 1))
    ctx.enqueue_function[ug_reset_merge_kernel[True]](
        csr.off.unsafe_ptr(), csr.col.unsafe_ptr(), nv.unsafe_ptr(),
        toff.unsafe_ptr(), tcol.unsafe_ptr(), tval.unsafe_ptr(),
        moff.unsafe_ptr(), mcount.unsafe_ptr(), ocol.unsafe_ptr(), oval.unsafe_ptr(),
        Int32(n), grid_dim=rg, block_dim=UG_TPB,
    )
    var out = SparseFuzzySimplicialGraph(
        n, graph.n_neighbors, graph.rhos.copy(), graph.sigmas.copy(),
        graph.directed_offsets.copy(), graph.directed_indices.copy(), graph.directed_values.copy(),
        _ug_get_offsets(ctx, moff, n + 1), _ug_get_u32(ctx, ocol, mnz), _ug_get_f32(ctx, oval, mnz),
    )
    _ = nv^
    _ = erow^
    _ = tcount^
    _ = keys^
    _ = order^
    _ = tkeys^
    _ = torder^
    _ = sort_counts^
    _ = toff^
    _ = tcol^
    _ = tval^
    _ = mcount^
    _ = no_off^
    _ = no_col^
    _ = no_val^
    _ = moff^
    _ = ocol^
    _ = oval^
    return out^


def reset_local_connectivity(ctx: DeviceContext, graph: SparseFuzzySimplicialGraph) raises -> SparseFuzzySimplicialGraph:
    """umap-learn `reset_local_connectivity` on the device: every row divided
    by its largest stored value, then S + S^T - S o S^T per cell, exact
    zeros eliminated (`_ug_reset_device`)."""
    var csr = _ug_csr_up(ctx, graph)
    return _ug_reset_device(ctx, graph, csr)


def categorical_intersection(
    ctx: DeviceContext, graph: SparseFuzzySimplicialGraph, target: List[Float32], far_dist: Float64,
    unknown_dist: Float64 = Float64(1.0),
) raises -> SparseFuzzySimplicialGraph:
    """umap-learn `discrete_metric_simplicial_set_intersection` (no target
    metric) on the device: each edge scaled by `ug_categorical_weight`,
    exact zeros eliminated (count, scan, write), then the reset."""
    var n = graph.n_samples
    if len(target) != n:
        raise Error("UMAP supervised target length differs from n_samples")
    var consts = ug_categorical_constants(far_dist, unknown_dist)
    var src = _ug_csr_up(ctx, graph)
    var h_t = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.synchronize()
    memcpy(dest=h_t.unsafe_ptr(), src=target.unsafe_ptr(), count=n)
    var d_t = ctx.enqueue_create_buffer[DType.float32](n)
    ctx.enqueue_copy(dst_buf=d_t, src_ptr=h_t.unsafe_ptr())
    var rg = _ug_blocks(n)
    var mcount = ctx.enqueue_create_buffer[DType.int32](n)
    var no_off = ctx.enqueue_create_buffer[DType.int32](1)
    var no_col = ctx.enqueue_create_buffer[DType.uint32](1)
    var no_val = ctx.enqueue_create_buffer[DType.float32](1)
    ctx.enqueue_function[ug_categorical_kernel[False]](
        src.off.unsafe_ptr(), src.col.unsafe_ptr(), src.val.unsafe_ptr(), d_t.unsafe_ptr(),
        no_off.unsafe_ptr(), mcount.unsafe_ptr(), no_col.unsafe_ptr(), no_val.unsafe_ptr(),
        Int32(n), consts[0], consts[1], grid_dim=rg, block_dim=UG_TPB,
    )
    var moff = _ug_scan(ctx, mcount, n)
    var mnz = Int(_ug_get_i32(ctx, moff, n, 1)[0])
    var ocol = ctx.enqueue_create_buffer[DType.uint32](max(mnz, 1))
    var oval = ctx.enqueue_create_buffer[DType.float32](max(mnz, 1))
    ctx.enqueue_function[ug_categorical_kernel[True]](
        src.off.unsafe_ptr(), src.col.unsafe_ptr(), src.val.unsafe_ptr(), d_t.unsafe_ptr(),
        moff.unsafe_ptr(), mcount.unsafe_ptr(), ocol.unsafe_ptr(), oval.unsafe_ptr(),
        Int32(n), consts[0], consts[1], grid_dim=rg, block_dim=UG_TPB,
    )
    _ = h_t^
    _ = d_t^
    _ = mcount^
    _ = no_off^
    _ = no_col^
    _ = no_val^
    _ = src^
    var scaled = UgDeviceCsr(moff^, ocol^, oval^, mnz)
    return _ug_reset_device(ctx, graph, scaled)


def _ug_min_stored_device(ctx: DeviceContext, mut csr: UgDeviceCsr) raises -> Float32:
    """A device CSR's smallest nonzero stored value, flushed (two words back)."""
    var pos = ctx.enqueue_create_buffer[DType.int32](1)
    var neg = ctx.enqueue_create_buffer[DType.int32](1)
    ctx.enqueue_memset(pos, UG_FLT_MAX_BITS)
    ctx.enqueue_memset(neg, UG_NO_NEG)
    if csr.nnz > 0:
        ctx.enqueue_function[ug_min_stored_kernel](
            csr.val.unsafe_ptr(), pos.unsafe_ptr(), neg.unsafe_ptr(), Int32(csr.nnz),
            grid_dim=_ug_blocks(csr.nnz), block_dim=UG_TPB,
        )
    var p = _ug_get_i32(ctx, pos, 0, 1)[0]
    var q = _ug_get_i32(ctx, neg, 0, 1)[0]
    _ = pos^
    _ = neg^
    return ug_min_stored(p, q)


def general_intersection(
    ctx: DeviceContext, left: SparseFuzzySimplicialGraph, right: SparseFuzzySimplicialGraph, weight: Float32,
) raises -> SparseFuzzySimplicialGraph:
    """umap-learn `general_simplicial_set_intersection` +
    `sparse.general_sset_intersection` (right_complement False) on the
    device: the union pattern of the two graphs, each cell
    `ug_intersect_cell` against the floors (half each graph's smallest
    stored value, at least 1e-8; umap-learn's graphs have had their explicit
    zeros eliminated, so a stored zero counts as absent); then the reset."""
    var n = left.n_samples
    if right.n_samples != n:
        raise Error("UMAP supervised target graph size differs")
    var lcsr = _ug_csr_up(ctx, left)
    var rcsr = _ug_csr_up(ctx, right)
    var left_min = ug_intersect_floor(_ug_min_stored_device(ctx, lcsr))
    var right_min = ug_intersect_floor(_ug_min_stored_device(ctx, rcsr))
    var ex = ug_intersect_expo(weight)
    var low = Int32(1) if ex[0] else Int32(0)
    var rg = _ug_blocks(n)
    var mcount = ctx.enqueue_create_buffer[DType.int32](n)
    var no_off = ctx.enqueue_create_buffer[DType.int32](1)
    var no_col = ctx.enqueue_create_buffer[DType.uint32](1)
    var no_val = ctx.enqueue_create_buffer[DType.float32](1)
    ctx.enqueue_function[ug_intersect_kernel[False]](
        lcsr.off.unsafe_ptr(), lcsr.col.unsafe_ptr(), lcsr.val.unsafe_ptr(),
        rcsr.off.unsafe_ptr(), rcsr.col.unsafe_ptr(), rcsr.val.unsafe_ptr(),
        no_off.unsafe_ptr(), mcount.unsafe_ptr(), no_col.unsafe_ptr(), no_val.unsafe_ptr(),
        Int32(n), left_min, right_min, low, ex[1], grid_dim=rg, block_dim=UG_TPB,
    )
    var moff = _ug_scan(ctx, mcount, n)
    var mnz = Int(_ug_get_i32(ctx, moff, n, 1)[0])
    var ocol = ctx.enqueue_create_buffer[DType.uint32](max(mnz, 1))
    var oval = ctx.enqueue_create_buffer[DType.float32](max(mnz, 1))
    ctx.enqueue_function[ug_intersect_kernel[True]](
        lcsr.off.unsafe_ptr(), lcsr.col.unsafe_ptr(), lcsr.val.unsafe_ptr(),
        rcsr.off.unsafe_ptr(), rcsr.col.unsafe_ptr(), rcsr.val.unsafe_ptr(),
        moff.unsafe_ptr(), mcount.unsafe_ptr(), ocol.unsafe_ptr(), oval.unsafe_ptr(),
        Int32(n), left_min, right_min, low, ex[1], grid_dim=rg, block_dim=UG_TPB,
    )
    _ = mcount^
    _ = no_off^
    _ = no_col^
    _ = no_val^
    _ = lcsr^
    _ = rcsr^
    var merged = UgDeviceCsr(moff^, ocol^, oval^, mnz)
    return _ug_reset_device(ctx, left, merged)
