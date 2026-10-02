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
CPU column's builder (`umap/host/sparse_graph_host.mojo`): `ug_row_rho`,
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
from std.memory import bitcast
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
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
def _ug_nz(dp: UG_F32P, base: Int, k: Int, which: Int) -> Float32:
    """The `which`-th (0-based) positive distance of the row, in rank order."""
    var seen = 0
    for j in range(k):
        var d = dp[base + j]
        if d > Float32(0.0):
            if seen == which:
                return d
            seen += 1
    return Float32(0.0)


def ug_row_rho(dp: UG_F32P, row: Int, k: Int, lc: Float32, tol: UInt64) -> Float32:
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
            var rho = _ug_nz(dp, base, k, index - 1)
            if sf64_gt(interp, tol):
                var diff = _ug_nz(dp, base, k, index) - rho
                rho = sf64_to_f32(
                    sf64_fma(interp, sf64_from_f32(diff), sf64_from_f32(rho))
                )
            return rho
        if cnt > 0:
            return sf64_to_f32(sf64_mul(interp, sf64_from_f32(first)))
        return Float32(0.0)
    if cnt > 0:
        return _ug_nz(dp, base, k, cnt - 1)
    return Float32(0.0)


@always_inline
def _ug_msum(dp: UG_F32P, base: Int, k: Int, rho: UInt64, sigma: UInt64) -> UInt64:
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
    while sf64_lt(_ug_msum(dp, base, k, rho, hi), target):
        hi = sf64_mul(hi, UG_TWO)
        if sf64_gt(hi, big):
            return SF64_NAN
    var lo = SF64_ZERO
    var sigma = hi
    var step = 0
    while step < 64:
        sigma = sf64_mul(sf64_add(lo, hi), UG_HALF)
        var value = _ug_msum(dp, base, k, rho, sigma)
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
    `Float32(exp(-delta / sigma))` in binary64."""
    if delta > Float32(0.0):
        return sf64_to_f32(
            sf64_exp(sf64_div(sf64_neg(sf64_from_f32(delta)), sf64_from_f32(sigma)))
        )
    return Float32(1.0)


@always_inline
def ug_merge_weight(a: Float32, b: Float32, mix: Float32) -> Float32:
    """`mix * (a + b - a b) + (1 - mix) * a b` with every product pinned and
    ONE rounding on the intersection's product (the fma)."""
    var intersection = identical_mul(a, b)
    var union = (a + b) - intersection
    return identical_mul_add(
        Float32(1.0) - mix, intersection, identical_mul(mix, union)
    )


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
    rhos[row] = ug_row_rho(dist, row, k, lc, tol)


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


def _ug_get_offsets(ctx: DeviceContext, buf: DeviceBuffer[DType.int32], n: Int) raises -> List[Int]:
    var h = _ug_get_i32(ctx, buf, 0, n)
    var out = List[Int](capacity=n)
    for i in range(n):
        out.append(Int(h[i]))
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
# target's graph set operations of umap-learn's `umap_.py`, host code both
# columns run, every CSR row in ascending column order.
# ---------------------------------------------------------------------------


def _with_csr(
    graph: SparseFuzzySimplicialGraph, var offsets: List[Int], var indices: List[UInt32], var values: List[Float32]
) -> SparseFuzzySimplicialGraph:
    return SparseFuzzySimplicialGraph(
        graph.n_samples, graph.n_neighbors, graph.rhos.copy(), graph.sigmas.copy(),
        graph.directed_offsets.copy(), graph.directed_indices.copy(), graph.directed_values.copy(),
        offsets^, indices^, values^,
    )


def reset_local_connectivity(graph: SparseFuzzySimplicialGraph) raises -> SparseFuzzySimplicialGraph:
    """umap-learn `reset_local_connectivity`: every row divided by its
    largest stored value (sklearn `normalize(norm='max')`, one float32
    division), then S + S^T - S o S^T per cell as (a + b) - a*b with the
    product pinned (no contraction), and exact zeros eliminated."""
    var n = graph.n_samples
    var nv = List[Float32](capacity=len(graph.values))
    for row in range(n):
        var mx = Float32(0.0)
        for e in range(graph.offsets[row], graph.offsets[row + 1]):
            if graph.values[e] > mx:
                mx = graph.values[e]
        for e in range(graph.offsets[row], graph.offsets[row + 1]):
            nv.append(graph.values[e] / mx if mx > Float32(0.0) else graph.values[e])
    # the transpose: rows scattered in ascending order, so each row of it is
    # column sorted
    var toff = List[Int](length=n + 1, fill=0)
    for col in graph.indices:
        toff[Int(col) + 1] += 1
    for i in range(n):
        toff[i + 1] += toff[i]
    var cursor = toff.copy()
    var tcol = List[UInt32](length=len(graph.indices), fill=UInt32(0))
    var tval = List[Float32](length=len(graph.indices), fill=Float32(0.0))
    for i in range(n):
        for at in range(graph.offsets[i], graph.offsets[i + 1]):
            var col = Int(graph.indices[at])
            tcol[cursor[col]] = UInt32(i)
            tval[cursor[col]] = nv[at]
            cursor[col] += 1
    var offsets = List[Int]()
    var indices = List[UInt32]()
    var values = List[Float32]()
    offsets.append(0)
    for i in range(n):
        var left = graph.offsets[i]
        var right = toff[i]
        while left < graph.offsets[i + 1] or right < toff[i + 1]:
            var lc = n
            var rc = n
            if left < graph.offsets[i + 1]:
                lc = Int(graph.indices[left])
            if right < toff[i + 1]:
                rc = Int(tcol[right])
            var col = min(lc, rc)
            var a = Float32(0.0)
            var b = Float32(0.0)
            if lc == col:
                a = nv[left]
                left += 1
            if rc == col:
                b = tval[right]
                right += 1
            var w = (a + b) - identical_mul(a, b)
            if w != Float32(0.0):
                indices.append(UInt32(col))
                values.append(w)
        offsets.append(len(indices))
    return _with_csr(graph, offsets^, indices^, values^)


def categorical_intersection(
    graph: SparseFuzzySimplicialGraph, target: List[Float32], far_dist: Float64,
    unknown_dist: Float64 = Float64(1.0),
) raises -> SparseFuzzySimplicialGraph:
    """umap-learn `discrete_metric_simplicial_set_intersection` (no target
    metric): an edge whose ends carry different labels is scaled by
    exp(-far_dist), one with an unknown label (-1) by exp(-unknown_dist),
    each as Float32(Float64(w) * exp) with the portable exp; exact zeros are
    eliminated, then `reset_local_connectivity`."""
    var n = graph.n_samples
    if len(target) != n:
        raise Error("UMAP supervised target length differs from n_samples")
    var far = identical_exp64(-far_dist)
    var unknown = identical_exp64(-unknown_dist)
    var offsets = List[Int]()
    var indices = List[UInt32]()
    var values = List[Float32]()
    offsets.append(0)
    for i in range(n):
        for e in range(graph.offsets[i], graph.offsets[i + 1]):
            var j = Int(graph.indices[e])
            var w = graph.values[e]
            if target[i] == Float32(-1.0) or target[j] == Float32(-1.0):
                w = Float32(Float64(w) * unknown)
            elif target[i] != target[j]:
                w = Float32(Float64(w) * far)
            if w != Float32(0.0):
                indices.append(UInt32(j))
                values.append(w)
        offsets.append(len(indices))
    return reset_local_connectivity(_with_csr(graph, offsets^, indices^, values^))


def _csr_at(offsets: List[Int], indices: List[UInt32], values: List[Float32], row: Int, col: Int) -> Tuple[Bool, Float32]:
    for e in range(offsets[row], offsets[row + 1]):
        if Int(indices[e]) == col:
            return (True, values[e])
    return (False, Float32(0.0))


def general_intersection(
    left: SparseFuzzySimplicialGraph, right: SparseFuzzySimplicialGraph, weight: Float32,
) raises -> SparseFuzzySimplicialGraph:
    """umap-learn `general_simplicial_set_intersection` +
    `sparse.general_sset_intersection` (right_complement False): the union
    pattern of the two graphs holding left + right; a cell where either side
    beats its floor (half its graph's smallest stored value, at least 1e-8,
    in Float64) becomes left * right^(w / (1 - w)) (w < 0.5) or
    left^((1 - w) / w) * right, Float64 with the portable pow, rounded once;
    then `reset_local_connectivity`."""
    var n = left.n_samples
    if right.n_samples != n:
        raise Error("UMAP supervised target graph size differs")
    # the smallest STORED value: umap-learn's graphs have had their explicit
    # zeros eliminated, so a stored zero here counts as absent
    var lmin_v = Float32(3.4028234663852886e38)
    for v in left.values:
        if v != Float32(0.0) and v < lmin_v:
            lmin_v = v
    var rmin_v = Float32(3.4028234663852886e38)
    for v in right.values:
        if v != Float32(0.0) and v < rmin_v:
            rmin_v = v
    var left_min = max(Float64(lmin_v) / 2.0, Float64(1.0e-8))
    var right_min = max(Float64(rmin_v) / 2.0, Float64(1.0e-8))
    var w64 = Float64(weight)
    var offsets = List[Int]()
    var indices = List[UInt32]()
    var values = List[Float32]()
    offsets.append(0)
    for i in range(n):
        var a = left.offsets[i]
        var b = right.offsets[i]
        while a < left.offsets[i + 1] or b < right.offsets[i + 1]:
            var ac = n
            var bc = n
            if a < left.offsets[i + 1]:
                ac = Int(left.indices[a])
            if b < right.offsets[i + 1]:
                bc = Int(right.indices[b])
            var col = min(ac, bc)
            var lv = Float32(0.0)
            var rv = Float32(0.0)
            var has_l = False
            var has_r = False
            if ac == col:
                lv = left.values[a]
                has_l = lv != Float32(0.0)
                a += 1
            if bc == col:
                rv = right.values[b]
                has_r = rv != Float32(0.0)
                b += 1
            var out = lv + rv
            var left_val = Float64(lv) if has_l else left_min
            var right_val = Float64(rv) if has_r else right_min
            if left_val > left_min or right_val > right_min:
                if w64 < 0.5:
                    out = Float32(left_val * identical_pow64(right_val, w64 / (1.0 - w64)))
                else:
                    out = Float32(identical_pow64(left_val, (1.0 - w64) / w64) * right_val)
            indices.append(UInt32(col))
            values.append(out)
        offsets.append(len(indices))
    return reset_local_connectivity(_with_csr(left, offsets^, indices^, values^))
