# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""IVF-PQ: THE PER-CELL ARITHMETIC, ONE SOURCE FOR THE GPU AND THE CPU
(lane/algos-ann, pass 1, 2026-09-27).

Reference: cuVS `cpp/src/neighbors/ivf_pq/ivf_pq_build.cuh` (build: coarse
k-means, residuals, `train_per_subset` codebooks, `process_and_fill_codes`)
and `ivf_pq_search.cuh` + `ivf_pq_compute_similarity_impl.cuh` (the coarse
probe select, the per-query lookup table, the code-sum scan, select_k), pinned
at ~/CascadeProjects/upstream/cuvs-v26.08.00.

Every function here computes ONE output cell with a fixed, sequential
reduction order and an index tie-break. The device kernels
(`x_ann/ivf_pq_device.mojo`) run one cell per thread; the host driver
(`x_ann/host/ivf_pq_host.mojo`) runs the SAME function in a loop. So GPU and
CPU execute one instruction sequence per cell and agree bit for bit by
construction (IDENTITY_PATHS.md "The rule": PIN).

THE FIXED-ORDER DESIGN vs cuVS's NONDETERMINISM
  * codebook training: cuVS runs `kmeans_balanced` per subspace with atomics
    in its centroid update. Here: Lloyd with a strided deterministic init, a
    fixed iteration count (no tolerance compare), assignment argmin with the
    LOWER code on an exact tie (row 22), and each centroid coordinate summed
    over its members in ascending row order by one thread.
  * the coarse quantizer: the same Lloyd cells over whole rows (one
    "subspace" of width dim, n_lists codes). cuVS trains it with
    kmeans_balanced (atomics); the tree's cluster/ k-means was the planned
    reuse, but a host build importing `ivf/host/ivf_host.mojo` beside new
    code hangs the Mojo compiler on the pod (0 CPU, futex wait; progress
    file), so pass 1 carries its own fixed-order Lloyd.
  * rotation: cuVS applies a random orthogonal rotation when
    `dim % pq_dim != 0`. Here: zero padding to `rot_dim = pq_dim * pq_len`
    (NOT_IMPLEMENTED.tsv), identity otherwise, the same as their default.
  * the lookup table: cuVS fills a LUT in shared memory (optionally fp16 /
    fp8) and sums entries per candidate. Here each LUT entry is the same
    ascending fused square sum, evaluated where it is read, and the entries
    are summed over subspaces in ascending order (float32 only).
  * top-k: cuVS's warpsort breaks equal distances by feed order. Here a
    sorted insertion under the total order (distance, original id), so the
    result does not depend on the scan order (rows 11 and 23).
"""

from std.memory import bitcast
from checks.numerics import ftz, identical_mul_add, identical_div

comptime F32P = MutPointer[Float32, MutAnyOrigin]
comptime I32P = MutPointer[Int32, MutAnyOrigin]


def pq_inf() -> Float32:
    return bitcast[DType.float32](UInt32(0x7F800000))


@always_inline
def pq_residual_cell(
    e: Int, x: F32P, centers: F32P, labels: I32P, dim: Int, rot_dim: Int, dst: F32P
):
    """out[i, c] = x[i, c] - centers[label_i, c], zero past `dim`."""
    var i = e // rot_dim
    var c = e % rot_dim
    var v = Float32(0.0)
    if c < dim:
        var l = Int(labels.unsafe_load(i))
        v = ftz(ftz(x.unsafe_load(i * dim + c)) - ftz(centers.unsafe_load(l * dim + c)))
    dst.unsafe_store(e, v)


@always_inline
def pq_subdist(a: F32P, a_off: Int, b: F32P, b_off: Int, pq_len: Int) -> Float32:
    """||a - b||^2 over one subspace: ascending, one fused step per coordinate."""
    var acc = Float32(0.0)
    for t in range(pq_len):
        var diff = ftz(ftz(a.unsafe_load(a_off + t)) - ftz(b.unsafe_load(b_off + t)))
        acc = ftz(identical_mul_add(diff, diff, acc))
    return acc


@always_inline
def pq_init_cell(
    e: Int, r: F32P, n: Int, rot_dim: Int, pq_len: Int, n_codes: Int, seed: Int, cb: F32P
):
    """Codebook cell (j, code, t) starts at the residual of a strided row."""
    var t = e % pq_len
    var jc = e // pq_len
    var code = jc % n_codes
    var j = jc // n_codes
    var row = (code * 40503 + seed * 7919 + j * 131) % n
    cb.unsafe_store(e, r.unsafe_load(row * rot_dim + j * pq_len + t))


@always_inline
def pq_assign_cell(
    e: Int, r: F32P, cb: F32P, pq_dim: Int, rot_dim: Int, pq_len: Int, n_codes: Int, codes: I32P
):
    """codes[i, j] = argmin over codes of the subspace distance, the lower
    code on an exact tie."""
    var i = e // pq_dim
    var j = e % pq_dim
    var best = 0
    var bd = pq_subdist(r, i * rot_dim + j * pq_len, cb, (j * n_codes) * pq_len, pq_len)
    for c in range(1, n_codes):
        var d = pq_subdist(r, i * rot_dim + j * pq_len, cb, (j * n_codes + c) * pq_len, pq_len)
        if d < bd:
            bd = d
            best = c
    codes.unsafe_store(e, Int32(best))


@always_inline
def pq_update_cell(
    e: Int, r: F32P, codes: I32P, n: Int, pq_dim: Int, rot_dim: Int, pq_len: Int,
    n_codes: Int, cb: F32P,
):
    """Codebook cell (j, code, t) = mean of its members' coordinate, rows
    summed ascending; an empty code keeps its previous value."""
    var t = e % pq_len
    var jc = e // pq_len
    var code = jc % n_codes
    var j = jc // n_codes
    var acc = Float32(0.0)
    var cnt = 0
    for i in range(n):
        if Int(codes.unsafe_load(i * pq_dim + j)) == code:
            acc = ftz(acc + ftz(r.unsafe_load(i * rot_dim + j * pq_len + t)))
            cnt += 1
    if cnt > 0:
        cb.unsafe_store(e, ftz(identical_div(acc, Float32(cnt))))


@always_inline
def pq_coarse_dist(q: F32P, q_off: Int, centers: F32P, l: Int, dim: Int) -> Float32:
    var acc = Float32(0.0)
    for c in range(dim):
        var diff = ftz(ftz(q.unsafe_load(q_off + c)) - ftz(centers.unsafe_load(l * dim + c)))
        acc = ftz(identical_mul_add(diff, diff, acc))
    return acc


@always_inline
def pq_lut_entry(
    q: F32P, q_off: Int, centers: F32P, l: Int, dim: Int, cb: F32P, j: Int,
    code: Int, pq_len: Int, n_codes: Int,
) -> Float32:
    """One lookup-table entry: ||(q - center_l)_j - cb[j, code]||^2, the query
    residual formed exactly as the build forms the row residual."""
    var acc = Float32(0.0)
    var base = (j * n_codes + code) * pq_len
    for t in range(pq_len):
        var c = j * pq_len + t
        var qr = Float32(0.0)
        if c < dim:
            qr = ftz(ftz(q.unsafe_load(q_off + c)) - ftz(centers.unsafe_load(l * dim + c)))
        var diff = ftz(qr - ftz(cb.unsafe_load(base + t)))
        acc = ftz(identical_mul_add(diff, diff, acc))
    return acc


@always_inline
def pq_better(d: Float32, id: Int32, sd: Float32, sid: Int32) -> Bool:
    """The total order (distance, id); an empty slot (sid < 0) is worst."""
    if sid < 0:
        return True
    return d < sd or (d == sd and id < sid)


@always_inline
def pq_insert(k: Int, base: Int, d: Float32, id: Int32, out_d: F32P, out_i: I32P):
    if not pq_better(d, id, out_d.unsafe_load(base + k - 1), out_i.unsafe_load(base + k - 1)):
        return
    var s = k - 1
    while s > 0 and pq_better(d, id, out_d.unsafe_load(base + s - 1), out_i.unsafe_load(base + s - 1)):
        out_d.unsafe_store(base + s, out_d.unsafe_load(base + s - 1))
        out_i.unsafe_store(base + s, out_i.unsafe_load(base + s - 1))
        s -= 1
    out_d.unsafe_store(base + s, d)
    out_i.unsafe_store(base + s, id)


@always_inline
def pq_search_cell(
    qi: Int, queries: F32P, dim: Int, centers: F32P, n_lists: Int, offsets: I32P,
    list_indices: I32P, codes: I32P, cb: F32P, pq_dim: Int, pq_len: Int, n_codes: Int,
    k: Int, n_probes: Int, out_d: F32P, out_i: I32P, out_n: I32P,
):
    """One query: the n_probes nearest lists under (distance, list id), then
    every row in them scored by its code sum, top-k under (distance, id).
    Short rows are filled with (+inf, -1)."""
    var base = qi * k
    var q_off = qi * dim
    for s in range(k):
        out_d.unsafe_store(base + s, pq_inf())
        out_i.unsafe_store(base + s, Int32(-1))
    var prev_d = Float32(0.0)
    var prev_l = -1
    var n_cand = 0
    for _ in range(n_probes):
        # the next list after (prev_d, prev_l) in the (distance, id) order
        var best_l = -1
        var best_d = Float32(0.0)
        for l in range(n_lists):
            var d = pq_coarse_dist(queries, q_off, centers, l, dim)
            var after = prev_l < 0 or d > prev_d or (d == prev_d and l > prev_l)
            if after and (best_l < 0 or d < best_d or (d == best_d and l < best_l)):
                best_l = l
                best_d = d
        if best_l < 0:
            break
        prev_l = best_l
        prev_d = best_d
        var start = Int(offsets.unsafe_load(best_l))
        var stop = Int(offsets.unsafe_load(best_l + 1))
        for slot in range(start, stop):
            var row = Int(list_indices.unsafe_load(slot))
            var total = Float32(0.0)
            for j in range(pq_dim):
                var code = Int(codes.unsafe_load(row * pq_dim + j))
                total = ftz(total + pq_lut_entry(
                    queries, q_off, centers, best_l, dim, cb, j, code, pq_len, n_codes
                ))
            pq_insert(k, base, total, Int32(row), out_d, out_i)
            n_cand += 1
    out_n.unsafe_store(qi, Int32(n_cand))


def pq_lists_from_labels(
    labels: List[Int32], n: Int, n_lists: Int, mut offsets: List[Int32], mut list_indices: List[Int32]
):
    """The CSR lists: offsets by a counting pass, rows ascending in each list."""
    offsets = List[Int32](length=n_lists + 1, fill=Int32(0))
    for i in range(n):
        offsets[Int(labels[i]) + 1] += 1
    for l in range(n_lists):
        offsets[l + 1] += offsets[l]
    list_indices = List[Int32](length=n, fill=Int32(0))
    var fill = List[Int](length=n_lists, fill=0)
    for i in range(n):
        var l = Int(labels[i])
        list_indices[Int(offsets[l]) + fill[l]] = Int32(i)
        fill[l] += 1


@fieldwise_init
struct IvfPqIndex(Movable):
    """The built index as host lists (both drivers return this)."""

    var n_lists: Int
    var dim: Int
    var n_rows: Int
    var pq_dim: Int
    var pq_len: Int
    var n_codes: Int
    var centers: List[Float32]
    var offsets: List[Int32]
    var list_indices: List[Int32]
    var codebooks: List[Float32]
    var codes: List[Int32]


def pq_validate(n: Int, dim: Int, n_lists: Int, pq_dim: Int, pq_bits: Int, pq_iters: Int) raises:
    if n <= 0 or dim <= 0:
        raise Error("IVF-PQ: positive dimensions required")
    if n_lists <= 0 or n_lists > n:
        raise Error("IVF-PQ: n_lists must be in [1, n_rows]")
    if pq_dim <= 0 or pq_dim > dim:
        raise Error("IVF-PQ: pq_dim must be in [1, dim]")
    if pq_bits < 1 or pq_bits > 8:
        raise Error("IVF-PQ: pq_bits must be in [1, 8]")
    if (1 << pq_bits) > n:
        raise Error("IVF-PQ: 2**pq_bits codes need at least that many rows")
    if pq_iters < 0:
        raise Error("IVF-PQ: pq_kmeans_n_iters must be >= 0")


def pq_len_of(dim: Int, pq_dim: Int) -> Int:
    return (dim + pq_dim - 1) // pq_dim
