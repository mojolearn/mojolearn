# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Derivative work: the upstream file and its pinned commit are recorded in this lane's PORTED_MAP.tsv and in this file's own docstring. See NOTICE.
"""`raft/sparse/op/detail/sort.h::coo_sort`, `raft/sparse/op/detail/
filter.cuh::coo_remove_scalar` and `raft/sparse/convert/detail/csr.cuh::
sorted_coo_to_csr`, as HOST index plumbing.

============ DEVIATION 775: THRUST/CUB INDEX PLUMBING RUNS ON THE HOST, AND
============ REPEATED (row, col) KEYS ARE REFUSED BY NAME ==================
THEIRS: `coo_sort` (`sort.h:62-70`) is `thrust::sort_by_key` over a
`(row, col)` zip with the `TupleComp` comparator (`sort.h:33-50`); NOT
stable, so two entries with the same key land in an unspecified order.
`coo_remove_scalar` is a per-row compaction kernel (`filter.cuh:39-85`)
behind two `exclusive_scan`s, reached by cuVS through the `raft::resources`
overload (`filter.cuh:197-250`); `sorted_coo_to_csr` is `coo_degree` plus
an `exclusive_scan` (`csr.cuh:78-90`). All integer
work; none of it rounds a float.
OURS: the same three operations on the host, over `List[Int32]`/
`List[Float32]`, because (a) the repository has no device sort to port them
onto without writing one, and (b) the outputs are pure functions of the
input regardless of where they run -- a sort to a TOTAL ORDER `(row, col,
original index)` has exactly one answer, a compaction of nonzeros in that
order has exactly one answer, and row offsets are a prefix sum of integer
counts. What a HOST sort cannot reproduce is thrust's unspecified order
among EQUAL keys, so instead of choosing one we REFUSE a COO whose sorted
form has two entries with the same `(row, col)`, by name: every downstream
stage (the degree fold, the diagonal lookup, the matvec's per-row
contraction) would otherwise be a function of a tie-break theirs never
defined. The kNN path cannot produce a repeat (`coo_symmetrize` emits each
ordered pair at most once); a precomputed graph that does gets the raise.
MEASURED: `check_spectral_refusals_host` plants a repeated pair and gets
the raise; `check_spectral_device_equals_oracle`'s hashed-graph fixture is
dedup'd on purpose and passes through.
============ DEVIATION 777: THE REFUSAL'S PLACE. `coo_sort` IS A PURE SORT
============ AND THE REPEATED-KEY REFUSAL MOVED ONTO THE LAPLACIAN'S INPUT
THE DEFECT, found by running the gate for the first time on 2026-08-23.
DEVIATION 775's refusal was spelled INSIDE `coo_sort`, and the kNN path
died on it with `repeated (row, col) pair (0, 0)` before a single bit was
compared. It was right about the data and wrong about the place:
`coo_symmetrize` writes into `2 * nnz` slots and leaves the unused ones at
the ZERO the caller memset them to, so its output legitimately carries a
run of `(0, 0, 0.0)` padding entries -- and cuVS sorts THAT
(`spectral_embedding.cu:95-101`) and only then compacts it with
`coo_remove_scalar(0)` (`:103-108`). A sort that refuses repeats cannot
stand where theirs sorts.
OURS: `coo_sort` is now their sort and nothing else. `refuse_repeated_keys`
carries DEVIATION 775's refusal and is called on the ROW-SORTED graph that
enters the Laplacian, on the device arm and in the oracle alike -- the one
place a repeat is actually a tie-break theirs never defined (the degree
fold sums both, `diagonal`'s scatter races, and the matvec walks both).
The padding never reaches it: `coo_remove_scalar(0)` has already run.
MEASURED: the refusal still fires on a planted repeat (through
`host_laplacian`, `check_spectral_refusals_host`), and the blobs kNN path
now reaches the card.
======================================================================
"""

from spectral.ported.sparse.coo import CooGraph


def coo_sort(g: CooGraph) raises -> CooGraph:
    """`sort.h:62-70` to the total order `(row, col, original index)`.

    A PURE SORT, as theirs is. It does NOT refuse repeated keys: see
    DEVIATION 777 below and `refuse_repeated_keys`, which carries the
    refusal at the one place the ambiguity bites."""
    var nnz = g.nnz()
    var order = List[Int]()
    for i in range(nnz):
        order.append(i)
    # Merge sort on a key that fits in one UInt64: (row << 32 | col), ties
    # by original index.
    var keys = List[UInt64]()
    for i in range(nnz):
        keys.append(
            (UInt64(Int(g.rows[i]) & 0xFFFFFFFF) << 32)
            | UInt64(Int(g.cols[i]) & 0xFFFFFFFF)
        )
    _sort_indices_by_key(order, keys)
    var rows = List[Int32]()
    var cols = List[Int32]()
    var vals = List[Float32]()
    for i in range(nnz):
        var src = order[i]
        rows.append(g.rows[src])
        cols.append(g.cols[src])
        vals.append(g.vals[src])
    return CooGraph(g.n, rows^, cols^, vals^)


def refuse_repeated_keys(g: CooGraph) raises:
    """DEVIATION 777: raise if a ROW-SORTED `g` holds two entries with the
    same `(row, col)`.

    Called on the Laplacian's INPUT (`compute_graph_laplacian` on the
    device, `host_laplacian` in the oracle), never inside `coo_sort`."""
    for i in range(1, g.nnz()):
        if g.rows[i] == g.rows[i - 1] and g.cols[i] == g.cols[i - 1]:
            raise Error(
                "connectivity_graph: repeated (row, col) pair ("
                + String(g.rows[i])
                + ", "
                + String(g.cols[i])
                + ") -- refused by name (DEVIATION 775)"
            )


def _sort_indices_by_key(mut order: List[Int], keys: List[UInt64]):
    """Bottom-up merge sort of `order` by `keys[order[i]]`, ties by the
    index itself (so the result is the unique total-order permutation)."""
    var n = len(order)
    if n < 2:
        return
    var tmp = List[Int]()
    for i in range(n):
        tmp.append(0)
    var width = 1
    while width < n:
        var lo = 0
        while lo < n:
            var mid = lo + width
            if mid > n:
                mid = n
            var hi = lo + 2 * width
            if hi > n:
                hi = n
            var i = lo
            var j = mid
            var k = lo
            while i < mid and j < hi:
                var a = order[i]
                var b = order[j]
                var take_left = keys[a] < keys[b] or (keys[a] == keys[b] and a < b)
                if take_left:
                    tmp[k] = a
                    i += 1
                else:
                    tmp[k] = b
                    j += 1
                k += 1
            while i < mid:
                tmp[k] = order[i]
                i += 1
                k += 1
            while j < hi:
                tmp[k] = order[j]
                j += 1
                k += 1
            lo = hi
        for i in range(n):
            order[i] = tmp[i]
        width *= 2


def coo_remove_scalar(g: CooGraph, scalar: Float32) -> CooGraph:
    """`filter.cuh:39-85`: keep every entry whose value `!= scalar`, in
    order. (`coo_remove_zeros` is this at `0.0`; `-0.0 != 0.0` is false, so
    a negative zero is removed too, exactly as theirs.)"""
    var rows = List[Int32]()
    var cols = List[Int32]()
    var vals = List[Float32]()
    for i in range(g.nnz()):
        if g.vals[i] != scalar:
            rows.append(g.rows[i])
            cols.append(g.cols[i])
            vals.append(g.vals[i])
    return CooGraph(g.n, rows^, cols^, vals^)


def sorted_coo_to_csr(g: CooGraph) -> List[Int32]:
    """`csr.cuh:78-90`, plus an `n + 1`-th entry holding `nnz`.

    THE TERMINATOR IS OURS. Theirs writes `m` row offsets and no
    terminator, and every reader compensates with `get_stop_idx`
    (`raft/sparse/detail/utils.h:97-105`), which returns `nnz` for the last
    row; `coo_symmetrize` allocates its `in_row_ind` at length `n`
    accordingly (`symmetrize.cuh:188`). Carrying the `n + 1` form here lets
    every per-row loop in this lane read `indptr[r + 1]` unconditionally,
    which is THE SAME NUMBERS by construction. (A previous version of this
    docstring credited the terminator to cuVS 25.08's `coo_to_csr_matrix`
    at `spectral_embedding.cu:134`; that function does not exist at 26.08,
    which builds the Laplacian straight from the COO.)"""
    var n = g.n
    var counts = List[Int32]()
    for _ in range(n):
        counts.append(Int32(0))
    for i in range(g.nnz()):
        counts[Int(g.rows[i])] += 1
    var indptr = List[Int32]()
    var acc = Int32(0)
    for r in range(n):
        indptr.append(acc)
        acc += counts[r]
    indptr.append(acc)
    return indptr^
