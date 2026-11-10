# SPDX-License-Identifier: Apache-2.0
# Native COO -> CSR for a caller's sparse input (lane P1, audit A #3): the
# (row, col, value) entries of an exact SciPy COO/CSC container (read from its
# stored buffers, bindings/cfb_sparse_read.mojo) grouped by row with a stable
# counting sort, split across the host's cores. It replaces the call to the
# caller's own `tocsr()` (SciPy's Python conversion) inside the native
# operation. `sparse_coo_to_csr` keeps the stored order within a row and keeps
# duplicates; `sparse_coo_to_csr_summed` (below) merges them as `tocsr()` does,
# and `sparse_dense_row_counts` / `sparse_dense_fill` take the nonzeros of a
# natively densified BSR / DIA / DOK / LIL container.
from std.sys.info import num_physical_cores
from core.host_parallel import host_parallelize

comptime _I32P = MutPointer[Int32, MutUntrackedOrigin]
comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
# Entries per host task: the sort is memory bound, so a task below a few
# thousand entries costs more to schedule than it saves (size reasoning, not
# a dataset dimension; a larger input fans out to every physical core).
comptime _MIN_ENTRIES_PER_TASK = 8192
# The per-task row histograms hold tasks * (rows + 1) int32 words; the task
# count is capped so they stay within this many bytes (memory reasoning: a
# very tall matrix uses fewer tasks rather than gigabytes of counters).
comptime _COUNTER_BYTES = 256 * 1024 * 1024


def sparse_coo_to_csr(n_rows: Int, n_cols: Int, nnz: Int, rows_addr: Int, cols_addr: Int, vals_addr: Int,
                      indptr_out: Int, indices_out: Int, vals_out: Int) -> Int:
    """Group the `nnz` int32 (row, col) entries at `rows_addr`/`cols_addr`
    (and their float32 values at `vals_addr`, skipped when 0) by row:
    `indptr_out` gets `n_rows + 1` int32 offsets, `indices_out` the columns
    and `vals_out` the values in row order, entries of one row in their
    input order. 0 on success; 1 when a row is outside `[0, n_rows)` or a
    column outside `[0, n_cols)` (outputs then partly written)."""
    if n_rows < 1 or n_cols < 1 or nnz < 0 or n_rows > 2147483647 or n_cols > 2147483647 or nnz > 2147483647:
        return 1
    var ip = _I32P(unsafe_from_address=indptr_out)
    if nnz == 0:
        for r in range(n_rows + 1):  # driver-loop: zero offsets of an empty matrix
            ip.unsafe_store(r, Int32(0))
        return 0
    var tasks = (nnz + _MIN_ENTRIES_PER_TASK - 1) // _MIN_ENTRIES_PER_TASK
    var cores = num_physical_cores()
    if tasks > cores:
        tasks = cores
    var cap = _COUNTER_BYTES // ((n_rows + 1) * 4)
    if tasks > cap:
        tasks = cap
    if tasks < 1:
        tasks = 1
    var chunk = (nnz + tasks - 1) // tasks
    # counts[t * n_rows + r]: entries of row r in task t's chunk, then the
    # output position of task t's next entry of row r.
    var counts = List[Int32](length=tasks * n_rows, fill=0)
    var cp = counts.unsafe_ptr()
    var failed = List[Int](length=tasks, fill=0)
    var fp = failed.unsafe_ptr()
    var rp = _I32P(unsafe_from_address=rows_addr)
    var colp = _I32P(unsafe_from_address=cols_addr)

    def _count_task(t: Int) {imm cp, imm fp, imm rp, imm colp, imm chunk, imm nnz, imm n_rows, imm n_cols}:
        var lo = t * chunk
        var hi = lo + chunk
        if hi > nnz:
            hi = nnz
        var base = t * n_rows
        for j in range(lo, hi):
            var r = Int(rp[j])
            var c = Int(colp[j])
            if r < 0 or r >= n_rows or c < 0 or c >= n_cols:
                fp.unsafe_store(t, 1)
                return
            cp.unsafe_store(base + r, cp[base + r] + 1)

    if tasks == 1:
        _count_task(0)
    else:
        host_parallelize(_count_task, tasks)
    for t in range(tasks):  # small-loop(tasks: at most one per core): the failure words of the join
        if failed[t] != 0:
            return 1
    # Row totals, row blocks across the cores.
    var row_tasks = (n_rows + _MIN_ENTRIES_PER_TASK - 1) // _MIN_ENTRIES_PER_TASK
    if row_tasks > cores:
        row_tasks = cores
    if row_tasks < 1:
        row_tasks = 1
    var row_chunk = (n_rows + row_tasks - 1) // row_tasks

    def _total_task(b: Int) {imm cp, imm ip, imm row_chunk, imm n_rows, imm tasks}:
        var lo = b * row_chunk
        var hi = lo + row_chunk
        if hi > n_rows:
            hi = n_rows
        for r in range(lo, hi):
            var total = Int32(0)
            for t in range(tasks):
                total += cp[t * n_rows + r]
            ip.unsafe_store(r + 1, total)

    if row_tasks == 1:
        _total_task(0)
    else:
        host_parallelize(_total_task, row_tasks)
    # The exclusive scan over rows: one pass of n_rows additions, a running
    # sum whose every step needs the previous one.
    ip.unsafe_store(0, Int32(0))
    for r in range(n_rows):  # driver-loop: prefix sum over rows (serial by nature: each offset needs the previous)
        ip.unsafe_store(r + 1, ip[r + 1] + ip[r])

    # Each task's starting position in each row, row blocks across the cores.
    def _offset_task(b: Int) {imm cp, imm ip, imm row_chunk, imm n_rows, imm tasks}:
        var lo = b * row_chunk
        var hi = lo + row_chunk
        if hi > n_rows:
            hi = n_rows
        for r in range(lo, hi):
            var at = ip[r]
            for t in range(tasks):
                var here = cp[t * n_rows + r]
                cp.unsafe_store(t * n_rows + r, at)
                at += here

    if row_tasks == 1:
        _offset_task(0)
    else:
        host_parallelize(_offset_task, row_tasks)
    var ix = _I32P(unsafe_from_address=indices_out)
    var vi = _F32P(unsafe_from_address=vals_addr)
    var vo = _F32P(unsafe_from_address=vals_out)
    var with_values = vals_addr != 0 and vals_out != 0

    def _scatter_task(t: Int) {imm cp, imm rp, imm colp, imm ix, imm vi, imm vo, imm with_values,
                               imm chunk, imm nnz, imm n_rows}:
        var lo = t * chunk
        var hi = lo + chunk
        if hi > nnz:
            hi = nnz
        var base = t * n_rows
        for j in range(lo, hi):
            var r = Int(rp[j])
            var at = Int(cp[base + r])
            cp.unsafe_store(base + r, Int32(at + 1))
            ix.unsafe_store(at, colp[j])
            if with_values:
                vo.unsafe_store(at, vi[j])

    if tasks == 1:
        _scatter_task(0)
    else:
        host_parallelize(_scatter_task, tasks)
    _ = len(counts)
    _ = len(failed)
    return 0


# ------------------------------------------------- duplicates summed, dense
# Integration lane i2 (2026-11-09), correction to P1_NEIGHBORS_GRAPH_TOCSR:
# SciPy's `coo.tocsr()` SUMS duplicate (row, col) entries; the grouping above
# keeps them, so a graph built from a COO adjacency with repeated edges gave
# wrong PageRank weights and degrees. `sparse_coo_to_csr_summed` groups the
# entries by column, then by row (two stable counting sorts, so entries of one
# row come out in ascending column and repeated (row, col) entries in their
# stored order), then merges each run of one column into one entry whose value
# is the sum of the run IN STORED ORDER, left to right: one fixed order on
# every host, so the same bits wherever it runs.


def _expand_major(n_major: Int, nnz: Int, ptr: _I32P, out_major: _I32P):
    """`out_major[k] = c` for every k in `[ptr[c], ptr[c + 1])`, column blocks
    across the cores."""
    var split = _row_blocks(n_major, n_major + nnz)
    var blocks = split[0]
    var chunk = split[1]

    def _task(b: Int) {imm ptr, imm out_major, imm chunk, imm n_major}:
        var lo = b * chunk
        var hi = lo + chunk
        if hi > n_major:
            hi = n_major
        for c in range(lo, hi):
            for k in range(Int(ptr[c]), Int(ptr[c + 1])):
                out_major.unsafe_store(k, Int32(c))

    if blocks == 1:
        _task(0)
    else:
        host_parallelize(_task, blocks)


def _row_blocks(n_rows: Int, work: Int) -> Tuple[Int, Int]:
    """(blocks, rows per block) for a pass over rows that touches `work`
    words in all, across the cores (one block per `_MIN_ENTRIES_PER_TASK`
    words, at most one per core, never more blocks than rows)."""
    var cores = num_physical_cores()
    var blocks = (work + _MIN_ENTRIES_PER_TASK - 1) // _MIN_ENTRIES_PER_TASK
    if blocks > n_rows:
        blocks = n_rows
    if blocks > cores:
        blocks = cores
    if blocks < 1:
        blocks = 1
    return (blocks, (n_rows + blocks - 1) // blocks)


def sparse_coo_to_csr_summed(n_rows: Int, n_cols: Int, nnz: Int, rows_addr: Int, cols_addr: Int, vals_addr: Int,
                             indptr_out: Int, indices_out: Int, vals_out: Int) -> Int:
    """`sparse_coo_to_csr` with duplicate (row, col) entries merged, as
    SciPy's `coo.tocsr()` does: `indptr_out` gets `n_rows + 1` offsets, the
    first m words of `indices_out` (and of `vals_out`, when both value
    addresses are nonzero) the ascending columns of each row and the summed
    values. `indices_out` / `vals_out` must hold `nnz` words. Returns m
    (the entry count after merging), or -1 when an entry lies outside the
    shape."""
    if n_rows < 1 or n_cols < 1 or nnz < 0 or nnz > 2147483647:
        return -1
    var with_values = vals_addr != 0 and vals_out != 0
    var nv = nnz if with_values else 0
    # Pass 1: grouped by column (rows as the minor index), stored order kept.
    var tptr = List[Int32](length=n_cols + 1, fill=0)
    var trow = List[Int32](length=nnz, fill=0)
    var tval = List[Float32](length=nv, fill=0)
    if sparse_coo_to_csr(n_cols, n_rows, nnz, cols_addr, rows_addr, vals_addr if with_values else 0,
                         Int(tptr.unsafe_ptr()), Int(trow.unsafe_ptr()),
                         Int(tval.unsafe_ptr()) if with_values else 0) != 0:
        return -1
    var tcol = List[Int32](length=nnz, fill=0)
    _expand_major(n_cols, nnz, _I32P(unsafe_from_address=Int(tptr.unsafe_ptr())),
                  _I32P(unsafe_from_address=Int(tcol.unsafe_ptr())))
    # Pass 2: grouped by row; within a row ascending column, ties in stored order.
    var gptr = List[Int32](length=n_rows + 1, fill=0)
    var gcol = List[Int32](length=nnz, fill=0)
    var gval = List[Float32](length=nv, fill=0)
    if sparse_coo_to_csr(n_rows, n_cols, nnz, Int(trow.unsafe_ptr()), Int(tcol.unsafe_ptr()),
                         Int(tval.unsafe_ptr()) if with_values else 0,
                         Int(gptr.unsafe_ptr()), Int(gcol.unsafe_ptr()),
                         Int(gval.unsafe_ptr()) if with_values else 0) != 0:
        return -1
    var gp = _I32P(unsafe_from_address=Int(gptr.unsafe_ptr()))
    var gc = _I32P(unsafe_from_address=Int(gcol.unsafe_ptr()))
    var gv = _F32P(unsafe_from_address=Int(gval.unsafe_ptr()))
    var ip = _I32P(unsafe_from_address=indptr_out)
    var ix = _I32P(unsafe_from_address=indices_out)
    var vo = _F32P(unsafe_from_address=vals_out)
    var split = _row_blocks(n_rows, n_rows + nnz)
    var blocks = split[0]
    var row_chunk = split[1]

    # Distinct columns per row.
    def _count_task(b: Int) {imm gp, imm gc, imm ip, imm row_chunk, imm n_rows}:
        var lo = b * row_chunk
        var hi = lo + row_chunk
        if hi > n_rows:
            hi = n_rows
        for r in range(lo, hi):
            var start = Int(gp[r])
            var stop = Int(gp[r + 1])
            var distinct = 0
            for k in range(start, stop):
                if k == start or gc[k] != gc[k - 1]:
                    distinct += 1
            ip.unsafe_store(r + 1, Int32(distinct))

    if blocks == 1:
        _count_task(0)
    else:
        host_parallelize(_count_task, blocks)
    ip.unsafe_store(0, Int32(0))
    for r in range(n_rows):  # driver-loop: prefix sum over rows (serial by nature: each offset needs the previous)
        ip.unsafe_store(r + 1, ip[r + 1] + ip[r])

    # One entry per run; the run's values summed in stored order.
    def _fill_task(b: Int) {imm gp, imm gc, imm gv, imm ip, imm ix, imm vo, imm with_values, imm row_chunk, imm n_rows}:
        var lo = b * row_chunk
        var hi = lo + row_chunk
        if hi > n_rows:
            hi = n_rows
        for r in range(lo, hi):
            var start = Int(gp[r])
            var stop = Int(gp[r + 1])
            var at = Int(ip[r]) - 1
            for k in range(start, stop):
                if k == start or gc[k] != gc[k - 1]:
                    at += 1
                    ix.unsafe_store(at, gc[k])
                    if with_values:
                        vo.unsafe_store(at, gv[k])
                elif with_values:
                    vo.unsafe_store(at, vo[at] + gv[k])

    if blocks == 1:
        _fill_task(0)
    else:
        host_parallelize(_fill_task, blocks)
    _ = len(tptr)
    _ = len(trow)
    _ = len(tval)
    _ = len(tcol)
    _ = len(gptr)
    _ = len(gcol)
    _ = len(gval)
    return Int(ip[n_rows])


def sparse_dense_row_counts(n_rows: Int, n_cols: Int, dense_addr: Int, indptr_out: Int) -> Int:
    """`indptr_out` (`n_rows + 1` int32 offsets) of the nonzero entries of the
    C-order float32 `n_rows` x `n_cols` matrix at `dense_addr` (a value is
    stored when it is not 0; NaN is stored); returns their count. Row blocks
    across the cores, then one serial prefix."""
    var dp = _F32P(unsafe_from_address=dense_addr)
    var ip = _I32P(unsafe_from_address=indptr_out)
    var split = _row_blocks(n_rows, n_rows * n_cols)
    var row_chunk = split[1]

    def _task(b: Int) {imm dp, imm ip, imm row_chunk, imm n_rows, imm n_cols}:
        var lo = b * row_chunk
        var hi = lo + row_chunk
        if hi > n_rows:
            hi = n_rows
        for r in range(lo, hi):
            var count = 0
            for c in range(n_cols):
                if dp[r * n_cols + c] != 0:
                    count += 1
            ip.unsafe_store(r + 1, Int32(count))

    if split[0] == 1:
        _task(0)
    else:
        host_parallelize(_task, split[0])
    ip.unsafe_store(0, Int32(0))
    for r in range(n_rows):  # driver-loop: prefix sum over rows (serial by nature: each offset needs the previous)
        ip.unsafe_store(r + 1, ip[r + 1] + ip[r])
    return Int(ip[n_rows])


def sparse_dense_fill(n_rows: Int, n_cols: Int, dense_addr: Int, indptr_addr: Int, indices_out: Int, vals_out: Int):
    """The nonzero entries counted by `sparse_dense_row_counts`, row by row in
    ascending column: columns into `indices_out`, values into `vals_out`
    (skipped when 0)."""
    var dp = _F32P(unsafe_from_address=dense_addr)
    var ip = _I32P(unsafe_from_address=indptr_addr)
    var ix = _I32P(unsafe_from_address=indices_out)
    var vo = _F32P(unsafe_from_address=vals_out)
    var with_values = vals_out != 0
    var split = _row_blocks(n_rows, n_rows * n_cols)
    var row_chunk = split[1]

    def _task(b: Int) {imm dp, imm ip, imm ix, imm vo, imm with_values, imm row_chunk, imm n_rows, imm n_cols}:
        var lo = b * row_chunk
        var hi = lo + row_chunk
        if hi > n_rows:
            hi = n_rows
        for r in range(lo, hi):
            var at = Int(ip[r])
            for c in range(n_cols):
                var v = dp[r * n_cols + c]
                if v != 0:
                    ix.unsafe_store(at, Int32(c))
                    if with_values:
                        vo.unsafe_store(at, v)
                    at += 1

    if split[0] == 1:
        _task(0)
    else:
        host_parallelize(_task, split[0])
