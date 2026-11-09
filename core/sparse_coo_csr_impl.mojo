# SPDX-License-Identifier: Apache-2.0
# Native COO -> CSR for a caller's sparse input (lane P1, audit A #3): the
# (row, col, value) entries of an exact SciPy COO/CSC container (read from its
# stored buffers, bindings/cfb_sparse_read.mojo) grouped by row with a stable
# counting sort, split across the host's cores. It replaces the call to the
# caller's own `tocsr()` (SciPy's Python conversion) inside the native
# operation. Entries keep their stored order within a row and duplicates are
# kept (each is one edge); no value arithmetic runs.
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
