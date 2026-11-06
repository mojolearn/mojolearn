# SPDX-License-Identifier: Apache-2.0
# Native CSR/CSC index expansion for the spectral sparse reader
# (bindings/cfb_sparse_read.mojo): the (row, col) int32 triples a compressed
# matrix's `tocoo()` makes, in its entry order (duplicates and unsorted
# minor indices kept, as `_sparsetools.expandptr` keeps them), read straight
# from the `indptr`/`indices` buffers. A malformed pointer or an out-of-range
# minor index returns 1 and the caller takes the matrix's own `tocoo()`, so
# SciPy's own refusal is raised unchanged.
from std.memory import bitcast
from std.sys.info import num_physical_cores
from core.host_parallel import host_parallelize

comptime _U8P = MutPointer[UInt8, MutUntrackedOrigin]
comptime _I32P = MutPointer[Int32, MutUntrackedOrigin]
# Major rows per host task: the expansion is memory bound, so a task below a
# few thousand rows costs more to schedule than it saves (size reasoning, not
# a dataset dimension; any larger matrix fans out to every physical core).
comptime _MIN_MAJOR_PER_TASK = 4096


def _index_word(address: Int, stride: Int, width: Int, native: Bool, i: Int) -> Int:
    """Item `i` of a signed int32/int64 index buffer, either byte order."""
    var bytes = _U8P(unsafe_from_address=address + i * stride)
    var raw = UInt64(0)
    for k in range(width):
        raw |= UInt64(bytes[k if native else width - 1 - k]) << UInt64(k * 8)
    if width == 4:
        return Int(bitcast[DType.int32](UInt32(raw)))
    return Int(bitcast[DType.int64](raw))


def cfb_sparse_expand(ptr_addr: Int, ptr_stride: Int, ptr_width: Int, ptr_native: Bool,
                      idx_addr: Int, idx_stride: Int, idx_width: Int, idx_native: Bool,
                      major: Int, minor: Int, count: Int, major_out: Int, minor_out: Int) -> Int:
    """Fill `major_out[j]` with the major index (row of a CSR, column of a
    CSC) of entry `j` and `minor_out[j]` with its stored minor index, for the
    `count` entries. 0 when the pointer is well formed (starts at 0, never
    decreases, ends at `count`) and every minor index is in `[0, minor)`;
    1 otherwise (outputs then partly written, never past `count`)."""
    if major < 1 or minor < 1 or count < 1 or major > 2147483647 or minor > 2147483647:
        return 1
    if _index_word(ptr_addr, ptr_stride, ptr_width, ptr_native, 0) != 0:
        return 1
    if _index_word(ptr_addr, ptr_stride, ptr_width, ptr_native, major) != count:
        return 1
    var tasks = (major + _MIN_MAJOR_PER_TASK - 1) // _MIN_MAJOR_PER_TASK
    var cores = num_physical_cores()
    if tasks > cores:
        tasks = cores
    if tasks < 1:
        tasks = 1
    var chunk = (major + tasks - 1) // tasks
    var failed = List[Int](length=tasks, fill=0)
    var fp = failed.unsafe_ptr()
    var mo = _I32P(unsafe_from_address=major_out)
    var no = _I32P(unsafe_from_address=minor_out)

    def _rows_task(c: Int) {imm fp, imm mo, imm no, imm chunk, imm major, imm minor, imm count,
                            imm ptr_addr, imm ptr_stride, imm ptr_width, imm ptr_native,
                            imm idx_addr, imm idx_stride, imm idx_width, imm idx_native}:
        var lo = c * chunk
        var hi = lo + chunk
        if hi > major:
            hi = major
        if lo >= hi:
            return
        var start = _index_word(ptr_addr, ptr_stride, ptr_width, ptr_native, lo)
        if start < 0 or start > count:
            fp.unsafe_store(c, 1)
            return
        for r in range(lo, hi):
            var end = _index_word(ptr_addr, ptr_stride, ptr_width, ptr_native, r + 1)
            if end < start or end > count:
                fp.unsafe_store(c, 1)
                return
            for j in range(start, end):
                var m = _index_word(idx_addr, idx_stride, idx_width, idx_native, j)
                if m < 0 or m >= minor:
                    fp.unsafe_store(c, 1)
                    return
                mo.unsafe_store(j, Int32(r))
                no.unsafe_store(j, Int32(m))
            start = end

    if tasks == 1:
        _rows_task(0)
    else:
        host_parallelize(_rows_task, tasks)
    _ = len(failed)
    for c in range(tasks):
        if failed[c] != 0:
            return 1
    return 0
