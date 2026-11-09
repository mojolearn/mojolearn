# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The one host finiteness scan of admitted float32 words (lane
fix-c1-c2-parallel-finite, 2026-11-08; docs/plans/REGRESSIONS_0866_VS_0855.md
causes C1 and C2).

Every public door that refuses NaN/infinity in float32 input reads one of
these. The test is the exponent-bits test the serial scans made
(`bits & 0x7F800000 == 0x7F800000`), on integer words, so no float operation
runs and the FTZ/DAZ state of a pool worker cannot change a verdict. The
verdict is "every word finite", the AND of the tasks' verdicts, so it does
not depend on the split or on the order the words are read in: no result and
no refusal changes, and no output bit is written.

The words are split into contiguous tasks over the host's cores
(`core/host_predict_threads.mojo`, with its grain floor: a small input runs
on the calling thread without touching the pool), each task a SIMD
any-non-finite reduction with a scalar tail. Loads are unaligned (alignment
1): a borrowed NumPy view may sit at any byte offset.

    host_all_finite_f32(p, n)                       n contiguous words
    host_all_finite_f64(p, n)                       n contiguous float64 words
    host_all_finite_f32_strided(addr, s0, s1, rows, cols)
                                                    a rows x cols view with
                                                    byte strides s0, s1
"""
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk

comptime _HFS_F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _HFS_F64P = MutPointer[Float64, MutUntrackedOrigin]
comptime _HFS_U32P = MutPointer[UInt32, MutUntrackedOrigin]
comptime _HFS_EXP = UInt32(0x7F800000)
#: SIMD words per load: 16 x 32 bits is one 512-bit vector, four 128-bit
#: NEON vectors or two 256-bit AVX2 vectors; LLVM splits it to the target.
comptime _HFS_W = 16


@always_inline
def _hfs_exp_mask[dt: DType]() -> Scalar[dt]:
    """The exponent field of the float whose bits are `dt` words."""
    comptime if dt == DType.uint64:
        return Scalar[dt](0x7FF0000000000000)
    else:
        return Scalar[dt](0x7F800000)


@always_inline
def _hfs_span_finite[dt: DType](addr: Int, n: Int) -> Bool:
    """True when none of the `n` contiguous `dt` words at byte address
    `addr` has all exponent bits set. Serial within the span; the callers
    split spans across cores."""
    comptime W = _HFS_W if dt == DType.uint32 else _HFS_W // 2
    var words = MutPointer[Scalar[dt], MutUntrackedOrigin](unsafe_from_address=addr)
    var exp1 = _hfs_exp_mask[dt]()
    var exp = SIMD[dt, W](exp1)
    var i = 0
    var body = n - n % W
    while i < body:
        var bits = words.unsafe_load[width=W, alignment=1](i) & exp
        if bits.eq(exp).reduce_or():
            return False
        i += W
    while i < n:
        if (words.unsafe_load[alignment=1](i) & exp1) == exp1:
            return False
        i += 1
    return True


def _hfs_all_finite[dt: DType](addr: Int, n: Int) -> Bool:
    """`_hfs_span_finite` of `n` contiguous words, split into contiguous
    tasks over the host's cores (grain floor applied)."""
    if n <= 0:
        return True
    var tasks = host_predict_task_count_work(n, n)
    if tasks <= 1:
        return _hfs_span_finite[dt](addr, n)
    var chunk = host_predict_chunk(n, tasks)
    # Round the chunk up to whole SIMD loads so only the last task has a tail.
    chunk = ((chunk + _HFS_W - 1) // _HFS_W) * _HFS_W
    var word_bytes = 8 if dt == DType.uint64 else 4
    var flags = List[UInt8](length=tasks, fill=UInt8(1))
    var flag_ptr = rebind[MutPointer[UInt8, MutUntrackedOrigin]](flags.unsafe_ptr())

    def _span(t: Int) {imm addr, imm n, imm chunk, imm word_bytes, imm flag_ptr}:
        var lo = t * chunk
        if lo >= n:
            return
        var hi = min(lo + chunk, n)
        if not _hfs_span_finite[dt](addr + lo * word_bytes, hi - lo):
            flag_ptr.unsafe_store(t, UInt8(0))

    host_parallelize(_span, tasks)
    var finite = True
    for t in range(tasks):  # driver-loop: over the scan tasks' flags (at most one per core)
        if flags[t] == UInt8(0):
            finite = False
    _ = flags^
    return finite


def host_all_finite_f32(p: _HFS_F32P, n: Int) -> Bool:
    """True when none of the `n` float32 words at `p` is a NaN or an
    infinity. `n <= 0` is True."""
    # parallel: finiteness scan across host cores (0.8.66 regression C2, serial scan)
    return _hfs_all_finite[DType.uint32](Int(p), n)


def host_all_finite_f64(p: _HFS_F64P, n: Int) -> Bool:
    """True when none of the `n` float64 words at `p` is a NaN or an
    infinity. `n <= 0` is True."""
    # parallel: finiteness scan across host cores (0.8.66 regression C1 class, serial scan)
    return _hfs_all_finite[DType.uint64](Int(p), n)


def host_all_finite_f32_strided(addr: Int, s0: Int, s1: Int, rows: Int, cols: Int) -> Bool:
    """True when none of the float32 words of the rows x cols view at byte
    address `addr` with byte strides `s0` (rows) and `s1` (columns) is a NaN
    or an infinity. A view whose words are contiguous (C order with no row
    padding, a single row, or a single column of unit stride) is scanned as
    one flat span; otherwise rows are split across the host's cores and a
    row with unit column stride is one SIMD span."""
    if rows <= 0 or cols <= 0:
        return True
    var row_unit = s1 == 4 or cols == 1
    if row_unit and (s0 == cols * 4 or rows == 1):
        return host_all_finite_f32(_HFS_F32P(unsafe_from_address=addr), rows * cols)
    # parallel: finiteness scan over row tasks across host cores (0.8.66 regression C1, 13 ns/element serial)
    var tasks = host_predict_task_count_work(rows, rows * cols)
    var chunk = host_predict_chunk(rows, tasks)
    var flags = List[UInt8](length=tasks, fill=UInt8(1))
    var flag_ptr = rebind[MutPointer[UInt8, MutUntrackedOrigin]](flags.unsafe_ptr())

    def _rows(t: Int) {imm addr, imm s0, imm s1, imm rows, imm cols, imm row_unit, imm chunk, imm flag_ptr}:
        var lo = t * chunk
        var hi = min(lo + chunk, rows)
        for r in range(lo, hi):  # native input marshaling: this task's input rows
            if row_unit:
                if not _hfs_span_finite[DType.uint32](addr + r * s0, cols):
                    flag_ptr.unsafe_store(t, UInt8(0))
                    return
            else:
                for c in range(cols):  # native input marshaling: strided words of one row
                    var w = _HFS_U32P(unsafe_from_address=addr + r * s0 + c * s1).unsafe_load[alignment=1](0)
                    if (w & _HFS_EXP) == _HFS_EXP:
                        flag_ptr.unsafe_store(t, UInt8(0))
                        return

    if tasks <= 1:
        _rows(0)
    else:
        host_parallelize(_rows, tasks)
    var finite = True
    for t in range(tasks):  # driver-loop: over the row tasks' flags (at most one per core)
        if flags[t] == UInt8(0):
            finite = False
    _ = flags^
    return finite
