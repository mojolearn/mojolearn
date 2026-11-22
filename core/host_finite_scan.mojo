# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The one host finiteness scan of admitted float32 words (lane
fix-c1-c2-parallel-finite, 2026-11-08; plan REGRESSIONS_0866_VS_0855 (git history)
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
    host_all_finite_view(addr, s0, s1, rows, cols, width, native)
                                                    the same for float16,
                                                    float32 or float64 words
                                                    in either byte order
    host_all_finite_nd(addr, shape, strides, width, native)
                                                    a view of any rank

THE THREE PREDICATES (lane fx-host-scans, 2026-11-18). The private host
scans of the families asked three different questions; each has one entry
here, all on bits, all one parallel SIMD pass with the same grain floor:

    all finite    no NaN and no infinity (exponent field all ones refuses):
                  `host_all_finite_*` above, `host_all_finite_list`,
                  `host_copy_f32_finite` (the test folded into a copy)
    any NaN       NaN refused or reported, infinity is an ordinary value
                  (exponent all ones and a nonzero fraction; `v != v`):
                  `host_any_nan_f32`, `host_any_nan_list`,
                  `host_any_nan_unit` (one unit of a caller's own split)
                  `host_copy_f32_any_nan` (an owned copy and its any-NaN verdict, one pass)
    any infinity  +inf or -inf refused, NaN is a missing value and passes
                  (magnitude bits exactly 0x7F800000):
                  `host_any_inf_f32`

A caller whose message names the first offending index asks the parallel
verdict first and, only on a refusal, `host_first_nonfinite_f32` (serial:
the error path).
"""
from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.host_predict_threads import HOST_MIN_TASK_WORK, host_predict_task_count_work, host_predict_chunk

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
def _hfs_span_finite_mask[dt: DType](addr: Int, n: Int, exp1: Scalar[dt]) -> Bool:
    """True when none of the `n` contiguous `dt` words at byte address
    `addr` has every bit of `exp1` set. `exp1` is the exponent field of the
    float as the word is loaded (byte-swapped for a non-native view).
    Serial within the span; the callers split spans across cores."""
    comptime W = _HFS_W * 2 if dt == DType.uint16 else (_HFS_W if dt == DType.uint32 else _HFS_W // 2)
    var words = MutPointer[Scalar[dt], MutUntrackedOrigin](unsafe_from_address=addr)
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


@always_inline
def _hfs_span_finite[dt: DType](addr: Int, n: Int) -> Bool:
    """`_hfs_span_finite_mask` with the native exponent field of `dt`."""
    return _hfs_span_finite_mask[dt](addr, n, _hfs_exp_mask[dt]())


def _hfs_all_finite[dt: DType](addr: Int, n: Int) -> Bool:
    """`_hfs_span_finite` of `n` contiguous words, split into contiguous
    tasks over the host's cores (grain floor applied)."""
    if n <= 0:
        return True
    if n < 2 * HOST_MIN_TASK_WORK:
        # The grain floor of host_predict_task_count_work, decided here from
        # `n` alone: a small scan (a parameter vector, a fitted statistic,
        # one step's state) reads no worker count and touches no pool.
        return _hfs_span_finite[dt](addr, n)
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


def host_all_finite_list(values: List[Float32]) -> Bool:
    """`host_all_finite_f32` over a List's storage (the List is the
    caller's borrow, alive through the call)."""
    return host_all_finite_f32(rebind[_HFS_F32P](values.unsafe_ptr()), len(values))


def host_first_nonfinite_f32(p: _HFS_F32P, n: Int) -> Int:
    """The first index in `[0, n)` whose float32 word is a NaN or an
    infinity (the same bits test), or -1. Serial by nature: it runs only
    after a parallel verdict refused, to name the index in the caller's
    message, and stops at the first hit."""
    var words = _HFS_U32P(unsafe_from_address=Int(p))
    for i in range(n):  # native input marshaling: the refusal's index, error path only
        if (words.unsafe_load[alignment=1](i) & _HFS_EXP) == _HFS_EXP:
            return i
    return -1


@always_inline
def _hfs_span_any_class(addr: Int, n: Int, nan: Bool) -> Bool:
    """True when any of the `n` contiguous float32 words at byte address
    `addr` is a NaN (`nan`) or an infinity (not `nan`), on bits: a NaN has
    the exponent field all ones and a nonzero fraction, an infinity has
    magnitude bits exactly 0x7F800000. Serial within the span; the caller
    splits spans across cores."""
    comptime W = _HFS_W
    var words = _HFS_U32P(unsafe_from_address=addr)
    var exp = SIMD[DType.uint32, W](0x7F800000)
    var i = 0
    var body = n - n % W
    if nan:
        var fraction = SIMD[DType.uint32, W](0x007FFFFF)
        var zero = SIMD[DType.uint32, W](0)
        while i < body:
            var w = words.unsafe_load[width=W, alignment=1](i)
            if (w & exp).eq(exp).select(w & fraction, zero).reduce_or() != UInt32(0):
                return True
            i += W
        while i < n:
            var w1 = words.unsafe_load[alignment=1](i)
            if (w1 & _HFS_EXP) == _HFS_EXP and (w1 & UInt32(0x007FFFFF)) != UInt32(0):
                return True
            i += 1
        return False
    var magnitude = SIMD[DType.uint32, W](0x7FFFFFFF)
    while i < body:
        if (words.unsafe_load[width=W, alignment=1](i) & magnitude).eq(exp).reduce_or():
            return True
        i += W
    while i < n:
        if (words.unsafe_load[alignment=1](i) & UInt32(0x7FFFFFFF)) == _HFS_EXP:
            return True
        i += 1
    return False


def _hfs_any_class(addr: Int, n: Int, nan: Bool) -> Bool:
    """`_hfs_span_any_class` of `n` contiguous words, split into contiguous
    tasks over the host's cores (grain floor applied). The verdict is the
    OR of the tasks' verdicts, so it does not depend on the split."""
    if n <= 0:
        return False
    if n < 2 * HOST_MIN_TASK_WORK:
        return _hfs_span_any_class(addr, n, nan)
    var tasks = host_predict_task_count_work(n, n)
    if tasks <= 1:
        return _hfs_span_any_class(addr, n, nan)
    var chunk = host_predict_chunk(n, tasks)
    chunk = ((chunk + _HFS_W - 1) // _HFS_W) * _HFS_W
    var flags = List[UInt8](length=tasks, fill=UInt8(0))
    var flag_ptr = rebind[MutPointer[UInt8, MutUntrackedOrigin]](flags.unsafe_ptr())

    def _span(t: Int) {imm addr, imm n, imm chunk, imm nan, imm flag_ptr}:
        var lo = t * chunk
        if lo >= n:
            return
        var hi = min(lo + chunk, n)
        if _hfs_span_any_class(addr + lo * 4, hi - lo, nan):
            flag_ptr.unsafe_store(t, UInt8(1))

    host_parallelize(_span, tasks)
    var found = False
    for t in range(tasks):  # driver-loop: over the scan tasks' flags (at most one per core)
        if flags[t] != UInt8(0):
            found = True
    _ = flags^
    return found


def host_any_nan_f32(p: _HFS_F32P, n: Int) -> Bool:
    """True when any of the `n` float32 words at `p` is a NaN. An infinity
    is an ordinary value here. `n <= 0` is False."""
    # parallel: NaN scan across host cores (was a private serial `v != v` loop per family)
    return _hfs_any_class(Int(p), n, True)


def host_any_nan_list(values: List[Float32]) -> Bool:
    """`host_any_nan_f32` over a List's storage."""
    return _hfs_any_class(Int(rebind[_HFS_F32P](values.unsafe_ptr())), len(values), True)


def host_any_nan_unit(values: List[Float32]) -> Bool:
    """The any-NaN test of ONE unit of a caller's own split (one feature
    column of a per-column quantization): the SIMD pass on the calling
    thread. Not split further because the caller already runs one such
    unit per core; a pool call from inside a pool task is not made."""
    return _hfs_span_any_class(Int(rebind[_HFS_F32P](values.unsafe_ptr())), len(values), True)


def host_any_inf_f32(p: _HFS_F32P, n: Int) -> Bool:
    """True when any of the `n` float32 words at `p` is +inf or -inf. A NaN
    (a missing value to the callers) passes. `n <= 0` is False."""
    # parallel: infinity scan across host cores (was a private serial loop)
    return _hfs_any_class(Int(p), n, False)


@always_inline
def _hfs_span_copy_finite(src: Int, dst: Int, n: Int) -> Bool:
    """Copy `n` float32 words from byte address `src` to `dst` and return
    True when every copied word is finite. The test is on bits, folded
    branch-free into the copy's own SIMD loads (lane fb12-kde-finite,
    2026-11-17, moved here from core/hostptr.mojo `copy_f32_finite`), so
    admission of a host input that is staged anyway costs no second pass.
    Every word is copied even when one is non-finite."""
    var src_ptr = _HFS_F32P(unsafe_from_address=src)
    var dst_ptr = _HFS_F32P(unsafe_from_address=dst)
    var hit = SIMD[DType.uint32, 8](0)
    var i = 0
    var body = n - n % 8
    while i < body:
        var v = src_ptr.unsafe_load[width=8](i)
        dst_ptr.unsafe_store[width=8](i, v)
        var e = bitcast[DType.uint32, 8](v) & SIMD[DType.uint32, 8](_HFS_EXP)
        hit |= e.eq(SIMD[DType.uint32, 8](_HFS_EXP)).cast[DType.uint32]()
        i += 8
    var tail = UInt32(0)
    while i < n:
        var w = src_ptr.unsafe_load(i)
        dst_ptr.unsafe_store(i, w)
        if (bitcast[DType.uint32](w) & _HFS_EXP) == _HFS_EXP:
            tail = 1
        i += 1
    return hit.reduce_or() == 0 and tail == 0


def host_copy_f32_finite(src: Int, dst: Int, n: Int) -> Bool:
    """Copy `n` float32 words from host address `src` to `dst` (the ranges
    must not overlap) and return True when every word is finite: the copy
    and the test are one pass, split into contiguous spans across the
    host's cores above the grain floor. Every word is copied whatever the
    verdict. A small copy runs on the calling thread and reads no worker
    count."""
    if n <= 0:
        return True
    if n < 2 * HOST_MIN_TASK_WORK:
        return _hfs_span_copy_finite(src, dst, n)
    # parallel: the staging copy and its finiteness verdict over spans across host cores
    var tasks = host_predict_task_count_work(n, n)
    if tasks <= 1:
        return _hfs_span_copy_finite(src, dst, n)
    var chunk = host_predict_chunk(n, tasks)
    chunk = ((chunk + _HFS_W - 1) // _HFS_W) * _HFS_W
    var flags = List[UInt8](length=tasks, fill=UInt8(1))
    var flag_ptr = rebind[MutPointer[UInt8, MutUntrackedOrigin]](flags.unsafe_ptr())

    def _span(t: Int) {imm src, imm dst, imm n, imm chunk, imm flag_ptr}:
        var lo = t * chunk
        if lo >= n:
            return
        var hi = min(lo + chunk, n)
        if not _hfs_span_copy_finite(src + lo * 4, dst + lo * 4, hi - lo):
            flag_ptr.unsafe_store(t, UInt8(0))

    host_parallelize(_span, tasks)
    var finite = True
    for t in range(tasks):  # driver-loop: over the copy tasks' flags (at most one per core)
        if flags[t] == UInt8(0):
            finite = False
    _ = flags^
    return finite


@always_inline
def _hfs_span_copy_any_nan(src: Int, dst: Int, n: Int) -> Bool:
    """Copy `n` float32 words from byte address `src` to `dst` and return
    True when any copied word is a NaN (exponent field all ones and a
    nonzero fraction, on bits; an infinity is an ordinary value). The test
    is folded branch-free into the copy's own SIMD loads, the form of
    `_hfs_span_copy_finite` with the NaN test of `_hfs_span_any_class`.
    Every word is copied whatever the verdict."""
    var src_ptr = _HFS_F32P(unsafe_from_address=src)
    var dst_ptr = _HFS_F32P(unsafe_from_address=dst)
    var exp = SIMD[DType.uint32, 8](_HFS_EXP)
    var fraction = SIMD[DType.uint32, 8](0x007FFFFF)
    var zero = SIMD[DType.uint32, 8](0)
    var hit = SIMD[DType.uint32, 8](0)
    var i = 0
    var body = n - n % 8
    while i < body:
        var v = src_ptr.unsafe_load[width=8](i)
        dst_ptr.unsafe_store[width=8](i, v)
        var w = bitcast[DType.uint32, 8](v)
        hit |= (w & exp).eq(exp).select(w & fraction, zero)
        i += 8
    var tail = UInt32(0)
    while i < n:
        var v1 = src_ptr.unsafe_load(i)
        dst_ptr.unsafe_store(i, v1)
        var w1 = bitcast[DType.uint32](v1)
        if (w1 & _HFS_EXP) == _HFS_EXP and (w1 & UInt32(0x007FFFFF)) != UInt32(0):
            tail = 1
        i += 1
    return hit.reduce_or() != UInt32(0) or tail != 0


def host_copy_f32_any_nan(src: Int, dst: Int, n: Int) -> Bool:
    """Copy `n` float32 words from host address `src` to `dst` (the ranges
    must not overlap) and return True when any word is a NaN: the copy and
    the test are one pass, split into contiguous spans across the host's
    cores above the grain floor (`host_copy_f32_finite`'s split). Every word
    is copied whatever the verdict; the verdict is the OR of the spans'
    verdicts, so it does not depend on the split."""
    if n <= 0:
        return False
    if n < 2 * HOST_MIN_TASK_WORK:
        return _hfs_span_copy_any_nan(src, dst, n)
    # parallel: the owned copy and its any-NaN verdict over spans across host cores
    var tasks = host_predict_task_count_work(n, n)
    if tasks <= 1:
        return _hfs_span_copy_any_nan(src, dst, n)
    var chunk = host_predict_chunk(n, tasks)
    chunk = ((chunk + _HFS_W - 1) // _HFS_W) * _HFS_W
    var flags = List[UInt8](length=tasks, fill=UInt8(0))
    var flag_ptr = rebind[MutPointer[UInt8, MutUntrackedOrigin]](flags.unsafe_ptr())

    def _span(t: Int) {imm src, imm dst, imm n, imm chunk, imm flag_ptr}:
        var lo = t * chunk
        if lo >= n:
            return
        var hi = min(lo + chunk, n)
        if _hfs_span_copy_any_nan(src + lo * 4, dst + lo * 4, hi - lo):
            flag_ptr.unsafe_store(t, UInt8(1))

    host_parallelize(_span, tasks)
    var found = False
    for t in range(tasks):  # driver-loop: over the copy tasks' flags (at most one per core)
        if flags[t] != UInt8(0):
            found = True
    _ = flags^
    return found


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


def _hfs_view_finite[dt: DType](addr: Int, outer: Int, inner: Int, s0: Int, s1: Int, exp1: Scalar[dt],
                                outer_rank: Int, dims_addr: Int, steps_addr: Int) -> Bool:
    """True when no word of a strided view has every bit of `exp1` set. The
    view is `outer` units of `inner` words `s1` bytes apart. Unit `u` starts
    at `addr + u * s0` when `outer_rank <= 1`; otherwise `u` is the C-order
    index over `outer_rank` axes whose sizes and byte strides are the Int
    words at `dims_addr` and `steps_addr`. The `outer * inner` words are
    split into equal contiguous ranges across the host's cores (grain floor
    applied), so a few long units and many short ones split alike; a range
    with `s1` equal to the word size is scanned as SIMD spans."""
    var word_bytes = 8 if dt == DType.uint64 else (2 if dt == DType.uint16 else 4)
    var total = outer * inner
    if outer <= 0 or inner <= 0:
        return True
    # parallel: finiteness scan over word ranges across host cores (was a serial byte-assembling scan)
    var tasks = max(host_predict_task_count_work(total, total), 1)
    var chunk = host_predict_chunk(total, tasks)
    var flags = List[UInt8](length=tasks, fill=UInt8(1))
    var flag_ptr = rebind[MutPointer[UInt8, MutUntrackedOrigin]](flags.unsafe_ptr())

    def _range(t: Int) {imm addr, imm inner, imm s0, imm s1, imm exp1, imm outer_rank, imm dims_addr,
                        imm steps_addr, imm word_bytes, imm total, imm chunk, imm flag_ptr}:
        var pos = t * chunk
        var hi = min(pos + chunk, total)
        if pos >= hi:
            return
        var u = pos // inner
        var c = pos - u * inner
        while pos < hi:  # native input marshaling: this task's units of the view
            var seg = min(inner - c, hi - pos)
            var base = addr + u * s0
            if outer_rank > 1:
                var dims = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=dims_addr)
                var steps = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=steps_addr)
                base = addr
                var remaining = u
                var axis = outer_rank - 1
                while axis >= 0:  # small-loop(admitted rank metadata)
                    var size = dims[axis]
                    base += (remaining % size) * steps[axis]
                    remaining //= size
                    axis -= 1
            base += c * s1
            if s1 == word_bytes:
                if not _hfs_span_finite_mask[dt](base, seg, exp1):
                    flag_ptr.unsafe_store(t, UInt8(0))
                    return
            else:
                for k in range(seg):  # native input marshaling: strided words of one unit
                    var w = MutPointer[Scalar[dt], MutUntrackedOrigin](unsafe_from_address=base + k * s1).unsafe_load[alignment=1](0)
                    if (w & exp1) == exp1:
                        flag_ptr.unsafe_store(t, UInt8(0))
                        return
            pos += seg
            u += 1
            c = 0

    if tasks <= 1:
        _range(0)
    else:
        host_parallelize(_range, tasks)
    var finite = True
    for t in range(tasks):  # driver-loop: over the scan tasks' flags (at most one per core)
        if flags[t] == UInt8(0):
            finite = False
    _ = flags^
    return finite


def _hfs_view_dispatch(addr: Int, outer: Int, inner: Int, s0: Int, s1: Int, width: Int, native: Bool,
                       outer_rank: Int, dims_addr: Int, steps_addr: Int) -> Bool:
    """`_hfs_view_finite` for a float of `width` bytes (2, 4 or 8). A view
    in the other byte order is tested with the byte-swapped exponent field,
    so no word is converted."""
    if width == 4:
        return _hfs_view_finite[DType.uint32](addr, outer, inner, s0, s1,
            UInt32(0x7F800000) if native else UInt32(0x0000807F), outer_rank, dims_addr, steps_addr)
    if width == 8:
        return _hfs_view_finite[DType.uint64](addr, outer, inner, s0, s1,
            UInt64(0x7FF0000000000000) if native else UInt64(0x000000000000F07F), outer_rank, dims_addr, steps_addr)
    return _hfs_view_finite[DType.uint16](addr, outer, inner, s0, s1,
        UInt16(0x7C00) if native else UInt16(0x007C), outer_rank, dims_addr, steps_addr)


def host_finite_width_supported(width: Int) -> Bool:
    """True for the float widths `host_all_finite_view` and
    `host_all_finite_nd` scan: 2, 4 and 8 bytes."""
    return width == 2 or width == 4 or width == 8


def host_all_finite_view(addr: Int, s0: Int, s1: Int, rows: Int, cols: Int, width: Int, native: Bool) -> Bool:
    """True when none of the floats of `width` bytes (2, 4 or 8) of the
    rows x cols view at byte address `addr` with byte strides `s0` (rows)
    and `s1` (columns) is a NaN or an infinity. `native` is False for a view
    in the other byte order. C-contiguous and Fortran-contiguous views, a
    single row and a single column are one flat run of words; any other
    view is rows of strided words. One parallel pass in every case."""
    if rows <= 0 or cols <= 0:
        return True
    if cols == 1:
        return _hfs_view_dispatch(addr, 1, rows, 0, width if rows == 1 else s0, width, native, 0, 0, 0)
    if rows == 1:
        return _hfs_view_dispatch(addr, 1, cols, 0, s1, width, native, 0, 0, 0)
    if (s1 == width and s0 == cols * width) or (s0 == width and s1 == rows * width):
        # The verdict does not depend on the order the words are read in.
        return _hfs_view_dispatch(addr, 1, rows * cols, 0, width, width, native, 0, 0, 0)
    return _hfs_view_dispatch(addr, rows, cols, s0, s1, width, native, 1, 0, 0)


def host_all_finite_nd(addr: Int, shape: List[Int], strides: List[Int], width: Int, native: Bool) -> Bool:
    """True when none of the floats of `width` bytes (2, 4 or 8) of the
    view at byte address `addr` with the given `shape` and byte `strides`
    (any rank, any sign) is a NaN or an infinity. A C-contiguous or
    Fortran-contiguous view is one flat run of words; otherwise the longest
    contiguous trailing block is the SIMD unit and the leading axes are
    walked. One parallel pass in every case."""
    var rank = len(shape)
    var count = 1
    for axis in range(rank):  # small-loop(admitted rank metadata)
        count *= shape[axis]
    if count <= 0:
        return True
    var c_order = True
    var f_order = True
    var expected_c = width
    var expected_f = width
    for i in range(rank):  # small-loop(admitted rank metadata)
        var back = rank - i - 1
        if shape[back] != 1 and strides[back] != expected_c:
            c_order = False
        expected_c *= shape[back]
        if shape[i] != 1 and strides[i] != expected_f:
            f_order = False
        expected_f *= shape[i]
    if c_order or f_order:
        return _hfs_view_dispatch(addr, 1, count, 0, width, width, native, 0, 0, 0)
    # The longest trailing block of axes whose words are adjacent in memory.
    var k = rank - 1
    var inner = 1
    var expected = width
    while k >= 0 and (shape[k] == 1 or strides[k] == expected):  # small-loop(admitted rank metadata)
        inner *= shape[k]
        expected *= shape[k]
        k -= 1
    var s1 = width
    if inner == 1:
        # No adjacent block: the innermost axis longer than one is a run of strided words.
        inner = shape[k]
        s1 = strides[k]
        k -= 1
    var outer_rank = k + 1
    if outer_rank <= 0:
        return _hfs_view_dispatch(addr, 1, inner, 0, s1, width, native, 0, 0, 0)
    if outer_rank == 1:
        return _hfs_view_dispatch(addr, shape[0], inner, strides[0], s1, width, native, 1, 0, 0)
    var dims = List[Int](length=outer_rank, fill=1)
    var steps = List[Int](length=outer_rank, fill=0)
    for axis in range(outer_rank):  # small-loop(admitted rank metadata)
        dims[axis] = shape[axis]
        steps[axis] = strides[axis]
    var dims_ptr = rebind[MutPointer[Int, MutUntrackedOrigin]](dims.unsafe_ptr())
    var steps_ptr = rebind[MutPointer[Int, MutUntrackedOrigin]](steps.unsafe_ptr())
    var finite = _hfs_view_dispatch(addr, count // inner, inner, 0, s1, width, native, outer_rank,
                                    Int(dims_ptr), Int(steps_ptr))
    _ = dims^
    _ = steps^
    return finite


def host_finite_status(address: Int, n: Int) -> Int32:
    """The finite status word of the `n` float32 words at host `address`
    (core/admitted_finite_status.mojo contract): 0 when all are finite, 1
    otherwise (the parallel SIMD scan; the CPU column of a deferred finite
    test). Moved here from core/admitted_finite_status.mojo (lane a6)."""
    return Int32(0) if host_all_finite_f32(_HFS_F32P(unsafe_from_address=address), n) else Int32(1)
