# SPDX-License-Identifier: Apache-2.0
"""The Python-free half of the boundary read kit (lane thin-unpack-kit,
2026-11-11; spec ~/mojolearn-evidence/briefs-20261111/_thin_unpack_shared.md).

`bindings/boundary_read.mojo` makes ONE `PyObject_GetBuffer` per array
argument and hands the ten Py_buffer words here. Everything after that is
plain Mojo: decoding the words into a layout (shape, strides, itemsize,
readonly, element kind), the contiguity test, and the ONE conversion copy a
caller makes only when the dtype or the layout demands it. No Python object,
no CPython call, so `checks/native_tests/boundary_layout_suite.mojo` tests it
natively with Mojo-owned memory standing in for an exporter.

Py_buffer (stable ABI, 64-bit): word 0 buf, 1 obj, 2 len, 3 itemsize,
4 readonly (low c_int) | ndim (high c_int), 5 format, 6 shape, 7 strides,
8 suboffsets, 9 internal. The same reading as
providers/python/borrowed_buffer_pyapi.mojo `AdmittedBorrowedBuffer`.
"""
from std.memory import unsafe_memcpy
from core.host_parallel import host_parallelize

comptime PY_BUFFER_WORDS = 10

# PyBUF_* request flags (CPython Include/pybuffer.h).
comptime PyBUF_SIMPLE = 0
comptime PyBUF_WRITABLE = 0x0001
comptime PyBUF_FORMAT = 0x0004
comptime PyBUF_ND = 0x0008
comptime PyBUF_STRIDES = 0x0018
comptime PyBUF_INDIRECT = 0x0118
comptime PyBUF_RECORDS_RO = 0x001C
comptime PyBUF_RECORDS = 0x001D
comptime PyBUF_FULL_RO = 0x011C

# Element kinds of a buffer format (the struct-module letter, any native
# byte-order prefix stripped); the width is the exported itemsize.
comptime BOUNDARY_KIND_FLOAT = 1
comptime BOUNDARY_KIND_INT = 2
comptime BOUNDARY_KIND_UINT = 3
comptime BOUNDARY_KIND_BOOL = 4
#: Fixed-width text storage, admitted only by `text_format_kind` (a text
#: view, bindings/boundary_read.mojo `text_buffer_view`): NumPy 'U' (UCS4
#: code points, NUL padded, struct format "w" / "<Nw") and NumPy 'S' (bytes,
#: NUL padded, "s" / "Ns"). The numeric gathers refuse both.
comptime BOUNDARY_KIND_UCS4 = 5
comptime BOUNDARY_KIND_BYTES = 6

#: Elements per conversion task: one task moves at least 256 KB of f32, so
#: the pool's dispatch cost is amortized whatever the array's shape (cost
#: reasoning, not a board dimension).
comptime BOUNDARY_COPY_CHUNK = 65536


@fieldwise_init
struct BufferLayout(Copyable, Movable):
    """The decoded layout of one exported buffer. `address` is the first
    element; `strides` are in bytes and may be negative or zero."""
    var address: Int
    var nbytes: Int
    var itemsize: Int
    var ndim: Int
    var shape: List[Int]
    var strides: List[Int]
    var readonly: Bool
    var count: Int


def buffer_layout_from_words(words: List[Int]) raises -> BufferLayout:
    """Decode the ten Py_buffer words a `PyObject_GetBuffer` filled. A NULL
    shape is a one-dimensional byte run (`len // itemsize` items); a NULL
    stride table is C order. An indirect (PIL-style) buffer is refused."""
    if len(words) < PY_BUFFER_WORDS:
        raise Error("BufferError: native buffer view needs ten Py_buffer words")
    var packed = UInt64(words[4])
    var readonly = (packed & UInt64(0xffffffff)) != 0
    var ndim = Int(packed >> 32)
    if ndim < 0 or ndim > 64:
        raise Error("BufferError: native buffer view has an invalid rank")
    var itemsize = words[3] if words[3] > 0 else 1
    var nbytes = words[2]
    var shape = List[Int]()
    var strides = List[Int]()
    if words[6] == 0:
        ndim = 1
        shape.append(nbytes // itemsize)
        strides.append(itemsize)
    else:
        var dims = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=words[6])
        for axis in range(ndim):  # driver-loop: over buffer rank metadata
            shape.append(dims[axis])
        if words[7] == 0:
            strides = c_strides(shape, itemsize)
        else:
            var steps = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=words[7])
            for axis in range(ndim):  # driver-loop: over buffer rank metadata
                strides.append(steps[axis])
    if words[8] != 0:
        var indirect = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=words[8])
        for axis in range(ndim):  # driver-loop: over buffer rank metadata
            if indirect[axis] >= 0:
                raise Error("BufferError: indirect buffers are not supported")
    var count = 1
    for axis in range(ndim):  # driver-loop: over buffer rank metadata
        if shape[axis] < 0:
            raise Error("BufferError: native buffer view has a negative extent")
        count *= shape[axis]
    if count * itemsize != nbytes:
        raise Error("BufferError: native buffer size differs from its shape")
    return BufferLayout(words[0], nbytes, itemsize, ndim, shape^, strides^, readonly, count)


def c_strides(shape: List[Int], itemsize: Int) -> List[Int]:
    """Byte strides of a C-order block of `shape`."""
    var strides = List[Int](length=len(shape), fill=0)
    var step = itemsize
    for k in range(len(shape)):  # driver-loop: over rank metadata
        var axis = len(shape) - 1 - k
        strides[axis] = step
        step *= shape[axis] if shape[axis] > 0 else 1
    return strides^


def layout_is_c_contiguous(layout: BufferLayout) -> Bool:
    """True when the elements are one C-order run (extent-1 axes and empty
    arrays ignore their strides, as NumPy's flag does)."""
    if layout.count == 0:
        return True
    var step = layout.itemsize
    for k in range(layout.ndim):  # driver-loop: over rank metadata
        var axis = layout.ndim - 1 - k
        if layout.shape[axis] != 1 and layout.strides[axis] != step:
            return False
        step *= layout.shape[axis]
    return True


def format_kind(format: String, itemsize: Int) raises -> Int:
    """The element kind of a struct-module buffer format of one native
    scalar ('f', '<d', '=q', '@B', '?', ...). Big-endian ('>', '!'),
    compound and non-numeric formats are refused."""
    var bytes = format.as_bytes()
    var start = 0
    if len(bytes) == 2:
        var prefix = bytes[0]
        if prefix == UInt8(ord(">")) or prefix == UInt8(ord("!")):
            raise Error("TypeError: big-endian buffers are not supported")
        if prefix != UInt8(ord("<")) and prefix != UInt8(ord("=")) and prefix != UInt8(ord("@")):
            raise Error("TypeError: unsupported buffer format '" + format + "'")
        start = 1
    elif len(bytes) != 1:
        if len(bytes) == 0:
            return BOUNDARY_KIND_UINT if itemsize == 1 else _unsupported(format)
        return _unsupported(format)
    var letter = bytes[start]
    if letter == UInt8(ord("f")) or letter == UInt8(ord("d")) or letter == UInt8(ord("e")):
        return BOUNDARY_KIND_FLOAT
    if (letter == UInt8(ord("b")) or letter == UInt8(ord("h")) or letter == UInt8(ord("i"))
            or letter == UInt8(ord("l")) or letter == UInt8(ord("q")) or letter == UInt8(ord("n"))):
        return BOUNDARY_KIND_INT
    if (letter == UInt8(ord("B")) or letter == UInt8(ord("H")) or letter == UInt8(ord("I"))
            or letter == UInt8(ord("L")) or letter == UInt8(ord("Q")) or letter == UInt8(ord("N"))):
        return BOUNDARY_KIND_UINT
    if letter == UInt8(ord("?")):
        return BOUNDARY_KIND_BOOL
    return _unsupported(format)


def text_format_kind(format: String) -> Int:
    """`BOUNDARY_KIND_UCS4` for a little-endian or native UCS4 text format
    ("w", "<10w", "=3w", "@w"), `BOUNDARY_KIND_BYTES` for a bytes format
    ("s", "8s"), 0 for anything else (a numeric format, a big-endian UCS4
    run, a compound format)."""
    var bytes = format.as_bytes()
    var n = len(bytes)
    if n == 0:
        return 0
    var start = 0
    var prefix = bytes[0]
    if prefix == UInt8(ord("<")) or prefix == UInt8(ord("=")) or prefix == UInt8(ord("@")):
        start = 1
    var letter = bytes[n - 1]
    for k in range(start, n - 1):  # driver-loop: over the format's repeat-count digits (metadata)
        if bytes[k] < UInt8(ord("0")) or bytes[k] > UInt8(ord("9")):
            return 0
    if n - 1 >= start:
        if letter == UInt8(ord("w")):
            return BOUNDARY_KIND_UCS4
        if letter == UInt8(ord("s")):
            return BOUNDARY_KIND_BYTES
    return 0


def _unsupported(format: String) raises -> Int:
    raise Error("TypeError: unsupported buffer format '" + format + "'")


def format_matches(kind: Int, itemsize: Int, typestr: StaticString) -> Bool:
    """True when (kind, itemsize) is the NumPy typestr "<f4", "<f8", "<i4",
    "<i8", "<u1", "|b1", ... of a little-endian target."""
    var want_kind = 0
    var bytes = typestr.as_bytes()
    if len(bytes) != 3:
        return False
    var letter = bytes[1]
    if letter == UInt8(ord("f")):
        want_kind = BOUNDARY_KIND_FLOAT
    elif letter == UInt8(ord("i")):
        want_kind = BOUNDARY_KIND_INT
    elif letter == UInt8(ord("u")):
        want_kind = BOUNDARY_KIND_UINT
    elif letter == UInt8(ord("b")):
        want_kind = BOUNDARY_KIND_BOOL
    return want_kind == kind and Int(bytes[2]) - ord("0") == itemsize


def element_offset(flat: Int, layout: BufferLayout) -> Int:
    """The byte offset of the `flat`-th element in C order."""
    var offset = 0
    var rest = flat
    for k in range(layout.ndim):  # driver-loop: over rank metadata
        var axis = layout.ndim - 1 - k
        var extent = layout.shape[axis]
        offset += (rest % extent) * layout.strides[axis]
        rest //= extent
    return offset


@always_inline
def element_address(flat: Int, layout: BufferLayout, contiguous: Bool) -> Int:
    """The address of the `flat`-th element in C order: `flat * itemsize`
    past the base for a C-contiguous layout (no div or mod), the strided
    `element_offset` walk only when the strides demand it. `contiguous` is
    `layout_is_c_contiguous(layout)`, read once per conversion."""
    if contiguous:
        return layout.address + flat * layout.itemsize
    return layout.address + element_offset(flat, layout)


@always_inline
def _load_f64(address: Int, kind: Int, itemsize: Int) -> Float64:
    if kind == BOUNDARY_KIND_FLOAT:
        if itemsize == 4:
            return Float64(MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=address)[0])
        if itemsize == 8:
            return MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=address)[0]
        return Float64(MutPointer[Float16, MutUntrackedOrigin](unsafe_from_address=address)[0])
    if kind == BOUNDARY_KIND_BOOL:
        return 1.0 if MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=address)[0] != 0 else 0.0
    if kind == BOUNDARY_KIND_UINT:
        if itemsize == 1:
            return Float64(MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=address)[0])
        if itemsize == 2:
            return Float64(MutPointer[UInt16, MutUntrackedOrigin](unsafe_from_address=address)[0])
        if itemsize == 4:
            return Float64(MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=address)[0])
        return Float64(MutPointer[UInt64, MutUntrackedOrigin](unsafe_from_address=address)[0])
    return Float64(_load_i64(address, kind, itemsize))


@always_inline
def _load_i64(address: Int, kind: Int, itemsize: Int) -> Int64:
    if kind == BOUNDARY_KIND_BOOL:
        return 1 if MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=address)[0] != 0 else 0
    if kind == BOUNDARY_KIND_UINT:
        if itemsize == 1:
            return Int64(MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=address)[0])
        if itemsize == 2:
            return Int64(MutPointer[UInt16, MutUntrackedOrigin](unsafe_from_address=address)[0])
        if itemsize == 4:
            return Int64(MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=address)[0])
        return Int64(MutPointer[UInt64, MutUntrackedOrigin](unsafe_from_address=address)[0])
    if itemsize == 1:
        return Int64(MutPointer[Int8, MutUntrackedOrigin](unsafe_from_address=address)[0])
    if itemsize == 2:
        return Int64(MutPointer[Int16, MutUntrackedOrigin](unsafe_from_address=address)[0])
    if itemsize == 4:
        return Int64(MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=address)[0])
    return MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=address)[0]


def _check_width(kind: Int, itemsize: Int) raises:
    if kind == BOUNDARY_KIND_UCS4 or kind == BOUNDARY_KIND_BYTES:
        raise Error("TypeError: a numeric buffer is required, got fixed-width text storage")
    var ok = itemsize == 1 or itemsize == 2 or itemsize == 4 or itemsize == 8
    if kind == BOUNDARY_KIND_FLOAT:
        ok = itemsize == 2 or itemsize == 4 or itemsize == 8
    elif kind == BOUNDARY_KIND_BOOL:
        ok = itemsize == 1
    if not ok:
        raise Error("TypeError: unsupported buffer element width " + String(itemsize))


def gather_f32(layout: BufferLayout, kind: Int, dst_address: Int) raises:
    """The ONE conversion copy into a C-order float32 block of
    `layout.count` elements at `dst_address` (the caller sized it). A
    float32 C-contiguous source is one memcpy; any other kind or layout is
    converted element by element in native code, split across the host pool
    in `BOUNDARY_COPY_CHUNK` runs. Rounding: float64 -> float32 is IEEE
    round-to-nearest, the NumPy `astype(float32)` result."""
    _check_width(kind, layout.itemsize)
    if layout.count == 0:
        return
    var contiguous = layout_is_c_contiguous(layout)
    if kind == BOUNDARY_KIND_FLOAT and layout.itemsize == 4 and contiguous:
        unsafe_memcpy(dest=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=dst_address),
            src=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=layout.address), count=layout.nbytes)
        return
    var units = (layout.count + BOUNDARY_COPY_CHUNK - 1) // BOUNDARY_COPY_CHUNK
    var lp = UnsafePointer(to=layout)
    var itemsize = layout.itemsize
    var count = layout.count
    def task(unit: Int) {imm lp, imm kind, imm itemsize, imm count, imm dst_address, imm contiguous}:
        var dst = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=dst_address)
        var stop = min(count, (unit + 1) * BOUNDARY_COPY_CHUNK)
        for flat in range(unit * BOUNDARY_COPY_CHUNK, stop):  # driver-loop: elements of one conversion chunk (native, no Python)
            var at = element_address(flat, lp[], contiguous)
            if kind == BOUNDARY_KIND_FLOAT and itemsize == 4:
                dst[flat] = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=at)[0]
            else:
                dst[flat] = Float32(_load_f64(at, kind, itemsize))
    host_parallelize(task, units)
    _ = layout  # the layout's shape/stride lists are read by every task above


def gather_f64(layout: BufferLayout, kind: Int, dst_address: Int) raises:
    """`gather_f32` into a C-order float64 block."""
    _check_width(kind, layout.itemsize)
    if layout.count == 0:
        return
    var contiguous = layout_is_c_contiguous(layout)
    if kind == BOUNDARY_KIND_FLOAT and layout.itemsize == 8 and contiguous:
        unsafe_memcpy(dest=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=dst_address),
            src=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=layout.address), count=layout.nbytes)
        return
    var units = (layout.count + BOUNDARY_COPY_CHUNK - 1) // BOUNDARY_COPY_CHUNK
    var lp = UnsafePointer(to=layout)
    var itemsize = layout.itemsize
    var count = layout.count
    def task(unit: Int) {imm lp, imm kind, imm itemsize, imm count, imm dst_address, imm contiguous}:
        var dst = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=dst_address)
        var stop = min(count, (unit + 1) * BOUNDARY_COPY_CHUNK)
        for flat in range(unit * BOUNDARY_COPY_CHUNK, stop):  # driver-loop: elements of one conversion chunk (native, no Python)
            dst[flat] = _load_f64(element_address(flat, lp[], contiguous), kind, itemsize)
    host_parallelize(task, units)
    _ = layout  # the layout's shape/stride lists are read by every task above


def gather_i64(layout: BufferLayout, kind: Int, dst_address: Int) raises:
    """`gather_f32` into a C-order int64 block, from integer or bool
    storage only (a float source is refused, never truncated)."""
    if kind == BOUNDARY_KIND_FLOAT:
        raise Error("TypeError: an integer buffer is required, got floating storage")
    _check_width(kind, layout.itemsize)
    if layout.count == 0:
        return
    var contiguous = layout_is_c_contiguous(layout)
    if kind == BOUNDARY_KIND_INT and layout.itemsize == 8 and contiguous:
        unsafe_memcpy(dest=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=dst_address),
            src=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=layout.address), count=layout.nbytes)
        return
    var units = (layout.count + BOUNDARY_COPY_CHUNK - 1) // BOUNDARY_COPY_CHUNK
    var lp = UnsafePointer(to=layout)
    var itemsize = layout.itemsize
    var count = layout.count
    def task(unit: Int) {imm lp, imm kind, imm itemsize, imm count, imm dst_address, imm contiguous}:
        var dst = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=dst_address)
        var stop = min(count, (unit + 1) * BOUNDARY_COPY_CHUNK)
        for flat in range(unit * BOUNDARY_COPY_CHUNK, stop):  # driver-loop: elements of one conversion chunk (native, no Python)
            dst[flat] = _load_i64(element_address(flat, lp[], contiguous), kind, itemsize)
    host_parallelize(task, units)
    _ = layout  # the layout's shape/stride lists are read by every task above


struct TextRows(Movable):
    """The elements of a fixed-width text buffer, in C order: element `i` is
    the code units `points[starts[i] : starts[i] + lengths[i]]` (UCS4 code
    points, or the bytes of an 'S' element widened), trailing NULs dropped,
    which is the value NumPy's `tolist()` / `str()` gives the element."""
    var points: List[UInt32]
    var starts: List[Int]
    var lengths: List[Int]
    var count: Int

    def __init__(out self, var points: List[UInt32], var starts: List[Int], var lengths: List[Int], count: Int):
        self.points = points^
        self.starts = starts^
        self.lengths = lengths^
        self.count = count


@always_inline
def _text_length(at: Int, kind: Int, width: Int) -> Int:
    var m = width
    if kind == BOUNDARY_KIND_UCS4:
        var wide = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=at)
        while m > 0 and wide[m - 1] == 0:
            m -= 1
    else:
        var narrow = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=at)
        while m > 0 and narrow[m - 1] == 0:
            m -= 1
    return m


def text_rows(layout: BufferLayout, kind: Int) raises -> TextRows:
    """Every element of a 'U' or 'S' buffer as one contiguous run of code
    units (`TextRows`); see `text_rows_into`."""
    var points = List[UInt32]()
    var starts = List[Int]()
    var lengths = List[Int]()
    var n = text_rows_into(layout, kind, points, starts, lengths)
    return TextRows(points^, starts^, lengths^, n)


def text_rows_into(layout: BufferLayout, kind: Int, mut points: List[UInt32], mut starts: List[Int],
                   mut lengths: List[Int]) raises -> Int:
    """`text_rows` into the caller's three lists (replaced), returning the
    element count. Two native passes split across the host pool in
    `BOUNDARY_COPY_CHUNK` runs (the lengths, then the copies), one prefix
    sum between them; no Python object per element."""
    if kind != BOUNDARY_KIND_UCS4 and kind != BOUNDARY_KIND_BYTES:
        raise Error("TypeError: a fixed-width text buffer is required")
    var width = layout.itemsize // 4 if kind == BOUNDARY_KIND_UCS4 else layout.itemsize
    var n = layout.count
    lengths = List[Int](length=n, fill=0)
    starts = List[Int](length=n, fill=0)
    if n == 0:
        points = List[UInt32]()
        return 0
    var units = (n + BOUNDARY_COPY_CHUNK - 1) // BOUNDARY_COPY_CHUNK
    var lp = UnsafePointer(to=layout)
    var lengths_address = Int(lengths.unsafe_ptr())
    var contiguous = layout_is_c_contiguous(layout)
    def measure(unit: Int) {imm lp, imm kind, imm width, imm n, imm lengths_address, imm contiguous}:
        var lens = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=lengths_address)
        var stop = min(n, (unit + 1) * BOUNDARY_COPY_CHUNK)
        for flat in range(unit * BOUNDARY_COPY_CHUNK, stop):  # driver-loop: elements of one text chunk (native, no Python)
            lens[flat] = _text_length(element_address(flat, lp[], contiguous), kind, width)
    host_parallelize(measure, units)
    var total = 0
    for i in range(n):  # driver-loop: prefix sum of the element lengths (native, no Python)
        starts[i] = total
        total += lengths[i]
    points = List[UInt32](length=total, fill=0)
    var points_address = Int(points.unsafe_ptr())
    var starts_address = Int(starts.unsafe_ptr())
    def fill_points(unit: Int) {imm lp, imm kind, imm n, imm lengths_address, imm starts_address, imm points_address,
                                imm contiguous}:
        var lens = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=lengths_address)
        var firsts = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=starts_address)
        var dst = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=points_address)
        var stop = min(n, (unit + 1) * BOUNDARY_COPY_CHUNK)
        for flat in range(unit * BOUNDARY_COPY_CHUNK, stop):  # driver-loop: elements of one text chunk (native, no Python)
            var at = element_address(flat, lp[], contiguous)
            var first = firsts[flat]
            if kind == BOUNDARY_KIND_UCS4:
                var wide = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=at)
                for k in range(lens[flat]):  # driver-loop: over one element's code points (native)
                    dst[first + k] = wide[k]
            else:
                var narrow = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=at)
                for k in range(lens[flat]):  # driver-loop: over one element's bytes (native)
                    dst[first + k] = UInt32(narrow[k])
    host_parallelize(fill_points, units)
    _ = layout  # the layout's shape/stride lists are read by every task above
    return n
