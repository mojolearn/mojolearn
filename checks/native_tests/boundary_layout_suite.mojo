# SPDX-License-Identifier: Apache-2.0
"""The Python-free half of the boundary read kit (core/boundary_layout.mojo;
lane thin-unpack-kit, 2026-11-11): Py_buffer word decoding, contiguity,
format kinds and the one conversion copy, with Mojo-owned memory standing in
for an exporter. No Python participates. Fixtures are independent units and
run across the native host worker pool.
"""
from core.host_parallel import host_parallelize
from core.boundary_layout import (
    BufferLayout, buffer_layout_from_words, layout_is_c_contiguous, format_kind, format_matches,
    gather_f32, gather_f64, gather_i64, c_strides, PY_BUFFER_WORDS,
    BOUNDARY_KIND_FLOAT, BOUNDARY_KIND_INT, BOUNDARY_KIND_UINT, BOUNDARY_KIND_BOOL,
    BOUNDARY_KIND_UCS4, BOUNDARY_KIND_BYTES, text_format_kind, text_rows,
)

# Fixtures 0-7 run across the pool; fixture 8 (a conversion that itself
# splits across the pool) runs after them on the calling thread, never as a
# nested pool task.
comptime BOUNDARY_LAYOUT_FIXTURES = 8


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def _words(buf: Int, nbytes: Int, itemsize: Int, ndim: Int, shape: Int, strides: Int,
           suboffsets: Int = 0) -> List[Int]:
    var words = List[Int](length=PY_BUFFER_WORDS, fill=0)
    words[0] = buf
    words[2] = nbytes
    words[3] = itemsize
    words[4] = Int(UInt64(ndim) << 32)  # readonly 0 (low c_int), ndim (high c_int)
    words[6] = shape
    words[7] = strides
    words[8] = suboffsets
    return words^


def _refuses(words: List[Int], text: String) -> Bool:
    try:
        _ = buffer_layout_from_words(words)
    except error:
        return String(error).find(text) >= 0
    return False


def fixture(unit: Int) raises:
    if unit == 0:
        # A 2x3 C-order float32 block, readonly flag set.
        var data: List[Float32] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        var shape: List[Int] = [2, 3]
        var strides: List[Int] = [12, 4]
        var words = _words(Int(data.unsafe_ptr()), 24, 4, 2, Int(shape.unsafe_ptr()), Int(strides.unsafe_ptr()))
        words[4] = words[4] | 1
        var layout = buffer_layout_from_words(words)
        require(layout.address == Int(data.unsafe_ptr()) and layout.count == 6 and layout.ndim == 2,
                "boundary_layout decodes address, count and rank")
        require(layout.shape[0] == 2 and layout.shape[1] == 3 and layout.strides[0] == 12, "boundary_layout shape/strides")
        require(layout.readonly, "boundary_layout readonly flag")
        require(layout_is_c_contiguous(layout), "boundary_layout C contiguity")
        _ = data^
        _ = shape^
        _ = strides^
    elif unit == 1:
        # NULL shape: one byte run of len // itemsize items; NULL strides: C order.
        var data = List[UInt8](length=16, fill=0)
        var flat = buffer_layout_from_words(_words(Int(data.unsafe_ptr()), 16, 1, 0, 0, 0))
        require(flat.ndim == 1 and flat.shape[0] == 16 and flat.strides[0] == 1, "boundary_layout NULL shape")
        var shape: List[Int] = [2, 2]
        var nd = buffer_layout_from_words(_words(Int(data.unsafe_ptr()), 16, 4, 2, Int(shape.unsafe_ptr()), 0))
        require(nd.strides[0] == 8 and nd.strides[1] == 4, "boundary_layout NULL strides are C order")
        var expected = c_strides(shape, 4)
        require(expected[0] == 8 and expected[1] == 4, "boundary_layout c_strides")
        _ = data^
        _ = shape^
    elif unit == 2:
        # Refusals: indirect storage, size mismatch, short word list.
        var data = List[Float32](length=4, fill=0.0)
        var shape: List[Int] = [4]
        var strides: List[Int] = [4]
        var sub: List[Int] = [0]
        require(_refuses(_words(Int(data.unsafe_ptr()), 16, 4, 1, Int(shape.unsafe_ptr()), Int(strides.unsafe_ptr()),
                                Int(sub.unsafe_ptr())), "indirect"), "boundary_layout refuses indirect buffers")
        require(_refuses(_words(Int(data.unsafe_ptr()), 12, 4, 1, Int(shape.unsafe_ptr()), Int(strides.unsafe_ptr())),
                         "size differs"), "boundary_layout refuses a size mismatch")
        require(_refuses(List[Int](length=3, fill=0), "ten Py_buffer words"), "boundary_layout refuses short words")
        _ = data^
        _ = shape^
        _ = strides^
        _ = sub^
    elif unit == 3:
        require(format_kind("f", 4) == BOUNDARY_KIND_FLOAT and format_kind("<d", 8) == BOUNDARY_KIND_FLOAT,
                "boundary_layout float formats")
        require(format_kind("=q", 8) == BOUNDARY_KIND_INT and format_kind("l", 8) == BOUNDARY_KIND_INT,
                "boundary_layout int formats")
        require(format_kind("@B", 1) == BOUNDARY_KIND_UINT and format_kind("?", 1) == BOUNDARY_KIND_BOOL,
                "boundary_layout uint/bool formats")
        var refused = 0
        for text in [">f", "Zf", "T{f:x:}", "s"]:  # driver-loop: over four fixed refused formats
            try:
                _ = format_kind(String(text), 4)
            except:
                refused += 1
        require(refused == 4, "boundary_layout refuses big-endian, complex, struct and bytes formats")
    elif unit == 4:
        require(format_matches(BOUNDARY_KIND_FLOAT, 4, "<f4") and not format_matches(BOUNDARY_KIND_FLOAT, 8, "<f4"),
                "boundary_layout format_matches float32")
        require(format_matches(BOUNDARY_KIND_INT, 8, "<i8") and format_matches(BOUNDARY_KIND_BOOL, 1, "|b1"),
                "boundary_layout format_matches int64/bool")
    elif unit == 5:
        # F-order 2x3 float64 -> C-order float32 (one strided conversion copy).
        var data: List[Float64] = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
        var shape: List[Int] = [2, 3]
        var strides: List[Int] = [8, 16]
        var layout = buffer_layout_from_words(_words(Int(data.unsafe_ptr()), 48, 8, 2, Int(shape.unsafe_ptr()),
                                                     Int(strides.unsafe_ptr())))
        require(not layout_is_c_contiguous(layout), "boundary_layout F order is not C contiguous")
        var dst = List[Float32](length=6, fill=0.0)
        gather_f32(layout, BOUNDARY_KIND_FLOAT, Int(dst.unsafe_ptr()))
        for i in range(6):  # driver-loop: over six fixture elements
            require(dst[i] == Float32(i + 1), "boundary_layout F->C float64->float32 element " + String(i))
        var wide = List[Float64](length=6, fill=0.0)
        gather_f64(layout, BOUNDARY_KIND_FLOAT, Int(wide.unsafe_ptr()))
        require(wide[1] == 2.0 and wide[3] == 4.0, "boundary_layout F->C float64 copy")
        _ = data^
        _ = shape^
        _ = strides^
    elif unit == 6:
        # A C-order float32 block is one memcpy, bit for bit (a negative zero survives).
        var data: List[Float32] = [-0.0, 1.5, -2.25]
        var shape: List[Int] = [3]
        var strides: List[Int] = [4]
        var layout = buffer_layout_from_words(_words(Int(data.unsafe_ptr()), 12, 4, 1, Int(shape.unsafe_ptr()),
                                                     Int(strides.unsafe_ptr())))
        var dst = List[UInt32](length=3, fill=7)
        gather_f32(layout, BOUNDARY_KIND_FLOAT, Int(dst.unsafe_ptr()))
        require(dst[0] == UInt32(0x80000000) and dst[1] == UInt32(0x3FC00000), "boundary_layout memcpy keeps bits")
        _ = data^
        _ = shape^
        _ = strides^
    elif unit == 7:
        # uint8 and bool -> int64; floating storage is refused, never truncated.
        var data: List[UInt8] = [0, 2, 255]
        var shape: List[Int] = [3]
        var strides: List[Int] = [1]
        var layout = buffer_layout_from_words(_words(Int(data.unsafe_ptr()), 3, 1, 1, Int(shape.unsafe_ptr()),
                                                     Int(strides.unsafe_ptr())))
        var dst = List[Int64](length=3, fill=0)
        gather_i64(layout, BOUNDARY_KIND_UINT, Int(dst.unsafe_ptr()))
        require(dst[0] == 0 and dst[1] == 2 and dst[2] == 255, "boundary_layout uint8 -> int64")
        gather_i64(layout, BOUNDARY_KIND_BOOL, Int(dst.unsafe_ptr()))
        require(dst[0] == 0 and dst[1] == 1 and dst[2] == 1, "boundary_layout bool -> int64")
        var refused = False
        try:
            gather_i64(layout, BOUNDARY_KIND_FLOAT, Int(dst.unsafe_ptr()))
        except:
            refused = True
        require(refused, "boundary_layout refuses float -> int64")
        # The GP integer label read: an F-order 2x2 int16 matrix -> C-order int64
        # (the walk's depth-first flatten order), negatives sign-extended.
        var small: List[Int16] = [-3, 7, 5, -1]
        var dims: List[Int] = [2, 2]
        var steps: List[Int] = [2, 4]
        var matrix = buffer_layout_from_words(_words(Int(small.unsafe_ptr()), 8, 2, 2, Int(dims.unsafe_ptr()),
                                                     Int(steps.unsafe_ptr())))
        var flat = List[Int64](length=4, fill=0)
        gather_i64(matrix, BOUNDARY_KIND_INT, Int(flat.unsafe_ptr()))
        require(flat[0] == -3 and flat[1] == 5 and flat[2] == 7 and flat[3] == -1, "boundary_layout F-order int16 -> C int64")
        _ = small^
        _ = dims^
        _ = steps^
        _ = data^
        _ = shape^
        _ = strides^
    else:
        # A strided conversion larger than one chunk runs across the pool.
        var n = 3 * 65536 + 5
        var data = List[Float64](length=2 * n, fill=0.0)
        for i in range(n):  # driver-loop: fixture fill (every other element)
            data[2 * i] = Float64(i)
        var shape: List[Int] = [n]
        var strides: List[Int] = [16]
        var layout = buffer_layout_from_words(_words(Int(data.unsafe_ptr()), n * 8, 8, 1, Int(shape.unsafe_ptr()),
                                                     Int(strides.unsafe_ptr())))
        var dst = List[Float32](length=n, fill=-1.0)
        gather_f32(layout, BOUNDARY_KIND_FLOAT, Int(dst.unsafe_ptr()))
        var bad = 0
        for i in range(n):  # driver-loop: check every converted element
            if dst[i] != Float32(i):
                bad += 1
        require(bad == 0, "boundary_layout chunked strided conversion: " + String(bad) + " wrong")
        _ = data^
        _ = shape^
        _ = strides^


def text_fixture() raises:
    """Fixed-width text storage (lane loops-prep-host): the text format
    kinds, a strided UCS4 vector and a bytes vector read by `text_rows`
    (trailing NULs dropped, as NumPy's tolist), and the numeric gathers'
    refusal of text storage. Runs after the pool fixtures: `text_rows`
    itself splits across the pool."""
    require(text_format_kind("w") == BOUNDARY_KIND_UCS4, "text kind w")
    require(text_format_kind("<3w") == BOUNDARY_KIND_UCS4, "text kind <3w")
    require(text_format_kind("=12w") == BOUNDARY_KIND_UCS4, "text kind =12w")
    require(text_format_kind("5s") == BOUNDARY_KIND_BYTES, "text kind 5s")
    require(text_format_kind("s") == BOUNDARY_KIND_BYTES, "text kind s")
    require(text_format_kind(">3w") == 0, "text kind >3w is not admitted")
    require(text_format_kind("<f") == 0 and text_format_kind("f") == 0, "numeric formats are not text")
    require(text_format_kind("3x") == 0 and text_format_kind("<") == 0 and text_format_kind("") == 0,
            "malformed formats are not text")
    # Three UCS4 elements of width 3 at a 24-byte stride (every other slot):
    # "ab", "", "xyz".
    var wide = List[UInt32](length=18, fill=0)
    wide[0] = 97
    wide[1] = 98
    wide[12] = 120
    wide[13] = 121
    wide[14] = 122
    wide[3] = 0x1F600  # padding slot between elements: never read
    var shape: List[Int] = [3]
    var strides: List[Int] = [24]
    var layout = buffer_layout_from_words(_words(Int(wide.unsafe_ptr()), 3 * 12, 12, 1, Int(shape.unsafe_ptr()),
                                                 Int(strides.unsafe_ptr())))
    var rows = text_rows(layout, BOUNDARY_KIND_UCS4)
    require(rows.count == 3, "text_rows ucs4 count")
    require(rows.lengths[0] == 2 and rows.lengths[1] == 0 and rows.lengths[2] == 3, "text_rows ucs4 lengths")
    require(rows.starts[0] == 0 and rows.starts[1] == 2 and rows.starts[2] == 2, "text_rows ucs4 starts")
    require(len(rows.points) == 5 and rows.points[0] == 97 and rows.points[1] == 98 and rows.points[2] == 120
            and rows.points[4] == 122, "text_rows ucs4 points")
    var refused = False
    try:
        var sink = List[Float64](length=3, fill=0.0)
        gather_f64(layout, BOUNDARY_KIND_UCS4, Int(sink.unsafe_ptr()))
        _ = sink^
    except:
        refused = True
    require(refused, "numeric gather refuses text storage")
    # Two bytes elements of width 4: "hi", "abcd" (an interior NUL is kept).
    var narrow = List[UInt8](length=8, fill=0)
    narrow[0] = 104
    narrow[1] = 105
    narrow[4] = 97
    narrow[5] = 0
    narrow[6] = 99
    narrow[7] = 100
    var bshape: List[Int] = [2]
    var bstrides: List[Int] = [4]
    var blayout = buffer_layout_from_words(_words(Int(narrow.unsafe_ptr()), 8, 4, 1, Int(bshape.unsafe_ptr()),
                                                  Int(bstrides.unsafe_ptr())))
    var brows = text_rows(blayout, BOUNDARY_KIND_BYTES)
    require(brows.lengths[0] == 2 and brows.lengths[1] == 4, "text_rows bytes lengths")
    require(brows.points[2] == 97 and brows.points[3] == 0 and brows.points[5] == 100, "text_rows bytes points")
    _ = wide^
    _ = shape^
    _ = strides^
    _ = narrow^
    _ = bshape^
    _ = bstrides^


def run_checks() raises:
    var failures = List[Int](length=BOUNDARY_LAYOUT_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            fixture(unit)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL boundary_layout fixture", unit, String(error))
    host_parallelize(task, BOUNDARY_LAYOUT_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "boundary_layout failed " + String(failed) + " fixtures")
    fixture(BOUNDARY_LAYOUT_FIXTURES)
    text_fixture()
    print("PASS boundary_layout:", BOUNDARY_LAYOUT_FIXTURES + 2, "Py_buffer decode / format / conversion / text fixtures")
