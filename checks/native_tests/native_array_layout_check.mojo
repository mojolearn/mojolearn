# SPDX-License-Identifier: Apache-2.0
"""Native allocation metadata: literal layouts, dtype widths and refusals.

All 45 independent fixtures use the existing host pool. Huge dimensions only
allocate short shape/stride lists; these tests never allocate their data span.
No Python objects, callbacks, dtype objects or product allocation are needed.
"""
from core.host_parallel import host_parallelize
from core.native_array_layout import NativeArrayLayout, native_array_itemsize, native_array_code, native_matrix_shape

comptime ARRAY_LAYOUT_FIXTURES = 45
comptime _MAX = 9223372036854775807


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def _layout(fixture_id: Int) raises:
    var shape = List[Int]()
    var expected = List[Int]()
    var order = String("C")
    var width = 4
    var count = 1
    var nbytes = 4
    if fixture_id == 1:
        order = "F"
        width = 8
        nbytes = 8
    elif fixture_id == 2:
        shape = [5]
        expected = [2]
        width = 2
        count = 5
        nbytes = 10
    elif fixture_id == 3 or fixture_id == 4:
        shape = [2, 3]
        expected = [12, 4]
        count = 6
        nbytes = 24
        if fixture_id == 4:
            order = "F"
            expected = [4, 8]
    elif fixture_id == 5 or fixture_id == 6:
        shape = [2, 1, 3]
        expected = [24, 24, 8]
        width = 8
        count = 6
        nbytes = 48
        if fixture_id == 6:
            order = "F"
            expected = [8, 16, 16]
    elif fixture_id == 7:
        shape = [0]
        expected = [4]
        count = 0
        nbytes = 0
    elif fixture_id == 8 or fixture_id == 9:
        shape = [2, 0, 3]
        expected = [0, 12, 4]
        count = 0
        nbytes = 0
        if fixture_id == 9:
            order = "F"
            expected = [4, 8, 0]
    elif fixture_id == 10:
        shape = [0, 2, 3]
        expected = [24, 12, 4]
        count = 0
        nbytes = 0
    elif fixture_id == 11:
        shape = [2, 3, 0]
        expected = [4, 8, 24]
        order = "F"
        count = 0
        nbytes = 0
    elif fixture_id == 12:
        # This order encounters the zero before the enormous dimension.
        # Each recorded stride stays representable; it must remain admitted.
        shape = [0, _MAX]
        expected = [4, 0]
        order = "F"
        count = 0
        nbytes = 0
    elif fixture_id == 13:
        shape = [_MAX, 0]
        expected = [0, 4]
        count = 0
        nbytes = 0
    elif fixture_id == 14:
        shape = [_MAX]
        expected = [1]
        width = 1
        count = _MAX
        nbytes = _MAX
    elif fixture_id == 15:
        shape = native_matrix_shape(7, 0)
        expected = [4]
        count = 7
        nbytes = 28
        require(len(shape) == 1 and shape[0] == 7, "cols=0 denotes a vector")
    elif fixture_id == 16:
        shape = native_matrix_shape(0, 5)
        expected = [20, 4]
        count = 0
        nbytes = 0
        require(len(shape) == 2 and shape[0] == 0 and shape[1] == 5, "empty matrix shape")
    elif fixture_id == 17:
        shape = native_matrix_shape(7, 1)
        expected = [4, 4]
        count = 7
        nbytes = 28
        require(len(shape) == 2 and shape[0] == 7 and shape[1] == 1, "one-column matrix stays rank two")
    var original = shape.copy()
    var layout = NativeArrayLayout(shape.copy(), width, order.copy())
    require(layout.count == count and layout.nbytes == nbytes, "layout element/byte count")
    require(layout.itemsize == width and layout.order == order, "layout item width/order")
    require(len(layout.shape) == len(original) and len(layout.strides) == len(expected), "layout rank")
    for axis in range(len(original)):
        require(layout.shape[axis] == original[axis] and layout.strides[axis] == expected[axis], "literal shape/stride expectation")
    # Input metadata and a copied plan own independent Lists. Mutation must
    # not change the first plan's stored shape/strides or scalar counts.
    var other = layout.copy()
    if len(shape) > 0:
        shape[0] = 99
        other.shape[0] = 77
        other.strides[0] = 123
        require(layout.shape[0] == original[0] and layout.strides[0] == expected[0], "layout metadata copy aliases input")
    require(layout.count == count and layout.nbytes == nbytes, "layout copy changed scalar counts")


def _dtype(fixture_id: Int) raises:
    var names: List[String] = ["<f8", "<i8", "<f4", "<i4", "<u4", "<f2", "<u2", "<u1", "<i1"]
    var widths: List[Int] = [8, 8, 4, 4, 4, 2, 2, 1, 1]
    var codes: List[String] = ["d", "q", "f", "i", "I", "H", "H", "B", "b"]
    require(native_array_itemsize(names[fixture_id]) == widths[fixture_id], "supported dtype width")
    require(native_array_code(names[fixture_id]) == codes[fixture_id], "supported buffer format code")
    var shape: List[Int] = [2, 3]
    var layout = NativeArrayLayout(shape^, widths[fixture_id])
    require(layout.count == 6 and layout.nbytes == 6 * widths[fixture_id], "dtype layout byte count")


def _dtype_refusal(fixture_id: Int) raises:
    var names: List[String] = ["<u8", "<i2", ">f4", "f4", "|u1", "<c8"]
    var refused_width = False
    var refused_code = False
    try:
        _ = native_array_itemsize(names[fixture_id])
    except error:
        refused_width = String(error).find("is not an Array dtype") >= 0
    try:
        _ = native_array_code(names[fixture_id])
    except error:
        refused_code = String(error).find("is not an Array dtype") >= 0
    require(refused_width and refused_code, "unsupported dtype mapping admitted")


def _layout_refusal(fixture_id: Int) raises:
    var shape: List[Int] = [1]
    var order = String("C")
    var width = 1
    var expected = String()
    if fixture_id == 0 or fixture_id == 1:
        shape = [-1]
        if fixture_id == 1:
            shape = [0, -1]
        expected = "negative dimensions are not allowed"
    elif fixture_id == 2:
        shape = [_MAX, 2]
        expected = "element count exceeds native storage"
    elif fixture_id == 3:
        shape = [_MAX]
        width = 2
        expected = "byte count exceeds native storage"
    elif fixture_id == 4 or fixture_id == 5:
        shape = List[Int]()
        width = 0 if fixture_id == 4 else -1
        expected = "byte count exceeds native storage"
    elif fixture_id == 6 or fixture_id == 7:
        # Total count is zero, but the chosen traversal multiplies itemsize
        # by MAX before reaching the zero axis. This must refuse, not wrap.
        shape = [0, _MAX]
        width = 2
        if fixture_id == 7:
            shape = [_MAX, 0]
            order = "F"
        expected = "strides exceed native storage"
    elif fixture_id == 8 or fixture_id == 9:
        order = "K" if fixture_id == 8 else "f"
        expected = "array order must be 'C' or 'F'"
    else:
        # C overflow comes from two axes with unit item width. F traversal
        # follows the count loop's order, so its separate near-boundary case
        # needs item width 2 to overflow before reaching the zero axis.
        shape = [0, _MAX // 2 + 1, 2]
        if fixture_id == 11:
            order = "F"
            shape = [_MAX // 2 + 1, 0, 2]
            width = 2
        expected = "strides exceed native storage"
    var refused = False
    var message = String()
    try:
        var layout = NativeArrayLayout(shape^, width, order^)
        _ = layout
    except error:
        refused = True
        message = String(error)
    require(refused and message.find(expected) >= 0, "wrong/absent layout refusal: " + message)


def run_fixture(fixture_id: Int) raises:
    if fixture_id < 18:
        _layout(fixture_id)
    elif fixture_id < 27:
        _dtype(fixture_id - 18)
    elif fixture_id < 33:
        _dtype_refusal(fixture_id - 27)
    else:
        _layout_refusal(fixture_id - 33)


def main() raises:
    var failures = List[Int](length=ARRAY_LAYOUT_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(fixture_id: Int) {imm fp}:
        try:
            run_fixture(fixture_id)
        except error:
            print("FAIL native_array_layout fixture", fixture_id, String(error))
            fp.unsafe_store(fixture_id, 1)
    host_parallelize(task, ARRAY_LAYOUT_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "native_array_layout failed " + String(failed) + " fixtures")
    print("PASS native_array_layout:", ARRAY_LAYOUT_FIXTURES, "native shape/stride/dtype/overflow fixtures, parallel")
