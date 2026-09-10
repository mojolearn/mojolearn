# SPDX-License-Identifier: Apache-2.0
"""Small resident lifecycle/identity check, no performance claim."""
from max.gpu.host import DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE
from checks.vendor import COMPILED_VENDOR
from std.memory import bitcast
from core.forest_inference import forest_predict_gpu
from core.forest_inference_model import resident_prepare, resident_predict, resident_release, ResidentForest, resident_predict_into


def check_workspace[RF_INPUT: Bool]() raises:
    var offsets: List[Int32] = [0, 3]
    var columns: List[Int32] = [0, -1, -1]
    var thresholds: List[Float32] = [2, 0, 0]
    var left: List[Int32] = [1, -1, -1]
    var leaves: List[Float32] = [0, 0, 0.25, 0.75, 0.75, 0.25]
    var model = ResidentForest(offsets, columns, thresholds, left, leaves, 2, 2)
    var sizes: List[Int] = [3, 3, 1, 5, 0, 5, 2]
    for step in range(len(sizes)):
        var rows = sizes[step]
        var x = List[Float32](length=rows * 2, fill=Float32(0))
        for row in range(rows):
            # Change values even when the shape is unchanged: stale input or
            # output must fail the independent split oracle.
            x[row * 2] = Float32(1 if (row + step) % 2 == 0 else 3)
        var actual = List[Float32](length=rows * 2, fill=Float32(-9))
        var reference = List[Float32](length=rows * 2, fill=Float32(-8))
        var previous_rows = model.workspace_rows
        model.predict_into[RF_INPUT](x.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
            reference.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), rows, 2, 2, False)
        if model.workspace_rows != previous_rows:
            raise Error("uncached prediction changed workspace")
        if rows > 0 and previous_rows == rows:
            var input_address = model.input_workspace.value().unsafe_ptr()
            var output_address = model.output_workspace.value().unsafe_ptr()
            model.predict_into[RF_INPUT](x.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
                actual.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), rows, 2, 2, True)
            if input_address != model.input_workspace.value().unsafe_ptr() or output_address != model.output_workspace.value().unsafe_ptr():
                raise Error("same-shape workspace was not reused")
        else:
            model.predict_into[RF_INPUT](x.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
                actual.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), rows, 2, 2, True)
        if model.workspace_rows != (rows if rows > 0 else previous_rows):
            raise Error("workspace resize/empty lifecycle mismatch")
        for row in range(rows):
            for output in range(2):
                var expected = Float32(0.25 if (row + step + output) % 2 == 0 else 0.75)
                var i = row * 2 + output
                if actual[i] != expected or bitcast[DType.uint32](actual[i]) != bitcast[DType.uint32](reference[i]):
                    raise Error("workspace stale data or reference mismatch")
    model.close()
    if model.input_workspace or model.output_workspace or model.workspace_rows != 0:
        raise Error("close retained workspace")
    print("WORKSPACE_PASS RF_INPUT", RF_INPUT, "cached/uncached resize/reuse/empty/changed-input/close")


def main() raises:
    check_workspace[True]()
    check_workspace[False]()
    print("RESIDENT_MODE", Int(GLOBAL_NUMERIC_MODE), "VENDOR", String(COMPILED_VENDOR))
    var off: List[Int32] = [0, 1]
    var col: List[Int32] = [-1]
    var thr: List[Float32] = [0]
    var left: List[Int32] = [-1]
    var leaf: List[Float32] = [0.25, 0.75]
    var x: List[Float32] = [1, 2, 3, 4, 5, 6]
    var h = resident_prepare[True](off, col, thr, left, leaf, 2, 2)
    var ctx = DeviceContext()
    var reference = forest_predict_gpu[True, True](ctx, off, col, thr, left, leaf, x, 3, 2, 2)
    for repeat in range(4):
        var direct = List[Float32](length=6, fill=Float32(-9))
        resident_predict_into[True](h,
            x.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
            direct.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), 3, 2, 2, repeat % 2 == 1)
        var actual = resident_predict[True](h, x, 3, 2, 2)
        for i in range(6):
            if bitcast[DType.uint32](direct[i]) != bitcast[DType.uint32](actual[i]):
                raise Error("borrowed-pointer/List output mismatch")
            if bitcast[DType.uint32](actual[i]) != bitcast[DType.uint32](reference[i]):
                raise Error("resident/transient output mismatch")
            if actual[i] != leaf[i % 2]:
                raise Error("resident independent stump oracle mismatch")
    # A real split makes input upload/addressing observable, unlike a stump.
    var split_off: List[Int32] = [0, 3]
    var split_col: List[Int32] = [0, -1, -1]
    var split_thr: List[Float32] = [2, 0, 0]
    var split_left: List[Int32] = [1, -1, -1]
    var split_leaf: List[Float32] = [0, 0, 0.25, 0.75, 0.75, 0.25]
    var split_handle = resident_prepare[True](split_off, split_col, split_thr,
        split_left, split_leaf, 2, 2)
    var split_out = List[Float32](length=6, fill=Float32(-9))
    resident_predict_into[True](split_handle,
        x.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
        split_out.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), 3, 2, 2)
    var split_reference = resident_predict[True](split_handle, x, 3, 2, 2)
    var split_expected: List[Float32] = [0.25, 0.75, 0.75, 0.25, 0.75, 0.25]
    for i in range(6):
        if split_out[i] != split_expected[i] or split_out[i] != split_reference[i]:
            raise Error("borrowed split input/output addressing mismatch")
    resident_release[True](split_handle)
    var rejected = False
    try:
        var unused = resident_predict[True](h, x, 3, 1, 2)
    except:
        rejected = True
    if not rejected:
        raise Error("resident dimension mismatch accepted")
    var bad = x.copy()
    bad[0] = bitcast[DType.float32](UInt32(0x7f800000))
    rejected = False
    try:
        var unused = resident_predict[True](h, bad, 3, 2, 2)
    except:
        rejected = True
    if not rejected:
        raise Error("nonfinite resident input accepted")
    var untouched = List[Float32](length=6, fill=Float32(-9))
    rejected = False
    try:
        resident_predict_into[True](h,
            bad.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
            untouched.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), 3, 2, 2, True)
    except:
        rejected = True
    if not rejected:
        raise Error("nonfinite borrowed input accepted")
    for value in untouched:
        if value != -9:
            raise Error("borrowed invalid input modified output before validation")
    resident_release[True](h)
    rejected = False
    try:
        var unused = resident_predict[True](h, x, 3, 2, 2)
    except:
        rejected = True
    if not rejected:
        raise Error("released resident handle accepted")
    rejected = False
    try:
        resident_release[True](h)
    except:
        rejected = True
    if not rejected:
        raise Error("double release accepted")
    var next = resident_prepare[True](off, col, thr, left, leaf, 2, 2)
    if h == next:
        raise Error("resident handle reused")
    resident_release[True](next)
    var bad_left: List[Int32] = [1]
    rejected = False
    try:
        var unused = resident_prepare[True](off, col, thr, bad_left, leaf, 2, 2)
    except:
        rejected = True
    if not rejected:
        raise Error("invalid resident graph accepted")
    # Finite leaves can overflow the fixed reduction. Direct output is written
    # before validation, but the public caller must receive an exception.
    var overflow_off: List[Int32] = [0, 1, 2]
    var overflow_col: List[Int32] = [-1, -1]
    var overflow_thr: List[Float32] = [0, 0]
    var overflow_left: List[Int32] = [-1, -1]
    var maximum = bitcast[DType.float32](UInt32(0x7f7fffff))
    var overflow_leaf: List[Float32] = [maximum, maximum]
    var overflow_handle = resident_prepare[True](overflow_off, overflow_col,
        overflow_thr, overflow_left, overflow_leaf, 2, 1)
    var overflow_out = List[Float32](length=3, fill=Float32(0))
    rejected = False
    try:
        resident_predict_into[True](overflow_handle,
            x.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
            overflow_out.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), 3, 2, 1, True)
    except:
        rejected = True
    if not rejected:
        raise Error("nonfinite borrowed output accepted")
    resident_release[True](overflow_handle)
    # Exercise owned destruction without explicit close, as on interpreter exit.
    var automatic = ResidentForest(off, col, thr, left, leaf, 2, 2)
    _ = automatic^
    _ = ctx^
    print("RESIDENT_FOREST_PASS repeated borrowed/List/transient/stump, release/stale/newID, dimensions/finite input+output/graph, destruction")
