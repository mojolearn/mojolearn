# SPDX-License-Identifier: Apache-2.0
"""Small resident lifecycle/identity check, no performance claim."""
from max.gpu.host import DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE
from checks.vendor import COMPILED_VENDOR
from std.memory import bitcast
from core.forest_inference import forest_predict_gpu
from core.forest_inference_model import resident_prepare, resident_predict, resident_release, ResidentForest


def main() raises:
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
    for _ in range(3):
        var actual = resident_predict[True](h, x, 3, 2, 2)
        for i in range(6):
            if bitcast[DType.uint32](actual[i]) != bitcast[DType.uint32](reference[i]):
                raise Error("resident/transient output mismatch")
            if actual[i] != leaf[i % 2]:
                raise Error("resident independent stump oracle mismatch")
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
    # Exercise owned destruction without explicit close, as on interpreter exit.
    var automatic = ResidentForest(off, col, thr, left, leaf, 2, 2)
    _ = automatic^
    _ = ctx^
    print("RESIDENT_FOREST_PASS repeated exact/transient/stump, release/stale/newID, dimensions/finite/graph, destruction")
