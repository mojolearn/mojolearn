# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""GradientBoosting binary probabilities and class codes against a float64 oracle (lane py2mojo-C, 2026-11-22;
ported from `checks/gbdt_binary_prediction_smoke.py`).

The margins are the original's: 257 evenly spaced values in [-7, 7] (np.linspace in float32: every value is exact,
the step is 14/256), then 0, -0, the two smallest subnormals, +-100 and +-Float32.max. The entry is the one the
public adapters' binary predict reaches: on the host column `gbdt_host_binary_margins_check` and
`gbdt_host_binary_prediction_into` (gbdt/host/gbdt_host_entries_impl.mojo, the host image's binary op), on the
device column `binary_prediction_device` (gbdt/binary_prediction.mojo, the GPU image's binary op). Assertions are
the original's: p[:, 1] against 1 / (1 + exp(-clip(raw, -700, 700))) in float64 to atol 2e-7, rtol 2e-6; the codes
are strict raw > 0 (+0, -0 and the negative subnormal are class 0, the positive subnormal class 1); each row sums
to 1 to atol 1e-7; every probability finite and in [0, 1]. The binding's numeric-mode readback and the dtype of
the returned NumPy array stay in Python (public boundary). The two entries are one call each over 265 margins
(each splits its elements across the host tasks or device threads); nothing else to split.
"""
from std.math import exp, isfinite, min, max
from std.memory import bitcast
from std.sys.compile import is_defined
from std.testing import TestSuite
from gbdt.host.gbdt_host_entries_impl import gbdt_host_binary_margins_check, gbdt_host_binary_prediction_into
from gbdt.binary_prediction import binary_prediction_device

comptime N = 265


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def margins() -> List[Float32]:
    var raw = List[Float32](length=N, fill=Float32(0))
    for i in range(257):  # small-loop(np.linspace(-7, 7, 257, dtype=float32): exact words)
        raw[i] = Float32(-7.0 + Float64(i) * (14.0 / 256.0))
    var tiny = bitcast[DType.float32](UInt32(1))
    var big = bitcast[DType.float32](UInt32(0x7F7FFFFF))
    var tail: List[Float32] = [0.0, -0.0, tiny, -tiny, 100.0, -100.0, big, -big]
    for i in range(8):  # small-loop(the eight edge margins)
        raw[257 + i] = tail[i]
    return raw^


def check(raw: List[Float32], p: List[Float32], c: List[Int32]) raises:
    for i in range(N):  # small-loop(oracle: 265 margins)
        var r = Float64(raw[i])
        var clipped = min(max(r, -700.0), 700.0)
        var want = 1.0 / (1.0 + exp(-clipped))
        var got = Float64(p[2 * i + 1])
        var label = "margin " + String(i) + " (" + String(raw[i]) + ")"
        require(abs(got - want) <= 2e-7 + 2e-6 * abs(want), label + ": p " + String(got) + " expected " + String(want))
        require(c[i] == (Int32(1) if raw[i] > Float32(0) else Int32(0)), label + ": class " + String(c[i]))
        var lo = p[2 * i]
        var hi = p[2 * i + 1]
        require(abs(Float64(lo) + Float64(hi) - 1.0) <= 1e-7, label + ": row sum " + String(Float64(lo) + Float64(hi)))
        require(isfinite(lo) and isfinite(hi), label + ": nonfinite probability")
        require(lo >= Float32(0) and lo <= Float32(1) and hi >= Float32(0) and hi <= Float32(1),
                label + ": probability outside [0, 1]")


def test_binary_probabilities_and_classes() raises:
    var raw = margins()
    var p = List[Float32](length=2 * N, fill=Float32(-1))
    var c = List[Int32](length=N, fill=Int32(-1))
    comptime if is_defined["MOJOLEARN_COLUMN_CPU"]():
        var rp = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(raw.unsafe_ptr()))
        gbdt_host_binary_margins_check(rp, N)
        gbdt_host_binary_prediction_into[True, DType.float32](
            rp, MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(p.unsafe_ptr())), N)
        gbdt_host_binary_prediction_into[False, DType.int32](
            rp, MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(c.unsafe_ptr())), N)
    else:
        binary_prediction_device[True, DType.float32](Int(raw.unsafe_ptr()), Int(p.unsafe_ptr()), N)
        binary_prediction_device[False, DType.int32](Int(raw.unsafe_ptr()), Int(c.unsafe_ptr()), N)
    check(raw, p, c)
    _ = raw^
    _ = p^
    _ = c^


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
