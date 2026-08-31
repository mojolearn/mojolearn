# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The HONEST row: raw floats to fitted model, quantization inside, both arms.

    pixi run -e bench mojo run -I . \
        bench/interleaved/end2end_interleaved.mojo <dir> <prefix> <rows> <feats>

Every other interleaved row quantizes OUTSIDE the timed region on both
arms -- the right protocol for comparing learners, and the one their own
model_evaluation methodology uses. This row is the one a USER
experiences: `train(X, y)` from raw floats against `catboost.Pool(X, y)`
+ `fit`, everything inside the timer on both sides, arms alternating in
one process. It exists because the prep-bill work
(`PREP_BILL_2026-08-21.md`) took our side of it from 24 s to ~7 s at
400k x 500, and a claim about the user-facing path needs the user-facing
measurement.

DATASET-GENERAL BY CONSTRUCTION: the harness takes any (prefix, rows,
feats) whose `<prefix>_Xcol.f32` (col-major raw floats) and
`<prefix>_y.f32` sit in the fixture dir. Nothing in either arm reads a
shape constant from anywhere but the arguments -- run it on epsilon,
covtype, synth alike; the standing rule is that optimizations are
structural and datasets are measurement vehicles.

Loss columns are each arm's train mse, comparable as NUMBERS since both
arms train the pinned RMSE settings on the same bytes; grids are derived
independently by each library (that is the point of the row), so mse
matches to a quality band rather than bit-for-bit.
"""
from max.gpu.host import DeviceContext
from std.python import Python, PythonObject
from std.time import perf_counter_ns

from gbdt.train import train

from std.sys import argv

comptime DEPTH = 6
comptime TREES = 20
comptime REPS = 3


def read_all(path: String) raises -> List[UInt8]:
    var f = open(path, "r")
    var b = f.read_bytes()
    f.close()
    return b^


def main() raises:
    var args = argv()
    if len(args) < 5:
        raise Error(
            "usage: ... <dir> <prefix> <rows> <feats>"
        )
    var d = String(args[1])
    var prefix = String(args[2])
    var n_rows = Int(String(args[3]))
    var n_feats = Int(String(args[4]))
    var ctx = DeviceContext()
    var sys = Python.import_module("sys")
    _ = sys.path.append("tools")
    var arm = Python.import_module("catboost_end2end_arm")

    var xbytes = read_all(d + "/" + prefix + "_Xcol.f32")
    if len(xbytes) != 4 * n_rows * n_feats:
        raise Error(prefix + "_Xcol.f32 missing or wrong size")
    var xp = xbytes.unsafe_ptr().unsafe_bitcast[Float32]()
    var ybytes = read_all(d + "/" + prefix + "_y.f32")
    var yp = ybytes.unsafe_ptr().unsafe_bitcast[Float32]()

    var x_colmajor = List[Float32](capacity=n_rows * n_feats)
    for i in range(n_rows * n_feats):
        x_colmajor.append(xp[i])
    var y = List[Float32](capacity=n_rows)
    for r in range(n_rows):
        y.append(yp[r])

    print(
        "end-to-end row:", prefix, n_rows, "x", n_feats,
        "@128, depth", DEPTH, ",", TREES,
        "trees, RAW FLOATS TO MODEL both arms, reps interleaved",
    )

    # warm-up, both arms, untimed
    _ = arm.fit_seconds_end2end(d, prefix, n_rows, n_feats, 128, 2, DEPTH)
    var wm = train(
        ctx, x_colmajor, y, n_rows, n_feats,
        border_count=128, n_estimators=2, max_depth=DEPTH,
        learning_rate=Float32(0.3), l2_leaf_reg=Float32(3.0),
    )
    _ = wm^

    for rep in range(REPS):
        var res = arm.fit_seconds_end2end(
            d, prefix, n_rows, n_feats, 128, TREES, DEPTH
        )
        var theirs_s = res[0].__float__()
        var their_mse = res[1].__float__()

        var t0 = perf_counter_ns()
        var tm = train(
            ctx, x_colmajor, y, n_rows, n_feats,
            border_count=128, n_estimators=TREES, max_depth=DEPTH,
            learning_rate=Float32(0.3), l2_leaf_reg=Float32(3.0),
        )
        var ours_s = Float64(perf_counter_ns() - t0) / 1e9
        var our_mse = tm.losses[len(tm.losses) - 1]
        _ = tm^
        print(
            "  rep", rep, " catboost-cpu", theirs_s,
            "s   ours-gpu", ours_s, "s   speedup", theirs_s / ours_s,
            "x", "  our mse", our_mse, " catboost mse", their_mse,
        )
