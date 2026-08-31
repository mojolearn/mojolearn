# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""What does Bayesian sampling cost OUR arm per tree? Interleaved, one process.

    pixi run -e bench mojo run -I . \
        checks/bootstrap_delta_probe.mojo <fixture_dir>

The 2026-08-21 epsilon rows read deterministic ~151 ms/tree and Bayesian
~200-234 -- an apparent +45-70 ms for a sampling step whose kernel is one
elementwise pass plus a one-block reduce (a few ms at most). But those
numbers came from DIFFERENT RUNS in DIFFERENT thermal windows (the
Bayesian run started on a box the Logloss run had just heated, and
climbed 200 -> 234 across its own reps), which is exactly the cross-time
A/B rule 7 forbids. This probe alternates the two arms inside one
process on the same fixtures, same cindex, same buffers:

    rep k:  fit(deterministic)  then  fit(bayesian, seed = k)

If the interleaved delta is a few ms, the anomaly was the window and
this file is the negative result. If it survives at tens of ms, the
next step is bisection: price `launch_bootstrap` alone, then the
estimator-under-bootstrap arm (RMSE Newton-1 skips the estimator only
WITHOUT bootstrap -- sampling shapes structure only, so the plain-weight
re-estimate must run), then tree-shape drift (reweighted splits change
partition balance, and subtraction cost follows the smaller-half sizes).

OUR ARM ONLY; no CatBoost, no cross-arm claim. epsilon at 128 borders,
20 trees depth 6, the pinned settings.
"""
from max.gpu.host import DeviceBuffer, DeviceContext
from std.time import perf_counter_ns

from gbdt.gpu_data.compressed_index_builder import build_layout
from gbdt.gpu_data.kernel.binarize import (
    WRITE_BLOCK_SIZE,
    write_compressed_index_kernel,
)
from gbdt.methods.doc_parallel_boosting import TAdditiveModel, fit

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
    if len(args) < 2:
        raise Error("usage: ... <fixture_dir>")
    var d = String(args[1])
    var n_rows = 400_000
    var ctx = DeviceContext()

    var folds = List[Int]()
    var f = open(d + "/epsilon_folds_128.txt", "r")
    var txt = f.read()
    f.close()
    for line in txt.split("\n"):
        if line.byte_length() > 0:
            folds.append(Int(line))
    var bins = read_all(d + "/epsilon_bins_128.u8")
    var ybytes = read_all(d + "/epsilon_y.f32")

    var lay = build_layout(folds)
    var cindex = ctx.enqueue_create_buffer[DType.uint32](
        n_rows * lay.columns
    )
    ctx.enqueue_memset(cindex, UInt32(0))
    var hb = ctx.enqueue_create_host_buffer[DType.uint8](n_rows)
    var dbins = ctx.enqueue_create_buffer[DType.uint8](n_rows)
    for feat in range(len(folds)):
        ref cf = lay.features[feat]
        var base = feat * n_rows
        for r in range(n_rows):
            hb.unsafe_ptr().unsafe_store(r, bins[base + r])
        ctx.enqueue_copy(dst_buf=dbins, src_ptr=hb.unsafe_ptr())
        ctx.enqueue_function[write_compressed_index_kernel](
            Int32(Int(cf.offset) * n_rows), cf.mask, cf.shift,
            dbins.unsafe_ptr(), Int32(n_rows), cindex.unsafe_ptr(),
            grid_dim=(n_rows + WRITE_BLOCK_SIZE - 1) // WRITE_BLOCK_SIZE,
            block_dim=(WRITE_BLOCK_SIZE, 1, 1),
        )
        ctx.synchronize()

    var targets = ctx.enqueue_create_buffer[DType.float32](n_rows)
    var weights = ctx.enqueue_create_buffer[DType.float32](n_rows)
    var ht = ctx.enqueue_create_host_buffer[DType.float32](n_rows)
    var hw = ctx.enqueue_create_host_buffer[DType.float32](n_rows)
    var yp = ybytes.unsafe_ptr().unsafe_bitcast[Float32]()
    for r in range(n_rows):
        ht.unsafe_ptr().unsafe_store(r, yp[r])
        hw.unsafe_ptr().unsafe_store(r, Float32(1.0))
    ctx.enqueue_copy(dst_buf=targets, src_ptr=ht.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=weights, src_ptr=hw.unsafe_ptr())
    ctx.synchronize()

    print(
        "bootstrap delta probe:", n_rows, "rows x", len(folds),
        "feats @128, depth", DEPTH, ",", TREES,
        "trees, arms alternated in one process",
    )

    # warm-up, both arms, untimed
    for arm in range(2):
        var wm = TAdditiveModel()
        _ = fit(
            wm, ctx, n_rows, folds, DEPTH, cindex, targets, weights,
            False, 2, Float32(0.3), Float32(3.0), True,
            bootstrap_bayesian=(arm == 1),
            bagging_temperature=Float32(1.0),
        )

    for rep in range(REPS):
        for arm in range(2):
            var model = TAdditiveModel()
            var t0 = perf_counter_ns()
            var losses = fit(
                model, ctx, n_rows, folds, DEPTH, cindex, targets,
                weights, False, TREES, Float32(0.3), Float32(3.0), True,
                bootstrap_bayesian=(arm == 1),
                bagging_temperature=Float32(1.0),
                random_seed=UInt64(rep),
            )
            var ms = Float64(perf_counter_ns() - t0) / 1e6 / Float64(
                TREES
            )
            print(
                "rep", rep,
                "  bayesian" if arm == 1 else "  deterministic",
                " ms/tree", ms, " mse", losses[len(losses) - 1],
            )
