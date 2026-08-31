# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Where does an epsilon tree's ~200 ms live? Fit t = a + b*R + c*F + d*R*F.

    pixi run -e bench python tools/shape_slices.py     # once
    pixi run -e bench mojo run -I . mojo_only/shape_sweep.mojo <fixture_dir>

The EPSILON_2026-08-21 row gave one attribution bit free: 254 borders cost
only ~4% more than 128, so the CELL-scaled phases (zero, scan, score,
convert -- everything proportional to bins) are small at this shape. This
sweep pins the rest by measurement instead of reasoning: five (rows,
feats) prefix-slices of the SAME fixture, one process, configs alternated
round-robin inside each rep so no config owns a thermal window. The
decomposition then falls out host-side:

    b (rows)        reorder, partition, per-row gradient work
    c (feats)       per-level per-feature passes independent of rows
    d (rows*feats)  the histogram accumulate's bin-matrix read traffic
    a               the per-tree launch/drain floor (known ~9-10 ms here)

OUR ARM ONLY -- this is attribution, not comparison, so no CatBoost arm
and no cross-arm claim. Absolute ms/tree here are NOT quotable against
anything outside this process; only the fitted structure is the result.
RMSE, depth 6, 20 trees, the pinned defaults of the interleaved harness.
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
comptime N_CONFIGS = 8


def read_all(path: String) raises -> List[UInt8]:
    var f = open(path, "r")
    var b = f.read_bytes()
    f.close()
    return b^


def build_cindex(
    ctx: DeviceContext,
    folds: List[Int],
    bins: List[UInt8],
    n_rows: Int,
) raises -> DeviceBuffer[DType.uint32]:
    var lay = build_layout(folds)
    var cindex = ctx.enqueue_create_buffer[DType.uint32](n_rows * lay.columns)
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
    return cindex^


def main() raises:
    var args = argv()
    if len(args) < 2:
        raise Error("usage: ... <fixture_dir>")
    var d = String(args[1])
    var ctx = DeviceContext()

    var stems = List[String]()
    var rows = List[Int]()
    stems.append(String("eps100k_500f"))
    rows.append(100_000)
    stems.append(String("eps100k_2000f"))
    rows.append(100_000)
    stems.append(String("eps400k_500f"))
    rows.append(400_000)
    stems.append(String("eps200k_1000f"))
    rows.append(200_000)
    stems.append(String("epsilon"))
    rows.append(400_000)
    # The subtraction reach check: same full-shape fixture, sibling
    # subtraction OFF. If this arm is not ~2x the d-term slower, the
    # switch is dead exactly where 83% of the tree's time goes.
    stems.append(String("epsilon"))
    rows.append(400_000)
    # The depth sweep: full shape at depths 2 and 4 beside the default 6.
    # Differencing gives per-level cost. Subtraction halves the ROWS built
    # per level as depth grows; if per-level cost does NOT fall to match,
    # the indexed reads' cache-line density is collapsing as leaves
    # shrink -- the read-density story, measured rather than assumed.
    stems.append(String("epsilon"))
    rows.append(400_000)
    stems.append(String("epsilon"))
    rows.append(400_000)

    var subtraction = List[Bool]()
    var depths = List[Int]()
    for i in range(N_CONFIGS):
        subtraction.append(i != 5)
        depths.append(DEPTH)
    depths[6] = 2
    depths[7] = 4

    var all_folds = List[List[Int]]()
    var cindexes = List[DeviceBuffer[DType.uint32]]()
    var targets = List[DeviceBuffer[DType.float32]]()
    var weights = List[DeviceBuffer[DType.float32]]()

    for i in range(N_CONFIGS):
        if i > 4:
            break  # configs 5..7 share the full-shape buffers of config 4
        var folds = List[Int]()
        var f = open(d + "/" + stems[i] + "_folds_128.txt", "r")
        var txt = f.read()
        f.close()
        for line in txt.split("\n"):
            if line.byte_length() > 0:
                folds.append(Int(line))
        var bins = read_all(d + "/" + stems[i] + "_bins_128.u8")
        cindexes.append(build_cindex(ctx, folds, bins, rows[i]))

        var ybytes = read_all(d + "/" + stems[i] + "_y.f32")
        var yp = ybytes.unsafe_ptr().unsafe_bitcast[Float32]()
        var t = ctx.enqueue_create_buffer[DType.float32](rows[i])
        var w = ctx.enqueue_create_buffer[DType.float32](rows[i])
        var ht = ctx.enqueue_create_host_buffer[DType.float32](rows[i])
        var hw = ctx.enqueue_create_host_buffer[DType.float32](rows[i])
        for r in range(rows[i]):
            ht.unsafe_ptr().unsafe_store(r, yp[r])
            hw.unsafe_ptr().unsafe_store(r, Float32(1.0))
        ctx.enqueue_copy(dst_buf=t, src_ptr=ht.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=w, src_ptr=hw.unsafe_ptr())
        ctx.synchronize()
        targets.append(t^)
        weights.append(w^)
        all_folds.append(folds^)
        print("built", stems[i], ":", rows[i], "rows x",
              len(all_folds[i]), "feats")

    # warm-up, every config once, untimed
    for i in range(N_CONFIGS):
        var bi = min(i, 4)
        var wm = TAdditiveModel()
        _ = fit(
            wm, ctx, rows[i], all_folds[bi], depths[i], cindexes[bi],
            targets[bi], weights[bi], False, 2, Float32(0.3), Float32(3.0),
            subtraction[i],
        )

    for rep in range(REPS):
        for i in range(N_CONFIGS):
            var bi = min(i, 4)
            var model = TAdditiveModel()
            var t0 = perf_counter_ns()
            var losses = fit(
                model, ctx, rows[i], all_folds[bi], depths[i], cindexes[bi],
                targets[bi], weights[bi], False, TREES, Float32(0.3),
                Float32(3.0), subtraction[i],
            )
            var ms = Float64(perf_counter_ns() - t0) / 1e6 / Float64(TREES)
            print(
                "rep", rep, " ", stems[i], " rows", rows[i], " feats",
                len(all_folds[bi]), " subtraction", subtraction[i],
                " depth", depths[i], " ms/tree", ms, " mse",
                losses[len(losses) - 1],
            )
