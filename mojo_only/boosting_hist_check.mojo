"""Why does the boosting dataset always pick bin-feature 0?

`boosting_check` grows trees that split on bin-feature 0 at every level and
every iteration, while the mixed-tree path through the SAME score kernel
picks varied splits. So the fault is the data or the single-policy path, and
this takes the boosting dataset's exact shape and checks two things in order:

1. is the HISTOGRAM right, cell by cell, against a host tally
2. is bin-feature 0 really the argmax of the SCORE, computed on the host
   from that same histogram

If (1) fails the score kernel is innocent. If (1) passes and (2) says some
other bin-feature wins, the score kernel is wrong on this data. If both say
bin-feature 0, then the split is genuinely best and the model is right to be
a stump, which would make the problem the DATASET and not the port.

Hand-arithmetic guessed wrong twice on this question, which is why this
computes rather than argues.
"""

from max.gpu.host import DeviceContext

from ported.gpu_data.compressed_index_builder import build_layout
from ported.gpu_data.feature_blocks import blocks_for
from ported.gpu_data.kernel.binarize import (
    WRITE_BLOCK_SIZE,
    write_compressed_index_kernel,
)
from ported.methods.greedy_subsets_searcher.greedy_search_helper import (
    launch_histograms_for_blocks,
    upload_blocks,
)
from ported.methods.greedy_subsets_searcher.kernel.histogram_utils import (
    scan_histograms_kernel,
)
from mojo_only.kernel_matrix import replicas_for


def check_boosting_histogram(n_rows: Int = 8192, n_features: Int = 16, n_folds: Int = 15) raises:
    var ctx = DeviceContext()
    var stat_count = 2

    var folds = List[Int]()
    for _ in range(n_features):
        folds.append(n_folds)

    var lay = build_layout(folds)
    var blocks = blocks_for(lay, n_rows)
    var dblocks = upload_blocks(ctx, blocks)
    var n_bf = lay.hist_cells

    var cindex = ctx.enqueue_create_buffer[DType.uint32](n_rows * lay.columns)
    var z = ctx.enqueue_create_host_buffer[DType.uint32](n_rows * lay.columns)
    for i in range(n_rows * lay.columns):
        z.unsafe_ptr().unsafe_store(i, UInt32(0))
    ctx.enqueue_copy(dst_buf=cindex, src_ptr=z.unsafe_ptr())
    ctx.synchronize()

    var host_bin = List[List[Int]]()
    var hb = ctx.enqueue_create_host_buffer[DType.uint8](n_rows)
    var bins = ctx.enqueue_create_buffer[DType.uint8](n_rows)
    for f in range(n_features):
        var col = List[Int]()
        ref cf = lay.features[f]
        for r in range(n_rows):
            var x = UInt32(r * 2654435761 + f * 40503 + 0x2545F491)
            x ^= x << 13
            x ^= x >> 17
            x ^= x << 5
            var v = Int(x % UInt32(folds[f]))
            col.append(v)
            hb.unsafe_ptr().unsafe_store(r, UInt8(v))
        host_bin.append(col^)
        ctx.enqueue_copy(dst_buf=bins, src_ptr=hb.unsafe_ptr())
        ctx.enqueue_function[write_compressed_index_kernel](
            Int32(Int(cf.offset) * n_rows), cf.mask, cf.shift,
            bins.unsafe_ptr(), Int32(n_rows), cindex.unsafe_ptr(),
            grid_dim=(n_rows + WRITE_BLOCK_SIZE - 1) // WRITE_BLOCK_SIZE,
            block_dim=WRITE_BLOCK_SIZE,
        )
        ctx.synchronize()

    # Exactly the boosting check's stats at iteration 1: weight 1, gradient y.
    var stats = ctx.enqueue_create_buffer[DType.float32](stat_count * n_rows)
    var hs = ctx.enqueue_create_host_buffer[DType.float32](
        stat_count * n_rows
    )
    var host_y = List[Float64]()
    for r in range(n_rows):
        var y = (
            Float64(host_bin[0][r]) * 1.5
            - Float64(host_bin[3][r]) * 0.75
            + Float64(host_bin[7][r]) * 0.5
        )
        if host_bin[0][r] > 7 and host_bin[3][r] > 7:
            y += 4.0
        host_y.append(y)
        hs.unsafe_ptr().unsafe_store(r, Float32(1.0))
        hs.unsafe_ptr().unsafe_store(n_rows + r, Float32(y))
    ctx.enqueue_copy(dst_buf=stats, src_ptr=hs.unsafe_ptr())

    var row_index = ctx.enqueue_create_buffer[DType.uint32](n_rows)
    var hi = ctx.enqueue_create_host_buffer[DType.uint32](n_rows)
    for r in range(n_rows):
        hi.unsafe_ptr().unsafe_store(r, UInt32(r))
    ctx.enqueue_copy(dst_buf=row_index, src_ptr=hi.unsafe_ptr())

    var max_leaves = 2
    var p_off = ctx.enqueue_create_buffer[DType.uint32](max_leaves)
    var p_sz = ctx.enqueue_create_buffer[DType.uint32](max_leaves)
    var ids = ctx.enqueue_create_buffer[DType.uint32](max_leaves)
    var dense = ctx.enqueue_create_buffer[DType.uint32](max_leaves)
    var ho = ctx.enqueue_create_host_buffer[DType.uint32](max_leaves)
    for i in range(max_leaves):
        ho.unsafe_ptr().unsafe_store(i, UInt32(0))
    ctx.enqueue_copy(dst_buf=p_off, src_ptr=ho.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=ids, src_ptr=ho.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=dense, src_ptr=ho.unsafe_ptr())
    ho.unsafe_ptr().unsafe_store(0, UInt32(n_rows))
    ctx.enqueue_copy(dst_buf=p_sz, src_ptr=ho.unsafe_ptr())

    var hist_cells = max_leaves * stat_count * n_bf
    var hist = ctx.enqueue_create_buffer[DType.float32](hist_cells)
    var acc = ctx.enqueue_create_buffer[DType.int32](hist_cells)
    var zf = ctx.enqueue_create_host_buffer[DType.float32](hist_cells)
    var zi = ctx.enqueue_create_host_buffer[DType.int32](hist_cells)
    for i in range(hist_cells):
        zf.unsafe_ptr().unsafe_store(i, Float32(0.0))
        zi.unsafe_ptr().unsafe_store(i, Int32(0))
    ctx.enqueue_copy(dst_buf=hist, src_ptr=zf.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=acc, src_ptr=zi.unsafe_ptr())

    var widest = 1
    for b in range(len(blocks)):
        var tf = 0
        for k in range(blocks[b].count()):
            tf += Int(blocks[b].folds[k])
        if tf > widest:
            widest = tf
    var block_hist = ctx.enqueue_create_buffer[DType.float32](
        max_leaves * stat_count * widest
    )
    ctx.synchronize()

    launch_histograms_for_blocks(
        ctx, dblocks, 0, 1, n_rows, stat_count, max_leaves,
        1, Float32(1.0),
        cindex, row_index, stats, p_off, p_sz, ids, dense,
        hist, acc, block_hist, n_bf,
    )
    ctx.synchronize()

    # ---- 1. the RAW histogram, before the scan ----
    var raw = ctx.enqueue_create_host_buffer[DType.float32](hist_cells)
    ctx.enqueue_copy(dst_ptr=raw.unsafe_ptr(), src_buf=hist)
    ctx.synchronize()

    var want_w = List[Float64]()
    var want_g = List[Float64]()
    for _ in range(n_bf):
        want_w.append(0.0)
        want_g.append(0.0)
    for f in range(n_features):
        var base = Int(lay.features[f].first_fold_index)
        for r in range(n_rows):
            var b = host_bin[f][r]
            if b < folds[f]:
                want_w[base + b] += 1.0
                want_g[base + b] += host_y[r]

    var wrong = 0
    for c in range(n_bf):
        var gw = Float64(raw.unsafe_ptr().unsafe_load(c))
        var gg = Float64(raw.unsafe_ptr().unsafe_load(n_bf + c))
        if gw != want_w[c]:
            wrong += 1
        elif abs(gg - want_g[c]) > 0.05 * (abs(want_g[c]) + 1.0):
            wrong += 1
    print("    raw histogram cells wrong:", wrong, "of", n_bf)
    for c in range(6):
        print("      bf", c, "got w",
              raw.unsafe_ptr().unsafe_load(c), "want", want_w[c],
              "| got g", raw.unsafe_ptr().unsafe_load(n_bf + c),
              "want", want_g[c])

    # ---- 2. the SCORE, computed on the host from the host tally ----
    var total_w = Float64(n_rows)
    var total_g = Float64(0.0)
    for r in range(n_rows):
        total_g += host_y[r]

    var lam = Float64(3.0)
    var best_bf = -1
    var best_score = Float64(0.0)
    var score0 = Float64(0.0)
    for f in range(n_features):
        var base = Int(lay.features[f].first_fold_index)
        var pw = Float64(0.0)
        var pg = Float64(0.0)
        for b in range(folds[f]):
            pw += want_w[base + b]
            pg += want_g[base + b]
            var wr = total_w - pw
            var gr = total_g - pg
            var sc = pg * pg / (pw + lam) + gr * gr / (wr + lam)
            if base + b == 0:
                score0 = sc
            if sc > best_score:
                best_score = sc
                best_bf = base + b
    print("    host argmax: bin-feature", best_bf, "score", best_score)
    print("    bin-feature 0 score           ", score0)

    if wrong != 0:
        raise Error(
            String("the histogram is wrong on this dataset: ") + String(wrong)
            + " of " + String(n_bf) + " cells"
        )
    print("  histogram is correct on the boosting dataset")
