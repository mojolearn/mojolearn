"""Read the flat histogram back for a MIXED dataset and check each slice.

The mixed tree conserves every row and refuses to split. Two inferences have
each found a real bug without finding this one, so this stops inferring.

It builds the depth-0 histogram for a dataset with all three policies and
compares EVERY policy's slice against a host count of the same bins. A slice
that is zero, or that holds another policy's numbers, names the failure
directly instead of leaving it to be deduced from leaf occupancy.
"""

from max.gpu.host import DeviceContext

from gbdt.gpu_data.compressed_index_builder import build_layout
from gbdt.gpu_data.feature_blocks import blocks_for
from gbdt.gpu_data.grid_policy import policy_name
from gbdt.gpu_data.kernel.binarize import (
    WRITE_BLOCK_SIZE,
    write_compressed_index_kernel,
)
from gbdt.methods.greedy_subsets_searcher.greedy_search_helper import (
    launch_histograms_for_blocks,
    upload_blocks,
)


def probe_mixed_histogram(binary: Int = 8, half: Int = 4, one: Int = 4, pre_bridge: Bool = False) raises:
    var ctx = DeviceContext()
    var n_rows = 2048
    var stat_count = 2

    var folds = List[Int]()
    for _ in range(binary):
        folds.append(1)
    for _ in range(half):
        folds.append(8)
    for _ in range(one):
        folds.append(64)
    var n_features = len(folds)

    var lay = build_layout(folds)
    var blocks = blocks_for(lay, n_rows)
    var dblocks = upload_blocks(ctx, blocks)

    var n_columns = lay.columns
    var cindex = ctx.enqueue_create_buffer[DType.uint32](n_rows * n_columns)
    var z = ctx.enqueue_create_host_buffer[DType.uint32](n_rows * n_columns)
    for i in range(n_rows * n_columns):
        z.unsafe_ptr().unsafe_store(i, UInt32(0))
    ctx.enqueue_copy(dst_buf=cindex, src_ptr=z.unsafe_ptr())
    ctx.synchronize()

    # Host copy of every row's bin per feature, so the expected counts are a
    # plain tally rather than a second derivation.
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

    var stats = ctx.enqueue_create_buffer[DType.float32](stat_count * n_rows)
    var hs = ctx.enqueue_create_host_buffer[DType.float32](
        stat_count * n_rows
    )
    for r in range(n_rows):
        hs.unsafe_ptr().unsafe_store(r, Float32(1.0))
        hs.unsafe_ptr().unsafe_store(n_rows + r, Float32(1.0))
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
    var ho = ctx.enqueue_create_host_buffer[DType.uint32](max_leaves)
    var hz2 = ctx.enqueue_create_host_buffer[DType.uint32](max_leaves)
    var hid = ctx.enqueue_create_host_buffer[DType.uint32](max_leaves)
    for i in range(max_leaves):
        ho.unsafe_ptr().unsafe_store(i, UInt32(0))
        hz2.unsafe_ptr().unsafe_store(i, UInt32(0))
        hid.unsafe_ptr().unsafe_store(i, UInt32(i))
    hz2.unsafe_ptr().unsafe_store(0, UInt32(n_rows))
    ctx.enqueue_copy(dst_buf=p_off, src_ptr=ho.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=p_sz, src_ptr=hz2.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=ids, src_ptr=hid.unsafe_ptr())

    var hist_cells = max_leaves * stat_count * lay.hist_cells
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

    # DENSE ids for the block scratch, `0..max_leaves`.
    var dense_ids = ctx.enqueue_create_buffer[DType.uint32](max_leaves)
    var h_dense = ctx.enqueue_create_host_buffer[DType.uint32](max_leaves)
    for i in range(max_leaves):
        h_dense.unsafe_ptr().unsafe_store(i, UInt32(i))
    ctx.enqueue_copy(dst_buf=dense_ids, src_ptr=h_dense.unsafe_ptr())
    ctx.synchronize()

    launch_histograms_for_blocks(
        ctx, dblocks, 0, 1, n_rows, stat_count, max_leaves, 1, Float32(1.0),
        cindex, row_index, stats, p_off, p_sz, ids, dense_ids, hist, acc,
        block_hist, lay.hist_cells, pre_bridge,
    )
    ctx.synchronize()

    # PRE-BRIDGE reads what the KERNEL wrote, in the per-block layout.
    # POST-BRIDGE reads the flat histogram the scorer sees. Comparing the two
    # separates a bad accumulation from a bad scatter.
    var out = ctx.enqueue_create_host_buffer[DType.float32](hist_cells)
    # PRE-BRIDGE truth lives in TWO places since the writebacks routed
    # every i32 one-byte cell through the accumulator (single-block
    # included) and the scratch went dead on that arm: the accumulator
    # where nonzero, the float scratch otherwise -- the same two-source
    # contract write_reduces_from_fixed_kernel documents. This probe
    # runs at fixed_scale = 1.0, so a raw accumulator cell IS the count.
    var out_acc = ctx.enqueue_create_host_buffer[DType.int32](hist_cells)
    if pre_bridge:
        ctx.enqueue_copy(dst_ptr=out.unsafe_ptr(), src_buf=block_hist)
        ctx.enqueue_copy(dst_ptr=out_acc.unsafe_ptr(), src_buf=acc)
    else:
        ctx.enqueue_copy(dst_ptr=out.unsafe_ptr(), src_buf=hist)
    ctx.synchronize()
    if pre_bridge:
        for i in range(hist_cells):
            var q = out_acc.unsafe_ptr().unsafe_load(i)
            if q != Int32(0):
                out.unsafe_ptr().unsafe_store(i, Float32(Int(q)))

    # Leaf 0, stat 0 (the weight plane): cell for (feature f, bin b) sits at
    # first_fold_index[f] + b in the flat array.
    print(
        "  reading",
        "BLOCK scratch (pre-bridge)" if pre_bridge else "FLAT (post-bridge)",
    )
    print("  feature  policy            bin0 device / host   status")
    var wrong = 0
    for f in range(n_features):
        ref cf = lay.features[f]
        var want = 0
        for r in range(n_rows):
            if host_bin[f][r] == 0:
                want += 1
        # Post-bridge: the flat index. Pre-bridge: the within-block index,
        # which for a single-policy probe is the same running total.
        var got = out.unsafe_ptr().unsafe_load(Int(cf.first_fold_index))
        var ok = abs(got - Float32(want)) < Float32(1e-3)
        if not ok:
            wrong += 1
        print(
            "   ", f, "   ", policy_name(lay.policy_of[f]),
            "   ", got, "/", want, "   ", "ok" if ok else "WRONG",
        )
    print("  wrong slices:", wrong, "of", n_features)
    if wrong != 0:
        raise Error("the mixed histogram is wrong")
    print("  every policy's slice lands in the right place")
