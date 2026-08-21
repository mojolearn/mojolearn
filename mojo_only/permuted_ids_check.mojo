"""Build histograms for a NON-CONTIGUOUS list of leaves and check every cell.

Every other histogram check in this tree passes leaf ids `0, 1, 2, ...`. That
is the identity permutation, and under the identity a DENSE index and a
LOOKED-UP index are the same number, so none of those checks can see the
difference between the two. The sibling subtraction is the first caller that
needs a non-identity list, which is why this check exists.

Their split, from `compute_hist_loop_two_stats.cuh`:

    const int partId = partIds[blockIdx.y];    // :511  READ the rows here
    ...
    hist.AddToGlobalMemory(..., blockIdx.y, ...);   // :554  WRITE here

so the partition is looked up and the scratch slot is DENSE, and
`WriteReducesHistogramsImpl` then reads the scratch at `blockIdx.y` and
writes the flat histogram at `histogramIds[blockIdx.y]`. Our port had the
kernels writing the scratch at the LOOKED-UP id, which only agrees with the
bridge when the list is the identity.

The check plants a different value in every cell, asks for leaves in the
order `[2, 0, 3]`, and requires

- every requested leaf to match a host tally, cell by cell
- leaf 1, which was NOT requested, to still hold its sentinel

The second requirement is the one the subtraction depends on: a level that
rebuilds a subset must leave the rest of the histogram alone.
"""

from max.gpu.host import DeviceContext

from gbdt.gpu_data.compressed_index_builder import build_layout
from gbdt.gpu_data.feature_blocks import blocks_for
from gbdt.gpu_data.kernel.binarize import (
    WRITE_BLOCK_SIZE,
    write_compressed_index_kernel,
)
from gbdt.methods.greedy_subsets_searcher.greedy_search_helper import (
    upload_scale,
    launch_histograms_for_blocks,
    upload_blocks,
)
from gbdt.methods.greedy_subsets_searcher.kernel.histogram_utils import (
    fixed_to_float_kernel,
)


def check_permuted_leaf_ids(depth: Int = 0, replicas: Int = 1, permute: Bool = True) raises:
    var ctx = DeviceContext()
    var n_rows = 2048
    var stat_count = 2
    var max_leaves = 4
    var sentinel = Float32(-7.0)

    var folds = List[Int]()
    for _ in range(8):
        folds.append(1)
    for _ in range(4):
        folds.append(8)
    for _ in range(4):
        folds.append(254)
    var n_features = len(folds)

    var lay = build_layout(folds)
    var blocks = blocks_for(lay, n_rows)
    var dblocks = upload_blocks(ctx, blocks)

    var cindex = ctx.enqueue_create_buffer[DType.uint32](n_rows * lay.columns)
    var z = ctx.enqueue_create_host_buffer[DType.uint32](n_rows * lay.columns)
    for i in range(n_rows * lay.columns):
        z.unsafe_ptr().unsafe_store(i, UInt32(0))
    ctx.enqueue_copy(dst_buf=cindex, src_ptr=z.unsafe_ptr())
    ctx.synchronize()

    # SCATTERED bins, hashed per row and feature, so a cell's expected value
    # differs from its neighbour's. A uniform plant would verify the total
    # and nothing about placement.
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
            var v = Int(x % UInt32(folds[f] + 1))
            if v > folds[f]:
                v = folds[f]
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

    # Four partitions of unequal size, so no two leaves can be confused by
    # their totals.
    var off = List[Int]()
    var siz = List[Int]()
    off.append(0)
    siz.append(500)
    off.append(500)
    siz.append(300)
    off.append(800)
    siz.append(700)
    off.append(1500)
    siz.append(548)

    var p_off = ctx.enqueue_create_buffer[DType.uint32](max_leaves)
    var p_sz = ctx.enqueue_create_buffer[DType.uint32](max_leaves)
    var ho = ctx.enqueue_create_host_buffer[DType.uint32](max_leaves)
    var hzz = ctx.enqueue_create_host_buffer[DType.uint32](max_leaves)
    for i in range(max_leaves):
        ho.unsafe_ptr().unsafe_store(i, UInt32(off[i]))
        hzz.unsafe_ptr().unsafe_store(i, UInt32(siz[i]))
    ctx.enqueue_copy(dst_buf=p_off, src_ptr=ho.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=p_sz, src_ptr=hzz.unsafe_ptr())

    # THE POINT: a permuted, non-contiguous request. Leaf 1 is absent.
    var want_ids = List[Int]()
    if permute:
        want_ids.append(2)
        want_ids.append(0)
        want_ids.append(3)
    else:
        # The CONTIGUOUS control. Separates "permuted ids are broken" from
        # "this configuration is broken regardless of the ids".
        want_ids.append(0)
        want_ids.append(1)
        want_ids.append(2)
    var n_live = len(want_ids)

    var ids = ctx.enqueue_create_buffer[DType.uint32](max_leaves)
    var hid = ctx.enqueue_create_host_buffer[DType.uint32](max_leaves)
    for i in range(max_leaves):
        hid.unsafe_ptr().unsafe_store(i, UInt32(0))
    for i in range(n_live):
        hid.unsafe_ptr().unsafe_store(i, UInt32(want_ids[i]))
    ctx.enqueue_copy(dst_buf=ids, src_ptr=hid.unsafe_ptr())

    var dense_ids = ctx.enqueue_create_buffer[DType.uint32](max_leaves)
    var h_dense = ctx.enqueue_create_host_buffer[DType.uint32](max_leaves)
    for i in range(max_leaves):
        h_dense.unsafe_ptr().unsafe_store(i, UInt32(i))
    ctx.enqueue_copy(dst_buf=dense_ids, src_ptr=h_dense.unsafe_ptr())

    var hist_cells = max_leaves * stat_count * lay.hist_cells
    var hist = ctx.enqueue_create_buffer[DType.float32](hist_cells)
    var acc = ctx.enqueue_create_buffer[DType.int32](hist_cells)
    var zf = ctx.enqueue_create_host_buffer[DType.float32](hist_cells)
    var zi = ctx.enqueue_create_host_buffer[DType.int32](hist_cells)
    for i in range(hist_cells):
        zf.unsafe_ptr().unsafe_store(i, sentinel)
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

    # `depth` picks the load path: 0 is the direct load, anything below it
    # is the GATHER variant, which reads rows through `row_index`. With
    # `row_index` the identity the two must agree cell for cell, so running
    # both is a differential on the gather path alone.
    var scale_keep = upload_scale(ctx, Float32(1.0))
    var scale_ptr = rebind[MutPointer[Float32, MutAnyOrigin]](
        scale_keep.unsafe_ptr()
    )
    launch_histograms_for_blocks(
        ctx, dblocks, depth, n_live, n_rows, stat_count, max_leaves,
        replicas, scale_ptr, cindex, row_index, stats, p_off, p_sz, ids,
        dense_ids,
        hist, acc, block_hist, lay.hist_cells,
    )
    ctx.enqueue_function[fixed_to_float_kernel](
        acc.unsafe_ptr(), hist.unsafe_ptr(), Int32(hist_cells), scale_ptr,
        grid_dim=(hist_cells + 255) // 256, block_dim=256,
    )
    ctx.synchronize()
    _ = scale_keep^

    var out = ctx.enqueue_create_host_buffer[DType.float32](hist_cells)
    ctx.enqueue_copy(dst_ptr=out.unsafe_ptr(), src_buf=hist)
    ctx.synchronize()

    print("  depth", depth, "( 0 = direct, >0 = gather ), replicas",
          replicas, "permuted", permute)

    var wrong = 0
    var checked = 0
    for k in range(n_live):
        var leaf = want_ids[k]
        # Host tally for THIS leaf's rows only.
        var tally = List[Int]()
        for _ in range(lay.hist_cells):
            tally.append(0)
        for r in range(off[leaf], off[leaf] + siz[leaf]):
            for f in range(n_features):
                var b = host_bin[f][r]
                if b < folds[f]:
                    tally[Int(lay.features[f].first_fold_index) + b] += 1

        var leaf_wrong = 0
        for c in range(lay.hist_cells):
            var got = out.unsafe_ptr().unsafe_load(
                leaf * stat_count * lay.hist_cells + c
            )
            if got != Float32(tally[c]):
                leaf_wrong += 1
            checked += 1
        wrong += leaf_wrong
        print("    leaf", leaf, "size", siz[leaf], ":", leaf_wrong,
              "wrong of", lay.hist_cells)

    # Leaf 1 was never requested. Anything other than the sentinel means the
    # build wrote into a leaf it was not asked for, which is exactly what
    # would corrupt a parent histogram the subtraction is about to read.
    # The bystander is whichever leaf was NOT asked for, which differs
    # between the permuted and contiguous arms. Hardcoding it to leaf 1 made
    # the contiguous arm assert against a leaf it had actually requested.
    var bystander = -1
    for leaf in range(max_leaves):
        var named = False
        for k in range(n_live):
            if want_ids[k] == leaf:
                named = True
        if not named:
            bystander = leaf
            break

    var untouched_wrong = 0
    if bystander >= 0:
        for c in range(lay.hist_cells):
            var got = out.unsafe_ptr().unsafe_load(
                bystander * stat_count * lay.hist_cells + c
            )
            if got != sentinel:
                untouched_wrong += 1
    print("    leaf", bystander, "(not requested):", untouched_wrong,
          "cells disturbed of", lay.hist_cells)

    print("  total wrong:", wrong, "of", checked)
    if wrong != 0:
        raise Error(
            String("permuted leaf ids produced ") + String(wrong)
            + " wrong cells; the dense/looked-up index split is broken"
        )
    if untouched_wrong != 0:
        raise Error(
            String("the build disturbed ") + String(untouched_wrong)
            + " cells of a leaf it was not asked to build"
        )
    print("  non-contiguous leaf ids land in the right slots")
