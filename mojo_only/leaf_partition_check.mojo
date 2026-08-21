"""Computing a row's leaf from the compressed index, against the searcher.

    pixi run check-leaf-partition

Their leaves estimator asks a dataset the tree was NOT grown on which leaf
each row falls in (`doc_parallel_leaves_estimator.cpp:45-49`,
`task.Model->ComputeBins(*task.DataSet, &bins)`). That is what every
permutation but one needs, and it is a second, independent answer to a
question the searcher already answers for free on the permutation it grew
the tree on.

**THE GATE IS THAT THE TWO ANSWERS AGREE, ROW FOR ROW.**

    run_tree_layout        grows the tree by MOVING ROWS: nine kernels per
                           level, partition growth, `row_index` left
                           bin-sorted with per-leaf offsets and sizes.

    compute_bins_for_model EVALUATES the finished tree: reads each row's
                           feature values out of the compressed index and
                           builds the leaf index bit by bit.

Nothing about the second path re-uses the first. They share the compressed
index and the split list and nothing else -- different kernels, different
memory, different order. So if the leaf sets agree for all 2^depth leaves,
both are right about where the rows went, and if they disagree, one of them
is wrong about a real fit.

This is the check that makes `partition_from_bins` usable for the other
permutations. Without it that path would be gated only by the fit it
produces, and a fit is exactly the thing that cannot tell you a row landed
in the wrong leaf: it would just be a slightly different model.

SABOTAGES:

    L1  one split's bin index moved by one     that the comparison is
                                               sensitive to the SPLITS and
                                               not just to the row count
    L2  the leaf bit order reversed            that it is sensitive to
                                               WHICH leaf, not just to the
                                               partition of rows -- a
                                               reversed bit order is a
                                               permutation of the leaves,
                                               so every leaf still has the
                                               right SIZE
"""

from max.gpu.host import DeviceContext

from gbdt.gpu_data.compressed_index_builder import build_layout
from gbdt.gpu_data.kernel.binarize import (
    WRITE_BLOCK_SIZE,
    write_compressed_index_kernel,
)
from gbdt.methods.greedy_subsets_searcher.greedy_search_helper import (
    TTreeWorkspace,
    run_tree_layout,
)
from gbdt.methods.leaves_estimation.doc_parallel_leaves_estimator import (
    compute_bins_for_model,
    partition_from_bins,
)
from gbdt.models.oblivious_model import TBinarySplit


def _leaf_of_row(part_offsets: List[Int], part_sizes: List[Int],
                 rows: List[UInt32], n_rows: Int) raises -> List[Int]:
    """Invert a partition: row -> the leaf it sits in.

    Comparing the two partitions row by row rather than leaf by leaf is
    deliberate. The two paths do NOT lay their leaves out in the same
    place -- the searcher's partitions sit in memory in the device's own
    order -- so a leaf-by-leaf comparison of offsets would fail on a
    difference that is not a difference. Where a row ENDED UP is the thing
    both paths agree about.
    """
    var out = List[Int]()
    for _ in range(n_rows):
        out.append(-1)
    for leaf in range(len(part_sizes)):
        for k in range(part_sizes[leaf]):
            var r = Int(rows[part_offsets[leaf] + k])
            if r < 0 or r >= n_rows:
                raise Error("partition holds row " + String(r))
            if out[r] != -1:
                raise Error("row " + String(r) + " appears twice")
            out[r] = leaf
    for r in range(n_rows):
        if out[r] == -1:
            raise Error("row " + String(r) + " is in no leaf")
    return out^


def check_leaf_partition() raises:
    var ctx = DeviceContext()
    var failures = 0

    # the mixed-width fixture: 8 binary, 4 half-byte, 4 one-byte, so the
    # bins kernel has to get three different masks and shifts right
    var n_rows = 4096
    var max_depth = 5
    var folds = List[Int]()
    for _ in range(8):
        folds.append(1)
    for _ in range(4):
        folds.append(8)
    for _ in range(4):
        folds.append(64)
    var n_features = len(folds)
    var n_columns = 3

    var cindex = ctx.enqueue_create_buffer[DType.uint32](n_rows * n_columns)
    var z = ctx.enqueue_create_host_buffer[DType.uint32](n_rows * n_columns)
    for i in range(n_rows * n_columns):
        z.unsafe_ptr().unsafe_store(i, UInt32(0))
    ctx.enqueue_copy(dst_buf=cindex, src_ptr=z.unsafe_ptr())
    ctx.synchronize()

    var lay = build_layout(folds)
    var hb = ctx.enqueue_create_host_buffer[DType.uint8](n_rows)
    var bins8 = ctx.enqueue_create_buffer[DType.uint8](n_rows)
    for f in range(n_features):
        ref cf = lay.features[f]
        for r in range(n_rows):
            var x = UInt32(r * 2654435761 + f * 40503 + 0x2545F491)
            x ^= x << 13
            x ^= x >> 17
            x ^= x << 5
            hb.unsafe_ptr().unsafe_store(
                r, UInt8(Int(x % UInt32(folds[f] + 1)))
            )
        ctx.enqueue_copy(dst_buf=bins8, src_ptr=hb.unsafe_ptr())
        ctx.enqueue_function[write_compressed_index_kernel](
            Int32(Int(cf.offset) * n_rows),
            cf.mask,
            cf.shift,
            bins8.unsafe_ptr(),
            Int32(n_rows),
            cindex.unsafe_ptr(),
            grid_dim=(n_rows + WRITE_BLOCK_SIZE - 1) // WRITE_BLOCK_SIZE,
            block_dim=WRITE_BLOCK_SIZE,
        )
        ctx.synchronize()

    # gradients that actually separate, so the tree splits on real features
    var stats = ctx.enqueue_create_buffer[DType.float32](2 * n_rows)
    var hs = ctx.enqueue_create_host_buffer[DType.float32](2 * n_rows)
    var tw = Float64(0.0)
    var tg = Float64(0.0)
    for r in range(n_rows):
        var g = Float64(-0.1)
        if (r % 4) == 0:
            g = 3.0
        hs.unsafe_ptr().unsafe_store(r, Float32(1.0))
        hs.unsafe_ptr().unsafe_store(n_rows + r, Float32(g))
        tw += 1.0
        tg += -g if g < 0.0 else g
    ctx.enqueue_copy(dst_buf=stats, src_ptr=hs.unsafe_ptr())

    var row_index = ctx.enqueue_create_buffer[DType.uint32](n_rows)
    var hi = ctx.enqueue_create_host_buffer[DType.uint32](n_rows)
    for r in range(n_rows):
        hi.unsafe_ptr().unsafe_store(r, UInt32(r))
    ctx.enqueue_copy(dst_buf=row_index, src_ptr=hi.unsafe_ptr())
    ctx.synchronize()

    var scratch_cursor = ctx.enqueue_create_buffer[DType.float32](1)
    var splits = List[TBinarySplit]()
    var leaf_values = List[Float32]()
    var leaf_offsets = List[Int]()
    var ws = List[TTreeWorkspace]()
    var sizes = run_tree_layout(
        ctx, n_rows, folds, max_depth, cindex, stats, row_index,
        scratch_cursor,
        Float32(tw), Float32(tg),
        splits, leaf_values, leaf_offsets,
        ws,
        export_offsets=True,
    )

    var nonempty = 0
    for i in range(len(sizes)):
        if sizes[i] > 0:
            nonempty += 1
    print(
        "  searcher: depth", max_depth, "->", len(sizes), "leaves,",
        nonempty, "non-empty",
    )
    if nonempty < 2:
        raise Error(
            "the fixture produced a tree that never split; the comparison"
            " below would be vacuous"
        )

    var h_rows = ctx.enqueue_create_host_buffer[DType.uint32](n_rows)
    ctx.enqueue_copy(dst_ptr=h_rows.unsafe_ptr(), src_buf=row_index)
    ctx.synchronize()
    var searcher_rows = List[UInt32]()
    for r in range(n_rows):
        searcher_rows.append(h_rows.unsafe_ptr().unsafe_load(r))
    var searcher_leaf = _leaf_of_row(
        leaf_offsets, sizes, searcher_rows, n_rows
    )

    print()
    print("-- the gate: evaluating the tree reproduces the growth --")
    var d_bins = ctx.enqueue_create_buffer[DType.uint32](n_rows)
    compute_bins_for_model(
        ctx, lay, splits, max_depth, cindex, n_rows, d_bins
    )
    var part = partition_from_bins(ctx, d_bins, n_rows, 1 << max_depth)
    var part_rows = List[UInt32]()
    var h_pr = ctx.enqueue_create_host_buffer[DType.uint32](n_rows)
    ctx.enqueue_copy(dst_ptr=h_pr.unsafe_ptr(), src_buf=part.row_index)
    ctx.synchronize()
    for r in range(n_rows):
        part_rows.append(h_pr.unsafe_ptr().unsafe_load(r))
    var evaluated_leaf = _leaf_of_row(
        part.offsets, part.sizes, part_rows, n_rows
    )

    var moved = 0
    for r in range(n_rows):
        if evaluated_leaf[r] != searcher_leaf[r]:
            moved += 1
    if moved != 0:
        print(
            "  FAIL", moved, "of", n_rows,
            "rows land in a different leaf than the searcher put them in",
        )
        failures += 1
    else:
        print(
            "  ok   all", n_rows,
            "rows land in the leaf the searcher put them in, over",
            nonempty, "non-empty leaves",
        )

    # and the sizes, which is a weaker statement but names the failure
    # differently if it ever fires alone
    var size_mismatch = 0
    for i in range(len(sizes)):
        if part.sizes[i] != sizes[i]:
            size_mismatch += 1
    if size_mismatch != 0:
        print(
            "  FAIL", size_mismatch, "leaves have a different size",
        )
        failures += 1
    else:
        print("  ok   every leaf has the size the searcher reported")

    print()
    print("-- sabotages --")
    # L1: one split's bin moved by one. Rows on the border of that split
    # change side, so the leaf sets must move.
    var sab = List[TBinarySplit]()
    for i in range(len(splits)):
        sab.append(splits[i])
    sab[0].bin_idx = sab[0].bin_idx + 1
    var d_bins1 = ctx.enqueue_create_buffer[DType.uint32](n_rows)
    compute_bins_for_model(
        ctx, lay, sab, max_depth, cindex, n_rows, d_bins1
    )
    var part1 = partition_from_bins(ctx, d_bins1, n_rows, 1 << max_depth)
    var h_pr1 = ctx.enqueue_create_host_buffer[DType.uint32](n_rows)
    ctx.enqueue_copy(dst_ptr=h_pr1.unsafe_ptr(), src_buf=part1.row_index)
    ctx.synchronize()
    var rows1 = List[UInt32]()
    for r in range(n_rows):
        rows1.append(h_pr1.unsafe_ptr().unsafe_load(r))
    var leaf1 = _leaf_of_row(part1.offsets, part1.sizes, rows1, n_rows)
    var moved1 = 0
    for r in range(n_rows):
        if leaf1[r] != searcher_leaf[r]:
            moved1 += 1
    if moved1 == 0:
        print(
            "  FAIL L1: moving split 0's bin by one changed no row's leaf",
        )
        failures += 1
    else:
        print(
            "  ok   L1", moved1, "of", n_rows,
            "rows move when split 0's bin moves by one",
        )

    # L2: reverse the bit order by reversing the split list. Every leaf
    # keeps a SIZE that some leaf had, so a size-only comparison survives
    # it -- which is exactly why the gate above is row-wise.
    var rev = List[TBinarySplit]()
    for i in range(len(splits)):
        rev.append(splits[len(splits) - 1 - i])
    var d_bins2 = ctx.enqueue_create_buffer[DType.uint32](n_rows)
    compute_bins_for_model(
        ctx, lay, rev, max_depth, cindex, n_rows, d_bins2
    )
    var part2 = partition_from_bins(ctx, d_bins2, n_rows, 1 << max_depth)
    var h_pr2 = ctx.enqueue_create_host_buffer[DType.uint32](n_rows)
    ctx.enqueue_copy(dst_ptr=h_pr2.unsafe_ptr(), src_buf=part2.row_index)
    ctx.synchronize()
    var rows2 = List[UInt32]()
    for r in range(n_rows):
        rows2.append(h_pr2.unsafe_ptr().unsafe_load(r))
    var leaf2 = _leaf_of_row(part2.offsets, part2.sizes, rows2, n_rows)
    var moved2 = 0
    for r in range(n_rows):
        if leaf2[r] != searcher_leaf[r]:
            moved2 += 1
    if moved2 == 0:
        print("  FAIL L2: reversing the bit order changed no row's leaf")
        failures += 1
    else:
        print(
            "  ok   L2", moved2, "of", n_rows,
            "rows move under a reversed bit order, which conserves every"
            " leaf SIZE",
        )

    print()
    if failures != 0:
        raise Error(
            "leaf partition check: " + String(failures) + " failures"
        )
    print("leaf partition: PASS")


def main() raises:
    check_leaf_partition()
