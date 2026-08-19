"""Check `copy_histograms_kernel` on its own.

It is the only one of the three histogram-maintenance kernels with no
standalone check, and it is the one the sibling subtraction leans on hardest:
`CopyHistogram(LeafIdToSplit, RightLeafIdAfterSplit, ...)`
(`split_points.cpp:326`) is what puts the parent histogram into BOTH
children's slots, which is what makes the pairing legal whichever child turns
out smaller. A wrong copy produces a plausible histogram, not a crash.

Every cell is planted with a DIFFERENT hashed value, so the check sees
placement and not just totals. It asserts three things:

- each destination leaf equals its source leaf, cell for cell
- each source leaf is unchanged
- every leaf that was not named is untouched
"""

from max.gpu.host import DeviceContext

from ported.methods.greedy_subsets_searcher.kernel.histogram_utils import (
    copy_histograms_kernel,
)


def check_copy_histograms() raises:
    var ctx = DeviceContext()
    var max_leaves = 8
    var stat_count = 2
    var bin_features = 173  # deliberately not a multiple of the block size
    var hist_size = bin_features * stat_count
    var cells = max_leaves * hist_size

    var hist = ctx.enqueue_create_buffer[DType.float32](cells)
    var h = ctx.enqueue_create_host_buffer[DType.float32](cells)

    # A distinct value per cell. A uniform plant would verify the total and
    # nothing about placement, which is the trap this repository keeps
    # falling into.
    for i in range(cells):
        var x = UInt32(i * 2654435761 + 0x9E3779B9)
        x ^= x << 13
        x ^= x >> 17
        x ^= x << 5
        h.unsafe_ptr().unsafe_store(i, Float32(Int(x % UInt32(100000))))
    ctx.enqueue_copy(dst_buf=hist, src_ptr=h.unsafe_ptr())

    # Copy 1 -> 5 and 3 -> 6. Non-contiguous, out of order, and leaving
    # leaves 0, 2, 4 and 7 alone.
    var src_ids = List[Int]()
    var dst_ids = List[Int]()
    src_ids.append(1)
    dst_ids.append(5)
    src_ids.append(3)
    dst_ids.append(6)
    var n_pairs = len(src_ids)

    var left = ctx.enqueue_create_buffer[DType.uint32](max_leaves)
    var right = ctx.enqueue_create_buffer[DType.uint32](max_leaves)
    var hl = ctx.enqueue_create_host_buffer[DType.uint32](max_leaves)
    var hr = ctx.enqueue_create_host_buffer[DType.uint32](max_leaves)
    for i in range(max_leaves):
        hl.unsafe_ptr().unsafe_store(i, UInt32(0))
        hr.unsafe_ptr().unsafe_store(i, UInt32(0))
    for i in range(n_pairs):
        hl.unsafe_ptr().unsafe_store(i, UInt32(src_ids[i]))
        hr.unsafe_ptr().unsafe_store(i, UInt32(dst_ids[i]))
    ctx.enqueue_copy(dst_buf=left, src_ptr=hl.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=right, src_ptr=hr.unsafe_ptr())
    ctx.synchronize()

    ctx.enqueue_function[copy_histograms_kernel](
        left.unsafe_ptr(), right.unsafe_ptr(),
        Int32(stat_count), Int32(bin_features), hist.unsafe_ptr(),
        grid_dim=((hist_size + 255) // 256, n_pairs, 1),
        block_dim=(256, 1, 1),
    )
    ctx.synchronize()

    var out = ctx.enqueue_create_host_buffer[DType.float32](cells)
    ctx.enqueue_copy(dst_ptr=out.unsafe_ptr(), src_buf=hist)
    ctx.synchronize()

    var copied_wrong = 0
    var source_wrong = 0
    for k in range(n_pairs):
        var s = src_ids[k]
        var d = dst_ids[k]
        for c in range(hist_size):
            var want = h.unsafe_ptr().unsafe_load(s * hist_size + c)
            if out.unsafe_ptr().unsafe_load(d * hist_size + c) != want:
                copied_wrong += 1
            if out.unsafe_ptr().unsafe_load(s * hist_size + c) != want:
                source_wrong += 1
        print("    copied leaf", s, "->", d)

    var bystander_wrong = 0
    for leaf in range(max_leaves):
        var named = False
        for k in range(n_pairs):
            if leaf == src_ids[k] or leaf == dst_ids[k]:
                named = True
        if named:
            continue
        for c in range(hist_size):
            if out.unsafe_ptr().unsafe_load(
                leaf * hist_size + c
            ) != h.unsafe_ptr().unsafe_load(leaf * hist_size + c):
                bystander_wrong += 1

    print("    destination cells wrong:", copied_wrong, "of",
          n_pairs * hist_size)
    print("    source cells disturbed:", source_wrong)
    print("    bystander cells disturbed:", bystander_wrong)

    if copied_wrong != 0:
        raise Error(
            String("copy_histograms wrote ") + String(copied_wrong)
            + " wrong cells"
        )
    if source_wrong != 0 or bystander_wrong != 0:
        raise Error("copy_histograms disturbed a leaf it was not given")
    print("  parent histogram lands in the right slot, nothing else moves")
