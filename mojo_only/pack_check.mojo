"""Does the packing actually round-trip on the GPU?

NO CATBOOST COUNTERPART: it is a correctness check, and this tree had none.
Every kernel so far is proven REACHABLE by `launch_probe` and proven correct
by nothing. Reachability is not correctness, exactly as compiling was not
reachability.

This checks the one property the whole port is FOR: that 32 binary features
survive being packed into a single `UInt32` and read back out. If this is
wrong, the read-density advantage is imaginary and everything downstream is
computing on garbage.
"""

from max.gpu.host import DeviceContext

from gbdt.gpu_data.grid_policy import (
    POLICY_BINARY,
    features_per_int,
    policy_mask,
    policy_shift,
)
from gbdt.gpu_data.kernel.binarize import (
    WRITE_BLOCK_SIZE,
    write_compressed_index_kernel,
)


def check_packing() raises:
    """Pack 32 binary features for a handful of rows, read them back.

    Each feature f is given the bin pattern `(row + f) % 2`, so a wrong shift
    or a wrong mask produces a mismatch rather than a plausible-looking
    answer. Every feature is written by its own launch, OR-ing into the same
    word, which is exactly how `TSharedCompressedIndexBuilder` drives it.
    """
    var ctx = DeviceContext()

    var n_rows = 64
    var n_features = features_per_int(POLICY_BINARY)  # 32
    print("  packing", n_features, "binary features x", n_rows, "rows")

    # ONE UInt32 column holds all 32 features. That is the whole point.
    var cindex = ctx.enqueue_create_buffer[DType.uint32](n_rows)
    var zero = ctx.enqueue_create_host_buffer[DType.uint32](n_rows)
    for i in range(n_rows):
        zero.unsafe_ptr().unsafe_store(i, UInt32(0))
    # The OR-write REQUIRES a zeroed destination; see binarize.mojo.
    ctx.enqueue_copy(dst_buf=cindex, src_ptr=zero.unsafe_ptr())

    var host_bins = ctx.enqueue_create_host_buffer[DType.uint8](n_rows)
    var bins = ctx.enqueue_create_buffer[DType.uint8](n_rows)

    for f in range(n_features):
        for r in range(n_rows):
            host_bins.unsafe_ptr().unsafe_store(r, UInt8((r + f) % 2))
        ctx.enqueue_copy(dst_buf=bins, src_ptr=host_bins.unsafe_ptr())
        var blocks = (n_rows + WRITE_BLOCK_SIZE - 1) // WRITE_BLOCK_SIZE
        ctx.enqueue_function[write_compressed_index_kernel](
            Int32(0),
            policy_mask(POLICY_BINARY),
            UInt32(policy_shift(POLICY_BINARY, f)),
            bins.unsafe_ptr(),
            Int32(n_rows),
            cindex.unsafe_ptr(),
            grid_dim=blocks,
            block_dim=WRITE_BLOCK_SIZE,
        )
        ctx.synchronize()

    var out = ctx.enqueue_create_host_buffer[DType.uint32](n_rows)
    ctx.enqueue_copy(dst_ptr=out.unsafe_ptr(), src_buf=cindex)
    ctx.synchronize()

    var mismatches = 0
    for r in range(n_rows):
        var word = out.unsafe_ptr().unsafe_load(r)
        for f in range(n_features):
            var shift = UInt32(policy_shift(POLICY_BINARY, f))
            var got = (word >> shift) & policy_mask(POLICY_BINARY)
            var want = UInt32((r + f) % 2)
            if got != want:
                mismatches += 1
    print("  row 0 word: ", out.unsafe_ptr().unsafe_load(0))
    print("  row 1 word: ", out.unsafe_ptr().unsafe_load(1))
    print(
        "  checked",
        n_rows * n_features,
        "feature values, mismatches:",
        mismatches,
    )
    if mismatches != 0:
        raise Error(
            "packing does NOT round-trip: "
            + String(mismatches)
            + " wrong feature values"
        )
    print("  32 features per UInt32 round-trip exactly")
