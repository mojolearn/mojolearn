# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The k-means labels as Int32, converted on the device (lane fam2-cluster,
2026-10-04).

`kmeans_fit_predict` / `kmeans_predict_device` write UInt32 labels; the
spectral entries return Int32. The conversion was a host loop over n after
the download; here it is one launch (one thread per label) and the download
lands in the result's own memory. Integer work: no bit moves.
`-D MOJOLEARN_IDN_SPECTRAL_LABELS_DEVICE_OFF=1` restores the host loop, as
does the master `-D MOJOLEARN_IDN_ALL_OFF=1`.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL

comptime LABELS_TPB = 256

comptime IDN_SPECTRAL_LABELS_DEVICE = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and not (
        is_defined["MOJOLEARN_IDN_SPECTRAL_LABELS_DEVICE_OFF"]()
        or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
    )
)


def labels_u32_to_i32_kernel(
    dst: MutPointer[Int32, MutAnyOrigin],
    src: MutPointer[UInt32, MutAnyOrigin],
    n_in: Int32,
):
    """`dst[i] = Int32(src[i])`, one thread per label."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_in):
        return
    dst.unsafe_store(i, Int32(Int(src.unsafe_load(i))))


def download_labels_i32(
    ctx: DeviceContext, mut d_labels: DeviceBuffer[DType.uint32], n: Int
) raises -> List[Int32]:
    """The `n` UInt32 labels in `d_labels` as a host `List[Int32]`, the
    conversion on the device."""
    var out = List[Int32](length=n, fill=Int32(0))
    if n <= 0:
        return out^
    var d_i32 = ctx.enqueue_create_buffer[DType.int32](n)
    ctx.enqueue_function[labels_u32_to_i32_kernel](
        d_i32.unsafe_ptr(), d_labels.unsafe_ptr(), Int32(n),
        grid_dim=((n + LABELS_TPB - 1) // LABELS_TPB, 1, 1),
        block_dim=(LABELS_TPB, 1, 1),
    )
    ctx.enqueue_copy(dst_ptr=out.unsafe_ptr(), src_buf=d_i32)
    ctx.synchronize()
    _ = d_i32^
    return out^
