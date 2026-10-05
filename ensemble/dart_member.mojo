# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""DART member fits read their inputs from the DART session on the device
(cpu4-forest, 2026-10-04).

Before: `x_trees_dart_step` downloaded each class's n-sized fit target to a
host array, and the member `RandomForestRegressor` fit staged it again (and,
under a bag or a column sample, gathered X and the target on the host side
of the call). Now the RF binding's `rf_regressor_fit_dart_export` fits the
member on the DART session's own buffers, on the DART context: the target
is the session's `target` plane for class `c` (a sub-buffer, no copy), and
X is the session's row-major X laid out column-major for the forest -- at
the bag rows and the sampled columns when the round has them -- by
`dart_member_x_kernel`. The values are the ones the host arrays held, so
the member forest is bit for bit the one the staged fit built.
"""

from std.sys.compile import is_defined as _rfx_is_defined
from std.gpu import block_dim, block_idx, thread_idx
from std.math import ceildiv
from max.gpu.host import DeviceBuffer, DeviceContext

from core.launch_clock import log_launch_ctx

comptime DART_MEMBER_TPB = 256


def dart_member_x_kernel(
    x: MutPointer[Float32, MutAnyOrigin],
    rows: MutPointer[Int32, MutAnyOrigin],
    cols: MutPointer[Int32, MutAnyOrigin],
    dst: MutPointer[Float32, MutAnyOrigin],
    d: Int32,
    m: Int32,
    dc: Int32,
    use_rows: Int32,
    use_cols: Int32,
):
    """`dst` (column-major `m x dc`) `[j * m + i] = x[rows[i] * d +
    cols[j]]` from the row-major `n x d` session X; identity rows / columns
    when the round has none. One thread per output element."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var mm = Int(m)
    if t >= mm * Int(dc):
        return
    var j = t // mm
    var i = t - j * mm
    var r = i
    if use_rows != Int32(0):
        r = Int(rows[unsafe_offset=i])
    var c = j
    if use_cols != Int32(0):
        c = Int(cols[unsafe_offset=j])
    dst[unsafe_offset=t] = x[unsafe_offset = r * Int(d) + c]


def dart_member_y_kernel(
    target: MutPointer[Float32, MutAnyOrigin],
    rows: MutPointer[Int32, MutAnyOrigin],
    dst: MutPointer[Float32, MutAnyOrigin],
    m: Int32,
):
    """The class target at the bag rows: `dst[i] = target[rows[i]]`."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(m):
        return
    dst[unsafe_offset=i] = target[unsafe_offset = Int(rows[unsafe_offset=i])]


def launch_dart_member_x(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    mut rows: DeviceBuffer[DType.int32],
    mut cols: DeviceBuffer[DType.int32],
    mut out: DeviceBuffer[DType.float32],
    d: Int,
    m: Int,
    dc: Int,
    use_rows: Bool,
    use_cols: Bool,
) raises:
    var total = m * dc
    if total <= 0:
        return
    log_launch_ctx(ctx, "dart_member_x")
    comptime if not _rfx_is_defined["RFX_38"]():
        ctx.enqueue_function[dart_member_x_kernel](
            x.unsafe_ptr(),
            rows.unsafe_ptr(),
            cols.unsafe_ptr(),
            out.unsafe_ptr(),
            Int32(d),
            Int32(m),
            Int32(dc),
            Int32(1) if use_rows else Int32(0),
            Int32(1) if use_cols else Int32(0),
            grid_dim=ceildiv(total, DART_MEMBER_TPB),
            block_dim=DART_MEMBER_TPB,
        )


def launch_dart_member_y(
    ctx: DeviceContext,
    mut target: DeviceBuffer[DType.float32],
    mut rows: DeviceBuffer[DType.int32],
    mut out: DeviceBuffer[DType.float32],
    m: Int,
) raises:
    if m <= 0:
        return
    log_launch_ctx(ctx, "dart_member_y")
    comptime if not _rfx_is_defined["RFX_39"]():
        ctx.enqueue_function[dart_member_y_kernel](
            target.unsafe_ptr(),
            rows.unsafe_ptr(),
            out.unsafe_ptr(),
            Int32(m),
            grid_dim=ceildiv(m, DART_MEMBER_TPB),
            block_dim=DART_MEMBER_TPB,
        )
