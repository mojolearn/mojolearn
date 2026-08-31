# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Derivative work: the upstream file and its pinned commit are recorded in this lane's PORTED_MAP.tsv and in this file's own docstring. See NOTICE.
"""`prepare_data`: difference a batch before a test or a filter.

PORT OF `cuml/cpp/src_prims/timeSeries/arima_helpers.cuh` at cuML 265b9da6
(v26.08.00), lines 194-240 ONLY (`prepare_data`). The stationarity test is
its one caller in this lane (`stationarity.cuh:330`). The rest of the file
-- `reduced_polynomial`, `prepare_future_data`, `finalize_forecast` /
`_undiff_kernel`, `batched_jones_transform` -- is reached by the ARIMA lane
and ported in `arima/ported/timeSeries/arima_helpers.mojo`, which imports
`prepare_data` from here rather than spelling it twice.

COPY, DO NOT IMPROVE. Their dispatch (`arima_helpers.cuh:219-239`):
`d + D == 1` -> one `batched_diff_kernel` with period `1` if `d` else `s`;
`d + D == 2` -> one `batched_second_diff_kernel` with `period1 = d ? 1 : s`,
`period2 = d == 2 ? 1 : s`; `d + D == 0` -> a copy. Threads per block is
their "quick heuristic" (256 past 512 rows, else 128): SCHEDULING, no fold.
`d + D <= 2` is enforced one layer up (`arima.pyx:313`) and RAISED here by
name rather than assumed.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from tsa.ported.linalg.batched.matrix import (
    batched_diff_host,
    batched_diff_kernel,
    batched_second_diff_host,
    batched_second_diff_kernel,
)


def prepare_data(
    ctx: DeviceContext,
    mut d_out: DeviceBuffer[DType.float32],
    mut d_in: DeviceBuffer[DType.float32],
    batch_size: Int,
    n_obs: Int,
    d: Int,
    D: Int,
    s: Int,
) raises:
    """`arima_helpers.cuh:209-240`. `d_out` holds `(n_obs - d - D*s) *
    batch_size` values on return."""
    if d + D > 2:
        raise Error(
            "prepare_data: d + D must be <= 2 (d=" + String(d) + ", D="
            + String(D) + "), refused by name (arima.pyx:313)"
        )
    if D > 0 and s < 2:
        raise Error(
            "prepare_data: seasonal differencing needs s >= 2 (s=" + String(s)
            + "), refused by name (arima.pyx:310)"
        )
    if d + D == 1:
        var period = 1 if d != 0 else s
        var tpb = 256 if (n_obs - period) > 512 else 128
        ctx.enqueue_function[batched_diff_kernel](
            d_in.unsafe_ptr(), d_out.unsafe_ptr(), Int32(n_obs), Int32(period),
            grid_dim=(batch_size, 1, 1), block_dim=(tpb, 1, 1),
        )
    elif d + D == 2:
        var period1 = 1 if d != 0 else s
        var period2 = 1 if d == 2 else s
        var tpb = 256 if (n_obs - period1 - period2) > 512 else 128
        ctx.enqueue_function[batched_second_diff_kernel](
            d_in.unsafe_ptr(), d_out.unsafe_ptr(), Int32(n_obs),
            Int32(period1), Int32(period2),
            grid_dim=(batch_size, 1, 1), block_dim=(tpb, 1, 1),
        )
    else:
        # `raft::copy(d_out, d_in, n_obs * batch_size)` -- their `d_in != d_out`
        # arm; the same-pointer arm cannot arise here because Mojo refuses an
        # aliased launch argument and every caller passes two buffers.
        var view_in = d_in.create_sub_buffer[DType.float32](0, n_obs * batch_size)
        var view_out = d_out.create_sub_buffer[DType.float32](0, n_obs * batch_size)
        ctx.enqueue_copy(dst_buf=view_out, src_buf=view_in)


def prepare_data_host(
    y: List[Float32], batch_size: Int, n_obs: Int, d: Int, D: Int, s: Int
) raises -> List[Float32]:
    """The host replay of the dispatch above (NOT a port; the oracle)."""
    if d + D > 2:
        raise Error("prepare_data_host: d + D must be <= 2")
    if d + D == 1:
        return batched_diff_host(y, batch_size, n_obs, 1 if d != 0 else s)
    elif d + D == 2:
        return batched_second_diff_host(
            y, batch_size, n_obs, 1 if d != 0 else s, 1 if d == 2 else s
        )
    var out = List[Float32]()
    for i in range(n_obs * batch_size):
        out.append(y[i])
    return out^


def n_obs_after_diff(n_obs: Int, d: Int, D: Int, s: Int) -> Int:
    return n_obs - d - s * D
