"""cuML `cpp/src/metrics/mutual_info_score.cu` (265b9da): `ML::Metrics::mutual_info_score` forwards to `raft::stats::mutual_info_score`. The `int` overloads (cuML's Python passes int32 labels)."""

from max.gpu.host import DeviceBuffer, DeviceContext

from metrics.ported.stats.detail.mutual_info_score import (
    mutual_info_score as _raft_mi,
)


def mutual_info_score(
    ctx: DeviceContext,
    mut y: DeviceBuffer[DType.int32],
    mut y_hat: DeviceBuffer[DType.int32],
    n: Int,
    lower_class_range: Int32,
    upper_class_range: Int32,
) raises -> Float64:
    """`double mutual_info_score(handle, y, y_hat, n, lower, upper)`."""
    return _raft_mi(ctx, y, y_hat, n, lower_class_range, upper_class_range)
