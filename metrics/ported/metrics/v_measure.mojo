"""cuML `cpp/src/metrics/v_measure.cu` (265b9da): `ML::Metrics::v_measure` forwards to `raft::stats::v_measure`. The `int` overloads (cuML's Python passes int32 labels)."""

from max.gpu.host import DeviceBuffer, DeviceContext

from metrics.ported.stats.detail.v_measure import v_measure as _raft_v


def v_measure(
    ctx: DeviceContext,
    mut y: DeviceBuffer[DType.int32],
    mut y_hat: DeviceBuffer[DType.int32],
    n: Int,
    lower_class_range: Int32,
    upper_class_range: Int32,
    beta: Float64 = 1.0,
) raises -> Float64:
    """`double v_measure(handle, y, y_hat, n, lower, upper, beta)`."""
    return _raft_v(
        ctx, y, y_hat, n, lower_class_range, upper_class_range, beta
    )
