"""cuML `cpp/src/metrics/entropy.cu` (265b9da): `ML::Metrics::entropy` forwards to `raft::stats::entropy`. The `int` overloads (cuML's Python passes int32 labels)."""

from max.gpu.host import DeviceBuffer, DeviceContext

from metrics.ported.stats.detail.entropy import entropy as _raft_entropy


def entropy(
    ctx: DeviceContext,
    mut y: DeviceBuffer[DType.int32],
    n: Int,
    lower_class_range: Int32,
    upper_class_range: Int32,
) raises -> Float64:
    """`double entropy(handle, const int* y, const int n, lower, upper)`."""
    return _raft_entropy(ctx, y, n, lower_class_range, upper_class_range)
