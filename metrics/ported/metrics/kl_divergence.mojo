"""cuML `cpp/src/metrics/kl_divergence.cu` (265b9da): `ML::Metrics::kl_divergence` forwards to `raft::stats::kl_divergence`. The `float` overload; the `double` one cannot run on Apple (no Float64 on device) and is refused by the Python surface (UNPORTED.tsv)."""

from max.gpu.host import DeviceBuffer, DeviceContext

from metrics.ported.stats.detail.kl_divergence import (
    kl_divergence as _raft_kl,
)


def kl_divergence(
    ctx: DeviceContext,
    mut y: DeviceBuffer[DType.float32],
    mut y_hat: DeviceBuffer[DType.float32],
    n: Int,
) raises -> Float32:
    """`float kl_divergence(handle, const float* y, const float* y_hat, int n)`."""
    return _raft_kl(ctx, y, y_hat, n)
