"""cuML `cpp/src/metrics/r2_score.cu` (265b9da): `ML::Metrics::r2_score_py` forwards to `raft::stats::r2_score`. The `int` overloads (cuML's Python passes int32 labels). The `float` overload; the `double` one cannot run on Apple (no Float64 on device) and is refused by the Python surface (UNPORTED.tsv)."""

from max.gpu.host import DeviceBuffer, DeviceContext

from metrics.ported.stats.detail.scores import r2_score as _raft_r2


def r2_score_py(
    ctx: DeviceContext,
    mut y: DeviceBuffer[DType.float32],
    mut y_hat: DeviceBuffer[DType.float32],
    n: Int,
) raises -> Float32:
    """`float r2_score_py(handle, float* y, float* y_hat, int n)`."""
    return _raft_r2(ctx, y, y_hat, n)
