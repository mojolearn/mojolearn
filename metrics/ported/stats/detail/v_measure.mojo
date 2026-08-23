"""RAFT `cpp/include/raft/stats/detail/v_measure.cuh` (ebf9268).

THEIRS (:27-50):

    h = homogeneity_score(truth, pred, ...)
    c = homogeneity_score(pred, truth, ...)            (= completeness)
    if (c + h == 0.0) v = 0.0
    else              v = (1 + beta) * h * c / (beta * h + c)

sklearn: `if homogeneity + completeness == 0.0: v = 0.0 else (1 + beta) *
homogeneity * completeness / (beta * homogeneity + completeness)` -- the
same, `beta = 1.0` default on both. Host Float64: two multiplies, a
multiply-add and a division; correctly rounded everywhere, and the
`beta * h + c` is spelled as a bare multiply then add on the host in both
modes (no device codegen can reach it; the host compiler does not
contract across this expression in Mojo's default). The NaN case of
theirs (h or c NaN from an MI/entropy NaN) propagates as theirs does.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from metrics.ported.stats.detail.homogeneity_score import homogeneity_score


def v_measure(
    ctx: DeviceContext,
    mut truth_cluster_array: DeviceBuffer[DType.int32],
    mut pred_cluster_array: DeviceBuffer[DType.int32],
    size: Int,
    lower_label_range: Int32,
    upper_label_range: Int32,
    beta: Float64 = 1.0,
) raises -> Float64:
    """`v_measure(truth, pred, size, lower, upper, stream, beta)` (:27-50)."""
    var computed_homogeneity = homogeneity_score(
        ctx,
        truth_cluster_array,
        pred_cluster_array,
        size,
        lower_label_range,
        upper_label_range,
    )
    var computed_completeness = homogeneity_score(
        ctx,
        pred_cluster_array,
        truth_cluster_array,
        size,
        lower_label_range,
        upper_label_range,
    )
    if computed_completeness + computed_homogeneity == 0.0:
        return 0.0
    var num = (1.0 + beta) * computed_homogeneity * computed_completeness
    var den = beta * computed_homogeneity + computed_completeness
    return num / den
