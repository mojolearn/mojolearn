"""RAFT `cpp/include/raft/stats/detail/homogeneity_score.cuh` (ebf9268).

THEIRS (:32-57):

    if (size == 0) return 1.0;
    MI = mutual_info_score(truth, pred, size, lower, upper)
    H  = entropy(truth, size, lower, upper)
    return H ? MI / H : 1.0

sklearn (`homogeneity_completeness_v_measure`): `homogeneity = MI /
entropy_C if entropy_C else 1.0` -- the same; `completeness` is the same
call with the two arrays swapped (`completeness_score.cu` does exactly
that, and so does `v_measure`). The division is one correctly-rounded
Float64 op on the host; the two operands carry DEVIATIONS 650/651.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from metrics.ported.stats.detail.entropy import entropy
from metrics.ported.stats.detail.mutual_info_score import mutual_info_score


def homogeneity_score(
    ctx: DeviceContext,
    mut truth_cluster_array: DeviceBuffer[DType.int32],
    mut pred_cluster_array: DeviceBuffer[DType.int32],
    size: Int,
    lower_label_range: Int32,
    upper_label_range: Int32,
) raises -> Float64:
    """`homogeneity_score(truth, pred, size, lower, upper, stream)` (:32-57)."""
    if size == 0:
        return 1.0
    var computed_mi = mutual_info_score(
        ctx,
        truth_cluster_array,
        pred_cluster_array,
        size,
        lower_label_range,
        upper_label_range,
    )
    var computed_entropy = entropy(
        ctx, truth_cluster_array, size, lower_label_range, upper_label_range
    )
    if computed_entropy != 0.0:
        return computed_mi / computed_entropy
    return 1.0
