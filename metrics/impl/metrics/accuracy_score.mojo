# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""cuML `cpp/src/metrics/accuracy_score.cu` (265b9da): `ML::Metrics::accuracy_score_py` forwards to `raft::stats::accuracy`. The `int` overloads (cuML's Python passes int32 labels)."""

from max.gpu.host import DeviceBuffer, DeviceContext

from metrics.impl.stats.detail.scores import accuracy_score as _raft_accuracy


def accuracy_score_py(
    ctx: DeviceContext,
    mut predictions: DeviceBuffer[DType.int32],
    mut ref_predictions: DeviceBuffer[DType.int32],
    n: Int,
) raises -> Float32:
    """`float accuracy_score_py(handle, const int* predictions, const int* ref_predictions, int n)`."""
    return _raft_accuracy(ctx, predictions, ref_predictions, n)
