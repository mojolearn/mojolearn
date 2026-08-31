# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`ML::KDE::score_samples`: cuML 26.08's C++ entry, over the 25.08 algorithm.

PORT OF cuML `cpp/include/cuml/neighbors/kde.hpp` and `cpp/src/kde/kde.cu`
at cuML `265b9da` (v26.08.00). Partial. Do not improve.

Their file is a delegation: it wraps the six pointers in mdspans and calls
`cuvs::distance::kde(handle, query, train, weights, output, bandwidth,
sum_weights, kernel, metric, metric_arg)` (`kde.cu:45-55`). That cuVS entry
is in cuVS 26.08, which this tree's pinned checkout (25.08, `94c2819`) does
not have, so the delegate here is `kde/derived/neighbors/kernel_density.mojo`
-- the 25.08 Python-layer algorithm the fused kernel reproduces. The SHAPE
of this entry is theirs: the `DensityKernelType` values, the argument order,
`weights` nullable, `sum_weights` supplied by the caller (`kde.hpp:44`:
"sum of `weights`, or `n_train` when weights is null"), `metric` as a
DistanceType value. `metric_arg` (Minkowski's `p`) is accepted and REFUSED
unless it is the default `2.0`, because no ported metric reads it.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from core.identity_trace import IdentityTrace
from kde.derived.neighbors.kernel_density import (
    KDE_ELEM_TPB,
    KDE_KERNEL_COSINE,
    KDE_KERNEL_EPANECHNIKOV,
    KDE_KERNEL_EXPONENTIAL,
    KDE_KERNEL_GAUSSIAN,
    KDE_KERNEL_LINEAR,
    KDE_KERNEL_TOPHAT,
    KDE_LSE_TPB,
    kde_score_samples_device,
)

#: `enum class DensityKernelType : int` (`kde.hpp:17-24`), their values.
comptime DensityKernelType_Gaussian = KDE_KERNEL_GAUSSIAN
comptime DensityKernelType_Tophat = KDE_KERNEL_TOPHAT
comptime DensityKernelType_Epanechnikov = KDE_KERNEL_EPANECHNIKOV
comptime DensityKernelType_Exponential = KDE_KERNEL_EXPONENTIAL
comptime DensityKernelType_Linear = KDE_KERNEL_LINEAR
comptime DensityKernelType_Cosine = KDE_KERNEL_COSINE


def score_samples(
    ctx: DeviceContext,
    mut query: DeviceBuffer[DType.float32],
    mut train: DeviceBuffer[DType.float32],
    mut weights: DeviceBuffer[DType.float32],
    has_weights: Bool,
    mut output: DeviceBuffer[DType.float32],
    n_query: Int,
    n_train: Int,
    n_features: Int,
    bandwidth: Float32,
    sum_weights: Float32,
    kernel: Int,
    metric: Int,
    metric_arg: Float32,
    mut trace: IdentityTrace,
    elem_tpb: Int = KDE_ELEM_TPB,
    lse_tpb: Int = KDE_LSE_TPB,
) raises:
    """`ML::KDE::score_samples<float>(handle, query, train, weights, output,
    n_query, n_train, n_features, bandwidth, sum_weights, kernel, metric,
    metric_arg)` (`kde.hpp:49-62`, `kde.cu:15-56`).

    `weights == nullptr` is `has_weights == False` (Mojo's launch refuses a
    null pointer argument, so the buffer is passed and ignored). The `T =
    double` instantiation is NOT ported: no float64 on the device column
    this tree is built on (DEVIATION 600).
    """
    if metric_arg != Float32(2.0):
        raise Error(
            "kde: metric_arg="
            + String(metric_arg)
            + " is read only by metric='minkowski', which is NOT PORTED;"
            " pass 2.0 (the default)"
        )
    kde_score_samples_device(
        ctx,
        train,
        query,
        weights,
        has_weights,
        sum_weights,
        n_train,
        n_query,
        n_features,
        bandwidth,
        kernel,
        metric,
        output,
        trace,
        elem_tpb,
        lse_tpb,
    )
