# SPDX-License-Identifier: Apache-2.0
"""Validated clustering configuration, separate from archive/public attributes."""
from core.spectral_native_api import SpectralFitOptions


@fieldwise_init
struct SpectralClusteringConfig(Copyable, Movable):
    var fit: SpectralFitOptions
    var affinity: String
    var assignment: Int
    var gamma: Float64


def spectral_assignment_method(name: String) raises -> Int:
    if name == "kmeans":
        return -1
    if name == "discretize":
        return 0
    if name == "cluster_qr":
        return 1
    raise Error("Spectral assignment requires an admitted method name")
