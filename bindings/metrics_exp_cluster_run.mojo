# SPDX-License-Identifier: Apache-2.0
# RUN part of bindings/metrics_exp_cluster.mojo: typed Mojo only (no Python
# imports). The binary64 epilogue of NMI and AMI.
from std.math import sqrt


def _generalized_average(U: Float64, V: Float64, method: Int) raises -> Float64:
    if method == 0:
        return V if V < U else U
    if method == 1:
        var x = U * V
        if x < 0.0:
            raise Error("math domain error")
        return sqrt(x)
    if method == 2:
        return (U + V) / 2
    return V if V > U else U
