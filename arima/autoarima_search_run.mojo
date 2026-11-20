# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""RUN body of `providers/python/autoarima_search_pyapi.mojo` (wave 8 run
split): the AutoARIMA information-criterion penalty, moved unchanged."""

from x_metrics.epilogue import portable_log_c


def _penalty(complexity: Int, criterion: String, observations: Int, d: Int, D: Int, period: Int) raises -> Float64:
    var size = Float64(complexity)
    var n = Float64(observations - d - period * D)
    if criterion == "aic":
        return 2.0 * size
    if criterion == "aicc":
        if n - size - 1.0 == 0:
            raise Error("ZeroDivisionError: AutoARIMA information criterion denominator is zero")
        return 2.0 * size + 2.0 * size * (size + 1.0) / (n - size - 1.0)
    return portable_log_c(n) * size
