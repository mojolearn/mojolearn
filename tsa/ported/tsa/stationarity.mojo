# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Derivative work: the upstream file and its pinned commit are recorded in this lane's PORTED_MAP.tsv and in this file's own docstring. See NOTICE.
"""`ML::Stationarity::kpss_test`: the public entry.

PORT OF `cuml/cpp/src/tsa/stationarity.cu` at cuML 265b9da6 (v26.08.00),
the `float` overload (`:36-48`) and its helper (`:17-32`): a pass-through
to `MLCommon::TimeSeries::kpss_test`. The `double` overload is not offered
(no Float64 on Metal; DEVIATION 670 in `arima/`). The Python surface
(`python/cuml/cuml/tsa/stationarity.pyx::kpss_test(y, d, D, s,
pval_threshold)`) maps one to one onto the arguments below.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from tsa.ported.timeSeries.stationarity import KPSS_ELEM_TPB, KpssResult
from tsa.ported.timeSeries.stationarity import kpss_test as _prim_kpss_test


def kpss_test(
    ctx: DeviceContext,
    mut d_y: DeviceBuffer[DType.float32],
    batch_size: Int,
    n_obs: Int,
    d: Int,
    D: Int,
    s: Int,
    pval_threshold: Float32 = 0.05,
    elem_tpb: Int = KPSS_ELEM_TPB,
) raises -> KpssResult:
    """`stationarity.cu:36-48`."""
    return _prim_kpss_test(ctx, d_y, batch_size, n_obs, d, D, s, pval_threshold, elem_tpb)
