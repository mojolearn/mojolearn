# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Choosing the differencing order `d` with the stationarity test.

PORT OF the "Choose the hyper-parameter d" block of
`python/cuml/cuml/tsa/auto_arima.pyx::AutoARIMA.search` at cuML 265b9da6
(v26.08.00), lines 318-343 ONLY. That block is HOST Python over the device
`kpss_test`:

    for d_ in d_options[:-1]:                 # d_options = 0 .. 2 - D
        mask = kpss_test(data_temp, d_, D_, s) # stationary after d_ diffs?
        out0, index0, out1, index1 = _divide_by_mask(data_temp, mask, id_temp)
        if out1 is not None: data_dD[(d_, D_)] = (out1, index1)   # decided
        if out0 is not None: (data_temp, id_temp) = (out0, index0) # continue
        else: break
    else:
        data_dD[(d_options[-1], D_)] = (data_temp, id_temp)        # the rest

`_divide_by_mask` (`auto_arima.pyx:552`, a `divide_by_mask` kernel in
`cpp/src/tsa/auto_arima.cu`) physically splits the batch so the next test
runs on the sub-batch of undecided series. OURS runs every round on the
FULL batch and masks on the host, which is the same answer because the
test is per series and its result is a pure function of the series bits --
the property `check_kpss_batch_composition_invariant` gates (two batch
compositions, same cell, same bytes). The splitting kernels
(`divide_by_mask`, `divide_by_min`, `build_division_map`, `merge_series`)
belong to auto_arima's p/q/P/Q search and are listed in `NOT_IMPLEMENTED.tsv`.

NOT PORTED, REFUSED BY NAME: the choice of `D` (`seasonal_test="seas"`
calls statsmodels' STL on the HOST, `seasonality.py:37-67`); `D` must be
given. The rest of `search` (the IC-driven p/q/P/Q/k grid over `ARIMA`
fits) is auto_arima's rung 4 and is not reached.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from tsa.derived.tsa.stationarity import kpss_test
from tsa.derived.timeSeries.stationarity import download_results


def select_d(
    ctx: DeviceContext,
    mut d_y: DeviceBuffer[DType.float32],
    batch_size: Int,
    n_obs: Int,
    D: Int,
    s: Int,
    d_max: Int = 2,
    pval_threshold: Float32 = 0.05,
) raises -> List[Int32]:
    """`d` per series in `0 .. d_max` (their `d_options = range(0, 2 - D + 1)`
    when the user passes a range; `d_max = 2 - D` is their default)."""
    if d_max < 0 or d_max + D > 2:
        raise Error(
            "select_d: d_max must satisfy 0 <= d_max <= 2 - D (d_max="
            + String(d_max) + ", D=" + String(D) + "), refused by name"
        )
    var chosen = List[Int32]()
    var decided = List[Bool]()
    for _ in range(batch_size):
        chosen.append(Int32(d_max))
        decided.append(False)
    var remaining = batch_size
    for d_ in range(d_max):
        if remaining == 0:
            break
        var res = kpss_test(ctx, d_y, batch_size, n_obs, d_, D, s, pval_threshold)
        var rs = download_results(ctx, res, batch_size)
        for b in range(batch_size):
            if not decided[b] and rs[0][b]:
                chosen[b] = Int32(d_)
                decided[b] = True
                remaining -= 1
    return chosen^
