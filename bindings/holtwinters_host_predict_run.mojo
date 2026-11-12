# SPDX-License-Identifier: Apache-2.0
"""RUN of the Holt-Winters host forecast and in-sample predict doors
(bindings/holtwinters_host_predict.mojo): the state guards and the typed
forecast/predict over the packed components, run by the doors with the GIL
released. No Python import (tools/python_free_dirs.sh). Lane rs-shared-infra-b
(2026-11-12): moved word for word; the kernels it calls are unchanged."""
from core.holtwinters_host_predict_impl import hw_store_values
from holtwinters.host.hw_predict import (
    hw_forecast_from_state_ptr,
    hw_predict_in_sample_ptr,
)
from holtwinters.impl.tsa.holtwinters_params import (
    SEASONAL_ADDITIVE,
    seasonal_from_name,
)


def _refuse_state(n: Int, batch_size: Int, frequency: Int, who: String) raises:
    """`holtwinters_forecast_ptr`'s guards (`holtwinters/estimator.mojo`),
    in its words and order, after `seasonal_from_name`."""
    if n <= frequency:
        raise Error(
            "holtwinters "
            + who
            + ": n ("
            + String(n)
            + ") must exceed frequency ("
            + String(frequency)
            + "); there would be no fitted components"
        )
    if batch_size < 1:
        raise Error(
            "holtwinters "
            + who
            + ": batch_size must be >= 1 (batch_size="
            + String(batch_size)
            + ")"
        )


def holtwinters_forecast_run(cp: MutPointer[Float32, MutUntrackedOrigin], op: MutPointer[Float32, MutUntrackedOrigin],
                             n: Int, batch_size: Int, frequency: Int, h: Int, sname: String) raises -> Int:
    """`holtwinters_forecast_binding`'s GIL-released body: the guards, then
    `h * batch_size` forecast values written to `op`. Returns that count."""
    var written = 0
    var st = seasonal_from_name(sname)
    _refuse_state(n, batch_size, frequency, "forecast")
    if h <= 0:
        raise Error("h must be > 0. Currently: " + String(h))
    var components_len = (n - frequency) * batch_size
    var fc = hw_forecast_from_state_ptr(
        cp,
        components_len,
        n,
        batch_size,
        frequency,
        st == SEASONAL_ADDITIVE,
        h,
    )
    hw_store_values(op, fc, h * batch_size)
    written = h * batch_size
    return written


def holtwinters_predict_run(cp: MutPointer[Float32, MutUntrackedOrigin], op: MutPointer[Float32, MutUntrackedOrigin],
                            n: Int, batch_size: Int, frequency: Int, start: Int, end: Int, sname: String) raises -> Int:
    """`holtwinters_predict_binding`'s GIL-released body: the guards, then
    `(end - start) * batch_size` in-sample predictions written to `op`.
    Returns that count."""
    var written = 0
    var st = seasonal_from_name(sname)
    _refuse_state(n, batch_size, frequency, "predict")
    if start < 0 or end <= start:
        raise Error(
            "holtwinters predict: need 0 <= start < end (start="
            + String(start)
            + ", end="
            + String(end)
            + ")"
        )
    if end > n:
        raise Error(
            "holtwinters predict: the in-sample prediction ends at n (end="
            + String(end)
            + ", n="
            + String(n)
            + "); later times are the forecast's"
        )
    var components_len = (n - frequency) * batch_size
    var out = hw_predict_in_sample_ptr(
        cp,
        components_len,
        n,
        batch_size,
        frequency,
        st == SEASONAL_ADDITIVE,
        start,
        end,
    )
    written = (end - start) * batch_size
    hw_store_values(op, out, written)
    return written
