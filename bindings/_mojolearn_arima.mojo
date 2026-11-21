# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPython boundary for the batched ARIMA lane.

AN ELEVENTH EXTENSION MODULE, and a separate one on purpose. The header of
`bindings/_mojolearn_estimators.mojo` states the reason and it is the same
reason here, an independently changing binding must not become a merge
point. `arima/` is cuML's `cpp/src/arima/`, the batched Kalman filter, the
Jones transform, `estimate_x0` and an own-written batched L-BFGS, and it
shares no code with the Holt-Winters and KPSS lanes that
`bindings/_mojolearn_tsa.mojo` carries. All eleven land in one wheel.

WHAT THIS CLOSES. `bindings/_mojolearn_tsa.mojo` says in its own header that
`arima/` is deliberately absent because the lane has no `fit` and an `ARIMA`
class would have to ask the caller for the answer before computing it. THAT
STOPPED BEING TRUE when `arima/impl/batched_fit.mojo` landed
(2026-09-01) with `estimate_x0`, the Householder QR least squares and the
batched L-BFGS, all gated by `arima/checks/fit_check.mojo` in both numeric
tiers. What was missing after that was only the Python door, and this file
is the middle third of it. THE STALE PARAGRAPH IN `_mojolearn_tsa.mojo` AND
THE MATCHING ONE IN `python/mojolearn/_tsa_impl.py` WERE DELETED 2026-09-02
(the deletion `arima/README.md`'s hand-off section requested); this
sentence used to say they were owed and edited by another lane's owner.

Arrays cross as borrowed NumPy addresses; all device buffers and contexts
live for one call and no pointer is retained. The Python wrapper owns the
arrays and keeps them alive for the duration of the call
(`python/mojolearn/_arrays.py` is where that contract is written down).

SCALARS ARRIVE AS ONE LIST, NOT AS SEPARATE ARGUMENTS.
`PythonModuleBuilder.def_function` infers its signature from arity and stops
working above roughly nine arguments, so buffer addresses go positionally
and every scalar goes in one `params` list. THE ORDER OF THAT LIST IS
WRITTEN OUT IN A COMMENT ON BOTH SIDES IN THE SAME WORDS. A silent
reordering is a wrong answer, not a failure, and it is the most dangerous
thing at this boundary because nine of the thirteen entries are small
non-negative integers that would all look plausible in each other's slots.
The length check on every entry below is what stands between a swapped pair
and a believable number, and the three lists are deliberately the same
length and the same tail so the two comments diff by eye.

WHAT IS REFUSED, AND WHERE. Nothing is refused in this file except a null
address and a params list of the wrong length. Every model refusal lives one
or two layers down and is raised there by name, which is what keeps it
reachable from every caller and not only from Python:

    n_exog < 0 or > 17   `arima/impl/tsa/arima_common.mojo::validate_order`
    non-finite exog      `arima/exog_layout.mojo`, naming the
                         series, row and regressor (DEVIATION 997)
    method css, css-ml   `arima/estimator.mojo::_refuse_method`
    rd > 8               `validate_order`, cuML's block-per-series Kalman
    r > 5                `validate_order`, cuML's Schur Lyapunov solver
    d + D > 2, s < 2,
    p q P Q > 8,
    an empty order       `validate_order`, with the reference's message
    a non-finite series  `batched_arima.mojo::_refuse_non_finite`, with the
                         flat index of the offender
    float64              by dtype, DEVIATION 670. Metal exposes no Float64
                         on the device and every kernel here is float32

DEVIATIONS 990-999 are this surface's. 990 to 993 are used and each is named
where it bites; 994 to 998 are the exogenous regressors' (lane/arima-exog,
2026-09-15: 994 EXOG_MAX, 995 the observation intercept's fold, 996 the
exog layout, 997 the non-finite exog refusal, 998 the saved-model format);
999 is unassigned.

`ARIMA` IS wired into `python/mojolearn/__init__.py` and
`_mojolearn_arima` IS registered in `python/mojolearn/_backend.py`'s
`_MODULES` and its `_build_script` dict. Both halves, because forgetting
either is silent: DEVIATION 869 records that an extension missing from
`_MODULES` is never re-pointed, so under `identical` a plain import resolves
to the FAST binary sitting beside it and returns fast arithmetic under the
identical label.

THE GIL is released around every device call, and nothing inside a
`GILReleased` block touches a `PythonObject`.
"""

# DEVIATION 2486: shared byte-preserving host copies.
from core.experiment_guards_series import SERIES_GUARDS_OK
from std.os import abort
from std.python import PythonObject
# TOMBSTONE: MOJOLEARN_ARIMA_FAST_K3_DF_PROBE (never measured, deleted) deleted 2026-11-09 by eff59d1b88; code
# recoverable at f59d913b0b (the `_arima_k3_df_probe` export). Same commit: MOJOLEARN_ARIMA_FAST_PRODUCT_DF_TAIL and its
# `arima_product_df_count` reach counter.
# Restore: git apply experiments/removed/MOJOLEARN_ARIMA_FAST_K3_DF_PROBE.patch
from std.python.bindings import PythonModuleBuilder

from checks.numerics import GLOBAL_NUMERIC_MODE
from checks.vendor import COMPILED_VENDOR

from arima.arima_ops_device import publish_arima_ops_device
from arima.resident_state import arima_resident_count


# TOMBSTONE: MOJOLEARN_ARIMA_FAST_GROUPS_CONCURRENT / _D_CONCURRENT (DROPPED-slower) deleted 2026-10-09 on
# lane/owed-deletions-D1 (arima_d_concurrent_enabled, arima_order_search_multi); code recoverable at b639a2bd2.
# Restore: git apply experiments/removed/MOJOLEARN_ARIMA_FAST_GROUPS_CONCURRENT.patch; record in docs/TOMBSTONES.md.

# TOMBSTONE: MOJOLEARN_ARIMA_FAST_CSS_SEARCH (DROPPED-quality) and MOJOLEARN_ARIMA_FAST_STEPWISE (DROPPED-slower) deleted
# 2026-10-09 on lane/owed-deletions-D1 (arima_order_search_css / arima_order_search_stepwise); code recoverable at b639a2bd2.
# Restore: git apply experiments/removed/MOJOLEARN_ARIMA_FAST_CSS_SEARCH.patch; record in docs/TOMBSTONES.md.


def arima_numeric_mode_binding() raises -> PythonObject:
    """THE BUILD'S TIER, as the `NUMERIC_*` code itself, 0 FAST, 1
    IDENTICAL. The same shape as `gbdt_numeric_mode`, and
    with the same reason: the wrapper reads it once and refuses to run if the
    binary it loaded disagrees with the mode the package asked for. A
    wrong-arm measurement that is correctly labelled by accident is the
    failure this prevents."""
    return PythonObject(GLOBAL_NUMERIC_MODE)


def arima_vendor_binding() raises -> PythonObject:
    """THE ACCELERATOR API THIS BINARY WAS COMPILED FOR, 'metal', 'cuda',
    'hip' or 'none'. A compile-time constant folded in from
    `checks/vendor.mojo`, the same shape as the tier read-back. The answer
    comes from the binary that actually loaded, never from the directory it
    sat in and never from the environment.
    `python/mojolearn/_backend.py` refuses at import when this disagrees
    with the vendor directory the set was loaded from."""
    return PythonObject(String(COMPILED_VENDOR))


def arima_resident_count_binding() raises -> PythonObject:
    """Live resident fitted states (the tools/ lifetime check reads it)."""
    return PythonObject(arima_resident_count())


def build__mojolearn_arima() -> PythonObject:
    comptime assert SERIES_GUARDS_OK, "invalid integrated experiment configuration (series guards)"
    try:
        var m = PythonModuleBuilder("_mojolearn_arima")
        m.def_function[arima_vendor_binding]("arima_vendor")
        m.def_function[arima_numeric_mode_binding]("arima_numeric_mode")
        m.def_function[arima_resident_count_binding]("arima_resident_count")
        # TOMBSTONE: dead Python exports arima_fit, arima_ic_from_loglike, arima_order_batch_enabled, arima_order_search, arima_order_caps, arima_order_search_device, arima_search_reuse_enabled, arima_order_search_fit, arima_fast_search_mode, arima_ndiffs_kpss, arima_predict, arima_forecast, arima_resident_prepare, arima_resident_release, arima_predict_resident, arima_forecast_resident (no product caller: the ARIMA and AutoARIMA owners reach these kernels through the published ARIMA operations table, core/arima_entry_table.mojo; lane dw-forecast-tail) deleted 2026-11-21; code recoverable at SHA_PLACEHOLDER.
        # Restore: git apply experiments/removed/dead_entries_arima_calls.patch
        # The ARIMA operations table of this image (core/arima_entry_table.mojo):
        # an estimator's route resolves it once.
        var published_module = m.finalize()
        publish_arima_ops_device()
        return published_module
    except e:
        abort(String("failed to create _mojolearn_arima: ", e))
