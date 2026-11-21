# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPython boundary for the time-series lanes: Holt-Winters and the KPSS test.

Kept in a separate extension, deliberately, so an independently changing
binding does not become a merge point (the same reason
`_mojolearn_estimators.mojo` is separate from `_mojolearn.mojo`). Arrays
cross as borrowed NumPy addresses; all device buffers and contexts live for
one call and no pointer is retained.

WHAT IS BEHIND EACH ENTRY, AND WHAT IS NOT

    holtwinters_fit       holtwinters/, DEVIATIONS 660-665 and 697-699.
                          cuML `ML::HoltWinters::fit`, the seasonal BFGS
                          arm, float32.
    holtwinters_forecast  the same lane's `ML::HoltWinters::forecast`.
    kpss_test             tsa/, DEVIATIONS 671-672. cuML
                          `ML::Stationarity::kpss_test`, float32.
    select_d              the same lane's implementation of auto_arima's "Choose the
                          hyper-parameter d" block.

`arima/` IS SERVED BY ITS OWN EXTENSION, `bindings/_mojolearn_arima.mojo`
(the eleventh), not by this one, so that two independently changing
bindings never become a merge point. A paragraph here used to say `arima/`
was "deliberately absent" from the package because the lane had no `fit`;
`estimate_x0` and the batched L-BFGS landed 2026-09-01, `mojolearn.ARIMA`
is exported from the package (`python/mojolearn/_arima_impl.py`), and the
false paragraph is DELETED, which is the correction `arima/README.md`'s
hand-off section asked for. There is still no `ARIMA` entry point in THIS
module, and there is not supposed to be.

THE ARGUMENT-COUNT RULE. `PythonModuleBuilder.def_function` infers its
signature from arity and stops working somewhere above nine arguments, so
buffer addresses go positionally and every scalar goes in ONE list. The
order of that list is written out in a comment on both sides of the
boundary IN THE SAME WORDS, here and in `python/mojolearn/_tsa_impl.py`.
A silently reordered list is a wrong answer rather than a failure, which
makes it the most dangerous thing at this boundary; the two comments are
meant to be diffable by eye.

THE GIL is released around every device call.
"""

# DEVIATION 2486: shared byte-preserving host copies.
from core.experiment_guards_series import SERIES_GUARDS_OK
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder

from checks.vendor import COMPILED_VENDOR
from checks.numerics import GLOBAL_NUMERIC_MODE

from holtwinters.resident_state import holtwinters_resident_count
from tsa.tsa_ops_device import publish_tsa_hw_ops_device


def holtwinters_resident_count_binding() raises -> PythonObject:
    """Live resident component sets (the tools/ lifetime check reads it)."""
    return PythonObject(holtwinters_resident_count())


def tsa_numeric_mode_binding() raises -> PythonObject:
    """THE BUILD'S TIER, as the `NUMERIC_*` code itself: 0 FAST, 1
    IDENTICAL; the same read-back as `svm_numeric_mode`.
    box-run-2 (2026-10-05): this binding exported no `*_numeric_mode`, so the
    wave native import smoke (tools/identical_wave_native_build.py) refused it
    in both arms. Read-back only; no kernel or bits change."""
    return PythonObject(GLOBAL_NUMERIC_MODE)


def tsa_vendor_binding() raises -> PythonObject:
    """THE ACCELERATOR API THIS BINARY WAS COMPILED FOR: 'metal', 'cuda',
    'hip' or 'none'. A compile-time constant folded in from
    `checks/vendor.mojo`, the same shape as the tier read-back
    (`gbdt_numeric_mode`): the answer comes from the binary that actually
    loaded, never from the directory it sat in or from the environment.
    `python/mojolearn/_backend.py` refuses at import when this disagrees
    with the vendor directory the set was loaded from."""
    return PythonObject(String(COMPILED_VENDOR))


# The module body. Its @export PyInit lives in bindings/entry/entry__mojolearn_tsa.mojo
# (the per-binding build) and in the merged roots tools/merged_bindings.py
# writes from bindings/merged_groups.json (one shared file per directory set).
def build__mojolearn_tsa() -> PythonObject:
    comptime assert SERIES_GUARDS_OK, "invalid integrated experiment configuration (series guards)"
    try:
        var m = PythonModuleBuilder("_mojolearn_tsa")
        m.def_function[tsa_vendor_binding]("tsa_vendor")
        m.def_function[tsa_numeric_mode_binding]("tsa_numeric_mode")
        # TOMBSTONE: dead Python exports holtwinters_fit, holtwinters_forecast, holtwinters_predict (no product caller: the ExponentialSmoothing owner reaches these kernels through the published Holt-Winters operations table, core/tsa_entry_table.mojo; lane dw-forecast-tail) deleted 2026-11-21; code recoverable at 8e1fedd052.
        # Restore: git apply experiments/removed/dead_entries_tsa_hw_calls.patch
        # The in-sample prediction runs on the device (one thread per cell);
        # the host bindings keep bindings/holtwinters_host_predict.mojo.
        m.def_function[holtwinters_resident_count_binding]("holtwinters_resident_count")
        # TOMBSTONE: dead Python exports holtwinters_resident_prepare, holtwinters_resident_release, holtwinters_query_resident (no caller: the ExponentialSmoothing owner reaches the resident operations through the published Holt-Winters operations table, core/tsa_entry_table.mojo; lane dw-forecast-tail) deleted 2026-11-20; code recoverable at 3b1a75cca0.
        # Restore: git apply experiments/removed/dead_entries_tsa_hw_resident.patch
        # TOMBSTONE: dead Python exports kpss_test, select_d (no product caller: the stationarity functions and AutoARIMA's search reach these kernels through the published tsa operations table, core/tsa_entry_table.mojo TSA_OP_KPSS_TEST / TSA_OP_SELECT_D; lane dw-forecast-tail) deleted 2026-11-21; code recoverable at 038b12431e.
        # Restore: git apply experiments/removed/dead_entries_tsa_kpss_calls.patch
        # The Holt-Winters operations table (core/tsa_entry_table.mojo): the
        # ExponentialSmoothing owner resolves it once per estimator and calls
        # through it; published once the module is ready.
        var published_module = m.finalize()
        publish_tsa_hw_ops_device()
        return published_module
    except e:
        abort(String("failed to create _mojolearn_tsa: ", e))
