# SPDX-License-Identifier: Apache-2.0
"""PACK of member FIT NOTICES (core/member_notices.mojo): the notice words a
fitted member state carried, raised as the warnings the family's own fit
door raises, in its words, category and order. The one raiser of every
caller (model selection units, the member set, recursive feature
elimination); no family PACK raises them (lane fx-audit 19 / 26).

SERIAL BY NATURE: a few warnings per unit, GIL held, on the public thread.
"""
from std.python import PythonObject
from bindings.classical_warnings_unpack import classical_warning
from bindings.kernel_methods_boundary_unpack import _km_warning_location
from bindings.metrics_native_support_door import metric_warn, metric_undefined
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call
from bindings.rf_fit_notices_pack import rf_fit_notices_word
from core.member_notices import (
    MEMBER_NOTICE_WORDS, MEMBER_NOTICE_FOREST, MEMBER_NOTICE_NYSTROEM_COMPONENTS, MEMBER_NOTICE_GP_RIDGE,
    MEMBER_NOTICE_PLS_RESIDUAL, MEMBER_NOTICE_SPLIT_LEAST, MEMBER_NOTICE_METRIC_PRF, MEMBER_NOTICE_METRIC_JACCARD,
    MEMBER_NOTICE_METRIC_BALANCED, MEMBER_NOTICE_METRIC_FEW, MEMBER_NOTICE_METRIC_TOP_K, MEMBER_NOTICE_R2_FEW,
)
from bindings.py_balanced import py_get


def _gp_ridge_notices(start: Float64, used: Float64, context: PythonObject) raises:
    # `_gpr_fit`'s ridge retries (bindings/gp_regressor_native_door.mojo): one
    # RuntimeWarning per tenfold step from the starting alpha to the one used.
    var b_mod = native_loaded_module("builtins")
    var alpha = start
    while alpha < used and alpha > 0.0:  # driver-loop: over the tenfold ridge retries (at most a few)
        var raised = min(alpha * 10.0, 1.0)
        var message = (String("mojolearn GaussianProcessRegressor: K + alpha*I is not positive ")
                       + "definite in float32 with alpha=" + String(py_call(b_mod.format, PythonObject(alpha), "g"))
                       + "; refitting with alpha=" + String(py_call(b_mod.format, PythonObject(raised), "g"))
                       + ". Pass a larger alpha to skip this step.")
        metric_warn(message, b_mod.RuntimeWarning, context)
        alpha = raised


def member_fit_notices_raise(words: List[Float64], context: PythonObject) raises:
    """Raise every notice of `words` (`member_notices_take`'s triples) with
    the member's `context`. Empty: nothing is read."""
    member_fit_notices_raise_with(words, context, context)


def _metric_notice(code: Int, a: Float64, b: Float64, context: PythonObject) raises:
    """A scorer's metric warning (lane mf-score5), in the public metric
    door's class and words (each door named beside its code in
    core/member_notices.mojo)."""
    if code == MEMBER_NOTICE_METRIC_PRF:
        var which = Int(a)
        var name = String("Precision") if which == 0 else (String("Recall") if which == 1 else String("F-score"))
        metric_undefined(name + " is ill-defined and being set to 0.0 due to zero "
                         + "denominator; use zero_division to control this behavior.", context)
    elif code == MEMBER_NOTICE_METRIC_JACCARD:
        metric_undefined("Jaccard is ill-defined and being set to 0.0 in labels with no true or "
                         + "predicted samples. Use `zero_division` parameter to control this behavior.", context)
    elif code == MEMBER_NOTICE_METRIC_BALANCED:
        metric_warn("y_pred contains classes not in y_true", native_loaded_module("builtins").UserWarning, context, 2)
    elif code == MEMBER_NOTICE_METRIC_FEW:
        var shown = String("R^2") if Int(a) == 0 else String("D^2")
        metric_undefined(shown + " score is not well-defined with less than two samples.", context)
    else:
        metric_undefined("'k' (" + String(Int(a)) + ") greater than or equal to 'n_classes' (" + String(Int(b))
                         + ") will result in a perfect score and is therefore meaningless.", context)


def member_fit_notices_raise_with(words: List[Float64], context: PythonObject,
                                  metric_context: PythonObject) raises:
    """`member_fit_notices_raise` with the context the scorers' metric
    warnings are raised with (`metric_context`: the search's own, which
    carries the metrics' warning class, as its scorers' calls do)."""
    var count = len(words) // MEMBER_NOTICE_WORDS
    for i in range(count):  # small-loop(a unit's notices: a few)
        var code = Int(words[MEMBER_NOTICE_WORDS * i])
        var a = words[MEMBER_NOTICE_WORDS * i + 1]
        var b = words[MEMBER_NOTICE_WORDS * i + 2]
        if code == MEMBER_NOTICE_FOREST:
            rf_fit_notices_word(Int(a), context)
        elif code == MEMBER_NOTICE_NYSTROEM_COMPONENTS:
            # `km_nys_fit_unpack` / sklearn Nystroem.fit.
            classical_warning("n_components > n_samples. This is not possible.\n"
                              + "n_components was set to n_samples, which results"
                              + " in inefficient evaluation of the full kernel.",
                              native_loaded_module("builtins").UserWarning, context,
                              _km_warning_location())
        elif code == MEMBER_NOTICE_GP_RIDGE:
            _gp_ridge_notices(a, b, context)
        elif code == MEMBER_NOTICE_PLS_RESIDUAL:
            # `pls_fit_pack`.
            metric_warn("y residual is constant at iteration " + String(Int(a)),
                        native_loaded_module("builtins").UserWarning, context)
        elif code == MEMBER_NOTICE_SPLIT_LEAST:
            # The splitter door's warning (bindings/msel_split_door.mojo
            # `msel_split_finish`: `msel_warn(plan.warning, UserWarning)`), in
            # `msel_plan_stratified`'s words.
            metric_warn("The least populated class in y has only " + String(Int(a)) + " members, which is less "
                        + "than n_splits=" + String(Int(b)) + ".",
                        native_loaded_module("builtins").UserWarning, context, 2)
        elif code >= MEMBER_NOTICE_METRIC_PRF and code <= MEMBER_NOTICE_METRIC_TOP_K:
            _metric_notice(code, a, b, metric_context)
        elif code == MEMBER_NOTICE_R2_FEW:
            # RidgeCV's split path (bindings/x_linear_cv_pack.mojo
            # `xl_ridge_cv_splits_finish` -> `_xl_undefined_warning`):
            # scikit-learn's UndefinedMetricWarning when sklearn.exceptions is
            # loaded, else the x_linear context's own category.
            var category = py_get(context, "xl_undefined_warning")
            var sk = py_call(native_loaded_module("sys").modules.get, PythonObject("sklearn.exceptions"))
            if sk is not None:
                category = py_call(native_loaded_module("builtins").getattr, sk, PythonObject("UndefinedMetricWarning"),
                                   category)
            if category is None:
                category = native_loaded_module("builtins").UserWarning
            metric_warn("R^2 score is not well-defined with less than two samples.", category, context, 2)
        else:
            raise Error("member ABI: unknown fit notice code " + String(code))
