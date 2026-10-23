# SPDX-License-Identifier: Apache-2.0
"""Import-time connection of the native owner of the linear expansion
estimators and the coordinate-descent pair (bindings/x_linear_native.mojo)
and its fixed ABI. Nothing here runs when a public method is called."""

#: The x_linear binding's entries the owner calls.
_X_LINEAR_ENTRIES = ("x_linear_fit", "x_linear_decision", "x_linear_decision_codes", "x_linear_class_prep",
                     "py2mojo_rows", "x_linear_spearman_sign")
_X_LINEAR_OPTIONAL = ("x_linear_numeric_mode", "x_linear_vendor")


def _refusal(error):
    # glue: a refusal recorded at import, raised again by the operation that needs it
    return type(error).__name__ + ": " + str(error)


def _linear_entries(module):
    # glue: retain checked compiled entries of one x_linear binding at connection
    from types import BuiltinFunctionType
    entries = {}
    for name in _X_LINEAR_ENTRIES + _X_LINEAR_OPTIONAL:  # glue: fixed native exports
        entry = getattr(module, name, None)
        if entry is None and name in _X_LINEAR_OPTIONAL:
            continue
        if not isinstance(entry, BuiltinFunctionType):
            raise TypeError(f"the x_linear owner requires compiled {name}")
        entries[name] = entry
    return entries


def connect(tables, solver=None):
    """`(call, context)` for the linear expansion estimators.

    `tables` holds the fit's plain name tables (the SGD loss, penalty and
    learning-rate codes, the summed-step losses, ElasticNet's selection
    codes); `solver` is `_solver_impl`'s `_mojolearn_solver` module (None
    for the x_linear estimators, which never call it)."""
    # glue: compiled entries and plain metadata are retained during import
    from types import BuiltinFunctionType
    from . import _backend
    from ._native_classical_bridge import _state_context
    from ._native_labels_bridge import label_context
    from .tokenizer import _binding

    context = _state_context()
    context.update(label_context())
    selected = _backend.default_mode()
    linear, metrics = {}, {}
    for mode in (selected, "fast" if selected == "identical" else "identical"):  # glue: the two numeric modes, connected once
        try:
            linear[mode] = _linear_entries(_backend.binding("_mojolearn_x_linear", mode))
        except (ImportError, AttributeError, TypeError, ValueError, RuntimeError) as error:
            linear[mode] = _refusal(error)
        try:
            module = _backend.binding("_mojolearn_x_metrics", mode)
            table = {}
            for name in ("x_metrics_run", "x_metrics_vendor", "x_metrics_numeric_mode",  # glue: fixed native exports
                         "x_metrics_run_out", "x_metrics_run_ranges"):
                entry = getattr(module, name, None)
                if entry is None and name in ("x_metrics_run_out", "x_metrics_run_ranges"):
                    continue
                if not isinstance(entry, BuiltinFunctionType):
                    raise TypeError(f"the x_linear owner's score requires compiled {name}")
                table[name] = entry
            metrics[mode] = table
        except (ImportError, AttributeError, TypeError, ValueError, RuntimeError) as error:
            metrics[mode] = _refusal(error)
    cd, host_predict = "ImportError: the coordinate-descent solver is not connected here", None
    if solver is not None:
        cd = _connect_cd(solver)
        # an explicit host Lasso / ElasticNet predicts through the classical
        # door's host entry (`_HostCD.predict`)
        from .linear_model import _native_classical, _CLASSICAL
        host_predict = (_native_classical, _CLASSICAL)
    call = _binding()._module.x_linear_native_call
    if not isinstance(call, BuiltinFunctionType):
        raise TypeError("the linear estimators require a compiled x_linear_native_call")
    import os
    context.update(xl_bindings=linear, classical_metrics=metrics, cd_solver=cd,
                   cd_host_predict=host_predict, environment=os.environ._data,
                   xl_sgd_clf_loss=dict(tables.get("sgd_clf_loss", {})), xl_sgd_reg_loss=dict(tables.get("sgd_reg_loss", {})),
                   xl_sgd_penalty=dict(tables.get("sgd_penalty", {})), xl_sgd_lr=dict(tables.get("sgd_lr", {})),
                   xl_sgd_sum_losses=frozenset(tables.get("sgd_sum_losses", ())),
                   cd_selection=dict(tables.get("cd_selection", {})))
    return call, context


def _connect_cd(solver):
    # glue: retain the solver binding's two compiled entries at connection
    from types import BuiltinFunctionType
    try:
        cd = {}
        for name in ("cd_fit", "cd_predict"):  # glue: fixed native exports
            entry = getattr(solver, name)
            if not isinstance(entry, BuiltinFunctionType):
                raise TypeError(f"ElasticNet requires compiled {name}")
            cd[name] = entry
        return cd
    except (ImportError, AttributeError, TypeError, ValueError, RuntimeError) as error:
        return _refusal(error)
