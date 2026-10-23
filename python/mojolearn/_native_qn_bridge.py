# SPDX-License-Identifier: Apache-2.0
"""Import-time connection of QNRegressor's native owner and its fixed ABI."""


def connect(losses):
    # glue: compiled entries and plain metadata are retained during import;
    # `losses` is QNRegressor's loss-name table.
    import os
    from types import BuiltinFunctionType
    from . import _backend
    from ._native_classical_bridge import _state_context
    from .tokenizer import _binding

    context = _state_context()
    connected = {}
    metric_bindings = {}
    selected = _backend.default_mode()
    for mode in (selected, "fast" if selected == "identical" else "identical"):  # glue: fixed numeric-mode ABI setup
        try:
            estimator = _backend.binding("_mojolearn_estimators", mode)
            entries = {}
            for name in ("estimators_vendor", "estimators_numeric_mode", "qn_fit", "qn_decision_function"):  # glue: fixed native exports
                entry = getattr(estimator, name)
                if not isinstance(entry, BuiltinFunctionType):
                    raise TypeError(f"QNRegressor requires compiled {name}")
                entries[name] = entry
            connected[mode] = entries
        except (ImportError, AttributeError, TypeError, ValueError, RuntimeError) as error:
            connected[mode] = type(error).__name__ + ": " + str(error)
        try:
            metrics = _backend.binding("_mojolearn_x_metrics", mode)
            metric_entries = {}
            for name in ("x_metrics_run", "x_metrics_vendor", "x_metrics_numeric_mode",  # glue: fixed native exports
                         "x_metrics_run_out", "x_metrics_run_ranges"):
                try:
                    entry = getattr(metrics, name)
                except (AttributeError, ImportError):
                    if name in ("x_metrics_run_out", "x_metrics_run_ranges"):
                        continue
                    raise
                if not isinstance(entry, BuiltinFunctionType):
                    raise TypeError(f"QNRegressor score requires compiled {name}")
                metric_entries[name] = entry
            metric_bindings[mode] = metric_entries
        except (ImportError, AttributeError, TypeError, ValueError, RuntimeError) as error:
            metric_bindings[mode] = type(error).__name__ + ": " + str(error)
    call = _binding()._module.qn_native_call
    if not isinstance(call, BuiltinFunctionType):
        raise TypeError("QNRegressor requires a compiled qn_native_call")
    from ._native_classical_bridge import _numpy_free_context
    # the fitted coefficient stores are mojolearn Arrays whatever is installed
    context.update(qn_bindings=connected, classical_metrics=metric_bindings, qn_losses=dict(losses),
                   environment=os.environ._data, output_context=_numpy_free_context())
    return call, context
