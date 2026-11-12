# SPDX-License-Identifier: Apache-2.0
"""Import-time connection of QNRegressor's native owner and its fixed ABI."""

#: The native tables' vendor words (core/estimators_entry_table.mojo
#: estimators_vendor_word applies the same mapping at publication).
_NATIVE_VENDORS = {"cpu": 0, "metal": 1, "cuda": 2, "hip": 3}


def connect(losses):
    # glue: compiled entries and plain metadata are retained during import;
    # `losses` is QNRegressor's loss-name table.
    import os
    from types import BuiltinFunctionType
    from . import _backend
    from ._native_classical_bridge import _state_context
    from .tokenizer import _binding

    from ._native_labels_bridge import label_context
    context = _state_context()
    # predict's float64 result (bindings/qn_native_door.mojo labels_cast) calls the
    # label context (context["labels_base_binding"], connected once); without
    # it every public predict raised ValueError('labels_helpers')
    # (0.8.66 AMD measurement smoke, algos/qn-reg on istella and taxi).
    context.update(label_context())
    from ._native_owner_lifetime import connect_owner_context
    connect_owner_context(context)
    connected = {}
    metric_bindings = {}
    selected = _backend.default_mode()
    for mode in (selected, "fast" if selected == "identical" else "identical"):  # glue: fixed numeric-mode ABI setup
        try:
            # glue: the estimators image's (tier, vendor) key of the native
            # estimators QN table (core/estimators_entry_table.mojo), read
            # once here; the owner calls the image's published entry with it,
            # never the binding's Python exports. `_module` retains the image.
            estimator = _backend.binding("_mojolearn_estimators", mode)
            tier = int(estimator.estimators_numeric_mode())  # import-time: the image's compiled tier
            vendor = _NATIVE_VENDORS.get(str(estimator.estimators_vendor()), -1)  # import-time: the image's vendor
            if vendor < 0 or tier not in (0, 1):
                raise ImportError("QNRegressor: the estimators image publishes no native QN table")
            connected[mode] = {"_module": estimator, "native_key": (tier, vendor)}
        except (ImportError, AttributeError, TypeError, ValueError, RuntimeError) as error:
            connected[mode] = type(error).__name__ + ": " + str(error)
        try:
            metrics = _backend.binding("_mojolearn_x_metrics", mode)
            metric_entries = {}
            for name in ("x_metrics_run", "x_metrics_vendor", "x_metrics_numeric_mode",  # glue: fixed native exports
                         "x_metrics_run_out", "x_metrics_run_ranges", "metrics_program_native_api_v1"):
                try:
                    entry = getattr(metrics, name)
                except (AttributeError, ImportError):
                    if name in ("x_metrics_run_out", "x_metrics_run_ranges"):
                        continue
                    raise
                if not isinstance(entry, BuiltinFunctionType):
                    raise TypeError(f"QNRegressor score requires compiled {name}")
                metric_entries[name] = entry
            metric_entries["_module"] = metric_entries["x_metrics_run"].__self__
            metric_bindings[mode] = metric_entries
        except (ImportError, AttributeError, TypeError, ValueError, RuntimeError) as error:
            metric_bindings[mode] = type(error).__name__ + ": " + str(error)
    call = _binding()._module.qn_native_call
    if not isinstance(call, BuiltinFunctionType):
        raise TypeError("QNRegressor requires a compiled qn_native_call")
    from ._native_classical_bridge import _numpy_free_context
    # the fitted coefficient stores are mojolearn Arrays whatever is installed
    context.update(qn_bindings=connected, classical_metrics=metric_bindings, qn_losses=dict(losses),
                   environment=os.environ._data, output_context=_numpy_free_context(),
                   qn_fold_key=_qn_fold_key())
    return call, context


def _qn_fold_key():
    # glue (lane a7c; table key since lane calltable-solver-estimators): the
    # (tier, vendor) key of the resident-fold QN entry the GPU base binding
    # publishes next to the fold store it reads (bindings/msel_device_unpack.mojo,
    # core/estimators_entry_table.mojo ESTIMATORS_TABLE_QN_FOLD); a build
    # without it (the CPU column) leaves None and the door gathers host rows.
    from ._buffer import _native
    try:
        _native('logistic_fold_native_api_v1')  # import-time: the base image carries the fold entry
        tier = int(_native('mojolearn_numeric_mode')())  # import-time: the base image's compiled tier
        vendor = _NATIVE_VENDORS.get(str(_native('mojolearn_vendor')()), -1)  # import-time: its vendor
    except (ImportError, AttributeError, KeyError, TypeError, ValueError, RuntimeError):
        return None
    return (tier, vendor) if vendor > 0 and tier in (0, 1) else None
