# SPDX-License-Identifier: Apache-2.0
"""Import-time connection of LogisticRegression's native owner
(bindings/logistic_native.mojo) and its fixed ABI."""

_ESTIMATOR_ENTRIES = ("estimators_vendor", "estimators_numeric_mode", "qn_fit", "qn_decision_function",
                      "qn_sigmoid", "qn_softmax")
_OPTIONAL_ENTRIES = ("qn_predict_binary", "qn_predict_multiclass", "py2mojo_rows", "py2mojo_linear_flags")
_X_TREES_ENTRIES = ("x_trees_code_counts", "x_trees_class_rows", "x_trees_mul_f32")


def _refusal(error):
    # glue: a refusal recorded at import as "<Class>: <message>"
    for base in (ImportError, AttributeError, TypeError, RuntimeError, ValueError):  # glue: the fixed refusal classes
        if isinstance(error, base):
            return f"{base.__name__}: {error}"
    return f"RuntimeError: {error}"


def connect():
    # glue: compiled entries and plain metadata are retained during import
    from types import BuiltinFunctionType
    from . import _backend
    from ._native_classical_bridge import _state_context
    from ._native_labels_bridge import label_context
    from .tokenizer import _binding

    context = _state_context()
    context.update(label_context())
    estimators = {}
    x_trees = {}
    selected = _backend.default_mode()
    for mode in (selected, "fast" if selected == "identical" else "identical"):  # glue: fixed numeric-mode ABI setup
        try:
            module = _backend.binding("_mojolearn_estimators", mode)
            entries = {}
            for name in _ESTIMATOR_ENTRIES + _OPTIONAL_ENTRIES:  # glue: fixed native exports
                entry = getattr(module, name, None)
                if entry is None and name in _OPTIONAL_ENTRIES:
                    continue
                if not isinstance(entry, BuiltinFunctionType):
                    raise TypeError(f"LogisticRegression requires compiled {name}")
                entries[name] = entry
            estimators[mode] = entries
        except Exception as error:  # noqa: BLE001 - recorded, raised at the first call
            estimators[mode] = _refusal(error)
        try:
            module = _backend.binding("_mojolearn_x_trees", mode)
            table = {}
            for name in _X_TREES_ENTRIES:  # glue: fixed native exports
                entry = getattr(module, name, None)
                if isinstance(entry, BuiltinFunctionType):
                    table[name] = entry
            x_trees[mode] = table
        except Exception as error:  # noqa: BLE001 - recorded, raised at the first call
            x_trees[mode] = _refusal(error)
    call = _binding()._module.logistic_native_call
    if not isinstance(call, BuiltinFunctionType):
        raise TypeError("LogisticRegression requires a compiled logistic_native_call")
    context.update(lr_bindings=estimators, lr_x_trees=x_trees, lr_host_contexts={})
    return call, context


def connect_host(cls, context, estimators_host, host_predict, host_context):
    # glue: called only by _classical_host at its explicit CPU API setup. The
    # explicit host class fits and scores on the CPU estimators binding
    # (identical only, `_HostBound._bind`); its predict is the classical host
    # inference owner, which its score calls too.
    from types import BuiltinFunctionType
    host = dict(context)
    entries = {}
    try:
        if isinstance(estimators_host, Exception):
            raise estimators_host
        # An inference-only host binary exports no qn_fit: the entries it
        # lacks are refused by name where an operation needs them.
        for name in _ESTIMATOR_ENTRIES + _OPTIONAL_ENTRIES:  # glue: fixed native exports
            entry = getattr(estimators_host, name, None)
            if entry is None and name != "estimators_numeric_mode":
                continue
            if not isinstance(entry, BuiltinFunctionType):
                raise TypeError(f"{cls.__name__} requires compiled {name}")
            entries[name] = entry
    except Exception as error:  # noqa: BLE001 - recorded, raised at the first call
        entries = _refusal(error)
    if not isinstance(host_predict, BuiltinFunctionType):
        raise TypeError(f"{cls.__name__} requires the compiled classical host inference owner")
    refusal = (f"ImportError: mojolearn: the host {cls.__name__} serves "
               "_mojolearn_estimators only, not _mojolearn_x_trees")
    host.update(lr_bindings={"identical": entries}, lr_x_trees={"identical": refusal},
                lr_host_model=True, lr_host_predict=(host_predict, host_context))
    context.setdefault("lr_host_contexts", {})[cls] = host
