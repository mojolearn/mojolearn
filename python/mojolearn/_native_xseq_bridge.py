# SPDX-License-Identifier: Apache-2.0
"""Import-time connection of the native owner of the sequence-expansion
estimators (bindings/xseq_native.mojo): the recurrent models, the
multi-layer perceptrons and the theta, Croston, ETS, GARCH, VAR and STL
forecasters. Each public method of those classes is one call,
`_XSEQ(self, "<operation>", args, _XSEQ_CONTEXT)`; Python here only
connects the compiled entries and fixed metadata once, at import."""

_ENTRY = None
_CONTEXT = None

#: the compiled `_mojolearn_x_sequence` entries the owner runs
_SEQUENCE_ENTRIES = ("x_sequence_vendor", "x_sequence_numeric_mode", "model_labels", "rnn_fit", "rnn_predict",
                     "mlp_fit", "mlp_predict", "theta", "croston", "croston_forecast", "ets", "garch", "var_fit",
                     "stl", "prophet_days", "prophet_features", "prophet_changepoints", "prophet_fit",
                     "prophet_predict")
#: the base-binding entries it runs (weight draws, row argmax, label thresholds, scans)
_BASE_ENTRIES = ("uniform_init_f32", "argmax_rows_f32", "threshold_labels_i64", "reduce_stat", "all_finite_f32")


def _unavailable(error):
    # glue: an install without the compiled owner keeps the package
    # importable; the operations then explain what is missing.
    message = f"mojolearn: the sequence estimators need the compiled host binding: {error}"

    def unavailable(*args):
        raise ModuleNotFoundError(message, name=getattr(error, "name", None))
    return unavailable


def connect():
    """`(entry, context)`: the compiled owner and the context every
    connected class shares. Connected once; later calls return the same pair."""
    global _ENTRY, _CONTEXT
    if _CONTEXT is not None:
        return _ENTRY, _CONTEXT
    import os
    from types import BuiltinFunctionType
    from ._optional_numpy import require_numpy
    # glue: the sequence owner returns ndarray results and raises NumPy's
    # LinAlgError; _x_sequence_model requires NumPy the same way.
    np = require_numpy('the sequence estimators')
    from . import _backend
    from ._buffer import _native
    try:
        from .tokenizer import _binding
        entry = _binding()._module.xseq_native_call
        if not isinstance(entry, BuiltinFunctionType):
            raise TypeError("the sequence estimators require the compiled xseq_native_call entry")
    except (ImportError, AttributeError, TypeError) as error:
        _ENTRY, _CONTEXT = _unavailable(error), {"xseq_unavailable": str(error)}
        return _ENTRY, _CONTEXT
    from ._x_sequence_model import _SEQUENTIAL_CONTEXT
    from ._training_impl import _OPT_CONTEXT
    selected = _backend.default_mode()
    bindings = {}
    for mode in (selected, "fast" if selected == "identical" else "identical"):  # glue: the two numeric modes, connected once
        try:
            module = _backend.binding("_mojolearn_x_sequence", mode)
            symbols = {}
            for name in _SEQUENCE_ENTRIES:  # glue: fixed compiled entry names
                symbol = getattr(module, name)
                if not isinstance(symbol, BuiltinFunctionType):
                    raise TypeError(f"the sequence estimators require a compiled {name}")
                symbols[name] = symbol
            bindings[mode] = symbols
        except (ImportError, AttributeError, ValueError, RuntimeError, TypeError) as error:
            bindings[mode] = type(error).__name__ + ": " + str(error)
    base = {}
    for name in _BASE_ENTRIES:  # glue: fixed compiled entry names
        symbol = _native(name)
        if not isinstance(symbol, BuiltinFunctionType):
            raise TypeError(f"the sequence estimators require a compiled {name}")
        base[name] = symbol
    metrics = {}
    for mode in (selected, "fast" if selected == "identical" else "identical"):  # glue: the two numeric modes, connected once
        try:
            module = _backend.binding("_mojolearn_metrics", mode)
            symbols = {}
            for name in ("r2_score", "r2_score_weighted", "metrics_numeric_mode"):  # glue: fixed compiled entry names
                symbol = getattr(module, name, None)
                if isinstance(symbol, BuiltinFunctionType):
                    symbols[name] = symbol
            metrics[mode] = symbols
        except (ImportError, AttributeError, ValueError, RuntimeError, TypeError) as error:
            metrics[mode] = type(error).__name__ + ": " + str(error)
    r2_options = None
    try:
        # the native metrics owner's multi-output R^2, where the install has it
        from . import _native_family_bridge
        r2_options = _native_family_bridge.connect()
    except ImportError:
        r2_options = None
    context = dict(_SEQUENTIAL_CONTEXT)
    context.update(xseq_bindings=bindings, xseq_base=base, optimizer_context=_OPT_CONTEXT,
                   os_urandom=os.urandom, np_asarray=np.asarray, xseq_linalg_error=np.linalg.LinAlgError,
                   xseq_metrics=metrics, xseq_r2_options=r2_options)
    _ENTRY, _CONTEXT = entry, context
    return _ENTRY, _CONTEXT


def register(**tables):
    """Connect a module's fixed lookup tables (and result classes) at its
    definition: `xseq_*` names the owner reads."""
    _, context = connect()
    if "xseq_unavailable" not in context:
        context.update(tables)
