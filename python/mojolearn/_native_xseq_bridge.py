# SPDX-License-Identifier: Apache-2.0
"""Import-time connection of the native owner of the sequence-expansion
estimators (bindings/xseq_native_pack.mojo): the recurrent models, the
multi-layer perceptrons and the theta, Croston, ETS, GARCH, VAR and STL
forecasters. Each public method of those classes is one call,
`_XSEQ(self, "<operation>", args, _XSEQ_CONTEXT)`; Python here only
connects the compiled entries and fixed metadata once, at import."""

_ENTRY = None
_CONTEXT = None

#: the compiled `_mojolearn_x_sequence` entries the owner runs
_SEQUENCE_ENTRIES = ("x_sequence_vendor", "x_sequence_numeric_mode", "model_native_api_v1", "model_labels", "rnn_fit", "rnn_predict",
                     "mlp_fit", "mlp_predict")
#: the sequence table's vendor words (core/xseq_entry_table.mojo
#: `xseq_vendor_word`, the mapping the images apply at publication)
_NATIVE_VENDORS = {"cpu": 0, "metal": 1, "cuda": 2, "hip": 3}
#: the base-binding entries it runs (weight draws, row argmax, label thresholds, scans)
_BASE_ENTRIES = ("uniform_init_f32", "argmax_rows_f32", "threshold_labels_i64", "reduce_stat", "all_finite_f32", "all_finite_f64")


def _unavailable(error):
    # glue: an install without the compiled owner keeps the package
    # importable; the operations then explain what is missing.
    message = f"mojolearn: the sequence estimators need the compiled host binding: {error}"

    def unavailable(*args):
        raise ModuleNotFoundError(message, name=getattr(error, "name", None))
    return unavailable


def _sequence_image(module):
    # glue: once per image at import: its fixed compiled entries, and its
    # (tier, vendor) key of the sequence table (lane calltable-xseq); the
    # native owner reads the image's published entries with the key.
    from types import BuiltinFunctionType
    symbols = {}
    for name in _SEQUENCE_ENTRIES:  # glue: fixed compiled entry names
        symbol = getattr(module, name)
        if not isinstance(symbol, BuiltinFunctionType):
            raise TypeError(f"the sequence estimators require a compiled {name}")
        symbols[name] = symbol
    vendor = _NATIVE_VENDORS.get(str(symbols["x_sequence_vendor"]()), -1)  # import-time: the image's vendor
    return symbols, (int(symbols["x_sequence_numeric_mode"]()), vendor)  # import-time: the image's compiled tier


def _metrics_image(module):
    # glue: once per image at import, the metrics image's (tier, vendor) key
    # of its native R^2 table (core/metrics_entry_table.mojo, lane
    # calltable-xseq); the sequence scores call that entry, not the image's
    # Python r2_score entries. The module is kept to pin the image.
    vendor = _NATIVE_VENDORS.get(str(module.metrics_vendor()), -1)  # import-time: the image's vendor
    return (int(module.metrics_numeric_mode()), vendor, module)  # import-time: the compiled tier


def _connect_cpu_device(context):
    # glue: `device="cpu"` beside a GPU set: the CPU identical images of the
    # sequence and metrics bindings, registered at import and loaded at the
    # first device="cpu" admission (fx-host-import,
    # `_native_family_bridge.cpu_load_later`); a recurrent or MLP
    # estimator's route holds them, and one restored without construction
    # connects them at its route resolution (bindings/xseq_native_common_door.mojo
    # `xseq_route`). A CPU-only installation has no second set: its one set
    # is the CPU's.
    from . import _backend
    if _backend._CPU_ONLY is not None:
        return
    from ._native_family_bridge import cpu_load_later
    cpu_load_later(context, lambda: _load_cpu_device(context))


def _load_cpu_device(context):
    # glue: once, at the first device="cpu" admission beside a GPU set. An
    # image that does not load records why, and the estimator that asks for
    # it says so.
    from . import _backend, host_surface
    if "xseq_cpu_bindings" in context:
        return
    errors = (ImportError, AttributeError, ValueError, RuntimeError, TypeError, OSError, KeyError)
    # The manifest's route table, read once here: `_backend._HOST_MODULES` is
    # read only by the CPU-only selector (tools/test_lane_select_narrowing.py).
    routed = host_surface.routed_modules()
    bindings = {}
    keys = {}
    metrics = {}
    try:
        module = _backend.load_host_module(routed["_mojolearn_x_sequence"])  # first CPU admission: the CPU image
        bindings["identical"], keys["identical"] = _sequence_image(module)
    except errors as error:
        bindings["identical"] = keys["identical"] = type(error).__name__ + ": " + str(error)
    try:
        metrics["identical"] = _metrics_image(
            _backend.load_host_module(routed["_mojolearn_metrics"]))  # first CPU admission: the CPU image
    except errors as error:
        metrics["identical"] = type(error).__name__ + ": " + str(error)
    context.update(xseq_cpu_bindings=bindings, xseq_cpu_native_keys=keys, xseq_cpu_metrics=metrics)


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
    keys = {}
    for mode in (selected, "fast" if selected == "identical" else "identical"):  # glue: the two numeric modes, connected once
        try:
            bindings[mode], keys[mode] = _sequence_image(_backend.binding("_mojolearn_x_sequence", mode))
        except (ImportError, AttributeError, ValueError, RuntimeError, TypeError) as error:
            bindings[mode] = type(error).__name__ + ": " + str(error)
            keys[mode] = bindings[mode]
    base = {}
    for name in _BASE_ENTRIES:  # glue: fixed compiled entry names
        symbol = _native(name)
        if not isinstance(symbol, BuiltinFunctionType):
            raise TypeError(f"the sequence estimators require a compiled {name}")
        base[name] = symbol
    # glue: once at import, the base helper table key the native doors read
    # (lane calltable-base-helpers); a binary without one records the refusal
    from ._buffer import base_native_key
    try:
        base["base_native_key"] = base_native_key()
    except Exception as error:  # noqa: BLE001 - recorded, raised at the first call
        base["base_native_key"] = type(error).__name__ + ": " + str(error)
    metrics = {}
    for mode in (selected, "fast" if selected == "identical" else "identical"):  # glue: the two numeric modes, connected once
        try:
            metrics[mode] = _metrics_image(_backend.binding("_mojolearn_metrics", mode))
        except (ImportError, AttributeError, ValueError, RuntimeError, TypeError) as error:
            metrics[mode] = type(error).__name__ + ": " + str(error)
    r2_options = None
    try:
        # the native metrics owner's multi-output R^2, where the install has it
        from . import _native_family_bridge
        r2_options = _native_family_bridge.connect()
    except ImportError:
        r2_options = None
    from ._native_family_bridge import shared_context
    # The classifier label entries are in the shared context from its creation
    # (_native_family_bridge.shared_context).
    lifecycle = shared_context()
    from ._native_owner_lifetime import connect_owner_context
    ownership = dict(lifecycle)
    connect_owner_context(ownership)
    context = dict(_SEQUENTIAL_CONTEXT)
    context.update(xseq_bindings=bindings, xseq_native_keys=keys, xseq_base=base, optimizer_context=_OPT_CONTEXT,
                   os_urandom=os.urandom, np_asarray=np.asarray, xseq_linalg_error=np.linalg.LinAlgError,
                   # the exact ndarray type whose read-only flag guards the
                   # fitted-weight records (bindings/xseq_neural_native_door.mojo)
                   xseq_ndarray=np.ndarray,
                   xseq_metrics=metrics, xseq_r2_options=r2_options, xseq_lifecycle_context=lifecycle,
                   xseq_forecast_ownership=ownership)
    _connect_cpu_device(context)
    _ENTRY, _CONTEXT = entry, context
    return _ENTRY, _CONTEXT


def register(**tables):
    """Connect a module's fixed lookup tables (and result classes) at its
    definition: `xseq_*` names the owner reads."""
    _, context = connect()
    if "xseq_unavailable" not in context:
        context.update(tables)
