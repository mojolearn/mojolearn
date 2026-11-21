# SPDX-License-Identifier: Apache-2.0
"""Import-time connection of the native owner of the stateless x_prep
transformers (Binarizer, Normalizer, PolynomialFeatures, VarianceThreshold and
the selectors' transform) and its fixed ABI."""

CONTEXT_CONNECTION = None

#: What the CPU template's `scaler_bindings["fast"]` holds: the doors say a
#: text entry as the reason the mode is not connected.
_FAST_ON_CPU = "device='cpu' runs in identical mode only; numeric_mode='fast' is for Apple GPUs"


def _entries(module):
    # glue: the fixed x_prep exports of one compiled image, connected during import
    from types import BuiltinFunctionType
    entries = {}
    for name in ("x_prep_run", "x_prep_numeric_mode", "x_prep_vendor", "x_prep_idn_fam",  # glue: fixed native exports connected during import
                 "x_prep_run_out", "x_prep_run_scratch", "x_prep_run_ranges",
                 "x_prep_dev_put", "x_prep_dev_free", "x_prep_dev_live", "x_prep_dev_take_rows",
                 "x_prep_host_column", "x_prep_ptimpute_flags"):
        try:
            entry = getattr(module, name)
        except (AttributeError, ImportError):
            if name not in ("x_prep_run", "x_prep_numeric_mode"):
                continue
            raise
        if not isinstance(entry, BuiltinFunctionType):
            raise TypeError(f"the stateless prep owner requires compiled {name}")
        entries[name] = entry
    entries["preprocessing_numeric_mode"] = entries["x_prep_numeric_mode"]
    entries.update(arena_binding=module, arena_binding_id=id(module))
    return entries


def _identities(connected):
    # glue: immutable identities of the compiled exports actually selected;
    # no assumed function name/module-owner convention is required.
    return tuple(
        (mode, tuple(table.items()) if type(table) is dict else None)
        for mode, table in connected.items()  # glue: freeze fixed import-time native ABI identities
    )


def _cpu_records(context):
    """The frozen records of the CPU template for `device="cpu"`: the
    owner's context with the host x_prep image as its identical binding and
    `vendor="cpu"`; the refusal text when the host image is not built, said
    when a `device="cpu"` estimator first runs. `context` is a copy of the
    owner's context as it was connected at import."""
    # glue: load the CPU image once, at the first device="cpu" admission
    from . import _backend
    try:
        table = {"identical": _entries(_backend.load_host_module("_mojolearn_x_prep_host")), "fast": _FAST_ON_CPU}
    except (ImportError, AttributeError, TypeError, ValueError, RuntimeError, OSError) as error:
        return type(error).__name__ + ": " + str(error)
    cpu = dict(context)
    cpu.update(vendor="cpu", scaler_bindings=table, prep_stateless_entry_identities=_identities(table))
    return tuple(cpu.items())


def _cpu_cell(context):
    """The one cell of the CPU template (bindings/prep_context_native_door.mojo
    `_cpu_template_records`), a dictionary filled in place. A CPU-only process: its
    `records` is None now (its own template already is the CPU one). A GPU
    process: empty until the first device="cpu" admission loads the host
    x_prep image (fx-audit 9: never at import), with `device_cpu_connect` for an
    estimator restored without construction to connect it at its first CPU
    route."""
    # glue: import-time registration, once
    if context["vendor"] == "cpu":
        return {"records": None}
    from ._native_family_bridge import cpu_load_later
    cell = {}
    snapshot = dict(context)  # the context as connected at import

    def load():
        # glue: once, at the first device="cpu" admission beside a GPU set
        if "records" not in cell:
            cell["records"] = _cpu_records(snapshot)
    cpu_load_later(cell, load)
    return cell


def connect(kinds):
    global CONTEXT_CONNECTION
    # glue: compiled entries and plain metadata are retained during import;
    # `kinds` maps each public class to the kind the owner dispatches on.
    import sys
    import mmap
    from types import BuiltinFunctionType
    from . import _backend
    from ._arena_io import DeviceRows, _Scope, _LOCAL
    from ._native_classical_bridge import _state_context, _numpy_free_context
    from .tokenizer import _binding
    from ._native_loaded_modules import connect_module

    context = _state_context()
    context["output_context"] = _numpy_free_context()
    connected = {}
    for mode in ("identical", "fast"):  # glue: fixed numeric-mode ABI setup
        try:
            connected[mode] = _entries(_backend.binding("_mojolearn_x_prep", mode))
        except (ImportError, AttributeError, TypeError, ValueError, RuntimeError) as error:
            connected[mode] = type(error).__name__ + ": " + str(error)
    gather = _backend.binding("_mojolearn", "identical").gather_rows_bytes
    call = _binding()._module.prep_stateless_native_call
    if any(not isinstance(entry, BuiltinFunctionType) for entry in (call, gather)):  # glue: fixed native ABI identity
        raise TypeError("the stateless prep owner requires compiled prep_stateless_native_call and row-gather entries")
    # The owners read the process mode from _backend's namespace.
    _backend.default_mode()
    context.update(arena_mmap=mmap.mmap, arena_local=_LOCAL,
                   arena_scope_class=_Scope, arena_scope_owner=vars(_Scope)['_owner'].__get__,
                   scaler_bindings=connected, scaler_host_contexts={},
                   maxabs_rows_class=DeviceRows, maxabs_rows_owner=vars(DeviceRows)['_owner'].__get__, maxabs_gather_rows=gather,
                   classical_getframe=sys._getframe, prep_stateless_kinds=dict(kinds))
    context["prep_stateless_entry_identities"] = _identities(connected)
    # glue: freeze native class/descriptor references and fixed mapping contents
    # during package connection. Native operations copy these records and never
    # execute this setup helper. Process namespaces keep their pinned identity.
    static_names = ("array_fields", "array_storage_slots", "buffer_codes",
                    "prep_stateless_kinds", "warning_category_slots")
    output = context["output_context"]
    CONTEXT_CONNECTION = (
        context, tuple(context.items()),
        tuple((key, context[key], tuple(context[key].items()))
              for key in static_names if key in context),  # glue: fixed import-time provider maps
        output, tuple(output.items()),
        tuple((key, output[key], tuple(output[key].items()))
              for key in static_names if key in output),  # glue: fixed import-time output provider maps
        # The CPU template's cell (`device="cpu"` beside a GPU set): the
        # fixed maps and the output context above are shared with it.
        _cpu_cell(context),
    )
    connect_module(__name__, sys.modules[__name__], ("CONTEXT_CONNECTION",))
    return call, context
