# SPDX-License-Identifier: Apache-2.0
"""Import-time connection of the shared native label, weight and score steps
(bindings/classical_labels_door.mojo) that the classical owners call inside their
one native operation: the GPU base binding whose label provider they
connect (or False on an install without one), the metrics binding of both
numeric modes and fixed type metadata. Nothing here
runs when a public method is called."""

_LABEL_CONTEXT = {}
_LABEL_CONTEXT_CPU = {}


def _refusal(error):
    # glue: a refusal recorded at import as "<Class>: <message>", raised again
    # by the native step that needs the entry
    for base in (ImportError, AttributeError, TypeError, RuntimeError, ValueError):  # glue: the fixed refusal classes
        if isinstance(error, base):
            return f"{base.__name__}: {error}"
    return f"RuntimeError: {error}"


def _base_binding():
    # glue: the GPU base binding whose `labels_native_api_v3` provider the
    # label steps connect once (core/labels_native_api.mojo), the module
    # `_buffer._native` resolves (always the identical binary); False on a
    # CPU-only install, where the native label path runs the same host bodies
    # in-process; a recorded refusal otherwise, including an ImportError of
    # the base binding on a GPU install (never a silent host column)
    from . import _backend
    if _backend._CPU_ONLY is not None:
        return False
    try:
        module = _backend.binding("_mojolearn", mode="identical")
    except ImportError as error:
        if _backend._CPU_ONLY is not None:
            # a CPU-only install (selected while resolving): the host column
            return False
        # glue: a GPU install whose base binding does not load never falls to
        # the host column silently; the refusal is raised by the first label
        # step that needs it
        return _refusal(ImportError(
            "mojolearn: the GPU base binding _mojolearn did not load on this GPU install, so the "
            f"native label steps cannot run ({error}); reinstall the GPU wheel for this machine"))
    except Exception as error:  # noqa: BLE001 - recorded, raised at the first call
        return _refusal(error)
    if getattr(module, "__name__", "").rpartition(".")[2].endswith("_host"):
        # the CPU-only routing of `_mojolearn` to the core host binding
        return False
    return module


def _metrics():
    # glue: the metrics binding's compiled entries of both numeric modes
    from types import BuiltinFunctionType
    from . import _backend
    selected = _backend.default_mode()
    connected = {}
    for mode in (selected, "fast" if selected == "identical" else "identical"):  # glue: the two numeric modes
        try:
            module = _backend.binding("_mojolearn_metrics", mode)
            table = {}
            for name in ("r2_score", "r2_score_weighted", "metrics_numeric_mode", "umap_numeric_mode"):  # glue: fixed native exports
                entry = getattr(module, name, None)
                if isinstance(entry, BuiltinFunctionType):
                    table[name] = entry
            connected[mode] = {"_mojolearn_metrics": table}
        except Exception as error:  # noqa: BLE001 - recorded, raised at the first call
            connected[mode] = {"_mojolearn_metrics": _refusal(error)}
    return connected


def label_context():
    """The context entries bindings/classical_labels_door.mojo reads, connected
    once; each owner's bridge merges them into its own context."""
    if not _LABEL_CONTEXT:
        from ._native_loaded_modules import REAL_TYPES, INTEGRAL_TYPES
        import types
        _LABEL_CONTEXT.update(
            labels_base_binding=_base_binding(), labels_bindings=_metrics(),
            labels_real_type=REAL_TYPES, labels_integral_type=INTEGRAL_TYPES,
            labels_namespace=types.SimpleNamespace,
        )
    return dict(_LABEL_CONTEXT)


def _metrics_cpu():
    # glue: the CPU identical metrics image's compiled entries, under both
    # mode words (device="cpu" is identical only); a recorded refusal when the
    # image does not load
    from types import BuiltinFunctionType
    from . import _backend
    try:
        module = _backend.load_host_module("_mojolearn_metrics_host")
        table = {}
        for name in ("r2_score", "r2_score_weighted", "metrics_numeric_mode", "umap_numeric_mode"):  # glue: fixed native exports
            entry = getattr(module, name, None)
            if isinstance(entry, BuiltinFunctionType):
                table[name] = entry
    except Exception as error:  # noqa: BLE001 - recorded, raised at the first call
        table = _refusal(error)
    return {"identical": {"_mojolearn_metrics": table}, "fast": {"_mojolearn_metrics": table}}


def label_context_cpu():
    """The label entries of a CPU scope beside a GPU set: the host label
    kernels (`labels_base_binding=False`, which the label steps read from the
    caller's context) and the CPU metrics tables. A family that builds its
    own CPU scope merges these; nothing here is a process state."""
    if "cpu" not in _LABEL_CONTEXT_CPU:
        _LABEL_CONTEXT_CPU["cpu"] = dict(labels_base_binding=False, labels_bindings=_metrics_cpu())
    return dict(_LABEL_CONTEXT_CPU["cpu"])


def cpu_scope(context, **entries):
    """Build `context["labels_cpu_scope"]`: the context a `device="cpu"`
    classifier of this family hands the label boundary beside a GPU set
    (bindings/labels_descriptor_boundary_door.mojo `labels_scope`). A copy of
    `context` with the host label entries and the family's own CPU `entries`
    (its CPU metrics tables, `vendor="cpu"`, ...). Call it LAST in the
    family's connection: the copy does not follow later updates of
    `context`. A CPU-only installation needs none and gets none."""
    # glue: import-time connection, once per family context
    from . import _backend
    if _backend._CPU_ONLY is not None:
        return None
    scope = dict(context)
    scope.pop("labels_cpu_scope", None)
    scope.update(label_context_cpu())
    scope.update(entries)
    context["labels_cpu_scope"] = scope
    return scope


#: `bindings/labels_native_entry_pack.mojo` operation codes (lane ne1-h-labels).
LABELS_OP_ENCODE = 0
LABELS_OP_UNION_ENCODE = 1
LABELS_OP_SCORE_FROM_CODES = 2
LABELS_OP_DECODE = 3
LABELS_OP_CLASS_WEIGHT_ROWS = 4
LABELS_OP_FLATTEN = 5
LABELS_OP_SORTED = 6
#: Lane py-admission (bindings/labels_admission_unpack.mojo): the label, target
#: and X refusals `_labels` and `_sk_validate` made in Python.
LABELS_OP_PREPARE_TARGETS = 7
LABELS_OP_ENCODE_NUMERIC = 8
LABELS_OP_UNIQUE_INVERSE = 9
LABELS_OP_ARGMAX_ROWS = 10
LABELS_OP_CLASS_INDEX = 11
ADMISSION_OP_CHECK_X = 12
ADMISSION_OP_CHECK_Y = 13
#: ADMISSION_OP_CHECK_Y flags.
ADMISSION_Y_COMPLEX = 1
ADMISSION_Y_SCAN = 2
ADMISSION_Y_CLASSIFIER = 4
ADMISSION_Y_MISSING = 8
#: `labels_encode` kind: the ORDER RULE classes of `_labels` and int32 codes.
LABELS_KIND_ORDER_RULE = 0

_LABELS_ENTRY = []


def labels_entry():
    """`(entry, context)`: the compiled label entry `labels_native_call`
    (bindings/labels_native_entry_pack.mojo) and the context it reads: the
    classical shared admission context, the label context above and the
    `classes_` list class. Connected once, at the first label call (the
    label module is imported before the binding and its classes exist)."""
    if not _LABELS_ENTRY:
        from types import BuiltinFunctionType
        from .tokenizer import _binding
        from ._native_classical_bridge import _state_context
        from ._labels import ClassLabels
        entry = _binding()._module.labels_native_call
        if not isinstance(entry, BuiltinFunctionType):
            raise TypeError("the label shells require the compiled labels_native_call entry")
        context = _state_context()
        context.update(label_context())
        context.update(classifier_labels_class=ClassLabels,
                       classifier_labels_namespace=dict(vars(ClassLabels)),
                       classifier_labels_mro=type.__dict__["__mro__"].__get__(ClassLabels))
        _LABELS_ENTRY.append((entry, context))
    return _LABELS_ENTRY[0]
