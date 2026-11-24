# SPDX-License-Identifier: Apache-2.0
"""Connection of the CPU images `device="cpu"` runs on (at the first
device="cpu" admission, never at import), for the
decomposition estimators (the family-local form of
`_native_classical_bridge.connect_cpu_device`)."""

#: The host binaries that hold the decomposition families' CPU descriptors.
X_DECOMP_HOST = "_mojolearn_x_decomp_host"
ESTIMATORS_HOST = "_mojolearn_estimators_host"


def connect_fold_bases(context, export):
    """Import the GPU base image (`_mojolearn`) of each numeric mode now, at
    the family's import, so its init publishes that mode's decomposition
    fold table (core/mojolearn_run.mojo `decomp_fold_table_publish`) before
    any member admission reads it (lane vdf-tail-a: a `fast` member's fold
    route no longer depends on whether something else imported the `fast`
    base image first). `context["decomp_fold_bases"]` records, per mode,
    whether that image loaded and exports `export` (its fold entry): the
    member door refuses a GPU route whose loaded image published no table,
    and only an install without the image keeps host rows."""
    # glue: import-time connection of the fixed native base images, once per family context
    from types import BuiltinFunctionType
    from . import _backend
    bases = {}
    for mode in ("identical", "fast"):  # glue: fixed numeric-mode images
        try:
            factory = getattr(_backend.binding("_mojolearn", mode), export)
        except (ImportError, AttributeError, TypeError, ValueError, RuntimeError):
            factory = None
        bases[mode] = isinstance(factory, BuiltinFunctionType)
    context["decomp_fold_bases"] = bases


def connect_cpu_device(context, basename, tables):
    """Register the CPU descriptor factories of `device="cpu"` beside the
    device factories already in `context`: the host image loads at the first
    device="cpu" admission beside a GPU set (fx-host-import,
    `_native_family_bridge.cpu_load_later`), never at import; an estimator
    restored without construction connects it at its first CPU route
    (bindings/decomp_device_route_door.mojo `decomp_cpu_table`). Nothing is
    registered in a CPU-only process."""
    # glue: import-time registration, once per family context
    if context.get("vendor") == "cpu" or "vendor" not in context:
        return
    from ._native_family_bridge import cpu_load_later
    cpu_load_later(context, lambda: load_cpu_device(context, basename, tables))


def load_cpu_device(context, basename, tables):
    """Connect the CPU descriptor factories of `device="cpu"` once, now,
    beside the device factories already in `context` (the deferred load of
    `connect_cpu_device`, or a caller already inside the one CPU connect).

    `tables` maps a factory's exported name in the host binary `basename` to
    the context key the compiled owner reads it under; each key receives
    `{"identical": factory}`, the same form as the device table of that
    family. The compiled owner resolves an estimator's route against the
    device table or this one, once per estimator
    (bindings/decomp_device_route_door.mojo). In a CPU-only process the
    selected family already is the CPU family and nothing is loaded. A host
    binary that is not built is kept as its refusal text, said when a
    `device="cpu"` estimator first runs; a factory the binary does not export
    is kept as None (the owner's "does not export" refusal)."""
    # glue: load the CPU image once and retain its compiled descriptor factories
    from types import BuiltinFunctionType
    from . import _backend
    if context.get("vendor") == "cpu" or "vendor" not in context:
        return
    if all(key in context for key in tables.values()):  # glue: fixed context keys, once
        return
    try:
        module = _backend.load_host_module(basename)
    except (ImportError, AttributeError, TypeError, ValueError, RuntimeError) as error:
        refusal = type(error).__name__ + ": " + str(error)
        for key in tables.values():  # glue: fixed context keys of one family group
            context[key] = {"identical": refusal}
        return
    for name, key in tables.items():  # glue: fixed factory names of one host binary
        factory = getattr(module, name, None)
        context[key] = {"identical": factory if isinstance(factory, BuiltinFunctionType) else None}
