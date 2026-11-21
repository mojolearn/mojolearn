# SPDX-License-Identifier: Apache-2.0
"""Import-time connection of the preprocessing owners' CPU contexts for `device="cpu"`."""

#: What `scaler_bindings["fast"]` holds in the CPU context: the doors say a
#: text entry as the reason the mode is not connected.
_FAST_ON_CPU = "device='cpu' runs in identical mode only; numeric_mode='fast' is for Apple GPUs"


def connect_cpu_device(context, cpu_entries):
    """Connect the CPU context of `device="cpu"` once, at import, beside the
    scaler's own connected context.

    `cpu_entries(load)` builds the scaler's entry dictionary from the host
    binaries (`load` is `_backend.load_host_module`), exactly as its
    `connect` builds one from the selected binaries. The CPU context is the
    scaler's context with those entries as its identical binding and
    `vendor="cpu"`; the compiled route owner
    (bindings/scaler_route_unpack.mojo, on the shared device route owner) resolves an estimator
    to this context or to its own, once per estimator. In a CPU-only process
    the scaler's own context already is the CPU one and nothing is loaded. A
    host binary that is not built is kept as its refusal text, said when a
    `device="cpu"` estimator first runs."""
    # glue: load the CPU images once and retain their entries beside the device ones
    from . import _backend
    # The builtins module the shared compiled device check takes
    # (providers/python/device_route_boundary_door.mojo `device_admit`): read
    # by the scaler doors only for a device that is not None and when an
    # estimator's route is resolved.
    import builtins
    context["prep_builtins"] = builtins
    if context["vendor"] == "cpu" or "prep_cpu_context" in context:
        return
    try:
        entries = cpu_entries(_backend.load_host_module)
        cpu = dict(context)
        # `prep_cpu_context` None: the CPU context names every key of its
        # source, so a key the source gains later is seen by the size alone
        # (bindings/scaler_route_unpack.mojo `_scaler_cpu_sync`).
        cpu.update(vendor="cpu", scaler_bindings={"identical": entries, "fast": _FAST_ON_CPU},
                   prep_cpu_context=None)
    except (ImportError, AttributeError, TypeError, ValueError, RuntimeError) as error:
        cpu = type(error).__name__ + ": " + str(error)
    context["prep_cpu_context"] = cpu


def connect_cpu_context(context, overrides):
    """Connect the CPU context of `device="cpu"` for a preprocessing owner
    whose device questions are context keys (the classifier owner, the
    selectors, the iterative imputer): the owner's context with the keys
    `overrides(load)` returns replaced, `vendor="cpu"` among them.

    `overrides` may only replace keys the context already holds: the CPU
    context then differs from its source in those values alone, and a key
    the source gains after import (another family connecting to it) is
    copied over when an estimator's route is next resolved. In a CPU-only
    process the owner's context already is the CPU one."""
    # glue: load the CPU images once and retain their entries beside the device ones
    from . import _backend
    import builtins
    context["prep_builtins"] = builtins
    if context["vendor"] == "cpu" or "prep_cpu_context" in context:
        return
    try:
        replaced = dict(overrides(_backend.load_host_module), vendor="cpu")
        if any(key not in context for key in replaced):  # glue: fixed override names, checked once at import
            raise TypeError("a CPU context may only replace keys of its source context")
        cpu = dict(context)
        cpu.update(replaced, prep_cpu_context=None)
    except (ImportError, AttributeError, TypeError, ValueError, RuntimeError, OSError) as error:
        cpu = type(error).__name__ + ": " + str(error)
    context["prep_cpu_context"] = cpu


def cpu_prog_context(prog_context, load, family="x_prep"):
    """The shared arena runner's context with the host image of `family` as
    its identical binding (the runner selects `prog_bindings[mode]`)."""
    # glue: the fixed program exports of the host image, connected once at import
    from types import BuiltinFunctionType
    from ._native_arena_prog import _FAMILIES
    module_name, mandatory, optional, _ = _FAMILIES[family]
    module = load(module_name + "_host")
    entries = {mandatory: getattr(module, mandatory)}
    for name in optional:  # glue: fixed optional native exports connected during import
        entry = getattr(module, name, None)
        if entry is not None:
            entries[name] = entry
    if any(not isinstance(entry, BuiltinFunctionType) for entry in entries.values()):  # glue: admit only compiled entries
        raise TypeError("arena programs require compiled host entries")
    entries.update(prog_family=family, arena_binding=module, arena_binding_id=id(module))
    cpu = dict(prog_context)
    cpu.update(vendor="cpu", prog_bindings={"identical": entries, "fast": "ValueError: " + _FAST_ON_CPU})
    return cpu


def cpu_label_overrides(context, load):
    """The label steps of a CPU context: the host bodies in-process (what a
    CPU-only installation runs, `labels_base_binding` False) and the host
    metrics image as the identical metrics table. Empty for a context
    without label steps."""
    # glue: the label context keys a CPU context replaces
    from types import BuiltinFunctionType
    if "labels_base_binding" not in context:
        return {}
    try:
        module = load("_mojolearn_metrics_host")
        table = {}
        for name in ("r2_score", "r2_score_weighted", "metrics_numeric_mode", "umap_numeric_mode"):  # glue: fixed native exports
            entry = getattr(module, name, None)
            if isinstance(entry, BuiltinFunctionType):
                table[name] = entry
    except (ImportError, AttributeError, TypeError, ValueError, RuntimeError, OSError) as error:
        table = type(error).__name__ + ": " + str(error)
    return dict(labels_base_binding=False,
                labels_bindings={"identical": {"_mojolearn_metrics": table},
                                 "fast": {"_mojolearn_metrics": "ValueError: " + _FAST_ON_CPU}})
