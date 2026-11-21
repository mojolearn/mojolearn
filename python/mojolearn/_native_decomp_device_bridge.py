# SPDX-License-Identifier: Apache-2.0
"""Import-time connection of the CPU images `device="cpu"` runs on, for the
decomposition estimators (the family-local form of
`_native_classical_bridge.connect_cpu_device`)."""

#: The host binaries that hold the decomposition families' CPU descriptors.
X_DECOMP_HOST = "_mojolearn_x_decomp_host"
ESTIMATORS_HOST = "_mojolearn_estimators_host"


def connect_cpu_device(context, basename, tables):
    """Connect the CPU descriptor factories of `device="cpu"` once, at import,
    beside the device factories already in `context`.

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
