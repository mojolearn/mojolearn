# SPDX-License-Identifier: Apache-2.0
"""Import-time connection of the CPU images of `device="cpu"` for the families
that select compiled bindings by name (mixture, cluster, neighbors, density).

On a GPU installation the device images of a family are connected by
`_native_family_bridge.connect_bindings`. `connect_cpu_device` loads, once, at
the family's import, the CPU identical image of each of the same names
(`_backend._HOST_MODULES`) beside them and records it in the shared context
under `family_cpu_binding_connections`, in the record form the compiled owner
already reads for a CPU-only install: `(module, module name, mode reader)`,
or `(refusal text,)` for an image that is not built. The compiled owner
resolves an estimator's route against the device table or this one, once per
estimator (providers/python/device_route_dw_neighbors_cluster_door.mojo).

In a CPU-only process the selected family already is the CPU family and
nothing is loaded here. Python only retains the loaded images; it selects
nothing per call."""


def connect_cpu_device(*names):
    """Connect the CPU image of each binding name once, at import."""
    # glue: load the CPU images once and retain them in the shared context
    from . import _backend
    from ._native_family_bridge import shared_context
    context = shared_context()
    if _backend._CPU_ONLY is not None or "family_unavailable" in context:
        return
    connections = context.setdefault("family_cpu_binding_connections", {})
    errors = (ImportError, AttributeError, TypeError, ValueError, RuntimeError, OSError, KeyError)
    for name in names:  # glue: fixed module-import binding names, never user data or product execution
        if name in connections:
            continue
        try:
            basename = _backend._HOST_MODULES[name]
            module = _backend.load_host_module(basename)  # import-time: the CPU image of an explicitly named binding
            connections[name] = (module, module.__name__, _backend._host_prefix(basename) + "_numeric_mode")
        except errors as error:
            connections[name] = (type(error).__name__ + ": " + str(error),)
