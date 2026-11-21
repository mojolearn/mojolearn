# SPDX-License-Identifier: Apache-2.0
"""Import-time connection of the scalers' CPU context for `device="cpu"`."""

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
    (providers/python/prep_route_boundary_door.mojo) resolves an estimator
    to this context or to its own, once per estimator. In a CPU-only process
    the scaler's own context already is the CPU one and nothing is loaded. A
    host binary that is not built is kept as its refusal text, said when a
    `device="cpu"` estimator first runs."""
    # glue: load the CPU images once and retain their entries beside the device ones
    from . import _backend
    # The exact `str` type and the builtin `repr` the compiled device check
    # uses for a value that is not None (the refusal shows the value's repr).
    context.update(prep_device_str=str, prep_device_repr=repr)
    if context["vendor"] == "cpu" or "prep_cpu_context" in context:
        return
    try:
        entries = cpu_entries(_backend.load_host_module)
        cpu = dict(context)
        cpu.update(vendor="cpu", scaler_bindings={"identical": entries, "fast": _FAST_ON_CPU})
    except (ImportError, AttributeError, TypeError, ValueError, RuntimeError) as error:
        cpu = type(error).__name__ + ": " + str(error)
    context["prep_cpu_context"] = cpu
