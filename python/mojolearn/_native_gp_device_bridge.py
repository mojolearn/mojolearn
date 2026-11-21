# SPDX-License-Identifier: Apache-2.0
"""Import-time connection of the CPU images a `device="cpu"` Gaussian process
estimator runs in a GPU installation.

The compiled owner resolves an estimator's route once per estimator
(providers/python/gp_device_route_pyapi_door.mojo): the device images of the
selected set, or the CPU images kept here under `gp_cpu_bindings`. Python
only loads those images and keeps them, once, at import, beside the device
images (`_native_family_bridge.connect_bindings`). In a CPU-only process the
selected set already is the CPU set and nothing is loaded. A CPU image that
is not built is kept as its refusal text, said when a `device="cpu"`
estimator first needs it."""

_ERRORS = (ImportError, AttributeError, TypeError, ValueError, RuntimeError, OSError)

#: The R^2 and accuracy programs `score` runs on a CPU route.
_METRICS_HOST = "_mojolearn_x_metrics_host"
_METRICS_FACTORY = "metrics_program_native_api_v1"


def _host_image(name):
    # glue: the CPU image a CPU-only process selects for `name` (the routed
    # host binding, else the inference-only one a wheel ships), loaded and
    # read back by the one host loader
    import os
    from . import _backend
    basename = _backend._HOST_MODULES.get(name)
    if not (basename and os.path.exists(_backend.host_module_path(basename))):
        fallback = _backend._HOST_INFERENCE_MODULES.get(name)
        basename = fallback if fallback and os.path.exists(_backend.host_module_path(fallback)) else None
    if basename is None:
        raise ImportError(f"mojolearn: device='cpu' needs the CPU binary of {name}, which is not built "
                          "in this installation")
    return _backend.load_host_module(basename)


def connect_cpu_device(context, *names):
    """Connect the CPU images of `names` (GPU binding names) and the CPU
    metrics program factory into `context["gp_cpu_bindings"]`, once."""
    # glue: load the CPU images once and retain them, at import
    from types import BuiltinFunctionType
    from . import _backend
    if _backend._CPU_ONLY is not None or "family_unavailable" in context:
        return
    cells = context.setdefault("gp_cpu_bindings", {})
    for name in names:  # glue: fixed module-import binding names, never user data or product execution
        if name in cells:
            continue
        try:
            cells[name] = _host_image(name)
        except _ERRORS as error:
            cells[name] = type(error).__name__ + ": " + str(error)
    if "metrics" not in cells:
        try:
            factory = getattr(_backend.load_host_module(_METRICS_HOST), _METRICS_FACTORY)
            if not isinstance(factory, BuiltinFunctionType):
                raise TypeError(f"device='cpu' score requires compiled {_METRICS_FACTORY}")
            cells["metrics"] = factory
        except _ERRORS as error:
            cells["metrics"] = type(error).__name__ + ": " + str(error)
