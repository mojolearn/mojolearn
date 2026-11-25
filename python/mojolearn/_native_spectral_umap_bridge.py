# SPDX-License-Identifier: Apache-2.0
"""Import-time connections for spectral and UMAP native operation owners."""
from types import BuiltinFunctionType, ModuleType

_CONNECTED = None
_SYMBOLS = {
    "metrics": ("_mojolearn_metrics", (
        "spectral_native_api_v1", "metrics_numeric_mode", "umap_numeric_mode", "umap_fit_transform", "umap_fit_transform_ex", "umap_transform",
        # The image's vendor word: spectral_embedding's resident-store slot
        # (bindings/cfb_spectral_door.mojo `_se_scope_slot`) names the store by
        # it; uncaptured, every call inside resident() raised AttributeError
        # (fn-spectral-embedding, L40S smoke of main 3cd641b37e; lane gs-fix).
        "metrics_vendor",
        # The by-name spectral fit, embedding and predict entries were deleted
        # 2026-11-23 (no caller: the doors take spectral_native_api_v1).
    )),
    "decomp": ("_mojolearn_x_decomp", (
        # The kernel-call provider the typed manifold kit (UMAP's random/PCA
        # start, bindings/mani_kit_door.mojo `mk_image`) and the spectral affinity
        # connect once; every kit primitive is one of its table operations
        # (x_decomp/descriptor_schema.mojo), so no other entry is read by name
        # (lane entangled-mkit, 2026-11-12: the 26 named x_decomp_* cells
        # this list carried for the Python-holding MKit are removed).
        "x_decomp_native_api_v1",
    )),
    "base": ("_mojolearn", (
        "spectral_preparation_native_api_v1",
    )),
    "cluster": ("_mojolearn_x_cluster", ("spectral_assign_native_api_v1",)),
}


def _capture(binding, name, symbols):
    # Resolve host proxies once. Retain exact compiled functions in an exact
    # module with no __getattr__, so native operations cannot reenter a proxy.
    module = ModuleType(name)
    for symbol in symbols:  # glue: import-time ABI connection only
        try:
            entry = getattr(binding, symbol)
        except (AttributeError, ImportError):
            continue  # native owner distinguishes optional and required entries
        if type(entry) is not BuiltinFunctionType:
            raise TypeError(f"{name}.{symbol} must be a compiled entry")
        setattr(module, symbol, entry)
    # The UMAP owner reads the image's UMAP table key from the module it is
    # handed (bindings/mani_umap_native_door.mojo `_umap_check_mode`); the
    # binding records it at PyInit. The captured route carries it too, or every
    # UMAP fit/transform refused "publishes no native UMAP table"
    # (smoke-triage 2026-11-15: lanes umap, x-decomp-umap-options; 5 test files).
    try:
        key = getattr(binding, "umap_native_key")
    except (AttributeError, ImportError):
        key = None
    if type(key) is tuple:
        module.umap_native_key = key
    return module


def _exact_owner(binding):
    # glue: an exact compiled module (no proxy, no module __getattr__) whose
    # provider entry it owns itself
    if type(binding) is not ModuleType or "__getattr__" in binding.__dict__:
        return False
    entry = binding.__dict__.get("x_decomp_native_api_v1")
    return type(entry) is BuiltinFunctionType and entry.__self__ is binding


def _host_owner(binding):
    # glue: on a CPU-only install `_backend.binding` returns the lazy
    # `_HostBinding` facade. Its entries belong to the host extension module,
    # so a `_capture` copy would not own `x_decomp_native_api_v1` and the
    # manifold kit's provider admission
    # (providers/python/neural_descriptor_pyapi_door.mojo `neural_descriptor_entry`)
    # refuses it. Resolve the facade to its exact host module at import, as
    # _native_family_bridge.connect_bindings does.
    from . import _backend
    return _backend.exact_host_module(binding)


def connect():
    global _CONNECTED
    if _CONNECTED is not None:
        return _CONNECTED
    from . import _backend
    from ._metrics_impl import _get_binding

    _backend.default_mode()  # initialize its live dictionary entry once
    routes = {}
    for group, (binding_name, symbols) in _SYMBOLS.items():  # glue: fixed import-time bindings
        choices = {}
        for mode in ("identical", "fast"):  # glue: retain offered tiers; no per-call loader
            if group == "base" and mode != "identical":
                continue
            try:
                binding = _get_binding(mode) if group == "metrics" else _backend.binding(binding_name, mode)
                if group == "decomp":
                    binding = _host_owner(binding)
                if group == "decomp" and _exact_owner(binding):
                    # The manifold kit connects the binding's kernel-call
                    # provider (x_decomp_native_api_v1), which is admitted
                    # only from the module that owns the entry, so the exact
                    # compiled module itself is retained, as
                    # _native_decomp_kit_bridge retains it.
                    choices[mode] = binding
                else:
                    choices[mode] = _capture(binding, binding_name + "." + mode, symbols)
            except (ImportError, AttributeError, ValueError, TypeError, RuntimeError) as error:
                # Missing tiers remain named refusals, never a Python fallback.
                choices[mode] = type(error).__name__ + ": " + str(error)
        routes[group] = choices
    _CONNECTED = routes
    return routes
