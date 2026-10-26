# SPDX-License-Identifier: Apache-2.0
"""Import-time connections for complete native neighbor queries."""
from types import BuiltinFunctionType, MethodType, ModuleType
from ._native_robust_bridge import native_method

_CONTEXTS = {}
_BINDING_ALIASES = {}
_SYMBOLS = {
    "base": ("_mojolearn", ("mojolearn_numeric_mode", "knn_search", "rbc_knn_search", "knn_index_prepare",
        "knn_index_release", "knn_search_resident", "radius_neighbors_count", "radius_neighbors_fill")),
    "sort": ("_mojolearn_x_neighbors", ("x_neighbors_numeric_mode", "xn_p2m_row_sort",)),
    "ivf": ("_mojolearn_ivf", ("ivf_numeric_mode", "ivf_flat_search", "ivf_flat_index_prepare",
        "ivf_flat_index_search", "ivf_flat_index_release", "ivf_flat_index_export")),
}


def _capture(binding, name, symbols):
    result = ModuleType(name)
    for symbol in symbols:  # glue: fixed import-time compiled exports
        try:
            entry = getattr(binding, symbol)
        except (ImportError, AttributeError):
            continue  # optional resident entries keep their established fallback
        if type(entry) is not BuiltinFunctionType:
            raise TypeError(f"{name}.{symbol} must be compiled")
        setattr(result, symbol, entry)
    _BINDING_ALIASES[id(binding)] = (binding, result)
    _BINDING_ALIASES[id(result)] = (result, result)
    return result


def _routes(groups, host_names=None):
    from . import _backend
    _backend.default_mode()  # initialize the live default once during import
    routes = {}
    for group in groups:  # glue: fixed query binding groups
        name, symbols = _SYMBOLS[group]
        choices = {}
        for mode in (("identical",) if host_names is not None else ("identical", "fast")):  # glue: import-time fixed supported numeric-mode routes
            try:
                binding = (_backend.load_host_module(host_names[name]) if host_names is not None  # cpu-route: import-time connection of explicit saved-model CPU entries
                           else _backend.binding(name, mode))
                choices[mode] = _capture(binding, name, symbols)
            except (ImportError, AttributeError, TypeError, ValueError, RuntimeError) as error:
                choices[mode] = type(error).__name__ + ": " + str(error)
        routes[group] = choices
    return routes


def connect(classes, context, archive_context, groups):
    from ._native_classical_bridge import _layout
    from .tokenizer import _binding
    module = _binding()._module
    call, attribute = module.neighbor_query_call, module.neighbor_query_attribute_call
    if type(call) is not BuiltinFunctionType or type(attribute) is not BuiltinFunctionType:
        raise TypeError("neighbor queries require compiled public entries")
    private = dict(context)
    private.update(neighbor_query_classes={}, neighbor_query_routes=_routes(groups),
                   neighbor_query_binding_aliases=_BINDING_ALIASES, neighbor_query_classifier_classes={},
                   neighbor_query_host_routes={}, classical_public_layouts={}, classical_public_installed=True,
                   host_archive_finalizer_descriptor=archive_context["host_archive_context"].get("host_archive_finalizer_descriptor"))
    descriptor = module.classical_attribute_descriptor(MethodType(attribute, private))
    private["classical_attribute_descriptor"] = descriptor
    for cls, kind in classes.items():  # glue: fixed public classes, connected after native lifetime setup
        cls.__getattribute__ = descriptor
        private["neighbor_query_classes"][cls] = (kind, False)
        private["classical_public_layouts"][cls] = _layout(cls, descriptor)
        _CONTEXTS[cls] = private
    return call, private


def connect_host(classes, host_names, host_archive):
    from ._native_classical_bridge import _layout
    # Each explicit CPU class carries the host archive's native finalizer in
    # place of its GPU base's (`_native_host_archive_bridge.connect`). The
    # shared query context admits it as the connected peer of the public
    # lifetime descriptor; without it every query and attribute read on a
    # loaded host model was refused as a Python finalizer hook.
    peer = host_archive.get("host_archive_finalizer_descriptor")
    for host, base in classes.items():  # glue: fixed supported saved-model classes
        context = _CONTEXTS[base]
        if peer is not None:
            context["host_archive_peer_finalizer_descriptor"] = peer
        kind = context["neighbor_query_classes"][base][0]
        groups = ("ivf",) if kind == "ivf" else ("base", "sort")
        context["neighbor_query_host_routes"].update(_routes(groups, host_names))
        descriptor = context["classical_attribute_descriptor"]
        host.__getattribute__ = descriptor
        context["neighbor_query_classes"][host] = (kind, True)
        if base in context["neighbor_query_classifier_classes"]:
            context["neighbor_query_classifier_classes"][host] = True
        context["classical_public_layouts"][host] = _layout(host, descriptor)
