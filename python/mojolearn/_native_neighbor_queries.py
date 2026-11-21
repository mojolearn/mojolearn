# SPDX-License-Identifier: Apache-2.0
"""Import-time connections for complete native neighbor queries."""
from types import BuiltinFunctionType, MethodType, ModuleType
from ._native_robust_bridge import native_method

_CONTEXTS = {}
_BINDING_ALIASES = {}
_SYMBOLS = {
    # The k-NN and radius queries call the image's neighbors operations table
    # (its `neighbors_native_key`, captured below), not exports by name.
    "base": ("_mojolearn", ("mojolearn_numeric_mode", "knn_index_prepare", "knn_index_release")),
    "sort": ("_mojolearn_x_neighbors", ("x_neighbors_numeric_mode", "x_neighbors_native_api_v1",)),
    "ivf": ("_mojolearn_ivf", ("ivf_numeric_mode", "ivf_flat_search", "ivf_flat_index_prepare",
        "ivf_flat_index_search", "ivf_flat_index_release", "ivf_flat_index_export", "ivf_flat_extend")),
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
    # The k-NN owner reads the image's neighbors table key from the module it
    # is handed (bindings/neighbors_knn_native_door.mojo knn_ops_key); the
    # binding records it at PyInit. The captured route carries it too, or every
    # resident/device query through a route refused "publishes no native
    # neighbors table" (lane lazy-import-fix).
    # On a CPU-only install `binding` is the `_HostBinding` facade, not an
    # exact module: read the key through it (it forwards to the host module
    # that published it), or every CPU k-NN/radius query refused (smoke-triage
    # 2026-11-15: lanes knn*, radius*, x-neighbors-metrics; 8 test files).
    try:
        key = getattr(binding, "neighbors_native_key")
    except (ImportError, AttributeError):
        key = None
    if type(key) is tuple:
        result.neighbors_native_key = key
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
                binding = (_backend.load_host_module(host_names[name]) if host_names is not None  # import-time: import-time connection of explicit saved-model CPU entries
                           else _backend.binding(name, mode))
                choices[mode] = _capture(binding, name, symbols)
            except (ImportError, AttributeError, TypeError, ValueError, RuntimeError) as error:
                choices[mode] = type(error).__name__ + ": " + str(error)
        routes[group] = choices
    return routes


def _connect_devices(groups, routes):
    """The alias rows of the modules an estimator's device route resolves to.

    A query takes its image from the estimator's binding route (the shared
    estimator door), then reads its entries from that module's captured export
    table. The route hands back the module of `family_binding_connections`
    (this installation's device; on a CPU-only installation the loaded host
    module, not its facade) or, for `device="cpu"` beside a GPU set, the CPU
    module of `family_binding_connections_cpu`. Each gets its row here, at
    import; the CPU module gets its own captured table. Returns the family
    owners' shared context, under which that route is resolved."""
    from ._native_family_bridge import shared_context
    shared = shared_context()
    connections = shared.get("family_binding_connections", {})
    cpu = shared.get("family_binding_connections_cpu", {})
    for group in groups:  # glue: fixed query binding groups, at import
        name, symbols = _SYMBOLS[group]
        if name not in connections:
            raise TypeError(f"neighbor queries: connect_bindings({name!r}) must run before the query connection")
        for mode, record in connections[name].items():  # glue: the two fixed numeric-mode records of one binding name
            module, route = record[0], routes[group].get(mode)
            if type(module) is ModuleType and type(route) is ModuleType and id(module) not in _BINDING_ALIASES:
                _BINDING_ALIASES[id(module)] = (module, route)
        module = cpu.get(name, ("",))[0]
        if type(module) is ModuleType and id(module) not in _BINDING_ALIASES:
            _capture(module, name, symbols)
    return shared


def connect(classes, context, archive_context, groups):
    from ._native_classical_bridge import _layout
    from .tokenizer import _binding
    module = _binding()._module
    call, attribute = module.neighbor_query_call, module.neighbor_query_attribute_call
    if type(call) is not BuiltinFunctionType or type(attribute) is not BuiltinFunctionType:
        raise TypeError("neighbor queries require compiled public entries")
    private = dict(context)
    routes = _routes(groups)
    # The captured tables of this installation's set are reached through the
    # alias rows only (`_connect_devices`); the door reads no route table.
    private.update(neighbor_query_classes={},
                   neighbor_query_family_context=_connect_devices(groups, routes),
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
