# SPDX-License-Identifier: Apache-2.0
"""Import-time connection of the Python-sharded multi-GPU drivers' native
owner (bindings/parallel_shard_native_door.mojo, on the native local process
pool). Each driver module registers its fixed classes, entries and contexts
here at its import; each public driver is then one call into it."""
from . import _native_process  # noqa: F401  (connects the native process pool)
from . import _backend
from .tokenizer import _binding

_native_parallel_shard = _binding()._module.parallel_shard_call


def _connect_binding(name, required, optional=()):
    # Connection only, before any operation or worker request. Resolve lazy
    # host proxies here and retain the exact compiled module. The native
    # owners admit only that module: they read its dictionary natively (never
    # a module __getattr__) and accept only builtin entries it owns, so a
    # copied module was refused by every worker ("native provider requires
    # compiled entry prepare_targets") and lost entries such as
    # kernel_methods_api_v1 that the owners read beyond the listed ones.
    from types import BuiltinFunctionType, ModuleType
    try:
        module = _backend.binding(name, "identical")
        module = _backend.exact_host_module(module)  # import-time: the CPU-only facade resolved to its compiled host module once
        if type(module) is not ModuleType:
            raise ImportError(f"parallel native binding requires the compiled {name} module")
        fields = vars(module)
        for key in required:  # glue: import-time check of a fixed set of compiled binding entries
            entry = fields.get(key)
            if entry is None:
                raise AttributeError(f"compiled {name} has no entry {key}")
            if type(entry) is not BuiltinFunctionType or entry.__self__ is not module:
                raise TypeError(f"parallel native binding requires compiled {name}.{key}")
        return module
    except (ImportError, AttributeError) as error:
        # A partial install stays importable; its native door reports this
        # error when the corresponding family is actually requested.
        return str(error)


_WORKER_BINDINGS = {
    "_mojolearn_kernel_methods": _connect_binding(
        "_mojolearn_kernel_methods", ("rbf_sampler_transform",), ("kernel_methods_numeric_mode",)),
    "_mojolearn_gp": _connect_binding(
        "_mojolearn_gp", ("gp_native_api_v1",), ("gp_numeric_mode",)),
    "_mojolearn": _connect_binding(
        "_mojolearn", ("prepare_targets", "gather_i64", "gather_f64")),
    "neighbors_base": _connect_binding(
        "_mojolearn", ("knn_classify", "knn_regress"),
        ("knn_index_prepare", "knn_index_release", "knn_classify_resident", "knn_regress_resident")),
    "_mojolearn_x_neighbors": _connect_binding("_mojolearn_x_neighbors", ("x_neighbors_native_api_v1",)),
    "neighbor_density": _connect_binding("_mojolearn_estimators", ("kde_score_samples",),
        ("kde_fit_prepare", "kde_fit_release", "kde_score_samples_resident")),
    # The shard worker calls the image's IVF-Flat operations table (the
    # module's `ivf_native_key`), not exports by name.
    "_mojolearn_ivf": _connect_binding("_mojolearn_ivf", (), ("ivf_numeric_mode",)),
}


def _table_keys():
    # glue: once at import, the (tier, vendor) keys of the images whose native
    # tables the owner calls (core/parallel_entry_table.mojo): the base image
    # (forest offset merge, reference top-k merge) and the IVF image (shard
    # plan, shard merge, distance finalize). An absent key makes the native
    # caller refuse when that family is actually requested.
    from ._native_table_key import connected_key
    keys = {"base": connected_key("_mojolearn"), "ivf": connected_key("_mojolearn_ivf")}
    if keys["ivf"] is not None and keys["ivf"][1] == 0:
        # A CPU install's IVF table is published by `_mojolearn_ivf_host`
        # alone (the search-only host binding publishes none): load it.
        try:
            _backend.load_host_module("_mojolearn_ivf_host")  # import-time: its init publishes the table
        except ImportError:
            pass
    return keys


_TABLE_KEYS = _table_keys()

#: Every GPU binding whose owner loops read a device plan and which registers
#: `device_plan_enter` / `device_plan_exit` / `device_plan_log`
#: (bindings/device_plan_binding_pack.mojo).
PLAN_BINDINGS = ('_mojolearn', '_mojolearn_estimators', '_mojolearn_gbdt', '_mojolearn_hdbscan', '_mojolearn_ivf',
                 '_mojolearn_linalg', '_mojolearn_metrics', '_mojolearn_mixture', '_mojolearn_resample',
                 '_mojolearn_rf', '_mojolearn_solver', '_mojolearn_trees', '_mojolearn_x_ann', '_mojolearn_x_cluster',
                 '_mojolearn_x_decomp', '_mojolearn_x_linear', '_mojolearn_x_neighbors', '_mojolearn_kernel_methods',
                 '_mojolearn_gp', '_mojolearn_svm')


def register(**fixed):
    """Add one driver module's fixed tables to the owner, once, at import."""
    # glue: import-time connection of compiled entries, classes and contexts.
    # Every driver's owner reads the family owners' shared `context` (array
    # types, codes, classes); it was registered only by the neighbors drivers,
    # so fit_scaler, the IVF shards and the GP classifier refused ("has no
    # context") in a process that had not imported those (fb6-smoke census).
    from ._native_family_bridge import shared_context
    _native_parallel_shard('register', dict(fixed, worker_bindings=_WORKER_BINDINGS,
                                          table_keys=_TABLE_KEYS, context=shared_context()))
