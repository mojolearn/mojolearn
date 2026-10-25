# SPDX-License-Identifier: Apache-2.0
"""Import-time connection of the Python-sharded multi-GPU drivers' native
owner (bindings/parallel_shard_native.mojo, on the native local process
pool). Each driver module registers its fixed classes, entries and contexts
here at its import; each public driver is then one call into it."""
from . import _native_process  # noqa: F401  (connects the native process pool)
from . import _backend
from .tokenizer import _binding

_native_parallel_shard = _binding()._module.parallel_shard_call


def _connect_binding(name, required, optional=()):
    # Connection only, before any operation or worker request. Resolve lazy
    # host proxies here and retain exclusively compiled entries. Native workers
    # must never invoke a module __getattr__ to resolve a missing entry.
    from types import BuiltinFunctionType, SimpleNamespace
    try:
        module = _backend.binding(name, "identical")
        entries = {"__name__": name}
        for key in required + optional:
            try:
                entry = getattr(module, key)
            except (ImportError, AttributeError):
                if key in required:
                    raise
                continue
            if type(entry) is not BuiltinFunctionType:
                raise TypeError(f"parallel native binding requires compiled {name}.{key}")
            entries[key] = entry
        return SimpleNamespace(**entries)
    except (ImportError, AttributeError) as error:
        # A partial install stays importable; its native door reports this
        # error when the corresponding family is actually requested.
        return str(error)


_WORKER_BINDINGS = {
    "_mojolearn_kernel_methods": _connect_binding(
        "_mojolearn_kernel_methods", ("rbf_sampler_transform",), ("kernel_methods_numeric_mode",)),
    "_mojolearn_gp": _connect_binding(
        "_mojolearn_gp", ("gpc_fit", "gpc_predict", "gpc_ovr_combine"), ("gp_numeric_mode",)),
    "_mojolearn": _connect_binding(
        "_mojolearn", ("prepare_targets", "gather_i64", "gather_f64", "msel_rebase_offsets_i32")),
}

#: Every GPU binding whose owner loops read a device plan and which registers
#: `device_plan_enter` / `device_plan_exit` / `device_plan_log`
#: (bindings/device_plan_binding.mojo).
PLAN_BINDINGS = ('_mojolearn', '_mojolearn_estimators', '_mojolearn_gbdt', '_mojolearn_hdbscan', '_mojolearn_ivf',
                 '_mojolearn_linalg', '_mojolearn_metrics', '_mojolearn_mixture', '_mojolearn_resample',
                 '_mojolearn_rf', '_mojolearn_solver', '_mojolearn_trees', '_mojolearn_x_ann', '_mojolearn_x_cluster',
                 '_mojolearn_x_decomp', '_mojolearn_x_linear', '_mojolearn_x_neighbors', '_mojolearn_kernel_methods',
                 '_mojolearn_gp', '_mojolearn_svm')


def register(**fixed):
    """Add one driver module's fixed tables to the owner, once, at import."""
    # glue: import-time connection of compiled entries, classes and contexts
    _native_parallel_shard('register', dict(fixed, worker_bindings=_WORKER_BINDINGS))
