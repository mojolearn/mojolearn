# SPDX-License-Identifier: Apache-2.0
"""Import-time connections for spectral and UMAP native operation owners."""
from types import BuiltinFunctionType, ModuleType

_CONNECTED = None
_SYMBOLS = {
    "metrics": ("_mojolearn_metrics", (
        "metrics_numeric_mode", "umap_numeric_mode", "umap_fit_transform", "umap_fit_transform_ex", "umap_transform",
        "spectral_fit_predict_dense", "spectral_fit_predict_graph", "spectral_fit_predict_graph_state",
        "spectral_fit_predict_dataset", "spectral_fit_predict_dataset_state", "spectral_predict",
        "spectral_embedding_dense", "spectral_embedding_graph", "spectral_embedding_dataset",
    )),
    "decomp": ("_mojolearn_x_decomp", (
        "x_decomp_dev_alloc", "x_decomp_dev_free", "x_decomp_dev_upload", "x_decomp_dev_download",
        "x_decomp_dev_ew", "x_decomp_dev_sqdist", "x_decomp_ew", "x_decomp_sqdist",
        # The random/PCA UMAP initializer uses the same native MKit cells.
        "x_decomp_rand", "x_decomp_dev_rand", "x_decomp_gemm", "x_decomp_dev_gemm",
        "x_decomp_colsum", "x_decomp_dev_colsum", "x_decomp_svd", "x_decomp_dev_svd",
        "x_decomp_move", "x_decomp_dev_move", "x_decomp_absmax_sign", "x_decomp_dev_absmax",
        "x_decomp_order_f", "x_decomp_dev_order_f", "x_decomp_reduce", "x_decomp_dev_reduce",
        "x_decomp_argmin_all", "x_decomp_dev_argmin_all",
    )),
    "base": ("_mojolearn", (
        "nonzero_f32_count", "nonzero_f32_fill", "nonzero_f64_count", "nonzero_f64_fill", "knn_affinity_f32",
    )),
    "cluster": ("_mojolearn_x_cluster", ("x_cluster_call",)),
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
    return module


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
                choices[mode] = _capture(binding, binding_name + "." + mode, symbols)
            except (ImportError, AttributeError, ValueError, TypeError, RuntimeError) as error:
                # Missing tiers remain named refusals, never a Python fallback.
                choices[mode] = type(error).__name__ + ": " + str(error)
        routes[group] = choices
    _CONNECTED = routes
    return routes
