# SPDX-License-Identifier: Apache-2.0
"""Import-time connection for the private native PCA owner."""


def connect(pca_class):
    # glue: connect native metadata and ABI entries once, before any operation
    from types import BuiltinFunctionType, GetSetDescriptorType
    from . import _backend
    from .linear_model import LinearRegression, Ridge
    from ._native_classical_bridge import connect as connect_classical
    from .tokenizer import _binding

    _, shared = connect_classical(Ridge, LinearRegression)
    context = dict(shared)
    descriptor = next((vars(base)["__dict__"] for base in pca_class.__mro__  # glue: connect the native instance dictionary descriptor
                       if "__dict__" in vars(base)), None)
    if type(pca_class) is not type or type(descriptor) is not GetSetDescriptorType:
        raise TypeError("PCA owner requires a standard native instance dictionary")
    connected = {}
    selected = _backend.default_mode()
    required = ("estimators_vendor", "estimators_numeric_mode", "pca_fit",
                "pca_transform", "inverse_transform", "pca_rank_finish")
    optional = ("pca_fit_full", "pca_whiten_transform", "pca_whiten_inverse_transform")
    if context["vendor"] == "cpu":
        decomp_names = ("x_decomp_pca_mle_rank",)  # cpu-route: connect only the explicit CPU rank entry on a CPU install
    else:
        decomp_names = ("x_decomp_dev_pca_mle_rank", "x_decomp_dev_alloc", "x_decomp_dev_free",
                        "x_decomp_dev_upload", "x_decomp_dev_download")
    randomized_names = ("ew", "gemm", "colsum", "rowsum", "move", "order_f", "svd", "rand", "orth", "absmax_sign")
    if context["vendor"] == "cpu":
        randomized_names += ("order_small",)  # cpu-route: host stable-order entry is connected only for CPU execution
    else:
        randomized_names += ("dev_ew", "dev_gemm", "dev_colsum", "dev_rowsum", "dev_move",
                             "dev_order_f", "dev_reduce", "dev_alloc", "dev_free", "dev_upload",
                             "dev_download", "dev_rand", "dev_orth", "dev_order_small", "dev_absmax")
    for mode in (selected, "fast" if selected == "identical" else "identical"):  # glue: fixed numeric-mode ABI connection
        try:
            binding = _backend.binding("_mojolearn_estimators", mode)
            entries = {}
            for name in required + optional:  # glue: retain checked compiled estimator symbols
                try:
                    entry = getattr(binding, name)
                except (AttributeError, ImportError):
                    if name in required:
                        raise
                    continue
                if not isinstance(entry, BuiltinFunctionType):
                    raise TypeError(f"PCA owner requires compiled {name}")
                entries[name] = entry
            try:
                decomp = _backend.binding("_mojolearn_x_decomp", mode)
                ranks = {}
                for name in decomp_names:  # glue: optional native rank-selection family
                    try:
                        entry = getattr(decomp, name)
                    except (AttributeError, ImportError):
                        continue
                    if not isinstance(entry, BuiltinFunctionType):
                        raise TypeError(f"PCA owner requires compiled {name}")
                    ranks[name] = entry
                entries["rank_bindings"] = ranks
                cells = {}
                for suffix in randomized_names + ("dev_svd", "s_flags", "dev_orth_ws"):  # glue: connect randomized PCA's fixed C primitive family
                    name = "x_decomp_" + suffix
                    try:
                        entry = getattr(decomp, name)
                    except (AttributeError, ImportError):
                        if suffix in randomized_names:
                            raise
                        continue
                    if not isinstance(entry, BuiltinFunctionType):
                        raise TypeError(f"PCA owner requires compiled {name}")
                    cells[name] = entry
                entries["randomized_bindings"] = cells
            except (ImportError, AttributeError, TypeError, ValueError, RuntimeError) as error:
                entries.setdefault("rank_bindings", type(error).__name__ + ": " + str(error))
                entries["randomized_bindings"] = type(error).__name__ + ": " + str(error)
            connected[mode] = entries
        except (ImportError, AttributeError, TypeError, ValueError, RuntimeError) as error:
            connected[mode] = type(error).__name__ + ": " + str(error)
    context.update(pca_class=pca_class, pca_dictionary_descriptor=descriptor, pca_bindings=connected,
                   pca_parameters=("copy", "iterated_power", "n_components", "n_oversamples",
                                   "numeric_mode", "power_iteration_normalizer", "random_state",
                                   "svd_solver", "tol", "whiten"),
                   pca_state=("components_", "mean_", "singular_values_", "explained_variance_",
                              "explained_variance_ratio_", "noise_variance_", "n_components_",
                              "n_features_in_", "n_samples_", "input_copied_",
                              "_mojolearn_fitted", "_mojolearn_fitting"))
    entry = _binding()._module.pca_native_call
    if not isinstance(entry, BuiltinFunctionType):
        raise TypeError("PCA owner requires a compiled pca_native_call")
    return entry, context
