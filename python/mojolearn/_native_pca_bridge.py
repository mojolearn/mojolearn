# SPDX-License-Identifier: Apache-2.0
"""Import-time connection for the native PCA owner."""


def connect(pca_class):
    # glue: connect native metadata and ABI entries once, before any operation
    from types import BuiltinFunctionType, GetSetDescriptorType
    from . import _backend
    from .linear_model import LinearRegression, Ridge
    from ._native_classical_bridge import connect as connect_classical, _numpy_free_context
    from .tokenizer import _binding

    # The PCA owner reads and returns ndarrays: it connects NumPy natively
    # when the caller has loaded it and refuses a fit or transform by name
    # otherwise (bindings/pca_native.mojo), so connecting never imports it.
    _, shared = connect_classical(Ridge, LinearRegression)
    context = dict(shared)
    descriptor = next((vars(base)["__dict__"] for base in pca_class.__mro__  # glue: connect the native instance dictionary descriptor
                       if "__dict__" in vars(base)), None)
    if type(pca_class) is not type or type(descriptor) is not GetSetDescriptorType:
        raise TypeError("PCA owner requires a standard native instance dictionary")
    connected = {}
    selected = _backend.default_mode()
    required = ("estimators_vendor", "estimators_numeric_mode", "pca_transform", "inverse_transform")
    # Additive exports: a build without one refuses the option that needs it
    # by name (bindings/pca_exact.mojo), not at import.
    optional = ("pca_fit_owned", "pca_fit_randomized", "pca_fit_full",
                "pca_whiten_transform", "pca_whiten_inverse_transform")
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
            connected[mode] = entries
        except (ImportError, AttributeError, TypeError, ValueError, RuntimeError) as error:
            connected[mode] = type(error).__name__ + ": " + str(error)
    # A densified SciPy sparse X is written as an Array (host_sparse_native).
    context["output_context"] = _numpy_free_context()
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


def connect_public(pca_class):
    """Connect the compiled owner of the public PCA class once, at import;
    returns its entry and context. An install without the compiled owner
    keeps the package importable and PCA explains what is missing."""
    try:
        return connect(pca_class)
    except ImportError as error:
        message = f"mojolearn: PCA is unavailable: {error}"

        def unavailable(*args):
            raise ModuleNotFoundError(message, name=getattr(error, "name", None))
        return unavailable, {"pca_unavailable": message}
