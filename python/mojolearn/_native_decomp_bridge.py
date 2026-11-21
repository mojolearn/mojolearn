# SPDX-License-Identifier: Apache-2.0
"""Import-time connection for the native owner of the decomposition
expansion's estimators (bindings/decomp_native_door.mojo)."""


def connect(classes, neighbors=False, metadata=None):
    # glue: connect the compiled decomp drivers once, before any operation;
    # `classes` maps each owned class's name to the class, `metadata` is
    # plain constant data the caller's module defines (merged as is)
    from types import BuiltinFunctionType
    from . import _backend
    from .linear_model import LinearRegression, Ridge
    from ._native_classical_bridge import connect as connect_classical, _numpy_free_context
    from .tokenizer import _binding
    from ._native_decomp_kit_bridge import _entries as _kit_entries

    _, shared = connect_classical(Ridge, LinearRegression)
    context = dict(shared)
    decomp = {}
    nmf = {}
    kpca_near = {}
    kpca_info = {}
    kpca_decomp = {}
    selected = _backend.default_mode()
    for mode in (selected, "fast" if selected == "identical" else "identical"):  # glue: fixed numeric-mode ABI connection
        try:
            binding = _backend.binding("_mojolearn_x_decomp", mode)
            # FastICA, FactorAnalysis, LatentDirichletAllocation and the random
            # projections run through the image's published operations table:
            # the kit's entries form (its table key and the pool entries a
            # resident fitted matrix is held through), no per-function map.
            decomp[mode] = _kit_entries(binding)
            # NMF runs through one typed descriptor, with no per-function map.
            factory = getattr(binding, "nmf_native_api_v1", None)
            nmf[mode] = factory if isinstance(factory, BuiltinFunctionType) else None
            factory = getattr(binding, "kpca_decomp_native_api_v1", None)
            kpca_decomp[mode] = factory if isinstance(factory, BuiltinFunctionType) else None
        except (ImportError, AttributeError, TypeError, ValueError, RuntimeError) as error:
            decomp[mode] = type(error).__name__ + ": " + str(error)
            nmf[mode] = decomp[mode]
            kpca_decomp[mode] = decomp[mode]
        if neighbors:
            try:
                binding = _backend.binding("_mojolearn_x_neighbors", mode)
                factory = getattr(binding, "x_neighbors_native_api_v1", None)
                kpca_near[mode] = factory if isinstance(factory, BuiltinFunctionType) else None
                factory = getattr(binding, "kpca_neighbors_native_api_v1", None)
                kpca_info[mode] = factory if isinstance(factory, BuiltinFunctionType) else None
            except (ImportError, AttributeError, TypeError, ValueError, RuntimeError) as error:
                kpca_near[mode] = type(error).__name__ + ": " + str(error)
                kpca_info[mode] = kpca_near[mode]
    # Fitted arrays and results are mojolearn Arrays, as `_M.out` made them.
    context["output_context"] = _numpy_free_context()
    context.update(decomp_classes=dict(classes), decomp_bindings=decomp, nmf_bindings=nmf,
                   kpca_neighbors_bindings=kpca_near, kpca_info_bindings=kpca_info,
                   kpca_decomp_bindings=kpca_decomp)
    context.update(metadata or {})
    # glue: the CPU image of `device="cpu"`, connected once beside the device set
    from ._native_decomp_device_bridge import connect_cpu_device, X_DECOMP_HOST
    connect_cpu_device(context, X_DECOMP_HOST, {"nmf_native_api_v1": "nmf_cpu_bindings",
                                                "kpca_decomp_native_api_v1": "kpca_decomp_cpu_bindings"})
    # glue: the same CPU image in the kit's entries form (its published table
    # key is (identical, cpu)); nothing is loaded in a CPU-only process
    if context.get("vendor") not in (None, "cpu"):
        try:
            cpu = _kit_entries(_backend.load_host_module(X_DECOMP_HOST))
        except (ImportError, AttributeError, TypeError, ValueError, RuntimeError, OSError) as error:
            cpu = type(error).__name__ + ": " + str(error)
        context["decomp_cpu_bindings"] = {"identical": cpu}
    if neighbors:
        connect_cpu_device(context, "_mojolearn_x_neighbors_host",
                           {"x_neighbors_native_api_v1": "kpca_neighbors_cpu_bindings",
                            "kpca_neighbors_native_api_v1": "kpca_info_cpu_bindings"})
    entry = _binding()._module.decomp_native_call
    if not isinstance(entry, BuiltinFunctionType):
        raise TypeError("the decomposition owner requires a compiled decomp_native_call")
    return entry, context


def connect_public(classes, neighbors=False, metadata=None):
    """Connect the compiled owner once, at import; returns its entry and
    context. An install without it keeps the package importable and the
    estimators explain what is missing."""
    try:
        return connect(classes, neighbors, metadata)
    except (ImportError, AttributeError) as error:
        message = f"mojolearn: the decomposition estimators are unavailable: {error}"

        def unavailable(*args):
            raise ModuleNotFoundError(message, name=getattr(error, "name", None))
        return unavailable, {"decomp_unavailable": message}
