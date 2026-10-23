# SPDX-License-Identifier: Apache-2.0
"""Import-time connection of the native decomposition owners that run the
x_decomp cells through bindings/decomp_kit_native.mojo (the dictionary
learning family, ...): their compiled door and the numeric-mode x_decomp
entries, retained once."""

# The x_decomp entries the native kit calls. A binding without an optional
# (resident) entry takes the host-address route, as `_Kit` does.
_REQUIRED = ("ew", "gemm", "colsum", "rowsum", "move", "order_f", "svd", "absmax_sign", "rand",
             "lasso_rows", "omp_rows", "lars_rows", "lu", "lu_solve", "dict_learning", "dict_minibatch")
_OPTIONAL = ("dev_ew", "dev_alloc", "dev_free", "dev_upload", "dev_download", "dev_gemm", "dev_colsum",
             "dev_rowsum", "dev_move", "dev_reduce", "dev_order_f", "dev_absmax", "dev_rand", "dev_svd",
             "dev_code_rows", "dev_lu", "pls_fit",
             "eigh", "dev_eigh", "reduce", "argmin_all", "dev_argmin_all", "select_smallest",
             "dev_select_smallest", "lu_aux", "dev_lu_aux", "mcd", "chi2_quantile", "consistency_factor",
             "dev_mcd_cov", "als_fit", "topn_desc", "gather")


def _entries(binding):
    # glue: retain checked compiled x_decomp symbols of one binding
    from types import BuiltinFunctionType
    entries = {}
    for suffix in _REQUIRED + _OPTIONAL:  # glue: fixed native exports
        name = "x_decomp_" + suffix
        try:
            entry = getattr(binding, name)
        except (AttributeError, ImportError):
            if suffix in _REQUIRED:
                raise
            continue
        if not isinstance(entry, BuiltinFunctionType):
            raise TypeError(f"the native decomposition owner requires compiled {name}")
        entries[name] = entry
    return entries


def connect(door, **classes):
    # glue: compiled entries and plain metadata are retained during import;
    # `door` names the tokenizer-host export, `classes` the classes the
    # owner tells apart natively.
    from types import BuiltinFunctionType
    from . import _backend
    from ._native_classical_bridge import _state_context, _numpy_free_context
    from .tokenizer import _binding

    context = _state_context()
    connected = {}
    selected = _backend.default_mode()
    for mode in (selected, "fast" if selected == "identical" else "identical"):  # glue: fixed numeric-mode ABI setup
        try:
            connected[mode] = _entries(_backend.binding("_mojolearn_x_decomp", mode))
        except (ImportError, AttributeError, TypeError, ValueError, RuntimeError, OSError) as error:
            connected[mode] = type(error).__name__ + ": " + str(error)
    # Every output and working buffer is a mojolearn Array, as `_M.out()` made.
    context.update(output_context=_numpy_free_context(), decomp_bindings=connected, **classes)
    entry = getattr(_binding()._module, door)
    if not isinstance(entry, BuiltinFunctionType):
        raise TypeError(f"the native decomposition owner requires a compiled {door}")
    return entry, context


def connect_public(door, **classes):
    """Connect a compiled decomposition owner once, at import; returns its
    entry and context. An install without the compiled owner keeps the
    package importable, and the estimators explain what is missing."""
    try:
        return connect(door, **classes)
    except (ImportError, AttributeError) as error:
        message = f"mojolearn: the native decomposition owner {door} is unavailable: {error}"

        def unavailable(*args):
            raise ModuleNotFoundError(message, name=getattr(error, "name", None))
        return unavailable, {"decomp_unavailable": message}
