# SPDX-License-Identifier: Apache-2.0
"""Import-time ABI connection for the incremental native classical owner."""


def connect(ridge_class, linear_class=None):
    # glue: connect compiled entries and exact native buffer classes once
    from types import BuiltinFunctionType
    from . import _backend
    from .tokenizer import _binding
    from ._x_sequence_model import _SEQUENTIAL_CONTEXT

    context = dict(_SEQUENTIAL_CONTEXT)
    connected = {}
    selected = _backend.default_mode()
    for mode in (selected, "fast" if selected == "identical" else "identical"):  # glue: native ABI connection at setup only
        try:
            estimator = _backend.binding("_mojolearn_estimators", mode)
            linear = _backend.binding("_mojolearn_x_linear", mode)
            entries = {}
            required = ("estimators_vendor", "estimators_numeric_mode", "lm_col_sums",
                        "lm_means_finish", "lm_center", "lm_intercept", "ridge_fit", "ols_predict")
            optional = ("linear_gram_fit", "linear_gram_solve_default", "ridge_fit_resident",
                        "ridge_resident_default", "lm_classical_stats", "lm_col_sums_pair",
                        "ols_fit_resident", "ols_normal_eq_default")
            if linear_class is not None:
                required += ("ols_fit", "lm_scale_rows", "py2mojo_rows")
            for name in required + optional:  # glue: retain raw native symbols at setup only
                try:
                    entry = getattr(estimator, name)
                except (AttributeError, ImportError):
                    if name in required:
                        raise
                    continue
                if not isinstance(entry, BuiltinFunctionType):
                    raise TypeError(f"classical owner requires compiled {name}")
                entries[name] = entry
            entry = linear.x_linear_fit
            if not isinstance(entry, BuiltinFunctionType):
                raise TypeError("classical owner requires compiled x_linear_fit")
            entries["x_linear_fit"] = entry
            if linear_class is not None:
                try:
                    decomp = _backend.binding("_mojolearn_x_decomp", mode)
                    cells = {}
                    required_cells = ("vendor", "numeric_mode", "tsqr_r", "ew", "gemm", "colsum", "rowsum",
                                      "move", "order_f", "svd", "rank_above", "orth_diag", "geqrf", "orgqr")
                    optional_cells = ("idn_flags", "ols_tsqr_r", "dev_ew", "dev_alloc", "dev_free", "dev_upload",
                                      "dev_download", "dev_gemm", "dev_colsum", "dev_rowsum", "dev_move",
                                      "dev_reduce", "dev_order_f", "dev_orth_diag", "dev_svd")
                    for suffix in required_cells + optional_cells:  # glue: cache only compiled decomp entry points during connection
                        name = "x_decomp_" + suffix
                        try:
                            entry = getattr(decomp, name)
                        except (AttributeError, ImportError):
                            if suffix in required_cells:
                                raise
                            continue
                        if not isinstance(entry, BuiltinFunctionType):
                            raise TypeError(f"classical TSQR owner requires compiled {name}")
                        cells[name] = entry
                    if "x_decomp_dev_ew" in cells:
                        for suffix in optional_cells[3:-1]:  # glue: complete native resident family admission
                            if "x_decomp_" + suffix not in cells:
                                raise ImportError(f"classical TSQR resident entry missing: {suffix}")
                    entries["decomp"] = cells
                except (ImportError, AttributeError, TypeError, ValueError, RuntimeError) as error:
                    entries["decomp"] = type(error).__name__ + ": " + str(error)
            connected[mode] = entries
        except (ImportError, AttributeError, TypeError, ValueError, RuntimeError) as error:
            connected[mode] = type(error).__name__ + ": " + str(error)
    context.update(ridge_class=ridge_class, linear_class=linear_class, classical_bindings=connected,
                   classical_fields=("numeric_mode", "fit_intercept", "alpha", "solver", "normalize",
                                     "coef_", "intercept_", "n_features_in_"))
    return _binding()._module.classical_native_call, context
