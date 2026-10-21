# SPDX-License-Identifier: Apache-2.0
"""Import-time ABI connection for the incremental native classical owner."""


def connect(ridge_class, linear_class=None):
    # glue: connect compiled entries and exact native buffer classes once
    from types import BuiltinFunctionType, MethodDescriptorType, WrapperDescriptorType
    import _codecs
    import io
    import os
    import zlib
    import numpy as np
    from . import _backend
    from .tokenizer import _binding
    from ._x_sequence_model import _SEQUENTIAL_CONTEXT

    try:
        import _bz2
    except ImportError:
        _bz2 = None
    try:
        import _lzma
    except ImportError:
        _lzma = None

    context = dict(_SEQUENTIAL_CONTEXT)
    archive_types = (io.BytesIO, io.FileIO, io.BufferedReader, io.BufferedWriter, io.BufferedRandom)
    archive_methods = []
    for cls in archive_types:  # glue: cache exact native file descriptors at ABI connection
        methods = {}
        for name in ("read", "write", "seek", "tell", "flush", "close"):  # glue: validate native descriptor origin once
            entry = getattr(cls, name)
            if not isinstance(entry, (MethodDescriptorType, WrapperDescriptorType)):
                raise TypeError(f"classical archives require native {cls.__name__}.{name}")
            methods[name] = entry
        archive_methods.append(methods)
    codec_entries = (zlib.crc32, zlib.decompress, _codecs.utf_32_le_encode, _codecs.utf_32_le_decode,
                     _codecs.unicode_escape_decode)
    if _lzma is not None:
        codec_entries += (_lzma._decode_filter_properties,)
    for entry in codec_entries:  # glue: validate format metadata C entries once
        if not isinstance(entry, BuiltinFunctionType):
            raise TypeError("classical archives require native CRC and Unicode metadata entries")
    decompressor_types = (() if _bz2 is None else (_bz2.BZ2Decompressor,)) + (() if _lzma is None else (_lzma.LZMADecompressor,))
    for cls in decompressor_types:  # glue: native archive codec admission at setup
        if (type(cls) is not type or not isinstance(cls.decompress, MethodDescriptorType)
                or not isinstance(cls.__new__, BuiltinFunctionType)
                or not isinstance(cls.__init__, WrapperDescriptorType)):
            raise TypeError("classical archives require native byte decompressor types")
    try:
        from compression import zstd
    except ImportError:
        zstd_class = None
    else:
        zstd_class = zstd.ZstdDecompressor
        if (type(zstd_class) is not type or not isinstance(zstd_class.decompress, MethodDescriptorType)
                or not isinstance(zstd_class.__new__, BuiltinFunctionType)
                or not isinstance(zstd_class.__init__, WrapperDescriptorType)):
            raise TypeError("classical archives require a native Zstandard byte decompressor")
    npy_dtypes = {name: np.dtype(name) for name in ("<f4", "<f8", "<i4", "<i8", "<u4", "<u1", "<f2", "=f4", "=f8", "=i4", "=i8", "=u4")}  # glue: connect fixed archive dtype metadata
    npy_dtypes.update({"|u1": np.dtype("u1"), "|b1": np.dtype("u1")})
    context.update(archive_file_types=archive_types, archive_file_methods=tuple(archive_methods),
                   archive_crc32=zlib.crc32, archive_utf32_encode=_codecs.utf_32_le_encode,
                   archive_utf32_decode=_codecs.utf_32_le_decode, archive_literal_decode=_codecs.unicode_escape_decode,
                   archive_inflate=zlib.decompress, archive_bzip=None if _bz2 is None else _bz2.BZ2Decompressor,
                   archive_lzma=None if _lzma is None else _lzma.LZMADecompressor,
                   archive_lzma_properties=None if _lzma is None else _lzma._decode_filter_properties,
                   archive_lzma1=None if _lzma is None else _lzma.FILTER_LZMA1,
                   archive_lzma_raw=None if _lzma is None else _lzma.FORMAT_RAW,
                   archive_zstd=zstd_class, archive_npy_dtypes=npy_dtypes,
                   archive_cp437=tuple(bytes(range(256)).decode("cp437")),
                   archive_npy_magic=b"\x93NUMPY", archive_zip_end=b"PK\x05\x06")
    connected = {}
    metric_bindings = {}
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
        try:
            metrics = _backend.binding("_mojolearn_x_metrics", mode)
            metric_entries = {}
            metric_names = ("x_metrics_run", "x_metrics_vendor", "x_metrics_numeric_mode",
                            "x_metrics_run_out", "x_metrics_run_ranges")
            for name in metric_names:  # glue: cache raw metric program entries once at package connection
                try:
                    entry = getattr(metrics, name)
                except (AttributeError, ImportError):
                    if name in ("x_metrics_run_out", "x_metrics_run_ranges"):
                        continue
                    raise
                if not isinstance(entry, BuiltinFunctionType):
                    raise TypeError(f"classical score requires compiled {name}")
                metric_entries[name] = entry
            metric_bindings[mode] = metric_entries
        except (ImportError, AttributeError, TypeError, ValueError, RuntimeError) as error:
            metric_bindings[mode] = type(error).__name__ + ": " + str(error)
    context.update(ridge_class=ridge_class, linear_class=linear_class, classical_bindings=connected,
                   classical_metrics=metric_bindings, environment=os.environ._data,
                   classical_fields=("numeric_mode", "fit_intercept", "alpha", "solver", "normalize",
                                     "coef_", "intercept_", "n_features_in_",
                                     "_mojolearn_fitted", "_mojolearn_fitting"))
    return _binding()._module.classical_native_call, context
