# SPDX-License-Identifier: Apache-2.0
"""Import-time ABI connection for the incremental native classical owner."""

_ERROR_ENTRY = None
_ERROR_CONTEXT = None
_ERROR_BRIDGES = {}


def _restore_not_fitted(message):
    return _ERROR_ENTRY(None, "restore_not_fitted", (message,), _ERROR_CONTEXT)


def _reduce_not_fitted(error):
    return _ERROR_ENTRY(error, "reduce_not_fitted", (), _ERROR_CONTEXT)


def _decomp_cells(decomp):
    # glue: cache only the compiled TSQR entry points of one decomposition
    # binding (the selected device's, or the CPU one for the Host* classes).
    from types import BuiltinFunctionType
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
    return cells


def connect_host(host_ridge_class, host_linear_class):
    # glue: connect the existing explicit Host* CPU ABI once at setup
    from types import BuiltinFunctionType

    entry, context = connect(host_ridge_class, host_linear_class)
    owner = object.__new__(host_ridge_class)
    object.__setattr__(owner, "numeric_mode", "identical")
    module = host_ridge_class._bind(owner, "_mojolearn_estimators")
    cpu = {}
    names = ("estimators_vendor", "estimators_numeric_mode", "ols_predict",
             "ols_fit", "ridge_fit", "lm_col_sums", "lm_means_finish",
             "lm_center", "lm_intercept", "lm_scale_rows", "py2mojo_rows")
    optional = ("linear_gram_fit", "linear_gram_solve_default", "ridge_fit_resident",
                "ridge_resident_default", "lm_classical_stats", "lm_col_sums_pair",
                "ols_fit_resident", "ols_normal_eq_default")
    for name in names + optional:  # glue: retain verified compiled CPU symbols at connection
        function = getattr(module, name, None)
        if function is None and name in optional:
            continue
        if not isinstance(function, BuiltinFunctionType):
            raise TypeError(f"explicit host model requires compiled {name}")
        cpu[name] = function
    if module.estimators_vendor() != "cpu" or module.estimators_numeric_mode() != 1:
        raise RuntimeError("explicit host model requires the CPU identical binary")
    # The TSQR and x_linear helpers are the CPU binaries too: the device
    # ones report the device vendor, which the native fit refuses beside
    # the CPU estimator entries (and would not give the CPU column's bits).
    from ._classical_host import _host_fit_helpers
    decomp, linear = _host_fit_helpers()
    try:
        cpu["decomp"] = _decomp_cells(decomp)
    except (ImportError, AttributeError, TypeError, ValueError, RuntimeError) as error:
        cpu["decomp"] = type(error).__name__ + ": " + str(error)
    linear_fit = getattr(linear, "x_linear_fit", None)
    if linear_fit is not None:  # only a weighted or one-column HostRidge fit needs it
        if not isinstance(linear_fit, BuiltinFunctionType):
            raise TypeError("explicit host model requires compiled x_linear_fit")
        cpu["x_linear_fit"] = linear_fit
    connected = {mode: dict(cpu) for mode in context["classical_bindings"]}  # glue: one CPU entry family per connected mode
    context.update(classical_bindings=connected, classical_explicit_cpu=True,
                   classical_metric_vendor=context["vendor"], vendor="cpu")
    return entry, context


_NPY_DTYPES = ("<f4", "<f8", "<i4", "<i8", "<u4", "<u1", "<f2", "=f4", "=f8", "=i4", "=i8", "=u4")


def _numpy():
    # glue: NumPy is optional and importing mojolearn never loads it. A
    # caller's ndarray exists only once NumPy is loaded, so the loaded module
    # decides: without it the owner reads and returns mojolearn Arrays, and
    # the native entries connect the NumPy context when NumPy is loaded later
    # (`_connect_numpy`).
    import sys
    return sys.modules.get("numpy")


def _archive_npy_dtypes(np):
    # glue: the archive dtypes the reader admits: NumPy dtypes with NumPy,
    # the Array typestr each one is stored as without it.
    if np is not None:
        npy_dtypes = {name: np.dtype(name) for name in _NPY_DTYPES}  # glue: connect fixed archive dtype metadata
        npy_dtypes.update({"|u1": np.dtype("u1"), "|b1": np.dtype("u1")})
    else:
        npy_dtypes = {name: "<" + name[1:] for name in _NPY_DTYPES}  # glue: connect fixed archive dtype metadata
        npy_dtypes.update({"|u1": "<u1", "|b1": "<u1"})
    return npy_dtypes


def _numpy_context(np):
    # glue: the NumPy admission types and dtypes (the Sequential context)
    # plus the NumPy archive dtypes.
    from ._x_sequence_model import _SEQUENTIAL_CONTEXT
    context = dict(_SEQUENTIAL_CONTEXT)
    context["archive_npy_dtypes"] = _archive_npy_dtypes(np)
    return context


def _connect_numpy(context):
    # glue: called by the native classical entries (bindings/
    # classical_attributes.mojo classical_connect_numpy) the first time they
    # run with NumPy loaded after a NumPy-free connection. Every entry the
    # NumPy-free connection installed and nothing changed since is replaced
    # by its NumPy counterpart; entries set after connection (the explicit
    # host vendor) are kept.
    import sys
    numpy_free = context.pop("classical_numpy_free")
    for key, value in _numpy_context(sys.modules["numpy"]).items():  # glue: connect the NumPy admission entries once
        if key not in context or context[key] is numpy_free.get(key, context):
            context[key] = value
    context.pop("classical_connect_numpy", None)


def _numpy_free_context():
    # glue: the admission types and Array entries of a NumPy-free install,
    # the NumPy-free subset of the Sequential context.
    from collections import OrderedDict, defaultdict
    from pathlib import PosixPath, PurePosixPath
    import mmap
    import sys
    from array import array
    from . import _backend
    from ._array import Array
    array_fields = {name: vars(Array)[name].__get__ for name in ("_mv", "shape", "dtype", "order")}  # glue: retain native slot descriptors once at package import
    return {
        "mapping_types": (dict, OrderedDict, defaultdict), "defaultdict_class": defaultdict,
        "native_default_factories": (list, tuple, dict, OrderedDict, defaultdict, bool, int, float, str, bytes, bytearray),
        "native_value_types": (type(None), str, bytes, bytearray, bool, int, float, range, Array, array, PosixPath,
                               PurePosixPath, type(iter([])), type(iter(())), type(iter(range(0))), type(reversed([]))),
        "buffer_owner_types": (type(None), bytes, bytearray, array, mmap.mmap),
        "scalar_types": (), "integer_types": (int,), "array_fields": array_fields,
        "array_type": array, "array_class": Array, "backend_fields": vars(_backend), "vendor": _backend.vendor(),
        "buffer_codes": {("b", 1): "b", ("i", 8): "q", ("u", 8): "Q", ("f", 8): "d"},
        "int64_min": -(2 ** 63), "int64_max": 2 ** 63 - 1, "uint64_max": 2 ** 64 - 1,
        "modules": sys.modules,
    }


def connect(ridge_class, linear_class=None):
    # glue: connect compiled entries and exact native buffer classes once
    global _ERROR_ENTRY, _ERROR_CONTEXT
    from types import BuiltinFunctionType, MethodDescriptorType, WrapperDescriptorType, ModuleType, MemberDescriptorType
    import _codecs
    import io
    import os
    import sys
    import zlib
    from . import _backend
    from .tokenizer import _binding
    from .exceptions import NotFittedError
    from ._native_classical_warnings import connect_warnings

    try:
        import _bz2
    except ImportError:
        _bz2 = None
    try:
        import _lzma
    except ImportError:
        _lzma = None

    np = _numpy()
    if np is not None:
        context = _numpy_context(np)
    else:
        context = _numpy_free_context()
        context["archive_npy_dtypes"] = _archive_npy_dtypes(None)
        context.update(classical_numpy_free=dict(context), classical_connect_numpy=_connect_numpy)
    context.update(not_fitted_class=NotFittedError, not_fitted_doc=NotFittedError.__doc__,
                   native_module_type=ModuleType, not_fitted_bridges=_ERROR_BRIDGES,
                   not_fitted_reduce=_reduce_not_fitted, not_fitted_restore=_restore_not_fitted,
                   active_not_fitted_class=NotFittedError)
    # The sklearn tag record layout is read natively from the loaded sklearn
    # module the first time sklearn asks for tags (bindings/classical_tags.mojo),
    # so connecting never imports the optional sklearn dependency.
    context.update(classical_tag_plan=None, native_member_descriptor=MemberDescriptorType)
    context.update(connect_warnings())
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
    context.update(archive_file_types=archive_types, archive_file_methods=tuple(archive_methods),
                   archive_crc32=zlib.crc32, archive_utf32_encode=_codecs.utf_32_le_encode,
                   archive_utf32_decode=_codecs.utf_32_le_decode, archive_literal_decode=_codecs.unicode_escape_decode,
                   archive_inflate=zlib.decompress, archive_bzip=None if _bz2 is None else _bz2.BZ2Decompressor,
                   archive_lzma=None if _lzma is None else _lzma.LZMADecompressor,
                   archive_lzma_properties=None if _lzma is None else _lzma._decode_filter_properties,
                   archive_lzma1=None if _lzma is None else _lzma.FILTER_LZMA1,
                   archive_lzma_raw=None if _lzma is None else _lzma.FORMAT_RAW,
                   archive_zstd=zstd_class,
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
                    entries["decomp"] = _decomp_cells(_backend.binding("_mojolearn_x_decomp", mode))
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
    kinds = {ridge_class: True}
    if linear_class is not None:
        kinds[linear_class] = False
    context.update(classical_kinds=kinds, classical_getframe=sys._getframe)
    context.update(ridge_class=ridge_class, linear_class=linear_class, classical_bindings=connected,
                   classical_metrics=metric_bindings, environment=os.environ._data,
                   ridge_name=ridge_class.__name__, linear_name="LinearRegression" if linear_class is None else linear_class.__name__,
                   ridge_saved_names=tuple(cls.__name__ for cls in ridge_class.__mro__),  # glue: preserve the declared saved-estimator ancestry
                   linear_saved_names=() if linear_class is None else tuple(cls.__name__ for cls in linear_class.__mro__),  # glue: preserve the declared saved-estimator ancestry
                   classical_fields=("numeric_mode", "fit_intercept", "alpha", "solver", "normalize",
                                     "coef_", "intercept_", "n_features_in_",
                                     "_mojolearn_fitted", "_mojolearn_fitting"))
    _ERROR_ENTRY, _ERROR_CONTEXT = _binding()._module.classical_native_call, context
    return _ERROR_ENTRY, context


def _layout(cls, attribute_descriptor):
    # glue: class-definition metadata the native attribute lookup reads; only
    # native descriptors and plain functions (whose binding is C) are bound.
    from types import (FunctionType, GetSetDescriptorType, MemberDescriptorType,
                       MethodDescriptorType, WrapperDescriptorType)
    mro = type.__dict__["__mro__"].__get__(cls)
    dictionaries = tuple(type.__dict__["__dict__"].__get__(base) for base in mro)  # glue: read class metadata once
    if type(cls) is not type or any(type(base) is not type for base in mro):  # glue: ordinary class layout admission
        raise TypeError("native classical models require ordinary Python classes")
    fields_descriptor = next((fields["__dict__"] for fields in dictionaries if "__dict__" in fields), None)  # glue: the instance dictionary member
    if type(fields_descriptor) is not GetSetDescriptorType:
        raise TypeError("native classical models require a native instance dictionary")
    names = {}
    for fields in dictionaries:  # glue: class lookup precedence
        for name, value in fields.items():  # glue: class attribute metadata
            names.setdefault(name, value)
    bindings = {}
    native = (GetSetDescriptorType, MemberDescriptorType, MethodDescriptorType, WrapperDescriptorType)
    for name, value in names.items():  # glue: bindable descriptor origin
        if type(value) in native or value is attribute_descriptor or type(value) is FunctionType:
            bindings[name] = (value, value.__get__)
        elif type(value) in (classmethod, staticmethod) and type(value.__func__) is FunctionType:
            bindings[name] = (value, value.__get__)
    return {"mro": mro, "fields_descriptor": fields_descriptor,
            "fields_get": fields_descriptor.__get__, "bindings": bindings}


def _unavailable(error):
    # glue: an install without the compiled owner keeps the package
    # importable; these classes then explain what is missing. (NumPy is
    # optional: without it the owner connects and returns Arrays.)
    message = f"mojolearn: LinearRegression and Ridge are unavailable: {error}"

    def unavailable(*args):
        raise ModuleNotFoundError(message, name=getattr(error, "name", None))
    return unavailable, {"classical_unavailable": message}


def connect_public(ridge_class, linear_class):
    """Connect the compiled owner of the public LinearRegression and Ridge
    classes once, at import; returns its entry and context."""
    from types import BuiltinFunctionType, MethodType
    try:
        entry, context = connect(ridge_class, linear_class)
        from .tokenizer import _binding
        module = _binding()._module
        attribute_entry = module.classical_attribute_call
        descriptor_entry = module.classical_attribute_descriptor
    except ImportError as error:
        return _unavailable(error)
    if not isinstance(entry, BuiltinFunctionType) or not isinstance(attribute_entry, BuiltinFunctionType) \
            or not isinstance(descriptor_entry, BuiltinFunctionType):
        raise TypeError("public classical methods require compiled ABI entries")
    descriptor = descriptor_entry(MethodType(attribute_entry, context))
    context.update(classical_attribute_descriptor=descriptor, classical_public_layouts={},
                   classical_public_installed=True)
    for cls in (ridge_class, linear_class):  # glue: the two public classes
        cls.__getattribute__ = descriptor
        context["classical_public_layouts"][cls] = _layout(cls, descriptor)
    return entry, context


def register_subclass(cls, context):
    """Connect a subclass of LinearRegression or Ridge when it is defined."""
    if context is None or "classical_attribute_descriptor" not in context:
        return
    host = context.get("classical_host_context")
    target = context
    if host is not None and any(issubclass(cls, base) for base in host["classical_kinds"]):  # glue: the two host classes
        target = host
    target["classical_kinds"][cls] = issubclass(cls, context["ridge_class"])
    try:
        context["classical_public_layouts"][cls] = _layout(cls, context["classical_attribute_descriptor"])
    except TypeError:
        # A class the native lookup cannot admit is refused when it is used.
        target["classical_kinds"].pop(cls, None)


def connect_host_public(host_ridge_class, host_linear_class, context):
    """Connect the explicit CPU host classes beside the public ones."""
    if "classical_attribute_descriptor" not in context:
        return
    entry, host_context = connect_host(host_ridge_class, host_linear_class)
    for cls in (host_ridge_class, host_linear_class):  # glue: the two host classes
        context["classical_kinds"].pop(cls, None)
    host_context.update(classical_public_layouts=context["classical_public_layouts"],
                        classical_public_installed=True,
                        classical_attribute_descriptor=context["classical_attribute_descriptor"],
                        classical_host_estimators={"Ridge": host_ridge_class,
                                                   "LinearRegression": host_linear_class})
    for cls in (host_ridge_class, host_linear_class):  # glue: the two host classes
        context["classical_public_layouts"][cls] = _layout(cls, context["classical_attribute_descriptor"])
    context["classical_host_context"] = host_context
