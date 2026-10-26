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


def connect_host(host_ridge_class, host_linear_class, helpers):
    # glue: connect the existing explicit Host* CPU ABI once at setup;
    # `helpers` is the CPU (decomposition, x_linear) binary pair
    # `_classical_host._host_fit_helpers` loads.
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
    decomp, linear = helpers
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


def _archive_npy_dtypes():
    # Package-import metadata for the NumPy-free archive reader. The native
    # owner replaces these with dtype objects if the caller loads NumPy.
    return {"<f4": "<f4", "<f8": "<f8", "<i4": "<i4", "<i8": "<i8",
            "<u4": "<u4", "<u1": "<u1", "<f2": "<f2", "=f4": "<f4",
            "=f8": "<f8", "=i4": "<i4", "=i8": "<i8", "=u4": "<u4",
            "|u1": "<u1", "|b1": "<u1"}


def _numpy_free_context():
    # glue: the admission types and Array entries of a NumPy-free install,
    # the NumPy-free subset of the Sequential context.
    from collections import OrderedDict, defaultdict
    from pathlib import PosixPath, PurePosixPath
    import ctypes
    import mmap
    import sys
    from array import array
    from . import _backend
    from ._array import Array, SUPPORTED_DTYPES
    array_fields = {name: vars(Array)[name].__get__ for name in ("_mv", "shape", "dtype", "order")}  # glue: retain native slot descriptors once at package import
    array_storage_slots = {name: vars(Array)[name] for name in ("_store", "_mv", "_addr", "_readonly", "shape", "dtype", "order")}  # glue: fixed native storage layout retained at package import
    return {
        "array_storage_slots": array_storage_slots, "array_storage_dtypes": SUPPORTED_DTYPES,
        "mapping_types": (dict, OrderedDict, defaultdict), "defaultdict_class": defaultdict,
        "native_bytes_type": bytes, "native_bytearray_type": bytearray,
        "native_default_factories": (list, tuple, dict, OrderedDict, defaultdict, bool, int, float, str, bytes, bytearray),
        "native_value_types": (type(None), str, bytes, bytearray, bool, int, float, range, Array, array, PosixPath,
                               PurePosixPath, type(iter([])), type(iter(())), type(iter(range(0))), type(reversed([]))),
        "buffer_owner_types": (type(None), bytes, bytearray, array, mmap.mmap),
        # The base of the library's native output store (_buffer._output_store):
        # admission accepts a ctypes array class made directly from it with no
        # Python buffer hook (bindings/sequential_admission.mojo).
        "ctypes_array_base": ctypes.Array,
        "scalar_types": (), "integer_types": (int,), "array_fields": array_fields,
        "array_type": array, "array_class": Array, "backend_fields": vars(_backend), "vendor": _backend.vendor(),
        "buffer_codes": {("b", 1): "b", ("i", 1): "b", ("i", 2): "h", ("i", 4): "i",
                         ("i", 8): "q", ("u", 1): "B", ("u", 2): "H", ("u", 4): "I",
                         ("u", 8): "Q", ("f", 2): "H", ("f", 4): "f", ("f", 8): "d"},
        "int64_min": -(2 ** 63), "int64_max": 2 ** 63 - 1, "uint64_max": 2 ** 64 - 1,
        "modules": sys.modules,
    }


def _state_context():
    # glue: connect compiled entries and exact native buffer classes once
    from types import BuiltinFunctionType, MethodDescriptorType, WrapperDescriptorType, ModuleType, MemberDescriptorType
    import _codecs
    import unicodedata
    import io
    import os
    import sys
    import zlib
    from pathlib import PosixPath, PurePosixPath, Path, PurePath
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

    # Both import orders use the same native admission setup. Do not copy
    # Sequential's cached context or install a Python first-call callback.
    context = _numpy_free_context()
    context["archive_npy_dtypes"] = _archive_npy_dtypes()
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
    if not isinstance(io.open, BuiltinFunctionType):
        raise TypeError("classical archives require native binary file opening")
    archive_methods = []
    for cls in archive_types:  # glue: cache exact native file descriptors at ABI connection
        methods = {}
        for name in ("read", "write", "seek", "tell", "flush", "close"):  # glue: validate native descriptor origin once
            entry = getattr(cls, name)
            if not isinstance(entry, (MethodDescriptorType, WrapperDescriptorType)):
                raise TypeError(f"classical archives require native {cls.__name__}.{name}")
            methods[name] = entry
        archive_methods.append(methods)
    codec_entries = (zlib.crc32, zlib.decompress, unicodedata.lookup, _codecs.utf_32_le_encode, _codecs.utf_32_le_decode,
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
                   archive_open=io.open, archive_bytes_type=bytes, archive_bytearray_type=bytearray,
                   archive_memoryview_type=memoryview, archive_attribute_error=AttributeError,
                   archive_path_types=(PosixPath, PurePosixPath),
                   archive_path_bases=(PosixPath, PurePosixPath, Path, PurePath, object),
                   archive_path_member_type=MemberDescriptorType,
                   archive_crc32=zlib.crc32, archive_utf32_encode=_codecs.utf_32_le_encode,
                   archive_utf32_decode=_codecs.utf_32_le_decode, archive_literal_decode=_codecs.unicode_escape_decode,
                   archive_unicode_lookup=unicodedata.lookup, archive_unicode_key_error=KeyError,
                   archive_unicode_decode_error=UnicodeDecodeError, archive_unicode_encode_error=UnicodeEncodeError,
                   archive_inflate=zlib.decompress, archive_bzip=None if _bz2 is None else _bz2.BZ2Decompressor,
                   archive_lzma=None if _lzma is None else _lzma.LZMADecompressor,
                   archive_lzma_properties=None if _lzma is None else _lzma._decode_filter_properties,
                   archive_lzma1=None if _lzma is None else _lzma.FILTER_LZMA1,
                   archive_lzma_raw=None if _lzma is None else _lzma.FORMAT_RAW,
                   archive_zstd=zstd_class,
                   archive_cp437=tuple(bytes(range(256)).decode("cp437")),
                   archive_npy_magic=b"\x93NUMPY", archive_zip_end=b"PK\x05\x06")
    return context


def connect(ridge_class, linear_class=None):
    # glue: connect compiled entries and exact native buffer classes once
    global _ERROR_ENTRY, _ERROR_CONTEXT
    from types import BuiltinFunctionType
    import os
    import sys
    from . import _backend
    from .tokenizer import _binding
    context = _state_context()
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
    layout = {"mro": mro, "fields_descriptor": fields_descriptor,
              "fields_get": fields_descriptor.__get__, "bindings": bindings,
              "attribute_error": AttributeError}
    if _IMPORT_LAYOUTS is not None:
        _IMPORT_LAYOUTS.append(layout)
    return layout


# Layouts taken while the package imports, before the native constructor and
# setter installers replace __init__/set_params on their classes (the neighbor
# query, host inference and other module-level connections run first).
# refresh_installed_members rebinds those members once, then stops recording.
_IMPORT_LAYOUTS = []


def refresh_installed_members(names=("__init__", "set_params")):
    # glue: called once by the package after every constructor/setter installer
    # and before any later snapshot; a stale binding would refuse the native
    # constructor as an unconnected descriptor.
    from types import FunctionType
    global _IMPORT_LAYOUTS
    layouts, _IMPORT_LAYOUTS = _IMPORT_LAYOUTS, None
    for layout in layouts or ():  # glue: import-time class layouts, once
        bindings = layout["bindings"]
        dictionaries = tuple(type.__dict__["__dict__"].__get__(base) for base in layout["mro"])  # glue: class metadata
        for name in names:  # glue: the fixed installer-replaced member names
            if name not in bindings:
                continue
            value = next((fields[name] for fields in dictionaries if name in fields), None)  # glue: class lookup precedence
            if type(value) is FunctionType and value is not bindings[name][0]:
                bindings[name] = (value, value.__get__)


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


def connect_host_public(host_ridge_class, host_linear_class, context, helpers):
    """Connect the explicit CPU host classes beside the public ones."""
    if "classical_attribute_descriptor" not in context:
        return
    entry, host_context = connect_host(host_ridge_class, host_linear_class, helpers)
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
