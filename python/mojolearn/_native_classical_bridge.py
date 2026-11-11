# SPDX-License-Identifier: Apache-2.0
"""Import-time ABI connection for the incremental native classical owner."""

_ERROR_ENTRY = None
_ERROR_CONTEXT = None
_ERROR_BRIDGES = {}


def _restore_not_fitted(message):
    return _ERROR_ENTRY(None, "restore_not_fitted", (message,), _ERROR_CONTEXT)


def _reduce_not_fitted(error):
    return _ERROR_ENTRY(error, "reduce_not_fitted", (), _ERROR_CONTEXT)


def _native_table_cells(module, prefix):
    # glue: create an immutable compiled connection owner once at import.
    # The compiled connector admits exact images/metadata and the published
    # endpoint; product operations never invoke the capsule factory.
    from .tokenizer import _binding
    families = {"estimators": (0, "linear_fit_native_api_v1"),
                "x_linear": (1, "ridge_fit_native_api_v1"),
                "x_decomp": (2, "x_decomp_native_api_v1"),
                "x_metrics": (3, "metrics_program_native_api_v1")}
    family, factory = families[prefix]
    owner = _binding()._module.linear_entry_connection_v1(
        family, getattr(module, prefix + "_numeric_mode"),
        getattr(module, prefix + "_vendor"), getattr(module, factory))
    return {"_native_entry_connection": owner}


def _decomp_cells(decomp):
    # glue: the selected CPU/device TSQR image publishes its native table
    # during initialization. No capsule factory is needed by converted OLS.
    return _native_table_cells(decomp, "x_decomp")


def connect_host(host_ridge_class, host_linear_class, helpers):
    # glue: connect the existing explicit Host* CPU ABI once at setup;
    # `helpers` is the CPU (decomposition, x_linear) binary pair
    # `_classical_host._host_fit_helpers` loads.
    from types import BuiltinFunctionType

    entry, context = connect(host_ridge_class, host_linear_class)
    owner = object.__new__(host_ridge_class)
    object.__setattr__(owner, "numeric_mode", "identical")
    module = host_ridge_class._bind(owner, "_mojolearn_estimators")
    cpu = _native_table_cells(module, "estimators")
    names = ("estimators_vendor", "estimators_numeric_mode", "ols_predict",
             "ols_fit", "ridge_fit", "lm_col_sums", "lm_means_finish",
             "lm_center", "lm_intercept", "lm_scale_rows", "py2mojo_rows")
    optional = ("linear_gram_fit", "linear_gram_solve_default", "ridge_fit_resident",
                "ridge_resident_default", "lm_classical_stats", "lm_col_sums_pair",
                "ols_fit_resident", "ols_normal_eq_default", "ols_predict_wide",
                "lm_ols_prep")
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
    weighted = _native_table_cells(linear, "x_linear")
    cpu["_native_weighted_connection"] = weighted["_native_entry_connection"]
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
    from ._array import Array, SUPPORTED_DTYPES, _ARRAY_CONTEXT
    array_fields = {name: vars(Array)[name].__get__ for name in ("_mv", "_store", "_base", "_pin", "_addr", "_readonly", "shape", "dtype", "ndim", "size", "nbytes", "itemsize", "strides", "order")}  # glue: retain native slot descriptors once at package import
    array_storage_slots = _ARRAY_CONTEXT["from_buffer_slots"]  # glue: reuse the complete native storage descriptor inventory retained at import
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


#: The compiled doors `classical_native_call` forwards to (2026-11-07 host
#: split, docs/build/HOT_GROUPS.md section 6): context key -> entry name on the
#: tokenizer host, where import setup (host_surface.HOST_IMPORT_ALIASES) put
#: the owner's entry, or its native refusal when that owner is not built.
_CLASSICAL_DOORS = (("classical_trees_door", "trees_native_call"),
                    ("classical_family_door", "family_native_call"))


def _classical_doors(binding):
    # glue: retain the separately compiled classical doors' entries once, at
    # import; the native door forwards to them and refuses when one is absent.
    from types import BuiltinFunctionType
    try:
        fields = vars(binding()._module)
    except ImportError:
        return {}
    doors = {}
    for key, name in _CLASSICAL_DOORS:  # glue: fixed door names, connected once at import
        entry = fields.get(name)
        if type(entry) is BuiltinFunctionType:
            doors[key] = entry
    return doors


def _state_context():
    # glue: connect compiled entries and exact native buffer classes once
    from types import BuiltinFunctionType, MethodDescriptorType, WrapperDescriptorType, ModuleType, MemberDescriptorType
    import io
    import os
    import sys
    from ._native_loaded_modules import PATH_TYPES, PATH_BASES
    from . import _backend
    from .tokenizer import _binding
    from .exceptions import NotFittedError
    from ._native_classical_warnings import connect_warnings

    # Both import orders use the same native admission setup. Do not copy
    # Sequential's cached context or install a Python first-call callback.
    context = _numpy_free_context()
    from ._native_owner_lifetime import connect_owner_context
    connect_owner_context(context)
    context["archive_npy_dtypes"] = _archive_npy_dtypes()
    context.update(not_fitted_class=NotFittedError, not_fitted_doc=NotFittedError.__doc__,
                   not_fitted_function_type=type(_reduce_not_fitted),
                   native_module_type=ModuleType, not_fitted_bridges=_ERROR_BRIDGES,
                   not_fitted_reduce=_reduce_not_fitted, not_fitted_restore=_restore_not_fitted,
                   active_not_fitted_class=NotFittedError)
    context.update(connect_warnings())
    context.update(_classical_doors(_binding))
    archive_types = (io.BytesIO, io.FileIO, io.BufferedReader, io.BufferedWriter, io.BufferedRandom)
    archive_mmap = context["owner_mmap_class"]
    if (type(archive_mmap) is not type or archive_mmap.__flags__ & 512
            or archive_mmap.__module__ != "mmap" or archive_mmap.__name__ != "mmap"
            or not isinstance(archive_mmap.__new__, BuiltinFunctionType)
            or not isinstance(archive_mmap.__init__, WrapperDescriptorType)):
        raise TypeError("classical archives require the native mmap owner")
    if not isinstance(io.open, BuiltinFunctionType):
        raise TypeError("classical archives require native binary file opening")
    archive_methods = []
    for cls in archive_types:  # glue: cache exact native file descriptors at ABI connection
        methods = {}
        for name in ("read", "readinto", "readable", "fileno", "write", "seek", "tell", "flush", "close"):  # glue: validate native descriptor origin once
            entry = getattr(cls, name)
            if not isinstance(entry, (MethodDescriptorType, WrapperDescriptorType)):
                raise TypeError(f"classical archives require native {cls.__name__}.{name}")
            methods[name] = entry
        if cls is io.BytesIO:
            entry = cls.getbuffer
            if not isinstance(entry, MethodDescriptorType):
                raise TypeError("classical archives require native BytesIO.getbuffer")
            methods["getbuffer"] = entry
        archive_methods.append(methods)
    context.update(archive_file_types=archive_types, archive_file_methods=tuple(archive_methods),
                   archive_python_minor=sys.version_info.minor, archive_python_micro=sys.version_info.micro,
                   archive_open=io.open, archive_bytes_type=bytes, archive_bytearray_type=bytearray,
                   archive_memoryview_type=memoryview, archive_attribute_error=AttributeError,
                   archive_path_types=PATH_TYPES,
                   archive_path_bases=PATH_BASES,
                   archive_path_member_type=MemberDescriptorType,
                   archive_unicode_decode_error=UnicodeDecodeError, archive_unicode_encode_error=UnicodeEncodeError)
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
            entries = _native_table_cells(estimator, "estimators")
            required = ("estimators_vendor", "estimators_numeric_mode", "lm_col_sums",
                        "lm_means_finish", "lm_center", "lm_intercept", "ridge_fit", "ols_predict")
            optional = ("linear_gram_fit", "linear_gram_solve_default", "ridge_fit_resident",
                        "ridge_resident_default", "lm_classical_stats", "lm_col_sums_pair",
                        "ols_fit_resident", "ols_normal_eq_default", "ols_predict_wide",
                "lm_ols_prep")
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
            weighted = _native_table_cells(linear, "x_linear")
            entries["_native_weighted_connection"] = weighted["_native_entry_connection"]
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
            metric_entries = _native_table_cells(metrics, "x_metrics")
            name = "metrics_program_native_api_v1"
            entry = getattr(metrics, name)
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
        if type(value) in native or (attribute_descriptor is not None and value is attribute_descriptor) or type(value) is FunctionType:
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
        dictionaries = tuple(type.__dict__["__dict__"].__get__(base) for base in layout["mro"])  # glue: read class metadata once
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
