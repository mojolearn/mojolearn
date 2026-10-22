# SPDX-License-Identifier: Apache-2.0
"""Import-time metadata connection for the native DeviceRows owner."""


def connect():
    # glue: retain native entry and immutable slot identities during import
    # The compound upload-error path constructs this exception natively.
    # Admit the immutable CPython builtin before retaining it: a replacement
    # installed before package import must never become a constructor hook.
    native_type = (0).__class__.__class__
    runtime_error = RuntimeError
    flags = native_type.__dict__["__flags__"].__get__
    if (native_type(runtime_error) is not native_type
            or flags(runtime_error) & 512 or not flags(runtime_error) & 256
            or native_type.__getattribute__(runtime_error, "__name__") != "RuntimeError"
            or native_type.__getattribute__(runtime_error, "__module__") != "builtins"):
        raise TypeError("DeviceCache requires the immutable native RuntimeError builtin")
    import_error = ImportError
    if (native_type(import_error) is not native_type
            or flags(import_error) & 512 or not flags(import_error) & 256
            or native_type.__getattribute__(import_error, "__name__") != "ImportError"
            or native_type.__getattribute__(import_error, "__module__") != "builtins"):
        raise TypeError("DeviceCache requires the immutable native ImportError builtin")
    from types import BuiltinFunctionType, ModuleType, MemberDescriptorType, SimpleNamespace
    import struct
    import re
    import sys
    from ._native_classical_warnings import connect_warnings
    from . import _arena_io, model_selection, _backend
    from ._buffer import _native
    from ._array import Array
    from ._native_classical_bridge import _numpy_free_context
    from ._native_owner_lifetime import connect_owner_context
    from .tokenizer import _binding

    call = _binding()._module.arena_rows_native_call
    if type(call) is not BuiltinFunctionType:
        raise TypeError("DeviceRows requires its compiled metadata entry")
    entries = {name: _native(name) for name in ('msel_put', 'msel_free', 'msel_take_rows')}  # glue: fixed model-selection snapshot entries, preserving the established identical base binding
    entries['cast'] = _native('cast_elements')
    if any(type(entry) is not BuiltinFunctionType for entry in entries.values()):  # glue: only connected native entries may own row preparation and gathering
        raise TypeError("DeviceRows requires compiled resident and cast entries")
    context = _numpy_free_context()
    connect_owner_context(context)
    # glue: describe the existing native half codec once; product copies use
    # Mojo bit operations and never call struct or convert through Python floats
    half_copy_canonical = struct.pack('<e', struct.unpack('<e', b'\x01\x7c')[0]) == b'\x00\x7e'
    # glue: retain native builtin constructors and exact scalar identities;
    # mutable builtins module names must never become product callbacks
    native_types = dict(type=type, object=object, list=list, tuple=tuple, dict=dict,
                        str=str, bytes=bytes, bytearray=bytearray, int=int,
                        float=float, bool=bool)
    context.update(arena_native_types=native_types, native_module_type=ModuleType, arena_memoryview=memoryview, arena_id=id,
                   arena_half_copy_canonical=half_copy_canonical,
                   arena_resident_class=model_selection._Resident,
                   arena_resident_dict=vars(model_selection._Resident)['__dict__'],
                   arena_resident_take=vars(model_selection._Resident)['take'],
                   arena_msel_put=entries['msel_put'], arena_msel_free=entries['msel_free'],
                   arena_msel_take_rows=entries['msel_take_rows'], arena_cast=entries['cast'],
                   maxabs_rows_class=_arena_io.DeviceRows,
                   maxabs_rows_dict=vars(_arena_io.DeviceRows)["__dict__"])
    # glue: retain ordinary metadata and native TLS access identities at import
    context.update(cache_type_mro=vars(type)["__mro__"].__get__,
                   cache_type_dict=vars(type)["__dict__"].__get__,
                   cache_object_getattribute=object.__getattribute__, cache_member_type=MemberDescriptorType,
                   cache_namespace_class=SimpleNamespace,
                   cache_namespace_getattribute=SimpleNamespace.__getattribute__,
                   cache_local_type=type(_arena_io._LOCAL),
                   cache_local_getattribute=type(_arena_io._LOCAL).__getattribute__,
                   cache_attribute_error=AttributeError,
                   cache_hash_types=(type(None), bool, int, float, complex, str, bytes))
    # glue: fixed scalar metadata/warning connection; no optional NumPy import
    scalar_context = dict(modules=sys.modules, module_type=ModuleType,
                          training_scalar_warnings=connect_warnings(),
                          training_scalar_locations={"cache_get": (_arena_io.DeviceCache.get.__code__.co_filename,
                                                                    _arena_io.DeviceCache.get.__code__.co_firstlineno,
                                                                    _arena_io.__name__)})
    # glue: retain the actual supported word-store entries once at package
    # connection; the CPU facade legitimately has no resident device store.
    # Do not admit arbitrary builtin callables by their type alone.
    store_routes = []
    live_entries = []
    for module_name, prefix in (("_mojolearn_x_prep", "x_prep"),  # glue: fixed import-time resident-store ABI
                                ("_mojolearn_x_metrics", "x_metrics")):
        for mode in ("identical", "fast"):  # glue: fixed supported numeric-mode connections
            try:
                module = _backend.binding(module_name, mode)
                put, free = getattr(module, prefix + "_dev_put"), getattr(module, prefix + "_dev_free")
            except (AttributeError, ImportError, ValueError):
                continue
            if type(put) is not BuiltinFunctionType or type(free) is not BuiltinFunctionType:
                raise TypeError("DeviceCache requires compiled native resident-store entries")
            store_routes.append((put, free))
            live = getattr(module, prefix + "_dev_live", None)
            if live is not None:
                if type(live) is not BuiltinFunctionType:
                    raise TypeError("DeviceCache requires a compiled native live-slot entry")
                live_entries.append(live)
    context.update(cache_scalar_context=scalar_context,
                   cache_store_routes=tuple(store_routes), cache_live_entries=tuple(live_entries),
                   cache_object_setattr=object.__setattr__,
                   cache_runtime_error=runtime_error,
                   cache_counter_types=(type(None), float, complex, str, bytes, bytearray,
                                        list, tuple, dict, set, frozenset, range),
                   cache_entry_types=(bool, int, float, complex, str, bytes, bytearray,
                                      list, tuple, dict, set, frozenset, range),
                   cache_type_flags=vars(type)["__flags__"].__get__,
                   cache_primitive_bases=(int, float, complex, str, bytes, bytearray),
                   cache_key_descriptor=vars(_arena_io.DeviceCache)["_key"],
                   cache_key_function=_arena_io.DeviceCache._key,
                   cache_little_endian=sys.byteorder == "little",
                   cache_deprecation_warning=DeprecationWarning,
                   cache_integer_types=(type(None), bool, int, float, complex, str, bytes,
                                        bytearray, list, tuple, dict, set, frozenset),
                   cache_itemsize_types=(type(None), object, bool, int, float, complex, str, bytes,
                                         bytearray, list, tuple, dict, set, frozenset, memoryview))
    # glue: preserve the CPU facade's by-name refusal without running its
    # Python __getattr__ during a public store capability probe.
    host_proxy = getattr(_backend, "_HostBinding", None)
    missing_proxy = getattr(_backend, "_NoGpuBinding", None)
    templates = ()
    no_cpu_entry = None
    if host_proxy is not None and missing_proxy is not None:
        no_cpu_entry = _backend.NoCpuEntry
        if type(no_cpu_entry) is not type or no_cpu_entry.__bases__ != (AttributeError,):
            raise TypeError("DeviceCache requires its ordinary NoCpuEntry exception class")
        if any(name in vars(no_cpu_entry) for name in ("__new__", "__init__", "__del__", "__getattribute__")):  # glue: never connect a Python exception-construction hook
            raise TypeError("DeviceCache cannot connect NoCpuEntry constructor hooks")
        markers = ("\0cache-module\0", "\0cache-entry\0", "\0cache-reason\0", "\0cache-host\0")
        pattern = "(" + "|".join(re.escape(marker) for marker in markers) + ")"  # glue: tokenize fixed CPU error metadata once during connection
        templates = tuple(tuple(markers.index(part) if part in markers else part  # glue: tokenize templates before inserting caller strings; marker-like caller text stays literal
                                for part in re.split(pattern, _backend._no_cpu_implementation(*markers[:3], basename)))  # glue: retain literal segments and slot indexes, not runtime Python formatting
                          for basename in (markers[3], None))  # glue: the two fixed host/missing-family refusal templates
    context.update(cache_cpu_host_proxy=host_proxy, cache_cpu_missing_proxy=missing_proxy,
                   cache_cpu_error_templates=templates, cache_no_cpu_entry=no_cpu_entry,
                   cache_import_error=import_error, cache_error_new=runtime_error.__mro__[-2].__new__,
                   cache_str_add=str.__add__,
                   cache_prefix_error_types=(type(None), bool, int, float, complex, bytes,
                                             bytearray, list, tuple, dict, set, frozenset, range))
    _arena_io._ROWS_CALL = call
    _arena_io._ROWS_CONTEXT = context
