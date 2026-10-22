# SPDX-License-Identifier: Apache-2.0
"""Import-time metadata connection for the native DeviceRows owner."""


def connect():
    # glue: retain native entry and immutable slot identities during import
    from types import BuiltinFunctionType, ModuleType, MemberDescriptorType
    import struct
    from . import _arena_io, model_selection
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
                   cache_local_type=type(_arena_io._LOCAL),
                   cache_local_getattribute=type(_arena_io._LOCAL).__getattribute__,
                   cache_attribute_error=AttributeError,
                   cache_hash_types=(type(None), bool, int, float, complex, str, bytes))
    _arena_io._ROWS_CALL = call
    _arena_io._ROWS_CONTEXT = context
