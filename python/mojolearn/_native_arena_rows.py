# SPDX-License-Identifier: Apache-2.0
"""Import-time metadata connection for the native DeviceRows owner."""


def connect():
    # glue: retain native entry and immutable slot identities during import
    from types import BuiltinFunctionType, MemberDescriptorType, ModuleType
    from . import _arena_io
    from ._array import Array
    from ._native_classical_bridge import _numpy_free_context
    from .tokenizer import _binding

    call = _binding()._module.arena_rows_native_call
    if type(call) is not BuiltinFunctionType:
        raise TypeError("DeviceRows requires its compiled metadata entry")
    slots = {name: vars(Array)[name] for name in ("shape", "size", "dtype", "itemsize", "ndim")}  # glue: native Array metadata slot identities
    if any(type(slot) is not MemberDescriptorType for slot in slots.values()):  # glue: refuse non-native metadata at connection
        raise TypeError("DeviceRows requires native Array slots")
    context = _numpy_free_context()
    context.update(native_module_type=ModuleType, arena_array_slots=slots,
                   maxabs_rows_class=_arena_io.DeviceRows,
                   maxabs_rows_dict=vars(_arena_io.DeviceRows)["__dict__"])
    _arena_io._ROWS_CALL = call
    _arena_io._ROWS_CONTEXT = context
