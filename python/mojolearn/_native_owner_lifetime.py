# SPDX-License-Identifier: Apache-2.0
"""Import-time identities for native owner replacement admission."""


def connect_owner_context(context):
    # glue: retain supported native slot/weak-reference identities at import
    from types import BuiltinFunctionType, GetSetDescriptorType, MemberDescriptorType, MethodType
    import mmap
    import weakref
    from ._array import Array

    slots = {name: vars(Array)[name] for name in ("_store", "_base", "_pin", "_mv", "_addr", "_readonly", "shape", "dtype", "ndim", "size", "nbytes", "itemsize", "strides", "order")}  # glue: every Array slot retains an owner
    if any(type(slot) is not MemberDescriptorType for slot in slots.values()):  # glue: connect only fixed native owner slots
        raise TypeError("native owner admission requires native Array slots")
    if type(weakref.getweakrefs) is not BuiltinFunctionType:
        raise TypeError("native owner admission requires native weak-reference inspection")
    callback = vars(weakref.ReferenceType)["__callback__"]
    if type(callback) is not GetSetDescriptorType:
        raise TypeError("native owner admission requires the native weak-reference callback descriptor")
    context.update(owner_array_slots=slots, owner_getweakrefs=weakref.getweakrefs,
                   owner_weakref_class=weakref.ReferenceType, owner_weakref_callback=callback,
                   owner_mmap_class=mmap.mmap, owner_method_class=MethodType)
