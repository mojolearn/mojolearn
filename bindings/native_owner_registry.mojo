# SPDX-License-Identifier: Apache-2.0
"""Exact compiled owner identities and native inspection of their retained pins.

Only audited compiled type registrars call register_native_owner, using the
canonical result of lookup_py_type_object after add_type/attachment. Nothing
here is exported to Python. Registration and admission hold the GIL.
"""
from std.python import PythonObject
from core.shared_slot import SharedSlot


# Request -1 returns the number of retained Python owners. Requests 0..count-1
# return borrowed PyObject addresses, kept alive by the inspected owner. -1 as
# a result refuses a failed inspection; Py_None has an ordinary nonzero address.
# This native C ABI cannot invoke Python dispatch or propagate Mojo exceptions.
comptime NativeOwnerPins = def(Int, Int) thin abi("C") -> Int


@fieldwise_init
struct NativeOwnerRegistration(Copyable, Movable):
    var cls: PythonObject
    var module: PythonObject
    var pins: NativeOwnerPins


struct NativeOwnerRegistry(Defaultable, Movable):
    var entries: Dict[Int, NativeOwnerRegistration]

    def __init__(out self):
        self.entries = Dict[Int, NativeOwnerRegistration]()


comptime NATIVE_OWNER_REGISTRY = SharedSlot[NativeOwnerRegistry, "MojolearnNativeOwnerRegistryV1"]


def register_native_owner(cls: PythonObject, module: PythonObject, pins: NativeOwnerPins) raises:
    # The retained class prevents recycled addresses. The module pins the
    # inspector's compiled image, including a second import namespace alias.
    # First registration owns the canonical inspector for this exact class.
    var registry = NATIVE_OWNER_REGISTRY.get_or_create_ptr()
    var address = Int(cls._obj_ptr)
    if address not in registry[].entries:
        registry[].entries[address] = NativeOwnerRegistration(cls, module, pins)


def registered_native_owner_pins(cls: PythonObject) raises -> Optional[NativeOwnerPins]:
    var registry = NATIVE_OWNER_REGISTRY.get_or_create_ptr()
    var found = registry[].entries.find(Int(cls._obj_ptr))
    if found and found.value().cls is cls:
        return found.value().pins
    return None


def native_owner_no_pins(value: Int, index: Int) abi("C") -> Int:
    # For audited owners consisting entirely of native words/handles.
    return 0 if index == -1 else -1
