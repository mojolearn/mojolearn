# SPDX-License-Identifier: Apache-2.0
"""Exact compiled owner identities and native inspection of their retained pins.

Only audited compiled type registrars call register_native_owner, using the
canonical result of lookup_py_type_object after the module is finalized. Nothing
here is exported to Python. Registration and admission hold the GIL.
"""
from std.python import PythonObject
from std.python.bindings import lookup_py_type_object
from std.python._cpython import PyObjectPtr
from core.shared_slot import SharedSlot


def borrowed_native_owner(address: Int) raises -> PythonObject:
    # Callers retain and admit this non-null PyObject address before inspection.
    # PyObjectPtr is an opaque SDK wrapper, constructed by its supported upcast;
    # it is not itself a Pointer with an unsafe_from_address initializer.
    var raw = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=address)
    return PythonObject(from_borrowed=PyObjectPtr(upcast_from=Optional(raw)))


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


def register_native_owner_type[T: AnyType](module: PythonObject, pins: NativeOwnerPins) raises:
    # Call only AFTER PythonModuleBuilder.finalize(). Under Mojo 1.2 add_type
    # only queues a PythonTypeBuilder; finalize() runs PyType_FromSpec and
    # records the type in MOJO_PYTHON_TYPE_OBJECTS, so a lookup between
    # add_type and finalize raises "No Python type object registered" and the
    # PyInit aborts (import smoke 2026-11-12, _mojolearn_estimators.mojo:1722).
    # On a second initialization the lookup returns the canonical class the
    # first one created, and register_native_owner keeps the first entry.
    register_native_owner(lookup_py_type_object[T](), module, pins)


def registered_native_owner_pins(cls: PythonObject) raises -> Optional[NativeOwnerPins]:
    var registry = NATIVE_OWNER_REGISTRY.get_or_create_ptr()
    var found = registry[].entries.find(Int(cls._obj_ptr))
    if found and found.value().cls is cls:
        return found.value().pins
    return None


def native_owner_no_pins(value: Int, index: Int) abi("C") -> Int:
    # For audited owners consisting entirely of native words/handles.
    return 0 if index == -1 else -1
