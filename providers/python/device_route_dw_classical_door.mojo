# SPDX-License-Identifier: Apache-2.0
"""UNPACK of an estimator's device and numeric mode into its native route owner,
for the families of lane dw-classical (QNRegressor first; the pattern is
providers/python/linear_route_boundary_door.mojo, the pilot's, generalised to
"one operations entry per image").

`device=None` (default: this installation's device), `device="gpu"` or
`device="cpu"` is a constructor parameter. Its value is stored as given,
checked where it is given (the constructor; `set_params` rebuilds through the
constructor) and read ONCE, at the estimator's first operation, into a
`FamilyRouteOwner` kept in the weak model-record table. The owner holds the
resolved entries (core/device_route_dw_classical.mojo) and the objects that
pin their images. Every later operation reads the owner and calls through it:
no per-call device read, mode String, table slot lookup or vendor question.

What a call still compares, as the pilot does: the estimator's
`numeric_mode` object and, when that is None, the process default mode object
(pointer compares against the owner's pins), and the connection context. A
changed mode resolves again; a new `__init__` drops the owner.

The device reader and every device word are the pilot's
(`linear_device_unpack`, `linear_device_installed`), imported.
"""
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder, lookup_py_type_object
from bindings.py_type_registry import py_type_absent, py_type_attach
from bindings.native_owner_registry_unpack import register_native_owner_type
from bindings.native_owner_registry_pack import borrowed_native_owner
from bindings.py_balanced import py_get
from bindings.py_model_records import records_get, records_put, records_drop, RECORDS_PREPARED
from core.device_route_dw_classical import FamilyRoute
from core.linear_route import LINEAR_DEVICE_CPU, LINEAR_DEVICE_INSTALL, linear_device_installed
from providers.python.linear_route_boundary_door import linear_device_unpack

# One slot for every family of this door: an estimator belongs to one family.
comptime _FAMILY_ROUTE_SLOT = "family_route_v1"


struct FamilyRouteOwner(Movable, Writable):
    var route: FamilyRoute
    var resolved: Bool
    # Why the family's second entry is not held (said by the operation that
    # needs it, in the words its table read raised).
    var second_refusal: String
    # The objects the route was resolved for. Held, so their addresses stay
    # theirs while this owner lives; a call compares identities only.
    var mode: PythonObject
    var default_mode: PythonObject
    var context: PythonObject
    # The entry dictionaries (and through them the extension modules and
    # connection owners) of every resolved table: the images stay loaded
    # while a call runs with the GIL released.
    var images: PythonObject

    def __init__(out self, device: Int):
        self.route = FamilyRoute(device)
        self.resolved = False
        self.second_refusal = String("")
        self.mode = PythonObject(None)
        self.default_mode = PythonObject(None)
        self.context = PythonObject(None)
        self.images = PythonObject(None)

    def write_to(self, mut writer: Some[Writer]):
        writer.write("FamilyRouteOwner(device=", self.route.device, ", vendor=", self.route.vendor, ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


def family_route_owner_pins(value: Int, index: Int) abi("C") -> Int:
    if not value or index < -1 or index > 3:
        return -1
    try:
        var owner = borrowed_native_owner(value)
        var native = owner.downcast_value_ptr[FamilyRouteOwner]()
        var result = 4
        if index == 0:
            result = Int(native[].mode._obj_ptr)
        elif index == 1:
            result = Int(native[].default_mode._obj_ptr)
        elif index == 2:
            result = Int(native[].context._obj_ptr)
        elif index == 3:
            result = Int(native[].images._obj_ptr)
        _ = owner
        return result
    except:
        return -1


def register_family_route_owner(mut module: PythonModuleBuilder) raises:
    var existing = PythonObject(None)
    try:
        existing = lookup_py_type_object[FamilyRouteOwner]()
    except error:
        if not py_type_absent(error):
            raise error
        _ = module.add_type[FamilyRouteOwner]("_FamilyRouteOwner")
        return
    py_type_attach(module.module, "_FamilyRouteOwner", existing)


def register_family_route_owner_identity(published: PythonObject) raises:
    # After PythonModuleBuilder.finalize() only: Mojo 1.2 creates the type
    # object in finalize() (bindings/native_owner_registry.mojo).
    register_native_owner_type[FamilyRouteOwner](published, family_route_owner_pins)


def family_device_request(fields: PythonObject, b: PythonObject) raises -> Int:
    """The one read of an estimator's `device` (a route miss). None, and an
    instance without the field (built without its constructor, or saved
    before the parameter existed), run this installation's device. `b` is
    the builtins module the calling door already admitted."""
    return linear_device_unpack(py_get(fields, "device"), b)


def family_device_admit(value: PythonObject, process_vendor: Int, b: PythonObject) raises:
    """Check a device where it is given (the constructor): an unknown value,
    and an explicit `'gpu'` in a CPU-only process (`process_vendor` 0), are
    refused in the pilot's words before the estimator changes. The default
    None is one identity compare inside `linear_device_unpack` and reads
    nothing else."""
    var device = linear_device_unpack(value, b)
    if device == LINEAR_DEVICE_INSTALL or device == LINEAR_DEVICE_CPU:
        return
    _ = linear_device_installed(device, process_vendor)


def family_route_clear(model: PythonObject) raises:
    """Forget the estimator's route (a new `__init__`): its next operation
    reads the device and the mode again."""
    records_drop(RECORDS_PREPARED, model, _FAMILY_ROUTE_SLOT)


def family_route_held(model: PythonObject, mode: PythonObject, default_mode: PythonObject,
    context: PythonObject) raises -> PythonObject:
    """The estimator's resolved route owner for this operation, or None on a
    miss (first operation, a changed numeric mode, another connection
    context). A hit is one record read and three pointer compares."""
    var owner = records_get(RECORDS_PREPARED, model, _FAMILY_ROUTE_SLOT)
    if owner is not None:
        var native = owner.downcast_value_ptr[FamilyRouteOwner]()
        if (native[].resolved and native[].mode is mode and native[].default_mode is default_mode
                and native[].context is context):
            return owner
    return PythonObject(None)


def family_route_keep(model: PythonObject, var built: FamilyRouteOwner) raises -> PythonObject:
    """Store a freshly resolved owner for the estimator and return it. An
    estimator that takes no weak reference keeps no owner: it resolves at
    every operation (records_put answers False and stores nothing)."""
    var fresh = PythonObject(alloc=built^)
    _ = records_put(RECORDS_PREPARED, model, _FAMILY_ROUTE_SLOT, fresh)
    return fresh


def family_route_value(owner: PythonObject) raises -> FamilyRoute:
    """The route of a held or freshly kept owner, by value. The caller keeps
    `owner` named until its operation returns: it pins the images."""
    return owner.downcast_value_ptr[FamilyRouteOwner]()[].route


def family_route_second_refuse(owner: PythonObject, fallback: String) raises:
    """Raise why this estimator's route holds no second entry (failure path
    of the one operation that needs it)."""
    var text = owner.downcast_value_ptr[FamilyRouteOwner]()[].second_refusal.copy()
    if text.byte_length() == 0:
        text = fallback.copy()
    raise Error(text)
