# SPDX-License-Identifier: Apache-2.0
"""UNPACK of an estimator's device and numeric mode into its native route owner,
for the families that select compiled bindings by name (mixture, cluster,
neighbors, density; lane dw-neighbors-cluster, 2026-11-20).

`device=None` (default: this installation's device), `device="gpu"` or
`device="cpu"` is a constructor parameter. Its value is stored as given,
checked when it is given (`family_route_device_admit`: constructor,
`set_params`) and read ONCE, at the estimator's first operation, into a
`FamilyRouteOwner` kept in the weak model-record table (read again only when
the stored `device` object is replaced). The owner holds, per
binding slot (core/device_route_dw_neighbors_cluster.mojo), the image that
answers: the device image of the selected tier, or the CPU identical image
connected at import (python/mojolearn/_native_device_bindings.py). Every
later operation reads the owner: no per-call device question, registry walk,
mode String or compiled-mode read-back.

What a call still compares, as the pilot's owner does
(providers/python/linear_route_boundary_door.mojo): the estimator's
`numeric_mode` object and, when that is None, the process default mode
object, and the connection context (three pointer compares), plus the stored
`device` object (one held-key read of the instance dictionary and a fourth
pointer compare). The fourth compare is here, and not in the pilot, because
the constructor and `set_params` of these families run in the shared
full-reset doors (bindings/full_reset_constructor_native_door.mojo,
full_reset_params_native_door.mojo), which this lane does not edit and which
therefore cannot drop the owner: a replaced `device` (or mode) object is how
the owner learns of a new `__init__`, a `set_params` or a direct assignment.
`family_route_clear` is there for a door that owns its constructor.

The owner replaces, for a family that calls `family_route_binding`, the
per-call `estimator_binding` (providers/python/estimator_binding_pyapi_door.mojo):
the mode String and its normalisation, the offerings and connections reads,
the process-wide `_CPU_ONLY` question, the live registry or `sys.modules`
read and the compiled-mode getter. That resolution still runs, once, on a
miss; it is called here, not copied.
"""
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder, lookup_py_type_object
from bindings.py_type_registry import py_type_absent, py_type_attach
from bindings.native_owner_registry_unpack import register_native_owner_type
from bindings.native_owner_registry_pack import borrowed_native_owner
from bindings.native_loaded_module_door import native_loaded_module
from bindings.mode_metadata_native_door import _fields
from bindings.py_balanced import py_call, py_get, py_item, py_type
from bindings.py_keys import py_key
from bindings.py_model_records import records_get, records_put, records_drop, RECORDS_PREPARED
from core.shared_slot import SharedSlot
from core.linear_route import (
    LINEAR_DEVICE_INSTALL, linear_device_code, linear_device_refusal, linear_device_installed,
)
from core.device_route_dw_neighbors_cluster import (
    FamilyDeviceRoute, FAMILY_ROUTE_SLOT_COUNT, family_route_binding_name, family_route_plan,
    family_route_unresolved, family_route_cpu_unconnected, family_route_vendor_is_cpu,
)
from providers.python.estimator_binding_pyapi_door import (
    _selected_mode, estimator_binding_mode, estimator_checked_mode,
)

comptime _FAMILY_ROUTE_SLOT = "family_device_route_v1"


struct FamilyRouteKeys(Defaultable, Movable):
    # Immutable interface names only: no model, state or user data
    # (the form of bindings/cluster_keys_unpack.mojo).
    var ready: Bool
    var k_numeric_mode: PythonObject
    var k_device: PythonObject
    var k_backend_fields: PythonObject
    var k__DEFAULT_MODE: PythonObject
    var k__CPU_ONLY: PythonObject
    var k_vendor: PythonObject
    var k_family_cpu_binding_connections: PythonObject

    def __init__(out self):
        self.ready = False
        self.k_numeric_mode = PythonObject(None)
        self.k_device = PythonObject(None)
        self.k_backend_fields = PythonObject(None)
        self.k__DEFAULT_MODE = PythonObject(None)
        self.k__CPU_ONLY = PythonObject(None)
        self.k_vendor = PythonObject(None)
        self.k_family_cpu_binding_connections = PythonObject(None)


comptime _FAMILY_ROUTE_KEYS = SharedSlot[FamilyRouteKeys, "MojolearnFamilyRouteKeysV1"]


def _family_route_keys_fill(nk: UnsafePointer[FamilyRouteKeys, MutUntrackedOrigin]) raises:
    # Once per process. `ready` is set last, so a fill that raised is run
    # again whole by the next call.
    nk[].k_numeric_mode = py_key("numeric_mode")
    nk[].k_device = py_key("device")
    nk[].k_backend_fields = py_key("backend_fields")
    nk[].k__DEFAULT_MODE = py_key("_DEFAULT_MODE")
    nk[].k__CPU_ONLY = py_key("_CPU_ONLY")
    nk[].k_vendor = py_key("vendor")
    nk[].k_family_cpu_binding_connections = py_key("family_cpu_binding_connections")
    nk[].ready = True


@always_inline
def family_route_keys() raises -> UnsafePointer[FamilyRouteKeys, MutUntrackedOrigin]:
    """The route door's keys: one slot fetch and one flag test."""
    var nk = _FAMILY_ROUTE_KEYS.get_or_create_ptr()
    if not nk[].ready:
        _family_route_keys_fill(nk)
    return nk


struct FamilyRouteOwner(Movable, Writable):
    var route: FamilyDeviceRoute
    # The objects the route was resolved for. Held, so their addresses stay
    # theirs while this owner lives; a call compares identities only.
    var mode: PythonObject
    var default_mode: PythonObject
    var context: PythonObject
    var device: PythonObject
    # Per binding slot: the image that answers (None until the estimator
    # first asks for that slot), and whether its compiled mode was read back.
    # The images stay loaded while a call runs with the GIL released.
    var modules: List[PythonObject]
    var checked: List[Bool]

    def __init__(out self, var route: FamilyDeviceRoute, mode: PythonObject, default_mode: PythonObject,
                 context: PythonObject, device: PythonObject):
        self.route = route^
        self.mode = mode
        self.default_mode = default_mode
        self.context = context
        self.device = device
        self.modules = List[PythonObject]()
        self.checked = List[Bool]()
        for slot in range(FAMILY_ROUTE_SLOT_COUNT):  # small-loop(binding slots): a fixed handful per owner, never data
            _ = slot
            self.modules.append(PythonObject(None))
            self.checked.append(False)

    def write_to(self, mut writer: Some[Writer]):
        writer.write("FamilyRouteOwner(device=", self.route.device, ", host_table=", self.route.host_table, ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


def family_route_owner_pins(value: Int, index: Int) abi("C") -> Int:
    if not value or index < -1 or index >= 4 + FAMILY_ROUTE_SLOT_COUNT:
        return -1
    try:
        var owner = borrowed_native_owner(value)
        var native = owner.downcast_value_ptr[FamilyRouteOwner]()
        var result = 4 + FAMILY_ROUTE_SLOT_COUNT
        if index == 0:
            result = Int(native[].mode._obj_ptr)
        elif index == 1:
            result = Int(native[].default_mode._obj_ptr)
        elif index == 2:
            result = Int(native[].context._obj_ptr)
        elif index == 3:
            result = Int(native[].device._obj_ptr)
        elif index >= 4:
            result = Int(native[].modules[index - 4]._obj_ptr)
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


def _family_route_cpu_only(context: PythonObject) raises -> Bool:
    """The one process question, asked at resolution and at a device check
    only: does this process hold a GPU set?"""
    var nk = family_route_keys()
    return py_item(py_item(context, nk[].k_backend_fields), nk[].k__CPU_ONLY) is not None


def family_route_device_unpack(value: PythonObject, b: PythonObject) raises -> Int:
    """The device code of a constructor value: None (this installation's
    device), or exact `'gpu'` or `'cpu'`; anything else is refused in the
    pilot's words (core/linear_route.mojo `linear_device_refusal`). The same
    reader as the pilot's `linear_device_unpack`, restated here so this
    family's binding does not compile the linear route door's closure.
    SHARED CHANGE WANTED: one reader in a leaf boundary file."""
    if value is None:
        return LINEAR_DEVICE_INSTALL
    if py_type(value) is b.str:
        var code = linear_device_code(String(value))
        if code >= 0:
            return code
    linear_device_refusal(String(py_call(b.repr, value)))
    return LINEAR_DEVICE_INSTALL


def family_route_device_admit(value: PythonObject, context: PythonObject) raises:
    """Check a device where it is given (constructor, `set_params`): an
    unknown value, and an explicit `'gpu'` in a CPU-only process, are refused
    in the pilot's words before the estimator changes. The default None is
    one identity compare and reads nothing."""
    if value is None:
        return
    var device = family_route_device_unpack(value, native_loaded_module("builtins"))
    _ = linear_device_installed(device, 0 if _family_route_cpu_only(context) else 1)


def family_route_clear(model: PythonObject) raises:
    """Forget the estimator's route (a new `__init__`, `set_params`): its next
    operation reads the device and the mode again."""
    records_drop(RECORDS_PREPARED, model, _FAMILY_ROUTE_SLOT)


def _family_route_build(device: PythonObject, mode: PythonObject, default_mode: PythonObject,
                        context: PythonObject) raises -> FamilyRouteOwner:
    var b = native_loaded_module("builtins")
    # The one decode of `device`. None, and an instance without the field
    # (unpickled from an older version, or loaded from a file), run this
    # installation's device.
    var requested = family_route_device_unpack(device, b)
    var word = _selected_mode(b, mode, context)
    var plan = family_route_plan(requested, _family_route_cpu_only(context), 1 if word == "identical" else 0)
    return FamilyRouteOwner(plan^, mode, default_mode, context, device)


def _family_route_slot(owner: PythonObject, slot: Int, check_mode: Bool) raises -> PythonObject:
    """Resolve one binding slot of a held route (the first time the
    estimator asks for it, or the first time it asks with the compiled-mode
    read-back) and keep it on the owner."""
    var native = owner.downcast_value_ptr[FamilyRouteOwner]()
    var name = family_route_binding_name(slot)
    var context = native[].context
    var module = PythonObject(None)
    if native[].route.host_table:
        # device='cpu' on a GPU installation: the CPU identical image of the
        # same name, loaded and read back at import
        # (python/mojolearn/_native_device_bindings.py connect_cpu_device).
        var nk = family_route_keys()
        var b = native_loaded_module("builtins")
        var table = py_get(context, nk[].k_family_cpu_binding_connections)
        if table is None:
            family_route_cpu_unconnected(name)
        var record = py_get(table, PythonObject(name))
        if record is None:
            family_route_cpu_unconnected(name)
        var image = py_item(record, PythonObject(0))
        if py_type(image) is b.str:
            # The import-time refusal of a CPU image that is not built.
            raise Error(String(image))
        if py_type(image) is not py_type(b):
            raise Error("TypeError: native host binding must be an exact extension module")
        module = image
        if check_mode:
            module = estimator_checked_mode(image, String(py_item(record, PythonObject(2))), name, String("identical"))
    else:
        # The process's own table: the device image of the selected tier on a
        # GPU installation, the CPU image in a CPU-only process. The shared
        # resolution, run once per estimator and slot.
        var selected = native[].mode
        module = estimator_binding_mode(selected, name, context, check_mode)
    native = owner.downcast_value_ptr[FamilyRouteOwner]()
    native[].modules[slot] = module
    native[].checked[slot] = check_mode
    return module


def family_route_owner(model: PythonObject, context: PythonObject) raises -> PythonObject:
    """The estimator's route owner for this operation. A hit is one record
    read and four pointer compares; a miss (first operation, a changed
    numeric mode or device object, an instance built without its
    constructor) decodes the device, resolves and stores it."""
    var nk = family_route_keys()
    var fields = _fields(model)
    var mode = py_get(fields, nk[].k_numeric_mode)
    var default_mode = PythonObject(None)
    if mode is None:
        default_mode = py_item(py_item(context, nk[].k_backend_fields), nk[].k__DEFAULT_MODE)
    var device = py_get(fields, nk[].k_device)
    var owner = records_get(RECORDS_PREPARED, model, _FAMILY_ROUTE_SLOT)
    if owner is not None:
        var native = owner.downcast_value_ptr[FamilyRouteOwner]()
        if (native[].mode is mode and native[].default_mode is default_mode and native[].context is context
                and native[].device is device):
            return owner
    var fresh = PythonObject(alloc=_family_route_build(device, mode, default_mode, context))
    # An estimator that takes no weak reference keeps no owner: it resolves
    # at every operation (records_put answers False and stores nothing).
    _ = records_put(RECORDS_PREPARED, model, _FAMILY_ROUTE_SLOT, fresh)
    return fresh


def family_route_owner_binding(owner: PythonObject, slot: Int, check_mode: Bool = True) raises -> PythonObject:
    """The image of `slot` on a route owner this operation already read: one
    list read on a hit."""
    var native = owner.downcast_value_ptr[FamilyRouteOwner]()
    var held = native[].modules[slot]
    if held is not None and (native[].checked[slot] or not check_mode):
        return held
    return _family_route_slot(owner, slot, check_mode)


def family_route_binding(model: PythonObject, slot: Int, context: PythonObject,
                         check_mode: Bool = True) raises -> PythonObject:
    """The image that answers `slot` for this estimator: the device-aware
    form of `estimator_binding(model, <name>, context, check_mode)`."""
    return family_route_owner_binding(family_route_owner(model, context), slot, check_mode)


def family_route_vendor_used(model: PythonObject, context: PythonObject) raises -> PythonObject:
    """`vendor_used()` of an estimator that takes `device=`: `cpu` when its
    route runs the CPU images, else the vendor word of this process's GPU
    set. Resolves the route on first use, with its refusals."""
    var owner = family_route_owner(model, context)
    if family_route_vendor_is_cpu(owner.downcast_value_ptr[FamilyRouteOwner]()[].route):
        return PythonObject("cpu")
    return py_item(context, family_route_keys()[].k_vendor)
