# SPDX-License-Identifier: Apache-2.0
"""UNPACK of a decomposition estimator's `device` into its native route owner.

`device=None` (default: this installation's device), `device="gpu"` or
`device="cpu"` is a constructor parameter of the decomposition estimators,
with the rule and the words of the Ridge pilot (core/linear_route.mojo). The
value is stored as given and read ONCE, at the estimator's first operation,
into a `DecompRouteOwner` kept in the weak model-record table. The owner holds
the resolved route (core/device_route_dw_decomp.mojo): the device and the
family's typed connection index of the device image or of the CPU image.
Every later operation reads the owner and calls through its connection.

What a call compares (pointer identities against the owner's pins, no value
is read): the estimator's mode object, its `device` object and the connected
context. The decomposition classes with a plain Python constructor and the
shared `set_params` have no native step at which an owner could be dropped,
so the `device` object is one of the compared identities: a `set_params`,
a `clone`, an unpickled estimator or a directly assigned `device` or
`numeric_mode` resolves again at its next operation. The compare replaces,
per call, the family's former selection: the mode word's strip/lower, the
binding table read and its key admission, the factory's provider lookup and
the vendor String build and compare.

The image of a resolved connection is pinned for the life of the process by
the family's provider pins (bindings/py_provider_pins.mojo), so a route by
value is safe to call through after the owner is gone.

Family-local form of providers/python/linear_route_boundary_door.mojo (the
pilot's owner, which holds the linear family's four entries). SHARED CHANGE
WANTED: one route owner for every family, holding a family-typed payload.
"""
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder, lookup_py_type_object
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_type_registry import py_type_absent, py_type_attach
from bindings.py_balanced import py_call, py_get, py_item, py_type
from bindings.py_keys import py_key
from bindings.py_model_records import records_get, records_put, records_drop, RECORDS_PREPARED
from core.shared_slot import SharedSlot
from core.linear_route import (
    LINEAR_DEVICE_CPU, LINEAR_DEVICE_INSTALL, linear_device_code, linear_device_refusal,
)
from core.device_route_dw_decomp import (
    DecompRoute, decomp_route_device, decomp_route_cpu_image, decomp_route_vendor,
)

comptime _DECOMP_ROUTE_SLOT = "decomp_route_v1"


struct DecompRouteKeys(Defaultable, Movable):
    # Immutable interface names only: no model, state or user data.
    var ready: Bool
    var k_device: PythonObject
    var k_vendor: PythonObject

    def __init__(out self):
        self.ready = False
        self.k_device = PythonObject(None)
        self.k_vendor = PythonObject(None)


comptime _DECOMP_ROUTE_KEYS = SharedSlot[DecompRouteKeys, "MojolearnDecompRouteKeysV1"]


def _decomp_route_keys_fill(rk: UnsafePointer[DecompRouteKeys, MutUntrackedOrigin]) raises:
    # Once per process. `ready` is set last, so a fill that raised is run
    # again whole by the next call.
    rk[].k_device = py_key("device")
    rk[].k_vendor = py_key("vendor")
    rk[].ready = True


@always_inline
def decomp_route_keys() raises -> UnsafePointer[DecompRouteKeys, MutUntrackedOrigin]:
    """The route's held keys: one slot fetch and one flag test."""
    var rk = _DECOMP_ROUTE_KEYS.get_or_create_ptr()
    if not rk[].ready:
        _decomp_route_keys_fill(rk)
    return rk


struct DecompRouteOwner(Movable, Writable):
    var route: DecompRoute
    var resolved: Bool
    # The objects the route was resolved for. Held, so their addresses stay
    # theirs while this owner lives; a call compares identities only.
    var mode: PythonObject
    var device: PythonObject
    var context: PythonObject
    # The compiled factory (or entry table) the connection was made from.
    var images: PythonObject

    def __init__(out self, route: DecompRoute, mode: PythonObject, device: PythonObject,
        context: PythonObject, images: PythonObject):
        self.route = route
        self.resolved = True
        self.mode = mode
        self.device = device
        self.context = context
        self.images = images

    def write_to(self, mut writer: Some[Writer]):
        writer.write("DecompRouteOwner(device=", self.route.device, ", connection=", self.route.connection, ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


def register_decomp_route_owner(mut module: PythonModuleBuilder) raises:
    """Register the owner's Python type in the ONE binding that allocates it
    (bindings/_mojolearn_runtime_decomp_host.mojo). A second initialization of
    that image attaches the existing type object."""
    var existing: PythonObject
    try:
        existing = lookup_py_type_object[DecompRouteOwner]()
    except error:
        if not py_type_absent(error):
            raise error
        _ = module.add_type[DecompRouteOwner]("_DecompRouteOwner")
        return
    py_type_attach(module.module, "_DecompRouteOwner", existing)


def decomp_device_unpack(value: PythonObject) raises -> Int:
    """The device code of a constructor value: None (this installation's
    device), or exact `'gpu'` or `'cpu'`. Anything else is refused in the
    pilot's words (core/linear_route.mojo), before any state changes."""
    if value is None:
        return LINEAR_DEVICE_INSTALL
    var b = native_loaded_module("builtins")
    if py_type(value) is b.str:
        var code = linear_device_code(String(value))
        if code >= 0:
            return code
    linear_device_refusal(String(py_call(b.repr, value)))
    return LINEAR_DEVICE_INSTALL


def decomp_device_admit(value: PythonObject, context: PythonObject) raises:
    """Check a device where it is given to a native constructor or a native
    `set_params`: an unknown value, and an explicit `'gpu'` in a CPU-only
    process, are refused in words before the estimator changes. The default
    None is one identity compare and reads nothing."""
    if value is None:
        return
    var device = decomp_device_unpack(value)
    if device == LINEAR_DEVICE_CPU:
        return
    var rk = decomp_route_keys()
    _ = decomp_route_device(device, String(py_item(context, rk[].k_vendor)))


@fieldwise_init
struct DecompRouteHit(Movable):
    """The per-call read. `found`: `route` is the estimator's resolved route.
    Otherwise `device` is the estimator's `device` object as stored (None
    when the field is absent), for the one resolution that follows."""
    var found: Bool
    var route: DecompRoute
    var device: PythonObject


def decomp_route_hit(model: PythonObject, fields: PythonObject, mode: PythonObject,
    context: PythonObject) raises -> DecompRouteHit:
    """The estimator's route for this operation: one held-key read of
    `device`, one record read and three pointer compares. `mode` is the mode
    object the family selects by (its `numeric_mode_`, or its `numeric_mode`
    with the process default when that is None)."""
    var rk = decomp_route_keys()
    var device = py_get(fields, rk[].k_device)
    var owner = records_get(RECORDS_PREPARED, model, _DECOMP_ROUTE_SLOT)
    if owner is not None:
        var native = owner.downcast_value_ptr[DecompRouteOwner]()
        if (native[].resolved and native[].mode is mode and native[].device is device
                and native[].context is context):
            var route = native[].route
            _ = owner  # the downcast owner stays alive through its final native read
            return DecompRouteHit(True, route, device)
    return DecompRouteHit(False, DecompRoute(), device)


def decomp_route_mode(fields: PythonObject, context: PythonObject) raises -> PythonObject:
    """The mode object a metadata read (`vendor_used`) selects by: the fitted
    `numeric_mode_` when the estimator has one, else its `numeric_mode`, else
    the process default. Not on a fit or query path."""
    var mode = py_get(fields, "numeric_mode_")
    if mode is None:
        mode = py_get(fields, "numeric_mode")
    if mode is None:
        mode = context["backend_fields"]["_DEFAULT_MODE"]
    return mode


def decomp_route_resolve_device(device: PythonObject, context: PythonObject) raises -> Int:
    """The device a stored `device` object runs on in this process (miss path
    only): None is this installation's device, an explicit `'gpu'` in a
    CPU-only process and an unknown value are refused in words."""
    var rk = decomp_route_keys()
    return decomp_route_device(decomp_device_unpack(device), String(py_item(context, rk[].k_vendor)))


def decomp_route_on_cpu_image(device: Int, context: PythonObject) raises -> Bool:
    """True when a route on `device` reads the family's CPU table (the host
    image loaded beside the device set), miss path only."""
    var rk = decomp_route_keys()
    return decomp_route_cpu_image(device, String(py_item(context, rk[].k_vendor)))


def decomp_route_expected_vendor(device: Int, context: PythonObject) raises -> String:
    """The vendor word the route's compiled image must report (miss path)."""
    var rk = decomp_route_keys()
    return decomp_route_vendor(device, String(py_item(context, rk[].k_vendor)))


def decomp_route_keep(model: PythonObject, route: DecompRoute, mode: PythonObject, device: PythonObject,
    context: PythonObject, images: PythonObject) raises:
    """Store a resolved route for the estimator (miss path: one owner
    allocation). An estimator that takes no weak reference keeps no owner and
    resolves at every operation (records_put stores nothing)."""
    var fresh = PythonObject(alloc=DecompRouteOwner(route, mode, device, context, images))
    _ = records_put(RECORDS_PREPARED, model, _DECOMP_ROUTE_SLOT, fresh)


def decomp_route_clear(model: PythonObject) raises:
    """Forget the estimator's route (a native `__init__` or `set_params`)."""
    records_drop(RECORDS_PREPARED, model, _DECOMP_ROUTE_SLOT)
