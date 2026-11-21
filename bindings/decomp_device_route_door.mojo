# SPDX-License-Identifier: Apache-2.0
"""The decomposition families' resolver glue over the ONE device route.

The rule and its words are core/device_route.mojo; the owner is
`DeviceRouteOwner` (providers/python/device_route_boundary_door.mojo, slot
`device_route_v1`). This file defines no owner, no rule and no refusal text.

A decomposition family reaches its compiled image through a typed connection
record (x_decomp/*_descriptor_boundary_run.mojo: the driver, compiled mode,
vendor and capabilities of one image), named by an index. The family's
resolver stores that index in the route's `flags` word (`decomp_route_keep`):
bits 0..15 the connection, bits 16..31 a second connection plus one (0: none).
`DecompRoute` is the view of those words a door reads.

THE RULE as these doors apply it: the device is checked where it is given (the
constructor, `set_params`) and read ONCE, when the estimator's route is
resolved (its first operation after construction, `set_params` or a changed
numeric mode). No public call reads or compares the device: a hit is one
record read and three pointer compares (mode object, the None default pin,
context).

The image of a resolved connection is pinned for the life of the process by
the family's provider pins (bindings/py_provider_pins.mojo), so a route by
value is safe to call through.
"""
from std.python import Python, PythonObject
from bindings.decomp_scalar_admission_unpack import decomp_scalar_repr
from bindings.py_balanced import py_get, py_item, py_type
from bindings.py_keys import py_key
from core.shared_slot import SharedSlot
from core.device_route import (
    DEVICE_CPU, DEVICE_GPU, DEVICE_INSTALL, device_code, device_refusal, device_installed,
    device_route_plan, device_vendor_code,
)
from providers.python.device_route_boundary_door import (
    DeviceRouteOwner, device_route_held, device_route_keep, device_route_value, device_route_clear,
    device_cpu_connect_late,
)

comptime _DECOMP_FLAG_BITS = 16
comptime _DECOMP_FLAG_MASK = 65535


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
    """The resolver's held keys: one slot fetch and one flag test."""
    var rk = _DECOMP_ROUTE_KEYS.get_or_create_ptr()
    if not rk[].ready:
        _decomp_route_keys_fill(rk)
    return rk


struct DecompRoute(ImplicitlyCopyable, Movable):
    """A door's view of one estimator's route: the resolved device
    (DEVICE_GPU or DEVICE_CPU), the family's typed connection index, the
    selected tier (1 identical, 0 fast) and a second connection of the same
    family where it has one (-1 otherwise)."""
    var device: Int
    var connection: Int
    var mode: Int
    var extra: Int

    def __init__(out self):
        self.device = DEVICE_GPU
        self.connection = -1
        self.mode = -1
        self.extra = -1

    def __init__(out self, device: Int, connection: Int, mode: Int, extra: Int = -1):
        self.device = device
        self.connection = connection
        self.mode = mode
        self.extra = extra


@fieldwise_init
struct DecompRouteHit(Movable):
    """The per-call read: `found` says `route` is the estimator's resolved route."""
    var found: Bool
    var route: DecompRoute


def _decomp_process_vendor(context: PythonObject) raises -> Int:
    var rk = decomp_route_keys()
    return device_vendor_code(String(py_item(context, rk[].k_vendor)))


def _decomp_device_unpack(value: PythonObject, context: PythonObject) raises -> Int:
    """The device code of a stored value, with the shared words
    (core/device_route.mojo). The refused value is shown by the
    decomposition doors' scalar repr: an admitted native scalar only."""
    if value is None:
        return DEVICE_INSTALL
    if py_type(value) is py_type(PythonObject("")):
        var code = device_code(String(value))
        if code >= 0:
            return code
    device_refusal(decomp_scalar_repr(value, context))
    return DEVICE_INSTALL


def decomp_device_admit(value: PythonObject, context: PythonObject) raises:
    """Check a device where it is given (a constructor, `set_params`): an
    unknown value, and an explicit `'gpu'` in a CPU-only process, are refused
    in the shared words before the estimator changes. The default None is
    one identity compare and reads nothing."""
    if value is None:
        return
    var device = _decomp_device_unpack(value, context)
    if device == DEVICE_CPU:
        # The first device="cpu" admission beside a GPU set connects the
        # family's CPU image and label scope (a context that loads them
        # lazily holds `device_cpu_connect`; lane fx-leftovers item 4). Once:
        # the key is cleared in place; other contexts read one None.
        _ = device_cpu_connect_late(context)
        return
    _ = device_installed(device, _decomp_process_vendor(context))


def decomp_route_clear(model: PythonObject) raises:
    """Forget the estimator's routes (a constructor, `set_params`): its next
    operation reads the device and the mode again."""
    device_route_clear(model)


def decomp_route_hit(model: PythonObject, mode: PythonObject, context: PythonObject) raises -> DecompRouteHit:
    """The estimator's route for this operation: one record read and three
    pointer compares. `mode` is the mode object the family selects by (its
    `numeric_mode_`, or its `numeric_mode` with the process default object
    when that is None). The device is not read."""
    var owner = device_route_held(model, mode, PythonObject(None), context)
    if owner is None:
        return DecompRouteHit(False, DecompRoute())
    var route = device_route_value(owner)
    _ = owner  # the owner stays alive through its final native read
    return DecompRouteHit(True, DecompRoute(route.device, route.flags & _DECOMP_FLAG_MASK, route.mode,
        ((route.flags >> _DECOMP_FLAG_BITS) & _DECOMP_FLAG_MASK) - 1))


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


def decomp_route_resolve_device(fields: PythonObject, tier: Int, context: PythonObject) raises -> Int:
    """THE one read of an estimator's `device` (a route miss): the device its
    request runs on in this process for the selected `tier` (1 identical, 0
    fast), by the shared plan. None, and an instance without the field, run
    this installation's device; an explicit `'gpu'` in a CPU-only process, an
    unknown value (an instance that got here without its constructor's
    check) and `'cpu'` with `fast` beside a device set are refused in the
    shared words."""
    var rk = decomp_route_keys()
    var request = _decomp_device_unpack(py_get(fields, rk[].k_device), context)
    return device_route_plan(request, _decomp_process_vendor(context), tier).device


def decomp_route_on_cpu_image(device: Int, context: PythonObject) raises -> Bool:
    """True when a route on `device` reads the family's CPU table (the host
    image loaded beside the device set). Miss path only."""
    return device == DEVICE_CPU and _decomp_process_vendor(context) != 0


def decomp_cpu_table(context: PythonObject, key: PythonObject) raises -> PythonObject:
    """The family's CPU table under `key` (a route on the CPU image), or
    None when it is not connected. The host image loads at the first
    device="cpu" admission (fx-host-import:
    python/mojolearn/_native_decomp_device_bridge.py, no longer at import);
    an estimator restored without construction connects it here, once.
    Miss path only (route resolution, once per estimator)."""
    var table = py_get(context, key)
    if table is None and device_cpu_connect_late(context):
        table = py_get(context, key)
    return table


def decomp_route_expected_vendor(device: Int, context: PythonObject) raises -> String:
    """The vendor word the route's compiled image must report. Miss path only."""
    if device == DEVICE_CPU:
        return String("cpu")
    var rk = decomp_route_keys()
    return String(py_item(context, rk[].k_vendor))


def decomp_route_keep(model: PythonObject, route: DecompRoute, mode: PythonObject, context: PythonObject,
    images: PythonObject) raises:
    """Store a resolved route in the estimator's `DeviceRouteOwner` (miss
    path: one owner allocation). The connection indices go into the route's
    family `flags` word; `images` is the compiled factory (or factories) the
    connection was made from."""
    if route.connection < 0 or route.connection >= _DECOMP_FLAG_MASK or route.extra + 1 >= _DECOMP_FLAG_MASK:
        raise Error("RuntimeError: decomposition connection index does not fit the route word")
    var built = DeviceRouteOwner(route.device)
    built.route.mode = route.mode
    built.route.vendor = 0 if route.device == DEVICE_CPU else _decomp_process_vendor(context)
    built.route.flags = route.connection | ((route.extra + 1) << _DECOMP_FLAG_BITS)
    built.resolved = True
    built.mode = mode
    built.context = context
    built.images = images
    _ = device_route_keep(model, built^)


def decomp_route_table_hit(model: PythonObject, mode: PythonObject, context: PythonObject) raises -> PythonObject:
    """The entries table of an estimator whose route holds one (the
    decomposition kit: bindings/decomp_kit_native_door.mojo), or None on a
    miss: one record read and three pointer compares. The device is not read."""
    var owner = device_route_held(model, mode, PythonObject(None), context)
    if owner is None:
        return owner
    var table = owner.downcast_value_ptr[DeviceRouteOwner]()[].table
    _ = owner  # the owner stays alive through its final native read
    return table


def decomp_route_keep_table(model: PythonObject, device: Int, tier: Int, mode: PythonObject, context: PythonObject,
    table: PythonObject) raises:
    """Store a kit estimator's resolved route: the device, the tier and the
    connected entries table of the image it resolved to (miss path: one
    owner allocation)."""
    var built = DeviceRouteOwner(device)
    built.route.mode = tier
    built.route.vendor = 0 if device == DEVICE_CPU else _decomp_process_vendor(context)
    built.resolved = True
    built.mode = mode
    built.context = context
    built.table = table
    _ = device_route_keep(model, built^)
