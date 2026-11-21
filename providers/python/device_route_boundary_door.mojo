# SPDX-License-Identifier: Apache-2.0
"""UNPACK of an estimator's device and numeric mode into its native route owner:
the ONE owner every family shares.

`device=None` (default: this installation's device), `device="gpu"` or
`device="cpu"` is a constructor parameter. Its value is stored as given,
checked where it is given (the constructor; `set_params`) and read ONCE, at
the estimator's first operation, into a `DeviceRouteOwner` kept in the weak
model-record table. The owner holds the resolved entries
(core/device_route.mojo) and the objects that pin their images. Every later
operation reads the owner and calls through it: no per-call device read, mode
String, table slot lookup or vendor question.

What a call still compares: the estimator's `numeric_mode` object and, when
that is None, the process default mode object (pointer compares against the
owner's pins), and the connection context. A changed mode resolves again; a
new `__init__` and `set_params` drop the owner.

One owner type serves both forms of a family door:
  * an entry route (slot `device_route_v1`): the family's resolver fills
    `route` with its resolved thin C entries (Ridge, LinearRegression,
    LogisticRegression, QNRegressor);
  * a binding route (slot `device_bindings_v1`): the shared estimator door
    (providers/python/estimator_binding_pyapi_door.mojo) records the compiled
    module each binding name resolved to for the estimator's device.
An estimator may hold one of each; they never share a record.

This owner replaces `LinearRouteOwner`, `LogisticRouteOwner` and
`FamilyRouteOwner` (three copies of the same pins, hit test and
registration). The family resolvers stay with their families.
"""
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder, lookup_py_type_object
from bindings.py_type_registry import py_type_absent, py_type_attach
from bindings.native_owner_registry_unpack import register_native_owner_type
from bindings.native_owner_registry_pack import borrowed_native_owner
from bindings.py_balanced import py_call, py_type, py_get, py_setitem
from bindings.py_model_records import records_get, records_put, records_drop, records_open, RECORDS_PREPARED
from core.device_route import (
    DeviceRoute, DEVICE_CPU, DEVICE_GPU, DEVICE_INSTALL, device_code, device_refusal, device_installed,
    device_cpu_absent,
)

# An estimator's entry route (one family owns an estimator).
comptime DEVICE_ROUTE_SLOT = "device_route_v1"
# An estimator's binding route (the shared estimator door).
comptime DEVICE_BINDINGS_SLOT = "device_bindings_v1"
comptime _DEVICE_ROUTE_FIXED_PINS = 7


struct DeviceRouteOwner(Movable, Writable):
    var route: DeviceRoute
    var resolved: Bool
    # Why an optional entry of the family is not held (said by the operation
    # that needs it, in the words its table read raised).
    var refusal: String
    # The objects the route was resolved for. Held, so their addresses stay
    # theirs while this owner lives; a call compares identities only.
    var mode: PythonObject
    var default_mode: PythonObject
    var context: PythonObject
    var cls: PythonObject
    # The entry dictionaries (and through them the extension modules and
    # connection owners) of every resolved table: the images stay loaded
    # while a call runs with the GIL released.
    var images: PythonObject
    # Family objects of the route's device (LogisticRegression: the context
    # its class runs in, and the class-weight table of the same device).
    var scope: PythonObject
    var table: PythonObject
    # The binding route: per resolved (binding name, checked, mode object,
    # default mode object) the compiled module of the route's device.
    var names: List[String]
    var checked: List[Bool]
    var modes: List[PythonObject]
    var defaults: List[PythonObject]
    var modules: List[PythonObject]

    def __init__(out self, device: Int):
        self.route = DeviceRoute(device)
        self.resolved = False
        self.refusal = String("")
        self.mode = PythonObject(None)
        self.default_mode = PythonObject(None)
        self.context = PythonObject(None)
        self.cls = PythonObject(None)
        self.images = PythonObject(None)
        self.scope = PythonObject(None)
        self.table = PythonObject(None)
        self.names = List[String]()
        self.checked = List[Bool]()
        self.modes = List[PythonObject]()
        self.defaults = List[PythonObject]()
        self.modules = List[PythonObject]()

    def write_to(self, mut writer: Some[Writer]):
        writer.write("DeviceRouteOwner(device=", self.route.device, ", vendor=", self.route.vendor, ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


def device_route_owner_pins(value: Int, index: Int) abi("C") -> Int:
    if not value or index < -1:
        return -1
    try:
        var owner = borrowed_native_owner(value)
        var native = owner.downcast_value_ptr[DeviceRouteOwner]()
        var held = len(native[].modules)
        var result = -1
        if index == -1:
            result = _DEVICE_ROUTE_FIXED_PINS + 3 * held
        elif index == 0:
            result = Int(native[].mode._obj_ptr)
        elif index == 1:
            result = Int(native[].default_mode._obj_ptr)
        elif index == 2:
            result = Int(native[].context._obj_ptr)
        elif index == 3:
            result = Int(native[].cls._obj_ptr)
        elif index == 4:
            result = Int(native[].images._obj_ptr)
        elif index == 5:
            result = Int(native[].scope._obj_ptr)
        elif index == 6:
            result = Int(native[].table._obj_ptr)
        elif index < _DEVICE_ROUTE_FIXED_PINS + held:
            result = Int(native[].modules[index - _DEVICE_ROUTE_FIXED_PINS]._obj_ptr)
        elif index < _DEVICE_ROUTE_FIXED_PINS + 2 * held:
            result = Int(native[].modes[index - _DEVICE_ROUTE_FIXED_PINS - held]._obj_ptr)
        elif index < _DEVICE_ROUTE_FIXED_PINS + 3 * held:
            result = Int(native[].defaults[index - _DEVICE_ROUTE_FIXED_PINS - 2 * held]._obj_ptr)
        _ = owner
        return result
    except:
        return -1


def register_device_route_owner(mut module: PythonModuleBuilder) raises:
    var existing = PythonObject(None)
    try:
        existing = lookup_py_type_object[DeviceRouteOwner]()
    except error:
        if not py_type_absent(error):
            raise error
        _ = module.add_type[DeviceRouteOwner]("_DeviceRouteOwner")
        return
    py_type_attach(module.module, "_DeviceRouteOwner", existing)


def register_device_route_owner_identity(published: PythonObject) raises:
    # After PythonModuleBuilder.finalize() only: Mojo 1.2 creates the type
    # object in finalize() (bindings/native_owner_registry.mojo).
    register_native_owner_type[DeviceRouteOwner](published, device_route_owner_pins)


def device_unpack(value: PythonObject, b: PythonObject) raises -> Int:
    """The device code of a constructor value: None (this installation's
    device), or exact `'gpu'` or `'cpu'`. Anything else is refused in words,
    before any state changes. `b` is the builtins module the calling door
    already admitted."""
    if value is None:
        return DEVICE_INSTALL
    if py_type(value) is b.str:
        var code = device_code(String(value))
        if code >= 0:
            return code
    device_refusal(String(py_call(b.repr, value)))
    return DEVICE_INSTALL


def device_request(fields: PythonObject, b: PythonObject) raises -> Int:
    """The one read of an estimator's `device` (a route miss). None, and an
    instance without the field (built without its constructor, or saved
    before the parameter existed), run this installation's device."""
    return device_unpack(py_get(fields, "device"), b)


def device_admit(value: PythonObject, process_vendor: Int, b: PythonObject) raises:
    """Check a device where it is given (the constructor, `set_params`): an
    unknown value, and an explicit `'gpu'` in a CPU-only process
    (`process_vendor` 0), are refused in words before the estimator changes.
    The default None is one identity compare inside `device_unpack` and
    reads nothing else."""
    var device = device_unpack(value, b)
    if device == DEVICE_INSTALL or device == DEVICE_CPU:
        return
    _ = device_installed(device, process_vendor)


def device_admit_class(model: PythonObject, value: PythonObject, cpu_only: Bool, cpu_classes: PythonObject,
    b: PythonObject, connect: PythonObject) raises:
    """`device_admit` for a class whose constructor is the shared wrapper
    (python/mojolearn/_mode.py `_install_numeric_mode`), and the drop of the
    estimator's binding route (a constructor run again on a live estimator).

    An explicit `'cpu'` on a GPU installation runs only for a class whose
    public operations take their images from the estimator's route.
    `cpu_classes` is the ONE table of that answer (class -> True, or the
    sentence why not: its family has no CPU implementation, or its door is
    not routed yet), generated at class definition from the host surface
    manifest (python/mojolearn/_device_table.py). A class the table does not
    hold is refused: nothing runs on a device its caller did not name.

    `connect` (the context's `device_cpu_connect`,
    python/mojolearn/_native_family_bridge.py `connect_cpu`) loads the host
    label and metrics images of the shared family context: called on the
    admitted CPU branch only, so a process with no device="cpu" estimator
    never loads them; once per process (it returns at a flag after)."""
    var device = device_unpack(value, b)
    records_drop(RECORDS_PREPARED, model, DEVICE_BINDINGS_SLOT)
    if device == DEVICE_INSTALL:
        return
    if device == DEVICE_GPU:
        _ = device_installed(device, 0 if cpu_only else 1)
        return
    if cpu_only:
        return
    var cls = py_type(model)
    var answer = py_get(cpu_classes, cls) if cpu_classes is not None else PythonObject(None)
    if answer is None:
        device_cpu_absent(String(cls.__name__), String("its CPU route is not connected in this installation"))
    if py_type(answer) is b.str:
        device_cpu_absent(String(cls.__name__), String(answer))
    if connect is not None:
        _ = py_call(connect)


def device_cpu_connect_late(context: PythonObject) raises -> Bool:
    """Admit the CPU device once for an estimator that reached a device="cpu"
    operation without construction (unpickle, copy, archive load: lane
    fx-leftovers item 3, 2026-11-21). Construction and set_params admit it
    in `device_admit_class`; a restored estimator never ran either, so its
    label steps and its score's metrics silently used the process device's
    images. The family contexts that load their CPU images lazily
    (python/mojolearn/_native_family_bridge.py `shared_context`,
    `cpu_scope_later`) hold `device_cpu_connect`; it is called here once
    (it returns at its own flag after) and then set to None in place, so a
    context it could not connect is never retried per call. Called only on
    the CPU branch when the CPU entry (`labels_cpu_scope`,
    `metrics_cpu_cells`) is missing: never on a GPU or connected call.
    True when the connection ran."""
    var connect = py_get(context, "device_cpu_connect")
    if connect is None:
        return False
    py_setitem(context, "device_cpu_connect", PythonObject(None))
    _ = py_call(connect)
    return True


def device_route_clear(model: PythonObject) raises:
    """Forget the estimator's routes (a new `__init__`, `set_params`): its
    next operation reads the device and the mode again."""
    records_drop(RECORDS_PREPARED, model, DEVICE_ROUTE_SLOT)
    records_drop(RECORDS_PREPARED, model, DEVICE_BINDINGS_SLOT)


def device_route_owner(model: PythonObject) raises -> PythonObject:
    """The estimator's entry-route owner, or None: one record read. For a
    family whose hit test is its own (LogisticRegression compares its class
    and its public context before the fields are admitted)."""
    return records_get(RECORDS_PREPARED, model, DEVICE_ROUTE_SLOT)


def device_route_held(model: PythonObject, mode: PythonObject, default_mode: PythonObject,
    context: PythonObject) raises -> PythonObject:
    """The estimator's resolved route owner for this operation, or None on a
    miss (first operation, a changed numeric mode, another connection
    context). A hit is one record read and three pointer compares."""
    var owner = records_get(RECORDS_PREPARED, model, DEVICE_ROUTE_SLOT)
    if owner is not None:
        var native = owner.downcast_value_ptr[DeviceRouteOwner]()
        if (native[].resolved and native[].mode is mode and native[].default_mode is default_mode
                and native[].context is context):
            return owner
    return PythonObject(None)


def device_route_keep(model: PythonObject, var built: DeviceRouteOwner) raises -> PythonObject:
    """Store a freshly resolved owner for the estimator and return it. An
    estimator that takes no weak reference keeps no owner: it resolves at
    every operation (records_put answers False and stores nothing)."""
    var fresh = PythonObject(alloc=built^)
    _ = records_put(RECORDS_PREPARED, model, DEVICE_ROUTE_SLOT, fresh)
    return fresh


def device_route_value(owner: PythonObject) raises -> DeviceRoute:
    """The route of a held or freshly kept owner, by value. The caller keeps
    `owner` named until its operation returns: it pins the images."""
    return owner.downcast_value_ptr[DeviceRouteOwner]()[].route


def device_route_refuse(owner: PythonObject, fallback: String) raises:
    """Raise why this estimator's route holds no optional entry (failure path
    of the one operation that needs it)."""
    var text = owner.downcast_value_ptr[DeviceRouteOwner]()[].refusal.copy()
    if text.byte_length() == 0:
        text = fallback.copy()
    raise Error(text)


def device_binding_held(model: PythonObject, name: String, check_mode: Bool, mode: PythonObject,
    default_mode: PythonObject, context: PythonObject) raises -> PythonObject:
    """The compiled module the estimator's binding route holds for `name`,
    or None on a miss. A hit is one record read, one pointer compare of the
    context and, per held binding (one or two per estimator), a name compare
    and two pointer compares."""
    var owner = records_get(RECORDS_PREPARED, model, DEVICE_BINDINGS_SLOT)
    if owner is None:
        return owner
    var native = owner.downcast_value_ptr[DeviceRouteOwner]()
    if native[].context is not context:
        return PythonObject(None)
    for i in range(len(native[].names)):  # small-loop(binding names of one estimator: one or two, never data)
        if (native[].modes[i] is mode and native[].defaults[i] is default_mode
                and (native[].checked[i] or not check_mode) and native[].names[i] == name):
            return native[].modules[i]
    return PythonObject(None)


def device_binding_device(model: PythonObject, fields: PythonObject, context: PythonObject, cpu_only: Bool,
    b: PythonObject) raises -> Int:
    """The device the estimator's binding route runs on (DEVICE_GPU or
    DEVICE_CPU): the route's, when one is held for this context, else the
    one read of `device` (None: this installation's device; an explicit
    `'gpu'` in a CPU-only process refused in words). A miss path only."""
    var owner = records_get(RECORDS_PREPARED, model, DEVICE_BINDINGS_SLOT)
    if owner is not None:
        var native = owner.downcast_value_ptr[DeviceRouteOwner]()
        if native[].context is context:
            return native[].route.device
    return device_installed(device_request(fields, b), 0 if cpu_only else 1)


def device_route_resolved(model: PythonObject) raises -> Int:
    """The device an estimator's route resolved to (DEVICE_GPU or
    DEVICE_CPU): its entry route's, else its binding route's; -1 when it
    holds neither yet. Two record reads at most; for metadata answers
    (`vendor_used`), never for a public operation's path."""
    var owner = records_get(RECORDS_PREPARED, model, DEVICE_ROUTE_SLOT)
    if owner is not None:
        var native = owner.downcast_value_ptr[DeviceRouteOwner]()
        if native[].resolved:
            return native[].route.device
    var bound = records_get(RECORDS_PREPARED, model, DEVICE_BINDINGS_SLOT)
    if bound is not None:
        return bound.downcast_value_ptr[DeviceRouteOwner]()[].route.device
    return -1


def device_binding_resolved(model: PythonObject, context: PythonObject) raises -> Int:
    """The device the estimator's binding route resolved to for this context
    (DEVICE_GPU or DEVICE_CPU), or -1 when the estimator holds no binding
    route yet. One record read; for a door that took its module through
    `estimator_binding(model, ...)` earlier in the same operation and needs
    the route's answer for something that is not a module (which fold store
    to read: `door_rows_store_routed`). Never a read of the parameter."""
    var owner = records_get(RECORDS_PREPARED, model, DEVICE_BINDINGS_SLOT)
    if owner is None:
        return -1
    var native = owner.downcast_value_ptr[DeviceRouteOwner]()
    if native[].context is not context:
        return -1
    return native[].route.device


def device_binding_keep(model: PythonObject, device: Int, name: String, check_mode: Bool, mode: PythonObject,
    default_mode: PythonObject, context: PythonObject, module: PythonObject) raises:
    """Record the module `name` resolved to on the estimator's binding route
    (a miss: first operation, a new binding name, a changed mode object). An
    estimator that takes no weak reference keeps nothing and resolves at
    every operation."""
    var owner = records_get(RECORDS_PREPARED, model, DEVICE_BINDINGS_SLOT)
    if owner is not None:
        var native = owner.downcast_value_ptr[DeviceRouteOwner]()
        # A retained fitted mode and the constructor's mode are two objects:
        # both stay held. Past a small bound (a caller that builds a new mode
        # object per call) the route starts again.
        if native[].context is context and native[].route.device == device and len(native[].names) < 16:
            native[].names.append(name.copy())
            native[].checked.append(check_mode)
            native[].modes.append(mode)
            native[].defaults.append(default_mode)
            native[].modules.append(module)
            return
    if records_open(model) == 0:
        # No weak reference, no record: nothing is allocated for it.
        return
    var built = DeviceRouteOwner(device)
    built.resolved = True
    built.context = context
    built.names.append(name.copy())
    built.checked.append(check_mode)
    built.modes.append(mode)
    built.defaults.append(default_mode)
    built.modules.append(module)
    var fresh = PythonObject(alloc=built^)
    _ = records_put(RECORDS_PREPARED, model, DEVICE_BINDINGS_SLOT, fresh)
