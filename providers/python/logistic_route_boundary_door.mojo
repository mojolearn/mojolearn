# SPDX-License-Identifier: Apache-2.0
"""UNPACK of a LogisticRegression's device and numeric mode into its native route owner.

The sibling of providers/python/linear_route_boundary_door.mojo for the
quasi-Newton entry, with the same rule and the same words: `device=None`
(default: this installation's device), `device="gpu"` or `device="cpu"` is a
constructor parameter, stored as given, checked when it is given
(constructor, `set_params`) and read ONCE, at the estimator's first
operation, into a `LogisticRouteOwner` kept in the weak model-record table.
The device reader and its refusals are the linear door's
(`linear_device_unpack`, core/linear_route.mojo); only the owner and what it
resolves are this family's.

The owner holds the resolved route (core/logistic_route.mojo: the thin entry
of the device image or of the CPU image and what that image said about
itself), the class-weight table of the same device, and the connection
context the estimator's class runs in (the explicit Host* class context or
the public one). Every later operation reads the owner and calls through it.

What a call still compares: the public connection context object and the
estimator's class (two pointer compares, before the fields are admitted),
then the estimator's `numeric_mode` object and, when that is None, the
process default mode object (two pointer compares). A changed mode resolves
again; `set_params` and a new `__init__` drop the owner.

The owner replaces, per public call: the walk of the estimator's class
hierarchy against the explicit host classes, the mode word build and its
normalisation, the table read by mode word, the provider admission and the
connection scan (twice per fit, up to three times per predict), the compiled
tier compare, the fold entry's host-class and mode tests.
"""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder, lookup_py_type_object
from bindings.py_type_registry import py_type_absent, py_type_attach
from bindings.native_owner_registry_unpack import register_native_owner_type
from bindings.native_owner_registry_pack import borrowed_native_owner
from bindings.py_balanced import py_call, py_type, py_get, py_item
from bindings.py_model_records import records_get, records_put, records_drop, RECORDS_PREPARED
from bindings.logistic_keys_unpack import logistic_keys
from bindings.logistic_descriptor_boundary_door import logistic_connection
from glm.logistic_descriptor_boundary_run import (
    logistic_mode, logistic_host, logistic_multiclass, logistic_driver,
)
from core.linear_metadata import linear_mode_name, linear_vendor_code
from core.linear_route import (
    LINEAR_DEVICE_CPU, LINEAR_DEVICE_INSTALL, linear_device_installed,
)
from core.logistic_native_api import QnFitRequest, QnQueryRequest
from core.logistic_route import LogisticRoute, logistic_route_admit, logistic_fit_run, logistic_query_run
from providers.python.linear_route_boundary_door import linear_device_unpack

comptime _LOGISTIC_ROUTE_SLOT = "logistic_route_v1"
comptime _LOGISTIC_ROUTE_PINS = 7


struct LogisticRouteOwner(Movable, Writable):
    var route: LogisticRoute
    # The objects the route was resolved for. Held, so their addresses stay
    # theirs while this owner lives; a call compares identities only.
    var cls: PythonObject
    var base: PythonObject
    var mode: PythonObject
    var default_mode: PythonObject
    # The connection context of the estimator's class: `base`, or the
    # explicit Host* class context selected from it.
    var context: PythonObject
    # The entry dictionary of the resolved image (the image stays loaded
    # while a call runs with the GIL released) and the class-weight table of
    # the same device, or the text of why that table is not connected (said
    # by a fit that takes class weights).
    var entries: PythonObject
    var xtrees: PythonObject

    def __init__(out self, var route: LogisticRoute, cls: PythonObject, base: PythonObject,
                 mode: PythonObject, default_mode: PythonObject, context: PythonObject,
                 entries: PythonObject, xtrees: PythonObject):
        self.route = route^
        self.cls = cls
        self.base = base
        self.mode = mode
        self.default_mode = default_mode
        self.context = context
        self.entries = entries
        self.xtrees = xtrees

    def write_to(self, mut writer: Some[Writer]):
        writer.write("LogisticRouteOwner(device=", self.route.device, ", vendor=", self.route.vendor, ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


def logistic_route_owner_pins(value: Int, index: Int) abi("C") -> Int:
    if not value or index < -1 or index >= _LOGISTIC_ROUTE_PINS:
        return -1
    try:
        var owner = borrowed_native_owner(value)
        var native = owner.downcast_value_ptr[LogisticRouteOwner]()
        var result = _LOGISTIC_ROUTE_PINS
        if index == 0:
            result = Int(native[].cls._obj_ptr)
        elif index == 1:
            result = Int(native[].base._obj_ptr)
        elif index == 2:
            result = Int(native[].mode._obj_ptr)
        elif index == 3:
            result = Int(native[].default_mode._obj_ptr)
        elif index == 4:
            result = Int(native[].context._obj_ptr)
        elif index == 5:
            result = Int(native[].entries._obj_ptr)
        elif index == 6:
            result = Int(native[].xtrees._obj_ptr)
        _ = owner
        return result
    except:
        return -1


def register_logistic_route_owner(mut module: PythonModuleBuilder) raises:
    var existing = PythonObject(None)
    try:
        existing = lookup_py_type_object[LogisticRouteOwner]()
    except error:
        if not py_type_absent(error):
            raise error
        _ = module.add_type[LogisticRouteOwner]("_LogisticRouteOwner")
        return
    py_type_attach(module.module, "_LogisticRouteOwner", existing)


def register_logistic_route_owner_identity(published: PythonObject) raises:
    # After PythonModuleBuilder.finalize() only: Mojo 1.2 creates the type
    # object in finalize() (bindings/native_owner_registry.mojo).
    register_native_owner_type[LogisticRouteOwner](published, logistic_route_owner_pins)


def logistic_device_admit(value: PythonObject, context: PythonObject, b: PythonObject) raises:
    """Check a device where it is given (constructor, `set_params`): an
    unknown value, and an explicit `'gpu'` in a CPU-only process, are refused
    in words before the estimator changes (the linear door's reader and
    words). The default None reads nothing here. The explicit Host* class
    runs on the CPU by class and takes any admitted value."""
    var device = linear_device_unpack(value, b)
    if device == LINEAR_DEVICE_INSTALL or device == LINEAR_DEVICE_CPU:
        return
    var nk = logistic_keys()
    if py_get(context, nk[].k_lr_host_model) is not None:
        return
    _ = linear_device_installed(device, linear_vendor_code(String(py_item(context, "vendor"))))


def logistic_route_clear(model: PythonObject) raises:
    """Forget the estimator's route (a new `__init__`, `set_params`): its next
    operation reads the device and the mode again."""
    records_drop(RECORDS_PREPARED, model, _LOGISTIC_ROUTE_SLOT)


def logistic_route_held(model: PythonObject, base: PythonObject) raises -> PythonObject:
    """The estimator's route owner when it was resolved for this connection
    context and this class, else None: one record read and two pointer
    compares, before the estimator's fields are admitted. The owner names the
    context the class runs in (`logistic_route_context`), so a hit needs no
    walk of the class hierarchy against the explicit host classes."""
    var owner = records_get(RECORDS_PREPARED, model, _LOGISTIC_ROUTE_SLOT)
    if owner is None:
        return owner
    var native = owner.downcast_value_ptr[LogisticRouteOwner]()
    if native[].base is base and native[].cls is py_type(model):
        return owner
    return PythonObject(None)


def logistic_route_context(owner: PythonObject) raises -> PythonObject:
    """The connection context the owner's estimator class runs in."""
    return owner.downcast_value_ptr[LogisticRouteOwner]()[].context


def logistic_route_xtrees(owner: PythonObject) raises -> PythonObject:
    """The class-weight table of the route's device. A table that is not
    connected is refused here, in the words recorded when it failed to load
    (a fit that takes class weights is the only caller)."""
    var table = owner.downcast_value_ptr[LogisticRouteOwner]()[].xtrees
    if py_type(table) is py_type(PythonObject("")):
        raise Error(String(table))
    return table


def _logistic_route_build(model: PythonObject, device: Int, mode: PythonObject, default_mode: PythonObject,
    base: PythonObject, context: PythonObject, b: PythonObject) raises -> LogisticRouteOwner:
    var nk = logistic_keys()
    var selected = mode
    if mode is None:
        selected = default_mode
    var explicit = py_get(context, nk[].k_lr_host_model) is not None
    var process_vendor = linear_vendor_code(String(py_item(context, "vendor")))
    var entries = PythonObject(None)
    var xtrees = PythonObject(None)
    var expected_mode = 1
    var resolved = LINEAR_DEVICE_CPU
    var cpu_table = False
    if explicit:
        # The explicit Host* class (mojolearn.host_model): CPU identical by
        # class, whatever `device` its inherited constructor stored.
        if mode is not None and (py_type(mode) is not b.str or String(mode) != "identical"):
            raise Error("mojolearn: " + String(py_type(model).__name__)
                + " runs IDENTICAL only on the host; this model was saved " + String(py_call(b.repr, mode)))
        var word = PythonObject("identical")
        entries = py_item(py_item(context, nk[].k_lr_bindings), word)
        xtrees = py_item(py_item(context, nk[].k_lr_x_trees), word)
    else:
        if py_type(selected) is not b.str:
            raise Error("numeric_mode must be fast, identical or None")
        var mode_name = linear_mode_name(String(selected))
        expected_mode = 1 if mode_name == "identical" else 0
        var word = PythonObject(mode_name)
        # None is this installation's device; an explicit 'gpu' in a CPU-only
        # process is refused in words (an instance that reached here without
        # its constructor's check: unpickled, or its field assigned).
        resolved = linear_device_installed(device, process_vendor)
        if resolved == LINEAR_DEVICE_CPU and process_vendor != 0:
            # A GPU installation: the CPU images, loaded beside the device
            # images at import (python/mojolearn/_native_logistic_bridge.py
            # connect_cpu_device).
            if expected_mode != 1:
                raise Error("ValueError: device='cpu' runs in identical mode only; numeric_mode='fast' is for Apple GPUs")
            var cpu = py_get(context, "lr_cpu_bindings")
            if cpu is None:
                raise Error("ImportError: device='cpu' is not connected in this installation")
            entries = py_item(cpu, word)
            xtrees = py_item(py_item(context, "lr_cpu_x_trees"), word)
            cpu_table = True
        else:
            # The process's own images: the device images for a GPU request,
            # the CPU images in a CPU-only process (where every request that
            # got here resolved to the CPU).
            entries = py_item(py_item(context, nk[].k_lr_bindings), word)
            xtrees = py_item(py_item(context, nk[].k_lr_x_trees), word)
    # A refusal recorded at import as "<Class>: <message>" is raised again.
    if py_type(entries) is b.str:
        raise Error(String(entries))
    # The connection admits the provider once per image and keeps its pin
    # for the process (bindings/logistic_descriptor_boundary_door.mojo).
    var connection = logistic_connection(entries)
    var route = logistic_route_admit(resolved, logistic_driver(connection), logistic_mode(connection),
        logistic_host(connection), logistic_multiclass(connection), expected_mode, process_vendor,
        cpu_table, explicit, py_get(context, nk[].k_logistic_fold_native_api_v1) is not None)
    return LogisticRouteOwner(route^, py_type(model), base, mode, default_mode, context, entries, xtrees)


@fieldwise_init
struct LogisticRouteCall(Movable):
    """One operation's route: the resolved entry by value, and the owner
    whose pins keep its image loaded until the operation returns."""
    var route: LogisticRoute
    var owner: PythonObject


def logistic_route_current(held: PythonObject, fields: PythonObject, context: PythonObject) raises -> Bool:
    """True when `held` (what `logistic_route_held` answered for this call)
    is the route of the estimator's numeric mode as it stands: the mode
    object and, when that is None, the process default mode object, compared
    by identity. False for None, a changed mode or a changed default."""
    if held is None:
        return False
    var nk = logistic_keys()
    var mode = py_get(fields, nk[].k_numeric_mode)
    var native = held.downcast_value_ptr[LogisticRouteOwner]()
    if native[].mode is not mode:
        return False
    if mode is not None:
        return native[].default_mode is None
    return native[].default_mode is py_item(py_item(context, nk[].k_backend_fields), nk[].k__DEFAULT_MODE)


def logistic_route_resolve(model: PythonObject, fields: PythonObject, base: PythonObject,
    context: PythonObject, b: PythonObject) raises -> PythonObject:
    """Resolve and store the estimator's route (first operation, a changed
    numeric mode, an instance built without its constructor): the one read
    of `device`. `b` is the builtins module the calling door admitted."""
    var nk = logistic_keys()
    var mode = py_get(fields, nk[].k_numeric_mode)
    var default_mode = PythonObject(None)
    if mode is None:
        default_mode = py_item(py_item(context, nk[].k_backend_fields), nk[].k__DEFAULT_MODE)
    # None, and an instance without the field (built without its
    # constructor, loaded, unpickled from an older version), run this
    # installation's device.
    var device = linear_device_unpack(py_get(fields, "device"), b)
    var fresh = PythonObject(alloc=_logistic_route_build(model, device, mode, default_mode, base, context, b))
    # An estimator that takes no weak reference keeps no owner: it resolves
    # at every operation (records_put answers False and stores nothing).
    _ = records_put(RECORDS_PREPARED, model, _LOGISTIC_ROUTE_SLOT, fresh)
    return fresh


def logistic_route_of(owner: PythonObject) raises -> LogisticRouteCall:
    """One operation's route out of its owner (by value; the owner rides along)."""
    return LogisticRouteCall(owner.downcast_value_ptr[LogisticRouteOwner]()[].route.copy(), owner)


def logistic_route_fit(routed: LogisticRouteCall, request: QnFitRequest) raises -> Int:
    """RUN of fit with the GIL released; the owner pins the image."""
    var iterations = 0
    try:
        with GILReleased(Python()):
            iterations = logistic_fit_run(routed.route, request)
    finally:
        _ = routed.owner
    return iterations


def logistic_route_query(routed: LogisticRouteCall, operation: Int, request: QnQueryRequest) raises:
    """RUN of one inference entry with the GIL released; the owner pins the image."""
    try:
        with GILReleased(Python()):
            logistic_query_run(routed.route, operation, request)
    finally:
        _ = routed.owner
