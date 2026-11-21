# SPDX-License-Identifier: Apache-2.0
"""The LogisticRegression resolver of the shared device route owner.

`device=None` (default: this installation's device), `device="gpu"` or
`device="cpu"` is a constructor parameter, stored as given, checked when it
is given (constructor, `set_params`) and read ONCE, at the estimator's first
operation, into the shared `DeviceRouteOwner`
(providers/python/device_route_boundary_door.mojo) kept in the weak
model-record table. The owner, its slot, the device reader and every device
word are the shared ones; only what the route resolves and this family's hit
test (class and public context before the fields are admitted) are here.
`LogisticRouteOwner` (its seven pins and registration) was deleted
2026-11-20 by lane dw-core; code recoverable at 3a99eaa0dc.

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
from bindings.py_balanced import py_call, py_type, py_get, py_item
from bindings.logistic_keys_unpack import logistic_keys
from bindings.logistic_descriptor_boundary_door import logistic_connection
from glm.logistic_descriptor_boundary_run import (
    logistic_mode, logistic_host, logistic_multiclass, logistic_driver,
)
from core.linear_metadata import linear_mode_name, linear_vendor_code
from core.device_route import DEVICE_CPU, DEVICE_INSTALL, device_installed, device_cpu_fast
from core.logistic_native_api import QnFitRequest, QnQueryRequest
from core.logistic_route import (
    LogisticRoute, logistic_route_admit, logistic_route_pack, logistic_route_view, logistic_fit_run,
    logistic_query_run,
)
from providers.python.device_route_boundary_door import (
    DeviceRouteOwner, device_unpack, device_request, device_route_owner, device_route_keep,
    device_cpu_connect_late,
)


def logistic_device_admit(value: PythonObject, context: PythonObject, b: PythonObject) raises:
    """Check a device where it is given (constructor, `set_params`): an
    unknown value, and an explicit `'gpu'` in a CPU-only process, are refused
    in words before the estimator changes (the linear door's reader and
    words). The default None reads nothing here. The explicit Host* class
    runs on the CPU by class and takes any admitted value."""
    var device = device_unpack(value, b)
    if device == DEVICE_INSTALL or device == DEVICE_CPU:
        return
    var nk = logistic_keys()
    if py_get(context, nk[].k_lr_host_model) is not None:
        return
    _ = device_installed(device, linear_vendor_code(String(py_item(context, "vendor"))))


def logistic_route_held(model: PythonObject, base: PythonObject) raises -> PythonObject:
    """The estimator's route owner when it was resolved for this connection
    context and this class, else None: one record read and two pointer
    compares, before the estimator's fields are admitted. The owner names the
    context the class runs in (`logistic_route_context`), so a hit needs no
    walk of the class hierarchy against the explicit host classes."""
    var owner = device_route_owner(model)
    if owner is None:
        return owner
    var native = owner.downcast_value_ptr[DeviceRouteOwner]()
    # `context` is the public connection context the route was resolved for;
    # `scope` is the context the estimator's class runs in.
    if native[].resolved and native[].context is base and native[].cls is py_type(model):
        return owner
    return PythonObject(None)


def logistic_route_context(owner: PythonObject) raises -> PythonObject:
    """The connection context the owner's estimator class runs in."""
    return owner.downcast_value_ptr[DeviceRouteOwner]()[].scope


def logistic_route_xtrees(owner: PythonObject) raises -> PythonObject:
    """The class-weight table of the route's device. A table that is not
    connected is refused here, in the words recorded when it failed to load
    (a fit that takes class weights is the only caller)."""
    var table = owner.downcast_value_ptr[DeviceRouteOwner]()[].table
    if py_type(table) is py_type(PythonObject("")):
        raise Error(String(table))
    return table


def _logistic_route_build(model: PythonObject, device: Int, mode: PythonObject, default_mode: PythonObject,
    base: PythonObject, context: PythonObject, b: PythonObject) raises -> DeviceRouteOwner:
    var nk = logistic_keys()
    var selected = mode
    if mode is None:
        selected = default_mode
    var explicit = py_get(context, nk[].k_lr_host_model) is not None
    var process_vendor = linear_vendor_code(String(py_item(context, "vendor")))
    var entries = PythonObject(None)
    var xtrees = PythonObject(None)
    var expected_mode = 1
    var resolved = DEVICE_CPU
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
        resolved = device_installed(device, process_vendor)
        if resolved == DEVICE_CPU and process_vendor != 0:
            # A GPU installation: the CPU images, loaded beside the device
            # images at the first device="cpu" admission (fx-host-import:
            # python/mojolearn/_native_logistic_bridge.py connect_cpu_device,
            # no longer at import); an estimator restored without
            # construction connects them here, once (both tables load
            # together).
            if expected_mode != 1:
                device_cpu_fast()
            var cpu = py_get(context, "lr_cpu_bindings")
            if cpu is None and device_cpu_connect_late(context):
                cpu = py_get(context, "lr_cpu_bindings")
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
    var owner = DeviceRouteOwner(resolved)
    owner.route = logistic_route_pack(route)
    owner.cls = py_type(model)
    owner.context = base
    owner.mode = mode
    owner.default_mode = default_mode
    owner.scope = context
    owner.images = entries
    owner.table = xtrees
    owner.resolved = True
    return owner^


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
    var native = held.downcast_value_ptr[DeviceRouteOwner]()
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
    var device = device_request(fields, b)
    # An estimator that takes no weak reference keeps no owner: it resolves
    # at every operation (device_route_keep stores nothing for it).
    return device_route_keep(model, _logistic_route_build(model, device, mode, default_mode, base, context, b))


def logistic_route_of(owner: PythonObject) raises -> LogisticRouteCall:
    """One operation's route out of its owner (by value; the owner rides along)."""
    return LogisticRouteCall(logistic_route_view(owner.downcast_value_ptr[DeviceRouteOwner]()[].route), owner)


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
