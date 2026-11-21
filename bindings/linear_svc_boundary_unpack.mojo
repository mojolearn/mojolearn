# SPDX-License-Identifier: Apache-2.0
"""Exact compiled provider admission and lifetime pins for typed LinearSVC."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from core.linear_svc_native_api import (
    LINEAR_SVC_NATIVE_ABI, LINEAR_SVC_NATIVE_COUNT, LinearSvcDriver,
    LinearSvcFitRequest, LinearSvcFitPlan, LinearSvcWeightsState,
)
from core.logistic_native_api import QnQueryRequest, QnFitRequest
from core.qn_single_fit import QnSingleFitPlan
from providers.python.neural_descriptor_pyapi_door import (
    NeuralDescriptorProvider, neural_descriptor_entry, neural_descriptor_connect_factory_once,
)
from core.device_route import DeviceRoute, DEVICE_CPU, DEVICE_GPU
from core.entry_table import EntryRef, entry_ref_word
from core.estimators_entry_table import estimators_vendor_word
from core.logistic_native_api import QN_NATIVE_MODE, QN_NATIVE_MULTICLASS_AVAILABLE
from providers.python.device_route_boundary_door import (
    DeviceRouteOwner, device_route_held, device_route_keep, device_route_value,
)
from providers.python.estimator_binding_pyapi_door import estimator_binding
from bindings.py_balanced import py_call, py_get, py_item
from bindings.svm_keys_unpack import svm_keys

# The words of a LinearSVC / LinearSVR route's `flags` (resolved once, with the
# entry): the binary's PY2MOJO rows bit (the driver's LINEAR_SVC_FLAGS answer
# & 2: the binary class-code query) and QN_NATIVE_MULTICLASS_AVAILABLE != 0.
comptime LINEAR_SVC_ROUTE_ROWS = 1
comptime LINEAR_SVC_ROUTE_MULTICLASS = 2
comptime _PY2MOJO_ROWS = 2


@fieldwise_init
struct LinearSvcConnection(Movable):
    """One operation's driver. `provider` pins the compiled images the entry
    lives in (the estimator's route owner on the door path, the connected
    provider on a resolve); `flags` is the route's LINEAR_SVC_ROUTE_* word."""
    var provider: PythonObject
    var driver: LinearSvcDriver
    var flags: Int


@fieldwise_init
struct LinearSvcWeights(Movable):
    # Python is retained only for admitted input storage. Generated ones,
    # class rows and multiplication output live in native lists.
    var input_pin: PythonObject
    var state: LinearSvcWeightsState


def linear_svc_connection(binding: PythonObject) raises -> LinearSvcConnection:
    var factory = neural_descriptor_entry(binding, "linear_svc_native_api_v1")
    var provider = neural_descriptor_connect_factory_once(factory,
        String(LINEAR_SVC_NATIVE_ABI), LINEAR_SVC_NATIVE_COUNT)
    var driver = LinearSvcDriver(provider.downcast_value_ptr[NeuralDescriptorProvider]()[].record[0].entry)
    var flags = 0
    if (driver.flags() & _PY2MOJO_ROWS) != 0:
        flags |= LINEAR_SVC_ROUTE_ROWS
    if driver.metadata(QN_NATIVE_MULTICLASS_AVAILABLE) != 0:
        flags |= LINEAR_SVC_ROUTE_MULTICLASS
    return LinearSvcConnection(provider, driver^, flags)


def _linear_svc_route_build(model: PythonObject, mode: PythonObject, default_mode: PythonObject,
    context: PythonObject) raises -> DeviceRouteOwner:
    """Resolve a LinearSVC / LinearSVR estimator's driver once: the
    estimators binding of its device and mode (the estimator's binding route:
    device='cpu' beside a GPU set answers the host image), that binding's
    `linear_svc_native_api_v1` provider, the binary's flag words and its tier
    and vendor. Runs at the estimator's first operation, after a changed mode
    and after a new `__init__` / `set_params` (both drop the route); never
    per call."""
    var binding = estimator_binding(model, "_mojolearn_estimators", context)
    var connection = linear_svc_connection(binding)
    var tier = connection.driver.metadata(QN_NATIVE_MODE)
    var vendor = estimators_vendor_word(String(py_call(neural_descriptor_entry(binding, "estimators_vendor"))))
    var owner = DeviceRouteOwner(DEVICE_CPU if vendor == 0 else DEVICE_GPU)
    owner.route.mode = tier
    owner.route.vendor = vendor
    owner.route.entry = EntryRef(connection.driver.entry, entry_ref_word(tier, vendor, LINEAR_SVC_NATIVE_COUNT))
    owner.route.flags = connection.flags
    owner.mode = mode
    owner.default_mode = default_mode
    owner.context = context
    owner.images = Python.tuple(binding, connection.provider)
    owner.resolved = True
    return owner^


def linear_svc_route(model: PythonObject, fields: PythonObject, context: PythonObject) raises -> LinearSvcConnection:
    """The estimator's driver for this operation through its held route. A
    hit is the mode read (and the default-mode read when it is None), one
    record read and three pointer compares; a miss resolves and keeps it.
    The returned connection pins the route owner (and so its images) until
    the operation drops it."""
    var nk = svm_keys()
    var mode = py_get(fields, nk[].k_numeric_mode, PythonObject(None))
    var default_mode = PythonObject(None)
    if mode is None:
        default_mode = py_item(py_item(context, nk[].k_backend_fields), nk[].k__DEFAULT_MODE)
    var owner = device_route_held(model, mode, default_mode, context)
    if owner is None:
        owner = device_route_keep(model, _linear_svc_route_build(model, mode, default_mode, context))
    var route = device_route_value(owner)
    return LinearSvcConnection(owner, LinearSvcDriver(route.entry.call), route.flags)


def linear_svc_run_fit(connection: LinearSvcConnection, request: LinearSvcFitRequest,
    mut plan: LinearSvcFitPlan) raises:
    try:
        with GILReleased(Python()):
            connection.driver.fit(request, plan)
    finally:
        _ = connection
        _ = plan


def linear_svc_run_query(connection: LinearSvcConnection, operation: Int,
    request: QnQueryRequest) raises:
    try:
        with GILReleased(Python()):
            connection.driver.qn().query(operation, request)
    finally:
        _ = connection


@fieldwise_init
struct LinearSvrFitResult(Movable):
    # Fitted public storage is retained at PACK; metadata is native.
    var weights: PythonObject
    var state: QnSingleFitPlan


def linear_svr_run_fit(connection: LinearSvcConnection, var request: QnFitRequest,
    mut plan: QnSingleFitPlan) raises:
    try:
        with GILReleased(Python()):
            plan.run(connection.driver.qn(), request^)
    finally:
        _ = connection
        _ = plan


@fieldwise_init
struct LinearSvcPrediction(Movable):
    var classes: PythonObject
    var codes: List[Int64]
