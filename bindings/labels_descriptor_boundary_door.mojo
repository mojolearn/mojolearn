# SPDX-License-Identifier: Apache-2.0
"""Label provider admission and image pins around typed native kernel calls (door).

The route word, the typed driver and the device-or-host dispatch are in
`bindings/labels_descriptor_boundary_run.mojo`."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.py_balanced import py_get, py_type, py_fields, py_call, py_setitem
from bindings.classical_keys_unpack import classical_keys
from core.device_route import DEVICE_CPU
from providers.python.device_route_boundary_door import device_route_resolved, device_cpu_connect_late
from core.shared_slot import SharedSlot
from core.hotpath_impl import HpStat
from core.labels_native_api import (
    LABELS_NATIVE_ABI_NAME, LABELS_NATIVE_FACTORY, LABELS_NATIVE_OPERATION_COUNT,
    LabelsNativeDriver, LabelsPrepareRequest, LabelsReduceRequest,
    LabelsArgmaxRequest, LabelsGatherRequest, LabelsRatioRequest, LabelsThresholdRequest,
)
from core.labels_native_host import labels_host_fsum
from bindings.labels_descriptor_boundary_run import (
    labels_route_state, labels_route_check, labels_route_device, labels_route_driver,
    labels_prepare_run, labels_reduce_run, labels_threshold_run, labels_argmax_run, labels_gather_run,
    labels_ratio_run,
)
from providers.python.neural_descriptor_pyapi_door import (
    NeuralDescriptorProvider, neural_descriptor_entry, neural_descriptor_connect_factory_once,
)


struct _LabelsProviderPin(Defaultable, Movable):
    # Python owns only the admitted extension image; the route word, dispatch
    # and all numerical state are native (bindings/labels_descriptor_boundary_run.mojo).
    var provider: PythonObject

    def __init__(out self):
        self.provider = PythonObject(None)


comptime _LABELS_PROVIDER_PIN = SharedSlot[_LabelsProviderPin, "MojolearnLabelsProviderPinV1"]


def _labels_route(context: PythonObject) raises -> Int:
    var state = labels_route_state()
    if state != 0:
        return state
    var base = py_get(context, "labels_base_binding")
    if base is None:
        raise Error("ImportError: mojolearn: the label steps need the label context (labels_base_binding)")
    if py_type(base) is py_type(PythonObject(False)):
        # `_labels_pin` answers a host context before it asks for the route.
        raise Error("RuntimeError: mojolearn: a host label context asked for the device label route")
    if py_type(base) is py_type(PythonObject("")):
        raise Error(String(base))
    var factory = neural_descriptor_entry(base, LABELS_NATIVE_FACTORY)
    var provider = neural_descriptor_connect_factory_once(factory, String(LABELS_NATIVE_ABI_NAME), LABELS_NATIVE_OPERATION_COUNT)
    var driver = LabelsNativeDriver(provider.downcast_value_ptr[NeuralDescriptorProvider]()[].record[0].entry)
    # RUN checks the compiled mode and capabilities before anything is stored;
    # the pin is stored first so the published device route always has it.
    labels_route_check(driver)
    _LABELS_PROVIDER_PIN.get_or_create_ptr()[].provider = provider
    labels_route_device(driver^)
    return 1


@fieldwise_init
struct _LabelsPin(Movable):
    var provider: PythonObject
    var driver: Optional[LabelsNativeDriver]


def _labels_pin(context: PythonObject) raises -> _LabelsPin:
    # THE CALLER'S CONTEXT names the label image, never a process state: a
    # context whose `labels_base_binding` is exactly False (a CPU-only
    # installation, and the CPU scope of a `device="cpu"` estimator beside a
    # GPU set: `_native_labels_bridge.cpu_scope`) runs the host kernels of
    # this image and connects nothing. One dictionary read with the interned
    # key and one pointer compare per label step (it was one route-word
    # read); every other context takes the device provider, connected once.
    if py_get(context, classical_keys()[].k_labels_base_binding) is PythonObject(False):
        return _LabelsPin(PythonObject(None), Optional[LabelsNativeDriver]())
    _ = _labels_route(context)
    return _LabelsPin(_LABELS_PROVIDER_PIN.get_or_create_ptr()[].provider, labels_route_driver())


def labels_prepare_targets(context: PythonObject, y_addr: Int, y_code: Int, n: Int,
    classes_addr: Int, codes_addr: Int, w_addr: Int, w_code: Int, w_flags: Int) raises -> Int:
    var request = LabelsPrepareRequest(y_addr, classes_addr, codes_addr, w_addr, y_code, n, w_code, w_flags)
    var pin = _labels_pin(context)
    var driver = pin.driver.copy()
    try:
        with GILReleased(Python()):
            return labels_prepare_run(driver^, request)
    finally:
        _ = pin


def labels_reduce_stat(context: PythonObject, addr: Int, code: Int, n: Int, what: Int) raises -> HpStat:
    var request = LabelsReduceRequest(addr, code, n, what)
    var pin = _labels_pin(context)
    var driver = pin.driver.copy()
    try:
        with GILReleased(Python()):
            return labels_reduce_run(driver^, request)
    finally:
        _ = pin


def labels_threshold(context: PythonObject, request: LabelsThresholdRequest) raises -> Int:
    var pin = _labels_pin(context)
    var driver = pin.driver.copy()
    try:
        with GILReleased(Python()):
            return labels_threshold_run(driver^, request)
    finally:
        _ = pin


def labels_argmax_rows(context: PythonObject, wide: Bool, scores_addr: Int, rows: Int, cols: Int, dst_addr: Int) raises:
    var request = LabelsArgmaxRequest(scores_addr, dst_addr, rows, cols)
    var pin = _labels_pin(context)
    var driver = pin.driver.copy()
    try:
        with GILReleased(Python()):
            labels_argmax_run(driver^, wide, request)
    finally:
        _ = pin


def labels_gather(context: PythonObject, kind: Int, table_addr: Int, n_table: Int,
    codes_addr: Int, n: Int, dst_addr: Int) raises:
    # The existing caller contract treats every kind other than 0/1 as float64.
    var native_kind = kind if kind == 0 or kind == 1 else 2
    var request = LabelsGatherRequest(table_addr, codes_addr, dst_addr, n_table, n)
    var pin = _labels_pin(context)
    var driver = pin.driver.copy()
    try:
        with GILReleased(Python()):
            labels_gather_run(driver^, native_kind, request)
    finally:
        _ = pin


def labels_class_ratio_f64(context: PythonObject, counts_addr: Int, k: Int, n: Int,
    mode: Int, out_addr: Int) raises -> Int:
    var request = LabelsRatioRequest(counts_addr, out_addr, k, n, mode)
    var pin = _labels_pin(context)
    var driver = pin.driver.copy()
    try:
        with GILReleased(Python()):
            return labels_ratio_run(driver^, request)
    finally:
        _ = pin


def labels_fsum_f64(src_addr: Int, n: Int, out_addr: Int) raises -> Int:
    with GILReleased(Python()):
        return labels_host_fsum(src_addr, n, out_addr)


def labels_scope(model: PythonObject, context: PythonObject) raises -> PythonObject:
    """The context a classifier's door hands the label boundary
    (bindings/classical_labels_door.mojo and the six steps here) for THIS
    estimator: `context` itself, or, for an estimator whose device is the CPU
    beside a GPU set, the family's CPU label scope
    (`context["labels_cpu_scope"]`: the same context with the host label
    kernels and the CPU metrics tables, built once at import by
    `_native_labels_bridge.cpu_scope`). The estimator's device is its
    resolved route's (two record reads at most); before its first operation
    resolved one, the admitted `device` parameter is read once. A family
    without a CPU scope (no CPU implementation, or a CPU-only installation,
    whose one context is already the host one) gets `context`.

    The call site changes only its argument: `classical_encode_labels(y,
    labels_scope(model, context))`."""
    var device = device_route_resolved(model)
    if device < 0:
        # No route yet: the parameter was checked where it was given, so an
        # exact 'cpu' string is the only value that names the CPU.
        var value = py_get(py_fields(model), "device")
        if value is None or py_type(value) is not py_type(PythonObject("")) or String(value) != "cpu":
            return context
    elif device != DEVICE_CPU:
        return context
    var scope = py_get(context, "labels_cpu_scope")
    if scope is None:
        # A restored estimator (no construction ran the device admission):
        # admit the CPU device once, then read the scope it built.
        if not device_cpu_connect_late(context):
            return context
        scope = py_get(context, "labels_cpu_scope")
        if scope is None:
            return context
    # The scope is a copy of the context it was made from. A family context
    # that grew after the copy (a module imported later added its tables), or
    # a context derived from it (the typed context a door builds once per
    # connected context, which carries the base's scope key), is copied
    # again, once per growth or derivation: the CPU branch only; two
    # dictionary reads, one pointer compare and one length read otherwise.
    var size = len(context)
    var source = py_get(scope, "labels_cpu_source")
    var held = py_get(scope, "labels_cpu_source_size")
    if source is context and held is not None and Int(py=held) == size:
        return scope
    var entries = py_get(context, "labels_cpu_entries")
    if entries is None:
        return scope
    var fresh = py_call(context.copy)
    _ = py_call(fresh.pop, PythonObject("labels_cpu_scope"), PythonObject(None))
    _ = py_call(fresh.update, entries)
    py_setitem(fresh, PythonObject("labels_cpu_source"), context)
    py_setitem(fresh, PythonObject("labels_cpu_source_size"), PythonObject(size))
    py_setitem(context, PythonObject("labels_cpu_scope"), fresh)
    return fresh


# The routed forms `labels_*_on(host, ...)` of the six steps (lane dw-prep, never
# called) were deleted 2026-11-21 by lane label-steps: a classifier routes its
# label steps by the context it hands them (`labels_scope` above).
