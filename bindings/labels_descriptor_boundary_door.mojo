# SPDX-License-Identifier: Apache-2.0
"""Label provider admission and image pins around typed native kernel calls (door).

The route word, the typed driver and the device-or-host dispatch are in
`bindings/labels_descriptor_boundary_run.mojo`."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.py_balanced import py_get, py_type
from core.shared_slot import SharedSlot
from core.hotpath_impl import HpStat
from core.labels_native_api import (
    LABELS_NATIVE_ABI_NAME, LABELS_NATIVE_FACTORY, LABELS_NATIVE_OPERATION_COUNT,
    LabelsNativeDriver, LabelsPrepareRequest, LabelsReduceRequest,
    LabelsArgmaxRequest, LabelsGatherRequest, LabelsRatioRequest, LabelsThresholdRequest,
)
from core.labels_native_host import labels_host_fsum
from bindings.labels_descriptor_boundary_run import (
    labels_route_state, labels_route_host, labels_route_check, labels_route_device, labels_route_driver,
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
        labels_route_host()
        return 2
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
