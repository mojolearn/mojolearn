# SPDX-License-Identifier: Apache-2.0
"""Label provider admission and image pins around typed native kernel calls."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.py_balanced import py_get, py_type
from core.shared_slot import SharedSlot
from core.hotpath_impl import HpStat
from core.labels_native_api import (
    LABELS_NATIVE_ABI_NAME, LABELS_NATIVE_FACTORY, LABELS_NATIVE_OPERATION_COUNT,
    LABELS_OP_MODE, LABELS_OP_CAPABILITIES, LABELS_CAP_DEVICE,
    LabelsNativeDriver, LabelsPrepareRequest, LabelsReduceRequest,
    LabelsArgmaxRequest, LabelsGatherRequest, LabelsRatioRequest,
)
from core.labels_native_host import LabelsHostKernels, labels_host_fsum
from providers.python.neural_descriptor_pyapi import (
    NeuralDescriptorProvider, neural_descriptor_entry, neural_descriptor_connect_factory_once,
)


struct _LabelsProvider(Defaultable, Movable):
    # Existing process route: 0 unconnected, 1 device, 2 host. Python owns only
    # the admitted extension image; dispatch and all numerical state are native.
    var state: Int
    var provider: PythonObject
    var driver: Optional[LabelsNativeDriver]

    def __init__(out self):
        self.state = 0
        self.provider = PythonObject(None)
        self.driver = Optional[LabelsNativeDriver]()


comptime _LABELS_PROVIDER = SharedSlot[_LabelsProvider, "MojolearnLabelsProviderV3", 3]


def _labels_route(context: PythonObject) raises -> Int:
    var slot = _LABELS_PROVIDER.get_or_create_ptr()
    if slot[].state != 0:
        return slot[].state
    var base = py_get(context, "labels_base_binding")
    if base is None:
        raise Error("ImportError: mojolearn: the label steps need the label context (labels_base_binding)")
    if py_type(base) is py_type(PythonObject(False)):
        slot[].state = 2
        return 2
    if py_type(base) is py_type(PythonObject("")):
        raise Error(String(base))
    var factory = neural_descriptor_entry(base, LABELS_NATIVE_FACTORY)
    var provider = neural_descriptor_connect_factory_once(factory, String(LABELS_NATIVE_ABI_NAME), LABELS_NATIVE_OPERATION_COUNT)
    var driver = LabelsNativeDriver(provider.downcast_value_ptr[NeuralDescriptorProvider]()[].entry)
    var mode = driver.metadata(LABELS_OP_MODE)
    if mode != 0 and mode != 1:
        raise Error("kit provider: unsupported compiled numeric mode")
    if driver.metadata(LABELS_OP_CAPABILITIES) != LABELS_CAP_DEVICE:
        raise Error("ImportError: mojolearn: the base binding's labels provider is not the device provider")
    slot[].provider = provider
    slot[].driver = Optional[LabelsNativeDriver](driver^)
    slot[].state = 1
    return 1


@fieldwise_init
struct _LabelsPin(Movable):
    var provider: PythonObject
    var driver: Optional[LabelsNativeDriver]


def _labels_pin(context: PythonObject) raises -> _LabelsPin:
    _ = _labels_route(context)
    var slot = _LABELS_PROVIDER.get_or_create_ptr()
    return _LabelsPin(slot[].provider, slot[].driver.copy())


def labels_prepare_targets(context: PythonObject, y_addr: Int, y_code: Int, n: Int,
    classes_addr: Int, codes_addr: Int, w_addr: Int, w_code: Int, w_flags: Int) raises -> Int:
    var request = LabelsPrepareRequest(y_addr, classes_addr, codes_addr, w_addr, y_code, n, w_code, w_flags)
    var pin = _labels_pin(context)
    var driver = pin.driver.copy()
    try:
        with GILReleased(Python()):
            return driver.value().prepare(request) if driver else LabelsHostKernels.prepare(request)
    finally:
        _ = pin


def labels_reduce_stat(context: PythonObject, addr: Int, code: Int, n: Int, what: Int) raises -> HpStat:
    var request = LabelsReduceRequest(addr, code, n, what)
    var pin = _labels_pin(context)
    var driver = pin.driver.copy()
    try:
        with GILReleased(Python()):
            return driver.value().reduce(request) if driver else LabelsHostKernels.reduce(request)
    finally:
        _ = pin


def labels_argmax_rows(context: PythonObject, wide: Bool, scores_addr: Int, rows: Int, cols: Int, dst_addr: Int) raises:
    var request = LabelsArgmaxRequest(scores_addr, dst_addr, rows, cols)
    var pin = _labels_pin(context)
    var driver = pin.driver.copy()
    try:
        with GILReleased(Python()):
            if driver:
                driver.value().argmax(wide, request)
            else:
                LabelsHostKernels.argmax(wide, request)
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
            if driver:
                driver.value().gather(native_kind, request)
            else:
                LabelsHostKernels.gather(native_kind, request)
    finally:
        _ = pin


def labels_class_ratio_f64(context: PythonObject, counts_addr: Int, k: Int, n: Int,
    mode: Int, out_addr: Int) raises -> Int:
    var request = LabelsRatioRequest(counts_addr, out_addr, k, n, mode)
    var pin = _labels_pin(context)
    var driver = pin.driver.copy()
    try:
        with GILReleased(Python()):
            return driver.value().ratio(request) if driver else LabelsHostKernels.ratio(request)
    finally:
        _ = pin


def labels_fsum_f64(src_addr: Int, n: Int, out_addr: Int) raises -> Int:
    with GILReleased(Python()):
        return labels_host_fsum(src_addr, n, out_addr)
