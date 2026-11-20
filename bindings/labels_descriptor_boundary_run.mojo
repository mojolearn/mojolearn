# SPDX-License-Identifier: Apache-2.0
"""Typed label route and dispatch (RUN): the process route word (0 unconnected,
1 device, 2 host), the connected device driver, and the device-or-host call of
each label kernel. The door (`bindings/labels_descriptor_boundary_door.mojo`)
admits the provider image, keeps its pin and releases the GIL around these
calls."""
from core.shared_slot import SharedSlot
from core.hotpath_impl import HpStat
from core.labels_native_api import (
    LABELS_OP_MODE, LABELS_OP_CAPABILITIES, LABELS_CAP_DEVICE,
    LabelsNativeDriver, LabelsPrepareRequest, LabelsReduceRequest,
    LabelsArgmaxRequest, LabelsGatherRequest, LabelsRatioRequest, LabelsThresholdRequest,
)
from core.labels_native_host import LabelsHostKernels


struct _LabelsRoute(Defaultable, Movable):
    var state: Int
    var driver: Optional[LabelsNativeDriver]

    def __init__(out self):
        self.state = 0
        self.driver = Optional[LabelsNativeDriver]()


# V5: the provider pin moved to the door (MojolearnLabelsProviderPinV1).
comptime _LABELS_ROUTE = SharedSlot[_LabelsRoute, "MojolearnLabelsProviderV5", 5]


def labels_route_state() raises -> Int:
    return _LABELS_ROUTE.get_or_create_ptr()[].state


def labels_route_host() raises:
    _LABELS_ROUTE.get_or_create_ptr()[].state = 2


def labels_route_check(driver: LabelsNativeDriver) raises:
    var mode = driver.metadata(LABELS_OP_MODE)
    if mode != 0 and mode != 1:
        raise Error("kit provider: unsupported compiled numeric mode")
    if driver.metadata(LABELS_OP_CAPABILITIES) != LABELS_CAP_DEVICE:
        raise Error("ImportError: mojolearn: the base binding's labels provider is not the device provider")


def labels_route_device(var driver: LabelsNativeDriver) raises:
    var slot = _LABELS_ROUTE.get_or_create_ptr()
    slot[].driver = Optional[LabelsNativeDriver](driver^)
    slot[].state = 1


def labels_route_driver() raises -> Optional[LabelsNativeDriver]:
    return _LABELS_ROUTE.get_or_create_ptr()[].driver.copy()


def labels_prepare_run(var driver: Optional[LabelsNativeDriver], request: LabelsPrepareRequest) raises -> Int:
    return driver.value().prepare(request) if driver else LabelsHostKernels.prepare(request)


def labels_reduce_run(var driver: Optional[LabelsNativeDriver], request: LabelsReduceRequest) raises -> HpStat:
    return driver.value().reduce(request) if driver else LabelsHostKernels.reduce(request)


def labels_threshold_run(var driver: Optional[LabelsNativeDriver], request: LabelsThresholdRequest) raises -> Int:
    return driver.value().threshold(request) if driver else LabelsHostKernels.threshold(request)


def labels_argmax_run(var driver: Optional[LabelsNativeDriver], wide: Bool, request: LabelsArgmaxRequest) raises:
    if driver:
        driver.value().argmax(wide, request)
    else:
        LabelsHostKernels.argmax(wide, request)


def labels_gather_run(var driver: Optional[LabelsNativeDriver], kind: Int, request: LabelsGatherRequest) raises:
    if driver:
        driver.value().gather(kind, request)
    else:
        LabelsHostKernels.gather(kind, request)


def labels_ratio_run(var driver: Optional[LabelsNativeDriver], request: LabelsRatioRequest) raises -> Int:
    return driver.value().ratio(request) if driver else LabelsHostKernels.ratio(request)
