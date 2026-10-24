# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""One process-lifetime DeviceContext per physical device for a binding
whose root follows a native multi-GPU driver's device plan
(core/device_plan.mojo; docs/plans/NATIVE_PARALLEL_DRIVER_20261024.md).
Kept out of core/neural_context.mojo so that shared module and its 31
bindings do not rebuild for one family's driver."""
from std.ffi import _Global
from max.gpu.host import DeviceContext

from core.neural_context import process_ctx


struct _DeviceContexts(Defaultable, Movable):
    var ids: List[Int]
    var contexts: List[DeviceContext]

    def __init__(out self):
        self.ids = List[Int]()
        self.contexts = List[DeviceContext]()


def process_ctx_on[name: StaticString, devices_name: StaticString](device_id: Int) raises -> DeviceContext:
    """The binding's process-lifetime context on physical device `device_id`
    (lane/native-parallel-driver-20261024). Device 0 is the binding's own
    slot (`process_ctx[name]`, `DeviceContext()` opens device 0); any other
    device gets one context of its own in slot `devices_name`, created on
    first use and kept for the process, so a native multi-GPU driver whose
    root is not device 0 (`core/device_plan.mojo::device_plan_root`) never
    creates and destroys a root context per call. Same rules as
    `process_ctx`: a device path only, every entry still synchronizes."""
    if device_id < 0:
        raise Error("process_ctx_on: negative device index " + String(device_id))
    if device_id == 0:
        return process_ctx[name]()
    comptime SLOT = _Global[StorageType=_DeviceContexts, name=devices_name, init_fn=_DeviceContexts.__init__]
    var slot = SLOT.get_or_create_ptr()
    for i in range(len(slot[].ids)):  # driver-loop: over the devices this binding has opened
        if slot[].ids[i] == device_id:
            return slot[].contexts[i].copy()
    var ctx = DeviceContext(device_id=device_id)
    slot[].contexts.append(ctx.copy())
    slot[].ids.append(device_id)
    return ctx^
