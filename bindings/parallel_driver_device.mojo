# SPDX-License-Identifier: Apache-2.0
"""The `device=` of a multi-GPU driver (lane vdf-parallel, 2026-11-24): the
family-local use of the ONE public-function rule (core/function_device.mojo)
by the `parallel_*` drivers and their driver classes.

A driver spreads one fit or query across the GPUs in `devices=` and has no
CPU form: HAS_GPU=True, HAS_CPU=False.

* None is today's route: the door's `device_unpack` answers DEVICE_INSTALL
  on one identity compare and nothing else is read (`parallel_driver_decide`
  is not called).
* "gpu" is today's route on a GPU installation; on a CPU-only one it is
  refused in the estimators' `device_gpu_absent` words.
* "cpu" beside a GPU set is refused in the shared `device_cpu_absent` words
  with the drivers' one sentence (bindings/msel_member_device_unpack.mojo
  `member_multi_gpu_refusal`); on a CPU-only installation it is the
  installation's device, today's route.

A driver handed an estimator applies `function_device_member` to it after
today's member refusal (a member that says 'cpu' beside a GPU set): a member
constructed with an explicit device that differs from the driver's explicit
one is refused before any partition or worker. Each read is once per public
driver call, never per shard, row or device.
"""
from std.python import PythonObject
from bindings.msel_member_device_unpack import member_multi_gpu_refusal
from bindings.py_balanced import py_type
from core.device_route import DEVICE_CPU, DEVICE_INSTALL
from core.function_device import FunctionDevice, function_device_decide, function_device_default, function_device_member
from providers.python.device_route_boundary_door import device_request

# The drivers' implementations: GPU kernels on the GPUs in `devices=`, no CPU form.
comptime PARALLEL_DRIVER_HAS_GPU = True
comptime PARALLEL_DRIVER_HAS_CPU = False


def parallel_driver_decide(request: Int, driver: String, cpu_only: Bool) raises -> FunctionDevice:
    """The device of one driver call from the admitted code of its `device`
    word (`device_unpack`). The door calls it only for a code that is not
    DEVICE_INSTALL, so `cpu_only` (the door's own record of a CPU-only
    installation) is read only for an explicit word."""
    if request == DEVICE_INSTALL:
        return function_device_default()
    if request == DEVICE_CPU and not cpu_only:
        member_multi_gpu_refusal(driver)
    return function_device_decide(request, cpu_only, PARALLEL_DRIVER_HAS_GPU, PARALLEL_DRIVER_HAS_CPU,
        "parallel " + driver)


def parallel_driver_member(device: FunctionDevice, model: PythonObject, fields: PythonObject, driver: String,
    b: PythonObject) raises:
    """The member rule of a driver handed an estimator `model` (instance
    dictionary `fields`): an explicit driver device that differs from the
    member's own explicit device is refused in the shared words. `device=None`
    on the driver is one compare and reads nothing."""
    if device.request == DEVICE_INSTALL:
        return
    var member = device_request(fields, b)
    if member == DEVICE_INSTALL or member == device.request:
        return
    _ = function_device_member(device.request, member, "parallel " + driver, String(py_type(model).__name__))
