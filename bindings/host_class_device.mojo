# SPDX-License-Identifier: Apache-2.0
"""The `device=` of a class whose only implementation is the host one (lane
vdf-parallel, 2026-11-24): the forward-only inference classes
(`LanguageModelInference`, `MLPInference`, `SambaInference`) run compiled
CPU code in every installation. The ONE public-function rule
(core/function_device.mojo) with HAS_GPU=False, HAS_CPU=True:

* None and "cpu" are today's route (the host code; nothing else is read);
* "gpu" is refused in the shared `function_device_gpu_unimplemented` words
  naming the class: its host route never runs in its place.

The installation's CPU-only record is not read: with no GPU implementation
the answer is the same on every installation. One read per construction,
never per call, row or token.
"""
from std.python import PythonObject
from core.device_route import DEVICE_INSTALL
from core.function_device import FunctionDevice, function_device_decide, function_device_default
from providers.python.device_route_boundary_door import device_unpack


def host_class_device(value: PythonObject, name: String, b: PythonObject) raises -> FunctionDevice:
    """The admitted `device` of a host-only class's constructor: `value` is
    the caller's word, `name` the public class (it leads the refusal words),
    `b` the builtins module the door already holds."""
    var request = device_unpack(value, b)
    if request == DEVICE_INSTALL:
        return function_device_default()
    # cpu_only=False: for a class with no GPU implementation the rule answers
    # 'gpu' before it reads it, and 'cpu' runs the host code either way.
    return function_device_decide(request, False, False, True, name)
