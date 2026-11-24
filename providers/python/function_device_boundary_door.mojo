# SPDX-License-Identifier: Apache-2.0
"""UNPACK of a public function's `device=` into its typed native value: the ONE
admission every public function that computes shares (lane vd-functions,
2026-11-24). The rule and its words are core/function_device.mojo; the read
of the word is the estimators' own (`device_unpack`,
providers/python/device_route_boundary_door.mojo): exact `None`, `'gpu'` or
`'cpu'`, anything else refused in the shared words before any work.

A door calls `function_device_admit` ONCE per public call, in its UNPACK, and
hands the `FunctionDevice` to the rest of its work:

* the compiled module of a binding name for that device:
  `estimator_binding_device` (providers/python/estimator_binding_pyapi_door.mojo,
  beside `estimator_binding_mode`, which it calls unchanged for every value
  but an explicit "cpu" beside a GPU set);
* the context its metric programs and label steps run in:
  `function_device_scope` (the family's CPU label scope for an explicit
  "cpu" beside a GPU set, else the context itself, unchanged).

`device=None` costs one identity compare: nothing else is read, and the
door's route is the one it ran before the parameter existed. The other
values read one more dictionary slot (`_CPU_ONLY`), once per call.
"""
from std.python import PythonObject
from bindings.py_balanced import py_get
from bindings.labels_descriptor_boundary_door import labels_cpu_scope
from core.device_route import DEVICE_INSTALL
from core.function_device import FunctionDevice, function_device_decide, function_device_default
from providers.python.device_route_boundary_door import device_unpack


def function_cpu_only(context: PythonObject) raises -> Bool:
    """True on a CPU-only installation (the backend's `_CPU_ONLY` record, the
    one every family door reads)."""
    var backend = context["backend_fields"]
    var record = py_get(backend, "_CPU_ONLY")
    return record is not None


def function_device_admit(value: PythonObject, name: String, has_gpu: Bool, has_cpu: Bool,
    context: PythonObject, b: PythonObject) raises -> FunctionDevice:
    """The device of one public function call. `value` is the caller's
    `device` argument; `name` is the public function's name (it leads the
    refusal words); `has_gpu` / `has_cpu` are the door's constants for which
    implementations the function has; `b` is the builtins module the door
    already admitted."""
    var request = device_unpack(value, b)
    if request == DEVICE_INSTALL:
        return function_device_default()
    return function_device_decide(request, function_cpu_only(context), has_gpu, has_cpu, name)


def function_device_scope(device: FunctionDevice, context: PythonObject) raises -> PythonObject:
    """The context a function's metric programs and label steps run in:
    `context` itself, or for an explicit "cpu" beside a GPU set the family's
    CPU label scope (the host label kernels and the CPU x_metrics runner
    entries; bindings/labels_descriptor_boundary_door.mojo
    `labels_cpu_scope`). A CPU scope that cannot be connected is refused in
    words: the call never runs the GPU images in its place."""
    if not device.host_beside:
        return context
    var scope = labels_cpu_scope(context)
    if scope is None:
        raise Error("ImportError: device='cpu' needs the CPU identical images beside the GPU set, "
            + "which are not connected in this installation")
    return scope
