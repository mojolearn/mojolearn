# SPDX-License-Identifier: Apache-2.0
"""The device rule of the neural model classes, as native words.

`device=None | "gpu" | "cpu"` on a block, stack or model constructor uses the
rule and the words of the Ridge pilot (core/linear_route.mojo): None is this
installation's device, an explicit `gpu` without a GPU set is refused, an
unknown value is refused. This file adds what is specific to the neural
classes:

- a class that is the CPU side by class (the `*Inference` classes) runs on
  the CPU whatever value its inherited constructor stored;
- `device="cpu"` in a GPU installation runs the identical host kernels only,
  so an explicit `numeric_mode="fast"` beside it is refused;
- where a GPU installation holds no host implementation of an operation
  (training: the host training images cannot be loaded beside their GPU
  siblings in one process), the operation is refused in the words below.

The rule is a metadata decision with no independent numerical work to
split; it runs at construction and at an owner record's miss, never per row.
"""
from core.linear_route import LINEAR_DEVICE_CPU, linear_device_installed

comptime NEURAL_ROUTE_GPU = 0
comptime NEURAL_ROUTE_CPU = 1


def neural_device_route(device: Int, cpu_class: Bool, process_vendor: Int, fast: Bool) raises -> Int:
    """The route a request runs on. `device` is a core/linear_route.mojo
    device code, `process_vendor` is 0 in a CPU-only process, `fast` is
    whether the constructor named `numeric_mode="fast"` explicitly."""
    if cpu_class:
        return NEURAL_ROUTE_CPU
    var resolved = linear_device_installed(device, process_vendor)
    if resolved != LINEAR_DEVICE_CPU:
        return NEURAL_ROUTE_GPU
    if process_vendor != 0 and fast:
        raise Error("ValueError: device='cpu' runs in identical mode only; numeric_mode='fast' is for Apple GPUs")
    return NEURAL_ROUTE_CPU


def neural_route_name(route: Int) -> String:
    return String("cpu") if route == NEURAL_ROUTE_CPU else String("gpu")


def neural_device_cpu_training_refusal(what: String) raises:
    """The one refusal of a training operation on `device="cpu"` in a GPU
    installation. `what` names the class and operation."""
    raise Error("NotImplementedError: mojolearn " + what + ": device='cpu' runs inference only in a GPU "
        + "installation; train with device='gpu' (or device=None)")


def neural_device_no_cpu_refusal(what: String) raises:
    """The one refusal of `device="cpu"` for a class with no host (CPU)
    implementation in this installation."""
    raise Error("NotImplementedError: mojolearn " + what + " has no CPU implementation; "
        + "use device='gpu' (or device=None)")
