# SPDX-License-Identifier: Apache-2.0
"""The device route of one decomposition estimator, as native words.

`device=None` (this installation's device), `device="gpu"` and `device="cpu"`
follow the one rule and the one wording of core/linear_route.mojo (the Ridge
pilot): this file adds no rule of its own. It holds what the decomposition
families share on top of that rule: the resolved route (the device and the
family's typed connection index, x_decomp/*_descriptor_boundary_run.mojo) and
the three questions a family's resolution asks once per estimator.

A decomposition family reaches its compiled image through a typed connection
record (the driver, the compiled mode, vendor and capabilities of one image).
`device="cpu"` on a GPU installation is the connection of the CPU image of the
same family (the host binary's descriptor factory, loaded beside the device
one at import). A route is resolved ONCE per estimator
(bindings/decomp_device_route_door.mojo) and every later operation calls
through its connection: no public operation tests the device, reads a binding
table or compares a vendor word.

Resolution is a metadata operation with no independent numerical work to
split; it runs at an estimator's first operation, never per row.

SHARED CHANGE WANTED: the device rule (`linear_device_*`, `LINEAR_DEVICE_*`)
is family neutral and lives under a linear name; move it to
core/device_route.mojo and have this file and core/linear_route.mojo import it.
"""
from core.linear_metadata import linear_vendor_code
from core.linear_route import LINEAR_DEVICE_CPU, LINEAR_DEVICE_GPU, linear_device_installed


struct DecompRoute(ImplicitlyCopyable, Movable):
    """One estimator's resolved route. `device` is LINEAR_DEVICE_GPU or
    LINEAR_DEVICE_CPU (never the request None); `connection` is the family's
    typed connection index; `mode` is the tier the estimator selected (1
    identical, 0 fast; the connection's image reports the same); `extra` is a
    second connection of the same family where it has one (-1 otherwise)."""
    var device: Int
    var connection: Int
    var mode: Int
    var extra: Int

    def __init__(out self):
        self.device = LINEAR_DEVICE_GPU
        self.connection = -1
        self.mode = -1
        self.extra = -1

    def __init__(out self, device: Int, connection: Int, mode: Int, extra: Int = -1):
        self.device = device
        self.connection = connection
        self.mode = mode
        self.extra = extra


def decomp_route_device(request: Int, process_vendor: String) raises -> Int:
    """The device a request runs on in a process whose selected vendor word
    is `process_vendor` (`cpu`: a CPU-only installation). None is the
    installation's device; an explicit `gpu` without a GPU set is refused in
    the pilot's words; `cpu` always runs."""
    return linear_device_installed(request, linear_vendor_code(process_vendor))


def decomp_route_cpu_image(device: Int, process_vendor: String) -> Bool:
    """True when the route runs the CPU image loaded BESIDE the device set (a
    `cpu` route of a GPU installation). In a CPU-only process the family's own
    table already is the CPU one."""
    return device == LINEAR_DEVICE_CPU and linear_vendor_code(process_vendor) != 0


def decomp_route_vendor(device: Int, process_vendor: String) -> String:
    """The vendor word the compiled image of a route must report."""
    if device == LINEAR_DEVICE_CPU:
        return String("cpu")
    return process_vendor


def decomp_route_cpu_identical(identical: Bool) raises:
    """The CPU image is the identical tier only (the pilot's words)."""
    if not identical:
        raise Error("ValueError: device='cpu' runs in identical mode only; numeric_mode='fast' is for Apple GPUs")


def decomp_route_cpu_absent(estimator: String) raises:
    """The one refusal of `device="cpu"` for a family with no CPU
    implementation of the operation."""
    raise Error("ValueError: device='cpu' is not available for " + estimator
        + ": it has no CPU implementation. Use device='gpu', or leave device=None")
