# SPDX-License-Identifier: Apache-2.0
"""The decomposition estimators' device route, as native words
(core/device_route_dw_decomp.mojo): the device a request runs on, which image
table a route reads, the vendor word its compiled image must report, and the
two refusals. Typed Mojo only; no Python module participates.

The rule and its words are the Ridge pilot's (core/linear_route.mojo); this
suite holds what the decomposition families add on top of it. The cases are
a handful of scalar comparisons with no independent numerical work to split.
"""
from core.linear_route import LINEAR_DEVICE_CPU, LINEAR_DEVICE_GPU, LINEAR_DEVICE_INSTALL
from core.device_route_dw_decomp import (
    DecompRoute, decomp_route_device, decomp_route_cpu_image, decomp_route_vendor,
    decomp_route_cpu_identical, decomp_route_cpu_absent,
)


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def _gpu_vendor(index: Int) -> String:
    if index == 0:
        return String("metal")
    if index == 1:
        return String("cuda")
    return String("hip")


def route_defaults() raises:
    var empty = DecompRoute()
    require(empty.connection == -1 and empty.extra == -1 and empty.mode == -1, "an empty route is not unresolved")
    var route = DecompRoute(LINEAR_DEVICE_CPU, 3, 1)
    require(route.device == LINEAR_DEVICE_CPU and route.connection == 3 and route.mode == 1 and route.extra == -1,
        "a route does not hold its words")
    var paired = DecompRoute(LINEAR_DEVICE_GPU, 2, 0, 5)
    require(paired.extra == 5 and paired.mode == 0, "a route does not hold its second connection")


def device_rules() raises:
    # None is the installation's device; cpu always runs; an explicit gpu
    # without a GPU set is refused in words that name the CPU.
    require(decomp_route_device(LINEAR_DEVICE_INSTALL, "cpu") == LINEAR_DEVICE_CPU,
        "the default device of a CPU-only process is not the CPU")
    require(decomp_route_device(LINEAR_DEVICE_CPU, "cpu") == LINEAR_DEVICE_CPU, "cpu did not run on a CPU-only process")
    require(not decomp_route_cpu_image(LINEAR_DEVICE_CPU, "cpu"),
        "a CPU-only process read a second CPU table (its own table already is the CPU one)")
    require(decomp_route_vendor(LINEAR_DEVICE_CPU, "cpu") == "cpu", "the CPU route of a CPU-only process is not cpu")
    for index in range(3):  # the three GPU vendor words
        var vendor = _gpu_vendor(index)
        require(decomp_route_device(LINEAR_DEVICE_INSTALL, vendor) == LINEAR_DEVICE_GPU,
            "the default device of a GPU process is not the GPU")
        require(decomp_route_device(LINEAR_DEVICE_GPU, vendor) == LINEAR_DEVICE_GPU
            and decomp_route_device(LINEAR_DEVICE_CPU, vendor) == LINEAR_DEVICE_CPU,
            "an explicit device did not keep its word on a GPU process")
        require(decomp_route_cpu_image(LINEAR_DEVICE_CPU, vendor) and not decomp_route_cpu_image(LINEAR_DEVICE_GPU, vendor),
            "the CPU image table is not the cpu route's alone on a GPU process")
        require(decomp_route_vendor(LINEAR_DEVICE_CPU, vendor) == "cpu" and decomp_route_vendor(LINEAR_DEVICE_GPU, vendor) == vendor,
            "a route's expected vendor word is not its device's")
    var refused = False
    try:
        _ = decomp_route_device(LINEAR_DEVICE_GPU, "cpu")
    except problem:
        require(String(problem).find("device='cpu'") >= 0, "the gpu refusal does not name device='cpu'")
        refused = True
    require(refused, "an explicit gpu ran on a CPU-only process")


def refusals() raises:
    decomp_route_cpu_identical(True)
    var refused = False
    try:
        decomp_route_cpu_identical(False)
    except problem:
        require(String(problem).find("identical mode only") >= 0, "the fast-on-cpu refusal lost its words")
        refused = True
    require(refused, "numeric_mode='fast' ran on the CPU image")
    refused = False
    try:
        decomp_route_cpu_absent("KernelPCA")
    except problem:
        require(String(problem).find("KernelPCA") >= 0 and String(problem).find("device='gpu'") >= 0,
            "the no-CPU-implementation refusal does not name the estimator and device='gpu'")
        refused = True
    require(refused, "a family with no CPU implementation was not refused")


def run_checks() raises:
    route_defaults()
    device_rules()
    refusals()
    print("PASS decomposition device route rules")
