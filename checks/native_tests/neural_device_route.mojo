# SPDX-License-Identifier: Apache-2.0
"""The neural model classes' device rule (core/device_route_dw_neural_models.mojo).

Host column: a metadata rule with no numerical work to split; one case.
"""
from checks.native_tests.runner import Check, Column, Report
from core.device_route_dw_neural_models import (
    NEURAL_ROUTE_CPU, NEURAL_ROUTE_GPU, neural_device_route, neural_route_name,
    neural_device_cpu_training_refusal, neural_device_no_cpu_refusal,
)
from core.linear_route import LINEAR_DEVICE_CPU, LINEAR_DEVICE_GPU, LINEAR_DEVICE_INSTALL


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def refused_with(device: Int, cpu_class: Bool, process_vendor: Int, fast: Bool, words: String) raises -> Bool:
    try:
        _ = neural_device_route(device, cpu_class, process_vendor, fast)
    except problem:
        require(String(problem).find(words) >= 0, "a device refusal lost its words: " + String(problem))
        return True
    return False


def run_checks() raises:
    # None is the installation's device.
    require(neural_device_route(LINEAR_DEVICE_INSTALL, False, 0, False) == NEURAL_ROUTE_CPU,
        "the default device of a CPU-only process is not the CPU")
    for vendor in range(1, 4):  # the three GPU vendor words
        require(neural_device_route(LINEAR_DEVICE_INSTALL, False, vendor, False) == NEURAL_ROUTE_GPU,
            "the default device of a GPU process is not the GPU")
        require(neural_device_route(LINEAR_DEVICE_GPU, False, vendor, False) == NEURAL_ROUTE_GPU
            and neural_device_route(LINEAR_DEVICE_CPU, False, vendor, False) == NEURAL_ROUTE_CPU,
            "an explicit device did not keep its word on a GPU process")
        # A fast mode named beside the default or the GPU device is not this rule's business.
        require(neural_device_route(LINEAR_DEVICE_INSTALL, False, vendor, True) == NEURAL_ROUTE_GPU
            and neural_device_route(LINEAR_DEVICE_GPU, False, vendor, True) == NEURAL_ROUTE_GPU,
            "a fast mode changed the GPU route")
        require(refused_with(LINEAR_DEVICE_CPU, False, vendor, True, "identical mode only"),
            "device='cpu' beside an explicit fast mode ran on a GPU process")
        # The CPU side by class runs on the CPU whatever value it stored.
        require(neural_device_route(LINEAR_DEVICE_GPU, True, vendor, False) == NEURAL_ROUTE_CPU
            and neural_device_route(LINEAR_DEVICE_INSTALL, True, vendor, True) == NEURAL_ROUTE_CPU,
            "a CPU class left the CPU")
    require(neural_device_route(LINEAR_DEVICE_CPU, False, 0, False) == NEURAL_ROUTE_CPU,
        "cpu did not run on a CPU-only process")
    # A CPU-only process runs its one set whatever the mode word (as before the rule).
    require(neural_device_route(LINEAR_DEVICE_CPU, False, 0, True) == NEURAL_ROUTE_CPU,
        "a CPU-only process refused its own device")
    require(refused_with(LINEAR_DEVICE_GPU, False, 0, False, "device='cpu'"),
        "an explicit gpu ran on a CPU-only process")
    require(neural_route_name(NEURAL_ROUTE_CPU) == "cpu" and neural_route_name(NEURAL_ROUTE_GPU) == "gpu",
        "route names are not the route words")
    var said = 0
    try:
        neural_device_cpu_training_refusal("neural block backward")
    except problem:
        require(String(problem).startswith("NotImplementedError: ") and String(problem).find("device='gpu'") >= 0,
            "the training refusal does not name device='gpu'")
        said += 1
    try:
        neural_device_no_cpu_refusal("SambaStack")
    except problem:
        require(String(problem).startswith("NotImplementedError: ") and String(problem).find("device='gpu'") >= 0,
            "the no-CPU refusal does not name device='gpu'")
        said += 1
    require(said == 2, "a refusal did not raise")


def run_cases(column: Column, mut report: Report):
    try:
        run_checks()
    except error:
        report.abort(error)


def register() -> Check:
    return Check("neural_device_route", cpu=True, gpu=False, serial="", run=run_cases)
