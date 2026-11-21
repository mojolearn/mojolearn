# SPDX-License-Identifier: Apache-2.0
"""The device rule of the binding-selecting families as native words
(core/device_route_dw_neighbors_cluster.mojo): what a `device` request
resolves to in a CPU-only process and beside a GPU set, when an estimator
runs the CPU images beside the device images (`host_table`), and the two
refusals (an explicit gpu without a GPU set, in the pilot's words; cpu beside
a GPU set in fast mode).

Not split across cores: a dozen constant-time rule evaluations with no
numerical work; the check counts as one case. No Python module participates.
"""
from checks.native_tests.runner import Check, Column, Report
from core.linear_route import LINEAR_DEVICE_CPU, LINEAR_DEVICE_GPU, LINEAR_DEVICE_INSTALL
from core.device_route_dw_neighbors_cluster import (
    FAMILY_ROUTE_SLOT_COUNT, family_route_binding_name, family_route_plan, family_route_unresolved,
    family_route_vendor_is_cpu,
)


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def run_checks() raises:
    # A CPU-only process: every request that runs resolves to the process's
    # own (CPU) table, never to a table beside it.
    var alone = family_route_plan(LINEAR_DEVICE_INSTALL, True, 1)
    require(alone.device == LINEAR_DEVICE_CPU and not alone.host_table and alone.requested == LINEAR_DEVICE_INSTALL,
        "the default device of a CPU-only process is not its own CPU table")
    var alone_cpu = family_route_plan(LINEAR_DEVICE_CPU, True, 1)
    require(alone_cpu.device == LINEAR_DEVICE_CPU and not alone_cpu.host_table,
        "cpu on a CPU-only process is not its own CPU table")
    require(family_route_vendor_is_cpu(alone) and family_route_vendor_is_cpu(alone_cpu),
        "a CPU-only route does not report the CPU")
    var refused = False
    try:
        _ = family_route_plan(LINEAR_DEVICE_GPU, True, 1)
    except problem:
        require(String(problem).find("device='cpu'") >= 0, "the gpu refusal does not name device='cpu'")
        refused = True
    require(refused, "an explicit gpu ran on a CPU-only process")
    # Beside a GPU set: None and gpu run the device images in either mode;
    # cpu runs the CPU images beside them, in identical mode only.
    for mode in range(2):  # fast, identical
        var installed = family_route_plan(LINEAR_DEVICE_INSTALL, False, mode)
        var gpu = family_route_plan(LINEAR_DEVICE_GPU, False, mode)
        require(installed.device == LINEAR_DEVICE_GPU and not installed.host_table and installed.mode == mode,
            "the default device of a GPU process is not the GPU")
        require(gpu.device == LINEAR_DEVICE_GPU and not gpu.host_table and gpu.requested == LINEAR_DEVICE_GPU,
            "an explicit gpu did not keep its word on a GPU process")
        require(not family_route_vendor_is_cpu(installed) and not family_route_vendor_is_cpu(gpu),
            "a device route reports the CPU")
    var beside = family_route_plan(LINEAR_DEVICE_CPU, False, 1)
    require(beside.device == LINEAR_DEVICE_CPU and beside.host_table and family_route_vendor_is_cpu(beside),
        "cpu beside a GPU set does not run the CPU images")
    var fast_refused = False
    try:
        _ = family_route_plan(LINEAR_DEVICE_CPU, False, 0)
    except problem:
        require(String(problem).find("identical mode only") >= 0, "the cpu fast refusal lost its words")
        fast_refused = True
    require(fast_refused, "cpu beside a GPU set ran in fast mode")
    # An unresolved route names no device; every slot has a binding name.
    var empty = family_route_unresolved()
    require(empty.device < 0 and not empty.host_table, "an unresolved route names a device")
    for slot in range(FAMILY_ROUTE_SLOT_COUNT):  # the fixed binding slots
        require(family_route_binding_name(slot).byte_length() > 0, "a binding slot has no name")
        for other in range(slot):  # earlier slots: names are distinct
            require(family_route_binding_name(slot) != family_route_binding_name(other), "two slots share a binding")
    require(family_route_binding_name(FAMILY_ROUTE_SLOT_COUNT).byte_length() == 0, "a slot past the table has a name")
    print("PASS family device route rules")


def run_cases(column: Column, mut report: Report):
    try:
        run_checks()
    except error:
        report.abort(error)


def register() -> Check:
    return Check("device_route_dw_neighbors_cluster", cpu=True, gpu=False, serial="", run=run_cases)
