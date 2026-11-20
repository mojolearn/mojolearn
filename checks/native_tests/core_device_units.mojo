# SPDX-License-Identifier: Apache-2.0
"""The multiprocessor-count readers answer on every device column without raising.

Metal refuses `DeviceAttribute.MULTIPROCESSOR_COUNT` ("not supported for Apple
GPU"); before fx-user-crash-scorer (2026-11-19) every GBDT fit that sized a
grid from it raised on Apple. Every such site now reads
`core.device_units.device_multiprocessor_count` (gbdt/train.mojo,
gbdt/gpu_util/*, gbdt/methods/**, the partition checks), and the attention
tile rule reads `attn_device_units`, which answers 0 on Apple at comptime.
Cases:
  0  `device_multiprocessor_count` returns without raising and is positive;
     on an Apple build it is the Apple column's core row.
  1  it answers the same value twice (the GBDT callers thread one read
     through a fit; a second read must agree).
  2  `attn_device_units` returns without raising: 0 on Apple, positive on
     NVIDIA and AMD.
The device column only: these read a GPU device context (the host column
never sizes a grid from a device count).
"""
from max.gpu.host import DeviceContext
from core.accelerator_target import TARGET_HAS_APPLE_GPU
from core.device_units import device_multiprocessor_count
from checks.hardware_matrix import gpu_cores_for
from checks.kernel_matrix import TARGET_COLUMN
from transformer.impl.llama.fused_attention import attn_device_units
from checks.native_tests.runner import Check, Column, Report, require


comptime DEVICE_UNITS_CASES = 3


def run_case(ctx: DeviceContext, index: Int) raises:
    var label = "device_units case " + String(index)
    if index == 0:
        var units = device_multiprocessor_count(ctx)
        require(units > 0, label + ": multiprocessor count " + String(units) + " is not positive")
        comptime if TARGET_HAS_APPLE_GPU:
            require(
                units == gpu_cores_for[TARGET_COLUMN](),
                label + ": Apple count " + String(units) + " is not the Apple column's core row",
            )
    elif index == 1:
        var first = device_multiprocessor_count(ctx)
        var second = device_multiprocessor_count(ctx)
        require(first == second, label + ": two reads disagree " + String(first) + " vs " + String(second))
    else:
        var units = attn_device_units(ctx)
        comptime if TARGET_HAS_APPLE_GPU:
            require(units == 0, label + ": attention units on Apple " + String(units) + ", expected 0")
        else:
            require(units > 0, label + ": attention units " + String(units) + " is not positive")


# One device context; the cases are submitted in order on it.
def _run_gpu(mut report: Report) raises:
    var cases = report.cases(DEVICE_UNITS_CASES)
    var ctx = DeviceContext()
    for unit in range(DEVICE_UNITS_CASES):  # driver-loop: over the ordered shared-context cases
        try:
            run_case(ctx, unit)
        except error:
            cases.fail(unit, error)
    _ = ctx


def run_cases(column: Column, mut report: Report):
    try:
        _run_gpu(report)
    except error:
        report.abort(error)


def register() -> Check:
    return Check("core_device_units", cpu=False, gpu=True, serial="one device context, cases in order",
                 run=run_cases)
