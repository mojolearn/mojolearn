# SPDX-License-Identifier: Apache-2.0
"""The device route of one tree-estimator operation, as native words (RUN: no Python).

The boundary (bindings/trees_route_boundary_door.mojo) reads the estimator's
`device` and its fitted-table-set marker once where a public operation
starts and hands them here as words. `trees_route_plan` answers which table
set the operation runs on, whether its end records the table set (a fit),
and which refusal applies. The device words and their rule are the pilot's
(core/linear_route.mojo): None is this installation's device, an explicit
`gpu` without a GPU set is refused, `cpu` always runs.

One plan per public operation: a metadata decision with no independent
numerical work to split; never per row.
"""
from core.linear_route import LINEAR_DEVICE_GPU, LINEAR_DEVICE_CPU

# The table set that fitted the estimator's state (its marker).
comptime TREES_FITTED_UNRECORDED = 0   # no fit recorded (never fitted, or built without a fit)
comptime TREES_FITTED_INSTALL = 1      # this installation's table set
comptime TREES_FITTED_CPU = 2          # the CPU table set of a GPU installation

comptime TREES_ROUTE_RUNS = 0
comptime TREES_ROUTE_GPU_ABSENT = 1        # explicit 'gpu' at a fit of a CPU-only installation
comptime TREES_ROUTE_FITTED_ON_CPU = 2     # fitted state of the CPU set asked through the device set
comptime TREES_ROUTE_FITTED_ON_INSTALL = 3 # fitted state of the device set asked through the CPU set


def trees_route_fits(name: String) -> Bool:
    """Whether operation `name` builds fitted state in the table set it runs
    on (`*_fit*`, an archive `load`); `is_fitted` only asks."""
    if name.find("is_fitted") >= 0:
        return False
    return name.find("fit") >= 0 or name.endswith("_load")


def trees_route_lifecycle(name: String) -> Bool:
    """Operations that read or replace parameters and never use fitted
    state: they run whatever the stored device and marker say."""
    return (name.find("params") >= 0 or name.find("construct") >= 0 or name.find("_init") >= 0
            or name.find("refresh") >= 0 or name.find("is_fitted") >= 0 or name.find("setstate") >= 0
            or name.find("vendor_used") >= 0)


def trees_route_strict(name: String) -> Bool:
    """Operations that check the stored parameters: an unknown `device` is
    refused in words there (the fits and `_validate_params`)."""
    return trees_route_fits(name) or name.find("validate_params") >= 0


struct TreesRoutePlan(ImplicitlyCopyable, Movable):
    # Run on the CPU table set of a GPU installation.
    var cpu: Bool
    # The operation's end writes the marker (a fit whose table set the
    # marker does not already name).
    var record: Bool
    # TREES_ROUTE_*: the refusal that applies, or TREES_ROUTE_RUNS.
    var refusal: Int

    def __init__(out self):
        self.cpu = False
        self.record = False
        self.refusal = TREES_ROUTE_RUNS


def trees_route_plan(device: Int, fitted: Int, cpu_only: Bool, fits: Bool, lifecycle: Bool) -> TreesRoutePlan:
    """The plan of one operation. `device` is a LINEAR_DEVICE_* request,
    `fitted` a TREES_FITTED_* marker, `cpu_only` whether the process has no
    GPU set, `fits` / `lifecycle` the operation's kind."""
    var plan = TreesRoutePlan()
    if cpu_only:
        # One table set, the CPU's. A marker written on a GPU installation
        # names a set this process does not have and is replaced by a fit.
        if device == LINEAR_DEVICE_GPU and fits:
            plan.refusal = TREES_ROUTE_GPU_ABSENT
            return plan
        plan.record = fits and fitted != TREES_FITTED_INSTALL
        return plan
    plan.cpu = device == LINEAR_DEVICE_CPU
    if fits:
        if plan.cpu:
            plan.record = fitted != TREES_FITTED_CPU
        else:
            plan.record = fitted != TREES_FITTED_INSTALL
        return plan
    if not lifecycle:
        if plan.cpu and fitted == TREES_FITTED_INSTALL:
            plan.refusal = TREES_ROUTE_FITTED_ON_INSTALL
        elif not plan.cpu and fitted == TREES_FITTED_CPU:
            plan.refusal = TREES_ROUTE_FITTED_ON_CPU
    return plan
