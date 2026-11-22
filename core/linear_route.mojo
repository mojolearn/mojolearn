# SPDX-License-Identifier: Apache-2.0
"""What a Ridge or LinearRegression estimator resolves into its device route.

The device rule, its words and the route value are the shared ones
(core/device_route.mojo); this module is only the family's resolver.
`device="gpu"` and `device="cpu"` name two published entry tables of the same
family (core/linear_entry_table.mojo): the device table of the selected
vendor and tier, and the CPU identical table. A route is resolved ONCE per
estimator (providers/python/linear_route_boundary_door.mojo) and holds the
resolved thin C entries; every later fit, predict and score calls through
them. No public operation tests the device, reads a table slot, builds a
schema String or compares an ABI name.

The family's slots of `DeviceRoute`: `entry` is the fit entry, `second` the
predict entry, `third` Ridge's weighted and one-column solve (`has_third`),
`fourth` LinearRegression's TSQR table (`has_fourth`), `fifth` the R^2
metrics program of a CPU route resolved beside a GPU set (`has_fifth`,
providers/python/metrics_route_boundary_door.mojo).

FORWARDING NAMES, for the lanes of the 2026-11-21 wave that were briefed to
reuse this module's device words while it still held them: `LINEAR_DEVICE_*`
and `linear_device_*` below forward to core/device_route.mojo. They hold no
rule of their own; they go when the wave's last importer moves
(`rg 'LINEAR_DEVICE_|linear_device_' bindings providers core checks`).

Resolution is a metadata operation with no independent numerical work to
split; it runs at an estimator's first operation, never per row.
"""
from core.entry_table import EntryRef, entry_ref_word
from core.linear_boundary_types import LINEAR_PREDICT_ABI
from core.linear_entry_table import (
    LinearEntryKey, LINEAR_TABLE_ESTIMATORS, LINEAR_TABLE_X_LINEAR, linear_table_entry,
)
from core.linear_fit_types import (
    LINEAR_FIT_ABI, LINEAR_FIT_OPERATION_COUNT, RIDGE_FIT_ABI, LF_MODE, LF_VENDOR,
)
from core.native_descriptor_call import native_descriptor_call
from core.device_route import (
    DeviceRoute, DEVICE_GPU, DEVICE_CPU, DEVICE_INSTALL, device_code, device_name, device_refusal,
    device_gpu_absent, device_installed,
)

comptime LINEAR_DEVICE_GPU = DEVICE_GPU
comptime LINEAR_DEVICE_CPU = DEVICE_CPU
comptime LINEAR_DEVICE_INSTALL = DEVICE_INSTALL


def linear_route_resolve(device: Int, key: LinearEntryKey, expected_vendor: Int,
    expected_mode: Int, explicit_host: Bool) raises -> DeviceRoute:
    """Read the family's fit and predict entries of `key` once and admit
    them: the compiled entry must report the vendor and tier the estimator
    selected. `explicit_host` keeps the explicit Host* classes' own wording."""
    var route = DeviceRoute(device)
    var fit = linear_table_entry[LINEAR_TABLE_ESTIMATORS](key, 0, String(LINEAR_FIT_ABI), LINEAR_FIT_OPERATION_COUNT)
    var vendor = native_descriptor_call(fit, LF_VENDOR, List[Int](), List[Int](), List[Float64]())
    var mode = native_descriptor_call(fit, LF_MODE, List[Int](), List[Int](), List[Float64]())
    if explicit_host:
        if vendor != 0 or mode != 1:
            raise Error("RuntimeError: the explicit host model requires its CPU identical binding")
    else:
        if vendor != expected_vendor:
            raise Error("RuntimeError: classical native vendor differs from selected execution")
        if mode != expected_mode:
            raise Error("RuntimeError: classical native mode differs from selected execution")
    var predict = linear_table_entry[LINEAR_TABLE_ESTIMATORS](key, 1, String(LINEAR_PREDICT_ABI), 1)
    route.mode = mode
    route.vendor = vendor
    route.entry = EntryRef(fit, entry_ref_word(key.mode, key.vendor, LINEAR_FIT_OPERATION_COUNT))
    route.second = EntryRef(predict, entry_ref_word(key.mode, key.vendor, 1))
    return route


def linear_route_weighted(mut route: DeviceRoute, key: LinearEntryKey, weighted: LinearEntryKey) raises:
    """Ridge's weighted and one-column solver entry (the x_linear table of
    the same tier and vendor)."""
    if weighted.mode != key.mode or weighted.vendor != key.vendor:
        raise Error("native Ridge weighted execution key differs")
    var entry = linear_table_entry[LINEAR_TABLE_X_LINEAR](weighted, 0, String(RIDGE_FIT_ABI), 1)
    route.third = EntryRef(entry, entry_ref_word(weighted.mode, weighted.vendor, 1))
    route.has_third = True
