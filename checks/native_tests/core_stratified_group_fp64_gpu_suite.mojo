# SPDX-License-Identifier: Apache-2.0
"""The GPU column's stratified group-fold plan: the group x class table on
the device (device_bincount2_i32), the plan on the host as the door runs it.
The six cases run in order on one DeviceContext (each table is one kernel
over all rows); the host oracles are cheap."""
from max.gpu.host import DeviceContext
from checks.native_tests.core_stratified_group_fp64_cases import (
    STRATIFIED_GROUP_FIXTURES, require, strat_case, load_labels, group_codes, first_seen, check_plan,
)
from core.hotpath_device import device_bincount2_i32
from core.hotpath_impl_rows import strat_group_plan_i32


def run_checks() raises:
    var ctx = DeviceContext()
    var failed = 0
    for unit in range(STRATIFIED_GROUP_FIXTURES):
        try:
            var c = strat_case(unit)
            var labels = load_labels(c)
            var groups = group_codes(c)
            var fs = first_seen(labels)
            var enc = fs[0].copy()
            var k = fs[1]
            var dist = List[Int64](length=c.groups * k, fill=-1)
            require(device_bincount2_i32(ctx, Int(enc.unsafe_ptr()), Int(groups.unsafe_ptr()), c.rows, k, c.groups,
                                         Int(dist.unsafe_ptr())), "the device table declined in-range codes")
            var folds = List[Int32](length=c.groups, fill=-1)
            var sizes = List[Int64](length=c.folds, fill=-1)
            var bad = strat_group_plan_i32(Int(dist.unsafe_ptr()), 0, Int(folds.unsafe_ptr()), Int(sizes.unsafe_ptr()), k, c.groups, c.folds)
            require(bad == 0, "every class has at least n_folds members")
            check_plan(c, enc, k, groups, folds)
            _ = dist
            _ = sizes
        except error:
            failed += 1
            print("FAIL core_stratified_group_fp64_gpu case", unit, String(error))
    require(failed == 0, "core_stratified_group_fp64_gpu failed " + String(failed) + " of " + String(STRATIFIED_GROUP_FIXTURES))
    print("PASS core_stratified_group_fp64_gpu:", STRATIFIED_GROUP_FIXTURES, "group-fold plans, ordered on one device")
