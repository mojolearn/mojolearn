# SPDX-License-Identifier: Apache-2.0
"""The host column's stratified group-fold plan; the six independent cases
run across host cores, each with its own buffers and failure slot."""
from checks.native_tests.core_stratified_group_fp64_cases import (
    STRATIFIED_GROUP_FIXTURES, require, strat_case, load_labels, group_codes, first_seen, check_plan,
)
from core.host_parallel import host_parallelize
from core.hotpath_impl_rows import bincount2_i32, strat_group_plan_i32


def check_unit(unit: Int) raises:
    var c = strat_case(unit)
    var labels = load_labels(c)
    var groups = group_codes(c)
    var fs = first_seen(labels)
    var enc = fs[0].copy()
    var k = fs[1]
    var dist = List[Int64](length=c.groups * k, fill=-1)
    _ = bincount2_i32(Int(enc.unsafe_ptr()), Int(groups.unsafe_ptr()), c.rows, k, c.groups, Int(dist.unsafe_ptr()))
    var folds = List[Int32](length=c.groups, fill=-1)
    var sizes = List[Int64](length=c.folds, fill=-1)
    var bad = strat_group_plan_i32(Int(dist.unsafe_ptr()), 0, Int(folds.unsafe_ptr()), Int(sizes.unsafe_ptr()), k, c.groups, c.folds)
    require(bad == 0, "every class has at least n_folds members")
    check_plan(c, enc, k, groups, folds)
    _ = dist
    _ = sizes


def run_checks() raises:
    var failures = List[Int](length=STRATIFIED_GROUP_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            check_unit(unit)
        except error:
            print("FAIL core_stratified_group_fp64_cpu case", unit, String(error))
            fp.unsafe_store(unit, 1)
    host_parallelize(task, STRATIFIED_GROUP_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "core_stratified_group_fp64_cpu failed " + String(failed) + " of " + String(STRATIFIED_GROUP_FIXTURES))
    print("PASS core_stratified_group_fp64_cpu:", STRATIFIED_GROUP_FIXTURES, "group-fold plans, parallel")
