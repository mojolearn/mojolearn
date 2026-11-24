# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""Lane mf-rfe6: the two typed pieces RFE's member path gained, with no Python.

 1. A search step (best_estimator_.feature_importances_): `meta_plan_leaf_importances` sets the forest leaf's
    importance word [10] on EVERY candidate of the search's plan entry (any may be best_estimator_) and
    answers True.
 2. A stack's final estimator (final_estimator_..., hop -1): only the LAST nested plan gets the word; the
    first (a forest too) keeps 0.
 3. Two steps (estimator_.final_estimator_.feature_importances_ through an RFE kind over a stack): the walk
    follows the nested META plan's handle (its params[1]) to the stack's entry and sets the final leaf.
 4. Not served, nothing set: a leaf that is not a tree or forest (a logistic plan), a hop past the entry's
    plans, a path with no step (hops of length 1), a non-META outer plan.
 5. `rfe_rows_gather`: True only with take_rows, ranges and a device scratch (has_out or has_scratch); False
    when any is missing (the door's programs then copied the fold's host rows).

FAIL-PROOF EDITS (each must make the named case fail): in core/ensemble_member.mojo `meta_plan_leaf_importances`
drop the `_search_kind` widening (case 1: the second candidate keeps 0); read `lo = 0` instead of `count - 1`
for a negative hop (case 2); return after the first step instead of following `params[1]` (case 3); drop the
family test on the leaf (case 4: the logistic plan's word [10] is set). In bindings/prep_select_run.mojo
`rfe_rows_gather` drop `(run.has_out or run.has_scratch)` (case 5).
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases_serial, require
from core.entry_table import entry_ref_unresolved
from core.ensemble_member import (
    meta_plan_mint, meta_plan_release, meta_plan_leaf_importances, _plan_first, _plan_final,
    META_SEARCH_CLASSIFIER, META_STACK_CLASSIFIER, META_STACK_REGRESSOR, META_RFE_REGRESSOR,
    META_BLOCK_FORWARD,
)
from core.member_abi import MemberPlan, MEMBER_BLOCK_FEATURE_IMPORTANCES
from core.member_families import MEMBER_FAMILY_ENSEMBLE, MEMBER_FAMILY_FOREST, MEMBER_FAMILY_LOGISTIC
from bindings.prep_classify_prog_run import ClsRunPlan
from bindings.prep_select_run import rfe_rows_gather

comptime CASES = 5


def _leaf(family: Int) -> MemberPlan:
    # A nested plan as the admission mints it: twelve parameter words, the importance word [10] at 0.
    var plan = MemberPlan()
    plan.family = family
    plan.params = List[Float64](length=12, fill=0.0)
    return plan^


def _meta(kind: Int, handle: Int) -> MemberPlan:
    # A META plan's six words (kind, plan handle, flag, folds, cap, reserved).
    var plan = MemberPlan()
    plan.family = MEMBER_FAMILY_ENSEMBLE
    plan.params = [Float64(kind), Float64(handle), 0.0, 5.0, 0.0, 0.0]
    return plan^


def _mint(var plans: List[MemberPlan]) raises -> Int:
    var masks = List[Int](length=len(plans), fill=0)
    var weights = List[Float64](length=len(plans), fill=1.0)
    return meta_plan_mint(plans^, weights^, List[Int](), masks^)


def _word(plan: MemberPlan) -> Float64:
    return plan.params[10]


def check_search() raises:
    var plans: List[MemberPlan] = [_leaf(MEMBER_FAMILY_FOREST), _leaf(MEMBER_FAMILY_FOREST)]
    var handle = _mint(plans^)
    var hops: List[Float64] = [0.0, Float64(MEMBER_BLOCK_FEATURE_IMPORTANCES)]
    require(meta_plan_leaf_importances(_meta(META_SEARCH_CLASSIFIER, handle), hops), "search: not served")
    require(_word(_plan_first(handle)) == 1.0, "search: the first candidate's word is not set")
    require(_word(_plan_final(handle)) == 1.0, "search: the last candidate's word is not set")
    meta_plan_release(handle)


def check_stack_final() raises:
    var plans: List[MemberPlan] = [_leaf(MEMBER_FAMILY_FOREST), _leaf(MEMBER_FAMILY_FOREST)]
    var handle = _mint(plans^)
    var hops: List[Float64] = [-1.0, Float64(MEMBER_BLOCK_FEATURE_IMPORTANCES)]
    require(meta_plan_leaf_importances(_meta(META_STACK_CLASSIFIER, handle), hops), "stack: not served")
    require(_word(_plan_final(handle)) == 1.0, "stack: the final estimator's word is not set")
    require(_word(_plan_first(handle)) == 0.0, "stack: a member that is not the final estimator was set")
    meta_plan_release(handle)


def check_two_steps() raises:
    var inner_plans: List[MemberPlan] = [_leaf(MEMBER_FAMILY_LOGISTIC), _leaf(MEMBER_FAMILY_FOREST)]
    var inner = _mint(inner_plans^)
    var outer_plans: List[MemberPlan] = [_meta(META_STACK_REGRESSOR, inner)]
    var outer = _mint(outer_plans^)
    var hops: List[Float64] = [0.0, -1.0, Float64(MEMBER_BLOCK_FEATURE_IMPORTANCES)]
    require(meta_plan_leaf_importances(_meta(META_RFE_REGRESSOR, outer), hops), "two steps: not served")
    require(_word(_plan_final(inner)) == 1.0, "two steps: the nested stack's final forest is not set")
    require(_word(_plan_first(inner)) == 0.0, "two steps: the nested stack's first member was set")
    meta_plan_release(outer)
    meta_plan_release(inner)


def check_not_served() raises:
    var plans: List[MemberPlan] = [_leaf(MEMBER_FAMILY_LOGISTIC)]
    var handle = _mint(plans^)
    var leaf_hops: List[Float64] = [0.0, Float64(MEMBER_BLOCK_FEATURE_IMPORTANCES)]
    require(not meta_plan_leaf_importances(_meta(META_STACK_CLASSIFIER, handle), leaf_hops),
            "a logistic leaf was served")
    require(_word(_plan_first(handle)) == 0.0, "a logistic leaf's word [10] was written")
    var past: List[Float64] = [3.0, Float64(MEMBER_BLOCK_FEATURE_IMPORTANCES)]
    require(not meta_plan_leaf_importances(_meta(META_STACK_CLASSIFIER, handle), past), "a hop past the plans")
    var none: List[Float64] = [Float64(META_BLOCK_FORWARD)]
    require(not meta_plan_leaf_importances(_meta(META_STACK_CLASSIFIER, handle), none), "a path with no step")
    require(not meta_plan_leaf_importances(_leaf(MEMBER_FAMILY_FOREST), leaf_hops), "a leaf as the outer plan")
    meta_plan_release(handle)


def check_rows_gather() raises:
    var key = entry_ref_unresolved()
    # ClsRunPlan(key, has_out, has_scratch, has_ranges, dev_store, take_rows)
    require(rfe_rows_gather(ClsRunPlan(key, True, False, True, True, True)), "has_out: the rows are not gathered")
    require(rfe_rows_gather(ClsRunPlan(key, False, True, True, True, True)), "has_scratch: not gathered")
    require(not rfe_rows_gather(ClsRunPlan(key, False, False, True, True, True)), "no device scratch: gathered")
    require(not rfe_rows_gather(ClsRunPlan(key, True, True, False, True, True)), "no ranges entry: gathered")
    require(not rfe_rows_gather(ClsRunPlan(key, True, True, True, True, False)), "no take-rows entry: gathered")


def test_mf_rfe6_rounds() raises:
    def run_case(which: Int) raises:
        if which == 0:
            check_search()
        elif which == 1:
            check_stack_final()
        elif which == 2:
            check_two_steps()
        elif which == 3:
            check_not_served()
        else:
            check_rows_gather()

    # Serial: the cases share the one plan store and read back what they set.
    check_cases_serial("mf_rfe6_rounds", CASES, run_case)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
