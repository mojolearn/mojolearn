# SPDX-License-Identifier: Apache-2.0
"""Lane mf-rfe7 (2026-11-23): RFE's importance sources after the door's own rounds (`_fit_clone`) were deleted.
Written, not run (WRITE CODE ONLY); IN_FILTER=mf_rfe7_sources.

  0 the kernel SVM kind's MEMBER_CAN_COEF (core/kernel_svm_member.mojo, MEMBER_OP_CAPABILITIES from the NEW
    words): set for kernel='linear' (code 0) on SVC and SVR, not for rbf (code 2); the family's union has it.
    Fail-proof: drop `common |= MEMBER_CAN_COEF` in the NEW-words branch (linear SVC loses the bit), or test
    `named[1]` against 2 (rbf gains it).
  1 `rfe_member_source` over a kernel SVM kind: with the bit 'auto' and 'coef_' read MEMBER_BLOCK_COEF squared;
    without it 'auto' / 'coef_' raise the SVM coef_ operation's words ("mojolearn SVC: coef_ is only available
    for kernel='linear'", SVR for a regressor mask) and 'feature_importances_' attrgetter's. Fail-proof: drop
    the MEMBER_FAMILY_KERNEL_SVM branch (rbf 'auto' gets RFE_NO_IMPORTANCES), or read the class from the
    mask the wrong way round (an SVC mask says SVR).
  2 every kind with neither block is refused (no family list): KernelRidge 'auto' RFE_NO_IMPORTANCES, an
    ExtraTrees mask 'feature_importances_' attrgetter's words. Fail-proof: answer RfeSource(0, False, "")
    after the two block tests.
  3 named feature-wide blocks (`rfe_named_block`, `rfe_member_source_named`): GaussianNB theta_ / var_,
    MultinomialNB feature_log_prob_ / feature_count_ (squared), CategoricalNB feature_log_prob_ refused, LDA
    and QDA means_, LDA xbar_, QDA xbar_ refused, an unknown name refused in attrgetter's words, the three
    plain getters as `rfe_member_source`. Fail-proof: swap _NB_BLOCK_THETA and _NB_BLOCK_VAR, drop the
    CategoricalNB test (it gets FAMILY+6), or drop `kind == _DA_KIND_LDA` (QDA xbar_ gets a block).
The end-to-end rounds (SVC linear, a named NB block, a META with a plain getter, the raised refusals) need the
SVM, NB and x_prep images: python/mojolearn/tests/test_mf_rfe7_members.py.
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, require
from core.member_abi import (
    MemberCall, MemberRef, member_call, MEMBER_OP_CAPABILITIES, MEMBER_CAN_FIT, MEMBER_CAN_EXPORT,
    MEMBER_CAN_COEF, MEMBER_CAN_IMPORTANCES, MEMBER_CAN_CLASSIFIER, MEMBER_BLOCK_COEF, MEMBER_BLOCK_FAMILY,
    MEMBER_BLOCK_FEATURE_IMPORTANCES,
)
from core.entry_table import EntryRef
from core.device_route import DeviceRoute, DEVICE_CPU
from core.member_families import (
    MEMBER_FAMILY_KERNEL_SVM, MEMBER_FAMILY_KERNEL_RIDGE, MEMBER_FAMILY_EXTRA_TREES, MEMBER_FAMILY_NAIVE_BAYES,
    MEMBER_FAMILY_DISCRIMINANT,
)
from core.kernel_svm_member import kernel_svm_member_v2
from bindings.prep_select_run import rfe_member_source, rfe_member_source_named, rfe_named_block, RFE_NO_IMPORTANCES

comptime CASES = 4


def _svm_mask(kind: Int, kernel: Int) raises -> Int:
    # The kind-exact mask from seventeen NEW words (no state): [0] kind, [1] kernel, [6] C, [12] probability.
    var words = List[Float64](length=17, fill=0.0)
    words[0] = Float64(kind)
    words[1] = Float64(kernel)
    words[6] = 1.0
    var call = MemberCall(DeviceRoute(DEVICE_CPU))
    call.params = Int(words.unsafe_ptr())
    call.param_count = len(words)
    var mask = member_call(MemberRef(EntryRef(kernel_svm_member_v2, 0), 0), MEMBER_OP_CAPABILITIES, call)
    _ = len(words)  # read through the call above
    return mask


def check_svm_bits() raises:
    require((_svm_mask(0, 0) & MEMBER_CAN_COEF) != 0, "linear SVC lacks MEMBER_CAN_COEF")
    require((_svm_mask(1, 0) & MEMBER_CAN_COEF) != 0, "linear SVR lacks MEMBER_CAN_COEF")
    require((_svm_mask(0, 2) & MEMBER_CAN_COEF) == 0, "rbf SVC has MEMBER_CAN_COEF")
    require((_svm_mask(1, 2) & MEMBER_CAN_COEF) == 0, "rbf SVR has MEMBER_CAN_COEF")
    var call = MemberCall(DeviceRoute(DEVICE_CPU))
    var union = member_call(MemberRef(EntryRef(kernel_svm_member_v2, 0), 0), MEMBER_OP_CAPABILITIES, call)
    require((union & MEMBER_CAN_COEF) != 0, "the kernel SVM family union lacks MEMBER_CAN_COEF")


def check_svm_source() raises:
    var base = MEMBER_CAN_FIT | MEMBER_CAN_EXPORT
    var linear = rfe_member_source(MEMBER_FAMILY_KERNEL_SVM, base | MEMBER_CAN_COEF | MEMBER_CAN_CLASSIFIER, "auto",
                                   "SVC")
    require(linear.block == MEMBER_BLOCK_COEF and linear.squared and linear.refusal.byte_length() == 0,
            "linear SVC 'auto': " + linear.refusal)
    var named = rfe_member_source(MEMBER_FAMILY_KERNEL_SVM, base | MEMBER_CAN_COEF, "coef_", "SVR")
    require(named.block == MEMBER_BLOCK_COEF and named.squared, "linear SVR 'coef_'")
    var rbf = rfe_member_source(MEMBER_FAMILY_KERNEL_SVM, base | MEMBER_CAN_CLASSIFIER, "auto", "SVC")
    require(rbf.block == 0 and rbf.refusal == "AttributeError: mojolearn SVC: coef_ is only available for kernel='linear'",
            "rbf SVC 'auto': " + rbf.refusal)
    var rbf_r = rfe_member_source(MEMBER_FAMILY_KERNEL_SVM, base, "coef_", "SVR")
    require(rbf_r.refusal == "AttributeError: mojolearn SVR: coef_ is only available for kernel='linear'",
            "rbf SVR 'coef_': " + rbf_r.refusal)
    var imp = rfe_member_source(MEMBER_FAMILY_KERNEL_SVM, base | MEMBER_CAN_COEF | MEMBER_CAN_CLASSIFIER,
                                "feature_importances_", "SVC")
    require(imp.refusal == "AttributeError: 'SVC' object has no attribute 'feature_importances_'",
            "SVC 'feature_importances_': " + imp.refusal)


def check_generic_refusal() raises:
    var base = MEMBER_CAN_FIT | MEMBER_CAN_EXPORT
    var kr = rfe_member_source(MEMBER_FAMILY_KERNEL_RIDGE, base, "auto", "KernelRidge")
    require(kr.block == 0 and kr.refusal == String(RFE_NO_IMPORTANCES), "KernelRidge 'auto': " + kr.refusal)
    var et = rfe_member_source(MEMBER_FAMILY_EXTRA_TREES, base, "feature_importances_", "ExtraTreesRegressor")
    require(et.refusal == "AttributeError: 'ExtraTreesRegressor' object has no attribute 'feature_importances_'",
            "ExtraTrees 'feature_importances_': " + et.refusal)
    # A kind WITH the block is not refused (the two block tests come first).
    var forest = rfe_member_source(MEMBER_FAMILY_EXTRA_TREES, base | MEMBER_CAN_IMPORTANCES, "auto", "X")
    require(forest.block == MEMBER_BLOCK_FEATURE_IMPORTANCES and not forest.squared, "a kind with importances")


def check_named_blocks() raises:
    var base = MEMBER_CAN_FIT | MEMBER_CAN_EXPORT | MEMBER_CAN_CLASSIFIER
    require(rfe_named_block(MEMBER_FAMILY_NAIVE_BAYES, 0, "theta_") == MEMBER_BLOCK_FAMILY, "GaussianNB theta_")
    require(rfe_named_block(MEMBER_FAMILY_NAIVE_BAYES, 0, "var_") == MEMBER_BLOCK_FAMILY + 1, "GaussianNB var_")
    require(rfe_named_block(MEMBER_FAMILY_NAIVE_BAYES, 0, "feature_log_prob_") == 0, "GaussianNB feature_log_prob_")
    require(rfe_named_block(MEMBER_FAMILY_NAIVE_BAYES, 1, "feature_log_prob_") == MEMBER_BLOCK_FAMILY + 6,
            "MultinomialNB feature_log_prob_")
    require(rfe_named_block(MEMBER_FAMILY_NAIVE_BAYES, 2, "feature_count_") == MEMBER_BLOCK_FAMILY + 8,
            "BernoulliNB feature_count_")
    require(rfe_named_block(MEMBER_FAMILY_NAIVE_BAYES, 4, "feature_log_prob_") == 0, "CategoricalNB feature_log_prob_")
    require(rfe_named_block(MEMBER_FAMILY_DISCRIMINANT, 0, "means_") == MEMBER_BLOCK_FAMILY, "LDA means_")
    require(rfe_named_block(MEMBER_FAMILY_DISCRIMINANT, 1, "means_") == MEMBER_BLOCK_FAMILY, "QDA means_")
    require(rfe_named_block(MEMBER_FAMILY_DISCRIMINANT, 0, "xbar_") == MEMBER_BLOCK_FAMILY + 2, "LDA xbar_")
    require(rfe_named_block(MEMBER_FAMILY_DISCRIMINANT, 1, "xbar_") == 0, "QDA xbar_")
    var theta = rfe_member_source_named(MEMBER_FAMILY_NAIVE_BAYES, 0, base, "theta_", "GaussianNB")
    require(theta.block == MEMBER_BLOCK_FAMILY and theta.squared and theta.refusal.byte_length() == 0, "theta_ source")
    var unknown = rfe_member_source_named(MEMBER_FAMILY_NAIVE_BAYES, 0, base, "sigma_", "GaussianNB")
    require(unknown.block == 0 and unknown.refusal == "AttributeError: 'GaussianNB' object has no attribute 'sigma_'",
            "an unknown name: " + unknown.refusal)
    var auto = rfe_member_source_named(MEMBER_FAMILY_NAIVE_BAYES, 0, base, "auto", "GaussianNB")
    require(auto.refusal == String(RFE_NO_IMPORTANCES), "GaussianNB 'auto' as rfe_member_source")


def test_mf_rfe7_sources() raises:
    def run_case(which: Int) raises:
        if which == 0:
            check_svm_bits()
        elif which == 1:
            check_svm_source()
        elif which == 2:
            check_generic_refusal()
        else:
            check_named_blocks()
    check_cases("mf_rfe7_sources", CASES, run_case)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
