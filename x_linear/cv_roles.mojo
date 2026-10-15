# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""A row's role in one cross-validation split (lane sk-last, 2026-10-15).

LassoCV / ElasticNetCV / LogisticRegressionCV read a fold description word
per row (`fid[i]`, the float32 words after the targets):

  * fid[i] >= 0: a partition (KFold, StratifiedKFold, GroupKFold, LeaveOneOut,
    ...): fid[i] is the row's fold id; split k's test rows are the rows with
    id k and its training rows every other row (the original form, the same
    tests, so the bits of every partition fit are unchanged);
  * fid[i] < 0: per-split roles (scikit-learn's ShuffleSplit, repeated or
    overlapping splits, splits that leave rows out): every word is -(n + 1)
    and split k's role of row i is the float32 word
    fid[(after + k) * n + i]: 0 neither, 1 training, 2 test. `after` is the
    number of n-word blocks the caller's layout keeps between the fold words
    and the roles (its weights). Python builds the roles natively
    (`mark_split_roles_f32`, bindings/hotpath_helpers.mojo).

A split < 0 is the refit: every row trains.
"""
from x_linear.ops import FP, ld

comptime CV_NEITHER = 0
comptime CV_TRAIN = 1
comptime CV_TEST = 2


@always_inline
def cv_role(fid: FP, i: Int, fold: Int, after: Int) -> Int:
    """Row i's role in split `fold` (CV_TRAIN, CV_TEST or CV_NEITHER);
    every row trains when fold < 0."""
    if fold < 0:
        return CV_TRAIN
    var c = ld(fid, i)
    if c >= Float32(0):
        return CV_TEST if Int(c) == fold else CV_TRAIN
    var n = Int(-c) - 1
    return Int(ld(fid, (after + fold) * n + i))


@always_inline
def cv_train(fid: FP, i: Int, fold: Int, after: Int) -> Bool:
    return cv_role(fid, i, fold, after) == CV_TRAIN


@always_inline
def cv_test(fid: FP, i: Int, fold: Int, after: Int) -> Bool:
    return fold >= 0 and cv_role(fid, i, fold, after) == CV_TEST
