# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`qn_params` and `qn_loss_type`: the solver's public parameter block.

PORT OF `cuml/cpp/include/cuml/linear_model/qn.h` at cuML `00094f7`. The
struct and the enum, with their defaults (`qn.h:80-96`). Do not improve.

The values a `mojolearn.LogisticRegression` hands in come from cuML's
Python layer, not from these defaults: `logistic_regression.py:275-340`
builds a `QN(loss='sigmoid', fit_intercept, l1_strength, l2_strength,
max_iter, linesearch_max_iter, tol)` and `solvers/qn.pyx:500-513` fills
`qn_params` from it with `grad_tol = tol`, `change_tol = delta if delta is
not None else tol * 0.01`, `lbfgs_memory = 5`, `penalty_normalized = True`.
`python/mojolearn/linear_model.py` reproduces that mapping and says so.
"""

comptime QN_LOSS_LOGISTIC = 0
comptime QN_LOSS_SQUARED = 1
comptime QN_LOSS_SOFTMAX = 2
comptime QN_LOSS_SVC_L1 = 3
comptime QN_LOSS_SVC_L2 = 4
comptime QN_LOSS_SVR_L1 = 5
comptime QN_LOSS_SVR_L2 = 6
comptime QN_LOSS_ABS = 7
comptime QN_LOSS_UNKNOWN = 99


@fieldwise_init
struct QNParams(ImplicitlyCopyable, Copyable, Movable):
    """`qn_params`. Field order and defaults are theirs (`qn.h:80-96`)."""

    var loss: Int
    var penalty_l1: Float64
    var penalty_l2: Float64
    var grad_tol: Float64
    var change_tol: Float64
    var max_iter: Int
    var linesearch_max_iter: Int
    var lbfgs_memory: Int
    var verbose: Int
    var fit_intercept: Bool
    var penalty_normalized: Bool

    @staticmethod
    def default() -> Self:
        return Self(
            QN_LOSS_UNKNOWN, 0.0, 0.0, 1e-4, 1e-5, 1000, 50, 5, 0, True, True
        )
