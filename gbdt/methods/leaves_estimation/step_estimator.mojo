# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# This lane MIRRORS CatBoost (pinned commit 54a8143a). Per-file provenance is in the root DERIVATION_MAP.tsv, in this file's own docstring, and in NOTICE; files with no CatBoost counterpart say so.
"""Backtracking acceptance tests for the Newton walker.

PORT OF `catboost/cuda/methods/leaves_estimation/step_estimator.{h,cpp}` at
CatBoost `54a8143a`. Transliterated. Do not improve.

Three acceptance rules behind one interface (`step_estimator.cpp:8-66`),
selected by `ELeavesEstimationStepBacktracking`:

  No             -> always satisfied (`TSkipStepEstimation`)
  AnyImprovement -> `FunctionValue <= nextFuncValue` (`TSimpleStepEstimator`)
                    -- everything downstream MAXIMIZES, so "improvement"
                    is a non-decrease, ties included
  Armijo         -> `nextFuncValue >= FunctionValue + C*step*DirGradDot`
                    with their literal `C = 1e-5` (`:36`); the gradient
                    argument of `IsSatisfied` is dead code in their file
                    too (`:59-62` is commented out)

`AnyImprovement` is the default (`TObliviousTreeLearnerOptions`,
`LeavesEstimationBacktrackingType`), and is what an untouched Logloss run
uses.

THE SHAPE DEVIATION, recorded: their three classes behind a virtual
interface become ONE struct switching on `kind`, because the walker is the
sole caller and Mojo host code has no reason to heap-allocate a vtable for
a two-branch predicate. Every comparison is theirs, bit for bit.
"""

comptime BACKTRACKING_NONE = 0
comptime BACKTRACKING_ANY_IMPROVEMENT = 1
comptime BACKTRACKING_ARMIJO = 2

#: their `const double C = 1e-5` (`step_estimator.cpp:36`)
comptime ARMIJO_C = 1e-5


@fieldwise_init
struct StepEstimator(Copyable, Movable):
    """One outer Newton round's acceptance rule, frozen at round start.

    `function_value` is the CURRENT point's value; `dir_grad_dot` is
    `sum(gradient[i] * direction[i])`, computed only for Armijo
    (`step_estimator.cpp:50-54`).
    """

    var kind: Int
    var function_value: Float64
    var dir_grad_dot: Float64

    def is_satisfied(self, step: Float64, next_func_value: Float64) -> Bool:
        if self.kind == BACKTRACKING_NONE:
            return True
        if self.kind == BACKTRACKING_ANY_IMPROVEMENT:
            return self.function_value <= next_func_value
        return next_func_value >= (
            self.function_value + ARMIJO_C * step * self.dir_grad_dot
        )


def create_step_estimator(
    kind: Int,
    current_value: Float64,
    gradient: List[Float64],
    direction: List[Float32],
) -> StepEstimator:
    """`CreateStepEstimator` (`step_estimator.cpp:69-89`)."""
    var dot = Float64(0.0)
    if kind == BACKTRACKING_ARMIJO:
        for i in range(len(gradient)):
            dot += gradient[i] * Float64(direction[i])
    return StepEstimator(kind, current_value, dot)
