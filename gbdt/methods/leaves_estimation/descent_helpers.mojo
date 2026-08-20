"""The Newton walker: iterate leaf values against an estimation oracle.

PORT OF `catboost/cuda/methods/leaves_estimation/descent_helpers.{h,cpp}` at
CatBoost `54a8143a`. Transliterated. Do not improve.

DIAGONAL ONLY. Their `TDirectionEstimator` switches on `HessianBlockSize`
(`descent_helpers.cpp:76-81`); the blocked arm solves per-block Cholesky
systems and is reached by MultiClass. Every pointwise loss is diagonal:

    MoveDirection[i] = Hessian[i] > 0
        ? Gradient[i] / (Hessian[i] + 1e-20f)     // :83-90
        : 0

CONTROL FLOW, kept quirk for quirk (`descent_helpers.cpp:128-204`):

* `Iterations == 1` short-circuits: one full step along the initial
  direction, `Regularize`, done -- the estimator never re-evaluates
  (`:151-156`). This is why an RMSE fit (Newton, 1 iteration) needs no
  walker: one step from zero IS the closed form.
* Otherwise each outer round freezes an acceptance rule at the current
  point, then tries steps 1, 1/2, 1/4, ... Every TRY costs one tick of
  the shared `iteration` counter; an ACCEPT costs one more and starts the
  next round. A run that has never accepted may keep halving to tick 100
  (`iteration < Iterations || (!updated && iteration < 100)`, `:179`).
* Only an ACCEPTED point gets second derivatives and becomes current
  (`:189-201`); the FINAL point is returned WITHOUT another regularize --
  it was regularized before its accepting evaluation.
* The walker moves in float: `point[leaf] += step * MoveDirection[leaf]`
  (`:69-71`) is a double product assigned into a float element, mirrored
  here as `Float32(Float64(p) + step * Float64(d))`.

The Langevin hooks at `:141-147` and `:190-196` are omitted, not stubbed;
`oracle_interface.mojo` records why and what putting them back requires.
"""

from gbdt.methods.leaves_estimation.oracle_interface import (
    LeavesEstimationOracle,
)
from gbdt.methods.leaves_estimation.step_estimator import (
    create_step_estimator,
)


def _diagonal_direction(
    gradient: List[Float64], hessian: List[Float64]
) -> List[Float32]:
    """`UpdateMoveDirectionDiagonal` (`descent_helpers.cpp:83-90`)."""
    var direction = List[Float32]()
    for i in range(len(gradient)):
        if hessian[i] > 0:
            direction.append(
                Float32(gradient[i] / (hessian[i] + Float64(1e-20)))
            )
        else:
            direction.append(Float32(0.0))
    return direction^


def _move(
    point: List[Float32], direction: List[Float32], step: Float64
) -> List[Float32]:
    """`MoveInOptimalDirection` (`descent_helpers.cpp:62-72`)."""
    var moved = List[Float32]()
    for i in range(len(point)):
        moved.append(
            Float32(Float64(point[i]) + step * Float64(direction[i]))
        )
    return moved^


def newton_like_walker_estimate[
    O: LeavesEstimationOracle
](
    mut oracle: O,
    iterations: Int,
    backtracking_type: Int,
    start_point: List[Float32],
) -> List[Float32]:
    """`TNewtonLikeWalker::Estimate` (`descent_helpers.cpp:128-204`)."""
    var point_dim = oracle.point_dim()

    # `startPoint.resize(Oracle.PointDim())` (`:131`)
    var cur_point = List[Float32]()
    for i in range(point_dim):
        cur_point.append(
            start_point[i] if i < len(start_point) else Float32(0.0)
        )

    # the initial evaluation (`:135-147`)
    var cur_value = Float64(0.0)
    var cur_grad = List[Float64]()
    var cur_hess = List[Float64]()
    oracle.move_to(cur_point)
    oracle.write_value_and_first_derivatives(cur_value, cur_grad)
    oracle.write_second_derivatives(cur_hess)
    var direction = _diagonal_direction(cur_grad, cur_hess)

    if iterations == 1:
        # `:151-156`: one full step, regularize, no re-evaluation
        var result = _move(cur_point, direction, 1.0)
        oracle.regularize(result)
        return result^

    var updated = False
    var iteration = 0
    while iteration < iterations:
        var est = create_step_estimator(
            backtracking_type, cur_value, cur_grad, direction
        )
        var step = Float64(1.0)
        var accepted = False
        var next_value = Float64(0.0)
        var next_grad = List[Float64]()
        while iteration < iterations or (
            (not updated) and iteration < 100
        ):
            var next_point = _move(cur_point, direction, step)
            oracle.regularize(next_point)
            oracle.move_to(next_point)
            oracle.write_value_and_first_derivatives(
                next_value, next_grad
            )
            if est.is_satisfied(step, next_value):
                # `:189-201`: only now are second ders paid for
                oracle.write_second_derivatives(cur_hess)
                cur_point = next_point.copy()
                cur_value = next_value
                cur_grad = next_grad.copy()
                direction = _diagonal_direction(cur_grad, cur_hess)
                iteration += 1
                updated = True
                accepted = True
                break
            # the for-increment arm: `++iteration, step /= 2` (`:179`)
            iteration += 1
            step /= 2
        if not accepted:
            break

    return cur_point^
