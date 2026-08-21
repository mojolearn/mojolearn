"""The Newton walker: iterate leaf values against an estimation oracle.

PORT OF `catboost/cuda/methods/leaves_estimation/descent_helpers.{h,cpp}` at
CatBoost `54a8143a`. Transliterated. Do not improve.

BOTH ARMS. Their `TDirectionEstimator::UpdateMoveDirection` switches on
`HessianBlockSize` (`descent_helpers.cpp:75-81`):

    == 1   DIAGONAL, every pointwise loss (`:82-90`)
             MoveDirection[i] = Hessian[i] > 0
                 ? Gradient[i] / (Hessian[i] + 1e-20f)
                 : 0

    >  1   BLOCKED, MultiClass (`:91-117`). The Hessian is
           `HessianBlockSize x HessianBlockSize` PER BLOCK and one dense
           Cholesky system is solved per block:
             sigma    <- Hessian[block]              (rowSize^2 doubles)
             solution <- Gradient[block]             (rowSize doubles)
             SolveLinearSystemCholesky(&sigma, &solution)
             MoveDirection[block] = (float)solution

           For MultiClass a BLOCK IS A LEAF and `rowSize` is
           `numClasses - 1`, so this is a six-by-six solve per leaf on a
           seven-class problem. Their `ParallelFor` over blocks (`:98`) is
           a serial loop here; the solves are independent and the answer
           does not depend on the order.

NOTE WHAT THE BLOCKED ARM DOES *NOT* DO: there is no `Hessian > 0` guard and
no `+ 1e-20`. Where the diagonal arm zeroes a direction whose curvature is
non-positive, the blocked arm hands the system to Cholesky and takes
whatever comes back -- including, when the factorization fails, the raw
gradient, because `dposv` leaves the right-hand side untouched and their
`CB_ENSURE(info >= 0)` passes on failure. `gbdt/lapack/linear_system.mojo`
DEVIATION 74 has the full account.

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
* BOTH EXITS RETURN `Oracle.MakeEstimationResult(...)` (`:153`, `:204`).
  That is the identity for every single-dimensional loss and the gauge
  projection for MultiClass, and it belongs to the walker rather than to
  its caller.
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
from gbdt.lapack.linear_system import solve_linear_system_cholesky


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


def _blocked_hessian_direction(
    gradient: List[Float64],
    hessian: List[Float64],
    hessian_block_size: Int,
) raises -> List[Float32]:
    """`UpdateMoveDirectionBlockedHessian` (`descent_helpers.cpp:91-117`).

    Their two `CB_ENSURE`s first (`:94-96`), because a shape mismatch here
    is a silent wrong answer rather than a crash:

        rowSize * rowSize * numBlocks == Hessian.size()
        rowSize * numBlocks           == Point.size()

    then one Cholesky solve per block. `sigma` is COPIED per block because
    the factorization is destructive and the caller's Hessian is reused by
    the next accepted point.
    """
    var row_size = hessian_block_size
    var num_blocks = len(gradient) // row_size
    if row_size * row_size * num_blocks != len(hessian):
        raise Error(
            "blocked hessian: rowSize^2 * numBlocks is "
            + String(row_size * row_size * num_blocks)
            + " but the Hessian holds " + String(len(hessian))
        )
    if row_size * num_blocks != len(gradient):
        raise Error(
            "blocked hessian: rowSize * numBlocks is "
            + String(row_size * num_blocks)
            + " but the gradient holds " + String(len(gradient))
        )

    var direction = List[Float32]()
    for _ in range(len(gradient)):
        direction.append(Float32(0.0))

    for block_id in range(num_blocks):
        var sigma = List[Float64]()
        for i in range(row_size * row_size):
            sigma.append(hessian[block_id * row_size * row_size + i])
        var solution = List[Float64]()
        for i in range(row_size):
            solution.append(gradient[block_id * row_size + i])

        _ = solve_linear_system_cholesky(sigma, solution)

        for i in range(row_size):
            direction[block_id * row_size + i] = Float32(solution[i])

    return direction^


def _update_move_direction(
    gradient: List[Float64],
    hessian: List[Float64],
    hessian_block_size: Int,
) raises -> List[Float32]:
    """`UpdateMoveDirection` (`descent_helpers.cpp:75-81`), the switch."""
    if hessian_block_size == 1:
        return _diagonal_direction(gradient, hessian)
    return _blocked_hessian_direction(
        gradient, hessian, hessian_block_size
    )


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
) raises -> List[Float32]:
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
    var direction = _update_move_direction(
        cur_grad, cur_hess, oracle.hessian_block_size()
    )

    if iterations == 1:
        # `:151-156`: one full step, regularize, no re-evaluation, and
        # `return Oracle.MakeEstimationResult(result)` -- the projection is
        # part of the RETURN, not of the caller.
        var result = _move(cur_point, direction, 1.0)
        oracle.regularize(result)
        return oracle.make_estimation_result(result)

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
                direction = _update_move_direction(
                    cur_grad, cur_hess, oracle.hessian_block_size()
                )
                iteration += 1
                updated = True
                accepted = True
                break
            # the for-increment arm: `++iteration, step /= 2` (`:179`)
            iteration += 1
            step /= 2
        if not accepted:
            break

    # `return Oracle.MakeEstimationResult(estimator.GetCurrentPoint().Point)`
    # (`:204`). Identity for every single-dimensional loss; for MultiClass
    # it is the gauge projection, and omitting it hands the caller a vector
    # one component too wide.
    return oracle.make_estimation_result(cur_point)
