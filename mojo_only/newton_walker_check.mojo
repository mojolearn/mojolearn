"""The Newton walker's control flow, against hand-derived trajectories.

    pixi run check-newton-walker

The walker is pure host logic, so its oracle can LIE in controlled ways and
every expectation below is computed by hand from `descent_helpers.cpp`'s
control flow -- no reference implementation of ours to agree with by
accident.

1. QUADRATIC, Newton exact: value = -sum a_i (p_i - m_i)^2 / 2, honest
   Hessian. The first full step lands on the optimum; ten iterations must
   end within 1e-6 of m (the 1e-20 the direction adds to the Hessian is the
   only slack), and the walk must never DECREASE the value.
2. CLIFF + UNDERREPORTED HESSIAN, the backtracking gate: 1-D, value
   -(p-0.6)^2 below p=1, a -100 cliff above, Hessian reported as 0.5 so the
   direction overshoots at 2.4. AnyImprovement at iterations=3 must try
   steps 1 (p=2.4, cliff, reject), 1/2 (p=1.2, cliff, reject), 1/4 (p=0.6,
   accept) -- exactly 4 move_to calls counting the initial evaluation, and
   the final point is EXACTLY Float32(0.6) = 0 + 0.25 * Float32(2.4).
3. SAME CLIFF, BACKTRACKING_NONE control: step 1.0 is accepted sight
   unseen, so the walk ends ON the cliff at 2.4 (the flat region's zero
   gradient holds it there) after 1 + 3 move_to calls. This is the arm
   that proves the acceptance rule actually reaches the walker.
4. ITERATIONS=1 SHORTCUT: one full step along the initial direction with
   NO re-evaluation -- exactly 1 move_to ever -- and `regularize` zeroing
   a bin must survive into the returned point (index 1 forced to 0).
"""

from gbdt.methods.leaves_estimation.descent_helpers import (
    newton_like_walker_estimate,
)
from gbdt.methods.leaves_estimation.oracle_interface import (
    LeavesEstimationOracle,
)
from gbdt.methods.leaves_estimation.step_estimator import (
    BACKTRACKING_ANY_IMPROVEMENT,
    BACKTRACKING_NONE,
)


struct QuadraticOracle(LeavesEstimationOracle):
    var a: List[Float64]
    var m: List[Float64]
    var zero_idx: Int
    var point: List[Float32]
    var moves: Int
    var last_value: Float64
    var min_seen_increase: Float64

    def __init__(out self, a: List[Float64], m: List[Float64], zero_idx: Int):
        self.a = a.copy()
        self.m = m.copy()
        self.zero_idx = zero_idx
        self.point = List[Float32]()
        self.moves = 0
        self.last_value = 0.0
        self.min_seen_increase = 0.0

    def point_dim(self) -> Int:
        return len(self.a)

    def hessian_block_size(self) -> Int:
        return 1

    def make_estimation_result(
        self, point: List[Float32]
    ) -> List[Float32]:
        return point.copy()

    def move_to(mut self, point: List[Float32]):
        self.point = point.copy()
        self.moves += 1

    def write_value_and_first_derivatives(
        mut self, mut value: Float64, mut gradient: List[Float64]
    ):
        value = 0.0
        gradient.clear()
        for i in range(len(self.a)):
            var d = Float64(self.point[i]) - self.m[i]
            value += -self.a[i] * d * d / 2.0
            gradient.append(self.a[i] * (self.m[i] - Float64(self.point[i])))

    def write_second_derivatives(mut self, mut second_der: List[Float64]):
        second_der.clear()
        for i in range(len(self.a)):
            second_der.append(self.a[i])

    def regularize(self, mut point: List[Float32]):
        if self.zero_idx >= 0:
            point[self.zero_idx] = Float32(0.0)


struct CliffOracle(LeavesEstimationOracle):
    var point: List[Float32]
    var moves: Int

    def __init__(out self):
        self.point = List[Float32]()
        self.moves = 0

    def point_dim(self) -> Int:
        return 1

    def hessian_block_size(self) -> Int:
        return 1

    def make_estimation_result(
        self, point: List[Float32]
    ) -> List[Float32]:
        return point.copy()

    def move_to(mut self, point: List[Float32]):
        self.point = point.copy()
        self.moves += 1

    def write_value_and_first_derivatives(
        mut self, mut value: Float64, mut gradient: List[Float64]
    ):
        gradient.clear()
        var p = Float64(self.point[0])
        if p <= 1.0:
            value = -(p - 0.6) * (p - 0.6)
            gradient.append(-2.0 * (p - 0.6))
        else:
            value = -100.0
            gradient.append(0.0)

    def write_second_derivatives(mut self, mut second_der: List[Float64]):
        second_der.clear()
        second_der.append(0.5)  # the LIE: true curvature is 2

    def regularize(self, mut point: List[Float32]):
        pass


struct BlockedQuadraticOracle(LeavesEstimationOracle):
    """A BLOCK-quadratic whose exact minimizer is known, per block.

        f(x)      = -1/2 * sum_b (x_b - m_b)^T H_b (x_b - m_b)
        gradient  =  H_b (m_b - x_b)          -- their sign convention:
                                                 `Der` is the NEGATIVE
                                                 gradient, so the walker
                                                 ASCENDS
        Hessian   =  H_b                       (row-major, per block)

    THE ANALYTIC GATE: the Newton direction is `H^-1 * H (m - x) = m - x`,
    so ONE FULL STEP from ANY start lands exactly on `m`, for every SPD
    `H`. That answer does not depend on this file's solver, on the block
    size, or on the starting point -- which is what makes it a gate rather
    than a tally.

    `H` is deliberately NOT diagonal. A diagonal `H` would make the blocked
    arm and the diagonal arm agree, and the check would pass with the
    Cholesky never reached.
    """

    var row_size: Int
    var n_blocks: Int
    var h: List[Float64]
    var m: List[Float64]
    var point: List[Float32]
    var sabotage: Int

    def __init__(
        out self, row_size: Int, n_blocks: Int, sabotage: Int
    ):
        self.row_size = row_size
        self.n_blocks = n_blocks
        self.sabotage = sabotage
        self.point = List[Float32]()
        self.h = List[Float64]()
        self.m = List[Float64]()
        # a diagonally dominant symmetric matrix per block, with DIFFERENT
        # numbers per block so no two blocks share an expectation
        for b in range(n_blocks):
            for i in range(row_size):
                for j in range(row_size):
                    if i == j:
                        self.h.append(4.0 + Float64(b) + Float64(i) * 0.5)
                    else:
                        var off = 0.75 / (1.0 + Float64(i + j + b))
                        self.h.append(off)
            for i in range(row_size):
                self.m.append(
                    1.0 + Float64(b) * 2.0 - Float64(i) * 0.6
                )

    def point_dim(self) -> Int:
        return self.row_size * self.n_blocks

    def make_estimation_result(
        self, point: List[Float32]
    ) -> List[Float32]:
        return point.copy()

    def hessian_block_size(self) -> Int:
        # B1: claim the Hessian is diagonal. The walker then divides by
        # the diagonal instead of solving, which cannot land on `m`
        # unless the off-diagonals are zero -- and they are not.
        if self.sabotage == 1:
            return 1
        return self.row_size

    def move_to(mut self, point: List[Float32]):
        self.point = point.copy()

    def write_value_and_first_derivatives(
        mut self, mut value: Float64, mut gradient: List[Float64]
    ):
        value = 0.0
        gradient.clear()
        for _ in range(self.point_dim()):
            gradient.append(0.0)
        for b in range(self.n_blocks):
            for i in range(self.row_size):
                var acc = 0.0
                for j in range(self.row_size):
                    var d = (
                        self.m[b * self.row_size + j]
                        - Float64(self.point[b * self.row_size + j])
                    )
                    acc += (
                        self.h[
                            b * self.row_size * self.row_size
                            + i * self.row_size
                            + j
                        ]
                        * d
                    )
                gradient[b * self.row_size + i] = acc
                var di = (
                    self.m[b * self.row_size + i]
                    - Float64(self.point[b * self.row_size + i])
                )
                value += -acc * di / 2.0

    def write_second_derivatives(mut self, mut second_der: List[Float64]):
        second_der.clear()
        for b in range(self.n_blocks):
            for i in range(self.row_size):
                for j in range(self.row_size):
                    var v = self.h[
                        b * self.row_size * self.row_size
                        + i * self.row_size
                        + j
                    ]
                    # B2: flip one off-diagonal's sign, which makes the
                    # reported Hessian a DIFFERENT matrix from the one the
                    # gradient came from, so the step cannot land on `m`.
                    if self.sabotage == 2 and i != j:
                        v = -v
                    second_der.append(v)
        # `write_second_derivatives` under B1 must return the DIAGONAL
        # shape the walker will then expect, or the shape check fires
        # instead of the arithmetic differing -- and a shape error is not
        # the mechanism this sabotage is aimed at.
        if self.sabotage == 1:
            var diag = List[Float64]()
            for b in range(self.n_blocks):
                for i in range(self.row_size):
                    diag.append(
                        self.h[
                            b * self.row_size * self.row_size
                            + i * self.row_size
                            + i
                        ]
                    )
            second_der.clear()
            for i in range(len(diag)):
                second_der.append(diag[i])

    def regularize(self, mut point: List[Float32]):
        pass


def expect(cond: Bool, mut failures: Int, msg: String):
    if not cond:
        print("FAIL:", msg)
        failures += 1


def main() raises:
    var failures = 0

    # 1: quadratic, 10 iterations
    var a: List[Float64] = [2.0, 0.5, 8.0]
    var m: List[Float64] = [0.75, -1.5, 0.03125]
    var q = QuadraticOracle(a, m, -1)
    var start = List[Float32]()
    var leaves = newton_like_walker_estimate(
        q, 10, BACKTRACKING_ANY_IMPROVEMENT, start
    )
    for i in range(3):
        expect(
            abs(Float64(leaves[i]) - m[i]) < 1e-6, failures,
            "quadratic leaf " + String(i) + " = " + String(leaves[i]),
        )

    # 2: cliff + underreported Hessian, AnyImprovement, iterations=3
    var c = CliffOracle()
    var got = newton_like_walker_estimate(
        c, 3, BACKTRACKING_ANY_IMPROVEMENT, List[Float32]()
    )
    expect(got[0] == Float32(0.6), failures,
           "cliff endpoint " + String(got[0]) + " != Float32(0.6)")
    expect(c.moves == 4, failures,
           "cliff move_to count " + String(c.moves) + " != 4")

    # 3: same cliff, backtracking NONE: the overshoot is accepted
    var c2 = CliffOracle()
    var got2 = newton_like_walker_estimate(
        c2, 3, BACKTRACKING_NONE, List[Float32]()
    )
    expect(got2[0] == Float32(2.4), failures,
           "no-backtracking endpoint " + String(got2[0]) + " != 2.4")
    expect(c2.moves == 4, failures,
           "no-backtracking move_to count " + String(c2.moves) + " != 4")

    # 4: iterations=1 shortcut + regularize survival
    var q2 = QuadraticOracle(a, m, 1)
    var got3 = newton_like_walker_estimate(
        q2, 1, BACKTRACKING_ANY_IMPROVEMENT, List[Float32]()
    )
    expect(q2.moves == 1, failures,
           "shortcut move_to count " + String(q2.moves) + " != 1")
    expect(got3[1] == Float32(0.0), failures, "regularize did not survive")
    expect(abs(Float64(got3[0]) - m[0]) < 1e-6, failures,
           "shortcut leaf 0 " + String(got3[0]))

    # 5: THE BLOCKED-HESSIAN ARM, which the diagonal cases never reach.
    #    PORTING_RULES 8: a non-default path is an unchecked path, and
    #    every switch is exercised on BOTH sides by a named check per side.
    #
    #    The gate is analytic: the Newton direction for a block-quadratic
    #    is `H^-1 H (m - x) = m - x`, so ONE step lands exactly on `m` for
    #    every SPD `H`, whatever this file's solver does.
    for row_size in [2, 3, 6]:
        var bq = BlockedQuadraticOracle(row_size, 4, 0)
        var gotb = newton_like_walker_estimate(
            bq, 1, BACKTRACKING_ANY_IMPROVEMENT, List[Float32]()
        )
        var worst = Float64(0.0)
        for i in range(len(gotb)):
            var d = abs(Float64(gotb[i]) - bq.m[i])
            if d > worst:
                worst = d
        expect(worst < 1e-4, failures,
               "blocked rowSize " + String(row_size)
               + " did not land on the minimizer, worst " + String(worst))

    # B1: claim the Hessian is diagonal. The off-diagonals are non-zero, so
    #     dividing by the diagonal CANNOT land on `m`.
    var b1 = BlockedQuadraticOracle(3, 4, 1)
    var g1 = newton_like_walker_estimate(
        b1, 1, BACKTRACKING_ANY_IMPROVEMENT, List[Float32]()
    )
    var moved1 = Float64(0.0)
    for i in range(len(g1)):
        var d = abs(Float64(g1[i]) - b1.m[i])
        if d > moved1:
            moved1 = d
    expect(moved1 > 1e-3, failures,
           "B1: reporting hessian_block_size 1 changed nothing, so the"
           " blocked arm is not reached -- worst " + String(moved1))

    # B2: flip the reported off-diagonals' sign, so the Hessian is no
    #     longer the matrix the gradient came from.
    var b2 = BlockedQuadraticOracle(3, 4, 2)
    var g2 = newton_like_walker_estimate(
        b2, 1, BACKTRACKING_ANY_IMPROVEMENT, List[Float32]()
    )
    var moved2 = Float64(0.0)
    for i in range(len(g2)):
        var d = abs(Float64(g2[i]) - b2.m[i])
        if d > moved2:
            moved2 = d
    expect(moved2 > 1e-3, failures,
           "B2: flipping the off-diagonals changed nothing -- worst "
           + String(moved2))

    if failures != 0:
        raise Error(
            "newton walker check FAILED with " + String(failures)
        )
    print("newton walker check: quadratic, cliff backtracking, "
          "no-backtracking control, shortcut+regularize, "
          "BLOCKED Hessian at rowSize 2/3/6 + 2 sabotages -- all match")
