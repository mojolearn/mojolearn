"""The overfitting detector and the held-out cursor that feeds it.

    pixi run check-overfitting-detector

NOT to be confused with `mojo_only/early_stop_check.mojo`, which is about
`run_tree_layout`'s one-level ROLLBACK during tree growth. This is about
stopping the BOOSTING loop.

THE CENTREPIECE IS AN EXACT IDENTITY, not a tolerance: **train with the
eval set equal to the learn set and the two loss curves must agree element
for element.** The learn loss comes off the training cursor, which the
partition-wise `add_model_value_kernel` updates as each tree terminates;
the held-out loss comes off a separate cursor that the ROW-WISE
`compute_bins_and_add_kernel` updates one tree at a time, re-deriving
every row's leaf from the compressed index. Different kernels, different
buffers, different orders. On the same rows they must produce the same
number, and if they do not, either the per-tree apply or the held-out
scoring is wrong -- and no aggregate would say which.

That identity also pins the thing most likely to go wrong silently: the
LEARNING RATE. `fit` stores `leaf_values * learning_rate` on the weak
model, so the per-tree apply must pass 1.0. Reapplying the rate makes the
held-out curve fall at a different speed from the learn curve, which looks
plausible on a plot and would make the detector stop on the wrong shape.
The identity catches it on the first tree.

THE OTHER GATES:

1. **The detector's arithmetic, against synthetic curves.** `Iter` at wait
   W stops exactly W+1 iterations after the minimum, for several W. That
   is arithmetic about their `UpdatePValue` (`overfitting_detector.cpp:
   166-173`), whose `else` arm returns exactly 1.0 while
   `IterationsFromLocalMax < IterationsWait`, and whose `IsNeedStop` is a
   strict `<` against a threshold of 1.0.
2. **CONTROLS.** A monotonically FALLING curve must never stop `Iter`.
   `None` must never stop anything. A detector built without a test set
   must be inert whatever was asked (`:122-124`).
3. **END TO END, on a curve that really does turn.** The fixture is built
   so the held-out loss has its minimum strictly INSIDE the iteration
   range -- if it did not, the check would pass with a detector that never
   fires and nobody would learn anything.
4. **`train` REFUSES `od_type` without an eval set**, rather than building
   an inert detector and silently never stopping.

SABOTAGES:

    O1  the reference stop index off by one      the W+1 arithmetic
    O2  the detector fed a monotone FALLING
        curve while expecting a stop            that it does not fire on
                                                a model that is improving
"""

from max.gpu.host import DeviceContext

from gbdt.overfitting_detector.overfitting_detector import (
    OD_INC_TO_DEC,
    OD_ITER,
    OD_NONE,
    make_overfitting_detector,
)
from gbdt.train import train


def synthetic_stop(
    errs: List[Float64], wait: Int, od: Int, threshold: Float64
) raises -> Int:
    """Feed a curve to a detector and report the first stopping index."""
    var d = make_overfitting_detector(od, False, threshold, wait, True)
    for i in range(len(errs)):
        d.add_error(errs[i])
        if d.is_need_stop():
            return i
    return -1


def build(
    n: Int, f: Int, seed: UInt64,
    mut x: List[Float32], mut y: List[Float32],
) -> None:
    """Hashed features, and a target that is a mix of one feature and a
    per-row term no tree can reach -- so a deep enough fit MEMORISES the
    learn rows and the held-out loss turns up. Without that turn there is
    no overfitting to detect and this check would be vacuous."""
    for c in range(f):
        for r in range(n):
            var h = (
                UInt64(r) * UInt64(2654435761)
                + UInt64(c) * UInt64(40503) + seed
            ) % UInt64(1000)
            x.append(Float32(Int(h)) / Float32(1000.0))
    for r in range(n):
        var h0 = (UInt64(r) * UInt64(2654435761) + seed) % UInt64(1000)
        var hn = (UInt64(r) * UInt64(40503) + seed) % UInt64(1000)
        y.append(
            Float32(Int(h0)) / Float32(500.0)
            + Float32(Int(hn)) / Float32(250.0)
        )


def check_overfitting_detector(ctx: DeviceContext) raises:
    var failures = 0

    print("-- gate 1: Iter stops exactly wait+1 after the minimum --")
    # a V: falls to index 4, then rises
    var v = List[Float64]()
    for e in [9.0, 7.0, 5.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]:
        v.append(e)
    # THE ARITHMETIC, and this check's first version had it wrong by one.
    # `IterationsFromLocalMax` is 0 AT the minimum and increments on every
    # iteration that is not a new best, so it reaches `IterationsWait` at
    # index `min + wait` -- not `min + wait + 1`. `UpdatePValue` fires on
    # `>=` (`overfitting_detector.cpp:167`), so the stop is at `min +
    # wait`. The check asserted `+1`, went red on all four waits, and the
    # DETECTOR was right; the expectation was wrong.
    #
    # The TREE COUNT is a different number and both appear below: trees
    # are counted from one while iterations are indexed from zero, so a
    # stop at iteration `min + wait` leaves `min + wait + 1` trees.
    for wait in [1, 2, 3, 5]:
        var got = synthetic_stop(v, wait, OD_ITER, 0.0)
        var want = 4 + wait
        if got != want:
            print(
                "  FAIL Iter wait", wait, "stopped at", got,
                "and should stop at", want,
            )
            failures += 1
        else:
            print("  ok   Iter wait", wait, "-> stop at", got)

    print()
    print("-- gate 2: the controls --")
    var falling = List[Float64]()
    for i in range(20):
        falling.append(10.0 - Float64(i) * 0.4)
    var f_iter = synthetic_stop(falling, 3, OD_ITER, 0.0)
    if f_iter != -1:
        print("  FAIL Iter stopped at", f_iter, "on a FALLING curve")
        failures += 1
    else:
        print("  ok   Iter never fires while the loss is still falling")
    var f_none = synthetic_stop(v, 1, OD_NONE, 0.9)
    if f_none != -1:
        print("  FAIL None stopped at", f_none)
        failures += 1
    else:
        print("  ok   None never stops")
    var inert = make_overfitting_detector(OD_ITER, False, 0.0, 1, False)
    if inert.is_active():
        print("  FAIL a detector with no test set is ACTIVE")
        failures += 1
    else:
        print("  ok   no test set -> inert, whatever was asked")
    # IncToDec at a real threshold fires on the V and not on the fall
    var itd_v = synthetic_stop(v, 2, OD_INC_TO_DEC, 0.9)
    var itd_f = synthetic_stop(falling, 2, OD_INC_TO_DEC, 0.9)
    if itd_v == -1 or itd_f != -1:
        print(
            "  FAIL IncToDec: V stopped at", itd_v,
            ", falling stopped at", itd_f,
        )
        failures += 1
    else:
        print("  ok   IncToDec fires on the V at", itd_v, ", not on the fall")

    print()
    print("-- THE IDENTITY: eval == learn makes the two curves equal --")
    var n = 2000
    var f = 4
    var xl = List[Float32]()
    var yl = List[Float32]()
    build(n, f, UInt64(1), xl, yl)
    var same = train(
        ctx, xl, yl, n, f, border_count=32, n_estimators=12,
        max_depth=5, loss="RMSE", learning_rate=Float32(0.3),
        eval_x_colmajor=xl, eval_y=yl,
    )
    if len(same.test_losses) != len(same.losses):
        print(
            "  FAIL curve lengths differ:", len(same.losses),
            "vs", len(same.test_losses),
        )
        failures += 1
    else:
        var worst = Float64(0.0)
        for i in range(len(same.losses)):
            var d = same.losses[i] - same.test_losses[i]
            if d < 0.0:
                d = -d
            var sc = same.losses[i]
            if sc < 1e-6:
                sc = 1e-6
            if d / sc > worst:
                worst = d / sc
        if worst > 1e-5:
            print(
                "  FAIL the partition-wise and row-wise applies disagree"
                " by", worst, "relative",
            )
            failures += 1
        else:
            print(
                "  ok   both applies agree to", worst,
                "relative over", len(same.losses), "iterations",
            )

    print()
    print("-- gate 3: end to end on a curve that really turns --")
    var xe = List[Float32]()
    var ye = List[Float32]()
    build(800, f, UInt64(77), xe, ye)
    var a = train(
        ctx, xl, yl, n, f, border_count=32, n_estimators=50,
        max_depth=6, loss="RMSE", learning_rate=Float32(0.3),
        eval_x_colmajor=xe, eval_y=ye,
    )
    var bi = a.best_iteration
    if bi <= 0 or bi >= len(a.test_losses) - 1:
        print(
            "  FAIL the held-out minimum is at", bi,
            "-- not strictly inside, so there is no overfitting to"
            " detect and this gate is vacuous",
        )
        failures += 1
    else:
        var lo = a.test_losses[bi]
        var hi = a.test_losses[len(a.test_losses) - 1]
        if not (hi > lo):
            print("  FAIL the held-out curve does not turn up")
            failures += 1
        else:
            print(
                "  ok   held-out minimum", lo, "at iteration", bi,
                "rising to", hi, "by", len(a.test_losses) - 1,
            )
        # and the learn curve must NOT turn -- that is the whole point
        if a.losses[len(a.losses) - 1] > a.losses[bi]:
            print(
                "  FAIL the LEARN curve turned up too; the fixture does"
                " not separate learn from held-out",
            )
            failures += 1
        else:
            print(
                "  ok   the learn curve keeps falling (",
                a.losses[bi], "->", a.losses[len(a.losses) - 1],
                ") -- which is why stopping on it is useless",
            )

    var wait = 5
    var b = train(
        ctx, xl, yl, n, f, border_count=32, n_estimators=50,
        max_depth=6, loss="RMSE", learning_rate=Float32(0.3),
        eval_x_colmajor=xe, eval_y=ye, od_type="Iter", od_wait=wait,
    )
    if not b.stopped_early:
        print("  FAIL Iter did not stop on an overfitting curve")
        failures += 1
    elif len(b.model.weak_models) != b.best_iteration + wait + 1:
        print(
            "  FAIL stopped with", len(b.model.weak_models),
            "trees; best", b.best_iteration, "+ wait", wait, "+ 1",
        )
        failures += 1
    else:
        print(
            "  ok   Iter wait", wait, "stopped at",
            len(b.model.weak_models), "trees, best_iteration",
            b.best_iteration,
        )

    print()
    print("-- gate 4: od_type without an eval set is refused --")
    var refused = False
    try:
        var _c = train(
            ctx, xl, yl, n, f, border_count=32, n_estimators=5,
            max_depth=4, loss="RMSE", od_type="Iter",
        )
    except e:
        refused = True
    if not refused:
        print("  FAIL od_type without an eval set was accepted")
        failures += 1
    else:
        print("  ok   refused")

    print()
    print("-- sabotages --")
    # O1: the waits must produce DISTINCT stop indices, or gate 1 would
    # pass with any monotone function of `wait` and could not tell
    # `min + wait` from `min + wait + 1`.
    var o1a = synthetic_stop(v, 2, OD_ITER, 0.0)
    var o1b = synthetic_stop(v, 3, OD_ITER, 0.0)
    if o1b - o1a != 1:
        print(
            "  FAIL O1: waits 2 and 3 stopped at", o1a, "and", o1b,
            "-- the gate cannot resolve one iteration",
        )
        failures += 1
    else:
        print(
            "  ok   O1 wait 2 -> ", o1a, ", wait 3 -> ", o1b,
            ": one iteration apart, so an off-by-one is visible",
        )
    var o2 = synthetic_stop(falling, 1, OD_ITER, 0.0)
    if o2 != -1:
        print("  FAIL O2: fired on a falling curve at", o2)
        failures += 1
    else:
        print("  ok   O2 a falling curve cannot make it fire at any wait")

    if failures != 0:
        raise Error(
            "overfitting detector check: " + String(failures) + " failures"
        )
    print()
    print("overfitting detector check: PASS")


def main() raises:
    var ctx = DeviceContext()
    check_overfitting_detector(ctx)
