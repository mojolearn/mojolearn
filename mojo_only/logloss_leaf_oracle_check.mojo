"""LOGLOSS LEAF VALUES AGAINST CATBOOST'S OWN, TEN NEWTON ITERATIONS AND ONE.

    pixi run -e bench logloss-leaf-oracle-gen   # bench/oracle_logloss_leaves.txt
    pixi run check-logloss-leaf-oracle          # this file

THE GAP THIS CLOSES
-------------------
CatBoost's default leaf estimator for Logloss is Newton with TEN iterations
and `AnyImprovement` backtracking (`catboost_options.cpp:157-164`, then
`:315-329`); RMSE gets one. This port implements the whole ten-iteration
descent walker and, until this file, NOTHING had compared its output to
CatBoost's.

`mojo_only/logloss_estimator_check.mojo` compares the device walker against
a float64 host reimplementation written in the same file. It is a good gate
for the device path and it has teeth -- truncating the simulation to one
iteration fails all twelve of its leaves. It is not a gate against
CatBoost: a ten-iteration Newton descent with backtracking has many places
to be subtly wrong in a way that a same-author reimplementation reproduces
faithfully. `mojo_only/loss_oracle_check.mojo` does compare leaves against
CatBoost, but only TREE 0 and only for the nine losses ported on 2026-08-21,
of which Logloss is not one.

WHICH ARM OF THEIRS, AND WHY THAT IS NOT A DETAIL HERE
------------------------------------------------------
Their CPU, because `task_type="GPU"` raises on Apple silicon. Everywhere
else in this repository that is a footnote; here it is load-bearing,
because CatBoost's CPU and GPU leaf estimators are two DIFFERENT
implementations of the same descent:

    theirs, CPU    `private/libs/algo/approx_calcer/gradient_walker.h:60-104`
                   -- `FastGradientWalker`, point in DOUBLE, accepts on
                   `valueAfterStep < lossValue` (STRICT, minimizing), loss
                   measured by the objective METRIC (mean, weighted)
    theirs, GPU    `cuda/methods/leaves_estimation/descent_helpers.cpp:128-204`
                   -- `TNewtonLikeWalker::Estimate`, point in FLOAT, accepts
                   on `FunctionValue <= nextFuncValue` (NON-STRICT,
                   maximizing), loss measured by the target kernel's
                   unnormalized `functionValue`
    ours           the GPU one, transliterated
                   (`gbdt/methods/leaves_estimation/descent_helpers.mojo`)

Both walk the same Newton direction, halve the same step, and run the same
number of outer rounds, so the comparison is legitimate; what it CANNOT do
is claim bitwise equality, and it does not.

The arithmetic that would otherwise make the two incomparable is the L2
denominator. Their CPU divides by `-sumDer2 + l2 * (sumAllWeights /
allDocCount)` (`private/libs/algo_helpers/online_predictor.h:165-169`)
where the GPU uses a raw lambda. THE FIXTURE HAS UNIT WEIGHTS, so that
scaling factor is exactly 1 and the two denominators are the same number.
A weighted fixture here would be comparing two different regularizers and
reporting the difference as a walker defect.

THE LEAF ORDER, WHICH IS THE HARD PART
--------------------------------------
A permutation of leaves that sums the same is exactly the failure mode a
green check hides, and this repository has hit that class before
(`[[uniform-test-data-hides-permutation]]`). The mapping is
`leaf = sum over levels of bit(level) << level`, level 0 the LEAST
significant bit, `level` being the position of the split in the tree's own
split list. It is established, not assumed, in three places:

  1. THEIR APPLIER computes exactly that:
     `indexesVec[docId] |= (binFeaturePtr[docId] >= borderVal) << depth`
     (`libs/model/cpu/evaluator_impl.cpp:26-40`), and their TRAINER builds
     the same index with `splitWeight = 1 << splitParams.Depth`
     (`private/libs/algo/index_calcer.cpp:330`).
  2. NOTHING PERMUTES IT INTO THE FILE. `TObliviousTreeBuilder::AddTree`
     appends `treeLeafValues` verbatim and keeps the split order it was
     given (`libs/model/model_build_helper.cpp:163-173`); `Build`
     re-indexes split IDENTIFIERS but reorders no tree
     (`:176-201`).
  3. THE GENERATOR RECOMPUTES IT AND THE RESIDUAL IS IN THE FIXTURE.
     `tools/catboost_logloss_leaf_oracle.py` rebuilds every training row's
     leaf under that convention, replays the whole model as
     `sum_t leaf_values[t][leaf(row, t)]`, and differences it against
     `model.predict(..., prediction_type="RawFormulaVal")`; separately it
     compares its reconstruction against `model.calc_leaf_indexes(pool)`
     cell by cell. Both residuals are written as the `armmap` record and
     THIS FILE REFUSES A FIXTURE WHOSE MAPPING IS NOT EXACT -- a fixture
     field no branch consumes is the defect
     `mojo_only/catboost_apply_check.mojo` exists to close.

Our side uses the same numbering, and `gbdt/models/oblivious_model.mojo`
records why: their growth convention (left child KEEPS slot `i`, right
child is appended at `leavesCount + i`, `split_properties_helper.cpp:861`)
puts a row with bits `(b_0, b_1, ...)` in slot `sum b_l << l`. The
comparison below TESTS that rather than trusting it -- see PLACEMENT.

PLACEMENT, NOT MULTISET
-----------------------
The fixture is generated so that no two leaf values inside one tree tie
(`min leaf gap` is in the `armmap` record and this file prints it). On top
of the per-leaf comparison, every arm is also scored against their leaves
under two nontrivial permutations of OUR leaf vector -- BIT-REVERSAL of the
leaf index, which is the exact defect a wrong bit order would produce, and
ROTATION BY ONE. Both must come back far outside the band. A comparison
that only summed, or that compared sorted values, would pass all three.

THE BAND, AND WHERE THE NUMBER COMES FROM
-----------------------------------------
    |ours - theirs| <= LEAF_ATOL + LEAF_RTOL * |theirs|
    LEAF_ATOL = 1e-5    LEAF_RTOL = 1e-3

There is no float64 on this device (`PORTING.md` 94) and their CPU arm
accumulates in double throughout, so the floor is set by our float32:

  * the oracle's per-leaf der/der2 reduces are float32 sums over ~375 rows
    per leaf at depth 3, so about `sqrt(375) * 2^-24 ~ 1.2e-6` relative on
    each of gradient and Hessian;
  * the Newton ratio inherits both, ~2.4e-6 relative;
  * the point is stored FLOAT on our side and on their GPU
    (`descent_helpers.cpp:69-71` assigns a double product into a float
    element) and DOUBLE on their CPU, one more 6e-8 relative per accepted
    step;
  * the cursor the next iteration reads is float32 on our side, and the
    learning-rate multiply adds a last rounding.

That puts the honest floor near 1e-5 relative. The band sits two orders
above it, which is slack for the CPU/GPU implementation difference above,
and it is still FOUR TIMES SMALLER than the fixture's minimum intra-tree
leaf gap (printed), so no permutation of leaves can pass it. The band is
stated here, derived from that arithmetic, and is NOT to be widened until a
number passes; a leaf outside it is a finding to price in `PORTING.md`.

WHAT IS GATED

  G0  the quantization grid, per feature per border, EXACT. A border
      disagreement moves every row and would be absorbed into a leaf
      verdict.
  G1  every tree's splits, per level, feature and bin, plus the first
      divergent tree. Leaf values of a tree whose STRUCTURE differs are
      not compared, because they are values of a different tree.
  L1  per-leaf agreement at TEN iterations -- their default -- worst
      absolute and worst relative per tree, and the count outside the band.
  L2  the same at ONE iteration, which must agree far more tightly. If L2
      is tight and L1 is loose, the walker's ITERATIONS are where the
      divergence lives, and that is the finding rather than a pass/fail.
  L3  a control that must DIFFER: OUR leaves at one iteration against
      THEIR leaves at ten. If L3 comes back inside the band, the
      comparison is not comparing what it claims to and every other line
      is void.
  P   the two permutation controls above.

THE BORDERS ARE NOT A VARIABLE HERE. The fixture is 3000 rows and
`train`'s `border_build_max_samples` is 200,000, so the border SUBSAMPLE
never fires and both arms binarize the whole column; G0 then checks the
resulting grid against CatBoost's own, cell for cell, and it comes back 0
wrong. Nothing below depends on the subsampling behaviour that was being
repaired elsewhere on 2026-08-21.

WHAT IT FOUND, 2026-08-21, AND WHY THIS CHECK IS RED
-----------------------------------------------------
    L2 (one iteration)    0 of 96 leaves outside the band
                          worst |dleaf| 4.6e-08, worst relative 7.1e-07
                          all 36 splits match, worst |dpred| 4.5e-07
    L1 (ten iterations)   3 of 96 leaves outside the band
                          worst |dleaf| 3.5e-03, worst relative 3.3e-03
                          all 36 splits STILL match, worst |dpred| 4.4e-03

That is the tight-L2 / loose-L1 signature, and the cause is located.

THE WALKS STOP AT DIFFERENT ITERATIONS. On tree 0 seven of eight leaves
agree to better than 1.2e-06 and ONE does not: leaf 2, the extreme leaf
(235 rows, nearly one class), whose Newton iterate is still moving after
six steps. An independent float64 transliteration of BOTH their walkers,
run on their own tree-0 partition at a zero cursor, says:

    steps  leaf 2 / learning_rate
      6    -3.47066423     <- CATBOOST'S ANSWER at 10 iterations (to 1.4e-07)
      8    -3.48222831     <- OUR ANSWER        at 10 iterations (to 6e-08)
     10    -3.48332208     <- what ten accepted steps actually give

CatBoost accepts SIX of its ten; we accept EIGHT. Neither is running ten.
Confirmed three ways: their leaf values are bit-identical at
`leaf_estimation_iterations` = 6, 7, 8, 9, 10, 11, 12, 20 and 40, so their
walk is frozen, not slow; at `leaf_estimation_backtracking="No"` -- the arm
with no acceptance test at all -- CatBoost does run all ten and lands on
-3.48332222, our number and the float64 number; and the mean Logloss is
strictly decreasing in exact arithmetic at every one of the ten steps
(step 7 improves it by 1.7e-07), so nothing has converged.

WHY THEIR WALK FREEZES, from their source. Their CPU backtracking test is
`valueAfterStep < lossValue`, STRICT (`gradient_walker.h:92`), and the
value is the objective METRIC. For Logloss the approxes are stored
exponentiated, and `TCrossEntropyMetric::EvalSingleThread`'s expApprox arm
computes every row through **`FastLogf`** (`libs/metrics/metric.cpp:
270-275`), whose own documentation states accuracy ~1e-05
(`library/cpp/fast_log/fast_log.h:79-86`). On ARM64 the SSE3 vector arm is
compiled out and returns an empty holder with `tailBegin = begin`
(`libs/helpers/short_vector_ops.h:245-257`), so EVERY row takes that
scalar `FastLogf` path. Once the true improvement in the summed loss
(5.0e-04 at step 7, over 3000 rows whose per-row terms each carry ~1e-05
of approximation error) drops into that noise, the strict test fails --
and after one failure the halved steps converge to a no-op, which can
NEVER strictly decrease anything, so the walk is frozen for good.

WHY OURS FREEZES LATER. Ours is their GPU walker, whose test is
`FunctionValue <= nextFuncValue`, NON-strict (`step_estimator.cpp:8-66`),
against the target kernel's `functionValue` computed in float32 on device.
Same failure mode, different precision and a different tie rule, so it
stalls two steps later.

WHAT THAT MEANS FOR THE VERDICT. This is NOT a defect in the port: the
walker, the direction, the halving and the acceptance rule are their GPU's
and the audit of `descent_helpers.mojo` against
`descent_helpers.cpp:128-204` stands. It is a CPU-versus-GPU divergence
inside CatBoost that any faithful GPU port inherits, and this machine
cannot run their GPU to check the other side. It is priced as
`PORTING.md` 140, and the check is left RED rather than given an
allowance for the three cells, because an allowance would also swallow the
next regression. THE NUMBER TO WATCH IS THREE. More than three cells
outside, or a divergence in a leaf that is NOT an extreme one, is new.

THE SABOTAGE TABLE, all five run 2026-08-21, all restored:

  S1  `_move` takes `0.5 * step`         L1 3->64 of 72 (worst 0.307),
                                         L2 0->16 of 16, L3 moved. Both
                                         arms also lose tree structure
                                         after tree 0, which is the
                                         structure gate firing too.
  S2  drop the `iteration < 100` escape  NOTHING MOVED, and that is
                                         honest: the first round accepts
                                         on this fixture, so `updated` is
                                         true before the escape can apply
                                         and the branch is unreachable
                                         here. It is DECORATIVE for this
                                         check.
  S3  skip the in-loop `regularize`      NOTHING MOVED, for a stated
                                         reason: every leaf here holds at
                                         least 235 rows, so none is under
                                         `MinLeafWeight` and
                                         `RegularizeImpl` is the identity.
                                         `mojo_only/logloss_estimator_check
                                         .mojo` plants an EMPTY leaf and
                                         is where that arm is gated.
  S4  AnyImprovement `<=` -> `<`         L1 3->4 outside. Small, and it is
                                         the point: their GPU accepts a
                                         tie and their CPU does not, and
                                         one leaf of ninety-six turns on
                                         exactly that character.
  S5  the `iterations == 1` short
      circuit steps 0.5 instead of 1.0   L2 0->16 of 16 and L3 moved; L1
                                         untouched, because the ten-
                                         iteration arm never takes that
                                         branch.
"""

from max.gpu.host import DeviceContext
from std.math import sqrt

from gbdt.models.model_text import parse_f32, parse_f64
from gbdt.options.catboost_options import LEAF_ESTIMATION_NEWTON
from gbdt.train import TrainedModel, predict_floats, train

comptime ORACLE = "bench/oracle_logloss_leaves.txt"

#: The band, derived in the module note from float32 accumulation. Do not
#: widen it to make a number pass.
comptime LEAF_ATOL = 1e-5
comptime LEAF_RTOL = 1e-3

#: The `armmap` residuals the generator wrote. Their mapping proof is exact
#: arithmetic on their own model, so anything but zero means the fixture was
#: written by a generator that did not establish its leaf order.
comptime MAP_RESIDUAL_MAX = 1e-9


def _split(line: String) -> List[String]:
    var out = List[String]()
    for tok in line.split(" "):
        var s = String(String(tok).strip())
        if s.byte_length() > 0:
            out.append(s^)
    return out^


struct Arm(Copyable, Movable):
    var name: String
    var iters: Int
    var backtracking: String
    var scale: Float64
    var bias: Float64
    var map_residual: Float64
    var index_mismatches: Int
    var have_route2: Bool
    var min_leaf_gap: Float64
    var split_feat: List[List[Int]]
    var split_border: List[List[Float32]]
    var leaves: List[List[Float64]]
    var pred: List[Float64]

    def __init__(out self, name: String, iters: Int, backtracking: String,
                 trees: Int):
        self.name = name
        self.iters = iters
        self.backtracking = backtracking
        self.scale = 1.0
        self.bias = 0.0
        self.map_residual = -1.0
        self.index_mismatches = -1
        self.have_route2 = False
        self.min_leaf_gap = 0.0
        self.split_feat = List[List[Int]]()
        self.split_border = List[List[Float32]]()
        self.leaves = List[List[Float64]]()
        self.pred = List[Float64]()
        for _ in range(trees):
            self.split_feat.append(List[Int]())
            self.split_border.append(List[Float32]())
            self.leaves.append(List[Float64]())


struct Fixture(Movable):
    var rows: Int
    var feats: Int
    var depth: Int
    var trees: Int
    var border_count: Int
    var learning_rate: Float32
    var l2: Float32
    var logloss_border: Float32
    var version: String
    var borders: List[List[Float32]]
    var xcol: List[Float32]
    var y: List[Float32]
    var arms: List[Arm]

    def __init__(out self):
        self.rows = 0
        self.feats = 0
        self.depth = 0
        self.trees = 0
        self.border_count = 0
        self.learning_rate = Float32(0.0)
        self.l2 = Float32(0.0)
        self.logloss_border = Float32(0.5)
        self.version = String("")
        self.borders = List[List[Float32]]()
        self.xcol = List[Float32]()
        self.y = List[Float32]()
        self.arms = List[Arm]()


def _arm_index(mut fx: Fixture, want: String) raises -> Int:
    for i in range(len(fx.arms)):
        if fx.arms[i].name == want:
            return i
    raise Error("a record names arm '" + want + "' before its `arm` line")


def load(path: String) raises -> Fixture:
    var fx = Fixture()
    var f = open(path, "r")
    var text = f.read()
    f.close()

    for line in text.splitlines():
        var s = String(String(line).strip())
        if s.byte_length() == 0 or s.startswith(String("#")):
            continue
        var t = _split(s)
        var kind = t[0]
        if kind == String("version"):
            fx.version = t[1]
        elif kind == String("dims"):
            fx.rows = Int(t[1])
            fx.feats = Int(t[2])
            fx.depth = Int(t[3])
            fx.trees = Int(t[4])
            fx.border_count = Int(t[5])
            for _ in range(fx.feats):
                fx.borders.append(List[Float32]())
        elif kind == String("hyper"):
            fx.learning_rate = parse_f32(t[1])
            fx.l2 = parse_f32(t[2])
        elif kind == String("border"):
            fx.logloss_border = parse_f32(t[1])
        elif kind == String("borders"):
            var fi = Int(t[1])
            for i in range(3, len(t)):
                fx.borders[fi].append(parse_f32(t[i]))
        elif kind == String("xcol"):
            # column major already, and the columns arrive in order
            for i in range(2, len(t)):
                fx.xcol.append(parse_f32(t[i]))
        elif kind == String("target"):
            for i in range(1, len(t)):
                fx.y.append(parse_f32(t[i]))
        elif kind == String("arm"):
            # arm <name> Logloss Newton <iters> <backtracking>
            if t[2] != String("Logloss"):
                raise Error("this check reads Logloss arms only, got " + t[2])
            if t[3] != String("Newton"):
                raise Error("this check reads Newton arms only, got " + t[3])
            fx.arms.append(Arm(t[1], Int(t[4]), t[5], fx.trees))
        elif kind == String("armscalebias"):
            var ai = _arm_index(fx, t[1])
            fx.arms[ai].scale = parse_f64(t[2])
            fx.arms[ai].bias = parse_f64(t[3])
        elif kind == String("armmap"):
            var ai2 = _arm_index(fx, t[1])
            fx.arms[ai2].map_residual = parse_f64(t[2])
            fx.arms[ai2].index_mismatches = Int(t[3])
            fx.arms[ai2].have_route2 = Int(t[4]) == 1
            fx.arms[ai2].min_leaf_gap = parse_f64(t[5])
        elif kind == String("armsplit"):
            var ai3 = _arm_index(fx, t[1])
            var ti = Int(t[2])
            fx.arms[ai3].split_feat[ti].append(Int(t[4]))
            fx.arms[ai3].split_border[ti].append(parse_f32(t[5]))
        elif kind == String("armleaves"):
            var ai4 = _arm_index(fx, t[1])
            var ti2 = Int(t[2])
            for i in range(3, len(t)):
                fx.arms[ai4].leaves[ti2].append(parse_f64(t[i]))
        elif kind == String("armpred"):
            var ai5 = _arm_index(fx, t[1])
            for i in range(2, len(t)):
                fx.arms[ai5].pred.append(parse_f64(t[i]))
        else:
            raise Error("unknown record '" + kind + "' in " + path)

    if fx.rows == 0 or len(fx.arms) == 0:
        raise Error(path + " carries no dims line or no arms")
    if len(fx.xcol) != fx.rows * fx.feats:
        raise Error(
            String("fixture declares ") + String(fx.rows) + " x "
            + String(fx.feats) + " and carries " + String(len(fx.xcol))
            + " cells"
        )
    return fx^


def _bin_of(borders: List[Float32], value: Float32) raises -> Int:
    """Their split border VALUE -> our bin index, by EXACT equality.

    Both sides carry the same float32 written at full precision by the same
    generator, so a near miss means the grids diverged and a nearest match
    would paper over exactly that.
    """
    for b in range(len(borders)):
        if borders[b] == value:
            return b
    raise Error(
        "CatBoost split on a border that is not in its own grid for that"
        " feature -- the fixture is internally inconsistent, which is a"
        " generator bug and not a learner bug"
    )


def _reverse_bits(leaf: Int, depth: Int) -> Int:
    """The permutation a wrong LEVEL-TO-BIT order would produce."""
    var out = 0
    for b in range(depth):
        if (leaf >> b) & 1 == 1:
            out |= 1 << (depth - 1 - b)
    return out


struct LeafDiff(Copyable, Movable):
    """One (arm, permutation) leaf comparison, per tree and in total."""

    var compared: Int
    var outside: Int
    var worst_abs: Float64
    var worst_rel: Float64
    var worst_tree: Int
    var worst_leaf: Int
    #: worst absolute difference within each tree, in tree order
    var per_tree_abs: List[Float64]
    var per_tree_rel: List[Float64]
    var per_tree_outside: List[Int]

    def __init__(out self):
        self.compared = 0
        self.outside = 0
        self.worst_abs = 0.0
        self.worst_rel = 0.0
        self.worst_tree = -1
        self.worst_leaf = -1
        self.per_tree_abs = List[Float64]()
        self.per_tree_rel = List[Float64]()
        self.per_tree_outside = List[Int]()


def compare_leaves(
    theirs: List[List[Float64]],
    ours: TrainedModel,
    depth: Int,
    trees: Int,
    structure_ok: List[Bool],
    permutation: Int,
) raises -> LeafDiff:
    """THEIR leaf values against OURS, per tree, per leaf, by PLACEMENT.

    `permutation` is 0 for the identity, 1 for a bit-reversal of our leaf
    index and 2 for a rotation by one. The last two are controls: they must
    come back far outside the band, which is what proves the identity
    comparison is scoring position and not a multiset.
    """
    var d = LeafDiff()
    var n_leaves = 1 << depth
    for ti in range(trees):
        var worst_a = Float64(0.0)
        var worst_r = Float64(0.0)
        var out_here = 0
        if structure_ok[ti] and ti < ours.model.size():
            ref theirs_t = theirs[ti]
            ref ours_t = ours.model.weak_models[ti].leaf_values
            if len(theirs_t) == n_leaves and len(ours_t) == n_leaves:
                for leaf in range(n_leaves):
                    var src = leaf
                    if permutation == 1:
                        src = _reverse_bits(leaf, depth)
                    elif permutation == 2:
                        src = (leaf + 1) % n_leaves
                    var mine = Float64(ours_t[src])
                    var yours = theirs_t[leaf]
                    var a = abs(mine - yours)
                    var band = LEAF_ATOL + LEAF_RTOL * abs(yours)
                    var r = a / (abs(yours) if abs(yours) > 1e-12 else 1e-12)
                    d.compared += 1
                    if a > worst_a:
                        worst_a = a
                    if r > worst_r:
                        worst_r = r
                    if a > band:
                        out_here += 1
                    if a > d.worst_abs:
                        d.worst_abs = a
                        d.worst_tree = ti
                        d.worst_leaf = leaf
                    if r > d.worst_rel:
                        d.worst_rel = r
        d.per_tree_abs.append(worst_a)
        d.per_tree_rel.append(worst_r)
        d.per_tree_outside.append(out_here)
        d.outside += out_here
    return d^


struct ArmResult(Movable):
    var border_wrong: Int
    var split_wrong: Int
    var split_total: Int
    var first_bad_tree: Int
    var structure_ok: List[Bool]
    var identity: LeafDiff
    var reversed_bits: LeafDiff
    var rotated: LeafDiff
    var pred_worst: Float64
    var our_leaves: List[List[Float64]]

    def __init__(out self):
        self.border_wrong = 0
        self.split_wrong = 0
        self.split_total = 0
        self.first_bad_tree = -1
        self.structure_ok = List[Bool]()
        self.identity = LeafDiff()
        self.reversed_bits = LeafDiff()
        self.rotated = LeafDiff()
        self.pred_worst = 0.0
        self.our_leaves = List[List[Float64]]()


def run_arm(ctx: DeviceContext, fx: Fixture, ai: Int) raises -> ArmResult:
    ref a = fx.arms[ai]
    var r = ArmResult()

    # SAME EVERYTHING EXCEPT THE DEVICE. Every option is the one the
    # generator pinned on their arm, `tools/catboost_arm.py:55-75`'s set,
    # with `leaf_estimation_iterations` taken from the fixture rather than
    # from either side's default -- ten is the point of the L1 arm and one
    # is the point of the L2 arm, and defaulting either side would compare
    # two different configurations.
    var tm = train(
        ctx, fx.xcol, fx.y, fx.rows, fx.feats,
        border_count=fx.border_count,
        n_estimators=fx.trees,
        max_depth=fx.depth,
        learning_rate=fx.learning_rate,
        l2_leaf_reg=fx.l2,
        bootstrap_type=String("No"),
        random_seed=UInt64(0),
        loss=String("Logloss"),
        loss_border=fx.logloss_border,
        leaf_estimation_iterations=a.iters,
        leaf_estimation_method=LEAF_ESTIMATION_NEWTON,
    )

    # G0. THE GRID, per cell, exact.
    for fi in range(fx.feats):
        ref theirs = fx.borders[fi]
        ref ours = tm.borders[fi]
        if len(ours) != len(theirs):
            r.border_wrong += len(ours) + len(theirs)
            continue
        for b in range(len(theirs)):
            if ours[b] != theirs[b]:
                r.border_wrong += 1

    # G1. SPLITS, per tree per level. A tree whose structure differs is
    # excluded from the leaf comparison: its leaves are the leaves of a
    # different tree and differencing them measures nothing.
    for ti in range(fx.trees):
        var their_n = len(a.split_feat[ti])
        var our_n = 0
        if ti < tm.model.size():
            our_n = len(tm.model.weak_models[ti].structure.splits)
        var n = their_n if their_n > our_n else our_n
        r.split_total += n
        var bad_here = 0
        for lvl in range(n):
            if lvl >= their_n or lvl >= our_n:
                bad_here += 1
                continue
            var tf = a.split_feat[ti][lvl]
            var tb = _bin_of(fx.borders[tf], a.split_border[ti][lvl])
            ref os = tm.model.weak_models[ti].structure.splits[lvl]
            if Int(os.feature_id) != tf or Int(os.bin_idx) != tb:
                bad_here += 1
        r.split_wrong += bad_here
        r.structure_ok.append(bad_here == 0)
        if bad_here != 0 and r.first_bad_tree < 0:
            r.first_bad_tree = ti

    r.identity = compare_leaves(
        a.leaves, tm, fx.depth, fx.trees, r.structure_ok, 0
    )
    r.reversed_bits = compare_leaves(
        a.leaves, tm, fx.depth, fx.trees, r.structure_ok, 1
    )
    r.rotated = compare_leaves(
        a.leaves, tm, fx.depth, fx.trees, r.structure_ok, 2
    )

    # OUR leaves kept for the L3 cross-arm control, which needs one arm's
    # leaves scored against the OTHER arm's expectation.
    for ti in range(tm.model.size()):
        var row = List[Float64]()
        ref lv = tm.model.weak_models[ti].leaf_values
        for i in range(len(lv)):
            row.append(Float64(lv[i]))
        r.our_leaves.append(row^)

    # The row-wise apply, kept because a leaf gate that passed while every
    # prediction moved would mean the leaves are not the leaves being used.
    var ours_pred = predict_floats(ctx, tm, fx.xcol, fx.rows)
    for row in range(fx.rows):
        var d = abs(Float64(ours_pred[row]) - a.pred[row])
        if d > r.pred_worst:
            r.pred_worst = d
    return r^


def report_leaves(label: String, d: LeafDiff, trees: Int) raises:
    print(
        "    " + label,
        "| compared", d.compared,
        "| outside band", d.outside,
        "| worst abs", d.worst_abs,
        "| worst rel", d.worst_rel,
        "| at tree", d.worst_tree, "leaf", d.worst_leaf,
    )
    var line = String("      per tree worst |dleaf|:")
    for ti in range(trees):
        line += " " + String(d.per_tree_abs[ti])
    print(line)
    var line2 = String("      per tree outside band:")
    for ti in range(trees):
        line2 += " " + String(d.per_tree_outside[ti])
    print(line2)


def main() raises:
    var fx = load(String(ORACLE))
    print(
        "LOGLOSS LEAF DIFFERENTIAL against CatBoost", fx.version,
        "CPU  --", fx.rows, "rows x", fx.feats, "features, depth",
        fx.depth, ",", fx.trees, "trees, border budget", fx.border_count,
    )
    print(
        "  band: |ours - theirs| <= ", LEAF_ATOL, "+", LEAF_RTOL,
        "* |theirs|   (derived in the module note; not tuned to pass)",
    )

    # THE FIXTURE'S OWN MAPPING PROOF, read rather than trusted. A record
    # no branch consumes is a record that can be wrong forever.
    for ai in range(len(fx.arms)):
        ref a = fx.arms[ai]
        print(
            "  arm", a.name, ": Newton x", a.iters, ",", a.backtracking,
            "| leaf-order replay residual", a.map_residual,
            "| calc_leaf_indexes mismatches", a.index_mismatches,
            "| min intra-tree leaf gap", a.min_leaf_gap,
        )
        if a.map_residual < 0.0 or a.map_residual > MAP_RESIDUAL_MAX:
            raise Error(
                "arm " + a.name + " carries no established leaf order:"
                " replaying CatBoost's own model under `leaf = sum bit_d"
                " << d` misses its own predictions by "
                + String(a.map_residual)
                + ". Every leaf comparison below would be scoring an"
                " unknown permutation"
            )
        if a.have_route2 and a.index_mismatches != 0:
            raise Error(
                "arm " + a.name + ": the reconstructed leaf index"
                " disagrees with CatBoost's own calc_leaf_indexes on "
                + String(a.index_mismatches) + " cells"
            )
        if a.scale != 1.0 or a.bias != 0.0:
            raise Error(
                "arm " + a.name + " has scale/bias "
                + String(a.scale) + "/" + String(a.bias)
                + "; every leaf comparison here assumes the identity"
            )
        # DISTINCTNESS, which is what a placement check needs from a
        # fixture. Two leaves in one tree carrying the SAME value are
        # interchangeable and a permutation that moves only those two is
        # invisible to any comparison. The generator refuses to write a
        # fixture with a tie; this reads the number back rather than
        # trusting it.
        #
        # WHAT THIS DOES NOT CLAIM. `min_leaf_gap` on the one-iteration arm
        # is 5.2e-04, which is the same order as the band on a leaf of
        # magnitude 0.5, so a TRANSPOSITION of that one closest pair could
        # sit inside the band. That is stated rather than engineered away:
        # the claim these controls support is about a PERMUTATION OF THE
        # WHOLE LEAF VECTOR, and it is enforced directly below by requiring
        # that a bit-reversal and a rotation each throw at least half the
        # leaves outside the band -- not by an inequality on the gap.
        if a.min_leaf_gap <= 0.0:
            raise Error(
                "arm " + a.name + " has two leaf values in one tree that"
                " tie exactly, so a permutation moving them is invisible"
                " and PLACEMENT cannot be checked on this fixture"
            )

    var ctx = DeviceContext()
    var results = List[ArmResult]()
    for ai in range(len(fx.arms)):
        print()
        print("  == " + fx.arms[ai].name + " ==")
        var r = run_arm(ctx, fx, ai)
        print(
            "    grid cells wrong", r.border_wrong,
            "| splits wrong", String(r.split_wrong) + "/"
            + String(r.split_total),
            "| first divergent tree", r.first_bad_tree,
            "| worst |dpred|", r.pred_worst,
        )
        report_leaves(String("L identity  "), r.identity, fx.trees)
        # A DIVERGENCE IS A DIAGNOSIS, NOT A NUMBER. The worst tree's
        # leaves printed side by side is where reading their source starts.
        if r.identity.worst_tree >= 0 and r.identity.outside != 0:
            var wt = r.identity.worst_tree
            print("      tree", wt, "leaves, theirs then ours:")
            for leaf in range(1 << fx.depth):
                print(
                    "       ", leaf, fx.arms[ai].leaves[wt][leaf],
                    r.our_leaves[wt][leaf],
                    "   d", r.our_leaves[wt][leaf]
                    - fx.arms[ai].leaves[wt][leaf],
                )
        report_leaves(String("P bit-rev   "), r.reversed_bits, fx.trees)
        report_leaves(String("P rotate-1  "), r.rotated, fx.trees)
        results.append(r^)

    # ---- L3, THE CONTROL THAT MUST DIFFER ------------------------------
    #
    # OUR leaves at ONE iteration against THEIR leaves at TEN. If this
    # lands inside the band then the two configurations produce the same
    # numbers on this fixture and NOTHING above is evidence about the
    # walker's iterations.
    var i10 = -1
    var i1 = -1
    for ai in range(len(fx.arms)):
        if fx.arms[ai].iters == 10:
            i10 = ai
        elif fx.arms[ai].iters == 1:
            i1 = ai
    if i10 < 0 or i1 < 0:
        raise Error("the fixture must carry both a 10- and a 1-iteration arm")

    var l3_outside = 0
    var l3_compared = 0
    var l3_worst = Float64(0.0)
    ref ours1 = results[i1].our_leaves
    ref theirs10 = fx.arms[i10].leaves
    var n_leaves = 1 << fx.depth
    for ti in range(fx.trees):
        # Only where BOTH arms reproduced the structure, so the control is
        # about leaf values and not about trees that already differ.
        if not (results[i1].structure_ok[ti] and results[i10].structure_ok[ti]):
            continue
        if ti >= len(ours1):
            continue
        for leaf in range(n_leaves):
            var mine = ours1[ti][leaf]
            var yours = theirs10[ti][leaf]
            l3_compared += 1
            var a3 = abs(mine - yours)
            if a3 > l3_worst:
                l3_worst = a3
            if a3 > LEAF_ATOL + LEAF_RTOL * abs(yours):
                l3_outside += 1
    print()
    print(
        "  L3 CONTROL (our 1-iteration leaves vs their 10-iteration"
        " leaves): compared", l3_compared, "| outside band", l3_outside,
        "| worst abs", l3_worst,
    )

    # ---- verdicts ------------------------------------------------------
    var failures = 0
    for ai in range(len(fx.arms)):
        ref a = fx.arms[ai]
        ref r = results[ai]
        if r.border_wrong != 0:
            print(
                "  FAIL", a.name, ": the grid does not agree with"
                " CatBoost's, so nothing downstream is comparable",
            )
            failures += 1
        if r.identity.compared == 0:
            print(
                "  FAIL", a.name, ": not one leaf was compared -- every"
                " tree's structure differs, so this check reached nothing",
            )
            failures += 1
        if r.identity.outside != 0:
            print(
                "  FAIL", a.name, ":", r.identity.outside, "of",
                r.identity.compared, "leaves are outside the band, worst",
                r.identity.worst_abs, "at tree", r.identity.worst_tree,
                "leaf", r.identity.worst_leaf,
            )
            failures += 1
        # PLACEMENT. A permutation of the WHOLE leaf vector must throw at
        # least half its cells outside the band. "At least one" would be
        # satisfied by a comparison that is only accidentally positioned.
        var half = r.identity.compared // 2
        if r.reversed_bits.outside < half or r.rotated.outside < half:
            print(
                "  FAIL", a.name, ": a permuted leaf vector left",
                r.reversed_bits.outside, "(bit-reversed) and",
                r.rotated.outside, "(rotated) of", r.identity.compared,
                "cells outside the band, fewer than the", half, "required,"
                " so this comparison is not testing PLACEMENT",
            )
            failures += 1
    if l3_compared == 0:
        print("  FAIL L3: the control compared nothing")
        failures += 1
    elif l3_outside == 0:
        print(
            "  FAIL L3: our 1-iteration leaves are indistinguishable from"
            " their 10-iteration leaves on this fixture, so L1 is not"
            " evidence about the ten iterations at all",
        )
        failures += 1

    if failures != 0:
        raise Error(
            "logloss leaf differential FAILED with " + String(failures)
            + " verdicts. Read the per-tree lines above: a tight L2 beside"
            " a loose L1 localises the divergence to the walker's"
            " ITERATIONS, and the numbers are the finding -- do not widen"
            " the band. THREE cells of ninety-six outside on the"
            " ten-iteration arm, all on extreme leaves, is the KNOWN state:"
            " PORTING.md 140, CatBoost's CPU walk freezes after six"
            " accepted steps where ours takes eight, because their"
            " acceptance test is measured through FastLogf and ours"
            " through a float32 device reduce. Anything more than that, or"
            " a divergence on a leaf that is not extreme, is new"
        )
    print()
    print(
        "  logloss leaf differential: every leaf of every tree agrees with"
        " CatBoost's own, at ten Newton iterations and at one, by"
        " PLACEMENT, with both permutation controls and the cross-arm"
        " control failing as they must"
    )
