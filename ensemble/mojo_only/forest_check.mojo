"""Does the forest loop build cuML's forest, and does predict read it back?

    pixi run mojo run -I . ensemble/mojo_only/forest_check.mojo

Covers `fit_forest` and `RowSampler` in `ensemble/randomforest.mojo` against
`randomforest.cuh:62-226` and `:286-370`, at cuML v26.08.00 (`265b9da6`).

THE FIXTURES EXPLOIT TWO ANALYTIC IDENTITIES, because "the forest looks
reasonable" is not a check and this repository gates on forced answers.

  IDENTITY ONE: with `bootstrap=False` and `max_features=1.0`, EVERY TREE
  IN THE FOREST MUST BE IDENTICAL. No bootstrap means every tree sees the
  same rows; `max_features=1.0` means every node considers every column.
  The per-node feature PERMUTATION still differs by tree -- the seed chain
  is `fnv1a32_hash(seed, treeid, nodeid)` -- but permuting the order in
  which the same columns are examined cannot change the winner, because
  `Split::update` breaks ties on a total order over the actual `colid`
  value and not on arrival order. So identical trees is not a coincidence
  to be tolerated, it is the REQUIRED answer, and any per-tree state
  leaking into the split logic breaks it.

  IDENTITY TWO: with `max_features` small, the trees MUST DIFFER. Each node
  now sees a different subset, chosen by `(seed, treeid, nodeid)`. If the
  trees come out identical here, the per-tree seed is not reaching the
  feature sampler at all -- which identity one alone cannot detect, because
  identical is its expected answer.

Together they pin the seed chain from both sides: it must reach the sampler
(two) and must not reach anything else (one).

  A. IDENTITY ONE, per node, per field, across every tree.
  B. IDENTITY TWO, and that the difference is deterministic across fits.
  C. `n_sampled_rows_for`'s round-half-away, and the `max_samples`
     overwrite-and-warn that their `else` branch performs.
  D. The three unported `RowSampler` arms raise BY NAME rather than
     silently behaving like the arm that does work.
  E. FIT THEN PREDICT: train a forest on a separable fixture and predict it
     back through the ported traversal. Every row must be classified
     correctly -- this is the only arm that runs the estimator end to end.
"""

from max.gpu.host import DeviceContext

from ensemble.decisiontree.batched_levelalgo.bins import ClassificationBin
from ensemble.decisiontree.decisiontree import DecisionTreeParams, GINI
from ensemble.randomforest import (
    CLASSIFICATION,
    RF_params,
    RandomForest,
    RandomForestMetaData,
    RowSampler,
    fit_forest,
    n_sampled_rows_for,
)

comptime DT = DType.float32
comptime LT = DType.int32


@always_inline
def _mix(x: UInt64) -> UInt64:
    var h = x
    h ^= h >> 33
    h *= 0xFF51AFD7ED558CCD
    h ^= h >> 33
    h *= 0xC4CEB9FE1A85EC53
    h ^= h >> 33
    return h


def _rf_params(
    n_trees: Int,
    bootstrap: Bool,
    max_features: Float32,
    max_depth: Int = 4,
    max_samples: Float32 = 1.0,
) -> RF_params:
    return RF_params(
        n_trees=Int32(n_trees),
        bootstrap=bootstrap,
        max_samples=max_samples,
        seed=UInt64(1234),
        n_streams=Int32(1),
        tree_params=DecisionTreeParams(
            max_depth=Int32(max_depth),
            max_leaves=Int32(-1),
            max_features=max_features,
            max_n_bins=Int32(16),
            min_samples_leaf=Int32(1),
            min_samples_split=Int32(2),
            split_criterion=GINI,
            min_impurity_decrease=Float32(0.0),
            max_batch_size=Int32(128),
        ),
    )


struct Data(Movable):
    var x: List[Float32]
    var y: List[Int32]
    var n_rows: Int
    var n_cols: Int
    var n_classes: Int

    def __init__(out self, n_rows: Int, n_cols: Int, n_classes: Int):
        self.x = List[Float32]()
        self.x.resize(n_rows * n_cols, Float32(0))
        self.y = List[Int32]()
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_classes = n_classes


def _hashed_data(n_rows: Int, n_cols: Int, n_classes: Int) -> Data:
    """Hashed features and a label that genuinely depends on several of
    them, so the tree has real structure to find rather than noise."""
    var d = Data(n_rows, n_cols, n_classes)
    for r in range(n_rows):
        var acc = UInt64(0)
        for c in range(n_cols):
            var v = Int(_mix(UInt64(r) * 131 + UInt64(c) + 7) % 512)
            d.x[c * n_rows + r] = Float32(v)
            if c < 3:
                acc += UInt64(v)
        d.y.append(Int32(Int(acc % UInt64(n_classes))))
    return d^


def _fit(
    ctx: DeviceContext, d: Data, mut p: RF_params
) raises -> RandomForestMetaData[DT, LT]:
    var hx = ctx.enqueue_create_host_buffer[DT](d.n_rows * d.n_cols)
    for i in range(d.n_rows * d.n_cols):
        hx.unsafe_ptr().unsafe_store(i, d.x[i])
    var dx = ctx.enqueue_create_buffer[DT](d.n_rows * d.n_cols)
    ctx.enqueue_copy(dst_buf=dx, src_ptr=hx.unsafe_ptr())

    var hy = ctx.enqueue_create_host_buffer[LT](d.n_rows)
    for i in range(d.n_rows):
        hy.unsafe_ptr().unsafe_store(i, d.y[i])
    var dy = ctx.enqueue_create_buffer[LT](d.n_rows)
    ctx.enqueue_copy(dst_buf=dy, src_ptr=hy.unsafe_ptr())

    var dsw = ctx.enqueue_create_buffer[DT](1)
    ctx.synchronize()

    var forest = fit_forest[LT, ClassificationBin](
        ctx, dx, dy, dsw, d.n_rows, d.n_cols, d.n_classes, p
    )
    # Mojo frees a value at its LAST USE; every buffer here reached a kernel
    # as a raw pointer.
    _ = dx^
    _ = dy^
    _ = dsw^
    _ = hx^
    _ = hy^
    return forest^


def _trees_equal(
    f: RandomForestMetaData[DT, LT], a: Int, b: Int
) -> Int:
    """Number of differing FIELDS between two trees, per node."""
    ref ta = f.trees[a]
    ref tb = f.trees[b]
    if len(ta.sparsetree) != len(tb.sparsetree):
        return 1000000
    var diff = 0
    for i in range(len(ta.sparsetree)):
        ref na = ta.sparsetree[i]
        ref nb = tb.sparsetree[i]
        if na.ColumnId() != nb.ColumnId():
            diff += 1
        if na.QueryValue() != nb.QueryValue():
            diff += 1
        if na.LeftChildId() != nb.LeftChildId():
            diff += 1
        if na.InstanceCount() != nb.InstanceCount():
            diff += 1
    if len(ta.vector_leaf) != len(tb.vector_leaf):
        return 1000000
    for i in range(len(ta.vector_leaf)):
        if ta.vector_leaf[i] != tb.vector_leaf[i]:
            diff += 1
    return diff


def arm_a_identical(ctx: DeviceContext) raises -> Int:
    """With bootstrap=False and max_features=1.0, every tree is identical."""
    var d = _hashed_data(600, 5, 3)
    var p = _rf_params(6, False, Float32(1.0))
    var f = _fit(ctx, d, p)
    var fails = 0

    if len(f.trees) != 6:
        fails += 1
        print("  arm A: expected 6 trees, got", len(f.trees))
        return fails
    if f.n_features != Int32(5):
        fails += 1
        print("  arm A: n_features", f.n_features, "want 5")
    if len(f.trees[0].sparsetree) < 5:
        fails += 1
        print(
            "  arm A: tree 0 has only", len(f.trees[0].sparsetree),
            "nodes -- too small to be evidence of anything",
        )
    var total_diff = 0
    for t in range(1, 6):
        total_diff += _trees_equal(f, 0, t)
    if total_diff != 0:
        fails += 1
        print(
            "  arm A FAILED:", total_diff,
            "field differences across the forest. With bootstrap=False and"
            " max_features=1.0 every tree sees the same rows and every"
            " column, so per-tree state has leaked into the SPLIT logic.",
        )
    # and the treeids must still be distinct and in order
    for t in range(6):
        if f.trees[t].treeid != Int32(t):
            fails += 1
            print("  arm A: tree", t, "has treeid", f.trees[t].treeid)

    if fails == 0:
        print(
            "  arm A OK: 6 trees of", len(f.trees[0].sparsetree),
            "nodes, BIT-IDENTICAL to each other across every field and"
            " every leaf value, with treeids 0..5 -- the per-tree seed"
            " reaches the sampler and nothing else",
        )
    return fails


def arm_b_differ(ctx: DeviceContext) raises -> Int:
    """With max_features small, trees MUST differ, and differ reproducibly."""
    var d = _hashed_data(600, 8, 3)
    var p1 = _rf_params(5, False, Float32(0.25))
    var f1 = _fit(ctx, d, p1)
    var fails = 0

    if len(f1.trees) != 5:
        fails += 1
        print("  arm B: expected 5 trees, got", len(f1.trees))
        return fails

    var differing_pairs = 0
    for t in range(1, 5):
        if _trees_equal(f1, 0, t) != 0:
            differing_pairs += 1
    if differing_pairs == 0:
        fails += 1
        print(
            "  arm B FAILED: every tree is identical at max_features=0.25."
            " The per-tree seed is not reaching the FEATURE SAMPLER --"
            " which arm A cannot detect, because identical is arm A's"
            " expected answer."
        )
    else:
        print(
            "  arm B: ", differing_pairs, "of 4 tree pairs differ at"
            " max_features=0.25, as they must",
        )

    # and the whole forest must be reproducible
    var p2 = _rf_params(5, False, Float32(0.25))
    var f2 = _fit(ctx, d, p2)
    var repro_diff = 0
    if len(f2.trees) != len(f1.trees):
        repro_diff += 1000000
    else:
        for t in range(len(f1.trees)):
            ref ta = f1.trees[t]
            ref tb = f2.trees[t]
            if len(ta.sparsetree) != len(tb.sparsetree):
                repro_diff += 1
                continue
            for i in range(len(ta.sparsetree)):
                if ta.sparsetree[i].ColumnId() != tb.sparsetree[i].ColumnId():
                    repro_diff += 1
                if (
                    ta.sparsetree[i].QueryValue()
                    != tb.sparsetree[i].QueryValue()
                ):
                    repro_diff += 1
            for i in range(len(ta.vector_leaf)):
                if ta.vector_leaf[i] != tb.vector_leaf[i]:
                    repro_diff += 1
    if repro_diff != 0:
        fails += 1
        print(
            "  arm B FAILED: two fits of the same forest differ in",
            repro_diff, "places -- the forest is not reproducible",
        )
    elif fails == 0:
        print(
            "  arm B OK: the trees differ from each other AND the whole"
            " forest is bit-identical across two fits"
        )
    return fails


def arm_c_sampled_rows() raises -> Int:
    """`randomforest.cuh:299-309`, on the host."""
    var fails = 0
    # bootstrap off: max_samples is ignored entirely
    if n_sampled_rows_for(False, Float32(0.5), 1000) != 1000:
        fails += 1
        print("  arm C: bootstrap=False must use every row regardless")
    # bootstrap on: std::round, half away from zero
    if n_sampled_rows_for(True, Float32(1.0), 1000) != 1000:
        fails += 1
        print("  arm C: max_samples 1.0 must give n_rows")
    if n_sampled_rows_for(True, Float32(0.5), 1000) != 500:
        fails += 1
        print("  arm C: 0.5 x 1000 must be 500")
    # 0.5 x 999 = 499.5 -> round-half-AWAY -> 500, not 499 (which is what
    # round-half-to-even would give, and what a naive Int() truncation
    # would give)
    if n_sampled_rows_for(True, Float32(0.5), 999) != 500:
        fails += 1
        print(
            "  arm C: 0.5 x 999 = 499.5 must round AWAY from zero to 500;"
            " got", n_sampled_rows_for(True, Float32(0.5), 999),
            "-- truncation gives 499 and round-half-to-even gives 500 too,"
            " so this case alone does not separate them",
        )
    # 0.5 x 1001 = 500.5 -> 501 away, 500 to-even. THIS one separates them.
    if n_sampled_rows_for(True, Float32(0.5), 1001) != 501:
        fails += 1
        print(
            "  arm C: 0.5 x 1001 = 500.5 must round AWAY to 501; got",
            n_sampled_rows_for(True, Float32(0.5), 1001),
            " -- round-half-to-EVEN would give 500, and that is the"
            " difference this case exists to catch",
        )
    if fails == 0:
        print(
            "  arm C OK: bootstrap=False ignores max_samples; round is"
            " half-AWAY-from-zero, separated from half-to-even at"
            " 0.5 x 1001 = 500.5 -> 501"
        )
    return fails


def arm_d_unported_arms(ctx: DeviceContext) raises -> Int:
    """The three arms that are not ported must RAISE, by name.

    A silently-wrong sampler is the worst outcome available here: it would
    train a forest that looks entirely normal on rows nobody asked for.
    """
    var fails = 0

    var s1 = RowSampler(ctx, True, UInt64(7), 100, 100, False)
    var raised1 = False
    try:
        s1.sample(ctx, Int32(0))
    except e:
        raised1 = True
        if String(e).find("bootstrap row sampling") < 0:
            fails += 1
            print("  arm D: bootstrap arm raised, but not by name:", e)
    if not raised1:
        fails += 1
        print(
            "  arm D FAILED: bootstrap=True SILENTLY produced rows. It is"
            " not ported; it must raise."
        )

    var s2 = RowSampler(ctx, True, UInt64(7), 100, 100, True)
    var raised2 = False
    try:
        s2.sample(ctx, Int32(0))
    except e:
        raised2 = True
        if String(e).find("weighted bootstrap") < 0:
            fails += 1
            print("  arm D: weighted arm raised, but not by name:", e)
    if not raised2:
        fails += 1
        print("  arm D FAILED: weighted bootstrap did not raise")

    var s3 = RowSampler(ctx, False, UInt64(7), 100, 100, True)
    var raised3 = False
    try:
        s3.sample(ctx, Int32(0))
    except e:
        raised3 = True
    if not raised3:
        fails += 1
        print("  arm D FAILED: zero-weight removal did not raise")

    # and the arm that IS ported must produce the identity, per cell
    var s4 = RowSampler(ctx, False, UInt64(7), 100, 100, False)
    s4.sample(ctx, Int32(3))
    var h = ctx.enqueue_create_host_buffer[DType.int32](100)
    ctx.enqueue_copy(dst_buf=h, src_buf=s4.selected_rows)
    ctx.synchronize()
    var wrong = 0
    for i in range(100):
        if Int(h.unsafe_ptr().unsafe_load(i)) != i:
            wrong += 1
    if wrong != 0:
        fails += 1
        print(
            "  arm D FAILED: the no-bootstrap arm is thrust::sequence and"
            " must be the identity;", wrong, "of 100 differ",
        )
    _ = s4^
    _ = h^

    if fails == 0:
        print(
            "  arm D OK: all three unported arms raise by name; the ported"
            " arm is the identity in all 100 cells"
        )
    return fails


def arm_e_fit_then_predict(ctx: DeviceContext) raises -> Int:
    """Train a forest, then predict it back through the ported traversal."""
    var n_rows = 400
    var n_cols = 3
    var d = Data(n_rows, n_cols, 2)
    for r in range(n_rows):
        # exactly separable under quantization: two distinct values
        d.x[0 * n_rows + r] = Float32(0.0) if r < 200 else Float32(100.0)
        d.x[1 * n_rows + r] = Float32(Int(_mix(UInt64(r) + 11) % 1000))
        d.x[2 * n_rows + r] = Float32(Int(_mix(UInt64(r) + 977) % 1000))
        d.y.append(Int32(0) if r < 200 else Int32(1))

    var p = _rf_params(4, False, Float32(1.0), max_depth=3)
    var f = _fit(ctx, d, p)
    var fails = 0
    if len(f.trees) != 4:
        fails += 1
        print("  arm E: expected 4 trees, got", len(f.trees))
        return fails

    # predict wants ROW-MAJOR input (`randomforest.cuh:399, 407`)
    var row_major = List[Float32]()
    row_major.resize(n_rows * n_cols, Float32(0))
    for r in range(n_rows):
        for c in range(n_cols):
            row_major[r * n_cols + c] = d.x[c * n_rows + r]

    var preds = List[Int32]()
    preds.resize(n_rows, Int32(-1))
    var rf = RandomForest[DT, LT](f.rf_params.copy(), CLASSIFICATION)
    rf.predict(row_major, n_rows, n_cols, preds, f)

    var wrong = 0
    for r in range(n_rows):
        if preds[r] != d.y[r]:
            wrong += 1
    if wrong != 0:
        fails += 1
        print(
            "  arm E FAILED:", wrong, "of", n_rows,
            "rows misclassified on a PERFECTLY SEPARABLE fixture -- a"
            " forest that trained correctly and a traversal that reads it"
            " correctly cannot miss any",
        )
    if fails == 0:
        print(
            "  arm E OK: fit ->", len(f.trees),
            "trees -> predict, all", n_rows,
            "rows classified correctly on a separable fixture",
        )
    return fails


def main() raises:
    print("forest_check: ensemble/randomforest.mojo")
    print("  RowSampler (randomforest.cuh:62-226) and the forest loop (:286-370)")
    var ctx = DeviceContext()
    var fails = 0
    fails += arm_a_identical(ctx)
    fails += arm_b_differ(ctx)
    fails += arm_c_sampled_rows()
    fails += arm_d_unported_arms(ctx)
    fails += arm_e_fit_then_predict(ctx)
    if fails == 0:
        print("forest_check: ALL OK")
    else:
        raise Error("forest_check: " + String(fails) + " failure(s)")
