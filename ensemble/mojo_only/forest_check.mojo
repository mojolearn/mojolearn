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
  D. `RowSampler`'s four arms. The bootstrap arm is compared PER CELL
     against RAFT's own compiled output (`ensemble/bench/philox_oracle.txt`,
     driven through cuML's real `fnv1a32(fnv1a32(basis, seed), tree_id)`
     chain), so it holds the seed chain, the bounds and the stride together
     rather than agreeing with the function it calls. The two
     weight-dependent arms still raise by name. The no-bootstrap arm is the
     identity.
  E. FIT THEN PREDICT: train a forest on a separable fixture and predict it
     back through the ported traversal. Every row must be classified
     correctly -- this is the only arm that runs the estimator end to end.
"""

from max.gpu.host import DeviceContext

from ensemble.decisiontree.batched_levelalgo.bins import ClassificationBin
from ensemble.decisiontree.batched_levelalgo.objectives import (
    ClassificationObjectiveFunction,
)
from ensemble.decisiontree.decisiontree import DecisionTreeParams, GINI
from core.philox import RNG_STRIDE
from ensemble.randomforest import (
    CLASSIFICATION,
    RF_params,
    RandomForest,
    RandomForestMetaData,
    RowSampler,
    check_random_seed,
    fit_forest,
    min_samples_leaf_fraction,
    min_samples_split_fraction,
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
    max_n_bins: Int = 16,
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
            max_n_bins=Int32(max_n_bins),
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

    var forest = fit_forest[
        ClassificationObjectiveFunction[DT, LT, ClassificationBin]
    ](ctx, dx, dy, dsw, d.n_rows, d.n_cols, d.n_classes, p)
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

    # The bootstrap arm is WIRED now, and is compared PER CELL against
    # RAFT's own compiled output rather than against itself. The oracle
    # rows come from `ensemble/bench/philox_oracle.txt`'s `e2e` section,
    # which drives RAFT's `uniformInt` through cuML's real seed chain
    # `fnv1a32(fnv1a32(basis, seed), tree_id)` -- so this holds the
    # SAMPLER's seed chain, bounds and stride together, and is not the
    # tautology "the sampler agrees with the function it calls".
    var oracle_rows = 0
    var oracle_wrong = 0
    var text: String
    with open("ensemble/bench/philox_oracle.txt", "r") as fh:
        text = fh.read()
    var lines = text.split("\n")
    for li in range(len(lines)):
        var line = String(lines[li])
        if line.byte_length() < 4 or String(line[byte=0]) != "e":
            continue
        var t = List[String]()
        var cur = String("")
        for i in range(line.byte_length()):
            var ch = String(line[byte=i])
            if ch == " ":
                if cur.byte_length() > 0:
                    t.append(cur)
                    cur = String("")
            else:
                cur += ch
        if cur.byte_length() > 0:
            t.append(cur)
        if len(t) < 8 or t[0] != "e2e":
            continue
        # `atol` raises above Int64's range and one oracle seed is
        # 0xDEADBEEFCAFEBABE. Parsed digit by digit in UInt64, where it
        # fits -- the same trap `shuffle_check` hit.
        var seed = UInt64(0)
        for di in range(t[1].byte_length()):
            seed = seed * 10 + UInt64(Int(atol(String(t[1][byte=di]))))
        var tree_id = Int(atol(t[2]))
        var nr = Int(atol(t[3]))
        var ns = Int(atol(t[4]))
        var stride = Int(atol(t[5]))
        var n_vals = len(t) - 7
        if n_vals <= 0:
            continue
        # SKIP rows taken at a non-default stride. The oracle deliberately
        # includes some -- `launch_uniform_int_ex` takes a stride so another
        # device's geometry is one call away (DEVIATION 184) -- but
        # `RowSampler` calls `launch_uniform_int`, which is pinned to
        # RNG_STRIDE. Feeding it a stride-256 row compared two different
        # launch geometries and reported 444 wrong rows; the port was right
        # and this filter is the fix. The mismatch began at index 256
        # exactly, which is what a stride difference looks like.
        if stride != RNG_STRIDE:
            continue
        # DEVIATION 400: the oracle's draws come from RAFT's TRUNCATING
        # seed chain, which our fixed chain matches exactly when the
        # seed's high word is zero. For a high-word oracle seed the same
        # RAFT parity is held by sampling with the LOW WORD (bit-equal
        # chain by the fix's own condition), and the fix's REACH is held
        # separately below: the full 64-bit seed must draw a DIFFERENT
        # sample than the truncated one, which is the whole point of
        # folding the high half.
        var hi = (seed >> 32) & 0xFFFFFFFF
        var parity_seed = seed if hi == 0 else (seed & 0xFFFFFFFF)
        var smp = RowSampler(ctx, True, parity_seed, nr, ns, False)
        smp.sample(ctx, Int32(tree_id))
        var hb = ctx.enqueue_create_host_buffer[DType.int32](ns)
        ctx.enqueue_copy(dst_buf=hb, src_buf=smp.selected_rows_[0])
        ctx.synchronize()
        oracle_rows += 1
        for i in range(min(n_vals, ns)):
            var want = Int(atol(t[7 + i]))
            if Int(hb.unsafe_ptr().unsafe_load(i)) != want:
                oracle_wrong += 1
                if oracle_wrong <= 2:
                    print(
                        "  arm D bootstrap MISMATCH seed", seed, "tree",
                        tree_id, "i", i, "got",
                        hb.unsafe_ptr().unsafe_load(i), "want", want,
                    )
        if hi != 0:
            var smp2 = RowSampler(ctx, True, seed, nr, ns, False)
            smp2.sample(ctx, Int32(tree_id))
            var hb2 = ctx.enqueue_create_host_buffer[DType.int32](ns)
            ctx.enqueue_copy(dst_buf=hb2, src_buf=smp2.selected_rows_[0])
            ctx.synchronize()
            var moved = 0
            for i in range(ns):
                if hb2.unsafe_ptr().unsafe_load(i) != hb.unsafe_ptr(
                ).unsafe_load(i):
                    moved += 1
            if moved == 0:
                oracle_wrong += 1
                print(
                    "  arm D DEVIATION 400 DID NOT REACH: seed",
                    seed, "tree", tree_id,
                    "drew the same rows as its truncated low word",
                )
            _ = smp2^
            _ = hb2^
        _ = smp^
        _ = hb^
    if oracle_rows < 5:
        fails += 1
        print(
            "  arm D: parsed only", oracle_rows,
            "oracle rows -- a check that reads nothing cannot fail",
        )
    if oracle_wrong != 0:
        fails += 1
        print(
            "  arm D FAILED:", oracle_wrong,
            "bootstrap rows disagree with RAFT's compiled output",
        )
    else:
        print(
            "  arm D: bootstrap sampler matches RAFT's own output per cell"
            " across", oracle_rows, "cuML call sites",
        )

    # Weighted bootstrap is PORTED now (`randomforest.cuh:125-138`), so it
    # must draw rather than raise -- and every drawn index must be a legal
    # row. Its distributional behaviour is checked in `sample_weight_check`;
    # what matters here is that all four arms of their dispatch produce
    # something usable.
    var s2 = RowSampler(ctx, True, UInt64(7), 100, 100, True)
    var w2 = List[Float32]()
    for i in range(100):
        w2.append(Float32(1 + (i % 3)))
    s2.prepare_weights(ctx, w2)
    s2.sample(ctx, Int32(0))
    var h2 = ctx.enqueue_create_host_buffer[DType.int32](100)
    ctx.enqueue_copy(dst_buf=h2, src_buf=s2.selected_rows_[0])
    ctx.synchronize()
    var oob = 0
    for i in range(100):
        var v = Int(h2.unsafe_ptr().unsafe_load(i))
        if v < 0 or v >= 100:
            oob += 1
    if oob != 0:
        fails += 1
        print(
            "  arm D FAILED: weighted bootstrap produced", oob,
            "row ids outside [0, 100) -- their upper_bound over the CDF"
            " cannot return one",
        )
    _ = s2^
    _ = h2^

    # The zero-weight arm (`randomforest.cuh:144-154`) IS PORTED now, so it
    # must NOT raise -- but it must also refuse to run before
    # `prepare_weights` has established which rows survive, because a
    # sampler that silently emitted `n_sampled_rows` identity indices there
    # would train on the zero-weight rows it was asked to drop. Its
    # behaviour proper is checked in `sample_weight_check`.
    var s3 = RowSampler(ctx, False, UInt64(7), 100, 100, True)
    var w3 = List[Float32]()
    for i in range(100):
        w3.append(Float32(0.0) if i % 5 == 0 else Float32(1.0))
    s3.prepare_weights(ctx, w3)
    s3.sample(ctx, Int32(0))
    if s3.n_selected != 80:
        fails += 1
        print(
            "  arm D: zero-weight removal selected", s3.n_selected,
            "rows, want 80",
        )
    _ = s3^

    # and the arm that IS ported must produce the identity, per cell
    var s4 = RowSampler(ctx, False, UInt64(7), 100, 100, False)
    s4.sample(ctx, Int32(3))
    var h = ctx.enqueue_create_host_buffer[DType.int32](100)
    ctx.enqueue_copy(dst_buf=h, src_buf=s4.selected_rows_[0])
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
            "  arm D OK: all four RowSampler arms -- uniform bootstrap"
            " matching RAFT per cell, weighted bootstrap drawing 100 legal"
            " rows, zero-weight removal selecting 80 of 100, and the"
            " no-bootstrap identity in all 100 cells"
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


def arm_f_bootstrapped_forest(ctx: DeviceContext) raises -> Int:
    """A forest trained cuML's DEFAULT way: bootstrap on.

    With bagging the trees must differ EVEN AT max_features=1.0 -- that is
    the whole point of bagging, and it is the arm that proves the sampler
    reaches the builder rather than merely producing plausible row ids in a
    buffer nobody reads. Arm A's identity is the same fixture with
    bootstrap OFF, so the two together separate "bagging works" from
    "bagging is wired but ignored".
    """
    var d = _hashed_data(600, 5, 3)
    var p = _rf_params(6, True, Float32(1.0))
    var f = _fit(ctx, d, p)
    var fails = 0

    if len(f.trees) != 6:
        fails += 1
        print("  arm F: expected 6 trees, got", len(f.trees))
        return fails
    var differing = 0
    for t in range(1, 6):
        if _trees_equal(f, 0, t) != 0:
            differing += 1
    if differing == 0:
        fails += 1
        print(
            "  arm F FAILED: every tree is identical WITH BOOTSTRAP ON at"
            " max_features=1.0. The row sample is being produced and then"
            " ignored -- arm A cannot see this, because identical is arm"
            " A's expected answer."
        )
    # and it must still be reproducible
    var p2 = _rf_params(6, True, Float32(1.0))
    var f2 = _fit(ctx, d, p2)
    var repro = 0
    if len(f2.trees) != len(f.trees):
        repro += 1
    else:
        for t in range(len(f.trees)):
            repro += _trees_equal_across(f, f2, t)
    if repro != 0:
        fails += 1
        print(
            "  arm F FAILED: two bootstrapped fits differ in", repro,
            "places -- the per-tree seed chain is not deterministic",
        )
    if fails == 0:
        print(
            "  arm F OK: bootstrap ON gives", differing,
            "of 5 tree pairs differing at max_features=1.0 (bagging is the"
            " only source of variation there), and two fits are"
            " bit-identical",
        )
    return fails


def _trees_equal_across(
    a: RandomForestMetaData[DT, LT],
    b: RandomForestMetaData[DT, LT],
    t: Int,
) -> Int:
    ref ta = a.trees[t]
    ref tb = b.trees[t]
    if len(ta.sparsetree) != len(tb.sparsetree):
        return 1
    var diff = 0
    for i in range(len(ta.sparsetree)):
        if ta.sparsetree[i].ColumnId() != tb.sparsetree[i].ColumnId():
            diff += 1
        if ta.sparsetree[i].QueryValue() != tb.sparsetree[i].QueryValue():
            diff += 1
    for i in range(len(ta.vector_leaf)):
        if ta.vector_leaf[i] != tb.vector_leaf[i]:
            diff += 1
    return diff


def arm_g_python_layer_params() raises -> Int:
    """`randomforest_common.pyx:520-536` and `internals/validation.py:73-79`.

    Four transforms cuML applies BETWEEN a user's arguments and the C++
    `RF_params`. They are not decoration: each one changes the model a
    user gets, and none of them is in their C++ layer, so a port that
    stops at the C-API shape silently drops all four.

    The expected values are arithmetic and written out by hand, not read
    back from the functions under test."""
    print()
    print("ARM G -- the Python layer's derived parameters")
    var wrong = 0

    # `:520-523` -- ceil(fraction * n_rows), NOT round and NOT truncate.
    # 0.1 * 400 = 40 exactly, so it must NOT become 41.
    var cases_leaf = [
        (0.1, 400, 40),
        (0.25, 401, 101),   # 100.25 -> 101
        (0.5, 401, 201),    # 200.5  -> 201, so it is not round-half-even
        (0.001, 100, 1),    # 0.1    -> 1, a fraction below one row
    ]
    for c in cases_leaf:
        var got = Int(min_samples_leaf_fraction(c[0], c[1]))
        print(
            "    min_samples_leaf_fraction(",
            c[0],
            ",",
            c[1],
            ") =",
            got,
            "want",
            c[2],
        )
        if got != c[2]:
            wrong += 1

    # `:524-527` -- the same, with max(2, ...) on top. Their
    # `validity_check` refuses below 2, so the floor is load-bearing.
    var cases_split = [(0.001, 100, 2), (0.1, 400, 40), (0.25, 401, 101)]
    for c in cases_split:
        var got = Int(min_samples_split_fraction(c[0], c[1]))
        print(
            "    min_samples_split_fraction(",
            c[0],
            ",",
            c[1],
            ") =",
            got,
            "want",
            c[2],
        )
        if got != c[2]:
            wrong += 1
    if Int(min_samples_leaf_fraction(0.001, 100)) != 1:
        wrong += 1
    else:
        print(
            "    ...and the floor is the only difference between them at"
            " 0.001 x 100: leaf 1, split 2"
        )

    # `validation.py:73-79` -- 0 and 2**32 - 1 in, -1 and 2**32 out.
    if check_random_seed(0) != UInt64(0):
        print("      FAIL: seed 0 rejected")
        wrong += 1
    if check_random_seed(4294967295) != UInt64(4294967295):
        print("      FAIL: seed 2**32 - 1 rejected")
        wrong += 1
    var refused = 0
    try:
        _ = check_random_seed(-1)
        print("      FAIL: seed -1 accepted")
        wrong += 1
    except:
        refused += 1
    try:
        _ = check_random_seed(4294967296)
        print("      FAIL: seed 2**32 accepted")
        wrong += 1
    except:
        refused += 1
    print(
        "    check_random_seed: 0 and 2**32-1 accepted,",
        refused,
        "of 2 out-of-range seeds refused -- this is WHY their one-round"
        " fnv1a32 fold of the seed is lossless",
    )

    if wrong == 0:
        print(
            "  arm G OK: ceil (not round, not truncate), the max(2, ...)"
            " floor, and the seed range that makes the fold lossless"
        )
    return wrong


def arm_h_n_bins_clamp(ctx: DeviceContext) raises -> Int:
    """`randomforest_common.pyx:529-536`. n_bins is clamped to n_rows, with
    a warning, INSIDE the fit -- the only place n_rows is known.

    Their C++ never checks it: `validity_check` only bounds n_bins to
    (0, 1024] (`decisiontree.cu:26-27`), so asking for 500 bins over 100
    rows passes every C-side gate and then asks the quantile pass for more
    bins than there are values to put in them.

    The clamp is observable two ways and this arm wants both: the params
    struct must come back MUTATED (theirs mutates too, which is what makes
    a later read see the corrected value), and the fit must still produce
    the same forest as asking for exactly n_rows bins in the first place.
    """
    print()
    print("ARM H -- n_bins is clamped to n_rows")
    var wrong = 0
    var d = _hashed_data(100, 4, 3)

    var p_big = _rf_params(2, False, Float32(1.0), max_n_bins=500)
    var f_big = _fit(ctx, d, p_big)
    print(
        "    asked for 500 bins over 100 rows -> params.max_n_bins is now",
        p_big.tree_params.max_n_bins,
    )
    if p_big.tree_params.max_n_bins != Int32(100):
        print("      FAIL: n_bins was not clamped to n_rows")
        wrong += 1

    var p_exact = _rf_params(2, False, Float32(1.0), max_n_bins=100)
    var f_exact = _fit(ctx, d, p_exact)
    if _trees_equal_across(f_big, f_exact, 0) != 0:
        print(
            "      FAIL: the clamped fit differs from asking for n_rows"
            " bins directly"
        )
        wrong += 1

    # and the clamp must not fire when it should not
    var p_small = _rf_params(2, False, Float32(1.0), max_n_bins=16)
    var f_small = _fit(ctx, d, p_small)
    if p_small.tree_params.max_n_bins != Int32(16):
        print("      FAIL: n_bins was clamped when it was already below n_rows")
        wrong += 1
    if _trees_equal_across(f_small, f_exact, 0) == 0:
        print(
            "      FAIL: 16 bins and 100 bins give the same forest, so the"
            " equality above proves nothing"
        )
        wrong += 1

    if wrong == 0:
        print(
            "  arm H OK: 500 bins over 100 rows became 100, gave the same"
            " forest as asking for 100, and 16 bins was left alone and"
            " gives a different forest"
        )
    return wrong


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
    fails += arm_f_bootstrapped_forest(ctx)
    fails += arm_g_python_layer_params()
    fails += arm_h_n_bins_clamp(ctx)
    if fails == 0:
        print("forest_check: ALL OK")
    else:
        raise Error("forest_check: " + String(fails) + " failure(s)")
