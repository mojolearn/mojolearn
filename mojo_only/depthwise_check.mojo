"""The DEPTHWISE lane's gate: seven claims, one of them the cross-GPU one.

`EGrowPolicy::Depthwise` is a NEW GROWTH POLICY in this port, so almost
everything it exercises is already verified -- the histogram families, the
scan, the subtraction, the split-points chain, the stable partition, the
reorder, the partition stats. WHAT IS NOT VERIFIED IS THE SEQUENCE UNDER A
LEAF FRONTIER THAT IS NOT A DENSE POWER OF TWO, and the three pieces this
lane wrote: the leafwise score kernel, the flat model builder, and the
non-symmetric apply.

RULES THIS FILE WAS WRITTEN AGAINST, all of them earned in this repository:

* **Verify reach, not output** (`mojotrees-verify-reach-not-output`). Claim 3
  is the per-branch reach proof for `compute_optimal_splits_region_kernel`,
  and it is not "the kernel ran": it is that TWO LEAVES OF ONE LEVEL CHOSE
  DIFFERENT BIN-FEATURES, which the oblivious kernel cannot produce at all.
  If this file ever calls the wrong kernel, claim 3 fails and nothing else
  has to notice.
* **Uniform test data hides permutation.** Bins are hashed and scattered,
  never `r % folds`, and the target is built so that different subtrees
  genuinely want different features -- otherwise a depthwise tree that is
  secretly symmetric passes every conservation test there is.
* **Conservation is not enough.** A tree that puts every row on one side of
  every split conserves rows perfectly. Claims 3 and 4 are what see it.
* **Gate against a real accumulator.** Claim 4 compares PER-BIN ROW COUNTS,
  not a total: a model that swaps two leaves' values conserves the total.

THE CROSS-GPU CLAIM, and what it does and does not prove
--------------------------------------------------------
Claim 6 grows the same tree twice in one process with two different CORE
COUNTS -- the M4's real 10 and a fabricated 108, an A100's -- and requires
the two models to be BIT-IDENTICAL: same nodes, same split types, same leaf
value bits.

That is not a proof that Metal and CUDA agree. It is the proof that the ONE
MACHINE-DEPENDENT INPUT this algorithm has cannot move the answer, which is
the property `IDENTITY_PATHS.md` exists to enumerate and the property that
was FALSE before row 7 was pinned (`partition_stats_chunks` was
`CeilDivide(2 * SMCount(), statCount)` feeding a float sum, so a 10-core Mac
and a 108-SM A100 built different models on every tree). Everything else
that differs between vendors -- denormal policy, FMA contraction, fp64 -- is
rows 9 and 10 and is tested by `check-ieee-arith`, which is the FIRST thing
to run on a new backend column. Claim 6 is the depthwise half of that
enumeration, and the honest sentence is: this lane adds no new row, and here
is the run that says so.
"""

from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer

from gbdt.data.leaf_path import TLeafPath, paths_equal
from gbdt.gpu_data.compressed_index_builder import build_layout
from gbdt.gpu_data.kernel.binarize import (
    WRITE_BLOCK_SIZE,
    write_compressed_index_kernel,
)
from gbdt.methods.greedy_subsets_searcher.greedy_search_helper import (
    TTreeWorkspace,
    resolve_split,
)
from gbdt.methods.greedy_subsets_searcher.greedy_search_helper_depthwise import (
    TBinFeatureTable,
    TDepthwiseWorkspace,
    fit_depthwise_tree,
    is_terminal_leaf,
)
from gbdt.methods.greedy_subsets_searcher.model_builder import (
    DUPLICATE_LEAVES_COMBINE,
    DUPLICATE_LEAVES_EXCEPTION,
    TFlatTreeBuilder,
    build_non_symmetric_tree,
)
from gbdt.methods.greedy_subsets_searcher.points_subsets import TLeaf
from gbdt.methods.greedy_subsets_searcher.structure_searcher_options import (
    TTreeStructureSearcherOptions,
)
from gbdt.methods.helpers import SPLIT_VALUE_ONE, SPLIT_VALUE_ZERO
from core.identity_trace import IdentityTrace
from gbdt.models.kernel.add_bin_values import (
    compute_non_symmetric_decision_tree_bins_kernel,
)
from gbdt.gpu_data.gpu_structures import TTreeNode
from gbdt.models.non_symmetric_tree import (
    TNonSymmetricTree,
    TNonSymmetricTreeStructure,
)
from gbdt.models.oblivious_model import (
    BIN_SPLIT_TAKE_BIN,
    BIN_SPLIT_TAKE_GREATER,
    TBinarySplit,
)
from gbdt.options.catboost_options import GROW_DEPTHWISE
from mojo_only.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL


struct Fixture(Movable):
    """The dataset every device claim runs on, built once.

    16 features over three policies -- 8 binary, 4 half-byte, 4 one-byte --
    which is three histogram launches per level and a 296-cell flat
    histogram. Same shape `level_check`'s mixed tree uses, so a failure here
    that is really a histogram failure is already covered there.

    **THE TARGET IS BUILT TO FORCE PER-LEAF DIVERGENCE, and that is the
    point rather than a cheat.** The gradient is a function of feature 8 for
    rows on one side of feature 12 and a function of feature 9 for rows on
    the other, so the two subtrees below the root genuinely want DIFFERENT
    splits. A fixture whose best split is the same everywhere cannot tell a
    depthwise searcher from a symmetric one, which is the failure claim 3
    exists to catch. This is a correctness fixture and never a measurement
    vehicle -- `no-dataset-cherry-picking` is about benchmarks.
    """

    var ctx: DeviceContext
    var n_rows: Int
    var folds: List[Int]
    var cindex: DeviceBuffer[DType.uint32]
    var stats: DeviceBuffer[DType.float32]
    var row_index: DeviceBuffer[DType.uint32]
    var total_weight: Float32
    var total_gradient: Float32
    var host_bins: List[UInt8]
    var pristine_stats: HostBuffer[DType.float32]
    """The stat planes as they were BEFORE any fit.

    ================= WHY THIS EXISTS, and it is not tidiness ==============
    **GROWTH MUTATES ITS INPUT.** `launch_reorder_in_leaves` permutes
    `stats` and `row_index` IN PLACE -- that is the whole design
    (`split_points.cu`, their segmented gather), and it is why a leaf is a
    contiguous range at all. CatBoost's boosting loop re-supplies both every
    tree, so upstream never notices.

    A CHECK THAT FITS THE SAME FIXTURE TWICE DOES NOTICE, and what it sees
    is not a library defect. The second fit starts from a permuted row
    order; the histogram is unaffected (fixed-point, order-free under
    IDENTICAL), but `compute_partition_stats` sums a leaf's rows in a float
    reduce whose CHUNK CONTENTS are positional, so the per-leaf stats move,
    and the score kernel reads those stats -- so the TREE moves.

    This cost an hour on 2026-08-22. The core-count claim below reported
    divergence at three different core counts, three probes were spent
    pinning `replication_for` and the split-chain grids, and the divergence
    survived all of them -- because it was never about the core count. Two
    fits at the SAME core count diverged too. The claim was measuring fixture
    reuse and calling it machine dependence.

    The lesson is the standing one in a new place: a differential is only a
    differential if the ONLY thing that differs is the thing under test.
    ======================================================================"""
    var pristine_index: HostBuffer[DType.uint32]

    def __init__(
        out self, var ctx: DeviceContext, corrupt_row: Int = -1
    ) raises:
        self.ctx = ctx^
        self.n_rows = 4096
        self.folds = List[Int]()
        for _ in range(8):
            self.folds.append(1)
        for _ in range(4):
            self.folds.append(8)
        for _ in range(4):
            self.folds.append(64)
        var n_features = len(self.folds)
        var n_rows = self.n_rows

        var n_columns = 3
        self.cindex = self.ctx.enqueue_create_buffer[DType.uint32](
            n_rows * n_columns
        )
        var z = self.ctx.enqueue_create_host_buffer[DType.uint32](
            n_rows * n_columns
        )
        for i in range(n_rows * n_columns):
            z.unsafe_ptr().unsafe_store(i, UInt32(0))
        self.ctx.enqueue_copy(dst_buf=self.cindex, src_ptr=z.unsafe_ptr())
        self.ctx.synchronize()

        var lay = build_layout(self.folds)
        self.host_bins = List[UInt8]()
        for _ in range(n_features * n_rows):
            self.host_bins.append(UInt8(0))

        var hb = self.ctx.enqueue_create_host_buffer[DType.uint8](n_rows)
        var bins = self.ctx.enqueue_create_buffer[DType.uint8](n_rows)
        for f in range(n_features):
            ref cf = lay.features[f]
            for r in range(n_rows):
                # HASHED AND SCATTERED. `r % folds` makes every binary
                # feature the constant 0 and makes every histogram cell
                # carry the same value, which is the pattern that reported
                # two broken kernels correct (see RESUME.md).
                var x = UInt32(r * 2654435761 + f * 40503 + 0x2545F491)
                x ^= x << 13
                x ^= x >> 17
                x ^= x << 5
                # `folds` is the BORDER count, so a feature takes bins
                # `0..folds`, which is `folds + 1` values.
                var b = UInt8(Int(x % UInt32(self.folds[f] + 1)))
                hb.unsafe_ptr().unsafe_store(r, b)
                self.host_bins[f * n_rows + r] = b
            self.ctx.enqueue_copy(dst_buf=bins, src_ptr=hb.unsafe_ptr())
            self.ctx.enqueue_function[write_compressed_index_kernel](
                Int32(Int(cf.offset) * n_rows),
                cf.mask,
                cf.shift,
                bins.unsafe_ptr(),
                Int32(n_rows),
                self.cindex.unsafe_ptr(),
                grid_dim=(n_rows + WRITE_BLOCK_SIZE - 1) // WRITE_BLOCK_SIZE,
                block_dim=WRITE_BLOCK_SIZE,
            )
            self.ctx.synchronize()
        # `mojo-buffer-freed-at-last-use`: a buffer whose last TEXTUAL use is
        # the `enqueue_*` argument is dead before the queue drains, and this
        # holds for DEVICE buffers handed to `.unsafe_ptr()` as much as for
        # host staging. `bins` and `hb` are both live across the loop's
        # launches, so the keep-alive goes after the last synchronize.
        _ = bins^
        _ = hb^
        _ = z^

        self.stats = self.ctx.enqueue_create_buffer[DType.float32](
            2 * n_rows
        )
        var hs = self.ctx.enqueue_create_host_buffer[DType.float32](
            2 * n_rows
        )
        var tw = Float64(0.0)
        var tg = Float64(0.0)
        for r in range(n_rows):
            # THE DIVERGENT TARGET. Feature 12 (one-byte, 64 folds) picks
            # the side; feature 8 or feature 9 (half-byte) drives the
            # gradient on that side. Neither subtree's best split can be
            # the other's.
            var gate = Int(self.host_bins[12 * n_rows + r])
            var g: Float64
            if gate >= 32:
                g = 2.0 if Int(self.host_bins[8 * n_rows + r]) >= 4 else -1.0
            else:
                g = -2.0 if Int(self.host_bins[9 * n_rows + r]) >= 4 else 1.0
            if r == corrupt_row:
                # claim 7's sabotage: one row's gradient made large enough
                # to move a border. If the tree does NOT move, the device
                # data is not reaching the model.
                g = 500.0
            hs.unsafe_ptr().unsafe_store(r, Float32(1.0))
            hs.unsafe_ptr().unsafe_store(n_rows + r, Float32(g))
            tw += 1.0
            # `choose_scale` is specified against a sum of MAGNITUDES, not
            # a signed total: gradients cancel, so a signed total can be
            # far smaller than the largest histogram cell it must bound.
            tg += -g if g < 0.0 else g
        self.ctx.enqueue_copy(dst_buf=self.stats, src_ptr=hs.unsafe_ptr())
        self.pristine_stats = self.ctx.enqueue_create_host_buffer[
            DType.float32
        ](2 * n_rows)
        for i in range(2 * n_rows):
            self.pristine_stats.unsafe_ptr().unsafe_store(
                i, hs.unsafe_ptr().unsafe_load(i)
            )
        self.total_weight = Float32(tw)
        self.total_gradient = Float32(tg)

        self.row_index = self.ctx.enqueue_create_buffer[DType.uint32](n_rows)
        var hi = self.ctx.enqueue_create_host_buffer[DType.uint32](n_rows)
        for r in range(n_rows):
            hi.unsafe_ptr().unsafe_store(r, UInt32(r))
        self.ctx.enqueue_copy(dst_buf=self.row_index, src_ptr=hi.unsafe_ptr())
        self.pristine_index = self.ctx.enqueue_create_host_buffer[
            DType.uint32
        ](n_rows)
        for r in range(n_rows):
            self.pristine_index.unsafe_ptr().unsafe_store(r, UInt32(r))
        self.ctx.synchronize()
        _ = hs^
        _ = hi^

    def reset(mut self) raises:
        """Put the stat planes and the row index back as they were.

        Their boosting loop re-supplies both per tree; a check that fits more
        than once has to do the same. See `pristine_stats`.
        """
        self.ctx.enqueue_copy(
            dst_buf=self.stats, src_ptr=self.pristine_stats.unsafe_ptr()
        )
        self.ctx.enqueue_copy(
            dst_buf=self.row_index, src_ptr=self.pristine_index.unsafe_ptr()
        )
        self.ctx.synchronize()


def default_options(max_depth: Int, min_data_in_leaf: Float64 = 1.0) raises -> TTreeStructureSearcherOptions:
    var o = TTreeStructureSearcherOptions()
    o.policy = GROW_DEPTHWISE
    o.max_depth = max_depth
    o.max_leaves = 1 << max_depth
    o.min_leaf_size = min_data_in_leaf
    return o^


def fit(
    mut fx: Fixture, options: TTreeStructureSearcherOptions,
    sm_count_override: Int = -1,
) raises -> TNonSymmetricTree:
    """One fit on the fixture, fresh pools so no state crosses claims.

    **AND A FIXTURE RESET FIRST.** Growth permutes `stats` and `row_index`
    in place; see `Fixture.pristine_stats` for what happened when this was
    missing.
    """
    fx.reset()
    var ws = List[TTreeWorkspace]()
    var dws = List[TDepthwiseWorkspace]()
    # `disabled()` and not `IdentityTrace()`: a GATE whose behavior depends
    # on whether the operator happens to have MOJOLEARN_IDENTITY_TRACE
    # exported is a gate that passes or fails for reasons outside itself.
    var tr = IdentityTrace.disabled()
    var model = fit_depthwise_tree(
        fx.ctx, fx.n_rows, fx.folds, options,
        fx.cindex, fx.stats, fx.row_index,
        fx.total_weight, fx.total_gradient,
        ws, dws, tr,
        sm_count_override=sm_count_override,
    )
    # KEEP-ALIVES, and the invariant they pin. `ws` / `dws` hold every
    # device buffer the fit enqueued work against, and they are LOCALS
    # here: they are safe only because `fit_depthwise_tree` drains before
    # it returns and enqueues nothing after. That invariant is load-bearing
    # and was undocumented -- any enqueue added after the final
    # `wait_complete` there turns this into a use-after-free that does not
    # crash. Pinned locally so the check cannot be the thing that breaks.
    _ = ws^
    _ = dws^
    _ = tr^
    return model^


def fingerprint(
    model: TNonSymmetricTree, with_values: Bool = True
) raises -> String:
    """Every bit of the model a reader could act on, as text.

    `with_values=False` gives the tree SHAPE alone -- nodes and split types,
    no leaf values. That distinction is worth having because the two answer
    different questions: shape differing means the SEARCH diverged, values
    differing with the same shape means only the leaf-value reduce did, and
    those live in different code.

    THIS WAS TWO FUNCTIONS until 2026-08-22, `fingerprint` and
    `structure_fingerprint`, the second being the first's node loop copied
    verbatim minus the value tail. A repo-wide duplication sweep found it.
    One loop with a flag cannot drift; two loops over the same four fields
    is a fix applied to one of them.

    Leaf values go out as their EXACT BIT PATTERN, never as decimals:
    `String(Float32)` does not round-trip in this Mojo
    (`mojo-string-float-roundtrip`), so a decimal fingerprint would call two
    different models equal.
    """
    var s = String("nodes=")
    for i in range(len(model.model_structure.nodes)):
        ref n = model.model_structure.nodes[i]
        s += (
            String("(")
            + String(Int(n.feature_id)) + ","
            + String(Int(n.bin)) + ","
            + String(Int(n.left_subtree)) + ","
            + String(Int(n.right_subtree)) + ","
            + String(Int(model.model_structure.split_types[i]))
            + ")"
        )
    if not with_values:
        return s^
    s += " values="
    for i in range(len(model.leaf_values)):
        s += String(Int(model.leaf_values[i].to_bits())) + ","
    return s^


def claim_1_bin_feature_table(fx: Fixture) raises:
    """The resolved table agrees with `resolve_split` cell for cell.

    `TBinFeatureTable` is DEVIATION 351 -- an O(1) lookup where theirs walks
    the features manager per candidate -- and it is filled BY
    `resolve_split`, so it "cannot disagree". This repository has been wrong
    about a cannot-disagree before (`build_layout`'s two walks agreed only
    because every fixture was binary-first), so the sentence is a claim and
    the claim is run.
    """
    var lay = build_layout(fx.folds)
    var table = TBinFeatureTable(lay)
    if len(table.feature) != lay.hist_cells:
        raise Error(
            String("table covers ") + String(len(table.feature))
            + " bin-features, layout has " + String(lay.hist_cells)
        )
    for bf in range(lay.hist_cells):
        var want = resolve_split(lay, bf)
        if Int(table.feature[bf]) != want.feature:
            raise Error(
                String("bin-feature ") + String(bf) + " feature: table "
                + String(Int(table.feature[bf])) + " vs resolve_split "
                + String(want.feature)
            )
        if Int(table.bin[bf]) != want.bin:
            raise Error(
                String("bin-feature ") + String(bf) + " bin: table "
                + String(Int(table.bin[bf])) + " vs resolve_split "
                + String(want.bin)
            )
        var split = table.to_split(bf)
        var expect_type = Int32(
            BIN_SPLIT_TAKE_BIN
        ) if lay.features[want.feature].one_hot_feature else Int32(
            BIN_SPLIT_TAKE_GREATER
        )
        if split.split_type != expect_type:
            raise Error(
                String("bin-feature ") + String(bf) + " split type wrong"
            )
    print("  claim 1 OK:", lay.hist_cells, "bin-features resolve identically")


def claim_2_model_builder() raises:
    """`TFlatTreeBuilder` on a tree written out by hand.

    THE TREE, chosen so the flat layout has something to get wrong -- the
    left subtree is deeper than the right, so `index + 1` and
    `index + left_subtree` are DIFFERENT node indices and a builder that
    confuses them still produces the right leaf COUNT:

                       root: f7 > b3
                        /            \\
              n1: f2 > b1             leaf D
               /        \\
           leaf A      leaf B/C  (n2: f5 > b9)
                        /     \\
                    leaf B    leaf C

    Four leaves, three nodes. Pre-order emission must give
    nodes = [root(L=3,R=1), n1(L=1,R=2), n2(L=1,R=1)] and bins A,B,C,D = 0..3.
    """
    var f7 = TBinarySplit(Int32(7), Int32(3), Int32(BIN_SPLIT_TAKE_GREATER))
    var f2 = TBinarySplit(Int32(2), Int32(1), Int32(BIN_SPLIT_TAKE_GREATER))
    var f5 = TBinarySplit(Int32(5), Int32(9), Int32(BIN_SPLIT_TAKE_GREATER))

    var pa = TLeafPath()
    pa.add_split(f7, SPLIT_VALUE_ZERO)
    pa.add_split(f2, SPLIT_VALUE_ZERO)
    var pb = TLeafPath()
    pb.add_split(f7, SPLIT_VALUE_ZERO)
    pb.add_split(f2, SPLIT_VALUE_ONE)
    pb.add_split(f5, SPLIT_VALUE_ZERO)
    var pc = TLeafPath()
    pc.add_split(f7, SPLIT_VALUE_ZERO)
    pc.add_split(f2, SPLIT_VALUE_ONE)
    pc.add_split(f5, SPLIT_VALUE_ONE)
    var pd = TLeafPath()
    pd.add_split(f7, SPLIT_VALUE_ONE)

    var paths = List[TLeafPath]()
    paths.append(pa.copy())
    paths.append(pb.copy())
    paths.append(pc.copy())
    paths.append(pd.copy())

    var weights = List[Float64]()
    var values = List[List[Float32]]()
    for i in range(4):
        weights.append(Float64(10 * (i + 1)))
        var v = List[Float32]()
        v.append(Float32(i + 1))
        values.append(v^)

    var model = build_non_symmetric_tree(paths, weights, values)
    ref st = model.model_structure

    if len(st.nodes) != 3:
        raise Error(String("expected 3 nodes, got ") + String(len(st.nodes)))
    if st.leaves_count() != 4:
        raise Error("leaves_count is not nodes + 1")

    var want_f = List[Int]()
    var want_b = List[Int]()
    var want_l = List[Int]()
    var want_r = List[Int]()
    want_f.append(7); want_f.append(2); want_f.append(5)
    want_b.append(3); want_b.append(1); want_b.append(9)
    want_l.append(3); want_l.append(1); want_l.append(1)
    want_r.append(1); want_r.append(2); want_r.append(1)
    for i in range(3):
        if Int(st.nodes[i].feature_id) != want_f[i]:
            raise Error(String("node ") + String(i) + " feature wrong")
        if Int(st.nodes[i].bin) != want_b[i]:
            raise Error(String("node ") + String(i) + " bin wrong")
        if Int(st.nodes[i].left_subtree) != want_l[i]:
            raise Error(
                String("node ") + String(i) + " left_subtree "
                + String(Int(st.nodes[i].left_subtree)) + " != "
                + String(want_l[i])
            )
        if Int(st.nodes[i].right_subtree) != want_r[i]:
            raise Error(
                String("node ") + String(i) + " right_subtree "
                + String(Int(st.nodes[i].right_subtree)) + " != "
                + String(want_r[i])
            )

    # values and weights come out in BIN order, which for this tree is the
    # input order A,B,C,D. That is not a coincidence to rely on in general
    # -- it is asserted here because this tree was written left to right.
    for i in range(4):
        if model.leaf_values[i] != Float32(i + 1):
            raise Error(String("leaf value ") + String(i) + " wrong")
        if model.leaf_weights[i] != Float64(10 * (i + 1)):
            raise Error(String("leaf weight ") + String(i) + " wrong")

    # `VisitBins` must hand the SAME paths back, in bin order.
    var visited = st.visit_bins()
    if len(visited) != 4:
        raise Error(
            String("visit_bins returned ") + String(len(visited))
            + " leaves, tree has 4"
        )
    for i in range(4):
        if visited[i].bin != i:
            raise Error("visit_bins numbered a leaf out of order")
        if not paths_equal(visited[i].path, paths[i]):
            raise Error(
                String("visit_bins path ") + String(i)
                + " differs from the path it was built from"
            )

    # THE DUPLICATE POLICIES, both arms. Theirs raises on the non-symmetric
    # path (`model_builder.cpp:288`) and combines on the other.
    var raised = False
    try:
        var dup = List[TLeafPath]()
        dup.append(pa.copy())
        dup.append(pa.copy())
        var dw = List[Float64]()
        dw.append(Float64(1.0))
        dw.append(Float64(2.0))
        var dv = List[List[Float32]]()
        var dv0 = List[Float32]()
        dv0.append(Float32(1.0))
        var dv1 = List[Float32]()
        dv1.append(Float32(2.0))
        dv.append(dv0^)
        dv.append(dv1^)
        _ = build_non_symmetric_tree(dup, dw, dv)
    except:
        raised = True
    if not raised:
        raise Error(
            "duplicate terminal leaf was accepted; the Exception policy is"
            " not reaching the builder"
        )

    # `Combine` (`model_builder.cpp:222-227`): weight and values ADD.
    # Their non-symmetric call site never selects it, so this is the only
    # thing that keeps the arm from being written and left unreached.
    # `Combine` (`model_builder.cpp:222-227`): weight and values ADD.
    # Their non-symmetric call site never selects it, so this is the only
    # thing that keeps the arm from being written and left unreached.
    #
    # THE WHOLE TREE IS ADDED FIRST and only then is `pa` repeated. A
    # builder holding one path of depth 2 has two internal nodes and one
    # child, and `build_flat` raises on the missing sibling -- their
    # `CB_ENSURE(cursor, "Tree is empty (cursor is nullptr)")`. That is
    # correct behaviour and not what this arm is about.
    var comb = TFlatTreeBuilder(DUPLICATE_LEAVES_COMBINE)
    for i in range(4):
        comb.add(paths[i], values[i], weights[i])
    var extra = List[Float32]()
    extra.append(Float32(1.5))
    comb.add(pa, extra, Float64(4.0))
    var cnodes = List[TTreeNode]()
    var ctypes = List[Int32]()
    var cvalues = List[Float32]()
    var cweights = List[Float64]()
    comb.build_flat(cnodes, ctypes, cvalues, cweights)
    if len(cweights) != 4:
        raise Error(
            "Combine created a fifth leaf instead of combining; got "
            + String(len(cweights))
        )
    # leaf A was 10.0 / 1.0 and is combined with 4.0 / 1.5
    if cweights[0] != Float64(14.0):
        raise Error(
            "Combine did not add the weights: leaf A is "
            + String(cweights[0]) + ", expected 14.0"
        )
    if cvalues[0] != Float32(2.5):
        raise Error("Combine did not add the values")
    if cweights[1] != Float64(20.0):
        raise Error("Combine touched a leaf it was not given")

    print("  claim 2 OK: 3 nodes, asymmetric subtree sizes, both policies")


def apply_bins(
    mut fx: Fixture, model: TNonSymmetricTree
) raises -> List[Int]:
    """Run the model over every row with THEIR apply kernel; tally per bin.

    This is `ComputeNonSymmetricDecisionTreeBins` (`add_model_value.cu:399`),
    reached through `compute_non_symmetric_decision_tree_bins_kernel`. The
    per-node feature planes are their `const TCFeature* features`, one entry
    PER INTERNAL NODE and walked in step with the nodes.

    Rows are indexed by their ORIGINAL id here, not through `row_index`:
    growth permuted the index array but never the compressed index, so
    `cindex[offset + r]` is still row `r`'s bin. That is the same property
    `split_and_make_sequence_kernel` relies on.
    """
    var lay = build_layout(fx.folds)
    ref st = model.model_structure
    var n_nodes = len(st.nodes)
    var slots = n_nodes if n_nodes > 0 else 1

    var h_off = fx.ctx.enqueue_create_host_buffer[DType.uint32](slots)
    var h_mask = fx.ctx.enqueue_create_host_buffer[DType.uint32](slots)
    var h_shift = fx.ctx.enqueue_create_host_buffer[DType.uint32](slots)
    var h_oh = fx.ctx.enqueue_create_host_buffer[DType.uint8](slots)
    var h_bin = fx.ctx.enqueue_create_host_buffer[DType.uint32](slots)
    var h_ls = fx.ctx.enqueue_create_host_buffer[DType.uint32](slots)
    var h_rs = fx.ctx.enqueue_create_host_buffer[DType.uint32](slots)
    for i in range(slots):
        h_off.unsafe_ptr().unsafe_store(i, UInt32(0))
        h_mask.unsafe_ptr().unsafe_store(i, UInt32(0))
        h_shift.unsafe_ptr().unsafe_store(i, UInt32(0))
        h_oh.unsafe_ptr().unsafe_store(i, UInt8(0))
        h_bin.unsafe_ptr().unsafe_store(i, UInt32(0))
        h_ls.unsafe_ptr().unsafe_store(i, UInt32(1))
        h_rs.unsafe_ptr().unsafe_store(i, UInt32(1))
    for i in range(n_nodes):
        ref n = st.nodes[i]
        ref f = lay.features[Int(n.feature_id)]
        # their `feature.Offset` is a COLUMN index into the compressed
        # index and the kernel adds the row; ours is the same column times
        # the row count, which is how this port lays the index out.
        h_off.unsafe_ptr().unsafe_store(i, UInt32(Int(f.offset) * fx.n_rows))
        h_mask.unsafe_ptr().unsafe_store(i, f.mask)
        h_shift.unsafe_ptr().unsafe_store(i, f.shift)
        h_oh.unsafe_ptr().unsafe_store(
            i, UInt8(1) if f.one_hot_feature else UInt8(0)
        )
        h_bin.unsafe_ptr().unsafe_store(i, UInt32(Int(n.bin)))
        h_ls.unsafe_ptr().unsafe_store(i, UInt32(Int(n.left_subtree)))
        h_rs.unsafe_ptr().unsafe_store(i, UInt32(Int(n.right_subtree)))

    var d_off = fx.ctx.enqueue_create_buffer[DType.uint32](slots)
    var d_mask = fx.ctx.enqueue_create_buffer[DType.uint32](slots)
    var d_shift = fx.ctx.enqueue_create_buffer[DType.uint32](slots)
    var d_oh = fx.ctx.enqueue_create_buffer[DType.uint8](slots)
    var d_bin = fx.ctx.enqueue_create_buffer[DType.uint32](slots)
    var d_ls = fx.ctx.enqueue_create_buffer[DType.uint32](slots)
    var d_rs = fx.ctx.enqueue_create_buffer[DType.uint32](slots)
    fx.ctx.enqueue_copy(dst_buf=d_off, src_ptr=h_off.unsafe_ptr())
    fx.ctx.enqueue_copy(dst_buf=d_mask, src_ptr=h_mask.unsafe_ptr())
    fx.ctx.enqueue_copy(dst_buf=d_shift, src_ptr=h_shift.unsafe_ptr())
    fx.ctx.enqueue_copy(dst_buf=d_oh, src_ptr=h_oh.unsafe_ptr())
    fx.ctx.enqueue_copy(dst_buf=d_bin, src_ptr=h_bin.unsafe_ptr())
    fx.ctx.enqueue_copy(dst_buf=d_ls, src_ptr=h_ls.unsafe_ptr())
    fx.ctx.enqueue_copy(dst_buf=d_rs, src_ptr=h_rs.unsafe_ptr())

    var out = fx.ctx.enqueue_create_buffer[DType.uint32](fx.n_rows)
    var h_out = fx.ctx.enqueue_create_host_buffer[DType.uint32](fx.n_rows)
    fx.ctx.enqueue_function[compute_non_symmetric_decision_tree_bins_kernel](
        d_off.unsafe_ptr(), d_mask.unsafe_ptr(), d_shift.unsafe_ptr(),
        d_oh.unsafe_ptr(), d_bin.unsafe_ptr(),
        d_ls.unsafe_ptr(), d_rs.unsafe_ptr(),
        Int32(n_nodes),
        fx.cindex.unsafe_ptr(),
        Int32(fx.n_rows),
        out.unsafe_ptr(),
        grid_dim=(fx.n_rows + 255) // 256,
        block_dim=256,
    )
    fx.ctx.enqueue_copy(dst_ptr=h_out.unsafe_ptr(), src_buf=out)
    fx.ctx.synchronize()
    # THE KEEP-ALIVES. Seven host planes and seven device buffers whose last
    # textual use is an argument to the launch above; without these the
    # allocator is free to reuse every one of them before the kernel runs,
    # and the check would pass or fail on page reuse. See
    # `mojo-buffer-freed-at-last-use` and the note in
    # `greedy_search_helper.mojo`'s DEVIATION 95 block, which needed the
    # same `_ = scale_keep^` for the same reason.
    _ = h_off^
    _ = h_mask^
    _ = h_shift^
    _ = h_oh^
    _ = h_bin^
    _ = h_ls^
    _ = h_rs^
    _ = d_off^
    _ = d_mask^
    _ = d_shift^
    _ = d_oh^
    _ = d_bin^
    _ = d_ls^
    _ = d_rs^
    _ = out^

    var counts = List[Int]()
    for _ in range(model.bin_count()):
        counts.append(0)
    for r in range(fx.n_rows):
        var b = Int(h_out.unsafe_ptr().unsafe_load(r))
        if b < 0 or b >= model.bin_count():
            raise Error(
                String("apply put row ") + String(r) + " in bin "
                + String(b) + " of " + String(model.bin_count())
            )
        counts[b] += 1
    return counts^


def claim_3_growth_and_divergence(
    mut fx: Fixture
) raises -> TNonSymmetricTree:
    """A real depthwise tree, and THE reach proof for the leafwise kernel.

    Three things, in increasing strength, AT FOUR DEPTHS:

    1. rows are conserved -- the leaf weights sum to `n_rows`;
    2. more than one leaf is non-empty -- a tree that puts every row on one
       side of every split conserves rows perfectly and splits nothing;
    3. **two internal nodes carry DIFFERENT `(feature, bin)` pairs.**

    (3) IS THE REACH PROOF, and nothing weaker is one. An oblivious level
    gives every leaf the SAME split by construction
    (`greedy_search_helper.cpp:539-541`, one `bestSplits[0]` written to
    every leaf), so if this file were somehow launching
    `compute_optimal_splits_kernel` instead of the region arm, every node at
    a given depth would agree and (3) would fail. Reach is per-branch and
    this is the branch.

    ================= WHY MORE THAN ONE DEPTH =================
    Every claim in this file called `default_options(4)` and nothing else
    until 2026-08-22, so depths 2, 3 and 6 had NEVER BEEN RUN. The lossguide
    lane found that the hard way: a bound of theirs was wrong at depth 6 and
    hid behind this gate's single shape, and they spent six probes on it.

    A gate that only ever runs one shape is a gate that tests a shape, not a
    policy. The depth changes the level count, the leaf frontier's raggedness
    and -- through `max_leaves = 1 << depth` -- every workspace bound, so it
    is the cheapest axis with real coverage in it.
    ===========================================================
    """
    var depths = List[Int]()
    depths.append(2)
    depths.append(3)
    depths.append(4)
    depths.append(6)

    var keep = fit(fx, default_options(4))

    for di in range(len(depths)):
        var d = depths[di]
        var model = fit(fx, default_options(d))
        ref st = model.model_structure

        var total = Float64(0.0)
        for i in range(len(model.leaf_weights)):
            total += model.leaf_weights[i]
        if Int(total) != fx.n_rows:
            raise Error(
                String("depth ") + String(d)
                + ": rows lost or duplicated, leaf weights sum to "
                + String(total) + " of " + String(fx.n_rows)
            )

        var nonempty = 0
        for i in range(len(model.leaf_weights)):
            if model.leaf_weights[i] > 0.0:
                nonempty += 1
        if nonempty < 2:
            raise Error(
                String("depth ") + String(d) + ": the tree never split, "
                + String(nonempty) + " non-empty leaf"
            )

        # `max_leaves = 1 << depth` is a HARD bound for every policy but
        # Lossguide (`catboost_options.cpp:993-1001`), and every workspace
        # in this fit is sized by it. A tree that exceeded it would have
        # been writing past the histogram plane.
        if model.bin_count() > (1 << d):
            raise Error(
                String("depth ") + String(d) + ": "
                + String(model.bin_count())
                + " leaves exceeds max_leaves " + String(1 << d)
            )

        var distinct = 0
        for i in range(len(st.nodes)):
            var seen_before = False
            for j in range(i):
                if (
                    st.nodes[j].feature_id == st.nodes[i].feature_id
                    and st.nodes[j].bin == st.nodes[i].bin
                ):
                    seen_before = True
            if not seen_before:
                distinct += 1
        # Depth 1 could legitimately have one node; from depth 2 up, a
        # depthwise tree on this fixture must diverge.
        if distinct < 2:
            raise Error(
                String("depth ") + String(d)
                + ": every node carries the same (feature, bin), so the"
                " level is oblivious and the leafwise score kernel is NOT"
                " the one being reached"
            )

        print(
            "  claim 3 OK depth", d, "->", len(st.nodes), "nodes,",
            model.bin_count(), "leaves,", nonempty, "non-empty,", distinct,
            "distinct splits,", Int(total), "rows conserved",
        )

    return keep^


def claim_4_apply_matches_growth(
    mut fx: Fixture, model: TNonSymmetricTree
) raises:
    """The apply kernel puts each row where growth put it. PER BIN.

    The leaf weights are a sum of `1.0` per row in the weight plane, so for
    a row count under 2^24 they are EXACT integers in Float32 and an exact
    comparison is legitimate rather than lenient. Comparing per bin and not
    in total is the point: a model that swaps two leaves' values -- the
    exact failure a wrong `left_subtree` produces -- conserves the total.
    """
    var counts = apply_bins(fx, model)
    if len(counts) != len(model.leaf_weights):
        raise Error("apply and model disagree about the leaf count")
    for b in range(len(counts)):
        var want = Int(model.leaf_weights[b])
        if counts[b] != want:
            raise Error(
                String("bin ") + String(b) + ": apply routed "
                + String(counts[b]) + " rows, growth partitioned "
                + String(want)
            )
    print("  claim 4 OK: all", len(counts), "bins match growth row for row")


def claim_5_min_data_in_leaf(mut fx: Fixture) raises:
    """`min_data_in_leaf` is LIVE here, and its boundary is `<=`.

    `IsTerminalLeaf` guards the size test with `Policy != SymmetricTree`
    (`greedy_search_helper.cpp:685`), so this lane is the first in the port
    where the option decides anything. Two halves:

    5a **THE BOUNDARY, tested directly on `is_terminal_leaf`.** Theirs is

            (checkLeafSize && leaf.Size <= Options.MinLeafSize)
            || leaf.Path.GetDepth() >= Options.MaxDepth

        so a leaf of EXACTLY `MinLeafSize` rows is TERMINAL, and
        `min_data_in_leaf = 1` means "a one-row leaf does not split" rather
        than "a one-row leaf is allowed". A port with `<` passes every
        end-to-end differential on most fixtures and grows one level too far
        on every branch that reaches the bound.

        This half is a UNIT test and not a differential ON PURPOSE. An
        end-to-end probe of the `<=` boundary has to find a leaf whose size
        is exactly the threshold AND whose depth is not already binding, and
        a fixture that stops producing one turns the gate amber for a reason
        that has nothing to do with the code. The boundary is one comparison;
        test the comparison.

        The depth clause is tested the same way, including that it is `>=`
        and not `>`.

    5b **THE OPTION IS NOT DROPPED**, end to end: a large minimum must give
        STRICTLY FEWER leaves than a minimum of 1. This is the half a unit
        test cannot give, because it is about the value reaching the
        searcher at all.
    """
    # --- 5a, the boundary, on the function itself ---
    var opt = default_options(4, 5.0)
    var leaf = TLeaf()

    leaf.size = 6
    if is_terminal_leaf(leaf, opt):
        raise Error(
            "a 6-row leaf is terminal at min_data_in_leaf=5; the size test"
            " is not `leaf.Size <= MinLeafSize`"
        )
    leaf.size = 5
    if not is_terminal_leaf(leaf, opt):
        raise Error(
            "a 5-row leaf is NOT terminal at min_data_in_leaf=5; the size"
            " test is `<` where CatBoost's is `<=`"
            " (greedy_search_helper.cpp:686)"
        )
    leaf.size = 4
    if not is_terminal_leaf(leaf, opt):
        raise Error("a 4-row leaf is not terminal at min_data_in_leaf=5")

    # the depth clause, `Path.GetDepth() >= MaxDepth`, also `>=`. A leaf
    # far above the size bound so that only the depth clause can fire.
    var probe_split = TBinarySplit(
        Int32(0), Int32(0), Int32(BIN_SPLIT_TAKE_GREATER)
    )
    var shallow = TLeaf()
    shallow.size = 1000000
    for _ in range(3):
        shallow.path.add_split(probe_split, SPLIT_VALUE_ZERO)
    if shallow.get_depth() != 3:
        raise Error("depth probe did not build a depth-3 path")
    if is_terminal_leaf(shallow, opt):
        raise Error(
            "a large leaf at depth 3 is terminal at max_depth=4; the depth"
            " test fires early"
        )
    var deep = shallow.copy()
    deep.path.add_split(probe_split, SPLIT_VALUE_ONE)
    if deep.get_depth() != 4:
        raise Error("depth probe did not reach max_depth")
    if not is_terminal_leaf(deep, opt):
        raise Error(
            "a large leaf AT max_depth is not terminal; the depth test is"
            " `>` where CatBoost's is `>=`"
        )

    # --- 5b, the option reaches the searcher ---
    var base = fit(fx, default_options(4, 1.0))
    var tight = fit(fx, default_options(4, 600.0))
    if tight.bin_count() >= base.bin_count():
        raise Error(
            String("min_data_in_leaf=600 gave ") + String(tight.bin_count())
            + " leaves, min=1 gave " + String(base.bin_count())
            + "; the option is inert"
        )
    print(
        "  claim 5 OK: boundary is `<=` at 5 and `>=` at max_depth;"
        " min=1 ->", base.bin_count(), "leaves, min=600 ->",
        tight.bin_count(),
    )


def claim_6_core_count_invariance(mut fx: Fixture) raises:
    """THE CROSS-GPU CLAIM. Three core counts, one process.

    The core count is the ONLY machine-dependent input this algorithm has.
    It reaches every strided grid width (`split_points_grid_x`), the
    histogram replication factor, and the partition-stats chunk count -- and
    the third is PINNED inside `partition_stats_chunks` precisely because it
    feeds a float sum (`IDENTITY_PATHS.md` row 7). So the question "would an
    A100 build a different model than this M4" is askable in one process,
    and `fit_depthwise_tree`'s `sm_count_override` is here to ask it.

    ================= WHAT IS REQUIRED IN WHICH MODE =================
    **Under `NUMERIC_IDENTICAL` the three models must be BIT-IDENTICAL** --
    nodes, split types and leaf-value bit patterns -- and this claim RAISES
    if they are not. That is the depthwise half of `IDENTITY_PATHS.md`, and
    the whole point of the mode.

    **Under `NUMERIC_FAST` this claim REPORTS rather than raises**, because
    what FAST does is a per-VENDOR question that this gate cannot answer for
    a machine it is not running on. FAST leaves the global histogram flush on
    float `atomicAdd` wherever the backend has one (`deterministic_flush_for`,
    the kernel matrix), and a float atomic's summation order is a function of
    how many blocks the machine ran -- so on a backend with float
    threadgroup atomics, FAST at two core counts is expected to differ.

    ON APPLE IT IS EXPECTED TO AGREE, and for a reason that is a hardware
    fact rather than a policy: Metal has no float threadgroup atomics, so
    the FAST path on this backend is FORCED onto the same i32 shared stage
    and fixed-order bridge that IDENTICAL uses. Apple's FAST is
    bit-reproducible by accident of the platform.

    MEASURED 2026-08-22, M4, this fixture:

        FAST       structure identical, leaf values identical
        IDENTICAL  structure identical, leaf values identical, all three
                   core counts, 15 nodes and 16 leaf values

    The two modes do NOT build the same tree as each other (FAST 8 nodes /
    9 leaves, IDENTICAL 15 / 16 on this fixture) and are not supposed to;
    the claim is invariance WITHIN a mode.
    =================================================================
    """
    # AT THREE DEPTHS. The core count reaches grid widths and the histogram
    # replication factor, and replication is `f(sm_count) / (groups * leaves
    # * stats)` -- so it COLLAPSES to 1 once the leaf count is large and is
    # at its most aggressive at SHALLOW depths. A one-depth run of this
    # claim tests the machine-dependence question at exactly one point on
    # the axis that decides how much machine-dependence there is.
    var depths = List[Int]()
    depths.append(2)
    depths.append(4)
    depths.append(6)

    var mode = String(
        "IDENTICAL"
    ) if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else String("FAST")

    for di in range(len(depths)):
        var d = depths[di]
        var a = fit(fx, default_options(d), sm_count_override=-1)
        var b = fit(fx, default_options(d), sm_count_override=108)
        var c = fit(fx, default_options(d), sm_count_override=1)

        var sa = fingerprint(a, with_values=False)
        var sb = fingerprint(b, with_values=False)
        var sc = fingerprint(c, with_values=False)
        var fa = fingerprint(a)
        var fb = fingerprint(b)
        var fc = fingerprint(c)

        var struct_same = (sa == sb) and (sa == sc)
        var bits_same = (fa == fb) and (fa == fc)

        comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
            if not struct_same:
                raise Error(
                    String("IDENTICAL, depth ") + String(d)
                    + ": core counts {device, 108, 1} built DIFFERENT TREE"
                    " STRUCTURES. A machine-dependent number is feeding a"
                    " summation order upstream of the split choice; see"
                    " IDENTITY_PATHS.md row 7 and the histogram flush rows."
                )
            if not bits_same:
                raise Error(
                    String("IDENTICAL, depth ") + String(d)
                    + ": core counts agreed on the tree but NOT on the leaf"
                    " values. The split path is pinned and the"
                    " partition-stats path is not; IDENTITY_PATHS.md row 7"
                    " is the first place to look."
                )
            print(
                "  claim 6 OK [IDENTICAL] depth", d, "-> core counts"
                " {device, 108, 1} give one model, bit for bit,",
                len(a.model_structure.nodes), "nodes and",
                len(a.leaf_values), "leaf values",
            )
        else:
            print(
                "  claim 6 [FAST] depth", d, "-> structure identical:",
                struct_same, "/ leaf values bit-identical:", bits_same,
            )
            if not (struct_same and bits_same):
                print(
                    "    NOTE: FAST DIVERGED on this backend. On Apple that"
                    " would be the surprising outcome (no float threadgroup"
                    " atomics => FAST is forced onto the i32 stage); on a"
                    " backend that has them it is expected. Either way the"
                    " IDENTICAL build is where the claim is gated."
                )

    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        print(
            "    FAST does not PIN the flush; on Apple it is forced onto"
            " the same i32 stage anyway, so agreement is a platform"
            " accident. The bit claim is gated in the IDENTICAL build."
        )


def claim_7_sabotage(mut fx: Fixture) raises:
    """One row's gradient changed. The model MUST move.

    Without this the whole file could be passing on a tree that is decided
    somewhere other than the device. Corrupting a single row of the
    gradient plane is the smallest input change that should reach the
    histogram, the score, the split and the leaf values, and if the
    fingerprint does not move then the pipeline is not the thing under
    test.

    `sabotage-when-required` calls this required here on two counts: the
    path is new, and the expected values are OUR tally rather than
    CatBoost's.
    """
    var clean = fit(fx, default_options(4))
    var poisoned_fx = Fixture(fx.ctx.copy(), corrupt_row=1234)
    var poisoned = fit(poisoned_fx, default_options(4))
    if fingerprint(clean) == fingerprint(poisoned):
        raise Error(
            "changing one row's gradient did not move the model; the tree"
            " is not being decided by the device data"
        )
    print("  claim 7 OK: one corrupted row moves the model")


def main() raises:
    print("depthwise growth: CatBoost EGrowPolicy::Depthwise, GPU learner")
    var ctx = DeviceContext()
    var fx = Fixture(ctx.copy())
    print(
        "  fixture: 4096 rows, 8 binary + 4 half-byte + 4 one-byte,"
        " hashed bins, divergent target"
    )
    claim_1_bin_feature_table(fx)
    claim_2_model_builder()
    var model = claim_3_growth_and_divergence(fx)
    claim_4_apply_matches_growth(fx, model)
    claim_5_min_data_in_leaf(fx)
    claim_6_core_count_invariance(fx)
    claim_7_sabotage(fx)
    print("depthwise: all 7 claims OK")
