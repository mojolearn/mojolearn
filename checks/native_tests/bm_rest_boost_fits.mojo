# SPDX-License-Identifier: Apache-2.0
"""RandomTreesEmbedding and DART host fits for the Mojo checks that replaced numeric Python tests (lane
bm-rest-rte, 2026-11-23). Importable; no `main()`, no `test_*`.

Every fit runs the product's typed RUN through the host images' published native entries, as the boost door
(bindings/trees_public_boost_door.mojo) calls them after its UNPACK:

- `rte_fit` / `rte_transform`: `rte_fit_run` (xtrees/trees_boost_run.mojo: the uniform targets, their float32
  words, the ExtraTreesRegressor request and XF_FIT in the ET host image), the image's export (what the door's PACK
  `forest_export_handle` copies), `rte_numbering_run` and `rte_transform_run`. The forest's configuration is what
  `_rte_fit` builds its member with: ExtraTreesRegressor(n_estimators, max_depth, max_features=1, the
  min_samples_* / max_leaf_nodes / min_impurity_decrease defaults, random_state=seed), as
  `checks/native_tests/bm_trees_et_fits.mojo` `et_config` writes the door's `_et_config` slots.
- `dart_fit` / `dart_raw`: `dart_starts`, `dart_round_admit`, `dart_member_request`, `dart_round_plan` and
  `dart_rounds_run` (bindings/trees_dart_rounds_run.mojo) over the x_trees glue and DART host entries and the RF
  host forest image (its one-tree DART member fit, core/mojolearn_rf_host_run.mojo `rf_host_fit_dart`); the raw
  score is the image's native DART predict (XD_PREDICT) over the per-tree words the door's `_dart_forest` builds.
  `DartSpec` holds the DARTRegressor / DARTClassifier constructor defaults (python/mojolearn/_expansion_trees.py);
  `dart_words_of` reads them as the door's UNPACK `dart_words` does.

Call `publish_host_images()` first (what the host bindings publish at module init). Each fit keeps its own
parallel builders; callers run independent fits as case-table rows across the host cores.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.trees_entry_table import TreesEntryKey
from core.forest_entry_table import ForestEntryKey, FOREST_TABLE_RF, FOREST_TABLE_ET
from core.forest_calls import xf_ref, xf_caps, xf_export_sizes, xf_export_into, xf_export_release
from core.forest_fit_request import XF_CAP_DEVICE_SCAN
from core.mojolearn_et_host_run import publish_et_host_forest
from core.mojolearn_rf_host_run import publish_rf_host_forest
from xtrees.glue_runs import publish_x_trees_glue
from xtrees.dart_runs_host import publish_x_trees_dart_host
from xtrees.glue_calls import xt_ref, xt_positive_codes, xt_softmax_rows, xt_transpose_f64
from xtrees.dart_calls import xd_ref, xd_predict
from xtrees.trees_forest_run import forest_export_counts
from xtrees.trees_boost_run import rte_fit_run, rte_numbering_run, rte_transform_run, _dart_thr
from bindings.trees_dart_rounds_run import (
    DartRounds, DartWords, DartMemberShape, dart_starts, dart_round_admit, dart_round_plan, dart_rounds_run,
    dart_member_request, DART_M_OFF_START, DART_M_NODE_START, DART_M_VAL_START,
)
from checks.native_tests.bm_trees_et_fits import EtSpec, et_config, ET_MSE
from checks.native_tests.py2mojo_forest_fits import Forest


def trees_key() -> TreesEntryKey:
    """The host images' key (this build's tier, the cpu vendor word)."""
    return TreesEntryKey(Int(GLOBAL_NUMERIC_MODE), 0)


def publish_host_images() raises:
    """The publications `_mojolearn_x_trees_host`, `_mojolearn_trees_host` and `_mojolearn_rf_host` make at module
    init: the x_trees glue and DART entries and the ET and RF host forest images. A repeated publication of the same
    entries is accepted (core/entry_table.mojo)."""
    publish_x_trees_glue("cpu")
    publish_x_trees_dart_host(0)
    publish_et_host_forest()
    publish_rf_host_forest()


# ---------------------------------------------------- RandomTreesEmbedding


def rte_spec(n_estimators: Int, max_depth: Int, seed: Int) -> EtSpec:
    """The embedding's forest (`_rte_fit`'s ExtraTreesRegressor settings): max_features=1 (one sampled feature per
    split: the count slot 1, fraction 0.0), squared error, no bootstrap, the other defaults."""
    var s = EtSpec(False, n_estimators, seed)
    s.max_depth = max_depth
    s.mf_spec = 1
    s.mf_fraction = 0.0
    s.criterion = ET_MSE
    return s^


struct Embedding(Movable):
    """A fitted embedding: its forest's export, the first node of every tree (`_tree_base`), every leaf's output
    column (`_node_col`) and the width (`n_output_features_`)."""
    var forest: Forest
    var tree_base: List[Int32]
    var node_col: List[Int32]
    var width: Int

    def __init__(out self, var forest: Forest, var tree_base: List[Int32], var node_col: List[Int32], width: Int):
        self.forest = forest^
        self.tree_base = tree_base^
        self.node_col = node_col^
        self.width = width


def rte_fit(s: EtSpec, x: List[Float32], n: Int, d: Int) raises -> Embedding:
    """`RandomTreesEmbedding(...).fit(X)` on a row-major float32 X."""
    var key = trees_key()
    var glue = xt_ref(key)
    var forest = xf_ref(ForestEntryKey(FOREST_TABLE_ET, key))
    var device_scan = (xf_caps(forest) & XF_CAP_DEVICE_SCAN) != 0
    var handle = rte_fit_run(glue, forest, et_config(s), Int(x.unsafe_ptr()), n, d, True, device_scan, s.seed)
    _ = len(x)  # the fit above read X through its address
    var offsets = List[Int32]()
    var colid = List[Int32]()
    var quesval = List[Float32]()
    var left = List[Int32]()
    var leaves = List[Float32]()
    var trees = 0
    var outputs = 0
    try:
        var sizes = xf_export_sizes(forest, handle)
        var counts = forest_export_counts(sizes)
        offsets = List[Int32](length=counts[0], fill=Int32(0))
        colid = List[Int32](length=counts[1], fill=Int32(0))
        quesval = List[Float32](length=counts[2], fill=Float32(0))
        left = List[Int32](length=counts[3], fill=Int32(0))
        leaves = List[Float32](length=counts[4], fill=Float32(0))
        var dst: List[Int] = [Int(offsets.unsafe_ptr()), Int(colid.unsafe_ptr()), Int(quesval.unsafe_ptr()),
                              Int(left.unsafe_ptr()), Int(leaves.unsafe_ptr())]
        xf_export_into(forest, sizes, dst)
        trees = sizes.meta[0]
        outputs = sizes.meta[1]
        _ = len(dst)
    finally:
        xf_export_release(forest, handle)
    # `_rte_fit`'s PACK reads: the node count at offsets[n_trees], the leaf numbering, the tree bases
    var nn = Int(offsets[trees])
    var node_col = List[Int32](length=max(nn, 1), fill=Int32(0))
    var width = rte_numbering_run(glue, Int(left.unsafe_ptr()), Int(node_col.unsafe_ptr()), nn)
    _ = len(left)
    var tree_base = List[Int32](capacity=trees)
    for t in range(trees):  # small-loop(the tree bases `_copy_i32(arrays[0], 0, n_trees)`: trees-sized)
        tree_base.append(offsets[t])
    var f = Forest(offsets^, colid^, quesval^, left^, leaves^, trees, outputs, d)
    return Embedding(f^, tree_base^, node_col^, width)


def rte_transform(e: Embedding, x: List[Float32], n: Int) raises -> List[Float64]:
    """`transform(X)`: the dense float64 one-hot, n x width row-major."""
    var glue = xt_ref(trees_key())
    var model = e.forest.model()
    var nodes = List[Int32](length=max(n * e.forest.trees, 1), fill=Int32(0))
    var out = List[Float64](length=max(n * e.width, 1), fill=0.0)
    rte_transform_run(glue, model, Int(x.unsafe_ptr()), Int(nodes.unsafe_ptr()), Int(e.tree_base.unsafe_ptr()),
                      Int(e.node_col.unsafe_ptr()), Int(out.unsafe_ptr()), n, e.forest.features, e.forest.trees,
                      e.width)
    _ = len(x)  # the calls above read these words through their addresses
    _ = len(nodes)
    _ = model^
    return out^


# --------------------------------------------------------------------- DART


struct DartSpec(Copyable, Movable):
    """DARTRegressor / DARTClassifier constructor arguments (python/mojolearn/_expansion_trees.py defaults; the
    seeds as `_seed` reads them: random_state None is 0)."""
    var n_estimators: Int
    var learning_rate: Float64
    var num_leaves: Int
    var max_depth: Int
    var min_child_samples: Int
    var reg_lambda: Float64
    var max_bin: Int
    var drop_rate: Float64
    var max_drop: Int
    var skip_drop: Float64
    var xgboost_dart_mode: Bool
    var uniform_drop: Bool
    var drop_seed: Int
    var seed: Int
    var reg_alpha: Float64
    var max_delta_step: Float64
    var colsample_bytree: Float64
    var subsample: Float64
    var subsample_freq: Int
    var feature_fraction_seed: Int
    var bagging_seed: Int

    def __init__(out self, n_estimators: Int = 100, seed: Int = 0):
        self.n_estimators = n_estimators
        self.learning_rate = 0.1
        self.num_leaves = 31
        self.max_depth = -1
        self.min_child_samples = 20
        self.reg_lambda = 0.0
        self.max_bin = 255
        self.drop_rate = 0.1
        self.max_drop = 50
        self.skip_drop = 0.5
        self.xgboost_dart_mode = False
        self.uniform_drop = False
        self.drop_seed = 4
        self.seed = seed
        self.reg_alpha = 0.0
        self.max_delta_step = 0.0
        self.colsample_bytree = 1.0
        self.subsample = 1.0
        self.subsample_freq = 0
        self.feature_fraction_seed = 2
        self.bagging_seed = 3


def dart_words_of(s: DartSpec) -> DartWords:
    """The door's UNPACK `dart_words` on typed settings: the row bag only when subsample_freq > 0 and
    subsample < 1 (its seed read only then), the column seed only when colsample_bytree < 1, the node cap
    2 * num_leaves - 1, skip_drop as the binding's 53-bit threshold (`_dart_thr`)."""
    var bag = s.subsample_freq > 0 and s.subsample < 1.0
    return DartWords(
        n_iters=s.n_estimators, node_cap=2 * s.num_leaves - 1, skip_thr=_dart_thr(s.skip_drop),
        uniform=1 if s.uniform_drop else 0, xgb=1 if s.xgboost_dart_mode else 0, max_drop=s.max_drop,
        drop_rate=s.drop_rate, lr=s.learning_rate, lam=s.reg_lambda, l1=s.reg_alpha, mds=s.max_delta_step,
        bag=bag, bag_freq=s.subsample_freq, bag_frac=s.subsample, bag_seed=s.bagging_seed if bag else 0,
        col_frac=s.colsample_bytree, col_seed=s.feature_fraction_seed if s.colsample_bytree < 1.0 else 0,
        col_seed_refusal=String(""), seed=s.seed, drop_seed=s.drop_seed)


struct DartModel(Movable):
    """A fitted DART ensemble: the round loop's members, the class starts (`init_score_`), the score width K
    (1 for a regressor and a binary classifier) and the feature count."""
    var rounds: DartRounds
    var starts: List[Float64]
    var k: Int
    var d: Int

    def __init__(out self, var rounds: DartRounds, var starts: List[Float64], k: Int, d: Int):
        self.rounds = rounds^
        self.starts = starts^
        self.k = k
        self.d = d

    def trees(self) -> Int:
        return min(self.rounds.count(), len(self.rounds.coefs))

    def size(self, t: Int) -> Int:
        """Tree t's node count (its one-tree offsets [0, size])."""
        var off = self.rounds.word(t, DART_M_OFF_START)
        return Int(self.rounds.offsets[off + 1]) - Int(self.rounds.offsets[off])


def dart_fit(s: DartSpec, x: List[Float32], n: Int, d: Int, y: List[Float32], k: Int, kind: Int) raises -> DartModel:
    """`_DARTBase._boost`: `y` the float32 targets (kind 0) or class codes (kind 1 binary with k 1, kind 2
    multiclass with k classes); `x` row-major float32."""
    var key = trees_key()
    var glue = xt_ref(key)
    var dart = xd_ref(key)
    var words = dart_words_of(s)
    var starts = dart_starts(glue, Int(y.unsafe_ptr()), n, k, kind)
    dart_round_admit(dart, words.n_iters, k, words.node_cap)
    var request = dart_member_request(DartMemberShape(max_depth=-1 if s.max_depth <= 0 else s.max_depth,
                                                      num_leaves=s.num_leaves, min_child_samples=s.min_child_samples,
                                                      max_bin=s.max_bin), words.seed, n, d)
    var forest = xf_ref(ForestEntryKey(FOREST_TABLE_RF, key.copy()))
    var plan = dart_round_plan(words, dart, key, forest, Int(x.unsafe_ptr()), Int(y.unsafe_ptr()), starts.copy(), n,
                               d, k, kind)
    var rounds = dart_rounds_run(plan, request^)
    _ = len(x)  # the round loop read X and y through their addresses
    _ = len(y)
    return DartModel(rounds^, starts^, k, d)


def dart_raw(m: DartModel, x: List[Float32], n: Int) raises -> List[Float64]:
    """`_DARTBase._raw`: the class-major raw scores (k * n float64) from the image's native DART predict over the
    per-tree words `_dart_forest` builds (each tree's colid / quesval / left_child / leaf-value addresses, its
    node count, its coefficient, and the class starts)."""
    var t_count = m.trees()
    var cols = List[Int64](length=max(t_count, 1), fill=0)
    var ques = List[Int64](length=max(t_count, 1), fill=0)
    var lefts = List[Int64](length=max(t_count, 1), fill=0)
    var vals = List[Int64](length=max(t_count, 1), fill=0)
    var sizes = List[Int64](length=max(t_count, 1), fill=0)
    var coefs = List[Float64](length=max(t_count, 1), fill=0.0)
    for t in range(t_count):  # small-loop(the per-tree words, trees-sized)
        var lo = Int(m.rounds.offsets[m.rounds.word(t, DART_M_OFF_START)])
        var node = m.rounds.word(t, DART_M_NODE_START)
        cols[t] = Int64(Int(m.rounds.colid.unsafe_ptr()) + 4 * (node + lo))
        ques[t] = Int64(Int(m.rounds.quesval.unsafe_ptr()) + 4 * (node + lo))
        lefts[t] = Int64(Int(m.rounds.left.unsafe_ptr()) + 4 * (node + lo))
        vals[t] = Int64(Int(m.rounds.values.unsafe_ptr()) + 4 * m.rounds.word(t, DART_M_VAL_START))
        sizes[t] = Int64(m.size(t))
        coefs[t] = m.rounds.coefs[t]
    var inits = m.starts.copy()
    var score = List[Float64](length=max(m.k * n, 1), fill=0.0)
    var dart = xd_ref(trees_key())
    _ = xd_predict(dart, Int(x.unsafe_ptr()), Int(score.unsafe_ptr()), Int(cols.unsafe_ptr()), Int(ques.unsafe_ptr()),
                   Int(lefts.unsafe_ptr()), Int(vals.unsafe_ptr()), Int(sizes.unsafe_ptr()), Int(coefs.unsafe_ptr()),
                   Int(inits.unsafe_ptr()), n, m.d, m.k, t_count)
    _ = len(x)  # the predict above read every word below through its address
    _ = len(cols)
    _ = len(ques)
    _ = len(lefts)
    _ = len(vals)
    _ = len(sizes)
    _ = len(coefs)
    _ = len(inits)
    return score^


def dart_labels(m: DartModel, x: List[Float32], n: Int) raises -> List[Int]:
    """DARTClassifier's predicted class codes: the positive raw scores (binary, `xt_positive_codes`) or the first
    maximum of each row's k class scores (multiclass)."""
    var raw = dart_raw(m, x, n)
    var out = List[Int](length=n, fill=0)
    if m.k == 1:
        var codes = List[Int32](length=max(n, 1), fill=Int32(0))
        _ = xt_positive_codes(xt_ref(trees_key()), Int(raw.unsafe_ptr()), Int(codes.unsafe_ptr()), n)
        _ = len(raw)
        for i in range(n):  # small-loop(oracle side: one prediction block)
            out[i] = Int(codes[i])
        return out^
    for i in range(n):  # small-loop(oracle side: the argmax of one prediction block)
        var best = 0
        for c in range(1, m.k):  # small-loop(the classes of one row)
            if raw[c * n + i] > raw[best * n + i]:
                best = c
        out[i] = best
    return out^


def dart_proba(m: DartModel, x: List[Float32], n: Int) raises -> List[Float64]:
    """Multiclass `predict_proba`: the raw scores as (n, k) rows (`xt_transpose_f64`), then `xt_softmax_rows`, as
    the door's `dart_predict_proba`."""
    var raw = dart_raw(m, x, n)
    var rows = List[Float64](length=max(n * m.k, 1), fill=0.0)
    var glue = xt_ref(trees_key())
    _ = xt_transpose_f64(glue, Int(raw.unsafe_ptr()), Int(rows.unsafe_ptr()), m.k, n)
    _ = len(raw)
    _ = xt_softmax_rows(glue, Int(rows.unsafe_ptr()), n, m.k)
    return rows^


def r2_f64(pred: List[Float64], want: List[Float32]) -> Float64:
    """scikit-learn's r2_score in float64."""
    var n = len(want)
    var mean = 0.0
    for i in range(n):  # small-loop(oracle: one held-out block)
        mean += Float64(want[i])
    mean /= Float64(n)
    var ss_res = 0.0
    var ss_tot = 0.0
    for i in range(n):  # small-loop(oracle: one held-out block)
        var e = Float64(want[i]) - pred[i]
        var t = Float64(want[i]) - mean
        ss_res += e * e
        ss_tot += t * t
    return 1.0 - ss_res / ss_tot
