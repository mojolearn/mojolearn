# SPDX-License-Identifier: Apache-2.0
"""GradientBoosting as a MEMBER (core/member_abi.mojo), family 16
(`MEMBER_FAMILY_GBDT`).

One entry, `gbdt_member_v2[I]`, published by the gbdt images (the GPU
`_mojolearn_gbdt` and the host `_mojolearn_gbdt_host`, beside their gbdt
entry: `gbdt_member_publish_image[I]`). It runs the estimator's EXISTING
typed fit RUN: the typed config words (bindings/gbdt_config_run.mojo, the
same `gbdt_fit_words` / `gbdt_fit_options` the fit door calls), then
`xg_fit` and `xg_model_dim` through `call.route.entry` (the gbdt entry of
the image; gbdt/fit_calls.mojo), and the estimator's own predict route:
the image's resident model (XG_RESIDENT_PREPARE once per fitted state, on
its first predict, XG_RESIDENT_PREDICT per call) where the image holds one
(XG_CAP_RESIDENT), else the per-call parse (XG_PREDICT / XG_PREDICT_MULTI)
of a column-major X, as `GradientBoosting.predict` / `predict_proba`. No
kernel is touched. Three kinds, one fitted form (the booster's model
text):
  0 `GradientBoostingClassifier` (binary Logloss; y int32 class codes 0 / 1
    with `call.classes` 2; predict int32 codes and predict_proba float32
    rows x 2 through the image's XG_BINARY transform of the margins, as the
    adapter's `_binary_output`; decision float32 rows x 1, the raw margin;
    score the unweighted accuracy, float32 count / rows as the adapter's
    metric; MEMBER_CAN_CLASSIFIER),
  1 `GradientBoosting` (the booster: its own predict returns raw scores),
  2 `GradientBoostingRegressor` (RMSE; predict float32 rows x 1; no score:
    the caller predicts and runs its own R^2),
  3 `ExperimentalTwoLevelFeatureFreq` (lane mf-meta5): its own fit, the
    estimator's two-level FeatureFreq RUN (`xg_feature_freq_fit`, the gbdt
    entry's `feature_freq_fit`, the body its door runs) after its data checks
    (gbdt/trees_gbdt_fit_run.mojo `gbdt_feature_freq_check` /
    `gbdt_weights_check`, the door's ONE BODY), then the booster's predict
    (raw float32 rows x 1). NEW words: [3, the GradientBoosting config, the
    learning rate, l2_leaf_reg, random_state, the S source columns, S, then
    the feature-list tail (0, 0)], read at UNPACK as its fit door reads them.
  4 `OrderedRMSE` (lane mf-hostent5): its own fit, the estimator's ordered
    RMSE RUN (`xg_ordered_fit`, the gbdt entry's `ordered_fit`, the body its
    door runs) after its data checks (gbdt/trees_gbdt_fit_run.mojo
    `gbdt_ordered_rows_check` / `gbdt_ordered_target_check` /
    `gbdt_ordered_order` / `gbdt_weights_check`, the door's ONE BODY), over
    the caller's `call.permutation` (int64 row ids: the fit's required
    `permutation`, read at the caller's UNPACK by the door's admission; not
    given: the fit's TypeError in Python's words), then the booster's predict
    (raw float32 rows x 1). NEW words: [4, the GradientBoosting config,
    n_estimators, max_depth, border_count, learning_rate, l2_leaf_reg, then
    the feature-list tail (0, 0)], read at UNPACK as its fit door reads them.
  5 `HostGBDT` (lane mf-hostent5): a saved model, PREDICT-ONLY (no
    MEMBER_CAN_FIT): its fitted state is IMPORTED from its own fields (the
    model text, its dimensions, bias and category map: the blocks below)
    and predicted on the host image by the booster's predict (raw float32
    rows x approx_dim) and, for Logloss / CrossEntropy, its sigmoid pair
    (float64 rows x 2), as `HostGBDT.predict` / `predict_proba`; the
    multiclass proba and a fit are refused in its words. NEW words: [5, a
    GradientBoosting config carrying the saved loss, (0, 0)].
The adapters' NEW words are their inner learner's config (the learner the
adapter's fit builds), so their fit is the learner's fit.

Route slots: `route.entry` the gbdt entry; `route.mode` / `route.vendor`
the image's key. `I.release(handle)` returns a resident model to the
image's registry (under the image's registry lock).

MEMBER_OP_NEW parameters (Float64 words): [0] kind (0 .. 5), then the
`GbdtConfig` words in `gbdt_config_words` order (bindings/gbdt_config_run.mojo:
ints as the bit pattern of their Int64, booleans 0/1, strings as a length
and one word per byte, lists as a length and their words), then the
feature-list tail (lane mf-img3): the C `cat_features` indices and the H
`one_hot_features` indices as the estimator holds them, then C, then H (the
last two words; 0, 0 for neither).
Views: X dense row-major host float32 (rows x features; finite or NaN:
nan_mode decides, as the estimator); y a host float32 column (the target,
or the class codes of a classification loss), or rows x n_targets for the
multi-target losses (MultiRMSE); optional `weights` a host float32 column
(MEMBER_CAN_SAMPLE_WEIGHT); optional `groups` the int32 or int64 group id of
each row (the query-wise losses' `group_id`: contiguous runs, a split
group refused in the fit door's words; PairLogit takes every pair within a
group, as the door with no `pairs`). Categorical columns hold numbers (their
Float32 words are the keys, gbdt/data/category_map.mojo, as the fit door
keys a float X), or, for a text X, the caller hands the keys its UNPACK read
with the door's encoding (`cat_keys`, lane mf-left4); the fit writes their
dense codes into its own column-major copy and keeps the map, and every
predict codes the query's columns through it (from `cat_keys` when given).
Since lane mf-left4 FIT also reads the door's held-out set (`eval_x` /
`eval_y`, with `eval_cat_keys`: early stopping, use_best_model and the test
loss curve as the door's fit) and PairLogit's explicit `pairs` /
`pairs_weight` (with `groups`, as the door requires). Refused in words: a row or column index, a device-home view, a nonzero
`fit_output`, a multi-column y for a one-target loss.
FAST (the route's tier word 0): GradientBoostingClassifier's class codes are
the learner's `predict_classes` (the resident direct-class kernel, else the
argmax of its predict_proba), as the adapter's `_binary_output` in fast.
Outputs: predict float32 rows x approx_dim (raw scores); predict_proba
float64 rows x 2 (Logloss, CrossEntropy) or float32 rows x classes
(MultiClass softmax, MultiClassOneVsAll sigmoid); other losses refuse it in
the estimator's words.

Fitted-state blocks (MEMBER_OP_EXPORT / MEMBER_OP_IMPORT), in index order:
  0 `MEMBER_BLOCK_FAMILY + 0` int32 1 x ceil(bytes / 4): the model text's
    UTF-8 bytes, four per word, little-endian, zero padded
  1 `MEMBER_BLOCK_FAMILY + 1` float64 1 x iterations: the learn loss curve
  2 `MEMBER_BLOCK_SCALARS` float64 [features, approx_dim, n_classes (0 =
    None), bias, best_iteration, stopped_early, learning_rate, text bytes,
    boosting (0 Plain, 1 Ordered)]
  3 `MEMBER_BLOCK_FAMILY + 2` int64 1 x words: the category map
    (`category_map_`), 0 rows for a model fitted without cat_features
  4 `MEMBER_BLOCK_FAMILY + 3` float64 1 x iterations: the held-out loss
    curve (`test_loss_curve_`, lane mf-left4), 0 rows for a fit without an
    eval set (None)
  5 `MEMBER_BLOCK_FEATURE_IMPORTANCES` float32 1 x features: the fit's
    `feature_importances_` (lane mf-imp6: `xg_importances` on the block the
    fit read, core/gbdt_importance.mojo), 0 rows for an imported model saved
    without them; an import takes float32 or the public float64 vector
An import needs blocks 0 to 2 (and 3 for a CTR model, before the scalars;
4 and 5 are optional); the block that carries MEMBER_IMPORT_DONE comes last.

LIFETIME: the state owns everything it reads after FIT (the model text and
the curves); it keeps no pointer into a fit buffer (no MEMBER_BORROWS_FIT).
The state is OWNED by its slot in a process-wide store of fixed chunks
(lock for mint, free and lookup only). SERIAL BY NATURE: the boosting rounds
(round t reads round t-1's trees) run inside the image; the member's own
per-row work (the code scan, the NaN scan, the column-major copy, the
widening cast) splits rows or columns over the host workers.
"""
from std.sys.info import size_of
from std.memory import unsafe_memcpy, bitcast
from std.math import isfinite
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.entry_table import EntryRef, entry_ref_gil_free, entry_ref_vendor
from core.gbdt_entry_table import gbdt_vendor_word
from gbdt.fit_request import GbdtFitArgs, gbdt_string_at
from gbdt.fit_calls import (
    xg_row_major, xg_model_dim, xg_fit, xg_resident, xg_resident_prepare, xg_resident_predict, xg_predict,
    xg_predict_multi, xg_sigmoid_pair, xg_binary, xg_feature_freq_fit, xg_ordered_fit, xg_importances,
)
from gbdt.trees_gbdt_run import gbdt_text_has_ctr
from gbdt.trees_gbdt_fit_run import (
    gbdt_bias_of, gbdt_check_group_rows, _pair_end_range, gbdt_feature_freq_check, gbdt_weights_check,
    gbdt_ordered_rows_check, gbdt_ordered_target_check, gbdt_ordered_order,
)
from gbdt.data.category_map import (
    CategoryMaps, CategoryKeyPtr, CategoryF32Ptr, category_keys_from_f32, category_block_columns,
    category_block_feature, category_apply_column,
)
from core.host_archive_gbdt import host_archive_gbdt_text
from core.host_helpers_impl import argmax_rows_f64
from core.label_words_impl import id_words_coded, group_runs_scan
from bindings.gbdt_config_run import (
    GbdtConfig, gbdt_config_of_words, gbdt_fit_words, gbdt_fit_options, gbdt_pad_tail, gbdt_push_int,
)
from bindings.trees_public_inputs_run import trees_finite_f32
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk
from core.member_abi import (
    MemberCall, MemberMatrix, MEMBER_F32, MEMBER_F64, MEMBER_I32, MEMBER_I64,
    MEMBER_OP_CAPABILITIES, MEMBER_OP_NEW, MEMBER_OP_FREE, MEMBER_OP_FIT, MEMBER_OP_PREDICT,
    MEMBER_OP_PREDICT_PROBA, MEMBER_OP_OUTPUTS, MEMBER_OP_EXPORT, MEMBER_OP_IMPORT, MEMBER_OP_DECISION,
    MEMBER_OP_SCORE,
    MEMBER_CAN_FIT, MEMBER_CAN_PREDICT, MEMBER_CAN_PREDICT_PROBA, MEMBER_CAN_EXPORT, MEMBER_CAN_IMPORT,
    MEMBER_CAN_SAMPLE_WEIGHT, MEMBER_HOME_DEVICE, MEMBER_GIL_FREE, MEMBER_CAN_DECISION, MEMBER_CAN_SCORE,
    MEMBER_CAN_CLASSIFIER, MEMBER_CAN_IMPORTANCES, MEMBER_BLOCK_FEATURE_IMPORTANCES,
    MEMBER_BLOCK_SCALARS, MEMBER_BLOCK_FAMILY, MEMBER_BLOCK_BY_INDEX, MEMBER_IMPORT_DONE,
    member_publish, member_call_from_spans, member_unsupported,
)
from core.member_families import MEMBER_FAMILY_GBDT
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report, native_descriptor_call
from core.metrics_entry_table import MR_R2
from core.shared_slot import SharedSlot, shared_slot_lock, shared_slot_unlock

comptime _STORE_NAME: StaticString = "MojolearnGbdtMemberStoreV2"
comptime _ESTIMATOR = "GradientBoosting"
#: the member kinds (the classes' `_mojolearn_member` words)
comptime _KIND_CLASSIFIER = 0
comptime _KIND_BOOSTER = 1
comptime _KIND_REGRESSOR = 2
# Lane mf-meta5: `ExperimentalTwoLevelFeatureFreq` (module note, KIND 3).
comptime _KIND_FEATURE_FREQ = 3
# Lane mf-hostent5: `OrderedRMSE` (its own fit) and `HostGBDT` (predict-only import) (module note, KINDS 4, 5).
comptime _KIND_ORDERED = 4
comptime _KIND_HOST = 5
#: OrderedRMSE's fit without `permutation` (Python's words for the missing keyword-only argument).
comptime _ORDERED_NO_PERMUTATION = "TypeError: OrderedRMSE.fit() missing 1 required keyword-only argument: 'permutation'"
comptime _CHUNK = 256
comptime _INDEX_SPAN = 16777216
comptime _BLOCKS = 6
comptime _SCALARS = 9
#: the image's predict modes (bindings/trees_public_gbdt_door.mojo `_PREDICT_*`)
comptime _PREDICT_RAW = 0
comptime _PREDICT_SOFTMAX = 1
comptime _PREDICT_SIGMOID = 2
comptime _PREDICT_SIGMOID_PAIR = 3
comptime _PREDICT_CLASSES_BINARY = 4

comptime _I32P = MutPointer[Int32, MutUntrackedOrigin]
comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _F64P = MutPointer[Float64, MutUntrackedOrigin]
comptime _CallPtr = MutPointer[MemberCall, MutUntrackedOrigin]


trait GbdtMemberImage:
    """What an image adds to the member: the return of a resident model to
    its registry (the host image holds none and never gets a handle)."""

    @staticmethod
    def release(handle: Int) raises:
        ...


struct GbdtMemberSlot(Copyable, Movable, Defaultable):
    var generation: Int
    var live: Bool
    var kind: Int
    var config: GbdtConfig
    var fitted: Bool
    var text: String
    var learn: List[Float64]
    var features: Int
    var dim: Int
    var classes: Int
    var bias: Float64
    var best_iteration: Int
    var stopped_early: Bool
    var rate: Float64
    var ordered: Bool
    var resident: Int
    # NEW's feature lists (lane mf-img3): `cat_features` and `one_hot_features`
    # as the estimator holds them (their order; the range test reads them).
    var cat_given: List[Int]
    var hot_given: List[Int]
    # The fitted category map (gbdt/data/category_map.mojo's stored words; the
    # estimator's `category_map_`); empty for a model without cat_features.
    var cat_map: List[UInt64]
    # The held-out loss curve (lane mf-left4: a fit given `call.eval_x`), the
    # estimator's `test_loss_curve_`; `has_test` False: None.
    var test: List[Float64]
    var has_test: Bool
    # `feature_importances_` (lane mf-imp6: float32 per input column, the
    # fit's xg_importances; empty for an imported model saved without them).
    var imp: List[Float32]
    # Kind 3 (lane mf-meta5): the FeatureFreq fit's source columns and settings.
    var ff_sources: List[Int]
    var ff_rate: Float64
    var ff_l2: Float64
    var ff_seed: Int
    # Kind 4 (lane mf-hostent5): the OrderedRMSE fit's settings.
    var od_trees: Int
    var od_depth: Int
    var od_border: Int
    var od_rate: Float64
    var od_l2: Float64
    # Export forms, built at export.
    var x_text: List[Int32]
    var x_scalars: List[Float64]

    def __init__(out self):
        self.generation = 0
        self.live = False
        self.kind = _KIND_BOOSTER
        self.config = GbdtConfig()
        self.fitted = False
        self.text = String("")
        self.learn = List[Float64]()
        self.features = 0
        self.dim = 0
        self.classes = 0
        self.bias = 0.0
        self.best_iteration = 0
        self.stopped_early = False
        self.rate = 0.0
        self.ordered = False
        self.resident = -1
        self.cat_given = List[Int]()
        self.hot_given = List[Int]()
        self.cat_map = List[UInt64]()
        self.test = List[Float64]()
        self.has_test = False
        self.imp = List[Float32]()
        self.ff_sources = List[Int]()
        self.ff_rate = 0.0
        self.ff_l2 = 0.0
        self.ff_seed = 0
        self.od_trees = 0
        self.od_depth = 0
        self.od_border = 0
        self.od_rate = 0.0
        self.od_l2 = 0.0
        self.x_text = List[Int32]()
        self.x_scalars = List[Float64]()


struct GbdtMemberStore(Defaultable, Movable):
    var synchronization: List[Int64]
    var chunks: List[List[GbdtMemberSlot]]
    var free: List[Int]
    var count: Int

    def __init__(out self):
        self.synchronization = List[Int64](length=2, fill=0)
        self.chunks = List[List[GbdtMemberSlot]]()
        self.free = List[Int]()
        self.count = 0


comptime _SlotPtr = MutPointer[GbdtMemberSlot, MutUntrackedOrigin]


def _slot_address(index: Int) raises -> Int:
    # Under the store's lock; a chunk's buffer never grows.
    var store = SharedSlot[GbdtMemberStore, _STORE_NAME, 2].get_or_create_ptr()
    return Int(store[].chunks[index // _CHUNK].unsafe_ptr()) + (index % _CHUNK) * size_of[GbdtMemberSlot]()


def _clear[I: GbdtMemberImage](slot: _SlotPtr):
    """The fitted state dropped; a resident model back to its image
    (non-raising)."""
    if slot[].resident >= 0:
        var handle = slot[].resident
        slot[].resident = -1
        try:
            I.release(handle)
        except:
            pass
    slot[].fitted = False
    slot[].text = String("")
    slot[].learn = List[Float64]()
    slot[].x_text = List[Int32]()
    slot[].x_scalars = List[Float64]()
    slot[].cat_map = List[UInt64]()
    slot[].test = List[Float64]()
    slot[].has_test = False
    slot[].imp = List[Float32]()
    slot[].features = 0
    slot[].dim = 0
    slot[].classes = 0


def _name(kind: Int) -> String:
    if kind == _KIND_CLASSIFIER:
        return String("GradientBoostingClassifier")
    if kind == _KIND_REGRESSOR:
        return String("GradientBoostingRegressor")
    if kind == _KIND_FEATURE_FREQ:
        return String("ExperimentalTwoLevelFeatureFreq")
    if kind == _KIND_ORDERED:
        return String("OrderedRMSE")
    if kind == _KIND_HOST:
        return String("HostGBDT")
    return String(_ESTIMATOR)


def _kind_word(word: Float64) raises -> Int:
    if word < 0.0 or word > 5.0 or word != Float64(Int(word)):
        raise Error("member ABI: a Gbdt member's kind word is 0, 1, 2, 3, 4 or 5")
    return Int(word)


def _feature_tail(words: _F64P, count: Int, mut cat: List[Int], mut hot: List[Int]) raises -> Int:
    """NEW's feature-list tail (lane mf-img3): `[..., cat indices, one-hot
    indices, C, H]` at the end of the words; the count of the words before
    it (the kind word and the config)."""
    if count < 4:
        raise Error("member ABI: a Gbdt member takes its kind word, its config words and its feature-list tail")
    var h = Int(words[count - 1])
    var c = Int(words[count - 2])
    if h < 0 or c < 0 or c + h + 4 > count:
        raise Error("member ABI: a Gbdt member's feature-list tail does not fit its words")
    var start = count - 2 - c - h
    for i in range(c):  # small-loop(cat_features)
        cat.append(Int(words[start + i]))
    for i in range(h):  # small-loop(one_hot_features)
        hot.append(Int(words[start + c + i]))
    return start


def _mint(words: _F64P, count: Int) raises -> Int:
    var cat = List[Int]()
    var hot = List[Int]()
    var head = _feature_tail(words, count, cat, hot)
    var kind = _kind_word(words[0])
    # Kind 3: the FeatureFreq words before the feature-list tail (module note).
    var sources = List[Int]()
    var ff_rate = 0.0
    var ff_l2 = 0.0
    var ff_seed = 0
    if kind == _KIND_FEATURE_FREQ:
        var s_count = Int(words[head - 1]) if head >= 1 else -1
        if s_count < 2 or head - 1 - s_count - 3 < 1:
            raise Error("member ABI: an ExperimentalTwoLevelFeatureFreq member takes its settings and source columns")
        for i in range(s_count):  # small-loop(the source columns)
            sources.append(Int(words[head - 1 - s_count + i]))
        var at = head - 1 - s_count - 3
        ff_rate = words[at]
        ff_l2 = words[at + 1]
        ff_seed = Int(words[at + 2])
        head = at
    # Kind 4: the five OrderedRMSE settings before the feature-list tail (module note).
    var od_trees = 0
    var od_depth = 0
    var od_border = 0
    var od_rate = 0.0
    var od_l2 = 0.0
    if kind == _KIND_ORDERED:
        if head - 5 < 2:
            raise Error("member ABI: an OrderedRMSE member takes its five settings")
        var at = head - 5
        od_trees = Int(words[at])
        od_depth = Int(words[at + 1])
        od_border = Int(words[at + 2])
        od_rate = words[at + 3]
        od_l2 = words[at + 4]
        head = at
    var copied = List[Float64](length=head, fill=0.0)
    unsafe_memcpy(dest=copied.unsafe_ptr(), src=words, count=head)
    var config = gbdt_config_of_words(copied^, 1)
    # The adapters fix their learner's loss (`_GBDTAdapter._loss`).
    if kind == _KIND_CLASSIFIER and config.loss != "Logloss":
        raise Error("member ABI: a GradientBoostingClassifier member fits loss='Logloss'")
    if (kind == _KIND_REGRESSOR or kind == _KIND_FEATURE_FREQ or kind == _KIND_ORDERED) and config.loss != "RMSE":
        raise Error("member ABI: a " + _name(kind) + " member fits loss='RMSE'")
    var store = SharedSlot[GbdtMemberStore, _STORE_NAME, 2].get_or_create_ptr()
    var lock = rebind[MutPointer[Int64, MutUntrackedOrigin]](store[].synchronization.unsafe_ptr())
    shared_slot_lock(lock)
    try:
        var index = -1
        if len(store[].free) > 0:
            index = store[].free.pop()
        else:
            if store[].count >= _INDEX_SPAN - 1:
                raise Error("RuntimeError: mojolearn: too many live Gbdt member states")
            if store[].count == len(store[].chunks) * _CHUNK:
                store[].chunks.append(List[GbdtMemberSlot](length=_CHUNK, fill=GbdtMemberSlot()))
            index = store[].count
            store[].count += 1
        var slot = _SlotPtr(unsafe_from_address=_slot_address(index))
        slot[].generation += 1
        slot[].live = True
        slot[].kind = kind
        slot[].config = config^
        slot[].cat_given = cat^
        slot[].hot_given = hot^
        slot[].ff_sources = sources^
        slot[].ff_rate = ff_rate
        slot[].ff_l2 = ff_l2
        slot[].ff_seed = ff_seed
        slot[].od_trees = od_trees
        slot[].od_depth = od_depth
        slot[].od_border = od_border
        slot[].od_rate = od_rate
        slot[].od_l2 = od_l2
        slot[].fitted = False
        slot[].resident = -1
        return slot[].generation * _INDEX_SPAN + index + 1
    finally:
        shared_slot_unlock(lock)


def _lookup(handle: Int, release: Bool) raises -> Int:
    """The slot address of a live handle; a released, foreign or made-up
    handle is refused. A released slot's fitted state is dropped by the
    caller (`_clear`) before the release."""
    var store = SharedSlot[GbdtMemberStore, _STORE_NAME, 2].get_or_create_ptr()
    var lock = rebind[MutPointer[Int64, MutUntrackedOrigin]](store[].synchronization.unsafe_ptr())
    shared_slot_lock(lock)
    try:
        var index = handle % _INDEX_SPAN - 1
        if handle <= 0 or index < 0 or index >= store[].count:
            raise Error("NotFittedError: this member state was released or never created")
        var address = _slot_address(index)
        var slot = _SlotPtr(unsafe_from_address=address)
        if not slot[].live or slot[].generation != handle // _INDEX_SPAN:
            raise Error("NotFittedError: this member state was released or never created")
        if release:
            slot[].live = False
            slot[].config = GbdtConfig()
            slot[].cat_given = List[Int]()
            slot[].hot_given = List[Int]()
            store[].free.append(index)
        return address
    finally:
        shared_slot_unlock(lock)


def _whole(call: _CallPtr) raises:
    if call[].row_index.rows > 0 or call[].column_index.rows > 0:
        raise Error("TypeError: a Gbdt member reads whole host views (no row or column index)")


def _dense(view: MemberMatrix, dtype: Int, what: String) raises:
    if view.home == MEMBER_HOME_DEVICE or not view.dense_host(dtype):
        var words = String("float32") if dtype == MEMBER_F32 else (
            String("float64") if dtype == MEMBER_F64 else String("int32"))
        raise Error("TypeError: a Gbdt member reads " + what + " as a dense host " + words + " view")


def _column_major(src: Int, rows: Int, cols: Int) -> List[Float32]:
    """A row-major float32 block as column-major words (the layout the
    image's column-major fit and per-call parse read; the fit door admits X
    in that order with the same one copy), split by columns over the host
    workers."""
    var out = List[Float32](length=max(rows * cols, 1), fill=0.0)
    var source = _F32P(unsafe_from_address=src)
    var target = _F32P(unsafe_from_address=Int(out.unsafe_ptr()))
    var tasks = host_predict_task_count_work(cols, rows * cols)
    var chunk = host_predict_chunk(cols, tasks)

    def task(unit: Int) {imm source, imm target, imm chunk, imm rows, imm cols}:
        for j in range(unit * chunk, min((unit + 1) * chunk, cols)):  # native data loop (one host task's columns)
            for i in range(rows):  # native data loop (the column's rows)
                target[j * rows + i] = source[i * cols + j]

    host_parallelize(task, tasks)
    return out^


def _has_nan(src: Int, count: Int) -> Bool:
    """Whether a float32 block holds a NaN (per-task flags)."""
    var tasks = host_predict_task_count_work(count, count)
    var chunk = host_predict_chunk(count, tasks)
    var flags = List[UInt8](length=max(tasks, 1), fill=0)
    var source = _F32P(unsafe_from_address=src)
    var marks = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(flags.unsafe_ptr()))

    def task(unit: Int) {imm source, imm marks, imm chunk, imm count}:
        for i in range(unit * chunk, min((unit + 1) * chunk, count)):  # native data loop (one host task's cells)
            if source[i] != source[i]:
                marks[unit] = 1
                return

    host_parallelize(task, tasks)
    for t in range(len(flags)):  # small-loop(tasks): one flag per host task
        if flags[t] != 0:
            return True
    return False


def _code_classes(src: Int, rows: Int, loss: String) raises -> Int:
    """`_codes` of the fit door on the member's float32 class codes: the
    number of distinct codes, refused in the door's words when a label is not
    a finite nonnegative integer, when fewer than two classes are present or
    when the codes are not 0..k-1 (two passes split over the host workers:
    the range, then the presence of each code)."""
    var tasks = host_predict_task_count_work(rows, rows)
    var chunk = host_predict_chunk(rows, tasks)
    var source = _F32P(unsafe_from_address=src)
    var lows = List[Float32](length=max(tasks, 1), fill=Float32(3.0e38))
    var highs = List[Float32](length=max(tasks, 1), fill=Float32(-1.0))
    var bads = List[UInt8](length=max(tasks, 1), fill=0)
    var lo_p = _F32P(unsafe_from_address=Int(lows.unsafe_ptr()))
    var hi_p = _F32P(unsafe_from_address=Int(highs.unsafe_ptr()))
    var bad_p = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(bads.unsafe_ptr()))

    def scan(unit: Int) {imm source, imm lo_p, imm hi_p, imm bad_p, imm chunk, imm rows}:
        for i in range(unit * chunk, min((unit + 1) * chunk, rows)):  # native data loop (one host task's rows)
            var v = source[i]
            # (a float32 at or above 2**24 is an integer; Int() of it is exact
            # below 1e15, and such a code is past the rows: not dense, below.)
            if not isfinite(v) or v < 0 or (v < Float32(1.0e15) and Float32(Int(v)) != v):
                bad_p[unit] = 1
                return
            lo_p[unit] = min(lo_p[unit], v)
            hi_p[unit] = max(hi_p[unit], v)

    host_parallelize(scan, tasks)
    var low = Float32(3.0e38)
    var high = Float32(-1.0)
    for t in range(len(bads)):  # small-loop(tasks): one range per host task
        if bads[t] != 0:
            raise Error("mojolearn: " + loss + " labels must be finite nonnegative integer class codes")
        low = min(low, lows[t])
        high = max(high, highs[t])
    if rows == 0 or low == high:
        raise Error("mojolearn: " + loss + " needs at least two classes")
    if high >= Float32(rows):
        raise Error("mojolearn: " + loss + " labels must be dense class codes 0..k-1; encode labels before fitting")
    var top = Int(high)
    if top >= rows:
        raise Error("mojolearn: " + loss + " labels must be dense class codes 0..k-1; encode labels before fitting")
    var width = top + 1
    var seen = List[UInt8](length=max(tasks, 1) * width, fill=0)
    var seen_p = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(seen.unsafe_ptr()))

    def mark(unit: Int) {imm source, imm seen_p, imm chunk, imm rows, imm width}:
        for i in range(unit * chunk, min((unit + 1) * chunk, rows)):  # native data loop (one host task's rows)
            seen_p[unit * width + Int(source[i])] = 1

    host_parallelize(mark, tasks)
    var k = 0
    for c in range(width):  # native data loop (the codes; tasks per code are a small-loop)
        var any = False
        for t in range(max(tasks, 1)):  # small-loop(tasks)
            if seen[t * width + c] != 0:
                any = True
                break
        if any:
            k += 1
    if k != width:
        raise Error("mojolearn: " + loss + " labels must be dense class codes 0..k-1; encode labels before fitting")
    return k


def _binary_codes(src: Int, rows: Int, classes: Int) raises -> List[Float32]:
    """GradientBoostingClassifier's targets: the caller's int32 codes of its
    two classes as float32 0 / 1 (the adapter's `_codes_f32`), refused in the
    adapter's words for one class or more than two; the cast split over the
    host workers."""
    if classes < 2:
        raise Error("Classifier can't train when only one class is present.")
    if classes > 2:
        raise Error("Only binary classification is supported. The type of the target is multiclass.")
    var out = List[Float32](length=max(rows, 1), fill=0.0)
    var tasks = host_predict_task_count_work(rows, rows)
    var chunk = host_predict_chunk(rows, tasks)
    var bads = List[UInt8](length=max(tasks, 1), fill=0)
    var source = _I32P(unsafe_from_address=src)
    var target = _F32P(unsafe_from_address=Int(out.unsafe_ptr()))
    var bad_p = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(bads.unsafe_ptr()))

    def task(unit: Int) {imm source, imm target, imm bad_p, imm chunk, imm rows}:
        for i in range(unit * chunk, min((unit + 1) * chunk, rows)):  # native data loop (one host task's rows)
            var c = source[i]
            if c < 0 or c > 1:
                bad_p[unit] = 1
                return
            target[i] = Float32(c)

    host_parallelize(task, tasks)
    for t in range(len(bads)):  # small-loop(tasks): one flag per host task
        if bads[t] != 0:
            raise Error("ValueError: a GradientBoostingClassifier member reads the class codes 0 and 1")
    return out^


def _multi_output(loss: String) -> Bool:
    return loss == "MultiClass" or loss == "MultiClassOneVsAll"


def _binary(loss: String) -> Bool:
    return loss == "Logloss" or loss == "CrossEntropy"


def _multi_regression(loss: String) -> Bool:
    """`MULTI_REGRESSION_LOSSES` (python/mojolearn/ensemble.py): y of
    n_targets columns."""
    return loss == "MultiRMSE"


def _sorted_unique(values: List[Int]) -> List[Int]:
    """`gbdt_category_columns`: the indices ascending, without repeats."""
    var out = List[Int]()
    for k in range(len(values)):  # small-loop(cat_features): sorted insert of one index
        var at = values[k]
        var pos = 0
        while pos < len(out) and out[pos] < at:  # small-loop(out)
            pos += 1
        if pos < len(out) and out[pos] == at:
            continue
        out.append(at)
        var i = len(out) - 1
        while i > pos:  # small-loop(out)
            out[i] = out[i - 1]
            i -= 1
        out[pos] = at
    return out^


def _feature_flags(slot: _SlotPtr, cols: Int) raises -> List[UInt32]:
    """`GradientBoosting._flags`: one word per feature (bit 0 categorical,
    bit 1 one-hot), the range refusal in its words and order (cat_features,
    then one_hot_features); one zero word when neither list is given."""
    if len(slot[].cat_given) == 0 and len(slot[].hot_given) == 0:
        return List[UInt32](length=1, fill=0)
    var out = List[UInt32](length=cols, fill=0)
    for which in range(2):  # small-loop(lists): the two feature-list arguments
        var name = String("cat_features") if which == 0 else String("one_hot_features")
        var count = len(slot[].cat_given) if which == 0 else len(slot[].hot_given)
        for k in range(count):  # small-loop(the feature-list indices)
            var at = slot[].cat_given[k] if which == 0 else slot[].hot_given[k]
            if at < 0 or at >= cols:
                raise Error("mojolearn: " + name + " index " + String(at) + " out of range for "
                            + String(cols) + " features")
            out[at] = out[at] | UInt32(1 if which == 0 else 2)
    return out^


def _keys_of(keys: MemberMatrix, n_cat: Int, rows: Int, what: String) raises -> Int:
    """The address of a caller's categorical keys (`MemberCall.cat_keys`:
    int64 words, one row per categorical column, one column per row of X),
    0 when not given (the columns' Float32 words are then the keys)."""
    if keys.rows == 0:
        return 0
    if not keys.dense_host(MEMBER_I64):
        raise Error("TypeError: a Gbdt member reads " + what + " as a dense host int64 view")
    if keys.rows != n_cat or keys.cols != rows:
        raise Error("member ABI: " + what + " is " + String(keys.rows) + " x " + String(keys.cols) + " for "
                    + String(n_cat) + " categorical columns of " + String(rows) + " rows")
    return keys.data


def _category_fit(columns: List[Int], cm: Int, rows: Int, keys: MemberMatrix) raises -> List[UInt64]:
    """`gbdt_category_fit` on the member's column-major float32 block: each
    categorical column keyed from its Float32 words, or (lane mf-left4) from
    the caller's keys of a text X (`keys`, the door's `from_cells` branch),
    its map fitted and its dense codes written over it (each column's rows
    on every worker); the stored map's words (the estimator's
    `category_map_`)."""
    var maps = CategoryMaps()
    var given = _keys_of(keys, len(columns), rows, "cat_keys")
    var scratch = List[UInt64](length=max(rows, 1) if given == 0 else 1, fill=0)
    var sp = CategoryKeyPtr(unsafe_from_address=Int(scratch.unsafe_ptr()))
    for c in range(len(columns)):  # driver-loop: over the categorical columns (each column's rows run on every worker)
        var column = CategoryF32Ptr(unsafe_from_address=cm + 4 * columns[c] * rows)
        if given != 0:
            maps.fit_column(columns[c], CategoryKeyPtr(unsafe_from_address=given) + c * rows, rows, column)
        else:
            category_keys_from_f32(column, rows, sp)
            maps.fit_column(columns[c], sp, rows, column)
    var words = List[UInt64](length=maps.block_words(), fill=0)
    maps.write_block(CategoryKeyPtr(unsafe_from_address=Int(words.unsafe_ptr())))
    _ = len(scratch)
    return words^


def _category_apply_map(cat_map: List[UInt64], features: Int, cm: Int, rows: Int, keys: MemberMatrix) raises:
    """`gbdt_category_apply` on a column-major float32 block (a query, or the
    fit's held-out set): the dense codes of the map `cat_map` over its
    categorical columns (a value the fit did not see takes the unseen code),
    keyed from the columns' Float32 words or from the caller's keys of a
    text X (`keys`, lane mf-left4); the map searched in place."""
    var block = CategoryKeyPtr(unsafe_from_address=Int(cat_map.unsafe_ptr()))
    var n_cat = category_block_columns(block, len(cat_map), features)
    var given = _keys_of(keys, n_cat, rows, "cat_keys")
    var scratch = List[UInt64](length=max(rows, 1) if given == 0 else 1, fill=0)
    var sp = CategoryKeyPtr(unsafe_from_address=Int(scratch.unsafe_ptr()))
    for c in range(n_cat):  # driver-loop: over the categorical columns (each column's rows run on every worker)
        var column = CategoryF32Ptr(unsafe_from_address=cm + 4 * category_block_feature(block, c) * rows)
        if given != 0:
            category_apply_column(block, c, CategoryKeyPtr(unsafe_from_address=given) + c * rows, rows, column)
        else:
            category_keys_from_f32(column, rows, sp)
            category_apply_column(block, c, sp, rows, column)
    _ = len(scratch)


def _group_sizes(call: _CallPtr, rows: Int) raises -> List[UInt32]:
    """`_group_sizes` of the caller's group ids (int32 or int64, one per
    row): the run lengths of consecutive equal ids, a split group refused in
    the fit door's words (`group_runs_scan`, the door's own scan). Restored by
    nn-integrate: lane mf-left4 deleted it while the grouped fit still calls it."""
    var view = call[].groups
    if view.home == MEMBER_HOME_DEVICE or not (view.dense_host(MEMBER_I64) or view.dense_host(MEMBER_I32)):
        raise Error("TypeError: a Gbdt member reads the group ids as a dense host int32 or int64 view")
    var n = view.rows * view.cols
    gbdt_check_group_rows(n, rows)
    var words = id_words_coded(view.data, 0 if view.dtype == MEMBER_I64 else 1, n)
    var sizes = List[Int]()
    var split = group_runs_scan(words, sizes)
    if split >= 0:
        raise Error("mojolearn: group Ids are not consecutive: the rows of group "
                    + String(words[split]) + " are split into more than one run (row " + String(split) + "); "
                    + "CatBoost requires every group's rows to be contiguous")
    var out = List[UInt32](capacity=len(sizes))
    for i in range(len(sizes)):  # native data loop (one word per group)
        out.append(UInt32(sizes[i]))
    return out^


def _category_apply(slot: _SlotPtr, cm: Int, rows: Int, keys: MemberMatrix) raises:
    """`_category_apply_map` with the state's fitted map."""
    _category_apply_map(slot[].cat_map, slot[].features, cm, rows, keys)


def _eval_scan(src: Int, rows: Int, codes: Bool) raises -> Tuple[Bool, Bool, Float32]:
    """One pass over the held-out targets split over the host workers (lane
    mf-left4): whether every value is the same (the door's `reduce_stat`
    min == max, `eval_constant`), whether one is not a finite nonnegative
    integer (read when `codes`: a classification loss's labels), and the
    largest value."""
    var tasks = host_predict_task_count_work(rows, rows)
    var chunk = host_predict_chunk(rows, tasks)
    var source = _F32P(unsafe_from_address=src)
    var lows = List[Float32](length=max(tasks, 1), fill=Float32(3.0e38))
    var highs = List[Float32](length=max(tasks, 1), fill=Float32(-3.0e38))
    var bads = List[UInt8](length=max(tasks, 1), fill=0)
    var lo_p = _F32P(unsafe_from_address=Int(lows.unsafe_ptr()))
    var hi_p = _F32P(unsafe_from_address=Int(highs.unsafe_ptr()))
    var bad_p = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(bads.unsafe_ptr()))

    def scan(unit: Int) {imm source, imm lo_p, imm hi_p, imm bad_p, imm chunk, imm rows, imm codes}:
        for i in range(unit * chunk, min((unit + 1) * chunk, rows)):  # native data loop (one host task's rows)
            var v = source[i]
            if codes and (not isfinite(v) or v < 0 or (v < Float32(1.0e15) and Float32(Int(v)) != v)):
                bad_p[unit] = 1
            lo_p[unit] = min(lo_p[unit], v)
            hi_p[unit] = max(hi_p[unit], v)

    host_parallelize(scan, tasks)
    var low = Float32(3.0e38)
    var high = Float32(-3.0e38)
    var bad = False
    for t in range(len(bads)):  # small-loop(tasks): one range per host task
        bad = bad or bads[t] != 0
        low = min(low, lows[t])
        high = max(high, highs[t])
    return (low == high, bad, high)


@always_inline
def _pair_end(data: Int, wide: Bool, at: Int) -> Int:
    if wide:
        return Int(MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=data)[at])
    return Int(_I32P(unsafe_from_address=data)[at])


def _pair_refuse(data: Int, wide: Bool, i: Int, rows: Int) raises:
    """The door's refusal of pair `i` (the first bad one), in its order."""
    var w0 = _pair_end(data, wide, 2 * i)
    _pair_end_range(w0, i, 0, rows)
    var w1 = _pair_end(data, wide, 2 * i + 1)
    _pair_end_range(w1, i, 1, rows)
    raise Error("mojolearn: pairs[" + String(i) + "] pairs row " + String(w0) + " with itself")


def _pairs_words(call: _CallPtr, rows: Int, mut flat: List[UInt32], mut weights: List[Float32]) raises:
    """The door's `_pairs_arrays` on the member's views (lane mf-left4): the
    uint32 winner / loser words, each end inside the fit's rows and never a
    row paired with itself, and the float32 pair weights (ones when not
    given), each finite and nonnegative; the refusal names the first bad
    pair or weight in the door's words. The pairs and the weights are split
    over the host workers (each task notes its first bad index; tasks are in
    index order, so the first task that noted one holds the door's)."""
    var view = call[].pairs
    if view.home == MEMBER_HOME_DEVICE or not (view.dense_host(MEMBER_I64) or view.dense_host(MEMBER_I32)):
        raise Error("TypeError: a Gbdt member reads pairs as a dense host int32 or int64 view")
    if view.cols != 2:
        raise Error("mojolearn: Length of pairs[0] isn't equal to 2.")
    var m = view.rows
    var data = view.data
    var wide = view.dtype == MEMBER_I64
    flat = List[UInt32](length=2 * m, fill=0)
    var tasks = host_predict_task_count_work(m, m)
    var chunk = host_predict_chunk(m, tasks)
    var firsts = List[Int](length=max(tasks, 1), fill=-1)
    var fp = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=Int(flat.unsafe_ptr()))
    var first_p = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=Int(firsts.unsafe_ptr()))

    def check(unit: Int) {imm data, imm wide, imm fp, imm first_p, imm chunk, imm m, imm rows}:
        for i in range(unit * chunk, min((unit + 1) * chunk, m)):  # native data loop (one host task's pairs)
            var w0 = _pair_end(data, wide, 2 * i)
            var w1 = _pair_end(data, wide, 2 * i + 1)
            if w0 < 0 or w0 >= rows or w1 < 0 or w1 >= rows or w0 == w1:
                first_p[unit] = i
                return
            fp[2 * i] = UInt32(w0)
            fp[2 * i + 1] = UInt32(w1)

    host_parallelize(check, tasks)
    for t in range(len(firsts)):  # small-loop(tasks): the first task that noted a bad pair
        if firsts[t] >= 0:
            _pair_refuse(data, wide, firsts[t], rows)
    weights = List[Float32](length=m, fill=Float32(1.0))
    var given = call[].pairs_weight
    if given.rows == 0:
        return
    _dense(given, MEMBER_F32, "pairs_weight")
    if given.rows * given.cols != m:
        raise Error("mojolearn: len(pairs_weight) = " + String(given.rows * given.cols) + " is not equal to "
                    + "len(pairs) = " + String(m) + " ")
    var source = _F32P(unsafe_from_address=given.data)
    var wp = _F32P(unsafe_from_address=Int(weights.unsafe_ptr()))
    var bads = List[Int](length=max(tasks, 1), fill=-1)
    var bad_p = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=Int(bads.unsafe_ptr()))

    def copy(unit: Int) {imm source, imm wp, imm bad_p, imm chunk, imm m}:
        for i in range(unit * chunk, min((unit + 1) * chunk, m)):  # native data loop (one host task's weights)
            var f = source[i]
            if not (isfinite(f) and f >= 0):
                bad_p[unit] = i
                return
            wp[i] = f

    host_parallelize(copy, tasks)
    for t in range(len(bads)):  # small-loop(tasks): the first task that noted a bad weight
        if bads[t] >= 0:
            raise Error("mojolearn: pairs_weight[" + String(bads[t]) + "] must be a finite nonnegative number")


def _fit[I: GbdtMemberImage](slot: _SlotPtr, call: _CallPtr) raises:
    """`gbdt_fit` of the door on the member's views: its checks in its
    order and words (the ones the member's views can reach), the typed
    request from the config, `xg_fit` and the model dimension. Since lane
    mf-img3 also cat_features (numeric categories, keyed from their Float32
    words as the door keys a float X), one_hot_features, the multi-target
    losses (y of n_targets columns) and the query-wise losses (the caller's
    group ids, `call.groups`). Since lane mf-left4 also the fit door's
    held-out set (`call.eval_x` / `eval_y`: its checks in its words, the
    category map applied to it, `eval_constant`, the test loss curve kept),
    PairLogit's explicit `pairs` / `pairs_weight`, and the categorical keys
    of a text X (`call.cat_keys` / `eval_cat_keys`, the door's encoding read
    at the caller's UNPACK)."""
    if slot[].kind == _KIND_FEATURE_FREQ:
        _fit_feature_freq[I](slot, call)
        return
    if slot[].kind == _KIND_ORDERED:
        _fit_ordered[I](slot, call)
        return
    if slot[].kind == _KIND_HOST:
        # A saved model has no fit (HostGBDT defines none: Python's words for the missing attribute).
        raise Error("AttributeError: 'HostGBDT' object has no attribute 'fit'")
    _whole(call)
    if call[].fit_output != 0:
        raise Error("TypeError: a Gbdt member has no fit output")
    _dense(call[].x, MEMBER_F32, "X")
    var cat_columns = _sorted_unique(slot[].cat_given)
    var rows = call[].x.rows
    var cols = call[].x.cols
    if rows == 0 or cols == 0:
        raise Error("mojolearn: fit requires at least one row and one feature")
    var kind = slot[].kind
    if kind == _KIND_CLASSIFIER:
        _dense(call[].y, MEMBER_I32, "y (class codes)")
    else:
        _dense(call[].y, MEMBER_F32, "y")
    ref config = slot[].config
    var grouped = call[].groups.rows > 0
    var target_dim = 1
    if _multi_regression(config.loss):
        target_dim = call[].y.cols
        if target_dim < 2:
            raise Error("mojolearn: " + config.loss + " needs y of shape (n_samples, "
                        + "n_targets) with n_targets >= 2, got " + String(target_dim) + " "
                        + "(CatBoost: \"Only one class found, can't learn "
                        + "multiclass objective\", multiclass_targets.h:167)")
        if call[].eval_x.rows > 0:
            raise Error("NotImplementedError: mojolearn: loss='" + config.loss + "' with eval_set is not "
                        + "carried: the held-out arm's target is one value per row")
        if grouped or call[].pairs.rows > 0:
            raise Error("mojolearn: loss='" + config.loss + "' does not read group_id or pairs")
    elif call[].y.cols != 1:
        raise Error("TypeError: a Gbdt member reads one target column")
    if call[].y.rows != rows:
        raise Error("mojolearn: y has " + String(call[].y.rows) + " values for " + String(rows) + " rows")
    var multi = _multi_output(config.loss)
    # The classifier's codes become the learner's float32 targets (the
    # adapter's `_codes_f32`); finite by construction. A multi-target y is
    # read column-major (the door admits it in that order).
    var codes = List[Float32]()
    var y_addr = call[].y.data
    if kind == _KIND_CLASSIFIER:
        codes = _binary_codes(call[].y.data, rows, call[].classes)
        y_addr = Int(codes.unsafe_ptr())
    elif target_dim > 1:
        codes = _column_major(call[].y.data, rows, target_dim)
        y_addr = Int(codes.unsafe_ptr())
    if kind != _KIND_CLASSIFIER and not trees_finite_f32(y_addr, rows * target_dim):
        if kind == _KIND_REGRESSOR:
            raise Error("y must be finite")
        raise Error("Input y contains NaN or infinity" + (" (the class labels)" if multi else "")
                    + " (GradientBoosting).")
    if config.nan_mode == "Forbidden" and not trees_finite_f32(call[].x.data, rows * cols):
        if _has_nan(call[].x.data, rows * cols):
            raise Error("mojolearn: nan_mode='Forbidden' but X contains NaN. "
                        + "CatBoost refuses this pair; this implementation would otherwise "
                        + "bin the NaNs silently with no NaN bin. Use "
                        + "nan_mode='Min' or 'Max', or clean the column.")
    var n_classes = 0
    if multi:
        n_classes = _code_classes(y_addr, rows, config.loss)
        if len(config.class_weights) != 0 and len(config.class_weights) != n_classes:
            raise Error("mojolearn: class_weights needs " + String(n_classes) + " entries for " + config.loss)
    var flags = _feature_flags(slot, cols)
    var n_flags = cols if len(slot[].cat_given) + len(slot[].hot_given) > 0 else 0
    var entry = entry_ref_gil_free(call[].route.entry)
    var row_major = False
    if n_flags == 0:
        try:
            row_major = xg_row_major(entry)
        except:
            row_major = False
    var scratch = List[Float32]()
    var x_addr = call[].x.data
    if not row_major:
        scratch = _column_major(call[].x.data, rows, cols)
        x_addr = Int(scratch.unsafe_ptr())
    # The category map of this fit: the dense codes replace the categorical
    # columns of the member's own column-major copy (the caller's X is never
    # written).
    var fit_map = len(cat_columns) != 0
    var fitted_map = List[UInt64]()
    if not fit_map and (call[].cat_keys.rows > 0 or call[].eval_cat_keys.rows > 0):
        raise Error("TypeError: a Gbdt member reads categorical keys only with cat_features")
    if fit_map:
        fitted_map = _category_fit(cat_columns, x_addr, rows, call[].cat_keys)
    var n_weights = 0
    var w_addr = call[].x.data
    if call[].weights.rows > 0:
        _dense(call[].weights, MEMBER_F32, "sample weights")
        if call[].weights.rows * call[].weights.cols != rows:
            raise Error("ValueError: sample_weight has " + String(call[].weights.rows * call[].weights.cols)
                        + " values for " + String(rows) + " rows")
        w_addr = call[].weights.data
        n_weights = rows
    # THE HELD-OUT SET (lane mf-left4): the fit door's `_eval_set` on the
    # member's views, in its order and words; its X column-major (the door
    # admits it F-order on either route) with the fit's category map over it.
    var n_eval = 0
    var eval_constant = False
    var eval_cm = List[Float32]()
    var eval_targets = List[Float32]()
    var ex_addr = x_addr
    var ey_addr = y_addr
    if call[].eval_x.rows > 0:
        _dense(call[].eval_x, MEMBER_F32, "eval_set X")
        if call[].eval_x.cols != cols:
            raise Error("mojolearn: eval_set X has " + String(call[].eval_x.cols) + " features "
                        + "for a fit on " + String(cols))
        n_eval = call[].eval_x.rows
        eval_cm = _column_major(call[].eval_x.data, n_eval, cols)
        if fit_map:
            _category_apply_map(fitted_map, cols, Int(eval_cm.unsafe_ptr()), n_eval, call[].eval_cat_keys)
        if kind == _KIND_CLASSIFIER:
            _dense(call[].eval_y, MEMBER_I32, "eval_set y (class codes)")
        else:
            _dense(call[].eval_y, MEMBER_F32, "eval_set y")
        if call[].eval_y.cols != 1:
            raise Error("TypeError: a Gbdt member reads one eval_set target column")
        if call[].eval_y.rows != n_eval:
            raise Error("mojolearn: eval_set y has " + String(call[].eval_y.rows) + " values for "
                        + String(n_eval) + " rows")
        ey_addr = call[].eval_y.data
        if kind == _KIND_CLASSIFIER:
            # the caller's int32 codes of the two classes, as the fit's y
            eval_targets = _binary_codes(call[].eval_y.data, n_eval, call[].classes)
            ey_addr = Int(eval_targets.unsafe_ptr())
        elif not trees_finite_f32(ey_addr, n_eval):
            raise Error("mojolearn: " + (String("eval_set y (the eval_set labels)") if multi else String("eval_set y"))
                        + " must be finite (no NaN or infinity)")
        var scanned = _eval_scan(ey_addr, n_eval, multi)
        if multi and (scanned[1] or Int(scanned[2]) >= n_classes):
            raise Error("mojolearn: eval_set labels must be integer class codes "
                        + "within the training class range")
        eval_constant = scanned[0]
        ex_addr = Int(eval_cm.unsafe_ptr())
    if call[].pairs_weight.rows > 0 and call[].pairs.rows == 0:
        raise Error("mojolearn: pairs_weight needs pairs")
    var paired = call[].pairs.rows > 0
    if paired:
        if config.loss != "PairLogit":
            raise Error("NotImplementedError: mojolearn: pairs are read by loss='PairLogit' only; "
                        + "loss='" + config.loss + "' does not use them")
        if not grouped:
            raise Error("NotImplementedError: mojolearn: pairs without group_id are not implemented: "
                        + "the CatBoost reference then regroups and reorders the "
                        + "whole pool by the pairs' connected components; pass "
                        + "group_id")
    var words = gbdt_fit_words(config, rows, cols, n_flags, n_weights, n_eval, eval_constant)
    if target_dim != 1:
        gbdt_pad_tail(words)
        gbdt_push_int(words, target_dim)
    var sizes = List[UInt32]()
    var pair_words = List[UInt32]()
    var pair_weights = List[Float32]()
    if grouped:
        sizes = _group_sizes(call, rows)
        gbdt_pad_tail(words)
        gbdt_push_int(words, Int(sizes.unsafe_ptr()))
        gbdt_push_int(words, len(sizes))
        if config.loss == "PairLogit":
            if not paired:
                # No pairs: every pair within a group, as the door with none.
                gbdt_push_int(words, Int(sizes.unsafe_ptr()))
                gbdt_push_int(words, -1)
                gbdt_push_int(words, Int(sizes.unsafe_ptr()))
            else:
                # The caller's pairs (lane mf-left4), as the door pushes them.
                _pairs_words(call, rows, pair_words, pair_weights)
                gbdt_push_int(words, Int(pair_words.unsafe_ptr()))
                gbdt_push_int(words, len(pair_weights))
                gbdt_push_int(words, Int(pair_weights.unsafe_ptr()))
    var options = gbdt_fit_options(config, rows)
    var rate = words.rate
    # The word lists move into the request by swap: moving the fields would leave
    # `words` and `options` partly destroyed.
    var fit_ints = List[Int]()
    swap(fit_ints, words.ints)
    var fit_floats = List[Float64]()
    swap(fit_floats, words.floats)
    var fit_bad = List[UInt8]()
    swap(fit_bad, words.bad)
    var fit_strs = List[String]()
    swap(fit_strs, options.strs)
    var args = GbdtFitArgs(x_addr, y_addr, w_addr, Int(flags.unsafe_ptr()), ex_addr, ey_addr,
                           fit_ints^, fit_floats^, fit_bad^, fit_strs^)
    var outcome = xg_fit(entry, row_major, args)
    # `feature_importances_` (lane mf-imp6): the door's RUN on the same block
    # the fit read (its layout, the categorical columns as their codes, the
    # fit's weights), on the route's column.
    var imp = xg_importances(entry, outcome.text, x_addr, rows, cols, row_major, w_addr, n_weights)
    _ = len(scratch)  # the column-major copy lives until the fit and the importances returned
    _ = len(codes)  # the classifier's float32 targets (or the column-major y) too
    _ = len(flags)
    _ = len(sizes)
    _ = len(eval_cm)  # the held-out X and targets the fit read
    _ = len(eval_targets)
    _ = len(pair_words)  # the pairs and their weights the fit read
    _ = len(pair_weights)
    var ctr = gbdt_text_has_ctr(outcome.text)
    if ctr and not fit_map:
        raise Error("RuntimeError: mojolearn: a Gbdt member fit returned a CTR model (no cat_features were given)")
    var dim: Int
    if ctr and entry_ref_vendor(entry) == 0:
        # The host image's CTR model: its dimension from the model text's own
        # parse (the door reads the same word, `host_archive_gbdt_state`).
        var parsed = host_archive_gbdt_text(outcome.text)
        dim = parsed.metadata["dim"]
    else:
        dim = xg_model_dim(entry, outcome.text)
    _clear[I](slot)
    swap(slot[].cat_map, fitted_map)
    swap(slot[].imp, imp)
    slot[].bias = gbdt_bias_of(outcome.text)
    swap(slot[].text, outcome.text)  # by swap: `outcome` is read after (moving a field out of it is refused)
    swap(slot[].learn, outcome.learn)
    # The held-out curve (`test_loss_curve_`: None without an eval set).
    slot[].has_test = n_eval > 0
    if n_eval > 0:
        swap(slot[].test, outcome.test)
    slot[].best_iteration = outcome.best_iteration
    slot[].stopped_early = outcome.stopped_early
    slot[].features = cols
    slot[].dim = dim
    slot[].classes = dim + 1 if config.loss == "MultiClass" else (dim if config.loss == "MultiClassOneVsAll" else 0)
    slot[].rate = rate
    slot[].ordered = options.boosting == "Ordered"
    slot[].fitted = True


def _fit_feature_freq[I: GbdtMemberImage](slot: _SlotPtr, call: _CallPtr) raises:
    """`ExperimentalTwoLevelFeatureFreq.fit(X, y, sample_weight=)` on the member's views: the door's
    checks in its order and words (`gbdt_feature_freq_check` over the member's column-major copy of X, as
    the door admits X in F order; y finite, one per row; `gbdt_weights_check`), then the estimator's own
    fit (`xg_feature_freq_fit` through the route's gbdt entry: the body its door runs) and the state the
    door's PACK publishes (one model dimension, bias 0, no curves)."""
    _whole(call)
    var name = _name(slot[].kind)
    if call[].fit_output != 0:
        raise Error("TypeError: a Gbdt member has no fit output")
    if (call[].eval_x.rows > 0 or call[].groups.rows > 0 or call[].pairs.rows > 0 or call[].cat_keys.rows > 0
            or call[].eval_cat_keys.rows > 0):
        raise Error("TypeError: an " + name + " member's fit reads X, y and sample weights only")
    _dense(call[].x, MEMBER_F32, "X")
    var rows = call[].x.rows
    var cols = call[].x.cols
    var xcm = _column_major(call[].x.data, rows, cols)
    gbdt_feature_freq_check(Int(xcm.unsafe_ptr()), rows, cols, slot[].ff_sources)
    _dense(call[].y, MEMBER_F32, "y")
    if call[].y.cols != 1 or call[].y.rows != rows or not trees_finite_f32(call[].y.data, rows):
        raise Error("mojolearn: y must be finite with one value per row")
    var w_addr = call[].y.data
    var n_weights = 0
    if call[].weights.rows > 0:
        _dense(call[].weights, MEMBER_F32, "sample weights")
        if call[].weights.rows * call[].weights.cols != rows:
            raise Error("mojolearn: sample_weight must have one value per row")
        gbdt_weights_check(call[].weights.data, rows)
        w_addr = call[].weights.data
        n_weights = rows
    var src = List[UInt32](length=len(slot[].ff_sources), fill=UInt32(0))
    for i in range(len(slot[].ff_sources)):  # small-loop(the source columns)
        src[i] = UInt32(slot[].ff_sources[i])
    var entry = entry_ref_gil_free(call[].route.entry)
    var text = xg_feature_freq_fit(entry, Int(xcm.unsafe_ptr()), call[].y.data, w_addr, Int(src.unsafe_ptr()), rows,
                                   cols, n_weights, len(src), slot[].ff_rate, slot[].ff_l2, slot[].ff_seed)
    var imp = xg_importances(entry, text, Int(xcm.unsafe_ptr()), rows, cols, False, w_addr, n_weights)
    _ = len(xcm)  # the column-major copy lives until the fit and the importances returned
    _ = len(src)
    _clear[I](slot)
    swap(slot[].imp, imp)
    slot[].text = text^
    slot[].learn = List[Float64]()
    slot[].has_test = False
    slot[].bias = 0.0
    slot[].best_iteration = 0
    slot[].stopped_early = False
    slot[].features = cols
    slot[].dim = 1
    slot[].classes = 0
    slot[].rate = slot[].ff_rate
    slot[].ordered = False
    slot[].fitted = True


def _fit_ordered[I: GbdtMemberImage](slot: _SlotPtr, call: _CallPtr) raises:
    """`OrderedRMSE.fit(X, y, permutation=, sample_weight=)` on the member's views (lane mf-hostent5): the
    door's checks in its order and words (`gbdt_ordered_rows_check` over the member's column-major copy of X, as
    the door admits X in F order; `gbdt_ordered_target_check`; the permutation through `gbdt_ordered_order`, its
    bijection test and uint32 cast; `gbdt_weights_check`), then the estimator's own fit (`xg_ordered_fit`
    through the route's gbdt entry: the body its door runs) and the state the door's PACK publishes (one model
    dimension, bias 0, no curves). The door's settings checks (`_ordered_options`) ran at the caller's UNPACK.
    No permutation given (every caller but one that hands the fit parameters): Python's TypeError for the
    missing keyword-only argument, as `OrderedRMSE.fit(X, y)` raises it."""
    _whole(call)
    var name = _name(slot[].kind)
    if call[].fit_output != 0:
        raise Error("TypeError: a Gbdt member has no fit output")
    if (call[].eval_x.rows > 0 or call[].groups.rows > 0 or call[].pairs.rows > 0 or call[].cat_keys.rows > 0
            or call[].eval_cat_keys.rows > 0):
        raise Error("TypeError: an " + name + " member's fit reads X, y, the permutation and sample weights only")
    if call[].permutation.rows == 0:
        raise Error(_ORDERED_NO_PERMUTATION)
    _dense(call[].x, MEMBER_F32, "X")
    var rows = call[].x.rows
    var cols = call[].x.cols
    var xcm = _column_major(call[].x.data, rows, cols)
    gbdt_ordered_rows_check(Int(xcm.unsafe_ptr()), rows, cols)
    _dense(call[].y, MEMBER_F32, "y")
    if call[].y.cols != 1 or call[].y.rows != rows:
        raise Error("mojolearn: y must be one-dimensional with one value per row")
    gbdt_ordered_target_check(call[].y.data, rows)
    var perm = call[].permutation
    if perm.home == MEMBER_HOME_DEVICE or not perm.dense_host(MEMBER_I64):
        raise Error("TypeError: a Gbdt member reads the permutation as a dense host int64 view")
    var order = gbdt_ordered_order(perm.data, perm.rows * perm.cols, rows)
    var w_addr = call[].y.data
    var n_weights = 0
    if call[].weights.rows > 0:
        _dense(call[].weights, MEMBER_F32, "sample weights")
        if call[].weights.rows * call[].weights.cols != rows:
            raise Error("mojolearn: sample_weight must have one value per row")
        gbdt_weights_check(call[].weights.data, rows)
        w_addr = call[].weights.data
        n_weights = rows
    var entry = entry_ref_gil_free(call[].route.entry)
    var text = xg_ordered_fit(entry, Int(xcm.unsafe_ptr()), call[].y.data, w_addr, Int(order.unsafe_ptr()), rows,
                              cols, n_weights, slot[].od_trees, slot[].od_depth, slot[].od_border, slot[].od_rate,
                              slot[].od_l2)
    var imp = xg_importances(entry, text, Int(xcm.unsafe_ptr()), rows, cols, False, w_addr, n_weights)
    _ = len(xcm)  # the column-major copy lives until the fit and the importances returned
    _ = len(order)
    _clear[I](slot)
    swap(slot[].imp, imp)
    slot[].text = text^
    slot[].learn = List[Float64]()
    slot[].has_test = False
    slot[].bias = 0.0
    slot[].best_iteration = 0
    slot[].stopped_early = False
    slot[].features = cols
    slot[].dim = 1
    slot[].classes = 0
    slot[].rate = slot[].od_rate
    slot[].ordered = False
    slot[].fitted = True


def _require_fitted(slot: _SlotPtr, before: String) raises:
    if not slot[].fitted:
        raise Error("RuntimeError: mojolearn: " + before + " before fit()")


def _x(slot: _SlotPtr, call: _CallPtr) raises:
    _whole(call)
    _dense(call[].x, MEMBER_F32, "X")
    if call[].x.cols != slot[].features:
        raise Error("mojolearn: model was fitted on " + String(slot[].features) + " features, got "
                    + String(call[].x.cols))


def _resident(slot: _SlotPtr, entry: EntryRef) raises -> Int:
    """The image's resident model of this state (prepared on the first
    predict, as `GradientBoosting._resident_handle`), or -1 on an image
    without one."""
    if slot[].resident >= 0:
        return slot[].resident
    if not xg_resident(entry):
        return -1
    slot[].resident = xg_resident_prepare(entry, Int(slot[].text.unsafe_ptr()), slot[].text.byte_length())
    return slot[].resident


def _apply(slot: _SlotPtr, call: _CallPtr, mode: Int, dst: Int) raises -> Int:
    """One predict in `mode` into `dst`: the resident model (row-major X as
    it is), else the per-call parse of a column-major copy. A model fitted
    with cat_features reads the column-major copy with the fitted map's
    codes over its categorical columns (the door's `_fitted_x`), on either
    route. The width per row written."""
    var entry = entry_ref_gil_free(call[].route.entry)
    var rows = call[].x.rows
    var handle = _resident(slot, entry)
    var coded = len(slot[].cat_map) > 0
    if handle >= 0 and not coded:
        return xg_resident_predict(entry, handle, call[].x.data, rows, dst, mode, True)
    var cm = _column_major(call[].x.data, rows, call[].x.cols)
    if coded:
        _category_apply(slot, Int(cm.unsafe_ptr()), rows, call[].cat_keys)
        if handle >= 0:
            var wrote = xg_resident_predict(entry, handle, Int(cm.unsafe_ptr()), rows, dst, mode, False)
            _ = len(cm)
            return wrote
    var text = Int(slot[].text.unsafe_ptr())
    var nbytes = slot[].text.byte_length()
    var width: Int
    if mode == _PREDICT_RAW and slot[].dim == 1:
        var wrote = xg_predict(entry, text, nbytes, Int(cm.unsafe_ptr()), rows, dst)
        if wrote != rows:
            raise Error("RuntimeError: mojolearn: predict wrote " + String(wrote) + " of " + String(rows) + " rows")
        width = 1
    else:
        width = xg_predict_multi(entry, text, nbytes, Int(cm.unsafe_ptr()), rows, dst, mode)
    _ = len(cm)
    return width


def _binary_out(slot: _SlotPtr, call: _CallPtr, probabilities: Bool, dst: Int) raises:
    """GradientBoostingClassifier's `_binary_output` into `dst` (int32 codes,
    rows, or float32 `[1 - p, p]`, rows x 2): the learner's raw margins (the
    route `_apply` takes), the adapter's finiteness refusal, then the image's
    XG_BINARY transform. X is checked by the caller."""
    var rows = call[].x.rows
    if rows == 0:
        return
    var raw = List[Float32](length=rows, fill=0.0)
    var width = _apply(slot, call, _PREDICT_RAW, Int(raw.unsafe_ptr()))
    if width != 1:
        raise Error("RuntimeError: mojolearn: a binary GBDT model wrote width " + String(width))
    if not trees_finite_f32(Int(raw.unsafe_ptr()), rows):
        raise Error("Classifier margins must be finite scalar Float32 values")
    var entry = entry_ref_gil_free(call[].route.entry)
    var wrote = xg_binary(entry, probabilities, Int(raw.unsafe_ptr()), dst, rows)
    _ = len(raw)
    if wrote != (2 * rows if probabilities else rows):
        raise Error("RuntimeError: GBDT binary output returned an unexpected row count")


def _sigmoid_pair(slot: _SlotPtr, call: _CallPtr, dst: Int) raises:
    """`GradientBoosting.predict_proba` of a binary loss into `dst` (float64
    rows x 2): the resident sigmoid pair, else the raw scores widened and
    the image's XG_SIGMOID_PAIR."""
    var rows = call[].x.rows
    var entry = entry_ref_gil_free(call[].route.entry)
    if _resident(slot, entry) >= 0:
        var width = _apply(slot, call, _PREDICT_SIGMOID_PAIR, dst)
        if width != 2:
            raise Error("RuntimeError: mojolearn: predict_proba wrote width " + String(width) + ", expected 2")
        return
    var raw32 = List[Float32](length=max(rows, 1), fill=0.0)
    _ = _apply(slot, call, _PREDICT_RAW, Int(raw32.unsafe_ptr()))
    var raw = List[Float64](length=max(rows, 1), fill=0.0)
    _widen(Int(raw32.unsafe_ptr()), Int(raw.unsafe_ptr()), rows)
    var wrote = xg_sigmoid_pair(entry, Int(raw.unsafe_ptr()), dst, rows)
    _ = len(raw)
    _ = len(raw32)
    if wrote != rows:
        raise Error("RuntimeError: mojolearn: gbdt_sigmoid_pair wrote " + String(wrote) + " of " + String(rows) + " rows")


def _narrow_codes(src: Int, dst: Int, rows: Int):
    """int64 class codes to the member's int32 codes, split over the host
    workers."""
    var tasks = host_predict_task_count_work(rows, rows)
    var chunk = host_predict_chunk(rows, tasks)
    var source = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=src)
    var target = _I32P(unsafe_from_address=dst)

    def task(unit: Int) {imm source, imm target, imm chunk, imm rows}:
        for i in range(unit * chunk, min((unit + 1) * chunk, rows)):  # native data loop (one host task's rows)
            target[i] = Int32(source[i])

    host_parallelize(task, tasks)


def _fast_codes(slot: _SlotPtr, call: _CallPtr, dst: Int) raises:
    """GradientBoostingClassifier's FAST class codes (`_binary_output` with
    numeric_mode_='fast': the learner's `predict_classes`, lane mf-img3):
    the resident direct-class kernel where the image holds the model, else
    the first-max argmax of the learner's `predict_proba`; int32 codes into
    `dst`."""
    var rows = call[].x.rows
    if rows == 0:
        return
    var wide = List[Int64](length=rows, fill=0)
    var entry = entry_ref_gil_free(call[].route.entry)
    if _resident(slot, entry) >= 0:
        var width = _apply(slot, call, _PREDICT_CLASSES_BINARY, Int(wide.unsafe_ptr()))
        if width != 1:
            raise Error("resident binary class prediction returned an invalid width")
    else:
        var proba = List[Float64](length=2 * rows, fill=0.0)
        _sigmoid_pair(slot, call, Int(proba.unsafe_ptr()))
        var tasks = host_predict_task_count_work(rows, 2 * rows)
        var chunk = host_predict_chunk(rows, tasks)
        var pp = Int(proba.unsafe_ptr())
        var wp = Int(wide.unsafe_ptr())

        def task(unit: Int) {imm pp, imm wp, imm chunk, imm rows}:
            var begin = unit * chunk
            var end = min(begin + chunk, rows)
            if end > begin:
                argmax_rows_f64(_F64P(unsafe_from_address=pp + begin * 16), end - begin, 2,
                    MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=wp + begin * 8))

        host_parallelize(task, tasks)
        _ = len(proba)
    _narrow_codes(Int(wide.unsafe_ptr()), dst, rows)
    _ = len(wide)


def _class_codes(slot: _SlotPtr, call: _CallPtr, dst: Int) raises:
    """GradientBoostingClassifier's int32 class codes (`_binary_output(X,
    False)`): FAST (the route's tier word 0) the learner's class kernels,
    else the margins through the image's XG_BINARY transform."""
    if call[].route.mode == 0:
        _fast_codes(slot, call, dst)
    else:
        _binary_out(slot, call, False, dst)


def _classify(slot: _SlotPtr, call: _CallPtr, probabilities: Bool) raises:
    """GradientBoostingClassifier's `predict` (int32 codes, rows x 1) and
    `predict_proba` (float32 rows x 2)."""
    _require_fitted(slot, String("predict_proba()") if probabilities else String("predict()"))
    _x(slot, call)
    if probabilities:
        _dense(call[].dst, MEMBER_F32, "the predict_proba output")
        if call[].dst.rows != call[].x.rows or call[].dst.cols != 2:
            raise Error("TypeError: a GradientBoostingClassifier member predict_proba writes rows x 2 float32")
    else:
        _dense(call[].dst, MEMBER_I32, "the predict output (class codes)")
        if call[].dst.rows != call[].x.rows or call[].dst.cols != 1:
            raise Error("TypeError: a GradientBoostingClassifier member predict writes rows x 1 int32 codes")
    if probabilities:
        _binary_out(slot, call, True, call[].dst.data)
    else:
        _class_codes(slot, call, call[].dst.data)


def _decision(slot: _SlotPtr, call: _CallPtr) raises:
    """GradientBoostingClassifier's `decision_function`: the learner's raw
    float32 margin, rows x 1 (one column: a binary decision)."""
    if slot[].kind != _KIND_CLASSIFIER:
        member_unsupported(_name(slot[].kind), MEMBER_OP_DECISION)
    _require_fitted(slot, "decision_function()")
    _x(slot, call)
    _dense(call[].dst, MEMBER_F32, "the decision output")
    if call[].dst.rows != call[].x.rows or call[].dst.cols != 1:
        raise Error("TypeError: a GradientBoostingClassifier member decision_function writes rows x 1 float32")
    if call[].x.rows == 0:
        return
    var width = _apply(slot, call, _PREDICT_RAW, call[].dst.data)
    if width != 1:
        raise Error("RuntimeError: mojolearn: a binary GBDT model wrote width " + String(width))


def _score(slot: _SlotPtr, call: _CallPtr) raises:
    """GradientBoostingClassifier's `score` without weights: the accuracy of
    the predicted codes against the caller's int32 codes, float32 count /
    rows widened (the adapter's `accuracy_score`; integer counts merged
    exactly over the host workers). A weighted score is the caller's metric
    (no MEMBER_CAN_WEIGHTED_SCORE)."""
    var name = _name(slot[].kind)
    if slot[].kind == _KIND_REGRESSOR:
        _score_r2(slot, call)
        return
    if slot[].kind != _KIND_CLASSIFIER:
        member_unsupported(name, MEMBER_OP_SCORE)
    if call[].weights.rows > 0:
        raise Error("NotImplementedError: mojolearn: a weighted score of a " + name
                    + " member is not published; the caller predicts and runs its own weighted metric")
    _require_fitted(slot, "score()")
    _x(slot, call)
    _dense(call[].y, MEMBER_I32, "y (class codes)")
    var rows = call[].x.rows
    if call[].y.cols != 1 or call[].y.rows != rows:
        raise Error("mojolearn metrics: y_true has " + String(call[].y.rows * call[].y.cols) + " entries and "
                    + "y_pred has " + String(rows) + "; they must be the same length")
    if rows <= 0:
        raise Error("accuracy_score: n must be positive, got " + String(rows) + " (0 / 0 is refused by name)")
    var codes = List[Int32](length=rows, fill=0)
    _class_codes(slot, call, Int(codes.unsafe_ptr()))
    var tasks = host_predict_task_count_work(rows, rows)
    var chunk = host_predict_chunk(rows, tasks)
    var partials = List[Int](length=max(tasks, 1), fill=0)
    var truth = _I32P(unsafe_from_address=call[].y.data)
    var guess = _I32P(unsafe_from_address=Int(codes.unsafe_ptr()))
    var part_p = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=Int(partials.unsafe_ptr()))

    def count(unit: Int) {imm truth, imm guess, imm part_p, imm chunk, imm rows}:
        var hits = 0
        for i in range(unit * chunk, min((unit + 1) * chunk, rows)):  # native data loop (one host task's rows)
            if truth[i] == guess[i]:
                hits += 1
        part_p[unit] = hits

    host_parallelize(count, tasks)
    var hits = 0
    for t in range(len(partials)):  # small-loop(tasks): one count per host task
        hits += partials[t]
    _ = len(codes)
    call[].value = Float64(Float32(hits) / Float32(rows))


def _score_r2(slot: _SlotPtr, call: _CallPtr) raises:
    """GradientBoostingRegressor's `score` without weights (lane w3f3-c,
    2026-11-24): the float32 R^2 kernel of the route's metrics image
    (core/metrics_entry_table.mojo MR_R2, `route.third`, resolved once by
    the member's UNPACK) over the raw float32 predict, the kernel the
    adapter's own score runs (bindings/trees_public_mixin_score_unpack.mojo
    `gbdt_adapter_score` -> `trees_r2` -> `r2_score`). Before, a search with
    scoring=None scored this class by the x_metrics R^2 program
    (`_own_score`'s R^2 row), another number than the estimator's own score
    (test_parallel_model_selection: cross_val_score serial vs the process
    workers' `estimator.score`). A weighted score is the caller's metric."""
    var name = _name(slot[].kind)
    if call[].weights.rows > 0:
        raise Error("NotImplementedError: mojolearn: a weighted score of a " + name
                    + " member is not published; the caller predicts and runs its own weighted metric")
    var route = call[].route
    if not route.has_third:
        raise Error("ImportError: mojolearn: the " + name + " member's score needs the R^2 entry on its route")
    _require_fitted(slot, "score()")
    _x(slot, call)
    _dense(call[].y, MEMBER_F32, "y")
    var rows = call[].x.rows
    if call[].y.cols != 1 or call[].y.rows <= 0:
        raise Error("y must be nonempty one-dimensional targets")
    if not trees_finite_f32(call[].y.data, call[].y.rows):
        raise Error("y must be finite")
    if call[].y.rows != rows:
        raise Error("mojolearn metrics: y_true has " + String(call[].y.rows) + " entries and "
                    + "y_pred has " + String(rows) + "; they must be the same length")
    var prediction = List[Float32](length=max(rows * slot[].dim, 1), fill=0.0)
    var width = _apply(slot, call, _PREDICT_RAW, Int(prediction.unsafe_ptr()))
    if width != 1:
        raise Error("RuntimeError: mojolearn: predict wrote width " + String(width) + ", expected 1")
    var value = List[Float64](length=1, fill=0.0)
    var words: List[Int] = [rows]
    var addresses: List[Int] = [call[].y.data, Int(prediction.unsafe_ptr()), Int(value.unsafe_ptr())]
    _ = native_descriptor_call(route.third.call, MR_R2, addresses, words, List[Float64]())
    call[].value = value[0]
    _ = len(prediction)  # read by the kernel through its address
    _ = len(value)


def _predict(slot: _SlotPtr, call: _CallPtr) raises:
    """`GradientBoosting.predict`: raw float32 scores, rows x approx_dim
    (the regressor's: rows x 1); the classifier's: `_classify`."""
    if slot[].kind == _KIND_CLASSIFIER:
        _classify(slot, call, False)
        return
    _require_fitted(slot, "predict()")
    _x(slot, call)
    _dense(call[].dst, MEMBER_F32, "the predict output")
    if call[].dst.rows != call[].x.rows or call[].dst.cols != slot[].dim:
        raise Error("TypeError: a Gbdt member predict writes rows x " + String(slot[].dim) + " float32")
    var width = _apply(slot, call, _PREDICT_RAW, call[].dst.data)
    if width != slot[].dim:
        raise Error("RuntimeError: mojolearn: predict wrote width " + String(width) + ", expected " + String(slot[].dim))


def _widen(src: Int, dst: Int, count: Int):
    """float32 raw scores to float64, split over the host workers."""
    var tasks = host_predict_task_count_work(count, count)
    var chunk = host_predict_chunk(count, tasks)
    var source = _F32P(unsafe_from_address=src)
    var target = _F64P(unsafe_from_address=dst)

    def task(unit: Int) {imm source, imm target, imm chunk, imm count}:
        for i in range(unit * chunk, min((unit + 1) * chunk, count)):  # native data loop (one host task's rows)
            target[i] = Float64(source[i])

    host_parallelize(task, tasks)


def _host_proba_refusal(loss: String) raises:
    """`HostGBDT.predict_proba`'s refusals, in its words (bindings/host_tree_native_door.mojo
    `host_tree_operation`): a multiclass loss, then any loss but Logloss / CrossEntropy."""
    if loss == "MultiClass" or loss == "MultiClassOneVsAll":
        raise Error("NotImplementedError: HostGBDT.predict_proba has no qualified multiclass transform")
    if not _binary(loss):
        raise Error("predict_proba is defined for Logloss and CrossEntropy")


def _proba(slot: _SlotPtr, call: _CallPtr) raises:
    """`GradientBoosting.predict_proba`: the same routes and dtypes (the
    classifier's: `_classify`; the regressor has none)."""
    if slot[].kind == _KIND_CLASSIFIER:
        _classify(slot, call, True)
        return
    if slot[].kind == _KIND_REGRESSOR or slot[].kind == _KIND_FEATURE_FREQ or slot[].kind == _KIND_ORDERED:
        member_unsupported(_name(slot[].kind), MEMBER_OP_PREDICT_PROBA)
    ref loss = slot[].config.loss
    var rows = call[].x.rows
    if slot[].kind == _KIND_HOST:
        _host_proba_refusal(loss)
    if _multi_output(loss):
        _require_fitted(slot, "predict_proba()")
        _x(slot, call)
        _dense(call[].dst, MEMBER_F32, "the predict_proba output")
        if call[].dst.rows != rows or call[].dst.cols != slot[].classes:
            raise Error("TypeError: a Gbdt member predict_proba writes rows x " + String(slot[].classes) + " float32")
        var mode = _PREDICT_SIGMOID if loss == "MultiClassOneVsAll" else _PREDICT_SOFTMAX
        var width = _apply(slot, call, mode, call[].dst.data)
        if width != slot[].classes:
            raise Error("RuntimeError: mojolearn: predict_proba wrote width " + String(width) + ", expected "
                        + String(slot[].classes))
        return
    if not _binary(loss):
        raise Error("mojolearn: predict_proba is defined for Logloss, "
                    + "CrossEntropy, MultiClass and MultiClassOneVsAll; this model was fitted with '"
                    + loss + "'. Use predict() and apply the link yourself.")
    _require_fitted(slot, "predict_proba()")
    _x(slot, call)
    _dense(call[].dst, MEMBER_F64, "the predict_proba output")
    if call[].dst.rows != rows or call[].dst.cols != 2:
        raise Error("TypeError: a Gbdt member predict_proba writes rows x 2 float64")
    _sigmoid_pair(slot, call, call[].dst.data)


def _block_of(index: Int) raises -> Int:
    if index < 0 or index >= _BLOCKS:
        raise Error("member ABI: a Gbdt member has six fitted blocks")
    if index == 5:
        return MEMBER_BLOCK_FEATURE_IMPORTANCES
    if index == 4:
        return MEMBER_BLOCK_FAMILY + 3
    if index == 3:
        return MEMBER_BLOCK_FAMILY + 2
    return MEMBER_BLOCK_SCALARS if index == 2 else MEMBER_BLOCK_FAMILY + index


def _export(slot: _SlotPtr, call: _CallPtr) raises:
    """A view of one fitted block (the slot owns the words; valid until the
    state's next fit, import or free)."""
    _require_fitted(slot, "export")
    var block = call[].block
    if (call[].flags & MEMBER_BLOCK_BY_INDEX) != 0:
        block = _block_of(block)
    if block == MEMBER_BLOCK_FAMILY:
        var nbytes = slot[].text.byte_length()
        if len(slot[].x_text) * 4 < nbytes or len(slot[].x_text) == 0:
            slot[].x_text = List[Int32](length=max((nbytes + 3) // 4, 1), fill=0)
            var into = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(slot[].x_text.unsafe_ptr()))
            var from_text = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(slot[].text.unsafe_ptr()))
            unsafe_memcpy(dest=into, src=from_text, count=nbytes)
        call[].dst = MemberMatrix(Int(slot[].x_text.unsafe_ptr()), 1, len(slot[].x_text), MEMBER_I32)
    elif block == MEMBER_BLOCK_FAMILY + 1:
        call[].dst = MemberMatrix(Int(slot[].learn.unsafe_ptr()), 1, len(slot[].learn), MEMBER_F64)
    elif block == MEMBER_BLOCK_FAMILY + 2:
        # The category map (no rows for a model fitted without cat_features).
        var words = len(slot[].cat_map)
        call[].dst = MemberMatrix(Int(slot[].cat_map.unsafe_ptr()), 1 if words > 0 else 0, words, MEMBER_I64)
    elif block == MEMBER_BLOCK_FAMILY + 3:
        # The held-out loss curve (lane mf-left4): no rows for a fit without
        # an eval set (`test_loss_curve_` None), else one row of its values.
        call[].dst = MemberMatrix(Int(slot[].test.unsafe_ptr()), 1 if slot[].has_test else 0, len(slot[].test),
                                  MEMBER_F64)
    elif block == MEMBER_BLOCK_FEATURE_IMPORTANCES:
        # `feature_importances_` (lane mf-imp6): float32 1 x features; no
        # rows for an imported model saved without them.
        var n = len(slot[].imp)
        call[].dst = MemberMatrix(Int(slot[].imp.unsafe_ptr()), 1 if n > 0 else 0, n, MEMBER_F32)
    elif block == MEMBER_BLOCK_SCALARS:
        slot[].x_scalars = [Float64(slot[].features), Float64(slot[].dim), Float64(slot[].classes), slot[].bias,
                            Float64(slot[].best_iteration), 1.0 if slot[].stopped_early else 0.0, slot[].rate,
                            Float64(slot[].text.byte_length()), 1.0 if slot[].ordered else 0.0]
        call[].dst = MemberMatrix(Int(slot[].x_scalars.unsafe_ptr()), 1, _SCALARS, MEMBER_F64)
    else:
        raise Error("member ABI: a Gbdt member has no fitted block " + String(block))
    call[].block = block
    call[].out_cols = _BLOCKS


def _import[I: GbdtMemberImage](slot: _SlotPtr, call: _CallPtr) raises:
    """Copy one fitted block in; MEMBER_IMPORT_DONE builds the model text
    and marks the state fitted."""
    var block = call[].block
    if slot[].fitted:
        _clear[I](slot)
    var view = call[].x
    if block == MEMBER_BLOCK_FAMILY:
        _dense(view, MEMBER_I32, "a fitted block")
        slot[].x_text = List[Int32](length=max(view.rows * view.cols, 1), fill=0)
        unsafe_memcpy(dest=slot[].x_text.unsafe_ptr(), src=_I32P(unsafe_from_address=view.data),
                      count=view.rows * view.cols)
    elif block == MEMBER_BLOCK_FAMILY + 1:
        _dense(view, MEMBER_F64, "a fitted block")
        slot[].learn = List[Float64](length=view.rows * view.cols, fill=0.0)
        unsafe_memcpy(dest=slot[].learn.unsafe_ptr(), src=_F64P(unsafe_from_address=view.data),
                      count=view.rows * view.cols)
    elif block == MEMBER_BLOCK_FAMILY + 2:
        # The category map of a model fitted with cat_features (optional; before
        # the scalars block).
        var words = view.rows * view.cols
        slot[].cat_map = List[UInt64](length=words, fill=0)
        if words > 0:
            _dense(view, MEMBER_I64, "a fitted block")
            unsafe_memcpy(dest=slot[].cat_map.unsafe_ptr(),
                          src=MutPointer[UInt64, MutUntrackedOrigin](unsafe_from_address=view.data), count=words)
    elif block == MEMBER_BLOCK_FAMILY + 3:
        # The held-out loss curve (optional; lane mf-left4): rows 0 = None.
        slot[].has_test = view.rows > 0
        slot[].test = List[Float64](length=view.rows * view.cols, fill=0.0)
        if view.rows * view.cols > 0:
            _dense(view, MEMBER_F64, "a fitted block")
            unsafe_memcpy(dest=slot[].test.unsafe_ptr(), src=_F64P(unsafe_from_address=view.data),
                          count=view.rows * view.cols)
    elif block == MEMBER_BLOCK_FEATURE_IMPORTANCES:
        # `feature_importances_` (optional; lane mf-imp6): float32, or the
        # public float64 vector (an exact widening of float32, narrowed back
        # exactly).
        var n = view.rows * view.cols
        slot[].imp = List[Float32](length=n, fill=Float32(0.0))
        if n > 0:
            if view.dense_host(MEMBER_F64):
                var src = _F64P(unsafe_from_address=view.data)
                for c in range(n):  # small-loop(one word per input column: the import narrowing)
                    slot[].imp[c] = Float32(src[c])
            else:
                _dense(view, MEMBER_F32, "a fitted block")
                unsafe_memcpy(dest=slot[].imp.unsafe_ptr(), src=_F32P(unsafe_from_address=view.data), count=n)
    elif block == MEMBER_BLOCK_SCALARS:
        _dense(view, MEMBER_F64, "a fitted block")
        if view.rows * view.cols != _SCALARS:
            raise Error("member ABI: the Gbdt scalars block is [features, approx_dim, n_classes, bias, "
                        + "best_iteration, stopped_early, learning_rate, text bytes, boosting]")
        var s = _F64P(unsafe_from_address=view.data)
        slot[].x_scalars = List[Float64](length=_SCALARS, fill=0.0)
        for k in range(_SCALARS):  # small-loop(scalars)
            slot[].x_scalars[k] = s[k]
    else:
        raise Error("member ABI: a Gbdt member has no fitted block " + String(block))
    if (call[].flags & MEMBER_IMPORT_DONE) == 0:
        return
    if len(slot[].x_scalars) != _SCALARS:
        raise Error("member ABI: a fitted Gbdt state needs its three blocks")
    var nbytes = Int(slot[].x_scalars[7])
    if nbytes < 1 or nbytes > len(slot[].x_text) * 4:
        raise Error("member ABI: the Gbdt model text block is shorter than its byte count")
    slot[].text = gbdt_string_at(Int(slot[].x_text.unsafe_ptr()), nbytes)
    slot[].features = Int(slot[].x_scalars[0])
    slot[].dim = Int(slot[].x_scalars[1])
    slot[].classes = Int(slot[].x_scalars[2])
    slot[].bias = slot[].x_scalars[3]
    slot[].best_iteration = Int(slot[].x_scalars[4])
    slot[].stopped_early = slot[].x_scalars[5] != 0.0
    slot[].rate = slot[].x_scalars[6]
    slot[].ordered = slot[].x_scalars[8] != 0.0
    if slot[].features < 1 or slot[].dim < 1:
        raise Error("member ABI: a fitted Gbdt state needs its features and model dimension")
    if len(slot[].imp) != 0 and len(slot[].imp) != slot[].features:
        raise Error("member ABI: the Gbdt feature importances block holds " + String(len(slot[].imp))
                    + " values for " + String(slot[].features) + " features")
    if slot[].kind != _KIND_BOOSTER and slot[].kind != _KIND_HOST and slot[].dim != 1:
        raise Error("member ABI: a " + _name(slot[].kind) + " state has one model dimension")
    if gbdt_text_has_ctr(slot[].text) and len(slot[].cat_map) == 0:
        raise Error("member ABI: a Gbdt CTR model (fitted with cat_features) is imported with its category map")
    if len(slot[].cat_map) > 0:
        _ = category_block_columns(CategoryKeyPtr(unsafe_from_address=Int(slot[].cat_map.unsafe_ptr())),
                                   len(slot[].cat_map), slot[].features)
    slot[].fitted = True


def _capabilities(kind: Int, loss: String) -> Int:
    """No `MEMBER_CAN_TEST_ROWS`: the fit tests y here and X only for
    nan_mode='Forbidden' (NaN; infinities reach the kernel), and predict
    does not test X (the classifier tests its margins, not X). No
    MEMBER_BORROWS_FIT: the state owns its text (the classifier's float32
    targets are freed when its fit returns). No MEMBER_CAN_NOTICES: the
    fits raise no warnings (the adapters' doors raise none either)."""
    # MEMBER_GIL_FREE (THE GIL RULE): every operation of both kinds is traced
    # (fx-gil-RESULT.md, "gbdt"): the route entry is marked `entry_ref_gil_free`
    # at every operation, the image's operations never touch the GIL, and the
    # store (SharedSlot) and the resident registry have their own lock words.
    # The adapter kinds (0, 2) reach nothing the booster's trace did not:
    # the same entry operations (XG_FIT, XG_CAPS, XG_MODEL_DIM, the predicts
    # and the resident prepare / predict) plus XG_BINARY, an operation of the
    # same traced entry (`gbdt_entry_dispatch` runs it bare under the bit;
    # `gbdt_binary_native` touches no CPython; its device context is
    # `process_ctx`, own lock, as XG_SIGMOID_PAIR's), the same store and the
    # same resident registry; their own work is host loops over plain words.
    # MEMBER_CAN_IMPORTANCES (lane mf-imp6): every fitting kind computes
    # `feature_importances_` at FIT (MEMBER_BLOCK_FEATURE_IMPORTANCES), so RFE
    # reads it (bindings/prep_select_run.mojo `rfe_member_source`). HostGBDT
    # (predict-only) exports it when its file carried it, without the bit.
    # No MEMBER_CAN_SCOPE and no MEMBER_CAN_ROW_INDEX (lane vdf-scope-trees,
    # 2026-11-24, audit): the gbdt images own no device store and no row
    # take, XG_FIT / XG_FIT_ROWMAJOR take a host X (the GPU image quantizes
    # it on the device after its own upload, GBDT_QUANT_DEVICE) and the
    # resident scope's caches are x_prep's (bindings/arena_cache_run.mojo).
    # What is owed: vdf-scope-trees-RESULT.md.
    var mask = (MEMBER_CAN_FIT | MEMBER_CAN_PREDICT | MEMBER_CAN_EXPORT | MEMBER_CAN_IMPORT
                | MEMBER_CAN_SAMPLE_WEIGHT | MEMBER_GIL_FREE | MEMBER_CAN_IMPORTANCES)
    if kind == _KIND_CLASSIFIER:
        return mask | MEMBER_CAN_PREDICT_PROBA | MEMBER_CAN_DECISION | MEMBER_CAN_SCORE | MEMBER_CAN_CLASSIFIER
    if kind == _KIND_REGRESSOR:
        # lane w3f3-c: its own score (`_score_r2`, the adapter's float32 R^2)
        return mask | MEMBER_CAN_SCORE
    if kind == _KIND_FEATURE_FREQ or kind == _KIND_ORDERED:
        return mask
    if kind == _KIND_HOST:
        # Predict-only (a saved model): no fit, no weights; its proba for the two binary losses only.
        var host = MEMBER_CAN_PREDICT | MEMBER_CAN_EXPORT | MEMBER_CAN_IMPORT | MEMBER_GIL_FREE
        if _binary(loss):
            return host | MEMBER_CAN_PREDICT_PROBA
        return host
    if _multi_output(loss) or _binary(loss):
        mask |= MEMBER_CAN_PREDICT_PROBA
    return mask


def _loss_of_words(call: _CallPtr) raises -> String:
    if not call[].params or call[].param_count < 2:
        raise Error("member ABI: a Gbdt member answers without a state from its parameter words")
    var words = _F64P(unsafe_from_address=call[].params)
    var n = Int(words[1])
    if n < 0 or 2 + n > call[].param_count:
        raise Error("member ABI: a Gbdt member's parameter words end early")
    var out = String(capacity_bytes=n)
    for k in range(n):  # small-loop(loss name bytes)
        out += chr(Int(words[2 + k]))
    return out^


def _member[I: GbdtMemberImage](operation: Int, address: Int) raises -> Int:
    var call = _CallPtr(unsafe_from_address=address)
    if operation == MEMBER_OP_CAPABILITIES:
        if call[].state == 0:
            if call[].params and call[].param_count >= 2:
                return _capabilities(_kind_word(_F64P(unsafe_from_address=call[].params)[0]), _loss_of_words(call))
            # The family's union: the booster's widest kind and the classifier.
            return _capabilities(_KIND_BOOSTER, String("MultiClass")) | _capabilities(_KIND_CLASSIFIER, String("Logloss"))
        var asked = _SlotPtr(unsafe_from_address=_lookup(call[].state, False))
        return _capabilities(asked[].kind, asked[].config.loss)
    if operation == MEMBER_OP_NEW:
        if not call[].params:
            raise Error("member ABI: a Gbdt member needs its parameter words")
        call[].state = _mint(_F64P(unsafe_from_address=call[].params), call[].param_count)
        return 0
    if operation == MEMBER_OP_FREE:
        var held = _SlotPtr(unsafe_from_address=_lookup(call[].state, False))
        _clear[I](held)
        _ = _lookup(call[].state, True)
        call[].state = 0
        return 0
    if operation == MEMBER_OP_OUTPUTS:
        # With a fitted state the state answers; without one the loss word:
        # a one-dimensional loss predicts one column, a binary proba two; the
        # width of a multi-class model is chosen by the fit (refused).
        var loss: String
        var kind: Int
        var dim = 0
        var classes = 0
        if call[].state != 0:
            var known = _SlotPtr(unsafe_from_address=_lookup(call[].state, False))
            loss = known[].config.loss
            kind = known[].kind
            if known[].fitted:
                dim = known[].dim
                classes = known[].classes
        else:
            loss = _loss_of_words(call)
            kind = _kind_word(_F64P(unsafe_from_address=call[].params)[0])
        if kind == _KIND_CLASSIFIER:
            # Two classes, one model dimension, before and after a fit.
            if call[].flags == MEMBER_OP_PREDICT:
                call[].out_cols = 1
                call[].out_dtype = MEMBER_I32
            elif call[].flags == MEMBER_OP_PREDICT_PROBA:
                call[].out_cols = 2
                call[].out_dtype = MEMBER_F32
            elif call[].flags == MEMBER_OP_DECISION:
                call[].out_cols = 1
                call[].out_dtype = MEMBER_F32
            else:
                member_unsupported(_name(kind), call[].flags)
            call[].classes = 2
            return 0
        if kind == _KIND_REGRESSOR or kind == _KIND_FEATURE_FREQ or kind == _KIND_ORDERED:
            if call[].flags != MEMBER_OP_PREDICT:
                member_unsupported(_name(kind), call[].flags)
            call[].out_cols = 1
            call[].out_dtype = MEMBER_F32
            call[].classes = 0
            return 0
        if dim == 0:
            if _multi_output(loss):
                raise Error("member ABI: the output width of a " + loss + " Gbdt member is chosen by its fit")
            dim = 1
        if call[].flags == MEMBER_OP_PREDICT_PROBA:
            if kind == _KIND_HOST:
                _host_proba_refusal(loss)
            if _binary(loss):
                call[].out_cols = 2
                call[].out_dtype = MEMBER_F64
            elif _multi_output(loss):
                call[].out_cols = classes
                call[].out_dtype = MEMBER_F32
            else:
                member_unsupported(_ESTIMATOR, MEMBER_OP_PREDICT_PROBA)
        elif call[].flags == MEMBER_OP_PREDICT:
            call[].out_cols = dim
            call[].out_dtype = MEMBER_F32
        else:
            member_unsupported(_ESTIMATOR, call[].flags)
        call[].classes = classes
        return 0
    var slot = _SlotPtr(unsafe_from_address=_lookup(call[].state, False))
    if operation == MEMBER_OP_FIT:
        _fit[I](slot, call)
    elif operation == MEMBER_OP_PREDICT:
        _predict(slot, call)
    elif operation == MEMBER_OP_PREDICT_PROBA:
        _proba(slot, call)
    elif operation == MEMBER_OP_EXPORT:
        _export(slot, call)
    elif operation == MEMBER_OP_IMPORT:
        _import[I](slot, call)
    elif operation == MEMBER_OP_DECISION:
        _decision(slot, call)
    elif operation == MEMBER_OP_SCORE:
        _score(slot, call)
    else:
        member_unsupported(_name(slot[].kind), operation)
    return 0


def gbdt_member_v2[I: GbdtMemberImage](operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
    result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """The member entry of GradientBoosting and its two adapters (schema
    `MEMBER_ABI`)."""
    try:
        var word = _member[I](operation, member_call_from_spans(operation, aa, na, np, nf))
        if result:
            MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = word
        return 0
    except error:
        native_descriptor_report(error, context, sink)
        return 1


def gbdt_member_publish_image[I: GbdtMemberImage](vendor: StaticString) raises:
    """The one publication line of a gbdt image's module initialization,
    after its gbdt entry: the image's tier and the vendor word of the
    string its entry is published under (-1 publishes nothing)."""
    var word = gbdt_vendor_word(vendor)
    if word < 0:
        return
    var store = SharedSlot[GbdtMemberStore, _STORE_NAME, 2].get_or_create_ptr()
    member_publish[MEMBER_FAMILY_GBDT](Int(GLOBAL_NUMERIC_MODE), word, Int(store), gbdt_member_v2[I])
