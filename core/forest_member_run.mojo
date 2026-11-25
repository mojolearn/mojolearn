# SPDX-License-Identifier: Apache-2.0
"""The forest member RUN shared by the Tree (13), Forest (14) and ExtraTrees
(15) families (core/member_abi.mojo). Each family file
(core/tree_member.mojo, core/forest_member.mojo, core/extra_trees_member.mojo)
names its own store and calls `fm_member[store, rf_layout, notices]` from its entry.

It runs the families' EXISTING typed forest RUN over the entry the caller's
route holds in `route.entry` (the forest entry of the image the estimator's
own doors call, core/forest_calls.mojo): the request
(bindings/forest_config_run.mojo `forest_request`), the fit core
(xtrees/trees_forest_run.mojo `forest_fit_core`, XF_FIT), the export into
the five model arrays (XF_EXPORT_*), and the predict route the estimator's
own predict takes (bindings/forest_predict_run.mojo `ForestState`: the
sequential walk, the host strict snapshot or the resident snapshot; the
fused resident labels where the image serves them). No kernel is touched.

MEMBER_OP_NEW parameters (Float64 words):
  [0] kind: 0 classifier, 1 regressor; 2 / 3 (lane mf-hostent5) the
      PREDICT-ONLY classifier / regressor of a saved model (`HostForest`): no
      fit, no score (its class has neither), its state IMPORTED from the saved
      arrays (the blocks below) and predicted by the same predict route as an
      imported RandomForest / ExtraTrees state on the host image (word [9]
      names its saved engine); the fit settings are unread (the admission
      writes three empty slots)
  [1] layout: 0 the RF slots (RandomForest, DecisionTree splitter='best'),
      1 the ET slots (ExtraTrees)
  [2] criterion (the XF_W_CRITERION word; -1 for the ET layout)
  [3] max_features form (FOREST_FEATURES_*), [4] its count, [5] its fraction
  [6] 1: a DecisionTreeRegressor left at n_bins=None (the image's bin count)
  [7] ET max_samples as a fraction of the fit's rows (0.0: the slot holds it)
  [8] 1: the fast tier; [9] 1: the parallel_groves predict engine
  [10] 1: every fit also computes the feature importances (the estimator's
      `fit(..., feature_importances=True)`), exported as
      `MEMBER_BLOCK_FEATURE_IMPORTANCES`. The admission writes 0; a caller
      that reads importances (recursive elimination, selection from a
      model) sets it to 1 in `plan.params` before `member_new`. RF layout
      only: the ExtraTrees fit computes none (a 1 there is refused).
  [11] 1: every fit also computes the out-of-bag outputs (the estimator's
      `oob_score=True`: `oob_decision_function_` or `oob_prediction_`, and
      `oob_score_`), in the same fit body the estimator's own fit runs
      (the ForestFit oob and score words). The admission writes it from the
      estimator (lane mf-oob3, 2026-11-23). RF layout only (a 1 on the ET
      layout is refused: the ExtraTrees fit has no out-of-bag mask).
  [12] S, the parameter slot count; then S triples
      [13 + 3k] the int reading of slot k as the Float64 bit pattern of its
      Int64 (std.memory.bitcast: seeds and counts travel exactly),
      [14 + 3k] its float reading, [15 + 3k] 1 when it has no reading.
  `FM_WORD_IMPORTANCES`, `FM_WORD_OOB`, `FM_WORD_SLOTS` and `FM_HEAD` name
  these positions: every caller that writes or rewrites a word uses them.
  (Lane mf-oob3: the out-of-bag word [11] was inserted ONCE for every
  caller; before it the slot count was word 11 and the triples began at 12.)
Views: X dense row-major host float32 (finite: the caller's admission, as
the estimator's own door; an image with the device scan tests it again);
y host int32 class codes 0..classes-1 (classifier, `call.classes` >= 2) or a
float32 target column (regressor); `call.weights` (optional) host float32,
one per row (finite and non-negative: the caller's admission, as the
estimator's `admit_weights`), multiplied by nothing (no class_weight: the
admission keeps the estimator's own doors for it). Refused in words: a
column index, a device-home view, a nonzero `fit_output`, a multi-column
y, a weighted score, and on a regressor of the RF layout weights together
with importances or the out-of-bag outputs (the estimator's own refusal). predict writes int32 codes
(classifier) or float32 values; predict_proba float32 rows x classes; score
(classifier only) the accuracy count / rows, unweighted.

THE FOLD ROWS (lane vdf-scope-trees, 2026-11-24; ExtraTrees and the
predictions lane vdf-scope-trees2a): `call.row_index` (host int32 or int64,
MEMBER_CAN_ROW_INDEX on a route whose image answers XF_CAP_DEV_ROWS, the
RandomForest and ExtraTrees GPU images) selects the rows of the WHOLE X, y
and weights. FIT: under a resident scope with a pin (MEMBER_CAN_SCOPE) X goes
once into the image's device store through the scope's cache of this family
(`arena_scope_store_put`, prefix "forest_rf" / "forest_et"; `pin_kept` as THE
PIN RULE says) and the image takes the rows on the device (XF_FIT_ROWS: the
same fit bodies on the plane a row-major upload of those rows makes, so the
same bits); without one the rows are gathered on the host. y and the weights
are gathered on the host (one word a row), in row tasks. PREDICT / SCORE:
under the same scope, on the resident route, the same store slot (a hit in
the scope's cache) and the image's resident snapshot reads the rows taken on
the device (XF_RESIDENT_ROWS: the image's store, fits and snapshots share
its one context); otherwise the rows are gathered on the host. A
classifier's score there reads the fold's y on the device against those
codes (XF_RESIDENT_ROWS mode 2, lane vdf-scope-trees3): one count comes back.
y and the weights of a fit are not kept in the store: they belong to no
pinned object (THE PIN RULE), so a scope cache entry keyed by their address
could be hit by a later buffer at the same address (the permutation test's
shuffled y under its enclosing scope); they are gathered per fold (one word a
row) and uploaded by the fit as before.

Fits, as the estimator's own fit door picks them: unweighted without
importances reads X row-major (the view itself, no copy); the ExtraTrees
weighted fit also reads it row-major; the RF layout's weighted fit and its
importance fit read X column-major, so the member writes the column-major
float32 copy the estimator's door writes for them (`order="F"`; the out-of-bag fit
too, as the estimator's oob fit reads `_x_for(X, False)`), split over
the host workers, alive until the launch returns. The RF regressor's
weighted fit also carries the largest weight (a parallel max).

Fitted-state blocks (MEMBER_OP_EXPORT / MEMBER_OP_IMPORT), in index order:
  0 `MEMBER_BLOCK_FAMILY + 0` offsets int32 1 x (trees + 1)
  1 `MEMBER_BLOCK_FAMILY + 1` colid int32 1 x nodes
  2 `MEMBER_BLOCK_FAMILY + 2` quesval float32 1 x nodes
  3 `MEMBER_BLOCK_FAMILY + 3` left_child int32 1 x nodes
  4 `MEMBER_BLOCK_FAMILY + 4` leaves float32 nodes x outputs
  5 `MEMBER_BLOCK_SCALARS` float64 [features, trees, nodes, outputs,
    classes, n_meta, meta words...] (meta: the export's metadata words)
  6 `MEMBER_BLOCK_FEATURE_IMPORTANCES` float32 1 x features, only when the
    fit computed them (or they were imported): the block count is then 7
An import needs the first six (importances optional, before the last); the
last one carries MEMBER_IMPORT_DONE.
OUT-OF-BAG OUTPUTS (word 11), by name only (never in the index walk, empty
after an import, as the estimator's arrays-only state has none):
  `FM_BLOCK_OOB` (`MEMBER_BLOCK_FAMILY + 5`) float64 rows x classes (a
    classifier's `oob_decision_function_`) or rows x 1 (a regressor's
    `oob_prediction_`), rows = the fit's rows
  `FM_BLOCK_OOB_SCORE` (`MEMBER_BLOCK_FAMILY + 6`) float64 1 x 1, `oob_score_`
A state fitted without word 11 refuses both in words.
FIT NOTICES (Forest: MEMBER_CAN_NOTICES; Tree has none since 2026-11-24): `MEMBER_BLOCK_NOTICES` by name,
one `MEMBER_NOTICE_FOREST` with the fit's notice mask (core/member_notices.mojo);
never in the index walk, empty after an import.

The state (the five arrays, the snapshot handle) is OWNED by its slot in a
process-wide store of fixed chunks: a slot never moves, the lock covers
mint, free and the handle lookup only. Members are independent: a caller
runs them on its own workers, one handle each. The snapshot is prepared once,
at the first predict / predict_proba / score after a fit or import, as the
estimator's own predict door does (a state only fitted and published or
freed uploads nothing); concurrent first predicts of one state build it
once under the slot's own lock word (`_ensure_prepared`). It is released at
the next fit, import or free.
Parallel: the ET classifier's code cast (host images only: an image with
XF_CAP_ET_CODES takes the int32 codes and makes the class ids on the device),
the column-major copy, the largest
weight, the argmax of the votes and the accuracy count split rows over the
host workers. Serial by nature: the
store's bookkeeping.
"""
from std.sys.info import size_of
from std.memory import unsafe_memcpy, bitcast
from core.entry_table import EntryRef, entry_ref_unresolved, entry_ref_gil_free
from core.forest_calls import (
    xf_caps, xf_export_sizes, xf_export_into, xf_export_release, xf_predict, xf_resident_prepare,
    xf_resident_predict, xf_resident_labels, xf_strict_prepare, xf_strict_predict, xf_snapshot_release,
)
from core.forest_fit_request import (
    XF_CAP_DEVICE_SCAN, XF_CAP_DEV_ROWS, XF_CAP_ET_CODES, XF_RESIDENT_RELEASE, XF_STRICT_RELEASE,
)
from core.forest_calls import xf_fit_rows, xf_resident_rows
from core.forest_entry_table import FOREST_TABLE_RF, FOREST_TABLE_ET
from core.forest_store_transport import forest_store_transport_ref, FOREST_STORE_PREFIX_RF, FOREST_STORE_PREFIX_ET
from bindings.arena_cache_run import arena_scope_store_put
from core.fold_x_word import fold_x_word
from core.host_parallel import host_widen_index_i64
from bindings.forest_config_run import ForestConfig, forest_request
from bindings.forest_predict_run import (
    ForestState, forest_labels_route, FOREST_ROUTE_SEQUENTIAL, FOREST_ROUTE_STRICT,
)
from xtrees.trees_forest_run import ForestFit, forest_fit_core, forest_export_counts
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk
from core.member_abi import (
    MEMBER_GIL_FREE,
    MemberCall, MemberMatrix, MEMBER_F32, MEMBER_F64, MEMBER_I32, MEMBER_I64, MEMBER_HOME_HOST,
    MEMBER_CAN_ROW_INDEX, MEMBER_CAN_SCOPE, MEMBER_SCOPE_NONE,
    MEMBER_OP_CAPABILITIES, MEMBER_OP_NEW, MEMBER_OP_FREE, MEMBER_OP_FIT, MEMBER_OP_PREDICT,
    MEMBER_OP_PREDICT_PROBA, MEMBER_OP_SCORE, MEMBER_OP_OUTPUTS, MEMBER_OP_EXPORT, MEMBER_OP_IMPORT,
    MEMBER_CAN_FIT, MEMBER_CAN_PREDICT, MEMBER_CAN_PREDICT_PROBA, MEMBER_CAN_SCORE, MEMBER_CAN_EXPORT,
    MEMBER_CAN_IMPORT, MEMBER_CAN_CLASSIFIER, MEMBER_CAN_SAMPLE_WEIGHT, MEMBER_CAN_IMPORTANCES, MEMBER_CAN_NOTICES,
    MEMBER_BLOCK_SCALARS, MEMBER_BLOCK_FEATURE_IMPORTANCES, MEMBER_BLOCK_FAMILY, MEMBER_BLOCK_BY_INDEX, MEMBER_IMPORT_DONE,
    MEMBER_BLOCK_NOTICES, member_unsupported,
)
from core.member_notices import member_notice_add, MEMBER_NOTICE_FOREST, MEMBER_NOTICE_WORDS
from core.shared_slot import SharedSlot, shared_slot_lock, shared_slot_unlock
from std.atomic import Atomic, Ordering

comptime _CHUNK = 256
comptime _INDEX_SPAN = 16777216
#: MEMBER_OP_NEW word positions (module docstring), shared by every caller
comptime FM_WORD_IMPORTANCES = 10
comptime FM_WORD_OOB = 11
comptime FM_WORD_SLOTS = 12
comptime FM_HEAD = 13
#: the out-of-bag output blocks, exported by name only
comptime FM_BLOCK_OOB = MEMBER_BLOCK_FAMILY + 5
comptime FM_BLOCK_OOB_SCORE = MEMBER_BLOCK_FAMILY + 6
comptime _HEAD = FM_HEAD
comptime _BLOCKS = 6
comptime _SLOTS_WORD = FM_WORD_SLOTS

comptime _I32P = MutPointer[Int32, MutUntrackedOrigin]
comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _F64P = MutPointer[Float64, MutUntrackedOrigin]


struct ForestMemberSlot(Copyable, Movable, Defaultable):
    var generation: Int
    var live: Bool
    var classifier: Bool
    var et: Bool
    var criterion: Int
    var form: Int
    var count: Int
    var fraction: Float64
    var bins_image: Bool
    var rows_fraction: Float64
    var fast: Bool
    var groves: Bool
    #: kinds 2 / 3 (lane mf-hostent5): a saved model, predict-only
    var predict_only: Bool
    var importances: Bool
    #: word 11: every fit computes the out-of-bag outputs
    var oob: Bool
    var ints: List[Int]
    var floats: List[Float64]
    var bad: List[UInt8]
    var fitted: Bool
    var features: Int
    var trees: Int
    var nodes: Int
    var outputs: Int
    var classes: Int
    var offsets: List[Int32]
    var colid: List[Int32]
    var quesval: List[Float32]
    var left: List[Int32]
    var leaves: List[Float32]
    var scalars: List[Float64]
    #: the fitted (or imported) feature importances; empty: none
    var imp: List[Float32]
    #: the last fit's out-of-bag outputs (word 11): `oob_rows` x `oob_cols`
    #: float64 words and the one score word; empty: none
    var oob_out: List[Float64]
    var oob_score: List[Float64]
    var oob_rows: Int
    var oob_cols: Int
    #: FIT NOTICES (core/member_notices.mojo): the RF layout's notice mask of
    #: the last fit (the third metadata word); empty after an import.
    var notices: List[Float64]
    #: the image the snapshot lives in, and the predict route words
    var key: EntryRef
    var route: Int
    var labels: Bool
    var snapshot: Int
    var release_op: Int
    #: 1 once the predict route words and snapshot above are built for the
    #: current fit or import (lazily, at the first predict / score; lane
    #: fx-audit, audit-trees F2), read with acquire; `prepare_lock` is the
    #: slot's own lock word for that one build
    var prepared: Int64
    var prepare_lock: Int64
    #: the XF_CAPS word of the member's route image, resolved ONCE at the
    #: state's admission (MEMBER_OP_NEW; lane vdf-scope-trees3); -1 when that
    #: entry call failed (then each use asks the image, as before)
    var caps: Int

    def __init__(out self):
        self.generation = 0
        self.live = False
        self.classifier = False
        self.et = False
        self.criterion = -1
        self.form = 0
        self.count = 0
        self.fraction = 0.0
        self.bins_image = False
        self.rows_fraction = 0.0
        self.fast = False
        self.groves = False
        self.predict_only = False
        self.importances = False
        self.oob = False
        self.ints = List[Int]()
        self.floats = List[Float64]()
        self.bad = List[UInt8]()
        self.fitted = False
        self.features = 0
        self.trees = 0
        self.nodes = 0
        self.outputs = 0
        self.classes = 0
        self.offsets = List[Int32]()
        self.colid = List[Int32]()
        self.quesval = List[Float32]()
        self.left = List[Int32]()
        self.leaves = List[Float32]()
        self.scalars = List[Float64]()
        self.imp = List[Float32]()
        self.oob_out = List[Float64]()
        self.oob_score = List[Float64]()
        self.oob_rows = 0
        self.oob_cols = 0
        self.notices = List[Float64]()
        self.key = entry_ref_unresolved()
        self.route = FOREST_ROUTE_SEQUENTIAL
        self.labels = False
        self.snapshot = 0
        self.release_op = 0
        self.prepared = 0
        self.prepare_lock = 0
        self.caps = -1


struct ForestMemberStore(Defaultable, Movable):
    var synchronization: List[Int64]
    var chunks: List[List[ForestMemberSlot]]
    var free: List[Int]
    var count: Int

    def __init__(out self):
        # Side-effect-free even when the process slot's initializers race.
        self.synchronization = List[Int64](length=2, fill=0)
        self.chunks = List[List[ForestMemberSlot]]()
        self.free = List[Int]()
        self.count = 0


comptime _SlotPtr = MutPointer[ForestMemberSlot, MutUntrackedOrigin]


def _slot_address[name: StaticString](index: Int) raises -> Int:
    # Under the store's lock. A chunk's buffer is allocated once and never
    # grows, so the address stays valid after the lock is released.
    var store = SharedSlot[ForestMemberStore, name, 2].get_or_create_ptr()
    return Int(store[].chunks[index // _CHUNK].unsafe_ptr()) + (index % _CHUNK) * size_of[ForestMemberSlot]()


def fm_store_address[name: StaticString]() raises -> Int:
    """A stable nonzero address of the store (the publisher word)."""
    return Int(SharedSlot[ForestMemberStore, name, 2].get_or_create_ptr())


def _mint[name: StaticString](words: _F64P, count: Int) raises -> Int:
    var slots = Int(words[_SLOTS_WORD])
    if count != _HEAD + 3 * slots or slots < 3:
        raise Error("member ABI: a forest member takes " + String(_HEAD) + " words and three per parameter slot")
    if words[0] != 0.0 and words[0] != 1.0 and words[0] != 2.0 and words[0] != 3.0:
        raise Error("member ABI: unknown forest member kind")
    if words[FM_WORD_IMPORTANCES] != 0.0 and words[1] != 0.0:
        raise Error("NotImplementedError: mojolearn: the ExtraTrees fit computes no feature importances")
    if words[FM_WORD_OOB] != 0.0 and words[FM_WORD_OOB] != 1.0:
        raise Error("member ABI: the forest member's out-of-bag word is 0 or 1")
    if words[FM_WORD_OOB] != 0.0 and words[1] != 0.0:
        raise Error("NotImplementedError: mojolearn: the ExtraTrees fit computes no out-of-bag outputs")
    var store = SharedSlot[ForestMemberStore, name, 2].get_or_create_ptr()
    var lock = rebind[MutPointer[Int64, MutUntrackedOrigin]](store[].synchronization.unsafe_ptr())
    shared_slot_lock(lock)
    try:
        var index = -1
        if len(store[].free) > 0:
            index = store[].free.pop()
        else:
            if store[].count >= _INDEX_SPAN - 1:
                raise Error("RuntimeError: mojolearn: too many live forest member states")
            if store[].count == len(store[].chunks) * _CHUNK:
                store[].chunks.append(List[ForestMemberSlot](length=_CHUNK, fill=ForestMemberSlot()))
            index = store[].count
            store[].count += 1
        var slot = _SlotPtr(unsafe_from_address=_slot_address[name](index))
        slot[].generation += 1
        slot[].live = True
        slot[].classifier = words[0] == 0.0 or words[0] == 2.0
        slot[].predict_only = words[0] >= 2.0
        slot[].et = words[1] != 0.0
        slot[].criterion = Int(words[2])
        slot[].form = Int(words[3])
        slot[].count = Int(words[4])
        slot[].fraction = words[5]
        slot[].bins_image = words[6] != 0.0
        slot[].rows_fraction = words[7]
        slot[].fast = words[8] != 0.0
        slot[].groves = words[9] != 0.0
        slot[].importances = words[FM_WORD_IMPORTANCES] != 0.0
        slot[].oob = words[FM_WORD_OOB] != 0.0
        slot[].ints = List[Int](length=slots, fill=0)
        slot[].floats = List[Float64](length=slots, fill=0.0)
        slot[].bad = List[UInt8](length=slots, fill=0)
        for k in range(slots):  # small-loop(slots): about twenty parameter slots, once per state
            slot[].ints[k] = Int(bitcast[DType.int64](words[_HEAD + 3 * k]))
            slot[].floats[k] = words[_HEAD + 3 * k + 1]
            slot[].bad[k] = UInt8(1) if words[_HEAD + 3 * k + 2] != 0.0 else UInt8(0)
        slot[].fitted = False
        slot[].imp = List[Float32]()
        _clear_oob(slot)
        slot[].snapshot = 0
        slot[].prepared = 0
        slot[].caps = -1
        return slot[].generation * _INDEX_SPAN + index + 1
    finally:
        shared_slot_unlock(lock)


def _lookup[name: StaticString](handle: Int, release: Bool) raises -> Int:
    """The slot address of a live handle. A released handle, a handle of
    another store or a made-up integer is refused: never a stale read."""
    var store = SharedSlot[ForestMemberStore, name, 2].get_or_create_ptr()
    var lock = rebind[MutPointer[Int64, MutUntrackedOrigin]](store[].synchronization.unsafe_ptr())
    shared_slot_lock(lock)
    try:
        var index = handle % _INDEX_SPAN - 1
        if handle <= 0 or index < 0 or index >= store[].count:
            raise Error("NotFittedError: this member state was released or never created")
        var address = _slot_address[name](index)
        var slot = _SlotPtr(unsafe_from_address=address)
        if not slot[].live or slot[].generation != handle // _INDEX_SPAN:
            raise Error("NotFittedError: this member state was released or never created")
        if release:
            slot[].live = False
            slot[].fitted = False
            slot[].ints = List[Int]()
            slot[].floats = List[Float64]()
            slot[].bad = List[UInt8]()
            slot[].offsets = List[Int32]()
            slot[].colid = List[Int32]()
            slot[].quesval = List[Float32]()
            slot[].left = List[Int32]()
            slot[].leaves = List[Float32]()
            slot[].scalars = List[Float64]()
            slot[].imp = List[Float32]()
            _clear_oob(slot)
            slot[].notices = List[Float64]()
            store[].free.append(index)
        return address
    finally:
        shared_slot_unlock(lock)


def _clear_oob(slot: _SlotPtr):
    slot[].oob_out = List[Float64]()
    slot[].oob_score = List[Float64]()
    slot[].oob_rows = 0
    slot[].oob_cols = 0


def _release_snapshot(slot: _SlotPtr):
    """The slot's snapshot back to its image (non-raising: free and refit)."""
    if slot[].snapshot == 0:
        return
    var handle = slot[].snapshot
    slot[].snapshot = 0
    try:
        xf_snapshot_release(slot[].key, slot[].release_op, handle)
    except:
        pass


def _dense(view: MemberMatrix, dtype: Int, what: String, estimator: String) raises:
    if not view.dense_host(dtype):
        var words = String("float32") if dtype == MEMBER_F32 else String("int32")
        raise Error("TypeError: a " + estimator + " member reads " + what + " as a dense host " + words + " view")


def _cast_codes(src: Int, dst: Int, rows: Int):
    """int32 class codes to the float32 codes the ET classifier reads
    (`codes.astype('<f4')`), split over the host workers."""
    var tasks = host_predict_task_count_work(rows, rows)
    var chunk = host_predict_chunk(rows, tasks)
    var source = _I32P(unsafe_from_address=src)
    var target = _F32P(unsafe_from_address=dst)

    def task(unit: Int) {imm source, imm target, imm chunk, imm rows}:
        for i in range(unit * chunk, min((unit + 1) * chunk, rows)):  # native data loop (one host task's rows)
            target[i] = Float32(source[i])

    host_parallelize(task, tasks)


def _column_major(src: Int, dst: Int, rows: Int, cols: Int):
    """The float32 rows x cols matrix at `src` (row-major) as its
    column-major copy at `dst` (the estimator's `order="F"` copy), split
    over the host workers by rows."""
    var tasks = host_predict_task_count_work(rows, rows * cols)
    var chunk = host_predict_chunk(rows, tasks)
    var source = _F32P(unsafe_from_address=src)
    var target = _F32P(unsafe_from_address=dst)

    def task(unit: Int) {imm source, imm target, imm chunk, imm rows, imm cols}:
        for i in range(unit * chunk, min((unit + 1) * chunk, rows)):  # native data loop (one host task's rows)
            for c in range(cols):  # native data loop (one row's columns)
                target[c * rows + i] = source[i * cols + c]

    host_parallelize(task, tasks)


def _largest(src: Int, rows: Int) -> Float64:
    """The largest of `rows` float32 weights (the estimator's
    `reduce_stat` max): per-task maxima, then their max (any order gives
    the same value). `rows` >= 1."""
    var tasks = host_predict_task_count_work(rows, rows)
    var chunk = host_predict_chunk(rows, tasks)
    var source = _F32P(unsafe_from_address=src)
    var tops = List[Float32](length=max(tasks, 1), fill=source[0])
    var top_words = _F32P(unsafe_from_address=Int(tops.unsafe_ptr()))

    def task(unit: Int) {imm source, imm top_words, imm chunk, imm rows}:
        var first = unit * chunk
        var last = min(first + chunk, rows)
        if first >= last:
            return
        var top = source[first]
        for i in range(first + 1, last):  # native data loop (one host task's rows)
            if source[i] > top:
                top = source[i]
        top_words[unit] = top

    host_parallelize(task, tasks)
    var best = tops[0]
    for t in range(1, len(tops)):  # small-loop(tasks): one maximum per host task
        if tops[t] > best:
            best = tops[t]
    return Float64(best)


def _argmax_rows(src: Int, dst: Int, rows: Int, width: Int):
    """The first-max column of each row of the float32 votes at `src` as
    int32 codes at `dst` (the estimator's `trees_argmax_rows`), split over
    the host workers."""
    var tasks = host_predict_task_count_work(rows, rows * width)
    var chunk = host_predict_chunk(rows, tasks)
    var votes = _F32P(unsafe_from_address=src)
    var codes = _I32P(unsafe_from_address=dst)

    def task(unit: Int) {imm votes, imm codes, imm chunk, imm rows, imm width}:
        for i in range(unit * chunk, min((unit + 1) * chunk, rows)):  # native data loop (one host task's rows)
            var best = 0
            for c in range(1, width):  # native data loop (one row's classes)
                if votes[i * width + c] > votes[i * width + best]:
                    best = c
            codes[i] = Int32(best)

    host_parallelize(task, tasks)


def _count_equal(left: Int, right: Int, rows: Int) -> Int:
    """The rows where two int32 code vectors agree: per-task counts, summed
    in task order (an integer count: any order gives the same value)."""
    var tasks = host_predict_task_count_work(rows, rows)
    var chunk = host_predict_chunk(rows, tasks)
    var counts = List[Int](length=max(tasks, 1), fill=0)
    var a = _I32P(unsafe_from_address=left)
    var b = _I32P(unsafe_from_address=right)
    var sums = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=Int(counts.unsafe_ptr()))

    def task(unit: Int) {imm a, imm b, imm sums, imm chunk, imm rows}:
        var hits = 0
        for i in range(unit * chunk, min((unit + 1) * chunk, rows)):  # native data loop (one host task's rows)
            if a[i] == b[i]:
                hits += 1
        sums[unit] = hits

    host_parallelize(task, tasks)
    var total = 0
    for t in range(len(counts)):  # small-loop(tasks): one count per host task
        total += counts[t]
    return total


def _prepare(slot: _SlotPtr, key: EntryRef) raises:
    """The predict route of the fitted state on the image `key` (the
    estimator's own `ForestState` rule) and, off the sequential route, its
    snapshot, owned by the slot."""
    var caps = _slot_caps(slot, key)
    var state = ForestState(key, 0, caps, slot[].fast, slot[].groves, slot[].features, slot[].trees, slot[].outputs)
    slot[].key = key
    slot[].route = state.route
    slot[].labels = slot[].classifier and forest_labels_route(state)
    if state.route == FOREST_ROUTE_SEQUENTIAL:
        return
    var model: List[Int] = [Int(slot[].offsets.unsafe_ptr()), Int(slot[].colid.unsafe_ptr()),
        Int(slot[].quesval.unsafe_ptr()), Int(slot[].left.unsafe_ptr()), Int(slot[].leaves.unsafe_ptr())]
    if state.route == FOREST_ROUTE_STRICT:
        slot[].snapshot = xf_strict_prepare(key, model, slot[].features, slot[].trees, slot[].outputs)
        slot[].release_op = XF_STRICT_RELEASE
    else:
        slot[].snapshot = xf_resident_prepare(key, model, slot[].features, slot[].trees, slot[].outputs,
            state.ordered)
        slot[].release_op = XF_RESIDENT_RELEASE


def _unprepared(slot: _SlotPtr, key: EntryRef):
    # A new fit or import: the route words wait for the first predict on
    # `key` (`_ensure_prepared`). Called after `_release_snapshot`, with the
    # state owned by the fitting caller (no predict runs concurrently).
    slot[].key = key
    slot[].route = FOREST_ROUTE_SEQUENTIAL
    slot[].labels = False
    slot[].prepared = 0


def _ensure_prepared(slot: _SlotPtr) raises:
    """The predict route and snapshot of the fitted state, built at its
    first predict / predict_proba / score (as the estimator's own predict
    door builds it, bindings/forest_predict_run.mojo), never at fit: a state
    only fitted and published or freed (bagging on a device route, RFE steps,
    freed stacking fits) uploads nothing. After the first call: one acquire
    load. Concurrent first predicts on one state build it once (the slot's
    own lock word)."""
    var flag = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(UnsafePointer(to=slot[].prepared)))
    if Atomic.load[ordering=Ordering.ACQUIRE](flag) != 0:
        return
    var lock = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(UnsafePointer(to=slot[].prepare_lock)))
    shared_slot_lock(lock)
    try:
        if Atomic.load[ordering=Ordering.ACQUIRE](flag) == 0:
            var key = slot[].key  # a copy: `_prepare` writes the slot's own word
            _prepare(slot, key)
            Atomic.store[ordering=Ordering.RELEASE](flag, Int64(1))
    finally:
        shared_slot_unlock(lock)


def _scalars(slot: _SlotPtr, meta: List[Int]):
    slot[].scalars = List[Float64](length=6 + len(meta), fill=0.0)
    slot[].scalars[0] = Float64(slot[].features)
    slot[].scalars[1] = Float64(slot[].trees)
    slot[].scalars[2] = Float64(slot[].nodes)
    slot[].scalars[3] = Float64(slot[].outputs)
    slot[].scalars[4] = Float64(slot[].classes)
    slot[].scalars[5] = Float64(len(meta))
    for k in range(len(meta)):  # small-loop(meta): at most XF_META_MAX export metadata words
        slot[].scalars[6 + k] = Float64(meta[k])


struct _FoldRows(Movable):
    """The fit's rows under `call.row_index` (lane vdf-scope-trees): `n`
    rows, `index` the address of their int64 numbers (the call's own, or
    `wide`, the int32 index widened), and `store_id` the slot of the whole X
    in the image's device store (-1: the rows are gathered on the host)."""
    var n: Int
    var index: Int
    var store_id: Int
    var wide: List[Int64]

    def __init__(out self):
        self.n = 0
        self.index = 0
        self.store_id = -1
        self.wide = List[Int64]()


def _dev_rows_route(call: MutPointer[MemberCall, MutUntrackedOrigin]) -> Bool:
    """The route's image keeps a device store, fits its rows and predicts
    them (XF_CAP_DEV_ROWS: the RandomForest and ExtraTrees GPU images). One
    entry call: asked by MEMBER_OP_CAPABILITIES without a state (once per
    member class) and at a state's admission (`_admit_caps`); a fit or a
    prediction reads the slot's word (`_slot_dev_rows`)."""
    try:
        return (xf_caps(call[].route.entry) & XF_CAP_DEV_ROWS) != 0
    except:
        return False


def _admit_caps(slot: _SlotPtr, call: MutPointer[MemberCall, MutUntrackedOrigin]):
    """The route image's XF_CAPS word, resolved ONCE per member state at its
    admission (MEMBER_OP_NEW; lane vdf-scope-trees3). Before, every predict,
    predict_proba and score under a row index asked the image again
    (`_scope_rows_route`), and every fit once more for the device scan."""
    try:
        slot[].caps = xf_caps(call[].route.entry)
    except:
        slot[].caps = -1


def _slot_caps(slot: _SlotPtr, key: EntryRef) raises -> Int:
    """The slot's admitted XF_CAPS word; an image that did not answer at
    admission is asked now (its refusal raises here, as before)."""
    if slot[].caps >= 0:
        return slot[].caps
    return xf_caps(key)


def _slot_dev_rows(slot: _SlotPtr, call: MutPointer[MemberCall, MutUntrackedOrigin]) -> Bool:
    """`_dev_rows_route` from the slot's admitted word (no entry call)."""
    if slot[].caps >= 0:
        return (slot[].caps & XF_CAP_DEV_ROWS) != 0
    return _dev_rows_route(call)


def _index_bad(index: Int, n: Int, base_rows: Int) -> Int:
    """The int64 row numbers outside 0 .. base_rows - 1, counted in row
    tasks across the host's cores (any order gives the same count)."""
    var tasks = host_predict_task_count_work(n, n)
    var chunk = host_predict_chunk(n, tasks)
    var counts = List[Int](length=max(tasks, 1), fill=0)
    var ids = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=index)
    var sums = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=Int(counts.unsafe_ptr()))

    def task(unit: Int) {imm ids, imm sums, imm chunk, imm n, imm base_rows}:
        var bad = 0
        for i in range(unit * chunk, min((unit + 1) * chunk, n)):  # native data loop (one host task's rows)
            var r = Int(ids[i])
            if r < 0 or r >= base_rows:
                bad += 1
        sums[unit] = bad
    host_parallelize(task, tasks)
    var total = 0
    for t in range(len(counts)):  # small-loop(tasks): one count per host task
        total += counts[t]
    return total


def _take_rows_f32(src: Int, index: Int, n: Int, width: Int, dst: Int):
    """Rows `index` (int64, `n` of them) of the float32 rows x `width`
    matrix at `src` into `dst`, in row tasks across the host's cores (a byte
    copy: the bytes the caller's own gather writes)."""
    var tasks = host_predict_task_count_work(n, n * width)
    var chunk = host_predict_chunk(n, tasks)
    var source = _F32P(unsafe_from_address=src)
    var target = _F32P(unsafe_from_address=dst)
    var ids = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=index)

    def task(unit: Int) {imm source, imm target, imm ids, imm chunk, imm n, imm width}:
        for i in range(unit * chunk, min((unit + 1) * chunk, n)):  # native data loop (one host task's rows)
            var r = Int(ids[i])
            for c in range(width):  # native data loop (one row's columns)
                target[i * width + c] = source[r * width + c]
    host_parallelize(task, tasks)


def _take_rows_i32(src: Int, index: Int, n: Int, dst: Int):
    """Rows `index` of the int32 column at `src` into `dst` (row tasks)."""
    var tasks = host_predict_task_count_work(n, n)
    var chunk = host_predict_chunk(n, tasks)
    var source = _I32P(unsafe_from_address=src)
    var target = _I32P(unsafe_from_address=dst)
    var ids = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=index)

    def task(unit: Int) {imm source, imm target, imm ids, imm chunk, imm n}:
        for i in range(unit * chunk, min((unit + 1) * chunk, n)):  # native data loop (one host task's rows)
            target[i] = source[Int(ids[i])]
    host_parallelize(task, tasks)


def _row_numbers(call: MutPointer[MemberCall, MutUntrackedOrigin], estimator: String,
                 mut wide: List[Int64]) raises -> Int:
    """The address of the call's row index as int64 numbers (an int32 index
    widened into `wide` across the host's cores), checked in range of X."""
    var given = call[].row_index
    if given.cols != 1 or given.home != MEMBER_HOME_HOST:
        raise Error("TypeError: a " + estimator + " member reads its row index as one host int32 or int64 column")
    var index = 0
    if given.dense_host(MEMBER_I64):
        index = given.data
    elif given.dense_host(MEMBER_I32):
        wide = List[Int64](unsafe_uninit_length=given.rows)
        host_widen_index_i64(given.data, Int(wide.unsafe_ptr()), given.rows)
        index = Int(wide.unsafe_ptr())
    else:
        raise Error("TypeError: a " + estimator + " member reads its row index as one host int32 or int64 column")
    if _index_bad(index, given.rows, call[].x.rows) != 0:
        raise Error("IndexError: a row index of the " + estimator + " member is outside X")
    return index


def _fold_rows(slot: _SlotPtr, call: MutPointer[MemberCall, MutUntrackedOrigin], estimator: String) raises -> _FoldRows:
    """THE FOLD ROWS of a fit (`_fit`): the row numbers and, under a
    resident scope with a pin on a route whose image keeps a device store,
    the store slot of the WHOLE X, put once per scope through the scope's
    cache of this family (bindings/arena_cache_run.mojo
    `arena_scope_store_put`; THE PIN RULE: `pin_kept` is written the moment
    the cache took the reference, before anything after it can raise)."""
    var fold = _FoldRows()
    fold.n = call[].row_index.rows
    fold.index = _row_numbers(call, estimator, fold.wide)
    if call[].scope != MEMBER_SCOPE_NONE and call[].pin != 0 and _slot_dev_rows(slot, call):
        fold.store_id = _scope_store_id(slot, call)
    return fold^


def _scope_store_id(slot: _SlotPtr, call: MutPointer[MemberCall, MutUntrackedOrigin]) raises -> Int:
    """The store slot of the WHOLE X in the image's device store, put once
    per scope through the scope's cache of this family (a hit after the
    first put: a fold's fit and its predictions share it). THE PIN RULE:
    `pin_kept` is written the moment the cache took the reference, before
    anything after it can raise."""
    var family = FOREST_TABLE_ET if slot[].et else FOREST_TABLE_RF
    var prefix = String(FOREST_STORE_PREFIX_ET) if slot[].et else String(FOREST_STORE_PREFIX_RF)
    var store = forest_store_transport_ref(family, call[].route.mode, call[].route.vendor)
    var put = arena_scope_store_put(call[].scope, store, prefix, call[].x.data,
                                    call[].x.rows * call[].x.cols, call[].pin)
    if put.pinned:
        call[].pin_kept = call[].pin
    return put.store_id


def _scope_y_word(slot: _SlotPtr, call: MutPointer[MemberCall, MutUntrackedOrigin]) raises -> Int:
    """THE PER-INPUT PINS for the fit's labels (lane vdf2-tail2;
    core/member_abi.mojo `MemberCall.y_pin`): the WHOLE base's labels (one
    4-byte word a row: the classifier's int32 codes or the regressor's
    float32 targets) put once per scope into the image's device store under
    the labels' own token, through the scope's cache of this family (a hit
    after the first put: every fold's fit shares it); their store slot as a
    NEGATIVE label word (THE FOLD X WORD numbering), which the image's
    XF_FIT_ROWS body takes rows of on the device (core/mojolearn_rf_run.mojo
    `rf_store_rows_y`, extratrees/mojolearn_trees_run.mojo
    `et_store_take_column`). THE PIN RULE: `y_pin_kept` is written the moment
    the cache took the reference."""
    var family = FOREST_TABLE_ET if slot[].et else FOREST_TABLE_RF
    var prefix = String(FOREST_STORE_PREFIX_ET) if slot[].et else String(FOREST_STORE_PREFIX_RF)
    var store = forest_store_transport_ref(family, call[].route.mode, call[].route.vendor)
    var put = arena_scope_store_put(call[].scope, store, prefix, call[].y.data, call[].y.rows, call[].y_pin)
    if put.pinned:
        call[].y_pin_kept = call[].y_pin
    return fold_x_word(put.store_id)


struct _Query(Movable):
    """A predict / score X (lane vdf-scope-trees2a): `x` its view (`data`
    0 when the rows stay on the device), and under `call.row_index` the
    int64 row numbers `index` (the call's own, or `wide`, the int32 index
    widened) with either `store_id`, the image's store slot of the WHOLE X
    whose rows the snapshot takes on the device, or -1 and `x_fold`, the
    rows gathered on the host."""
    var x: MemberMatrix
    var store_id: Int
    var index: Int
    var x_fold: List[Float32]
    var wide: List[Int64]

    def __init__(out self, x: MemberMatrix):
        self.x = x
        self.store_id = -1
        self.index = 0
        self.x_fold = List[Float32]()
        self.wide = List[Int64]()


def _scope_rows_route(slot: _SlotPtr, call: MutPointer[MemberCall, MutUntrackedOrigin]) raises -> Bool:
    """A prediction reads the fold's rows on the device: a resident scope
    with a pin, an image with a device store, and the state's resident
    snapshot route (the sequential and strict routes take a host X)."""
    if call[].scope == MEMBER_SCOPE_NONE or call[].pin == 0 or not _slot_dev_rows(slot, call):
        return False
    _ensure_prepared(slot)
    return slot[].route != FOREST_ROUTE_SEQUENTIAL and slot[].route != FOREST_ROUTE_STRICT


def _query_rows(slot: _SlotPtr, call: MutPointer[MemberCall, MutUntrackedOrigin], estimator: String) raises -> _Query:
    """A predict / score X: the call's own view, or under `call.row_index`
    its rows: kept on the device under a resident scope (`_scope_rows_route`:
    the snapshot takes them out of the image's store, XF_RESIDENT_ROWS), else
    gathered across the host's cores."""
    if call[].column_index.rows > 0:
        raise Error("TypeError: a " + estimator + " member reads every column (no column index)")
    _dense(call[].x, MEMBER_F32, "X", estimator)
    var q = _Query(call[].x)
    if call[].row_index.rows == 0:
        return q^
    var n = call[].row_index.rows
    var cols = call[].x.cols
    q.index = _row_numbers(call, estimator, q.wide)
    if _scope_rows_route(slot, call):
        q.store_id = _scope_store_id(slot, call)
        q.x = MemberMatrix(0, n, cols, MEMBER_F32)
        return q^
    q.x_fold = List[Float32](length=n * cols, fill=0.0)
    _take_rows_f32(call[].x.data, q.index, n, cols, Int(q.x_fold.unsafe_ptr()))
    q.x = MemberMatrix(Int(q.x_fold.unsafe_ptr()), n, cols, MEMBER_F32)
    return q^


def _fit(slot: _SlotPtr, call: MutPointer[MemberCall, MutUntrackedOrigin], estimator: String) raises:
    # THE GIL RULE: the route entry is marked once per operation; the image
    # then runs its fit and walks with no release (core/forest_entry.mojo).
    # The marked ref is what the slot keeps for its predicts.
    var key = entry_ref_gil_free(call[].route.entry)
    var cols = call[].x.cols
    if call[].column_index.rows > 0:
        raise Error("TypeError: a " + estimator + " member reads every column (no column index)")
    if call[].fit_output != 0:
        member_unsupported(estimator, call[].fit_output)
    _dense(call[].x, MEMBER_F32, "X", estimator)
    if call[].y.cols != 1:
        raise Error("ValueError: a " + estimator + " member fits one target column")
    # THE FOLD ROWS (lane vdf-scope-trees, MEMBER_CAN_ROW_INDEX): X, y and the
    # weights are the WHOLE base and `call.row_index` the fit's rows. Under a
    # resident scope with a pin, X goes ONCE into the image's device store
    # through the scope's cache and the image takes the rows on the device
    # (XF_FIT_ROWS); otherwise the rows are gathered here across the host's
    # cores and the fit is the plain one. The weights (one word a row) are
    # gathered here; y too, except on XF_FIT_ROWS with y's own pin (lane
    # vdf2-tail2: `_scope_y_word`, the labels taken on the device).
    var fold = _FoldRows()
    var base_rows = call[].x.rows
    var rows = base_rows
    if call[].row_index.rows > 0:
        fold = _fold_rows(slot, call, estimator)
        rows = fold.n
    if call[].y.rows != base_rows:
        raise Error("ValueError: Found input variables with inconsistent numbers of samples")
    var weighted = call[].weights.rows > 0
    if weighted:
        _dense(call[].weights, MEMBER_F32, "the sample weights", estimator)
        if call[].weights.rows != base_rows or call[].weights.cols != 1:
            raise Error("ValueError: sample_weight.shape == (" + String(call[].weights.rows)
                + ",), expected (" + String(base_rows) + ",)!")
    var classes = 0
    if slot[].classifier:
        _dense(call[].y, MEMBER_I32, "y (class codes)", estimator)
        classes = call[].classes
        if classes < 2:
            raise Error("ValueError: y has fewer than 2 classes")
    else:
        _dense(call[].y, MEMBER_F32, "y", estimator)
        if weighted and slot[].oob and not slot[].et:
            # The estimator's own refusal (`_rf_fit_regressor`). Lane mf-rfe6: a weighted fit with the importance
            # fit is served (the same fit, then the same importance body over the fitted trees on both columns:
            # core/mojolearn_rf_run.mojo `compute_feature_importances_host`, core/mojolearn_rf_host_run.mojo
            # `rf_host_importance`), as the classifier's was.
            raise Error("NotImplementedError: sample_weight is not implemented together with oob_score "
                        + "or a tree-ID shard")
    _release_snapshot(slot)
    slot[].fitted = False
    slot[].imp = List[Float32]()
    _clear_oob(slot)
    # The fit's own views: the call's, or the fold's gathered words.
    var x_rows_addr = call[].x.data
    var y_addr = call[].y.data
    var w_addr = call[].weights.data if weighted else 0
    var y_fold = List[Int32]()
    var t_fold = List[Float32]()
    var w_fold = List[Float32]()
    var x_fold = List[Float32]()
    # The ET classifier on an image answering XF_CAP_ET_CODES (the ET GPU
    # images; lane vdf-scope-trees3) hands its int32 codes as they are.
    var y_codes = slot[].classifier and slot[].et and (_slot_caps(slot, key) & XF_CAP_ET_CODES) != 0
    # The labels on the device (lane vdf2-tail2): on the XF_FIT_ROWS route
    # inside a scope with the labels' own pin, the image takes the fold's
    # labels out of its store (no host gather); the ET classifier only when
    # it hands int32 codes (the store's words are the caller's bytes).
    var y_dev = (fold.store_id >= 0 and call[].scope != MEMBER_SCOPE_NONE and call[].y_pin != 0
                 and (not (slot[].classifier and slot[].et) or y_codes))
    if fold.n > 0:
        if y_dev:
            y_addr = _scope_y_word(slot, call)
        elif slot[].classifier:
            y_fold = List[Int32](length=rows, fill=0)
            _take_rows_i32(call[].y.data, fold.index, rows, Int(y_fold.unsafe_ptr()))
            y_addr = Int(y_fold.unsafe_ptr())
        else:
            t_fold = List[Float32](length=rows, fill=0.0)
            _take_rows_f32(call[].y.data, fold.index, rows, 1, Int(t_fold.unsafe_ptr()))
            y_addr = Int(t_fold.unsafe_ptr())
        if weighted:
            w_fold = List[Float32](length=rows, fill=0.0)
            _take_rows_f32(call[].weights.data, fold.index, rows, 1, Int(w_fold.unsafe_ptr()))
            w_addr = Int(w_fold.unsafe_ptr())
        if fold.store_id < 0:
            x_fold = List[Float32](length=rows * cols, fill=0.0)
            _take_rows_f32(call[].x.data, fold.index, rows, cols, Int(x_fold.unsafe_ptr()))
            x_rows_addr = Int(x_fold.unsafe_ptr())
    var on_device = fold.store_id >= 0
    # The ET classifier reads float32 codes; the cast lives until the launch
    # returns. An image answering XF_CAP_ET_CODES (the ET GPU images; lane
    # vdf-scope-trees3) takes the int32 codes as they are and makes the class
    # ids on the device (XF_SHAPE_Y_CODES): no host cast. The host images keep it.
    var codes = List[Float32]()
    if slot[].classifier and slot[].et and not y_codes:
        codes = List[Float32](length=rows, fill=0.0)
        _cast_codes(y_addr, Int(codes.unsafe_ptr()), rows)
        y_addr = Int(codes.unsafe_ptr())
    # The fit the estimator's own door picks (module docstring): the RF
    # layout's weighted, importance and out-of-bag fits read X column-major.
    # XF_FIT_ROWS makes its own column-major plane on the device.
    var column_major = not on_device and not slot[].et and (weighted or slot[].importances or slot[].oob)
    var x_cols = List[Float32]()
    var x_addr = x_rows_addr
    if column_major:
        x_cols = List[Float32](length=rows * cols, fill=0.0)
        _column_major(x_rows_addr, Int(x_cols.unsafe_ptr()), rows, cols)
        x_addr = Int(x_cols.unsafe_ptr())
    if on_device:
        x_addr = fold.index  # XF_FIT_ROWS: x = the int64 row ids
    var imp = List[Float32]()
    if slot[].importances:
        imp = List[Float32](length=cols, fill=0.0)
    # The out-of-bag outputs the estimator's oob fit writes (zero-filled, as
    # its `admitted_array` destinations): rows x classes (classifier) or rows
    # float64 words, and the one score word. Moved into the slot after the
    # launch (their last named use).
    var oob_cols = classes if slot[].classifier else 1
    var oob_out = List[Float64]()
    var oob_score = List[Float64]()
    if slot[].oob:
        oob_out = List[Float64](length=rows * oob_cols, fill=0.0)
        oob_score = List[Float64](length=1, fill=0.0)
    var w_max = 0.0
    if weighted and not slot[].classifier and not slot[].et:
        w_max = _largest(w_addr, rows)
    var fit = ForestFit(slot[].classifier, rowmajor_ok=not column_major, weights=w_addr,
                        oob=Int(oob_out.unsafe_ptr()) if slot[].oob else 0,
                        score=Int(oob_score.unsafe_ptr()) if slot[].oob else 0,
                        importance=Int(imp.unsafe_ptr()) if slot[].importances else 0, weight_max=w_max)
    var config = ForestConfig(slot[].et, slot[].ints.copy(), slot[].floats.copy(), slot[].bad.copy())
    config.criterion = slot[].criterion
    config.features.form = slot[].form
    config.features.count = slot[].count
    config.features.fraction = slot[].fraction
    config.bins_image = slot[].bins_image
    config.rows_fraction = slot[].rows_fraction
    # `unweighted` as the ET door passes it: a weighted ET classifier fit
    # clears the class-weight slot (its class weights travel as row weights).
    var request = forest_request(config, key, rows, cols, classes, x_addr, y_addr, fit,
        weighted and slot[].classifier and slot[].et)
    request.y_codes = y_codes
    var handle: Int
    if on_device:
        request.session = fold.store_id
        handle = xf_fit_rows(key, slot[].classifier, request)
    else:
        var device_scan = (_slot_caps(slot, key) & XF_CAP_DEVICE_SCAN) != 0
        handle = forest_fit_core(key, slot[].classifier, not column_major, request, device_scan)
    _ = len(codes)
    _ = len(x_cols)
    _ = len(x_fold)
    _ = len(y_fold)
    _ = len(t_fold)
    _ = len(w_fold)
    _ = len(fold.wide)
    try:
        var sizes = xf_export_sizes(key, handle)
        var counts = forest_export_counts(sizes)
        slot[].offsets = List[Int32](length=counts[0], fill=0)
        slot[].colid = List[Int32](length=counts[1], fill=0)
        slot[].quesval = List[Float32](length=counts[2], fill=0.0)
        slot[].left = List[Int32](length=counts[3], fill=0)
        slot[].leaves = List[Float32](length=counts[4], fill=0.0)
        var dst: List[Int] = [Int(slot[].offsets.unsafe_ptr()), Int(slot[].colid.unsafe_ptr()),
            Int(slot[].quesval.unsafe_ptr()), Int(slot[].left.unsafe_ptr()), Int(slot[].leaves.unsafe_ptr())]
        xf_export_into(key, sizes, dst)
        slot[].features = cols
        slot[].trees = sizes.trees
        slot[].nodes = sizes.nodes
        slot[].outputs = sizes.outputs
        slot[].classes = classes
        _scalars(slot, sizes.meta)
        # The RF layout's third metadata word is the `ensemble/rf_notices.mojo`
        # mask the door's `_rf_fit_notices` reads (ExtraTrees': its depth cap).
        slot[].notices = List[Float64]()
        if not slot[].et and len(sizes.meta) >= 3 and sizes.meta[2] != 0:
            member_notice_add(slot[].notices, MEMBER_NOTICE_FOREST, Float64(sizes.meta[2]))
    finally:
        xf_export_release(key, handle)
    if slot[].importances:
        slot[].imp = imp^
    if slot[].oob:
        slot[].oob_out = oob_out^
        slot[].oob_score = oob_score^
        slot[].oob_rows = rows
        slot[].oob_cols = oob_cols
    _unprepared(slot, key)
    slot[].fitted = True


def _require_fitted(slot: _SlotPtr, estimator: String) raises:
    if not slot[].fitted:
        raise Error("NotFittedError: this " + estimator + " member is not fitted yet")


def _width(slot: _SlotPtr, x: MemberMatrix, estimator: String) raises:
    if x.cols != slot[].features:
        raise Error("ValueError: X has " + String(x.cols) + " features, but " + estimator
            + " is expecting " + String(slot[].features) + " features as input.")


def _vote(slot: _SlotPtr, q: _Query, dst: Int) raises:
    """The rows' votes or values (flat float32 at `dst`) on the state's route."""
    _ensure_prepared(slot)
    var x = q.x
    var wrote = 0
    if q.store_id >= 0:
        # XF_RESIDENT_ROWS: the fold's rows taken on the device (resident route)
        wrote = xf_resident_rows(slot[].key, 0, slot[].snapshot, q.store_id, q.index, dst, x.rows, x.cols,
                                 slot[].outputs)
    elif slot[].route == FOREST_ROUTE_SEQUENTIAL:
        var model: List[Int] = [Int(slot[].offsets.unsafe_ptr()), Int(slot[].colid.unsafe_ptr()),
            Int(slot[].quesval.unsafe_ptr()), Int(slot[].left.unsafe_ptr()), Int(slot[].leaves.unsafe_ptr())]
        # The RF regressor's sequential walk averages (`rf_predict_reg`); the
        # other kinds vote (`rf_predict_proba` / `et_predict`).
        wrote = xf_predict(slot[].key, model, x.data, dst, x.rows, x.cols, slot[].trees, slot[].outputs,
            not slot[].classifier and not slot[].et)
    elif slot[].route == FOREST_ROUTE_STRICT:
        wrote = xf_strict_predict(slot[].key, slot[].snapshot, x.data, dst, x.rows, x.cols, slot[].outputs)
    else:
        wrote = xf_resident_predict(slot[].key, slot[].snapshot, x.data, dst, x.rows, x.cols, slot[].outputs)
    if wrote != x.rows:
        raise Error("RuntimeError: forest prediction wrote " + String(wrote) + " of " + String(x.rows) + " rows")


def _codes(slot: _SlotPtr, q: _Query, dst: Int) raises:
    """A classifier's int32 class codes at `dst`: the resident fused labels
    where the image serves them, else the first-max argmax of the vote."""
    _ensure_prepared(slot)
    var x = q.x
    if slot[].labels:
        var wrote = 0
        if q.store_id >= 0:
            wrote = xf_resident_rows(slot[].key, 1, slot[].snapshot, q.store_id, q.index, dst, x.rows, x.cols,
                                     slot[].outputs)
        else:
            wrote = xf_resident_labels(slot[].key, slot[].snapshot, x.data, dst, x.rows, x.cols, slot[].outputs)
        if wrote != x.rows:
            raise Error("RuntimeError: forest label prediction wrote " + String(wrote) + " of " + String(x.rows) + " rows")
        return
    var votes = List[Float32](length=x.rows * slot[].outputs, fill=0.0)
    _vote(slot, q, Int(votes.unsafe_ptr()))
    _argmax_rows(Int(votes.unsafe_ptr()), dst, x.rows, slot[].outputs)
    _ = len(votes)


def _predict(slot: _SlotPtr, call: MutPointer[MemberCall, MutUntrackedOrigin], proba: Bool, estimator: String) raises:
    _require_fitted(slot, estimator)
    var q = _query_rows(slot, call, estimator)
    var x = q.x
    _width(slot, x, estimator)
    var rows = x.rows
    if proba:
        if not slot[].classifier:
            member_unsupported(estimator, MEMBER_OP_PREDICT_PROBA)
        _dense(call[].dst, MEMBER_F32, "the probability buffer", estimator)
        if call[].dst.rows != rows or call[].dst.cols != slot[].outputs:
            raise Error("ValueError: the probability buffer holds one row of class probabilities per row of X")
        _vote(slot, q, call[].dst.data)
    elif call[].dst.rows != rows or call[].dst.cols != 1:
        raise Error("ValueError: the prediction buffer holds one value per row of X")
    elif slot[].classifier:
        _dense(call[].dst, MEMBER_I32, "the prediction buffer (class codes)", estimator)
        _codes(slot, q, call[].dst.data)
    else:
        _dense(call[].dst, MEMBER_F32, "the prediction buffer", estimator)
        _vote(slot, q, call[].dst.data)
    _ = len(q.x_fold)  # the walk above read the gathered rows
    _ = len(q.wide)


def _score(slot: _SlotPtr, call: MutPointer[MemberCall, MutUntrackedOrigin], estimator: String) raises:
    """A classifier's accuracy (the estimator's `score`): matching codes /
    rows. A regressor's R^2 is not published: the caller predicts and runs
    its own metric."""
    _require_fitted(slot, estimator)
    if not slot[].classifier:
        member_unsupported(estimator, MEMBER_OP_SCORE)
    if call[].weights.rows > 0:
        raise Error("NotImplementedError: mojolearn: a weighted score of a " + estimator
            + " member is not published; the caller predicts and runs its own weighted metric")
    var q = _query_rows(slot, call, estimator)
    var x = q.x
    _width(slot, x, estimator)
    _dense(call[].y, MEMBER_I32, "y (class codes)", estimator)
    var rows = x.rows
    if call[].y.rows != call[].x.rows or call[].y.cols != 1:
        raise Error("ValueError: Found input variables with inconsistent numbers of samples")
    # Under a row index y is the base's column, read through the same rows.
    var y_fold = List[Int32]()
    var y_addr = call[].y.data
    if call[].row_index.rows > 0:
        var index = q.index
        y_fold = List[Int32](length=rows, fill=0)
        _take_rows_i32(call[].y.data, index, rows, Int(y_fold.unsafe_ptr()))
        y_addr = Int(y_fold.unsafe_ptr())
    var codes = List[Int32]()
    var hits: Int
    _ensure_prepared(slot)
    if q.store_id >= 0 and slot[].labels and rows > 0:
        # XF_RESIDENT_ROWS mode 2 (lane vdf-scope-trees3): the fold's y is
        # read ON THE DEVICE against the codes the snapshot made of the rows
        # it took there; one word (the matching count) comes back, no codes.
        hits = xf_resident_rows(slot[].key, 2, slot[].snapshot, q.store_id, q.index, y_addr, rows, x.cols,
                                slot[].outputs)
    else:
        codes = List[Int32](length=rows, fill=0)
        _codes(slot, q, Int(codes.unsafe_ptr()))
        hits = _count_equal(Int(codes.unsafe_ptr()), y_addr, rows)
    _ = len(codes)
    _ = len(q.x_fold)
    _ = len(y_fold)
    _ = len(q.wide)
    call[].value = Float64(hits) / Float64(rows) if rows > 0 else 0.0


def _block_count(slot: _SlotPtr) -> Int:
    return _BLOCKS + 1 if len(slot[].imp) > 0 else _BLOCKS


def _block_of(slot: _SlotPtr, index: Int) raises -> Int:
    if index < 0 or index >= _block_count(slot):
        raise Error("member ABI: this forest member has " + String(_block_count(slot)) + " fitted blocks")
    if index == 6:
        return MEMBER_BLOCK_FEATURE_IMPORTANCES
    return MEMBER_BLOCK_SCALARS if index == 5 else MEMBER_BLOCK_FAMILY + index


def _export(slot: _SlotPtr, call: MutPointer[MemberCall, MutUntrackedOrigin], estimator: String) raises:
    """A view of one fitted block. The slot owns the words; they stay valid
    until this state's next fit, import or free."""
    _require_fitted(slot, estimator)
    var block = call[].block
    if (call[].flags & MEMBER_BLOCK_BY_INDEX) != 0:
        block = _block_of(slot, block)
    if block == MEMBER_BLOCK_FAMILY:
        call[].dst = MemberMatrix(Int(slot[].offsets.unsafe_ptr()), 1, len(slot[].offsets), MEMBER_I32)
    elif block == MEMBER_BLOCK_FAMILY + 1:
        call[].dst = MemberMatrix(Int(slot[].colid.unsafe_ptr()), 1, len(slot[].colid), MEMBER_I32)
    elif block == MEMBER_BLOCK_FAMILY + 2:
        call[].dst = MemberMatrix(Int(slot[].quesval.unsafe_ptr()), 1, len(slot[].quesval), MEMBER_F32)
    elif block == MEMBER_BLOCK_FAMILY + 3:
        call[].dst = MemberMatrix(Int(slot[].left.unsafe_ptr()), 1, len(slot[].left), MEMBER_I32)
    elif block == MEMBER_BLOCK_FAMILY + 4:
        call[].dst = MemberMatrix(Int(slot[].leaves.unsafe_ptr()), slot[].nodes, slot[].outputs, MEMBER_F32)
    elif block == MEMBER_BLOCK_SCALARS:
        call[].dst = MemberMatrix(Int(slot[].scalars.unsafe_ptr()), 1, len(slot[].scalars), MEMBER_F64)
    elif block == MEMBER_BLOCK_FEATURE_IMPORTANCES and len(slot[].imp) > 0:
        call[].dst = MemberMatrix(Int(slot[].imp.unsafe_ptr()), 1, len(slot[].imp), MEMBER_F32)
    elif block == MEMBER_BLOCK_FEATURE_IMPORTANCES:
        raise Error("AttributeError: this " + estimator + " member was fitted without feature importances "
            + "(parameter word 10)")
    elif (block == FM_BLOCK_OOB or block == FM_BLOCK_OOB_SCORE) and len(slot[].oob_score) == 0:
        raise Error("AttributeError: this " + estimator + " member was fitted without out-of-bag outputs "
            + "(parameter word 11)")
    elif block == FM_BLOCK_OOB:
        call[].dst = MemberMatrix(Int(slot[].oob_out.unsafe_ptr()), slot[].oob_rows, slot[].oob_cols, MEMBER_F64)
    elif block == FM_BLOCK_OOB_SCORE:
        call[].dst = MemberMatrix(Int(slot[].oob_score.unsafe_ptr()), 1, 1, MEMBER_F64)
    else:
        raise Error("member ABI: a forest member has no fitted block " + String(block))
    call[].block = block
    call[].out_cols = _block_count(slot)


def _copy_i32(view: MemberMatrix, estimator: String) raises -> List[Int32]:
    _dense(view, MEMBER_I32, "a fitted block", estimator)
    var count = view.rows * view.cols
    var words = List[Int32](length=count, fill=0)
    unsafe_memcpy(dest=words.unsafe_ptr(), src=_I32P(unsafe_from_address=view.data), count=count)
    return words^


def _copy_f32(view: MemberMatrix, estimator: String) raises -> List[Float32]:
    _dense(view, MEMBER_F32, "a fitted block", estimator)
    var count = view.rows * view.cols
    var words = List[Float32](length=count, fill=0.0)
    unsafe_memcpy(dest=words.unsafe_ptr(), src=_F32P(unsafe_from_address=view.data), count=count)
    return words^


def _import(slot: _SlotPtr, call: MutPointer[MemberCall, MutUntrackedOrigin], estimator: String) raises:
    """Copy one fitted block in; MEMBER_IMPORT_DONE checks the six blocks
    against each other (the estimator's `forest_shape_check`), prepares the
    predict route and marks the state fitted."""
    var block = call[].block
    if block != 0 or (call[].flags & MEMBER_IMPORT_DONE) == 0:
        _release_snapshot(slot)
        slot[].fitted = False
    if block == MEMBER_BLOCK_FAMILY:
        # The first block of an import: importances only when imported again;
        # an imported state has no out-of-bag outputs.
        slot[].imp = List[Float32]()
        _clear_oob(slot)
        slot[].offsets = _copy_i32(call[].x, estimator)
    elif block == MEMBER_BLOCK_FAMILY + 1:
        slot[].colid = _copy_i32(call[].x, estimator)
    elif block == MEMBER_BLOCK_FAMILY + 2:
        slot[].quesval = _copy_f32(call[].x, estimator)
    elif block == MEMBER_BLOCK_FAMILY + 3:
        slot[].left = _copy_i32(call[].x, estimator)
    elif block == MEMBER_BLOCK_FAMILY + 4:
        slot[].leaves = _copy_f32(call[].x, estimator)
    elif block == MEMBER_BLOCK_FEATURE_IMPORTANCES:
        if slot[].et:
            raise Error("member ABI: an ExtraTrees member has no feature importances block")
        slot[].imp = _copy_f32(call[].x, estimator)
    elif block == MEMBER_BLOCK_SCALARS:
        if not call[].x.dense_host(MEMBER_F64) or call[].x.rows * call[].x.cols < 6:
            raise Error("member ABI: the forest scalars block is host float64 words "
                + "[features, trees, nodes, outputs, classes, n_meta, meta...]")
        var count = call[].x.rows * call[].x.cols
        var source = _F64P(unsafe_from_address=call[].x.data)
        if count != 6 + Int(source[5]):
            raise Error("member ABI: the forest scalars block's metadata count disagrees with its length")
        slot[].scalars = List[Float64](length=count, fill=0.0)
        unsafe_memcpy(dest=slot[].scalars.unsafe_ptr(), src=source, count=count)
        slot[].notices = List[Float64]()
    elif block != 0:
        raise Error("member ABI: a forest member has no fitted block " + String(block))
    if (call[].flags & MEMBER_IMPORT_DONE) == 0:
        return
    if len(slot[].scalars) < 6:
        raise Error("member ABI: a fitted forest state needs its six blocks")
    slot[].features = Int(slot[].scalars[0])
    slot[].trees = Int(slot[].scalars[1])
    slot[].nodes = Int(slot[].scalars[2])
    slot[].outputs = Int(slot[].scalars[3])
    slot[].classes = Int(slot[].scalars[4])
    var nodes = slot[].nodes
    var last = -1
    if len(slot[].offsets) > 0:
        last = Int(slot[].offsets[len(slot[].offsets) - 1])
    if (slot[].features < 1 or slot[].trees < 1 or slot[].outputs < 1 or len(slot[].offsets) != slot[].trees + 1
            or nodes < 1 or len(slot[].colid) != nodes or len(slot[].quesval) != nodes or len(slot[].left) != nodes
            or len(slot[].leaves) != nodes * slot[].outputs or last != nodes):
        raise Error("forest model array shapes do not match metadata")
    if slot[].classifier and slot[].classes != slot[].outputs:
        raise Error("member ABI: a forest classifier's class count differs from its outputs")
    if len(slot[].imp) > 0 and len(slot[].imp) != slot[].features:
        raise Error("member ABI: the forest feature importances block holds one word per feature")
    _unprepared(slot, entry_ref_gil_free(call[].route.entry))
    slot[].fitted = True


def _capabilities(classifier: Bool, importances: Bool, weights: Bool, notices: Bool,
                  predict_only: Bool = False) -> Int:
    """The bits a kind honors. `weights` False: the kind's fit refuses
    `call.weights` (the RF-layout regressor with the out-of-bag fit, the
    estimator's own refusal), so it is not advertised. No
    `MEMBER_CAN_TEST_ROWS`: the predict entries (XF_PREDICT,
    XF_RESIDENT_PREDICT, XF_RESIDENT_LABELS, XF_STRICT_PREDICT) never test X,
    and only images with XF_CAP_DEVICE_SCAN test it at fit; the caller's
    admission scans. No `MEMBER_CAN_DEVICE_VIEWS`: every forest predict entry
    takes a HOST address of X and uploads it itself (no entry op binds a
    resident store id), so a device-home X is refused (`_dense`).
    `notices`: the kind carries its fit notices (Forest; `fm_member`)."""
    # MEMBER_CAN_ROW_INDEX and MEMBER_CAN_SCOPE are added by `fm_member` on a
    # route whose image answers XF_CAP_DEV_ROWS (lanes vdf-scope-trees, -trees2a).
    # MEMBER_GIL_FREE (THE GIL RULE): every operation of the six kinds is
    # traced (fx-gil-RESULT.md, "forest / tree / extra_trees: HAS the bit"):
    # the route entry is marked `entry_ref_gil_free` at fit and import and
    # kept so, the four images run their fits and walks bare under the bit,
    # and the member store, the export, resident, grove and strict
    # registries and the workspace pool each have their own lock word.
    # Withdrawn at 45e90deaa7 and restored by lane pv-race (2026-11-22): the
    # random crash of parallel meta fits was the export registry's
    # pre-lock snapshot read (core/forest_export_run.mojo, THE TABLE IS
    # BEHIND A POINTER), now fixed.
    var mask = MEMBER_CAN_FIT | MEMBER_CAN_PREDICT | MEMBER_CAN_EXPORT | MEMBER_CAN_IMPORT | MEMBER_GIL_FREE
    if predict_only:
        # A saved model (kinds 2 / 3): predict (and a classifier's predict_proba) over its imported arrays.
        var held = MEMBER_CAN_PREDICT | MEMBER_CAN_EXPORT | MEMBER_CAN_IMPORT | MEMBER_GIL_FREE
        if classifier:
            held |= MEMBER_CAN_PREDICT_PROBA | MEMBER_CAN_CLASSIFIER
        return held
    if notices:
        mask |= MEMBER_CAN_NOTICES
    if weights:
        mask |= MEMBER_CAN_SAMPLE_WEIGHT
    if importances:
        mask |= MEMBER_CAN_IMPORTANCES
    if classifier:
        mask |= MEMBER_CAN_PREDICT_PROBA | MEMBER_CAN_SCORE | MEMBER_CAN_CLASSIFIER
    return mask


def fm_member[name: StaticString, rf_layout: Bool, notices: Bool](operation: Int, address: Int,
                                                                   estimator: String) raises -> Int:
    """One member call on the store `name` (the family's own store).
    `rf_layout`: the family fits on the RF slots (Tree, Forest), whose fit
    computes feature importances; the ExtraTrees family does not.
    `notices`: the family's fit can raise a notice (Forest: max_samples
    without bootstrap, partial out-of-bag scores). Tree's only notice was the
    n_bins clamp, no longer raised (lane w3-fix-h, 2026-11-24), so it does
    not advertise MEMBER_CAN_NOTICES and its callers skip the notice read."""
    var call = MutPointer[MemberCall, MutUntrackedOrigin](unsafe_from_address=address)
    if operation == MEMBER_OP_CAPABILITIES:
        # THE FOLD ROWS (lane vdf-scope-trees): MEMBER_CAN_ROW_INDEX and
        # MEMBER_CAN_SCOPE on a route whose image keeps a device store
        # (XF_CAP_DEV_ROWS: the RandomForest and ExtraTrees GPU images; one
        # entry call per capability question, asked once per member class).
        if call[].state == 0:
            var held = _capabilities(True, rf_layout, True, notices)
            if _dev_rows_route(call):
                held |= MEMBER_CAN_ROW_INDEX | MEMBER_CAN_SCOPE
            return held
        var known = _SlotPtr(unsafe_from_address=_lookup[name](call[].state, False))
        # Kind AND settings exact: the RF-layout regressor whose every fit
        # computes out-of-bag outputs refuses weights (`_fit`; lane mf-rfe6:
        # the importance fit takes them).
        var weights = known[].classifier or known[].et or not known[].oob
        var mask = _capabilities(known[].classifier, known[].importances or len(known[].imp) > 0, weights,
                                 notices and not known[].et, known[].predict_only)
        if not known[].predict_only and _slot_dev_rows(known, call):
            mask |= MEMBER_CAN_ROW_INDEX | MEMBER_CAN_SCOPE
        return mask
    if operation == MEMBER_OP_NEW:
        if not call[].params:
            raise Error("member ABI: a forest member needs its parameter words")
        call[].state = _mint[name](_F64P(unsafe_from_address=call[].params), call[].param_count)
        _admit_caps(_SlotPtr(unsafe_from_address=_lookup[name](call[].state, False)), call)
        return 0
    if operation == MEMBER_OP_FREE:
        var held = _SlotPtr(unsafe_from_address=_lookup[name](call[].state, False))
        _release_snapshot(held)
        _ = _lookup[name](call[].state, True)
        call[].state = 0
        return 0
    if operation == MEMBER_OP_OUTPUTS:
        # With a state the state answers; with none, the kind word of the NEW
        # parameters (word 0) and the caller's `call.classes` (item 10).
        var proba = call[].flags == MEMBER_OP_PREDICT_PROBA
        var classifier: Bool
        if call[].state != 0:
            var known = _SlotPtr(unsafe_from_address=_lookup[name](call[].state, False))
            call[].classes = known[].classes
            classifier = known[].classifier
        else:
            if not call[].params or call[].param_count < 1:
                raise Error("member ABI: a forest member answers OUTPUTS without a state from its parameter words")
            var word = _F64P(unsafe_from_address=call[].params)[0]
            classifier = word == 0.0 or word == 2.0
        call[].out_cols = call[].classes if proba else 1
        # predict: int32 codes (classifier) or float32 values; predict_proba
        # (classifier only) float32. Another operation stays unsaid.
        if call[].flags == MEMBER_OP_PREDICT:
            call[].out_dtype = MEMBER_I32 if classifier else MEMBER_F32
        elif proba and classifier:
            call[].out_dtype = MEMBER_F32
        return 0
    var slot = _SlotPtr(unsafe_from_address=_lookup[name](call[].state, False))
    if slot[].predict_only and (operation == MEMBER_OP_FIT or operation == MEMBER_OP_SCORE):
        # A saved model has no fit and no score (HostForest defines neither: Python's words).
        raise Error("AttributeError: 'HostForest' object has no attribute '"
                    + (String("fit") if operation == MEMBER_OP_FIT else String("score")) + "'")
    if operation == MEMBER_OP_FIT:
        _fit(slot, call, estimator)
    elif operation == MEMBER_OP_PREDICT:
        _predict(slot, call, False, estimator)
    elif operation == MEMBER_OP_PREDICT_PROBA:
        _predict(slot, call, True, estimator)
    elif operation == MEMBER_OP_SCORE:
        _score(slot, call, estimator)
    elif operation == MEMBER_OP_EXPORT and call[].block == MEMBER_BLOCK_NOTICES and (
            call[].flags & MEMBER_BLOCK_BY_INDEX) == 0:
        # FIT NOTICES, by name (never in the index walk).
        call[].dst = MemberMatrix(Int(slot[].notices.unsafe_ptr()), len(slot[].notices) // MEMBER_NOTICE_WORDS,
                                  MEMBER_NOTICE_WORDS, MEMBER_F64)
    elif operation == MEMBER_OP_EXPORT:
        _export(slot, call, estimator)
    elif operation == MEMBER_OP_IMPORT:
        _import(slot, call, estimator)
    else:
        member_unsupported(estimator, operation)
    return 0
