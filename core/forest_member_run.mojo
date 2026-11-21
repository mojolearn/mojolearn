# SPDX-License-Identifier: Apache-2.0
"""The forest member RUN shared by the Tree (13), Forest (14) and ExtraTrees
(15) families (core/member_abi.mojo). Each family file
(core/tree_member.mojo, core/forest_member.mojo, core/extra_trees_member.mojo)
names its own store and calls `fm_member[store, rf_layout]` from its entry.

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
  [0] kind: 0 classifier, 1 regressor
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
  [11] S, the parameter slot count; then S triples
      [12 + 3k] the int reading of slot k as the Float64 bit pattern of its
      Int64 (std.memory.bitcast: seeds and counts travel exactly),
      [13 + 3k] its float reading, [14 + 3k] 1 when it has no reading.
Views: X dense row-major host float32 (finite: the caller's admission, as
the estimator's own door; an image with the device scan tests it again);
y host int32 class codes 0..classes-1 (classifier, `call.classes` >= 2) or a
float32 target column (regressor); `call.weights` (optional) host float32,
one per row (finite and non-negative: the caller's admission, as the
estimator's `admit_weights`), multiplied by nothing (no class_weight: the
admission keeps the estimator's own doors for it). Refused in words: a row
or column index, a device-home view, a nonzero `fit_output`, a multi-column
y, a weighted score, and on a regressor of the RF layout weights together
with importances (the estimator's own refusal). predict writes int32 codes
(classifier) or float32 values; predict_proba float32 rows x classes; score
(classifier only) the accuracy count / rows, unweighted.

Fits, as the estimator's own fit door picks them: unweighted without
importances reads X row-major (the view itself, no copy); the ExtraTrees
weighted fit also reads it row-major; the RF layout's weighted fit and its
importance fit read X column-major, so the member writes the column-major
float32 copy the estimator's door writes for them (`order="F"`), split over
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
FIT NOTICES (Tree, Forest: MEMBER_CAN_NOTICES): `MEMBER_BLOCK_NOTICES` by name,
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
Parallel: the ET classifier's code cast, the column-major copy, the largest
weight, the argmax of the votes and the accuracy count split rows over the
host workers. Serial by nature: the
store's bookkeeping.
"""
from std.sys.info import size_of
from std.memory import unsafe_memcpy, bitcast
from core.entry_table import EntryRef, entry_ref_unresolved
from core.forest_calls import (
    xf_caps, xf_export_sizes, xf_export_into, xf_export_release, xf_predict, xf_resident_prepare,
    xf_resident_predict, xf_resident_labels, xf_strict_prepare, xf_strict_predict, xf_snapshot_release,
)
from core.forest_fit_request import XF_CAP_DEVICE_SCAN, XF_RESIDENT_RELEASE, XF_STRICT_RELEASE
from bindings.forest_config_run import ForestConfig, forest_request
from bindings.forest_predict_run import (
    ForestState, forest_labels_route, FOREST_ROUTE_SEQUENTIAL, FOREST_ROUTE_STRICT,
)
from xtrees.trees_forest_run import ForestFit, forest_fit_core, forest_export_counts
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk
from core.member_abi import (
    MemberCall, MemberMatrix, MEMBER_F32, MEMBER_F64, MEMBER_I32,
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
comptime _HEAD = 12
comptime _BLOCKS = 6
comptime _SLOTS_WORD = 11

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
    var importances: Bool
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
        self.importances = False
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
        self.notices = List[Float64]()
        self.key = entry_ref_unresolved()
        self.route = FOREST_ROUTE_SEQUENTIAL
        self.labels = False
        self.snapshot = 0
        self.release_op = 0
        self.prepared = 0
        self.prepare_lock = 0


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
    if words[0] != 0.0 and words[0] != 1.0:
        raise Error("member ABI: unknown forest member kind")
    if words[10] != 0.0 and words[1] != 0.0:
        raise Error("NotImplementedError: mojolearn: the ExtraTrees fit computes no feature importances")
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
        slot[].classifier = words[0] == 0.0
        slot[].et = words[1] != 0.0
        slot[].criterion = Int(words[2])
        slot[].form = Int(words[3])
        slot[].count = Int(words[4])
        slot[].fraction = words[5]
        slot[].bins_image = words[6] != 0.0
        slot[].rows_fraction = words[7]
        slot[].fast = words[8] != 0.0
        slot[].groves = words[9] != 0.0
        slot[].importances = words[10] != 0.0
        slot[].ints = List[Int](length=slots, fill=0)
        slot[].floats = List[Float64](length=slots, fill=0.0)
        slot[].bad = List[UInt8](length=slots, fill=0)
        for k in range(slots):  # small-loop(slots): about twenty parameter slots, once per state
            slot[].ints[k] = Int(bitcast[DType.int64](words[_HEAD + 3 * k]))
            slot[].floats[k] = words[_HEAD + 3 * k + 1]
            slot[].bad[k] = UInt8(1) if words[_HEAD + 3 * k + 2] != 0.0 else UInt8(0)
        slot[].fitted = False
        slot[].imp = List[Float32]()
        slot[].snapshot = 0
        slot[].prepared = 0
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
            slot[].notices = List[Float64]()
            store[].free.append(index)
        return address
    finally:
        shared_slot_unlock(lock)


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


def _whole(call: MutPointer[MemberCall, MutUntrackedOrigin], estimator: String) raises:
    if call[].row_index.rows > 0 or call[].column_index.rows > 0:
        raise Error("TypeError: a " + estimator + " member reads whole host views (no row or column index)")


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
    var caps = xf_caps(key)
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


def _fit(slot: _SlotPtr, call: MutPointer[MemberCall, MutUntrackedOrigin], estimator: String) raises:
    var key = call[].route.entry
    var rows = call[].x.rows
    var cols = call[].x.cols
    _whole(call, estimator)
    if call[].fit_output != 0:
        member_unsupported(estimator, call[].fit_output)
    _dense(call[].x, MEMBER_F32, "X", estimator)
    if call[].y.cols != 1:
        raise Error("ValueError: a " + estimator + " member fits one target column")
    if call[].y.rows != rows:
        raise Error("ValueError: Found input variables with inconsistent numbers of samples")
    var weighted = call[].weights.rows > 0
    if weighted:
        _dense(call[].weights, MEMBER_F32, "the sample weights", estimator)
        if call[].weights.rows != rows or call[].weights.cols != 1:
            raise Error("ValueError: sample_weight.shape == (" + String(call[].weights.rows)
                + ",), expected (" + String(rows) + ",)!")
    var classes = 0
    if slot[].classifier:
        _dense(call[].y, MEMBER_I32, "y (class codes)", estimator)
        classes = call[].classes
        if classes < 2:
            raise Error("ValueError: y has fewer than 2 classes")
    else:
        _dense(call[].y, MEMBER_F32, "y", estimator)
        if weighted and slot[].importances and not slot[].et:
            # The estimator's own refusal (`_rf_fit_regressor`).
            raise Error("NotImplementedError: sample_weight is not implemented together with oob_score, feature "
                        + "importances or a tree-ID shard")
    _release_snapshot(slot)
    slot[].fitted = False
    slot[].imp = List[Float32]()
    # The ET classifier reads float32 codes; the cast lives until the launch returns.
    var codes = List[Float32]()
    var y_addr = call[].y.data
    if slot[].classifier and slot[].et:
        codes = List[Float32](length=rows, fill=0.0)
        _cast_codes(call[].y.data, Int(codes.unsafe_ptr()), rows)
        y_addr = Int(codes.unsafe_ptr())
    # The fit the estimator's own door picks (module docstring): the RF
    # layout's weighted and importance fits read X column-major.
    var w_addr = call[].weights.data if weighted else 0
    var column_major = not slot[].et and (weighted or slot[].importances)
    var x_cols = List[Float32]()
    var x_addr = call[].x.data
    if column_major:
        x_cols = List[Float32](length=rows * cols, fill=0.0)
        _column_major(call[].x.data, Int(x_cols.unsafe_ptr()), rows, cols)
        x_addr = Int(x_cols.unsafe_ptr())
    var imp = List[Float32]()
    if slot[].importances:
        imp = List[Float32](length=cols, fill=0.0)
    var w_max = 0.0
    if weighted and not slot[].classifier and not slot[].et:
        w_max = _largest(w_addr, rows)
    var fit = ForestFit(slot[].classifier, rowmajor_ok=not column_major, weights=w_addr,
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
    var device_scan = (xf_caps(key) & XF_CAP_DEVICE_SCAN) != 0
    var handle = forest_fit_core(key, slot[].classifier, not column_major, request, device_scan)
    _ = len(codes)
    _ = len(x_cols)
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
    _unprepared(slot, key)
    slot[].fitted = True


def _require_fitted(slot: _SlotPtr, estimator: String) raises:
    if not slot[].fitted:
        raise Error("NotFittedError: this " + estimator + " member is not fitted yet")


def _width(slot: _SlotPtr, call: MutPointer[MemberCall, MutUntrackedOrigin], estimator: String) raises:
    _whole(call, estimator)
    _dense(call[].x, MEMBER_F32, "X", estimator)
    if call[].x.cols != slot[].features:
        raise Error("ValueError: X has " + String(call[].x.cols) + " features, but " + estimator
            + " is expecting " + String(slot[].features) + " features as input.")


def _vote(slot: _SlotPtr, x: MemberMatrix, dst: Int) raises:
    """The rows' votes or values (flat float32 at `dst`) on the state's route."""
    _ensure_prepared(slot)
    var wrote = 0
    if slot[].route == FOREST_ROUTE_SEQUENTIAL:
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


def _codes(slot: _SlotPtr, x: MemberMatrix, dst: Int) raises:
    """A classifier's int32 class codes at `dst`: the resident fused labels
    where the image serves them, else the first-max argmax of the vote."""
    _ensure_prepared(slot)
    if slot[].labels:
        var wrote = xf_resident_labels(slot[].key, slot[].snapshot, x.data, dst, x.rows, x.cols, slot[].outputs)
        if wrote != x.rows:
            raise Error("RuntimeError: forest label prediction wrote " + String(wrote) + " of " + String(x.rows) + " rows")
        return
    var votes = List[Float32](length=x.rows * slot[].outputs, fill=0.0)
    _vote(slot, x, Int(votes.unsafe_ptr()))
    _argmax_rows(Int(votes.unsafe_ptr()), dst, x.rows, slot[].outputs)
    _ = len(votes)


def _predict(slot: _SlotPtr, call: MutPointer[MemberCall, MutUntrackedOrigin], proba: Bool, estimator: String) raises:
    _require_fitted(slot, estimator)
    _width(slot, call, estimator)
    var rows = call[].x.rows
    if proba:
        if not slot[].classifier:
            member_unsupported(estimator, MEMBER_OP_PREDICT_PROBA)
        _dense(call[].dst, MEMBER_F32, "the probability buffer", estimator)
        if call[].dst.rows != rows or call[].dst.cols != slot[].outputs:
            raise Error("ValueError: the probability buffer holds one row of class probabilities per row of X")
        _vote(slot, call[].x, call[].dst.data)
        return
    if call[].dst.rows != rows or call[].dst.cols != 1:
        raise Error("ValueError: the prediction buffer holds one value per row of X")
    if slot[].classifier:
        _dense(call[].dst, MEMBER_I32, "the prediction buffer (class codes)", estimator)
        _codes(slot, call[].x, call[].dst.data)
    else:
        _dense(call[].dst, MEMBER_F32, "the prediction buffer", estimator)
        _vote(slot, call[].x, call[].dst.data)


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
    _width(slot, call, estimator)
    _dense(call[].y, MEMBER_I32, "y (class codes)", estimator)
    var rows = call[].x.rows
    if call[].y.rows != rows or call[].y.cols != 1:
        raise Error("ValueError: Found input variables with inconsistent numbers of samples")
    var codes = List[Int32](length=rows, fill=0)
    _codes(slot, call[].x, Int(codes.unsafe_ptr()))
    var hits = _count_equal(Int(codes.unsafe_ptr()), call[].y.data, rows)
    _ = len(codes)
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
        # The first block of an import: importances only when imported again.
        slot[].imp = List[Float32]()
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
    _unprepared(slot, call[].route.entry)
    slot[].fitted = True


def _capabilities(classifier: Bool, importances: Bool, weights: Bool, notices: Bool) -> Int:
    """The bits a kind honors. `weights` False: the kind's fit refuses
    `call.weights` (the RF-layout regressor with the importance fit, the
    estimator's own refusal), so it is not advertised. No
    `MEMBER_CAN_TEST_ROWS`: the predict entries (XF_PREDICT,
    XF_RESIDENT_PREDICT, XF_RESIDENT_LABELS, XF_STRICT_PREDICT) never test X,
    and only images with XF_CAP_DEVICE_SCAN test it at fit; the caller's
    admission scans. No `MEMBER_CAN_DEVICE_VIEWS`: every forest predict entry
    takes a HOST address of X and uploads it itself (no entry op binds a
    resident store id), so a device-home X is refused (`_dense`).
    `notices`: the RF layout (Tree, Forest) carries its fit notices."""
    var mask = MEMBER_CAN_FIT | MEMBER_CAN_PREDICT | MEMBER_CAN_EXPORT | MEMBER_CAN_IMPORT
    if notices:
        mask |= MEMBER_CAN_NOTICES
    if weights:
        mask |= MEMBER_CAN_SAMPLE_WEIGHT
    if importances:
        mask |= MEMBER_CAN_IMPORTANCES
    if classifier:
        mask |= MEMBER_CAN_PREDICT_PROBA | MEMBER_CAN_SCORE | MEMBER_CAN_CLASSIFIER
    return mask


def fm_member[name: StaticString, rf_layout: Bool](operation: Int, address: Int, estimator: String) raises -> Int:
    """One member call on the store `name` (the family's own store).
    `rf_layout`: the family fits on the RF slots (Tree, Forest), whose fit
    computes feature importances; the ExtraTrees family does not."""
    var call = MutPointer[MemberCall, MutUntrackedOrigin](unsafe_from_address=address)
    if operation == MEMBER_OP_CAPABILITIES:
        if call[].state == 0:
            return _capabilities(True, rf_layout, True, rf_layout)
        var known = _SlotPtr(unsafe_from_address=_lookup[name](call[].state, False))
        # Kind AND settings exact: the RF-layout regressor whose every fit
        # computes importances refuses weights (`_fit`).
        var weights = known[].classifier or known[].et or not known[].importances
        return _capabilities(known[].classifier, known[].importances or len(known[].imp) > 0, weights,
                             not known[].et)
    if operation == MEMBER_OP_NEW:
        if not call[].params:
            raise Error("member ABI: a forest member needs its parameter words")
        call[].state = _mint[name](_F64P(unsafe_from_address=call[].params), call[].param_count)
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
            classifier = _F64P(unsafe_from_address=call[].params)[0] == 0.0
        call[].out_cols = call[].classes if proba else 1
        # predict: int32 codes (classifier) or float32 values; predict_proba
        # (classifier only) float32. Another operation stays unsaid.
        if call[].flags == MEMBER_OP_PREDICT:
            call[].out_dtype = MEMBER_I32 if classifier else MEMBER_F32
        elif proba and classifier:
            call[].out_dtype = MEMBER_F32
        return 0
    var slot = _SlotPtr(unsafe_from_address=_lookup[name](call[].state, False))
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
