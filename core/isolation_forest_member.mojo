# SPDX-License-Identifier: Apache-2.0
"""IsolationForest as a MEMBER (core/member_abi.mojo; family 18,
`MEMBER_FAMILY_ISOLATION_FOREST`).

One entry, `isolation_forest_member_v2`, published by each svm image
(`_mojolearn_svm` on a GPU vendor, `_mojolearn_svm_host` on the CPU column)
beside its iforest entry. It runs the family's typed RUN, the iforest entry of
core/iforest_entry.mojo, through the route slot it is handed: `entry` = that
image's iforest entry (the caller's admit resolves it from the estimator's
tier). No kernel is touched; the fit and the scores are the estimator's own
(`fit` is one fresh forest scored on one row and kept resident under its
token, as the estimator's door does; `fit_output` PREDICT / DECISION /
SCORE_SAMPLES is its `fit_predict`: one fresh forest that scores X).

MEMBER_OP_NEW parameters (Float64 words, 13): [0] kind (0 IsolationForest),
[1] n_estimators, [2] max_samples mode (0 'auto', 1 an int, 2 a fraction),
[3] max_samples int, [4] max_samples fraction, [5] max_depth (-1 None),
[6] max_features mode (0 a fraction, 1 an int), [7] max_features int,
[8] max_features fraction, [9] bootstrap (0 / 1), [10] random_state (None 0),
[11] contamination 'auto' (0 / 1), [12] contamination (read when [11] is 0).
The words are the ones the estimator's own parameter checks resolve; the
admit runs those checks.

Views: X dense host row-major float32 (fit and every query). FIT reads no y
and no sample weights (the estimator takes neither): a given y or weights
view is refused. PREDICT writes host int32 labels (1 inlier, -1 outlier);
DECISION and SCORE_SAMPLES write host float32, or float64 where the image
writes the float64 scores itself (IF_CAP_SCORES_F64 on the resident route).
Row and column indexes and device-home views are refused.

THE TRAINING VIEW IS BORROWED, NOT COPIED (ADDED 18:35Z): the forest is keyed
on the training matrix's address and a resident forest the store evicted (or
an image without the resident route, which refits per call) refits from it,
so the caller keeps X alive until FREE, exactly as the estimator keeps the
caller's float32 X as `_x`. FREE drops the resident forest (IF_OP_RELEASE).

DEVICE PLAN (item 28): the family's admit reads the public thread's plan
once (`device_plan_current("MOJOLEARN_IFOREST_DEVICE_COUNT")`) into
`plan.devices`; FIT copies `call.devices` (device ids, rank order) into the
slot, and every run of that state (the fit, a refit after eviction, the
export) hands them to the image (IF_P_DEVICES / IF_P_DEVICE_COUNT): more
than one id runs the estimator's whole-tree shards over exactly those
devices; none or one is `single` (IF_P_SINGLE), the route's device. The
member asks no thread plan and no environment variable, so a caller's worker
thread runs on the plan the public thread had.

FITTED-STATE BLOCKS (EXPORT / IMPORT, index order):
  0 MEMBER_BLOCK_SCALARS float64 1 x 3: offset_, max_samples_, n_features_in_;
  1 MEMBER_BLOCK_FAMILY + 0 int32 1 x (trees + 1): offsets (tree t's first node);
  2 MEMBER_BLOCK_FAMILY + 1 int32 1 x nodes: feature (< 0 marks a leaf);
  3 MEMBER_BLOCK_FAMILY + 2 float32 1 x nodes: threshold (a leaf's path length);
  4 MEMBER_BLOCK_FAMILY + 3 int32 1 x nodes: left (tree-local node ids);
  5 MEMBER_BLOCK_FAMILY + 4 int32 1 x nodes: right.
The compact saved form of isolation_forest/forest_io.mojo (`save()`'s
`_if_forest`). EXPORT of a fitted state reads it out of the image once
(IF_OP_EXPORT, the resident forest the scores use) into lists the slot
owns; the views stay valid until the state's next fit, import or free.
IMPORT copies the five arrays in; an imported state scores through
IF_OP_FOREST_RUN (uploaded once, then resident under its own token): the
same kernels as a fitted one, so the same bits.

The member state lives in a process-wide store of fixed chunks: a slot never
moves once minted, so a call runs on its slot without the store's lock; the
lock covers mint, free and the handle lookup. Members are independent; the
image entry is thread-safe (its resident store holds its own lock), so a
caller runs one handle per worker. Serial by nature here: the store's
bookkeeping.
"""
from std.memory import unsafe_memcpy
from std.sys.info import size_of
from core.entry_table import EntryRef, entry_ref_unresolved
from core.iforest_entry import (
    IFForestRunRequest, IFForestSpans, IFRunRequest, IF_CAP_RESIDENT, IF_CAP_SCORES_F64, IF_EXPORT_INFO,
    if_entry_caps, if_entry_export, if_entry_forest_run, if_entry_release, if_entry_run,
)
from core.member_abi import (
    MemberCall, MemberMatrix, MEMBER_F32, MEMBER_F64, MEMBER_I32,
    MEMBER_OP_CAPABILITIES, MEMBER_OP_NEW, MEMBER_OP_FREE, MEMBER_OP_FIT, MEMBER_OP_PREDICT,
    MEMBER_OP_DECISION, MEMBER_OP_SCORE_SAMPLES, MEMBER_OP_OUTPUTS, MEMBER_OP_EXPORT, MEMBER_OP_IMPORT,
    MEMBER_CAN_FIT, MEMBER_CAN_PREDICT, MEMBER_CAN_DECISION, MEMBER_CAN_SCORE_SAMPLES,
    MEMBER_CAN_EXPORT, MEMBER_CAN_IMPORT,
    MEMBER_BLOCK_SCALARS, MEMBER_BLOCK_FAMILY, MEMBER_BLOCK_BY_INDEX, MEMBER_IMPORT_DONE,
    member_publish, member_call_from_spans, member_unsupported,
)
from core.member_families import MEMBER_FAMILY_ISOLATION_FOREST
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.shared_slot import SharedSlot, shared_slot_lock, shared_slot_unlock

comptime _CHUNK = 256
comptime _INDEX_SPAN = 16777216
comptime _NAME = "IsolationForest"
comptime _WANT_SCORE_SAMPLES = 0
comptime _WANT_DECISION = 1
comptime _WANT_PREDICT = 2
comptime _WANT_WIDE_BASE = 3
# No MEMBER_CAN_TEST_ROWS: only images with IF_CAP_DEVICE_SCAN test the
# training X, and the host image's queries are not scanned; the caller's
# admission scans. No sample weight, no y (the estimator takes neither).
comptime _CAPS = (MEMBER_CAN_FIT | MEMBER_CAN_PREDICT | MEMBER_CAN_DECISION | MEMBER_CAN_SCORE_SAMPLES
                  | MEMBER_CAN_EXPORT | MEMBER_CAN_IMPORT)
comptime _BLOCKS = 6
comptime _B_OFFSETS = MEMBER_BLOCK_FAMILY
comptime _B_FEATURE = MEMBER_BLOCK_FAMILY + 1
comptime _B_THRESHOLD = MEMBER_BLOCK_FAMILY + 2
comptime _B_LEFT = MEMBER_BLOCK_FAMILY + 3
comptime _B_RIGHT = MEMBER_BLOCK_FAMILY + 4


struct IsolationForestMemberForest(Copyable, Defaultable, Movable):
    """The compact forest a state exported or imported (the slot owns it)."""

    var scalars: List[Float64]
    var offsets: List[Int32]
    var feature: List[Int32]
    var threshold: List[Float32]
    var left: List[Int32]
    var right: List[Int32]

    def __init__(out self):
        self.scalars = List[Float64]()
        self.offsets = List[Int32]()
        self.feature = List[Int32]()
        self.threshold = List[Float32]()
        self.left = List[Int32]()
        self.right = List[Int32]()

    def spans(self) -> IFForestSpans:
        return IFForestSpans(Int(self.offsets.unsafe_ptr()), Int(self.feature.unsafe_ptr()),
                             Int(self.threshold.unsafe_ptr()), Int(self.left.unsafe_ptr()),
                             Int(self.right.unsafe_ptr()), len(self.offsets) - 1, len(self.feature))


struct IsolationForestMemberSlot(Copyable, Defaultable, Movable):
    var generation: Int
    var live: Bool
    var words: IFRunRequest
    """The knobs from NEW; addresses, shapes and the resident form per call."""
    var entry: EntryRef
    """The iforest entry the fit ran through (FREE releases through it)."""
    var fitted: Bool
    var resident: Bool
    var wide: Bool
    var token: Int
    var train: Int
    var n_train: Int
    var n_features: Int
    var offset: Float64
    var max_samples: Int
    var info: List[Float64]
    var scratch: List[Float32]
    var imported: Bool
    """True: the state holds a saved forest (IMPORT), no training view."""
    var forest: IsolationForestMemberForest
    """The compact forest (IMPORT's copy, or EXPORT's read-out; empty until then)."""
    var devices: List[Int]
    """The device ids of the last FIT's `call.devices` (rank order; empty or
    one id: the route's one device)."""

    def __init__(out self):
        self.generation = 0
        self.live = False
        self.words = IFRunRequest()
        self.entry = entry_ref_unresolved()
        self.fitted = False
        self.resident = False
        self.wide = False
        self.token = 0
        self.train = 0
        self.n_train = 0
        self.n_features = 0
        self.offset = -0.5
        self.max_samples = 0
        self.info = List[Float64](length=IF_EXPORT_INFO, fill=0.0)
        self.scratch = List[Float32](length=1, fill=0.0)
        self.imported = False
        self.forest = IsolationForestMemberForest()
        self.devices = List[Int]()


struct IsolationForestMemberStore(Defaultable, Movable):
    var synchronization: List[Int64]
    var chunks: List[List[IsolationForestMemberSlot]]
    var free: List[Int]
    var count: Int

    def __init__(out self):
        # Side-effect-free even when the process slot's initializers race.
        self.synchronization = List[Int64](length=2, fill=0)
        self.chunks = List[List[IsolationForestMemberSlot]]()
        self.free = List[Int]()
        self.count = 0


comptime _STORE = SharedSlot[IsolationForestMemberStore, "MojolearnIsolationForestMemberStoreV2", 2]


def _slot_address(index: Int) raises -> Int:
    # Under the store's lock. A chunk's buffer is allocated once and never
    # grows, so the address stays valid after the lock is released.
    var store = _STORE.get_or_create_ptr()
    return Int(store[].chunks[index // _CHUNK].unsafe_ptr()) + (index % _CHUNK) * size_of[IsolationForestMemberSlot]()


def _mint(words: IFRunRequest) raises -> Int:
    var store = _STORE.get_or_create_ptr()
    var lock = rebind[MutPointer[Int64, MutUntrackedOrigin]](store[].synchronization.unsafe_ptr())
    shared_slot_lock(lock)
    try:
        var index = -1
        if len(store[].free) > 0:
            index = store[].free.pop()
        else:
            if store[].count >= _INDEX_SPAN - 1:
                raise Error("RuntimeError: mojolearn: too many live IsolationForest member states")
            if store[].count == len(store[].chunks) * _CHUNK:
                store[].chunks.append(List[IsolationForestMemberSlot](length=_CHUNK, fill=IsolationForestMemberSlot()))
            index = store[].count
            store[].count += 1
        var slot = MutPointer[IsolationForestMemberSlot, MutUntrackedOrigin](unsafe_from_address=_slot_address(index))
        slot[].generation += 1
        slot[].live = True
        slot[].words = words.copy()
        slot[].entry = entry_ref_unresolved()
        slot[].fitted = False
        slot[].imported = False
        slot[].token = 0
        slot[].train = 0
        return slot[].generation * _INDEX_SPAN + index + 1
    finally:
        shared_slot_unlock(lock)


def _lookup(handle: Int, release: Bool, mut token: Int, mut entry: EntryRef,
            mut dropped: IsolationForestMemberForest) raises -> Int:
    """The slot address of a live handle. A released handle, a handle of
    another store or a made-up integer is refused: never a stale read. On
    release, `token` / `entry` receive the resident forest to drop and
    `dropped` the slot's compact forest (both freed after the lock)."""
    var store = _STORE.get_or_create_ptr()
    var lock = rebind[MutPointer[Int64, MutUntrackedOrigin]](store[].synchronization.unsafe_ptr())
    shared_slot_lock(lock)
    try:
        var index = handle % _INDEX_SPAN - 1
        if handle <= 0 or index < 0 or index >= store[].count:
            raise Error("NotFittedError: this member state was released or never created")
        var address = _slot_address(index)
        var slot = MutPointer[IsolationForestMemberSlot, MutUntrackedOrigin](unsafe_from_address=address)
        if not slot[].live or slot[].generation != handle // _INDEX_SPAN:
            raise Error("NotFittedError: this member state was released or never created")
        if release:
            token = slot[].token if slot[].resident else 0
            entry = slot[].entry
            slot[].live = False
            slot[].fitted = False
            slot[].token = 0
            slot[].train = 0
            slot[].imported = False
            swap(slot[].forest, dropped)
            store[].free.append(index)
        return address
    finally:
        shared_slot_unlock(lock)


def _whole(call: MutPointer[MemberCall, MutUntrackedOrigin]) raises:
    if call[].row_index.rows > 0 or call[].column_index.rows > 0:
        raise Error("TypeError: an " + _NAME + " member reads whole host views (no row or column index)")


def _x(view: MemberMatrix, what: String) raises:
    if not view.dense_host(MEMBER_F32):
        raise Error("TypeError: an " + _NAME + " member reads " + what + " as a dense host float32 view")
    if view.rows <= 0 or view.cols <= 0:
        raise Error("ValueError: an " + _NAME + " member needs at least one row and one feature in " + what)


def _want(slot: MutPointer[IsolationForestMemberSlot, MutUntrackedOrigin], operation: Int, dst: MemberMatrix,
          rows: Int) raises -> Int:
    """The run's `want` for `operation` into `dst` (one value per row)."""
    if dst.rows != rows or dst.cols != 1:
        raise Error("ValueError: the output buffer holds one value per row of X")
    if operation == MEMBER_OP_PREDICT:
        if not dst.dense_host(MEMBER_I32):
            raise Error("TypeError: an " + _NAME + " member writes predict as host int32 labels")
        return _WANT_PREDICT
    var want = _WANT_DECISION if operation == MEMBER_OP_DECISION else _WANT_SCORE_SAMPLES
    if dst.dense_host(MEMBER_F32):
        return want
    if dst.dense_host(MEMBER_F64) and slot[].wide:
        return want + _WANT_WIDE_BASE
    raise Error("TypeError: an " + _NAME + " member writes scores as host float32"
                + (" or float64" if slot[].wide else "") + " (this image)")


def _plan_words(slot: MutPointer[IsolationForestMemberSlot, MutUntrackedOrigin], mut r: IFRunRequest):
    """The slot's device plan into a run's words: several ids run over those
    devices; none or one is the route's one device (`single`)."""
    r.single = True
    var count = len(slot[].devices)
    r.devices = Int(slot[].devices.unsafe_ptr()) if count > 1 else 0
    r.device_count = count if count > 1 else 0


def _run(slot: MutPointer[IsolationForestMemberSlot, MutUntrackedOrigin], entry: EntryRef, query: Int,
         n_query: Int, want: Int, out_f32: Int, out_i32: Int, fresh: Bool) raises:
    var r = slot[].words.copy()
    r.train = slot[].train
    r.n_train = slot[].n_train
    r.n_features = slot[].n_features
    r.query = query
    r.n_query = n_query
    r.want = want
    r.out_f32 = out_f32
    r.out_i32 = out_i32
    r.info = Int(slot[].info.unsafe_ptr())
    r.resident = slot[].resident
    r.token = 0 if fresh else slot[].token
    _plan_words(slot, r)  # the admit's plan (call.devices at FIT); no thread plan, no environment
    _ = if_entry_run(entry, r)
    slot[].offset = slot[].info[0]
    slot[].max_samples = Int(slot[].info[1])
    if slot[].resident:
        slot[].token = Int(slot[].info[3])


def _forest_run(slot: MutPointer[IsolationForestMemberSlot, MutUntrackedOrigin], query: Int, n_query: Int,
                want: Int, out_f32: Int, out_i32: Int) raises:
    """One score of an IMPORTED state (IF_OP_FOREST_RUN): the forest the
    slot owns, uploaded on the first call and resident under its token."""
    var r = IFForestRunRequest(slot[].forest.spans(), query, out_f32, out_i32, Int(slot[].info.unsafe_ptr()),
                               slot[].n_features, n_query, slot[].max_samples, slot[].offset, want,
                               slot[].token if slot[].resident else 0)
    _ = if_entry_forest_run(slot[].entry, r)
    if slot[].resident:
        slot[].token = Int(slot[].info[3])


def _fit(slot: MutPointer[IsolationForestMemberSlot, MutUntrackedOrigin],
         call: MutPointer[MemberCall, MutUntrackedOrigin]) raises:
    _whole(call)
    _x(call[].x, "X")
    if call[].y.rows > 0:
        raise Error("TypeError: an " + _NAME + " member fits no y (an unsupervised estimator)")
    if call[].weights.rows > 0:
        raise Error("TypeError: an " + _NAME + " member takes no sample_weight")
    var entry = call[].route.entry
    var caps = if_entry_caps(entry)
    # A refit of this state drops the forest the previous fit kept.
    if slot[].fitted and slot[].resident and slot[].token != 0:
        if_entry_release(slot[].entry, slot[].token)
    slot[].fitted = False
    slot[].imported = False
    slot[].forest = IsolationForestMemberForest()  # a previous EXPORT's read-out or IMPORT's forest
    slot[].entry = entry
    slot[].resident = (caps & IF_CAP_RESIDENT) != 0
    slot[].wide = slot[].resident and (caps & IF_CAP_SCORES_F64) != 0
    slot[].token = 0
    slot[].train = call[].x.data
    slot[].n_train = call[].x.rows
    slot[].n_features = call[].x.cols
    # The plan the admit read on the public thread (item 28), kept for this
    # state's refits and export; the slot's list keeps its capacity.
    slot[].devices.clear()
    if call[].device_count > 1 and call[].devices != 0:
        var ids = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=call[].devices)
        for k in range(call[].device_count):  # driver-loop: over the plan's ranks (at most 64)
            slot[].devices.append(ids[k])
    var output = call[].fit_output
    if output == 0:
        # The estimator's `fit`: one fresh forest scored on one row, kept.
        _run(slot, entry, call[].x.data, 1, _WANT_SCORE_SAMPLES, Int(slot[].scratch.unsafe_ptr()), 0, True)
    elif output == MEMBER_OP_PREDICT or output == MEMBER_OP_DECISION or output == MEMBER_OP_SCORE_SAMPLES:
        # Its `fit_predict`: one fresh forest that scores X.
        var want = _want(slot, output, call[].dst, call[].x.rows)
        _run(slot, entry, call[].x.data, call[].x.rows, want,
             call[].dst.data if want != _WANT_PREDICT else 0, call[].dst.data if want == _WANT_PREDICT else 0, True)
    else:
        member_unsupported(_NAME, output)
    slot[].fitted = True


def _query(slot: MutPointer[IsolationForestMemberSlot, MutUntrackedOrigin],
           call: MutPointer[MemberCall, MutUntrackedOrigin], operation: Int) raises:
    if not slot[].fitted:
        raise Error("NotFittedError: this " + _NAME + " member is not fitted yet")
    _whole(call)
    _x(call[].x, "X")
    if call[].x.cols != slot[].n_features:
        raise Error("ValueError: X has " + String(call[].x.cols) + " features, but this " + _NAME
                    + " member was fitted with " + String(slot[].n_features))
    var want = _want(slot, operation, call[].dst, call[].x.rows)
    var out_f32 = call[].dst.data if want != _WANT_PREDICT else 0
    var out_i32 = call[].dst.data if want == _WANT_PREDICT else 0
    if slot[].imported:
        _forest_run(slot, call[].x.data, call[].x.rows, want, out_f32, out_i32)
    else:
        _run(slot, slot[].entry, call[].x.data, call[].x.rows, want, out_f32, out_i32, False)


def _read_out(slot: MutPointer[IsolationForestMemberSlot, MutUntrackedOrigin]) raises:
    """A FITTED state's compact forest into the slot's lists, once
    (IF_OP_EXPORT: the sizes, then the copy into lists of exactly that size;
    the resident forest the scores use, refit from the borrowed training
    view only if the store evicted it)."""
    if len(slot[].forest.offsets) > 0:
        return
    var r = slot[].words.copy()
    r.train = slot[].train
    r.n_train = slot[].n_train
    r.n_features = slot[].n_features
    r.info = Int(slot[].info.unsafe_ptr())
    r.token = slot[].token
    _plan_words(slot, r)
    _ = if_entry_export(slot[].entry, r, IFForestSpans(0, 0, 0, 0, 0, 0, 0))
    var trees = Int(slot[].info[4])
    var total = Int(slot[].info[5])
    r.token = Int(slot[].info[3])
    var forest = IsolationForestMemberForest()
    forest.offsets = List[Int32](length=trees + 1, fill=Int32(0))
    forest.feature = List[Int32](length=total, fill=Int32(0))
    forest.threshold = List[Float32](length=total, fill=Float32(0))
    forest.left = List[Int32](length=total, fill=Int32(0))
    forest.right = List[Int32](length=total, fill=Int32(0))
    _ = if_entry_export(slot[].entry, r, forest.spans())
    if slot[].resident:
        slot[].token = Int(slot[].info[3])
    slot[].offset = slot[].info[0]
    slot[].max_samples = Int(slot[].info[1])
    forest.scalars = [slot[].offset, Float64(slot[].max_samples), Float64(slot[].n_features)]
    slot[].forest = forest^


def _view(forest: IsolationForestMemberForest, block: Int) raises -> MemberMatrix:
    if block == MEMBER_BLOCK_SCALARS:
        return MemberMatrix(Int(forest.scalars.unsafe_ptr()), 1, len(forest.scalars), MEMBER_F64)
    if block == _B_OFFSETS:
        return MemberMatrix(Int(forest.offsets.unsafe_ptr()), 1, len(forest.offsets), MEMBER_I32)
    if block == _B_FEATURE:
        return MemberMatrix(Int(forest.feature.unsafe_ptr()), 1, len(forest.feature), MEMBER_I32)
    if block == _B_THRESHOLD:
        return MemberMatrix(Int(forest.threshold.unsafe_ptr()), 1, len(forest.threshold), MEMBER_F32)
    if block == _B_LEFT:
        return MemberMatrix(Int(forest.left.unsafe_ptr()), 1, len(forest.left), MEMBER_I32)
    if block == _B_RIGHT:
        return MemberMatrix(Int(forest.right.unsafe_ptr()), 1, len(forest.right), MEMBER_I32)
    raise Error("member ABI: an " + _NAME + " member has no fitted block " + String(block))


def _export(slot: MutPointer[IsolationForestMemberSlot, MutUntrackedOrigin],
            call: MutPointer[MemberCall, MutUntrackedOrigin]) raises:
    """A view of one fitted block; the slot owns the words until its next
    fit, import or free."""
    if not slot[].fitted:
        raise Error("NotFittedError: this " + _NAME + " member is not fitted yet")
    if not slot[].imported:
        _read_out(slot)
    var block = call[].block
    if (call[].flags & MEMBER_BLOCK_BY_INDEX) != 0:
        if block < 0 or block >= _BLOCKS:
            raise Error("member ABI: an " + _NAME + " member has " + String(_BLOCKS) + " fitted blocks")
        block = MEMBER_BLOCK_SCALARS if block == 0 else MEMBER_BLOCK_FAMILY + block - 1
    call[].dst = _view(slot[].forest, block)
    call[].block = block
    call[].out_cols = _BLOCKS


def _words_in(view: MemberMatrix, dtype: Int, what: String) raises -> Int:
    """The element count of a dense host block of `dtype`."""
    if not view.dense_host(dtype) or view.rows != 1 or view.cols < 1:
        raise Error("member ABI: the " + _NAME + " " + what + " block is one dense host row")
    return view.cols


def _copy_i32(view: MemberMatrix, what: String) raises -> List[Int32]:
    var count = _words_in(view, MEMBER_I32, what)
    var words = List[Int32](length=count, fill=Int32(0))
    unsafe_memcpy(dest=words.unsafe_ptr(), src=MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=view.data),
                  count=count)
    return words^


def _import(slot: MutPointer[IsolationForestMemberSlot, MutUntrackedOrigin],
            call: MutPointer[MemberCall, MutUntrackedOrigin]) raises:
    """Copy one fitted block in; MEMBER_IMPORT_DONE checks the forest is
    whole and marks the state fitted (an imported state: no training view)."""
    var block = call[].block
    if slot[].fitted or not slot[].imported:
        # The first block of an import into a fitted (or imported) state:
        # drop what it left resident, so a new forest whose lists reuse the
        # old addresses can never match the old resident key.
        if slot[].fitted and slot[].resident and slot[].token != 0:
            if_entry_release(slot[].entry, slot[].token)
        slot[].fitted = False
        slot[].imported = True
        slot[].token = 0
        slot[].train = 0
        slot[].n_train = 0
        slot[].forest = IsolationForestMemberForest()
    if block == MEMBER_BLOCK_SCALARS:
        var count = _words_in(call[].x, MEMBER_F64, "scalars")
        if count != 3:
            raise Error("member ABI: the " + _NAME + " scalars block is three float64 words")
        var w = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call[].x.data)
        slot[].forest.scalars = [w[0], w[1], w[2]]
    elif block == _B_OFFSETS:
        slot[].forest.offsets = _copy_i32(call[].x, "offsets")
    elif block == _B_FEATURE:
        slot[].forest.feature = _copy_i32(call[].x, "feature")
    elif block == _B_THRESHOLD:
        var count = _words_in(call[].x, MEMBER_F32, "threshold")
        var words = List[Float32](length=count, fill=Float32(0))
        unsafe_memcpy(dest=words.unsafe_ptr(),
                      src=MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=call[].x.data), count=count)
        slot[].forest.threshold = words^
    elif block == _B_LEFT:
        slot[].forest.left = _copy_i32(call[].x, "left")
    elif block == _B_RIGHT:
        slot[].forest.right = _copy_i32(call[].x, "right")
    elif block != 0 or (call[].flags & MEMBER_IMPORT_DONE) == 0:
        raise Error("member ABI: an " + _NAME + " member has no fitted block " + String(block))
    if (call[].flags & MEMBER_IMPORT_DONE) == 0:
        return
    var total = len(slot[].forest.feature)
    if (len(slot[].forest.scalars) != 3 or len(slot[].forest.offsets) < 2 or total < 1
            or len(slot[].forest.threshold) != total or len(slot[].forest.left) != total
            or len(slot[].forest.right) != total):
        raise Error("member ABI: a fitted " + _NAME + " state needs its scalars and five forest blocks of matching lengths")
    var n_features = Int(slot[].forest.scalars[2])
    if n_features < 1:
        raise Error("member ABI: the " + _NAME + " scalars block names no features")
    var entry = call[].route.entry
    var caps = if_entry_caps(entry)
    slot[].entry = entry
    slot[].resident = (caps & IF_CAP_RESIDENT) != 0
    slot[].wide = slot[].resident and (caps & IF_CAP_SCORES_F64) != 0
    slot[].offset = slot[].forest.scalars[0]
    slot[].max_samples = Int(slot[].forest.scalars[1])
    slot[].n_features = n_features
    slot[].fitted = True


def _new(call: MutPointer[MemberCall, MutUntrackedOrigin]) raises:
    if call[].param_count != 13 or not call[].params:
        raise Error("member ABI: an IsolationForest member takes thirteen parameters")
    var w = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call[].params)
    if w[0] != 0.0:
        raise Error("member ABI: unknown IsolationForest member kind")
    var r = IFRunRequest()
    r.n_estimators = Int(w[1])
    r.max_samples_mode = Int(w[2])
    r.max_samples_int = Int(w[3])
    r.max_samples_frac = w[4]
    r.max_depth = Int(w[5])
    r.max_features_mode = Int(w[6])
    r.max_features_int = Int(w[7])
    r.max_features_frac = w[8]
    r.bootstrap = w[9] != 0.0
    r.random_state = Int(w[10])
    r.contamination_auto = w[11] != 0.0
    r.contamination = w[12]
    if r.n_estimators < 1:
        raise Error("mojolearn IsolationForest: n_estimators must be at least 1, got " + String(r.n_estimators))
    call[].state = _mint(r)


def _isolation_forest_member(operation: Int, address: Int) raises -> Int:
    var call = MutPointer[MemberCall, MutUntrackedOrigin](unsafe_from_address=address)
    if operation == MEMBER_OP_CAPABILITIES:
        return _CAPS
    if operation == MEMBER_OP_NEW:
        _new(call)
        return 0
    if operation == MEMBER_OP_OUTPUTS:
        # One value per row on every published operation, with or without
        # a state (item 10): no width is chosen by the fit.
        call[].out_cols = 1
        call[].classes = 0
        # predict: int32 labels (1 / -1); decision and score_samples: float32
        # (served on every image; float64 only where the image writes it).
        if call[].flags == MEMBER_OP_PREDICT:
            call[].out_dtype = MEMBER_I32
        elif call[].flags == MEMBER_OP_DECISION or call[].flags == MEMBER_OP_SCORE_SAMPLES:
            call[].out_dtype = MEMBER_F32
        return 0
    if operation == MEMBER_OP_FREE:
        var token = 0
        var entry = entry_ref_unresolved()
        var dropped = IsolationForestMemberForest()
        _ = _lookup(call[].state, True, token, entry, dropped)
        call[].state = 0
        if token != 0:
            if_entry_release(entry, token)
        _ = dropped^  # the slot's compact forest, freed after the lock
        return 0
    var token = 0
    var entry = entry_ref_unresolved()
    var unused = IsolationForestMemberForest()
    var slot = MutPointer[IsolationForestMemberSlot, MutUntrackedOrigin](
        unsafe_from_address=_lookup(call[].state, False, token, entry, unused))
    if operation == MEMBER_OP_FIT:
        _fit(slot, call)
    elif operation == MEMBER_OP_PREDICT or operation == MEMBER_OP_DECISION or operation == MEMBER_OP_SCORE_SAMPLES:
        _query(slot, call, operation)
    elif operation == MEMBER_OP_EXPORT:
        _export(slot, call)
    elif operation == MEMBER_OP_IMPORT:
        _import(slot, call)
    else:
        member_unsupported(_NAME, operation)
    return 0


def isolation_forest_member_v2(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
    result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """The member entry of IsolationForest (schema `MEMBER_ABI`)."""
    try:
        var word = _isolation_forest_member(operation, member_call_from_spans(operation, aa, na, np, nf))
        if result:
            MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = word
        return 0
    except error:
        native_descriptor_report(error, context, sink)
        return 1


def isolation_forest_member_publish(mode: Int, vendor: Int, publisher: Int) raises:
    """Publish this image's IsolationForest member entry. One line in each
    svm image's module initialization, after its iforest entry."""
    member_publish[MEMBER_FAMILY_ISOLATION_FOREST](mode, vendor, publisher, isolation_forest_member_v2)
