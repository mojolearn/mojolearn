# SPDX-License-Identifier: Apache-2.0
"""KNeighborsClassifier and KNeighborsRegressor as MEMBERS
(core/member_abi.mojo, MEMBER-ABI-V2.md).

One entry, `neighbors_member_v2`, published by each image that publishes the
neighbors operations table (core/neighbors_entry_table.mojo): the base
binding `_mojolearn` (the device column) and `_mojolearn_core_host` (the CPU
column), beside that table. It runs the family's existing typed k-NN RUN
(neighbors/neighbors_knn_run.mojo `knn_classify_run` / `knn_regress_run`,
core/neighbors_entry_table.mojo `nb_index_prepare_call` /
`nb_index_release_call`) through the ONE route slot it reads: `entry` = the
neighbors operations entry of the image the route resolved to
(`route.vendor` 0: the CPU column, no resident index; otherwise the device
column, whose resident copy of the index is minted at the state's first
query and reused by every later one, as the estimator's own door does). No
kernel is touched. NearestNeighbors and RadiusNeighbors have no predict and
are not members (kneighbors graphs are not covered by the ABI).

MEMBER_OP_NEW parameters (Float64 words, count 6): [0] kind (0
KNeighborsClassifier, 1 KNeighborsRegressor), [1] n_neighbors, [2] metric
code (the brute-force distance codes of neighbors/neighbors_knn_run.mojo:
0 sqeuclidean, 1 euclidean, 2 cosine, 3 manhattan, 6 inner_product,
7 chebyshev, 8 canberra, 9 minkowski, 10 correlation, 14 braycurtis,
15 jensenshannon), [3] metric argument (Minkowski's p; 2.0 otherwise),
[4] weights (0 uniform, 1 distance), [5] query_tile.

Views (all host, dense row-major; a device-home view, a row or column index
and a sample weight are refused in words): X float32 rows x features.
Classifier: y = class CODES 0 .. classes-1 as one int32 column
(`MEMBER_I32`), `call.classes` the class count at fit / import; predict
writes int32 codes (rows x 1); predict_proba writes float32 rows x classes
(a class absent from the fitted rows has probability 0). Regressor: y one
float32 column; predict writes float32 rows x 1. One output only (no
MEMBER_CAN_MULTI_OUTPUT). The caller's UNPACK admits X and y (finite,
shapes) as for every member; the member tests no word again. OUTPUTS with
no state answers from the NEW words: predict_proba `call.classes` columns
(the classifier), predict `max(1, y.cols)`; `out_dtype` int32 for the
classifier's predict, float32 otherwise. MEMBER_CAN_TEST_ROWS is clear (no
finite test in the RUN). Score is not
published: the caller predicts and runs its own metric.

Fitted-state blocks (MEMBER_OP_EXPORT / MEMBER_OP_IMPORT), in index order:
`MEMBER_BLOCK_FAMILY` (64) float32 rows x features, the fitted points;
`MEMBER_BLOCK_FAMILY + 1` (65) 1 x rows, the targets (int32 codes for the
classifier, float32 for the regressor). An import needs both;
`MEMBER_IMPORT_DONE` reads `call.classes` for the classifier.

The fitted state (the class set present in the targets, the resident index
handle, the per-state query buffers) is OWNED here, in a process-wide store
of fixed chunks (the core/linear_member.mojo form): a slot never moves once
minted; the store's lock covers only mint, free and the handle lookup.

LIFETIME: a fit BORROWS the caller's X and y views, exactly as the
estimator's own fit borrows the user's array (k-NN is a lazy learner: the
fitted points ARE X). The caller keeps the words of X and y alive and
unchanged until this state's next fit, import or free; no copy is made per
fit (a copy per fit is a copy per fold inside cross-validation). The first
MEMBER_OP_EXPORT of a block copies that block once into the slot, and the
state reads the slot's copy from then on, so an exported view lives until
the state's next fit, import or free whatever the caller does with X. An
import copies its blocks in. Parallel: the
class-presence scan of the codes and the probability scatter split their
rows across the host's cores; the search and vote are the device or host
entry's.
"""
from std.sys.info import size_of
from std.memory import unsafe_memcpy
from core.device_route import DeviceRoute
from core.entry_table import EntryRef, entry_ref_unresolved, entry_ref_same
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk
from core.member_abi import (
    MemberCall, MemberMatrix, MEMBER_F32, MEMBER_I32,
    MEMBER_OP_CAPABILITIES, MEMBER_OP_NEW, MEMBER_OP_FREE, MEMBER_OP_FIT, MEMBER_OP_PREDICT,
    MEMBER_OP_PREDICT_PROBA, MEMBER_OP_OUTPUTS, MEMBER_OP_EXPORT, MEMBER_OP_IMPORT,
    MEMBER_CAN_FIT, MEMBER_CAN_PREDICT, MEMBER_CAN_PREDICT_PROBA, MEMBER_CAN_EXPORT, MEMBER_CAN_IMPORT,
    MEMBER_CAN_CLASSIFIER,
    MEMBER_BLOCK_FAMILY, MEMBER_BLOCK_BY_INDEX, MEMBER_IMPORT_DONE,
    member_publish, member_call_from_spans, member_unsupported,
)
from core.member_families import MEMBER_FAMILY_NEIGHBORS
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.neighbors_entry_table import nb_index_prepare_call, nb_index_release_call
from core.shared_slot import SharedSlot, shared_slot_lock, shared_slot_unlock
from neighbors.neighbors_knn_run import (
    KnnDistWords, KnnQueryRequest, KnnQueryOutcome, knn_classify_run, knn_regress_run,
    knn_check_metric_input, knn_check_width,
)

comptime _CHUNK = 256
comptime _INDEX_SPAN = 16777216
comptime NEIGHBORS_BLOCK_POINTS = MEMBER_BLOCK_FAMILY
comptime NEIGHBORS_BLOCK_TARGETS = MEMBER_BLOCK_FAMILY + 1
comptime _I32P = MutPointer[Int32, MutUntrackedOrigin]
comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _U8P = MutPointer[UInt8, MutUntrackedOrigin]


struct NeighborsMemberSlot(Copyable, Movable, Defaultable):
    var generation: Int
    var live: Bool
    var regressor: Bool
    var k: Int
    var metric: Int
    var metric_arg: Float64
    var weights: Int
    var query_tile: Int
    var fitted: Bool
    var rows: Int
    var features: Int
    var classes: Int
    # The words every query reads: the caller's borrowed views after a fit,
    # the slot's own lists below after an import or an export.
    var points_addr: Int
    var targets_addr: Int
    # Slot-owned copies: filled by an import, or by the first export of a
    # borrowed block; empty while the state borrows.
    var points: List[Float32]
    var codes: List[Int32]
    var targets: List[Float32]
    # The class codes present in the fitted rows, ascending (the entry's
    # vote sizes its class set by them), and their count as the entry's
    # one-output class-count list (allocated at fit, never per query).
    var present: List[Int32]
    var counts: List[Int]
    # Per-state buffers a query reuses: the entry's class set, the one-word
    # sentinel of the output it does not write, the probabilities of the
    # present classes when some class is absent.
    var uniq: List[Int32]
    var one_i32: List[Int32]
    var one_f32: List[Float32]
    var proba: List[Float32]
    # The resident copy of `points` on the route's device and the entry
    # that minted it (released through its minter).
    var handle: Int
    var handle_entry: EntryRef

    def __init__(out self):
        self.generation = 0
        self.live = False
        self.regressor = False
        self.k = 5
        self.metric = 1
        self.metric_arg = 2.0
        self.weights = 0
        self.query_tile = 0
        self.fitted = False
        self.rows = 0
        self.features = 0
        self.classes = 0
        self.points_addr = 0
        self.targets_addr = 0
        self.points = List[Float32]()
        self.codes = List[Int32]()
        self.targets = List[Float32]()
        self.present = List[Int32]()
        self.counts = List[Int]()
        self.uniq = List[Int32]()
        self.one_i32 = List[Int32](length=1, fill=Int32(0))
        self.one_f32 = List[Float32](length=1, fill=Float32(0.0))
        self.proba = List[Float32]()
        self.handle = 0
        self.handle_entry = entry_ref_unresolved()


struct NeighborsMemberStore(Defaultable, Movable):
    var synchronization: List[Int64]
    var chunks: List[List[NeighborsMemberSlot]]
    var free: List[Int]
    var count: Int

    def __init__(out self):
        # Side-effect-free even when the process slot's initializers race.
        self.synchronization = List[Int64](length=2, fill=0)
        self.chunks = List[List[NeighborsMemberSlot]]()
        self.free = List[Int]()
        self.count = 0


comptime _STORE = SharedSlot[NeighborsMemberStore, "MojolearnNeighborsMemberStoreV2", 2]
comptime _SlotPtr = MutPointer[NeighborsMemberSlot, MutUntrackedOrigin]
comptime _CallPtr = MutPointer[MemberCall, MutUntrackedOrigin]


def _slot_address(index: Int) raises -> Int:
    # Under the store's lock; a chunk's buffer never grows.
    var store = _STORE.get_or_create_ptr()
    return Int(store[].chunks[index // _CHUNK].unsafe_ptr()) + (index % _CHUNK) * size_of[NeighborsMemberSlot]()


def _mint(words: MutPointer[Float64, MutUntrackedOrigin]) raises -> Int:
    var store = _STORE.get_or_create_ptr()
    var lock = rebind[MutPointer[Int64, MutUntrackedOrigin]](store[].synchronization.unsafe_ptr())
    shared_slot_lock(lock)
    try:
        var index = -1
        if len(store[].free) > 0:
            index = store[].free.pop()
        else:
            if store[].count >= _INDEX_SPAN - 1:
                raise Error("RuntimeError: mojolearn: too many live k-NN member states")
            if store[].count == len(store[].chunks) * _CHUNK:
                store[].chunks.append(List[NeighborsMemberSlot](length=_CHUNK, fill=NeighborsMemberSlot()))
            index = store[].count
            store[].count += 1
        var slot = _SlotPtr(unsafe_from_address=_slot_address(index))
        slot[].generation += 1
        slot[].live = True
        slot[].regressor = words[0] == 1.0
        slot[].k = Int(words[1])
        slot[].metric = Int(words[2])
        slot[].metric_arg = words[3]
        slot[].weights = Int(words[4])
        slot[].query_tile = Int(words[5])
        slot[].fitted = False
        slot[].handle = 0
        slot[].handle_entry = entry_ref_unresolved()
        return slot[].generation * _INDEX_SPAN + index + 1
    finally:
        shared_slot_unlock(lock)


def _lookup(handle: Int, release: Bool) raises -> Int:
    """The slot address of a live handle; a released, foreign or made-up
    handle is refused (never a stale read)."""
    var store = _STORE.get_or_create_ptr()
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
            slot[].fitted = False
            slot[].points_addr = 0
            slot[].targets_addr = 0
            slot[].points = List[Float32]()
            slot[].codes = List[Int32]()
            slot[].targets = List[Float32]()
            slot[].present = List[Int32]()
            slot[].counts = List[Int]()
            slot[].uniq = List[Int32]()
            slot[].proba = List[Float32]()
            store[].free.append(index)
        return address
    finally:
        shared_slot_unlock(lock)


def _name(slot: _SlotPtr) -> String:
    return String("KNeighborsRegressor") if slot[].regressor else String("KNeighborsClassifier")


def _whole(call: _CallPtr, estimator: String) raises:
    if call[].row_index.rows > 0 or call[].column_index.rows > 0:
        raise Error("TypeError: a " + estimator + " member reads whole host views (no row or column index)")


def _dense(view: MemberMatrix, dtype: Int, what: String, estimator: String) raises:
    if not view.dense_host(dtype):
        raise Error("TypeError: a " + estimator + " member reads " + what + " as a dense host "
            + (String("int32") if dtype == MEMBER_I32 else String("float32")) + " view")


def _release_index(slot: _SlotPtr):
    """Drop the resident copy of the points through the entry that minted
    it. Quiet: a handle the image already released is nothing to free."""
    if slot[].handle == 0:
        return
    var handle = slot[].handle
    slot[].handle = 0
    try:
        nb_index_release_call(slot[].handle_entry.call, handle)
    except:
        pass
    slot[].handle_entry = entry_ref_unresolved()


def _present_codes(slot: _SlotPtr) raises:
    """The class codes present in the fitted codes (ascending), with a code
    outside 0 .. classes-1 refused. Parallel: each task marks the codes of
    its rows in its own presence row; the merge reads tasks x classes bytes."""
    var rows = slot[].rows
    var classes = slot[].classes
    var tasks = max(host_predict_task_count_work(rows, rows), 1)
    var chunk = host_predict_chunk(rows, tasks)
    var marks = List[UInt8](length=tasks * classes, fill=UInt8(0))
    var bad = List[UInt8](length=tasks, fill=UInt8(0))
    var marks_ptr = _U8P(unsafe_from_address=Int(marks.unsafe_ptr()))
    var bad_ptr = _U8P(unsafe_from_address=Int(bad.unsafe_ptr()))
    var codes_ptr = _I32P(unsafe_from_address=slot[].targets_addr)

    def _scan(t: Int) {imm marks_ptr, imm bad_ptr, imm codes_ptr, imm rows, imm classes, imm chunk}:
        var lo = t * chunk
        var hi = min(lo + chunk, rows)
        for i in range(lo, hi):  # member fit: this task's rows, one presence mark each
            var v = Int(codes_ptr.unsafe_load(i))
            if v < 0 or v >= classes:
                bad_ptr.unsafe_store(t, UInt8(1))
                return
            marks_ptr.unsafe_store(t * classes + v, UInt8(1))

    if tasks <= 1:
        _scan(0)
    else:
        host_parallelize(_scan, tasks)
    for t in range(tasks):  # driver-loop: over the tasks' refusal words
        if bad[t] != 0:
            raise Error("ValueError: a k-NN classifier member reads class codes 0 .. "
                + String(classes - 1) + " in y")
    var present = List[Int32]()
    for c in range(classes):  # driver-loop: over classes, merging the tasks' presence rows
        for t in range(tasks):  # driver-loop: over tasks (cores)
            if marks[t * classes + c] != 0:
                present.append(Int32(c))
                break
    _ = marks^
    _ = bad^
    slot[].present = present^
    var counts: List[Int] = [len(slot[].present)]
    slot[].counts = counts^
    slot[].uniq = List[Int32](length=len(slot[].present), fill=Int32(0))


def _fit(slot: _SlotPtr, call: _CallPtr) raises:
    var estimator = _name(slot)
    _whole(call, estimator)
    if call[].fit_output != 0:
        member_unsupported(estimator, call[].fit_output)
    if call[].weights.rows > 0:
        raise Error("TypeError: a " + estimator + " member fits no sample_weight")
    _dense(call[].x, MEMBER_F32, "X", estimator)
    var rows = call[].x.rows
    var cols = call[].x.cols
    if call[].y.cols != 1:
        raise Error("ValueError: a " + estimator + " member fits one target column")
    if call[].y.rows != rows:
        raise Error("ValueError: X and y have different numbers of rows")
    if slot[].regressor:
        _dense(call[].y, MEMBER_F32, "y", estimator)
    else:
        _dense(call[].y, MEMBER_I32, "y (class codes)", estimator)
        if call[].classes < 1:
            raise Error("ValueError: a " + estimator + " member needs the class count at fit")
    knn_check_metric_input(estimator, slot[].metric, call[].x.data, rows * cols)
    _release_index(slot)
    slot[].fitted = False
    # Borrow, never copy (LIFETIME in the module docstring). A copy a
    # previous import or export left in the slot is dropped.
    slot[].points = List[Float32]()
    slot[].codes = List[Int32]()
    slot[].targets = List[Float32]()
    slot[].points_addr = call[].x.data
    slot[].targets_addr = call[].y.data
    slot[].rows = rows
    slot[].features = cols
    if not slot[].regressor:
        slot[].classes = call[].classes
        _present_codes(slot)
    slot[].fitted = True


def _request(slot: _SlotPtr, call: _CallPtr) raises -> KnnQueryRequest:
    """One query's typed request, with the estimator's predict checks (its
    width and k wording) and the resident index of a device route."""
    var estimator = _name(slot)
    if not slot[].fitted:
        raise Error("NotFittedError: mojolearn: call fit before predict")
    _whole(call, estimator)
    _dense(call[].x, MEMBER_F32, "X", estimator)
    knn_check_width(call[].x.cols, slot[].features)
    if slot[].k < 1 or slot[].k > slot[].rows:
        raise Error("mojolearn: n_neighbors must be in [1, " + String(slot[].rows) + "], got " + String(slot[].k))
    knn_check_metric_input(estimator, slot[].metric, call[].x.data, call[].x.rows * call[].x.cols)
    var request = KnnQueryRequest(slot[].points_addr, call[].x.data, slot[].rows, call[].x.rows,
        slot[].features, slot[].k, slot[].query_tile,
        KnnDistWords(slot[].metric, slot[].metric_arg, slot[].weights))
    var route = call[].route
    if route.vendor != 0:
        # The index stays on the device: minted at this state's first query
        # (the estimator's own rule), reused while the state keeps its fit.
        if slot[].handle != 0 and not entry_ref_same(slot[].handle_entry, route.entry):
            _release_index(slot)
        if slot[].handle == 0:
            slot[].handle = nb_index_prepare_call(route.entry.call, slot[].points_addr, 0,
                slot[].rows, slot[].features)
            slot[].handle_entry = route.entry
        request.resident = True
        request.handle = slot[].handle
    return request^


def _predict(slot: _SlotPtr, call: _CallPtr) raises:
    var estimator = _name(slot)
    var request = _request(slot, call)
    var rows = call[].x.rows
    var entry = call[].route.entry.call
    var outcome = KnnQueryOutcome()
    if slot[].regressor:
        _dense(call[].dst, MEMBER_F32, "the prediction buffer", estimator)
        if call[].dst.rows != rows or call[].dst.cols != 1:
            raise Error("ValueError: the prediction buffer holds one value per row of X")
        outcome = knn_regress_run(entry, request, slot[].targets_addr, call[].dst.data, 1)
    else:
        _dense(call[].dst, MEMBER_I32, "the prediction buffer (class codes)", estimator)
        if call[].dst.rows != rows or call[].dst.cols != 1:
            raise Error("ValueError: the prediction buffer holds one class code per row of X")
        outcome = knn_classify_run(entry, request, slot[].targets_addr, call[].dst.data,
            Int(slot[].one_f32.unsafe_ptr()), Int(slot[].uniq.unsafe_ptr()), False, slot[].counts)
    _ = outcome^


def _predict_proba(slot: _SlotPtr, call: _CallPtr) raises:
    var estimator = _name(slot)
    if slot[].regressor:
        member_unsupported(estimator, MEMBER_OP_PREDICT_PROBA)
    var request = _request(slot, call)
    var rows = call[].x.rows
    var classes = slot[].classes
    var present = len(slot[].present)
    _dense(call[].dst, MEMBER_F32, "the probability buffer", estimator)
    if call[].dst.rows != rows or call[].dst.cols != classes:
        raise Error("ValueError: the probability buffer holds one row of " + String(classes)
            + " class probabilities per row of X")
    var target = call[].dst.data
    if present != classes:
        # Some class is absent from the fitted rows: the entry writes the
        # present classes' columns, scattered below into the class columns.
        if len(slot[].proba) < rows * present:
            slot[].proba = List[Float32](length=rows * present, fill=Float32(0.0))
        target = Int(slot[].proba.unsafe_ptr())
    var outcome = knn_classify_run(call[].route.entry.call, request, slot[].targets_addr,
        Int(slot[].one_i32.unsafe_ptr()), target, Int(slot[].uniq.unsafe_ptr()), True, slot[].counts)
    _ = outcome^
    if present == classes:
        return
    var dst_ptr = _F32P(unsafe_from_address=call[].dst.data)
    var src_ptr = _F32P(unsafe_from_address=target)
    var present_ptr = _I32P(unsafe_from_address=Int(slot[].present.unsafe_ptr()))
    var tasks = max(host_predict_task_count_work(rows, rows * classes), 1)
    var chunk = host_predict_chunk(rows, tasks)

    def _scatter(t: Int) {imm dst_ptr, imm src_ptr, imm present_ptr, imm rows, imm classes, imm present, imm chunk}:
        var lo = t * chunk
        var hi = min(lo + chunk, rows)
        for r in range(lo, hi):  # member predict_proba: this task's rows
            for c in range(classes):
                dst_ptr.unsafe_store(r * classes + c, Float32(0.0))
            for j in range(present):
                dst_ptr.unsafe_store(r * classes + Int(present_ptr.unsafe_load(j)),
                    src_ptr.unsafe_load(r * present + j))

    if tasks <= 1:
        _scatter(0)
    else:
        host_parallelize(_scatter, tasks)


def _own_points(slot: _SlotPtr):
    """Copy borrowed points into the slot once (first export); the state
    reads its own copy from then on. A resident device index is a copy of
    the same words and stays valid."""
    var count = slot[].rows * slot[].features
    if slot[].points_addr == Int(slot[].points.unsafe_ptr()) and len(slot[].points) == count:
        return
    var owned = List[Float32](length=count, fill=Float32(0.0))
    unsafe_memcpy(dest=owned.unsafe_ptr(), src=_F32P(unsafe_from_address=slot[].points_addr), count=count)
    slot[].points = owned^
    slot[].points_addr = Int(slot[].points.unsafe_ptr())


def _own_targets(slot: _SlotPtr):
    """Copy borrowed targets (float32) or class codes (int32) into the slot
    once (first export)."""
    var count = slot[].rows
    if slot[].regressor:
        if slot[].targets_addr == Int(slot[].targets.unsafe_ptr()) and len(slot[].targets) == count:
            return
        var owned = List[Float32](length=count, fill=Float32(0.0))
        unsafe_memcpy(dest=owned.unsafe_ptr(), src=_F32P(unsafe_from_address=slot[].targets_addr), count=count)
        slot[].targets = owned^
        slot[].targets_addr = Int(slot[].targets.unsafe_ptr())
    else:
        if slot[].targets_addr == Int(slot[].codes.unsafe_ptr()) and len(slot[].codes) == count:
            return
        var owned = List[Int32](length=count, fill=Int32(0))
        unsafe_memcpy(dest=owned.unsafe_ptr(), src=_I32P(unsafe_from_address=slot[].targets_addr), count=count)
        slot[].codes = owned^
        slot[].targets_addr = Int(slot[].codes.unsafe_ptr())


def _export(slot: _SlotPtr, call: _CallPtr) raises:
    """A view of one fitted block. A block the state still borrows from the
    caller's fit views is copied into the slot here, once; the slot owns the
    words until this state's next fit, import or free."""
    if not slot[].fitted:
        raise Error("NotFittedError: this " + _name(slot) + " member is not fitted yet")
    var block = call[].block
    if (call[].flags & MEMBER_BLOCK_BY_INDEX) != 0:
        if block < 0 or block > 1:
            raise Error("member ABI: a k-NN member has two fitted blocks")
        block = NEIGHBORS_BLOCK_POINTS + block
    if block == NEIGHBORS_BLOCK_POINTS:
        _own_points(slot)
        call[].dst = MemberMatrix(Int(slot[].points.unsafe_ptr()), slot[].rows, slot[].features)
    elif block == NEIGHBORS_BLOCK_TARGETS:
        _own_targets(slot)
        if slot[].regressor:
            call[].dst = MemberMatrix(Int(slot[].targets.unsafe_ptr()), 1, slot[].rows)
        else:
            call[].dst = MemberMatrix(Int(slot[].codes.unsafe_ptr()), 1, slot[].rows, MEMBER_I32)
    else:
        raise Error("member ABI: a k-NN member has no fitted block " + String(block))
    call[].block = block
    call[].out_cols = 2
    call[].classes = slot[].classes


def _import(slot: _SlotPtr, call: _CallPtr) raises:
    """Copy one fitted block in; MEMBER_IMPORT_DONE completes the state."""
    var estimator = _name(slot)
    if call[].block == NEIGHBORS_BLOCK_POINTS:
        _dense(call[].x, MEMBER_F32, "the fitted points", estimator)
        var count = call[].x.rows * call[].x.cols
        _release_index(slot)
        slot[].fitted = False
        slot[].points_addr = 0
        slot[].points = List[Float32](length=count, fill=Float32(0.0))
        unsafe_memcpy(dest=slot[].points.unsafe_ptr(), src=_F32P(unsafe_from_address=call[].x.data), count=count)
        slot[].rows = call[].x.rows
        slot[].features = call[].x.cols
    elif call[].block == NEIGHBORS_BLOCK_TARGETS:
        var count = call[].x.rows * call[].x.cols
        slot[].fitted = False
        slot[].targets_addr = 0
        if slot[].regressor:
            _dense(call[].x, MEMBER_F32, "the fitted targets", estimator)
            slot[].targets = List[Float32](length=count, fill=Float32(0.0))
            unsafe_memcpy(dest=slot[].targets.unsafe_ptr(), src=_F32P(unsafe_from_address=call[].x.data),
                count=count)
        else:
            _dense(call[].x, MEMBER_I32, "the fitted class codes", estimator)
            slot[].codes = List[Int32](length=count, fill=Int32(0))
            unsafe_memcpy(dest=slot[].codes.unsafe_ptr(), src=_I32P(unsafe_from_address=call[].x.data),
                count=count)
    elif call[].block != 0 or (call[].flags & MEMBER_IMPORT_DONE) == 0:
        raise Error("member ABI: a k-NN member has no fitted block " + String(call[].block))
    if (call[].flags & MEMBER_IMPORT_DONE) == 0:
        return
    var targets = len(slot[].targets) if slot[].regressor else len(slot[].codes)
    if slot[].rows == 0 or len(slot[].points) != slot[].rows * slot[].features or targets != slot[].rows:
        raise Error("member ABI: a fitted k-NN state needs its points and one target per point")
    slot[].points_addr = Int(slot[].points.unsafe_ptr())
    slot[].targets_addr = Int(slot[].targets.unsafe_ptr()) if slot[].regressor else Int(slot[].codes.unsafe_ptr())
    if not slot[].regressor:
        if call[].classes < 1:
            raise Error("member ABI: a k-NN classifier import needs the class count")
        slot[].classes = call[].classes
        _present_codes(slot)
    slot[].fitted = True


def _out_dtype(regressor: Bool, asked: Int) -> Int:
    """The element type of `dst` for the asked operation: the classifier's
    predict writes int32 codes, every other output float32."""
    if not regressor and asked == MEMBER_OP_PREDICT:
        return MEMBER_I32
    return MEMBER_F32


# MEMBER_CAN_TEST_ROWS: none: bit clear. The k-NN RUN makes no finite test of
# X or y: `knn_check_metric_input` (neighbors/neighbors_knn_run.mojo:285)
# refuses only negative X for jensenshannon, and the search entries return
# no status. The caller's admission makes the finite scan.


def _neighbors_member(operation: Int, address: Int) raises -> Int:
    var call = _CallPtr(unsafe_from_address=address)
    comptime _REGRESSOR = MEMBER_CAN_FIT | MEMBER_CAN_PREDICT | MEMBER_CAN_EXPORT | MEMBER_CAN_IMPORT
    comptime _CLASSIFIER = _REGRESSOR | MEMBER_CAN_PREDICT_PROBA | MEMBER_CAN_CLASSIFIER
    if operation == MEMBER_OP_CAPABILITIES:
        if call[].state == 0:
            return _CLASSIFIER
        var kind = _SlotPtr(unsafe_from_address=_lookup(call[].state, False))
        return _REGRESSOR if kind[].regressor else _CLASSIFIER
    if operation == MEMBER_OP_NEW:
        if call[].param_count != 6 or not call[].params:
            raise Error("member ABI: a k-NN member takes six parameters (kind, n_neighbors, metric, "
                + "metric argument, weights, query_tile)")
        var words = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call[].params)
        if words[0] != 0.0 and words[0] != 1.0:
            raise Error("member ABI: unknown k-NN member kind")
        if words[4] != 0.0 and words[4] != 1.0:
            raise Error("member ABI: unknown k-NN weights code")
        call[].state = _mint(words)
        return 0
    if operation == MEMBER_OP_FREE:
        var held = _SlotPtr(unsafe_from_address=_lookup(call[].state, False))
        _release_index(held)
        _ = _lookup(call[].state, True)
        call[].state = 0
        return 0
    if operation == MEMBER_OP_OUTPUTS and call[].state == 0:
        # No state (member_outputs): answer from the NEW words, as a state made
        # from them and fitted on `x.cols` features, `y.cols` targets would.
        if call[].param_count != 6 or not call[].params:
            raise Error("member ABI: a k-NN member takes six parameters (kind, n_neighbors, metric, "
                + "metric argument, weights, query_tile)")
        var words = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call[].params)
        if words[0] == 1.0:
            if call[].flags == MEMBER_OP_PREDICT_PROBA:
                raise Error("member ABI: KNeighborsRegressor has no predict_proba")
            call[].classes = 0
        call[].out_cols = call[].classes if call[].flags == MEMBER_OP_PREDICT_PROBA else max(1, call[].y.cols)
        call[].out_dtype = _out_dtype(words[0] == 1.0, call[].flags)
        return 0
    var slot = _SlotPtr(unsafe_from_address=_lookup(call[].state, False))
    if operation == MEMBER_OP_OUTPUTS:
        call[].classes = 0 if slot[].regressor else (slot[].classes if slot[].fitted else call[].classes)
        call[].out_cols = call[].classes if call[].flags == MEMBER_OP_PREDICT_PROBA else 1
        call[].out_dtype = _out_dtype(slot[].regressor, call[].flags)
        return 0
    if operation == MEMBER_OP_FIT:
        _fit(slot, call)
    elif operation == MEMBER_OP_PREDICT:
        _predict(slot, call)
    elif operation == MEMBER_OP_PREDICT_PROBA:
        _predict_proba(slot, call)
    elif operation == MEMBER_OP_EXPORT:
        _export(slot, call)
    elif operation == MEMBER_OP_IMPORT:
        _import(slot, call)
    else:
        member_unsupported(_name(slot), operation)
    return 0


def neighbors_member_v2(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
    result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """The member entry of KNeighborsClassifier and KNeighborsRegressor
    (schema `MEMBER_ABI`)."""
    try:
        var word = _neighbors_member(operation, member_call_from_spans(operation, aa, na, np, nf))
        if result:
            MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = word
        return 0
    except error:
        native_descriptor_report(error, context, sink)
        return 1


def neighbors_member_publish(mode: Int, vendor: Int, publisher: Int) raises:
    """Publish this image's k-NN member entry. One line in the image's
    module initialization, after its neighbors operations table."""
    member_publish[MEMBER_FAMILY_NEIGHBORS](mode, vendor, publisher, neighbors_member_v2)
