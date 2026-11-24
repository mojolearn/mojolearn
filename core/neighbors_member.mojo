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
kernel is touched.

Lane mf-member8 (2026-11-23): NearestNeighbors (kind 2) and RadiusNeighbors
(kind 3) are FIT + EXPORT members (no predict: their kneighbors / radius
graphs are not member operations). Their fit is the estimator's own: the
fitted points ARE X (bindings/neighbors_knn_native_door.mojo `_nn_fit`: the
metric input check, nothing else; providers/python/radius_public_lifecycle_
pyapi_door.mojo `radius_fit`: X admitted finite by the caller, nothing else),
borrowed as the k-NN kinds borrow it; `y` is not read (scikit-learn's
`fit(X, y=None)` ignores it). One block, the points. A caller that fits one
as a final estimator (RFE with no elimination round, scikit-learn's
`_fit` with n_features_to_select >= n_features) PACKs `_index`,
`n_samples_fit_`, `n_features_in_` as the estimator's own fit writes them.

MEMBER_OP_NEW parameters (Float64 words, count 6): [0] kind (0
KNeighborsClassifier, 1 KNeighborsRegressor, 2 NearestNeighbors, 3
RadiusNeighbors; kinds 2 and 3 read only [2], -1 for no metric input check),
[1] n_neighbors, [2] metric
code (the brute-force distance codes of neighbors/neighbors_knn_run.mojo:
0 sqeuclidean, 1 euclidean, 2 cosine, 3 manhattan, 6 inner_product,
7 chebyshev, 8 canberra, 9 minkowski, 10 correlation, 14 braycurtis,
15 jensenshannon), [3] metric argument (Minkowski's p; 2.0 otherwise),
[4] weights (0 uniform, 1 distance), [5] query_tile.

Views (all host, dense row-major; a device-home view, a column index and a
sample weight are refused in words): X float32 rows x features. A host int32
/ int64 ROW INDEX (MEMBER_CAN_ROW_INDEX, the k-NN kinds; lane
vdf-scope-cluster2a) selects the fold's rows of X and y: on the device column
inside a resident scope with X pinned (MEMBER_CAN_SCOPE) X stays resident once
per search and the fit's index and the queries are the fold's rows gathered
on the device (THE FOLD INDEX, NB_OP_INDEX_PREPARE_FOLD; THE FOLD QUERIES,
the resident entries' NbFold); otherwise the rows are gathered on the host
once per call. y's rows are gathered into the slot either way.
Classifier: y = class CODES 0 .. classes-1 as one int32 column
(`MEMBER_I32`), `call.classes` the class count at fit / import; predict
writes int32 codes (rows x 1); predict_proba writes float32 rows x classes
(a class absent from the fitted rows has probability 0). Regressor: y one
float32 column; predict writes float32 rows x 1. THE 2-D y (lane w3f3-k,
MEMBER_CAN_MULTI_OUTPUT, core/member_abi.mojo layout 4), as the estimators'
own multi-output fit and predict (scikit-learn's KNeighbors* take a 2-D y):
y rows x outputs (the classifier's codes per output, `call.class_counts`
each output's count); predict writes rows x outputs, predict_proba rows x
the counts' sum (each output's columns in turn); the same RUN calls with
`outputs` target columns. The caller's UNPACK admits X and y (finite,
shapes) as for every member; the member tests no word again. OUTPUTS with
no state answers from the NEW words: predict_proba `call.classes` columns
(the classifier), predict `max(1, y.cols)`; `out_dtype` int32 for the
classifier's predict, float32 otherwise. MEMBER_CAN_TEST_ROWS is clear (no
finite test in the RUN). Score is not
published: the caller predicts and runs its own metric.

Fitted-state blocks (MEMBER_OP_EXPORT / MEMBER_OP_IMPORT), in index order:
`MEMBER_BLOCK_FAMILY` (64) float32 rows x features, the fitted points;
`MEMBER_BLOCK_FAMILY + 1` (65) outputs x rows, the targets (int32 codes for
the classifier, float32 for the regressor; one row per output). An import
needs both; `MEMBER_IMPORT_DONE` reads `call.classes` for a one-output
classifier, `call.class_counts` (one count per output) for several.

The fitted state (the class set present in the targets, the resident index
handle, the per-state query buffers) is OWNED here, in a process-wide store
of fixed chunks (the core/linear_member.mojo form): a slot never moves once
minted; the store's lock covers only mint, free and the handle lookup.

LIFETIME: a fit BORROWS the caller's X and y views, exactly as the
estimator's own fit borrows the user's array (k-NN is a lazy learner: the
fitted points ARE X). The caller keeps the words of X and y alive and
unchanged until this state's next fit, import or free; no copy is made per
fit (a copy per fit is a copy per fold inside cross-validation), but for a
y of several outputs: the RUN reads `outputs` contiguous target columns
(POLICY 6 of neighbors/estimator.mojo), so that fit copies y once into the
slot in that layout (the estimator's own fit makes the same transpose). The first
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
    MEMBER_CAN_CLASSIFIER, MEMBER_CAN_MULTI_OUTPUT,
    MEMBER_BLOCK_FAMILY, MEMBER_BLOCK_BY_INDEX, MEMBER_IMPORT_DONE,
    member_publish, member_call_from_spans, member_unsupported,
)
from core.member_abi import MEMBER_BORROWS_FIT, MEMBER_EXPORT_OWN, MEMBER_GIL_DOOR, MEMBER_IGNORES_Y
from core.member_families import MEMBER_FAMILY_NEIGHBORS
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.neighbors_entry_table import nb_index_prepare_call, nb_index_release_call
# Lane vdf-scope-cluster2a (2026-11-24): THE FOLD INDEX and THE FOLD QUERIES.
from core.neighbors_entry_table import nb_index_prepare_fold_call, nb_fold_ready_call, NbFold
from core.member_abi import (
    MEMBER_CAN_ROW_INDEX, MEMBER_CAN_SCOPE, MEMBER_SCOPE_NONE, MEMBER_I64, MEMBER_HOME_DEVICE,
)
from core.host_parallel import host_widen_index_i64
from core.admitted_rows_gather import admitted_gather_rows_f32, ARG_OUT_OF_RANGE
from core.kmeans_fold_transport import kmeans_fold_store_ref
from bindings.arena_cache_run import arena_scope_store_put, arena_cache_put, arena_cache_free
from core.shared_slot import SharedSlot, shared_slot_lock, shared_slot_unlock
from neighbors.neighbors_knn_run import (
    KnnDistWords, KnnQueryRequest, KnnQueryOutcome, knn_classify_run, knn_regress_run,
    knn_check_metric_input, knn_check_width, knn_metric_reads_host_queries,
)

comptime _CHUNK = 256
#: The scope cache's family prefix of the k-NN folds (on model selection's fold store, the KMeans fold store's
#: transport entry: core/kmeans_fold_transport.mojo).
comptime NB_FOLD_PREFIX = "neighbors_fold"
comptime _INDEX_SPAN = 16777216
comptime NEIGHBORS_BLOCK_POINTS = MEMBER_BLOCK_FAMILY
comptime NEIGHBORS_BLOCK_TARGETS = MEMBER_BLOCK_FAMILY + 1
comptime _I32P = MutPointer[Int32, MutUntrackedOrigin]
comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _U8P = MutPointer[UInt8, MutUntrackedOrigin]
comptime _IntP = MutPointer[Int, MutUntrackedOrigin]


struct NeighborsMemberSlot(Copyable, Movable, Defaultable):
    var generation: Int
    var live: Bool
    var regressor: Bool
    # Lane mf-member8: kinds 2 (NearestNeighbors) and 3 (RadiusNeighbors) fit
    # the points only (no targets, no predict).
    var unsupervised: Bool
    var kind: Int
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
    # THE 2-D y (lane w3f3-k): the target columns (1: the borrowed column),
    # and the running offsets of each output's class count (the classifier)
    # and of its present classes (`outputs + 1` words each, made at fit /
    # import, never per query).
    var outputs: Int
    var class_off: List[Int]
    var present_off: List[Int]
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
    # THE FOLD INDEX (lane vdf-scope-cluster2a): a fit under a row index in a
    # resident scope builds the resident index on the device from the fold's
    # rows; the slot then holds no host points: `fold_base` (the borrowed
    # host X, `fold_base_rows` rows) and the fold's int64 row numbers, which
    # the first export (or a re-mint on another entry) gathers on the host.
    var fold_base: Int
    var fold_base_rows: Int
    var fold_rows: List[Int64]
    # THE FOLD QUERIES: set by `_indexed_query` for one query only.
    var query_fold: NbFold

    def __init__(out self):
        self.generation = 0
        self.live = False
        self.regressor = False
        self.unsupervised = False
        self.kind = 0
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
        self.outputs = 1
        self.class_off = List[Int]()
        self.present_off = List[Int]()
        self.uniq = List[Int32]()
        self.one_i32 = List[Int32](length=1, fill=Int32(0))
        self.one_f32 = List[Float32](length=1, fill=Float32(0.0))
        self.proba = List[Float32]()
        self.handle = 0
        self.handle_entry = entry_ref_unresolved()
        self.fold_base = 0
        self.fold_base_rows = 0
        self.fold_rows = List[Int64]()
        self.query_fold = NbFold.none()


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
        slot[].kind = Int(words[0])
        slot[].unsupervised = words[0] == 2.0 or words[0] == 3.0
        slot[].k = Int(words[1])
        slot[].metric = Int(words[2])
        slot[].metric_arg = words[3]
        slot[].weights = Int(words[4])
        slot[].query_tile = Int(words[5])
        slot[].fitted = False
        slot[].outputs = 1
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
            slot[].class_off = List[Int]()
            slot[].present_off = List[Int]()
            slot[].uniq = List[Int32]()
            slot[].proba = List[Float32]()
            store[].free.append(index)
        return address
    finally:
        shared_slot_unlock(lock)


def _name(slot: _SlotPtr) -> String:
    if slot[].kind == 2:
        return String("NearestNeighbors")
    if slot[].kind == 3:
        return String("RadiusNeighbors")
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


def _set_class_counts(slot: _SlotPtr, call: _CallPtr, outputs: Int) raises:
    """Each output's class count (lane w3f3-k): `call.classes` for one
    output, the `outputs` words at `call.class_counts` for several (layout
    4), kept as their running offsets `class_off` (in the slot's own list,
    its capacity reused fit after fit: no allocation per fit)."""
    if outputs > 1 and call[].class_counts == 0:
        raise Error("ValueError: a " + _name(slot) + " member needs each output's class count")
    var given = _IntP(unsafe_from_address=call[].class_counts)
    slot[].class_off.clear()
    slot[].class_off.append(0)
    for j in range(outputs):  # small-loop(the outputs' class counts)
        var count = call[].classes if outputs == 1 else given.unsafe_load(j)
        if count < 1:
            raise Error("ValueError: a " + _name(slot) + " member needs each output's class count")
        slot[].class_off.append(slot[].class_off[j] + count)
    slot[].classes = call[].classes if outputs == 1 else 0


def _present_codes(slot: _SlotPtr) raises:
    """The class codes present in the fitted codes (ascending), per output
    (the slot's `outputs` code columns, each output's count from
    `class_off`), with a code outside 0 .. count-1 refused. Parallel:
    each task marks the codes of its rows in its own presence row; the merge
    reads tasks x (the counts' sum) bytes. `counts` is the entry's class
    count per output (the present classes), `present` their codes, output
    after output."""
    var rows = slot[].rows
    var outputs = slot[].outputs
    var total = slot[].class_off[outputs]
    var tasks = max(host_predict_task_count_work(rows, rows * outputs), 1)
    var chunk = host_predict_chunk(rows, tasks)
    var marks = List[UInt8](length=tasks * total, fill=UInt8(0))
    var bad = List[UInt8](length=tasks, fill=UInt8(0))
    var marks_ptr = _U8P(unsafe_from_address=Int(marks.unsafe_ptr()))
    var bad_ptr = _U8P(unsafe_from_address=Int(bad.unsafe_ptr()))
    var codes_ptr = _I32P(unsafe_from_address=slot[].targets_addr)
    var off_ptr = _IntP(unsafe_from_address=Int(slot[].class_off.unsafe_ptr()))

    def _scan(t: Int) {imm marks_ptr, imm bad_ptr, imm codes_ptr, imm off_ptr, imm rows, imm outputs, imm total,
                       imm chunk}:
        var lo = t * chunk
        var hi = min(lo + chunk, rows)
        for i in range(lo, hi):  # member fit: this task's rows, one presence mark per output
            for j in range(outputs):
                var first = off_ptr.unsafe_load(j)
                var v = Int(codes_ptr.unsafe_load(j * rows + i))
                if v < 0 or v >= off_ptr.unsafe_load(j + 1) - first:
                    bad_ptr.unsafe_store(t, UInt8(1))
                    return
                marks_ptr.unsafe_store(t * total + first + v, UInt8(1))

    if tasks <= 1:
        _scan(0)
    else:
        host_parallelize(_scan, tasks)
    for t in range(tasks):  # driver-loop: over the tasks' refusal words
        if bad[t] != 0:
            if outputs == 1:
                raise Error("ValueError: a k-NN classifier member reads class codes 0 .. "
                    + String(slot[].classes - 1) + " in y")
            raise Error("ValueError: a k-NN classifier member reads each output's class codes 0 .. count-1 in y")
    var present = List[Int32]()
    # The slot's own count lists, their capacity reused fit after fit.
    slot[].counts.clear()
    slot[].present_off.clear()
    slot[].present_off.append(0)
    for j in range(outputs):  # driver-loop: over the outputs
        var found = 0
        for c in range(slot[].class_off[j + 1] - slot[].class_off[j]):  # driver-loop: over classes, merging the tasks' presence rows
            for t in range(tasks):  # driver-loop: over tasks (cores)
                if marks[t * total + slot[].class_off[j] + c] != 0:
                    present.append(Int32(c))
                    found += 1
                    break
        slot[].counts.append(found)
        slot[].present_off.append(slot[].present_off[j] + found)
    _ = marks^
    _ = bad^
    slot[].present = present^
    slot[].uniq = List[Int32](length=len(slot[].present), fill=Int32(0))


def _wide_index(index: MemberMatrix, estimator: String) raises -> List[Int64]:
    """A host int32 / int64 row index as owned int64 row numbers (one widening
    pass in row tasks for int32)."""
    if index.home == MEMBER_HOME_DEVICE or index.cols != 1 or not (
            index.dense_host(MEMBER_I32) or index.dense_host(MEMBER_I64)):
        raise Error("TypeError: a " + estimator + " member reads a row index as a dense host int32 or int64 column")
    var n = index.rows
    var wide = List[Int64](length=max(n, 1), fill=Int64(0))
    if index.dtype == MEMBER_I64:
        unsafe_memcpy(dest=wide.unsafe_ptr(), src=MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=index.data),
                      count=n)
    else:
        host_widen_index_i64(index.data, Int(wide.unsafe_ptr()), n)
    return wide^


def _gather_rows(base: Int, base_rows: Int, width: Int, wide: List[Int64], n: Int, dst: Int,
                 estimator: String) raises:
    """Rows `wide[0:n]` of the host words at `base` (`width` 4-byte words a
    row, float32 or int32: words are copied, never converted) into `dst`, in
    row tasks across the host's cores. An index outside the base is refused."""
    var status = admitted_gather_rows_f32[DType.int64](base, base_rows, width, Int(wide.unsafe_ptr()), n, dst)
    if status == ARG_OUT_OF_RANGE:
        raise Error("IndexError: a " + estimator + " member's row index holds a row outside X's "
                    + String(base_rows) + " rows")


def _fold_route(slot: _SlotPtr, call: _CallPtr) raises -> Bool:
    """THE FOLD ROUTE: a call inside a resident scope with X pinned, on the
    device column, whose resident index lives on the fold store's context,
    and a metric that reads no host rows (cosine's zero-row and
    Jensen-Shannon's sign refusals read host words: those keep the host
    gather)."""
    if call[].scope == MEMBER_SCOPE_NONE or call[].pin == 0 or call[].route.vendor == 0:
        return False
    if knn_metric_reads_host_queries(slot[].metric):
        return False
    return nb_fold_ready_call(call[].route.entry.call)


def _put_fold(call: _CallPtr, wide: List[Int64], n: Int) raises -> NbFold:
    """The whole X found in (or put ONCE into) the scope's fold-store cache
    (`call.pin_kept` on publish: THE PIN RULE) and this call's int64 row
    numbers put (the caller frees `.index` after the call)."""
    var x = call[].x
    var store = kmeans_fold_store_ref(call[].route.mode, call[].route.vendor)
    var put = arena_scope_store_put(call[].scope, store, NB_FOLD_PREFIX, x.data, x.rows * x.cols, call[].pin)
    if put.pinned:
        call[].pin_kept = 1
    var index_id = arena_cache_put(store, Int(wide.unsafe_ptr()), 2 * n)
    return NbFold(1, put.store_id, x.rows, index_id)


def _fit_points(slot: _SlotPtr, call: _CallPtr) raises:
    """Lane mf-member8: NearestNeighbors / RadiusNeighbors `fit(X, y=None)`:
    the estimator's own fit over the caller's admitted X (its metric input
    check for NearestNeighbors, word [2] >= 0), the points borrowed, `y`
    not read."""
    var estimator = _name(slot)
    _whole(call, estimator)
    if call[].fit_output != 0:
        member_unsupported(estimator, call[].fit_output)
    if call[].weights.rows > 0:
        raise Error("TypeError: a " + estimator + " member fits no sample_weight")
    _dense(call[].x, MEMBER_F32, "X", estimator)
    var rows = call[].x.rows
    var cols = call[].x.cols
    if slot[].metric >= 0:
        knn_check_metric_input(estimator, slot[].metric, call[].x.data, rows * cols)
    _release_index(slot)
    slot[].fitted = False
    slot[].points = List[Float32]()
    slot[].fold_base = 0
    slot[].fold_base_rows = 0
    slot[].fold_rows = List[Int64]()
    slot[].points_addr = call[].x.data
    slot[].targets_addr = 0
    slot[].rows = rows
    slot[].features = cols
    slot[].fitted = True


def _fit(slot: _SlotPtr, call: _CallPtr) raises:
    if slot[].unsupervised:
        _fit_points(slot, call)
        return
    var estimator = _name(slot)
    if call[].column_index.rows > 0:
        raise Error("TypeError: a " + estimator + " member reads every column of X (no column index)")
    var indexed = call[].row_index.rows > 0
    if call[].fit_output != 0:
        member_unsupported(estimator, call[].fit_output)
    if call[].weights.rows > 0:
        raise Error("TypeError: a " + estimator + " member fits no sample_weight")
    _dense(call[].x, MEMBER_F32, "X", estimator)
    var base_rows = call[].x.rows
    var rows = call[].row_index.rows if indexed else base_rows
    var cols = call[].x.cols
    # THE 2-D y (lane w3f3-k): `outputs` target columns.
    var outputs = call[].y.cols
    if outputs < 1:
        raise Error("ValueError: a " + estimator + " member fits one target column or more")
    if call[].y.rows != base_rows:
        raise Error("ValueError: X and y have different numbers of rows")
    if slot[].regressor:
        _dense(call[].y, MEMBER_F32, "y", estimator)
    else:
        _dense(call[].y, MEMBER_I32, "y (class codes)", estimator)
        if outputs == 1 and call[].classes < 1:
            raise Error("ValueError: a " + estimator + " member needs the class count at fit")
    if not indexed:
        knn_check_metric_input(estimator, slot[].metric, call[].x.data, rows * cols)
    _release_index(slot)
    slot[].fitted = False
    # Borrow, never copy (LIFETIME in the module docstring). A copy a
    # previous import or export left in the slot is dropped.
    slot[].points = List[Float32]()
    slot[].codes = List[Int32]()
    slot[].targets = List[Float32]()
    slot[].fold_base = 0
    slot[].fold_base_rows = 0
    slot[].fold_rows = List[Int64]()
    slot[].points_addr = call[].x.data
    slot[].targets_addr = call[].y.data
    slot[].rows = rows
    slot[].features = cols
    slot[].outputs = outputs
    if indexed:
        _fit_indexed(slot, call, estimator, rows, cols, outputs)
    elif outputs > 1:
        # Several outputs: the RUN's contiguous target columns, copied once.
        _target_columns(slot, call[].y.data, rows, outputs)
    if not slot[].regressor:
        _set_class_counts(slot, call, outputs)
        _present_codes(slot)
    slot[].fitted = True


def _fit_indexed(slot: _SlotPtr, call: _CallPtr, estimator: String, rows: Int, cols: Int, outputs: Int) raises:
    """A fit under a host row index (MEMBER_CAN_ROW_INDEX, lane
    vdf-scope-cluster2a). THE FOLD INDEX on the fold route (`_fold_route`):
    the whole X found in (or put once into) the scope's cache, the fold's row
    numbers put for this call, and the resident index built from them ON THE
    DEVICE (NB_OP_INDEX_PREPARE_FOLD: the same rows in the same order as the
    host gather, so the same index words and the same neighbours); no host
    points, no per-fold upload of X. Otherwise the fold's points are gathered
    once into the slot (row tasks). Either way the fold's targets (rows x
    outputs words) are gathered into the slot (row tasks), then laid out as
    the RUN reads them."""
    var wide = _wide_index(call[].row_index, estimator)
    var x = call[].x
    if _fold_route(slot, call):
        var fold = _put_fold(call, wide, rows)
        var store = kmeans_fold_store_ref(call[].route.mode, call[].route.vendor)
        var status = List[Int32](length=2, fill=Int32(0))
        var handle = 0
        try:
            handle = nb_index_prepare_fold_call(call[].route.entry.call, Int(status.unsafe_ptr()), fold, rows, cols)
        finally:
            arena_cache_free(store, fold.index)
        if handle == 0:
            if status[1] != 0:
                raise Error("IndexError: a " + estimator + " member's row index holds a row outside X's "
                            + String(x.rows) + " rows")
            raise Error("ValueError: Input X contains NaN or infinity.")
        slot[].handle = handle
        slot[].handle_entry = call[].route.entry
        # The resident index holds the fold's rows. `points_addr` names the
        # borrowed base only as the entries' non-null index word: a metric on
        # this route reads no host index rows (`_fold_route`); a re-mint or an
        # export gathers the rows first (`_own_points`).
        slot[].fold_base = x.data
        slot[].fold_base_rows = x.rows
        slot[].points_addr = x.data
    else:
        slot[].points = List[Float32](length=max(rows * cols, 1), fill=Float32(0.0))
        _gather_rows(x.data, x.rows, cols, wide, rows, Int(slot[].points.unsafe_ptr()), estimator)
        slot[].points_addr = Int(slot[].points.unsafe_ptr())
        knn_check_metric_input(estimator, slot[].metric, slot[].points_addr, rows * cols)
    # The targets: words of y's rows (float32 targets or int32 codes).
    var gathered = List[Int32](length=max(rows * outputs, 1), fill=Int32(0))
    _gather_rows(call[].y.data, call[].y.rows, outputs, wide, rows, Int(gathered.unsafe_ptr()), estimator)
    if slot[].fold_base != 0:
        slot[].fold_rows = wide.copy()  # rows x 8 bytes: the export's gather list
    _ = len(wide)
    if outputs > 1:
        _target_columns(slot, Int(gathered.unsafe_ptr()), rows, outputs)
        _ = len(gathered)
    elif slot[].regressor:
        slot[].targets = List[Float32](length=max(rows, 1), fill=Float32(0.0))
        unsafe_memcpy(dest=slot[].targets.unsafe_ptr(), src=_F32P(unsafe_from_address=Int(gathered.unsafe_ptr())),
                      count=rows)
        slot[].targets_addr = Int(slot[].targets.unsafe_ptr())
        _ = len(gathered)
    else:
        slot[].codes = gathered^
        slot[].targets_addr = Int(slot[].codes.unsafe_ptr())


def _target_columns(slot: _SlotPtr, source: Int, rows: Int, outputs: Int):
    """THE 2-D y (lane w3f3-k): the caller's rows x `outputs` targets at
    `source` as the k-NN RUN reads them, `outputs` contiguous columns of
    `rows` words (POLICY 6 of neighbors/estimator.mojo; the estimator's own
    fit makes the same layout with `knn_transpose_targets_run`), owned by the
    slot (codes for the classifier, float32 targets for the regressor). An
    exact copy, once per fit of a multi-output y. Parallel: each task moves
    its own rows."""
    var count = rows * outputs
    var tasks = max(host_predict_task_count_work(rows, count), 1)
    var chunk = host_predict_chunk(rows, tasks)
    if slot[].regressor:
        slot[].targets = List[Float32](length=count, fill=Float32(0.0))
        var dst_f = _F32P(unsafe_from_address=Int(slot[].targets.unsafe_ptr()))
        var src_f = _F32P(unsafe_from_address=source)

        def _move_f(t: Int) {imm dst_f, imm src_f, imm rows, imm outputs, imm chunk}:
            var lo = t * chunk
            var hi = min(lo + chunk, rows)
            for i in range(lo, hi):  # member fit: this task's rows, one word per output
                for j in range(outputs):
                    dst_f.unsafe_store(j * rows + i, src_f.unsafe_load(i * outputs + j))

        if tasks <= 1:
            _move_f(0)
        else:
            host_parallelize(_move_f, tasks)
        slot[].targets_addr = Int(slot[].targets.unsafe_ptr())
        return
    slot[].codes = List[Int32](length=count, fill=Int32(0))
    var dst_i = _I32P(unsafe_from_address=Int(slot[].codes.unsafe_ptr()))
    var src_i = _I32P(unsafe_from_address=source)

    def _move_i(t: Int) {imm dst_i, imm src_i, imm rows, imm outputs, imm chunk}:
        var lo = t * chunk
        var hi = min(lo + chunk, rows)
        for i in range(lo, hi):  # member fit: this task's rows, one code per output
            for j in range(outputs):
                dst_i.unsafe_store(j * rows + i, src_i.unsafe_load(i * outputs + j))

    if tasks <= 1:
        _move_i(0)
    else:
        host_parallelize(_move_i, tasks)
    slot[].targets_addr = Int(slot[].codes.unsafe_ptr())


def _request(slot: _SlotPtr, call: _CallPtr) raises -> KnnQueryRequest:
    """One query's typed request, with the estimator's predict checks (its
    width and k wording) and the resident index of a device route."""
    var estimator = _name(slot)
    if not slot[].fitted:
        raise Error("NotFittedError: mojolearn: call fit before predict")
    _whole(call, estimator)
    # THE FOLD QUERIES (`_indexed_query`): the queries are the fold's rows of
    # the scope's resident X, gathered on the device by the resident entry.
    var folded = slot[].query_fold.has_fold != 0
    if not folded:
        _dense(call[].x, MEMBER_F32, "X", estimator)
    knn_check_width(call[].x.cols, slot[].features)
    if slot[].k < 1 or slot[].k > slot[].rows:
        raise Error("mojolearn: n_neighbors must be in [1, " + String(slot[].rows) + "], got " + String(slot[].k))
    if not folded:
        knn_check_metric_input(estimator, slot[].metric, call[].x.data, call[].x.rows * call[].x.cols)
    var request = KnnQueryRequest(slot[].points_addr, 0 if folded else call[].x.data, slot[].rows, call[].x.rows,
        slot[].features, slot[].k, slot[].query_tile,
        KnnDistWords(slot[].metric, slot[].metric_arg, slot[].weights))
    if folded:
        request.fold = slot[].query_fold.copy()
    var route = call[].route
    if route.vendor != 0:
        # The index stays on the device: minted at this state's first query
        # (the estimator's own rule), reused while the state keeps its fit.
        if slot[].handle != 0 and not entry_ref_same(slot[].handle_entry, route.entry):
            _release_index(slot)
        if slot[].handle == 0:
            # A fold-built index is re-minted from the fold's host rows.
            _own_points(slot)
            request.index = slot[].points_addr
            slot[].handle = nb_index_prepare_call(route.entry.call, slot[].points_addr, 0,
                slot[].rows, slot[].features)
            slot[].handle_entry = route.entry
        request.resident = True
        request.handle = slot[].handle
    return request^


def _fold_outcome(outcome: KnnQueryOutcome, slot: _SlotPtr) raises:
    """A fold query's range word (the entry's device gather of the query
    rows): an index outside X is refused in the member's words."""
    if outcome.range_status != 0:
        raise Error("IndexError: a " + _name(slot) + " member's row index holds a row outside X")


def _query(slot: _SlotPtr, call: _CallPtr, operation: Int) raises:
    if operation == MEMBER_OP_PREDICT:
        _predict(slot, call)
    else:
        _predict_proba(slot, call)


def _indexed_query(slot: _SlotPtr, call: _CallPtr, operation: Int) raises:
    """PREDICT / PREDICT_PROBA under a host row index (MEMBER_CAN_ROW_INDEX,
    lane vdf-scope-cluster2a). THE FOLD QUERIES on the fold route: the whole
    X found in (or put once into) the scope's cache, the fold's row numbers
    put for this call and freed after it, and the resident entry gathers the
    query rows on the device (`NbFold`, the entry's existing fold form): no
    host gather, no per-fold upload of X. Otherwise the fold's query rows are
    gathered once on the host (row tasks) and run as plain host words. The
    caller's views are restored whether the query returns or raises."""
    var estimator = _name(slot)
    if not slot[].fitted:
        raise Error("NotFittedError: mojolearn: call fit before predict")
    if call[].column_index.rows > 0:
        raise Error("TypeError: a " + estimator + " member reads every column of X (no column index)")
    _dense(call[].x, MEMBER_F32, "X", estimator)
    var index = call[].row_index
    var x = call[].x
    var n = index.rows
    var wide = _wide_index(index, estimator)
    if _fold_route(slot, call):
        var fold = _put_fold(call, wide, n)
        var store = kmeans_fold_store_ref(call[].route.mode, call[].route.vendor)
        var queries = MemberMatrix(fold.base, n, x.cols)
        queries.home = MEMBER_HOME_DEVICE
        call[].x = queries
        call[].row_index = MemberMatrix()
        slot[].query_fold = fold.copy()
        try:
            _query(slot, call, operation)
        finally:
            call[].x = x
            call[].row_index = index
            slot[].query_fold = NbFold.none()
            arena_cache_free(store, fold.index)
        _ = len(wide)
        return
    var rows = List[Float32](length=max(n * x.cols, 1), fill=Float32(0.0))
    _gather_rows(x.data, x.rows, x.cols, wide, n, Int(rows.unsafe_ptr()), estimator)
    call[].x = MemberMatrix(Int(rows.unsafe_ptr()), n, x.cols)
    call[].row_index = MemberMatrix()
    try:
        _query(slot, call, operation)
    finally:
        call[].x = x
        call[].row_index = index
    _ = len(rows) + len(wide)


def _predict(slot: _SlotPtr, call: _CallPtr) raises:
    var estimator = _name(slot)
    var request = _request(slot, call)
    var rows = call[].x.rows
    var entry = call[].route.entry.call
    var outcome = KnnQueryOutcome()
    # THE 2-D y (lane w3f3-k): the entry writes rows x outputs, row-major.
    var outputs = slot[].outputs
    if slot[].regressor:
        _dense(call[].dst, MEMBER_F32, "the prediction buffer", estimator)
        if call[].dst.rows != rows or call[].dst.cols != outputs:
            raise Error("ValueError: the prediction buffer holds one value per row of X"
                        + (String("") if outputs == 1 else String(" and output")))
        outcome = knn_regress_run(entry, request, slot[].targets_addr, call[].dst.data, outputs)
    else:
        _dense(call[].dst, MEMBER_I32, "the prediction buffer (class codes)", estimator)
        if call[].dst.rows != rows or call[].dst.cols != outputs:
            raise Error("ValueError: the prediction buffer holds one class code per row of X"
                        + (String("") if outputs == 1 else String(" and output")))
        outcome = knn_classify_run(entry, request, slot[].targets_addr, call[].dst.data,
            Int(slot[].one_f32.unsafe_ptr()), Int(slot[].uniq.unsafe_ptr()), False, slot[].counts)
    _fold_outcome(outcome, slot)


def _predict_proba(slot: _SlotPtr, call: _CallPtr) raises:
    var estimator = _name(slot)
    if slot[].regressor:
        member_unsupported(estimator, MEMBER_OP_PREDICT_PROBA)
    if slot[].outputs > 1:
        _predict_proba_outputs(slot, call)
        return
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
    _fold_outcome(outcome, slot)
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


def _predict_proba_outputs(slot: _SlotPtr, call: _CallPtr) raises:
    """THE 2-D y's predict_proba (lane w3f3-k): the entry's per-output
    blocks (rows x the output's present classes, output after output,
    neighbors/estimator.mojo `out_proba_ptr`) into `dst`, rows x the class
    counts' sum, each output's columns in turn (layout 4: scikit-learn's
    list of per-output probabilities side by side); a class absent from the
    fitted rows has probability 0. Parallel over rows."""
    var estimator = _name(slot)
    var request = _request(slot, call)
    var rows = call[].x.rows
    var outputs = slot[].outputs
    var width = slot[].class_off[outputs]
    var present = slot[].present_off[outputs]
    _dense(call[].dst, MEMBER_F32, "the probability buffer", estimator)
    if call[].dst.rows != rows or call[].dst.cols != width:
        raise Error("ValueError: the probability buffer holds one row of " + String(width)
            + " class probabilities (every output's) per row of X")
    if len(slot[].proba) < rows * present:
        slot[].proba = List[Float32](length=rows * present, fill=Float32(0.0))
    var target = Int(slot[].proba.unsafe_ptr())
    var outcome = knn_classify_run(call[].route.entry.call, request, slot[].targets_addr,
        Int(slot[].one_i32.unsafe_ptr()), target, Int(slot[].uniq.unsafe_ptr()), True, slot[].counts)
    _fold_outcome(outcome, slot)
    var dst_ptr = _F32P(unsafe_from_address=call[].dst.data)
    var src_ptr = _F32P(unsafe_from_address=target)
    var present_ptr = _I32P(unsafe_from_address=Int(slot[].present.unsafe_ptr()))
    var class_off = _IntP(unsafe_from_address=Int(slot[].class_off.unsafe_ptr()))
    var present_off = _IntP(unsafe_from_address=Int(slot[].present_off.unsafe_ptr()))
    var tasks = max(host_predict_task_count_work(rows, rows * width), 1)
    var chunk = host_predict_chunk(rows, tasks)

    def _scatter(t: Int) {imm dst_ptr, imm src_ptr, imm present_ptr, imm class_off, imm present_off, imm rows,
                          imm outputs, imm width, imm chunk}:
        var lo = t * chunk
        var hi = min(lo + chunk, rows)
        for r in range(lo, hi):  # member predict_proba: this task's rows
            for c in range(width):
                dst_ptr.unsafe_store(r * width + c, Float32(0.0))
            for j in range(outputs):
                var first = present_off.unsafe_load(j)
                var m = present_off.unsafe_load(j + 1) - first
                var block = rows * first + r * m
                var column = r * width + class_off.unsafe_load(j)
                for q in range(m):
                    dst_ptr.unsafe_store(column + Int(present_ptr.unsafe_load(first + q)),
                        src_ptr.unsafe_load(block + q))

    if tasks <= 1:
        _scatter(0)
    else:
        host_parallelize(_scatter, tasks)


def _own_points(slot: _SlotPtr):
    """Copy borrowed points into the slot once (first export); the state
    reads its own copy from then on. A resident device index is a copy of
    the same words and stays valid."""
    var count = slot[].rows * slot[].features
    if slot[].fold_base != 0:
        # THE FOLD INDEX: the fold's rows gathered from the borrowed base once.
        var gathered = List[Float32](length=max(count, 1), fill=Float32(0.0))
        try:
            _gather_rows(slot[].fold_base, slot[].fold_base_rows, slot[].features, slot[].fold_rows, slot[].rows,
                         Int(gathered.unsafe_ptr()), _name(slot))
        except:
            pass  # the index was range-checked when the fit gathered it on the device
        slot[].points = gathered^
        slot[].points_addr = Int(slot[].points.unsafe_ptr())
        slot[].fold_base = 0
        slot[].fold_base_rows = 0
        slot[].fold_rows = List[Int64]()
        return
    if slot[].points_addr == Int(slot[].points.unsafe_ptr()) and len(slot[].points) == count:
        return
    var owned = List[Float32](length=count, fill=Float32(0.0))
    unsafe_memcpy(dest=owned.unsafe_ptr(), src=_F32P(unsafe_from_address=slot[].points_addr), count=count)
    slot[].points = owned^
    slot[].points_addr = Int(slot[].points.unsafe_ptr())


def _own_targets(slot: _SlotPtr):
    """Copy borrowed targets (float32) or class codes (int32) into the slot
    once (first export); a multi-output fit's columns are the slot's own
    already (lane w3f3-k)."""
    var count = slot[].rows * slot[].outputs
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
    if (call[].flags & MEMBER_EXPORT_OWN) != 0:
        # THE LIFETIME RULE: both borrowed blocks, once; no view is answered.
        _own_points(slot)
        if not slot[].unsupervised:
            _own_targets(slot)
        return
    var block = call[].block
    var count = 1 if slot[].unsupervised else 2
    if (call[].flags & MEMBER_BLOCK_BY_INDEX) != 0:
        if block < 0 or block >= count:
            raise Error("member ABI: a " + _name(slot) + " member has " + String(count) + " fitted blocks")
        block = NEIGHBORS_BLOCK_POINTS + block
    if block == NEIGHBORS_BLOCK_TARGETS and slot[].unsupervised:
        raise Error("member ABI: a " + _name(slot) + " member has no fitted targets")
    if block == NEIGHBORS_BLOCK_POINTS:
        _own_points(slot)
        call[].dst = MemberMatrix(Int(slot[].points.unsafe_ptr()), slot[].rows, slot[].features)
    elif block == NEIGHBORS_BLOCK_TARGETS:
        _own_targets(slot)
        # One row per output (lane w3f3-k: outputs x rows).
        if slot[].regressor:
            call[].dst = MemberMatrix(Int(slot[].targets.unsafe_ptr()), slot[].outputs, slot[].rows)
        else:
            call[].dst = MemberMatrix(Int(slot[].codes.unsafe_ptr()), slot[].outputs, slot[].rows, MEMBER_I32)
    else:
        raise Error("member ABI: a k-NN member has no fitted block " + String(block))
    call[].block = block
    call[].out_cols = count
    call[].classes = 0 if slot[].unsupervised else slot[].classes


def _import(slot: _SlotPtr, call: _CallPtr) raises:
    """Copy one fitted block in; MEMBER_IMPORT_DONE completes the state."""
    var estimator = _name(slot)
    if call[].block == NEIGHBORS_BLOCK_POINTS:
        _dense(call[].x, MEMBER_F32, "the fitted points", estimator)
        var count = call[].x.rows * call[].x.cols
        _release_index(slot)
        slot[].fitted = False
        slot[].points_addr = 0
        slot[].fold_base = 0
        slot[].fold_base_rows = 0
        slot[].fold_rows = List[Int64]()
        slot[].points = List[Float32](length=count, fill=Float32(0.0))
        unsafe_memcpy(dest=slot[].points.unsafe_ptr(), src=_F32P(unsafe_from_address=call[].x.data), count=count)
        slot[].rows = call[].x.rows
        slot[].features = call[].x.cols
    elif call[].block == NEIGHBORS_BLOCK_TARGETS and not slot[].unsupervised:
        var count = call[].x.rows * call[].x.cols
        slot[].fitted = False
        slot[].targets_addr = 0
        # One row per output (lane w3f3-k: outputs x rows).
        slot[].outputs = max(call[].x.rows, 1)
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
    if slot[].unsupervised:
        # Lane mf-member8: the points are the whole fitted state.
        if slot[].rows == 0 or len(slot[].points) != slot[].rows * slot[].features:
            raise Error("member ABI: a fitted " + _name(slot) + " state needs its points")
        slot[].points_addr = Int(slot[].points.unsafe_ptr())
        slot[].fitted = True
        return
    var targets = len(slot[].targets) if slot[].regressor else len(slot[].codes)
    if (slot[].rows == 0 or len(slot[].points) != slot[].rows * slot[].features
            or targets != slot[].rows * slot[].outputs):
        raise Error("member ABI: a fitted k-NN state needs its points and one target per point and output")
    slot[].points_addr = Int(slot[].points.unsafe_ptr())
    slot[].targets_addr = Int(slot[].targets.unsafe_ptr()) if slot[].regressor else Int(slot[].codes.unsafe_ptr())
    if not slot[].regressor:
        if slot[].outputs == 1 and call[].classes < 1:
            raise Error("member ABI: a k-NN classifier import needs the class count")
        _set_class_counts(slot, call, slot[].outputs)
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
    # MEMBER_BORROWS_FIT (both kinds): a fitted state reads the caller's X
    # and y views until MEMBER_EXPORT_OWN, or the export of both blocks, has
    # copied them (LIFETIME in the module docstring).
    # MEMBER_GIL_DOOR (THE GIL RULE, core/member_abi.mojo; fx-gil-RESULT.md,
    # round 4): still one member operation at a time on the public thread;
    # the caller releases the GIL around it, as the family's own door
    # (bindings/neighbors_knn_native_door.mojo) does around the same RUN
    # functions (`nb_index_prepare_call`, `nb_index_release_call`,
    # `knn_classify_run`, `knn_regress_run`). The two checks the door runs
    # with the GIL held read the call's own words; the store has its own
    # lock. NOT `MEMBER_GIL_FREE`: the resident index and the images behind
    # the route are not traced for parallel workers.
    # MEMBER_CAN_MULTI_OUTPUT (lane w3f3-k): both kinds take a 2-D y, as
    # scikit-learn's KNeighborsClassifier / KNeighborsRegressor and the
    # estimators' own fit do (THE 2-D y in the module docstring).
    comptime _REGRESSOR = (MEMBER_CAN_FIT | MEMBER_CAN_PREDICT | MEMBER_CAN_EXPORT | MEMBER_CAN_IMPORT
        | MEMBER_BORROWS_FIT | MEMBER_GIL_DOOR | MEMBER_CAN_MULTI_OUTPUT)
    comptime _CLASSIFIER = _REGRESSOR | MEMBER_CAN_PREDICT_PROBA | MEMBER_CAN_CLASSIFIER
    # Lane mf-member8: NearestNeighbors / RadiusNeighbors fit, export and
    # import their points; no predict, no coef_ or importances.
    comptime _POINTS = (MEMBER_CAN_FIT | MEMBER_CAN_EXPORT | MEMBER_CAN_IMPORT | MEMBER_BORROWS_FIT | MEMBER_GIL_DOOR
        | MEMBER_IGNORES_Y)
    if operation == MEMBER_OP_CAPABILITIES:
        # ROW_INDEX | SCOPE (lane vdf-scope-cluster2a): the k-NN kinds take a
        # host row index (a fold) on both columns; on the device column a
        # pinned X in a resident scope stays resident and the index and the
        # queries are the fold's rows gathered on the device.
        var folds = MEMBER_CAN_ROW_INDEX | (MEMBER_CAN_SCOPE if call[].route.vendor != 0 else 0)
        if call[].state == 0:
            return _CLASSIFIER | folds
        var kind = _SlotPtr(unsafe_from_address=_lookup(call[].state, False))
        if kind[].unsupervised:
            return _POINTS
        return (_REGRESSOR if kind[].regressor else _CLASSIFIER) | folds
    if operation == MEMBER_OP_NEW:
        if call[].param_count != 6 or not call[].params:
            raise Error("member ABI: a k-NN member takes six parameters (kind, n_neighbors, metric, "
                + "metric argument, weights, query_tile)")
        var words = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call[].params)
        if words[0] != 0.0 and words[0] != 1.0 and words[0] != 2.0 and words[0] != 3.0:
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
        if words[0] == 2.0:
            raise Error("member ABI: NearestNeighbors has no predict")
        if words[0] == 3.0:
            raise Error("member ABI: RadiusNeighbors has no predict")
        if words[0] == 1.0:
            if call[].flags == MEMBER_OP_PREDICT_PROBA:
                raise Error("member ABI: KNeighborsRegressor has no predict_proba")
            call[].classes = 0
        var proba_cols = call[].classes
        if call[].y.cols > 1 and call[].class_counts != 0:
            # THE 2-D y (lane w3f3-k): every output's columns in turn.
            var given = _IntP(unsafe_from_address=call[].class_counts)
            proba_cols = 0
            for j in range(call[].y.cols):  # small-loop(the outputs' class counts)
                proba_cols += given.unsafe_load(j)
        call[].out_cols = proba_cols if call[].flags == MEMBER_OP_PREDICT_PROBA else max(1, call[].y.cols)
        call[].out_dtype = _out_dtype(words[0] == 1.0, call[].flags)
        return 0
    var slot = _SlotPtr(unsafe_from_address=_lookup(call[].state, False))
    if slot[].unsupervised and (operation == MEMBER_OP_PREDICT or operation == MEMBER_OP_PREDICT_PROBA):
        # scikit-learn's own failure (round b): no predict on these classes.
        raise Error("AttributeError: '" + _name(slot) + "' object has no attribute '"
                    + ("predict" if operation == MEMBER_OP_PREDICT else "predict_proba") + "'")
    if slot[].unsupervised and operation == MEMBER_OP_OUTPUTS:
        member_unsupported(_name(slot), operation)
    if operation == MEMBER_OP_OUTPUTS:
        if slot[].fitted and slot[].outputs > 1:
            # THE 2-D y (lane w3f3-k): rows x outputs, or every output's
            # class columns in turn.
            call[].classes = 0
            call[].out_cols = (slot[].class_off[slot[].outputs]
                               if call[].flags == MEMBER_OP_PREDICT_PROBA and not slot[].regressor
                               else slot[].outputs)
            call[].out_dtype = _out_dtype(slot[].regressor, call[].flags)
            return 0
        call[].classes = 0 if slot[].regressor else (slot[].classes if slot[].fitted else call[].classes)
        call[].out_cols = call[].classes if call[].flags == MEMBER_OP_PREDICT_PROBA else 1
        call[].out_dtype = _out_dtype(slot[].regressor, call[].flags)
        return 0
    if operation == MEMBER_OP_FIT:
        _fit(slot, call)
    elif (operation == MEMBER_OP_PREDICT or operation == MEMBER_OP_PREDICT_PROBA) and call[].row_index.rows > 0:
        _indexed_query(slot, call, operation)
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
