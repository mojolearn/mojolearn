# SPDX-License-Identifier: Apache-2.0
"""The XNeighbors family as MEMBERS (core/member_abi.mojo, MEMBER-ABI-V2.md).

One entry, `xneighbors_member_v2`, published by each x_neighbors image
(bindings/_mojolearn_x_neighbors.mojo: the device column, its
`COMPILED_VENDOR`; bindings/_mojolearn_x_neighbors_host.mojo: the CPU column,
vendor 0). It runs the family's existing typed RUN through the ONE route slot
it reads: `entry` = the x_neighbors native operations entry of the image the
route resolved to (`x_neighbors_native_api_v1`, the `XnDriver` entry every
x_neighbors door uses). No kernel is touched.

Kinds (MEMBER_OP_NEW word 0): 0 NearestCentroid, 1 LocalOutlierFactor. The
other classes of the family are not published yet (MEMBER-ABI-V2.md family
12).

NearestCentroid (kind 0). RUN: neighbors/nearest_centroid_run.mojo
(`nc_fit_count_run`, `nc_fit_priors_run`, `nc_fit_run`, `nc_labels_run`,
`nc_decision_run`, `nc_proba_run`), in the order the estimator's own door
(bindings/neighbors_local_native_door.mojo `_nc_fit`) runs them; empirical
priors are the host label ratio (core/array_helpers_impl.mojo
`class_ratio_f64` mode 1, count / rows: the host label kernel's own step).
MEMBER_OP_NEW parameters (Float64 words, count 6, or 6 + classes with given
priors): [0] kind 0, [1] metric (0 euclidean, 1 manhattan), [2] priors (0
empirical, 1 uniform, 2 given), [3] shrink_threshold (0 none), [4] the image's
fused capability (0 / 1), [5] the image's wide-probability capability (0 /
1), [6 ..] the given priors, one per class (checked against `call.classes`
at fit, normalized by the fit).
Views (all host, dense row-major; a device-home view, a row or column index,
a sample weight and a nonzero `fit_output` are refused in words): X float32
rows x features; y = class CODES 0 .. classes-1 as one int32 column
(`MEMBER_I32`), `call.classes` the class count (at least two, and every class
present in the fit rows: the estimator fits the classes it sees, so a fold
missing a class is refused in words). predict writes int32 codes (rows x 1);
decision (euclidean only) writes float32 rows x 1 for two classes (the second
class's score minus the first's), else rows x classes; predict_proba
(euclidean only) writes rows x classes as float32 or float64 (the dtype of
`dst`; the image computes one width and the other is cast from a slot-owned
scratch). Score is not published: the caller predicts and runs its own metric.
Fitted-state blocks (EXPORT / IMPORT), in index order: 64 centroids float32
classes x features, 65 within_class_std_dev float32 1 x features, 66
deviations float32 classes x features, 67 class_prior float64 1 x classes.
An import needs all four; `MEMBER_IMPORT_DONE` reads `call.classes`.

LocalOutlierFactor (kind 1). RUN: neighbors/lof_run.mojo (`lof_fit_run`,
`lof_fit_labels_run`, `lof_query_run`), as the estimator's door
(bindings/neighbors_local_native_door.mojo `_lof_fit`, `_lof_query`) runs
them. MEMBER_OP_NEW parameters (count 7): [0] kind 1, [1] n_neighbors, [2]
novelty (0 / 1), [3] contamination 'auto' (0 / 1), [4] contamination (when
not auto), [5] the unfused-kNN switch, [6] the old-items switch (the two
context flags the door reads). Views: X float32 rows x features (host,
dense; no y, no weights, no index, no device view). FIT with `fit_output =
MEMBER_OP_PREDICT` (novelty 0 only: fit_predict) also writes the fit rows'
labels to `dst` as int64 (-1 outlier, 1 inlier), rows x 1. With novelty 1:
PREDICT (int64 labels), DECISION and SCORE_SAMPLES (float32, rows x 1).
Blocks: 64 the fitted rows float32 rows x features, 65 their k-distances
float32 rows x k, 66 local reachability densities float32 1 x rows, 67
negative_outlier_factor float32 1 x rows, 68 the offset (float64 1 x 1).

LIFETIME: a NearestCentroid fit reads X and y only during the call (its
fitted state is the centroids it computes); nothing of X is kept or copied.
A LocalOutlierFactor fit BORROWS X as its fitted rows, as the estimator's
fit keeps the user's array (`_fit_X`): the caller keeps X alive and
unchanged until the state's next fit, import or free; the first export of
block 64 copies it into the slot once, and the state reads that copy from
then on.

The fitted state is OWNED here, in a process-wide store of fixed chunks (the
core/linear_member.mojo form): a slot never moves once minted; the store's
lock covers only mint, free and the handle lookup. Per-state scratch (the
predicted codes, a probability cast) is grown once and reused by every later
query of the state.
"""
from std.sys.info import size_of
from std.memory import unsafe_memcpy
from core.array_helpers_impl import class_ratio_f64
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk
from core.member_abi import (
    MemberCall, MemberMatrix, MEMBER_F32, MEMBER_F64, MEMBER_I32,
    MEMBER_OP_CAPABILITIES, MEMBER_OP_NEW, MEMBER_OP_FREE, MEMBER_OP_FIT, MEMBER_OP_PREDICT,
    MEMBER_OP_PREDICT_PROBA, MEMBER_OP_DECISION, MEMBER_OP_OUTPUTS, MEMBER_OP_EXPORT, MEMBER_OP_IMPORT,
    MEMBER_OP_SCORE_SAMPLES, MEMBER_I64, MEMBER_CAN_SCORE_SAMPLES,
    MEMBER_CAN_FIT, MEMBER_CAN_PREDICT, MEMBER_CAN_PREDICT_PROBA, MEMBER_CAN_DECISION,
    MEMBER_CAN_EXPORT, MEMBER_CAN_IMPORT, MEMBER_CAN_CLASSIFIER,
    MEMBER_BLOCK_FAMILY, MEMBER_BLOCK_BY_INDEX, MEMBER_IMPORT_DONE,
    member_publish, member_call_from_spans, member_unsupported,
)
from core.member_families import MEMBER_FAMILY_XNEIGHBORS
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.admitted_refusals import admitted_finite_refuse
from core.nearest_centroid_plan import NcLabelCodes, NcFitPlan
from neighbors.lof_run import (
    LofFitRequest, LofFitOutcome, LofQueryRequest, lof_fit_run, lof_fit_labels_run, lof_query_run,
)
from core.shared_slot import SharedSlot, shared_slot_lock, shared_slot_unlock
from core.neighbors_native_driver import XnDriver
from neighbors.nearest_centroid_run import (
    NC_PRIORS_EMPIRICAL, NC_PRIORS_UNIFORM, NC_PRIORS_GIVEN,
    NcFitRequest, NcFitOutcome, NcQueryRequest,
    nc_fit_count_run, nc_fit_priors_run, nc_fit_run, nc_labels_run, nc_decision_run, nc_proba_wide, nc_proba_run,
)
from neighbors.neighbors_local_run import (
    LOF_RESULT_SCORES, LOF_RESULT_DECISION, LOF_RESULT_LABELS, lof_check_fit_rows, lof_fit_neighbors,
    lof_check_novelty, lof_refuse_fit_predict_novelty, local_check_n_neighbors,
    nc_check_classes, nc_check_priors_size, nc_check_variance, nc_refuse_decision_metric,
)

comptime _CHUNK = 256
comptime _INDEX_SPAN = 16777216
comptime XNEIGHBORS_KIND_NEAREST_CENTROID = 0
comptime XNEIGHBORS_KIND_LOF = 1
comptime XNEIGHBORS_KIND_COUNT = 2
comptime NC_BLOCK_CENTROIDS = MEMBER_BLOCK_FAMILY
comptime NC_BLOCK_STD = MEMBER_BLOCK_FAMILY + 1
comptime NC_BLOCK_DEVIATIONS = MEMBER_BLOCK_FAMILY + 2
comptime NC_BLOCK_PRIOR = MEMBER_BLOCK_FAMILY + 3
comptime _NC_BLOCKS = 4
comptime LOF_BLOCK_ROWS = MEMBER_BLOCK_FAMILY
comptime LOF_BLOCK_DISTANCES = MEMBER_BLOCK_FAMILY + 1
comptime LOF_BLOCK_DENSITIES = MEMBER_BLOCK_FAMILY + 2
comptime LOF_BLOCK_SCORES = MEMBER_BLOCK_FAMILY + 3
comptime LOF_BLOCK_OFFSET = MEMBER_BLOCK_FAMILY + 4
comptime _LOF_BLOCKS = 5
comptime _I32P = MutPointer[Int32, MutUntrackedOrigin]
comptime _I64P = MutPointer[Int64, MutUntrackedOrigin]
comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _F64P = MutPointer[Float64, MutUntrackedOrigin]


struct XNeighborsMemberSlot(Copyable, Movable, Defaultable):
    var generation: Int
    var live: Bool
    var kind: Int
    var fitted: Bool
    var rows: Int
    var features: Int
    var classes: Int
    # NearestCentroid words (kind 0).
    var euclidean: Bool
    var priors: Int
    var shrink: Float64
    var fused: Bool
    var wide: Bool
    var given: List[Float64]
    var centroids: List[Float32]
    var std: List[Float32]
    var deviations: List[Float32]
    var prior: List[Float64]
    # Per-state query scratch, grown once: the predicted codes and the
    # probabilities of the other width when `dst` differs from the image's.
    var pred: List[Int32]
    var proba32: List[Float32]
    var proba64: List[Float64]
    # LocalOutlierFactor words (kind 1). `rows_addr` = the borrowed fit X,
    # or `fit_rows` after an import or the first export.
    var k: Int
    var novelty: Bool
    var auto: Bool
    var contamination: Float64
    var unfused: Bool
    var old_items: Bool
    var neighbors: Int
    var offset: Float64
    var rows_addr: Int
    var fit_rows: List[Float32]
    var distances: List[Float32]
    var densities: List[Float32]
    var scores: List[Float32]
    var offset_word: List[Float64]

    def __init__(out self):
        self.generation = 0
        self.live = False
        self.kind = 0
        self.fitted = False
        self.rows = 0
        self.features = 0
        self.classes = 0
        self.euclidean = True
        self.priors = NC_PRIORS_UNIFORM
        self.shrink = 0.0
        self.fused = False
        self.wide = False
        self.given = List[Float64]()
        self.centroids = List[Float32]()
        self.std = List[Float32]()
        self.deviations = List[Float32]()
        self.prior = List[Float64]()
        self.pred = List[Int32]()
        self.proba32 = List[Float32]()
        self.proba64 = List[Float64]()
        self.k = 20
        self.novelty = False
        self.auto = True
        self.contamination = 0.0
        self.unfused = False
        self.old_items = False
        self.neighbors = 0
        self.offset = 0.0
        self.rows_addr = 0
        self.fit_rows = List[Float32]()
        self.distances = List[Float32]()
        self.densities = List[Float32]()
        self.scores = List[Float32]()
        self.offset_word = List[Float64](length=1, fill=0.0)


struct XNeighborsMemberStore(Defaultable, Movable):
    var synchronization: List[Int64]
    var chunks: List[List[XNeighborsMemberSlot]]
    var free: List[Int]
    var count: Int

    def __init__(out self):
        # Side-effect-free even when the process slot's initializers race.
        self.synchronization = List[Int64](length=2, fill=0)
        self.chunks = List[List[XNeighborsMemberSlot]]()
        self.free = List[Int]()
        self.count = 0


comptime _STORE = SharedSlot[XNeighborsMemberStore, "MojolearnXNeighborsMemberStoreV2", 2]
comptime _SlotPtr = MutPointer[XNeighborsMemberSlot, MutUntrackedOrigin]
comptime _CallPtr = MutPointer[MemberCall, MutUntrackedOrigin]


def _slot_address(index: Int) raises -> Int:
    # Under the store's lock; a chunk's buffer never grows.
    var store = _STORE.get_or_create_ptr()
    return Int(store[].chunks[index // _CHUNK].unsafe_ptr()) + (index % _CHUNK) * size_of[XNeighborsMemberSlot]()


def _clear(slot: _SlotPtr):
    slot[].fitted = False
    slot[].given = List[Float64]()
    slot[].centroids = List[Float32]()
    slot[].std = List[Float32]()
    slot[].deviations = List[Float32]()
    slot[].prior = List[Float64]()
    slot[].pred = List[Int32]()
    slot[].proba32 = List[Float32]()
    slot[].proba64 = List[Float64]()
    slot[].rows_addr = 0
    slot[].fit_rows = List[Float32]()
    slot[].distances = List[Float32]()
    slot[].densities = List[Float32]()
    slot[].scores = List[Float32]()


def _mint(words: _F64P, count: Int) raises -> Int:
    var store = _STORE.get_or_create_ptr()
    var lock = rebind[MutPointer[Int64, MutUntrackedOrigin]](store[].synchronization.unsafe_ptr())
    shared_slot_lock(lock)
    try:
        var index = -1
        if len(store[].free) > 0:
            index = store[].free.pop()
        else:
            if store[].count >= _INDEX_SPAN - 1:
                raise Error("RuntimeError: mojolearn: too many live XNeighbors member states")
            if store[].count == len(store[].chunks) * _CHUNK:
                store[].chunks.append(List[XNeighborsMemberSlot](length=_CHUNK, fill=XNeighborsMemberSlot()))
            index = store[].count
            store[].count += 1
        var slot = _SlotPtr(unsafe_from_address=_slot_address(index))
        slot[].generation += 1
        slot[].live = True
        _clear(slot)
        slot[].kind = Int(words[0])
        if slot[].kind == XNEIGHBORS_KIND_NEAREST_CENTROID:
            slot[].euclidean = words[1] == 0.0
            slot[].priors = Int(words[2])
            slot[].shrink = words[3]
            slot[].fused = words[4] != 0.0
            slot[].wide = words[5] != 0.0
            for i in range(6, count):  # driver-loop: over the given priors (one word per class)
                slot[].given.append(words[i])
        else:
            slot[].k = Int(words[1])
            slot[].novelty = words[2] != 0.0
            slot[].auto = words[3] != 0.0
            slot[].contamination = words[4]
            slot[].unfused = words[5] != 0.0
            slot[].old_items = (words[6] != 0.0) and not slot[].unfused
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
            _clear(slot)
            store[].free.append(index)
        return address
    finally:
        shared_slot_unlock(lock)


def _name(slot: _SlotPtr) -> String:
    return String("NearestCentroid") if slot[].kind == XNEIGHBORS_KIND_NEAREST_CENTROID else String(
        "LocalOutlierFactor")


def _whole(call: _CallPtr, estimator: String) raises:
    if call[].row_index.rows > 0 or call[].column_index.rows > 0:
        raise Error("TypeError: a " + estimator + " member reads whole host views (no row or column index)")


def _dense(view: MemberMatrix, dtype: Int, what: String, estimator: String) raises:
    if not view.dense_host(dtype):
        var word = String("int32") if dtype == MEMBER_I32 else (
            String("float64") if dtype == MEMBER_F64 else String("float32"))
        raise Error("TypeError: a " + estimator + " member reads " + what + " as a dense host " + word + " view")


# ------------------------------------------------------------ NearestCentroid


def _nc_fit(slot: _SlotPtr, call: _CallPtr) raises:
    var estimator = _name(slot)
    _whole(call, estimator)
    if call[].fit_output != 0:
        member_unsupported(estimator, call[].fit_output)
    if call[].weights.rows > 0:
        raise Error("TypeError: a " + estimator + " member fits no sample_weight")
    _dense(call[].x, MEMBER_F32, "X", estimator)
    _dense(call[].y, MEMBER_I32, "y (class codes)", estimator)
    var rows = call[].x.rows
    var features = call[].x.cols
    if call[].y.cols != 1 or call[].y.rows != rows:
        raise Error("X and y have different numbers of rows")
    var classes = call[].classes
    nc_check_classes(classes)
    if slot[].priors == NC_PRIORS_GIVEN:
        nc_check_priors_size(len(slot[].given), classes)
    var driver = XnDriver(call[].route.entry.call)
    slot[].fitted = False
    slot[].pred = List[Int32]()
    slot[].proba32 = List[Float32]()
    slot[].proba64 = List[Float64]()
    # The four fitted arrays are the state's own (the estimator's door
    # allocates the same four as its published attributes).
    slot[].prior = List[Float64](length=classes, fill=0.0)
    slot[].std = List[Float32](length=features, fill=Float32(0.0))
    slot[].centroids = List[Float32](length=classes * features, fill=Float32(0.0))
    slot[].deviations = List[Float32](length=classes * features, fill=Float32(0.0))
    if slot[].priors == NC_PRIORS_GIVEN:
        unsafe_memcpy(dest=slot[].prior.unsafe_ptr(), src=slot[].given.unsafe_ptr(), count=classes)
    var request = NcFitRequest(call[].x.data, call[].y.data, rows, features, classes,
        slot[].euclidean, slot[].priors, slot[].shrink)
    request.prior = Int(slot[].prior.unsafe_ptr())
    request.stats = Int(slot[].std.unsafe_ptr())
    request.centroids = Int(slot[].centroids.unsafe_ptr())
    request.deviations = Int(slot[].deviations.unsafe_ptr())
    var plan = NcFitPlan(rows, features, classes)
    nc_fit_count_run(driver, request, plan)
    for c in range(classes):  # driver-loop: over classes (the fit's class counts)
        if plan.wide_counts[c] == 0:
            raise Error("ValueError: a " + estimator + " member needs every class code 0 .. "
                + String(classes - 1) + " present in its fit rows (class " + String(c) + " is absent)")
    if slot[].priors == NC_PRIORS_EMPIRICAL:
        # The label family's ratio step, mode 1: Float64(count) / Float64(rows).
        _ = class_ratio_f64(_I64P(unsafe_from_address=Int(plan.wide_counts.unsafe_ptr())),
            _F64P(unsafe_from_address=request.prior), classes, rows, 1)
    else:
        nc_fit_priors_run(request)
    var outcome = nc_fit_run(driver, request, plan)
    if outcome.status != 0:
        admitted_finite_refuse(outcome.status, estimator)
    nc_check_variance(outcome.constant)
    _ = plan^
    slot[].rows = rows
    slot[].features = features
    slot[].classes = classes
    slot[].fitted = True


def _nc_request(slot: _SlotPtr, call: _CallPtr, scored: Bool) raises -> NcQueryRequest:
    var estimator = _name(slot)
    if not slot[].fitted:
        raise Error("NotFittedError: mojolearn: call fit before predict")
    _whole(call, estimator)
    if scored and not slot[].euclidean:
        nc_refuse_decision_metric()
    _dense(call[].x, MEMBER_F32, "X", estimator)
    if call[].x.cols != slot[].features:
        raise Error("ValueError: X has " + String(call[].x.cols) + " features, but " + estimator
            + " is expecting " + String(slot[].features) + " features as input.")
    return NcQueryRequest(call[].x.data, Int(slot[].centroids.unsafe_ptr()), Int(slot[].std.unsafe_ptr()),
        Int(slot[].prior.unsafe_ptr()), call[].x.rows, slot[].features, slot[].classes,
        slot[].euclidean, slot[].fused, slot[].wide)


def _nc_predict(slot: _SlotPtr, call: _CallPtr) raises:
    var estimator = _name(slot)
    var request = _nc_request(slot, call, False)
    var rows = request.rows
    _dense(call[].dst, MEMBER_I32, "the prediction buffer (class codes)", estimator)
    if call[].dst.rows != rows or call[].dst.cols != 1:
        raise Error("ValueError: the prediction buffer holds one class code per row of X")
    if rows == 0:
        return
    # The RUN writes into its code holder; the state's buffer (grown once)
    # stands in for it, so a query allocates one word, not one per row.
    if len(slot[].pred) < rows:
        slot[].pred = List[Int32](length=rows, fill=Int32(0))
    var codes = NcLabelCodes(1)
    swap(codes.values, slot[].pred)
    var status = 0
    try:
        status = nc_labels_run(XnDriver(call[].route.entry.call), request, codes)
    finally:
        swap(codes.values, slot[].pred)
    _ = codes^
    if status != 0:
        admitted_finite_refuse(status, estimator)
    unsafe_memcpy(dest=_I32P(unsafe_from_address=call[].dst.data), src=slot[].pred.unsafe_ptr(), count=rows)


def _nc_decision(slot: _SlotPtr, call: _CallPtr) raises:
    var estimator = _name(slot)
    var request = _nc_request(slot, call, True)
    var width = 1 if slot[].classes == 2 else slot[].classes
    _dense(call[].dst, MEMBER_F32, "the decision buffer", estimator)
    if call[].dst.rows != request.rows or call[].dst.cols != width:
        raise Error("ValueError: the decision buffer holds " + String(width) + " scores per row of X")
    var status = nc_decision_run(XnDriver(call[].route.entry.call), request, call[].dst.data)
    if status != 0:
        admitted_finite_refuse(status, estimator)


def _nc_proba(slot: _SlotPtr, call: _CallPtr) raises:
    var estimator = _name(slot)
    var request = _nc_request(slot, call, True)
    var rows = request.rows
    var classes = slot[].classes
    var wide = nc_proba_wide(request, False)
    var asked_wide = call[].dst.dtype == MEMBER_F64
    _dense(call[].dst, MEMBER_F64 if asked_wide else MEMBER_F32, "the probability buffer", estimator)
    if call[].dst.rows != rows or call[].dst.cols != classes:
        raise Error("ValueError: the probability buffer holds one row of " + String(classes)
            + " class probabilities per row of X")
    var count = rows * classes
    var target = call[].dst.data
    if wide != asked_wide:
        # The image computes the other width: into the state's scratch,
        # cast below (parallel over rows).
        if wide:
            if len(slot[].proba64) < count:
                slot[].proba64 = List[Float64](length=count, fill=0.0)
            target = Int(slot[].proba64.unsafe_ptr())
        else:
            if len(slot[].proba32) < count:
                slot[].proba32 = List[Float32](length=count, fill=Float32(0.0))
            target = Int(slot[].proba32.unsafe_ptr())
    var status = nc_proba_run(XnDriver(call[].route.entry.call), request, target, False)
    if status != 0:
        admitted_finite_refuse(status, estimator)
    if wide == asked_wide or count == 0:
        return
    var src = target
    var dst = call[].dst.data
    var tasks = max(host_predict_task_count_work(rows, count), 1)
    var chunk = host_predict_chunk(rows, tasks)

    def _cast(t: Int) {imm src, imm dst, imm rows, imm classes, imm chunk, imm wide}:
        var lo = t * chunk * classes
        var hi = min((t + 1) * chunk, rows) * classes
        if wide:
            var s = _F64P(unsafe_from_address=src)
            var d = _F32P(unsafe_from_address=dst)
            for i in range(lo, hi):  # member predict_proba: this task's words
                d.unsafe_store(i, Float32(s.unsafe_load(i)))
        else:
            var s = _F32P(unsafe_from_address=src)
            var d = _F64P(unsafe_from_address=dst)
            for i in range(lo, hi):  # member predict_proba: this task's words
                d.unsafe_store(i, Float64(s.unsafe_load(i)))

    if tasks <= 1:
        _cast(0)
    else:
        host_parallelize(_cast, tasks)


def _nc_block(slot: _SlotPtr, block: Int) raises -> MemberMatrix:
    if block == NC_BLOCK_CENTROIDS:
        return MemberMatrix(Int(slot[].centroids.unsafe_ptr()), slot[].classes, slot[].features)
    if block == NC_BLOCK_STD:
        return MemberMatrix(Int(slot[].std.unsafe_ptr()), 1, slot[].features)
    if block == NC_BLOCK_DEVIATIONS:
        return MemberMatrix(Int(slot[].deviations.unsafe_ptr()), slot[].classes, slot[].features)
    if block == NC_BLOCK_PRIOR:
        return MemberMatrix(Int(slot[].prior.unsafe_ptr()), 1, slot[].classes, MEMBER_F64)
    raise Error("member ABI: a NearestCentroid member has no fitted block " + String(block))


def _nc_export(slot: _SlotPtr, call: _CallPtr) raises:
    """A view of one fitted block; the slot owns the words until this
    state's next fit, import or free."""
    if not slot[].fitted:
        raise Error("NotFittedError: this NearestCentroid member is not fitted yet")
    var block = call[].block
    if (call[].flags & MEMBER_BLOCK_BY_INDEX) != 0:
        if block < 0 or block >= _NC_BLOCKS:
            raise Error("member ABI: a NearestCentroid member has four fitted blocks")
        block = NC_BLOCK_CENTROIDS + block
    call[].dst = _nc_block(slot, block)
    call[].block = block
    call[].out_cols = _NC_BLOCKS
    call[].classes = slot[].classes


def _nc_import(slot: _SlotPtr, call: _CallPtr) raises:
    """Copy one fitted block in; MEMBER_IMPORT_DONE completes the state."""
    var estimator = _name(slot)
    var block = call[].block
    var count = call[].x.rows * call[].x.cols
    if block == NC_BLOCK_PRIOR:
        _dense(call[].x, MEMBER_F64, "the class priors", estimator)
        slot[].fitted = False
        slot[].prior = List[Float64](length=count, fill=0.0)
        unsafe_memcpy(dest=slot[].prior.unsafe_ptr(), src=_F64P(unsafe_from_address=call[].x.data), count=count)
    elif block == NC_BLOCK_CENTROIDS or block == NC_BLOCK_STD or block == NC_BLOCK_DEVIATIONS:
        _dense(call[].x, MEMBER_F32, "a fitted block", estimator)
        slot[].fitted = False
        var words = List[Float32](length=count, fill=Float32(0.0))
        unsafe_memcpy(dest=words.unsafe_ptr(), src=_F32P(unsafe_from_address=call[].x.data), count=count)
        if block == NC_BLOCK_CENTROIDS:
            slot[].centroids = words^
            slot[].features = call[].x.cols
        elif block == NC_BLOCK_STD:
            slot[].std = words^
        else:
            slot[].deviations = words^
    elif block != 0 or (call[].flags & MEMBER_IMPORT_DONE) == 0:
        raise Error("member ABI: a NearestCentroid member has no fitted block " + String(block))
    if (call[].flags & MEMBER_IMPORT_DONE) == 0:
        return
    var classes = call[].classes
    var features = slot[].features
    nc_check_classes(classes)
    if (len(slot[].prior) != classes or len(slot[].centroids) != classes * features
            or len(slot[].deviations) != classes * features or len(slot[].std) != features or features == 0):
        raise Error("member ABI: a fitted NearestCentroid state needs its centroids, deviations, "
            + "standard deviations and one prior per class")
    slot[].classes = classes
    slot[].pred = List[Int32]()
    slot[].proba32 = List[Float32]()
    slot[].proba64 = List[Float64]()
    slot[].fitted = True


# ------------------------------------------------------------ LocalOutlierFactor


def _lof_fit(slot: _SlotPtr, call: _CallPtr) raises:
    var estimator = _name(slot)
    _whole(call, estimator)
    if call[].weights.rows > 0:
        raise Error("TypeError: a " + estimator + " member fits no sample_weight")
    var labels = call[].fit_output == MEMBER_OP_PREDICT
    if call[].fit_output != 0 and not labels:
        member_unsupported(estimator, call[].fit_output)
    if labels and slot[].novelty:
        lof_refuse_fit_predict_novelty()
    _dense(call[].x, MEMBER_F32, "X", estimator)
    var rows = call[].x.rows
    var features = call[].x.cols
    lof_check_fit_rows(rows, features)
    if labels:
        _dense(call[].dst, MEMBER_I64, "the label buffer", estimator)
        if call[].dst.rows != rows or call[].dst.cols != 1:
            raise Error("ValueError: the label buffer holds one label per row of X")
    var k = lof_fit_neighbors(slot[].k, rows)
    slot[].fitted = False
    # Borrow the rows (LIFETIME); a copy a previous import or export left is dropped.
    slot[].fit_rows = List[Float32]()
    slot[].rows_addr = call[].x.data
    slot[].distances = List[Float32](length=rows * k, fill=Float32(0.0))
    slot[].densities = List[Float32](length=rows, fill=Float32(0.0))
    slot[].scores = List[Float32](length=rows, fill=Float32(0.0))
    var request = LofFitRequest(call[].x.data, rows, features, k, slot[].unfused, slot[].old_items,
        slot[].auto, slot[].contamination)
    request.distances = Int(slot[].distances.unsafe_ptr())
    request.densities = Int(slot[].densities.unsafe_ptr())
    request.scores = Int(slot[].scores.unsafe_ptr())
    var driver = XnDriver(call[].route.entry.call)
    var outcome = lof_fit_run(driver, request)
    if outcome.status != 0:
        admitted_finite_refuse(outcome.status, estimator)
    if labels:
        lof_fit_labels_run(driver, request.scores, rows, outcome.offset, call[].dst.data)
    slot[].rows = rows
    slot[].features = features
    slot[].neighbors = k
    slot[].offset = outcome.offset
    slot[].fitted = True


def _lof_query(slot: _SlotPtr, call: _CallPtr, kind: Int, what: String) raises:
    var estimator = _name(slot)
    lof_check_novelty(slot[].novelty, what)
    if not slot[].fitted:
        raise Error("NotFittedError: mojolearn: call fit before " + what)
    _whole(call, estimator)
    _dense(call[].x, MEMBER_F32, "X", estimator)
    if call[].x.cols != slot[].features:
        raise Error("ValueError: X has " + String(call[].x.cols) + " features, but " + estimator
            + " is expecting " + String(slot[].features) + " features as input.")
    var rows = call[].x.rows
    _dense(call[].dst, MEMBER_I64 if kind == LOF_RESULT_LABELS else MEMBER_F32, "the result buffer", estimator)
    if call[].dst.rows != rows or call[].dst.cols != 1:
        raise Error("ValueError: the result buffer holds one value per row of X")
    var request = LofQueryRequest(call[].x.data, slot[].rows_addr, Int(slot[].distances.unsafe_ptr()),
        Int(slot[].densities.unsafe_ptr()), rows, slot[].rows, slot[].features, slot[].neighbors, slot[].rows,
        slot[].unfused, slot[].old_items, kind, 0.0 if kind == LOF_RESULT_SCORES else -slot[].offset)
    request.dst = call[].dst.data
    var status = lof_query_run(XnDriver(call[].route.entry.call), request)
    admitted_finite_refuse(status, estimator)


def _lof_own_rows(slot: _SlotPtr):
    """Copy the borrowed fit rows into the slot once (first export)."""
    var count = slot[].rows * slot[].features
    if slot[].rows_addr == Int(slot[].fit_rows.unsafe_ptr()) and len(slot[].fit_rows) == count:
        return
    var owned = List[Float32](length=count, fill=Float32(0.0))
    unsafe_memcpy(dest=owned.unsafe_ptr(), src=_F32P(unsafe_from_address=slot[].rows_addr), count=count)
    slot[].fit_rows = owned^
    slot[].rows_addr = Int(slot[].fit_rows.unsafe_ptr())


def _lof_export(slot: _SlotPtr, call: _CallPtr) raises:
    """A view of one fitted block; the slot owns the words until this
    state's next fit, import or free (the borrowed rows are copied here)."""
    if not slot[].fitted:
        raise Error("NotFittedError: this LocalOutlierFactor member is not fitted yet")
    var block = call[].block
    if (call[].flags & MEMBER_BLOCK_BY_INDEX) != 0:
        if block < 0 or block >= _LOF_BLOCKS:
            raise Error("member ABI: a LocalOutlierFactor member has five fitted blocks")
        block = LOF_BLOCK_ROWS + block
    if block == LOF_BLOCK_ROWS:
        _lof_own_rows(slot)
        call[].dst = MemberMatrix(slot[].rows_addr, slot[].rows, slot[].features)
    elif block == LOF_BLOCK_DISTANCES:
        call[].dst = MemberMatrix(Int(slot[].distances.unsafe_ptr()), slot[].rows, slot[].neighbors)
    elif block == LOF_BLOCK_DENSITIES:
        call[].dst = MemberMatrix(Int(slot[].densities.unsafe_ptr()), 1, slot[].rows)
    elif block == LOF_BLOCK_SCORES:
        call[].dst = MemberMatrix(Int(slot[].scores.unsafe_ptr()), 1, slot[].rows)
    elif block == LOF_BLOCK_OFFSET:
        slot[].offset_word[0] = slot[].offset
        call[].dst = MemberMatrix(Int(slot[].offset_word.unsafe_ptr()), 1, 1, MEMBER_F64)
    else:
        raise Error("member ABI: a LocalOutlierFactor member has no fitted block " + String(block))
    call[].block = block
    call[].out_cols = _LOF_BLOCKS


def _lof_import(slot: _SlotPtr, call: _CallPtr) raises:
    """Copy one fitted block in; MEMBER_IMPORT_DONE completes the state."""
    var estimator = _name(slot)
    var block = call[].block
    var count = call[].x.rows * call[].x.cols
    if block == LOF_BLOCK_OFFSET:
        _dense(call[].x, MEMBER_F64, "the offset", estimator)
        if count != 1:
            raise Error("member ABI: a LocalOutlierFactor offset is one float64 word")
        slot[].fitted = False
        slot[].offset = _F64P(unsafe_from_address=call[].x.data)[0]
    elif block >= LOF_BLOCK_ROWS and block <= LOF_BLOCK_SCORES:
        _dense(call[].x, MEMBER_F32, "a fitted block", estimator)
        slot[].fitted = False
        var words = List[Float32](length=count, fill=Float32(0.0))
        unsafe_memcpy(dest=words.unsafe_ptr(), src=_F32P(unsafe_from_address=call[].x.data), count=count)
        if block == LOF_BLOCK_ROWS:
            slot[].fit_rows = words^
            slot[].rows_addr = Int(slot[].fit_rows.unsafe_ptr())
            slot[].rows = call[].x.rows
            slot[].features = call[].x.cols
        elif block == LOF_BLOCK_DISTANCES:
            slot[].distances = words^
            slot[].neighbors = call[].x.cols
        elif block == LOF_BLOCK_DENSITIES:
            slot[].densities = words^
        else:
            slot[].scores = words^
    elif block != 0 or (call[].flags & MEMBER_IMPORT_DONE) == 0:
        raise Error("member ABI: a LocalOutlierFactor member has no fitted block " + String(block))
    if (call[].flags & MEMBER_IMPORT_DONE) == 0:
        return
    var rows = slot[].rows
    if (rows == 0 or len(slot[].fit_rows) != rows * slot[].features or slot[].neighbors < 1
            or len(slot[].distances) != rows * slot[].neighbors or len(slot[].densities) != rows
            or len(slot[].scores) != rows):
        raise Error("member ABI: a fitted LocalOutlierFactor state needs its rows, k-distances, densities, "
            + "scores and offset")
    slot[].fitted = True


# ------------------------------------------------------------ dispatch


comptime _NC_EUCLIDEAN = (MEMBER_CAN_FIT | MEMBER_CAN_PREDICT | MEMBER_CAN_PREDICT_PROBA | MEMBER_CAN_DECISION
    | MEMBER_CAN_EXPORT | MEMBER_CAN_IMPORT | MEMBER_CAN_CLASSIFIER)
comptime _NC_MANHATTAN = MEMBER_CAN_FIT | MEMBER_CAN_PREDICT | MEMBER_CAN_EXPORT | MEMBER_CAN_IMPORT | MEMBER_CAN_CLASSIFIER


comptime _LOF_NOVELTY = (MEMBER_CAN_FIT | MEMBER_CAN_PREDICT | MEMBER_CAN_DECISION | MEMBER_CAN_SCORE_SAMPLES
    | MEMBER_CAN_EXPORT | MEMBER_CAN_IMPORT)
comptime _LOF_OUTLIER = MEMBER_CAN_FIT | MEMBER_CAN_EXPORT | MEMBER_CAN_IMPORT   # fit_predict: FIT + fit_output


def _capabilities(slot: _SlotPtr) -> Int:
    if slot[].kind == XNEIGHBORS_KIND_LOF:
        return _LOF_NOVELTY if slot[].novelty else _LOF_OUTLIER
    return _NC_EUCLIDEAN if slot[].euclidean else _NC_MANHATTAN


def _new(call: _CallPtr) raises:
    if call[].param_count < 1 or not call[].params:
        raise Error("member ABI: an XNeighbors member takes its kind as the first parameter")
    var words = _F64P(unsafe_from_address=call[].params)
    var kind = Int(words[0])
    if Float64(kind) != words[0] or kind < 0 or kind >= XNEIGHBORS_KIND_COUNT:
        raise Error("member ABI: unknown XNeighbors member kind")
    if kind == XNEIGHBORS_KIND_NEAREST_CENTROID:
        if call[].param_count < 6:
            raise Error("member ABI: a NearestCentroid member takes six parameters (kind, metric, priors, "
                + "shrink_threshold, fused, wide) and the given priors")
        if words[1] != 0.0 and words[1] != 1.0:
            raise Error("member ABI: unknown NearestCentroid metric code")
        if words[2] != 0.0 and words[2] != 1.0 and words[2] != 2.0:
            raise Error("member ABI: unknown NearestCentroid priors code")
        if (words[2] == 2.0) != (call[].param_count > 6):
            raise Error("member ABI: NearestCentroid given priors come as words 6 .. 6 + classes")
    else:
        if call[].param_count != 7:
            raise Error("member ABI: a LocalOutlierFactor member takes seven parameters (kind, n_neighbors, "
                + "novelty, contamination auto, contamination, unfused, old_items)")
        local_check_n_neighbors(Int(words[1]))
    call[].state = _mint(words, call[].param_count)


def _xneighbors_member(operation: Int, address: Int) raises -> Int:
    var call = _CallPtr(unsafe_from_address=address)
    if operation == MEMBER_OP_CAPABILITIES:
        if call[].state == 0:
            return _NC_EUCLIDEAN | _LOF_NOVELTY
        return _capabilities(_SlotPtr(unsafe_from_address=_lookup(call[].state, False)))
    if operation == MEMBER_OP_NEW:
        _new(call)
        return 0
    if operation == MEMBER_OP_FREE:
        _ = _lookup(call[].state, True)
        call[].state = 0
        return 0
    var slot = _SlotPtr(unsafe_from_address=_lookup(call[].state, False))
    if slot[].kind == XNEIGHBORS_KIND_LOF:
        if operation == MEMBER_OP_OUTPUTS:
            call[].classes = 0
            call[].out_cols = 1
        elif operation == MEMBER_OP_FIT:
            _lof_fit(slot, call)
        elif operation == MEMBER_OP_PREDICT:
            _lof_query(slot, call, LOF_RESULT_LABELS, "predict")
        elif operation == MEMBER_OP_DECISION:
            _lof_query(slot, call, LOF_RESULT_DECISION, "decision_function")
        elif operation == MEMBER_OP_SCORE_SAMPLES:
            _lof_query(slot, call, LOF_RESULT_SCORES, "score_samples")
        elif operation == MEMBER_OP_EXPORT:
            _lof_export(slot, call)
        elif operation == MEMBER_OP_IMPORT:
            _lof_import(slot, call)
        else:
            member_unsupported(_name(slot), operation)
        return 0
    if operation == MEMBER_OP_OUTPUTS:
        var classes = slot[].classes if slot[].fitted else call[].classes
        call[].classes = classes
        if call[].flags == MEMBER_OP_PREDICT_PROBA:
            call[].out_cols = classes
        elif call[].flags == MEMBER_OP_DECISION:
            call[].out_cols = 1 if classes == 2 else classes
        else:
            call[].out_cols = 1
        return 0
    if operation == MEMBER_OP_FIT:
        _nc_fit(slot, call)
    elif operation == MEMBER_OP_PREDICT:
        _nc_predict(slot, call)
    elif operation == MEMBER_OP_DECISION:
        _nc_decision(slot, call)
    elif operation == MEMBER_OP_PREDICT_PROBA:
        _nc_proba(slot, call)
    elif operation == MEMBER_OP_EXPORT:
        _nc_export(slot, call)
    elif operation == MEMBER_OP_IMPORT:
        _nc_import(slot, call)
    else:
        member_unsupported(_name(slot), operation)
    return 0


def xneighbors_member_v2(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
    result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """The member entry of the XNeighbors family (schema `MEMBER_ABI`)."""
    try:
        var word = _xneighbors_member(operation, member_call_from_spans(operation, aa, na, np, nf))
        if result:
            MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = word
        return 0
    except error:
        native_descriptor_report(error, context, sink)
        return 1


def xneighbors_member_publish(mode: Int, vendor: Int, publisher: Int) raises:
    """Publish this image's XNeighbors member entry. One line in each
    x_neighbors image's module initialization; a build with no GPU vendor
    (`vendor` -1) publishes nothing."""
    if vendor < 0:
        return
    member_publish[MEMBER_FAMILY_XNEIGHBORS](mode, vendor, publisher, xneighbors_member_v2)
