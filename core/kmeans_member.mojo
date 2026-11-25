# SPDX-License-Identifier: Apache-2.0
"""KMeans as a MEMBER (core/member_abi.mojo, MEMBER-ABI-V2.md).

One entry, `kmeans_member_v2`, published by each image that carries the
KMeans kernel table (`kmeans_member_publish`): the device image `_mojolearn`
(core/mojolearn_run.mojo `KMEANS_NATIVE_TABLE`) and the host image
`_mojolearn_core_host` (core/mojolearn_core_host_run.mojo). It runs the
family's existing typed RUN (`KMeansNativeDriver`, core/kmeans_native_api.mojo)
over the KMeans entry of the `DeviceRoute` it is handed: `route.entry` is the
KMeans native entry of the member's image, `route.flags` that entry's
capability word (`KMEANS_CAP_*`, read once by the caller's UNPACK). No other
route slot is read. No kernel is touched.

MEMBER_OP_NEW parameters (Float64 words, 8, or more for explicit starting
centers): [0] n_clusters, [1] init (0 'k-means++', 1 'random', 2 explicit
starting centers), [2] n_init, [3] max_iter, [4] tol,
[5] random_state as the BITS of an int64 (exact for every seed:
`bitcast[DType.float64](Int64(seed))`), [6] metric (0 'euclidean' /
'l2_expanded', 1 'l2_sqrt_expanded'), [7] oversampling_factor.
With init 2 (lane mf-clu6) the starting centers follow as DATA read once at
the caller's UNPACK: [8] 1 given / 0 absent (`init='array'` without
`init_centroids`), [9] their rows, [10] their columns, then their float32
words packed two per word (core/cluster_member_words.mojo). The fit raises
the class's own door's refusals in its order (bindings/kmeans_native_pack.mojo
`_kmeans_fit`): n_clusters above the rows, the absent centers, their shape
against (n_clusters, features of X), a NaN or an infinity in them; then it
copies them into the state's centers, which the kernel refines in place.
Kind 0 KMeans (a clusterer, not a classifier: `y` is not read, `classes` 0).

Views: X dense row-major host float32; sample weights a host float32 vector
(MEMBER_CAN_SAMPLE_WEIGHT; negative weights are the estimator's own rule).
On an image with the resident fold entry (`KMEANS_CAP_FOLD`, the GPU images)
X may also be a DEVICE-home view (`data` = the base store id, `rows` = the
base's rows, `offset` 0) with a device-home int64 `row_index` (the store of
the fold's row numbers): the member gathers on the device, as the
estimator's own door does for a model-selection fold. Its weights (lane
vdf-scope-cluster2a) are the whole device-home float32 vector of the base's
rows, gathered by the same index on the device. A HOST-home X with a host int32 / int64 `row_index` (a caller's host
fold or drawn rows: MEMBER_CAN_ROW_INDEX means both homes) is gathered here
once per call, rows (and the weights' rows) in row tasks across the host's
cores (core/admitted_rows_gather.mojo), then runs the host route as gathered
host words would (lane fx-audit 7); inside a resident scope on the fold image
it runs the device fold instead (`_scoped_indexed`), weighted or not. A
column index or a device view on a host image: refused in words.
X is tested finite by the kernel over its own upload (fit and every query,
host and device images: `test_finite` is always set); a NaN or an infinity is
refused in admit_x's wording. Given sample weights are scanned here (one
parallel host scan) and refused in the estimator's words, so the kind has
MEMBER_CAN_TEST_ROWS: a caller makes no finite scan of X for it.

Outputs: predict writes the cluster ids as int32 (MEMBER_I32, rows x 1);
transform writes float32 distances (rows x n_clusters) in the model's metric.
`call.fit_output = MEMBER_OP_PREDICT` writes the training labels in the fit
call (fit_predict); `MEMBER_OP_TRANSFORM` writes the fitted rows' distances
(fit_transform: one fused call where the image has it, `KMEANS_CAP_JOINT`).
Score (MEMBER_OP_SCORE, lane mf-kmeans5): `call.value` = scikit-learn's
`KMeans.score(X, sample_weight)`, minus the inertia of X on the state's
centers: the image's KMEANS_SCORE op, the fit's post-loop cost over the given
rows (weights optional, MEMBER_CAN_WEIGHTED_SCORE; a device fold gathers its
weights on the device), on the resident centers of a GPU image as a query is.

Fitted-state blocks (MEMBER_OP_EXPORT / MEMBER_OP_IMPORT), in index order:
0 `MEMBER_BLOCK_FAMILY` float32 n_clusters x features, the centers;
1 `MEMBER_BLOCK_SCALARS` float64 [inertia, n_iter, sum_scale, weight_scale];
2 `MEMBER_BLOCK_FAMILY + 1` int32 1 x rows, the training labels (empty after
an import that did not give them). An import needs the first two.

The fitted state is OWNED here, in a process-wide store of fixed chunks: a
slot never moves once minted, so an operation runs on its slot without the
store's lock (the lock covers mint, free and the handle lookup). A state on a
GPU image keeps its centers resident after its first query (one upload per
fitted state, as the estimator's own resident holder), released at its next
fit, import or free. Serial by nature here: the store's bookkeeping.
"""
from std.memory import bitcast, unsafe_memcpy
from std.sys.info import size_of
from core.admitted_refusals import admitted_finite_refuse, door_rows_refuse
from core.kmeans_native_api import (
    KMeansNativeDriver, KMeansFitRequest, KMeansFitResult, KMeansQueryRequest, KMeansScoreRequest, KMeansFoldStore,
    KMEANS_CAP_JOINT, KMEANS_CAP_RESIDENT, KMEANS_CAP_FOLD,
)
from core.member_abi import (
    MemberCall, MemberMatrix, MEMBER_F32, MEMBER_F64, MEMBER_I32, MEMBER_I64, MEMBER_HOME_DEVICE,
    MEMBER_OP_CAPABILITIES, MEMBER_OP_NEW, MEMBER_OP_FREE, MEMBER_OP_FIT, MEMBER_OP_PREDICT,
    MEMBER_OP_TRANSFORM, MEMBER_OP_OUTPUTS, MEMBER_OP_EXPORT, MEMBER_OP_IMPORT, MEMBER_OP_SCORE,
    MEMBER_CAN_FIT, MEMBER_CAN_PREDICT, MEMBER_CAN_TRANSFORM, MEMBER_CAN_EXPORT, MEMBER_CAN_IMPORT,
    MEMBER_CAN_SCORE, MEMBER_CAN_WEIGHTED_SCORE,
    MEMBER_CAN_SAMPLE_WEIGHT, MEMBER_CAN_DEVICE_VIEWS, MEMBER_CAN_ROW_INDEX, MEMBER_CAN_TEST_ROWS,
    MEMBER_HOME_HOST, MEMBER_BLOCK_SCALARS, MEMBER_BLOCK_FAMILY, MEMBER_BLOCK_BY_INDEX, MEMBER_IMPORT_DONE,
    member_publish, member_call_from_spans, member_unsupported,
)
from core.member_families import MEMBER_FAMILY_KMEANS
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.shared_slot import SharedSlot, shared_slot_lock, shared_slot_unlock
from core.host_finite_scan import host_all_finite_f32
from core.cluster_member_words import member_words_take_f32, member_words_f32_slots
from core.admitted_rows_gather import admitted_gather_rows_f32, ARG_OUT_OF_RANGE
from core.member_abi import MEMBER_CAN_SCOPE, MEMBER_SCOPE_NONE
from core.host_parallel import host_widen_index_i64
# lane vdf-scope-cluster (2026-11-24): the scope's per-family store (lane vdf-scope-run,
# bindings/arena_cache_run.mojo) on the fold image's store (core/kmeans_fold_transport.mojo).
from core.kmeans_fold_transport import kmeans_fold_store_ref, KMEANS_FOLD_PREFIX
from bindings.arena_cache_run import arena_scope_store_put, arena_cache_put, arena_cache_free

comptime _CHUNK = 256
comptime _INDEX_SPAN = 16777216
comptime _EST = "KMeans"
comptime _WORDS = 8
comptime _INIT_START = 2
comptime _START_AT = 11
comptime _BLOCK_CENTERS = MEMBER_BLOCK_FAMILY
comptime _BLOCK_LABELS = MEMBER_BLOCK_FAMILY + 1


struct KMeansMemberSlot(Copyable, Movable, Defaultable):
    var generation: Int
    var live: Bool
    var clusters: Int
    var init: Int
    var restarts: Int
    var iterations: Int
    var seed: Int
    var metric: Int
    var tolerance: Float64
    var oversampling: Float64
    var fitted: Bool
    var features: Int
    # The resident center handle of a GPU image (-1: none minted).
    var handle: Int
    var centers: List[Float32]
    var labels: List[Int32]
    var scalars: List[Float64]
    # Explicit starting centers (init 2, lane mf-clu6): given, shape, words.
    var start_given: Bool
    var start_rows: Int
    var start_cols: Int
    var start: List[Float32]

    def __init__(out self):
        self.generation = 0
        self.live = False
        self.clusters = 0
        self.init = 0
        self.restarts = 1
        self.iterations = 0
        self.seed = 0
        self.metric = 0
        self.tolerance = 0.0
        self.oversampling = 0.0
        self.fitted = False
        self.features = 0
        self.handle = -1
        self.centers = List[Float32]()
        self.labels = List[Int32]()
        self.scalars = List[Float64]()
        self.start_given = False
        self.start_rows = 0
        self.start_cols = 0
        self.start = List[Float32]()


struct KMeansMemberStore(Defaultable, Movable):
    var synchronization: List[Int64]
    var chunks: List[List[KMeansMemberSlot]]
    var free: List[Int]
    var count: Int

    def __init__(out self):
        # Side-effect-free even when the process slot's initializers race.
        self.synchronization = List[Int64](length=2, fill=0)
        self.chunks = List[List[KMeansMemberSlot]]()
        self.free = List[Int]()
        self.count = 0


comptime _STORE = SharedSlot[KMeansMemberStore, "MojolearnKMeansMemberStoreV2", 2]
comptime _Slot = MutPointer[KMeansMemberSlot, MutUntrackedOrigin]
comptime _Call = MutPointer[MemberCall, MutUntrackedOrigin]


def _slot_address(index: Int) raises -> Int:
    # Under the store's lock. A chunk's buffer is allocated once and never
    # grows, so the address stays valid after the lock is released.
    var store = _STORE.get_or_create_ptr()
    return Int(store[].chunks[index // _CHUNK].unsafe_ptr()) + (index % _CHUNK) * size_of[KMeansMemberSlot]()


def _mint(words: MutPointer[Float64, MutUntrackedOrigin], count: Int) raises -> Int:
    var store = _STORE.get_or_create_ptr()
    var lock = rebind[MutPointer[Int64, MutUntrackedOrigin]](store[].synchronization.unsafe_ptr())
    shared_slot_lock(lock)
    try:
        var index = -1
        if len(store[].free) > 0:
            index = store[].free.pop()
        else:
            if store[].count >= _INDEX_SPAN - 1:
                raise Error("RuntimeError: mojolearn: too many live KMeans member states")
            if store[].count == len(store[].chunks) * _CHUNK:
                store[].chunks.append(List[KMeansMemberSlot](length=_CHUNK, fill=KMeansMemberSlot()))
            index = store[].count
            store[].count += 1
        var slot = _Slot(unsafe_from_address=_slot_address(index))
        slot[].generation += 1
        slot[].live = True
        slot[].clusters = Int(words[0])
        slot[].init = Int(words[1])
        slot[].restarts = Int(words[2])
        slot[].iterations = Int(words[3])
        slot[].tolerance = words[4]
        slot[].seed = Int(bitcast[DType.int64](words[5]))
        slot[].metric = Int(words[6])
        slot[].oversampling = words[7]
        slot[].fitted = False
        slot[].features = 0
        slot[].handle = -1
        slot[].start_given = False
        slot[].start_rows = 0
        slot[].start_cols = 0
        slot[].start = List[Float32]()
        if slot[].init == _INIT_START and count > _WORDS:
            slot[].start_given = words[8] != 0.0
            slot[].start_rows = Int(words[9])
            slot[].start_cols = Int(words[10])
            slot[].start = member_words_take_f32(Int(words), _START_AT, slot[].start_rows * slot[].start_cols)
        return slot[].generation * _INDEX_SPAN + index + 1
    finally:
        shared_slot_unlock(lock)


def _lookup(handle: Int, release: Bool) raises -> Int:
    """The slot address of a live handle. A released handle, a handle of
    another store or a made-up integer is refused: never a stale read."""
    var store = _STORE.get_or_create_ptr()
    var lock = rebind[MutPointer[Int64, MutUntrackedOrigin]](store[].synchronization.unsafe_ptr())
    shared_slot_lock(lock)
    try:
        var index = handle % _INDEX_SPAN - 1
        if handle <= 0 or index < 0 or index >= store[].count:
            raise Error("NotFittedError: this member state was released or never created")
        var address = _slot_address(index)
        var slot = _Slot(unsafe_from_address=address)
        if not slot[].live or slot[].generation != handle // _INDEX_SPAN:
            raise Error("NotFittedError: this member state was released or never created")
        if release:
            slot[].live = False
            slot[].fitted = False
            slot[].handle = -1
            slot[].centers = List[Float32]()
            slot[].labels = List[Int32]()
            slot[].scalars = List[Float64]()
            slot[].start = List[Float32]()
            store[].free.append(index)
        return address
    finally:
        shared_slot_unlock(lock)


def _release_resident(slot: _Slot, call: _Call) raises:
    """Release the slot's resident centers (a GPU image's handle), before a
    new fit or import and at free."""
    if slot[].handle < 0:
        return
    var handle = slot[].handle
    slot[].handle = -1
    KMeansNativeDriver(call[].route.entry.call).release(handle)


def _fold(call: _Call) raises -> KMeansFoldStore:
    """The device fold of X (a device-home X with a device-home int64 row
    index, on an image with the fold entry), or an inactive store for a
    whole dense host X. Every other view is refused in words."""
    if call[].column_index.rows > 0:
        raise Error("TypeError: a KMeans member reads every column of X (no column index)")
    if call[].x.home != MEMBER_HOME_DEVICE:
        # A host row index never reaches here: `_host_indexed` gathered it.
        if not call[].x.dense_host(MEMBER_F32):
            raise Error("TypeError: a KMeans member reads X as a dense host float32 view")
        return KMeansFoldStore()
    if (call[].route.flags & KMEANS_CAP_FOLD) == 0:
        raise Error("TypeError: this KMeans member's image has no device fold entry; hand over host words")
    if call[].x.offset != 0 or call[].x.dtype != MEMBER_F32:
        raise Error("TypeError: a KMeans member reads a device-home X as a whole float32 store (offset 0)")
    if (call[].row_index.rows <= 0 or call[].row_index.home != MEMBER_HOME_DEVICE
            or call[].row_index.dtype != MEMBER_I64):
        raise Error("TypeError: a device-home X of a KMeans member needs a device-home int64 row index")
    var weights_slot = -1
    if call[].weights.rows > 0:
        # THE WEIGHTED FOLD (lane vdf-scope-cluster2a): the whole weight
        # vector resident in the same store, gathered by the same index on
        # the device (its finite test there, refused in the estimator's words).
        var w = call[].weights
        if (w.home != MEMBER_HOME_DEVICE or w.offset != 0 or w.dtype != MEMBER_F32 or w.cols != 1
                or w.rows != call[].x.rows):
            raise Error("TypeError: a weighted KMeans fold reads sample_weight as a whole device-home float32"
                + " vector of X's rows (offset 0)")
        weights_slot = w.data
    return KMeansFoldStore(call[].x.data, call[].x.rows, call[].row_index.data, weights_slot)


def _rows(call: _Call, fold: KMeansFoldStore) -> Int:
    return call[].row_index.rows if fold.active() else call[].x.rows


def _refuse(finite: Int, range_error: Int, fold: KMeansFoldStore) raises:
    if fold.active():
        door_rows_refuse(finite, range_error, _EST, fold.rows)
    else:
        admitted_finite_refuse(finite, _EST)


def _fit(slot: _Slot, call: _Call) raises:
    var fold = _fold(call)
    var output = call[].fit_output
    if output != 0 and output != MEMBER_OP_PREDICT and output != MEMBER_OP_TRANSFORM:
        member_unsupported(_EST, output)
    var rows = _rows(call, fold)
    var cols = call[].x.cols
    var k = slot[].clusters
    if k > rows:
        raise Error("mojolearn: n_clusters=" + String(k) + " exceeds n_samples=" + String(rows))
    if slot[].init == _INIT_START:
        _start_check(slot, k, cols)
    var weights = _weights(call, fold, rows, "fit")
    var weight_count = rows if (weights != 0 or fold.weights >= 0) else 0
    var distances = 0
    var joint = (output == MEMBER_OP_TRANSFORM and not fold.active()
        and (call[].route.flags & KMEANS_CAP_JOINT) != 0)
    if output == MEMBER_OP_PREDICT:
        _destination(call, rows, 1, MEMBER_I32, "the label buffer")
    elif output == MEMBER_OP_TRANSFORM:
        _destination(call, rows, k, MEMBER_F32, "the distance buffer")
        if joint:
            distances = call[].dst.data
    _release_resident(slot, call)
    slot[].fitted = False
    slot[].centers = List[Float32](length=k * cols, fill=0.0)
    if slot[].init == _INIT_START:
        # A COPY, as the door's: the kernel refines the centers in place and
        # the state's starting words serve its next fit.
        unsafe_memcpy(dest=slot[].centers.unsafe_ptr(), src=slot[].start.unsafe_ptr(), count=k * cols)
    slot[].labels = List[Int32](length=rows, fill=0)
    var request = KMeansFitRequest(0 if fold.active() else call[].x.data, Int(slot[].centers.unsafe_ptr()),
        Int(slot[].labels.unsafe_ptr()), weights, distances, rows, cols, k, weight_count, slot[].iterations,
        slot[].seed, slot[].restarts, slot[].init, slot[].metric, True, slot[].tolerance, slot[].oversampling,
        fold)
    var result = KMeansNativeDriver(call[].route.entry.call).fit(request)
    _refuse(result.finite, result.range_error, fold)
    slot[].features = cols
    slot[].scalars = List[Float64](length=4, fill=0.0)
    slot[].scalars[0] = result.inertia
    slot[].scalars[1] = Float64(result.iterations)
    slot[].scalars[2] = result.sum_scale
    slot[].scalars[3] = result.weight_scale
    slot[].fitted = True
    if output == MEMBER_OP_PREDICT:
        # The kernel's labels are the state's (labels_ block); the caller's
        # buffer gets one copy of the int32 ids.
        unsafe_memcpy(dest=MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=call[].dst.data),
            src=slot[].labels.unsafe_ptr(), count=rows)
    elif output == MEMBER_OP_TRANSFORM and not joint:
        _query(slot, call, True)


def _start_check(slot: _Slot, k: Int, cols: Int) raises:
    """The class's own refusals of explicit starting centers, its words and
    order (bindings/kmeans_native_pack.mojo `_kmeans_fit`): absent, shape
    against (n_clusters, features of X), then a NaN or an infinity."""
    if not slot[].start_given:
        raise Error("mojolearn: init='array' needs init_centroids")
    if slot[].start_rows != k or slot[].start_cols != cols:
        raise Error("mojolearn: init_centroids must be (" + String(k) + ", " + String(cols) + "), got ("
            + String(slot[].start_rows) + ", " + String(slot[].start_cols) + ")")
    if not host_all_finite_f32(MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(slot[].start.unsafe_ptr())),
            k * cols):
        raise Error("Input init_centroids contains NaN or infinity (" + String(_EST) + ").")


def _weights(call: _Call, fold: KMeansFoldStore, rows: Int, what: String) raises -> Int:
    """The address of the call's host float32 sample weights (0: none),
    admitted for the fit or the score in the estimator's words."""
    if call[].weights.rows <= 0:
        return 0
    if fold.active():
        # The fold's weights are its store slot (`_fold`), never a host address.
        return 0
    if not call[].weights.dense_host(MEMBER_F32):
        raise Error("TypeError: a KMeans member reads sample_weight as a dense host float32 view")
    if call[].weights.rows != rows or call[].weights.cols != 1:
        raise Error("ValueError: sample_weight and X have different numbers of rows")
    if not host_all_finite_f32(MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=call[].weights.data),
            rows):
        # The estimator's own words (bindings/admitted_family_door.mojo FAMILY_NONFINITE_WEIGHTS).
        raise Error("Input sample_weight contains NaN or infinity.")
    return call[].weights.data


def _destination(call: _Call, rows: Int, cols: Int, dtype: Int, what: String) raises:
    if not call[].dst.dense_host(dtype):
        raise Error("TypeError: a KMeans member writes " + what + " as a dense host "
            + (String("int32") if dtype == MEMBER_I32 else String("float32")) + " view")
    if call[].dst.rows != rows or call[].dst.cols != cols:
        raise Error("ValueError: " + what + " holds " + String(rows) + " x " + String(cols) + " values")


def _query(slot: _Slot, call: _Call, transform: Bool) raises:
    if not slot[].fitted:
        raise Error("RuntimeError: this estimator is not fitted yet")
    var fold = _fold(call)
    var rows = _rows(call, fold)
    var k = slot[].clusters
    if call[].x.cols != slot[].features:
        raise Error("mojolearn: X has " + String(call[].x.cols) + " features, KMeans was fitted with "
            + String(slot[].features))
    if transform:
        _destination(call, rows, k, MEMBER_F32, "the distance buffer")
    else:
        _destination(call, rows, 1, MEMBER_I32, "the label buffer")
    var driver = KMeansNativeDriver(call[].route.entry.call)
    # A fold query uploads the host centers on the store's context (as the
    # estimator's own door); a whole query on a GPU image reads the state's
    # resident copy, minted once at the first query.
    if not fold.active() and slot[].handle < 0 and (call[].route.flags & KMEANS_CAP_RESIDENT) != 0:
        slot[].handle = driver.prepare(Int(slot[].centers.unsafe_ptr()), k, slot[].features)
    var handle = -1 if fold.active() else slot[].handle
    var request = KMeansQueryRequest(0 if fold.active() else call[].x.data, Int(slot[].centers.unsafe_ptr()),
        call[].dst.data, rows, slot[].features, k, slot[].metric, True, transform, handle, fold)
    var result = driver.query(request)
    _refuse(result.finite, result.range_error, fold)


def _score(slot: _Slot, call: _Call) raises:
    """MEMBER_OP_SCORE: `call.value` = minus the inertia of the call's X on
    the state's centers (scikit-learn's `KMeans.score`; `y` is not read).
    The image's KMEANS_SCORE op over the same three homes as `_query`."""
    if not slot[].fitted:
        raise Error("RuntimeError: this estimator is not fitted yet")
    var fold = _fold(call)
    var rows = _rows(call, fold)
    var k = slot[].clusters
    if call[].x.cols != slot[].features:
        raise Error("mojolearn: X has " + String(call[].x.cols) + " features, KMeans was fitted with "
            + String(slot[].features))
    var weights = _weights(call, fold, rows, "score")
    var weight_count = rows if (weights != 0 or fold.weights >= 0) else 0
    var driver = KMeansNativeDriver(call[].route.entry.call)
    if not fold.active() and slot[].handle < 0 and (call[].route.flags & KMEANS_CAP_RESIDENT) != 0:
        slot[].handle = driver.prepare(Int(slot[].centers.unsafe_ptr()), k, slot[].features)
    var handle = -1 if fold.active() else slot[].handle
    var request = KMeansScoreRequest(0 if fold.active() else call[].x.data, Int(slot[].centers.unsafe_ptr()),
        weights, rows, slot[].features, k, weight_count, slot[].metric, True, handle, fold)
    var result = driver.score(request)
    _refuse(result.finite, result.range_error, fold)
    call[].value = -result.inertia


def _export(slot: _Slot, call: _Call) raises:
    """A view of one fitted block. The slot owns the words; they stay valid
    until this state's next fit, import or free."""
    if not slot[].fitted:
        raise Error("NotFittedError: this KMeans member is not fitted yet")
    var block = call[].block
    if (call[].flags & MEMBER_BLOCK_BY_INDEX) != 0:
        if block < 0 or block > 2:
            raise Error("member ABI: a KMeans member has three fitted blocks")
        block = _BLOCK_CENTERS if block == 0 else (MEMBER_BLOCK_SCALARS if block == 1 else _BLOCK_LABELS)
    if block == _BLOCK_CENTERS:
        call[].dst = MemberMatrix(Int(slot[].centers.unsafe_ptr()), slot[].clusters, slot[].features)
    elif block == MEMBER_BLOCK_SCALARS:
        call[].dst = MemberMatrix(Int(slot[].scalars.unsafe_ptr()), 1, len(slot[].scalars), MEMBER_F64)
    elif block == _BLOCK_LABELS:
        var count = len(slot[].labels)
        call[].dst = MemberMatrix(Int(slot[].labels.unsafe_ptr()) if count > 0 else 0, 1 if count > 0 else 0,
            count, MEMBER_I32)
    else:
        raise Error("member ABI: a KMeans member has no fitted block " + String(block))
    call[].block = block
    call[].out_cols = 3


def _import(slot: _Slot, call: _Call) raises:
    """Copy one fitted block in; MEMBER_IMPORT_DONE completes the state."""
    if call[].block == _BLOCK_CENTERS:
        if not call[].x.dense_host(MEMBER_F32):
            raise Error("member ABI: the KMeans centers block is a dense host float32 matrix")
        if call[].x.rows != slot[].clusters:
            raise Error("ValueError: the fitted centers have " + String(call[].x.rows)
                + " rows, n_clusters is " + String(slot[].clusters))
        _release_resident(slot, call)
        slot[].fitted = False
        var count = call[].x.rows * call[].x.cols
        slot[].centers = List[Float32](length=count, fill=0.0)
        unsafe_memcpy(dest=slot[].centers.unsafe_ptr(),
            src=MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=call[].x.data), count=count)
        slot[].features = call[].x.cols
        slot[].labels = List[Int32]()
    elif call[].block == MEMBER_BLOCK_SCALARS:
        if not call[].x.dense_host(MEMBER_F64) or call[].x.rows * call[].x.cols != 4:
            raise Error("member ABI: the KMeans scalars block is four host float64 words"
                + " (inertia, n_iter, sum_scale, weight_scale)")
        var source = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call[].x.data)
        slot[].scalars = List[Float64](length=4, fill=0.0)
        for i in range(4):  # driver-loop: four scalar words
            slot[].scalars[i] = source[i]
    elif call[].block == _BLOCK_LABELS:
        if not call[].x.dense_host(MEMBER_I32):
            raise Error("member ABI: the KMeans labels block is a host int32 vector")
        var count = call[].x.rows * call[].x.cols
        slot[].labels = List[Int32](length=count, fill=0)
        unsafe_memcpy(dest=slot[].labels.unsafe_ptr(),
            src=MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=call[].x.data), count=count)
    elif call[].block != 0 or (call[].flags & MEMBER_IMPORT_DONE) == 0:
        raise Error("member ABI: a KMeans member has no fitted block " + String(call[].block))
    if (call[].flags & MEMBER_IMPORT_DONE) == 0:
        return
    if slot[].features == 0 or len(slot[].centers) != slot[].clusters * slot[].features or len(slot[].scalars) != 4:
        raise Error("member ABI: a fitted KMeans state needs its centers and scalars blocks")
    slot[].fitted = True


def _capabilities(call: _Call) -> Int:
    var mask = (MEMBER_CAN_FIT | MEMBER_CAN_PREDICT | MEMBER_CAN_TRANSFORM | MEMBER_CAN_EXPORT | MEMBER_CAN_IMPORT
        | MEMBER_CAN_SAMPLE_WEIGHT | MEMBER_CAN_TEST_ROWS | MEMBER_CAN_SCORE | MEMBER_CAN_WEIGHTED_SCORE)
    if (call[].route.flags & KMEANS_CAP_FOLD) != 0:
        # SCOPE (lane vdf-scope-cluster): on the fold image a pinned host X under a host row index is kept
        # resident in the caller's scope (the fold store kind) and every fold is taken on the device.
        mask |= MEMBER_CAN_DEVICE_VIEWS | MEMBER_CAN_ROW_INDEX | MEMBER_CAN_SCOPE
    return mask


def _gather_host(source: MemberMatrix, index: MemberMatrix, width: Int, mut dst: List[Float32]) raises:
    # Rows `index` of the dense host float32 `source` (`width` words a row)
    # into `dst`, in row tasks across the host's cores. Non-finite words are
    # the operation's own test (TEST_ROWS); an index outside X is refused.
    var rows = index.rows
    dst = List[Float32](unsafe_uninit_length=max(rows * width, 1))
    var status: Int
    if index.dtype == MEMBER_I64:
        status = admitted_gather_rows_f32[DType.int64](source.data, source.rows, width, index.data, rows,
                                                       Int(dst.unsafe_ptr()))
    else:
        status = admitted_gather_rows_f32[DType.int32](source.data, source.rows, width, index.data, rows,
                                                       Int(dst.unsafe_ptr()))
    if status == ARG_OUT_OF_RANGE:
        raise Error("IndexError: a KMeans member's row index holds a row outside X's " + String(source.rows)
            + " rows")


def _scoped_indexed(slot: _Slot, call: _Call, operation: Int) raises:
    """THE SCOPED FOLD (MEMBER_CAN_SCOPE, lane vdf-scope-cluster; weighted: lane vdf-scope-cluster2a): a FIT / PREDICT / TRANSFORM / SCORE
    over a pinned host X with a host row index, inside the caller's resident scope, on the fold image. The whole X
    is found in (or put once into) the scope's fold-store cache, so every fold and candidate of one search reads ONE
    upload (`call.pin_kept` = 1 the moment the cache keeps the caller's pin: THE PIN RULE); the fold's int64 row
    numbers are put for this call only (rows x 8 bytes) and freed after it; the operation then runs the image's
    fold kernels (cluster/kmeans_fold.mojo) exactly as a device-home fold, which gather the same rows in the same
    order as `_gather_host`: same bytes as the host-index path and the CPU column (the kernels test finite words
    and the index range over the gather, refused in the door's words). The caller's views are restored whether
    the operation returns or raises."""
    var index = call[].row_index
    var x = call[].x
    var weights = call[].weights
    var n = index.rows
    if weights.rows > 0 and (not weights.dense_host(MEMBER_F32) or weights.cols != 1 or weights.rows != x.rows):
        raise Error("TypeError: a KMeans member reads sample_weight as a dense host float32 view of X's rows")
    var store = kmeans_fold_store_ref(call[].route.mode, call[].route.vendor)
    var put = arena_scope_store_put(call[].scope, store, KMEANS_FOLD_PREFIX, x.data, x.rows * x.cols, call[].pin)
    if put.pinned:
        call[].pin_kept = 1
    # THE WEIGHTED FOLD (lane vdf-scope-cluster2a): the whole weight vector,
    # gathered on the device by the fold's index, beside X's rows. THE
    # PER-INPUT PINS (lane vdf-scope-pins): with the weights' own token it is
    # found in (or put once into) the same scope cache as X, kept resident for
    # every fold and candidate (`call.weights_pin_kept` = 1 the moment the
    # cache keeps it); without one it is put for this call and freed after.
    var weights_id = -1
    var weights_owned = False
    if weights.rows > 0:
        if call[].weights_pin != 0 and call[].scope != MEMBER_SCOPE_NONE:
            var wput = arena_scope_store_put(call[].scope, store, KMEANS_FOLD_PREFIX, weights.data, x.rows,
                                             call[].weights_pin)
            if wput.pinned:
                call[].weights_pin_kept = 1
            weights_id = wput.store_id
        else:
            weights_id = arena_cache_put(store, weights.data, x.rows)
            weights_owned = True
    var wide = List[Int64]()
    var index_address = index.data
    if index.dtype == MEMBER_I32:
        wide = List[Int64](unsafe_uninit_length=max(n, 1))
        host_widen_index_i64(index.data, Int(wide.unsafe_ptr()), n)
        index_address = Int(wide.unsafe_ptr())
    var index_id = arena_cache_put(store, index_address, 2 * n)  # int64 rows: two 4-byte words each
    _ = wide^  # the put waited for its copy: the host words may go
    var device_x = MemberMatrix(put.store_id, x.rows, x.cols)
    device_x.home = MEMBER_HOME_DEVICE
    var device_index = MemberMatrix(index_id, n, 1, MEMBER_I64)
    device_index.home = MEMBER_HOME_DEVICE
    call[].x = device_x
    call[].row_index = device_index
    if weights_id >= 0:
        var device_weights = MemberMatrix(weights_id, x.rows, 1)
        device_weights.home = MEMBER_HOME_DEVICE
        call[].weights = device_weights
    try:
        if operation == MEMBER_OP_FIT:
            _fit(slot, call)
        elif operation == MEMBER_OP_SCORE:
            _score(slot, call)
        else:
            _query(slot, call, operation == MEMBER_OP_TRANSFORM)
    finally:
        call[].x = x
        call[].row_index = index
        call[].weights = weights
        arena_cache_free(store, index_id)
        if weights_owned:
            arena_cache_free(store, weights_id)


def _host_indexed(slot: _Slot, call: _Call, operation: Int) raises:
    """FIT / PREDICT / TRANSFORM / SCORE over a host-home X with a host row index:
    the indexed rows of X (and of the weights) gathered once into this call's
    own words, the plain host route run on them, the caller's views restored
    (whether the operation returns or raises)."""
    var index = call[].row_index
    if index.home != MEMBER_HOME_HOST or index.data == 0 or index.cols != 1 or (
            index.dtype != MEMBER_I64 and index.dtype != MEMBER_I32):
        raise Error("TypeError: a KMeans member reads a host row index as one host int32 or int64 column")
    var x = call[].x
    var weights = call[].weights
    if not x.dense_host(MEMBER_F32):
        raise Error("TypeError: a KMeans member reads X as a dense host float32 view")
    if (call[].scope != MEMBER_SCOPE_NONE and call[].pin != 0
            and (call[].route.flags & KMEANS_CAP_FOLD) != 0):
        _scoped_indexed(slot, call, operation)
        return
    var x_rows = List[Float32]()
    _gather_host(x, index, x.cols, x_rows)
    var w_rows = List[Float32]()
    if weights.rows > 0:
        if not weights.dense_host(MEMBER_F32) or weights.cols != 1 or weights.rows != x.rows:
            raise Error("TypeError: a KMeans member reads sample_weight as a dense host float32 view of X's rows")
        _gather_host(weights, index, 1, w_rows)
    call[].x = MemberMatrix(Int(x_rows.unsafe_ptr()), index.rows, x.cols)
    if weights.rows > 0:
        call[].weights = MemberMatrix(Int(w_rows.unsafe_ptr()), index.rows, 1)
    call[].row_index = MemberMatrix()
    try:
        if operation == MEMBER_OP_FIT:
            _fit(slot, call)
        elif operation == MEMBER_OP_SCORE:
            _score(slot, call)
        else:
            _query(slot, call, operation == MEMBER_OP_TRANSFORM)
    finally:
        call[].x = x
        call[].weights = weights
        call[].row_index = index
    # The gathered words outlive the operation above.
    _ = x_rows^
    _ = w_rows^


def _kmeans_member(operation: Int, address: Int) raises -> Int:
    var call = _Call(unsafe_from_address=address)
    if operation == MEMBER_OP_CAPABILITIES:
        return _capabilities(call)
    if operation == MEMBER_OP_NEW:
        if call[].param_count < _WORDS or not call[].params:
            raise Error("member ABI: a KMeans member takes eight parameters (n_clusters, init, n_init, max_iter,"
                + " tol, random_state, metric, oversampling_factor)")
        var words = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call[].params)
        if words[0] < 1.0:
            raise Error("ValueError: n_clusters must be at least 1, got " + String(Int(words[0])))
        if words[1] != 0.0 and words[1] != 1.0 and words[1] != Float64(_INIT_START):
            raise Error("member ABI: a KMeans member's init is 0 ('k-means++'), 1 ('random') or 2 (explicit"
                + " starting centers)")
        var want = _WORDS
        if words[1] == Float64(_INIT_START):
            if call[].param_count < _START_AT:
                raise Error("member ABI: explicit KMeans starting centers need their given, rows and columns words")
            want = _START_AT + member_words_f32_slots(Int(words[9]) * Int(words[10]))
        if call[].param_count != want:
            raise Error("member ABI: this KMeans member takes " + String(want) + " parameter words, got "
                + String(call[].param_count))
        if words[6] != 0.0 and words[6] != 1.0:
            raise Error("member ABI: unknown KMeans member metric code " + String(Int(words[6])))
        call[].state = _mint(words, call[].param_count)
        return 0
    if operation == MEMBER_OP_FREE:
        var held = _Slot(unsafe_from_address=_lookup(call[].state, False))
        try:
            _release_resident(held, call)
        finally:
            _ = _lookup(call[].state, True)
        call[].state = 0
        return 0
    if operation == MEMBER_OP_OUTPUTS:
        var clusters = 0
        if call[].state != 0:
            clusters = _Slot(unsafe_from_address=_lookup(call[].state, False))[].clusters
        elif call[].params and call[].param_count >= _WORDS:
            clusters = Int(MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call[].params)[0])
        call[].out_cols = clusters if call[].flags == MEMBER_OP_TRANSFORM else 1
        call[].classes = 0
        # predict: int32 cluster ids; transform: float32 distances.
        if call[].flags == MEMBER_OP_PREDICT:
            call[].out_dtype = MEMBER_I32
        elif call[].flags == MEMBER_OP_TRANSFORM:
            call[].out_dtype = MEMBER_F32
        return 0
    var slot = _Slot(unsafe_from_address=_lookup(call[].state, False))
    if ((operation == MEMBER_OP_FIT or operation == MEMBER_OP_PREDICT or operation == MEMBER_OP_TRANSFORM
            or operation == MEMBER_OP_SCORE)
            and call[].x.home != MEMBER_HOME_DEVICE and call[].row_index.rows > 0):
        _host_indexed(slot, call, operation)
    elif operation == MEMBER_OP_FIT:
        _fit(slot, call)
    elif operation == MEMBER_OP_PREDICT:
        _query(slot, call, False)
    elif operation == MEMBER_OP_TRANSFORM:
        _query(slot, call, True)
    elif operation == MEMBER_OP_SCORE:
        _score(slot, call)
    elif operation == MEMBER_OP_EXPORT:
        _export(slot, call)
    elif operation == MEMBER_OP_IMPORT:
        _import(slot, call)
    else:
        member_unsupported(_EST, operation)
    return 0


def kmeans_member_v2(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
    result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """The member entry of KMeans (schema `MEMBER_ABI`)."""
    try:
        var word = _kmeans_member(operation, member_call_from_spans(operation, aa, na, np, nf))
        if result:
            MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = word
        return 0
    except error:
        native_descriptor_report(error, context, sink)
        return 1


def kmeans_member_publish(mode: Int, vendor: Int, publisher: Int) raises:
    """Publish this image's KMeans member entry. One line in the image's
    module initialization, after its KMeans kernel table exists; a build
    with no vendor (`vendor` < 0) publishes nothing."""
    if vendor < 0:
        return
    member_publish[MEMBER_FAMILY_KMEANS](mode, vendor, publisher, kmeans_member_v2)
