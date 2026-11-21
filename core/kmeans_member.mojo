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

MEMBER_OP_NEW parameters (Float64 words, 8): [0] n_clusters, [1] init
(0 'k-means++', 1 'random'; explicit starting centers are not a word and the
class keeps its own doors for them), [2] n_init, [3] max_iter, [4] tol,
[5] random_state as the BITS of an int64 (exact for every seed:
`bitcast[DType.float64](Int64(seed))`), [6] metric (0 'euclidean' /
'l2_expanded', 1 'l2_sqrt_expanded'), [7] oversampling_factor.
Kind 0 KMeans (a clusterer, not a classifier: `y` is not read, `classes` 0).

Views: X dense row-major host float32; sample weights a host float32 vector
(MEMBER_CAN_SAMPLE_WEIGHT; negative weights are the estimator's own rule).
On an image with the resident fold entry (`KMEANS_CAP_FOLD`, the GPU images)
X may also be a DEVICE-home view (`data` = the base store id, `rows` = the
base's rows, `offset` 0) with a device-home int64 `row_index` (the store of
the fold's row numbers): the member gathers on the device, as the
estimator's own door does for a model-selection fold. Such a fit takes no
weights. A host-home row or column index, a device view on a host image, a
weighted device fit: refused in words (the caller hands gathered host words).
X is tested finite by the kernel over its own upload; a NaN or an infinity is
refused in admit_x's wording.

Outputs: predict writes the cluster ids as int32 (MEMBER_I32, rows x 1);
transform writes float32 distances (rows x n_clusters) in the model's metric.
`call.fit_output = MEMBER_OP_PREDICT` writes the training labels in the fit
call (fit_predict); `MEMBER_OP_TRANSFORM` writes the fitted rows' distances
(fit_transform: one fused call where the image has it, `KMEANS_CAP_JOINT`).
No score (the class has none).

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
    KMeansNativeDriver, KMeansFitRequest, KMeansFitResult, KMeansQueryRequest, KMeansFoldStore,
    KMEANS_CAP_JOINT, KMEANS_CAP_RESIDENT, KMEANS_CAP_FOLD,
)
from core.member_abi import (
    MemberCall, MemberMatrix, MEMBER_F32, MEMBER_F64, MEMBER_I32, MEMBER_I64, MEMBER_HOME_DEVICE,
    MEMBER_OP_CAPABILITIES, MEMBER_OP_NEW, MEMBER_OP_FREE, MEMBER_OP_FIT, MEMBER_OP_PREDICT,
    MEMBER_OP_TRANSFORM, MEMBER_OP_OUTPUTS, MEMBER_OP_EXPORT, MEMBER_OP_IMPORT,
    MEMBER_CAN_FIT, MEMBER_CAN_PREDICT, MEMBER_CAN_TRANSFORM, MEMBER_CAN_EXPORT, MEMBER_CAN_IMPORT,
    MEMBER_CAN_SAMPLE_WEIGHT, MEMBER_CAN_DEVICE_VIEWS, MEMBER_CAN_ROW_INDEX,
    MEMBER_BLOCK_SCALARS, MEMBER_BLOCK_FAMILY, MEMBER_BLOCK_BY_INDEX, MEMBER_IMPORT_DONE,
    member_publish, member_call_from_spans, member_unsupported,
)
from core.member_families import MEMBER_FAMILY_KMEANS
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.shared_slot import SharedSlot, shared_slot_lock, shared_slot_unlock

comptime _CHUNK = 256
comptime _INDEX_SPAN = 16777216
comptime _EST = "KMeans"
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
        if call[].row_index.rows > 0:
            raise Error("TypeError: a KMeans member reads a row index only over a device-home X"
                + " (a host fold is handed over as gathered host words)")
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
    return KMeansFoldStore(call[].x.data, call[].x.rows, call[].row_index.data)


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
    var weights = 0
    var weight_count = 0
    if call[].weights.rows > 0:
        if fold.active():
            raise Error("TypeError: a weighted KMeans member fit reads host views (no device fold)")
        if not call[].weights.dense_host(MEMBER_F32):
            raise Error("TypeError: a KMeans member reads sample_weight as a dense host float32 view")
        if call[].weights.rows != rows or call[].weights.cols != 1:
            raise Error("ValueError: sample_weight and X have different numbers of rows")
        weights = call[].weights.data
        weight_count = rows
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
        | MEMBER_CAN_SAMPLE_WEIGHT)
    if (call[].route.flags & KMEANS_CAP_FOLD) != 0:
        mask |= MEMBER_CAN_DEVICE_VIEWS | MEMBER_CAN_ROW_INDEX
    return mask


def _kmeans_member(operation: Int, address: Int) raises -> Int:
    var call = _Call(unsafe_from_address=address)
    if operation == MEMBER_OP_CAPABILITIES:
        return _capabilities(call)
    if operation == MEMBER_OP_NEW:
        if call[].param_count != 8 or not call[].params:
            raise Error("member ABI: a KMeans member takes eight parameters (n_clusters, init, n_init, max_iter,"
                + " tol, random_state, metric, oversampling_factor)")
        var words = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call[].params)
        if words[0] < 1.0:
            raise Error("ValueError: n_clusters must be at least 1, got " + String(Int(words[0])))
        if words[1] != 0.0 and words[1] != 1.0:
            raise Error("member ABI: a KMeans member's init is 0 ('k-means++') or 1 ('random')")
        if words[6] != 0.0 and words[6] != 1.0:
            raise Error("member ABI: unknown KMeans member metric code " + String(Int(words[6])))
        call[].state = _mint(words)
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
        elif call[].params and call[].param_count == 8:
            clusters = Int(MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call[].params)[0])
        call[].out_cols = clusters if call[].flags == MEMBER_OP_TRANSFORM else 1
        call[].classes = 0
        return 0
    var slot = _Slot(unsafe_from_address=_lookup(call[].state, False))
    if operation == MEMBER_OP_FIT:
        _fit(slot, call)
    elif operation == MEMBER_OP_PREDICT:
        _query(slot, call, False)
    elif operation == MEMBER_OP_TRANSFORM:
        _query(slot, call, True)
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
