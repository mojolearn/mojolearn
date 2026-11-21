# SPDX-License-Identifier: Apache-2.0
"""The x_cluster estimators as MEMBERS (core/member_abi.mojo, MEMBER-ABI-V2.md).

One entry, `xcluster_member_v2`, published by each x_cluster image
(`xcluster_member_publish`): the device image `_mojolearn_x_cluster`
(bindings/_mojolearn_x_cluster.mojo, `KMEANS_VARIANTS_DEVICE_API`) and the
host image `_mojolearn_x_cluster_host` (vendor 0). It runs the family's
existing typed RUN over the entries of the `DeviceRoute` it is handed:
`route.entry` is the image's KMeans-variants entry
(core/kmeans_variants_api.mojo `KMeansVariantsDriver`: the bisecting fit and
tree descent, the centers queries, the resident centers), `route.flags` that
entry's capability word (`VARIANTS_CAP_*`, read once by the caller's
UNPACK). No other route slot is read by the kinds published here. No kernel
is touched.

Kinds (MEMBER_OP_NEW word [0]): 1 BisectingKMeans. Kinds 0 MiniBatchKMeans,
2 MeanShift, 3 OPTICS, 4 AffinityPropagation, 5 BayesianGaussianMixture are
reserved for this family and refused until published.

MEMBER_OP_NEW parameters (Float64 words):
  kind 1 BisectingKMeans (8 words): [0] 1, [1] n_clusters, [2] init
  (0 'random', 1 'k-means++'), [3] n_init, [4] max_iter, [5] tol,
  [6] random_state as the BITS of an int64 (`bitcast[DType.float64](Int64(seed))`,
  the estimator's own seed rule: None is 0, the low 63 bits of an int),
  [7] bisecting_strategy (0 'biggest_inertia', 1 'largest_cluster').
A clusterer, not a classifier: `y` is not read, `classes` is 0.

Views: X dense row-major host float32, BORROWED for the call (never copied:
the kernel reads the caller's words, as the estimator's own door reads the
admitted array); sample weights (fit only) a dense host float32 vector
(MEMBER_CAN_SAMPLE_WEIGHT; negative weights are the estimator's own rule).
X is admitted finite by the caller's UNPACK, as by the estimator's own door.
No row or column index and no device-home view: refused in words (the caller
hands gathered host words).

Outputs: predict writes the leaf cluster ids as int32 (MEMBER_I32, rows x 1;
the bisection-tree descent of `BisectingKMeans.predict`); transform writes
float32 distances to the leaf centers (rows x n_clusters); score sets
`call.value` to minus the inertia of X under the leaf centers (no weights:
the class's score counts every row once, so a weight is refused).
`call.fit_output = MEMBER_OP_PREDICT` writes the training labels in the fit
call (fit_predict); `MEMBER_OP_TRANSFORM` the fitted rows' distances.

Fitted-state blocks (MEMBER_OP_EXPORT / MEMBER_OP_IMPORT), in index order:
0 `MEMBER_BLOCK_FAMILY` float32 n_clusters x features, the leaf centers;
1 `MEMBER_BLOCK_SCALARS` float64 [inertia];
2 `MEMBER_BLOCK_FAMILY + 1` int32 1 x rows, the training labels (empty after
an import that did not give them);
3 `MEMBER_BLOCK_FAMILY + 2` float32 nodes x features, the tree centers;
4 `MEMBER_BLOCK_FAMILY + 3` int32 nodes x 3, the tree (left, right, label).
An import needs blocks 0, 1, 3 and 4.

The fitted state is OWNED here, in a process-wide store of fixed chunks: a
slot never moves once minted, so an operation runs on its slot without the
store's lock (the lock covers mint, free and the handle lookup). A state on a
GPU image keeps its leaf centers resident after its first centers query (one
upload per fitted state, as the estimator's own resident lease), released at
its next fit, import or free. Serial by nature here: the store's bookkeeping.
"""
from std.memory import bitcast, unsafe_memcpy
from std.sys.info import size_of
from core.kmeans_variants_api import (
    KMeansVariantsDriver, VariantsSpan, VariantsResultSlot, VariantsQueryRequest, BisectFitRequest,
    BisectPredictRequest, VARIANTS_CAP_RESIDENT,
)
from core.member_abi import (
    MemberCall, MemberMatrix, MEMBER_F32, MEMBER_F64, MEMBER_I32, MEMBER_HOME_DEVICE,
    MEMBER_OP_CAPABILITIES, MEMBER_OP_NEW, MEMBER_OP_FREE, MEMBER_OP_FIT, MEMBER_OP_PREDICT,
    MEMBER_OP_TRANSFORM, MEMBER_OP_SCORE, MEMBER_OP_OUTPUTS, MEMBER_OP_EXPORT, MEMBER_OP_IMPORT,
    MEMBER_CAN_FIT, MEMBER_CAN_PREDICT, MEMBER_CAN_TRANSFORM, MEMBER_CAN_SCORE, MEMBER_CAN_EXPORT,
    MEMBER_CAN_IMPORT, MEMBER_CAN_SAMPLE_WEIGHT,
    MEMBER_BLOCK_SCALARS, MEMBER_BLOCK_FAMILY, MEMBER_BLOCK_BY_INDEX, MEMBER_IMPORT_DONE,
    member_publish, member_call_from_spans, member_unsupported,
)
from core.member_families import MEMBER_FAMILY_XCLUSTER
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.shared_slot import SharedSlot, shared_slot_lock, shared_slot_unlock

comptime _CHUNK = 256
comptime _INDEX_SPAN = 16777216
comptime _KIND_BISECT = 1
comptime _BLOCK_CENTERS = MEMBER_BLOCK_FAMILY
comptime _BLOCK_LABELS = MEMBER_BLOCK_FAMILY + 1
comptime _BLOCK_TREE_CENTERS = MEMBER_BLOCK_FAMILY + 2
comptime _BLOCK_TREE_NODES = MEMBER_BLOCK_FAMILY + 3
# The query kinds of the resident centers query (x_cluster/resident.mojo XC_RES_*).
comptime _RES_PREDICT = 0
comptime _RES_SCORE = 1
comptime _RES_TRANSFORM = 2


struct XClusterMemberSlot(Copyable, Movable, Defaultable):
    var generation: Int
    var live: Bool
    var kind: Int
    var words: List[Float64]
    var fitted: Bool
    var features: Int
    # The resident leaf-center handle of a GPU image (-1: none minted).
    var handle: Int
    var centers: List[Float32]
    var labels: List[Int32]
    var tree_centers: List[Float32]
    var tree_nodes: List[Int32]
    var scalars: List[Float64]

    def __init__(out self):
        self.generation = 0
        self.live = False
        self.kind = -1
        self.words = List[Float64]()
        self.fitted = False
        self.features = 0
        self.handle = -1
        self.centers = List[Float32]()
        self.labels = List[Int32]()
        self.tree_centers = List[Float32]()
        self.tree_nodes = List[Int32]()
        self.scalars = List[Float64]()

    def clear(mut self):
        self.fitted = False
        self.features = 0
        self.centers = List[Float32]()
        self.labels = List[Int32]()
        self.tree_centers = List[Float32]()
        self.tree_nodes = List[Int32]()
        self.scalars = List[Float64]()


struct XClusterMemberStore(Defaultable, Movable):
    var synchronization: List[Int64]
    var chunks: List[List[XClusterMemberSlot]]
    var free: List[Int]
    var count: Int

    def __init__(out self):
        # Side-effect-free even when the process slot's initializers race.
        self.synchronization = List[Int64](length=2, fill=0)
        self.chunks = List[List[XClusterMemberSlot]]()
        self.free = List[Int]()
        self.count = 0


comptime _STORE = SharedSlot[XClusterMemberStore, "MojolearnXClusterMemberStoreV2", 2]
comptime _Slot = MutPointer[XClusterMemberSlot, MutUntrackedOrigin]
comptime _Call = MutPointer[MemberCall, MutUntrackedOrigin]
comptime _Words = MutPointer[Float64, MutUntrackedOrigin]


def _name(kind: Int) -> String:
    if kind == _KIND_BISECT:
        return "BisectingKMeans"
    return "an x_cluster estimator"


def _param_count(kind: Int) raises -> Int:
    """The NEW word count of a published kind; every other kind is refused."""
    if kind == _KIND_BISECT:
        return 8
    raise Error("member ABI: x_cluster member kind " + String(kind) + " is not published")


def _slot_address(index: Int) raises -> Int:
    # Under the store's lock. A chunk's buffer is allocated once and never
    # grows, so the address stays valid after the lock is released.
    var store = _STORE.get_or_create_ptr()
    return Int(store[].chunks[index // _CHUNK].unsafe_ptr()) + (index % _CHUNK) * size_of[XClusterMemberSlot]()


def _mint(words: _Words, count: Int) raises -> Int:
    var store = _STORE.get_or_create_ptr()
    var lock = rebind[MutPointer[Int64, MutUntrackedOrigin]](store[].synchronization.unsafe_ptr())
    shared_slot_lock(lock)
    try:
        var index = -1
        if len(store[].free) > 0:
            index = store[].free.pop()
        else:
            if store[].count >= _INDEX_SPAN - 1:
                raise Error("RuntimeError: mojolearn: too many live x_cluster member states")
            if store[].count == len(store[].chunks) * _CHUNK:
                store[].chunks.append(List[XClusterMemberSlot](length=_CHUNK, fill=XClusterMemberSlot()))
            index = store[].count
            store[].count += 1
        var slot = _Slot(unsafe_from_address=_slot_address(index))
        slot[].generation += 1
        slot[].live = True
        slot[].kind = Int(words[0])
        slot[].words = List[Float64](length=count, fill=0.0)
        for i in range(count):  # driver-loop: the state's parameter words
            slot[].words[i] = words[i]
        slot[].clear()
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
            slot[].handle = -1
            slot[].clear()
            slot[].words = List[Float64]()
            store[].free.append(index)
        return address
    finally:
        shared_slot_unlock(lock)


def _driver(call: _Call) -> KMeansVariantsDriver:
    return KMeansVariantsDriver(call[].route.entry.call)


def _release_resident(slot: _Slot, call: _Call) raises:
    """Release the slot's resident leaf centers (a GPU image's handle), before
    a new fit or import and at free."""
    if slot[].handle < 0:
        return
    var handle = slot[].handle
    slot[].handle = -1
    _driver(call).release(handle)


def _whole_x(call: _Call, kind: Int) raises:
    """X is a whole dense host float32 view: no index, no device view."""
    if call[].row_index.rows > 0 or call[].column_index.rows > 0:
        raise Error("TypeError: a " + _name(kind) + " member reads no row or column index"
            + " (a fold is handed over as gathered host words)")
    if call[].x.home == MEMBER_HOME_DEVICE or not call[].x.dense_host(MEMBER_F32):
        raise Error("TypeError: a " + _name(kind) + " member reads X as a dense host float32 view")


def _destination(call: _Call, rows: Int, cols: Int, dtype: Int, what: String) raises:
    if not call[].dst.dense_host(dtype):
        raise Error("TypeError: an x_cluster member writes " + what + " as a dense host "
            + (String("int32") if dtype == MEMBER_I32 else String("float32")) + " view")
    if call[].dst.rows != rows or call[].dst.cols != cols:
        raise Error("ValueError: " + what + " holds " + String(rows) + " x " + String(cols) + " values")


def _copy_i32(dst: Int, values: List[Int32], count: Int):
    if count > 0:
        unsafe_memcpy(dest=MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=dst),
            src=values.unsafe_ptr(), count=count)


def _copy_f32(dst: Int, values: List[Float32], count: Int):
    if count > 0:
        unsafe_memcpy(dest=MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=dst),
            src=values.unsafe_ptr(), count=count)


# ---------------------------------------------------------------- BisectingKMeans
def _bisect_fit(slot: _Slot, call: _Call) raises:
    var kind = slot[].kind
    _whole_x(call, kind)
    var output = call[].fit_output
    if output != 0 and output != MEMBER_OP_PREDICT and output != MEMBER_OP_TRANSFORM:
        member_unsupported(_name(kind), output)
    var rows = call[].x.rows
    var cols = call[].x.cols
    var k = Int(slot[].words[1])
    if k > rows:
        raise Error("mojolearn: n_clusters=" + String(k) + " exceeds n_samples=" + String(rows))
    var weights = VariantsSpan(0, 0)
    if call[].weights.rows > 0:
        if not call[].weights.dense_host(MEMBER_F32):
            raise Error("TypeError: a BisectingKMeans member reads sample_weight as a dense host float32 view")
        if call[].weights.rows * call[].weights.cols != rows:
            raise Error("ValueError: sample_weight and X have different numbers of rows")
        weights = VariantsSpan(call[].weights.data, rows)
    if output == MEMBER_OP_PREDICT:
        _destination(call, rows, 1, MEMBER_I32, "the label buffer")
    elif output == MEMBER_OP_TRANSFORM:
        _destination(call, rows, k, MEMBER_F32, "the distance buffer")
    _release_resident(slot, call)
    slot[].clear()
    var w = slot[].words.unsafe_ptr()
    var request = BisectFitRequest(VariantsSpan(call[].x.data, rows * cols), weights, rows, cols, k,
        Int(w[3]), w[2] == 0.0, Int(w[4]), Int(bitcast[DType.int64](w[6])), w[7] != 0.0, w[5])
    var result = VariantsResultSlot()
    _driver(call).bisect_fit(request, result)
    # The entry's result lists become the state's blocks (no copy).
    swap(slot[].centers, result.storage[0].centers)
    swap(slot[].labels, result.storage[0].labels)
    swap(slot[].tree_centers, result.storage[0].tree_centers)
    swap(slot[].tree_nodes, result.storage[0].tree_nodes)
    var scalars: List[Float64] = [result.storage[0].inertia]
    slot[].scalars = scalars^
    slot[].features = cols
    slot[].fitted = True
    if output == MEMBER_OP_PREDICT:
        _copy_i32(call[].dst.data, slot[].labels, rows)
    elif output == MEMBER_OP_TRANSFORM:
        _centers_query(slot, call, _RES_TRANSFORM)


def _bisect_predict(slot: _Slot, call: _Call) raises:
    var rows = _query_rows(slot, call)
    _destination(call, rows, 1, MEMBER_I32, "the label buffer")
    var request = BisectPredictRequest(VariantsSpan(call[].x.data, rows * slot[].features),
        VariantsSpan(Int(slot[].tree_centers.unsafe_ptr()), len(slot[].tree_centers)),
        VariantsSpan(Int(slot[].tree_nodes.unsafe_ptr()) if len(slot[].tree_nodes) > 0 else 0,
            len(slot[].tree_nodes)), rows, slot[].features)
    var result = VariantsResultSlot()
    _driver(call).bisect_predict(request, result)
    _copy_i32(call[].dst.data, result.storage[0].labels, rows)


# ---------------------------------------------------------------- centers queries
def _query_rows(slot: _Slot, call: _Call) raises -> Int:
    if not slot[].fitted:
        raise Error("RuntimeError: this estimator is not fitted yet")
    _whole_x(call, slot[].kind)
    if call[].x.cols != slot[].features:
        raise Error("mojolearn: X has " + String(call[].x.cols) + " features, " + _name(slot[].kind)
            + " was fitted with " + String(slot[].features))
    return call[].x.rows


def _centers_query(slot: _Slot, call: _Call, query: Int) raises:
    """predict / transform / score against the state's centers: the resident
    copy on a GPU image (minted once per fitted state), else the centers
    handed over, as the estimator's own `_CentersMixin` door."""
    var rows = _query_rows(slot, call)
    var k = len(slot[].centers) // slot[].features if slot[].features > 0 else 0
    var destination = 0
    if query == _RES_PREDICT:
        _destination(call, rows, 1, MEMBER_I32, "the label buffer")
        destination = call[].dst.data
    elif query == _RES_TRANSFORM:
        _destination(call, rows, k, MEMBER_F32, "the distance buffer")
        destination = call[].dst.data
    elif call[].weights.rows > 0:
        raise Error("TypeError: the score of a " + _name(slot[].kind) + " member counts every row once"
            + " (no sample_weight)")
    var driver = _driver(call)
    var result = VariantsResultSlot()
    if slot[].handle < 0 and (call[].route.flags & VARIANTS_CAP_RESIDENT) != 0:
        slot[].handle = driver.prepare(Int(slot[].centers.unsafe_ptr()), k, slot[].features)
    if slot[].handle >= 0:
        var total = driver.resident_query(slot[].handle, query, call[].x.data, rows, k, slot[].features,
            destination, result)
        if query == _RES_SCORE:
            call[].value = -total
        return
    var request = VariantsQueryRequest(VariantsSpan(call[].x.data, rows * slot[].features),
        VariantsSpan(Int(slot[].centers.unsafe_ptr()), len(slot[].centers)), rows, k, slot[].features,
        query == _RES_SCORE)
    driver.query(request, query == _RES_TRANSFORM, result)
    if query == _RES_SCORE:
        call[].value = -result.storage[0].inertia
    elif query == _RES_TRANSFORM:
        _copy_f32(destination, result.storage[0].distances, rows * k)
    else:
        _copy_i32(destination, result.storage[0].labels, rows)


# ---------------------------------------------------------------- fitted blocks
def _block_count(kind: Int) -> Int:
    return 5 if kind == _KIND_BISECT else 0


def _block_at(kind: Int, index: Int) raises -> Int:
    if index < 0 or index >= _block_count(kind):
        raise Error("member ABI: a " + _name(kind) + " member has " + String(_block_count(kind))
            + " fitted blocks")
    if index == 0:
        return _BLOCK_CENTERS
    if index == 1:
        return MEMBER_BLOCK_SCALARS
    if index == 2:
        return _BLOCK_LABELS
    if index == 3:
        return _BLOCK_TREE_CENTERS
    return _BLOCK_TREE_NODES


def _f32_view(values: List[Float32], rows: Int, cols: Int) -> MemberMatrix:
    if len(values) == 0:
        return MemberMatrix(0, 0, cols)
    return MemberMatrix(Int(values.unsafe_ptr()), rows, cols)


def _i32_view(values: List[Int32], rows: Int, cols: Int) -> MemberMatrix:
    if len(values) == 0:
        return MemberMatrix(0, 0, cols, MEMBER_I32)
    return MemberMatrix(Int(values.unsafe_ptr()), rows, cols, MEMBER_I32)


def _export(slot: _Slot, call: _Call) raises:
    """A view of one fitted block. The slot owns the words; they stay valid
    until this state's next fit, import or free."""
    var kind = slot[].kind
    if not slot[].fitted:
        raise Error("NotFittedError: this " + _name(kind) + " member is not fitted yet")
    var block = call[].block
    if (call[].flags & MEMBER_BLOCK_BY_INDEX) != 0:
        block = _block_at(kind, block)
    var d = slot[].features
    if block == _BLOCK_CENTERS:
        call[].dst = _f32_view(slot[].centers, len(slot[].centers) // d, d)
    elif block == MEMBER_BLOCK_SCALARS:
        call[].dst = MemberMatrix(Int(slot[].scalars.unsafe_ptr()), 1, len(slot[].scalars), MEMBER_F64)
    elif block == _BLOCK_LABELS:
        call[].dst = _i32_view(slot[].labels, 1, len(slot[].labels))
    elif block == _BLOCK_TREE_CENTERS and kind == _KIND_BISECT:
        call[].dst = _f32_view(slot[].tree_centers, len(slot[].tree_centers) // d, d)
    elif block == _BLOCK_TREE_NODES and kind == _KIND_BISECT:
        call[].dst = _i32_view(slot[].tree_nodes, len(slot[].tree_nodes) // 3, 3)
    else:
        raise Error("member ABI: a " + _name(kind) + " member has no fitted block " + String(block))
    call[].block = block
    call[].out_cols = _block_count(kind)


def _take_f32(call: _Call, what: String) raises -> List[Float32]:
    if not call[].x.dense_host(MEMBER_F32):
        raise Error("member ABI: the x_cluster " + what + " block is a dense host float32 matrix")
    var count = call[].x.rows * call[].x.cols
    var values = List[Float32](length=count, fill=0.0)
    if count > 0:
        unsafe_memcpy(dest=values.unsafe_ptr(),
            src=MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=call[].x.data), count=count)
    return values^


def _take_i32(call: _Call, what: String) raises -> List[Int32]:
    if not call[].x.dense_host(MEMBER_I32):
        raise Error("member ABI: the x_cluster " + what + " block is a dense host int32 matrix")
    var count = call[].x.rows * call[].x.cols
    var values = List[Int32](length=count, fill=0)
    if count > 0:
        unsafe_memcpy(dest=values.unsafe_ptr(),
            src=MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=call[].x.data), count=count)
    return values^


def _import(slot: _Slot, call: _Call) raises:
    """Copy one fitted block in; MEMBER_IMPORT_DONE completes the state."""
    var kind = slot[].kind
    var block = call[].block
    if block == _BLOCK_CENTERS:
        _release_resident(slot, call)
        slot[].fitted = False
        slot[].centers = _take_f32(call, "centers")
        slot[].features = call[].x.cols
        slot[].labels = List[Int32]()
    elif block == MEMBER_BLOCK_SCALARS:
        if not call[].x.dense_host(MEMBER_F64) or call[].x.rows * call[].x.cols != 1:
            raise Error("member ABI: the " + _name(kind) + " scalars block is one host float64 word (inertia)")
        var scalars: List[Float64] = [_Words(unsafe_from_address=call[].x.data)[0]]
        slot[].scalars = scalars^
    elif block == _BLOCK_LABELS:
        slot[].labels = _take_i32(call, "labels")
    elif block == _BLOCK_TREE_CENTERS and kind == _KIND_BISECT:
        slot[].tree_centers = _take_f32(call, "tree centers")
    elif block == _BLOCK_TREE_NODES and kind == _KIND_BISECT:
        if call[].x.cols != 3:
            raise Error("member ABI: the BisectingKMeans tree block holds three int32 words per node")
        slot[].tree_nodes = _take_i32(call, "tree")
    elif block != 0 or (call[].flags & MEMBER_IMPORT_DONE) == 0:
        raise Error("member ABI: a " + _name(kind) + " member has no fitted block " + String(block))
    if (call[].flags & MEMBER_IMPORT_DONE) == 0:
        return
    var d = slot[].features
    var k = Int(slot[].words[1])
    if d == 0 or len(slot[].centers) != k * d or len(slot[].scalars) != 1:
        raise Error("member ABI: a fitted " + _name(kind) + " state needs its centers and scalars blocks")
    if kind == _KIND_BISECT and (len(slot[].tree_nodes) == 0
            or len(slot[].tree_centers) != (len(slot[].tree_nodes) // 3) * d):
        raise Error("member ABI: a fitted BisectingKMeans state needs its tree centers and tree blocks")
    slot[].fitted = True


# ---------------------------------------------------------------- the entry
def _capabilities(kind: Int) -> Int:
    # Every published kind (state 0: the family's union, today kind 1 only).
    _ = kind
    return (MEMBER_CAN_FIT | MEMBER_CAN_PREDICT | MEMBER_CAN_TRANSFORM | MEMBER_CAN_SCORE | MEMBER_CAN_EXPORT
        | MEMBER_CAN_IMPORT | MEMBER_CAN_SAMPLE_WEIGHT)


def _xcluster_member(operation: Int, address: Int) raises -> Int:
    var call = _Call(unsafe_from_address=address)
    if operation == MEMBER_OP_CAPABILITIES:
        if call[].state != 0:
            return _capabilities(_Slot(unsafe_from_address=_lookup(call[].state, False))[].kind)
        return _capabilities(-1)
    if operation == MEMBER_OP_NEW:
        if not call[].params or call[].param_count < 1:
            raise Error("member ABI: an x_cluster member's first parameter word is its kind")
        var words = _Words(unsafe_from_address=call[].params)
        var kind = Int(words[0])
        var count = _param_count(kind)
        if call[].param_count != count:
            raise Error("member ABI: a " + _name(kind) + " member takes " + String(count) + " parameter words")
        if words[1] < 1.0:
            raise Error("ValueError: n_clusters must be at least 1, got " + String(Int(words[1])))
        if kind == _KIND_BISECT and words[3] < 1.0:
            raise Error("ValueError: n_init must be at least 1, got " + String(Int(words[3])))
        call[].state = _mint(words, count)
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
            clusters = Int(_Slot(unsafe_from_address=_lookup(call[].state, False))[].words[1])
        elif call[].params and call[].param_count >= 2:
            clusters = Int(_Words(unsafe_from_address=call[].params)[1])
        call[].out_cols = clusters if call[].flags == MEMBER_OP_TRANSFORM else 1
        call[].classes = 0
        return 0
    var slot = _Slot(unsafe_from_address=_lookup(call[].state, False))
    var kind = slot[].kind
    if operation == MEMBER_OP_FIT and kind == _KIND_BISECT:
        _bisect_fit(slot, call)
    elif operation == MEMBER_OP_PREDICT and kind == _KIND_BISECT:
        _bisect_predict(slot, call)
    elif operation == MEMBER_OP_TRANSFORM:
        _centers_query(slot, call, _RES_TRANSFORM)
    elif operation == MEMBER_OP_SCORE:
        _centers_query(slot, call, _RES_SCORE)
    elif operation == MEMBER_OP_EXPORT:
        _export(slot, call)
    elif operation == MEMBER_OP_IMPORT:
        _import(slot, call)
    else:
        member_unsupported(_name(kind), operation)
    return 0


def xcluster_member_v2(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
    result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """The member entry of the x_cluster estimators (schema `MEMBER_ABI`)."""
    try:
        var word = _xcluster_member(operation, member_call_from_spans(operation, aa, na, np, nf))
        if result:
            MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = word
        return 0
    except error:
        native_descriptor_report(error, context, sink)
        return 1


def xcluster_member_publish(mode: Int, vendor: Int, publisher: Int) raises:
    """Publish this image's x_cluster member entry. One line in the image's
    module initialization, after its variants kernel table exists; a build
    with no vendor (`vendor` < 0) publishes nothing."""
    if vendor < 0:
        return
    member_publish[MEMBER_FAMILY_XCLUSTER](mode, vendor, publisher, xcluster_member_v2)
