# SPDX-License-Identifier: Apache-2.0
"""DBSCAN as a MEMBER (core/member_abi.mojo, MEMBER-ABI-V2.md).

One entry, `density_member_v2`, published by each estimators image beside
its DBSCAN kernel table (`density_member_publish`): the device image
`_mojolearn_estimators` (`DBSCAN_DEVICE_API`) and the host image
`_mojolearn_estimators_host` (`DBSCAN_HOST_API`, vendor 0). It runs the
family's existing typed RUN (`DBSCANNativeDriver`, core/dbscan_native_api.mojo:
fit, core count, core gather, the core-row queries and their resident copy)
over the DBSCAN entry of the `DeviceRoute` it is handed: `route.entry` is the
image's `dbscan_native_api_v1` entry, `route.flags` that entry's capability
word (`DBSCAN_CAP_*`, read once by the caller's UNPACK). No other route slot
is read. No kernel is touched.

Kinds (MEMBER_OP_NEW word [0]): 1 DBSCAN with prediction_data=True (the
class has `predict` only then). Kind 0 KernelDensity is reserved and refused
(the class has no predict).

MEMBER_OP_NEW parameters (Float64 words, 7): [0] 1, [1] eps, [2] min_samples,
[3] metric code (the estimator's `dbscan_metric_code`; 'precomputed' is not a
member: its words are tied to the training rows), [4] algorithm code
(`dbscan_method_code`), [5] max_iterations (0 for None), [6]
max_mbytes_per_batch (0: the provider's own budget).
A clusterer, not a classifier: `y` is not read, `classes` is 0.

Views: X dense row-major host float32, BORROWED for the call (never copied:
the entry reads the caller's words; the fitted state keeps only the core
rows the class's `components_` keeps), admitted finite by the caller's
UNPACK. Sample weights (fit only) a dense host float32 vector
(MEMBER_CAN_SAMPLE_WEIGHT). No row or column index, no device-home view:
refused in words.

Outputs: predict writes int32 labels (rows x 1): the nearest core sample's
label within eps, -1 for noise, every row -1 when the fit found no core
sample. `call.fit_output = MEMBER_OP_PREDICT` writes the training labels
(`labels_`, fit_predict) in the fit call.

Fitted-state blocks (MEMBER_OP_EXPORT / MEMBER_OP_IMPORT), in index order:
0 `MEMBER_BLOCK_FAMILY` int32 1 x rows, the training labels (empty after an
import that did not give them);
1 `MEMBER_BLOCK_FAMILY + 1` int32 1 x m, the core sample indices;
2 `MEMBER_BLOCK_FAMILY + 2` float32 m x features, the core rows (components);
3 `MEMBER_BLOCK_FAMILY + 3` int32 1 x m, the core rows' labels;
4 `MEMBER_BLOCK_SCALARS` float64 [n_iter, features].
An import needs blocks 1, 2, 3 and 4.

The fitted state is OWNED here, in a process-wide store of fixed chunks: a
slot never moves once minted, so an operation runs on its slot without the
store's lock (the lock covers mint, free and the handle lookup). A state on a
GPU image keeps its core rows resident after its first query (one upload per
fitted state, as the estimator's own resident owner), released at its next
fit, import or free. Serial by nature here: the store's bookkeeping.
"""
from std.memory import unsafe_memcpy
from std.sys.info import size_of
from cluster.dbscan_door_run import dbscan_fit_request, dbscan_core_request
from core.dbscan_native_api import (
    DBSCANNativeDriver, DBSCANFitWorkspace, DBSCANGatherRequest, DBSCANReferenceRequest, DBSCANQueryRequest,
    DBSCANResidentQuery, DBSCAN_CAP_RESIDENT, dbscan_empty_labels,
)
from core.member_abi import (
    MemberCall, MemberMatrix, MEMBER_F32, MEMBER_F64, MEMBER_I32, MEMBER_HOME_DEVICE,
    MEMBER_OP_CAPABILITIES, MEMBER_OP_NEW, MEMBER_OP_FREE, MEMBER_OP_FIT, MEMBER_OP_PREDICT,
    MEMBER_OP_OUTPUTS, MEMBER_OP_EXPORT, MEMBER_OP_IMPORT,
    MEMBER_CAN_FIT, MEMBER_CAN_PREDICT, MEMBER_CAN_EXPORT, MEMBER_CAN_IMPORT, MEMBER_CAN_SAMPLE_WEIGHT,
    MEMBER_BLOCK_SCALARS, MEMBER_BLOCK_FAMILY, MEMBER_BLOCK_BY_INDEX, MEMBER_IMPORT_DONE,
    member_publish, member_call_from_spans, member_unsupported,
)
from core.member_families import MEMBER_FAMILY_DENSITY
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.shared_slot import SharedSlot, shared_slot_lock, shared_slot_unlock

comptime _CHUNK = 256
comptime _INDEX_SPAN = 16777216
comptime _EST = "DBSCAN"
comptime _KIND_DBSCAN = 1
comptime _BLOCK_LABELS = MEMBER_BLOCK_FAMILY
comptime _BLOCK_INDICES = MEMBER_BLOCK_FAMILY + 1
comptime _BLOCK_COMPONENTS = MEMBER_BLOCK_FAMILY + 2
comptime _BLOCK_CORE_LABELS = MEMBER_BLOCK_FAMILY + 3
comptime _I32P = MutPointer[Int32, MutUntrackedOrigin]


struct DensityMemberSlot(Copyable, Movable, Defaultable):
    var generation: Int
    var live: Bool
    var eps: Float64
    var min_samples: Int
    var metric: Int
    var method: Int
    var cap: Int
    var budget: Int
    var fitted: Bool
    var features: Int
    # The resident core-row handle of a GPU image (-1: none minted).
    var handle: Int
    var labels: List[Int32]
    var indices: List[Int32]
    var components: List[Float32]
    var core_labels: List[Int32]
    var scalars: List[Float64]

    def __init__(out self):
        self.generation = 0
        self.live = False
        self.eps = 0.0
        self.min_samples = 0
        self.metric = 0
        self.method = 0
        self.cap = 0
        self.budget = 0
        self.fitted = False
        self.features = 0
        self.handle = -1
        self.labels = List[Int32]()
        self.indices = List[Int32]()
        self.components = List[Float32]()
        self.core_labels = List[Int32]()
        self.scalars = List[Float64]()

    def clear(mut self):
        self.fitted = False
        self.features = 0
        self.labels = List[Int32]()
        self.indices = List[Int32]()
        self.components = List[Float32]()
        self.core_labels = List[Int32]()
        self.scalars = List[Float64]()


struct DensityMemberStore(Defaultable, Movable):
    var synchronization: List[Int64]
    var chunks: List[List[DensityMemberSlot]]
    var free: List[Int]
    var count: Int

    def __init__(out self):
        # Side-effect-free even when the process slot's initializers race.
        self.synchronization = List[Int64](length=2, fill=0)
        self.chunks = List[List[DensityMemberSlot]]()
        self.free = List[Int]()
        self.count = 0


comptime _STORE = SharedSlot[DensityMemberStore, "MojolearnDensityMemberStoreV2", 2]
comptime _Slot = MutPointer[DensityMemberSlot, MutUntrackedOrigin]
comptime _Call = MutPointer[MemberCall, MutUntrackedOrigin]
comptime _Words = MutPointer[Float64, MutUntrackedOrigin]


def _slot_address(index: Int) raises -> Int:
    # Under the store's lock. A chunk's buffer is allocated once and never
    # grows, so the address stays valid after the lock is released.
    var store = _STORE.get_or_create_ptr()
    return Int(store[].chunks[index // _CHUNK].unsafe_ptr()) + (index % _CHUNK) * size_of[DensityMemberSlot]()


def _mint(words: _Words) raises -> Int:
    var store = _STORE.get_or_create_ptr()
    var lock = rebind[MutPointer[Int64, MutUntrackedOrigin]](store[].synchronization.unsafe_ptr())
    shared_slot_lock(lock)
    try:
        var index = -1
        if len(store[].free) > 0:
            index = store[].free.pop()
        else:
            if store[].count >= _INDEX_SPAN - 1:
                raise Error("RuntimeError: mojolearn: too many live DBSCAN member states")
            if store[].count == len(store[].chunks) * _CHUNK:
                store[].chunks.append(List[DensityMemberSlot](length=_CHUNK, fill=DensityMemberSlot()))
            index = store[].count
            store[].count += 1
        var slot = _Slot(unsafe_from_address=_slot_address(index))
        slot[].generation += 1
        slot[].live = True
        slot[].eps = words[1]
        slot[].min_samples = Int(words[2])
        slot[].metric = Int(words[3])
        slot[].method = Int(words[4])
        slot[].cap = Int(words[5])
        slot[].budget = Int(words[6])
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
            store[].free.append(index)
        return address
    finally:
        shared_slot_unlock(lock)


def _driver(call: _Call) -> DBSCANNativeDriver:
    return DBSCANNativeDriver(call[].route.entry.call)


def _release_resident(slot: _Slot, call: _Call) raises:
    """Release the slot's resident core rows, before a new fit or import and
    at free."""
    if slot[].handle < 0:
        return
    var handle = slot[].handle
    slot[].handle = -1
    _ = _driver(call).release(handle)


def _whole_x(call: _Call) raises:
    if call[].row_index.rows > 0 or call[].column_index.rows > 0:
        raise Error("TypeError: a DBSCAN member reads no row or column index"
            + " (a fold is handed over as gathered host words)")
    if call[].x.home == MEMBER_HOME_DEVICE or not call[].x.dense_host(MEMBER_F32):
        raise Error("TypeError: a DBSCAN member reads X as a dense host float32 view")


def _labels_out(call: _Call, rows: Int) raises:
    if not call[].dst.dense_host(MEMBER_I32):
        raise Error("TypeError: a DBSCAN member writes the label buffer as a dense host int32 view")
    if call[].dst.rows != rows or call[].dst.cols != 1:
        raise Error("ValueError: the label buffer holds " + String(rows) + " x 1 values")


def _fit(slot: _Slot, call: _Call) raises:
    _whole_x(call)
    var output = call[].fit_output
    var rows = call[].x.rows
    var d = call[].x.cols
    if output == MEMBER_OP_PREDICT:
        _labels_out(call, rows)
    elif output != 0:
        member_unsupported(_EST, output)
    var weights = 0
    if call[].weights.rows > 0:
        if not call[].weights.dense_host(MEMBER_F32):
            raise Error("TypeError: a DBSCAN member reads sample_weight as a dense host float32 view")
        if call[].weights.rows * call[].weights.cols != rows:
            raise Error("ValueError: sample_weight and X have different numbers of rows")
        weights = call[].weights.data
    _release_resident(slot, call)
    slot[].clear()
    slot[].labels = List[Int32](length=rows, fill=0)
    var workspace = DBSCANFitWorkspace(rows)
    var driver = _driver(call)
    var labels_address = Int(slot[].labels.unsafe_ptr())
    var request = dbscan_fit_request(call[].x.data, labels_address, weights, workspace.address(), rows, d,
        slot[].min_samples, slot[].budget, slot[].cap, slot[].method, slot[].metric, slot[].eps)
    var n_iter = driver.fit(request)
    # The class's own count and gather passes, in its order.
    var m = driver.core_count(dbscan_core_request(call[].x.data, labels_address, workspace.address(), rows, d))
    slot[].indices = List[Int32](length=m, fill=0)
    slot[].components = List[Float32](length=m * d, fill=0.0)
    slot[].core_labels = List[Int32](length=m, fill=0)
    _ = driver.core_gather(DBSCANGatherRequest(call[].x.data, labels_address, workspace.address(),
        Int(slot[].indices.unsafe_ptr()), Int(slot[].components.unsafe_ptr()), Int(slot[].core_labels.unsafe_ptr()),
        rows, d, m))
    _ = workspace^
    var scalars: List[Float64] = [Float64(n_iter), Float64(d)]
    slot[].scalars = scalars^
    slot[].features = d
    slot[].fitted = True
    if output == MEMBER_OP_PREDICT:
        unsafe_memcpy(dest=_I32P(unsafe_from_address=call[].dst.data), src=slot[].labels.unsafe_ptr(), count=rows)


def _predict(slot: _Slot, call: _Call) raises:
    if not slot[].fitted:
        raise Error("mojolearn DBSCAN.predict: this DBSCAN instance is not fitted yet; call fit first")
    _whole_x(call)
    var queries = call[].x.rows
    var d = slot[].features
    if call[].x.cols != d:
        raise Error("ValueError: DBSCAN.predict: X has " + String(call[].x.cols) + " features, the fit had "
            + String(d))
    _labels_out(call, queries)
    var m = len(slot[].core_labels)
    if m == 0:
        dbscan_empty_labels(call[].dst.data, queries)
        return
    var driver = _driver(call)
    if slot[].handle < 0 and (call[].route.flags & DBSCAN_CAP_RESIDENT) != 0:
        slot[].handle = driver.prepare(DBSCANReferenceRequest(Int(slot[].components.unsafe_ptr()),
            Int(slot[].indices.unsafe_ptr()), Int(slot[].core_labels.unsafe_ptr()), m, d, slot[].metric))
    if slot[].handle >= 0:
        _ = driver.resident(DBSCANResidentQuery(call[].x.data, call[].dst.data, slot[].handle, queries, d,
            slot[].metric, True, slot[].eps))
        return
    # The class's own scratch for the unused chosen reference (one per query).
    var chosen = List[Int32](length=queries, fill=0)
    _ = driver.query(DBSCANQueryRequest(Int(slot[].components.unsafe_ptr()), Int(slot[].indices.unsafe_ptr()),
        Int(slot[].core_labels.unsafe_ptr()), call[].x.data, call[].dst.data, Int(chosen.unsafe_ptr()), m,
        queries, d, slot[].metric, True, slot[].eps))
    _ = chosen^


def _i32_view(values: List[Int32]) -> MemberMatrix:
    var count = len(values)
    return MemberMatrix(Int(values.unsafe_ptr()) if count > 0 else 0, 1 if count > 0 else 0, count, MEMBER_I32)


def _export(slot: _Slot, call: _Call) raises:
    """A view of one fitted block. The slot owns the words; they stay valid
    until this state's next fit, import or free."""
    if not slot[].fitted:
        raise Error("NotFittedError: this DBSCAN member is not fitted yet")
    var block = call[].block
    if (call[].flags & MEMBER_BLOCK_BY_INDEX) != 0:
        if block < 0 or block > 4:
            raise Error("member ABI: a DBSCAN member has five fitted blocks")
        block = MEMBER_BLOCK_SCALARS if block == 4 else MEMBER_BLOCK_FAMILY + block
    if block == _BLOCK_LABELS:
        call[].dst = _i32_view(slot[].labels)
    elif block == _BLOCK_INDICES:
        call[].dst = _i32_view(slot[].indices)
    elif block == _BLOCK_CORE_LABELS:
        call[].dst = _i32_view(slot[].core_labels)
    elif block == _BLOCK_COMPONENTS:
        var m = len(slot[].core_labels)
        call[].dst = MemberMatrix(Int(slot[].components.unsafe_ptr()) if m > 0 else 0, m, slot[].features)
    elif block == MEMBER_BLOCK_SCALARS:
        call[].dst = MemberMatrix(Int(slot[].scalars.unsafe_ptr()), 1, 2, MEMBER_F64)
    else:
        raise Error("member ABI: a DBSCAN member has no fitted block " + String(block))
    call[].block = block
    call[].out_cols = 5


def _take_i32(call: _Call, what: String) raises -> List[Int32]:
    var count = call[].x.rows * call[].x.cols
    if count > 0 and not call[].x.dense_host(MEMBER_I32):
        raise Error("member ABI: the DBSCAN " + what + " block is a dense host int32 vector")
    var values = List[Int32](length=count, fill=0)
    if count > 0:
        unsafe_memcpy(dest=values.unsafe_ptr(), src=_I32P(unsafe_from_address=call[].x.data), count=count)
    return values^


def _import(slot: _Slot, call: _Call) raises:
    """Copy one fitted block in; MEMBER_IMPORT_DONE completes the state."""
    var block = call[].block
    if block != 0 or (call[].flags & MEMBER_IMPORT_DONE) == 0:
        _release_resident(slot, call)
        slot[].fitted = False
    if block == _BLOCK_LABELS:
        slot[].labels = _take_i32(call, "labels")
    elif block == _BLOCK_INDICES:
        slot[].indices = _take_i32(call, "core sample indices")
    elif block == _BLOCK_CORE_LABELS:
        slot[].core_labels = _take_i32(call, "core labels")
    elif block == _BLOCK_COMPONENTS:
        var count = call[].x.rows * call[].x.cols
        if count > 0 and not call[].x.dense_host(MEMBER_F32):
            raise Error("member ABI: the DBSCAN components block is a dense host float32 matrix")
        slot[].components = List[Float32](length=count, fill=0.0)
        if count > 0:
            unsafe_memcpy(dest=slot[].components.unsafe_ptr(),
                src=MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=call[].x.data), count=count)
    elif block == MEMBER_BLOCK_SCALARS:
        if not call[].x.dense_host(MEMBER_F64) or call[].x.rows * call[].x.cols != 2:
            raise Error("member ABI: the DBSCAN scalars block is two host float64 words (n_iter, features)")
        var source = _Words(unsafe_from_address=call[].x.data)
        var scalars: List[Float64] = [source[0], source[1]]
        slot[].scalars = scalars^
        slot[].features = Int(source[1])
    elif block != 0 or (call[].flags & MEMBER_IMPORT_DONE) == 0:
        raise Error("member ABI: a DBSCAN member has no fitted block " + String(block))
    if (call[].flags & MEMBER_IMPORT_DONE) == 0:
        return
    var m = len(slot[].core_labels)
    if (slot[].features <= 0 or len(slot[].scalars) != 2 or len(slot[].indices) != m
            or len(slot[].components) != m * slot[].features):
        raise Error("member ABI: a fitted DBSCAN state needs its core indices, core rows, core labels and"
            + " scalars blocks")
    slot[].fitted = True


def _density_member(operation: Int, address: Int) raises -> Int:
    var call = _Call(unsafe_from_address=address)
    if operation == MEMBER_OP_CAPABILITIES:
        return MEMBER_CAN_FIT | MEMBER_CAN_PREDICT | MEMBER_CAN_EXPORT | MEMBER_CAN_IMPORT | MEMBER_CAN_SAMPLE_WEIGHT
    if operation == MEMBER_OP_NEW:
        if not call[].params or call[].param_count != 7:
            raise Error("member ABI: a DBSCAN member takes seven parameter words (kind, eps, min_samples, metric,"
                + " algorithm, max_iterations, max_mbytes_per_batch)")
        var words = _Words(unsafe_from_address=call[].params)
        if Int(words[0]) != _KIND_DBSCAN:
            raise Error("member ABI: density member kind " + String(Int(words[0])) + " is not published")
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
        call[].out_cols = 1
        call[].classes = 0
        return 0
    var slot = _Slot(unsafe_from_address=_lookup(call[].state, False))
    if operation == MEMBER_OP_FIT:
        _fit(slot, call)
    elif operation == MEMBER_OP_PREDICT:
        _predict(slot, call)
    elif operation == MEMBER_OP_EXPORT:
        _export(slot, call)
    elif operation == MEMBER_OP_IMPORT:
        _import(slot, call)
    else:
        member_unsupported(_EST, operation)
    return 0


def density_member_v2(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
    result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """The member entry of the Density family (schema `MEMBER_ABI`)."""
    try:
        var word = _density_member(operation, member_call_from_spans(operation, aa, na, np, nf))
        if result:
            MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = word
        return 0
    except error:
        native_descriptor_report(error, context, sink)
        return 1


def density_member_publish(mode: Int, vendor: Int, publisher: Int) raises:
    """Publish this image's Density member entry. One line in the image's
    module initialization, after its DBSCAN kernel table exists; a build with
    no vendor (`vendor` < 0) publishes nothing."""
    if vendor < 0:
        return
    member_publish[MEMBER_FAMILY_DENSITY](mode, vendor, publisher, density_member_v2)
