# SPDX-License-Identifier: Apache-2.0
"""AgglomerativeClustering as a MEMBER (core/member_abi.mojo, MEMBER-ABI-V2.md).

One entry, `hierarchy_member_v2`, published by each estimators image beside
its DBSCAN kernel table (`hierarchy_member_publish`): the device image
`_mojolearn_estimators` (`DBSCAN_DEVICE_API`) and the host image
`_mojolearn_estimators_host` (vendor 0). The class's predict runs on that
image's labeled-reference queries, so its member lives there. It runs the
family's existing typed RUN over the entries of the `DeviceRoute` it is
handed: `route.entry` is the image's `dbscan_native_api_v1` entry
(core/dbscan_native_api.mojo `DBSCANNativeDriver`: prepare / resident /
query / release of the labeled reference rows, the requests of
core/agglo_native_api.mojo), `route.flags` that entry's capability word
(`DBSCAN_CAP_*`, read once by the caller's UNPACK); `route.second` is the
solver image's `hierarchy_solver_native_api_v1` entry (`AggloSolverDriver`,
cuML's single-linkage Boruvka fit, the class's own solver route). No other
route slot is read. No kernel is touched.

Kind (MEMBER_OP_NEW word [0]): 0 AgglomerativeClustering in the form that
has a predict and fits on the solver route: linkage='single', metric
'euclidean' or 'l2', no connectivity matrix (None or 'pairwise'), no
distance_threshold, compute_distances=False, prediction_data=True. Every
other instance is not a member (no predict, or the x_cluster tree route):
the caller's admit answers False and the class keeps its own doors.

MEMBER_OP_NEW parameters (Float64 words, 3): [0] 0, [1] n_clusters,
[2] 1 when the process may build the MST without the dense graph past the
pairwise row bound (`MOJOLEARN_NUMERIC_MODE=fast` on Apple, read by the
caller's UNPACK exactly as the class's `_fast_mst`; the 64-feature bound
is applied here), else 0.
A clusterer, not a classifier: `y` is not read, `classes` is 0.

Views: X dense row-major host float32, BORROWED (never copied at fit): the
solver reads the caller's words, and predict reads the training rows
through the same view, so THE CALLER KEEPS X ALIVE UNTIL THIS STATE IS
FREED OR REFITTED (the member ABI's view lifetime). MEMBER_OP_EXPORT of the
reference block copies the rows into the state once (the class's own fit
copies them into `_fit_X` on every fit); from then on the state reads its
own copy. No sample weights, no row or column index, no device-home view:
refused in words.

Outputs: predict writes int32 labels (rows x 1): the label of the nearest
training row (squared euclidean, ties to the lowest label then the lowest
row), DEVIATION 2740, the class's own rule. `call.fit_output =
MEMBER_OP_PREDICT` writes the training labels in the fit call
(fit_predict).

Fitted-state blocks (MEMBER_OP_EXPORT / MEMBER_OP_IMPORT), in index order:
0 `MEMBER_BLOCK_FAMILY` int32 1 x rows, the training labels;
1 `MEMBER_BLOCK_FAMILY + 1` int32 (rows - 1) x 2, the merge tree
(`children_`; empty after an import that did not give it);
2 `MEMBER_BLOCK_FAMILY + 2` float32 rows x features, the reference rows;
3 `MEMBER_BLOCK_SCALARS` float64 [n_clusters_, n_boruvka_rounds_,
n_connected_components_].
An import needs blocks 0, 2 and 3.

The fitted state is OWNED here, in a process-wide store of fixed chunks: a
slot never moves once minted, so an operation runs on its slot without the
store's lock (the lock covers mint, free and the handle lookup). A state on a
GPU image keeps its reference rows resident after its first query (one
upload per fitted state, as the estimator's own resident owner), released at
its next fit, import or free. Serial by nature here: the store's bookkeeping.
"""
from std.memory import unsafe_memcpy
from std.sys.info import size_of
from cluster.cfb_agglo_run import (
    agglo_check_solver_rows, agglo_check_solver_clusters, agglo_solver_request, agglo_check_reference_width,
)
from core.agglo_native_api import (
    AggloSolverDriver, AggloSolverScratch, agglo_reference_request, agglo_query_request, agglo_resident_query,
)
from core.agglo_native_state import AGGLO_PAIRWISE_MAX_ROWS
from core.dbscan_native_api import DBSCANNativeDriver, DBSCAN_CAP_RESIDENT
from core.member_abi import (
    MemberCall, MemberMatrix, MEMBER_F32, MEMBER_F64, MEMBER_I32, MEMBER_HOME_DEVICE,
    MEMBER_OP_CAPABILITIES, MEMBER_OP_NEW, MEMBER_OP_FREE, MEMBER_OP_FIT, MEMBER_OP_PREDICT,
    MEMBER_OP_OUTPUTS, MEMBER_OP_EXPORT, MEMBER_OP_IMPORT,
    MEMBER_CAN_FIT, MEMBER_CAN_PREDICT, MEMBER_CAN_EXPORT, MEMBER_CAN_IMPORT,
    MEMBER_BLOCK_SCALARS, MEMBER_BLOCK_FAMILY, MEMBER_BLOCK_BY_INDEX, MEMBER_IMPORT_DONE,
    member_publish, member_call_from_spans, member_unsupported,
)
from core.member_families import MEMBER_FAMILY_HIERARCHY
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.shared_slot import SharedSlot, shared_slot_lock, shared_slot_unlock

comptime _CHUNK = 256
comptime _INDEX_SPAN = 16777216
comptime _EST = "AgglomerativeClustering"
comptime _KIND_AGGLO = 0
comptime _BLOCK_LABELS = MEMBER_BLOCK_FAMILY
comptime _BLOCK_CHILDREN = MEMBER_BLOCK_FAMILY + 1
comptime _BLOCK_REFS = MEMBER_BLOCK_FAMILY + 2
comptime _SCALARS = 3
comptime _I32P = MutPointer[Int32, MutUntrackedOrigin]
comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]


struct HierarchyMemberSlot(Copyable, Movable, Defaultable):
    var generation: Int
    var live: Bool
    var clusters: Int
    var fast_mst: Bool
    var fitted: Bool
    var rows: Int
    var features: Int
    # The reference rows: the caller's borrowed X after a fit (`borrowed`,
    # alive until free or refit), else the state's own copy in `refs`.
    var borrowed: Int
    var refs: List[Float32]
    # The resident reference handle of a GPU image (-1: none minted).
    var handle: Int
    var labels: List[Int32]
    var children: List[Int32]
    var scalars: List[Float64]

    def __init__(out self):
        self.generation = 0
        self.live = False
        self.clusters = 0
        self.fast_mst = False
        self.fitted = False
        self.rows = 0
        self.features = 0
        self.borrowed = 0
        self.refs = List[Float32]()
        self.handle = -1
        self.labels = List[Int32]()
        self.children = List[Int32]()
        self.scalars = List[Float64]()

    def clear(mut self):
        self.fitted = False
        self.rows = 0
        self.features = 0
        self.borrowed = 0
        self.refs = List[Float32]()
        self.labels = List[Int32]()
        self.children = List[Int32]()
        self.scalars = List[Float64]()

    def reference_address(self) -> Int:
        if self.borrowed != 0:
            return self.borrowed
        return Int(self.refs.unsafe_ptr()) if len(self.refs) > 0 else 0


struct HierarchyMemberStore(Defaultable, Movable):
    var synchronization: List[Int64]
    var chunks: List[List[HierarchyMemberSlot]]
    var free: List[Int]
    var count: Int

    def __init__(out self):
        # Side-effect-free even when the process slot's initializers race.
        self.synchronization = List[Int64](length=2, fill=0)
        self.chunks = List[List[HierarchyMemberSlot]]()
        self.free = List[Int]()
        self.count = 0


comptime _STORE = SharedSlot[HierarchyMemberStore, "MojolearnHierarchyMemberStoreV2", 2]
comptime _Slot = MutPointer[HierarchyMemberSlot, MutUntrackedOrigin]
comptime _Call = MutPointer[MemberCall, MutUntrackedOrigin]
comptime _Words = MutPointer[Float64, MutUntrackedOrigin]


def _slot_address(index: Int) raises -> Int:
    # Under the store's lock. A chunk's buffer is allocated once and never
    # grows, so the address stays valid after the lock is released.
    var store = _STORE.get_or_create_ptr()
    return Int(store[].chunks[index // _CHUNK].unsafe_ptr()) + (index % _CHUNK) * size_of[HierarchyMemberSlot]()


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
                raise Error("RuntimeError: mojolearn: too many live AgglomerativeClustering member states")
            if store[].count == len(store[].chunks) * _CHUNK:
                store[].chunks.append(List[HierarchyMemberSlot](length=_CHUNK, fill=HierarchyMemberSlot()))
            index = store[].count
            store[].count += 1
        var slot = _Slot(unsafe_from_address=_slot_address(index))
        slot[].generation += 1
        slot[].live = True
        slot[].clusters = Int(words[1])
        slot[].fast_mst = words[2] != 0.0
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
    """Release the slot's resident reference rows, before a new fit or
    import, when the reference rows move, and at free."""
    if slot[].handle < 0:
        return
    var handle = slot[].handle
    slot[].handle = -1
    _ = _driver(call).release(handle)


def _whole_x(call: _Call) raises:
    if call[].row_index.rows > 0 or call[].column_index.rows > 0:
        raise Error("TypeError: an AgglomerativeClustering member reads no row or column index"
            + " (a fold is handed over as gathered host words)")
    if call[].x.home == MEMBER_HOME_DEVICE or not call[].x.dense_host(MEMBER_F32):
        raise Error("TypeError: an AgglomerativeClustering member reads X as a dense host float32 view")


def _labels_out(call: _Call, rows: Int) raises:
    if not call[].dst.dense_host(MEMBER_I32):
        raise Error("TypeError: an AgglomerativeClustering member writes the label buffer as a dense host"
            + " int32 view")
    if call[].dst.rows != rows or call[].dst.cols != 1:
        raise Error("ValueError: the label buffer holds " + String(rows) + " x 1 values")


def _fit(slot: _Slot, call: _Call) raises:
    """The class's solver route (`_agglo_legacy`): its row and cluster
    checks, one solver call on the borrowed X."""
    _whole_x(call)
    if call[].weights.rows > 0:
        raise Error("TypeError: AgglomerativeClustering takes no sample_weight")
    var output = call[].fit_output
    var rows = call[].x.rows
    var d = call[].x.cols
    if output == MEMBER_OP_PREDICT:
        _labels_out(call, rows)
    elif output != 0:
        member_unsupported(_EST, output)
    if not call[].route.has_second:
        raise Error("member ABI: an AgglomerativeClustering member's route holds no solver entry")
    agglo_check_solver_rows(rows, d, rows > AGGLO_PAIRWISE_MAX_ROWS and slot[].fast_mst and d <= 64)
    agglo_check_solver_clusters(slot[].clusters, rows)
    _release_resident(slot, call)
    slot[].clear()
    slot[].children = List[Int32](length=(rows - 1) * 2, fill=0)
    slot[].labels = List[Int32](length=rows, fill=0)
    var info = AggloSolverScratch()
    var request = agglo_solver_request(call[].x.data, Int(slot[].children.unsafe_ptr()),
        Int(slot[].labels.unsafe_ptr()), info.address(), rows, d, slot[].clusters)
    _ = AggloSolverDriver(call[].route.second.call).fit(request)
    var scalars: List[Float64] = [Float64(slot[].clusters), Float64(info.rounds()), Float64(info.components())]
    _ = info^
    slot[].scalars = scalars^
    slot[].rows = rows
    slot[].features = d
    slot[].borrowed = call[].x.data
    slot[].fitted = True
    if output == MEMBER_OP_PREDICT:
        unsafe_memcpy(dest=_I32P(unsafe_from_address=call[].dst.data), src=slot[].labels.unsafe_ptr(), count=rows)


def _predict(slot: _Slot, call: _Call) raises:
    """`_agglo_predict`: the label of each query's nearest reference row, on
    the resident copy of a GPU image (minted once per fitted state), else
    the host query with the class's per-query scratch."""
    if not slot[].fitted:
        raise Error("mojolearn AgglomerativeClustering.predict: prediction data was not stored. Fit with"
            + " AgglomerativeClustering(prediction_data=True)")
    _whole_x(call)
    var queries = call[].x.rows
    var d = slot[].features
    agglo_check_reference_width(d, call[].x.cols)
    _labels_out(call, queries)
    var driver = _driver(call)
    var refs = slot[].reference_address()
    var labels = Int(slot[].labels.unsafe_ptr())
    if slot[].handle < 0 and (call[].route.flags & DBSCAN_CAP_RESIDENT) != 0:
        slot[].handle = driver.prepare(agglo_reference_request(refs, labels, slot[].rows, d))
    if slot[].handle >= 0:
        _ = driver.resident(agglo_resident_query(call[].x.data, call[].dst.data, slot[].handle, queries, d))
        return
    var chosen = List[Int32](length=queries, fill=0)
    _ = driver.query(agglo_query_request(refs, labels, call[].x.data, call[].dst.data, Int(chosen.unsafe_ptr()),
        slot[].rows, queries, d))
    _ = chosen^


def _i32_view(values: List[Int32], rows: Int, cols: Int) -> MemberMatrix:
    if len(values) == 0:
        return MemberMatrix(0, 0, cols, MEMBER_I32)
    return MemberMatrix(Int(values.unsafe_ptr()), rows, cols, MEMBER_I32)


def _own_refs(slot: _Slot):
    """Copy the borrowed reference rows into the state, once (the class's
    `_fit_copy`); the state reads its own copy from then on. The resident
    copy (if any) holds the same words and stays."""
    if slot[].borrowed == 0:
        return
    var count = slot[].rows * slot[].features
    slot[].refs = List[Float32](length=count, fill=0.0)
    if count > 0:
        unsafe_memcpy(dest=slot[].refs.unsafe_ptr(), src=_F32P(unsafe_from_address=slot[].borrowed), count=count)
    slot[].borrowed = 0


def _export(slot: _Slot, call: _Call) raises:
    """A view of one fitted block. The slot owns the words; they stay valid
    until this state's next fit, import or free."""
    if not slot[].fitted:
        raise Error("NotFittedError: this AgglomerativeClustering member is not fitted yet")
    var block = call[].block
    if (call[].flags & MEMBER_BLOCK_BY_INDEX) != 0:
        if block < 0 or block > 3:
            raise Error("member ABI: an AgglomerativeClustering member has four fitted blocks")
        block = MEMBER_BLOCK_SCALARS if block == 3 else MEMBER_BLOCK_FAMILY + block
    if block == _BLOCK_LABELS:
        call[].dst = _i32_view(slot[].labels, 1, len(slot[].labels))
    elif block == _BLOCK_CHILDREN:
        call[].dst = _i32_view(slot[].children, len(slot[].children) // 2, 2)
    elif block == _BLOCK_REFS:
        _own_refs(slot)
        var count = slot[].rows * slot[].features
        call[].dst = MemberMatrix(Int(slot[].refs.unsafe_ptr()) if count > 0 else 0, slot[].rows, slot[].features)
    elif block == MEMBER_BLOCK_SCALARS:
        call[].dst = MemberMatrix(Int(slot[].scalars.unsafe_ptr()), 1, _SCALARS, MEMBER_F64)
    else:
        raise Error("member ABI: an AgglomerativeClustering member has no fitted block " + String(block))
    call[].block = block
    call[].out_cols = 4


def _take_i32(call: _Call, what: String) raises -> List[Int32]:
    var count = call[].x.rows * call[].x.cols
    if count > 0 and not call[].x.dense_host(MEMBER_I32):
        raise Error("member ABI: the AgglomerativeClustering " + what + " block is dense host int32")
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
    elif block == _BLOCK_CHILDREN:
        if call[].x.rows * call[].x.cols > 0 and call[].x.cols != 2:
            raise Error("member ABI: the AgglomerativeClustering tree block holds two int32 words per merge")
        slot[].children = _take_i32(call, "children")
    elif block == _BLOCK_REFS:
        var count = call[].x.rows * call[].x.cols
        if count > 0 and not call[].x.dense_host(MEMBER_F32):
            raise Error("member ABI: the AgglomerativeClustering reference block is a dense host float32 matrix")
        slot[].borrowed = 0
        slot[].refs = List[Float32](length=count, fill=0.0)
        if count > 0:
            unsafe_memcpy(dest=slot[].refs.unsafe_ptr(), src=_F32P(unsafe_from_address=call[].x.data), count=count)
        slot[].rows = call[].x.rows
        slot[].features = call[].x.cols
    elif block == MEMBER_BLOCK_SCALARS:
        if not call[].x.dense_host(MEMBER_F64) or call[].x.rows * call[].x.cols != _SCALARS:
            raise Error("member ABI: the AgglomerativeClustering scalars block is three host float64 words"
                + " (n_clusters_, n_boruvka_rounds_, n_connected_components_)")
        var source = _Words(unsafe_from_address=call[].x.data)
        var scalars: List[Float64] = [source[0], source[1], source[2]]
        slot[].scalars = scalars^
    elif block != 0 or (call[].flags & MEMBER_IMPORT_DONE) == 0:
        raise Error("member ABI: an AgglomerativeClustering member has no fitted block " + String(block))
    if (call[].flags & MEMBER_IMPORT_DONE) == 0:
        return
    if (slot[].features <= 0 or slot[].rows <= 0 or len(slot[].labels) != slot[].rows
            or len(slot[].refs) != slot[].rows * slot[].features or len(slot[].scalars) != _SCALARS):
        raise Error("member ABI: a fitted AgglomerativeClustering state needs its labels, reference rows and"
            + " scalars blocks")
    slot[].fitted = True


def _hierarchy_member(operation: Int, address: Int) raises -> Int:
    var call = _Call(unsafe_from_address=address)
    if operation == MEMBER_OP_CAPABILITIES:
        # No MEMBER_CAN_TEST_ROWS: X is admitted finite by the caller (the
        # solver and labeled-reference entries do not test it).
        return MEMBER_CAN_FIT | MEMBER_CAN_PREDICT | MEMBER_CAN_EXPORT | MEMBER_CAN_IMPORT
    if operation == MEMBER_OP_NEW:
        if not call[].params or call[].param_count != 3:
            raise Error("member ABI: an AgglomerativeClustering member takes three parameter words (kind,"
                + " n_clusters, fast_mst)")
        var words = _Words(unsafe_from_address=call[].params)
        if Int(words[0]) != _KIND_AGGLO:
            raise Error("member ABI: hierarchy member kind " + String(Int(words[0])) + " is not published")
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
        if call[].flags == MEMBER_OP_PREDICT:
            call[].out_dtype = MEMBER_I32  # int32 labels, with or without a state
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


def hierarchy_member_v2(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
    result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """The member entry of the Hierarchy family (schema `MEMBER_ABI`)."""
    try:
        var word = _hierarchy_member(operation, member_call_from_spans(operation, aa, na, np, nf))
        if result:
            MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = word
        return 0
    except error:
        native_descriptor_report(error, context, sink)
        return 1


def hierarchy_member_publish(mode: Int, vendor: Int, publisher: Int) raises:
    """Publish this image's Hierarchy member entry. One line in the image's
    module initialization, after its DBSCAN kernel table exists; a build with
    no vendor (`vendor` < 0) publishes nothing."""
    if vendor < 0:
        return
    member_publish[MEMBER_FAMILY_HIERARCHY](mode, vendor, publisher, hierarchy_member_v2)
