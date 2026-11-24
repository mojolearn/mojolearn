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

Kind (MEMBER_OP_NEW word [0]): 0 AgglomerativeClustering, every setting
its fit admits (lane mf-clu6). The class's two numerical owners are kept as
the class picks them (`AggloOptions.solver_route`): the solver route
(single linkage, euclidean, no connectivity matrix, no distance_threshold,
compute_distances=False: cuML's Boruvka fit, `route.second`) and the
x_cluster tree route (`_agglo_x`: every linkage and metric, a connectivity
matrix, a threshold, the per-merge distances: `route.third`, the x_cluster
image's `hierarchy_cluster_native_api_v1` entry, and for metric='cosine'
`route.fourth`, the x_neighbors image's entry for the zero-row test). A
fit with prediction_data=False, or a tree route whose linkage/metric has no
predict (`agglo_x_predicts`), keeps no reference rows: its predict is the
class's own refusal.

MEMBER_OP_NEW parameters (Float64 words, 17 + data): [0] 0, [1] n_clusters
(0 when None), [2] 1 when the process may build the MST without the dense
graph past the pairwise row bound (`MOJOLEARN_NUMERIC_MODE=fast` on Apple,
read by the caller's UNPACK exactly as the class's `_fast_mst`; the
64-feature bound is applied here), else 0, [3] linkage code, [4] metric
code, [5] constrained (a connectivity matrix), [6] has distance_threshold,
[7] compute_distances, [8] prediction_data, [9] full-tree mode, [10] the
threshold, [11] n_clusters given, [12] the connectivity matrix's side (> 0;
-1 not square; 0 unread), [13] its float32 word count, [14] its edge count,
[15] its edge mode (1 dense, 2 sparse COO triples), [16] the byte count of
the refusal its admission raised (0: none); then the float32 words packed
two per word (core/cluster_member_words.mojo), then the refusal's bytes. The
matrix is read once at UNPACK; its refusals (a callable included: the class
never calls it) and its shape check against X are raised at fit, where the
class's `_agglo_x` makes them.
A clusterer, not a classifier: `y` is not read, `classes` is 0.

Views: X dense row-major host float32, BORROWED (never copied at fit): the
solver reads the caller's words, and predict reads the training rows
through the same view, so THE CALLER KEEPS X ALIVE UNTIL THIS STATE IS
FREED OR REFITTED (the member ABI's view lifetime). MEMBER_OP_EXPORT of the
reference block copies the rows into the state once (the class's own fit
copies them into `_fit_X` on every fit); from then on the state reads its
own copy. A row index (lane vdf-scope-cluster8) names a model-selection
fold: taken on the device into an x_prep store slot that the solver, the
x_cluster tree entry, the x_neighbors zero-row test and the labeled-reference
predict read as a negative X word (the state keeps it as THE FITTED-ROWS
SLOT), or gathered on the host cores, the state owning the words. No sample
weights, no column index, no device-home view: refused in words.

Outputs: predict writes int32 labels (rows x 1): the label of the nearest
training row (squared euclidean, ties to the lowest label then the lowest
row), DEVIATION 2740, the class's own rule. `call.fit_output =
MEMBER_OP_PREDICT` writes the training labels in the fit call
(fit_predict).

Fitted-state blocks (MEMBER_OP_EXPORT / MEMBER_OP_IMPORT), in index order:
0 `MEMBER_BLOCK_FAMILY` int32 1 x rows, the training labels;
1 `MEMBER_BLOCK_FAMILY + 1` int32 merges x 2, the merge tree
(`children_`; empty after an import that did not give it);
2 `MEMBER_BLOCK_FAMILY + 2` float32 rows x features, the reference rows
(0 x features when the fit keeps none);
3 `MEMBER_BLOCK_SCALARS` float64 [n_clusters_, n_boruvka_rounds_,
n_connected_components_];
4 `MEMBER_BLOCK_FAMILY + 3` float32 1 x merges, the per-merge distances
(export only; empty unless the tree route computed them).
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
    agglo_check_tree_rows, agglo_check_precomputed, agglo_checks_zero_rows, agglo_check_zero_rows,
    agglo_zero_rows_request, agglo_check_requested_clusters, agglo_cluster_request, agglo_publishes_distances,
    agglo_cut_counts, agglo_fitted_clusters, agglo_x_predicts,
)
from core.agglo_native_api import (
    AggloSolverDriver, AggloSolverScratch, agglo_reference_request, agglo_query_request, agglo_resident_query,
    AggloClusterDriver, AggloClusterScratch, AggloZeroRowsDriver, AggloZeroRowsScratch,
)
from core.agglo_native_state import AGGLO_PAIRWISE_MAX_ROWS, AggloOptions, agglo_tree_plan, agglo_predicts
from core.cluster_member_words import (
    member_words_f32_slots, member_words_byte_slots, member_words_take_f32, member_words_take_text,
)
from core.neighbors_native_driver import XnDriver
from core.dbscan_native_api import DBSCANNativeDriver, DBSCAN_CAP_RESIDENT
from core.member_abi import (
    MemberCall, MemberMatrix, MEMBER_F32, MEMBER_F64, MEMBER_I32, MEMBER_HOME_DEVICE,
    MEMBER_OP_CAPABILITIES, MEMBER_OP_NEW, MEMBER_OP_FREE, MEMBER_OP_FIT, MEMBER_OP_PREDICT,
    MEMBER_OP_OUTPUTS, MEMBER_OP_EXPORT, MEMBER_OP_IMPORT,
    MEMBER_CAN_FIT, MEMBER_CAN_PREDICT, MEMBER_CAN_EXPORT, MEMBER_CAN_IMPORT,
    MEMBER_BLOCK_SCALARS, MEMBER_BLOCK_FAMILY, MEMBER_BLOCK_BY_INDEX, MEMBER_IMPORT_DONE,
    member_publish, member_call_from_spans, member_unsupported,
)
from core.member_abi import MEMBER_BORROWS_FIT, MEMBER_EXPORT_OWN
from core.member_families import MEMBER_FAMILY_HIERARCHY
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.shared_slot import SharedSlot, shared_slot_lock, shared_slot_unlock
from core.member_abi import MEMBER_CAN_ROW_INDEX, MEMBER_CAN_SCOPE
from core.member_fold_x import MemberFoldX, member_fold_x_open, member_fold_x_close, member_fold_store_ready
from core.member_scope_rows import MemberScopeRows, member_scope_rows_in
from core.device_route import DEVICE_GPU
from core.entry_table import EntryRef, entry_ref_unresolved
from core.fold_x_word import fold_x_word
from core.dbscan_native_api import DBSCAN_CAP_FOLD_X
from core.native_descriptor_fixed_call import native_descriptor_call_fixed
from hierarchy.solver_descriptor_schema import HIERARCHY_SOLVER_CAPABILITIES, HIERARCHY_SOLVER_CAP_FOLD_X
from hierarchy.cluster_descriptor_schema import HIERARCHY_CLUSTER_CAPABILITIES, HIERARCHY_CLUSTER_CAP_FOLD_X
from bindings.arena_cache_run import arena_cache_free, arena_store_read

comptime _CHUNK = 256
comptime _INDEX_SPAN = 16777216
comptime _EST = "AgglomerativeClustering"
comptime _KIND_AGGLO = 0
comptime _BLOCK_LABELS = MEMBER_BLOCK_FAMILY
comptime _BLOCK_CHILDREN = MEMBER_BLOCK_FAMILY + 1
comptime _BLOCK_REFS = MEMBER_BLOCK_FAMILY + 2
comptime _BLOCK_DISTANCES = MEMBER_BLOCK_FAMILY + 3
comptime _SCALARS = 3
comptime _HEADER = 17
comptime _I32P = MutPointer[Int32, MutUntrackedOrigin]
comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]


struct HierarchyMemberSlot(Copyable, Movable, Defaultable):
    var generation: Int
    var live: Bool
    var clusters: Int
    var fast_mst: Bool
    # The class's settings (NEW words 3-11) and its connectivity matrix as
    # read at UNPACK (words 12-16 and the data behind them).
    var linkage: Int
    var metric: Int
    var constrained: Bool
    var has_threshold: Bool
    var compute_distances: Bool
    var prediction_data: Bool
    var full_tree: Int
    var threshold: Float64
    var has_clusters: Bool
    var conn_side: Int
    var conn_edges: Int
    var conn_mode: Int
    var conn_words: List[Float32]
    var conn_error: String
    # Whether the fitted state keeps reference rows (predict is served).
    var predicting: Bool
    var fitted: Bool
    var rows: Int
    var features: Int
    # The reference rows: the caller's borrowed X after a fit (`borrowed`,
    # alive until free or refit), else the state's own copy in `refs`.
    var borrowed: Int
    var refs: List[Float32]
    # THE FITTED-ROWS SLOT (lane vdf-scope-cluster8): a fold fit taken on the device keeps its reference rows as
    # that slot of the x_prep store `rows_store` (-1: none; `borrowed` / `refs` then hold none).
    var rows_slot: Int
    var rows_store: EntryRef
    # The resident reference handle of a GPU image (-1: none minted).
    var handle: Int
    var labels: List[Int32]
    var children: List[Int32]
    var distances: List[Float32]
    var scalars: List[Float64]

    def __init__(out self):
        self.generation = 0
        self.live = False
        self.clusters = 0
        self.fast_mst = False
        self.linkage = 3
        self.metric = -1
        self.constrained = False
        self.has_threshold = False
        self.compute_distances = False
        self.prediction_data = True
        self.full_tree = 0
        self.threshold = 0.0
        self.has_clusters = True
        self.conn_side = 0
        self.conn_edges = 0
        self.conn_mode = 0
        self.conn_words = List[Float32]()
        self.conn_error = String("")
        self.predicting = False
        self.fitted = False
        self.rows = 0
        self.features = 0
        self.borrowed = 0
        self.refs = List[Float32]()
        self.rows_slot = -1
        self.rows_store = entry_ref_unresolved()
        self.handle = -1
        self.labels = List[Int32]()
        self.children = List[Int32]()
        self.distances = List[Float32]()
        self.scalars = List[Float64]()

    def options(self) -> AggloOptions:
        return AggloOptions(self.linkage, self.metric, self.constrained, self.has_threshold,
            self.compute_distances, self.prediction_data, self.full_tree)

    def clear(mut self):
        self.fitted = False
        self.predicting = False
        self.rows = 0
        self.features = 0
        self.borrowed = 0
        self.refs = List[Float32]()
        self.labels = List[Int32]()
        self.children = List[Int32]()
        self.distances = List[Float32]()
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
        slot[].linkage = Int(words[3])
        slot[].metric = Int(words[4])
        slot[].constrained = words[5] != 0.0
        slot[].has_threshold = words[6] != 0.0
        slot[].compute_distances = words[7] != 0.0
        slot[].prediction_data = words[8] != 0.0
        slot[].full_tree = Int(words[9])
        slot[].threshold = words[10]
        slot[].has_clusters = words[11] != 0.0
        slot[].conn_side = Int(words[12])
        var conn_count = Int(words[13])
        slot[].conn_edges = Int(words[14])
        slot[].conn_mode = Int(words[15])
        slot[].conn_words = member_words_take_f32(Int(words), _HEADER, conn_count)
        slot[].conn_error = member_words_take_text(Int(words), _HEADER + member_words_f32_slots(conn_count),
            Int(words[16]))
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


def _drop_rows_slot(slot: _Slot) raises:
    """Free THE FITTED-ROWS SLOT (a new fit, an import, a free or the first export of the rows replace it). The
    resident reference set (if any) holds its own device copy."""
    if slot[].rows_slot >= 0:
        var held = slot[].rows_slot
        slot[].rows_slot = -1
        arena_cache_free(slot[].rows_store, held)


def _caps(entry: EntryRef, operation: Int) -> Int:
    """An entry's capability word (0 when it answers none)."""
    try:
        var a: Array[Int, 0] = []
        var p: Array[Int, 0] = []
        var f: Array[Float64, 0] = []
        return native_descriptor_call_fixed(entry.call, operation, a, p, f)
    except:
        return 0


def _fold_ready(slot: _Slot, call: _Call) -> Bool:
    """THE FOLD X WORD on this fit (lane vdf-scope-cluster8): a GPU route whose image ships the x_prep store, whose
    fit entry takes the word (the solver route: HIERARCHY_SOLVER_CAP_FOLD_X on `route.second`; the tree route:
    HIERARCHY_CLUSTER_CAP_FOLD_X on `route.third`, with the device x_neighbors zero-row test of a GPU route), and,
    when the fit keeps prediction data, whose predict entry reads the word too (DBSCAN_CAP_FOLD_X with
    DBSCAN_CAP_RESIDENT on `route.entry`)."""
    if call[].route.device != DEVICE_GPU or not member_fold_store_ready(call):
        return False
    if slot[].prediction_data and ((call[].route.flags & DBSCAN_CAP_FOLD_X) == 0
            or (call[].route.flags & DBSCAN_CAP_RESIDENT) == 0):
        return False
    if slot[].options().solver_route():
        return call[].route.has_second and (
            _caps(call[].route.second, HIERARCHY_SOLVER_CAPABILITIES) & HIERARCHY_SOLVER_CAP_FOLD_X) != 0
    return call[].route.has_third and (
        _caps(call[].route.third, HIERARCHY_CLUSTER_CAPABILITIES) & HIERARCHY_CLUSTER_CAP_FOLD_X) != 0


def _whole_x(call: _Call) raises:
    if call[].row_index.rows > 0 or call[].column_index.rows > 0:
        raise Error("TypeError: an AgglomerativeClustering member reads no column index")
    if call[].x.home == MEMBER_HOME_DEVICE or not call[].x.dense_host(MEMBER_F32):
        raise Error("TypeError: an AgglomerativeClustering member reads X as a dense host float32 view")


def _labels_out(call: _Call, rows: Int) raises:
    if not call[].dst.dense_host(MEMBER_I32):
        raise Error("TypeError: an AgglomerativeClustering member writes the label buffer as a dense host"
            + " int32 view")
    if call[].dst.rows != rows or call[].dst.cols != 1:
        raise Error("ValueError: the label buffer holds " + String(rows) + " x 1 values")


def _fit(slot: _Slot, call: _Call) raises:
    """`_agglo_fit`: the class's route predicate picks its owner, the solver
    route (`_agglo_legacy`) or the x_cluster tree route (`_agglo_x`), both on
    the borrowed X. Under a row index (MEMBER_CAN_ROW_INDEX, lane vdf-scope-cluster8) the fold's rows: taken ON
    THE DEVICE into a slot of the x_prep store (`_fold_ready`: the scope's one copy of X), whose negative word the
    fit entry reads and which the state keeps as THE FITTED-ROWS SLOT for predict; else gathered on the host
    cores, the state owning the gathered words."""
    if call[].weights.rows > 0:
        raise Error("TypeError: AgglomerativeClustering takes no sample_weight")
    if call[].column_index.rows > 0 or call[].x.home == MEMBER_HOME_DEVICE or not call[].x.dense_host(MEMBER_F32):
        raise Error("TypeError: an AgglomerativeClustering member reads X as a dense host float32 view, no column"
            + " index")
    var output = call[].fit_output
    var rows = call[].row_index.rows if call[].row_index.rows > 0 else call[].x.rows
    if output == MEMBER_OP_PREDICT:
        _labels_out(call, rows)
    elif output != 0:
        member_unsupported(_EST, output)
    if call[].row_index.rows == 0:
        _fit_words(slot, call, call[].x.data, call[].x.rows, call[].x.cols)
        slot[].borrowed = call[].x.data if slot[].predicting else 0
    elif _fold_ready(slot, call):
        var fold = MemberFoldX()
        member_fold_x_open(call, fold, _EST)
        var kept = False
        try:
            _fit_words(slot, call, fold.word(), fold.rows, fold.cols)
            if slot[].predicting:
                slot[].rows_slot = fold.slot
                slot[].rows_store = fold.store
                fold.slot = -1  # the state owns the slot now
                kept = True
        finally:
            if not kept:
                member_fold_x_close(fold)
    else:
        var gathered = member_scope_rows_in(call[], call[].route.entry, "x_prep", False, _EST)
        _fit_words(slot, call, gathered.source, gathered.rows, gathered.cols)
        if slot[].predicting:
            swap(slot[].refs, gathered.gathered)  # the state owns the gathered rows (nothing borrowed)
        _ = gathered.rows  # the gathered rows stay alive through the fit
    if output == MEMBER_OP_PREDICT:
        unsafe_memcpy(dest=_I32P(unsafe_from_address=call[].dst.data), src=slot[].labels.unsafe_ptr(), count=rows)


def _fit_words(slot: _Slot, call: _Call, x_word: Int, rows: Int, d: Int) raises:
    """The class's route predicate picks its owner over X at `x_word` (`rows` x `d`: host words, or a negative
    fold X word); the state borrows nothing here (the caller sets `borrowed` / the kept rows)."""
    if slot[].options().solver_route():
        _fit_solver(slot, call, x_word, rows, d)
    else:
        _fit_tree(slot, call, x_word, rows, d)
    slot[].borrowed = 0


def _fit_solver(slot: _Slot, call: _Call, x_word: Int, rows: Int, d: Int) raises:
    """`_agglo_legacy`: its row and cluster checks, one solver call."""
    if not call[].route.has_second:
        raise Error("member ABI: an AgglomerativeClustering member's route holds no solver entry")
    agglo_check_solver_rows(rows, d, rows > AGGLO_PAIRWISE_MAX_ROWS and slot[].fast_mst and d <= 64)
    agglo_check_solver_clusters(slot[].clusters, rows)
    _release_resident(slot, call)
    _drop_rows_slot(slot)
    slot[].clear()
    slot[].children = List[Int32](length=(rows - 1) * 2, fill=0)
    slot[].labels = List[Int32](length=rows, fill=0)
    var info = AggloSolverScratch()
    var request = agglo_solver_request(x_word, Int(slot[].children.unsafe_ptr()),
        Int(slot[].labels.unsafe_ptr()), info.address(), rows, d, slot[].clusters)
    _ = AggloSolverDriver(call[].route.second.call).fit(request)
    var scalars: List[Float64] = [Float64(slot[].clusters), Float64(info.rounds()), Float64(info.components())]
    _ = info^
    slot[].scalars = scalars^
    slot[].rows = rows
    slot[].features = d
    slot[].predicting = slot[].prediction_data
    slot[].fitted = True


def _connectivity_check(slot: _Slot, rows: Int) raises:
    """`hierarchy_connectivity`'s refusals, in its order, for the matrix read
    at UNPACK: the shape against X (a sparse matrix's shape is read before
    its storage), then what its admission refused."""
    if slot[].conn_side < 0 or (slot[].conn_side > 0 and slot[].conn_side != rows):
        raise Error("Wrong shape for connectivity matrix when X has " + String(rows) + " samples")
    if slot[].conn_error.byte_length() > 0:
        raise Error(slot[].conn_error)
    if slot[].conn_side != rows:
        raise Error("Wrong shape for connectivity matrix when X has " + String(rows) + " samples")


def _fit_tree(slot: _Slot, call: _Call, x_word: Int, n: Int, d: Int) raises:
    """`_agglo_x`: the x_cluster route's checks in its order, one cluster fit
    (`route.third`), the zero-row test for cosine on `route.fourth` (both read a
    negative fold X word on the device)."""
    var options = slot[].options()
    if not call[].route.has_third:
        raise Error("member ABI: an AgglomerativeClustering member's route holds no x_cluster entry")
    agglo_check_tree_rows(n)
    agglo_check_precomputed(options.metric, n, d)
    if agglo_checks_zero_rows(options.metric):
        agglo_check_zero_rows(d == 0)
        if not call[].route.has_fourth:
            raise Error("member ABI: an AgglomerativeClustering member's route holds no x_neighbors entry")
        var zrow = AggloZeroRowsScratch(n)
        AggloZeroRowsDriver(XnDriver(call[].route.fourth.call)).run(
            agglo_zero_rows_request(x_word, zrow.address(), n, d))
        agglo_check_zero_rows(zrow.any_zero())
        _ = zrow^
    var requested = 0
    if slot[].has_clusters:
        requested = slot[].clusters
        agglo_check_requested_clusters(requested, n)
    var plan = agglo_tree_plan(options, n, requested)
    var merges = plan.merges
    var conn_address = 0
    var conn_count = 0
    var edges = 0
    var edge_mode = 0
    if options.constrained:
        _connectivity_check(slot, n)
        conn_count = len(slot[].conn_words)
        conn_address = Int(slot[].conn_words.unsafe_ptr()) if conn_count > 0 else 0
        edges = slot[].conn_edges
        edge_mode = slot[].conn_mode
    var threshold = slot[].threshold if options.has_threshold else Float64(0)
    _release_resident(slot, call)
    _drop_rows_slot(slot)
    slot[].clear()
    slot[].children = List[Int32](length=merges * 2, fill=0)
    slot[].labels = List[Int32](length=n, fill=0)
    slot[].distances = List[Float32](length=merges, fill=0.0)
    var scratch = AggloClusterScratch()
    var request = agglo_cluster_request(x_word, conn_address, Int(slot[].children.unsafe_ptr()),
        Int(slot[].labels.unsafe_ptr()), Int(slot[].distances.unsafe_ptr()), scratch.address(), n * d, conn_count,
        n, d, options, edges, plan, edge_mode, threshold)
    AggloClusterDriver(call[].route.third.call).fit(request)
    _ = len(slot[].conn_words)  # the edge words the fit read stay alive through it
    var components = scratch.components()
    var cut_clusters = scratch.clusters() if agglo_cut_counts(options, plan) else 0
    _ = scratch^
    if not agglo_publishes_distances(options):
        slot[].distances = List[Float32]()
    var clusters = agglo_fitted_clusters(options, plan, cut_clusters, requested)
    # n_boruvka_rounds_ is -1 on this route, as `_agglo_x` publishes it.
    var scalars: List[Float64] = [Float64(clusters), -1.0, Float64(components)]
    slot[].scalars = scalars^
    slot[].rows = n
    slot[].features = d
    slot[].predicting = agglo_x_predicts(options)
    slot[].fitted = True


def _predict(slot: _Slot, call: _Call) raises:
    """`_agglo_predict`: the label of each query's nearest reference row, on
    the resident copy of a GPU image (minted once per fitted state), else
    the host query with the class's per-query scratch."""
    if slot[].fitted and not agglo_predicts(slot[].linkage, slot[].metric):
        # `_agglo_predict`'s first refusal, in its words.
        raise Error("NotImplementedError: mojolearn AgglomerativeClustering.predict: only single linkage under"
            + " the euclidean metric has a predict rule (DEVIATION 2740)")
    if not slot[].fitted or not slot[].predicting:
        raise Error("mojolearn AgglomerativeClustering.predict: prediction data was not stored. Fit with"
            + " AgglomerativeClustering(prediction_data=True)")
    if call[].row_index.rows == 0:
        _whole_x(call)
        _predict_words(slot, call, call[].x.data, call[].x.rows, call[].x.cols)
        return
    if call[].column_index.rows > 0 or call[].x.home == MEMBER_HOME_DEVICE or not call[].x.dense_host(MEMBER_F32):
        raise Error("TypeError: an AgglomerativeClustering member reads X as a dense host float32 view, no column"
            + " index")
    # MEMBER_CAN_ROW_INDEX (lane vdf-scope-cluster8): the fold's queries taken on the device (DBSCAN_CAP_FOLD_X)
    # out of the scope's one copy of X, else gathered on the host cores.
    if (call[].route.device == DEVICE_GPU and (call[].route.flags & DBSCAN_CAP_FOLD_X) != 0
            and (call[].route.flags & DBSCAN_CAP_RESIDENT) != 0 and member_fold_store_ready(call)):
        var fold = MemberFoldX()
        member_fold_x_open(call, fold, _EST)
        try:
            _predict_words(slot, call, fold.word(), fold.rows, fold.cols)
        finally:
            member_fold_x_close(fold)
        return
    var gathered = member_scope_rows_in(call[], call[].route.entry, "x_prep", False, _EST)
    _predict_words(slot, call, gathered.source, gathered.rows, gathered.cols)
    _ = gathered.rows  # the gathered queries stay alive through the query call


def _predict_words(slot: _Slot, call: _Call, q_word: Int, queries: Int, cols: Int) raises:
    """`_agglo_predict` over the queries at `q_word` (host words, or a negative fold X word); the reference rows
    are the state's host words or THE FITTED-ROWS SLOT (its negative word, DBSCAN_CAP_FOLD_X)."""
    var d = slot[].features
    agglo_check_reference_width(d, cols)
    _labels_out(call, queries)
    var driver = _driver(call)
    var refs = fold_x_word(slot[].rows_slot) if slot[].rows_slot >= 0 else slot[].reference_address()
    var labels = Int(slot[].labels.unsafe_ptr())
    if slot[].handle < 0 and (call[].route.flags & DBSCAN_CAP_RESIDENT) != 0:
        slot[].handle = driver.prepare(agglo_reference_request(refs, labels, slot[].rows, d))
    if slot[].handle >= 0:
        _ = driver.resident(agglo_resident_query(q_word, call[].dst.data, slot[].handle, queries, d))
        return
    if refs < 0 or q_word < 0:
        raise Error("member ABI: an AgglomerativeClustering state's device-resident rows need a resident predict"
            + " entry (DBSCAN_CAP_RESIDENT)")
    var chosen = List[Int32](length=queries, fill=0)
    _ = driver.query(agglo_query_request(refs, labels, q_word, call[].dst.data, Int(chosen.unsafe_ptr()),
        slot[].rows, queries, d))
    _ = chosen^


def _i32_view(values: List[Int32], rows: Int, cols: Int) -> MemberMatrix:
    if len(values) == 0:
        return MemberMatrix(0, 0, cols, MEMBER_I32)
    return MemberMatrix(Int(values.unsafe_ptr()), rows, cols, MEMBER_I32)


def _own_refs(slot: _Slot) raises:
    """Copy the borrowed reference rows into the state, once (the class's
    `_fit_copy`); the state reads its own copy from then on. The resident
    copy (if any) holds the same words and stays. THE FITTED-ROWS SLOT is read
    home once (`arena_store_read`) and freed."""
    if slot[].rows_slot >= 0:
        var words = slot[].rows * slot[].features
        slot[].refs = List[Float32](length=words, fill=0.0)
        if words > 0:
            arena_store_read(slot[].rows_store, slot[].rows_slot, Int(slot[].refs.unsafe_ptr()), words)
        _drop_rows_slot(slot)
        slot[].borrowed = 0
        return
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
    if (call[].flags & MEMBER_EXPORT_OWN) != 0:
        # THE LIFETIME RULE: the training rows, once; no view is answered.
        _own_refs(slot)
        return
    var block = call[].block
    if (call[].flags & MEMBER_BLOCK_BY_INDEX) != 0:
        if block < 0 or block > 4:
            raise Error("member ABI: an AgglomerativeClustering member has five fitted blocks")
        block = MEMBER_BLOCK_SCALARS if block == 3 else (_BLOCK_DISTANCES if block == 4 else MEMBER_BLOCK_FAMILY + block)
    if block == _BLOCK_LABELS:
        call[].dst = _i32_view(slot[].labels, 1, len(slot[].labels))
    elif block == _BLOCK_CHILDREN:
        call[].dst = _i32_view(slot[].children, len(slot[].children) // 2, 2)
    elif block == _BLOCK_REFS:
        _own_refs(slot)
        if not slot[].predicting:
            # No reference rows kept (the class publishes `_fit_X` None).
            call[].dst = MemberMatrix(0, 0, slot[].features)
        else:
            var count = slot[].rows * slot[].features
            call[].dst = MemberMatrix(Int(slot[].refs.unsafe_ptr()) if count > 0 else 0, slot[].rows,
                slot[].features)
    elif block == MEMBER_BLOCK_SCALARS:
        call[].dst = MemberMatrix(Int(slot[].scalars.unsafe_ptr()), 1, _SCALARS, MEMBER_F64)
    elif block == _BLOCK_DISTANCES:
        var merges = len(slot[].distances)
        call[].dst = MemberMatrix(Int(slot[].distances.unsafe_ptr()) if merges > 0 else 0, 1 if merges > 0 else 0,
            merges)
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
        _drop_rows_slot(slot)
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
    slot[].predicting = True
    slot[].fitted = True


def _hierarchy_member(operation: Int, address: Int) raises -> Int:
    var call = _Call(unsafe_from_address=address)
    if operation == MEMBER_OP_CAPABILITIES:
        # No MEMBER_CAN_TEST_ROWS: X is admitted finite by the caller (the
        # solver and labeled-reference entries do not test it).
        # MEMBER_BORROWS_FIT: a fitted state reads the caller's X (its
        # training rows) until an export of the rows block, or
        # MEMBER_EXPORT_OWN, copied them.
        # MEMBER_CAN_ROW_INDEX (lane vdf-scope-cluster8): a fold's rows (gathered on the host cores, or taken on
        # the device: MEMBER_CAN_SCOPE on a GPU route whose image ships the x_prep store).
        var caps = (MEMBER_CAN_FIT | MEMBER_CAN_PREDICT | MEMBER_CAN_EXPORT | MEMBER_CAN_IMPORT | MEMBER_BORROWS_FIT
            | MEMBER_CAN_ROW_INDEX)
        if call[].route.device == DEVICE_GPU and member_fold_store_ready(call):
            caps |= MEMBER_CAN_SCOPE
        return caps
    if operation == MEMBER_OP_NEW:
        if not call[].params or call[].param_count < 1:
            raise Error("member ABI: a hierarchy member's first parameter word is its kind")
        var words = _Words(unsafe_from_address=call[].params)
        if Int(words[0]) != _KIND_AGGLO:
            raise Error("member ABI: hierarchy member kind " + String(Int(words[0])) + " is not published")
        if call[].param_count < _HEADER or call[].param_count != (_HEADER
                + member_words_f32_slots(Int(words[13])) + member_words_byte_slots(Int(words[16]))):
            raise Error("member ABI: an AgglomerativeClustering member takes seventeen parameter words (kind,"
                + " n_clusters, fast_mst, linkage, metric, constrained, has_threshold, compute_distances,"
                + " prediction_data, full_tree, threshold, has_clusters, and the connectivity side, word count,"
                + " edges, mode and refusal length), then the connectivity words and the refusal's bytes")
        call[].state = _mint(words)
        return 0
    if operation == MEMBER_OP_FREE:
        var held = _Slot(unsafe_from_address=_lookup(call[].state, False))
        try:
            _release_resident(held, call)
            _drop_rows_slot(held)
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
