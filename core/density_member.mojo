# SPDX-License-Identifier: Apache-2.0
"""DBSCAN and KernelDensity as MEMBERS (core/member_abi.mojo, MEMBER-ABI-V2.md).

One entry, `density_member_v2`, published by each estimators image beside
its DBSCAN kernel table (`density_member_publish`): the device image
`_mojolearn_estimators` (`DBSCAN_DEVICE_API`) and the host image
`_mojolearn_estimators_host` (`DBSCAN_HOST_API`, vendor 0). It runs the
family's existing typed RUN (`DBSCANNativeDriver`, core/dbscan_native_api.mojo:
fit, core count, core gather, the core-row queries and their resident copy)
over the DBSCAN entry of the `DeviceRoute` it is handed: `route.entry` is the
image's `dbscan_native_api_v1` entry, `route.flags` that entry's capability
word (`DBSCAN_CAP_*`, read once by the caller's UNPACK). For KernelDensity
`route.entry` is the image's `kde_native_api_v1` entry (core/kde_native_api.mojo
`KDENativeDriver`: prepare / release / query / resident query, the requests
of neighbors/neighbors_kde_run.mojo) and `route.flags` its `KDE_CAP_*` word.
No other route slot is read. No kernel is touched.

Kinds (MEMBER_OP_NEW word [0]): 1 DBSCAN (its `predict` serves only a fit
with prediction_data=True; otherwise it is the class's own refusal);
0 KernelDensity (fit, score_samples, score:
no predict; a grid search over its bandwidth scores it).

KernelDensity NEW parameters (Float64 words, 4): [0] 0, [1] bandwidth (a
float; a rule name is the class's own refusal), [2] kernel (0 'gaussian',
1 'tophat', 2 'epanechnikov', 3 'exponential', 4 'linear', 5 'cosine'),
[3] metric (0 'euclidean', 1 'l2', 2 'sqeuclidean', 3 'l1', 4 'cityblock',
5 'manhattan', 6 'chebyshev', 7 'cosine', 8 'minkowski'). metric_params
is not a word: the class's validation admits only a single Minkowski p=2,
which its door then does not hand to the entry either. A kernel or metric
name outside these tables is -1 and its text follows (lane mf-clu6):
[4] the kernel name's byte count, [5] the metric name's, then each name's
UTF-8 bytes packed from a fresh word (core/cluster_member_words.mojo). The
names are built once at NEW and handed to the entry by name, as the class's
door hands them, so an unknown name meets the entry's own refusal. KernelDensity is LAZY: fit BORROWS X
and the sample weights (no copy: the class's own fit copies X into `_x` on
every fit), so THE CALLER KEEPS X AND THE WEIGHTS ALIVE AND UNCHANGED UNTIL
THIS STATE IS FREED OR REFITTED (the member ABI's view lifetime); the first
MEMBER_OP_EXPORT of the training block copies X (and the weights) into the
state once, and the state reads its own copy from then on. Weights are
checked as the class checks them (finite, none negative, one positive).
score_samples writes float32 log densities (rows x 1); score sets
`call.value` to their sum, folded by the entry (the class's `score`). A
state on a KDE_CAP_RESIDENT image uploads its training set once (first
query) and releases it at refit, import or free. Blocks by index: 0
`MEMBER_BLOCK_FAMILY + 4` float32 rows x features, the training rows;
1 `MEMBER_BLOCK_FAMILY + 5` float32 1 x rows, the weights (empty: none);
2 `MEMBER_BLOCK_SCALARS` float64 [rows, features]. An import needs 0 and 2.

DBSCAN MEMBER_OP_NEW parameters (Float64 words, 7 or 8): [0] 1, [1] eps,
[2] min_samples, [3] metric code (the estimator's `dbscan_metric_code`;
'precomputed' reads X as the n x n distance matrix and is refused in the
door's words when it is not square or holds a negative distance), [4]
algorithm code (`dbscan_method_code`), [5] max_iterations (0 for None), [6]
max_mbytes_per_batch (0: the provider's own budget), [7] prediction_data
(0/1; absent: 1). The estimator's settings refuse prediction_data=True with
'precomputed', so a precomputed member never predicts (lane mf-clu6).
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
from cluster.dbscan_door_run import dbscan_fit_request, dbscan_core_request, dbscan_check_x
from core.admitted_input import admitted_check_weights_f32
from core.admitted_refusals import admitted_finite_refuse
from core.host_finite_scan import host_all_finite_f32
from core.cluster_member_words import member_words_byte_slots, member_words_take_text
from core.kde_native_api import KDENativeDriver, KDE_CAP_RESIDENT, KDE_CAP_DEV_STORE
from core.member_abi import MEMBER_CAN_ROW_INDEX, MEMBER_CAN_SCOPE, MEMBER_SCOPE_NONE, MEMBER_I64
from core.member_scope_rows import MemberScopeRows, member_scope_rows, member_scope_rows_done
from core.admitted_rows_gather import admitted_gather_rows_f32, ARG_OUT_OF_RANGE
from neighbors.neighbors_kde_run import (
    kde_fit_key, kde_prepare_request, kde_query_request, kde_resident_query, kde_refuse_unfitted, kde_refuse_features,
)
from core.dbscan_native_api import (
    DBSCANNativeDriver, DBSCANFitWorkspace, DBSCANGatherRequest, DBSCANReferenceRequest, DBSCANQueryRequest,
    DBSCANResidentQuery, DBSCAN_CAP_RESIDENT, dbscan_empty_labels,
)
from core.member_abi import (
    MemberCall, MemberMatrix, MEMBER_F32, MEMBER_F64, MEMBER_I32, MEMBER_HOME_DEVICE,
    MEMBER_OP_CAPABILITIES, MEMBER_OP_NEW, MEMBER_OP_FREE, MEMBER_OP_FIT, MEMBER_OP_PREDICT,
    MEMBER_OP_OUTPUTS, MEMBER_OP_EXPORT, MEMBER_OP_IMPORT, MEMBER_OP_SCORE, MEMBER_OP_SCORE_SAMPLES,
    MEMBER_CAN_FIT, MEMBER_CAN_PREDICT, MEMBER_CAN_EXPORT, MEMBER_CAN_IMPORT, MEMBER_CAN_SAMPLE_WEIGHT,
    MEMBER_CAN_SCORE, MEMBER_CAN_SCORE_SAMPLES,
    MEMBER_BLOCK_SCALARS, MEMBER_BLOCK_FAMILY, MEMBER_BLOCK_BY_INDEX, MEMBER_IMPORT_DONE,
    member_publish, member_call_from_spans, member_unsupported,
)
from core.member_abi import MEMBER_BORROWS_FIT, MEMBER_EXPORT_OWN
from core.member_families import MEMBER_FAMILY_DENSITY
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.shared_slot import SharedSlot, shared_slot_lock, shared_slot_unlock

comptime _CHUNK = 256
comptime _INDEX_SPAN = 16777216
comptime _EST = "DBSCAN"
comptime _KIND_DBSCAN = 1
comptime _KIND_KDE = 0
comptime _BLOCK_TRAIN = MEMBER_BLOCK_FAMILY + 4
comptime _BLOCK_WEIGHTS = MEMBER_BLOCK_FAMILY + 5
comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _BLOCK_LABELS = MEMBER_BLOCK_FAMILY
comptime _BLOCK_INDICES = MEMBER_BLOCK_FAMILY + 1
comptime _BLOCK_COMPONENTS = MEMBER_BLOCK_FAMILY + 2
comptime _BLOCK_CORE_LABELS = MEMBER_BLOCK_FAMILY + 3
comptime _I32P = MutPointer[Int32, MutUntrackedOrigin]


struct DensityMemberSlot(Copyable, Movable, Defaultable):
    var generation: Int
    var live: Bool
    var kind: Int
    # KernelDensity: its words, the names built once at NEW, the training
    # rows and weights (borrowed addresses after a fit, else the state's own
    # copies below; 0: none) and their counts.
    var bandwidth: Float64
    var kernel: String
    var kde_metric: String
    var train: Int
    var weights: Int
    var rows: Int
    var train_owned: List[Float32]
    var weights_owned: List[Float32]
    var eps: Float64
    var min_samples: Int
    var metric: Int
    var method: Int
    var cap: Int
    var budget: Int
    var predicting: Bool
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
        self.kind = -1
        self.bandwidth = 0.0
        self.kernel = String("")
        self.kde_metric = String("")
        self.train = 0
        self.weights = 0
        self.rows = 0
        self.train_owned = List[Float32]()
        self.weights_owned = List[Float32]()
        self.eps = 0.0
        self.min_samples = 0
        self.metric = 0
        self.method = 0
        self.cap = 0
        self.budget = 0
        self.predicting = True
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
        self.train = 0
        self.weights = 0
        self.rows = 0
        self.train_owned = List[Float32]()
        self.weights_owned = List[Float32]()
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
                raise Error("RuntimeError: mojolearn: too many live DBSCAN member states")
            if store[].count == len(store[].chunks) * _CHUNK:
                store[].chunks.append(List[DensityMemberSlot](length=_CHUNK, fill=DensityMemberSlot()))
            index = store[].count
            store[].count += 1
        var slot = _Slot(unsafe_from_address=_slot_address(index))
        slot[].generation += 1
        slot[].live = True
        slot[].kind = Int(words[0])
        if slot[].kind == _KIND_KDE:
            slot[].bandwidth = words[1]
            if words[2] < 0.0 or words[3] < 0.0:
                var kernel_bytes = Int(words[4])
                var metric_at = 6 + member_words_byte_slots(kernel_bytes)
                slot[].kernel = (member_words_take_text(Int(words), 6, kernel_bytes) if words[2] < 0.0
                    else _kde_kernel_name(Int(words[2])))
                slot[].kde_metric = (member_words_take_text(Int(words), metric_at, Int(words[5]))
                    if words[3] < 0.0 else _kde_metric_name(Int(words[3])))
            else:
                slot[].kernel = _kde_kernel_name(Int(words[2]))
                slot[].kde_metric = _kde_metric_name(Int(words[3]))
        else:
            slot[].eps = words[1]
            slot[].min_samples = Int(words[2])
            slot[].metric = Int(words[3])
            slot[].method = Int(words[4])
            slot[].cap = Int(words[5])
            slot[].budget = Int(words[6])
            slot[].predicting = count < 8 or words[7] != 0.0
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


def _kde_kernel_name(code: Int) raises -> String:
    var names: List[String] = ["gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine"]
    if code < 0 or code >= len(names):
        raise Error("member ABI: KernelDensity kernel code " + String(code) + " is not one of 0 .. 5")
    return names[code]


def _kde_metric_name(code: Int) raises -> String:
    var names: List[String] = ["euclidean", "l2", "sqeuclidean", "l1", "cityblock", "manhattan", "chebyshev",
        "cosine", "minkowski"]
    if code < 0 or code >= len(names):
        raise Error("member ABI: KernelDensity metric code " + String(code) + " is not one of 0 .. 8")
    return names[code]


def _release_resident(slot: _Slot, call: _Call) raises:
    """Release the slot's resident copy (DBSCAN core rows, KernelDensity
    training set), before a new fit or import and at free."""
    if slot[].handle < 0:
        return
    var handle = slot[].handle
    slot[].handle = -1
    if slot[].kind == _KIND_KDE:
        KDENativeDriver(call[].route.entry.call).release(handle)
        return
    _ = _driver(call).release(handle)


def _whole_x(call: _Call, what: String = _EST) raises:
    if call[].row_index.rows > 0 or call[].column_index.rows > 0:
        raise Error("TypeError: a " + what + " member reads no row or column index"
            + " (a fold is handed over as gathered host words)")
    if call[].x.home == MEMBER_HOME_DEVICE or not call[].x.dense_host(MEMBER_F32):
        raise Error("TypeError: a " + what + " member reads X as a dense host float32 view")


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
    # metric='precomputed': the door's square and sign refusals, before the
    # weights as `_dbscan_fit` makes them (the caller admitted X finite, so
    # the finite scan is not repeated).
    dbscan_check_x(call[].x.data, rows, d, True, slot[].metric)
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
    if not slot[].predicting:
        # The class's own refusal (bindings/dbscan_public_native_unpack.mojo
        # `dbscan_predict_components`).
        raise Error("mojolearn DBSCAN.predict: prediction data was not stored. Fit with "
                    + "DBSCAN(prediction_data=True), which keeps the core samples this "
                    + "rule needs (DEVIATION 2740)")
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


# ---------------------------------------------------------------- KernelDensity
def _kde_fit(slot: _Slot, call: _Call) raises:
    """`KernelDensity.fit`: the weights checks, then the training set is the
    borrowed view (no copy). Under a row index: `_kde_fit_indexed`."""
    if call[].row_index.rows > 0:
        _kde_fit_indexed(slot, call)
        return
    _whole_x(call, "KernelDensity")
    if call[].fit_output != 0:
        member_unsupported("KernelDensity", call[].fit_output)
    var rows = call[].x.rows
    var d = call[].x.cols
    var weights = 0
    if call[].weights.rows > 0:
        if not call[].weights.dense_host(MEMBER_F32):
            raise Error("TypeError: a KernelDensity member reads sample_weight as a dense host float32 view")
        if call[].weights.rows * call[].weights.cols != rows:
            raise Error("sample_weight.shape == (" + String(call[].weights.rows * call[].weights.cols)
                + ",), expected (" + String(rows) + ",)!")
        _kde_weights_check(_F32P(unsafe_from_address=call[].weights.data), rows)
        weights = call[].weights.data
    _release_resident(slot, call)
    slot[].clear()
    slot[].train = call[].x.data
    slot[].weights = weights
    slot[].rows = rows
    slot[].features = d
    var scalars: List[Float64] = [Float64(rows), Float64(d)]
    slot[].scalars = scalars^
    slot[].fitted = True


comptime _KDE_PREFIX = "estimators"


def _kde_weights_check(w: _F32P, rows: Int) raises:
    if not host_all_finite_f32(w, rows):
        raise Error("Input sample_weight contains NaN or infinity.")
    # The class's parallel sign test (a vector weight is not written).
    if not admitted_check_weights_f32(w, w, rows, False):
        raise Error("Sample weights must contain at least one non-zero number.")


def _kde_gather(call: _Call, base: Int, width: Int, dst: Int) raises:
    """Rows `call.row_index` of the host base at `base` (`call.x.rows` rows of `width` words) into `dst`, in row
    tasks across the host's cores; an index outside X is refused."""
    var index = call[].row_index
    var status: Int
    if index.dtype == MEMBER_I32:
        status = admitted_gather_rows_f32[DType.int32](base, call[].x.rows, width, index.data, index.rows, dst)
    else:
        status = admitted_gather_rows_f32[DType.int64](base, call[].x.rows, width, index.data, index.rows, dst)
    if status == ARG_OUT_OF_RANGE:
        raise Error("IndexError: a KernelDensity member's row index holds a row outside X's "
            + String(call[].x.rows) + " rows")


def _kde_store_ok(slot: _Slot, call: _Call) -> Bool:
    """THE RESIDENT-STORE ROUTE of a KernelDensity call (lane vdf-scope-cluster2b): the GPU image with its store
    (`KDE_CAP_DEV_STORE`, the transport entry in `route.second`), a resident image (`KDE_CAP_RESIDENT`), the
    caller's scope with X's pin, and a metric without a host rule on the rows (cosine and sqeuclidean keep the
    host form, kde/resident_fit.mojo `kde_host_rules_needed`)."""
    var flags = call[].route.flags
    return ((flags & KDE_CAP_DEV_STORE) != 0 and (flags & KDE_CAP_RESIDENT) != 0 and call[].route.has_second
        and call[].scope != MEMBER_SCOPE_NONE and call[].pin != 0
        and slot[].kde_metric != "cosine" and slot[].kde_metric != "sqeuclidean")


def _kde_fit_indexed(slot: _Slot, call: _Call) raises:
    """`KernelDensity.fit` over the rows of a host row index (MEMBER_CAN_ROW_INDEX, lane vdf-scope-cluster2b). The
    state OWNS its training rows here (gathered on the host cores, and the weights' rows likewise), so nothing is
    borrowed past the call. On the resident-store route the same rows are also taken ON THE DEVICE out of the
    scope's one copy of X and prepared from that slot at once (`prepare_store`: no upload of the fold); the
    prepared handle holds the same words the first query would have uploaded, so the same bits."""
    if call[].fit_output != 0:
        member_unsupported("KernelDensity", call[].fit_output)
    if call[].column_index.rows > 0 or call[].x.home == MEMBER_HOME_DEVICE:
        raise Error("TypeError: a KernelDensity member reads X as a dense host float32 view, no column index")
    if not call[].x.dense_host(MEMBER_F32):
        raise Error("TypeError: a KernelDensity member reads X as a dense host float32 view")
    var index = call[].row_index
    if index.home == MEMBER_HOME_DEVICE or (index.dtype != MEMBER_I32 and index.dtype != MEMBER_I64):
        raise Error("TypeError: a KernelDensity member reads a host int32 or int64 row index")
    var rows = index.rows
    var d = call[].x.cols
    var weights = List[Float32]()
    if call[].weights.rows > 0:
        if not call[].weights.dense_host(MEMBER_F32):
            raise Error("TypeError: a KernelDensity member reads sample_weight as a dense host float32 view")
        if call[].weights.rows * call[].weights.cols != call[].x.rows:
            raise Error("sample_weight.shape == (" + String(call[].weights.rows * call[].weights.cols)
                + ",), expected (" + String(call[].x.rows) + ",)!")
        weights = List[Float32](unsafe_uninit_length=max(rows, 1))
        _kde_gather(call, call[].weights.data, 1, Int(weights.unsafe_ptr()))
        _kde_weights_check(_F32P(unsafe_from_address=Int(weights.unsafe_ptr())), rows)
    _release_resident(slot, call)
    slot[].clear()
    # The state's own training rows: the index's rows of X, gathered on the host cores.
    slot[].train_owned = List[Float32](unsafe_uninit_length=max(rows * d, 1))
    _kde_gather(call, call[].x.data, d, Int(slot[].train_owned.unsafe_ptr()))
    slot[].train = Int(slot[].train_owned.unsafe_ptr()) if rows > 0 else 0
    if call[].weights.rows > 0:
        slot[].weights_owned = weights^
        slot[].weights = Int(slot[].weights_owned.unsafe_ptr())
    slot[].rows = rows
    slot[].features = d
    var scalars: List[Float64] = [Float64(rows), Float64(d)]
    slot[].scalars = scalars^
    slot[].fitted = True
    if not _kde_store_ok(slot, call):
        return
    var device = member_scope_rows(call[], _KDE_PREFIX, True, "KernelDensity")
    try:
        var key = kde_fit_key(device.source, rows, d, slot[].weights, slot[].kernel, slot[].kde_metric,
            slot[].bandwidth)
        slot[].handle = KDENativeDriver(call[].route.entry.call).prepare_store(
            kde_prepare_request(key, slot[].weights != 0))
    finally:
        member_scope_rows_done(call[], device)


def _kde_scores(slot: _Slot, call: _Call, want_total: Bool) raises:
    """`KernelDensity._scores`: the resident training set of a
    KDE_CAP_RESIDENT image (prepared once per fitted state), else the direct
    query over the state's training rows. Under a row index (lane
    vdf-scope-cluster2b) the queries are the index's rows: taken on the device
    out of the scope's one copy of X on the resident-store route
    (`resident_store`, no upload), else gathered on the host cores."""
    if not slot[].fitted:
        kde_refuse_unfitted()
    var x = _kde_query_rows(slot, call)
    try:
        _kde_scores_rows(slot, call, x, want_total)
    finally:
        member_scope_rows_done(call[], x)


def _kde_query_rows(slot: _Slot, call: _Call) raises -> MemberScopeRows:
    if call[].row_index.rows == 0:
        _whole_x(call, "KernelDensity")
        return MemberScopeRows(call[].x.data, call[].x.rows, call[].x.cols, False, List[Float32]())
    if call[].column_index.rows > 0 or call[].x.home == MEMBER_HOME_DEVICE:
        raise Error("TypeError: a KernelDensity member reads X as a dense host float32 view, no column index")
    return member_scope_rows(call[], _KDE_PREFIX, _kde_store_ok(slot, call), "KernelDensity")


def _kde_scores_rows(slot: _Slot, call: _Call, x: MemberScopeRows, want_total: Bool) raises:
    var d = slot[].features
    if x.cols != d:
        kde_refuse_features(String(x.cols), String(d))
    var queries = x.rows
    var scratch = List[Float32]()
    var dst = 0
    if want_total:
        if call[].weights.rows > 0:
            raise Error("TypeError: the score of a KernelDensity member counts every row once (no sample_weight)")
        # The entry writes the per-row scores beside the total (the class's
        # door allocates the same buffer per call).
        scratch = List[Float32](length=queries, fill=0.0)
        dst = Int(scratch.unsafe_ptr()) if queries > 0 else 0
    else:
        if not call[].dst.dense_host(MEMBER_F32):
            raise Error("TypeError: a KernelDensity member writes score_samples as a dense host float32 view")
        if call[].dst.rows != queries or call[].dst.cols != 1:
            raise Error("ValueError: the score buffer holds " + String(queries) + " x 1 values")
        dst = call[].dst.data
    var driver = KDENativeDriver(call[].route.entry.call)
    var has_weights = slot[].weights != 0
    if slot[].handle < 0 and (call[].route.flags & KDE_CAP_RESIDENT) != 0:
        var key = kde_fit_key(slot[].train, slot[].rows, d, slot[].weights, slot[].kernel, slot[].kde_metric,
            slot[].bandwidth)
        slot[].handle = driver.prepare(kde_prepare_request(key, has_weights))
    var total = Float64(0)
    if slot[].handle >= 0:
        var status = List[Int32](length=1, fill=Int32(0))
        var request = kde_resident_query(slot[].handle, x.source, dst, Int(status.unsafe_ptr()),
            queries, d, slot[].bandwidth, want_total, slot[].kernel, slot[].kde_metric)
        total = driver.resident_store(request) if x.resident else driver.resident(request)
        admitted_finite_refuse(Int(status[0]), "KernelDensity")
        _ = status^
    else:
        total = driver.query(kde_query_request(slot[].train, x.source, slot[].weights, dst, slot[].rows,
            queries, d, slot[].bandwidth, has_weights, want_total, slot[].kernel, slot[].kde_metric))
    _ = scratch^
    if want_total:
        call[].value = total


def _kde_own(slot: _Slot):
    """Copy the borrowed training rows and weights into the state, once
    (the class's fit snapshot); the state reads its own copies from then on.
    A resident copy holds the same words and stays."""
    var count = slot[].rows * slot[].features
    if slot[].train != 0 and slot[].train != Int(slot[].train_owned.unsafe_ptr()):
        slot[].train_owned = List[Float32](length=count, fill=0.0)
        if count > 0:
            unsafe_memcpy(dest=slot[].train_owned.unsafe_ptr(), src=_F32P(unsafe_from_address=slot[].train),
                count=count)
        slot[].train = Int(slot[].train_owned.unsafe_ptr()) if count > 0 else 0
    if slot[].weights != 0 and slot[].weights != Int(slot[].weights_owned.unsafe_ptr()):
        slot[].weights_owned = List[Float32](length=slot[].rows, fill=0.0)
        unsafe_memcpy(dest=slot[].weights_owned.unsafe_ptr(), src=_F32P(unsafe_from_address=slot[].weights),
            count=slot[].rows)
        slot[].weights = Int(slot[].weights_owned.unsafe_ptr())


def _kde_export(slot: _Slot, call: _Call) raises:
    if not slot[].fitted:
        raise Error("NotFittedError: this KernelDensity member is not fitted yet")
    if (call[].flags & MEMBER_EXPORT_OWN) != 0:
        # THE LIFETIME RULE: the rows and the weights, once; no view is answered.
        _kde_own(slot)
        return
    var block = call[].block
    if (call[].flags & MEMBER_BLOCK_BY_INDEX) != 0:
        if block < 0 or block > 2:
            raise Error("member ABI: a KernelDensity member has three fitted blocks")
        block = MEMBER_BLOCK_SCALARS if block == 2 else (_BLOCK_TRAIN if block == 0 else _BLOCK_WEIGHTS)
    _kde_own(slot)
    if block == _BLOCK_TRAIN:
        call[].dst = MemberMatrix(slot[].train, slot[].rows if slot[].train != 0 else 0, slot[].features)
    elif block == _BLOCK_WEIGHTS:
        call[].dst = MemberMatrix(slot[].weights, 1 if slot[].weights != 0 else 0,
            slot[].rows if slot[].weights != 0 else 0)
    elif block == MEMBER_BLOCK_SCALARS:
        call[].dst = MemberMatrix(Int(slot[].scalars.unsafe_ptr()), 1, 2, MEMBER_F64)
    else:
        raise Error("member ABI: a KernelDensity member has no fitted block " + String(block))
    call[].block = block
    call[].out_cols = 3


def _kde_import(slot: _Slot, call: _Call) raises:
    var block = call[].block
    if block != 0 or (call[].flags & MEMBER_IMPORT_DONE) == 0:
        _release_resident(slot, call)
        slot[].fitted = False
    var count = call[].x.rows * call[].x.cols
    if block == _BLOCK_TRAIN or block == _BLOCK_WEIGHTS:
        if count > 0 and not call[].x.dense_host(MEMBER_F32):
            raise Error("member ABI: the KernelDensity blocks are dense host float32 matrices")
        var values = List[Float32](length=count, fill=0.0)
        if count > 0:
            unsafe_memcpy(dest=values.unsafe_ptr(), src=_F32P(unsafe_from_address=call[].x.data), count=count)
        if block == _BLOCK_TRAIN:
            slot[].train_owned = values^
            slot[].train = Int(slot[].train_owned.unsafe_ptr()) if count > 0 else 0
            slot[].rows = call[].x.rows
            slot[].features = call[].x.cols
        else:
            slot[].weights_owned = values^
            slot[].weights = Int(slot[].weights_owned.unsafe_ptr()) if count > 0 else 0
    elif block == MEMBER_BLOCK_SCALARS:
        if not call[].x.dense_host(MEMBER_F64) or count != 2:
            raise Error("member ABI: the KernelDensity scalars block is two host float64 words (rows, features)")
        var source = _Words(unsafe_from_address=call[].x.data)
        var scalars: List[Float64] = [source[0], source[1]]
        slot[].scalars = scalars^
    elif block != 0 or (call[].flags & MEMBER_IMPORT_DONE) == 0:
        raise Error("member ABI: a KernelDensity member has no fitted block " + String(block))
    if (call[].flags & MEMBER_IMPORT_DONE) == 0:
        return
    if (slot[].train == 0 or slot[].rows <= 0 or slot[].features <= 0 or len(slot[].scalars) != 2
            or (slot[].weights != 0 and len(slot[].weights_owned) != slot[].rows)):
        raise Error("member ABI: a fitted KernelDensity state needs its training rows and scalars blocks")
    slot[].fitted = True


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
        # No MEMBER_CAN_TEST_ROWS: X is admitted finite by the caller (the
        # DBSCAN and KDE entries do not test it; only KDE weights are checked here).
        var dbscan = MEMBER_CAN_FIT | MEMBER_CAN_PREDICT | MEMBER_CAN_EXPORT | MEMBER_CAN_IMPORT | MEMBER_CAN_SAMPLE_WEIGHT
        # MEMBER_BORROWS_FIT: a fitted KernelDensity state reads the caller's
        # X and weights until an export (or MEMBER_EXPORT_OWN) copied them.
        # DBSCAN owns its state (its core rows are gathered at fit).
        # MEMBER_CAN_ROW_INDEX (lane vdf-scope-cluster2b): a KernelDensity call reads a host row index (its fit
        # then owns the gathered rows); MEMBER_CAN_SCOPE only on the GPU image's route with its resident store,
        # where a search's folds are taken on the device out of one copy of X. DBSCAN keeps gathered folds.
        var kde = (MEMBER_CAN_FIT | MEMBER_CAN_SCORE_SAMPLES | MEMBER_CAN_SCORE | MEMBER_CAN_EXPORT | MEMBER_CAN_IMPORT
            | MEMBER_CAN_SAMPLE_WEIGHT | MEMBER_BORROWS_FIT | MEMBER_CAN_ROW_INDEX)
        if ((call[].route.flags & KDE_CAP_DEV_STORE) != 0 and (call[].route.flags & KDE_CAP_RESIDENT) != 0
                and call[].route.has_second):
            kde |= MEMBER_CAN_SCOPE
        if call[].state != 0:
            var kind = _Slot(unsafe_from_address=_lookup(call[].state, False))[].kind
            return kde if kind == _KIND_KDE else dbscan
        return dbscan | kde
    if operation == MEMBER_OP_NEW:
        if not call[].params or call[].param_count < 1:
            raise Error("member ABI: a density member's first parameter word is its kind")
        var words = _Words(unsafe_from_address=call[].params)
        var kind = Int(words[0])
        if kind == _KIND_KDE:
            var want = 4
            if call[].param_count >= 4 and (words[2] < 0.0 or words[3] < 0.0):
                # A name outside the tables: its two byte counts, then its bytes.
                want = 6
                if call[].param_count >= 6:
                    want += (member_words_byte_slots(Int(words[4]) if words[2] < 0.0 else 0)
                        + member_words_byte_slots(Int(words[5]) if words[3] < 0.0 else 0))
            if call[].param_count != want:
                raise Error("member ABI: a KernelDensity member takes four parameter words (kind, bandwidth,"
                    + " kernel, metric), and a name's byte words after them when its code is -1")
            if not (words[1] > 0.0):
                raise Error("ValueError: bandwidth must be positive, got " + String(words[1]))
        elif kind == _KIND_DBSCAN:
            if call[].param_count != 7 and call[].param_count != 8:
                raise Error("member ABI: a DBSCAN member takes seven or eight parameter words (kind, eps,"
                    + " min_samples, metric, algorithm, max_iterations, max_mbytes_per_batch[, prediction_data])")
        else:
            raise Error("member ABI: density member kind " + String(kind) + " is not published")
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
        call[].out_cols = 1
        call[].classes = 0
        # The kind: the state's, else NEW word 0. DBSCAN predict: int32
        # labels; KernelDensity score_samples: float32 log densities.
        var which = -1
        if call[].state != 0:
            which = _Slot(unsafe_from_address=_lookup(call[].state, False))[].kind
        elif call[].params and call[].param_count >= 1:
            which = Int(_Words(unsafe_from_address=call[].params)[0])
        if which == _KIND_DBSCAN and call[].flags == MEMBER_OP_PREDICT:
            call[].out_dtype = MEMBER_I32
        elif which == _KIND_KDE and call[].flags == MEMBER_OP_SCORE_SAMPLES:
            call[].out_dtype = MEMBER_F32
        return 0
    var slot = _Slot(unsafe_from_address=_lookup(call[].state, False))
    if slot[].kind == _KIND_KDE:
        if operation == MEMBER_OP_FIT:
            _kde_fit(slot, call)
        elif operation == MEMBER_OP_SCORE_SAMPLES:
            _kde_scores(slot, call, False)
        elif operation == MEMBER_OP_SCORE:
            _kde_scores(slot, call, True)
        elif operation == MEMBER_OP_EXPORT:
            _kde_export(slot, call)
        elif operation == MEMBER_OP_IMPORT:
            _kde_import(slot, call)
        else:
            member_unsupported("KernelDensity", operation)
        return 0
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
