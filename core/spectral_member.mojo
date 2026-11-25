# SPDX-License-Identifier: Apache-2.0
"""SpectralClustering as a MEMBER (core/member_abi.mojo, MEMBER-ABI-V2.md).

One entry, `spectral_member_v2`, published by each metrics image beside its
spectral kernel table (`spectral_member_publish`): the device image
`_mojolearn_metrics` (vendor of `COMPILED_VENDOR`) and the host image
`_mojolearn_metrics_host` (vendor 0). The class's predict runs on that
image's spectral queries, so its member lives there. It runs the family's
existing typed RUN over the entries of the `DeviceRoute` it is handed:
`route.entry` is the FIT image's `spectral_native_api_v1` entry
(core/spectral_native_api.mojo `SpectralDriver`, the class's fit binding:
`_mojolearn_metrics` in the process default mode), `route.second` the
PREDICT image's `spectral_native_api_v1` entry (the class's predict binding:
the host binding of the estimator's route, else `_mojolearn_metrics` in the
estimator's mode) and `route.flags` that entry's capability word
(`SPECTRAL_CAP_*`), `route.third` the x_cluster image's
`spectral_assign_native_api_v1` entry (core/spectral_assignment_api.mojo
`SpectralAssignmentDriver`, read only for assign_labels 'discretize' /
'cluster_qr'). No other route slot is read. No kernel is touched.

Kind 1 (lane mf-member8): SpectralEmbedding, fit and export only (the
class has no transform and no predict), the stages of its own fit
(bindings/cfb_spectral_door.mojo `_se_execute`) on the same entries: the
dataset route on X ('nearest_neighbors'), or the n x n affinity read
('precomputed', square), built ('rbf' on `route.fifth`,
'precomputed_nearest_neighbors' on `route.fourth`), to the fit entry's
EMBED operation dense or as COO triples, with its refusals in its words.
NEW words (15): [0] 1, [1] 0, [2] n_components, [3] 0, [4] n_neighbors
(as given; auto below), [5] eigen_tol, [6] seed bits, [7] -1, [8] affinity
code, [9] 0, [10] gamma (as given; auto below), [11] dense fit entry,
[12] drop_first, [13] norm_laplacian, [14] 1 automatic n_neighbors
(max(n // 10, 1)) + 2 automatic gamma (1 / n_features). Blocks: the
embedding (`MEMBER_BLOCK_FAMILY + 1`), the built 'rbf' affinity
(`+ 7`), SCALARS [n_components_, n_neighbors_ (-1: none published),
n_features_in_ (-1: none published), gamma_ (-1: none)].

Kind (MEMBER_OP_NEW word [0]): 0 SpectralClustering, every affinity its
fit admits, with or without prediction data (lane mf-clu6; the class's own
checks decide the rest at the caller's UNPACK). 'nearest_neighbors' runs the
dataset route on X; 'precomputed' reads X as the n x n affinity;
'rbf' builds the affinity on the x_decomp image's RBF cells (`route.fifth`)
and 'precomputed_nearest_neighbors' reads X as the n x n distance matrix and
builds the k-NN connectivity affinity on the preparation entry of the base
image (`route.fourth`), as `_sc_fit` does. A graph affinity goes to the fit
entry dense when that entry serves dense matrices (NEW word [11]), else as
the COO triples the same preparation entry counts and fills. predict
serves 'nearest_neighbors' and 'precomputed' fits kept with prediction data
(the class's rule); otherwise it is the class's own refusal.

MEMBER_OP_NEW parameters (Float64 words, 12): [0] 0, [1] n_clusters,
[2] n_components (resolved: n_clusters for None), [3] n_init,
[4] n_neighbors, [5] eigen_tol (positive; 'auto' refused at UNPACK, DEVIATION 890), [6] the seed as the BITS of
an int64 (`bitcast[DType.float64](Int64(seed))`, the class's own seed
rule), [7] assign_labels code (-1 'kmeans', 0 'discretize', 1 'cluster_qr'),
[8] affinity (0 'nearest_neighbors', 1 'precomputed', 2 'rbf',
3 'precomputed_nearest_neighbors'), [9] prediction_data (keep_state),
[10] gamma (the resolved `_gamma`; 'rbf' only), [11] 1 when the fit entry
serves a dense affinity (`SPECTRAL_CAP_DENSE`).
A clusterer, not a classifier: `y` is not read, `classes` is 0.

Views: X dense row-major host float32, BORROWED (never copied at fit): the
fit entry reads the caller's words, and predict's Nystrom extension reads the
training rows through the same view, so THE CALLER KEEPS X ALIVE UNTIL THIS
STATE IS FREED OR REFITTED (the member ABI's view lifetime). MEMBER_OP_EXPORT
of the training block copies the rows into the state once (the class's own
fit copies them into `_fit_X` on every fit); from then on the state reads its
own copy. A model-selection fold fit on the metrics store whose predict entry reads that store
(SPECTRAL_CAP_DEV_QUERY, lane vdf-scope-cluster8) keeps the fold's rows as THE FITTED-ROWS SLOT instead: no
host gather; predict's k-NN search reads that slot (and a fold of queries taken there) on the device; the
first export of the rows block reads it home once and frees it. No finite scan here (MEMBER_CAN_TEST_ROWS clear: the caller's
admission scans, as the class's door does). No sample weights, no row or
column index, no device-home view: refused in words.

Outputs: predict writes int32 labels (rows x 1): the nearest fitted k-means
centroid of each query's Nystrom embedding (the class's `predict`).
`call.fit_output = MEMBER_OP_PREDICT` writes the training labels in the fit
call (fit_predict, the class's labels_).

Fitted-state blocks (MEMBER_OP_EXPORT / MEMBER_OP_IMPORT), in index order:
0 `MEMBER_BLOCK_FAMILY` int32 1 x rows, the training labels;
1 `MEMBER_BLOCK_FAMILY + 1` float32 rows x components, the embedding;
2 `MEMBER_BLOCK_FAMILY + 2` float32 1 x components, the eigenvalues;
3 `MEMBER_BLOCK_FAMILY + 3` float32 rows x components, the eigenvectors;
4 `MEMBER_BLOCK_FAMILY + 4` float32 1 x rows, the degree diagonal;
5 `MEMBER_BLOCK_FAMILY + 5` float32 clusters x components, the centroids;
6 `MEMBER_BLOCK_FAMILY + 6` float32 rows x features, the training rows
(the affinity X for 'precomputed'; 0 x features when a 'nearest_neighbors'
fit keeps no prediction data);
7 `MEMBER_BLOCK_SCALARS` float64 [n_components_, n_iter_assign_ (-1: none)];
8 `MEMBER_BLOCK_FAMILY + 7` float32 rows x rows, the built affinity ('rbf',
'precomputed_nearest_neighbors'; export only).
An import needs blocks 2 to 7 (labels and embedding are optional; a
'precomputed' state has no training rows).

The fitted state is OWNED here, in a process-wide store of fixed chunks: a
slot never moves once minted, so an operation runs on its slot without the
store's lock (the lock covers mint, free and the handle lookup). A state on
an image with SPECTRAL_CAP_RESIDENT keeps its prediction state resident after
its first query (one upload per fitted state, as the class's resident lease),
released at its next fit, import or free. Serial by nature here: the store's
bookkeeping.
"""
from std.memory import bitcast, unsafe_memcpy
from std.sys.info import size_of
from core.member_abi import (
    MemberCall, MemberMatrix, MEMBER_F32, MEMBER_F64, MEMBER_I32, MEMBER_HOME_DEVICE,
    MEMBER_OP_CAPABILITIES, MEMBER_OP_NEW, MEMBER_OP_FREE, MEMBER_OP_FIT, MEMBER_OP_PREDICT,
    MEMBER_OP_OUTPUTS, MEMBER_OP_EXPORT, MEMBER_OP_IMPORT,
    MEMBER_CAN_FIT, MEMBER_CAN_PREDICT, MEMBER_CAN_EXPORT, MEMBER_CAN_IMPORT,
    MEMBER_BLOCK_SCALARS, MEMBER_BLOCK_FAMILY, MEMBER_BLOCK_BY_INDEX, MEMBER_IMPORT_DONE,
    member_publish, member_call_from_spans, member_unsupported,
)
from core.member_abi import MEMBER_BORROWS_FIT, MEMBER_EXPORT_OWN, MEMBER_IGNORES_Y
from core.member_families import MEMBER_FAMILY_SPECTRAL
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.shared_slot import SharedSlot, shared_slot_lock, shared_slot_unlock
from core.spectral_assignment_api import (
    SpectralAssignmentDriver, SpectralAssignmentOutput, SpectralAssignmentRequest,
)
from core.spectral_native_api import (
    SpectralDriver, SpectralInput, SpectralStateView, SpectralFitOptions, SpectralFitRequest,
    SpectralQueryShape, SpectralQueryRequest, SpectralPrepareRequest, SpectralResidentQueryRequest,
    SPECTRAL_DATASET, SPECTRAL_CAP_RESIDENT,
)
from spectral.cfb_spectral_run import _sc_check_shape, _se_check_shape
from core.spectral_native_api import SpectralEmbeddingRequest, SpectralEmbeddingOptions
from core.spectral_native_api import SPECTRAL_DENSE, SPECTRAL_GRAPH
from core.spectral_native_api import (
    SPECTRAL_RBF, SPECTRAL_KNN, SPECTRAL_KNN_REFUSED, SPECTRAL_CAP_AFFINITY, SPECTRAL_CAPABILITIES,
)
from core.spectral_native_api import SPECTRAL_DATASET_STORE, SPECTRAL_CAP_DEV_STORE, SPECTRAL_CAP_DEV_QUERY
from core.entry_table import EntryRef, entry_ref_mode, entry_ref_vendor, entry_ref_unresolved
from core.fold_x_word import fold_x_word
from bindings.arena_cache_run import arena_cache_free, arena_store_read
from core.metrics_store_ref import metrics_store_find, METRICS_STORE_PREFIX
from core.member_scope_rows import MemberScopeRows, member_scope_rows_in, member_scope_rows_done_in
from core.member_abi import MEMBER_CAN_ROW_INDEX, MEMBER_CAN_SCOPE, MEMBER_SCOPE_NONE
from core.spectral_preparation_api import (
    SpectralPreparationDriver, SpectralCOOCountRequest, SpectralCOOFillRequest, SpectralKNNRequest,
    SpectralAffinityStatus, SpectralCOOStorage,
)
from core.spectral_affinity_plan import SpectralRBFRequest, spectral_rbf_affinity
from core.native_descriptor_fixed_call import native_descriptor_call_fixed
from core.family_cast_impl import family_min_f32
from core.host_finite_scan import host_all_finite_f32
from x_decomp.descriptor_schema import XD_CAPABILITIES, XD_CAP_RESIDENT

comptime _CHUNK = 256
comptime _INDEX_SPAN = 16777216
comptime _EST = "SpectralClustering"
comptime _KIND_SPECTRAL = 0
comptime _WORDS = 12
comptime _KIND_EMBEDDING = 1   # lane mf-member8: SpectralEmbedding
comptime _SE_WORDS = 15
comptime _SE_EST = "SpectralEmbedding"
comptime _BLOCKS = 9
comptime _AFF_NN = 0
comptime _AFF_PRECOMPUTED = 1
comptime _AFF_RBF = 2
comptime _AFF_PNN = 3
comptime _BLOCK_LABELS = MEMBER_BLOCK_FAMILY
comptime _BLOCK_EMBEDDING = MEMBER_BLOCK_FAMILY + 1
comptime _BLOCK_EIGENVALUES = MEMBER_BLOCK_FAMILY + 2
comptime _BLOCK_EIGENVECTORS = MEMBER_BLOCK_FAMILY + 3
comptime _BLOCK_DIAGONAL = MEMBER_BLOCK_FAMILY + 4
comptime _BLOCK_CENTROIDS = MEMBER_BLOCK_FAMILY + 5
comptime _BLOCK_TRAINING = MEMBER_BLOCK_FAMILY + 6
comptime _BLOCK_AFFINITY = MEMBER_BLOCK_FAMILY + 7
comptime _SCALARS = 2
comptime _I32P = MutPointer[Int32, MutUntrackedOrigin]
comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]


struct SpectralMemberSlot(Copyable, Movable, Defaultable):
    var generation: Int
    var live: Bool
    # Lane mf-member8: 0 SpectralClustering, 1 SpectralEmbedding (its two
    # flags and its automatic-setting word).
    var kind: Int
    var drop_first: Bool
    var norm_laplacian: Bool
    var automatic: Int
    var clusters: Int
    var components: Int
    var initializations: Int
    var neighbors: Int
    var tolerance: Float64
    var seed: Int
    var assignment: Int
    var affinity: Int
    var keep_state: Bool
    var gamma: Float64
    var fit_dense: Bool
    # The affinity an 'rbf' / 'precomputed_nearest_neighbors' fit built.
    var affinity_values: List[Float32]
    var fitted: Bool
    var rows: Int
    var features: Int
    # The training rows: the caller's borrowed X after a fit (`borrowed`,
    # alive until free or refit), else the state's own copy in `refs`.
    var borrowed: Int
    var refs: List[Float32]
    # The resident prediction state of a SPECTRAL_CAP_RESIDENT image (-1: none).
    var handle: Int
    # THE FITTED-ROWS SLOT (lane vdf-scope-cluster8): a fold fit on the metrics store keeps its training rows
    # there (-1: none; `refs` / `borrowed` then hold none), in the store `rows_store`.
    var rows_slot: Int
    var rows_store: EntryRef
    var labels: List[Int32]
    var embedding: List[Float32]
    var eigenvalues: List[Float32]
    var eigenvectors: List[Float32]
    var diagonal: List[Float32]
    var centroids: List[Float32]
    var scalars: List[Float64]

    def __init__(out self):
        self.generation = 0
        self.live = False
        self.kind = _KIND_SPECTRAL
        self.drop_first = True
        self.norm_laplacian = True
        self.automatic = 0
        self.clusters = 0
        self.components = 0
        self.initializations = 0
        self.neighbors = 0
        self.tolerance = 0.0
        self.seed = 0
        self.assignment = -1
        self.affinity = _AFF_NN
        self.keep_state = True
        self.gamma = 0.0
        self.fit_dense = False
        self.affinity_values = List[Float32]()
        self.fitted = False
        self.rows = 0
        self.features = 0
        self.borrowed = 0
        self.refs = List[Float32]()
        self.handle = -1
        self.rows_slot = -1
        self.rows_store = entry_ref_unresolved()
        self.labels = List[Int32]()
        self.embedding = List[Float32]()
        self.eigenvalues = List[Float32]()
        self.eigenvectors = List[Float32]()
        self.diagonal = List[Float32]()
        self.centroids = List[Float32]()
        self.scalars = List[Float64]()

    def clear(mut self):
        self.fitted = False
        self.rows = 0
        self.features = 0
        self.borrowed = 0
        self.refs = List[Float32]()
        self.labels = List[Int32]()
        self.embedding = List[Float32]()
        self.eigenvalues = List[Float32]()
        self.eigenvectors = List[Float32]()
        self.diagonal = List[Float32]()
        self.centroids = List[Float32]()
        self.affinity_values = List[Float32]()
        self.scalars = List[Float64]()

    def training_address(self) -> Int:
        if self.borrowed != 0:
            return self.borrowed
        return Int(self.refs.unsafe_ptr()) if len(self.refs) > 0 else 0

    def state_view(self) -> SpectralStateView:
        return SpectralStateView(Int(self.eigenvalues.unsafe_ptr()), Int(self.eigenvectors.unsafe_ptr()),
            Int(self.diagonal.unsafe_ptr()), Int(self.centroids.unsafe_ptr()))


struct SpectralMemberStore(Defaultable, Movable):
    var synchronization: List[Int64]
    var chunks: List[List[SpectralMemberSlot]]
    var free: List[Int]
    var count: Int

    def __init__(out self):
        # Side-effect-free even when the process slot's initializers race.
        self.synchronization = List[Int64](length=2, fill=0)
        self.chunks = List[List[SpectralMemberSlot]]()
        self.free = List[Int]()
        self.count = 0


comptime _STORE = SharedSlot[SpectralMemberStore, "MojolearnSpectralMemberStoreV2", 2]
comptime _Slot = MutPointer[SpectralMemberSlot, MutUntrackedOrigin]
comptime _Call = MutPointer[MemberCall, MutUntrackedOrigin]
comptime _Words = MutPointer[Float64, MutUntrackedOrigin]


def _slot_address(index: Int) raises -> Int:
    # Under the store's lock. A chunk's buffer is allocated once and never
    # grows, so the address stays valid after the lock is released.
    var store = _STORE.get_or_create_ptr()
    return Int(store[].chunks[index // _CHUNK].unsafe_ptr()) + (index % _CHUNK) * size_of[SpectralMemberSlot]()


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
                raise Error("RuntimeError: mojolearn: too many live SpectralClustering member states")
            if store[].count == len(store[].chunks) * _CHUNK:
                store[].chunks.append(List[SpectralMemberSlot](length=_CHUNK, fill=SpectralMemberSlot()))
            index = store[].count
            store[].count += 1
        var slot = _Slot(unsafe_from_address=_slot_address(index))
        slot[].generation += 1
        slot[].live = True
        slot[].clusters = Int(words[1])
        slot[].components = Int(words[2])
        slot[].initializations = Int(words[3])
        slot[].neighbors = Int(words[4])
        slot[].tolerance = words[5]
        slot[].seed = Int(bitcast[DType.int64](words[6]))
        slot[].assignment = Int(words[7])
        slot[].affinity = Int(words[8])
        slot[].keep_state = words[9] != 0.0
        slot[].gamma = words[10]
        slot[].fit_dense = words[11] != 0.0
        slot[].kind = Int(words[0])
        if slot[].kind == _KIND_EMBEDDING:
            slot[].drop_first = words[12] != 0.0
            slot[].norm_laplacian = words[13] != 0.0
            slot[].automatic = Int(words[14])
        slot[].clear()
        slot[].handle = -1
        slot[].rows_slot = -1
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


def _query_driver(call: _Call) raises -> SpectralDriver:
    if not call[].route.has_second:
        raise Error("member ABI: a SpectralClustering member's route holds no predict entry")
    return SpectralDriver(call[].route.second.call)


def _release_resident(slot: _Slot, call: _Call) raises:
    """Release the slot's resident prediction state, before a new fit or
    import, and at free."""
    if slot[].handle < 0:
        return
    var handle = slot[].handle
    slot[].handle = -1
    _query_driver(call).release(handle)


def _drop_rows_slot(slot: _Slot) raises:
    """Free THE FITTED-ROWS SLOT (a new fit, an import, a free or the first export of the rows replace it). The
    resident prediction state holds its own device copy, so it may stay."""
    if slot[].rows_slot >= 0:
        var held = slot[].rows_slot
        slot[].rows_slot = -1
        arena_cache_free(slot[].rows_store, held)


def _query_store_ok(call: _Call) -> Bool:
    """The predict entry (`route.second`, capability word `route.flags`) reads the metrics store's slots
    (SPECTRAL_CAP_DEV_QUERY with SPECTRAL_CAP_RESIDENT) on the fit entry's image (the same (mode, vendor), so the
    same store and context): a fold fit may then keep its training rows as THE FITTED-ROWS SLOT."""
    if not call[].route.has_second:
        return False
    var flags = call[].route.flags
    if (flags & SPECTRAL_CAP_DEV_QUERY) == 0 or (flags & SPECTRAL_CAP_RESIDENT) == 0:
        return False
    return (entry_ref_mode(call[].route.second) == entry_ref_mode(call[].route.entry)
        and entry_ref_vendor(call[].route.second) == entry_ref_vendor(call[].route.entry))


def _whole_x(call: _Call) raises:
    if call[].row_index.rows > 0 or call[].column_index.rows > 0:
        raise Error("TypeError: a SpectralClustering member reads no row or column index"
            + " (a fold is handed over as gathered host words)")
    if call[].x.home == MEMBER_HOME_DEVICE or not call[].x.dense_host(MEMBER_F32):
        raise Error("TypeError: a SpectralClustering member reads X as a dense host float32 view")


def _labels_out(call: _Call, rows: Int) raises:
    if not call[].dst.dense_host(MEMBER_I32):
        raise Error("TypeError: a SpectralClustering member writes the label buffer as a dense host int32 view")
    if call[].dst.rows != rows or call[].dst.cols != 1:
        raise Error("ValueError: the label buffer holds " + String(rows) + " x 1 values")


def _preparation(call: _Call) raises -> SpectralPreparationDriver:
    if not call[].route.has_fourth:
        raise Error("member ABI: a SpectralClustering member's route holds no preparation entry")
    return SpectralPreparationDriver(call[].route.fourth.call)


def _rbf_affinity(slot: _Slot, call: _Call, n: Int, d: Int, x_addr: Int) raises:
    """`_rbf_affinity`: the x_decomp image's sqdist -> scale -> exp cells into
    the state's n x n affinity (`spectral_rbf_run`'s request, its resident
    word read from the entry's capabilities)."""
    if not call[].route.has_fifth:
        raise Error("member ABI: a SpectralClustering member's route holds no x_decomp entry")
    var entry = call[].route.fifth.call
    var a: Array[Int, 0] = []
    var p: Array[Int, 0] = []
    var f: Array[Float64, 0] = []
    var capabilities = native_descriptor_call_fixed(entry, XD_CAPABILITIES, a, p, f)
    slot[].affinity_values = List[Float32](length=n * n, fill=0.0)
    spectral_rbf_affinity(entry, SpectralRBFRequest(x_addr, Int(slot[].affinity_values.unsafe_ptr()), n, d,
        slot[].gamma, (capabilities & XD_CAP_RESIDENT) != 0))


def _knn_affinity(slot: _Slot, call: _Call, n: Int, d: Int, x_addr: Int) raises:
    """`_knn_affinity` of a dense precomputed distance matrix: the base
    image's `knn_affinity_f32` (connectivity, symmetrized), its refusals."""
    if d != n:
        raise Error("mojolearn SpectralEmbedding: the precomputed distance matrix must be square")
    slot[].affinity_values = List[Float32](length=n * n, fill=0.0)
    if n == 0:
        return
    var k = slot[].neighbors
    var status = SpectralAffinityStatus()
    _preparation(call).knn(SpectralKNNRequest(x_addr, 0, 0, 0, Int(slot[].affinity_values.unsafe_ptr()),
        status.address(), n, k, 0, False))
    _knn_status_refuse(status, k)


def _knn_status_refuse(status: SpectralAffinityStatus, k: Int) raises:
    """The k-NN affinity's three status words as the class's refusals (the base image's builder and the fit
    entry's fused builder write the same words)."""
    var code = Int(status.words[0])
    var bad_row = Int(status.words[1])
    var stored = Int(status.words[2])
    if code == 1:
        raise Error("mojolearn SpectralEmbedding: a precomputed distance is negative or NaN")
    if code == 2:
        raise Error("mojolearn SpectralEmbedding: row " + String(bad_row) + " of the precomputed graph has "
                    + String(stored) + " stored distances, fewer than n_neighbors=" + String(k))


def _fused_affinity(call: _Call, fit_dense: Bool) raises -> Bool:
    """Whether the FIT entry builds an 'rbf' / 'precomputed_nearest_neighbors' affinity on its own device and
    fits (or embeds) from it there (`SPECTRAL_CAP_AFFINITY` on a dense fit entry; lane vdf-misc4's estimator
    route, applied to the member by lane vdf-scope-cluster4): the n x n crosses once, into the state's affinity
    block. Was: built by the x_decomp (rbf) or base (knn) image into the state's host words and uploaded again
    by the fit entry. One metadata call per fit."""
    if not fit_dense:
        return False
    return (SpectralDriver(call[].route.entry.call).metadata(SPECTRAL_CAPABILITIES) & SPECTRAL_CAP_AFFINITY) != 0


def _coo(call: _Call, address: Int, n: Int, built: Bool) raises -> SpectralCOOStorage:
    """The COO triples of the dense n x n affinity at `address` (the
    preparation entry's float32 nonzero count and fill; exact zeros dropped),
    then the class's refusals of the COO route in its order."""
    var nnz = 0
    if n * n > 0:
        nnz = _preparation(call).count(SpectralCOOCountRequest(address, n * n, False))
    var room = SpectralCOOStorage(nnz)
    if nnz > 0:
        var wrote = _preparation(call).fill(SpectralCOOFillRequest(address, Int(room.rows.unsafe_ptr()),
            Int(room.columns.unsafe_ptr()), Int(room.values.unsafe_ptr()), n, n, nnz, False))
        if wrote != nnz:
            raise Error("RuntimeError: mojolearn: nonzero_f32_fill wrote " + String(wrote) + " of "
                        + String(nnz) + " entries")
    if nnz == 0:
        if built:
            # `_dense_coo_span`'s refusal of an affinity it built with no entry.
            raise Error("mojolearn: rows is empty, shape (0,): Found array with 0 sample(s) (shape=(0,)) while"
                        + " a minimum of 1 is required.")
        raise Error("mojolearn SpectralClustering: the precomputed affinity matrix has no nonzero entries")
    if not host_all_finite_f32(_F32P(unsafe_from_address=Int(room.values.unsafe_ptr())), nnz):
        raise Error("mojolearn SpectralClustering: the precomputed affinity "
                    + "matrix has a non-finite entry (the implemented path refuses "
                    + "it by name, because sqrt of a non-finite degree is a "
                    + "NaN and no NaN may reach a recorded value)")
    if family_min_f32(_F32P(unsafe_from_address=Int(room.values.unsafe_ptr())), nnz) < 0:
        raise Error("mojolearn SpectralClustering: the precomputed affinity "
                    + "matrix has a negative entry (refused by name: sqrt of a "
                    + "negative degree is a NaN in cuVS too)")
    return room^


def _fit(slot: _Slot, call: _Call) raises:
    """`_sc_fit`: the affinity's route (the dataset route on X, or the n x n
    affinity read or built, to the fit entry dense or as COO), its shape
    checks in the class's order, one fit entry call on the borrowed X
    (keeping the prediction state when asked), then the assignment entry for
    'discretize' / 'cluster_qr' (the class's 30 restarts, 20 iterations)."""
    if call[].row_index.rows == 0:
        _whole_x(call)
        _fit_rows(slot, call, call[].x.data, call[].x.rows, call[].x.cols, -1, List[Float32](), False,
            entry_ref_unresolved())
        return
    if call[].column_index.rows > 0 or call[].x.home == MEMBER_HOME_DEVICE:
        raise Error("TypeError: a SpectralClustering member reads X as a dense host float32 view, no column index")
    # MEMBER_CAN_ROW_INDEX (lane vdf-scope-cluster4): the fold's rows. THE RESIDENT-STORE ROUTE: the dataset
    # affinity on a GPU fit entry with the store (SPECTRAL_CAP_DEV_STORE, the metrics image's store resolved from
    # the fit route's (mode, vendor)), the caller's scope with X's pin: the fold is taken on the device out of the
    # scope's one copy of X and the fit reads that slot. Gathered on the host cores otherwise. A state that keeps
    # its training rows for predict keeps THAT SLOT (lane vdf-scope-cluster8) when the predict entry reads the
    # store's slots (`_query_store_ok`: the Nystrom extension's k-NN search then reads the rows on the device);
    # it gathers them on the host cores only when the predict entry reads host rows.
    var store = _spectral_store(slot, call)
    var resident = Bool(store)
    var keeps_nn = slot[].affinity == _AFF_NN and slot[].keep_state
    var keep_slot = resident and keeps_nn and _query_store_ok(call)
    var gathered = MemberScopeRows(0, 0, 0, False, List[Float32]())
    if not resident or (keeps_nn and not keep_slot):
        gathered = member_scope_rows_in(call[], call[].route.entry, METRICS_STORE_PREFIX, False, _EST)
    # The gathered words move to the fit (a swap: the aggregate stays whole); their address is unchanged.
    var owned = List[Float32]()
    swap(owned, gathered.gathered)
    if not resident:
        _fit_rows(slot, call, gathered.source, gathered.rows, gathered.cols, -1, owned^, False,
            entry_ref_unresolved())
        return
    var taken = member_scope_rows_in(call[], store.value(), METRICS_STORE_PREFIX, True, _EST)
    var kept = False
    try:
        _fit_rows(slot, call, gathered.source, taken.rows, taken.cols, taken.source, owned^, keep_slot,
            store.value())
        kept = keep_slot
    finally:
        if not kept:
            member_scope_rows_done_in(store.value(), taken)


def _store_affinity(slot: _Slot, caps: Int) -> Bool:
    """The affinities whose fit reads X from a metrics-store slot: the dataset route ('nearest_neighbors',
    SPECTRAL_DATASET_STORE), and 'rbf' when the fit entry builds it on its own device (the fused route,
    SPECTRAL_CAP_AFFINITY on a dense fit entry: `rbf_affinity_device` reads the slot word in place, lane
    vdf-scope-cluster8). 'precomputed' / 'precomputed_nearest_neighbors' read X as an n x n matrix, whose fold is
    a row AND column selection: a row index alone cannot name it, so they keep the caller's words."""
    if slot[].affinity == _AFF_NN:
        return True
    return slot[].affinity == _AFF_RBF and slot[].fit_dense and (caps & SPECTRAL_CAP_AFFINITY) != 0


def _spectral_store(slot: _Slot, call: _Call) -> Optional[EntryRef]:
    """The metrics image's resident store when a SpectralClustering fit may read a fold from it: an affinity of
    `_store_affinity`, a fit entry answering SPECTRAL_CAP_DEV_STORE, the caller's scope with X's pin, and the store
    published for the fit route's (mode, vendor) (core/metrics_store_ref.mojo)."""
    if slot[].kind != _KIND_SPECTRAL:
        return Optional[EntryRef]()
    if call[].scope == MEMBER_SCOPE_NONE or call[].pin == 0:
        return Optional[EntryRef]()
    try:
        var caps = SpectralDriver(call[].route.entry.call).metadata(SPECTRAL_CAPABILITIES)
        if (caps & SPECTRAL_CAP_DEV_STORE) == 0 or not _store_affinity(slot, caps):
            return Optional[EntryRef]()
    except:
        return Optional[EntryRef]()
    return metrics_store_find(entry_ref_mode(call[].route.entry), entry_ref_vendor(call[].route.entry))


def _fit_rows(slot: _Slot, call: _Call, x_addr: Int, rows: Int, d: Int, store_slot: Int,
              var owned: List[Float32], keep_slot: Bool, rows_store: EntryRef) raises:
    """The SpectralClustering fit over host rows at `x_addr` (`rows` x `d`: the caller's X, or a gathered fold
    whose words `owned` holds; 0 when only a slot holds the fold) or, for the dataset route, over slot
    `store_slot` of the metrics store (`>= 0`; SPECTRAL_DATASET_STORE: the same kernels on the same words).
    `keep_slot`: on success the state keeps `store_slot` (of `rows_store`) as THE FITTED-ROWS SLOT, its training
    rows for predict (the caller frees it otherwise)."""
    if call[].weights.rows > 0:
        raise Error("TypeError: SpectralClustering takes no sample_weight")
    var output = call[].fit_output
    if output == MEMBER_OP_PREDICT:
        _labels_out(call, rows)
    elif output != 0:
        member_unsupported(_EST, output)
    var clusters = slot[].clusters
    var k = slot[].components
    var affinity = slot[].affinity
    if slot[].assignment >= 0 and not call[].route.has_third:
        raise Error("member ABI: a SpectralClustering member's route holds no assignment entry")
    _release_resident(slot, call)
    _drop_rows_slot(slot)
    slot[].clear()
    var n = rows
    var coo = SpectralCOOStorage(0)
    var source_input: SpectralInput
    var status = SpectralAffinityStatus()
    var fused = ((affinity == _AFF_RBF or affinity == _AFF_PNN) and clusters <= n and k < n
        and (affinity == _AFF_RBF or (d == n and n > 0)) and _fused_affinity(call, slot[].fit_dense))
    if store_slot >= 0 and not (affinity == _AFF_NN or (fused and affinity == _AFF_RBF)):
        _sc_check_shape(clusters, n, k)  # the class's refusal first (the only case the fused route declines)
        raise Error("member ABI: a SpectralClustering fold on the metrics store serves the dataset route and the"
            + " device-built 'rbf' affinity only")
    if fused:
        # The fit entry builds the affinity on its device into the state's block and fits from it there; a fold
        # in the metrics store is named by its negative X word (`rbf_affinity_device` reads it in place).
        slot[].affinity_values = List[Float32](length=n * n, fill=0.0)
        if affinity == _AFF_RBF:
            source_input = SpectralInput(fold_x_word(store_slot) if store_slot >= 0 else x_addr, 0, 0, 0,
                SPECTRAL_RBF, n, d, 0, Int(slot[].affinity_values.unsafe_ptr()), 0, slot[].gamma)
        else:
            source_input = SpectralInput(x_addr, 0, 0, 0, SPECTRAL_KNN, n, slot[].neighbors, 0,
                Int(slot[].affinity_values.unsafe_ptr()), status.address(), 0.0)
    elif affinity == _AFF_NN:
        _sc_check_shape(clusters, rows, k)
        if slot[].neighbors > rows:
            raise Error("mojolearn SpectralClustering: n_neighbors=" + String(slot[].neighbors)
                        + " exceeds n_samples=" + String(rows))
        if store_slot >= 0:
            source_input = SpectralInput(store_slot, 0, 0, 0, SPECTRAL_DATASET_STORE, rows, d, 0)
        else:
            source_input = SpectralInput(x_addr, 0, 0, 0, SPECTRAL_DATASET, rows, d, 0)
    else:
        var dense_address = x_addr
        if affinity == _AFF_PRECOMPUTED:
            if rows != d:
                raise Error("mojolearn SpectralClustering: with affinity='precomputed', X "
                            + "must be a square affinity matrix or a SciPy sparse matrix, "
                            + "got shape (" + String(rows) + ", " + String(d) + ")")
        elif affinity == _AFF_RBF:
            _rbf_affinity(slot, call, rows, d, x_addr)
            dense_address = Int(slot[].affinity_values.unsafe_ptr())
        else:
            _knn_affinity(slot, call, rows, d, x_addr)
            dense_address = Int(slot[].affinity_values.unsafe_ptr())
        if slot[].fit_dense:
            _sc_check_shape(clusters, n, k)
            source_input = SpectralInput(dense_address, 0, 0, 0, SPECTRAL_DENSE, n, 0, 0)
        else:
            coo = _coo(call, dense_address, n, affinity != _AFF_PRECOMPUTED)
            _sc_check_shape(clusters, n, k)
            source_input = SpectralInput(0, Int(coo.rows.unsafe_ptr()), Int(coo.columns.unsafe_ptr()),
                Int(coo.values.unsafe_ptr()), SPECTRAL_GRAPH, n, 0, len(coo.values))
    var keep = slot[].keep_state
    slot[].labels = List[Int32](length=n, fill=0)
    slot[].embedding = List[Float32](length=n * k, fill=0.0)
    if keep:
        slot[].eigenvalues = List[Float32](length=k, fill=0.0)
        slot[].eigenvectors = List[Float32](length=n * k, fill=0.0)
        slot[].diagonal = List[Float32](length=n, fill=0.0)
        slot[].centroids = List[Float32](length=clusters * k, fill=0.0)
    var view = slot[].state_view() if keep else SpectralStateView(0, 0, 0, 0)
    var options = SpectralFitOptions(clusters, k, slot[].initializations, slot[].neighbors, slot[].tolerance,
        slot[].seed, keep)
    var request = SpectralFitRequest(source_input, Int(slot[].labels.unsafe_ptr()), Int(slot[].embedding.unsafe_ptr()),
        view, options)
    var n_out = SpectralDriver(call[].route.entry.call).fit(request)
    _ = len(coo.values)  # the COO triples the fit read stay alive through it
    if fused and n_out == SPECTRAL_KNN_REFUSED:
        _knn_status_refuse(status, slot[].neighbors)
    _ = status^
    if n_out != k:
        raise Error("RuntimeError: mojolearn SpectralClustering: the kernel returned " + String(n_out) + " "
                    + "embedding columns where " + String(k) + " were expected")
    var iterations = -1
    if slot[].assignment >= 0:
        var assign = SpectralAssignmentRequest(Int(slot[].embedding.unsafe_ptr()), n * k, n, k,
            slot[].assignment, slot[].seed, 30, 20)
        var result = SpectralAssignmentOutput()
        SpectralAssignmentDriver(call[].route.third.call).assign(assign, result)
        if len(result.storage[0].labels) != n:
            raise Error("RuntimeError: mojolearn SpectralClustering: the assignment returned "
                        + String(len(result.storage[0].labels)) + " labels where " + String(n) + " were expected")
        if slot[].assignment == 0:
            iterations = result.storage[0].iterations
        swap(slot[].labels, result.storage[0].labels)
        _ = result^
    var scalars: List[Float64] = [Float64(k), Float64(iterations)]
    slot[].scalars = scalars^
    slot[].rows = n
    slot[].features = d
    # The training rows the state keeps: the dataset route's X with prediction
    # data (`_fit_X`), a precomputed X always (its `affinity_matrix_`).
    var keeps_rows = (affinity == _AFF_NN and keep) or affinity == _AFF_PRECOMPUTED
    if keeps_rows and keep_slot and store_slot >= 0:
        # THE FITTED-ROWS SLOT: the fold's rows stay on the device; nothing is borrowed or copied.
        slot[].rows_slot = store_slot
        slot[].rows_store = rows_store
        slot[].borrowed = 0
    elif keeps_rows and len(owned) > 0:
        # A gathered fold: the state owns its training rows (nothing is borrowed past the call).
        slot[].refs = owned^
        slot[].borrowed = 0
    else:
        slot[].borrowed = x_addr if keeps_rows else 0
    slot[].fitted = True
    if output == MEMBER_OP_PREDICT:
        unsafe_memcpy(dest=_I32P(unsafe_from_address=call[].dst.data), src=slot[].labels.unsafe_ptr(), count=n)


def _se_coo(call: _Call, address: Int, n: Int, precomputed: Bool) raises -> SpectralCOOStorage:
    """Lane mf-member8: `_se_execute`'s COO route of a dense n x n affinity
    (the preparation entry's nonzero count and fill), its refusals in its
    words: an empty built affinity (`_dense_coo_span`), an empty given one,
    then (precomputed only: an 'rbf' affinity is finite and >= 0 by
    construction) a non-finite or negative entry."""
    var nnz = 0
    if n * n > 0:
        nnz = _preparation(call).count(SpectralCOOCountRequest(address, n * n, False))
    var room = SpectralCOOStorage(nnz)
    if nnz > 0:
        var wrote = _preparation(call).fill(SpectralCOOFillRequest(address, Int(room.rows.unsafe_ptr()),
            Int(room.columns.unsafe_ptr()), Int(room.values.unsafe_ptr()), n, n, nnz, False))
        if wrote != nnz:
            raise Error("RuntimeError: mojolearn: nonzero_f32_fill wrote " + String(wrote) + " of "
                        + String(nnz) + " entries")
    if nnz == 0:
        if not precomputed:
            raise Error("mojolearn: rows is empty, shape (0,): Found array with 0 sample(s) (shape=(0,)) while"
                        + " a minimum of 1 is required.")
        raise Error("mojolearn SpectralEmbedding: the precomputed affinity matrix has no nonzero entries")
    if precomputed:
        if not host_all_finite_f32(_F32P(unsafe_from_address=Int(room.values.unsafe_ptr())), nnz):
            raise Error("mojolearn SpectralEmbedding: the precomputed affinity matrix has a non-finite entry")
        if family_min_f32(_F32P(unsafe_from_address=Int(room.values.unsafe_ptr())), nnz) < 0:
            raise Error("mojolearn SpectralEmbedding: the precomputed affinity "
                        + "matrix has a negative entry (refused by name: sqrt of a "
                        + "negative degree is a NaN)")
    return room^


def _se_store(slot: _Slot, call: _Call) -> Optional[EntryRef]:
    """`_spectral_store` for a SpectralEmbedding fit: the dataset affinity ('nearest_neighbors', lane
    vdf-scope-cluster6) and the device-built 'rbf' affinity (lane vdf-scope-cluster8) read from the metrics image's
    resident store (`_store_affinity`)."""
    if slot[].kind != _KIND_EMBEDDING:
        return Optional[EntryRef]()
    if call[].scope == MEMBER_SCOPE_NONE or call[].pin == 0:
        return Optional[EntryRef]()
    try:
        var caps = SpectralDriver(call[].route.entry.call).metadata(SPECTRAL_CAPABILITIES)
        if (caps & SPECTRAL_CAP_DEV_STORE) == 0 or not _store_affinity(slot, caps):
            return Optional[EntryRef]()
    except:
        return Optional[EntryRef]()
    return metrics_store_find(entry_ref_mode(call[].route.entry), entry_ref_vendor(call[].route.entry))


def _fit_se(slot: _Slot, call: _Call) raises:
    """`SpectralEmbedding.fit(X)` over the whole X, or (MEMBER_CAN_ROW_INDEX, lane vdf-scope-cluster6) over a
    model-selection fold: taken on the device out of the scope's one copy of X when the affinity is the dataset
    route and the fit entry reads the metrics store (SPECTRAL_DATASET_STORE), else gathered on the host cores.
    The state owns its embedding: nothing points into the fold after the call."""
    if call[].row_index.rows == 0:
        _fit_se_rows(slot, call, -1)
        return
    if call[].column_index.rows > 0 or call[].x.home == MEMBER_HOME_DEVICE:
        raise Error("TypeError: a SpectralEmbedding member reads X as a dense host float32 view, no column index")
    var x = call[].x
    var index = call[].row_index
    var store = _se_store(slot, call)
    var failure = String("")
    if store:
        var taken = member_scope_rows_in(call[], store.value(), METRICS_STORE_PREFIX, True, _SE_EST)
        try:
            call[].x = MemberMatrix(0, taken.rows, taken.cols)
            call[].row_index = MemberMatrix()
            _fit_se_rows(slot, call, taken.source)
        except error:
            failure = String(error)
        call[].x = x
        call[].row_index = index
        member_scope_rows_done_in(store.value(), taken)
    else:
        var gathered = member_scope_rows_in(call[], call[].route.entry, METRICS_STORE_PREFIX, False, _SE_EST)
        try:
            call[].x = MemberMatrix(gathered.source, gathered.rows, gathered.cols)
            call[].row_index = MemberMatrix()
            _fit_se_rows(slot, call, -1)
        except error:
            failure = String(error)
        call[].x = x
        call[].row_index = index
        _ = gathered.rows  # the gathered rows stay alive through the fit
    if failure.byte_length() > 0:
        raise Error(failure)


def _fit_se_rows(slot: _Slot, call: _Call, store_slot: Int) raises:
    """Lane mf-member8: `SpectralEmbedding.fit(X)` (bindings/cfb_spectral_door.mojo
    `_se_execute`): the affinity's route and shape checks in its order, one
    EMBED call of the fit entry (`route.entry`), the embedding and the
    attributes its fit publishes kept (SCALARS). `store_slot >= 0`: the rows are that slot of the metrics store
    (the dataset affinity only; SPECTRAL_DATASET_STORE, the same kernels on the same words; `call.x` then names
    only the fold's shape)."""
    if store_slot < 0:
        _whole_x(call)
    if call[].weights.rows > 0:
        raise Error("TypeError: SpectralEmbedding takes no sample_weight")
    if call[].fit_output != 0:
        member_unsupported(_SE_EST, call[].fit_output)
    var rows = call[].x.rows
    var d = call[].x.cols
    var k = slot[].components
    var n_lanczos = k + 1 if slot[].drop_first else k
    var affinity = slot[].affinity
    var auto_neighbors = (slot[].automatic & 1) != 0
    var auto_gamma = (slot[].automatic & 2) != 0
    slot[].clear()
    var n = rows
    var neighbors_out = -1
    var features_out = -1
    var gamma_out = -1.0
    var coo = SpectralCOOStorage(0)
    var source_input: SpectralInput
    var neighbors_word = 0
    var status = SpectralAffinityStatus()
    var fused = ((affinity == _AFF_RBF or affinity == _AFF_PNN) and n > 0
        and (affinity == _AFF_RBF or d == n) and _fused_affinity(call, slot[].fit_dense))
    if store_slot >= 0 and not (affinity == _AFF_NN or (fused and affinity == _AFF_RBF)):
        raise Error("member ABI: a SpectralEmbedding fold on the metrics store serves the dataset route and the"
            + " device-built 'rbf' affinity only")
    if fused:
        # The fit entry builds the affinity on its device and embeds from it there (lane vdf-misc4's
        # `_se_execute` route); the 'rbf' affinity crosses once, into the state's published block.
        _se_check_shape(k, n, n_lanczos)
        neighbors_out = max(n // 10, 1) if auto_neighbors else slot[].neighbors
        slot[].affinity_values = List[Float32](length=n * n, fill=0.0)
        if affinity == _AFF_RBF:
            var gamma = 1.0 / Float64(d) if auto_gamma else slot[].gamma
            gamma_out = gamma
            features_out = d
            source_input = SpectralInput(fold_x_word(store_slot) if store_slot >= 0 else call[].x.data, 0, 0, 0,
                SPECTRAL_RBF, n, d, 0, Int(slot[].affinity_values.unsafe_ptr()), 0, gamma)
        else:
            source_input = SpectralInput(call[].x.data, 0, 0, 0, SPECTRAL_KNN, n, neighbors_out, 0,
                Int(slot[].affinity_values.unsafe_ptr()), status.address(), 0.0)
    elif affinity == _AFF_NN:
        _se_check_shape(k, n, n_lanczos)
        var neighbors = max(n // 10, 1) if auto_neighbors else slot[].neighbors
        if neighbors > n:
            raise Error("mojolearn SpectralEmbedding: n_neighbors=" + String(neighbors) + " "
                        + "exceeds n_samples=" + String(n))
        neighbors_out = neighbors
        neighbors_word = neighbors
        features_out = d
        if store_slot >= 0:
            source_input = SpectralInput(store_slot, 0, 0, 0, SPECTRAL_DATASET_STORE, rows, d, 0)
        else:
            source_input = SpectralInput(call[].x.data, 0, 0, 0, SPECTRAL_DATASET, rows, d, 0)
    else:
        var dense_address = call[].x.data
        if affinity == _AFF_PNN:
            var saved = slot[].neighbors
            slot[].neighbors = max(n // 10, 1) if auto_neighbors else saved
            _knn_affinity(slot, call, rows, d, call[].x.data)
            slot[].neighbors = saved
            dense_address = Int(slot[].affinity_values.unsafe_ptr())
        elif affinity == _AFF_PRECOMPUTED:
            # The class's square check (`_se_execute`), for the dense and the COO route alike.
            if rows != d:
                raise Error("mojolearn SpectralEmbedding: with affinity='precomputed', X "
                            + "must be a square affinity matrix or a SciPy sparse matrix, "
                            + "got shape (" + String(rows) + ", " + String(d) + ")")
        else:
            var gamma = 1.0 / Float64(d) if auto_gamma else slot[].gamma
            gamma_out = gamma
            features_out = d
            var kept = slot[].gamma
            slot[].gamma = gamma
            _rbf_affinity(slot, call, rows, d, call[].x.data)
            slot[].gamma = kept
            dense_address = Int(slot[].affinity_values.unsafe_ptr())
        neighbors_out = max(n // 10, 1) if auto_neighbors else slot[].neighbors
        if slot[].fit_dense:
            _se_check_shape(k, n, n_lanczos)
            source_input = SpectralInput(dense_address, 0, 0, 0, SPECTRAL_DENSE, n, 0, 0)
        else:
            coo = _se_coo(call, dense_address, n, affinity == _AFF_PRECOMPUTED)
            _se_check_shape(k, n, n_lanczos)
            source_input = SpectralInput(0, Int(coo.rows.unsafe_ptr()), Int(coo.columns.unsafe_ptr()),
                Int(coo.values.unsafe_ptr()), SPECTRAL_GRAPH, n, 0, len(coo.values))
    slot[].embedding = List[Float32](length=n * k, fill=0.0)
    var request = SpectralEmbeddingRequest(source_input, Int(slot[].embedding.unsafe_ptr()),
        SpectralEmbeddingOptions(n_lanczos, k, neighbors_word, slot[].norm_laplacian, slot[].drop_first,
            slot[].seed, slot[].tolerance))
    var n_out = SpectralDriver(call[].route.entry.call).embed(request)
    _ = len(coo.values)  # the COO triples the embedding read stay alive through it
    if fused and n_out == SPECTRAL_KNN_REFUSED:
        _knn_status_refuse(status, neighbors_out)
    _ = status^
    if n_out != k:
        raise Error("RuntimeError: mojolearn SpectralEmbedding: the kernel returned " + String(n_out) + " "
                    + "embedding columns where " + String(k) + " were expected")
    # The 'rbf' affinity is published (affinity_matrix_); the other built one is not.
    if affinity != _AFF_RBF:
        slot[].affinity_values = List[Float32]()
    var scalars: List[Float64] = [Float64(k), Float64(neighbors_out), Float64(features_out), gamma_out]
    slot[].scalars = scalars^
    slot[].rows = n
    slot[].features = d
    slot[].borrowed = 0
    slot[].fitted = True


def _import_se(slot: _Slot, call: _Call) raises:
    """Round b (lane mf-member8): a fitted SpectralEmbedding held by a
    meta-estimator's state: the embedding (rows x n_components), the built
    'rbf' affinity (optional) and SCALARS [n_components_, n_neighbors_,
    n_features_in_, gamma_]; MEMBER_IMPORT_DONE needs the embedding and the
    scalars, of n_components_ columns."""
    var block = call[].block
    slot[].fitted = False
    if block == _BLOCK_EMBEDDING:
        slot[].embedding = _take_f32(call, "embedding", -1, 0)
        slot[].rows = call[].x.rows
    elif block == _BLOCK_AFFINITY:
        slot[].affinity_values = _take_f32(call, "affinity", -1, 0)
    elif block == MEMBER_BLOCK_SCALARS:
        if not call[].x.dense_host(MEMBER_F64) or call[].x.rows * call[].x.cols != 4:
            raise Error("member ABI: the SpectralEmbedding scalars block is four host float64 words")
        var source = _Words(unsafe_from_address=call[].x.data)
        var scalars: List[Float64] = [source[0], source[1], source[2], source[3]]
        slot[].scalars = scalars^
    elif block != 0 or (call[].flags & MEMBER_IMPORT_DONE) == 0:
        raise Error("member ABI: a SpectralEmbedding member has no fitted block " + String(block))
    if (call[].flags & MEMBER_IMPORT_DONE) == 0:
        return
    var k = slot[].components
    if (len(slot[].scalars) != 4 or Int(slot[].scalars[0]) != k or slot[].rows <= 0
            or len(slot[].embedding) != slot[].rows * k):
        raise Error("member ABI: a fitted SpectralEmbedding state needs its embedding and scalars blocks")
    slot[].features = max(Int(slot[].scalars[2]), 0)
    slot[].fitted = True


def _predict(slot: _Slot, call: _Call) raises:
    """`_sc_predict` (affinity='nearest_neighbors'): the Nystrom extension of
    the fitted eigenvectors, then the nearest fitted centroid; on the
    resident state of a SPECTRAL_CAP_RESIDENT image (prepared once per
    fitted state), else the plain query. The embedding is per-call scratch,
    as the class's own predict allocates it."""
    if not slot[].fitted or not slot[].keep_state:
        raise Error("mojolearn SpectralClustering.predict: prediction data was not stored. Fit with"
            + " SpectralClustering(prediction_data=True), which keeps the eigenpairs, degrees and"
            + " centroids the Nystrom extension needs (DEVIATION 2860)")
    # Under a row index (lane vdf-scope-cluster4) the queries are the index's rows, gathered on the host cores;
    # a state holding THE FITTED-ROWS SLOT under the caller's scope with X's pin takes them ON THE DEVICE out of
    # the scope's one copy of X instead (lane vdf-scope-cluster8): the predict entry's k-NN search reads both
    # slots there (SPECTRAL_CAP_DEV_QUERY).
    var qx = MemberScopeRows(call[].x.data, call[].x.rows, call[].x.cols, False, List[Float32]())
    if call[].row_index.rows == 0:
        _whole_x(call)
    else:
        if call[].column_index.rows > 0 or call[].x.home == MEMBER_HOME_DEVICE:
            raise Error("TypeError: a SpectralClustering member reads X as a dense host float32 view, no column"
                + " index")
        var on_store = (slot[].rows_slot >= 0 and slot[].affinity == _AFF_NN and call[].scope != MEMBER_SCOPE_NONE
            and call[].pin != 0)
        qx = member_scope_rows_in(call[], slot[].rows_store if on_store else call[].route.entry,
            METRICS_STORE_PREFIX, on_store, _EST)
    try:
        _predict_rows(slot, call, qx)
    finally:
        member_scope_rows_done_in(slot[].rows_store, qx)
    _ = qx.rows  # the gathered queries stay alive through the query calls above


def _predict_rows(slot: _Slot, call: _Call, qx: MemberScopeRows) raises:
    """`_predict` over the queries `qx`: host rows, or (`qx.resident`) a slot of the metrics store, named to the
    predict entry by its negative word (core/fold_x_word.mojo). The training rows are the state's host words,
    or THE FITTED-ROWS SLOT, named the same way to `prepare`."""
    var queries = qx.rows
    var d = slot[].features
    var code = 0
    if slot[].affinity == _AFF_PRECOMPUTED:
        # `_sc_predict`'s precomputed arm: X is the affinity between the new
        # rows and the training rows (finite: admitted by the caller).
        if qx.cols != slot[].rows:
            raise Error("mojolearn SpectralClustering.predict: with affinity="
                        + "'precomputed', X must be the affinity between the new rows "
                        + "and the " + String(slot[].rows) + " training rows, shape (n_new, "
                        + String(slot[].rows) + "); got shape (" + String(queries) + ", " + String(qx.cols)
                        + ")")
        if queries < 1:
            raise Error("mojolearn SpectralClustering.predict: X has no rows; refused by name")
        if family_min_f32(_F32P(unsafe_from_address=qx.source), queries * qx.cols) < 0:
            raise Error("mojolearn SpectralClustering.predict: the affinity has a "
                        + "negative entry; refused by name, as the fit refuses one")
        d = 0
        code = 1
    else:
        if qx.cols != d:
            raise Error("ValueError: X has " + String(qx.cols) + " features, but"
                + " SpectralClustering.predict is expecting " + String(d) + " features as input")
        if queries < 1:
            raise Error("mojolearn SpectralClustering.predict: X has no rows; refused by name")
    _labels_out(call, queries)
    var k = slot[].components
    var shape = SpectralQueryShape(slot[].rows, queries, d, k, slot[].clusters, slot[].neighbors, code)
    var driver = _query_driver(call)
    var training = slot[].training_address() if code == 0 else 0
    if code == 0 and slot[].rows_slot >= 0:
        training = fold_x_word(slot[].rows_slot)
    var source = fold_x_word(qx.source) if qx.resident else qx.source
    var embedding = List[Float32](length=queries * k, fill=0.0)
    if slot[].handle < 0 and (call[].route.flags & SPECTRAL_CAP_RESIDENT) != 0:
        slot[].handle = driver.prepare(SpectralPrepareRequest(training, slot[].state_view(), shape))
    if slot[].handle >= 0:
        _ = driver.query_resident(SpectralResidentQueryRequest(slot[].handle, source, call[].dst.data,
            Int(embedding.unsafe_ptr()), shape))
    else:
        if training < 0 or source < 0:
            raise Error("member ABI: a SpectralClustering state's device-resident rows need a resident predict"
                + " entry (SPECTRAL_CAP_RESIDENT)")
        _ = driver.query(SpectralQueryRequest(source, training, slot[].state_view(), call[].dst.data,
            Int(embedding.unsafe_ptr()), shape))
    _ = embedding^


def _f32_view(values: List[Float32], rows: Int, cols: Int) -> MemberMatrix:
    if len(values) == 0:
        return MemberMatrix(0, 0, cols)
    return MemberMatrix(Int(values.unsafe_ptr()), rows, cols)


def _own_refs(slot: _Slot) raises:
    """Copy the borrowed training rows into the state, once (the class's
    `_fit_X` copy); the state reads its own copy from then on. The resident
    state (if any) holds the same words and stays. THE FITTED-ROWS SLOT is
    read home once (`arena_store_read`) and freed."""
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
        raise Error("NotFittedError: this SpectralClustering member is not fitted yet")
    if (call[].flags & MEMBER_EXPORT_OWN) != 0:
        # THE LIFETIME RULE: the training rows, once; no view is answered.
        _own_refs(slot)
        return
    var block = call[].block
    if (call[].flags & MEMBER_BLOCK_BY_INDEX) != 0:
        if block < 0 or block >= _BLOCKS:
            raise Error("member ABI: a SpectralClustering member has nine fitted blocks")
        block = MEMBER_BLOCK_SCALARS if block == 7 else (_BLOCK_AFFINITY if block == 8 else MEMBER_BLOCK_FAMILY + block)
    var n = slot[].rows
    var k = slot[].components
    if block == _BLOCK_LABELS:
        if len(slot[].labels) == 0:
            call[].dst = MemberMatrix(0, 0, 0, MEMBER_I32)
        else:
            call[].dst = MemberMatrix(Int(slot[].labels.unsafe_ptr()), 1, len(slot[].labels), MEMBER_I32)
    elif block == _BLOCK_EMBEDDING:
        call[].dst = _f32_view(slot[].embedding, len(slot[].embedding) // k if k > 0 else 0, k)
    elif block == _BLOCK_EIGENVALUES:
        call[].dst = _f32_view(slot[].eigenvalues, 1, k)
    elif block == _BLOCK_EIGENVECTORS:
        call[].dst = _f32_view(slot[].eigenvectors, n, k)
    elif block == _BLOCK_DIAGONAL:
        call[].dst = _f32_view(slot[].diagonal, 1, n)
    elif block == _BLOCK_CENTROIDS:
        call[].dst = _f32_view(slot[].centroids, slot[].clusters, k)
    elif block == _BLOCK_TRAINING:
        _own_refs(slot)
        call[].dst = _f32_view(slot[].refs, n, slot[].features)
    elif block == _BLOCK_AFFINITY:
        call[].dst = _f32_view(slot[].affinity_values, n, n)
    elif block == MEMBER_BLOCK_SCALARS:
        call[].dst = MemberMatrix(Int(slot[].scalars.unsafe_ptr()), 1, len(slot[].scalars), MEMBER_F64)
    else:
        raise Error("member ABI: a SpectralClustering member has no fitted block " + String(block))
    call[].block = block
    call[].out_cols = _BLOCKS


def _take_f32(call: _Call, what: String, rows: Int, cols: Int) raises -> List[Float32]:
    var count = call[].x.rows * call[].x.cols
    if count > 0 and not call[].x.dense_host(MEMBER_F32):
        raise Error("member ABI: the SpectralClustering " + what + " block is dense host float32")
    if rows >= 0 and count != rows * cols:
        raise Error("member ABI: the SpectralClustering " + what + " block holds " + String(rows * cols)
            + " float32 words")
    var values = List[Float32](length=count, fill=0.0)
    if count > 0:
        unsafe_memcpy(dest=values.unsafe_ptr(), src=_F32P(unsafe_from_address=call[].x.data), count=count)
    return values^


def _import(slot: _Slot, call: _Call) raises:
    """Copy one fitted block in; MEMBER_IMPORT_DONE completes the state. The
    training block comes first (it sets rows and features)."""
    var block = call[].block
    if block != 0 or (call[].flags & MEMBER_IMPORT_DONE) == 0:
        _release_resident(slot, call)
        _drop_rows_slot(slot)
        slot[].fitted = False
    var k = slot[].components
    if block == _BLOCK_LABELS:
        var count = call[].x.rows * call[].x.cols
        if count > 0 and not call[].x.dense_host(MEMBER_I32):
            raise Error("member ABI: the SpectralClustering labels block is dense host int32")
        slot[].labels = List[Int32](length=count, fill=0)
        if count > 0:
            unsafe_memcpy(dest=slot[].labels.unsafe_ptr(), src=_I32P(unsafe_from_address=call[].x.data), count=count)
    elif block == _BLOCK_EMBEDDING:
        slot[].embedding = _take_f32(call, "embedding", -1, 0)
    elif block == _BLOCK_EIGENVALUES:
        slot[].eigenvalues = _take_f32(call, "eigenvalues", 1, k)
    elif block == _BLOCK_EIGENVECTORS:
        slot[].eigenvectors = _take_f32(call, "eigenvectors", slot[].rows, k)
    elif block == _BLOCK_DIAGONAL:
        slot[].diagonal = _take_f32(call, "degree diagonal", 1, slot[].rows)
    elif block == _BLOCK_CENTROIDS:
        slot[].centroids = _take_f32(call, "centroids", slot[].clusters, k)
    elif block == _BLOCK_TRAINING:
        slot[].borrowed = 0
        slot[].refs = _take_f32(call, "training rows", -1, 0)
        slot[].rows = call[].x.rows
        slot[].features = call[].x.cols
    elif block == MEMBER_BLOCK_SCALARS:
        if not call[].x.dense_host(MEMBER_F64) or call[].x.rows * call[].x.cols != _SCALARS:
            raise Error("member ABI: the SpectralClustering scalars block is two host float64 words"
                + " (n_components_, n_iter_assign_)")
        var source = _Words(unsafe_from_address=call[].x.data)
        var scalars: List[Float64] = [source[0], source[1]]
        slot[].scalars = scalars^
    elif block != 0 or (call[].flags & MEMBER_IMPORT_DONE) == 0:
        raise Error("member ABI: a SpectralClustering member has no fitted block " + String(block))
    if (call[].flags & MEMBER_IMPORT_DONE) == 0:
        return
    var n = slot[].rows
    # A precomputed state keeps no training rows: its training block is
    # n x 0 (it carries n only).
    var precomputed = slot[].affinity == _AFF_PRECOMPUTED
    var rows_ok = (len(slot[].refs) == 0) if precomputed else (
        slot[].features > 0 and len(slot[].refs) == n * slot[].features)
    if (not rows_ok or n <= 0
            or len(slot[].eigenvalues) != k or len(slot[].eigenvectors) != n * k
            or len(slot[].diagonal) != n or len(slot[].centroids) != slot[].clusters * k
            or len(slot[].scalars) != _SCALARS):
        raise Error("member ABI: a fitted SpectralClustering state needs its training rows, eigenvalues,"
            + " eigenvectors, degree diagonal, centroids and scalars blocks")
    slot[].fitted = True


def _spectral_member(operation: Int, address: Int) raises -> Int:
    var call = _Call(unsafe_from_address=address)
    if operation == MEMBER_OP_CAPABILITIES:
        # No MEMBER_CAN_TEST_ROWS: X is admitted finite by the caller (the
        # spectral and assignment entries do not test it).
        # MEMBER_BORROWS_FIT: a fitted state reads the caller's X (its
        # training rows) until an export of the rows block, or
        # MEMBER_EXPORT_OWN, copied them.
        # Lane mf-member8: a SpectralEmbedding state fits and exports (it
        # owns its embedding; it reads no fit buffer after FIT).
        if call[].state != 0:
            var held = _Slot(unsafe_from_address=_lookup(call[].state, False))
            if held[].kind == _KIND_EMBEDDING:
                # MEMBER_CAN_ROW_INDEX (lane vdf-scope-cluster6): a fold's rows (gathered, or taken on the device
                # from the metrics store: MEMBER_CAN_SCOPE, as SpectralClustering below).
                var se = MEMBER_CAN_FIT | MEMBER_CAN_EXPORT | MEMBER_CAN_IMPORT | MEMBER_IGNORES_Y | MEMBER_CAN_ROW_INDEX
                try:
                    var se_caps = SpectralDriver(call[].route.entry.call).metadata(SPECTRAL_CAPABILITIES)
                    if (se_caps & SPECTRAL_CAP_DEV_STORE) != 0 and metrics_store_find(
                            entry_ref_mode(call[].route.entry), entry_ref_vendor(call[].route.entry)):
                        se |= MEMBER_CAN_SCOPE
                except:
                    pass
                return se
        # MEMBER_CAN_ROW_INDEX (lane vdf-scope-cluster4): a SpectralClustering call reads a host row index (a
        # gathered fold owns its rows); MEMBER_CAN_SCOPE when the fit entry reads the metrics image's resident
        # store (SPECTRAL_CAP_DEV_STORE and the store published for its (mode, vendor)): a search's dataset-route
        # folds are then taken on the device out of one copy of X.
        var sc = (MEMBER_CAN_FIT | MEMBER_CAN_PREDICT | MEMBER_CAN_EXPORT | MEMBER_CAN_IMPORT | MEMBER_BORROWS_FIT
            | MEMBER_CAN_ROW_INDEX)
        try:
            var caps = SpectralDriver(call[].route.entry.call).metadata(SPECTRAL_CAPABILITIES)
            if (caps & SPECTRAL_CAP_DEV_STORE) != 0 and metrics_store_find(
                    entry_ref_mode(call[].route.entry), entry_ref_vendor(call[].route.entry)):
                sc |= MEMBER_CAN_SCOPE
        except:
            pass  # a route without a fit entry answers no MEMBER_CAN_SCOPE
        return sc
    if operation == MEMBER_OP_NEW and call[].params != 0 and call[].param_count == _SE_WORDS:
        var se_words = _Words(unsafe_from_address=call[].params)
        if Int(se_words[0]) != _KIND_EMBEDDING:
            raise Error("member ABI: a SpectralEmbedding member's first word is kind 1")
        call[].state = _mint(se_words)
        return 0
    if operation == MEMBER_OP_NEW:
        if not call[].params or call[].param_count != _WORDS:
            raise Error("member ABI: a SpectralClustering member takes twelve parameter words (kind, n_clusters,"
                + " n_components, n_init, n_neighbors, eigen_tol, seed bits, assign_labels code, affinity code,"
                + " prediction_data, gamma, dense fit entry)")
        var words = _Words(unsafe_from_address=call[].params)
        if Int(words[0]) != _KIND_SPECTRAL:
            raise Error("member ABI: spectral member kind " + String(Int(words[0])) + " is not published")
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
    if slot[].kind == _KIND_EMBEDDING and operation == MEMBER_OP_FIT:
        _fit_se(slot, call)
    elif slot[].kind == _KIND_EMBEDDING and operation == MEMBER_OP_IMPORT:
        _import_se(slot, call)
    elif slot[].kind == _KIND_EMBEDDING and operation == MEMBER_OP_PREDICT:
        # scikit-learn's own failure (round b): SpectralEmbedding has no predict.
        raise Error("AttributeError: 'SpectralEmbedding' object has no attribute 'predict'")
    elif slot[].kind == _KIND_EMBEDDING and operation != MEMBER_OP_EXPORT:
        member_unsupported(_SE_EST, operation)
    elif operation == MEMBER_OP_FIT:
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


def spectral_member_v2(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
    result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """The member entry of the Spectral family (schema `MEMBER_ABI`)."""
    try:
        var word = _spectral_member(operation, member_call_from_spans(operation, aa, na, np, nf))
        if result:
            MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = word
        return 0
    except error:
        native_descriptor_report(error, context, sink)
        return 1


def spectral_member_publish(mode: Int, vendor: Int, publisher: Int) raises:
    """Publish this image's Spectral member entry. One line in the image's
    module initialization, after its spectral kernel table exists; a build
    with no vendor (`vendor` < 0) publishes nothing."""
    if vendor < 0:
        return
    member_publish[MEMBER_FAMILY_SPECTRAL](mode, vendor, publisher, spectral_member_v2)
