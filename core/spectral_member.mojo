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

Kind (MEMBER_OP_NEW word [0]): 0 SpectralClustering in the form that has a
predict: affinity='nearest_neighbors', prediction_data=True (the class's
own checks decide the rest at the caller's UNPACK). 'precomputed' words are
tied to the training rows, and 'rbf' / 'precomputed_nearest_neighbors' have
no predict: the caller's admit answers False and the class keeps its doors.

MEMBER_OP_NEW parameters (Float64 words, 8): [0] 0, [1] n_clusters,
[2] n_components (resolved: n_clusters for None), [3] n_init,
[4] n_neighbors, [5] eigen_tol (0 for 'auto'), [6] the seed as the BITS of
an int64 (`bitcast[DType.float64](Int64(seed))`, the class's own seed
rule), [7] assign_labels code (-1 'kmeans', 0 'discretize', 1 'cluster_qr').
A clusterer, not a classifier: `y` is not read, `classes` is 0.

Views: X dense row-major host float32, BORROWED (never copied at fit): the
fit entry reads the caller's words, and predict's Nystrom extension reads the
training rows through the same view, so THE CALLER KEEPS X ALIVE UNTIL THIS
STATE IS FREED OR REFITTED (the member ABI's view lifetime). MEMBER_OP_EXPORT
of the training block copies the rows into the state once (the class's own
fit copies them into `_fit_X` on every fit); from then on the state reads its
own copy. No finite scan here (MEMBER_CAN_TEST_ROWS clear: the caller's
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
6 `MEMBER_BLOCK_FAMILY + 6` float32 rows x features, the training rows;
7 `MEMBER_BLOCK_SCALARS` float64 [n_components_, n_iter_assign_ (-1: none)].
An import needs blocks 2 to 7 (labels and embedding are optional).

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
from spectral.cfb_spectral_run import _sc_check_shape

comptime _CHUNK = 256
comptime _INDEX_SPAN = 16777216
comptime _EST = "SpectralClustering"
comptime _KIND_SPECTRAL = 0
comptime _WORDS = 8
comptime _BLOCKS = 8
comptime _BLOCK_LABELS = MEMBER_BLOCK_FAMILY
comptime _BLOCK_EMBEDDING = MEMBER_BLOCK_FAMILY + 1
comptime _BLOCK_EIGENVALUES = MEMBER_BLOCK_FAMILY + 2
comptime _BLOCK_EIGENVECTORS = MEMBER_BLOCK_FAMILY + 3
comptime _BLOCK_DIAGONAL = MEMBER_BLOCK_FAMILY + 4
comptime _BLOCK_CENTROIDS = MEMBER_BLOCK_FAMILY + 5
comptime _BLOCK_TRAINING = MEMBER_BLOCK_FAMILY + 6
comptime _SCALARS = 2
comptime _I32P = MutPointer[Int32, MutUntrackedOrigin]
comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]


struct SpectralMemberSlot(Copyable, Movable, Defaultable):
    var generation: Int
    var live: Bool
    var clusters: Int
    var components: Int
    var initializations: Int
    var neighbors: Int
    var tolerance: Float64
    var seed: Int
    var assignment: Int
    var fitted: Bool
    var rows: Int
    var features: Int
    # The training rows: the caller's borrowed X after a fit (`borrowed`,
    # alive until free or refit), else the state's own copy in `refs`.
    var borrowed: Int
    var refs: List[Float32]
    # The resident prediction state of a SPECTRAL_CAP_RESIDENT image (-1: none).
    var handle: Int
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
        self.clusters = 0
        self.components = 0
        self.initializations = 0
        self.neighbors = 0
        self.tolerance = 0.0
        self.seed = 0
        self.assignment = -1
        self.fitted = False
        self.rows = 0
        self.features = 0
        self.borrowed = 0
        self.refs = List[Float32]()
        self.handle = -1
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


def _fit(slot: _Slot, call: _Call) raises:
    """The class's dataset route (`_sc_fit`, affinity='nearest_neighbors',
    prediction_data=True): its shape checks, one fit entry call on the
    borrowed X keeping the prediction state, then the assignment entry for
    'discretize' / 'cluster_qr' (the class's 30 restarts, 20 iterations)."""
    _whole_x(call)
    if call[].weights.rows > 0:
        raise Error("TypeError: SpectralClustering takes no sample_weight")
    var output = call[].fit_output
    var rows = call[].x.rows
    var d = call[].x.cols
    if output == MEMBER_OP_PREDICT:
        _labels_out(call, rows)
    elif output != 0:
        member_unsupported(_EST, output)
    var clusters = slot[].clusters
    var k = slot[].components
    _sc_check_shape(clusters, rows, k)
    if slot[].neighbors > rows:
        raise Error("mojolearn SpectralClustering: n_neighbors=" + String(slot[].neighbors)
                    + " exceeds n_samples=" + String(rows))
    if slot[].assignment >= 0 and not call[].route.has_third:
        raise Error("member ABI: a SpectralClustering member's route holds no assignment entry")
    _release_resident(slot, call)
    slot[].clear()
    slot[].labels = List[Int32](length=rows, fill=0)
    slot[].embedding = List[Float32](length=rows * k, fill=0.0)
    slot[].eigenvalues = List[Float32](length=k, fill=0.0)
    slot[].eigenvectors = List[Float32](length=rows * k, fill=0.0)
    slot[].diagonal = List[Float32](length=rows, fill=0.0)
    slot[].centroids = List[Float32](length=clusters * k, fill=0.0)
    var options = SpectralFitOptions(clusters, k, slot[].initializations, slot[].neighbors, slot[].tolerance,
        slot[].seed, True)
    var request = SpectralFitRequest(SpectralInput(call[].x.data, 0, 0, 0, SPECTRAL_DATASET, rows, d, 0),
        Int(slot[].labels.unsafe_ptr()), Int(slot[].embedding.unsafe_ptr()), slot[].state_view(), options)
    var n_out = SpectralDriver(call[].route.entry.call).fit(request)
    if n_out != k:
        raise Error("RuntimeError: mojolearn SpectralClustering: the kernel returned " + String(n_out) + " "
                    + "embedding columns where " + String(k) + " were expected")
    var iterations = -1
    if slot[].assignment >= 0:
        var assign = SpectralAssignmentRequest(Int(slot[].embedding.unsafe_ptr()), rows * k, rows, k,
            slot[].assignment, slot[].seed, 30, 20)
        var result = SpectralAssignmentOutput()
        SpectralAssignmentDriver(call[].route.third.call).assign(assign, result)
        if len(result.storage[0].labels) != rows:
            raise Error("RuntimeError: mojolearn SpectralClustering: the assignment returned "
                        + String(len(result.storage[0].labels)) + " labels where " + String(rows) + " were expected")
        if slot[].assignment == 0:
            iterations = result.storage[0].iterations
        swap(slot[].labels, result.storage[0].labels)
        _ = result^
    var scalars: List[Float64] = [Float64(k), Float64(iterations)]
    slot[].scalars = scalars^
    slot[].rows = rows
    slot[].features = d
    slot[].borrowed = call[].x.data
    slot[].fitted = True
    if output == MEMBER_OP_PREDICT:
        unsafe_memcpy(dest=_I32P(unsafe_from_address=call[].dst.data), src=slot[].labels.unsafe_ptr(), count=rows)


def _predict(slot: _Slot, call: _Call) raises:
    """`_sc_predict` (affinity='nearest_neighbors'): the Nystrom extension of
    the fitted eigenvectors, then the nearest fitted centroid; on the
    resident state of a SPECTRAL_CAP_RESIDENT image (prepared once per
    fitted state), else the plain query. The embedding is per-call scratch,
    as the class's own predict allocates it."""
    if not slot[].fitted:
        raise Error("mojolearn SpectralClustering.predict: prediction data was not stored. Fit with"
            + " SpectralClustering(prediction_data=True), which keeps the eigenpairs, degrees and"
            + " centroids the Nystrom extension needs (DEVIATION 2860)")
    _whole_x(call)
    var queries = call[].x.rows
    var d = slot[].features
    if call[].x.cols != d:
        raise Error("ValueError: X has " + String(call[].x.cols) + " features, but SpectralClustering.predict"
            + " is expecting " + String(d) + " features as input")
    if queries < 1:
        raise Error("mojolearn SpectralClustering.predict: X has no rows; refused by name")
    _labels_out(call, queries)
    var k = slot[].components
    var shape = SpectralQueryShape(slot[].rows, queries, d, k, slot[].clusters, slot[].neighbors, 0)
    var driver = _query_driver(call)
    var training = slot[].training_address()
    var embedding = List[Float32](length=queries * k, fill=0.0)
    if slot[].handle < 0 and (call[].route.flags & SPECTRAL_CAP_RESIDENT) != 0:
        slot[].handle = driver.prepare(SpectralPrepareRequest(training, slot[].state_view(), shape))
    if slot[].handle >= 0:
        _ = driver.query_resident(SpectralResidentQueryRequest(slot[].handle, call[].x.data, call[].dst.data,
            Int(embedding.unsafe_ptr()), shape))
    else:
        _ = driver.query(SpectralQueryRequest(call[].x.data, training, slot[].state_view(), call[].dst.data,
            Int(embedding.unsafe_ptr()), shape))
    _ = embedding^


def _f32_view(values: List[Float32], rows: Int, cols: Int) -> MemberMatrix:
    if len(values) == 0:
        return MemberMatrix(0, 0, cols)
    return MemberMatrix(Int(values.unsafe_ptr()), rows, cols)


def _own_refs(slot: _Slot):
    """Copy the borrowed training rows into the state, once (the class's
    `_fit_X` copy); the state reads its own copy from then on. The resident
    state (if any) holds the same words and stays."""
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
    var block = call[].block
    if (call[].flags & MEMBER_BLOCK_BY_INDEX) != 0:
        if block < 0 or block >= _BLOCKS:
            raise Error("member ABI: a SpectralClustering member has eight fitted blocks")
        block = MEMBER_BLOCK_SCALARS if block == _BLOCKS - 1 else MEMBER_BLOCK_FAMILY + block
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
    elif block == MEMBER_BLOCK_SCALARS:
        call[].dst = MemberMatrix(Int(slot[].scalars.unsafe_ptr()), 1, _SCALARS, MEMBER_F64)
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
    if (slot[].features <= 0 or n <= 0 or len(slot[].refs) != n * slot[].features
            or len(slot[].eigenvalues) != k or len(slot[].eigenvectors) != n * k
            or len(slot[].diagonal) != n or len(slot[].centroids) != slot[].clusters * k
            or len(slot[].scalars) != _SCALARS):
        raise Error("member ABI: a fitted SpectralClustering state needs its training rows, eigenvalues,"
            + " eigenvectors, degree diagonal, centroids and scalars blocks")
    slot[].fitted = True


def _spectral_member(operation: Int, address: Int) raises -> Int:
    var call = _Call(unsafe_from_address=address)
    if operation == MEMBER_OP_CAPABILITIES:
        return MEMBER_CAN_FIT | MEMBER_CAN_PREDICT | MEMBER_CAN_EXPORT | MEMBER_CAN_IMPORT
    if operation == MEMBER_OP_NEW:
        if not call[].params or call[].param_count != _WORDS:
            raise Error("member ABI: a SpectralClustering member takes eight parameter words (kind, n_clusters,"
                + " n_components, n_init, n_neighbors, eigen_tol, seed bits, assign_labels code)")
        var words = _Words(unsafe_from_address=call[].params)
        if Int(words[0]) != _KIND_SPECTRAL:
            raise Error("member ABI: spectral member kind " + String(Int(words[0])) + " is not published")
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
