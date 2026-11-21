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
UNPACK). `route.second` (MeanShift, AffinityPropagation) is the image's centroid-fit entry
(core/cluster_centroid_fit_api.mojo `CentroidFitDriver`). `route.third`
(BayesianGaussianMixture) is the image's typed BGMM entry
(core/bgmm_native_api.mojo `BgmmDriver`: ENTRY_BGMM fit, ENTRY_BGMM_SCORE).
No other route slot is read by the kinds published here. No kernel is touched.

Kinds (MEMBER_OP_NEW word [0]): 0 MiniBatchKMeans, 1 BisectingKMeans,
2 MeanShift, 4 AffinityPropagation, 5 BayesianGaussianMixture. Kind 3
OPTICS (the class has no predict: not a member) is reserved and refused.

MEMBER_OP_NEW parameters (Float64 words):
  kind 0 MiniBatchKMeans (11 words): [0] 0, [1] n_clusters, [2] init
  (0 'k-means++', 1 'random'; explicit starting centers are not a word and
  the class keeps its own doors for them), [3] n_init ('auto' resolved by
  the caller's UNPACK with the estimator's own rule), [4] max_iter, [5] tol,
  [6] random_state as int64 BITS (as kind 1), [7] batch_size,
  [8] max_no_improvement (-1 for None), [9] init_size (0 for None),
  [10] reassignment_ratio.
  kind 1 BisectingKMeans (8 words): [0] 1, [1] n_clusters, [2] init
  (0 'random', 1 'k-means++'), [3] n_init, [4] max_iter, [5] tol,
  [6] random_state as the BITS of an int64 (`bitcast[DType.float64](Int64(seed))`,
  the estimator's own seed rule: None is 0, the low 63 bits of an int),
  [7] bisecting_strategy (0 'biggest_inertia', 1 'largest_cluster').
  kind 2 MeanShift (6 words): [0] 2, [1] bandwidth (0 for None: the entry
  estimates it), [2] cluster_all (0/1), [3] max_iter, [4] bin_seeding (0/1),
  [5] min_bin_freq. Explicit seeds are not words: the class keeps its own
  doors for them. MeanShift takes no sample weights and has no transform or
  score (refused in words).
  kind 4 AffinityPropagation (7 words): [0] 4, [1] damping, [2] max_iter,
  [3] convergence_iter, [4] random_state as int64 BITS, [5] preference mode
  (0 the median similarity, 1 a scalar), [6] the scalar preference. A
  per-sample preference and affinity='precomputed' are not members (their
  words are tied to the training rows): the class keeps its own doors. No
  sample weights, no transform or score (refused in words). Predict writes
  -1 for every row when the fit found no exemplar, as the class does.
  kind 5 BayesianGaussianMixture (13 words): [0] 5, [1] n_components,
  [2] covariance_type (0 'full', 1 'tied', 2 'diag', 3 'spherical'),
  [3] init_params (0 'kmeans', 1 'random', 2 'k-means++',
  3 'random_from_data'), [4] weight_concentration_prior_type
  (1 'dirichlet_process', 0 'dirichlet_distribution'),
  [5] weight_concentration_prior (-1 for None), [6] mean_precision_prior
  (-1 for None), [7] degrees_of_freedom_prior (-1 for None), [8] reg_covar,
  [9] tol, [10] max_iter, [11] n_init, [12] random_state as int64 BITS (the
  estimator's own seed rule). An array mean_prior or covariance_prior and
  warm_start=True are not words: the class keeps its own doors for them.
  No sample weights, no transform (refused in words). Predict writes the
  first largest log responsibility per row (int32), predict_proba the
  responsibilities (float32 rows x n_components), score_samples the
  per-row log likelihood (float32 rows x 1) and score `call.value` = the
  entry's float-float fold of the per-row log likelihoods over the row
  count, as the class's own `score`.
A clusterer, not a classifier: `y` is not read, `classes` is 0.

Views: X dense row-major host float32, BORROWED for the call (never copied:
the kernel reads the caller's words, as the estimator's own door reads the
admitted array); sample weights (fit only) a dense host float32 vector
(MEMBER_CAN_SAMPLE_WEIGHT; negative weights are the estimator's own rule).
X is admitted finite by the caller's UNPACK, as by the estimator's own door.
No row or column index and no device-home view: refused in words (the caller
hands gathered host words).

Outputs: predict writes cluster ids as int32 (MEMBER_I32, rows x 1): the
nearest center for kind 0, the bisection-tree descent of
`BisectingKMeans.predict` for kind 1; transform writes
float32 distances to the leaf centers (rows x n_clusters); score sets
`call.value` to minus the inertia of X under the leaf centers (no weights:
the class's score counts every row once, so a weight is refused).
`call.fit_output = MEMBER_OP_PREDICT` writes the training labels in the fit
call (fit_predict); `MEMBER_OP_TRANSFORM` the fitted rows' distances.

Fitted-state blocks (MEMBER_OP_EXPORT / MEMBER_OP_IMPORT), in index order:
0 `MEMBER_BLOCK_FAMILY` float32 n_clusters x features, the (leaf) centers;
1 `MEMBER_BLOCK_SCALARS` float64: kind 0 [inertia, n_steps, n_iter],
kind 1 [inertia];
2 `MEMBER_BLOCK_FAMILY + 1` int32 1 x rows, the training labels (empty after
an import that did not give them);
kind 0: 3 `MEMBER_BLOCK_FAMILY + 4` float32 1 x n_clusters, the counts
(empty after an import that did not give them). An import needs 0 and 1.
kind 2: SCALARS [bandwidth, n_iter]; centers are n_found x features. An
import needs 0 and 1.
kind 4: SCALARS [n_iter]; 3 `MEMBER_BLOCK_FAMILY + 5` int32 1 x k, the
exemplar indices; 4 `MEMBER_BLOCK_FAMILY + 6` float32 rows x rows, the
affinity matrix; 5 `MEMBER_BLOCK_FAMILY + 7` float32 1 x m, the final message
diagonal (empty after an import). Centers may be empty (no exemplar). An
import needs 0 and 1.
kind 1: 3 `MEMBER_BLOCK_FAMILY + 2` float32 nodes x features, the tree
centers; 4 `MEMBER_BLOCK_FAMILY + 3` int32 nodes x 3, the tree (left, right,
label). An import needs blocks 0, 1, 3 and 4.
kind 5 (18 blocks): index j = 0 .. 15 is `MEMBER_BLOCK_FAMILY + 8 + j`, the
entry's float block f[j] as float32 1 x len (bgmm_entry's order: weights,
means, full covariances, full precision factors, wc0, wc1, mean_precision,
degrees_of_freedom, log constants, mean_prior, full covariance_prior, nk,
log_det_chol, then covariances, precision factors and covariance prior in
the covariance type's shape); 16 `MEMBER_BLOCK_SCALARS` float64 [lower_bound,
n_iter, converged, weight_concentration_prior, mean_precision_prior,
degrees_of_freedom_prior]; 17 `MEMBER_BLOCK_FAMILY + 1` int32 1 x rows, the
training labels. An import needs f[1] (means, as n_components x features),
f[3] (full precision factors), f[8] (log constants) and the scalars; the
other blocks are empty after an import that did not give them.
Not published: MEMBER_OP_PARTIAL_FIT of kind 0 (the class's own
`partial_fit` door keeps its stream state on the model).

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
    BisectPredictRequest, MiniBatchFitRequest, VARIANTS_CAP_RESIDENT, VARIANTS_CAP_DEVSCAN,
)
from core.admitted_weights import admitted_fill
from core.bgmm_native_api import BgmmDriver, BgmmFitRequest, BgmmResultSlot
from core.cluster_centroid_fit_api import (
    CentroidFitDriver, CentroidFitResultSlot, MeanShiftFitRequest, AffinityFitRequest,
)
from core.member_abi import (
    MemberCall, MemberMatrix, MEMBER_F32, MEMBER_F64, MEMBER_I32, MEMBER_HOME_DEVICE,
    MEMBER_OP_CAPABILITIES, MEMBER_OP_NEW, MEMBER_OP_FREE, MEMBER_OP_FIT, MEMBER_OP_PREDICT,
    MEMBER_OP_TRANSFORM, MEMBER_OP_SCORE, MEMBER_OP_OUTPUTS, MEMBER_OP_EXPORT, MEMBER_OP_IMPORT,
    MEMBER_OP_PREDICT_PROBA, MEMBER_OP_SCORE_SAMPLES,
    MEMBER_CAN_FIT, MEMBER_CAN_PREDICT, MEMBER_CAN_TRANSFORM, MEMBER_CAN_SCORE, MEMBER_CAN_EXPORT,
    MEMBER_CAN_IMPORT, MEMBER_CAN_SAMPLE_WEIGHT, MEMBER_CAN_PREDICT_PROBA, MEMBER_CAN_SCORE_SAMPLES,
    MEMBER_BLOCK_SCALARS, MEMBER_BLOCK_FAMILY, MEMBER_BLOCK_BY_INDEX, MEMBER_IMPORT_DONE,
    member_publish, member_call_from_spans, member_unsupported,
)
from core.member_families import MEMBER_FAMILY_XCLUSTER
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.shared_slot import SharedSlot, shared_slot_lock, shared_slot_unlock

comptime _CHUNK = 256
comptime _INDEX_SPAN = 16777216
comptime _KIND_MINIBATCH = 0
comptime _KIND_BISECT = 1
comptime _KIND_MEANSHIFT = 2
comptime _KIND_AFFINITY = 4
comptime _KIND_BGMM = 5
# BayesianGaussianMixture: the entry's sixteen float blocks f[0 .. 15], named
# _BLOCK_BGMM + j; f[1] means, f[3] full precision factors, f[8] constants.
comptime _BGMM_BLOCKS = 16
comptime _BLOCK_BGMM = MEMBER_BLOCK_FAMILY + 8
comptime _BGMM_SCALARS = 6
comptime _BLOCK_CENTERS = MEMBER_BLOCK_FAMILY
comptime _BLOCK_LABELS = MEMBER_BLOCK_FAMILY + 1
comptime _BLOCK_TREE_CENTERS = MEMBER_BLOCK_FAMILY + 2
comptime _BLOCK_TREE_NODES = MEMBER_BLOCK_FAMILY + 3
comptime _BLOCK_COUNTS = MEMBER_BLOCK_FAMILY + 4
comptime _BLOCK_EXEMPLARS = MEMBER_BLOCK_FAMILY + 5
comptime _BLOCK_AFFINITY = MEMBER_BLOCK_FAMILY + 6
comptime _BLOCK_MESSAGES = MEMBER_BLOCK_FAMILY + 7
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
    var counts: List[Float32]
    var exemplars: List[Int32]
    var affinity: List[Float32]
    var messages: List[Float32]
    var scalars: List[Float64]
    # BayesianGaussianMixture: the entry's float blocks (f[0 .. 15]) and the
    # score entry's model words (means, full precision factors, constants
    # end to end), laid out ONCE per fitted state (the door lays them out on
    # every query).
    var mixture: List[List[Float32]]
    var score_words: List[Float32]

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
        self.counts = List[Float32]()
        self.exemplars = List[Int32]()
        self.affinity = List[Float32]()
        self.messages = List[Float32]()
        self.scalars = List[Float64]()
        self.mixture = List[List[Float32]]()
        self.score_words = List[Float32]()

    def clear(mut self):
        self.fitted = False
        self.features = 0
        self.centers = List[Float32]()
        self.labels = List[Int32]()
        self.tree_centers = List[Float32]()
        self.tree_nodes = List[Int32]()
        self.counts = List[Float32]()
        self.exemplars = List[Int32]()
        self.affinity = List[Float32]()
        self.messages = List[Float32]()
        self.scalars = List[Float64]()
        self.mixture = List[List[Float32]]()
        self.score_words = List[Float32]()


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
    if kind == _KIND_MINIBATCH:
        return "MiniBatchKMeans"
    if kind == _KIND_BISECT:
        return "BisectingKMeans"
    if kind == _KIND_MEANSHIFT:
        return "MeanShift"
    if kind == _KIND_AFFINITY:
        return "AffinityPropagation"
    if kind == _KIND_BGMM:
        return "BayesianGaussianMixture"
    return "an x_cluster estimator"


def _param_count(kind: Int) raises -> Int:
    """The NEW word count of a published kind; every other kind is refused."""
    if kind == _KIND_MINIBATCH:
        return 11
    if kind == _KIND_BISECT:
        return 8
    if kind == _KIND_MEANSHIFT:
        return 6
    if kind == _KIND_AFFINITY:
        return 7
    if kind == _KIND_BGMM:
        return 13
    raise Error("member ABI: x_cluster member kind " + String(kind) + " is not published")


def _fixed_k(kind: Int) -> Bool:
    """The k-means kinds fix n_clusters (word 1); the others find theirs."""
    return kind == _KIND_MINIBATCH or kind == _KIND_BISECT


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


# ---------------------------------------------------------------- shared fit checks
def _fit_weights(call: _Call, kind: Int, rows: Int) raises -> VariantsSpan:
    """The caller's sample weights as a borrowed span (no copy), or none."""
    if call[].weights.rows == 0:
        return VariantsSpan(0, 0)
    if not call[].weights.dense_host(MEMBER_F32):
        raise Error("TypeError: a " + _name(kind) + " member reads sample_weight as a dense host float32 view")
    if call[].weights.rows * call[].weights.cols != rows:
        raise Error("ValueError: sample_weight and X have different numbers of rows")
    return VariantsSpan(call[].weights.data, rows)


def _fit_output(call: _Call, kind: Int, rows: Int, k: Int) raises -> Int:
    var output = call[].fit_output
    if output == MEMBER_OP_PREDICT:
        _destination(call, rows, 1, MEMBER_I32, "the label buffer")
    elif output == MEMBER_OP_TRANSFORM:
        _destination(call, rows, k, MEMBER_F32, "the distance buffer")
    elif output != 0:
        member_unsupported(_name(kind), output)
    return output


def _fit_done(slot: _Slot, call: _Call, output: Int, rows: Int) raises:
    if output == MEMBER_OP_PREDICT:
        _copy_i32(call[].dst.data, slot[].labels, rows)
    elif output == MEMBER_OP_TRANSFORM:
        _centers_query(slot, call, _RES_TRANSFORM)


# ---------------------------------------------------------------- MiniBatchKMeans
def _minibatch_fit(slot: _Slot, call: _Call) raises:
    var kind = slot[].kind
    _whole_x(call, kind)
    var rows = call[].x.rows
    var cols = call[].x.cols
    var w = slot[].words.unsafe_ptr()
    var k = Int(w[1])
    if k > rows:
        raise Error("mojolearn: n_clusters=" + String(k) + " exceeds n_samples=" + String(rows))
    # No explicit starting centers (admit refused them), so the entry's
    # auxiliary words are the weights alone: the caller's span, borrowed.
    var weights = _fit_weights(call, kind, rows)
    var output = _fit_output(call, kind, rows, k)
    _release_resident(slot, call)
    slot[].clear()
    var request = MiniBatchFitRequest(VariantsSpan(call[].x.data, rows * cols), weights, rows, cols, k,
        Int(w[4]), Int(w[7]), Int(w[8]), Int(w[9]), Int(w[3]), False, Int(bitcast[DType.int64](w[6])),
        w[2] != 0.0, weights.count > 0, (call[].route.flags & VARIANTS_CAP_DEVSCAN) != 0, w[5], w[10])
    var result = VariantsResultSlot()
    try:
        _driver(call).minibatch_fit(request, result)
    except error:
        if String(error).find("contains NaN or infinity") >= 0:
            raise Error("mojolearn: X contains NaN or infinity")
        raise error
    swap(slot[].centers, result.storage[0].centers)
    swap(slot[].counts, result.storage[0].counts)
    swap(slot[].labels, result.storage[0].labels)
    var scalars: List[Float64] = [result.storage[0].inertia, Float64(result.storage[0].steps),
        Float64(result.storage[0].iterations)]
    slot[].scalars = scalars^
    slot[].features = cols
    slot[].fitted = True
    _fit_done(slot, call, output, rows)


# ---------------------------------------------------------------- MeanShift
def _meanshift_fit(slot: _Slot, call: _Call) raises:
    var kind = slot[].kind
    _whole_x(call, kind)
    if call[].weights.rows > 0:
        raise Error("TypeError: MeanShift takes no sample_weight")
    var output = call[].fit_output
    if output == MEMBER_OP_PREDICT:
        _destination(call, call[].x.rows, 1, MEMBER_I32, "the label buffer")
    elif output != 0:
        member_unsupported(_name(kind), output)
    if not call[].route.has_second:
        raise Error("member ABI: a MeanShift member's route holds no centroid-fit entry")
    var rows = call[].x.rows
    var cols = call[].x.cols
    var w = slot[].words.unsafe_ptr()
    _release_resident(slot, call)
    slot[].clear()
    var request = MeanShiftFitRequest(VariantsSpan(call[].x.data, rows * cols), VariantsSpan(0, 0), rows, cols, 0,
        w[2] != 0.0, Int(w[3]), w[4] != 0.0, Int(w[5]), w[1])
    var result = CentroidFitResultSlot()
    CentroidFitDriver(call[].route.second.call).meanshift(request, result)
    swap(slot[].centers, result.storage[0].centers)
    swap(slot[].labels, result.storage[0].labels)
    var scalars: List[Float64] = [result.storage[0].bandwidth, Float64(result.storage[0].iterations)]
    slot[].scalars = scalars^
    slot[].features = cols
    slot[].fitted = True
    _fit_done(slot, call, output, rows)


# ---------------------------------------------------------------- AffinityPropagation
def _affinity_fit(slot: _Slot, call: _Call) raises:
    var kind = slot[].kind
    _whole_x(call, kind)
    if call[].weights.rows > 0:
        raise Error("TypeError: AffinityPropagation takes no sample_weight")
    var output = call[].fit_output
    if output == MEMBER_OP_PREDICT:
        _destination(call, call[].x.rows, 1, MEMBER_I32, "the label buffer")
    elif output != 0:
        member_unsupported(_name(kind), output)
    if not call[].route.has_second:
        raise Error("member ABI: an AffinityPropagation member's route holds no centroid-fit entry")
    var rows = call[].x.rows
    var cols = call[].x.cols
    var w = slot[].words.unsafe_ptr()
    _release_resident(slot, call)
    slot[].clear()
    var request = AffinityFitRequest(VariantsSpan(call[].x.data, rows * cols), VariantsSpan(0, 0), rows, cols,
        False, Int(w[5]), Int(w[2]), Int(w[3]), Int(bitcast[DType.int64](w[4])), w[1], w[6])
    var result = CentroidFitResultSlot()
    CentroidFitDriver(call[].route.second.call).affinity(request, result)
    swap(slot[].centers, result.storage[0].centers)
    swap(slot[].labels, result.storage[0].labels)
    swap(slot[].exemplars, result.storage[0].center_indices)
    swap(slot[].affinity, result.storage[0].affinity)
    swap(slot[].messages, result.storage[0].message_diagonal)
    var scalars: List[Float64] = [Float64(result.storage[0].iterations)]
    slot[].scalars = scalars^
    slot[].features = cols
    slot[].fitted = True
    _fit_done(slot, call, output, rows)


def _affinity_predict(slot: _Slot, call: _Call) raises:
    """Nearest exemplar; every row -1 when the fit found none."""
    if slot[].fitted and len(slot[].centers) == 0:
        var rows = _query_rows(slot, call)
        _destination(call, rows, 1, MEMBER_I32, "the label buffer")
        admitted_fill[DType.int32](MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=call[].dst.data),
            rows, Int32(-1))
        return
    _centers_query(slot, call, _RES_PREDICT)


# ---------------------------------------------------------------- BayesianGaussianMixture
def _bgmm_driver(call: _Call) raises -> BgmmDriver:
    if not call[].route.has_third:
        raise Error("member ABI: a BayesianGaussianMixture member's route holds no mixture entry")
    return BgmmDriver(call[].route.third.call)


def _bgmm_ready(slot: _Slot) raises:
    """Check the scoring blocks and lay out the score entry's model words
    (means, full precision factors, constants) once for this state."""
    var k = Int(slot[].words[1])
    var d = slot[].features
    var whole = len(slot[].mixture) == _BGMM_BLOCKS and d > 0 and len(slot[].scalars) == _BGMM_SCALARS
    if whole:
        whole = (len(slot[].mixture[1]) == k * d and len(slot[].mixture[3]) == k * d * d
            and len(slot[].mixture[8]) == k)
    if not whole:
        raise Error("member ABI: a fitted BayesianGaussianMixture state needs its means, precision factors,"
            + " constants and scalars blocks")
    var words = List[Float32](length=k * d + k * d * d + k, fill=0.0)
    _copy_f32(Int(words.unsafe_ptr()), slot[].mixture[1], k * d)
    _copy_f32(Int(words.unsafe_ptr()) + 4 * k * d, slot[].mixture[3], k * d * d)
    _copy_f32(Int(words.unsafe_ptr()) + 4 * (k * d + k * d * d), slot[].mixture[8], k)
    slot[].score_words = words^
    slot[].fitted = True


def _bgmm_fit(slot: _Slot, call: _Call) raises:
    """ENTRY_BGMM on the borrowed X: the words `bgmm_fit_stage` reads, laid
    out as `bgmm_entry_ints` / `bgmm_entry_floats` (mixture/mixture_door_run.mojo)
    lay them out, with no mean or covariance prior, no warm state and no
    plain-mode init words (the admit refused those instances)."""
    var kind = slot[].kind
    _whole_x(call, kind)
    if call[].weights.rows > 0:
        raise Error("TypeError: BayesianGaussianMixture takes no sample_weight")
    var rows = call[].x.rows
    var cols = call[].x.cols
    var output = call[].fit_output
    if output == MEMBER_OP_PREDICT:
        _destination(call, rows, 1, MEMBER_I32, "the label buffer")
    elif output != 0:
        member_unsupported(_name(kind), output)
    var driver = _bgmm_driver(call)
    var w = slot[].words.unsafe_ptr()
    var k = Int(w[1])
    if k < 1 or rows < k:
        raise Error("Expected n_samples >= n_components but got n_components = " + String(k)
            + ", n_samples = " + String(rows))
    slot[].clear()
    # ORDER MATCHES x_cluster/entries.mojo::bgmm_entry (ip, then fp).
    var ints: List[Int] = [rows, cols, k, 1 if w[4] != 0.0 else 0, Int(w[10]), Int(w[11]), Int(w[3]),
        Int(bitcast[DType.int64](w[12])), 0, 0, Int(w[2]), 0, 0, 0, 0, 0]
    var floats: List[Float64] = [w[5], w[6], w[7], w[8], w[9], 0.0]
    var request = BgmmFitRequest(VariantsSpan(call[].x.data, rows * cols), VariantsSpan(0, 0), ints^, floats^)
    var result = BgmmResultSlot()
    driver.fit(request, result)
    if len(result.storage[0].f) < _BGMM_BLOCKS or len(result.storage[0].i) < 1:
        raise Error("member ABI: the BGMM entry returned fewer blocks than bgmm_entry documents")
    # The entry's result lists become the state's blocks (no copy).
    swap(slot[].mixture, result.storage[0].f)
    swap(slot[].labels, result.storage[0].i[0])
    swap(slot[].scalars, result.storage[0].s)
    slot[].features = cols
    _bgmm_ready(slot)
    if output == MEMBER_OP_PREDICT:
        _copy_i32(call[].dst.data, slot[].labels, rows)


def _bgmm_query(slot: _Slot, call: _Call, operation: Int) raises:
    """ENTRY_BGMM_SCORE on the borrowed X and the state's model words, the
    flags of the class's own query: predict 0 (labels), predict_proba 1
    (f[2]), score_samples 0 (f[1]), score 2 (s[0] over the row count)."""
    var rows = _query_rows(slot, call)
    var k = Int(slot[].words[1])
    var flags = 0
    if operation == MEMBER_OP_PREDICT:
        _destination(call, rows, 1, MEMBER_I32, "the label buffer")
    elif operation == MEMBER_OP_PREDICT_PROBA:
        _destination(call, rows, k, MEMBER_F32, "the responsibility buffer")
        flags = 1
    elif operation == MEMBER_OP_SCORE_SAMPLES:
        _destination(call, rows, 1, MEMBER_F32, "the log-likelihood buffer")
    else:
        if call[].weights.rows > 0:
            raise Error("TypeError: the score of a BayesianGaussianMixture member counts every row once"
                + " (no sample_weight)")
        flags = 2
    var result = BgmmResultSlot()
    _bgmm_driver(call).score(VariantsSpan(call[].x.data, rows * slot[].features),
        VariantsSpan(Int(slot[].score_words.unsafe_ptr()), len(slot[].score_words)), rows, slot[].features, k,
        flags, result)
    if operation == MEMBER_OP_PREDICT:
        _copy_i32(call[].dst.data, result.storage[0].i[0], rows)
    elif operation == MEMBER_OP_PREDICT_PROBA:
        _copy_f32(call[].dst.data, result.storage[0].f[2], rows * k)
    elif operation == MEMBER_OP_SCORE_SAMPLES:
        _copy_f32(call[].dst.data, result.storage[0].f[1], rows)
    else:
        call[].value = result.storage[0].s[0] / Float64(rows)


def _bgmm_block_at(index: Int) raises -> Int:
    if index < 0 or index >= _BGMM_BLOCKS + 2:
        raise Error("member ABI: a BayesianGaussianMixture member has " + String(_BGMM_BLOCKS + 2)
            + " fitted blocks")
    if index < _BGMM_BLOCKS:
        return _BLOCK_BGMM + index
    return MEMBER_BLOCK_SCALARS if index == _BGMM_BLOCKS else _BLOCK_LABELS


def _bgmm_export(slot: _Slot, call: _Call, block: Int) raises:
    if block >= _BLOCK_BGMM and block < _BLOCK_BGMM + _BGMM_BLOCKS:
        var j = block - _BLOCK_BGMM
        if j == 1:
            var d = slot[].features
            call[].dst = _f32_view(slot[].mixture[1], len(slot[].mixture[1]) // d, d)
        else:
            call[].dst = _f32_view(slot[].mixture[j], 1, len(slot[].mixture[j]))
    elif block == MEMBER_BLOCK_SCALARS:
        call[].dst = MemberMatrix(Int(slot[].scalars.unsafe_ptr()), 1, len(slot[].scalars), MEMBER_F64)
    elif block == _BLOCK_LABELS:
        call[].dst = _i32_view(slot[].labels, 1, len(slot[].labels))
    else:
        raise Error("member ABI: a BayesianGaussianMixture member has no fitted block " + String(block))
    call[].block = block
    call[].out_cols = _BGMM_BLOCKS + 2


def _bgmm_import(slot: _Slot, call: _Call) raises:
    var block = call[].block
    if (call[].flags & MEMBER_BLOCK_BY_INDEX) != 0:
        block = _bgmm_block_at(block)
    if len(slot[].mixture) != _BGMM_BLOCKS:
        slot[].fitted = False
        slot[].mixture = List[List[Float32]]()
        for _ in range(_BGMM_BLOCKS):  # driver-loop: the sixteen fitted blocks
            slot[].mixture.append(List[Float32]())
    if block >= _BLOCK_BGMM and block < _BLOCK_BGMM + _BGMM_BLOCKS:
        slot[].fitted = False
        var j = block - _BLOCK_BGMM
        slot[].mixture[j] = _take_f32(call, "mixture")
        if j == 1:
            slot[].features = call[].x.cols
            slot[].labels = List[Int32]()
    elif block == MEMBER_BLOCK_SCALARS:
        if not call[].x.dense_host(MEMBER_F64) or call[].x.rows * call[].x.cols != _BGMM_SCALARS:
            raise Error("member ABI: the BayesianGaussianMixture scalars block is " + String(_BGMM_SCALARS)
                + " host float64 words")
        var source = _Words(unsafe_from_address=call[].x.data)
        slot[].scalars = List[Float64](length=_BGMM_SCALARS, fill=0.0)
        for i in range(_BGMM_SCALARS):  # driver-loop: six scalar words
            slot[].scalars[i] = source[i]
    elif block == _BLOCK_LABELS:
        slot[].labels = _take_i32(call, "labels")
    elif block != 0 or (call[].flags & MEMBER_IMPORT_DONE) == 0:
        raise Error("member ABI: a BayesianGaussianMixture member has no fitted block " + String(block))
    if (call[].flags & MEMBER_IMPORT_DONE) != 0:
        _bgmm_ready(slot)


# ---------------------------------------------------------------- BisectingKMeans
def _bisect_fit(slot: _Slot, call: _Call) raises:
    var kind = slot[].kind
    _whole_x(call, kind)
    var rows = call[].x.rows
    var cols = call[].x.cols
    var k = Int(slot[].words[1])
    if k > rows:
        raise Error("mojolearn: n_clusters=" + String(k) + " exceeds n_samples=" + String(rows))
    var weights = _fit_weights(call, kind, rows)
    var output = _fit_output(call, kind, rows, k)
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
    _fit_done(slot, call, output, rows)


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
    if kind == _KIND_AFFINITY:
        return 6
    if kind == _KIND_MEANSHIFT:
        return 3
    if kind == _KIND_MINIBATCH:
        return 4
    return 5 if kind == _KIND_BISECT else 0


def _scalar_count(kind: Int) -> Int:
    if kind == _KIND_AFFINITY:
        return 1
    if kind == _KIND_MEANSHIFT:
        return 2
    return 3 if kind == _KIND_MINIBATCH else 1


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
    if kind == _KIND_MINIBATCH:
        return _BLOCK_COUNTS
    if kind == _KIND_AFFINITY:
        return _BLOCK_EXEMPLARS if index == 3 else (_BLOCK_AFFINITY if index == 4 else _BLOCK_MESSAGES)
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
    if kind == _KIND_BGMM:
        if (call[].flags & MEMBER_BLOCK_BY_INDEX) != 0:
            block = _bgmm_block_at(block)
        _bgmm_export(slot, call, block)
        return
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
    elif block == _BLOCK_COUNTS and kind == _KIND_MINIBATCH:
        call[].dst = _f32_view(slot[].counts, 1, len(slot[].counts))
    elif block == _BLOCK_EXEMPLARS and kind == _KIND_AFFINITY:
        call[].dst = _i32_view(slot[].exemplars, 1, len(slot[].exemplars))
    elif block == _BLOCK_AFFINITY and kind == _KIND_AFFINITY:
        var n = len(slot[].labels)
        call[].dst = _f32_view(slot[].affinity, n, len(slot[].affinity) // n if n > 0 else 0)
    elif block == _BLOCK_MESSAGES and kind == _KIND_AFFINITY:
        call[].dst = _f32_view(slot[].messages, 1, len(slot[].messages))
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
    if kind == _KIND_BGMM:
        _bgmm_import(slot, call)
        return
    var block = call[].block
    if block == _BLOCK_CENTERS:
        _release_resident(slot, call)
        slot[].fitted = False
        slot[].centers = _take_f32(call, "centers")
        slot[].features = call[].x.cols
        slot[].labels = List[Int32]()
    elif block == MEMBER_BLOCK_SCALARS:
        var count = _scalar_count(kind)
        if not call[].x.dense_host(MEMBER_F64) or call[].x.rows * call[].x.cols != count:
            raise Error("member ABI: the " + _name(kind) + " scalars block is " + String(count)
                + " host float64 words")
        var source = _Words(unsafe_from_address=call[].x.data)
        slot[].scalars = List[Float64](length=count, fill=0.0)
        for i in range(count):  # driver-loop: at most three scalar words
            slot[].scalars[i] = source[i]
    elif block == _BLOCK_LABELS:
        slot[].labels = _take_i32(call, "labels")
    elif block == _BLOCK_TREE_CENTERS and kind == _KIND_BISECT:
        slot[].tree_centers = _take_f32(call, "tree centers")
    elif block == _BLOCK_TREE_NODES and kind == _KIND_BISECT:
        if call[].x.cols != 3:
            raise Error("member ABI: the BisectingKMeans tree block holds three int32 words per node")
        slot[].tree_nodes = _take_i32(call, "tree")
    elif block == _BLOCK_COUNTS and kind == _KIND_MINIBATCH:
        slot[].counts = _take_f32(call, "counts")
    elif block == _BLOCK_EXEMPLARS and kind == _KIND_AFFINITY:
        slot[].exemplars = _take_i32(call, "exemplar indices")
    elif block != 0 or (call[].flags & MEMBER_IMPORT_DONE) == 0:
        raise Error("member ABI: a " + _name(kind) + " member has no fitted block " + String(block))
    if (call[].flags & MEMBER_IMPORT_DONE) == 0:
        return
    var d = slot[].features
    # MeanShift and AffinityPropagation find their own center count (none
    # at all for an AffinityPropagation that found no exemplar); the
    # k-means kinds fix it.
    var k = Int(slot[].words[1]) if _fixed_k(kind) else (len(slot[].centers) // d if d > 0 else 0)
    var least = 0 if kind == _KIND_AFFINITY else 1
    if d == 0 or k < least or len(slot[].centers) != k * d or len(slot[].scalars) != _scalar_count(kind):
        raise Error("member ABI: a fitted " + _name(kind) + " state needs its centers and scalars blocks")
    if kind == _KIND_BISECT and (len(slot[].tree_nodes) == 0
            or len(slot[].tree_centers) != (len(slot[].tree_nodes) // 3) * d):
        raise Error("member ABI: a fitted BisectingKMeans state needs its tree centers and tree blocks")
    slot[].fitted = True


# ---------------------------------------------------------------- the entry
def _capabilities(kind: Int) -> Int:
    # A state's kind, or (kind -1, state 0) the family's union.
    if kind == _KIND_MEANSHIFT or kind == _KIND_AFFINITY:
        return MEMBER_CAN_FIT | MEMBER_CAN_PREDICT | MEMBER_CAN_EXPORT | MEMBER_CAN_IMPORT
    var bgmm = (MEMBER_CAN_FIT | MEMBER_CAN_PREDICT | MEMBER_CAN_PREDICT_PROBA | MEMBER_CAN_SCORE_SAMPLES
        | MEMBER_CAN_SCORE | MEMBER_CAN_EXPORT | MEMBER_CAN_IMPORT)
    if kind == _KIND_BGMM:
        return bgmm
    var centers = (MEMBER_CAN_FIT | MEMBER_CAN_PREDICT | MEMBER_CAN_TRANSFORM | MEMBER_CAN_SCORE
        | MEMBER_CAN_EXPORT | MEMBER_CAN_IMPORT | MEMBER_CAN_SAMPLE_WEIGHT)
    if kind < 0:
        return centers | bgmm
    return centers


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
        if _fixed_k(kind) and words[1] < 1.0:
            raise Error("ValueError: n_clusters must be at least 1, got " + String(Int(words[1])))
        if kind == _KIND_BGMM and words[1] < 1.0:
            raise Error("ValueError: n_components must be at least 1, got " + String(Int(words[1])))
        if kind == _KIND_BGMM and (words[10] < 1.0 or words[11] < 1.0):
            raise Error("n_init and max_iter must be >= 1")
        if _fixed_k(kind) and words[3] < 1.0:
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
        # Only the k-means kinds transform, with n_clusters columns; only
        # BayesianGaussianMixture answers predict_proba, n_components columns.
        var clusters = 0
        var asked = MEMBER_OP_TRANSFORM
        var which = -1
        if call[].state != 0:
            var held = _Slot(unsafe_from_address=_lookup(call[].state, False))
            which = held[].kind
            if _fixed_k(which) or which == _KIND_BGMM:
                clusters = Int(held[].words[1])
        elif call[].params and call[].param_count >= 2:
            var words = _Words(unsafe_from_address=call[].params)
            which = Int(words[0])
            if _fixed_k(which) or which == _KIND_BGMM:
                clusters = Int(words[1])
        if which == _KIND_BGMM:
            asked = MEMBER_OP_PREDICT_PROBA
        call[].out_cols = clusters if call[].flags == asked else 1
        call[].classes = 0
        return 0
    var slot = _Slot(unsafe_from_address=_lookup(call[].state, False))
    var kind = slot[].kind
    if operation == MEMBER_OP_FIT and kind == _KIND_MINIBATCH:
        _minibatch_fit(slot, call)
    elif operation == MEMBER_OP_PREDICT and kind == _KIND_MINIBATCH:
        _centers_query(slot, call, _RES_PREDICT)
    elif operation == MEMBER_OP_FIT and kind == _KIND_MEANSHIFT:
        _meanshift_fit(slot, call)
    elif operation == MEMBER_OP_PREDICT and kind == _KIND_MEANSHIFT:
        _centers_query(slot, call, _RES_PREDICT)
    elif operation == MEMBER_OP_FIT and kind == _KIND_AFFINITY:
        _affinity_fit(slot, call)
    elif operation == MEMBER_OP_PREDICT and kind == _KIND_AFFINITY:
        _affinity_predict(slot, call)
    elif operation == MEMBER_OP_FIT and kind == _KIND_BISECT:
        _bisect_fit(slot, call)
    elif operation == MEMBER_OP_PREDICT and kind == _KIND_BISECT:
        _bisect_predict(slot, call)
    elif operation == MEMBER_OP_FIT and kind == _KIND_BGMM:
        _bgmm_fit(slot, call)
    elif kind == _KIND_BGMM and (operation == MEMBER_OP_PREDICT or operation == MEMBER_OP_PREDICT_PROBA
            or operation == MEMBER_OP_SCORE_SAMPLES or operation == MEMBER_OP_SCORE):
        _bgmm_query(slot, call, operation)
    elif operation == MEMBER_OP_TRANSFORM and _fixed_k(kind):
        _centers_query(slot, call, _RES_TRANSFORM)
    elif operation == MEMBER_OP_SCORE and _fixed_k(kind):
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
