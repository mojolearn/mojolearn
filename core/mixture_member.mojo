# SPDX-License-Identifier: Apache-2.0
"""GaussianMixture as a MEMBER (core/member_abi.mojo, MEMBER-ABI-V2.md).

One entry, `mixture_member_v2`, published by each mixture image that can fit
(`mixture_member_publish`): the device image `_mojolearn_mixture`
(`GMM_NATIVE_DEVICE_API`) and the host image `_mojolearn_mixture_host`
(`GMM_NATIVE_HOST_API`, vendor 0). The inference-only CPU image
`_mojolearn_mixture_infer_host` has no fit and publishes nothing. It runs the
family's existing typed RUN (`GmmDriver`, core/mixture_native_api.mojo:
`gmm_fit`, the scoring entries) over the entry of the `DeviceRoute` it is
handed: `route.entry` is the image's `mixture_native_api_v1` entry,
`route.flags` that entry's capability word (`MIXTURE_CAP_*`, read once by the
caller's UNPACK). No other route slot is read. No kernel is touched.

Kind 0 GaussianMixture, the default fit only (covariance_type='full',
n_init=1, no warm start, no explicit weights/means/precisions starts, and
init_params 'kmeans' or 'random'): the option-parity fit of the other
settings runs through another entry and the class keeps its own doors for it.
A clusterer, not a classifier: `y` is not read, `classes` is 0.

MEMBER_OP_NEW parameters (Float64 words, 6): [0] n_components, [1] max_iter,
[2] tol, [3] reg_covar, [4] random_state as the BITS of an int64
(`bitcast[DType.float64](Int64(seed))`, the estimator's own integer reading),
[5] init_params (0 'kmeans', 1 'random'). The entry decodes the two option
names by name; the state holds them as strings built once at NEW.

Views: X dense row-major host float32, BORROWED for the call (never copied:
the entry reads the caller's words, as the estimator's own door reads the
admitted array), admitted finite by the caller's UNPACK. No sample weights,
no row or column index, no device-home view: refused in words.

Outputs: predict writes component ids as int32 (rows x 1); predict_proba
float32 rows x n_components; score_samples float32 rows x 1 (the per-row
log-likelihood). `call.fit_output = MEMBER_OP_PREDICT` writes predict of the
fitted model for the training rows (fit_predict: in the fit's own crossing
where the image has `MIXTURE_CAP_FIT_PREDICT`, else a second entry call). No
score (the caller averages score_samples itself).

Fitted-state blocks (MEMBER_OP_EXPORT / MEMBER_OP_IMPORT), in index order:
0 `MEMBER_BLOCK_FAMILY` float32 1 x k, the weights;
1 `MEMBER_BLOCK_FAMILY + 1` float32 k x d, the means;
2 `MEMBER_BLOCK_FAMILY + 2` float32 k x (d*d), the covariances;
3 `MEMBER_BLOCK_FAMILY + 3` float32 k x (d*d), the precision Cholesky factors;
4 `MEMBER_BLOCK_FAMILY + 4` float32 1 x k, the log-determinants of the factors;
5 `MEMBER_BLOCK_SCALARS` float64 [n_iter, converged (0/1), lower_bound].
An import needs all six.

The fitted state is OWNED here, in a process-wide store of fixed chunks: a
slot never moves once minted, so an operation runs on its slot without the
store's lock (the lock covers mint, free and the handle lookup). Serial by
nature here: the store's bookkeeping.
"""
from std.memory import bitcast, unsafe_memcpy
from std.sys.info import size_of
from core.member_abi import (
    MemberCall, MemberMatrix, MEMBER_F32, MEMBER_F64, MEMBER_I32, MEMBER_HOME_DEVICE,
    MEMBER_OP_CAPABILITIES, MEMBER_OP_NEW, MEMBER_OP_FREE, MEMBER_OP_FIT, MEMBER_OP_PREDICT,
    MEMBER_OP_PREDICT_PROBA, MEMBER_OP_SCORE_SAMPLES, MEMBER_OP_OUTPUTS, MEMBER_OP_EXPORT, MEMBER_OP_IMPORT,
    MEMBER_CAN_FIT, MEMBER_CAN_PREDICT, MEMBER_CAN_PREDICT_PROBA, MEMBER_CAN_SCORE_SAMPLES, MEMBER_CAN_EXPORT,
    MEMBER_CAN_IMPORT, MEMBER_BLOCK_SCALARS, MEMBER_BLOCK_FAMILY, MEMBER_BLOCK_BY_INDEX, MEMBER_IMPORT_DONE,
    member_publish, member_call_from_spans, member_unsupported,
)
from core.member_families import MEMBER_FAMILY_MIXTURE
from core.mixture_native_api import (
    GmmDriver, GmmFitRequest, GmmScoreRequest, MIXTURE_NATIVE_PREDICT, MIXTURE_NATIVE_PREDICT_PROBA,
    MIXTURE_NATIVE_SCORE_SAMPLES, MIXTURE_CAP_FIT_PREDICT,
)
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.shared_slot import SharedSlot, shared_slot_lock, shared_slot_unlock

comptime _CHUNK = 256
comptime _INDEX_SPAN = 16777216
comptime _EST = "GaussianMixture"
comptime _BLOCKS = 6


struct MixtureMemberSlot(Copyable, Movable, Defaultable):
    var generation: Int
    var live: Bool
    var components: Int
    var max_iter: Int
    var tolerance: Float64
    var reg_covar: Float64
    var seed: Int
    var covariance_name: String
    var init_name: String
    var fitted: Bool
    var features: Int
    var weights: List[Float32]
    var means: List[Float32]
    var covariances: List[Float32]
    var precisions: List[Float32]
    var log_det: List[Float32]
    var scalars: List[Float64]

    def __init__(out self):
        self.generation = 0
        self.live = False
        self.components = 0
        self.max_iter = 0
        self.tolerance = 0.0
        self.reg_covar = 0.0
        self.seed = 0
        self.covariance_name = String("full")
        self.init_name = String("kmeans")
        self.fitted = False
        self.features = 0
        self.weights = List[Float32]()
        self.means = List[Float32]()
        self.covariances = List[Float32]()
        self.precisions = List[Float32]()
        self.log_det = List[Float32]()
        self.scalars = List[Float64]()

    def clear(mut self):
        self.fitted = False
        self.features = 0
        self.weights = List[Float32]()
        self.means = List[Float32]()
        self.covariances = List[Float32]()
        self.precisions = List[Float32]()
        self.log_det = List[Float32]()
        self.scalars = List[Float64]()


struct MixtureMemberStore(Defaultable, Movable):
    var synchronization: List[Int64]
    var chunks: List[List[MixtureMemberSlot]]
    var free: List[Int]
    var count: Int

    def __init__(out self):
        # Side-effect-free even when the process slot's initializers race.
        self.synchronization = List[Int64](length=2, fill=0)
        self.chunks = List[List[MixtureMemberSlot]]()
        self.free = List[Int]()
        self.count = 0


comptime _STORE = SharedSlot[MixtureMemberStore, "MojolearnMixtureMemberStoreV2", 2]
comptime _Slot = MutPointer[MixtureMemberSlot, MutUntrackedOrigin]
comptime _Call = MutPointer[MemberCall, MutUntrackedOrigin]
comptime _Words = MutPointer[Float64, MutUntrackedOrigin]


def _slot_address(index: Int) raises -> Int:
    # Under the store's lock. A chunk's buffer is allocated once and never
    # grows, so the address stays valid after the lock is released.
    var store = _STORE.get_or_create_ptr()
    return Int(store[].chunks[index // _CHUNK].unsafe_ptr()) + (index % _CHUNK) * size_of[MixtureMemberSlot]()


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
                raise Error("RuntimeError: mojolearn: too many live GaussianMixture member states")
            if store[].count == len(store[].chunks) * _CHUNK:
                store[].chunks.append(List[MixtureMemberSlot](length=_CHUNK, fill=MixtureMemberSlot()))
            index = store[].count
            store[].count += 1
        var slot = _Slot(unsafe_from_address=_slot_address(index))
        slot[].generation += 1
        slot[].live = True
        slot[].components = Int(words[0])
        slot[].max_iter = Int(words[1])
        slot[].tolerance = words[2]
        slot[].reg_covar = words[3]
        slot[].seed = Int(bitcast[DType.int64](words[4]))
        slot[].covariance_name = String("full")
        slot[].init_name = String("random") if words[5] != 0.0 else String("kmeans")
        slot[].clear()
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
            slot[].clear()
            store[].free.append(index)
        return address
    finally:
        shared_slot_unlock(lock)


def _whole_x(call: _Call) raises:
    if call[].row_index.rows > 0 or call[].column_index.rows > 0:
        raise Error("TypeError: a GaussianMixture member reads no row or column index"
            + " (a fold is handed over as gathered host words)")
    if call[].x.home == MEMBER_HOME_DEVICE or not call[].x.dense_host(MEMBER_F32):
        raise Error("TypeError: a GaussianMixture member reads X as a dense host float32 view")


def _destination(call: _Call, rows: Int, cols: Int, dtype: Int, what: String) raises:
    if not call[].dst.dense_host(dtype):
        raise Error("TypeError: a GaussianMixture member writes " + what + " as a dense host "
            + (String("int32") if dtype == MEMBER_I32 else String("float32")) + " view")
    if call[].dst.rows != rows or call[].dst.cols != cols:
        raise Error("ValueError: " + what + " holds " + String(rows) + " x " + String(cols) + " values")


def _fit(slot: _Slot, call: _Call) raises:
    _whole_x(call)
    if call[].weights.rows > 0:
        raise Error("TypeError: GaussianMixture takes no sample_weight")
    var output = call[].fit_output
    var rows = call[].x.rows
    var d = call[].x.cols
    var k = slot[].components
    if output == MEMBER_OP_PREDICT:
        _destination(call, rows, 1, MEMBER_I32, "the label buffer")
    elif output != 0:
        member_unsupported(_EST, output)
    slot[].clear()
    slot[].weights = List[Float32](length=k, fill=0.0)
    slot[].means = List[Float32](length=k * d, fill=0.0)
    slot[].covariances = List[Float32](length=k * d * d, fill=0.0)
    slot[].precisions = List[Float32](length=k * d * d, fill=0.0)
    slot[].log_det = List[Float32](length=k, fill=0.0)
    slot[].scalars = List[Float64](length=3, fill=0.0)
    var fused = output == MEMBER_OP_PREDICT and (call[].route.flags & MIXTURE_CAP_FIT_PREDICT) != 0
    var request = GmmFitRequest(call[].x.data, Int(slot[].weights.unsafe_ptr()), Int(slot[].means.unsafe_ptr()),
        Int(slot[].covariances.unsafe_ptr()), Int(slot[].precisions.unsafe_ptr()),
        Int(slot[].log_det.unsafe_ptr()), Int(slot[].scalars.unsafe_ptr()), call[].dst.data if fused else 0,
        Int(slot[].covariance_name.as_c_string_slice().unsafe_ptr()),
        Int(slot[].init_name.as_c_string_slice().unsafe_ptr()), rows, d, k, slot[].max_iter, slot[].seed,
        slot[].tolerance, slot[].reg_covar)
    var n_iter = GmmDriver(call[].route.entry.call).fit(request)
    # The entry's scalars word 0 is the caller's to fill: n_iter_ is its result.
    slot[].scalars[0] = Float64(n_iter)
    slot[].features = d
    slot[].fitted = True
    if output == MEMBER_OP_PREDICT and not fused:
        _score(slot, call, MIXTURE_NATIVE_PREDICT)


def _score(slot: _Slot, call: _Call, operation: Int) raises:
    if not slot[].fitted:
        raise Error("RuntimeError: " + String(_EST) + ": call fit before predict")
    _whole_x(call)
    var rows = call[].x.rows
    var d = slot[].features
    if call[].x.cols != d:
        raise Error(String(_EST) + ": X has " + String(call[].x.cols) + " features, the fit had " + String(d))
    if operation == MIXTURE_NATIVE_PREDICT:
        _destination(call, rows, 1, MEMBER_I32, "the label buffer")
    elif operation == MIXTURE_NATIVE_PREDICT_PROBA:
        _destination(call, rows, slot[].components, MEMBER_F32, "the probability buffer")
    else:
        _destination(call, rows, 1, MEMBER_F32, "the log-likelihood buffer")
    # ORDER MATCHES bindings/_mojolearn_mixture.mojo::_rebuild_model.
    var request = GmmScoreRequest(Int(slot[].weights.unsafe_ptr()), Int(slot[].means.unsafe_ptr()),
        Int(slot[].covariances.unsafe_ptr()), Int(slot[].precisions.unsafe_ptr()),
        Int(slot[].log_det.unsafe_ptr()), call[].x.data, call[].dst.data, slot[].components, d,
        Int(slot[].scalars[0]), slot[].scalars[1] != 0.0, rows, slot[].scalars[2])
    GmmDriver(call[].route.entry.call).score(operation, request)


def _block_list(slot: _Slot, block: Int) raises -> Int:
    """The address of block `block`'s list (f32), its rows and columns via
    `_block_shape`."""
    if block == MEMBER_BLOCK_FAMILY:
        return Int(slot[].weights.unsafe_ptr())
    if block == MEMBER_BLOCK_FAMILY + 1:
        return Int(slot[].means.unsafe_ptr())
    if block == MEMBER_BLOCK_FAMILY + 2:
        return Int(slot[].covariances.unsafe_ptr())
    if block == MEMBER_BLOCK_FAMILY + 3:
        return Int(slot[].precisions.unsafe_ptr())
    if block == MEMBER_BLOCK_FAMILY + 4:
        return Int(slot[].log_det.unsafe_ptr())
    raise Error("member ABI: a GaussianMixture member has no fitted block " + String(block))


def _block_rows(slot: _Slot, block: Int) -> Int:
    return 1 if (block == MEMBER_BLOCK_FAMILY or block == MEMBER_BLOCK_FAMILY + 4) else slot[].components


def _block_cols(slot: _Slot, block: Int) -> Int:
    var d = slot[].features
    if block == MEMBER_BLOCK_FAMILY or block == MEMBER_BLOCK_FAMILY + 4:
        return slot[].components
    return d if block == MEMBER_BLOCK_FAMILY + 1 else d * d


def _export(slot: _Slot, call: _Call) raises:
    """A view of one fitted block. The slot owns the words; they stay valid
    until this state's next fit, import or free."""
    if not slot[].fitted:
        raise Error("NotFittedError: this GaussianMixture member is not fitted yet")
    var block = call[].block
    if (call[].flags & MEMBER_BLOCK_BY_INDEX) != 0:
        if block < 0 or block >= _BLOCKS:
            raise Error("member ABI: a GaussianMixture member has six fitted blocks")
        block = MEMBER_BLOCK_SCALARS if block == 5 else MEMBER_BLOCK_FAMILY + block
    if block == MEMBER_BLOCK_SCALARS:
        call[].dst = MemberMatrix(Int(slot[].scalars.unsafe_ptr()), 1, 3, MEMBER_F64)
    else:
        call[].dst = MemberMatrix(_block_list(slot, block), _block_rows(slot, block), _block_cols(slot, block))
    call[].block = block
    call[].out_cols = _BLOCKS


def _take(call: _Call, count: Int, what: String) raises -> List[Float32]:
    if not call[].x.dense_host(MEMBER_F32) or call[].x.rows * call[].x.cols != count:
        raise Error("member ABI: the GaussianMixture " + what + " block is " + String(count)
            + " dense host float32 words")
    var values = List[Float32](length=count, fill=0.0)
    unsafe_memcpy(dest=values.unsafe_ptr(),
        src=MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=call[].x.data), count=count)
    return values^


def _import(slot: _Slot, call: _Call) raises:
    """Copy one fitted block in; MEMBER_IMPORT_DONE completes the state. The
    means block comes first: it sets the feature count."""
    var k = slot[].components
    var block = call[].block
    if block == MEMBER_BLOCK_FAMILY + 1:
        slot[].fitted = False
        if call[].x.rows != k:
            raise Error("ValueError: the fitted means have " + String(call[].x.rows) + " rows, n_components is "
                + String(k))
        slot[].features = call[].x.cols
        slot[].means = _take(call, k * call[].x.cols, "means")
    elif block == MEMBER_BLOCK_FAMILY:
        slot[].weights = _take(call, k, "weights")
    elif block == MEMBER_BLOCK_FAMILY + 4:
        slot[].log_det = _take(call, k, "log-determinant")
    elif block == MEMBER_BLOCK_FAMILY + 2:
        slot[].covariances = _take(call, k * slot[].features * slot[].features, "covariances")
    elif block == MEMBER_BLOCK_FAMILY + 3:
        slot[].precisions = _take(call, k * slot[].features * slot[].features, "precisions")
    elif block == MEMBER_BLOCK_SCALARS:
        if not call[].x.dense_host(MEMBER_F64) or call[].x.rows * call[].x.cols != 3:
            raise Error("member ABI: the GaussianMixture scalars block is three host float64 words"
                + " (n_iter, converged, lower_bound)")
        var source = _Words(unsafe_from_address=call[].x.data)
        slot[].scalars = List[Float64](length=3, fill=0.0)
        for i in range(3):  # driver-loop: three scalar words
            slot[].scalars[i] = source[i]
    elif block != 0 or (call[].flags & MEMBER_IMPORT_DONE) == 0:
        raise Error("member ABI: a GaussianMixture member has no fitted block " + String(block))
    if (call[].flags & MEMBER_IMPORT_DONE) == 0:
        return
    var d = slot[].features
    if (d == 0 or len(slot[].weights) != k or len(slot[].means) != k * d or len(slot[].log_det) != k
            or len(slot[].covariances) != k * d * d or len(slot[].precisions) != k * d * d
            or len(slot[].scalars) != 3):
        raise Error("member ABI: a fitted GaussianMixture state needs its six blocks")
    slot[].fitted = True


def _mixture_member(operation: Int, address: Int) raises -> Int:
    var call = _Call(unsafe_from_address=address)
    if operation == MEMBER_OP_CAPABILITIES:
        # No MEMBER_CAN_TEST_ROWS: X is admitted finite by the caller (the
        # GMM entry does not test it).
        return (MEMBER_CAN_FIT | MEMBER_CAN_PREDICT | MEMBER_CAN_PREDICT_PROBA | MEMBER_CAN_SCORE_SAMPLES
            | MEMBER_CAN_EXPORT | MEMBER_CAN_IMPORT)
    if operation == MEMBER_OP_NEW:
        if call[].param_count != 6 or not call[].params:
            raise Error("member ABI: a GaussianMixture member takes six parameters (n_components, max_iter,"
                + " tol, reg_covar, random_state, init_params)")
        var words = _Words(unsafe_from_address=call[].params)
        if words[0] < 1.0:
            raise Error("ValueError: n_components must be at least 1, got " + String(Int(words[0])))
        if words[5] != 0.0 and words[5] != 1.0:
            raise Error("member ABI: a GaussianMixture member's init_params is 0 ('kmeans') or 1 ('random')")
        call[].state = _mint(words)
        return 0
    if operation == MEMBER_OP_FREE:
        _ = _lookup(call[].state, True)
        call[].state = 0
        return 0
    if operation == MEMBER_OP_OUTPUTS:
        var components = 0
        if call[].state != 0:
            components = _Slot(unsafe_from_address=_lookup(call[].state, False))[].components
        elif call[].params and call[].param_count == 6:
            components = Int(_Words(unsafe_from_address=call[].params)[0])
        call[].out_cols = components if call[].flags == MEMBER_OP_PREDICT_PROBA else 1
        call[].classes = 0
        # predict: int32 component ids; predict_proba, score_samples: float32.
        if call[].flags == MEMBER_OP_PREDICT:
            call[].out_dtype = MEMBER_I32
        elif call[].flags == MEMBER_OP_PREDICT_PROBA or call[].flags == MEMBER_OP_SCORE_SAMPLES:
            call[].out_dtype = MEMBER_F32
        return 0
    var slot = _Slot(unsafe_from_address=_lookup(call[].state, False))
    if operation == MEMBER_OP_FIT:
        _fit(slot, call)
    elif operation == MEMBER_OP_PREDICT:
        _score(slot, call, MIXTURE_NATIVE_PREDICT)
    elif operation == MEMBER_OP_PREDICT_PROBA:
        _score(slot, call, MIXTURE_NATIVE_PREDICT_PROBA)
    elif operation == MEMBER_OP_SCORE_SAMPLES:
        _score(slot, call, MIXTURE_NATIVE_SCORE_SAMPLES)
    elif operation == MEMBER_OP_EXPORT:
        _export(slot, call)
    elif operation == MEMBER_OP_IMPORT:
        _import(slot, call)
    else:
        member_unsupported(_EST, operation)
    return 0


def mixture_member_v2(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
    result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """The member entry of GaussianMixture (schema `MEMBER_ABI`)."""
    try:
        var word = _mixture_member(operation, member_call_from_spans(operation, aa, na, np, nf))
        if result:
            MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = word
        return 0
    except error:
        native_descriptor_report(error, context, sink)
        return 1


def mixture_member_publish(mode: Int, vendor: Int, publisher: Int) raises:
    """Publish this image's GaussianMixture member entry. One line in the
    image's module initialization, after its mixture kernel table exists; a
    build with no vendor (`vendor` < 0) publishes nothing."""
    if vendor < 0:
        return
    member_publish[MEMBER_FAMILY_MIXTURE](mode, vendor, publisher, mixture_member_v2)
