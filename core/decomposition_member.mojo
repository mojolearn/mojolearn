# SPDX-License-Identifier: Apache-2.0
"""PCA and TruncatedSVD as MEMBERS (core/member_abi.mojo).

One entry, `decomposition_member_v2`, published by each estimators image
(`decomposition_member_publish`, beside the linear member). It runs the
family's existing typed RUN, the PCA and TruncatedSVD native drivers
(core/pca_native_api.mojo `PCANativeDriver`, core/tsvd_native_api.mojo
`TSVDNativeDriver`), over the entries of the `DeviceRoute` it is handed, the
same entries the estimators' own doors call (bindings/pca_fit_native_door.mojo,
bindings/tsvd_native_door.mojo). No kernel is touched.

Route slots read: PCA: `entry` the PCA native entry of the estimator's image
(`flags` its PCA_CAP_* word). TruncatedSVD: `entry` the estimators image's
TruncatedSVD entry, `second` the x_decomp image's (randomized and TSQR arms;
`has_second` False when that image did not connect); `flags` bits
DECOMP_ROUTE_FUSED (entry has the fused fit), DECOMP_ROUTE_TSQR (the x_decomp
image takes the TSQR fit: its qfix bit and TSVD_CAP_TSQR), DECOMP_ROUTE_TSQR_MISSING
(qfix asks for the TSQR fit and the image does not export it: refused as the
door refuses), DECOMP_ROUTE_RANDOMIZED (the x_decomp image has the randomized
fit), DECOMP_ROUTE_RESIDENT (the entry keeps components resident: the state
opens one lease at its first transform and every transform / inverse reads
that device copy, as the door's lease does; closed at refit, import, free). No score is published (neither class has one).

MEMBER_OP_NEW parameters (Float64 words, 9):
  [0] kind: 0 PCA, 1 TruncatedSVD
  [1] n_components: -1 None (min(n_samples, n_features)), else the count
  [2] keep mode (PCA): 0 the count in [1], 1 a fraction in (0, 1) ([3]), 2 'mle'
  [3] the fraction (keep mode 1)
  [4] solver: 0 covariance ('auto', 'covariance_eigh', 'jacobi'), 1 'full', 2 'randomized'
  [5] whiten (PCA): 0 / 1
  [6] n_oversamples (randomized)
  [7] power iterations (randomized; PCA iterated_power, TruncatedSVD n_iter): -1 'auto'
  [8] seed (randomized): random_state & 0xFFFFFFFF, 0 for None

Views: X dense row-major host float32; no y (ignored), no sample weights (a
given weight is refused: no MEMBER_CAN_SAMPLE_WEIGHT), no row or column
index, no device-home view (the caller hands over gathered host words). The
transform of an unwhitened PCA and of TruncatedSVD tests X's finite words in
the kernel, over its own upload, as the doors do; the caller's UNPACK admits
an inverse_transform's input (and a whitened PCA transform's) finite, as the
doors' admission does. `call.fit_output = MEMBER_OP_TRANSFORM` writes the
training rows' projection to `dst` (rows x n_components) in the fit call;
PCA refuses it for a fraction or 'mle' n_components (the width is known only
after the fit: the caller fits, asks OUTPUTS, then transforms).

Fitted-state blocks (MEMBER_OP_EXPORT / MEMBER_OP_IMPORT), in index order:
  0 `MEMBER_BLOCK_SCALARS` float64: PCA [noise_variance_, n_components_,
    n_features_in_, n_samples_]; TruncatedSVD [n_components_, n_features_in_]
  1 `DECOMP_BLOCK_COMPONENTS` (MEMBER_BLOCK_FAMILY) float32 n_components_ x n_features_in_
  then PCA only: `DECOMP_BLOCK_MEAN` float32 1 x n_features_in_
  then `DECOMP_BLOCK_VARIANCE`, `DECOMP_BLOCK_RATIO`, `DECOMP_BLOCK_SINGULAR`
    float32 1 x n_components_ (explained_variance_, explained_variance_ratio_,
    singular_values_).
PCA has 6 blocks, TruncatedSVD 5; an import needs every block of its kind.

The fitted state is OWNED here, in a process-wide store of fixed chunks (the
linear member's store, core/linear_member.mojo): a slot never moves once
minted, the lock covers only mint, free and the handle lookup. Members are
independent: a caller runs folds and candidates on its own workers, one
handle each. Serial by nature here: the store's bookkeeping.
"""
from std.sys.info import size_of
from std.memory import unsafe_memcpy
from core.admitted_refusals import admitted_finite_refuse
from core.pca_native_api import PCAFitRequest, PCAFitResult, PCAQueryRequest, PCAFoldStore, PCANativeDriver
from core.tsvd_native_api import (
    TSVDFitRequest, TSVDQueryRequest, TSVDNativeDriver, TSVD_GRAM, TSVD_FUSED, TSVD_EXPLAINED,
    TSVD_RANDOMIZED, TSVD_TSQR,
)
from bindings.pca_exact_run import pca_refusal, pca_too_small
from x_decomp.tsvd_run import _too_small as tsvd_too_small
from core.member_abi import (
    MemberCall, MemberMatrix, MEMBER_F32, MEMBER_F64,
    MEMBER_OP_CAPABILITIES, MEMBER_OP_NEW, MEMBER_OP_FREE, MEMBER_OP_FIT, MEMBER_OP_OUTPUTS,
    MEMBER_OP_TRANSFORM, MEMBER_OP_INVERSE_TRANSFORM, MEMBER_OP_EXPORT, MEMBER_OP_IMPORT,
    MEMBER_CAN_FIT, MEMBER_CAN_TRANSFORM, MEMBER_CAN_INVERSE_TRANSFORM, MEMBER_CAN_EXPORT, MEMBER_CAN_IMPORT,
    MEMBER_BLOCK_SCALARS, MEMBER_BLOCK_FAMILY, MEMBER_BLOCK_BY_INDEX, MEMBER_IMPORT_DONE,
    member_publish, member_call_from_spans, member_unsupported,
)
from core.member_families import MEMBER_FAMILY_DECOMPOSITION
from core.native_descriptor_abi import NativeDescriptorError
from core.entry_table import EntryRef, entry_ref_unresolved
from core.native_descriptor_call import native_descriptor_report
from core.shared_slot import SharedSlot, shared_slot_lock, shared_slot_unlock

comptime DECOMP_KIND_PCA = 0
comptime DECOMP_KIND_TSVD = 1
comptime DECOMP_PARAM_COUNT = 9

comptime DECOMP_BLOCK_COMPONENTS = MEMBER_BLOCK_FAMILY
comptime DECOMP_BLOCK_MEAN = MEMBER_BLOCK_FAMILY + 1
comptime DECOMP_BLOCK_VARIANCE = MEMBER_BLOCK_FAMILY + 2
comptime DECOMP_BLOCK_RATIO = MEMBER_BLOCK_FAMILY + 3
comptime DECOMP_BLOCK_SINGULAR = MEMBER_BLOCK_FAMILY + 4

comptime DECOMP_ROUTE_FUSED = 1
comptime DECOMP_ROUTE_TSQR = 2
comptime DECOMP_ROUTE_TSQR_MISSING = 4
comptime DECOMP_ROUTE_RANDOMIZED = 8
#: The TruncatedSVD entry keeps components resident under a lease
#: (TSVD_CAP_RESIDENT): the member's transforms read the lease's device copy,
#: as the door's (bindings/tsvd_native_door.mojo `_resident_lease`).
comptime DECOMP_ROUTE_RESIDENT = 16

#: x_decomp/tsqr_core.mojo TS_MAX_N: the widest X the TSQR takes (the
#: TruncatedSVD door's `_tsqr_connection` rule, a kernel width limit).
comptime _TS_MAX_N = 512

comptime _CHUNK = 256
comptime _INDEX_SPAN = 16777216


struct DecompositionMemberSlot(Copyable, Movable, Defaultable):
    var generation: Int
    var live: Bool
    var words: List[Float64]
    var fitted: Bool
    var keep: Int
    var features: Int
    var fit_rows: Int
    var noise: Float64
    var components: List[Float32]
    var mean: List[Float32]
    var variance: List[Float32]
    var ratio: List[Float32]
    var singular: List[Float32]
    var scalars: List[Float64]
    # TruncatedSVD: the resident lease of this state's components (0 none)
    # and the entry that opened it (the one that closes it).
    var lease: Int
    var lease_entry: EntryRef

    def __init__(out self):
        self.lease = 0
        self.lease_entry = entry_ref_unresolved()
        self.generation = 0
        self.live = False
        self.words = List[Float64]()
        self.fitted = False
        self.keep = 0
        self.features = 0
        self.fit_rows = 0
        self.noise = 0.0
        self.components = List[Float32]()
        self.mean = List[Float32]()
        self.variance = List[Float32]()
        self.ratio = List[Float32]()
        self.singular = List[Float32]()
        self.scalars = List[Float64]()

    def kind(self) -> Int:
        return Int(self.words[0])

    def clear(mut self):
        self.fitted = False
        self.keep = 0
        self.features = 0
        self.fit_rows = 0
        self.noise = 0.0
        self.components = List[Float32]()
        self.mean = List[Float32]()
        self.variance = List[Float32]()
        self.ratio = List[Float32]()
        self.singular = List[Float32]()
        self.scalars = List[Float64]()


struct DecompositionMemberStore(Defaultable, Movable):
    var synchronization: List[Int64]
    var chunks: List[List[DecompositionMemberSlot]]
    var free: List[Int]
    var count: Int

    def __init__(out self):
        # Side-effect-free even when the process slot's initializers race.
        self.synchronization = List[Int64](length=2, fill=0)
        self.chunks = List[List[DecompositionMemberSlot]]()
        self.free = List[Int]()
        self.count = 0


comptime _STORE = SharedSlot[DecompositionMemberStore, "MojolearnDecompositionMemberStoreV2", 2]
comptime _Slot = MutPointer[DecompositionMemberSlot, MutUntrackedOrigin]
comptime _Call = MutPointer[MemberCall, MutUntrackedOrigin]


def _slot_address(index: Int) raises -> Int:
    # Under the store's lock. A chunk's buffer is allocated once and never
    # grows, so the address stays valid after the lock is released.
    var store = _STORE.get_or_create_ptr()
    return Int(store[].chunks[index // _CHUNK].unsafe_ptr()) + (index % _CHUNK) * size_of[DecompositionMemberSlot]()


def _mint(var words: List[Float64]) raises -> Int:
    var store = _STORE.get_or_create_ptr()
    var lock = rebind[MutPointer[Int64, MutUntrackedOrigin]](store[].synchronization.unsafe_ptr())
    shared_slot_lock(lock)
    try:
        var index = -1
        if len(store[].free) > 0:
            index = store[].free.pop()
        else:
            if store[].count >= _INDEX_SPAN - 1:
                raise Error("RuntimeError: mojolearn: too many live decomposition member states")
            if store[].count == len(store[].chunks) * _CHUNK:
                store[].chunks.append(List[DecompositionMemberSlot](length=_CHUNK, fill=DecompositionMemberSlot()))
            index = store[].count
            store[].count += 1
        var slot = _Slot(unsafe_from_address=_slot_address(index))
        slot[].generation += 1
        slot[].live = True
        slot[].words = words^
        slot[].clear()
        return slot[].generation * _INDEX_SPAN + index + 1
    finally:
        shared_slot_unlock(lock)


def _lookup(handle: Int, release: Bool) raises -> Int:
    """The slot address of a live handle. A released handle, a handle of
    another store or a made-up integer is refused: never a stale read."""
    var store = _STORE.get_or_create_ptr()
    var lock = rebind[MutPointer[Int64, MutUntrackedOrigin]](store[].synchronization.unsafe_ptr())
    var address = 0
    var lease = 0
    var lease_entry = entry_ref_unresolved()
    shared_slot_lock(lock)
    try:
        var index = handle % _INDEX_SPAN - 1
        if handle <= 0 or index < 0 or index >= store[].count:
            raise Error("NotFittedError: this member state was released or never created")
        address = _slot_address(index)
        var slot = _Slot(unsafe_from_address=address)
        if not slot[].live or slot[].generation != handle // _INDEX_SPAN:
            raise Error("NotFittedError: this member state was released or never created")
        if release:
            # The lease is taken out under the lock and closed after it: the
            # close is the kernel store's, never under this store's lock.
            lease = slot[].lease
            lease_entry = slot[].lease_entry
            slot[].lease = 0
            slot[].lease_entry = entry_ref_unresolved()
            slot[].live = False
            slot[].clear()
            slot[].words = List[Float64]()
            store[].free.append(index)
    finally:
        shared_slot_unlock(lock)
    if lease != 0:
        TSVDNativeDriver(lease_entry.call).close(lease)
    return address


def _drop_lease(slot: _Slot) raises:
    """Close this state's resident lease before its components change (a
    refit, an import): the resident copy is keyed by the components' address
    and size, which a new list may reuse."""
    if slot[].lease == 0:
        return
    var lease = slot[].lease
    slot[].lease = 0
    var entry = slot[].lease_entry
    slot[].lease_entry = entry_ref_unresolved()
    TSVDNativeDriver(entry.call).close(lease)


def _lease(slot: _Slot, call: _Call) raises -> Int:
    """The resident lease of this state's components (opened at its first
    transform, as the door opens its lease at the first transform of a fitted
    model), 0 when the entry keeps no resident copy (the per-call route)."""
    if slot[].lease != 0 or (call[].route.flags & DECOMP_ROUTE_RESIDENT) == 0:
        return slot[].lease
    var entry = call[].route.entry
    var lease = TSVDNativeDriver(entry.call).open()
    if lease > 0:
        slot[].lease = lease
        slot[].lease_entry = entry
    return slot[].lease


def _name(slot: _Slot) -> String:
    return String("PCA") if slot[].kind() == DECOMP_KIND_PCA else String("TruncatedSVD")


def _dense(view: MemberMatrix, what: String, estimator: String) raises:
    if not view.dense_host(MEMBER_F32):
        raise Error("TypeError: a " + estimator + " member reads " + what + " as a dense host float32 view")


def _whole(call: _Call, estimator: String) raises:
    if call[].row_index.rows > 0 or call[].column_index.rows > 0:
        raise Error("TypeError: a " + estimator + " member reads whole host views (no row or column index)")


def _prefix(mut words: List[Float32], count: Int) -> List[Float32]:
    var part = List[Float32](length=count, fill=0.0)
    if count > 0:
        unsafe_memcpy(dest=part.unsafe_ptr(), src=words.unsafe_ptr(), count=count)
    return part^


def _fit_checks(call: _Call, estimator: String) raises:
    _whole(call, estimator)
    _dense(call[].x, "X", estimator)
    if call[].weights.rows > 0:
        raise Error("TypeError: a " + estimator + " member takes no sample_weight")
    if call[].fit_output != 0 and call[].fit_output != MEMBER_OP_TRANSFORM:
        member_unsupported(estimator, call[].fit_output)


def _fit_pca(slot: _Slot, call: _Call) raises:
    var estimator = String("PCA")
    _fit_checks(call, estimator)
    var route = call[].route
    var rows = call[].x.rows
    var cols = call[].x.cols
    var keep_mode = Int(slot[].words[2])
    var solver = Int(slot[].words[4])
    var randomized = solver == 2
    var dense = solver == 1 and cols >= 2
    if rows < 2 or (randomized and cols < 2):
        raise Error(pca_too_small(rows, cols))
    if dense and rows < cols and route.mode != 1:
        raise Error("NotImplementedError: mojolearn PCA: svd_solver='full' needs at least as many samples as "
                    + "features and got " + String(rows) + " x " + String(cols) + ". The dense route is an R-SVD, "
                    + "which needs a tall matrix; svd_solver='covariance_eigh' handles this shape "
                    + "(decomposition/NOT_IMPLEMENTED.tsv, DEVIATION 593)")
    var count = min(rows, cols)
    var fraction = Float64(0)
    if keep_mode == 2:
        if rows < cols:
            raise Error("n_components='mle' is only supported if n_samples >= n_features")
    elif keep_mode == 1:
        fraction = slot[].words[3]
    else:
        if slot[].words[1] >= 0.0:
            count = Int(slot[].words[1])
        if count < 1 or count > cols:
            raise Error("mojolearn n_components must be in [1, " + String(cols) + "], got " + String(count))
    if dense and count > min(rows, cols):
        raise Error("full SVD n_components cannot exceed min(n_samples, n_features)")
    if randomized and count >= min(rows, cols):
        raise Error("svd_solver='randomized' needs n_components < min(n_samples, n_features)")
    var projection = 0
    if call[].fit_output == MEMBER_OP_TRANSFORM:
        if keep_mode != 0:
            raise Error("NotImplementedError: a PCA member writes fit_transform's output in the fit only for an "
                        + "integer n_components; fit, ask its outputs, then transform")
        _dense(call[].dst, "the projection buffer", estimator)
        if call[].dst.rows != rows or call[].dst.cols != count:
            raise Error("ValueError: the projection buffer holds n_components values per row of X")
        projection = call[].dst.data
    slot[].clear()
    slot[].components = List[Float32](length=count * cols, fill=0.0)
    slot[].mean = List[Float32](length=cols, fill=0.0)
    slot[].variance = List[Float32](length=count, fill=0.0)
    slot[].ratio = List[Float32](length=count, fill=0.0)
    slot[].singular = List[Float32](length=count, fill=0.0)
    var iterations = Int(slot[].words[7])
    var request = PCAFitRequest(call[].x.data, Int(slot[].components.unsafe_ptr()), Int(slot[].mean.unsafe_ptr()),
        Int(slot[].variance.unsafe_ptr()), Int(slot[].ratio.unsafe_ptr()), Int(slot[].singular.unsafe_ptr()),
        projection, rows, cols, count, dense, keep_mode, slot[].words[5] != 0.0, randomized,
        Int(slot[].words[6]), max(iterations, 0), iterations < 0, Int(slot[].words[8]), True, fraction,
        PCAFoldStore())
    var fitted = _pca_fit(PCANativeDriver(route.entry.call), request, randomized)
    admitted_finite_refuse(fitted.finite, estimator)
    var keep = fitted.keep
    if keep != count:
        slot[].components = _prefix(slot[].components, keep * cols)
        slot[].variance = _prefix(slot[].variance, keep)
        slot[].ratio = _prefix(slot[].ratio, keep)
        slot[].singular = _prefix(slot[].singular, keep)
    slot[].keep = keep
    slot[].features = cols
    slot[].fit_rows = rows
    slot[].noise = fitted.noise
    slot[].scalars = [fitted.noise, Float64(keep), Float64(cols), Float64(rows)]
    slot[].fitted = True


def _pca_fit(driver: PCANativeDriver, request: PCAFitRequest, randomized: Bool) raises -> PCAFitResult:
    # The randomized arm's refusals take the PCA prefix, as the door's do.
    if not randomized:
        return driver.fit(request)
    try:
        return driver.fit(request)
    except error:
        var failure = pca_refusal(error^)
        raise failure^


def _fit_tsvd(slot: _Slot, call: _Call) raises:
    var estimator = String("TruncatedSVD")
    _fit_checks(call, estimator)
    var route = call[].route
    var rows = call[].x.rows
    var cols = call[].x.cols
    if rows < 2 or cols < 2:
        raise Error(tsvd_too_small(rows, cols))
    var nc = min(rows, cols) if slot[].words[1] < 0.0 else Int(slot[].words[1])
    if nc < 1 or nc > cols:
        raise Error("mojolearn n_components must be in [1, " + String(cols) + "], got " + String(nc))
    if call[].fit_output == MEMBER_OP_TRANSFORM:
        _dense(call[].dst, "the projection buffer", estimator)
        if call[].dst.rows != rows or call[].dst.cols != nc:
            raise Error("ValueError: the projection buffer holds n_components values per row of X")
    _drop_lease(slot)
    slot[].clear()
    slot[].components = List[Float32](length=nc * cols, fill=0.0)
    slot[].variance = List[Float32](length=nc, fill=0.0)
    slot[].ratio = List[Float32](length=nc, fill=0.0)
    slot[].singular = List[Float32](length=nc, fill=0.0)
    var request = TSVDFitRequest(call[].x.data, Int(slot[].components.unsafe_ptr()),
        Int(slot[].singular.unsafe_ptr()), Int(slot[].variance.unsafe_ptr()), Int(slot[].ratio.unsafe_ptr()),
        rows, cols, nc, 0, 0, False, 0, True, TSVD_GRAM, PCAFoldStore())
    var flags = route.flags
    if Int(slot[].words[4]) == 2:
        if not route.has_second or (flags & DECOMP_ROUTE_RANDOMIZED) == 0:
            raise Error("NotImplementedError: mojolearn TruncatedSVD: this build of _mojolearn_x_decomp does not export "
                        + "x_decomp_tsvd_randomized; rebuild it with bindings/build_x_decomp.sh")
        var iterations = Int(slot[].words[7])
        request.oversamples = Int(slot[].words[6])
        request.iterations = max(iterations, 0)
        request.auto_iterations = iterations < 0
        request.seed = Int(slot[].words[8])
        request.route = TSVD_RANDOMIZED
        var result = TSVDNativeDriver(route.second.call).fit(request)
        admitted_finite_refuse(result.finite, estimator)
    else:
        var tsqr = (route.has_second and (flags & (DECOMP_ROUTE_TSQR | DECOMP_ROUTE_TSQR_MISSING)) != 0
            and cols <= _TS_MAX_N and rows >= cols and nc <= cols)
        var driver = TSVDNativeDriver(route.entry.call)
        var explained = True
        if tsqr:
            if (flags & DECOMP_ROUTE_TSQR) == 0:
                raise Error("NotImplementedError: mojolearn TruncatedSVD: this build of _mojolearn_x_decomp does not export "
                            + "x_decomp_tsvd_tsqr; rebuild it with bindings/build_x_decomp.sh")
            request.route = TSVD_TSQR
            var result = TSVDNativeDriver(route.second.call).fit(request)
            admitted_finite_refuse(result.finite, estimator)
        else:
            var fused = (flags & DECOMP_ROUTE_FUSED) != 0
            request.route = TSVD_FUSED if fused else TSVD_GRAM
            var result = driver.fit(request)
            admitted_finite_refuse(result.finite, estimator)
            explained = not fused
        if explained:
            request.route = TSVD_EXPLAINED
            request.test_finite = False
            _ = driver.fit(request)
    slot[].keep = nc
    slot[].features = cols
    slot[].fit_rows = rows
    slot[].scalars = [Float64(nc), Float64(cols)]
    slot[].fitted = True
    if call[].fit_output == MEMBER_OP_TRANSFORM:
        _query(slot, call, False)


def _query(slot: _Slot, call: _Call, inverse: Bool) raises:
    """transform (X -> rows x n_components_) or inverse_transform (rows x
    n_components_ -> rows x n_features_in_) through the route's entry."""
    var estimator = _name(slot)
    if not slot[].fitted:
        raise Error("NotFittedError: this " + estimator + " member is not fitted yet")
    _whole(call, estimator)
    _dense(call[].x, "X", estimator)
    _dense(call[].dst, "the output buffer", estimator)
    var rows = call[].x.rows
    var width_in = slot[].keep if inverse else slot[].features
    var width_out = slot[].features if inverse else slot[].keep
    if call[].x.cols != width_in:
        if inverse:
            raise Error("mojolearn " + estimator + " component count differs from fit")
        raise Error("X has " + String(call[].x.cols) + " features, but " + estimator + " is expecting "
                    + String(slot[].features) + " features as input.")
    if call[].dst.rows != rows or call[].dst.cols != width_out:
        raise Error("ValueError: the output buffer holds " + String(width_out) + " values per row of X")
    if slot[].kind() == DECOMP_KIND_PCA:
        var whiten = slot[].words[5] != 0.0
        var request = PCAQueryRequest(call[].x.data, Int(slot[].mean.unsafe_ptr()),
            Int(slot[].components.unsafe_ptr()), 0, call[].dst.data, rows, slot[].features, slot[].keep,
            slot[].fit_rows, inverse, whiten, PCAFoldStore())
        var driver = PCANativeDriver(call[].route.entry.call)
        if whiten:
            if slot[].fit_rows < 2:
                raise Error("PCA whitening requires fit rows >=2")
            request.singular = Int(slot[].singular.unsafe_ptr())
            try:
                _ = driver.query(request)
            except error:
                var failure = pca_refusal(error^)
                raise failure^
        elif inverse:
            _ = driver.query(request)
        else:
            var status = driver.query(request)
            admitted_finite_refuse(status.finite, estimator)
    else:
        # The door's resident route: components_ read from the lease's device
        # copy (uploaded once per fitted state), never uploaded per call.
        var lease = _lease(slot, call)
        var request = TSVDQueryRequest(call[].x.data, Int(slot[].components.unsafe_ptr()), call[].dst.data,
            rows, slot[].features, slot[].keep, inverse, not inverse, lease > 0, lease)
        var status = TSVDNativeDriver(call[].route.entry.call).query(request)
        if not inverse:
            admitted_finite_refuse(status, estimator)


def _block_name(kind: Int, index: Int) raises -> Int:
    if index == 0:
        return MEMBER_BLOCK_SCALARS
    if index == 1:
        return DECOMP_BLOCK_COMPONENTS
    if kind == DECOMP_KIND_PCA:
        if index >= 2 and index <= 5:
            return DECOMP_BLOCK_COMPONENTS + index - 1
    elif index >= 2 and index <= 4:
        return DECOMP_BLOCK_COMPONENTS + index
    raise Error("member ABI: a decomposition member has no fitted block index " + String(index))


def _export(slot: _Slot, call: _Call) raises:
    """A view of one fitted block. The slot owns the words; they stay valid
    until this state's next fit, import or free."""
    if not slot[].fitted:
        raise Error("NotFittedError: this " + _name(slot) + " member is not fitted yet")
    var pca = slot[].kind() == DECOMP_KIND_PCA
    var block = call[].block
    if (call[].flags & MEMBER_BLOCK_BY_INDEX) != 0:
        block = _block_name(slot[].kind(), block)
    var keep = slot[].keep
    var cols = slot[].features
    if block == MEMBER_BLOCK_SCALARS:
        call[].dst = MemberMatrix(Int(slot[].scalars.unsafe_ptr()), 1, len(slot[].scalars), MEMBER_F64)
    elif block == DECOMP_BLOCK_COMPONENTS:
        call[].dst = MemberMatrix(Int(slot[].components.unsafe_ptr()), keep, cols)
    elif block == DECOMP_BLOCK_MEAN and pca:
        call[].dst = MemberMatrix(Int(slot[].mean.unsafe_ptr()), 1, cols)
    elif block == DECOMP_BLOCK_VARIANCE:
        call[].dst = MemberMatrix(Int(slot[].variance.unsafe_ptr()), 1, keep)
    elif block == DECOMP_BLOCK_RATIO:
        call[].dst = MemberMatrix(Int(slot[].ratio.unsafe_ptr()), 1, keep)
    elif block == DECOMP_BLOCK_SINGULAR:
        call[].dst = MemberMatrix(Int(slot[].singular.unsafe_ptr()), 1, keep)
    else:
        raise Error("member ABI: a " + _name(slot) + " member has no fitted block " + String(block))
    call[].block = block
    call[].out_cols = 6 if pca else 5


def _words(view: MemberMatrix, estimator: String) raises -> List[Float32]:
    _dense(view, "a fitted block", estimator)
    var count = view.rows * view.cols
    var words = List[Float32](length=count, fill=0.0)
    unsafe_memcpy(dest=words.unsafe_ptr(),
        src=MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=view.data), count=count)
    return words^


def _import(slot: _Slot, call: _Call) raises:
    """Copy one fitted block in; MEMBER_IMPORT_DONE completes the state."""
    var estimator = _name(slot)
    var pca = slot[].kind() == DECOMP_KIND_PCA
    var block = call[].block
    _drop_lease(slot)
    if block != 0 or (call[].flags & MEMBER_IMPORT_DONE) == 0:
        slot[].fitted = False
    if block == MEMBER_BLOCK_SCALARS:
        var size = 4 if pca else 2
        if not call[].x.dense_host(MEMBER_F64) or call[].x.rows * call[].x.cols != size:
            raise Error("member ABI: the " + estimator + " scalars block is " + String(size) + " host float64 words")
        var source = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call[].x.data)
        slot[].scalars = List[Float64](length=size, fill=0.0)
        for index in range(size):  # driver-loop: two or four fitted scalars
            slot[].scalars[index] = source[index]
    elif block == DECOMP_BLOCK_COMPONENTS:
        slot[].components = _words(call[].x, estimator)
    elif block == DECOMP_BLOCK_MEAN and pca:
        slot[].mean = _words(call[].x, estimator)
    elif block == DECOMP_BLOCK_VARIANCE:
        slot[].variance = _words(call[].x, estimator)
    elif block == DECOMP_BLOCK_RATIO:
        slot[].ratio = _words(call[].x, estimator)
    elif block == DECOMP_BLOCK_SINGULAR:
        slot[].singular = _words(call[].x, estimator)
    elif block != 0 or (call[].flags & MEMBER_IMPORT_DONE) == 0:
        raise Error("member ABI: a " + estimator + " member has no fitted block " + String(block))
    if (call[].flags & MEMBER_IMPORT_DONE) == 0:
        return
    if len(slot[].scalars) != (4 if pca else 2):
        raise Error("member ABI: a fitted " + estimator + " state needs its scalars block")
    var keep = Int(slot[].scalars[1] if pca else slot[].scalars[0])
    var cols = Int(slot[].scalars[2] if pca else slot[].scalars[1])
    if (keep < 0 or cols < 1 or len(slot[].components) != keep * cols or len(slot[].variance) != keep
            or len(slot[].ratio) != keep or len(slot[].singular) != keep or (pca and len(slot[].mean) != cols)):
        raise Error("member ABI: a fitted " + estimator + " state needs every fitted block, of the fitted sizes")
    slot[].keep = keep
    slot[].features = cols
    slot[].noise = slot[].scalars[0] if pca else 0.0
    slot[].fit_rows = Int(slot[].scalars[3]) if pca else 0
    slot[].fitted = True


def _outputs(call: _Call) raises:
    """`call.out_cols` of transform / inverse_transform. Before a fit the
    transform width is known only for an integer n_components."""
    call[].classes = 0
    if call[].state == 0:
        raise Error("member ABI: a decomposition member's output width needs its state")
    var slot = _Slot(unsafe_from_address=_lookup(call[].state, False))
    var asked = call[].flags
    if asked != MEMBER_OP_TRANSFORM and asked != MEMBER_OP_INVERSE_TRANSFORM:
        member_unsupported(_name(slot), asked)
    if slot[].fitted:
        call[].out_cols = slot[].keep if asked == MEMBER_OP_TRANSFORM else slot[].features
        return
    if asked == MEMBER_OP_TRANSFORM and Int(slot[].words[2]) == 0 and slot[].words[1] >= 0.0:
        call[].out_cols = Int(slot[].words[1])
        return
    raise Error("NotFittedError: this " + _name(slot) + " member's output width is known after its fit")


def _decomposition_member(operation: Int, address: Int) raises -> Int:
    var call = _Call(unsafe_from_address=address)
    if operation == MEMBER_OP_CAPABILITIES:
        return MEMBER_CAN_FIT | MEMBER_CAN_TRANSFORM | MEMBER_CAN_INVERSE_TRANSFORM | MEMBER_CAN_EXPORT | MEMBER_CAN_IMPORT
    if operation == MEMBER_OP_NEW:
        if call[].param_count != DECOMP_PARAM_COUNT or not call[].params:
            raise Error("member ABI: a decomposition member takes nine parameters")
        var source = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call[].params)
        if source[0] != 0.0 and source[0] != 1.0:
            raise Error("member ABI: unknown decomposition member kind")
        var words = List[Float64](length=DECOMP_PARAM_COUNT, fill=0.0)
        for index in range(DECOMP_PARAM_COUNT):  # driver-loop: nine parameter words
            words[index] = source[index]
        call[].state = _mint(words^)
        return 0
    if operation == MEMBER_OP_FREE:
        _ = _lookup(call[].state, True)
        call[].state = 0
        return 0
    if operation == MEMBER_OP_OUTPUTS:
        _outputs(call)
        return 0
    var slot = _Slot(unsafe_from_address=_lookup(call[].state, False))
    if operation == MEMBER_OP_FIT:
        if slot[].kind() == DECOMP_KIND_PCA:
            _fit_pca(slot, call)
        else:
            _fit_tsvd(slot, call)
    elif operation == MEMBER_OP_TRANSFORM:
        _query(slot, call, False)
    elif operation == MEMBER_OP_INVERSE_TRANSFORM:
        _query(slot, call, True)
    elif operation == MEMBER_OP_EXPORT:
        _export(slot, call)
    elif operation == MEMBER_OP_IMPORT:
        _import(slot, call)
    else:
        member_unsupported(_name(slot), operation)
    return 0


def decomposition_member_v2(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
    result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """The member entry of PCA and TruncatedSVD (schema `MEMBER_ABI`)."""
    try:
        var word = _decomposition_member(operation, member_call_from_spans(operation, aa, na, np, nf))
        if result:
            MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = word
        return 0
    except error:
        native_descriptor_report(error, context, sink)
        return 1


def decomposition_member_publish(mode: Int, vendor: Int, publisher: Int) raises:
    """Publish this image's decomposition member entry. One line in the
    image's module initialization, after its kernel tables are published."""
    member_publish[MEMBER_FAMILY_DECOMPOSITION](mode, vendor, publisher, decomposition_member_v2)
