# SPDX-License-Identifier: Apache-2.0
"""The Discriminant family as MEMBERS (core/member_abi.mojo):
LinearDiscriminantAnalysis (kind 0) and QuadraticDiscriminantAnalysis
(kind 1).

One entry, `discriminant_member_v2`, published by each x_prep image beside
its program transport (`discriminant_member_publish`,
x_prep/transport_device.mojo and x_prep/transport_host.mojo). A member runs
the estimator's own x_prep programs: the typed builders the door builds
through (bindings/prep_classify_prog_run.mojo `cls_lda_svd_program`,
`cls_lda_cov_program`, `cls_qda_fit_program`, `cls_lda_transform_program`,
`cls_lda_pair_program`, `cls_score_program`) and the typed run the
NaiveBayes members run (`cls_borrowed_run`) over the transport entry of the
`DeviceRoute` it is handed. Same programs, same stages, same order as the
door: no bit moves. No kernel is touched.

Route slots read: those of core/naive_bayes_member.mojo (`entry` the x_prep
transport entry; `flags` NB_ROUTE_OUT, _SCRATCH, _RANGES, _DEV_STORE,
_TAKE_ROWS, _PROBA64).

MEMBER_OP_NEW parameters (Float64 words, 13 + P):
  [0] kind: 0 LinearDiscriminantAnalysis, 1 QuadraticDiscriminantAnalysis
  [1] solver: 0 'svd', 1 'lsqr' (kind 0 only), 2 'eigen'
  [2] shrinkage given 0 / 1 (never with 'svd': refused at admission)
  [3] the shrinkage (-1.0: 'auto')
  [4] tol
  [5] store_covariance 0 / 1
  [6] kind 0: n_components (-1.0: None); kind 1: reg_param
  [7] blocked statistics 0 / 1 (MOJOLEARN_XPREP_BLOCKED and the image's
      IDN_STATS_BLOCKED bit)
  [8] one-pass class statistics 0 / 1 (the switch and IDN_CLASS_ONEPASS)
  [9] the image's `x_prep_idn_fam2` word (0 with the switch off)
  [10] R3 work region 0 / 1 (`xprep_r3("work")`)
  [11] kind 0: the image's C04 bit 0 / 1 (centered matmul of transform)
  [12] P: the count of the `priors` parameter (0: priors=None)
  [13 .. 13 + P) the priors (non-negative: checked at admission; P must
      equal the class count at fit)

Views: X dense row-major host float32 (finite: the caller's admission tests
it; the programs do not); `y` the class CODES 0..classes-1, one host
column, int32 (read in place) or int64; `call.classes` the class count at
FIT. No sample weights (the estimators take none). PREDICT writes codes,
one column, int32 or int64 as `dst` says. PREDICT_PROBA writes rows x
classes, float64 on a route with NB_ROUTE_PROBA64, else float32. DECISION
writes float32: one column with two classes, else rows x classes.
TRANSFORM (kind 0, solver 'svd' or 'eigen') writes float32 rows x
min(max components, rank). `call.row_index`, `call.scope` / `call.pin`: as
the NaiveBayes members (THE PIN RULE: the run writes `call.pin_kept`). No
score, no column index, no device-home view, no fit_output.

Fitted-state blocks (MEMBER_OP_EXPORT / MEMBER_OP_IMPORT), by index; a block
the fitted state does not have exports as an empty view (rows 0) and is not
imported:
  0 `MEMBER_BLOCK_SCALARS` float64 [n_features_in_, classes, rank, max components]
  1 `DA_BLOCK_MEANS` float32 classes x features (means_)
  2 `DA_BLOCK_PRIORS` 1 x classes (priors_)
  3 `DA_BLOCK_XBAR` 1 x features (LDA xbar_)
  4 `DA_BLOCK_COEF` classes x features (LDA `_coef`)
  5 `DA_BLOCK_INTER` 1 x classes (LDA `_inter`)
  6 `DA_BLOCK_SCAL` features x features (LDA `_scal_full`; not 'lsqr')
  7 `DA_BLOCK_EVR` 1 x max components (LDA explained_variance_ratio_; not 'lsqr')
  8 `DA_BLOCK_COV` LDA features x features (covariance_; 'svd' only when
     stored); QDA classes x features^2 (covariance_, when stored)
  9 `DA_BLOCK_PAIR_COEF` 1 x features (LDA coef_ with two classes)
  10 `DA_BLOCK_PAIR_INTER` 1 x 1 (LDA intercept_ with two classes)
  11 `DA_BLOCK_ROT` classes x features^2 (QDA `_rot`)
  12 `DA_BLOCK_LOGC` 1 x classes (QDA `_logc`)
  13 `DA_BLOCK_ROTATIONS` classes x features^2 (QDA rotations_)
  14 `DA_BLOCK_SCALINGS` classes x features (QDA scalings_)

By NAME only (lane mf-shaprfe, 2026-11-23; the kind's mask has MEMBER_CAN_COEF): LDA's
`MEMBER_BLOCK_COEF` / `MEMBER_BLOCK_INTERCEPT` are its public coef_ / intercept_, the
words of blocks 9 / 10 with two classes, else of blocks 4 / 5 (RFE reads coef_ this way,
bindings/prep_select_run.mojo `rfe_member_importances`). QDA exports neither.

QDA's rank refusal names the class by its CODE (the member has no labels).

The fitted state is OWNED here, in a process-wide store of fixed chunks (the
linear member's store). Serial by nature here: the store's bookkeeping and
the label passes of core/naive_bayes_member.mojo `_fit_views`; the programs
are the family's and run on the route's device.
"""
from std.sys.info import size_of
from std.memory import unsafe_memcpy
from bindings.prep_classify_prog_run import (
    ClsProgram, ClsBorrowed, ClsDaMode, CLS_NONE, CLS_W_PREDICT, CLS_W_PROBA, CLS_KIND_LDA, CLS_KIND_QDA,
    CLS_LDA_SVD, CLS_LDA_LSQR, CLS_LDA_EIGEN,
    cls_lda_svd_program, cls_lda_cov_program, cls_lda_transform_program, cls_lda_pair_program, cls_qda_fit_program,
    cls_score_program, cls_borrowed_run,
)
from core.naive_bayes_member import (
    NB_ROUTE_RANGES, NB_ROUTE_DEV_STORE, NB_ROUTE_PROBA64,
    _Rows, _rows, _x_source, _plain, _copy_out, _fit_views, _fit_shape, _run_plan, _route_gathers,
    nb_wide_index,
)
from core.member_abi import (
    MemberCall, MemberMatrix, MEMBER_F32, MEMBER_F64, MEMBER_I32, MEMBER_I64,
    MEMBER_OP_CAPABILITIES, MEMBER_OP_NEW, MEMBER_OP_FREE, MEMBER_OP_FIT, MEMBER_OP_PREDICT,
    MEMBER_OP_PREDICT_PROBA, MEMBER_OP_DECISION, MEMBER_OP_TRANSFORM, MEMBER_OP_OUTPUTS, MEMBER_OP_EXPORT,
    MEMBER_OP_IMPORT,
    MEMBER_CAN_FIT, MEMBER_CAN_PREDICT, MEMBER_CAN_PREDICT_PROBA, MEMBER_CAN_DECISION, MEMBER_CAN_TRANSFORM,
    MEMBER_CAN_EXPORT, MEMBER_CAN_IMPORT, MEMBER_CAN_CLASSIFIER, MEMBER_CAN_ROW_INDEX, MEMBER_CAN_SCOPE,
    MEMBER_CAN_COEF, MEMBER_BLOCK_COEF, MEMBER_BLOCK_INTERCEPT,
    MEMBER_BLOCK_SCALARS, MEMBER_BLOCK_FAMILY, MEMBER_BLOCK_BY_INDEX, MEMBER_IMPORT_DONE,
    member_publish, member_call_from_spans, member_unsupported,
)
from core.member_families import MEMBER_FAMILY_DISCRIMINANT
from core.host_parallel import host_widen_index_i64
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.shared_slot import SharedSlot, shared_slot_lock, shared_slot_unlock

comptime DA_KIND_LDA = 0
comptime DA_KIND_QDA = 1
#: The parameter words before the priors.
comptime DA_PARAM_HEAD = 13
comptime DA_SCALARS = 4

comptime DA_BLOCK_MEANS = MEMBER_BLOCK_FAMILY
comptime DA_BLOCK_PRIORS = MEMBER_BLOCK_FAMILY + 1
comptime DA_BLOCK_XBAR = MEMBER_BLOCK_FAMILY + 2
comptime DA_BLOCK_COEF = MEMBER_BLOCK_FAMILY + 3
comptime DA_BLOCK_INTER = MEMBER_BLOCK_FAMILY + 4
comptime DA_BLOCK_SCAL = MEMBER_BLOCK_FAMILY + 5
comptime DA_BLOCK_EVR = MEMBER_BLOCK_FAMILY + 6
comptime DA_BLOCK_COV = MEMBER_BLOCK_FAMILY + 7
comptime DA_BLOCK_PAIR_COEF = MEMBER_BLOCK_FAMILY + 8
comptime DA_BLOCK_PAIR_INTER = MEMBER_BLOCK_FAMILY + 9
comptime DA_BLOCK_ROT = MEMBER_BLOCK_FAMILY + 10
comptime DA_BLOCK_LOGC = MEMBER_BLOCK_FAMILY + 11
comptime DA_BLOCK_ROTATIONS = MEMBER_BLOCK_FAMILY + 12
comptime DA_BLOCK_SCALINGS = MEMBER_BLOCK_FAMILY + 13
#: The family blocks (the scalars block is index 0 before them).
comptime DA_BLOCKS = 14

comptime _CHUNK = 256
comptime _INDEX_SPAN = 16777216


struct DiscriminantMemberSlot(Copyable, Movable, Defaultable):
    var generation: Int
    var live: Bool
    var words: List[Float64]
    var fitted: Bool
    var features: Int
    var classes: Int
    #: The element type `y` had at fit (the type predict answers by default).
    var label_dtype: Int
    var scalars: List[Float64]
    #: The DA_BLOCKS family blocks (an absent block is empty) and their shapes.
    var blocks: List[List[Float32]]
    var block_rows: List[Int]
    var block_cols: List[Int]

    def __init__(out self):
        self.generation = 0
        self.live = False
        self.words = List[Float64]()
        self.fitted = False
        self.features = 0
        self.classes = 0
        self.label_dtype = MEMBER_I64
        self.scalars = List[Float64]()
        self.blocks = List[List[Float32]]()
        self.block_rows = List[Int]()
        self.block_cols = List[Int]()

    def kind(self) -> Int:
        return Int(self.words[0])

    def solver(self) -> Int:
        return Int(self.words[1])

    def clear(mut self):
        self.fitted = False
        self.features = 0
        self.classes = 0
        self.scalars = List[Float64]()
        var blocks = List[List[Float32]](capacity=DA_BLOCKS)
        for _ in range(DA_BLOCKS):  # driver-loop: over the family's fourteen fitted blocks
            blocks.append(List[Float32]())
        self.blocks = blocks^
        self.block_rows = List[Int](length=DA_BLOCKS, fill=0)
        self.block_cols = List[Int](length=DA_BLOCKS, fill=0)

    def put(mut self, block: Int, var words: List[Float32], rows: Int, cols: Int):
        var index = block - MEMBER_BLOCK_FAMILY
        self.blocks[index] = words^
        self.block_rows[index] = rows
        self.block_cols[index] = cols

    def size(self, block: Int) -> Int:
        return len(self.blocks[block - MEMBER_BLOCK_FAMILY])

    def address(self, block: Int) -> Int:
        return Int(self.blocks[block - MEMBER_BLOCK_FAMILY].unsafe_ptr())


struct DiscriminantMemberStore(Defaultable, Movable):
    var synchronization: List[Int64]
    var chunks: List[List[DiscriminantMemberSlot]]
    var free: List[Int]
    var count: Int

    def __init__(out self):
        # Side-effect-free even when the process slot's initializers race.
        self.synchronization = List[Int64](length=2, fill=0)
        self.chunks = List[List[DiscriminantMemberSlot]]()
        self.free = List[Int]()
        self.count = 0


comptime _STORE = SharedSlot[DiscriminantMemberStore, "MojolearnDiscriminantMemberStoreV2", 2]
comptime _Slot = MutPointer[DiscriminantMemberSlot, MutUntrackedOrigin]
comptime _Call = MutPointer[MemberCall, MutUntrackedOrigin]
comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _F64P = MutPointer[Float64, MutUntrackedOrigin]
comptime _I32P = MutPointer[Int32, MutUntrackedOrigin]
comptime _I64P = MutPointer[Int64, MutUntrackedOrigin]


def _slot_address(index: Int) raises -> Int:
    # Under the store's lock. A chunk's buffer is allocated once and never
    # grows, so the address stays valid after the lock is released.
    var store = _STORE.get_or_create_ptr()
    return Int(store[].chunks[index // _CHUNK].unsafe_ptr()) + (index % _CHUNK) * size_of[DiscriminantMemberSlot]()


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
                raise Error("RuntimeError: mojolearn: too many live Discriminant member states")
            if store[].count == len(store[].chunks) * _CHUNK:
                store[].chunks.append(List[DiscriminantMemberSlot](length=_CHUNK, fill=DiscriminantMemberSlot()))
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
            slot[].words = List[Float64]()
            store[].free.append(index)
        return address
    finally:
        shared_slot_unlock(lock)


def _kind_name(kind: Int) -> String:
    if kind == DA_KIND_QDA:
        return String("QuadraticDiscriminantAnalysis")
    return String("LinearDiscriminantAnalysis")


def discriminant_kind_published(kind: Int) -> Bool:
    """The kinds this entry serves (a class is marked when its kind is here)."""
    return kind == DA_KIND_LDA or kind == DA_KIND_QDA


def _family_capabilities(flags: Int) -> Int:
    # No MEMBER_CAN_TEST_ROWS: the programs do not refuse a non-finite word
    # of X; the caller's admission tests it. No sample weights.
    var mask = (MEMBER_CAN_FIT | MEMBER_CAN_PREDICT | MEMBER_CAN_PREDICT_PROBA | MEMBER_CAN_DECISION
                | MEMBER_CAN_EXPORT | MEMBER_CAN_IMPORT | MEMBER_CAN_CLASSIFIER)
    if (flags & NB_ROUTE_DEV_STORE) != 0 and (flags & NB_ROUTE_RANGES) != 0:
        mask |= MEMBER_CAN_SCOPE
    if _route_gathers(flags):
        mask |= MEMBER_CAN_ROW_INDEX
    return mask


def _mode(slot: _Slot) -> ClsDaMode:
    var p = slot[].words.unsafe_ptr()
    return ClsDaMode(p[7] != 0.0, p[8] != 0.0, Int(p[9]))


def _priors(slot: _Slot, K: Int, estimator: String) raises -> List[Float64]:
    var p = slot[].words.unsafe_ptr()
    var prior_count = Int(p[12])
    if prior_count != 0 and prior_count != K:
        raise Error("mojolearn: " + estimator + ": number of priors must match number of classes")
    var priors = List[Float64](capacity=prior_count)
    for i in range(prior_count):  # driver-loop: over the user prior list (class-count sized)
        priors.append(p[DA_PARAM_HEAD + i])
    return priors^


def _fit_rows(call: _Call, estimator: String) raises -> _Rows:
    var rows = _rows(call, estimator)
    _fit_shape(call, rows, estimator)
    if call[].weights.rows > 0:
        raise Error("TypeError: a " + estimator + " member takes no sample weights")
    return rows


def _fit_lda(slot: _Slot, call: _Call) raises:
    """LinearDiscriminantAnalysis.fit: the door's fit program (`_lda_fit`,
    `_lda_cov_solver`) over the caller's views, its refusals in its words,
    the fitted blocks copied out of the arena into the slot."""
    var estimator = String("LinearDiscriminantAnalysis")
    var rows = _fit_rows(call, estimator)
    var n = rows.n
    var d = rows.d
    var K = call[].classes
    if K < 2 or n <= K:
        raise Error("mojolearn: LinearDiscriminantAnalysis needs at least two classes and more samples than classes")
    var p = slot[].words.unsafe_ptr()
    var solver = slot[].solver()
    var maxc = min(K - 1, d)
    var asked = Int(p[6])
    if asked > maxc:
        raise Error("mojolearn: n_components cannot be larger than min(n_features, n_classes - 1)")
    var components = maxc if asked < 0 else asked
    var views = _fit_views(call, rows, estimator)
    var priors = _priors(slot, K, estimator)
    var program = ClsProgram(p[10] != 0.0)
    var eigen = solver == CLS_LDA_EIGEN
    # record order: X, the class codes, then (eigen) the one-class code vector
    var sources = List[ClsBorrowed](capacity=3)
    sources.append(_x_source(call, rows))
    sources.append(_plain(views.codes_address, n))
    var zeros = List[Float32]()
    var o = cls_lda_svd_program(program, _mode(slot), n, d, K, p[4], priors, p[5] != 0.0) if solver == CLS_LDA_SVD else (
        cls_lda_cov_program(program, _mode(slot), n, d, K, eigen, p[2] != 0.0, p[3], priors))
    if solver != CLS_LDA_SVD and eigen:
        zeros = List[Float32](length=n, fill=Float32(0))
        sources.append(_plain(Int(zeros.unsafe_ptr()), n))
    # THE PIN RULE (core/member_abi.mojo `MemberCall.pin`): the run writes
    # `call.pin_kept` at the moment the scope's cache takes the caller's
    # reference, so the caller reads it even when a later stage raises.
    var done = cls_borrowed_run(_run_plan(call), program, sources, call[].scope, call[].pin_kept)
    _ = views.keep()
    _ = len(zeros)
    var rank = 0
    if solver == CLS_LDA_SVD:
        rank = Int(done.arena.f32(o.meta + 2)[0])
    elif eigen:
        var values = done.arena.f32(o.e)
        for j in range(d):  # driver-loop: over the within-class covariance eigenvalues (feature count)
            if values[j] <= 0:
                raise Error("mojolearn: the within-class covariance is not positive definite (the reference's eigh(Sb, Sw) fails); set shrinkage")
        rank = d
    slot[].clear()
    slot[].put(DA_BLOCK_MEANS, _copy_out(done.arena.f32(o.mean), K * d), K, d)
    slot[].put(DA_BLOCK_PRIORS, _copy_out(done.arena.f32(o.priors), K), 1, K)
    slot[].put(DA_BLOCK_XBAR, _copy_out(done.arena.f32(o.xbar), d), 1, d)
    slot[].put(DA_BLOCK_COEF, _copy_out(done.arena.f32(o.coef), K * d), K, d)
    slot[].put(DA_BLOCK_INTER, _copy_out(done.arena.f32(o.inter), K), 1, K)
    if o.scal != CLS_NONE:
        slot[].put(DA_BLOCK_SCAL, _copy_out(done.arena.f32(o.scal), d * d), d, d)
        if components > 0:
            slot[].put(DA_BLOCK_EVR, _copy_out(done.arena.f32(o.evr), components), 1, components)
    if o.sw != CLS_NONE:
        slot[].put(DA_BLOCK_COV, _copy_out(done.arena.f32(o.sw), d * d), d, d)
    if K == 2:
        slot[].put(DA_BLOCK_PAIR_COEF, _copy_out(done.arena.f32(o.cd), d), 1, d)
        slot[].put(DA_BLOCK_PAIR_INTER, _copy_out(done.arena.f32(o.ci), 1), 1, 1)
    slot[].scalars = [Float64(d), Float64(K), Float64(rank), Float64(components)]
    slot[].features = d
    slot[].classes = K
    slot[].label_dtype = views.label_dtype
    slot[].fitted = True


def _fit_qda(slot: _Slot, call: _Call) raises:
    """QuadraticDiscriminantAnalysis.fit: the door's fit program
    (`_qda_fit`) over the caller's views, its count and rank refusals, the
    fitted blocks copied out of the arena into the slot."""
    var estimator = String("QuadraticDiscriminantAnalysis")
    var rows = _fit_rows(call, estimator)
    var n = rows.n
    var d = rows.d
    var K = call[].classes
    if K < 2:
        raise Error("mojolearn: QuadraticDiscriminantAnalysis needs at least two classes")
    var p = slot[].words.unsafe_ptr()
    var eigen = slot[].solver() == CLS_LDA_EIGEN
    var views = _fit_views(call, rows, estimator)
    var priors = _priors(slot, K, estimator)
    var program = ClsProgram(p[10] != 0.0)
    var o = cls_qda_fit_program(program, _mode(slot), n, d, K, eigen, p[6], p[2] != 0.0, p[3], priors, p[5] != 0.0)
    var sources = List[ClsBorrowed](capacity=2)
    sources.append(_x_source(call, rows))
    sources.append(_plain(views.codes_address, n))
    # THE PIN RULE: see `_fit_lda`.
    var done = cls_borrowed_run(_run_plan(call), program, sources, call[].scope, call[].pin_kept)
    _ = views.keep()
    var counts = done.arena.f32(o.cnt)
    for k in range(K):  # driver-loop: over the class counts (class count)
        if counts[k] < 2:
            raise Error("mojolearn: y has only 1 sample in a class, covariance is ill defined")
    var scalings = done.arena.f32(o.s2)
    for k in range(K):  # driver-loop: rank check per class before raising (class count)
        var above = 0
        for j in range(d):  # driver-loop: over one class's scalings (feature count)
            if Float64(scalings[k * d + j]) > p[4]:
                above += 1
        if above < d:
            raise Error("mojolearn: the covariance matrix of class code " + String(k) + " is not full rank. Increase the value of `"
                        + (String("shrinkage") if eigen else String("reg_param")) + "` to reduce the collinearity.")
    slot[].clear()
    slot[].put(DA_BLOCK_MEANS, _copy_out(done.arena.f32(o.mean), K * d), K, d)
    slot[].put(DA_BLOCK_PRIORS, _copy_out(done.arena.f32(o.priors), K), 1, K)
    slot[].put(DA_BLOCK_ROT, _copy_out(done.arena.f32(o.rot), K * d * d), K, d * d)
    slot[].put(DA_BLOCK_LOGC, _copy_out(done.arena.f32(o.logc), K), 1, K)
    slot[].put(DA_BLOCK_ROTATIONS, _copy_out(done.arena.f32(o.evec), K * d * d), K, d * d)
    slot[].put(DA_BLOCK_SCALINGS, _copy_out(scalings, K * d), K, d)
    if o.keep != CLS_NONE:
        slot[].put(DA_BLOCK_COV, _copy_out(done.arena.f32(o.keep), K * d * d), K, d * d)
    # `scalings` points into the arena: `done` is named after its last read
    # (without a kept covariance its last named use was the rotations' copy).
    _ = done^
    slot[].scalars = [Float64(d), Float64(K), Float64(d), 0.0]
    slot[].features = d
    slot[].classes = K
    slot[].label_dtype = views.label_dtype
    slot[].fitted = True


def _fitted_rows(slot: _Slot, call: _Call, estimator: String) raises -> _Rows:
    if not slot[].fitted:
        raise Error("NotFittedError: This " + estimator + " instance is not fitted yet. Call 'fit' with appropriate "
                    + "arguments before using this estimator.")
    var rows = _rows(call, estimator)
    if rows.d != slot[].features:
        raise Error("mojolearn: X has " + String(rows.d) + " features, but " + estimator + " was fitted with "
                    + String(slot[].features))
    if rows.n < 1:
        raise Error("ValueError: a " + estimator + " member reads at least one row")
    if not call[].dst.given() or call[].dst.home != 0 or call[].dst.stride != call[].dst.cols or call[].dst.offset != 0:
        raise Error("TypeError: a " + estimator + " member writes a dense host view")
    return rows


def _float_dst(call: _Call, n: Int, cols: Int, what: String) raises:
    if call[].dst.rows != n or call[].dst.cols != cols or call[].dst.dtype != MEMBER_F32:
        raise Error("ValueError: the " + what + " buffer holds " + String(cols) + " float32 values per row of X")


def _pair_decision(slot: _Slot, call: _Call, rows: _Rows) raises:
    """LDA's two-class decision function: the door's one-column program
    over the coefficient difference."""
    var n = rows.n
    var d = rows.d
    var p = slot[].words.unsafe_ptr()
    var program = ClsProgram(p[10] != 0.0)
    var dst = cls_lda_pair_program(program, n, d)
    var sources = List[ClsBorrowed](capacity=3)
    sources.append(_x_source(call, rows))
    sources.append(_plain(slot[].address(DA_BLOCK_PAIR_COEF), d))
    sources.append(_plain(slot[].address(DA_BLOCK_PAIR_INTER), 1))
    # THE PIN RULE: see `_fit_lda`.
    var done = cls_borrowed_run(_run_plan(call), program, sources, call[].scope, call[].pin_kept)
    unsafe_memcpy(dest=_F32P(unsafe_from_address=call[].dst.data), src=done.arena.f32(dst), count=n)


def _score(slot: _Slot, call: _Call, operation: Int) raises:
    """predict / predict_proba / decision_function: the door's scoring
    program (`_base_scores`, `_decision_function`) with the slot's blocks
    as its fitted records."""
    var kind = slot[].kind()
    var estimator = _kind_name(kind)
    var rows = _fitted_rows(slot, call, estimator)
    var n = rows.n
    var d = rows.d
    var K = slot[].classes
    var proba = operation == MEMBER_OP_PREDICT_PROBA
    var decision = operation == MEMBER_OP_DECISION
    var pair = decision and K == 2
    var p64 = proba and (call[].route.flags & NB_ROUTE_PROBA64) != 0
    if decision:
        _float_dst(call, n, 1 if pair else K, "decision")
    elif proba:
        if call[].dst.rows != n or call[].dst.cols != K or call[].dst.dtype != (MEMBER_F64 if p64 else MEMBER_F32):
            raise Error("ValueError: the probability buffer holds " + String(K) + (" float64" if p64 else " float32")
                        + " values per row of X")
    elif (call[].dst.rows != n or call[].dst.cols != 1
            or (call[].dst.dtype != MEMBER_I32 and call[].dst.dtype != MEMBER_I64)):
        raise Error("ValueError: the prediction buffer holds one int32 or int64 class code per row of X")
    if pair and kind == DA_KIND_LDA:
        _pair_decision(slot, call, rows)
        return
    var p = slot[].words.unsafe_ptr()
    # The class's fitted records in the order `cls_jll_stages` creates them.
    var a = _plain(slot[].address(DA_BLOCK_COEF), K * d)
    var b = _plain(slot[].address(DA_BLOCK_INTER), K)
    var c = _plain(0, 0)
    var program_kind = CLS_KIND_LDA
    if kind == DA_KIND_QDA:
        a = _plain(slot[].address(DA_BLOCK_MEANS), K * d)
        b = _plain(slot[].address(DA_BLOCK_ROT), K * d * d)
        c = _plain(slot[].address(DA_BLOCK_LOGC), K)
        program_kind = CLS_KIND_QDA
    var want = 0
    if proba:
        want = CLS_W_PROBA
    elif not decision:
        want = CLS_W_PREDICT
    var program = ClsProgram(p[10] != 0.0)
    var o = cls_score_program(program, program_kind, CLS_NONE, n, d, K, a.words, b.words, c.words, want, p64, pair,
                              False, False, 0.0, 0)
    var sources = List[ClsBorrowed](capacity=4)
    sources.append(_x_source(call, rows))
    sources.append(a)
    sources.append(b)
    if c.words > 0:
        sources.append(c)
    # THE PIN RULE: see `_fit_lda`.
    var done = cls_borrowed_run(_run_plan(call), program, sources, call[].scope, call[].pin_kept)
    if decision:
        # QDA's two-class answer is the n differences at `am`; else the scores
        unsafe_memcpy(dest=_F32P(unsafe_from_address=call[].dst.data), src=done.arena.f32(o.am if pair else o.jll),
                      count=n if pair else n * K)
        return
    if proba:
        # float64 probabilities are two arena words each
        unsafe_memcpy(dest=_F32P(unsafe_from_address=call[].dst.data), src=done.arena.f32(o.pp),
                      count=(2 if p64 else 1) * n * K)
        return
    # `found` points into the run's arena: `done` is named again after each
    # read (Mojo frees a local at its last named use; the arena is freed with
    # it). Proving suite 2026-11-22: a member's codes were read from the freed
    # arena and Voting / MultiOutput over a device="cpu" member refused
    # "code out of range".
    var found = _I32P(unsafe_from_address=done.arena.address + 4 * o.am)
    if call[].dst.dtype == MEMBER_I32:
        unsafe_memcpy(dest=_I32P(unsafe_from_address=call[].dst.data), src=found, count=n)
        _ = done^
        return
    # int64 output: the codes widened in chunks across the host's cores
    # (core/naive_bayes_member.mojo does the same), not one row at a time.
    host_widen_index_i64(Int(found), call[].dst.data, n)
    _ = done^


def _transform_width(slot: _Slot) -> Int:
    return min(Int(slot[].scalars[3]), Int(slot[].scalars[2]))


def _transform(slot: _Slot, call: _Call) raises:
    """LinearDiscriminantAnalysis.transform: the door's program
    (`_lda_transform`) with xbar_ and the scalings as its records."""
    var estimator = String("LinearDiscriminantAnalysis")
    if slot[].kind() != DA_KIND_LDA:
        member_unsupported(_kind_name(slot[].kind()), MEMBER_OP_TRANSFORM)
    var rows = _fitted_rows(slot, call, estimator)
    if slot[].solver() == CLS_LDA_LSQR:
        raise Error("NotImplementedError: mojolearn: transform not implemented for 'lsqr' solver (use 'svd' or 'eigen').")
    var n = rows.n
    var d = rows.d
    var mc = _transform_width(slot)
    if mc < 1 or slot[].size(DA_BLOCK_SCAL) != d * d or slot[].size(DA_BLOCK_XBAR) != d:
        raise Error("ValueError: this LinearDiscriminantAnalysis member holds no component to project on")
    _float_dst(call, n, mc, "transform")
    var p = slot[].words.unsafe_ptr()
    var eigen = slot[].solver() == CLS_LDA_EIGEN
    var program = ClsProgram(p[10] != 0.0)
    var dst = cls_lda_transform_program(program, n, d, mc, eigen, not eigen and p[11] != 0.0)
    var sources = List[ClsBorrowed](capacity=3)
    sources.append(_x_source(call, rows))
    sources.append(_plain(slot[].address(DA_BLOCK_XBAR), d))
    sources.append(_plain(slot[].address(DA_BLOCK_SCAL), d * d))
    # THE PIN RULE: see `_fit_lda`.
    var done = cls_borrowed_run(_run_plan(call), program, sources, call[].scope, call[].pin_kept)
    unsafe_memcpy(dest=_F32P(unsafe_from_address=call[].dst.data), src=done.arena.f32(dst), count=n * mc)


def _export(slot: _Slot, call: _Call) raises:
    """A view of one fitted block (empty when the state has none). The slot
    owns the words; they stay valid until this state's next fit, import or
    free."""
    var estimator = _kind_name(slot[].kind())
    if not slot[].fitted:
        raise Error("NotFittedError: this " + estimator + " member is not fitted yet")
    var block = call[].block
    if (call[].flags & MEMBER_BLOCK_BY_INDEX) != 0:
        if block < 0 or block > DA_BLOCKS:
            raise Error("member ABI: a " + estimator + " member has no fitted block index " + String(block))
        block = MEMBER_BLOCK_SCALARS if block == 0 else MEMBER_BLOCK_FAMILY + block - 1
    if block == MEMBER_BLOCK_COEF or block == MEMBER_BLOCK_INTERCEPT:
        # LDA's PUBLIC coef_ / intercept_, the words its fit door publishes
        # (bindings/prep_classify_native_door.mojo `_lda_publish_coef`): the
        # pair rows (1 x features, 1 x 1) with two classes, else `_coef`
        # (classes x features) and `_inter` (1 x classes). By name only (never
        # in a MEMBER_BLOCK_BY_INDEX walk: the same words are blocks 4/5 and
        # 9/10 there). QDA has neither.
        if slot[].kind() != DA_KIND_LDA:
            raise Error("member ABI: a " + estimator + " member has no fitted block " + String(block))
        var pair = slot[].classes == 2
        var family_block = DA_BLOCK_COEF if block == MEMBER_BLOCK_COEF else DA_BLOCK_INTER
        if pair:
            family_block = DA_BLOCK_PAIR_COEF if block == MEMBER_BLOCK_COEF else DA_BLOCK_PAIR_INTER
        var index = family_block - MEMBER_BLOCK_FAMILY
        call[].dst = MemberMatrix(slot[].address(family_block), slot[].block_rows[index], slot[].block_cols[index])
    elif block == MEMBER_BLOCK_SCALARS:
        call[].dst = MemberMatrix(Int(slot[].scalars.unsafe_ptr()), 1, len(slot[].scalars), MEMBER_F64)
    elif block >= MEMBER_BLOCK_FAMILY and block < MEMBER_BLOCK_FAMILY + DA_BLOCKS:
        var index = block - MEMBER_BLOCK_FAMILY
        if len(slot[].blocks[index]) == 0:
            call[].dst = MemberMatrix(0, 0, 0)
        else:
            call[].dst = MemberMatrix(slot[].address(block), slot[].block_rows[index], slot[].block_cols[index])
    else:
        raise Error("member ABI: a " + estimator + " member has no fitted block " + String(block))
    call[].block = block
    call[].out_cols = DA_BLOCKS + 1


def _holds(slot: _Slot, block: Int, words: Int) -> Bool:
    return slot[].size(block) == words


def _import(slot: _Slot, call: _Call) raises:
    """Copy one fitted block in; MEMBER_IMPORT_DONE completes the state."""
    var kind = slot[].kind()
    var estimator = _kind_name(kind)
    var block = call[].block
    if slot[].fitted:
        slot[].clear()
    if block == MEMBER_BLOCK_SCALARS:
        if not call[].x.dense_host(MEMBER_F64) or call[].x.rows * call[].x.cols != DA_SCALARS:
            raise Error("member ABI: the " + estimator + " scalars block is four host float64 words "
                        + "(n_features_in_, classes, rank, max components)")
        var given = _F64P(unsafe_from_address=call[].x.data)
        var scalars = List[Float64](capacity=DA_SCALARS)
        for i in range(DA_SCALARS):  # driver-loop: over the four scalar words
            scalars.append(given[i])
        slot[].scalars = scalars^
    elif block >= MEMBER_BLOCK_FAMILY and block < MEMBER_BLOCK_FAMILY + DA_BLOCKS:
        var size = call[].x.rows * call[].x.cols
        if size < 1 or not call[].x.dense_host(MEMBER_F32):
            raise Error("TypeError: a " + estimator + " member reads a fitted block as a dense host float32 view")
        slot[].put(block, _copy_out(_F32P(unsafe_from_address=call[].x.data), size), call[].x.rows, call[].x.cols)
    elif block != 0 or (call[].flags & MEMBER_IMPORT_DONE) == 0:
        raise Error("member ABI: a " + estimator + " member has no fitted block " + String(block))
    if (call[].flags & MEMBER_IMPORT_DONE) == 0:
        return
    if len(slot[].scalars) != DA_SCALARS:
        raise Error("member ABI: a fitted " + estimator + " state needs its scalars block")
    var d = Int(slot[].scalars[0])
    var K = Int(slot[].scalars[1])
    var whole = d >= 1 and K >= 2
    if kind == DA_KIND_LDA:
        whole = whole and _holds(slot, DA_BLOCK_COEF, K * d) and _holds(slot, DA_BLOCK_INTER, K)
        if K == 2:
            whole = whole and _holds(slot, DA_BLOCK_PAIR_COEF, d) and _holds(slot, DA_BLOCK_PAIR_INTER, 1)
        # the transform's records, when the state has them
        whole = whole and (slot[].size(DA_BLOCK_SCAL) == 0 or _holds(slot, DA_BLOCK_SCAL, d * d))
        whole = whole and (slot[].size(DA_BLOCK_XBAR) == 0 or _holds(slot, DA_BLOCK_XBAR, d))
    else:
        whole = (whole and _holds(slot, DA_BLOCK_MEANS, K * d) and _holds(slot, DA_BLOCK_ROT, K * d * d)
                 and _holds(slot, DA_BLOCK_LOGC, K))
    if not whole:
        raise Error("member ABI: a fitted " + estimator + " state needs its scoring blocks, of the fitted sizes")
    slot[].features = d
    slot[].classes = K
    slot[].fitted = True


def _outputs(call: _Call) raises:
    """`call.out_cols`, `call.out_dtype` and `call.classes` of predict,
    predict_proba, decision_function and transform, with a state or without
    one (the NEW words in `call.params`, `call.classes`, `call.x.cols`)."""
    var asked = call[].flags
    var classes = call[].classes
    var label_dtype = MEMBER_I64
    var kind = DA_KIND_LDA
    var solver = CLS_LDA_SVD
    var components = -1
    var fitted = False
    var width = 0
    if call[].state != 0:
        var slot = _Slot(unsafe_from_address=_lookup(call[].state, False))
        kind = slot[].kind()
        solver = slot[].solver()
        components = Int(slot[].words[6]) if kind == DA_KIND_LDA else -1
        if slot[].fitted:
            fitted = True
            classes = slot[].classes
            label_dtype = slot[].label_dtype
            width = _transform_width(slot)
    elif call[].param_count >= DA_PARAM_HEAD and call[].params:
        var source = _F64P(unsafe_from_address=call[].params)
        kind = Int(source[0])
        solver = Int(source[1])
        components = Int(source[6]) if kind == DA_KIND_LDA else -1
    var estimator = _kind_name(kind)
    if asked == MEMBER_OP_PREDICT:
        call[].out_cols = 1
        call[].out_dtype = label_dtype
    elif asked == MEMBER_OP_PREDICT_PROBA:
        call[].out_cols = classes
        call[].out_dtype = MEMBER_F64 if (call[].route.flags & NB_ROUTE_PROBA64) != 0 else MEMBER_F32
    elif asked == MEMBER_OP_DECISION:
        call[].out_cols = 1 if classes == 2 else classes
        call[].out_dtype = MEMBER_F32
    elif asked == MEMBER_OP_TRANSFORM and kind == DA_KIND_LDA and solver != CLS_LDA_LSQR:
        if fitted:
            call[].out_cols = width
        elif solver == CLS_LDA_EIGEN and classes >= 2 and call[].x.cols >= 1:
            # 'eigen' keeps every direction: the width is the parameters'
            call[].out_cols = min(classes - 1, call[].x.cols) if components < 0 else components
        else:
            raise Error("member ABI: the width of a LinearDiscriminantAnalysis transform is chosen by its fit "
                        + "(solver 'svd': the rank), or needs the class and feature counts ('eigen')")
        call[].out_dtype = MEMBER_F32
    else:
        member_unsupported(estimator, asked)
    call[].classes = classes


def _new(call: _Call) raises:
    """MEMBER_OP_NEW: the parameter words checked (kind, solver, counts)
    and kept in a new state."""
    if call[].param_count < DA_PARAM_HEAD or not call[].params:
        raise Error("member ABI: a Discriminant member takes 13 parameter words and its priors")
    var source = _F64P(unsafe_from_address=call[].params)
    var kind = Int(source[0])
    if Float64(kind) != source[0] or not discriminant_kind_published(kind):
        raise Error("member ABI: Discriminant member kind " + String(source[0]) + " is not published")
    var estimator = _kind_name(kind)
    var prior_count = Int(source[12])
    if Float64(prior_count) != source[12] or prior_count < 0 or call[].param_count != DA_PARAM_HEAD + prior_count:
        raise Error("member ABI: a " + estimator + " member takes 13 parameter words and its priors")
    var solver = Int(source[1])
    var known = solver == CLS_LDA_SVD or solver == CLS_LDA_EIGEN or (solver == CLS_LDA_LSQR and kind == DA_KIND_LDA)
    if Float64(solver) != source[1] or not known:
        raise Error("member ABI: " + estimator + " solver word " + String(source[1]) + " is not a solver")
    if solver == CLS_LDA_SVD and source[2] != 0.0:
        raise Error("NotImplementedError: mojolearn: shrinkage not supported with 'svd' solver.")
    var words = List[Float64](length=call[].param_count, fill=0.0)
    for index in range(call[].param_count):  # driver-loop: the parameter words
        words[index] = source[index]
    call[].state = _mint(words^)


def _discriminant_member(operation: Int, address: Int) raises -> Int:
    var call = _Call(unsafe_from_address=address)
    if operation == MEMBER_OP_CAPABILITIES:
        # Kind-exact with a state: only LDA with solver 'svd' or 'eigen'
        # transforms; LDA (every solver) exports MEMBER_BLOCK_COEF (its
        # public coef_, lane mf-shaprfe). QDA has no coef_ (scikit-learn's
        # neither). Without a state, the family's union.
        var mask = _family_capabilities(call[].route.flags)
        if call[].state == 0:
            return mask | MEMBER_CAN_TRANSFORM | MEMBER_CAN_COEF
        var asked = _Slot(unsafe_from_address=_lookup(call[].state, False))
        if asked[].kind() == DA_KIND_LDA:
            mask |= MEMBER_CAN_COEF
            if asked[].solver() != CLS_LDA_LSQR:
                mask |= MEMBER_CAN_TRANSFORM
        return mask
    if operation == MEMBER_OP_NEW:
        _new(call)
        return 0
    if operation == MEMBER_OP_FREE:
        _ = _lookup(call[].state, True)
        call[].state = 0
        return 0
    if operation == MEMBER_OP_OUTPUTS:
        _outputs(call)
        return 0
    var slot = _Slot(unsafe_from_address=_lookup(call[].state, False))
    # A host int32 row index: widened once (core/naive_bayes_member.mojo
    # `nb_wide_index`), the caller's view restored after.
    var given = call[].row_index
    var wide = List[Int64]()
    var widened = nb_wide_index(call, wide)
    try:
        if operation == MEMBER_OP_FIT:
            if slot[].kind() == DA_KIND_LDA:
                _fit_lda(slot, call)
            else:
                _fit_qda(slot, call)
        elif (operation == MEMBER_OP_PREDICT or operation == MEMBER_OP_PREDICT_PROBA
                or operation == MEMBER_OP_DECISION):
            _score(slot, call, operation)
        elif operation == MEMBER_OP_TRANSFORM:
            _transform(slot, call)
        elif operation == MEMBER_OP_EXPORT:
            _export(slot, call)
        elif operation == MEMBER_OP_IMPORT:
            _import(slot, call)
        else:
            member_unsupported(_kind_name(slot[].kind()), operation)
    finally:
        if widened:
            call[].row_index = given
    _ = wide^
    return 0


def discriminant_member_v2(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
    result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """The member entry of the Discriminant family (schema `MEMBER_ABI`)."""
    try:
        var word = _discriminant_member(operation, member_call_from_spans(operation, aa, na, np, nf))
        if result:
            MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = word
        return 0
    except error:
        native_descriptor_report(error, context, sink)
        return 1


def discriminant_member_publish(mode: Int, vendor: Int, publisher: Int) raises:
    """Publish this image's Discriminant member entry. One line in the
    image's module initialization, after its program transport is published."""
    member_publish[MEMBER_FAMILY_DISCRIMINANT](mode, vendor, publisher, discriminant_member_v2)
