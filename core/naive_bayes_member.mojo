# SPDX-License-Identifier: Apache-2.0
"""The NaiveBayes family as MEMBERS (core/member_abi.mojo): GaussianNB
(kind 0) so far; MultinomialNB, BernoulliNB, ComplementNB and CategoricalNB
are not published yet.

One entry, `naive_bayes_member_v2`, published by each x_prep image beside its
program transport (`naive_bayes_member_publish`, x_prep/transport_device.mojo
and x_prep/transport_host.mojo). A member runs the estimator's own x_prep
programs: the typed builders the door builds through
(bindings/prep_classify_prog_run.mojo `cls_gnb_fit_program`,
`cls_score_program`) and the typed run the door runs
(`cls_borrowed_run`: `cls_run_route`, `cls_input_choice`, `cls_prog_run`) over
the transport entry of the `DeviceRoute` it is handed. Same programs, same
stages, same order as the door: no bit moves. No kernel is touched.

Route slots read: `entry` the x_prep transport entry of the image; `flags`
NB_ROUTE_OUT (x_prep_run_out), NB_ROUTE_SCRATCH (x_prep_run_scratch),
NB_ROUTE_RANGES (x_prep_run_ranges), NB_ROUTE_DEV_STORE (x_prep_dev_put and
x_prep_dev_free), NB_ROUTE_TAKE_ROWS (x_prep_dev_take_rows),
NB_ROUTE_PROBA64 (the image exports x_prep_proba64), NB_ROUTE_CLASS_ONEPASS
(`x_prep_idn_fam` & IDN_CLASS_ONEPASS of an identical image).

MEMBER_OP_NEW parameters, GaussianNB (Float64 words, 5 + P):
  [0] kind: 0 GaussianNB
  [1] var_smoothing (checked at admission)
  [2] 2048-row statistics blocks 0 / 1 (MOJOLEARN_XPREP_BLOCKED)
  [3] R3 work region 0 / 1 (`xprep_r3("work")`)
  [4] P: the count of the `priors` parameter (0: priors=None)
  [5 .. 5 + P) the priors (non-negative, summing to one: checked at
      admission; P must equal the class count at fit)

Views: X dense row-major host float32 (finite: the caller's admission
tests it); `y` the class CODES 0..classes-1, one host column, int32 (read
in place) or int64; `call.classes` the class count at FIT (at least one);
sample weights one host float32 column (`MEMBER_CAN_SAMPLE_WEIGHT`). PREDICT
writes codes, one column, int32 or int64 as `dst` says. PREDICT_PROBA
writes rows x classes, float64 on a route with NB_ROUTE_PROBA64, else
float32 (`MEMBER_OP_OUTPUTS` answers which). No score, no decision function,
no partial fit, no column index, no device-home view.

`call.row_index` (`MEMBER_CAN_ROW_INDEX`, answered only on a route that can
gather: NB_ROUTE_RANGES and NB_ROUTE_TAKE_ROWS with device scratch): host
int64 row numbers; X is then the WHOLE base and its rows are taken on the
device (the door's DeviceRows gather); y and the weights are the base's
columns, read through the index. `call.scope` / `call.pin`
(`MEMBER_CAN_SCOPE`, answered on a route with NB_ROUTE_DEV_STORE and
NB_ROUTE_RANGES): with a scope and `pin != 0`, X (the whole base under a
row index) stays resident in the scope's x_prep cache, the door's cache, so
the next call of any member or door over the same buffer is one lookup. When
the cache took the pin's reference the member writes `call.pin_kept`
nonzero at that moment (THE PIN RULE, core/member_abi.mojo); the caller
keeps and later drops every reference no cache took.

Fitted-state blocks (MEMBER_OP_EXPORT / MEMBER_OP_IMPORT), GaussianNB:
  0 `MEMBER_BLOCK_SCALARS` float64 [n_features_in_, classes, epsilon_]
  1 `NB_BLOCK_THETA` (MEMBER_BLOCK_FAMILY) float32 classes x features (theta_)
  2 `NB_BLOCK_VAR` float32 classes x features (var_)
  3 `NB_BLOCK_CONST` float32 1 x classes (the per-class constant `_const`)
  4 `NB_BLOCK_CLASS_COUNT` float32 1 x classes (class_count_)
  5 `NB_BLOCK_CLASS_PRIOR` float32 1 x classes (class_prior_)
  6 `NB_BLOCK_RAW_VAR` float32 classes x features (the unsmoothed variances `_raw_var`)

The fitted state is OWNED here, in a process-wide store of fixed chunks (the
linear member's store): a slot never moves once minted, the lock covers only
mint, free and the handle lookup. Serial by nature here: the store's
bookkeeping, and the two label passes of an indexed or int64 `y` (one word
per row, beside a program that reads every word of X); the programs are the
family's and run on the route's device.
"""
from std.sys.info import size_of
from std.memory import unsafe_memcpy
from bindings.prep_classify_prog_run import (
    ClsProgram, ClsRunPlan, ClsBorrowed, CLS_NONE, CLS_KIND_GNB, CLS_W_PREDICT, CLS_W_PROBA,
    cls_gnb_fit_program, cls_score_program, cls_borrowed_run,
)
from core.member_abi import (
    MemberCall, MemberMatrix, MEMBER_F32, MEMBER_F64, MEMBER_I32, MEMBER_I64,
    MEMBER_OP_CAPABILITIES, MEMBER_OP_NEW, MEMBER_OP_FREE, MEMBER_OP_FIT, MEMBER_OP_PREDICT,
    MEMBER_OP_PREDICT_PROBA, MEMBER_OP_OUTPUTS, MEMBER_OP_EXPORT, MEMBER_OP_IMPORT,
    MEMBER_CAN_FIT, MEMBER_CAN_PREDICT, MEMBER_CAN_PREDICT_PROBA, MEMBER_CAN_EXPORT, MEMBER_CAN_IMPORT,
    MEMBER_CAN_SAMPLE_WEIGHT, MEMBER_CAN_CLASSIFIER, MEMBER_CAN_ROW_INDEX, MEMBER_CAN_SCOPE,
    MEMBER_BLOCK_SCALARS, MEMBER_BLOCK_FAMILY, MEMBER_BLOCK_BY_INDEX, MEMBER_IMPORT_DONE,
    member_publish, member_call_from_spans, member_unsupported,
)
from core.member_families import MEMBER_FAMILY_NAIVE_BAYES
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.shared_slot import SharedSlot, shared_slot_lock, shared_slot_unlock

comptime NB_KIND_GAUSSIAN = 0
#: GaussianNB's parameter words before its priors.
comptime NB_GAUSSIAN_PARAM_HEAD = 5

comptime NB_BLOCK_THETA = MEMBER_BLOCK_FAMILY
comptime NB_BLOCK_VAR = MEMBER_BLOCK_FAMILY + 1
comptime NB_BLOCK_CONST = MEMBER_BLOCK_FAMILY + 2
comptime NB_BLOCK_CLASS_COUNT = MEMBER_BLOCK_FAMILY + 3
comptime NB_BLOCK_CLASS_PRIOR = MEMBER_BLOCK_FAMILY + 4
comptime NB_BLOCK_RAW_VAR = MEMBER_BLOCK_FAMILY + 5
comptime NB_GAUSSIAN_BLOCKS = 7
comptime NB_GAUSSIAN_SCALARS = 3

comptime NB_ROUTE_OUT = 1
comptime NB_ROUTE_SCRATCH = 2
comptime NB_ROUTE_RANGES = 4
comptime NB_ROUTE_DEV_STORE = 8
comptime NB_ROUTE_TAKE_ROWS = 16
comptime NB_ROUTE_PROBA64 = 32
comptime NB_ROUTE_CLASS_ONEPASS = 64

comptime _CHUNK = 256
comptime _INDEX_SPAN = 16777216


struct NaiveBayesMemberSlot(Copyable, Movable, Defaultable):
    var generation: Int
    var live: Bool
    var words: List[Float64]
    var fitted: Bool
    var features: Int
    var classes: Int
    #: The element type `y` had at fit (the type predict answers by default).
    var label_dtype: Int
    var scalars: List[Float64]
    var theta: List[Float32]
    var variance: List[Float32]
    var constant: List[Float32]
    var class_count: List[Float32]
    var class_prior: List[Float32]
    var raw_variance: List[Float32]

    def __init__(out self):
        self.generation = 0
        self.live = False
        self.words = List[Float64]()
        self.fitted = False
        self.features = 0
        self.classes = 0
        self.label_dtype = MEMBER_I64
        self.scalars = List[Float64]()
        self.theta = List[Float32]()
        self.variance = List[Float32]()
        self.constant = List[Float32]()
        self.class_count = List[Float32]()
        self.class_prior = List[Float32]()
        self.raw_variance = List[Float32]()

    def kind(self) -> Int:
        return Int(self.words[0])

    def clear(mut self):
        self.fitted = False
        self.features = 0
        self.classes = 0
        self.scalars = List[Float64]()
        self.theta = List[Float32]()
        self.variance = List[Float32]()
        self.constant = List[Float32]()
        self.class_count = List[Float32]()
        self.class_prior = List[Float32]()
        self.raw_variance = List[Float32]()


struct NaiveBayesMemberStore(Defaultable, Movable):
    var synchronization: List[Int64]
    var chunks: List[List[NaiveBayesMemberSlot]]
    var free: List[Int]
    var count: Int

    def __init__(out self):
        # Side-effect-free even when the process slot's initializers race.
        self.synchronization = List[Int64](length=2, fill=0)
        self.chunks = List[List[NaiveBayesMemberSlot]]()
        self.free = List[Int]()
        self.count = 0


comptime _STORE = SharedSlot[NaiveBayesMemberStore, "MojolearnNaiveBayesMemberStoreV2", 2]
comptime _Slot = MutPointer[NaiveBayesMemberSlot, MutUntrackedOrigin]
comptime _Call = MutPointer[MemberCall, MutUntrackedOrigin]
comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _F64P = MutPointer[Float64, MutUntrackedOrigin]
comptime _I32P = MutPointer[Int32, MutUntrackedOrigin]
comptime _I64P = MutPointer[Int64, MutUntrackedOrigin]


def _slot_address(index: Int) raises -> Int:
    # Under the store's lock. A chunk's buffer is allocated once and never
    # grows, so the address stays valid after the lock is released.
    var store = _STORE.get_or_create_ptr()
    return Int(store[].chunks[index // _CHUNK].unsafe_ptr()) + (index % _CHUNK) * size_of[NaiveBayesMemberSlot]()


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
                raise Error("RuntimeError: mojolearn: too many live NaiveBayes member states")
            if store[].count == len(store[].chunks) * _CHUNK:
                store[].chunks.append(List[NaiveBayesMemberSlot](length=_CHUNK, fill=NaiveBayesMemberSlot()))
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
    if kind == NB_KIND_GAUSSIAN:
        return String("GaussianNB")
    return String("NaiveBayes")


def _run_plan(call: _Call) -> ClsRunPlan:
    # The image as the typed run reads it, from the route's words.
    var flags = call[].route.flags
    var dev_store = (flags & NB_ROUTE_DEV_STORE) != 0
    return ClsRunPlan(call[].route.entry, (flags & NB_ROUTE_OUT) != 0, (flags & NB_ROUTE_SCRATCH) != 0,
                      (flags & NB_ROUTE_RANGES) != 0, dev_store, dev_store and (flags & NB_ROUTE_TAKE_ROWS) != 0)


def _route_gathers(flags: Int) -> Bool:
    # Every program of this family gathers on such a route: device scratch
    # makes the range runner the route of a program with scratch words too.
    return ((flags & NB_ROUTE_RANGES) != 0 and (flags & NB_ROUTE_DEV_STORE) != 0
            and (flags & NB_ROUTE_TAKE_ROWS) != 0 and (flags & (NB_ROUTE_OUT | NB_ROUTE_SCRATCH)) != 0)


def _capabilities(flags: Int) -> Int:
    var mask = (MEMBER_CAN_FIT | MEMBER_CAN_PREDICT | MEMBER_CAN_PREDICT_PROBA | MEMBER_CAN_EXPORT
                | MEMBER_CAN_IMPORT | MEMBER_CAN_SAMPLE_WEIGHT | MEMBER_CAN_CLASSIFIER)
    if (flags & NB_ROUTE_DEV_STORE) != 0 and (flags & NB_ROUTE_RANGES) != 0:
        mask |= MEMBER_CAN_SCOPE
    if _route_gathers(flags):
        mask |= MEMBER_CAN_ROW_INDEX
    return mask


struct _Rows(ImplicitlyCopyable, Movable):
    """X as one operation reads it: `n` rows of `d` float32 words, the whole
    buffer (`base_rows` rows) and, under a row index, the int64 row numbers."""
    var n: Int
    var d: Int
    var base_rows: Int
    var index: Int

    def __init__(out self, n: Int, d: Int, base_rows: Int, index: Int):
        self.n = n
        self.d = d
        self.base_rows = base_rows
        self.index = index


def _rows(call: _Call, estimator: String) raises -> _Rows:
    if call[].column_index.rows > 0:
        raise Error("TypeError: a " + estimator + " member reads every column (no column index)")
    if not call[].x.dense_host(MEMBER_F32):
        raise Error("TypeError: a " + estimator + " member reads X as a dense host float32 view")
    var base_rows = call[].x.rows
    var d = call[].x.cols
    if call[].row_index.rows == 0:
        return _Rows(base_rows, d, base_rows, 0)
    if not call[].row_index.dense_host(MEMBER_I64) or call[].row_index.cols != 1:
        raise Error("TypeError: a " + estimator + " member reads its row index as one host int64 column")
    if not _route_gathers(call[].route.flags):
        raise Error("TypeError: a " + estimator + " member takes a row index only on a device that gathers rows")
    return _Rows(call[].row_index.rows, d, base_rows, call[].row_index.data)


def _x_source(call: _Call, rows: _Rows) -> ClsBorrowed:
    return ClsBorrowed(call[].x.data, rows.base_rows * rows.d, call[].pin, rows.index,
                       rows.n if rows.index != 0 else 0, rows.d)


def _plain(address: Int, words: Int) -> ClsBorrowed:
    return ClsBorrowed(address, words, 0, 0, 0, 0)


def _copy_out(source: _F32P, count: Int) -> List[Float32]:
    var words = List[Float32](length=count, fill=Float32(0))
    if count > 0:
        unsafe_memcpy(dest=words.unsafe_ptr(), src=source, count=count)
    return words^


def _fit_gaussian(slot: _Slot, call: _Call) raises:
    """GaussianNB.fit: the door's fit program (`_gnb_fit`) over the caller's
    views; the fitted blocks are copied out of the arena into the slot."""
    var estimator = String("GaussianNB")
    if call[].fit_output != 0:
        member_unsupported(estimator, call[].fit_output)
    var rows = _rows(call, estimator)
    var n = rows.n
    var d = rows.d
    var K = call[].classes
    if n < 1 or d < 1:
        raise Error("ValueError: a GaussianNB member fits at least one row and one feature")
    if K < 1:
        raise Error("ValueError: a GaussianNB member fits at least one class (call.classes)")
    var label_dtype = call[].y.dtype
    if (not (call[].y.dense_host(MEMBER_I32) or call[].y.dense_host(MEMBER_I64)) or call[].y.cols != 1
            or call[].y.rows != rows.base_rows):
        raise Error("TypeError: a GaussianNB member reads y as one host int32 or int64 column of class codes, "
                    + "one per row of X")
    var weighted = call[].weights.rows > 0
    if weighted and (not call[].weights.dense_host(MEMBER_F32) or call[].weights.cols != 1
                     or call[].weights.rows != rows.base_rows):
        raise Error("TypeError: a GaussianNB member reads sample weights as one host float32 column, one per row of X")
    var p = slot[].words.unsafe_ptr()
    var prior_count = Int(p[4])
    if prior_count != 0 and prior_count != K:
        raise Error("mojolearn: GaussianNB: number of priors must match number of classes")
    var priors = List[Float64](capacity=prior_count)
    for i in range(prior_count):  # driver-loop: over the user prior list (class-count sized)
        priors.append(p[NB_GAUSSIAN_PARAM_HEAD + i])
    # The class codes as the program's int32 words, in row order: read in
    # place, or narrowed / taken through the row index (one word per row).
    var codes = List[Int32]()
    var codes_address = call[].y.data
    if rows.index != 0 or label_dtype == MEMBER_I64:
        codes = List[Int32](length=n, fill=Int32(0))
        var index = _I64P(unsafe_from_address=rows.index if rows.index != 0 else call[].y.data)
        if label_dtype == MEMBER_I64:
            var wide = _I64P(unsafe_from_address=call[].y.data)
            for i in range(n):  # driver-loop: one label word per row (see the module docstring)
                codes[i] = Int32(wide[Int(index[i]) if rows.index != 0 else i])
        else:
            var narrow = _I32P(unsafe_from_address=call[].y.data)
            for i in range(n):  # driver-loop: one label word per row (see the module docstring)
                codes[i] = narrow[Int(index[i])]
        codes_address = Int(codes.unsafe_ptr())
    var taken = List[Float32]()
    var weights_address = call[].weights.data
    if weighted and rows.index != 0:
        taken = List[Float32](length=n, fill=Float32(0))
        var index = _I64P(unsafe_from_address=rows.index)
        var given = _F32P(unsafe_from_address=call[].weights.data)
        for i in range(n):  # driver-loop: one weight word per row (see the module docstring)
            taken[i] = given[Int(index[i])]
        weights_address = Int(taken.unsafe_ptr())
    var blocked = p[2] != 0.0
    var onepass = blocked and (call[].route.flags & NB_ROUTE_CLASS_ONEPASS) != 0
    var program = ClsProgram(p[3] != 0.0)
    var o = cls_gnb_fit_program(program, n, d, K, weighted, p[1], prior_count != 0, priors, blocked, onepass)
    var sources = List[ClsBorrowed](capacity=3)
    sources.append(_x_source(call, rows))
    sources.append(_plain(codes_address, n))
    if weighted:
        sources.append(_plain(weights_address, n))
    # THE PIN RULE (core/member_abi.mojo `MemberCall.pin`): the run writes
    # `call.pin_kept` at the moment the scope's cache takes the caller's
    # reference, so the caller reads it even when a later stage raises.
    var done = cls_borrowed_run(_run_plan(call), program, sources, call[].scope, call[].pin_kept)
    # The addressed words stay live through the run.
    _ = len(codes)
    _ = len(taken)
    slot[].clear()
    slot[].theta = _copy_out(done.arena.f32(o.theta), K * d)
    slot[].variance = _copy_out(done.arena.f32(o.var_), K * d)
    slot[].constant = _copy_out(done.arena.f32(o.cst), K)
    slot[].class_count = _copy_out(done.arena.f32(o.cnt), K)
    slot[].class_prior = _copy_out(done.arena.f32(o.prior), K)
    slot[].raw_variance = _copy_out(done.arena.f32(o.raw), K * d)
    slot[].scalars = [Float64(d), Float64(K), Float64(done.arena.f32(o.eps)[0])]
    slot[].features = d
    slot[].classes = K
    slot[].label_dtype = label_dtype
    slot[].fitted = True


def _score_gaussian(slot: _Slot, call: _Call, operation: Int) raises:
    """GaussianNB predict / predict_proba: the door's scoring program
    (`_base_scores`) with the slot's blocks as its fitted records."""
    var estimator = String("GaussianNB")
    if not slot[].fitted:
        raise Error("NotFittedError: This GaussianNB instance is not fitted yet. Call 'fit' with appropriate "
                    + "arguments before using this estimator.")
    var rows = _rows(call, estimator)
    var n = rows.n
    var d = slot[].features
    var K = slot[].classes
    if rows.d != d:
        raise Error("mojolearn: X has " + String(rows.d) + " features, but GaussianNB was fitted with " + String(d))
    if n < 1:
        raise Error("ValueError: a GaussianNB member scores at least one row")
    var proba = operation == MEMBER_OP_PREDICT_PROBA
    var p64 = proba and (call[].route.flags & NB_ROUTE_PROBA64) != 0
    if not call[].dst.given() or call[].dst.home != 0 or call[].dst.stride != call[].dst.cols or call[].dst.offset != 0:
        raise Error("TypeError: a GaussianNB member writes a dense host view")
    if proba:
        if call[].dst.rows != n or call[].dst.cols != K or call[].dst.dtype != (MEMBER_F64 if p64 else MEMBER_F32):
            raise Error("ValueError: the probability buffer holds " + String(K) + (" float64" if p64 else " float32")
                        + " values per row of X")
    elif (call[].dst.rows != n or call[].dst.cols != 1
            or (call[].dst.dtype != MEMBER_I32 and call[].dst.dtype != MEMBER_I64)):
        raise Error("ValueError: the prediction buffer holds one int32 or int64 class code per row of X")
    var p = slot[].words.unsafe_ptr()
    var program = ClsProgram(p[3] != 0.0)
    var o = cls_score_program(program, CLS_KIND_GNB, CLS_NONE, n, d, K, K * d, K * d, K,
                              CLS_W_PROBA if proba else CLS_W_PREDICT, p64, False, False, False, 0.0, 0)
    var sources = List[ClsBorrowed](capacity=4)
    sources.append(_x_source(call, rows))
    sources.append(_plain(Int(slot[].theta.unsafe_ptr()), K * d))
    sources.append(_plain(Int(slot[].variance.unsafe_ptr()), K * d))
    sources.append(_plain(Int(slot[].constant.unsafe_ptr()), K))
    # THE PIN RULE (core/member_abi.mojo `MemberCall.pin`): the run writes
    # `call.pin_kept` at the moment the scope's cache takes the caller's
    # reference, so the caller reads it even when a later stage raises.
    var done = cls_borrowed_run(_run_plan(call), program, sources, call[].scope, call[].pin_kept)
    if proba:
        # float64 probabilities are two arena words each
        unsafe_memcpy(dest=_F32P(unsafe_from_address=call[].dst.data), src=done.arena.f32(o.pp),
                      count=(2 if p64 else 1) * n * K)
        return
    var found = _I32P(unsafe_from_address=done.arena.address + 4 * o.am)
    if call[].dst.dtype == MEMBER_I32:
        unsafe_memcpy(dest=_I32P(unsafe_from_address=call[].dst.data), src=found, count=n)
        return
    var wide = _I64P(unsafe_from_address=call[].dst.data)
    for i in range(n):  # driver-loop: one class code per row, widened for an int64 buffer
        wide[i] = Int64(found[i])


def _block_view(slot: _Slot, block: Int) raises -> MemberMatrix:
    var K = slot[].classes
    var d = slot[].features
    if block == NB_BLOCK_THETA:
        return MemberMatrix(Int(slot[].theta.unsafe_ptr()), K, d)
    if block == NB_BLOCK_VAR:
        return MemberMatrix(Int(slot[].variance.unsafe_ptr()), K, d)
    if block == NB_BLOCK_CONST:
        return MemberMatrix(Int(slot[].constant.unsafe_ptr()), 1, K)
    if block == NB_BLOCK_CLASS_COUNT:
        return MemberMatrix(Int(slot[].class_count.unsafe_ptr()), 1, K)
    if block == NB_BLOCK_CLASS_PRIOR:
        return MemberMatrix(Int(slot[].class_prior.unsafe_ptr()), 1, K)
    if block == NB_BLOCK_RAW_VAR:
        return MemberMatrix(Int(slot[].raw_variance.unsafe_ptr()), K, d)
    raise Error("member ABI: a GaussianNB member has no fitted block " + String(block))


def _export(slot: _Slot, call: _Call) raises:
    """A view of one fitted block. The slot owns the words; they stay valid
    until this state's next fit, import or free."""
    if not slot[].fitted:
        raise Error("NotFittedError: this GaussianNB member is not fitted yet")
    var block = call[].block
    if (call[].flags & MEMBER_BLOCK_BY_INDEX) != 0:
        if block < 0 or block >= NB_GAUSSIAN_BLOCKS:
            raise Error("member ABI: a GaussianNB member has no fitted block index " + String(block))
        block = MEMBER_BLOCK_SCALARS if block == 0 else NB_BLOCK_THETA + block - 1
    if block == MEMBER_BLOCK_SCALARS:
        call[].dst = MemberMatrix(Int(slot[].scalars.unsafe_ptr()), 1, len(slot[].scalars), MEMBER_F64)
    else:
        call[].dst = _block_view(slot, block)
    call[].block = block
    call[].out_cols = NB_GAUSSIAN_BLOCKS


def _import(slot: _Slot, call: _Call) raises:
    """Copy one fitted block in; MEMBER_IMPORT_DONE completes the state."""
    var block = call[].block
    if slot[].fitted:
        slot[].clear()
    if block == MEMBER_BLOCK_SCALARS:
        if not call[].x.dense_host(MEMBER_F64) or call[].x.rows * call[].x.cols != NB_GAUSSIAN_SCALARS:
            raise Error("member ABI: the GaussianNB scalars block is 3 host float64 words "
                        + "(n_features_in_, classes, epsilon_)")
        var given = _F64P(unsafe_from_address=call[].x.data)
        slot[].scalars = [given[0], given[1], given[2]]
    elif block >= NB_BLOCK_THETA and block <= NB_BLOCK_RAW_VAR:
        var size = call[].x.rows * call[].x.cols
        if size < 1 or not call[].x.dense_host(MEMBER_F32):
            raise Error("TypeError: a GaussianNB member reads a fitted block as a dense host float32 view")
        var words = _copy_out(_F32P(unsafe_from_address=call[].x.data), size)
        if block == NB_BLOCK_THETA:
            slot[].theta = words^
        elif block == NB_BLOCK_VAR:
            slot[].variance = words^
        elif block == NB_BLOCK_CONST:
            slot[].constant = words^
        elif block == NB_BLOCK_CLASS_COUNT:
            slot[].class_count = words^
        elif block == NB_BLOCK_CLASS_PRIOR:
            slot[].class_prior = words^
        else:
            slot[].raw_variance = words^
    elif block != 0 or (call[].flags & MEMBER_IMPORT_DONE) == 0:
        raise Error("member ABI: a GaussianNB member has no fitted block " + String(block))
    if (call[].flags & MEMBER_IMPORT_DONE) == 0:
        return
    if len(slot[].scalars) != NB_GAUSSIAN_SCALARS:
        raise Error("member ABI: a fitted GaussianNB state needs its scalars block")
    var d = Int(slot[].scalars[0])
    var K = Int(slot[].scalars[1])
    if (d < 1 or K < 1 or len(slot[].theta) != K * d or len(slot[].variance) != K * d or len(slot[].constant) != K
            or len(slot[].class_count) != K or len(slot[].class_prior) != K or len(slot[].raw_variance) != K * d):
        raise Error("member ABI: a fitted GaussianNB state needs every fitted block, of the fitted sizes")
    slot[].features = d
    slot[].classes = K
    slot[].fitted = True


def _outputs(call: _Call) raises:
    """`call.out_cols`, `call.out_dtype` and `call.classes` of predict and
    predict_proba, with a state (its fitted class count and label type) or
    without one (the caller's `call.classes`)."""
    var asked = call[].flags
    var classes = call[].classes
    var label_dtype = MEMBER_I64
    if call[].state != 0:
        var slot = _Slot(unsafe_from_address=_lookup(call[].state, False))
        if slot[].fitted:
            classes = slot[].classes
            label_dtype = slot[].label_dtype
    if asked == MEMBER_OP_PREDICT:
        call[].out_cols = 1
        call[].out_dtype = label_dtype
    elif asked == MEMBER_OP_PREDICT_PROBA:
        call[].out_cols = classes
        call[].out_dtype = MEMBER_F64 if (call[].route.flags & NB_ROUTE_PROBA64) != 0 else MEMBER_F32
    else:
        member_unsupported(String("GaussianNB"), asked)
    call[].classes = classes


def _naive_bayes_member(operation: Int, address: Int) raises -> Int:
    var call = _Call(unsafe_from_address=address)
    if operation == MEMBER_OP_CAPABILITIES:
        # One published kind: the family's union is its mask. Scope and row
        # index depend on the route the caller resolved, never on the state.
        return _capabilities(call[].route.flags)
    if operation == MEMBER_OP_NEW:
        if call[].param_count < 1 or not call[].params:
            raise Error("member ABI: a NaiveBayes member takes its parameter words")
        var source = _F64P(unsafe_from_address=call[].params)
        var kind = Int(source[0])
        if Float64(kind) != source[0] or kind != NB_KIND_GAUSSIAN:
            raise Error("member ABI: NaiveBayes member kind " + String(source[0]) + " is not published")
        if call[].param_count < NB_GAUSSIAN_PARAM_HEAD:
            raise Error("member ABI: a GaussianNB member takes five parameter words and its priors")
        var prior_count = Int(source[4])
        if (Float64(prior_count) != source[4] or prior_count < 0
                or call[].param_count != NB_GAUSSIAN_PARAM_HEAD + prior_count):
            raise Error("member ABI: a GaussianNB member takes five parameter words and its priors")
        if not (source[1] >= 0.0):
            raise Error("member ABI: GaussianNB var_smoothing must be a non-negative number")
        var words = List[Float64](length=call[].param_count, fill=0.0)
        for index in range(call[].param_count):  # driver-loop: the parameter words
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
        _fit_gaussian(slot, call)
    elif operation == MEMBER_OP_PREDICT or operation == MEMBER_OP_PREDICT_PROBA:
        _score_gaussian(slot, call, operation)
    elif operation == MEMBER_OP_EXPORT:
        _export(slot, call)
    elif operation == MEMBER_OP_IMPORT:
        _import(slot, call)
    else:
        member_unsupported(_kind_name(slot[].kind()), operation)
    return 0


def naive_bayes_member_v2(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
    result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """The member entry of the NaiveBayes family (schema `MEMBER_ABI`)."""
    try:
        var word = _naive_bayes_member(operation, member_call_from_spans(operation, aa, na, np, nf))
        if result:
            MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = word
        return 0
    except error:
        native_descriptor_report(error, context, sink)
        return 1


def naive_bayes_member_publish(mode: Int, vendor: Int, publisher: Int) raises:
    """Publish this image's NaiveBayes member entry. One line in the image's
    module initialization, after its program transport is published."""
    member_publish[MEMBER_FAMILY_NAIVE_BAYES](mode, vendor, publisher, naive_bayes_member_v2)
