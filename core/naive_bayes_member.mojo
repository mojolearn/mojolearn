# SPDX-License-Identifier: Apache-2.0
"""The NaiveBayes family as MEMBERS (core/member_abi.mojo): GaussianNB
(kind 0), MultinomialNB (kind 1), BernoulliNB (kind 2) and ComplementNB
(kind 3); CategoricalNB is not published yet (`naive_bayes_kind_published`).

One entry, `naive_bayes_member_v2`, published by each x_prep image beside its
program transport (`naive_bayes_member_publish`, x_prep/transport_device.mojo
and x_prep/transport_host.mojo). A member runs the estimator's own x_prep
programs: the typed builders the door builds through
(bindings/prep_classify_prog_run.mojo `cls_gnb_fit_program`,
`cls_dnb_counts`, `cls_dnb_params`, `cls_score_program`) and the typed run
the door runs (`cls_borrowed_run`: `cls_run_route`, `cls_input_choice`,
`cls_prog_run`) over the transport entry of the `DeviceRoute` it is handed.
Same programs, same stages, same order as the door: no bit moves. No kernel
is touched.

Route slots read: `entry` the x_prep transport entry of the image; `flags`
NB_ROUTE_OUT (x_prep_run_out), NB_ROUTE_SCRATCH (x_prep_run_scratch),
NB_ROUTE_RANGES (x_prep_run_ranges), NB_ROUTE_DEV_STORE (x_prep_dev_put and
x_prep_dev_free), NB_ROUTE_TAKE_ROWS (x_prep_dev_take_rows),
NB_ROUTE_PROBA64 (the image exports x_prep_proba64), NB_ROUTE_CLASS_ONEPASS
(`x_prep_idn_fam` & IDN_CLASS_ONEPASS of an identical image),
NB_ROUTE_NB_ONEPASS (`x_prep_idn_int` & IDN_NB_ONEPASS of an identical
image; the discrete kinds' one-pass count).

MEMBER_OP_NEW parameters, GaussianNB (Float64 words, 5 + P):
  [0] kind: 0 GaussianNB
  [1] var_smoothing (checked at admission)
  [2] 2048-row statistics blocks 0 / 1 (MOJOLEARN_XPREP_BLOCKED)
  [3] R3 work region 0 / 1 (`xprep_r3("work")`)
  [4] P: the count of the `priors` parameter (0: priors=None)
  [5 .. 5 + P) the priors (non-negative, summing to one: checked at
      admission; P must equal the class count at fit)

MEMBER_OP_NEW parameters, the discrete kinds (Float64 words, 9 + P):
  [0] kind: 1 MultinomialNB, 2 BernoulliNB, 3 ComplementNB
  [1] alpha (positive: checked at admission)
  [2] 2048-row statistics blocks 0 / 1 (MOJOLEARN_XPREP_BLOCKED)
  [3] R3 work region 0 / 1
  [4] P: the count of the `class_prior` parameter (0: None)
  [5] fit_prior 0 / 1 (read when P is 0)
  [6] binarize given 0 / 1 (BernoulliNB; 0 for the other kinds)
  [7] the binarize threshold
  [8] norm 0 / 1 (ComplementNB)
  [9 .. 9 + P) the class priors (non-negative: checked at admission; P
      must equal the class count at fit)

Views: X dense row-major host float32 (finite: the caller's admission
tests it; MultinomialNB and ComplementNB refuse a negative word at fit, in
the estimator's words); `y` the class CODES 0..classes-1, one host column,
int32 (read in place) or int64; `call.classes` the class count at FIT (at
least one); sample weights one host float32 column
(`MEMBER_CAN_SAMPLE_WEIGHT`). PREDICT writes codes, one column, int32 or
int64 as `dst` says. PREDICT_PROBA writes rows x classes, float64 on a
route with NB_ROUTE_PROBA64, else float32 (`MEMBER_OP_OUTPUTS` answers
which). No score, no decision function, no partial fit, no column index, no
device-home view, no sparse X (the estimator's CSR route stays its door's).

`call.row_index` (`MEMBER_CAN_ROW_INDEX`, answered only on a route that can
gather: NB_ROUTE_RANGES and NB_ROUTE_TAKE_ROWS with device scratch): host
int64 row numbers, or int32 (a meta-estimator's draws: widened once per call
in chunks across the host's cores, `nb_wide_index`); X is then the WHOLE base and its rows are taken on the
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

Fitted-state blocks, the discrete kinds:
  0 `MEMBER_BLOCK_SCALARS` float64 [n_features_in_, classes]
  1 `NB_BLOCK_LOG_PROB` float32 classes x features (feature_log_prob_)
  2 `NB_BLOCK_LOG_PRIOR` float32 1 x classes (class_log_prior_)
  3 `NB_BLOCK_FEATURE_COUNT` float32 classes x features (feature_count_)
  4 `NB_BLOCK_CLASS_COUNT` float32 1 x classes (class_count_)
  5 `NB_BLOCK_W` float32 classes x features (BernoulliNB `_w`)
  6 `NB_BLOCK_BIAS` float32 1 x classes (BernoulliNB `_bias`)

The fitted state is OWNED here, in a process-wide store of fixed chunks (the
linear member's store): a slot never moves once minted, the lock covers only
mint, free and the handle lookup. Serial by nature here: the store's
bookkeeping. The label / weight takes of an indexed or int64 `y`, the int32
widen of a row index and of int64 output codes run in row tasks across the
host's cores; the programs are the family's and run on the route's device.
"""
from std.sys.info import size_of
from std.memory import unsafe_memcpy
from bindings.prep_classify_prog_run import (
    ClsProgram, ClsRunPlan, ClsBorrowed, CLS_NONE, CLS_W_PREDICT, CLS_W_PROBA,
    CLS_PRIOR_GIVEN, CLS_PRIOR_FIT, CLS_PRIOR_UNIFORM,
    cls_gnb_fit_program, cls_dnb_counts, cls_dnb_params, cls_score_program, cls_borrowed_run,
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
from core.host_parallel import host_widen_index_i64, host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk

comptime NB_KIND_GAUSSIAN = 0
comptime NB_KIND_MULTINOMIAL = 1  # the scoring program's CLS_KIND_MNB
comptime NB_KIND_BERNOULLI = 2    # CLS_KIND_BNB
comptime NB_KIND_COMPLEMENT = 3   # CLS_KIND_CNB
#: The parameter words of a discrete kind before its class priors.
comptime NB_DISCRETE_PARAM_HEAD = 9
#: GaussianNB's parameter words before its priors.
comptime NB_GAUSSIAN_PARAM_HEAD = 5

comptime NB_BLOCK_THETA = MEMBER_BLOCK_FAMILY
comptime NB_BLOCK_VAR = MEMBER_BLOCK_FAMILY + 1
comptime NB_BLOCK_CONST = MEMBER_BLOCK_FAMILY + 2
comptime NB_BLOCK_CLASS_COUNT = MEMBER_BLOCK_FAMILY + 3
comptime NB_BLOCK_CLASS_PRIOR = MEMBER_BLOCK_FAMILY + 4
comptime NB_BLOCK_RAW_VAR = MEMBER_BLOCK_FAMILY + 5
comptime NB_BLOCK_LOG_PROB = MEMBER_BLOCK_FAMILY + 6
comptime NB_BLOCK_LOG_PRIOR = MEMBER_BLOCK_FAMILY + 7
comptime NB_BLOCK_FEATURE_COUNT = MEMBER_BLOCK_FAMILY + 8
comptime NB_BLOCK_W = MEMBER_BLOCK_FAMILY + 9
comptime NB_BLOCK_BIAS = MEMBER_BLOCK_FAMILY + 10
comptime NB_GAUSSIAN_SCALARS = 3
comptime NB_DISCRETE_SCALARS = 2

comptime NB_ROUTE_OUT = 1
comptime NB_ROUTE_SCRATCH = 2
comptime NB_ROUTE_RANGES = 4
comptime NB_ROUTE_DEV_STORE = 8
comptime NB_ROUTE_TAKE_ROWS = 16
comptime NB_ROUTE_PROBA64 = 32
comptime NB_ROUTE_CLASS_ONEPASS = 64
comptime NB_ROUTE_NB_ONEPASS = 128

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
    #: The discrete kinds: feature_log_prob_, class_log_prior_,
    #: feature_count_ and, BernoulliNB, `_w` and `_bias`.
    var log_prob: List[Float32]
    var log_prior: List[Float32]
    var feature_count: List[Float32]
    var w: List[Float32]
    var bias: List[Float32]

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
        self.log_prob = List[Float32]()
        self.log_prior = List[Float32]()
        self.feature_count = List[Float32]()
        self.w = List[Float32]()
        self.bias = List[Float32]()

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
        self.log_prob = List[Float32]()
        self.log_prior = List[Float32]()
        self.feature_count = List[Float32]()
        self.w = List[Float32]()
        self.bias = List[Float32]()


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
    if kind == NB_KIND_MULTINOMIAL:
        return String("MultinomialNB")
    if kind == NB_KIND_BERNOULLI:
        return String("BernoulliNB")
    if kind == NB_KIND_COMPLEMENT:
        return String("ComplementNB")
    return String("NaiveBayes")


def naive_bayes_kind_published(kind: Int) -> Bool:
    """The kinds this entry serves (a class is marked when its kind is here)."""
    return kind >= NB_KIND_GAUSSIAN and kind <= NB_KIND_COMPLEMENT


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
    # No MEMBER_CAN_TEST_ROWS, for all four kinds (read 2026-11-21): the
    # stage kernels these programs run (x_prep/stats.mojo, x_prep/blocked.mojo,
    # x_prep/fastnb.mojo) and the transports (x_prep/transport_host.mojo,
    # x_prep/transport_device.mojo) hold no finite test and no refusal of a
    # NaN or infinite word of X, y or the weights; the only refusal after a
    # run is the negative minimum of MultinomialNB / ComplementNB
    # (`_fit_discrete`), which a NaN passes. The caller's admission tests.
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


def nb_wide_index(call: _Call, mut wide: List[Int64]) -> Bool:
    """A dense host int32 row index (a meta-estimator's draws, bagging's)
    widened into `wide` as the int64 row numbers every kind reads, the
    call's view pointed at it: one pass in chunks across the host's cores.
    True when the caller must restore `call.row_index` after the operation;
    any other index is left for `_rows` to read or refuse."""
    var index = call[].row_index
    if index.rows <= 0 or index.cols != 1 or not index.dense_host(MEMBER_I32):
        return False
    wide = List[Int64](unsafe_uninit_length=index.rows)
    host_widen_index_i64(index.data, Int(wide.unsafe_ptr()), index.rows)
    call[].row_index = MemberMatrix(Int(wide.unsafe_ptr()), index.rows, 1, MEMBER_I64)
    return True


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
        raise Error("TypeError: a " + estimator + " member reads its row index as one host int32 or int64 column")
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


struct _FitViews(Movable):
    """y and the sample weights of one fit as the program's records read
    them: n int32 class codes and n float32 weights in row order, in place
    (the caller's words) or in the lists this value owns (an int64 `y`, a
    row index). The addresses stay valid while this value is alive."""
    var codes: List[Int32]
    var taken: List[Float32]
    var codes_address: Int
    var weights_address: Int
    var weighted: Bool
    var label_dtype: Int

    def __init__(out self, var codes: List[Int32], var taken: List[Float32], codes_address: Int,
                 weights_address: Int, weighted: Bool, label_dtype: Int):
        self.codes = codes^
        self.taken = taken^
        self.codes_address = codes_address
        self.weights_address = weights_address
        self.weighted = weighted
        self.label_dtype = label_dtype

    def keep(self) -> Int:
        # Named after the run: the addressed words stay live through it.
        return len(self.codes) + len(self.taken)


def _take_codes(src: Int, wide: Bool, index: Int, dst: Int, n: Int):
    """`dst[i]` (int32) = the class code of row `index[i]` (row i when `index`
    is 0) of `src` (int32, or int64 narrowed: codes are class indices, exact),
    in row tasks across the host's cores (lane fx-audit 30: was a serial loop
    over rows). The door hands the kernels int32 codes its label encode wrote
    and gathers a fold's rows with the fold (no separate pass)."""
    var tasks = host_predict_task_count_work(n, n)
    var chunk = host_predict_chunk(n, tasks)

    def task(unit: Int) {imm src, imm wide, imm index, imm dst, imm n, imm chunk}:
        var out = _I32P(unsafe_from_address=dst)
        var rows = _I64P(unsafe_from_address=index)
        var narrow = _I32P(unsafe_from_address=src)
        var codes64 = _I64P(unsafe_from_address=src)
        for i in range(unit * chunk, min((unit + 1) * chunk, n)):  # native data loop (one host task's rows)
            var r = Int(rows[i]) if index != 0 else i
            out[i] = Int32(codes64[r]) if wide else narrow[r]

    host_parallelize(task, tasks)


def _take_weights(src: Int, index: Int, dst: Int, n: Int):
    """`dst[i]` = the float32 weight of row `index[i]` of `src`, in row tasks
    across the host's cores (lane fx-audit 30)."""
    var tasks = host_predict_task_count_work(n, n)
    var chunk = host_predict_chunk(n, tasks)

    def task(unit: Int) {imm src, imm index, imm dst, imm n, imm chunk}:
        var out = _F32P(unsafe_from_address=dst)
        var rows = _I64P(unsafe_from_address=index)
        var given = _F32P(unsafe_from_address=src)
        for i in range(unit * chunk, min((unit + 1) * chunk, n)):  # native data loop (one host task's rows)
            out[i] = given[Int(rows[i])]

    host_parallelize(task, tasks)


def _fit_views(call: _Call, rows: _Rows, estimator: String) raises -> _FitViews:
    """The view checks of y and the weights (every kind's fit), and their
    words in row order: read in place, or narrowed / taken through the row
    index (one word per row)."""
    var n = rows.n
    var label_dtype = call[].y.dtype
    if (not (call[].y.dense_host(MEMBER_I32) or call[].y.dense_host(MEMBER_I64)) or call[].y.cols != 1
            or call[].y.rows != rows.base_rows):
        raise Error("TypeError: a " + estimator + " member reads y as one host int32 or int64 column of class codes, "
                    + "one per row of X")
    var weighted = call[].weights.rows > 0
    if weighted and (not call[].weights.dense_host(MEMBER_F32) or call[].weights.cols != 1
                     or call[].weights.rows != rows.base_rows):
        raise Error("TypeError: a " + estimator + " member reads sample weights as one host float32 column, "
                    + "one per row of X")
    var codes = List[Int32]()
    var codes_address = call[].y.data
    if rows.index != 0 or label_dtype == MEMBER_I64:
        codes = List[Int32](unsafe_uninit_length=max(n, 1))
        _take_codes(call[].y.data, label_dtype == MEMBER_I64, rows.index, Int(codes.unsafe_ptr()), n)
        codes_address = Int(codes.unsafe_ptr())
    var taken = List[Float32]()
    var weights_address = call[].weights.data
    if weighted and rows.index != 0:
        taken = List[Float32](unsafe_uninit_length=max(n, 1))
        _take_weights(call[].weights.data, rows.index, Int(taken.unsafe_ptr()), n)
        weights_address = Int(taken.unsafe_ptr())
    return _FitViews(codes^, taken^, codes_address, weights_address, weighted, label_dtype)


def _fit_shape(call: _Call, rows: _Rows, estimator: String) raises:
    if call[].fit_output != 0:
        member_unsupported(estimator, call[].fit_output)
    if rows.n < 1 or rows.d < 1:
        raise Error("ValueError: a " + estimator + " member fits at least one row and one feature")
    if call[].classes < 1:
        raise Error("ValueError: a " + estimator + " member fits at least one class (call.classes)")


def _priors(slot: _Slot, head: Int, K: Int, estimator: String) raises -> List[Float64]:
    # The prior words of the state's parameters (word 4: their count).
    var p = slot[].words.unsafe_ptr()
    var prior_count = Int(p[4])
    if prior_count != 0 and prior_count != K:
        raise Error("mojolearn: " + estimator + ": number of priors must match number of classes")
    var priors = List[Float64](capacity=prior_count)
    for i in range(prior_count):  # driver-loop: over the user prior list (class-count sized)
        priors.append(p[head + i])
    return priors^


def _fit_gaussian(slot: _Slot, call: _Call) raises:
    """GaussianNB.fit: the door's fit program (`_gnb_fit`) over the caller's
    views; the fitted blocks are copied out of the arena into the slot."""
    var estimator = String("GaussianNB")
    var rows = _rows(call, estimator)
    _fit_shape(call, rows, estimator)
    var n = rows.n
    var d = rows.d
    var K = call[].classes
    var views = _fit_views(call, rows, estimator)
    var p = slot[].words.unsafe_ptr()
    var priors = _priors(slot, NB_GAUSSIAN_PARAM_HEAD, K, estimator)
    var has_priors = len(priors) != 0
    var blocked = p[2] != 0.0
    var onepass = blocked and (call[].route.flags & NB_ROUTE_CLASS_ONEPASS) != 0
    var program = ClsProgram(p[3] != 0.0)
    var o = cls_gnb_fit_program(program, n, d, K, views.weighted, p[1], has_priors, priors, blocked, onepass)
    var sources = List[ClsBorrowed](capacity=3)
    sources.append(_x_source(call, rows))
    sources.append(_plain(views.codes_address, n))
    if views.weighted:
        sources.append(_plain(views.weights_address, n))
    # THE PIN RULE (core/member_abi.mojo `MemberCall.pin`): the run writes
    # `call.pin_kept` at the moment the scope's cache takes the caller's
    # reference, so the caller reads it even when a later stage raises.
    var done = cls_borrowed_run(_run_plan(call), program, sources, call[].scope, call[].pin_kept)
    _ = views.keep()
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
    slot[].label_dtype = views.label_dtype
    slot[].fitted = True


def _fit_discrete(slot: _Slot, call: _Call) raises:
    """MultinomialNB / BernoulliNB / ComplementNB fit: the door's dense fit
    program (`_fit_counts`, `_prior_stages`, the class's `_params`) over the
    caller's views, the door's refusal of a negative word (MultinomialNB,
    ComplementNB), the fitted blocks copied out of the arena."""
    var kind = slot[].kind()
    var estimator = _kind_name(kind)
    var rows = _rows(call, estimator)
    _fit_shape(call, rows, estimator)
    var n = rows.n
    var d = rows.d
    var K = call[].classes
    var views = _fit_views(call, rows, estimator)
    var p = slot[].words.unsafe_ptr()
    var priors = _priors(slot, NB_DISCRETE_PARAM_HEAD, K, estimator)
    var prior = CLS_PRIOR_UNIFORM
    if len(priors) != 0:
        prior = CLS_PRIOR_GIVEN
    elif p[5] != 0.0:
        prior = CLS_PRIOR_FIT
    var blocked = p[2] != 0.0
    var onepass = blocked and (call[].route.flags & NB_ROUTE_NB_ONEPASS) != 0
    var program = ClsProgram(p[3] != 0.0)
    var c = cls_dnb_counts(program, kind, CLS_NONE, n, d, K, views.weighted, p[6] != 0.0, p[7], onepass, blocked,
                           prior, priors)
    var o = cls_dnb_params(program, kind, c, p[1], p[8] != 0.0)
    # record order: X, the weights, the class codes
    var sources = List[ClsBorrowed](capacity=3)
    sources.append(_x_source(call, rows))
    if views.weighted:
        sources.append(_plain(views.weights_address, n))
    sources.append(_plain(views.codes_address, n))
    # THE PIN RULE (core/member_abi.mojo `MemberCall.pin`): the run writes
    # `call.pin_kept` at the moment the scope's cache takes the caller's
    # reference, so the caller reads it even when a later stage raises.
    var done = cls_borrowed_run(_run_plan(call), program, sources, call[].scope, call[].pin_kept)
    _ = views.keep()
    if kind == NB_KIND_MULTINOMIAL or kind == NB_KIND_COMPLEMENT:
        var minimum = done.arena.f32(c.st + 3 * d)
        for j in range(d):  # driver-loop: over the per-column minimum statistics (feature count)
            if minimum[j] < 0:
                raise Error("mojolearn: Negative values in data passed to " + estimator + " (input X)")
    slot[].clear()
    slot[].class_count = _copy_out(done.arena.f32(c.cnt), K)
    slot[].feature_count = _copy_out(done.arena.f32(c.fc), K * d)
    slot[].log_prior = _copy_out(done.arena.f32(c.clp), K)
    slot[].log_prob = _copy_out(done.arena.f32(o.flp), K * d)
    if kind == NB_KIND_BERNOULLI:
        slot[].w = _copy_out(done.arena.f32(o.w), K * d)
        slot[].bias = _copy_out(done.arena.f32(o.bias), K)
    slot[].scalars = [Float64(d), Float64(K)]
    slot[].features = d
    slot[].classes = K
    slot[].label_dtype = views.label_dtype
    slot[].fitted = True


def _score(slot: _Slot, call: _Call, operation: Int) raises:
    """predict / predict_proba: the door's scoring program (`_base_scores`)
    with the slot's blocks as its fitted records."""
    var kind = slot[].kind()
    var estimator = _kind_name(kind)
    if not slot[].fitted:
        raise Error("NotFittedError: This " + estimator + " instance is not fitted yet. Call 'fit' with appropriate "
                    + "arguments before using this estimator.")
    var rows = _rows(call, estimator)
    var n = rows.n
    var d = slot[].features
    var K = slot[].classes
    if rows.d != d:
        raise Error("mojolearn: X has " + String(rows.d) + " features, but " + estimator + " was fitted with "
                    + String(d))
    if n < 1:
        raise Error("ValueError: a " + estimator + " member scores at least one row")
    var proba = operation == MEMBER_OP_PREDICT_PROBA
    var p64 = proba and (call[].route.flags & NB_ROUTE_PROBA64) != 0
    if not call[].dst.given() or call[].dst.home != 0 or call[].dst.stride != call[].dst.cols or call[].dst.offset != 0:
        raise Error("TypeError: a " + estimator + " member writes a dense host view")
    if proba:
        if call[].dst.rows != n or call[].dst.cols != K or call[].dst.dtype != (MEMBER_F64 if p64 else MEMBER_F32):
            raise Error("ValueError: the probability buffer holds " + String(K) + (" float64" if p64 else " float32")
                        + " values per row of X")
    elif (call[].dst.rows != n or call[].dst.cols != 1
            or (call[].dst.dtype != MEMBER_I32 and call[].dst.dtype != MEMBER_I64)):
        raise Error("ValueError: the prediction buffer holds one int32 or int64 class code per row of X")
    var p = slot[].words.unsafe_ptr()
    # The class's fitted records in the order `cls_jll_stages` creates them
    # (a record of no words is not created: no source for it).
    var a = _plain(Int(slot[].theta.unsafe_ptr()), K * d)
    var b = _plain(Int(slot[].variance.unsafe_ptr()), K * d)
    var c = _plain(Int(slot[].constant.unsafe_ptr()), K)
    var has_threshold = False
    var threshold = 0.0
    if kind == NB_KIND_BERNOULLI:
        a = _plain(Int(slot[].w.unsafe_ptr()), K * d)
        b = _plain(Int(slot[].bias.unsafe_ptr()), K)
        c = _plain(0, 0)
        has_threshold = p[6] != 0.0
        threshold = p[7]
    elif kind != NB_KIND_GAUSSIAN:
        a = _plain(Int(slot[].log_prob.unsafe_ptr()), K * d)
        # ComplementNB adds the class log prior only with one class
        b = _plain(Int(slot[].log_prior.unsafe_ptr()), K if (kind == NB_KIND_MULTINOMIAL or K == 1) else 0)
        c = _plain(0, 0)
    var program = ClsProgram(p[3] != 0.0)
    var o = cls_score_program(program, kind, CLS_NONE, n, d, K, a.words, b.words, c.words,
                              CLS_W_PROBA if proba else CLS_W_PREDICT, p64, False, False, has_threshold, threshold, 0)
    var sources = List[ClsBorrowed](capacity=4)
    sources.append(_x_source(call, rows))
    sources.append(a)
    if b.words > 0:
        sources.append(b)
    if c.words > 0:
        sources.append(c)
    # THE PIN RULE (core/member_abi.mojo `MemberCall.pin`): the run writes
    # `call.pin_kept` at the moment the scope's cache takes the caller's
    # reference, so the caller reads it even when a later stage raises.
    var done = cls_borrowed_run(_run_plan(call), program, sources, call[].scope, call[].pin_kept)
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
    # int64 output: the codes widened in chunks across the host's cores.
    host_widen_index_i64(Int(found), call[].dst.data, n)
    _ = done^


def _block_count(kind: Int) -> Int:
    if kind == NB_KIND_GAUSSIAN or kind == NB_KIND_BERNOULLI:
        return 7
    return 5


def _scalar_count(kind: Int) -> Int:
    return NB_GAUSSIAN_SCALARS if kind == NB_KIND_GAUSSIAN else NB_DISCRETE_SCALARS


def _block_at(kind: Int, index: Int) -> Int:
    # The block of an index in the kind's order (the module docstring).
    if index == 0:
        return MEMBER_BLOCK_SCALARS
    if kind == NB_KIND_GAUSSIAN:
        return NB_BLOCK_THETA + index - 1
    if index == 1:
        return NB_BLOCK_LOG_PROB
    if index == 2:
        return NB_BLOCK_LOG_PRIOR
    if index == 3:
        return NB_BLOCK_FEATURE_COUNT
    if index == 4:
        return NB_BLOCK_CLASS_COUNT
    return NB_BLOCK_W if index == 5 else NB_BLOCK_BIAS


def _has_block(kind: Int, block: Int) -> Bool:
    for index in range(1, _block_count(kind)):  # driver-loop: over the kind's at most six family blocks
        if _block_at(kind, index) == block:
            return True
    return False


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
    if block == NB_BLOCK_LOG_PROB:
        return MemberMatrix(Int(slot[].log_prob.unsafe_ptr()), K, d)
    if block == NB_BLOCK_LOG_PRIOR:
        return MemberMatrix(Int(slot[].log_prior.unsafe_ptr()), 1, K)
    if block == NB_BLOCK_FEATURE_COUNT:
        return MemberMatrix(Int(slot[].feature_count.unsafe_ptr()), K, d)
    if block == NB_BLOCK_W:
        return MemberMatrix(Int(slot[].w.unsafe_ptr()), K, d)
    return MemberMatrix(Int(slot[].bias.unsafe_ptr()), 1, K)


def _export(slot: _Slot, call: _Call) raises:
    """A view of one fitted block. The slot owns the words; they stay valid
    until this state's next fit, import or free."""
    var kind = slot[].kind()
    var estimator = _kind_name(kind)
    if not slot[].fitted:
        raise Error("NotFittedError: this " + estimator + " member is not fitted yet")
    var block = call[].block
    if (call[].flags & MEMBER_BLOCK_BY_INDEX) != 0:
        if block < 0 or block >= _block_count(kind):
            raise Error("member ABI: a " + estimator + " member has no fitted block index " + String(block))
        block = _block_at(kind, block)
    if block == MEMBER_BLOCK_SCALARS:
        call[].dst = MemberMatrix(Int(slot[].scalars.unsafe_ptr()), 1, len(slot[].scalars), MEMBER_F64)
    elif _has_block(kind, block):
        call[].dst = _block_view(slot, block)
    else:
        raise Error("member ABI: a " + estimator + " member has no fitted block " + String(block))
    call[].block = block
    call[].out_cols = _block_count(kind)


def _import(slot: _Slot, call: _Call) raises:
    """Copy one fitted block in; MEMBER_IMPORT_DONE completes the state."""
    var kind = slot[].kind()
    var estimator = _kind_name(kind)
    var block = call[].block
    if slot[].fitted:
        slot[].clear()
    if block == MEMBER_BLOCK_SCALARS:
        var count = _scalar_count(kind)
        if not call[].x.dense_host(MEMBER_F64) or call[].x.rows * call[].x.cols != count:
            raise Error("member ABI: the " + estimator + " scalars block is " + String(count)
                        + " host float64 words (n_features_in_, classes" + (", epsilon_)" if count == 3 else ")"))
        var given = _F64P(unsafe_from_address=call[].x.data)
        var scalars = List[Float64](capacity=count)
        for i in range(count):  # driver-loop: over the two or three scalar words
            scalars.append(given[i])
        slot[].scalars = scalars^
    elif _has_block(kind, block):
        var size = call[].x.rows * call[].x.cols
        if size < 1 or not call[].x.dense_host(MEMBER_F32):
            raise Error("TypeError: a " + estimator + " member reads a fitted block as a dense host float32 view")
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
        elif block == NB_BLOCK_RAW_VAR:
            slot[].raw_variance = words^
        elif block == NB_BLOCK_LOG_PROB:
            slot[].log_prob = words^
        elif block == NB_BLOCK_LOG_PRIOR:
            slot[].log_prior = words^
        elif block == NB_BLOCK_FEATURE_COUNT:
            slot[].feature_count = words^
        elif block == NB_BLOCK_W:
            slot[].w = words^
        else:
            slot[].bias = words^
    elif block != 0 or (call[].flags & MEMBER_IMPORT_DONE) == 0:
        raise Error("member ABI: a " + estimator + " member has no fitted block " + String(block))
    if (call[].flags & MEMBER_IMPORT_DONE) == 0:
        return
    if len(slot[].scalars) != _scalar_count(kind):
        raise Error("member ABI: a fitted " + estimator + " state needs its scalars block")
    var d = Int(slot[].scalars[0])
    var K = Int(slot[].scalars[1])
    var whole = d >= 1 and K >= 1 and len(slot[].class_count) == K
    if kind == NB_KIND_GAUSSIAN:
        whole = (whole and len(slot[].theta) == K * d and len(slot[].variance) == K * d and len(slot[].constant) == K
                 and len(slot[].class_prior) == K and len(slot[].raw_variance) == K * d)
    else:
        whole = (whole and len(slot[].log_prob) == K * d and len(slot[].log_prior) == K
                 and len(slot[].feature_count) == K * d)
        if kind == NB_KIND_BERNOULLI:
            whole = whole and len(slot[].w) == K * d and len(slot[].bias) == K
    if not whole:
        raise Error("member ABI: a fitted " + estimator + " state needs every fitted block, of the fitted sizes")
    slot[].features = d
    slot[].classes = K
    slot[].fitted = True


def _outputs(call: _Call) raises:
    """`call.out_cols`, `call.out_dtype` and `call.classes` of predict and
    predict_proba, with a state (its fitted class count and label type) or
    without one (the caller's `call.classes`; every kind answers alike)."""
    var asked = call[].flags
    var classes = call[].classes
    var label_dtype = MEMBER_I64
    var estimator = String("NaiveBayes")
    if call[].state != 0:
        var slot = _Slot(unsafe_from_address=_lookup(call[].state, False))
        estimator = _kind_name(slot[].kind())
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
        member_unsupported(estimator, asked)
    call[].classes = classes


def _new(call: _Call) raises:
    """MEMBER_OP_NEW: the parameter words checked (kind, counts, ranges)
    and kept in a new state."""
    if call[].param_count < 1 or not call[].params:
        raise Error("member ABI: a NaiveBayes member takes its parameter words")
    var source = _F64P(unsafe_from_address=call[].params)
    var kind = Int(source[0])
    if Float64(kind) != source[0] or not naive_bayes_kind_published(kind):
        raise Error("member ABI: NaiveBayes member kind " + String(source[0]) + " is not published")
    var estimator = _kind_name(kind)
    var head = NB_GAUSSIAN_PARAM_HEAD if kind == NB_KIND_GAUSSIAN else NB_DISCRETE_PARAM_HEAD
    var shape = "member ABI: a " + estimator + " member takes " + String(head) + " parameter words and its priors"
    if call[].param_count < head:
        raise Error(shape)
    var prior_count = Int(source[4])
    if Float64(prior_count) != source[4] or prior_count < 0 or call[].param_count != head + prior_count:
        raise Error(shape)
    if kind == NB_KIND_GAUSSIAN:
        if not (source[1] >= 0.0):
            raise Error("member ABI: GaussianNB var_smoothing must be a non-negative number")
    elif not (source[1] > 0.0):
        raise Error("member ABI: " + estimator + " alpha must be a positive number")
    var words = List[Float64](length=call[].param_count, fill=0.0)
    for index in range(call[].param_count):  # driver-loop: the parameter words
        words[index] = source[index]
    call[].state = _mint(words^)


def _naive_bayes_member(operation: Int, address: Int) raises -> Int:
    var call = _Call(unsafe_from_address=address)
    if operation == MEMBER_OP_CAPABILITIES:
        # Every published kind honors the same bits. Scope and row index
        # depend on the route the caller resolved, never on the state.
        return _capabilities(call[].route.flags)
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
    # A host int32 row index (a meta-estimator's draws): widened once to the
    # int64 row numbers every kind reads, the caller's view restored after.
    var given = call[].row_index
    var wide = List[Int64]()
    var widened = nb_wide_index(call, wide)
    try:
        if operation == MEMBER_OP_FIT:
            if slot[].kind() == NB_KIND_GAUSSIAN:
                _fit_gaussian(slot, call)
            else:
                _fit_discrete(slot, call)
        elif operation == MEMBER_OP_PREDICT or operation == MEMBER_OP_PREDICT_PROBA:
            _score(slot, call, operation)
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
