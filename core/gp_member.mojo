# SPDX-License-Identifier: Apache-2.0
"""The member entry of the Gp family (MEMBER_FAMILY_GP = 23, schema
`MEMBER_ABI`; recipe MEMBER-ABI-V2.md, form core/kernel_ridge_member.mojo).
Python-free RUN: it dispatches to the family's typed GP provider requests
(core/gp_native_api.mojo `GPDriver`: `gpr_optimize`, `gpr_fit`, `gpr_query`;
`gpc_fit_all` / `gpc_fit`, `gpc_query_all` / `gpc_query`, `gpc_combine`)
and the typed target normalization (core/gp_native_normalization.mojo
`gp_normalize_targets`), with every entry read from `call[].route`.

KINDS: 0 GaussianProcessRegressor, 1 GaussianProcessClassifier.

NEW PARAMETER WORDS (float64; header of 12, then the kernel's postfix table):
  0 kind
  1 regressor: alpha (the estimator's `_alpha`, the starting ridge);
    classifier: max_iter_predict (>= 1). Words 2 to 7 are 0 for the
    classifier, and so are F and B (its kernel is used as given: targets
    all 0).
  2 normalize_y (0 / 1)
  3 optimize (1: optimizer='fmin_l_bfgs_b' and the kernel has free
    hyperparameters; 0: the kernel is used as given)
  4 n_restarts_optimizer
  5 seed, low 32 bits; 6 seed, high 32 bits (random_state; None is 0)
  7 normalization direct word (the scaler provider's direct route, 0 / 1)
  8 N, postfix node count; 9 S, length-scale word count; 10 F, free
    hyperparameter values; 11 B, bound words (2 per free value)
  then kinds[N] (node kind codes), parameters[N] (each node's parameter),
  lengths[N] (each node's length-scale count), scales[S] (in node order),
  targets[N] (0 fixed node, 1 a free node whose value is its parameter,
  2 a free node whose values are its length scales), bounds[B] (lower,
  upper per free value, in postfix order).
  Total: 12 + 4 N + S + B words.

ROUTE SLOTS: `route.entry` the GP provider table entry (`route.mode`,
`route.flags` = its capability word GP_CAP_*); `route.second` the scaler
provider entry and `route.third` the preprocessing program entry (both set
only when normalize_y is 1). No metrics slot: score is not published (the
caller's metric, as for SVR).

VIEWS: FIT x float32 (rows x features), y float32 (rows x 1), dense host
views, no weights, no row / column index, no fit_output. PREDICT and
PREDICT_STD: x float32 (rows x features) -> dst float32 (rows x 1): the
predictive mean / the predictive standard deviation.
Classifier: FIT y = int32 class codes 0 .. classes-1 (rows x 1;
`call.classes` at fit, at least 2). PREDICT -> dst int32 codes (rows x 1);
PREDICT_PROBA -> dst float64 (rows x classes). One binary Laplace fit
(class 1 against 0) for two classes, else one per class (one-vs-rest).

LIFETIME (both kinds): the fitted state borrows the caller's fit X view as X_train_ (no
copy at fit; the provider's resident lease reads it on the GPU route), and,
when normalize_y is 0, its y view as y_train_. The caller keeps both alive
and unchanged until this state's next FIT, IMPORT or FREE. IMPORT copies.

CLASSIFIER BLOCKS (MEMBER_BLOCK_BY_INDEX order; m = binary fits): 0 scalars
(16 float64: kind, rows, features, classes, m, max_iter_predict, then 0,
word 15 = N); 1 X_train_ (rows x features f32); 2 the fits' y_train_ (m x
rows f32); 3 their L_ (m * rows x rows f32); 4 pi_ (m x rows); 5 W_sr_ (m x
rows); 6 kernel node parameters (N f32); 7 length scales (max(1, S) f32);
8 the fits' (log marginal likelihood, n_iter, nb) (m x 3 f64).

REGRESSOR BLOCKS (MEMBER_BLOCK_BY_INDEX order): 0 scalars (16 float64: kind, rows,
features, info, nb, logdet, y.alpha, log marginal likelihood, y mean, y std,
normalize_y, alpha used, optimized, F, optimizer runs, N); 1 X_train_
(rows x features f32); 2 y_train_ (rows f32, normalized when normalize_y);
3 L_ (rows x rows f32); 4 alpha_ (rows f32); 5 the fitted kernel's node
parameters (N f32); 6 its length scales (max(1, S) f32); 7 theta (F f64);
8 the optimized free values (F f64); 9 the optimizer runs (runs x 4 f64:
the provider's restart records). Blocks 7 to 9 are empty when not optimized.
FIT NOTICES (`MEMBER_BLOCK_NOTICES`, by name, MEMBER_CAN_NOTICES): the regressor's
ridge retries as one `MEMBER_NOTICE_GP_RIDGE` (starting alpha, alpha used); the
caller raises the RuntimeWarnings (core/member_notices.mojo).

SERIAL BY NATURE: one member call drives one provider request at a time
(the optimizer's restarts and the factorization run inside the provider,
on every core / the GPU); a caller runs its members in parallel.
"""
from std.memory import unsafe_memcpy
from std.sys.info import size_of
from core.admitted_refusals import admitted_finite_wording, admitted_finite_refuse
from core.entry_table import EntryRef, entry_ref_unresolved
from core.gp_native_api import (
    GPDriver, GPRFitRequest, GPRQueryRequest, GPROptimizeRequest,
    GPCFitRequest, GPCFitAllRequest, GPCQueryRequest, GPCQueryAllRequest, GPCCombineRequest,
    GP_CAP_OVR, GP_CAP_POINTER, GP_CAP_RESIDENT, GP_CAP_OPTIMIZE,
)
from core.gp_native_normalization import GPRNormalizeRequest, gp_normalize_targets
from core.gp_native_storage import GPKernelStorage, GPCQueryScratch
from core.hotpath_impl import cast_elements, HP_I32, HP_I64
from core.host_finite_scan import host_all_finite_f32
from core.member_abi import (
    MemberCall, MemberMatrix, MEMBER_F32, MEMBER_F64, MEMBER_I32,
    MEMBER_OP_CAPABILITIES, MEMBER_OP_NEW, MEMBER_OP_FREE, MEMBER_OP_FIT, MEMBER_OP_PREDICT,
    MEMBER_OP_PREDICT_STD, MEMBER_OP_PREDICT_PROBA, MEMBER_OP_OUTPUTS, MEMBER_OP_EXPORT, MEMBER_OP_IMPORT,
    MEMBER_CAN_FIT, MEMBER_CAN_PREDICT, MEMBER_CAN_PREDICT_STD, MEMBER_CAN_PREDICT_PROBA, MEMBER_CAN_CLASSIFIER,
    MEMBER_CAN_EXPORT, MEMBER_CAN_IMPORT, MEMBER_CAN_TEST_ROWS, MEMBER_CAN_NOTICES,
    MEMBER_BLOCK_SCALARS, MEMBER_BLOCK_FAMILY, MEMBER_BLOCK_BY_INDEX, MEMBER_BLOCK_NOTICES, MEMBER_IMPORT_DONE,
    member_publish, member_call_from_spans, member_unsupported,
)
from core.member_abi import MEMBER_BORROWS_FIT, MEMBER_EXPORT_OWN
from core.member_families import MEMBER_FAMILY_GP
from core.member_notices import member_notice_add, MEMBER_NOTICE_GP_RIDGE, MEMBER_NOTICE_WORDS
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.scaler_native_api import ScalerDriver, PrepProgramDriver
from core.shared_slot import SharedSlot, shared_slot_lock, shared_slot_unlock

comptime _CHUNK = 256
comptime _INDEX_SPAN = 16777216
comptime _HEADER = 12
comptime _SCALARS = 16
comptime _BLOCKS = 10
comptime _B_X = MEMBER_BLOCK_FAMILY
comptime _B_Y = MEMBER_BLOCK_FAMILY + 1
comptime _B_L = MEMBER_BLOCK_FAMILY + 2
comptime _B_DUAL = MEMBER_BLOCK_FAMILY + 3
comptime _B_KPARAMS = MEMBER_BLOCK_FAMILY + 4
comptime _B_KSCALES = MEMBER_BLOCK_FAMILY + 5
comptime _B_THETA = MEMBER_BLOCK_FAMILY + 6
comptime _B_VALUES = MEMBER_BLOCK_FAMILY + 7
comptime _B_RUNS = MEMBER_BLOCK_FAMILY + 8
comptime _B_PI = MEMBER_BLOCK_FAMILY + 9
comptime _B_WSR = MEMBER_BLOCK_FAMILY + 10
comptime _B_MEMBERS = MEMBER_BLOCK_FAMILY + 11
comptime _GPC_BLOCKS = 9
comptime _KIND_GPR = 0
comptime _KIND_GPC = 1
comptime _GPR = "GaussianProcessRegressor"
comptime _GPC = "GaussianProcessClassifier"


struct GpMemberSlot(Copyable, Movable, Defaultable):
    var generation: Int
    var live: Bool
    var params: List[Float64]
    # The unfitted kernel (NEW's postfix table) and its free-parameter words.
    var base: GPKernelStorage
    var flags: List[Int32]
    var bounds: List[Float32]
    var targets: List[Int]
    var fitted: Bool
    var rows: Int
    var cols: Int
    var info: Int
    var nb: Int
    var logdet: Float64
    var ydotalpha: Float64
    var lml: Float64
    var y_mean: Float64
    var y_std: Float64
    var alpha_used: Float64
    var optimized: Bool
    # X_train_ / y_train_: the caller's fit views (LIFETIME) or owned words.
    var x_fit: Int
    var x_owned: List[Float32]
    var y_fit: Int
    var y_owned: List[Float32]
    var factor: List[Float32]
    var dual: List[Float32]
    var kernel: GPKernelStorage
    var theta: List[Float64]
    var values: List[Float64]
    var runs: List[Float64]
    var run_count: Int
    var scalars: List[Float64]
    # FIT NOTICES (core/member_notices.mojo): the regressor's ridge retries,
    # set by `_fit`, cleared by the next fit or import.
    var notices: List[Float64]
    # The provider's resident lease of X_train_, L_ and alpha_ (GPU route)
    # and the entry that closes it. 0: none held.
    var lease: Int
    var lease_entry: EntryRef
    # GaussianProcessClassifier (kind 1): classes, binary Laplace fits (one,
    # or one per class past two), their columns, the fits' float32 words
    # back to back (y_train_, pi_, W_sr_: members x rows; L_ in `factor`:
    # members x rows x rows), their (lml, n_iter, nb) words and the address
    # tables the provider reads (built once per fit or import).
    var classes: Int
    var members: Int
    var columns: List[Int]
    var y01: List[Float32]
    var pis: List[Float32]
    var wsrs: List[Float32]
    var member_scalars: List[Float64]
    var t_targets: List[Int]
    var t_factors: List[Int]
    var t_pis: List[Int]
    var t_wsrs: List[Int]
    var t_scalars: List[Int]

    def __init__(out self):
        self.generation = 0
        self.live = False
        self.params = List[Float64]()
        self.base = GPKernelStorage()
        self.flags = List[Int32]()
        self.bounds = List[Float32]()
        self.targets = List[Int]()
        self.fitted = False
        self.rows = 0
        self.cols = 0
        self.info = 0
        self.nb = 0
        self.logdet = 0.0
        self.ydotalpha = 0.0
        self.lml = 0.0
        self.y_mean = 0.0
        self.y_std = 1.0
        self.alpha_used = 0.0
        self.optimized = False
        self.x_fit = 0
        self.x_owned = List[Float32]()
        self.y_fit = 0
        self.y_owned = List[Float32]()
        self.factor = List[Float32]()
        self.dual = List[Float32]()
        self.kernel = GPKernelStorage()
        self.theta = List[Float64]()
        self.values = List[Float64]()
        self.runs = List[Float64]()
        self.run_count = 0
        self.scalars = List[Float64]()
        self.notices = List[Float64]()
        self.lease = 0
        self.lease_entry = entry_ref_unresolved()
        self.classes = 0
        self.members = 0
        self.columns = List[Int]()
        self.y01 = List[Float32]()
        self.pis = List[Float32]()
        self.wsrs = List[Float32]()
        self.member_scalars = List[Float64]()
        self.t_targets = List[Int]()
        self.t_factors = List[Int]()
        self.t_pis = List[Int]()
        self.t_wsrs = List[Int]()
        self.t_scalars = List[Int]()

    def clear(mut self):
        self.fitted = False
        self.notices = List[Float64]()
        self.rows = 0
        self.cols = 0
        self.info = 0
        self.nb = 0
        self.logdet = 0.0
        self.ydotalpha = 0.0
        self.lml = 0.0
        self.y_mean = 0.0
        self.y_std = 1.0
        self.alpha_used = 0.0
        self.optimized = False
        self.x_fit = 0
        self.x_owned = List[Float32]()
        self.y_fit = 0
        self.y_owned = List[Float32]()
        self.factor = List[Float32]()
        self.dual = List[Float32]()
        self.kernel = GPKernelStorage()
        self.theta = List[Float64]()
        self.values = List[Float64]()
        self.runs = List[Float64]()
        self.run_count = 0
        self.scalars = List[Float64]()
        self.classes = 0
        self.members = 0
        self.columns = List[Int]()
        self.y01 = List[Float32]()
        self.pis = List[Float32]()
        self.wsrs = List[Float32]()
        self.member_scalars = List[Float64]()
        self.t_targets = List[Int]()
        self.t_factors = List[Int]()
        self.t_pis = List[Int]()
        self.t_wsrs = List[Int]()
        self.t_scalars = List[Int]()


struct GpMemberStore(Defaultable, Movable):
    var synchronization: List[Int64]
    var chunks: List[List[GpMemberSlot]]
    var free: List[Int]
    var count: Int

    def __init__(out self):
        # Side-effect-free even when the process slot's initializers race.
        self.synchronization = List[Int64](length=2, fill=0)
        self.chunks = List[List[GpMemberSlot]]()
        self.free = List[Int]()
        self.count = 0


comptime _STORE = SharedSlot[GpMemberStore, "MojolearnGpMemberStoreV2", 2]
comptime _Slot = MutPointer[GpMemberSlot, MutUntrackedOrigin]
comptime _Call = MutPointer[MemberCall, MutUntrackedOrigin]
comptime _F64P = MutPointer[Float64, MutUntrackedOrigin]


def _slot_address(index: Int) raises -> Int:
    # Under the store's lock. A chunk's buffer is allocated once and never
    # grows, so the address stays valid after the lock is released.
    var store = _STORE.get_or_create_ptr()
    return Int(store[].chunks[index // _CHUNK].unsafe_ptr()) + (index % _CHUNK) * size_of[GpMemberSlot]()


def _mint(var params: List[Float64], var base: GPKernelStorage, var flags: List[Int32], var bounds: List[Float32],
          var targets: List[Int]) raises -> Int:
    var store = _STORE.get_or_create_ptr()
    var lock = rebind[MutPointer[Int64, MutUntrackedOrigin]](store[].synchronization.unsafe_ptr())
    shared_slot_lock(lock)
    try:
        var index = -1
        if len(store[].free) > 0:
            index = store[].free.pop()
        else:
            if store[].count >= _INDEX_SPAN - 1:
                raise Error("RuntimeError: mojolearn: too many live Gp member states")
            if store[].count == len(store[].chunks) * _CHUNK:
                store[].chunks.append(List[GpMemberSlot](length=_CHUNK, fill=GpMemberSlot()))
            index = store[].count
            store[].count += 1
        var slot = _Slot(unsafe_from_address=_slot_address(index))
        slot[].generation += 1
        slot[].live = True
        slot[].params = params^
        slot[].base = base^
        slot[].flags = flags^
        slot[].bounds = bounds^
        slot[].targets = targets^
        slot[].clear()
        return slot[].generation * _INDEX_SPAN + index + 1
    finally:
        shared_slot_unlock(lock)


def _close_lease(entry: EntryRef, lease: Int):
    """Close a resident lease. Never raises (FREE never raises)."""
    try:
        GPDriver(entry.call).lease_close(lease)
    except:
        pass


def _drop_lease(slot: _Slot):
    """Before the fitted state changes (fit, import): its device lease goes."""
    if slot[].lease != 0:
        var lease = slot[].lease
        slot[].lease = 0
        _close_lease(slot[].lease_entry, lease)


def _lookup(handle: Int, release: Bool) raises -> Int:
    """The slot address of a live handle. A released handle, a handle of
    another store or a made-up integer is refused: never a stale read. A
    release closes the lease after the lock is released."""
    var store = _STORE.get_or_create_ptr()
    var lock = rebind[MutPointer[Int64, MutUntrackedOrigin]](store[].synchronization.unsafe_ptr())
    var address = 0
    var lease = 0
    var entry = entry_ref_unresolved()
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
            lease = slot[].lease
            entry = slot[].lease_entry
            slot[].lease = 0
            slot[].live = False
            slot[].clear()
            slot[].params = List[Float64]()
            slot[].base = GPKernelStorage()
            slot[].flags = List[Int32]()
            slot[].bounds = List[Float32]()
            slot[].targets = List[Int]()
            store[].free.append(index)
    finally:
        shared_slot_unlock(lock)
    if lease != 0:
        _close_lease(entry, lease)
    return address


def _f32p(address: Int) -> MutPointer[Float32, MutUntrackedOrigin]:
    return MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=address)


def _whole(call: _Call) raises:
    if call[].row_index.rows > 0 or call[].column_index.rows > 0:
        raise Error("TypeError: a Gp member reads whole host views (no row or column index)")


def _dense(view: MemberMatrix, dtype: Int, what: String) raises:
    if not view.dense_host(dtype):
        var word = String("float32") if dtype == MEMBER_F32 else (
            String("int32") if dtype == MEMBER_I32 else String("float64"))
        raise Error("TypeError: a Gp member reads " + what + " as a dense host " + word + " view")


def _finite(view: MemberMatrix, name: String) raises:
    """The estimator's admission test of a host float32 view (finite)."""
    _finite_of(view, name, _GPR)


def _finite_of(view: MemberMatrix, name: String, est: String) raises:
    var count = view.rows * view.cols
    if count > 0 and not host_all_finite_f32(_f32p(view.data), count):
        raise Error("ValueError: " + admitted_finite_wording(est, name))


def _parse(call: _Call) raises -> Int:
    """NEW: check the parameter words, build the kernel table, mint."""
    var count = call[].param_count
    if count < _HEADER or not call[].params:
        raise Error("member ABI: a Gp member takes at least " + String(_HEADER) + " parameters")
    var source = _F64P(unsafe_from_address=call[].params)
    var kind = Int(source[0])
    if kind != _KIND_GPR and kind != _KIND_GPC:
        raise Error("member ABI: unknown Gp member kind " + String(kind))
    var nodes = Int(source[8])
    var scale_words = Int(source[9])
    var free = Int(source[10])
    var bound_words = Int(source[11])
    if nodes < 1 or scale_words < 0 or free < 0 or bound_words < 0 or count != _HEADER + 4 * nodes + scale_words + bound_words:
        raise Error("member ABI: the Gp member's kernel table does not match its counts")
    var words = List[Float64](length=count, fill=0.0)
    for i in range(count):  # small-loop(the member's parameter words)
        words[i] = source[i]
    var base = GPKernelStorage()
    var flags = List[Int32]()
    var targets = List[Int]()
    var at = _HEADER
    for i in range(nodes):  # driver-loop: over the kernel's postfix nodes
        base.kinds.append(Int32(Int(words[at + i])))
        base.parameters.append(Float32(words[at + nodes + i]))
        base.lengths.append(Int32(Int(words[at + 2 * nodes + i])))
    at += 3 * nodes
    for i in range(scale_words):  # driver-loop: over the kernel's length-scale words
        base.scales.append(Float32(words[at + i]))
    at += scale_words
    var taken = 0
    for i in range(nodes):  # driver-loop: over the kernel's postfix nodes
        var target = Int(words[at + i])
        if target == 1:
            taken += 1
        elif target == 2:
            taken += Int(base.lengths[i])
        elif target != 0:
            raise Error("member ABI: a Gp member kernel node target is 0, 1 or 2")
        targets.append(target)
        flags.append(Int32(0 if target == 0 else 1))
    if taken != free:
        raise Error("member ABI: the Gp member's free values do not match its kernel table")
    if kind == _KIND_GPC and (free != 0 or bound_words != 0 or Int(words[1]) < 1):
        # The classifier's kernel is used as given (optimizer=None) and its
        # Newton loop runs at least once (max_iter_predict >= 1).
        raise Error("member ABI: a Gp classifier member takes a fixed kernel and max_iter_predict >= 1")
    at += nodes
    var bounds = List[Float32]()
    for i in range(bound_words):  # driver-loop: over the free values' bound words
        bounds.append(Float32(words[at + i]))
    base.finish()
    return _mint(words^, base^, flags^, bounds^, targets^)


def _with_values(slot: _Slot, values: List[Float64]) -> GPKernelStorage:
    """The base kernel with its free entries taken from `values` in postfix
    order (`gp_replace`'s walk, on the native table)."""
    var kernel = slot[].base.copy()
    var pos = 0
    var offset = 0
    for i in range(len(slot[].targets)):  # driver-loop: over the kernel's postfix nodes
        var length = Int(kernel.lengths[i])
        if slot[].targets[i] == 1:
            kernel.parameters[i] = Float32(values[pos])
            pos += 1
        elif slot[].targets[i] == 2:
            for j in range(length):  # driver-loop: over one node's length scales
                kernel.scales[offset + j] = Float32(values[pos])
                pos += 1
        offset += length
    return kernel^


def _fit(slot: _Slot, call: _Call) raises:
    _whole(call)
    if call[].fit_output != 0:
        member_unsupported(_GPR, call[].fit_output)
    if call[].weights.rows > 0:
        raise Error("TypeError: GaussianProcessRegressor.fit takes no sample_weight")
    _dense(call[].x, MEMBER_F32, "X")
    _dense(call[].y, MEMBER_F32, "y")
    var n = call[].x.rows
    var d = call[].x.cols
    if n < 1:
        raise Error("ValueError: " + String(_GPR) + ": X has no rows")
    if call[].y.rows != n or call[].y.cols != 1:
        raise Error(String(_GPR) + ": y has " + String(call[].y.rows) + " entries, X has " + String(n) + " rows")
    _finite(call[].x, "X")
    _finite(call[].y, "y")
    _drop_lease(slot)
    slot[].clear()
    var route = call[].route
    var driver = GPDriver(route.entry.call)
    var params = slot[].params.copy()
    var targets = call[].y.data
    if params[2] != 0.0:
        # `_gpr_normalize`: the scaler provider's StandardScaler fit and
        # transform of y (route slots two and three).
        if not route.has_second or not route.has_third:
            raise Error("member ABI: a Gp member with normalize_y needs the scaler route slots")
        slot[].y_owned = List[Float32](length=n, fill=Float32(0))
        var answer = gp_normalize_targets(ScalerDriver(route.second.call), PrepProgramDriver(route.third.call),
            GPRNormalizeRequest(call[].y.data, Int(slot[].y_owned.unsafe_ptr()), n, params[7] != 0.0))
        slot[].y_mean = answer.mean
        slot[].y_std = answer.stddev
        targets = Int(slot[].y_owned.unsafe_ptr())
    var alpha = params[1]
    var kernel = slot[].base.copy()
    if params[3] != 0.0:
        if (route.flags & GP_CAP_OPTIMIZE) == 0:
            raise Error("ImportError: the connected GP provider does not support gpr_optimize")
        var free = Int(params[10])
        var restarts = Int(params[4])
        var n_runs = 1 + restarts
        var theta = List[Float64](length=max(free, 1), fill=0.0)
        var values = List[Float64](length=max(free, 1), fill=0.0)
        var runs = List[Float64](length=max(4 * n_runs, 1), fill=0.0)
        _ = driver.gpr_optimize(GPROptimizeRequest(call[].x.data, targets, Int(slot[].flags.unsafe_ptr()),
            Int(slot[].bounds.unsafe_ptr()), Int(theta.unsafe_ptr()), Int(values.unsafe_ptr()),
            Int(runs.unsafe_ptr()), n, d, slot[].base.view(), alpha, restarts, Int(params[5]), Int(params[6])))
        kernel = _with_values(slot, values)
        slot[].theta = theta^
        slot[].values = values^
        slot[].runs = runs^
        slot[].run_count = n_runs
        slot[].optimized = True
    slot[].factor = List[Float32](length=n * n, fill=Float32(0))
    slot[].dual = List[Float32](length=n, fill=Float32(0))
    # info, nb, logdet, ydotalpha, lml -- in that order.
    var scalars = List[Float64](length=5, fill=0.0)
    var info = 0
    while True:  # driver-loop: over the tenfold ridge retries (alpha up to 1.0)
        info = driver.gpr_fit(GPRFitRequest(call[].x.data, targets, Int(slot[].factor.unsafe_ptr()),
            Int(slot[].dual.unsafe_ptr()), Int(scalars.unsafe_ptr()), n, d, kernel.view(), alpha,
            (route.flags & GP_CAP_POINTER) == GP_CAP_POINTER))
        # `_gpr_fit`'s rule: a positive ridge too small for float32 is raised
        # tenfold up to 1.0. The caller raises the RuntimeWarning for each
        # step from the fit notice below (alpha used vs. the starting alpha).
        if info == 0 or not alpha > 0.0 or alpha >= 1.0:
            break
        alpha = min(alpha * 10.0, 1.0)
    if alpha > params[1]:
        member_notice_add(slot[].notices, MEMBER_NOTICE_GP_RIDGE, params[1], alpha)
    slot[].rows = n
    slot[].cols = d
    slot[].info = info
    slot[].nb = Int(scalars[1])
    slot[].logdet = scalars[2]
    slot[].ydotalpha = scalars[3]
    slot[].lml = scalars[4]
    slot[].alpha_used = alpha
    slot[].x_fit = call[].x.data
    slot[].y_fit = targets
    slot[].kernel = kernel^
    slot[].fitted = True
    _ = scalars^


def _query(slot: _Slot, call: _Call, want_std: Bool) raises:
    if not slot[].fitted:
        raise Error("NotFittedError: this GaussianProcessRegressor member is not fitted yet")
    _whole(call)
    _dense(call[].x, MEMBER_F32, "X")
    var q = call[].x.rows
    var d = slot[].cols
    if call[].x.cols != d:
        raise Error(String(_GPR) + ": X has " + String(call[].x.cols) + " features, fit saw " + String(d))
    _dense(call[].dst, MEMBER_F32, "the output buffer")
    if call[].dst.rows != q or call[].dst.cols != 1:
        raise Error("ValueError: the output buffer holds one value per row of X")
    if q == 0:
        return
    var route = call[].route
    var caps = route.flags
    var resident = (caps & GP_CAP_RESIDENT) == GP_CAP_RESIDENT
    if not resident:
        # These entries return no status word: the query is tested here.
        _finite(call[].x, "X")
    var driver = GPDriver(route.entry.call)
    var lease = 0
    var way = 1 if (caps & GP_CAP_POINTER) == GP_CAP_POINTER else 0
    if resident:
        # X_train_, alpha_ and L_ cross to the device once per fitted state
        # (the estimator's `gp_resident_lease`): the lease is opened at the
        # first resident query and closed at the next fit, import or free.
        if slot[].lease == 0:
            slot[].lease = driver.lease_open()
            slot[].lease_entry = route.entry
        lease = slot[].lease
        way = 2
    var mean = List[Float32]()
    var variance = List[Float32]()
    var clamped = List[Int32]()
    var mean_address = call[].dst.data
    var var_address = 0
    var std_address = 0
    var clamped_address = 0
    if want_std:
        mean = List[Float32](length=q, fill=Float32(0))
        variance = List[Float32](length=q, fill=Float32(0))
        clamped = List[Int32](length=q, fill=Int32(0))
        mean_address = Int(mean.unsafe_ptr())
        var_address = Int(variance.unsafe_ptr())
        std_address = call[].dst.data
        clamped_address = Int(clamped.unsafe_ptr())
    var normalize = slot[].params[2] != 0.0
    var answer = driver.gpr_query(GPRQueryRequest(slot[].x_fit, Int(slot[].factor.unsafe_ptr()),
        Int(slot[].dual.unsafe_ptr()), call[].x.data, mean_address, var_address, std_address, clamped_address,
        slot[].rows, d, q, slot[].kernel.view(), slot[].info, want_std, normalize, slot[].y_std, slot[].y_mean,
        way, lease, -1, 0, -1))
    if resident:
        # The entry's status word: 0 = every query word finite (the
        # estimator's `gp_query_refuse`).
        admitted_finite_refuse(answer.finite, _GPR)
    _ = mean^
    _ = variance^
    _ = clamped^


def _own_views(slot: _Slot):
    """THE LIFETIME RULE (MEMBER_EXPORT_OWN): copy the borrowed X_train_
    (and the regressor's borrowed y_train_) into the state, once; the state
    reads its own words from then on. A device lease opened over the
    borrowed words is closed: the next resident query opens one over the
    state's own (one upload, only for a state that is kept past its fit
    buffers). Two copies of independent blocks, each one memcpy."""
    var moved = False
    var count = slot[].rows * slot[].cols
    if slot[].x_fit != 0 and (slot[].x_fit != Int(slot[].x_owned.unsafe_ptr()) or len(slot[].x_owned) != count):
        var owned = List[Float32](length=count, fill=Float32(0))
        if count > 0:
            unsafe_memcpy(dest=owned.unsafe_ptr(), src=_f32p(slot[].x_fit), count=count)
        slot[].x_owned = owned^
        slot[].x_fit = Int(slot[].x_owned.unsafe_ptr())
        moved = True
    var n = slot[].rows
    if slot[].y_fit != 0 and (slot[].y_fit != Int(slot[].y_owned.unsafe_ptr()) or len(slot[].y_owned) != n):
        var kept = List[Float32](length=n, fill=Float32(0))
        if n > 0:
            unsafe_memcpy(dest=kept.unsafe_ptr(), src=_f32p(slot[].y_fit), count=n)
        slot[].y_owned = kept^
        slot[].y_fit = Int(slot[].y_owned.unsafe_ptr())
        moved = True
    if moved:
        _drop_lease(slot)


def _export(slot: _Slot, call: _Call) raises:
    """A view of one fitted block. X_train_ (and y_train_ without
    normalize_y) is the caller's fit view or the imported words; the rest is
    owned here. Valid until this state's next fit, import or free."""
    if not slot[].fitted:
        raise Error("NotFittedError: this GaussianProcessRegressor member is not fitted yet")
    if (call[].flags & MEMBER_EXPORT_OWN) != 0:
        _own_views(slot)
        return
    var block = call[].block
    if (call[].flags & MEMBER_BLOCK_BY_INDEX) != 0:
        if block < 0 or block >= _BLOCKS:
            raise Error("member ABI: this Gp member has " + String(_BLOCKS) + " fitted blocks")
        block = MEMBER_BLOCK_SCALARS if block == 0 else MEMBER_BLOCK_FAMILY + block - 1
    var free = len(slot[].theta) if slot[].optimized else 0
    if block == MEMBER_BLOCK_SCALARS:
        slot[].scalars = List[Float64](length=_SCALARS, fill=0.0)
        slot[].scalars[0] = slot[].params[0]
        slot[].scalars[1] = Float64(slot[].rows)
        slot[].scalars[2] = Float64(slot[].cols)
        slot[].scalars[3] = Float64(slot[].info)
        slot[].scalars[4] = Float64(slot[].nb)
        slot[].scalars[5] = slot[].logdet
        slot[].scalars[6] = slot[].ydotalpha
        slot[].scalars[7] = slot[].lml
        slot[].scalars[8] = slot[].y_mean
        slot[].scalars[9] = slot[].y_std
        slot[].scalars[10] = slot[].params[2]
        slot[].scalars[11] = slot[].alpha_used
        slot[].scalars[12] = 1.0 if slot[].optimized else 0.0
        slot[].scalars[13] = Float64(free)
        slot[].scalars[14] = Float64(slot[].run_count)
        slot[].scalars[15] = Float64(len(slot[].kernel.kinds))
        call[].dst = MemberMatrix(Int(slot[].scalars.unsafe_ptr()), 1, _SCALARS, MEMBER_F64)
    elif block == _B_X:
        call[].dst = MemberMatrix(slot[].x_fit, slot[].rows, slot[].cols)
    elif block == _B_Y:
        call[].dst = MemberMatrix(slot[].y_fit, 1, slot[].rows)
    elif block == _B_L:
        call[].dst = MemberMatrix(Int(slot[].factor.unsafe_ptr()), slot[].rows, slot[].rows)
    elif block == _B_DUAL:
        call[].dst = MemberMatrix(Int(slot[].dual.unsafe_ptr()), 1, slot[].rows)
    elif block == _B_KPARAMS:
        call[].dst = MemberMatrix(Int(slot[].kernel.parameters.unsafe_ptr()), 1, len(slot[].kernel.parameters))
    elif block == _B_KSCALES:
        call[].dst = MemberMatrix(Int(slot[].kernel.scales.unsafe_ptr()), 1, len(slot[].kernel.scales))
    elif block == _B_THETA:
        call[].dst = MemberMatrix(Int(slot[].theta.unsafe_ptr()), 1 if free > 0 else 0, free, MEMBER_F64)
    elif block == _B_VALUES:
        call[].dst = MemberMatrix(Int(slot[].values.unsafe_ptr()), 1 if free > 0 else 0, free, MEMBER_F64)
    elif block == _B_RUNS:
        var runs = slot[].run_count if slot[].optimized else 0
        call[].dst = MemberMatrix(Int(slot[].runs.unsafe_ptr()), runs, 4, MEMBER_F64)
    else:
        raise Error("member ABI: a Gp member has no fitted block " + String(block))
    call[].block = block
    call[].out_cols = _BLOCKS


def _copy_f32(view: MemberMatrix, what: String) raises -> List[Float32]:
    _dense(view, MEMBER_F32, what)
    var count = view.rows * view.cols
    var words = List[Float32](length=max(1, count), fill=Float32(0))
    if count > 0:
        unsafe_memcpy(dest=words.unsafe_ptr(), src=_f32p(view.data), count=count)
    return words^


def _import(slot: _Slot, call: _Call) raises:
    """Copy one fitted block in; MEMBER_IMPORT_DONE completes the state.
    The scalars come first; blocks 1 (X_train_), 3 (L_), 4 (alpha_), 5 and
    6 (the fitted kernel) are required, the rest optional."""
    _drop_lease(slot)
    var block = call[].block
    if block == MEMBER_BLOCK_SCALARS:
        _dense(call[].x, MEMBER_F64, "the scalars block")
        if call[].x.rows * call[].x.cols != _SCALARS:
            raise Error("member ABI: the Gp scalars block is sixteen float64 words")
        var words = _F64P(unsafe_from_address=call[].x.data)
        if Int(words[0]) != Int(slot[].params[0]):
            raise Error("member ABI: the Gp scalars block holds another kind")
        if Int(words[15]) != len(slot[].base.kinds):
            raise Error("member ABI: the Gp scalars block holds another kernel")
        slot[].clear()
        slot[].rows = Int(words[1])
        slot[].cols = Int(words[2])
        slot[].info = Int(words[3])
        slot[].nb = Int(words[4])
        slot[].logdet = words[5]
        slot[].ydotalpha = words[6]
        slot[].lml = words[7]
        slot[].y_mean = words[8]
        slot[].y_std = words[9]
        slot[].alpha_used = words[11]
        slot[].kernel = slot[].base.copy()
    elif block == _B_X:
        if call[].x.rows != slot[].rows or call[].x.cols != slot[].cols:
            raise Error("member ABI: the Gp X_train_ block does not match the scalars block")
        slot[].x_owned = _copy_f32(call[].x, "X_train_")
        slot[].x_fit = Int(slot[].x_owned.unsafe_ptr())
    elif block == _B_Y:
        if call[].x.rows * call[].x.cols != slot[].rows:
            raise Error("member ABI: the Gp y_train_ block does not match the scalars block")
        slot[].y_owned = _copy_f32(call[].x, "y_train_")
        slot[].y_fit = Int(slot[].y_owned.unsafe_ptr())
    elif block == _B_L:
        if call[].x.rows * call[].x.cols != slot[].rows * slot[].rows:
            raise Error("member ABI: the Gp L_ block does not match the scalars block")
        slot[].factor = _copy_f32(call[].x, "L_")
    elif block == _B_DUAL:
        if call[].x.rows * call[].x.cols != slot[].rows:
            raise Error("member ABI: the Gp alpha_ block does not match the scalars block")
        slot[].dual = _copy_f32(call[].x, "alpha_")
    elif block == _B_KPARAMS:
        if call[].x.rows * call[].x.cols != len(slot[].kernel.parameters):
            raise Error("member ABI: the Gp kernel parameter block does not match the kernel table")
        slot[].kernel.parameters = _copy_f32(call[].x, "the kernel parameters")
    elif block == _B_KSCALES:
        if call[].x.rows * call[].x.cols != len(slot[].kernel.scales):
            raise Error("member ABI: the Gp kernel length-scale block does not match the kernel table")
        slot[].kernel.scales = _copy_f32(call[].x, "the kernel length scales")
    elif block != 0 or (call[].flags & MEMBER_IMPORT_DONE) == 0:
        raise Error("member ABI: a Gp member imports no fitted block " + String(block))
    if (call[].flags & MEMBER_IMPORT_DONE) == 0:
        return
    if slot[].rows < 1 or slot[].x_fit == 0 or len(slot[].factor) != slot[].rows * slot[].rows or len(slot[].dual) != slot[].rows:
        raise Error("member ABI: a fitted Gp state needs its scalars, X_train_, L_ and alpha_ blocks")
    slot[].fitted = True


def _gpc_tables(slot: _Slot):
    """The provider's address tables over the classifier's fitted words, built
    once per fit or import (the words do not move until the next one)."""
    var m = slot[].members
    var n = slot[].rows
    slot[].t_targets = List[Int](length=m, fill=0)
    slot[].t_factors = List[Int](length=m, fill=0)
    slot[].t_pis = List[Int](length=m, fill=0)
    slot[].t_wsrs = List[Int](length=m, fill=0)
    slot[].t_scalars = List[Int](length=m, fill=0)
    for i in range(m):  # driver-loop: over the binary Laplace fits (their addresses)
        slot[].t_targets[i] = Int(slot[].y01.unsafe_ptr() + i * n)
        slot[].t_factors[i] = Int(slot[].factor.unsafe_ptr() + i * n * n)
        slot[].t_pis[i] = Int(slot[].pis.unsafe_ptr() + i * n)
        slot[].t_wsrs[i] = Int(slot[].wsrs.unsafe_ptr() + i * n)
        slot[].t_scalars[i] = Int(slot[].member_scalars.unsafe_ptr() + 3 * i)


def _gpc_layout(slot: _Slot, n: Int, d: Int, k: Int):
    """Columns and storage of a classifier with `k` classes on `n` x `d`:
    one binary fit (class 1 against 0) for two classes, else one per class
    (`_gpc_fit`'s one-vs-rest columns)."""
    var m = 1 if k == 2 else k
    slot[].columns = List[Int](length=m, fill=0)
    if k == 2:
        slot[].columns[0] = 1
    else:
        for c in range(k):  # driver-loop: over the one-vs-rest classes
            slot[].columns[c] = c
    slot[].rows = n
    slot[].cols = d
    slot[].classes = k
    slot[].members = m


def _gpc_fit(slot: _Slot, call: _Call) raises:
    """`_gpc_fit`: every binary Laplace fit of the classifier, through the
    provider's one-vs-rest entry when it has one (GP_CAP_OVR: one upload of
    X, one kernel matrix), else one provider fit per column."""
    _whole(call)
    if call[].fit_output != 0:
        member_unsupported(_GPC, call[].fit_output)
    if call[].weights.rows > 0:
        raise Error("TypeError: GaussianProcessClassifier.fit takes no sample_weight")
    _dense(call[].x, MEMBER_F32, "X")
    _dense(call[].y, MEMBER_I32, "y (the class codes)")
    var n = call[].x.rows
    var d = call[].x.cols
    if n < 1:
        raise Error("ValueError: " + String(_GPC) + ": X has no rows")
    if call[].y.rows != n or call[].y.cols != 1:
        raise Error(String(_GPC) + ": y has " + String(call[].y.rows * call[].y.cols) + " entries, X has "
                    + String(n) + " rows")
    var k = call[].classes
    if k < 2:
        raise Error(String(_GPC) + " requires 2 or more distinct classes; got " + String(k) + " class")
    _finite_of(call[].x, "X", _GPC)
    _drop_lease(slot)
    slot[].clear()
    _gpc_layout(slot, n, d, k)
    var m = slot[].members
    slot[].y01 = List[Float32](length=m * n, fill=Float32(0))
    slot[].pis = List[Float32](length=m * n, fill=Float32(0))
    slot[].wsrs = List[Float32](length=m * n, fill=Float32(0))
    slot[].factor = List[Float32](length=m * n * n, fill=Float32(0))
    slot[].member_scalars = List[Float64](length=3 * m, fill=0.0)
    slot[].kernel = slot[].base.copy()
    _gpc_tables(slot)
    var route = call[].route
    var driver = GPDriver(route.entry.call)
    var max_iter = Int(slot[].params[1])
    if (route.flags & GP_CAP_OVR) == GP_CAP_OVR:
        _ = driver.gpc_fit_all(GPCFitAllRequest(call[].x.data, call[].y.data, Int(slot[].columns.unsafe_ptr()),
            Int(slot[].t_targets.unsafe_ptr()), Int(slot[].t_factors.unsafe_ptr()), Int(slot[].t_pis.unsafe_ptr()),
            Int(slot[].t_wsrs.unsafe_ptr()), Int(slot[].t_scalars.unsafe_ptr()), n, d, slot[].kernel.view(),
            max_iter, m))
    else:
        var scalars = List[Float64](length=3, fill=0.0)
        for i in range(m):  # driver-loop: one provider fit per binary column (the estimator's route without the OVR entry)
            var iterations = driver.gpc_fit(GPCFitRequest(call[].x.data, call[].y.data, slot[].t_factors[i],
                slot[].t_pis[i], slot[].t_wsrs[i], Int(scalars.unsafe_ptr()), n, d, slot[].kernel.view(), max_iter,
                True, slot[].columns[i], slot[].t_targets[i]))
            slot[].member_scalars[3 * i] = scalars[0]
            slot[].member_scalars[3 * i + 1] = Float64(iterations)
            slot[].member_scalars[3 * i + 2] = scalars[2]
        _ = scalars^
    slot[].x_fit = call[].x.data
    slot[].fitted = True


def _gpc_query(slot: _Slot, call: _Call, proba: Bool) raises:
    """`predict` (int32 codes, rows x 1) or `predict_proba` (float64, rows x
    classes): `_gpc_predict_all` through the one-vs-rest entry, else
    `_gpc_latent` per fit and `_gpc_ovr_combine` past two classes."""
    if not slot[].fitted:
        raise Error("NotFittedError: this GaussianProcessClassifier member is not fitted yet")
    _whole(call)
    _dense(call[].x, MEMBER_F32, "X")
    var q = call[].x.rows
    var d = slot[].cols
    if call[].x.cols != d:
        raise Error(String(_GPC) + ": X has " + String(call[].x.cols) + " features, fit saw " + String(d))
    var k = slot[].classes
    var m = slot[].members
    var n = slot[].rows
    if proba:
        _dense(call[].dst, MEMBER_F64, "the output buffer")
        if call[].dst.rows != q or call[].dst.cols != k:
            raise Error("ValueError: the output buffer holds one probability per class for each row of X")
    else:
        _dense(call[].dst, MEMBER_I32, "the output buffer")
        if call[].dst.rows != q or call[].dst.cols != 1:
            raise Error("ValueError: the output buffer holds one class code per row of X")
    if q == 0:
        return
    var route = call[].route
    var caps = route.flags
    var driver = GPDriver(route.entry.call)
    # predict: the entries write int64 codes, narrowed into `dst` below.
    var wide = List[Int64]()
    if (caps & GP_CAP_OVR) == GP_CAP_OVR:
        var lease = -1
        if (caps & GP_CAP_RESIDENT) == GP_CAP_RESIDENT:
            # X_train_ and the fits' words cross to the device once per
            # fitted state (the estimator's `gp_resident_lease`).
            if slot[].lease == 0:
                slot[].lease = driver.lease_open()
                slot[].lease_entry = route.entry
            lease = slot[].lease
        var destination = call[].dst.data
        if not proba:
            wide = List[Int64](length=q, fill=0)
            destination = Int(wide.unsafe_ptr())
        var answer = driver.gpc_query_all(GPCQueryAllRequest(slot[].x_fit, call[].x.data, destination,
            Int(slot[].t_targets.unsafe_ptr()), Int(slot[].t_pis.unsafe_ptr()), Int(slot[].t_wsrs.unsafe_ptr()),
            Int(slot[].t_factors.unsafe_ptr()), n, d, q, slot[].kernel.view(), 2 if proba else 1, m, lease,
            -1, 0, -1))
        admitted_finite_refuse(answer.finite, _GPC)
    else:
        # These entries return no status word: the query is tested here
        # (the estimator's `gp_query_scan`).
        _finite_of(call[].x, "X", _GPC)
        if m == 1:
            var scratch = GPCQueryScratch(q, 0)
            var destination = call[].dst.data
            if not proba:
                wide = List[Int64](length=q, fill=0)
                destination = Int(wide.unsafe_ptr())
            _ = driver.gpc_query(GPCQueryRequest(slot[].x_fit, slot[].t_targets[0], slot[].t_pis[0],
                slot[].t_wsrs[0], slot[].t_factors[0], call[].x.data, Int(scratch.mean.unsafe_ptr()),
                Int(scratch.variance.unsafe_ptr()), Int(scratch.probability.unsafe_ptr()), n, d, q,
                slot[].kernel.view(), proba, 2 if proba else 1, destination))
            _ = scratch^
        else:
            var columns = List[List[Float64]]()
            for i in range(m):  # driver-loop: one latent pass per binary fit (the estimator's route without the OVR entry)
                var scratch = GPCQueryScratch(q, 0)
                _ = driver.gpc_query(GPCQueryRequest(slot[].x_fit, slot[].t_targets[i], slot[].t_pis[i],
                    slot[].t_wsrs[i], slot[].t_factors[i], call[].x.data, Int(scratch.mean.unsafe_ptr()),
                    Int(scratch.variance.unsafe_ptr()), Int(scratch.probability.unsafe_ptr()), n, d, q,
                    slot[].kernel.view(), True, 0, 0))
                var probability = List[Float64]()
                swap(probability, scratch.probability)
                columns.append(probability^)
            var addresses = List[Int](length=m, fill=0)
            for i in range(m):  # driver-loop: one address per class column
                addresses[i] = Int(columns[i].unsafe_ptr())
            var private_proba = List[Float64]()
            var private_codes = List[Int32]()
            var probability_address = call[].dst.data
            var codes_address = call[].dst.data
            if proba:
                private_codes = List[Int32](length=q, fill=Int32(0))
                codes_address = Int(private_codes.unsafe_ptr())
            else:
                private_proba = List[Float64](length=q * k, fill=0.0)
                probability_address = Int(private_proba.unsafe_ptr())
            _ = driver.gpc_combine(GPCCombineRequest(Int(addresses.unsafe_ptr()), probability_address,
                codes_address, q, k))
            _ = columns^
            _ = addresses^
            _ = private_proba^
            _ = private_codes^
    if not proba and len(wide) == q:
        # The exact narrowing of the entries' codes (each below `classes`).
        _ = cast_elements(Int(wide.unsafe_ptr()), HP_I64, call[].dst.data, HP_I32, q)
    _ = wide^


def _gpc_block(block: Int) raises -> Int:
    """MEMBER_BLOCK_BY_INDEX -> the classifier's block word."""
    if block < 0 or block >= _GPC_BLOCKS:
        raise Error("member ABI: this Gp classifier member has " + String(_GPC_BLOCKS) + " fitted blocks")
    if block == 0:
        return MEMBER_BLOCK_SCALARS
    if block <= 3:
        return MEMBER_BLOCK_FAMILY + block - 1
    if block == 4:
        return _B_PI
    if block == 5:
        return _B_WSR
    if block == 6:
        return _B_KPARAMS
    if block == 7:
        return _B_KSCALES
    return _B_MEMBERS


def _gpc_export(slot: _Slot, call: _Call) raises:
    """A view of one fitted classifier block; X_train_ is the caller's fit
    view or the imported words. Valid until the next fit, import or free."""
    if not slot[].fitted:
        raise Error("NotFittedError: this GaussianProcessClassifier member is not fitted yet")
    if (call[].flags & MEMBER_EXPORT_OWN) != 0:
        # The classifier borrows X_train_ only (its 0 / 1 targets are owned).
        _own_views(slot)
        return
    var block = call[].block
    if (call[].flags & MEMBER_BLOCK_BY_INDEX) != 0:
        block = _gpc_block(block)
    var n = slot[].rows
    var m = slot[].members
    if block == MEMBER_BLOCK_SCALARS:
        slot[].scalars = List[Float64](length=_SCALARS, fill=0.0)
        slot[].scalars[0] = slot[].params[0]
        slot[].scalars[1] = Float64(n)
        slot[].scalars[2] = Float64(slot[].cols)
        slot[].scalars[3] = Float64(slot[].classes)
        slot[].scalars[4] = Float64(m)
        slot[].scalars[5] = slot[].params[1]
        slot[].scalars[15] = Float64(len(slot[].kernel.kinds))
        call[].dst = MemberMatrix(Int(slot[].scalars.unsafe_ptr()), 1, _SCALARS, MEMBER_F64)
    elif block == _B_X:
        call[].dst = MemberMatrix(slot[].x_fit, n, slot[].cols)
    elif block == _B_Y:
        call[].dst = MemberMatrix(Int(slot[].y01.unsafe_ptr()), m, n)
    elif block == _B_L:
        call[].dst = MemberMatrix(Int(slot[].factor.unsafe_ptr()), m * n, n)
    elif block == _B_PI:
        call[].dst = MemberMatrix(Int(slot[].pis.unsafe_ptr()), m, n)
    elif block == _B_WSR:
        call[].dst = MemberMatrix(Int(slot[].wsrs.unsafe_ptr()), m, n)
    elif block == _B_KPARAMS:
        call[].dst = MemberMatrix(Int(slot[].kernel.parameters.unsafe_ptr()), 1, len(slot[].kernel.parameters))
    elif block == _B_KSCALES:
        call[].dst = MemberMatrix(Int(slot[].kernel.scales.unsafe_ptr()), 1, len(slot[].kernel.scales))
    elif block == _B_MEMBERS:
        call[].dst = MemberMatrix(Int(slot[].member_scalars.unsafe_ptr()), m, 3, MEMBER_F64)
    else:
        raise Error("member ABI: a Gp classifier member has no fitted block " + String(block))
    call[].block = block
    call[].out_cols = _GPC_BLOCKS


def _gpc_import(slot: _Slot, call: _Call) raises:
    """Copy one fitted classifier block in; MEMBER_IMPORT_DONE completes the
    state. The scalars come first; X_train_, y_train_, L_, pi_, W_sr_ and the
    fitted kernel are required, the fits' (lml, n_iter, nb) words optional."""
    _drop_lease(slot)
    var block = call[].block
    var n = slot[].rows
    var m = slot[].members
    var count = call[].x.rows * call[].x.cols
    if block == MEMBER_BLOCK_SCALARS:
        _dense(call[].x, MEMBER_F64, "the scalars block")
        if count != _SCALARS:
            raise Error("member ABI: the Gp scalars block is sixteen float64 words")
        var words = _F64P(unsafe_from_address=call[].x.data)
        if Int(words[0]) != Int(slot[].params[0]):
            raise Error("member ABI: the Gp scalars block holds another kind")
        if Int(words[15]) != len(slot[].base.kinds):
            raise Error("member ABI: the Gp scalars block holds another kernel")
        var k = Int(words[3])
        if k < 2 or Int(words[4]) != (1 if k == 2 else k) or Int(words[1]) < 1:
            raise Error("member ABI: the Gp classifier scalars block holds no valid class count")
        slot[].clear()
        _gpc_layout(slot, Int(words[1]), Int(words[2]), k)
        slot[].kernel = slot[].base.copy()
    elif block == _B_X:
        if call[].x.rows != n or call[].x.cols != slot[].cols:
            raise Error("member ABI: the Gp X_train_ block does not match the scalars block")
        slot[].x_owned = _copy_f32(call[].x, "X_train_")
        slot[].x_fit = Int(slot[].x_owned.unsafe_ptr())
    elif block == _B_Y:
        if count != m * n:
            raise Error("member ABI: the Gp classifier y_train_ block does not match the scalars block")
        slot[].y01 = _copy_f32(call[].x, "y_train_")
    elif block == _B_L:
        if count != m * n * n:
            raise Error("member ABI: the Gp classifier L_ block does not match the scalars block")
        slot[].factor = _copy_f32(call[].x, "L_")
    elif block == _B_PI:
        if count != m * n:
            raise Error("member ABI: the Gp classifier pi_ block does not match the scalars block")
        slot[].pis = _copy_f32(call[].x, "pi_")
    elif block == _B_WSR:
        if count != m * n:
            raise Error("member ABI: the Gp classifier W_sr_ block does not match the scalars block")
        slot[].wsrs = _copy_f32(call[].x, "W_sr_")
    elif block == _B_KPARAMS:
        if count != len(slot[].kernel.parameters):
            raise Error("member ABI: the Gp kernel parameter block does not match the kernel table")
        slot[].kernel.parameters = _copy_f32(call[].x, "the kernel parameters")
    elif block == _B_KSCALES:
        if count != len(slot[].kernel.scales):
            raise Error("member ABI: the Gp kernel length-scale block does not match the kernel table")
        slot[].kernel.scales = _copy_f32(call[].x, "the kernel length scales")
    elif block == _B_MEMBERS:
        _dense(call[].x, MEMBER_F64, "the fits' scalar block")
        if count != 3 * m:
            raise Error("member ABI: the Gp classifier fit scalars block is three float64 words per fit")
        slot[].member_scalars = List[Float64](length=3 * m, fill=0.0)
        var words = _F64P(unsafe_from_address=call[].x.data)
        for i in range(3 * m):  # small-loop(three words per binary fit)
            slot[].member_scalars[i] = words[i]
    elif block != 0 or (call[].flags & MEMBER_IMPORT_DONE) == 0:
        raise Error("member ABI: a Gp classifier member imports no fitted block " + String(block))
    if (call[].flags & MEMBER_IMPORT_DONE) == 0:
        return
    if (slot[].rows < 1 or slot[].x_fit == 0 or len(slot[].y01) != m * n or len(slot[].factor) != m * n * n
            or len(slot[].pis) != m * n or len(slot[].wsrs) != m * n):
        raise Error("member ABI: a fitted Gp classifier state needs its scalars, X_train_, y_train_, L_, pi_ "
                    + "and W_sr_ blocks")
    if len(slot[].member_scalars) != 3 * m:
        slot[].member_scalars = List[Float64](length=3 * m, fill=0.0)
    _gpc_tables(slot)
    slot[].fitted = True


def _gpc(slot: _Slot) -> Bool:
    return Int(slot[].params[0]) == _KIND_GPC


# MEMBER_CAN_TEST_ROWS (both kinds): FIT tests X and the regressor's y
# (`_fit`, `_gpc_fit`: `_finite`); every query tests X, through the resident
# entry's status word or here (`_query`, `_gpc_query`). Codes are integers.
comptime _CAN_COMMON = (MEMBER_CAN_FIT | MEMBER_CAN_PREDICT | MEMBER_CAN_EXPORT | MEMBER_CAN_IMPORT
    | MEMBER_CAN_TEST_ROWS)
<<<<<<< HEAD
comptime _CAN_GPR = _CAN_COMMON | MEMBER_CAN_PREDICT_STD | MEMBER_CAN_NOTICES
comptime _CAN_GPC = _CAN_COMMON | MEMBER_CAN_PREDICT_PROBA | MEMBER_CAN_CLASSIFIER
=======
# MEMBER_BORROWS_FIT (both kinds): the fitted state reads the caller's fit X
# (and the regressor's y without normalize_y) at every query, and its export
# hands those words back themselves; only MEMBER_EXPORT_OWN (`_own_views`) or
# an import makes the state own them (LIFETIME in the module docstring).
comptime _CAN_GPR = _CAN_COMMON | MEMBER_CAN_PREDICT_STD | MEMBER_BORROWS_FIT
comptime _CAN_GPC = _CAN_COMMON | MEMBER_CAN_PREDICT_PROBA | MEMBER_CAN_CLASSIFIER | MEMBER_BORROWS_FIT
>>>>>>> origin/preinteg/wave2-20261113


def _gp_outputs(call: _Call, classifier: Bool, classes: Int) raises:
    """OUTPUTS of either kind, with or without a state: the regressor's
    predict / predict_std one float32 column; the classifier's predict one
    int32 code column, predict_proba float64 x classes. Another operation is
    refused, never sized."""
    var op = call[].flags
    if classifier:
        if op == MEMBER_OP_PREDICT:
            call[].out_cols = 1
            call[].out_dtype = MEMBER_I32
        elif op == MEMBER_OP_PREDICT_PROBA:
            call[].out_cols = classes
            call[].out_dtype = MEMBER_F64
        else:
            member_unsupported(_GPC, op)
        call[].classes = classes
        return
    if op != MEMBER_OP_PREDICT and op != MEMBER_OP_PREDICT_STD:
        member_unsupported(_GPR, op)
    call[].out_cols = 1
    call[].out_dtype = MEMBER_F32
    call[].classes = 0


def _gp_member(operation: Int, address: Int) raises -> Int:
    var call = _Call(unsafe_from_address=address)
    if operation == MEMBER_OP_CAPABILITIES:
        if call[].state == 0:
            # Kind-exact from the NEW words when given (item 17).
            if call[].params and call[].param_count >= 1:
                return _CAN_GPC if Int(_F64P(unsafe_from_address=call[].params)[0]) == _KIND_GPC else _CAN_GPR
            return _CAN_GPR | _CAN_GPC
        # A live state answers for its own kind.
        var held = _Slot(unsafe_from_address=_lookup(call[].state, False))
        return _CAN_GPC if _gpc(held) else _CAN_GPR
    if operation == MEMBER_OP_NEW:
        call[].state = _parse(call)
        return 0
    if operation == MEMBER_OP_OUTPUTS and call[].state == 0:
        # Without a state, from the NEW words' kind: the regressor answers
        # one column for predict and predict_std; the classifier one code
        # column for predict and `call.classes` for predict_proba.
        var kind = _KIND_GPR
        if call[].params and call[].param_count >= 1:
            kind = Int(_F64P(unsafe_from_address=call[].params)[0])
        var classifier = kind == _KIND_GPC
        _gp_outputs(call, classifier, call[].classes)
        return 0
    if operation == MEMBER_OP_FREE:
        _ = _lookup(call[].state, True)
        call[].state = 0
        return 0
    var slot = _Slot(unsafe_from_address=_lookup(call[].state, False))
    var classifier = _gpc(slot)
    if operation == MEMBER_OP_EXPORT and call[].block == MEMBER_BLOCK_NOTICES and (call[].flags & MEMBER_BLOCK_BY_INDEX) == 0:
        # FIT NOTICES, by name, either kind (the classifier's are empty).
        call[].dst = MemberMatrix(Int(slot[].notices.unsafe_ptr()), len(slot[].notices) // MEMBER_NOTICE_WORDS,
                                  MEMBER_NOTICE_WORDS, MEMBER_F64)
        return 0
    if operation == MEMBER_OP_OUTPUTS:
        _gp_outputs(call, classifier, slot[].classes if slot[].classes > 0 else call[].classes)
        return 0
    if classifier:
        if operation == MEMBER_OP_FIT:
            _gpc_fit(slot, call)
        elif operation == MEMBER_OP_PREDICT:
            _gpc_query(slot, call, False)
        elif operation == MEMBER_OP_PREDICT_PROBA:
            _gpc_query(slot, call, True)
        elif operation == MEMBER_OP_EXPORT:
            _gpc_export(slot, call)
        elif operation == MEMBER_OP_IMPORT:
            _gpc_import(slot, call)
        else:
            member_unsupported(_GPC, operation)
        return 0
    if operation == MEMBER_OP_FIT:
        _fit(slot, call)
    elif operation == MEMBER_OP_PREDICT:
        _query(slot, call, False)
    elif operation == MEMBER_OP_PREDICT_STD:
        _query(slot, call, True)
    elif operation == MEMBER_OP_EXPORT:
        _export(slot, call)
    elif operation == MEMBER_OP_IMPORT:
        _import(slot, call)
    else:
        member_unsupported(_GPR, operation)
    return 0


def gp_member_v2(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
    result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """The member entry of the Gp family (schema `MEMBER_ABI`)."""
    try:
        var word = _gp_member(operation, member_call_from_spans(operation, aa, na, np, nf))
        if result:
            MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = word
        return 0
    except error:
        native_descriptor_report(error, context, sink)
        return 1


def gp_member_publish(mode: Int, vendor: Int, publisher: Int) raises:
    """Publish this image's Gp member entry. One line in each GP image's
    module initialization, after its GP table is published."""
    member_publish[MEMBER_FAMILY_GP](mode, vendor, publisher, gp_member_v2)
