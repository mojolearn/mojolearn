# SPDX-License-Identifier: Apache-2.0
"""SVC and SVR as MEMBERS (core/member_abi.mojo, family
`MEMBER_FAMILY_KERNEL_SVM`), after core/linear_member.mojo.

One entry, `kernel_svm_member_v2`, published by each SVM image
(`kernel_svm_member_publish`: bindings/_mojolearn_svm.mojo and
bindings/_mojolearn_svm_host.mojo). It runs the family's existing typed RUN
(svm/svc_kernel_fit_run.mojo `svc_fit_run` / `svc_fit_finish_run`,
svm/svc_kernel_native_run.mojo `svc_query_run`, core/svr_native_api.mojo
`SvrNativeDriver`) over the entries of the `DeviceRoute` it is handed:
`entry` the image's shared SVC table (short-form solve, host-words query,
epilogues, row glue, Platt, scale_gamma's limbs, the one-vs-one layout),
`second` the SVC device table (a GPU image only: the device-X binary solve),
`third` the SVR table. `route.flags` bit 1: the
-D MOJOLEARN_CLASSICAL_NATIVE_DISPATCH B arm (the SVR request's candidate
word). No kernel is touched.

KINDS: 0 SVC (a classifier: MEMBER_CAN_CLASSIFIER), 1 SVR.

MEMBER_OP_NEW parameters (17 Float64 words): [0] kind; [1] kernel code
(svm/operation_metadata.mojo codes; 'precomputed' is not published);
[2] degree; [3] coef0; [4] gamma kind (0 a value, 1 'auto', 2 'scale');
[5] gamma (kind 0); [6] C; [7] tol; [8] epsilon (SVR); [9] max_iter;
[10] nochange_steps; [11] cache_size (MiB); [12] probability (0 / 1, SVC);
[13] decision_function_shape (0 'ovr', 1 'ovo'); [14] break_ties (0 / 1);
[15] random_state's low 32 bits, [16] its high 32 bits (0, 0 for None).
class_weight is not published (the boundary keeps the estimator's own doors).

VIEWS: X dense row-major host float32. SVC: y = int32 class codes
0 .. classes-1 (`call.classes` at fit); predict writes int32 codes;
decision_function writes float32 (binary: 1 column; 'ovo': one per pair) or
float64 ('ovr': one per class); predict_proba float64 (rows x classes).
SVR: y float32, predict float32 (one column). Sample weights: host float64
or float32, one per row. Anything else is refused in words (a device-home
view, a row or column index, a nonzero `fit_output`). Score: SVC accuracy
(unweighted; MEMBER_CAN_WEIGHTED_SCORE is not published). SVR publishes no
score: the caller predicts and runs its own metric.

FITTED-STATE BLOCKS, in index order (count 9 + 3 x machines):
 0 `MEMBER_BLOCK_SCALARS` float64 [kind, classes, gamma, machines, total
   supports, features, label0, label1, probability];
 1 `MEMBER_BLOCK_INTERCEPT` float32 1 x machines (the public intercept_);
 2 FAMILY+0 support_ int32 1 x total; 3 FAMILY+1 support_vectors_ float32
   total x features; 4 FAMILY+2 dual_coef_ float32 (1 or classes-1) x total;
 5 FAMILY+3 n_support int32 1 x (classes, or 1); 6 FAMILY+4 n_iter int32
   1 x machines; 7 FAMILY+5 the machine table float64 machines x 4 [i, j,
   bias, supports]; 8 FAMILY+6 the sigmoid parameters float64 1 x 2 x pairs
   (A, B per pair; zero rows without probability);
 then per machine q: FAMILY+16+3q its duals float32 1 x s, FAMILY+17+3q its
   support rows float32 s x features, FAMILY+18+3q its support int32 1 x s.
An import needs blocks 0, 7, 8 (with probability) and each machine's three;
the rest default to the machines' own (binary and SVR) or stay empty.

The fitted state is OWNED here, in a process-wide store of fixed chunks
(the linear member's store): the lock covers only mint, free and lookup.
Serial by nature here: the store's bookkeeping; the SVM RUN's pair and fold
loops are serial by nature (svm/svc_kernel_fit_run.mojo), and its launches
are parallel inside. The member's own row work (dtype casts, the
break_ties argmax, the accuracy count) runs in row tasks across the cores.
"""
from std.math import isnan
from std.memory import unsafe_memcpy
from std.sys.info import size_of
from core.admitted_cast_elements import admitted_cast_elements
from core.hotpath_impl import HP_F32, HP_F64, HP_I32, HP_I64
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk
from core.member_abi import (
    MemberCall, MemberMatrix, MEMBER_F32, MEMBER_F64, MEMBER_I32,
    MEMBER_OP_CAPABILITIES, MEMBER_OP_NEW, MEMBER_OP_FREE, MEMBER_OP_FIT, MEMBER_OP_PREDICT,
    MEMBER_OP_PREDICT_PROBA, MEMBER_OP_DECISION, MEMBER_OP_SCORE, MEMBER_OP_OUTPUTS,
    MEMBER_OP_EXPORT, MEMBER_OP_IMPORT,
    MEMBER_CAN_FIT, MEMBER_CAN_PREDICT, MEMBER_CAN_PREDICT_PROBA, MEMBER_CAN_DECISION, MEMBER_CAN_SCORE,
    MEMBER_CAN_EXPORT, MEMBER_CAN_IMPORT, MEMBER_CAN_SAMPLE_WEIGHT, MEMBER_CAN_CLASSIFIER,
    MEMBER_BLOCK_SCALARS, MEMBER_BLOCK_INTERCEPT, MEMBER_BLOCK_FAMILY, MEMBER_BLOCK_BY_INDEX, MEMBER_IMPORT_DONE,
    member_publish, member_call_from_spans, member_unsupported,
)
from core.member_families import MEMBER_FAMILY_KERNEL_SVM
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.shared_slot import SharedSlot, shared_slot_lock, shared_slot_unlock
from core.svc_native_api import SvcDrivers
from core.svr_native_api import SvrFitPlan, SvrFitRequest, SvrQueryRequest, SvrNativeDriver
from svm.operation_metadata import SvcOperationSettings
from svm.svc_kernel_fit_run import (
    SvcFitPlan, SvcFitResult, SvcPairFit, svc_fit_run, svc_fit_finish_run, svc_gamma_run, svc_bounds_run,
    SVC_FIT_PAIR_LABELS,
)
from svm.svc_kernel_native_run import SvcMachine, SvcQueryPlan, svc_query_run, SVC_EPILOGUE_NONE, SVC_KERNEL_PRECOMPUTED

comptime _CHUNK = 256
comptime _INDEX_SPAN = 16777216
comptime _PARAMS = 17
comptime _KIND_SVC = 0
comptime _KIND_SVR = 1
# `svc_pair_epilogue`'s per-row modes (svm/impl/svc_rows.mojo; the same
# words as bindings/svc_kernel_rows_unpack.mojo _EPI_*: plain words here so
# this RUN file imports no binding).
comptime _EPI_OVO = 0
comptime _EPI_OVR = 1
comptime _EPI_VOTES = 2
comptime _EPI_PROBA = 3
comptime _EPI_BINARY = 5
# The fixed blocks before the machines' own (docstring order).
comptime _FIXED_BLOCKS = 9
comptime _B_SUPPORT = MEMBER_BLOCK_FAMILY
comptime _B_VECTORS = MEMBER_BLOCK_FAMILY + 1
comptime _B_DUAL = MEMBER_BLOCK_FAMILY + 2
comptime _B_N_SUPPORT = MEMBER_BLOCK_FAMILY + 3
comptime _B_N_ITER = MEMBER_BLOCK_FAMILY + 4
comptime _B_MACHINES = MEMBER_BLOCK_FAMILY + 5
comptime _B_PROB = MEMBER_BLOCK_FAMILY + 6
comptime _B_MACHINE0 = MEMBER_BLOCK_FAMILY + 16
comptime _SCALARS = 9


def _f32(values: List[Float32]) -> Int:
    return Int(values.unsafe_ptr()) if len(values) else 0


def _i32(values: List[Int32]) -> Int:
    return Int(values.unsafe_ptr()) if len(values) else 0


def _f64(values: List[Float64]) -> Int:
    return Int(values.unsafe_ptr()) if len(values) else 0


def _round_f32(value: Float64) -> Float64:
    return Float64(Float32(value))


struct KernelSvmMemberSlot(Copyable, Movable, Defaultable):
    var generation: Int
    var live: Bool
    var params: List[Float64]
    var fitted: Bool
    var classes: Int
    var features: Int
    var gamma: Float64
    var label0: Float64
    var label1: Float64
    var probability: Bool
    # One fitted machine per class pair (SVC one-vs-one), else one.
    var machines: List[SvcPairFit]
    # The public layout (support_, support_vectors_, dual_coef_, n_support_,
    # intercept_, n_iter_) and the sigmoid parameters, A and B per pair.
    var support: List[Int32]
    var vectors: List[Float32]
    var dual: List[Float32]
    var n_support: List[Int32]
    var intercept: List[Float32]
    var n_iter: List[Int32]
    var prob: List[Float64]
    var scalars: List[Float64]
    var table: List[Float64]

    def __init__(out self):
        self.generation = 0
        self.live = False
        self.params = List[Float64]()
        self.fitted = False
        self.classes = 0
        self.features = 0
        self.gamma = 1.0
        self.label0 = 0.0
        self.label1 = 1.0
        self.probability = False
        self.machines = List[SvcPairFit]()
        self.support = List[Int32]()
        self.vectors = List[Float32]()
        self.dual = List[Float32]()
        self.n_support = List[Int32]()
        self.intercept = List[Float32]()
        self.n_iter = List[Int32]()
        self.prob = List[Float64]()
        self.scalars = List[Float64]()
        self.table = List[Float64]()

    def clear(mut self):
        self.fitted = False
        self.classes = 0
        self.features = 0
        self.machines = List[SvcPairFit]()
        self.support = List[Int32]()
        self.vectors = List[Float32]()
        self.dual = List[Float32]()
        self.n_support = List[Int32]()
        self.intercept = List[Float32]()
        self.n_iter = List[Int32]()
        self.prob = List[Float64]()
        self.scalars = List[Float64]()
        self.table = List[Float64]()


struct KernelSvmMemberStore(Defaultable, Movable):
    var synchronization: List[Int64]
    var chunks: List[List[KernelSvmMemberSlot]]
    var free: List[Int]
    var count: Int

    def __init__(out self):
        # Side-effect-free even when the process slot's initializers race.
        self.synchronization = List[Int64](length=2, fill=0)
        self.chunks = List[List[KernelSvmMemberSlot]]()
        self.free = List[Int]()
        self.count = 0


comptime _STORE = SharedSlot[KernelSvmMemberStore, "MojolearnKernelSvmMemberStoreV2", 2]


def _slot_address(index: Int) raises -> Int:
    # Under the store's lock. A chunk's buffer is allocated once and never
    # grows, so the address stays valid after the lock is released.
    var store = _STORE.get_or_create_ptr()
    return Int(store[].chunks[index // _CHUNK].unsafe_ptr()) + (index % _CHUNK) * size_of[KernelSvmMemberSlot]()


def _mint(var params: List[Float64]) raises -> Int:
    var store = _STORE.get_or_create_ptr()
    var lock = rebind[MutPointer[Int64, MutUntrackedOrigin]](store[].synchronization.unsafe_ptr())
    shared_slot_lock(lock)
    try:
        var index = -1
        if len(store[].free) > 0:
            index = store[].free.pop()
        else:
            if store[].count >= _INDEX_SPAN - 1:
                raise Error("RuntimeError: mojolearn: too many live kernel SVM member states")
            if store[].count == len(store[].chunks) * _CHUNK:
                store[].chunks.append(List[KernelSvmMemberSlot](length=_CHUNK, fill=KernelSvmMemberSlot()))
            index = store[].count
            store[].count += 1
        var slot = MutPointer[KernelSvmMemberSlot, MutUntrackedOrigin](unsafe_from_address=_slot_address(index))
        slot[].generation += 1
        slot[].live = True
        slot[].params = params^
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
        var slot = MutPointer[KernelSvmMemberSlot, MutUntrackedOrigin](unsafe_from_address=address)
        if not slot[].live or slot[].generation != handle // _INDEX_SPAN:
            raise Error("NotFittedError: this member state was released or never created")
        if release:
            slot[].live = False
            slot[].clear()
            slot[].params = List[Float64]()
            store[].free.append(index)
        return address
    finally:
        shared_slot_unlock(lock)


comptime _Slot = MutPointer[KernelSvmMemberSlot, MutUntrackedOrigin]
comptime _Call = MutPointer[MemberCall, MutUntrackedOrigin]


def _svc(slot: _Slot) -> Bool:
    return Int(slot[].params[0]) == _KIND_SVC


def _name(slot: _Slot) -> String:
    return String("SVC") if _svc(slot) else String("SVR")


def _settings(slot: _Slot, features: Int, gamma: Float64) -> SvcOperationSettings:
    var p = slot[].params.copy()
    return SvcOperationSettings(Int(p[1]), Int(p[2]), Int(p[9]), Int(p[10]), features, gamma, p[6], p[7], p[3], p[11])


def _drivers(call: _Call) -> SvcDrivers:
    var route = call[].route
    return SvcDrivers(route.entry.call, route.second.call if route.has_second else route.entry.call, route.has_second)


def _whole(call: _Call, estimator: String) raises:
    if call[].row_index.rows > 0 or call[].column_index.rows > 0:
        raise Error("TypeError: a " + estimator + " member reads whole host views (no row or column index)")


def _dense(view: MemberMatrix, dtype: Int, what: String, estimator: String) raises:
    if not view.dense_host(dtype):
        var word = String("float32") if dtype == MEMBER_F32 else (String("float64") if dtype == MEMBER_F64 else String("int32"))
        raise Error("TypeError: a " + estimator + " member reads " + what + " as a dense host " + word + " view")


def _cast(source: Int, source_code: Int, dst: Int, dst_code: Int, count: Int) raises:
    """An exact element conversion in row tasks (core/admitted_cast_elements.mojo)."""
    if count > 0 and admitted_cast_elements(source, source_code, dst, dst_code, count) != 0:
        raise Error("ValueError: a member view holds a value its element type cannot convert")


def _weights(call: _Call, rows: Int, estimator: String, mut held: List[Float64]) raises -> Int:
    """The float64 sample weights the bounds read (0: none)."""
    if call[].weights.rows == 0:
        return 0
    if call[].weights.rows != rows or call[].weights.cols != 1:
        raise Error("ValueError: sample_weight and X have different numbers of rows")
    if call[].weights.dense_host(MEMBER_F64):
        return call[].weights.data
    _dense(call[].weights, MEMBER_F32, "sample_weight", estimator)
    held = List[Float64](length=rows, fill=0.0)
    _cast(call[].weights.data, HP_F32, Int(held.unsafe_ptr()), HP_F64, rows)
    return Int(held.unsafe_ptr())


def _flagged(first: Float64, second: Float64, estimator: String) raises:
    # The device entry's finite and range words of X (admit_x's wording).
    if Int(first) != 0 or Int(second) != 0:
        raise Error("ValueError: Input X contains NaN or infinity (" + estimator + ").")


def _fit_svc(slot: _Slot, call: _Call) raises:
    var rows = call[].x.rows
    var cols = call[].x.cols
    var classes = call[].classes
    _dense(call[].y, MEMBER_I32, "y (the class codes)", "SVC")
    if call[].y.rows != rows or call[].y.cols != 1:
        raise Error("ValueError: X and y have different numbers of rows")
    if classes < 2:
        raise Error("ValueError: mojolearn SVC: y has " + String(classes) + " class; at least two are needed")
    var drivers = _drivers(call)
    var p = slot[].params.copy()
    var probability = p[12] != 0.0
    var gamma_kind = Int(p[4])
    var held = List[Float64]()
    var sample_weights = _weights(call, rows, "SVC", held)
    # The solver's float32 labels of a binary machine: the codes 0.0 / 1.0
    # (the estimator's own non-numeric label route; codes keep the sorted
    # class order, so the solver's +1 is class 1 either way).
    var labels = List[Float32](length=rows if classes == 2 else 0, fill=Float32(0))
    if classes == 2:
        _cast(call[].y.data, HP_I32, _f32(labels), HP_F32, rows)
    var device_x = drivers.has_device and classes == 2 and not probability
    var plan = SvcFitPlan(_settings(slot, cols, 1.0))
    if device_x:
        if gamma_kind == 1:
            plan.gamma = 1.0 / Float64(cols)
        elif gamma_kind == 2:
            plan.gamma = 1.0
            plan.scale_gamma = True
        else:
            plan.gamma = p[5]
    else:
        if gamma_kind == 2 and rows * cols == 0:
            raise Error("ValueError: gamma='scale' needs at least one cell of X")
        plan.gamma_kind = gamma_kind
        plan.gamma = p[5]
    plan.x = call[].x.data
    plan.rows = rows
    plan.cols = cols
    plan.cells = rows * cols
    plan.device_form = device_x
    plan.classes = classes
    plan.labels = _f32(labels)
    plan.codes = call[].y.data
    plan.weighted = sample_weights != 0
    plan.sample_weights = sample_weights
    plan.class_weights = 0
    plan.probability = probability
    plan.seed = (UInt64(Int(p[16])) << UInt64(32)) | UInt64(Int(p[15]))
    var result = SvcFitResult()
    svc_fit_run(drivers, plan, result)
    if classes > 2:
        if result.status == SVC_FIT_PAIR_LABELS:
            raise Error("RuntimeError: mojolearn SVC: the solver's label pair for class codes (" + String(result.bad_i)
                + ", " + String(result.bad_j) + ") is not (0.0, 1.0); the class mapping cannot be trusted")
    else:
        if device_x:
            _flagged(result.solved.info[5], result.solved.info[6], "SVC")
        if _round_f32(result.solved.info[3]) != 0.0 or _round_f32(result.solved.info[4]) != 1.0:
            raise Error("RuntimeError: mojolearn SVC: the solver's label pair is not the codes (0, 1); "
                + "the class mapping cannot be trusted")
    slot[].clear()
    var total = result.total if classes > 2 else result.supports
    slot[].vectors = List[Float32](length=total * cols, fill=Float32(0))
    svc_fit_finish_run(drivers, plan, _f32(slot[].vectors), result)
    slot[].classes = classes
    slot[].features = cols
    slot[].gamma = result.gamma
    slot[].probability = probability
    if probability:
        var pairs = classes * (classes - 1) // 2
        slot[].prob = List[Float64](length=2 * pairs, fill=0.0)
        for q in range(pairs):  # driver-loop: over the class pairs (their sigmoid parameters)
            slot[].prob[2 * q] = result.prob_a[q]
            slot[].prob[2 * q + 1] = result.prob_b[q]
    if classes > 2:
        # The pair machines move out of the result element by element (no
        # buffer copy and no partial move of the result).
        var popped = List[SvcPairFit](capacity=len(result.pairs))
        while len(result.pairs) > 0:  # driver-loop: over the class pairs (moved, not copied)
            popped.append(result.pairs.pop())
        while len(popped) > 0:  # driver-loop: over the class pairs (back in pair order)
            slot[].machines.append(popped.pop())
        slot[].support = List[Int32](length=total, fill=Int32(0))
        if total > 0:
            unsafe_memcpy(dest=slot[].support.unsafe_ptr(), src=result.support.unsafe_ptr(), count=total)
        slot[].dual = List[Float32](length=(classes - 1) * total, fill=Float32(0))
        if total > 0:
            unsafe_memcpy(dest=slot[].dual.unsafe_ptr(), src=result.dual_all.unsafe_ptr(), count=(classes - 1) * total)
        slot[].n_support = result.n_support.copy()
        var pairs = classes * (classes - 1) // 2
        slot[].intercept = List[Float32](length=pairs, fill=Float32(0))
        slot[].n_iter = List[Int32](length=pairs, fill=Int32(0))
        for q in range(pairs):  # driver-loop: over the class pairs (their public intercept and iterations)
            slot[].intercept[q] = result.intercept[q]
            slot[].n_iter[q] = result.n_iter[q]
        slot[].label0 = 0.0
        slot[].label1 = 1.0
    else:
        var n_sv = result.supports
        var dual = List[Float32](length=n_sv, fill=Float32(0))
        var support = List[Int32](length=n_sv, fill=Int32(0))
        if n_sv > 0:
            unsafe_memcpy(dest=dual.unsafe_ptr(), src=result.solved.dual.unsafe_ptr(), count=n_sv)
            unsafe_memcpy(dest=support.unsafe_ptr(), src=result.solved.support.unsafe_ptr(), count=n_sv)
            unsafe_memcpy(dest=slot[].vectors.unsafe_ptr(), src=result.solved.vectors.unsafe_ptr(), count=n_sv * cols)
        var bias = _round_f32(result.solved.info[0])
        slot[].machines.append(SvcPairFit(0, 1, n_sv, cols, dual^, slot[].vectors.copy(), support^, bias,
            Int(result.solved.info[2])))
        slot[].label0 = _round_f32(result.solved.info[3])
        slot[].label1 = _round_f32(result.solved.info[4])
        _binary_layout(slot)
    slot[].fitted = True
    _ = labels^
    _ = held^


def _binary_layout(slot: _Slot):
    """The public layout of one machine (binary SVC, SVR): its own arrays."""
    var machine = slot[].machines[0].copy()
    slot[].support = machine.support.copy()
    slot[].vectors = machine.vectors.copy()
    slot[].dual = machine.dual.copy()
    slot[].n_support = List[Int32](length=1, fill=Int32(machine.supports))
    slot[].intercept = List[Float32](length=1, fill=Float32(machine.bias))
    slot[].n_iter = List[Int32](length=1, fill=Int32(machine.n_iter))


def _fit_svr(slot: _Slot, call: _Call) raises:
    var rows = call[].x.rows
    var cols = call[].x.cols
    _dense(call[].y, MEMBER_F32, "y", "SVR")
    if call[].y.rows != rows or call[].y.cols != 1:
        raise Error("ValueError: X and y have different numbers of rows")
    var drivers = _drivers(call)
    var route = call[].route
    if not route.has_third:
        raise Error("ImportError: mojolearn: the SVR member needs the SVR table on its route")
    var p = slot[].params.copy()
    var gamma_kind = Int(p[4])
    var held = List[Float64]()
    var sample_weights = _weights(call, rows, "SVR", held)
    var candidate = (route.flags & 1) != 0
    var device_x = drivers.has_device
    var gamma = p[5]
    var scale_gamma = False
    if device_x:
        if gamma_kind == 1:
            gamma = 1.0 / Float64(cols)
        elif gamma_kind == 2:
            gamma = 1.0
            scale_gamma = True
    elif gamma_kind == 2 and rows * cols == 0:
        raise Error("ValueError: gamma='scale' needs at least one cell of X")
    var bounds = List[Float32](length=rows if sample_weights != 0 else 0, fill=Float32(0))
    var plan = SvrFitPlan(rows, cols)
    var device_form = device_x and not candidate
    var request = SvrFitRequest(call[].x.data, call[].y.data, Int(plan.dual.unsafe_ptr()),
        Int(plan.support.unsafe_ptr()), Int(plan.vectors.unsafe_ptr()), Int(plan.info.unsafe_ptr()), _f32(bounds),
        rows, cols, Int(p[1]), Int(p[9]), Int(p[10]), Int(p[2]), -1, 0, -1, device_form, scale_gamma, candidate,
        gamma, p[6], p[8], p[7], p[3])
    if not device_x:
        request.gamma = svc_gamma_run(drivers, gamma_kind, p[5], call[].x.data, rows * cols, cols)
    if sample_weights != 0:
        svc_bounds_run(drivers, rows, sample_weights, 0, 0, 0, p[6], _f32(bounds))
    var count = SvrNativeDriver(route.third.call).fit(request)
    gamma = request.gamma
    if device_form:
        _flagged(plan.info[3], plan.info[4], "SVR")
        gamma = plan.info[5]
    slot[].clear()
    var dual = List[Float32](length=count, fill=Float32(0))
    var support = List[Int32](length=count, fill=Int32(0))
    var vectors = List[Float32](length=count * cols, fill=Float32(0))
    if count > 0:
        unsafe_memcpy(dest=dual.unsafe_ptr(), src=plan.dual.unsafe_ptr(), count=count)
        unsafe_memcpy(dest=support.unsafe_ptr(), src=plan.support.unsafe_ptr(), count=count)
        unsafe_memcpy(dest=vectors.unsafe_ptr(), src=plan.vectors.unsafe_ptr(), count=count * cols)
    slot[].machines.append(SvcPairFit(0, 0, count, cols, dual^, vectors^, support^, _round_f32(plan.info[0]),
        Int(plan.info[2])))
    slot[].features = cols
    slot[].gamma = gamma
    _binary_layout(slot)
    slot[].fitted = True
    _ = bounds^
    _ = held^


def _fit(slot: _Slot, call: _Call) raises:
    var estimator = _name(slot)
    _whole(call, estimator)
    if call[].fit_output != 0:
        member_unsupported(estimator, call[].fit_output)
    _dense(call[].x, MEMBER_F32, "X", estimator)
    if _svc(slot):
        _fit_svc(slot, call)
    else:
        _fit_svr(slot, call)


def _fitted(slot: _Slot, call: _Call) raises:
    var estimator = _name(slot)
    if not slot[].fitted:
        raise Error("NotFittedError: this " + estimator + " member is not fitted yet")
    _whole(call, estimator)
    _dense(call[].x, MEMBER_F32, "X", estimator)
    if call[].x.cols != slot[].features:
        raise Error("ValueError: X has " + String(call[].x.cols) + " features, fit saw " + String(slot[].features))


def _outputs(slot: _Slot, operation: Int) -> Int:
    """Columns of `dst` per row for `operation` (MEMBER_OP_OUTPUTS)."""
    var k = slot[].classes
    if not _svc(slot) or operation == MEMBER_OP_PREDICT:
        return 1
    if operation == MEMBER_OP_PREDICT_PROBA:
        return k
    if k <= 2:
        return 1
    return k * (k - 1) // 2 if slot[].params[13] != 0.0 else k


def _svc_plan(slot: _Slot, call: _Call, settings: SvcOperationSettings, mut status: List[Int32],
              mut codes: List[Int32]) -> SvcQueryPlan:
    var plan = SvcQueryPlan(settings)
    plan.x = call[].x.data
    plan.rows = call[].x.rows
    plan.cols = call[].x.cols
    plan.status = Int(status.unsafe_ptr())
    plan.label0 = slot[].label0
    plan.label1 = slot[].label1
    for m in range(len(slot[].machines)):  # driver-loop: over the fitted machines (their descriptors)
        var n = slot[].machines[m].supports
        plan.machines.append(SvcMachine(_f32(slot[].machines[m].dual) if n > 0 else 0,
            _f32(slot[].machines[m].vectors) if n > 0 else 0, n, slot[].machines[m].bias, 0, 0))
    var pairs = len(slot[].machines)
    codes = List[Int32](length=max(1, 2 * pairs), fill=Int32(0))
    if slot[].classes > 2:
        for q in range(pairs):  # driver-loop: over the class pairs (their two class codes)
            codes[2 * q] = Int32(slot[].machines[q].i)
            codes[2 * q + 1] = Int32(slot[].machines[q].j)
    else:
        codes[1] = Int32(1)
    plan.pairs = pairs
    plan.pair_codes = Int(codes.unsafe_ptr())
    plan.classes = slot[].classes
    return plan^


def _argmax_rows(scores: Int, rows: Int, cols: Int, dst: Int) raises:
    """break_ties: the row argmax of the 'ovr' scores as int32 codes (the
    first maximum; a NaN is taken where it first appears, numpy's rule)."""
    var tasks = host_predict_task_count_work(rows, rows * cols)
    var chunk = host_predict_chunk(rows, max(tasks, 1))
    var source = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=scores)
    var dst_ptr = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=dst)

    def _rows(t: Int) {imm source, imm dst_ptr, imm rows, imm cols, imm chunk}:
        var lo = t * chunk
        var hi = min(rows, lo + chunk)
        for r in range(lo, hi):
            var best = 0
            var value = source[r * cols]
            if not isnan(value):
                for c in range(1, cols):
                    var v = source[r * cols + c]
                    if isnan(v):
                        best = c
                        break
                    if v > value:
                        best = c
                        value = v
            dst_ptr.unsafe_store(r, Int32(best))

    if rows > 0:
        host_parallelize(_rows, max(tasks, 1))


def _query_svc(slot: _Slot, call: _Call, operation: Int, dst: Int) raises:
    """One SVC inference into `dst` (predict: int32 codes; decision and
    proba: the documented float type)."""
    var rows = call[].x.rows
    var k = slot[].classes
    var binary = k <= 2
    if operation == MEMBER_OP_PREDICT_PROBA and not slot[].probability:
        raise Error("AttributeError: mojolearn SVC: predict_proba is not available when probability=False "
            + "(scikit-learn's rule); fit with probability=True")
    var status = List[Int32](length=2, fill=Int32(0))
    var codes = List[Int32]()
    var plan = _svc_plan(slot, call, _settings(slot, slot[].features, slot[].gamma), status, codes)
    var pairs = len(slot[].machines)
    var decisions = List[Float32]()
    var wide = List[Int64]()
    var scores = List[Float64]()
    var argmax = False
    if binary and operation == MEMBER_OP_DECISION:
        plan.decisions = dst
    else:
        decisions = List[Float32](length=max(1, pairs * rows), fill=Float32(0))
        plan.decisions = _f32(decisions)
        if operation == MEMBER_OP_PREDICT:
            if binary:
                plan.predict_class = True
                plan.epilogue = _EPI_BINARY
                plan.epilogue_label1 = slot[].label1
            elif slot[].params[14] != 0.0:
                plan.epilogue = _EPI_OVR
                argmax = True
            else:
                plan.epilogue = _EPI_VOTES
            if argmax:
                scores = List[Float64](length=max(1, rows * k), fill=0.0)
                plan.dst = _f64(scores)
            else:
                wide = List[Int64](length=max(1, rows), fill=Int64(0))
                plan.dst = Int(wide.unsafe_ptr())
        elif operation == MEMBER_OP_PREDICT_PROBA:
            plan.epilogue = _EPI_PROBA
            # The sigmoid parameters (A, B per pair), read by the slot in place.
            if rows > 0 and len(slot[].prob) > 0:
                plan.ab = _f64(slot[].prob)
            plan.dst = dst
        else:
            plan.epilogue = _EPI_OVO if slot[].params[13] != 0.0 else _EPI_OVR
            plan.dst = dst
    if not svc_query_run(_drivers(call), plan):
        raise Error("ValueError: Input X contains NaN or infinity (SVC).")
    if operation == MEMBER_OP_PREDICT:
        if argmax:
            _argmax_rows(_f64(scores), rows, k, dst)
        else:
            _cast(Int(wide.unsafe_ptr()), HP_I64, dst, HP_I32, rows)
    _ = status^
    _ = codes^
    _ = decisions^
    _ = wide^
    _ = scores^


def _predict(slot: _Slot, call: _Call, operation: Int) raises:
    _fitted(slot, call)
    var rows = call[].x.rows
    var estimator = _name(slot)
    var cols = _outputs(slot, operation)
    var dtype = MEMBER_F32
    if _svc(slot):
        if operation == MEMBER_OP_PREDICT:
            dtype = MEMBER_I32
        elif operation == MEMBER_OP_PREDICT_PROBA or (slot[].classes > 2 and slot[].params[13] == 0.0):
            dtype = MEMBER_F64
    _dense(call[].dst, dtype, "the output buffer", estimator)
    if call[].dst.rows != rows or call[].dst.cols != cols:
        raise Error("ValueError: the output buffer holds " + String(cols) + " value(s) per row of X")
    if _svc(slot):
        _query_svc(slot, call, operation, call[].dst.data)
        return
    var route = call[].route
    if not route.has_third:
        raise Error("ImportError: mojolearn: the SVR member needs the SVR table on its route")
    var p = slot[].params.copy()
    var status = List[Int32](length=2, fill=Int32(0))
    var n = slot[].machines[0].supports
    var request = SvrQueryRequest(call[].x.data, _f32(slot[].machines[0].dual) if n > 0 else 0,
        _f32(slot[].machines[0].vectors) if n > 0 else 0, call[].dst.data, Int(status.unsafe_ptr()), rows,
        slot[].features, n, Int(p[1]), Int(p[2]), 0, 0, -1, 0, -1, slot[].machines[0].bias, slot[].gamma, p[11], p[3])
    _ = SvrNativeDriver(route.third.call).query(request, False)
    _flagged(Float64(status[0]), Float64(status[1]), "SVR")


def _score(slot: _Slot, call: _Call) raises:
    """SVC accuracy: the matched codes over the rows (an exact count; the
    row tasks' partial counts sum to the same integer in any order)."""
    if not _svc(slot):
        member_unsupported("SVR", MEMBER_OP_SCORE)
    _fitted(slot, call)
    var rows = call[].x.rows
    _dense(call[].y, MEMBER_I32, "y (the class codes)", "SVC")
    if call[].y.rows != rows or call[].y.cols != 1:
        raise Error("ValueError: mojolearn: y must be 1-D with one entry per row of X")
    if call[].weights.rows > 0:
        raise Error("NotImplementedError: mojolearn: a weighted score of an SVC member is not published; "
            + "the caller predicts and runs its own weighted metric")
    var predicted = List[Int32](length=max(1, rows), fill=Int32(0))
    _query_svc(slot, call, MEMBER_OP_PREDICT, _i32(predicted))
    var tasks = max(host_predict_task_count_work(rows, rows), 1)
    var chunk = host_predict_chunk(rows, tasks)
    var counts = List[Int](length=tasks, fill=0)
    var count_ptr = rebind[MutPointer[Int, MutUntrackedOrigin]](counts.unsafe_ptr())
    var got = rebind[MutPointer[Int32, MutUntrackedOrigin]](predicted.unsafe_ptr())
    var want = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=call[].y.data)

    def _count(t: Int) {imm got, imm want, imm count_ptr, imm rows, imm chunk}:
        var lo = t * chunk
        var hi = min(rows, lo + chunk)
        var matched = 0
        for r in range(lo, hi):
            if got[r] == want[r]:
                matched += 1
        count_ptr.unsafe_store(t, matched)

    if rows > 0:
        host_parallelize(_count, tasks)
    var matched = 0
    for t in range(tasks):  # driver-loop: over the row tasks' counts (at most one per core)
        matched += counts[t]
    call[].value = Float64(matched) / Float64(rows) if rows > 0 else 0.0
    _ = predicted^
    _ = counts^


def _block_view(slot: _Slot, block: Int) raises -> MemberMatrix:
    var k = slot[].classes
    var machines = len(slot[].machines)
    var total = len(slot[].support)
    if block == MEMBER_BLOCK_SCALARS:
        slot[].scalars = List[Float64](length=_SCALARS, fill=0.0)
        slot[].scalars[0] = slot[].params[0]
        slot[].scalars[1] = Float64(k)
        slot[].scalars[2] = slot[].gamma
        slot[].scalars[3] = Float64(machines)
        slot[].scalars[4] = Float64(total)
        slot[].scalars[5] = Float64(slot[].features)
        slot[].scalars[6] = slot[].label0
        slot[].scalars[7] = slot[].label1
        slot[].scalars[8] = Float64(1.0) if slot[].probability else Float64(0.0)
        return MemberMatrix(_f64(slot[].scalars), 1, _SCALARS, MEMBER_F64)
    if block == MEMBER_BLOCK_INTERCEPT:
        return MemberMatrix(_f32(slot[].intercept), 1, len(slot[].intercept))
    if block == _B_SUPPORT:
        return MemberMatrix(_i32(slot[].support), 1, total, MEMBER_I32)
    if block == _B_VECTORS:
        return MemberMatrix(_f32(slot[].vectors), total, slot[].features)
    if block == _B_DUAL:
        var dual_rows = k - 1 if k > 2 else 1
        return MemberMatrix(_f32(slot[].dual), dual_rows, total)
    if block == _B_N_SUPPORT:
        return MemberMatrix(_i32(slot[].n_support), 1, len(slot[].n_support), MEMBER_I32)
    if block == _B_N_ITER:
        return MemberMatrix(_i32(slot[].n_iter), 1, len(slot[].n_iter), MEMBER_I32)
    if block == _B_MACHINES:
        slot[].table = List[Float64](length=4 * machines, fill=0.0)
        for m in range(machines):  # driver-loop: over the fitted machines (their table rows)
            slot[].table[4 * m] = Float64(slot[].machines[m].i)
            slot[].table[4 * m + 1] = Float64(slot[].machines[m].j)
            slot[].table[4 * m + 2] = slot[].machines[m].bias
            slot[].table[4 * m + 3] = Float64(slot[].machines[m].supports)
        return MemberMatrix(_f64(slot[].table), machines, 4, MEMBER_F64)
    if block == _B_PROB:
        return MemberMatrix(_f64(slot[].prob), 1 if len(slot[].prob) else 0, len(slot[].prob), MEMBER_F64)
    var q = (block - _B_MACHINE0) // 3
    if block >= _B_MACHINE0 and q < machines:
        var n = slot[].machines[q].supports
        var part = (block - _B_MACHINE0) % 3
        if part == 0:
            return MemberMatrix(_f32(slot[].machines[q].dual), 1, n)
        if part == 1:
            return MemberMatrix(_f32(slot[].machines[q].vectors), n, slot[].machines[q].vector_cols)
        return MemberMatrix(_i32(slot[].machines[q].support), 1, n, MEMBER_I32)
    raise Error("member ABI: a kernel SVM member has no fitted block " + String(block))


def _export(slot: _Slot, call: _Call) raises:
    """A view of one fitted block. The slot owns the words; they stay valid
    until this state's next fit, import or free."""
    if not slot[].fitted:
        raise Error("NotFittedError: this " + _name(slot) + " member is not fitted yet")
    var count = _FIXED_BLOCKS + 3 * len(slot[].machines)
    var block = call[].block
    if (call[].flags & MEMBER_BLOCK_BY_INDEX) != 0:
        if block < 0 or block >= count:
            raise Error("member ABI: this kernel SVM member has " + String(count) + " fitted blocks")
        if block == 0:
            block = MEMBER_BLOCK_SCALARS
        elif block == 1:
            block = MEMBER_BLOCK_INTERCEPT
        elif block < _FIXED_BLOCKS:
            block = MEMBER_BLOCK_FAMILY + block - 2
        else:
            block = _B_MACHINE0 + block - _FIXED_BLOCKS
    call[].dst = _block_view(slot, block)
    call[].block = block
    call[].out_cols = count


def _copy_f32(view: MemberMatrix, what: String) raises -> List[Float32]:
    if view.rows == 0:
        return List[Float32]()
    _dense(view, MEMBER_F32, what, "kernel SVM")
    var count = view.rows * view.cols
    var words = List[Float32](length=count, fill=Float32(0))
    unsafe_memcpy(dest=words.unsafe_ptr(), src=MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=view.data),
        count=count)
    return words^


def _copy_i32(view: MemberMatrix, what: String) raises -> List[Int32]:
    if view.rows == 0:
        return List[Int32]()
    _dense(view, MEMBER_I32, what, "kernel SVM")
    var count = view.rows * view.cols
    var words = List[Int32](length=count, fill=Int32(0))
    unsafe_memcpy(dest=words.unsafe_ptr(), src=MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=view.data),
        count=count)
    return words^


def _copy_f64(view: MemberMatrix, what: String) raises -> List[Float64]:
    if view.rows == 0:
        return List[Float64]()
    _dense(view, MEMBER_F64, what, "kernel SVM")
    var count = view.rows * view.cols
    var words = List[Float64](length=count, fill=0.0)
    unsafe_memcpy(dest=words.unsafe_ptr(), src=MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=view.data),
        count=count)
    return words^


def _import(slot: _Slot, call: _Call) raises:
    """Copy one fitted block in; MEMBER_IMPORT_DONE completes the state.
    The machine table (block FAMILY+5) comes before the machines' own."""
    var block = call[].block
    if block == MEMBER_BLOCK_SCALARS:
        var words = _copy_f64(call[].x, "the scalars block")
        if len(words) != _SCALARS or Int(words[0]) != Int(slot[].params[0]):
            raise Error("member ABI: the kernel SVM scalars block is nine float64 words of this member's kind")
        slot[].clear()
        slot[].classes = Int(words[1])
        slot[].gamma = words[2]
        slot[].features = Int(words[5])
        slot[].label0 = words[6]
        slot[].label1 = words[7]
        slot[].probability = words[8] != 0.0
        slot[].scalars = words^
    elif block == MEMBER_BLOCK_INTERCEPT:
        slot[].intercept = _copy_f32(call[].x, "intercept_")
    elif block == _B_SUPPORT:
        slot[].support = _copy_i32(call[].x, "support_")
    elif block == _B_VECTORS:
        slot[].vectors = _copy_f32(call[].x, "support_vectors_")
    elif block == _B_DUAL:
        slot[].dual = _copy_f32(call[].x, "dual_coef_")
    elif block == _B_N_SUPPORT:
        slot[].n_support = _copy_i32(call[].x, "n_support_")
    elif block == _B_N_ITER:
        slot[].n_iter = _copy_i32(call[].x, "n_iter_")
    elif block == _B_PROB:
        slot[].prob = _copy_f64(call[].x, "the sigmoid parameters")
    elif block == _B_MACHINES:
        var table = _copy_f64(call[].x, "the machine table")
        if call[].x.cols != 4:
            raise Error("member ABI: the kernel SVM machine table has four words per machine")
        slot[].machines = List[SvcPairFit]()
        for m in range(call[].x.rows):  # driver-loop: over the fitted machines (empty until their blocks)
            slot[].machines.append(SvcPairFit(Int(table[4 * m]), Int(table[4 * m + 1]), Int(table[4 * m + 3]),
                slot[].features, List[Float32](), List[Float32](), List[Int32](), table[4 * m + 2], 0))
    elif block >= _B_MACHINE0 and (block - _B_MACHINE0) // 3 < len(slot[].machines):
        var q = (block - _B_MACHINE0) // 3
        var part = (block - _B_MACHINE0) % 3
        if part == 0:
            slot[].machines[q].dual = _copy_f32(call[].x, "a machine's duals")
        elif part == 1:
            slot[].machines[q].vectors = _copy_f32(call[].x, "a machine's support rows")
        else:
            slot[].machines[q].support = _copy_i32(call[].x, "a machine's support")
    elif block != 0 or (call[].flags & MEMBER_IMPORT_DONE) == 0:
        raise Error("member ABI: a kernel SVM member has no fitted block " + String(block))
    if (call[].flags & MEMBER_IMPORT_DONE) == 0:
        return
    if len(slot[].scalars) != _SCALARS or len(slot[].machines) == 0:
        raise Error("member ABI: a fitted kernel SVM state needs its scalars and machine table blocks")
    for m in range(len(slot[].machines)):  # driver-loop: over the fitted machines (completeness)
        var n = slot[].machines[m].supports
        if len(slot[].machines[m].dual) != n or len(slot[].machines[m].vectors) != n * slot[].features:
            raise Error("member ABI: kernel SVM machine " + String(m) + " lacks its duals or support rows")
    var k = slot[].classes
    if _svc(slot) and slot[].probability and len(slot[].prob) != k * (k - 1):
        raise Error("member ABI: a probability SVC state needs its sigmoid parameters block")
    if len(slot[].machines) == 1 and len(slot[].support) == 0:
        _binary_layout(slot)
    slot[].fitted = True


def _kernel_svm_member(operation: Int, address: Int) raises -> Int:
    var call = _Call(unsafe_from_address=address)
    if operation == MEMBER_OP_CAPABILITIES:
        var common = MEMBER_CAN_FIT | MEMBER_CAN_PREDICT | MEMBER_CAN_EXPORT | MEMBER_CAN_IMPORT | MEMBER_CAN_SAMPLE_WEIGHT
        var svc = (MEMBER_CAN_PREDICT_PROBA | MEMBER_CAN_DECISION | MEMBER_CAN_SCORE | MEMBER_CAN_CLASSIFIER)
        if call[].state == 0:
            return common | svc
        var slot = _Slot(unsafe_from_address=_lookup(call[].state, False))
        return common | svc if _svc(slot) else common
    if operation == MEMBER_OP_NEW:
        if call[].param_count != _PARAMS or not call[].params:
            raise Error("member ABI: a kernel SVM member takes " + String(_PARAMS) + " parameters")
        var source = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call[].params)
        var words = List[Float64](length=_PARAMS, fill=0.0)
        for i in range(_PARAMS):  # small-loop(the member's parameter words)
            words[i] = source[i]
        if words[0] != 0.0 and words[0] != 1.0:
            raise Error("member ABI: unknown kernel SVM member kind")
        if Int(words[1]) == SVC_KERNEL_PRECOMPUTED:
            raise Error("member ABI: a kernel SVM member with kernel='precomputed' is not published")
        if words[6] <= 0.0:
            raise Error("ValueError: C must be positive, got " + String(words[6]))
        call[].state = _mint(words^)
        return 0
    if operation == MEMBER_OP_FREE:
        _ = _lookup(call[].state, True)
        call[].state = 0
        return 0
    var slot = _Slot(unsafe_from_address=_lookup(call[].state, False))
    if operation == MEMBER_OP_OUTPUTS:
        call[].out_cols = _outputs(slot, call[].flags)
        call[].classes = slot[].classes if _svc(slot) else 0
        return 0
    if operation == MEMBER_OP_FIT:
        _fit(slot, call)
    elif operation == MEMBER_OP_PREDICT or ((operation == MEMBER_OP_DECISION
            or operation == MEMBER_OP_PREDICT_PROBA) and _svc(slot)):
        _predict(slot, call, operation)
    elif operation == MEMBER_OP_SCORE:
        _score(slot, call)
    elif operation == MEMBER_OP_EXPORT:
        _export(slot, call)
    elif operation == MEMBER_OP_IMPORT:
        _import(slot, call)
    else:
        member_unsupported(_name(slot), operation)
    return 0


def kernel_svm_member_v2(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
    result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """The member entry of SVC and SVR (schema `MEMBER_ABI`)."""
    try:
        var word = _kernel_svm_member(operation, member_call_from_spans(operation, aa, na, np, nf))
        if result:
            MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = word
        return 0
    except error:
        native_descriptor_report(error, context, sink)
        return 1


def kernel_svm_member_publish(mode: Int, vendor: Int, publisher: Int) raises:
    """Publish this image's kernel SVM member entry. One line in the SVM
    image's module initialization."""
    member_publish[MEMBER_FAMILY_KERNEL_SVM](mode, vendor, publisher, kernel_svm_member_v2)
