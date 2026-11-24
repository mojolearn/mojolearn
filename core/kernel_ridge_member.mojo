# SPDX-License-Identifier: Apache-2.0
"""KernelRidge as a MEMBER (core/member_abi.mojo, family
`MEMBER_FAMILY_KERNEL_RIDGE`), after core/linear_member.mojo.

One entry, `kernel_ridge_member_v2`, published by each kernel methods image
(`kernel_ridge_member_publish`: bindings/_mojolearn_kernel_methods.mojo and
bindings/_mojolearn_kernel_methods_host.mojo). It runs the family's existing
typed RUN (kernel_methods/native_call_run.mojo `KmCall` requests: the
estimator's own door's operations, in its order) over `route.entry`, the
image's kernel methods provider table (both tiers: a `fast` estimator's own
door keeps its list and pointer entries, its member admission takes the
image's table, which runs the same bodies; lane mf-img3).
`route.flags` is the provider's capability word (operation 14): bit 1 =
the pointer fit and the device-resident predict (a GPU image), as the
estimator's door reads it. No kernel is touched.

KIND: 0 KernelRidge (a regressor).

MEMBER_OP_NEW parameters (6 Float64 words): [0] kernel code (0 linear,
1 poly, 2 rbf, 3 sigmoid, 4 precomputed (lane mf-adm3: X is the kernel
matrix, square at fit, query rows x training rows at predict, handed to the
provider with code 4 as the estimator's own door hands it), 5 laplacian,
6 cosine, 7 chi2, 8 additive_chi2); [1] degree; [2] gamma kind (0 a value,
1 None: 1 / n_features at fit, 1.0 for chi2); [3] gamma (kind 0); [4] coef0;
[5] alpha.

VIEWS: X dense row-major host float32 (the caller's; NOT copied: see
LIFETIME). y host float32 rows x targets (targets >= 1). Sample weights
host float64 or float32, one per row. predict writes float32 rows x
targets. Anything else is refused in words (a device-home view, a row or
column index, a nonzero `fit_output`). X, y and the query are tested for
NaN or infinity on the host before the provider call (the estimator's own
UNPACK test: the provider route returns no status word), and the chi2
kernels refuse negative X, as the estimator does. No score is published:
the caller predicts and runs its own metric (as the SVR member).

LIFETIME (no copy of X at fit): a fitted state reads X_fit_ from the
caller's fit view until its next fit, import or free; the caller keeps that
view alive until then (MEMBER-ABI-V2 "NO COPY OF X AT FIT"). An imported
state owns its X_fit_ words. On a GPU route the first predict uploads
X_fit_ and dual_coef_ once (the provider's resident prepare, the
estimator's own route) and every later predict reads the device copy; the
handle is released before a refit, an import or a free.

FITTED-STATE BLOCKS, in index order (count 3):
 0 `MEMBER_BLOCK_SCALARS` float64 1 x 9 [kernel, degree, gamma (resolved),
   coef0, alpha, rows, features, targets, info];
 1 FAMILY+0 X_fit_ float32 rows x features;
 2 FAMILY+1 dual_coef_ float32 rows x targets.
An import needs all three (DONE on the last).

The fitted state is OWNED here, in a process-wide store of fixed chunks
(the linear member's store): the lock covers only mint, free and lookup.
Serial by nature here: the store's bookkeeping and one provider call per
operation (the provider's kernels are parallel inside); the host tests of
X are row tasks across the cores (core/host_finite_scan.mojo).
"""
from std.memory import unsafe_memcpy
from std.sys.info import size_of
from core.admitted_cast_elements import admitted_cast_elements
from core.admitted_refusals import admitted_finite_wording
from core.entry_table import EntryRef, entry_ref_unresolved
from core.family_cast_impl import family_min_f32
from core.host_finite_scan import host_all_finite_f32, host_all_finite_f64
from core.hotpath_impl import HP_F32, HP_F64
from core.member_abi import (
    MemberCall, MemberMatrix, MEMBER_F32, MEMBER_F64,
    MEMBER_OP_CAPABILITIES, MEMBER_OP_NEW, MEMBER_OP_FREE, MEMBER_OP_FIT, MEMBER_OP_PREDICT,
    MEMBER_OP_OUTPUTS, MEMBER_OP_EXPORT, MEMBER_OP_IMPORT,
    MEMBER_CAN_FIT, MEMBER_CAN_PREDICT, MEMBER_CAN_EXPORT, MEMBER_CAN_IMPORT, MEMBER_CAN_SAMPLE_WEIGHT,
    MEMBER_CAN_TEST_ROWS, MEMBER_CAN_ROW_INDEX, MEMBER_CAN_SCOPE,
    MEMBER_BLOCK_SCALARS, MEMBER_BLOCK_FAMILY, MEMBER_BLOCK_BY_INDEX, MEMBER_IMPORT_DONE,
    member_publish, member_call_from_spans, member_unsupported,
)
from core.member_abi import MEMBER_BORROWS_FIT, MEMBER_EXPORT_OWN
from core.member_families import MEMBER_FAMILY_KERNEL_RIDGE
from core.device_route import DEVICE_GPU
from core.member_fold_x import (
    MemberFoldX, member_fold_x_open, member_fold_x_close,
    member_fold_take_into, member_fold_store_ready,
)
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report, native_descriptor_leased_call
from core.shared_slot import SharedSlot, shared_slot_lock, shared_slot_unlock
from kernel_methods.native_call_run import (
    KmCall, KmKernel, km_call, km_krr_fit_call, km_krr_predict_call, km_krr_prepare_call,
    KM_OP_KRR_PREDICT, KM_OP_KRR_PREDICT_RESIDENT,
)

comptime _CHUNK = 256
comptime _INDEX_SPAN = 16777216
comptime _PARAMS = 6
comptime _SCALARS = 9
comptime _BLOCKS = 3
comptime _B_X_FIT = MEMBER_BLOCK_FAMILY
comptime _B_DUAL = MEMBER_BLOCK_FAMILY + 1
# kernel_methods/descriptor_schema.mojo operation 5: release a resident fit.
comptime _KM_OP_KRR_RESIDENT_RELEASE = 5
comptime _KERNEL_PRECOMPUTED = 4
comptime _KERNEL_CHI2 = 7
comptime _KERNEL_ADDITIVE_CHI2 = 8
comptime _WHERE = "KernelRidge"


struct KernelRidgeMemberSlot(Copyable, Movable, Defaultable):
    var generation: Int
    var live: Bool
    var params: List[Float64]
    var fitted: Bool
    var rows: Int
    var cols: Int
    var targets: Int
    var gamma: Float64
    var info: Int
    # X_fit_: the caller's fit view (borrowed, LIFETIME) or `x_owned`.
    var x_fit: Int
    var x_owned: List[Float32]
    var dual: List[Float32]
    var scalars: List[Float64]
    # The provider's resident copy of X_fit_ and dual_coef_ (GPU route), and
    # the entry that releases it. 0: none held.
    var handle: Int
    var handle_entry: EntryRef

    def __init__(out self):
        self.generation = 0
        self.live = False
        self.params = List[Float64]()
        self.fitted = False
        self.rows = 0
        self.cols = 0
        self.targets = 0
        self.gamma = 0.0
        self.info = 0
        self.x_fit = 0
        self.x_owned = List[Float32]()
        self.dual = List[Float32]()
        self.scalars = List[Float64]()
        self.handle = 0
        self.handle_entry = entry_ref_unresolved()

    def clear(mut self):
        self.fitted = False
        self.rows = 0
        self.cols = 0
        self.targets = 0
        self.info = 0
        self.x_fit = 0
        self.x_owned = List[Float32]()
        self.dual = List[Float32]()
        self.scalars = List[Float64]()


struct KernelRidgeMemberStore(Defaultable, Movable):
    var synchronization: List[Int64]
    var chunks: List[List[KernelRidgeMemberSlot]]
    var free: List[Int]
    var count: Int

    def __init__(out self):
        # Side-effect-free even when the process slot's initializers race.
        self.synchronization = List[Int64](length=2, fill=0)
        self.chunks = List[List[KernelRidgeMemberSlot]]()
        self.free = List[Int]()
        self.count = 0


comptime _STORE = SharedSlot[KernelRidgeMemberStore, "MojolearnKernelRidgeMemberStoreV2", 2]
comptime _Slot = MutPointer[KernelRidgeMemberSlot, MutUntrackedOrigin]
comptime _Call = MutPointer[MemberCall, MutUntrackedOrigin]


def _slot_address(index: Int) raises -> Int:
    # Under the store's lock. A chunk's buffer is allocated once and never
    # grows, so the address stays valid after the lock is released.
    var store = _STORE.get_or_create_ptr()
    return Int(store[].chunks[index // _CHUNK].unsafe_ptr()) + (index % _CHUNK) * size_of[KernelRidgeMemberSlot]()


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
                raise Error("RuntimeError: mojolearn: too many live KernelRidge member states")
            if store[].count == len(store[].chunks) * _CHUNK:
                store[].chunks.append(List[KernelRidgeMemberSlot](length=_CHUNK, fill=KernelRidgeMemberSlot()))
            index = store[].count
            store[].count += 1
        var slot = _Slot(unsafe_from_address=_slot_address(index))
        slot[].generation += 1
        slot[].live = True
        slot[].params = params^
        slot[].clear()
        return slot[].generation * _INDEX_SPAN + index + 1
    finally:
        shared_slot_unlock(lock)


def _run(entry: NativeDescriptorCall, call: KmCall) raises -> Int:
    """One provider request (the estimator's `km_call_run`, on the route's
    entry instead of a Python-held record)."""
    return native_descriptor_leased_call(entry, 0, call.operation, call.a, call.p, call.f)


def _release_handle(entry: EntryRef, handle: Int):
    """Drop a resident fit. Never raises (FREE never raises)."""
    try:
        var call = km_call(_KM_OP_KRR_RESIDENT_RELEASE)
        call.address(handle)
        _ = _run(entry.call, call)
    except:
        pass


def _drop_handle(slot: _Slot):
    """Before the fitted state changes (fit, import): its device copy goes."""
    if slot[].handle != 0:
        var handle = slot[].handle
        slot[].handle = 0
        _release_handle(slot[].handle_entry, handle)


def _lookup(handle: Int, release: Bool) raises -> Int:
    """The slot address of a live handle. A released handle, a handle of
    another store or a made-up integer is refused: never a stale read. A
    release drops the resident fit after the lock is released."""
    var store = _STORE.get_or_create_ptr()
    var lock = rebind[MutPointer[Int64, MutUntrackedOrigin]](store[].synchronization.unsafe_ptr())
    var address = 0
    var resident = 0
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
            resident = slot[].handle
            entry = slot[].handle_entry
            slot[].handle = 0
            slot[].live = False
            slot[].clear()
            slot[].params = List[Float64]()
            store[].free.append(index)
    finally:
        shared_slot_unlock(lock)
    if resident != 0:
        _release_handle(entry, resident)
    return address


def _f32p(address: Int) -> MutPointer[Float32, MutUntrackedOrigin]:
    return MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=address)


def _whole(call: _Call) raises:
    if call[].row_index.rows > 0 or call[].column_index.rows > 0:
        raise Error("TypeError: a KernelRidge member reads whole host views (no row or column index)")


def _dense(view: MemberMatrix, dtype: Int, what: String) raises:
    if not view.dense_host(dtype):
        var word = String("float32") if dtype == MEMBER_F32 else String("float64")
        raise Error("TypeError: a KernelRidge member reads " + what + " as a dense host " + word + " view")


def _kernel(slot: _Slot) -> KmKernel:
    return KmKernel(Int(slot[].params[0]), Int(slot[].params[1]), slot[].gamma, slot[].params[4])


def _tested(view: MemberMatrix, kernel: Int, name: String) raises:
    """The estimator's UNPACK tests of a host float32 view: finite (the
    provider route returns no status word), and x >= 0 for the chi2
    kernels (`_km_chi2_input`)."""
    var count = view.rows * view.cols
    if count == 0:
        return
    if not host_all_finite_f32(_f32p(view.data), count):
        raise Error("ValueError: " + admitted_finite_wording(_WHERE, name))
    if name == "X" and (kernel == _KERNEL_CHI2 or kernel == _KERNEL_ADDITIVE_CHI2):
        if Float64(family_min_f32(_f32p(view.data), count)) < 0.0:
            raise Error("mojolearn KernelRidge: X contains negative values (the chi2 kernels need x >= 0)")


def _weights(call: _Call, rows: Int, mut held: List[Float64]) raises -> Int:
    """The float64 sample weights the provider reads (0: none)."""
    if call[].weights.rows == 0:
        return 0
    if call[].weights.rows != rows or call[].weights.cols != 1:
        raise Error("ValueError: sample_weight and X have different numbers of rows")
    # The finite test of the weights (one parallel pass): with X and y
    # tested at fit and X at predict, the member tests every row it reads.
    if call[].weights.dense_host(MEMBER_F64):
        if not host_all_finite_f64(MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call[].weights.data), rows):
            raise Error("ValueError: " + admitted_finite_wording(_WHERE, "sample_weight"))
        return call[].weights.data
    _dense(call[].weights, MEMBER_F32, "sample_weight")
    if not host_all_finite_f32(_f32p(call[].weights.data), rows):
        raise Error("ValueError: " + admitted_finite_wording(_WHERE, "sample_weight"))
    held = List[Float64](length=rows, fill=0.0)
    if admitted_cast_elements(call[].weights.data, HP_F32, Int(held.unsafe_ptr()), HP_F64, rows) != 0:
        raise Error("ValueError: a member view holds a value its element type cannot convert")
    return Int(held.unsafe_ptr())


def _fit(slot: _Slot, call: _Call) raises:
    _whole(call)
    if call[].fit_output != 0:
        member_unsupported(_WHERE, call[].fit_output)
    _dense(call[].x, MEMBER_F32, "X")
    _dense(call[].y, MEMBER_F32, "y")
    var n = call[].x.rows
    var d = call[].x.cols
    var t = call[].y.cols
    if call[].y.rows != n or t < 1:
        raise Error("mojolearn KernelRidge: y has " + String(call[].y.rows) + " rows, X has " + String(n))
    var kernel = Int(slot[].params[0])
    if kernel == _KERNEL_PRECOMPUTED and d != n:
        raise Error("mojolearn KernelRidge: kernel='precomputed' needs the square "
                    + "n x n kernel matrix as X, got " + String(n) + " x " + String(d))
    var fold_x = call[].x.data < 0  # THE FOLD X WORD (`_fold_op`): tested on the device
    if not fold_x:
        _tested(call[].x, kernel, "X")
    _tested(call[].y, kernel, "y")
    var held = List[Float64]()
    var sample_weights = _weights(call, n, held)
    # `_km_gamma`: None is 1 / n_features (1.0 for chi2).
    var gamma = slot[].params[3]
    if slot[].params[2] != 0.0:
        gamma = 1.0 if kernel == _KERNEL_CHI2 else 1.0 / Float64(d)
    _drop_handle(slot)
    slot[].clear()
    slot[].gamma = gamma
    var dual = List[Float32](length=max(1, n * t), fill=Float32(0))
    var scalars = List[Float64](length=1, fill=0.0)
    var route = call[].route
    var call_km = km_krr_fit_call((route.flags & 1) != 0, call[].x.data, call[].y.data, Int(dual.unsafe_ptr()),
        Int(scalars.unsafe_ptr()), n, d, t, _kernel(slot), slot[].params[5])
    var owned = List[Float32]()
    if fold_x:
        # the pointer fit copies the fold's rows it read back as X_fit_
        # (x_fit_out; kernel_methods/native_descriptor_dispatch.mojo) and
        # refuses a non-finite fold there; the state owns them.
        owned = List[Float32](unsafe_uninit_length=max(n * d, 1))
        call_km.address(sample_weights)
        call_km.address(Int(owned.unsafe_ptr()))
    elif sample_weights != 0:
        call_km.address(sample_weights)
    var info = _run(route.entry.call, call_km)
    slot[].rows = n
    slot[].cols = d
    slot[].targets = t
    slot[].info = info
    slot[].x_fit = call[].x.data
    if fold_x:
        slot[].x_owned = owned^
        slot[].x_fit = Int(slot[].x_owned.unsafe_ptr())
    slot[].dual = dual^
    slot[].fitted = True
    _ = scalars^
    _ = held^


def _resident(slot: _Slot, call: _Call) raises -> Int:
    """The provider's device copy of X_fit_ and dual_coef_ (GPU route),
    uploaded once per fitted state, as the estimator's own door keeps one."""
    if slot[].handle == 0:
        var upload = km_krr_prepare_call(slot[].x_fit, Int(slot[].dual.unsafe_ptr()), slot[].rows, slot[].cols,
            slot[].targets)
        slot[].handle = _run(call[].route.entry.call, upload)
        slot[].handle_entry = call[].route.entry
    return slot[].handle


def _predict(slot: _Slot, call: _Call) raises:
    if not slot[].fitted:
        raise Error("NotFittedError: this KernelRidge member is not fitted yet")
    _whole(call)
    _dense(call[].x, MEMBER_F32, "X")
    var q = call[].x.rows
    var d = slot[].cols
    var t = slot[].targets
    if call[].x.cols != d:
        raise Error("mojolearn KernelRidge: X has " + String(call[].x.cols) + " features, the fit had " + String(d))
    _dense(call[].dst, MEMBER_F32, "the output buffer")
    if call[].dst.rows != q or call[].dst.cols != t:
        raise Error("ValueError: the output buffer holds " + String(t) + " value(s) per row of X")
    if q == 0:
        return
    var kernel = Int(slot[].params[0])
    if call[].x.data >= 0:  # a fold X word (`_fold_op`) is tested on the device by the predict entry
        _tested(call[].x, kernel, "X")
    var route = call[].route
    var alpha = slot[].params[5]
    if (route.flags & 1) != 0:
        var handle = _resident(slot, call)
        var resident = km_krr_predict_call(KM_OP_KRR_PREDICT_RESIDENT, slot[].rows, d, t, _kernel(slot), alpha,
            slot[].info, q)
        resident.address(handle)
        resident.address(call[].x.data)
        resident.address(call[].dst.data)
        _ = _run(route.entry.call, resident)
        return
    # ORDER MATCHES bindings/_mojolearn_kernel_methods.mojo::kernel_ridge_predict_binding:
    # x_fit, dual, x_new, out.
    var plain = km_krr_predict_call(KM_OP_KRR_PREDICT, slot[].rows, d, t, _kernel(slot), alpha, slot[].info, q)
    plain.address(slot[].x_fit)
    plain.address(Int(slot[].dual.unsafe_ptr()))
    plain.address(call[].x.data)
    plain.address(call[].dst.data)
    _ = _run(route.entry.call, plain)


def _own_fit(slot: _Slot):
    """THE LIFETIME RULE (MEMBER_EXPORT_OWN): copy the borrowed X_fit_ into
    the state, once (one memcpy); the state reads its own words from then
    on. A resident device handle holds a copy of the same words and stays."""
    var count = slot[].rows * slot[].cols
    if slot[].x_fit == 0 or (slot[].x_fit == Int(slot[].x_owned.unsafe_ptr()) and len(slot[].x_owned) == count):
        return
    var owned = List[Float32](length=count, fill=Float32(0))
    if count > 0:
        unsafe_memcpy(dest=owned.unsafe_ptr(), src=_f32p(slot[].x_fit), count=count)
    slot[].x_owned = owned^
    slot[].x_fit = Int(slot[].x_owned.unsafe_ptr())


def _export(slot: _Slot, call: _Call) raises:
    """A view of one fitted block. X_fit_ is the caller's fit view (or the
    imported words); the rest is owned here. Valid until this state's next
    fit, import or free."""
    if not slot[].fitted:
        raise Error("NotFittedError: this KernelRidge member is not fitted yet")
    if (call[].flags & MEMBER_EXPORT_OWN) != 0:
        _own_fit(slot)
        return
    var block = call[].block
    if (call[].flags & MEMBER_BLOCK_BY_INDEX) != 0:
        if block < 0 or block >= _BLOCKS:
            raise Error("member ABI: this KernelRidge member has " + String(_BLOCKS) + " fitted blocks")
        block = MEMBER_BLOCK_SCALARS if block == 0 else MEMBER_BLOCK_FAMILY + block - 1
    if block == MEMBER_BLOCK_SCALARS:
        slot[].scalars = List[Float64](length=_SCALARS, fill=0.0)
        slot[].scalars[0] = slot[].params[0]
        slot[].scalars[1] = slot[].params[1]
        slot[].scalars[2] = slot[].gamma
        slot[].scalars[3] = slot[].params[4]
        slot[].scalars[4] = slot[].params[5]
        slot[].scalars[5] = Float64(slot[].rows)
        slot[].scalars[6] = Float64(slot[].cols)
        slot[].scalars[7] = Float64(slot[].targets)
        slot[].scalars[8] = Float64(slot[].info)
        call[].dst = MemberMatrix(Int(slot[].scalars.unsafe_ptr()), 1, _SCALARS, MEMBER_F64)
    elif block == _B_X_FIT:
        call[].dst = MemberMatrix(slot[].x_fit, slot[].rows, slot[].cols)
    elif block == _B_DUAL:
        call[].dst = MemberMatrix(Int(slot[].dual.unsafe_ptr()), slot[].rows, slot[].targets)
    else:
        raise Error("member ABI: a KernelRidge member has no fitted block " + String(block))
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
    The scalars come first."""
    _drop_handle(slot)
    var block = call[].block
    if block == MEMBER_BLOCK_SCALARS:
        _dense(call[].x, MEMBER_F64, "the scalars block")
        if call[].x.rows * call[].x.cols != _SCALARS:
            raise Error("member ABI: the KernelRidge scalars block is nine float64 words")
        var words = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call[].x.data)
        if Int(words[0]) != Int(slot[].params[0]):
            raise Error("member ABI: the KernelRidge scalars block holds another kernel")
        slot[].clear()
        slot[].gamma = words[2]
        slot[].rows = Int(words[5])
        slot[].cols = Int(words[6])
        slot[].targets = Int(words[7])
        slot[].info = Int(words[8])
    elif block == _B_X_FIT:
        if call[].x.rows != slot[].rows or call[].x.cols != slot[].cols:
            raise Error("member ABI: the KernelRidge X_fit_ block does not match the scalars block")
        slot[].x_owned = _copy_f32(call[].x, "X_fit_")
        slot[].x_fit = Int(slot[].x_owned.unsafe_ptr())
    elif block == _B_DUAL:
        if call[].x.rows * call[].x.cols != slot[].rows * slot[].targets:
            raise Error("member ABI: the KernelRidge dual_coef_ block does not match the scalars block")
        slot[].dual = _copy_f32(call[].x, "dual_coef_")
    elif block != 0 or (call[].flags & MEMBER_IMPORT_DONE) == 0:
        raise Error("member ABI: a KernelRidge member has no fitted block " + String(block))
    if (call[].flags & MEMBER_IMPORT_DONE) == 0:
        return
    if slot[].rows < 1 or slot[].targets < 1 or slot[].x_fit == 0 or len(slot[].dual) == 0:
        raise Error("member ABI: a fitted KernelRidge state needs its scalars, X_fit_ and dual_coef_ blocks")
    slot[].fitted = True


def _kernel_ridge_member(operation: Int, address: Int) raises -> Int:
    var call = _Call(unsafe_from_address=address)
    if operation == MEMBER_OP_CAPABILITIES:
        # TEST_ROWS: `_tested` X and y at fit, X at predict, `_weights` the
        # sample weights; one parallel pass each.
        # MEMBER_BORROWS_FIT: the fitted state reads the caller's fit X as
        # X_fit_ at every predict and its export hands those words back
        # themselves; MEMBER_EXPORT_OWN (`_own_fit`) or an import owns them.
        var bits = (MEMBER_CAN_FIT | MEMBER_CAN_PREDICT | MEMBER_CAN_EXPORT | MEMBER_CAN_IMPORT | MEMBER_CAN_SAMPLE_WEIGHT
            | MEMBER_CAN_TEST_ROWS | MEMBER_BORROWS_FIT)
        # MEMBER_CAN_ROW_INDEX | MEMBER_CAN_SCOPE (lane vdf-scope-linear2): a
        # GPU image's pointer fit and device predict read THE FOLD X WORD
        # through DeviceQueryX (core/device_query_x.mojo); `_fold_op`.
        # Lane vdf-scope-linear3: the chi2 kernels too (their x >= 0 test of a
        # fold slot runs on the device, kernel_methods/estimator.mojo
        # `krr_chi2_fold_check`); a 'precomputed' state answers no fold: its X
        # is a kernel matrix, whose fold is a row AND column take.
        if _route_fold(call) and not (call[].state != 0 and _precomputed_state(call[].state)):
            bits |= MEMBER_CAN_ROW_INDEX | MEMBER_CAN_SCOPE
        return bits
    if operation == MEMBER_OP_NEW:
        if call[].param_count != _PARAMS or not call[].params:
            raise Error("member ABI: a KernelRidge member takes " + String(_PARAMS) + " parameters")
        var source = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call[].params)
        var words = List[Float64](length=_PARAMS, fill=0.0)
        for i in range(_PARAMS):  # small-loop(the member's parameter words)
            words[i] = source[i]
        call[].state = _mint(words^)
        return 0
    if operation == MEMBER_OP_FREE:
        _ = _lookup(call[].state, True)
        call[].state = 0
        return 0
    if operation == MEMBER_OP_OUTPUTS:
        # predict: float32, one column per target (the fitted state's, else
        # the caller's `y.cols`, item 10). Another operation is refused.
        if call[].flags != MEMBER_OP_PREDICT:
            member_unsupported(_WHERE, call[].flags)
        var targets = max(1, call[].y.cols)
        if call[].state != 0:
            var held = _Slot(unsafe_from_address=_lookup(call[].state, False))
            if held[].fitted:
                targets = held[].targets
        call[].out_cols = targets
        call[].out_dtype = MEMBER_F32
        call[].classes = 0
        return 0
    var slot = _Slot(unsafe_from_address=_lookup(call[].state, False))
    if (call[].row_index.rows > 0 and call[].column_index.rows == 0 and call[].route.device == DEVICE_GPU
            and (operation == MEMBER_OP_FIT or operation == MEMBER_OP_PREDICT)):
        _fold_op(slot, call, operation)
        return 0
    if operation == MEMBER_OP_FIT:
        _fit(slot, call)
    elif operation == MEMBER_OP_PREDICT:
        _predict(slot, call)
    elif operation == MEMBER_OP_EXPORT:
        _export(slot, call)
    elif operation == MEMBER_OP_IMPORT:
        _import(slot, call)
    else:
        member_unsupported(_WHERE, operation)
    return 0


def _route_fold(call: _Call) -> Bool:
    """A GPU image with the pointer fit and the device predict (flags bit
    1) that ships the x_prep family store. Asked with the capabilities."""
    return call[].route.device == DEVICE_GPU and (call[].route.flags & 1) != 0 and member_fold_store_ready(call)


def _precomputed_state(state: Int) -> Bool:
    """Whether the live state's kernel is 'precomputed' (False on a bad word:
    the capability answer stays the family's)."""
    try:
        var held = _Slot(unsafe_from_address=_lookup(state, False))
        return Int(held[].params[0]) == _KERNEL_PRECOMPUTED
    except:
        return False


def _fold_op(slot: _Slot, call: _Call, operation: Int) raises:
    """FIT / PREDICT over the rows `call.row_index` of the WHOLE host X
    (core/member_fold_x.mojo): X resident in the caller's scope, the fold's
    rows taken on the device and read by DeviceQueryX in the provider's
    entries (THE FOLD X WORD; its finite test, and for the chi2 kernels its
    x >= 0 test, on the device: lane vdf-scope-linear3). 'precomputed' is
    refused: its X is a kernel matrix, a row-only fold of it is no kernel
    matrix of the fold (the caller takes rows and columns itself; the
    member answers no fold for it). y and the weights are taken bitwise
    through the index on the host. Views restored before this returns or
    raises."""
    var x = call[].x
    var y = call[].y
    var weights = call[].weights
    var index = call[].row_index
    var fold = MemberFoldX()
    var host_y = List[UInt64]()
    var host_w = List[UInt64]()
    if Int(slot[].params[0]) == _KERNEL_PRECOMPUTED:
        raise Error("TypeError: a KernelRidge member with kernel='precomputed' takes no row index (its X is the "
                    + "kernel matrix: the caller selects its rows and columns)")
    if not _route_fold(call):
        raise Error("TypeError: a KernelRidge member reads a row index only on a GPU route with the pointer fit "
                    + "and the x_prep store")
    var failure = String("")
    try:
        member_fold_x_open(call, fold, _WHERE)
        call[].x = MemberMatrix(fold.word(), fold.rows, fold.cols)
        if operation == MEMBER_OP_FIT:
            if not y.dense_host(MEMBER_F32) or y.rows != x.rows:
                raise Error("mojolearn KernelRidge: y has " + String(y.rows) + " rows, X has " + String(x.rows))
            call[].y = member_fold_take_into(y, fold, host_y)
            if weights.rows > 0:
                if weights.rows != x.rows or weights.cols != 1 or not weights.dense_host(weights.dtype):
                    raise Error("ValueError: sample_weight and X have different numbers of rows")
                call[].weights = member_fold_take_into(weights, fold, host_w)
        call[].row_index = MemberMatrix()
        if operation == MEMBER_OP_FIT:
            _fit(slot, call)
        else:
            _predict(slot, call)
    except error:
        failure = String(error)
    call[].x = x
    call[].y = y
    call[].weights = weights
    call[].row_index = index
    try:
        member_fold_x_close(fold)
    except close_error:
        if failure.byte_length() == 0:
            failure = "RuntimeError: a KernelRidge member's fold slot was not freed: " + String(close_error)
    _ = host_y^
    _ = host_w^
    if failure.byte_length() > 0:
        raise Error(failure)


def kernel_ridge_member_v2(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
    result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """The member entry of KernelRidge (schema `MEMBER_ABI`)."""
    try:
        var word = _kernel_ridge_member(operation, member_call_from_spans(operation, aa, na, np, nf))
        if result:
            MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = word
        return 0
    except error:
        native_descriptor_report(error, context, sink)
        return 1


def kernel_ridge_member_publish(mode: Int, vendor: Int, publisher: Int) raises:
    """Publish this image's KernelRidge member entry. One line in each
    kernel methods image's module initialization."""
    member_publish[MEMBER_FAMILY_KERNEL_RIDGE](mode, vendor, publisher, kernel_ridge_member_v2)
