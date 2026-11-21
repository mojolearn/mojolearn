# SPDX-License-Identifier: Apache-2.0
"""Nystroem and RBFSampler as MEMBERS (core/member_abi.mojo, family
`MEMBER_FAMILY_KERNEL_APPROX`), after core/linear_member.mojo.

One entry, `kernel_approx_member_v2`, published by each kernel methods
image (`kernel_approx_member_publish`: bindings/_mojolearn_kernel_methods.mojo
and bindings/_mojolearn_kernel_methods_host.mojo). It runs the family's
existing typed RUN (kernel_methods/native_call_run.mojo `KmCall` requests:
the estimator's own door's operations, in its order) over `route.entry`,
the image's kernel methods provider table (`identical` only: a `fast` build
has no provider and the boundary keeps the estimator's own doors).
`route.flags` is the provider's capability word (operation 14): bit 2 =
Nystroem's pointer fit, as the estimator's door reads it. No kernel is
touched.

KINDS: 0 Nystroem, 1 RBFSampler (transformers: fit, transform).

MEMBER_OP_NEW parameters (9 Float64 words): [0] kind; [1] kernel code
(Nystroem: 0 linear, 1 poly, 2 rbf, 3 sigmoid, 5 laplacian, 6 cosine,
7 chi2, 8 additive_chi2; RBFSampler: 2); [2] degree; [3] gamma kind (0 a
value, 1 None: 1 / n_features at fit, 1.0 for chi2; 2 'scale', RBFSampler:
the provider's exact limbs of X at fit); [4] gamma (kind 0); [5] coef0;
[6] n_components; [7] random_state's word, low 32 bits, [8] its high 32
bits (two's complement, as `km_seed` answers it).

VIEWS: X dense row-major host float32 (read during the call only: no copy,
nothing kept). transform writes float32 rows x n_components. Anything else
is refused in words (a device-home view, a row or column index, a nonzero
`fit_output`: the caller fits, then transforms). The chi2 kernels refuse
negative X (`_km_chi2_input`), X is tested for NaN or infinity on the host
(the estimator's admission test). Nystroem's n_components above the rows
becomes the rows (scikit-learn's rule; its warning is the estimator's own
door's, not the member's).

FITTED-STATE BLOCKS, in index order:
 0 `MEMBER_BLOCK_SCALARS` float64 1 x 12 [kind, kernel, degree, gamma
   (resolved), coef0, n_components (effective), features, sweeps, sigma,
   scale, seed low, seed high];
 Nystroem (count 6): FAMILY+0 components_ float32 q x d; FAMILY+1
   component_indices_ int32 1 x q; FAMILY+2 normalization_ float32 q x q;
   FAMILY+3 eigenvalues_ float32 1 x q; FAMILY+4 eigenvectors_ float32 q x q.
 RBFSampler (count 3): FAMILY+0 random_weights_ float32 d x q; FAMILY+1
   random_offset_ float32 1 x q.
An import needs every block of its kind (DONE on the last).

The fitted state is OWNED here, in a process-wide store of fixed chunks
(the linear member's store): the lock covers only mint, free and lookup.
Serial by nature here: the store's bookkeeping and one provider call per
operation (the provider's kernels are parallel inside); the host test of X
is row tasks across the cores (core/host_finite_scan.mojo).
"""
from std.memory import unsafe_memcpy
from std.sys.info import size_of
from core.admitted_refusals import admitted_finite_wording
from core.family_cast_impl import family_min_f32
from core.host_finite_scan import host_all_finite_f32
from core.member_abi import (
    MemberCall, MemberMatrix, MEMBER_F32, MEMBER_F64, MEMBER_I32,
    MEMBER_OP_CAPABILITIES, MEMBER_OP_NEW, MEMBER_OP_FREE, MEMBER_OP_FIT, MEMBER_OP_TRANSFORM,
    MEMBER_OP_OUTPUTS, MEMBER_OP_EXPORT, MEMBER_OP_IMPORT,
    MEMBER_CAN_FIT, MEMBER_CAN_TRANSFORM, MEMBER_CAN_EXPORT, MEMBER_CAN_IMPORT, MEMBER_CAN_TEST_ROWS,
    MEMBER_BLOCK_SCALARS, MEMBER_BLOCK_FAMILY, MEMBER_BLOCK_BY_INDEX, MEMBER_IMPORT_DONE,
    member_publish, member_call_from_spans, member_unsupported,
)
from core.member_families import MEMBER_FAMILY_KERNEL_APPROX
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report, native_descriptor_leased_call
from core.shared_slot import SharedSlot, shared_slot_lock, shared_slot_unlock
from kernel_methods.native_call_run import (
    KmCall, KmKernel, km_nys_fit_call, km_nys_transform_call, km_rbf_fit_call, km_rbf_transform_call,
    km_scale_gamma_words, km_scale_gamma_value, KM_OP_NYS_FIT, KM_OP_NYS_FIT_PTR, KM_OP_SCALE_GAMMA_LIMBS,
)

comptime _CHUNK = 256
comptime _INDEX_SPAN = 16777216
comptime _PARAMS = 9
comptime _SCALARS = 12
comptime _KIND_NYSTROEM = 0
comptime _KIND_RBF = 1
comptime _KERNEL_PRECOMPUTED = 4
comptime _KERNEL_CHI2 = 7
comptime _KERNEL_ADDITIVE_CHI2 = 8
comptime _B_FIRST = MEMBER_BLOCK_FAMILY


struct KernelApproxMemberSlot(Copyable, Movable, Defaultable):
    var generation: Int
    var live: Bool
    var params: List[Float64]
    var fitted: Bool
    var cols: Int
    var components: Int
    var gamma: Float64
    var sweeps: Int
    var sigma: Float64
    var scale: Float64
    # Nystroem: components_, normalization_, eigenvalues_, eigenvectors_,
    # component_indices_. RBFSampler: random_weights_ (first),
    # random_offset_ (second).
    var first: List[Float32]
    var second: List[Float32]
    var third: List[Float32]
    var fourth: List[Float32]
    var indices: List[Int32]
    var scalars: List[Float64]

    def __init__(out self):
        self.generation = 0
        self.live = False
        self.params = List[Float64]()
        self.fitted = False
        self.cols = 0
        self.components = 0
        self.gamma = 0.0
        self.sweeps = 0
        self.sigma = 0.0
        self.scale = 0.0
        self.first = List[Float32]()
        self.second = List[Float32]()
        self.third = List[Float32]()
        self.fourth = List[Float32]()
        self.indices = List[Int32]()
        self.scalars = List[Float64]()

    def clear(mut self):
        self.fitted = False
        self.cols = 0
        self.components = 0
        self.sweeps = 0
        self.sigma = 0.0
        self.scale = 0.0
        self.first = List[Float32]()
        self.second = List[Float32]()
        self.third = List[Float32]()
        self.fourth = List[Float32]()
        self.indices = List[Int32]()
        self.scalars = List[Float64]()


struct KernelApproxMemberStore(Defaultable, Movable):
    var synchronization: List[Int64]
    var chunks: List[List[KernelApproxMemberSlot]]
    var free: List[Int]
    var count: Int

    def __init__(out self):
        # Side-effect-free even when the process slot's initializers race.
        self.synchronization = List[Int64](length=2, fill=0)
        self.chunks = List[List[KernelApproxMemberSlot]]()
        self.free = List[Int]()
        self.count = 0


comptime _STORE = SharedSlot[KernelApproxMemberStore, "MojolearnKernelApproxMemberStoreV2", 2]
comptime _Slot = MutPointer[KernelApproxMemberSlot, MutUntrackedOrigin]
comptime _Call = MutPointer[MemberCall, MutUntrackedOrigin]


def _slot_address(index: Int) raises -> Int:
    # Under the store's lock. A chunk's buffer is allocated once and never
    # grows, so the address stays valid after the lock is released.
    var store = _STORE.get_or_create_ptr()
    return Int(store[].chunks[index // _CHUNK].unsafe_ptr()) + (index % _CHUNK) * size_of[KernelApproxMemberSlot]()


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
                raise Error("RuntimeError: mojolearn: too many live kernel approximation member states")
            if store[].count == len(store[].chunks) * _CHUNK:
                store[].chunks.append(List[KernelApproxMemberSlot](length=_CHUNK, fill=KernelApproxMemberSlot()))
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
            slot[].params = List[Float64]()
            store[].free.append(index)
        return address
    finally:
        shared_slot_unlock(lock)


def _run(entry: NativeDescriptorCall, call: KmCall) raises -> Int:
    """One provider request (the estimator's `km_call_run`, on the route's
    entry instead of a Python-held record)."""
    return native_descriptor_leased_call(entry, 0, call.operation, call.a, call.p, call.f)


def _nystroem(slot: _Slot) -> Bool:
    return Int(slot[].params[0]) == _KIND_NYSTROEM


def _name(slot: _Slot) -> String:
    return String("Nystroem") if _nystroem(slot) else String("RBFSampler")


def _seed(slot: _Slot) -> Int:
    return (Int(slot[].params[8]) << 32) | (Int(slot[].params[7]) & 0xFFFFFFFF)


def _kernel(slot: _Slot) -> KmKernel:
    return KmKernel(Int(slot[].params[1]), Int(slot[].params[2]), slot[].gamma, slot[].params[5])


def _f32(values: List[Float32]) -> Int:
    return Int(values.unsafe_ptr())


def _f32p(address: Int) -> MutPointer[Float32, MutUntrackedOrigin]:
    return MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=address)


def _x(slot: _Slot, call: _Call) raises:
    """The view checks and the estimator's admission tests of X."""
    var estimator = _name(slot)
    if call[].row_index.rows > 0 or call[].column_index.rows > 0:
        raise Error("TypeError: a " + estimator + " member reads whole host views (no row or column index)")
    if not call[].x.dense_host(MEMBER_F32):
        raise Error("TypeError: a " + estimator + " member reads X as a dense host float32 view")
    var count = call[].x.rows * call[].x.cols
    if count == 0:
        return
    if not host_all_finite_f32(_f32p(call[].x.data), count):
        raise Error("ValueError: " + admitted_finite_wording(estimator, "X"))
    var kernel = Int(slot[].params[1])
    if _nystroem(slot) and (kernel == _KERNEL_CHI2 or kernel == _KERNEL_ADDITIVE_CHI2):
        if Float64(family_min_f32(_f32p(call[].x.data), count)) < 0.0:
            raise Error("mojolearn Nystroem: X contains negative values (the chi2 kernels need x >= 0)")


def _fit(slot: _Slot, call: _Call) raises:
    if call[].fit_output != 0:
        member_unsupported(_name(slot), call[].fit_output)
    if call[].weights.rows > 0:
        # No MEMBER_CAN_SAMPLE_WEIGHT: a given weight is refused, never ignored.
        raise Error("member ABI: a " + _name(slot) + " member takes no sample weights")
    _x(slot, call)
    var n = call[].x.rows
    var d = call[].x.cols
    var q = Int(slot[].params[6])
    var gamma_kind = Int(slot[].params[3])
    var kernel = Int(slot[].params[1])
    var route = call[].route
    var gamma = slot[].params[4]
    if gamma_kind == 1:
        gamma = 1.0 if kernel == _KERNEL_CHI2 else 1.0 / Float64(d)
    elif gamma_kind == 2:
        # `_km_scale_gamma`: the provider's exact limbs of X's cells, the one
        # rounding in exact integer math.
        var cells = n * d
        if cells == 0:
            raise Error("gamma='scale' needs at least one cell of X")
        var words = km_scale_gamma_words()
        var limbs = KmCall(KM_OP_SCALE_GAMMA_LIMBS, 1, 0)
        limbs.address(call[].x.data)
        limbs.address(Int(words.unsafe_ptr()))
        limbs.p[0] = cells
        _ = _run(route.entry.call, limbs)
        gamma = km_scale_gamma_value(words, cells, d)
    slot[].clear()
    slot[].gamma = gamma
    slot[].cols = d
    if _nystroem(slot):
        # scikit-learn's rule: n_components above the rows becomes the rows.
        if q > n:
            q = n
        var components = List[Float32](length=max(1, q * d), fill=Float32(0))
        var indices = List[Int32](length=max(1, q), fill=Int32(0))
        var normalization = List[Float32](length=max(1, q * q), fill=Float32(0))
        var eigenvalues = List[Float32](length=max(1, q), fill=Float32(0))
        var eigenvectors = List[Float32](length=max(1, q * q), fill=Float32(0))
        var scalars = List[Float64](length=1, fill=0.0)
        var operation = KM_OP_NYS_FIT_PTR if (route.flags & 2) != 0 else KM_OP_NYS_FIT
        var request = km_nys_fit_call(operation, call[].x.data, _f32(components), Int(indices.unsafe_ptr()),
            _f32(normalization), _f32(eigenvalues), _f32(eigenvectors), Int(scalars.unsafe_ptr()), n, d,
            _kernel(slot), q, _seed(slot))
        slot[].sweeps = _run(route.entry.call, request)
        slot[].first = components^
        slot[].second = normalization^
        slot[].third = eigenvalues^
        slot[].fourth = eigenvectors^
        slot[].indices = indices^
        _ = scalars^
    else:
        var weights = List[Float32](length=max(1, d * q), fill=Float32(0))
        var offset = List[Float32](length=max(1, q), fill=Float32(0))
        var scalars = List[Float64](length=2, fill=0.0)
        var request = km_rbf_fit_call(_f32(weights), _f32(offset), Int(scalars.unsafe_ptr()), d, q, gamma,
            _seed(slot))
        _ = _run(route.entry.call, request)
        slot[].sigma = scalars[0]
        slot[].scale = scalars[1]
        slot[].first = weights^
        slot[].second = offset^
    slot[].components = q
    slot[].fitted = True


def _transform(slot: _Slot, call: _Call) raises:
    var estimator = _name(slot)
    if not slot[].fitted:
        raise Error("NotFittedError: this " + estimator + " member is not fitted yet")
    _x(slot, call)
    var m = call[].x.rows
    var d = slot[].cols
    var q = slot[].components
    if call[].x.cols != d:
        raise Error("mojolearn " + estimator + ": X has " + String(call[].x.cols) + " features, the fit had "
            + String(d))
    if not call[].dst.dense_host(MEMBER_F32) or call[].dst.rows != m or call[].dst.cols != q:
        raise Error("TypeError: a " + estimator + " member writes a dense host float32 rows x n_components buffer")
    if m == 0:
        return
    var entry = call[].route.entry.call
    if _nystroem(slot):
        var request = km_nys_transform_call(_f32(slot[].first), Int(slot[].indices.unsafe_ptr()), _f32(slot[].second),
            _f32(slot[].third), _f32(slot[].fourth), call[].x.data, call[].dst.data, q, d, _kernel(slot), _seed(slot),
            slot[].sweeps, m)
        _ = _run(entry, request)
        return
    var request = km_rbf_transform_call(_f32(slot[].first), _f32(slot[].second), call[].x.data, call[].dst.data,
        d, q, slot[].gamma, _seed(slot), slot[].sigma, slot[].scale, m)
    _ = _run(entry, request)


def _blocks(slot: _Slot) -> Int:
    return 6 if _nystroem(slot) else 3


def _export(slot: _Slot, call: _Call) raises:
    """A view of one fitted block. The slot owns the words; they stay valid
    until this state's next fit, import or free."""
    if not slot[].fitted:
        raise Error("NotFittedError: this " + _name(slot) + " member is not fitted yet")
    var count = _blocks(slot)
    var block = call[].block
    if (call[].flags & MEMBER_BLOCK_BY_INDEX) != 0:
        if block < 0 or block >= count:
            raise Error("member ABI: this " + _name(slot) + " member has " + String(count) + " fitted blocks")
        block = MEMBER_BLOCK_SCALARS if block == 0 else _B_FIRST + block - 1
    var q = slot[].components
    var d = slot[].cols
    if block == MEMBER_BLOCK_SCALARS:
        slot[].scalars = List[Float64](length=_SCALARS, fill=0.0)
        slot[].scalars[0] = slot[].params[0]
        slot[].scalars[1] = slot[].params[1]
        slot[].scalars[2] = slot[].params[2]
        slot[].scalars[3] = slot[].gamma
        slot[].scalars[4] = slot[].params[5]
        slot[].scalars[5] = Float64(q)
        slot[].scalars[6] = Float64(d)
        slot[].scalars[7] = Float64(slot[].sweeps)
        slot[].scalars[8] = slot[].sigma
        slot[].scalars[9] = slot[].scale
        slot[].scalars[10] = slot[].params[7]
        slot[].scalars[11] = slot[].params[8]
        call[].dst = MemberMatrix(Int(slot[].scalars.unsafe_ptr()), 1, _SCALARS, MEMBER_F64)
    elif _nystroem(slot) and block == _B_FIRST:
        call[].dst = MemberMatrix(_f32(slot[].first), q, d)
    elif _nystroem(slot) and block == _B_FIRST + 1:
        call[].dst = MemberMatrix(Int(slot[].indices.unsafe_ptr()), 1, q, MEMBER_I32)
    elif _nystroem(slot) and block == _B_FIRST + 2:
        call[].dst = MemberMatrix(_f32(slot[].second), q, q)
    elif _nystroem(slot) and block == _B_FIRST + 3:
        call[].dst = MemberMatrix(_f32(slot[].third), 1, q)
    elif _nystroem(slot) and block == _B_FIRST + 4:
        call[].dst = MemberMatrix(_f32(slot[].fourth), q, q)
    elif not _nystroem(slot) and block == _B_FIRST:
        call[].dst = MemberMatrix(_f32(slot[].first), d, q)
    elif not _nystroem(slot) and block == _B_FIRST + 1:
        call[].dst = MemberMatrix(_f32(slot[].second), 1, q)
    else:
        raise Error("member ABI: a " + _name(slot) + " member has no fitted block " + String(block))
    call[].block = block
    call[].out_cols = count


def _copy_f32(view: MemberMatrix, count: Int, what: String) raises -> List[Float32]:
    if not view.dense_host(MEMBER_F32) or view.rows * view.cols != count:
        raise Error("member ABI: the " + what + " block is a dense host float32 view of " + String(count)
            + " words")
    var words = List[Float32](length=max(1, count), fill=Float32(0))
    if count > 0:
        unsafe_memcpy(dest=words.unsafe_ptr(), src=_f32p(view.data), count=count)
    return words^


def _import(slot: _Slot, call: _Call) raises:
    """Copy one fitted block in; MEMBER_IMPORT_DONE completes the state.
    The scalars come first."""
    var block = call[].block
    var q = slot[].components
    var d = slot[].cols
    if block == MEMBER_BLOCK_SCALARS:
        if not call[].x.dense_host(MEMBER_F64) or call[].x.rows * call[].x.cols != _SCALARS:
            raise Error("member ABI: the kernel approximation scalars block is twelve float64 words")
        var words = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call[].x.data)
        if Int(words[0]) != Int(slot[].params[0]):
            raise Error("member ABI: the scalars block holds another kernel approximation kind")
        slot[].clear()
        slot[].gamma = words[3]
        slot[].components = Int(words[5])
        slot[].cols = Int(words[6])
        slot[].sweeps = Int(words[7])
        slot[].sigma = words[8]
        slot[].scale = words[9]
    elif _nystroem(slot) and block == _B_FIRST:
        slot[].first = _copy_f32(call[].x, q * d, "components_")
    elif _nystroem(slot) and block == _B_FIRST + 1:
        if not call[].x.dense_host(MEMBER_I32) or call[].x.rows * call[].x.cols != q:
            raise Error("member ABI: the component_indices_ block is a dense host int32 view of " + String(q)
                + " words")
        slot[].indices = List[Int32](length=max(1, q), fill=Int32(0))
        if q > 0:
            unsafe_memcpy(dest=slot[].indices.unsafe_ptr(),
                src=MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=call[].x.data), count=q)
    elif _nystroem(slot) and block == _B_FIRST + 2:
        slot[].second = _copy_f32(call[].x, q * q, "normalization_")
    elif _nystroem(slot) and block == _B_FIRST + 3:
        slot[].third = _copy_f32(call[].x, q, "eigenvalues_")
    elif _nystroem(slot) and block == _B_FIRST + 4:
        slot[].fourth = _copy_f32(call[].x, q * q, "eigenvectors_")
    elif not _nystroem(slot) and block == _B_FIRST:
        slot[].first = _copy_f32(call[].x, d * q, "random_weights_")
    elif not _nystroem(slot) and block == _B_FIRST + 1:
        slot[].second = _copy_f32(call[].x, q, "random_offset_")
    elif block != 0 or (call[].flags & MEMBER_IMPORT_DONE) == 0:
        raise Error("member ABI: a " + _name(slot) + " member has no fitted block " + String(block))
    if (call[].flags & MEMBER_IMPORT_DONE) == 0:
        return
    var complete = q > 0 and d > 0 and len(slot[].first) > 0 and len(slot[].second) > 0
    if _nystroem(slot):
        complete = complete and len(slot[].third) > 0 and len(slot[].fourth) > 0 and len(slot[].indices) > 0
    if not complete:
        raise Error("member ABI: a fitted " + _name(slot) + " state needs every block of its kind")
    slot[].fitted = True


def _kernel_approx_member(operation: Int, address: Int) raises -> Int:
    var call = _Call(unsafe_from_address=address)
    if operation == MEMBER_OP_CAPABILITIES:
        # TEST_ROWS: `_x` tests X at fit and transform (no y, no weights).
        return MEMBER_CAN_FIT | MEMBER_CAN_TRANSFORM | MEMBER_CAN_EXPORT | MEMBER_CAN_IMPORT | MEMBER_CAN_TEST_ROWS
    if operation == MEMBER_OP_NEW:
        if call[].param_count != _PARAMS or not call[].params:
            raise Error("member ABI: a kernel approximation member takes " + String(_PARAMS) + " parameters")
        var source = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call[].params)
        var words = List[Float64](length=_PARAMS, fill=0.0)
        for i in range(_PARAMS):  # small-loop(the member's parameter words)
            words[i] = source[i]
        if words[0] != 0.0 and words[0] != 1.0:
            raise Error("member ABI: unknown kernel approximation member kind")
        if Int(words[1]) == _KERNEL_PRECOMPUTED:
            raise Error("member ABI: kernel='precomputed' is refused by name (Nystroem's own rule)")
        if Int(words[6]) < 1:
            raise Error("mojolearn: n_components must be positive, got " + String(Int(words[6])))
        call[].state = _mint(words^)
        return 0
    if operation == MEMBER_OP_FREE:
        _ = _lookup(call[].state, True)
        call[].state = 0
        return 0
    if operation == MEMBER_OP_OUTPUTS:
        # transform: float32 x n_components. A fitted state answers its own
        # width; without one RBFSampler answers n_components from its NEW
        # words, while Nystroem's width is chosen by the fit
        # (min(n_components, rows)) and is refused in words.
        if call[].flags != MEMBER_OP_TRANSFORM:
            member_unsupported("a kernel approximation member", call[].flags)
        var width = 0
        if call[].state != 0:
            var held = _Slot(unsafe_from_address=_lookup(call[].state, False))
            if held[].fitted:
                width = held[].components
            elif not _nystroem(held):
                width = Int(held[].params[6])
        elif call[].param_count == _PARAMS and call[].params:
            var named = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call[].params)
            if named[0] == 1.0:
                width = Int(named[6])
        if width == 0:
            raise Error("member ABI: Nystroem's transform width is chosen by its fit (min(n_components, rows)); "
                + "ask the fitted state")
        call[].out_cols = width
        call[].out_dtype = MEMBER_F32
        call[].classes = 0
        return 0
    var slot = _Slot(unsafe_from_address=_lookup(call[].state, False))
    if operation == MEMBER_OP_FIT:
        _fit(slot, call)
    elif operation == MEMBER_OP_TRANSFORM:
        _transform(slot, call)
    elif operation == MEMBER_OP_EXPORT:
        _export(slot, call)
    elif operation == MEMBER_OP_IMPORT:
        _import(slot, call)
    else:
        member_unsupported(_name(slot), operation)
    return 0


def kernel_approx_member_v2(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
    result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """The member entry of Nystroem and RBFSampler (schema `MEMBER_ABI`)."""
    try:
        var word = _kernel_approx_member(operation, member_call_from_spans(operation, aa, na, np, nf))
        if result:
            MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = word
        return 0
    except error:
        native_descriptor_report(error, context, sink)
        return 1


def kernel_approx_member_publish(mode: Int, vendor: Int, publisher: Int) raises:
    """Publish this image's kernel approximation member entry. One line in
    each kernel methods image's module initialization."""
    member_publish[MEMBER_FAMILY_KERNEL_APPROX](mode, vendor, publisher, kernel_approx_member_v2)
