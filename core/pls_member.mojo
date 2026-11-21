# SPDX-License-Identifier: Apache-2.0
"""PLSRegression, PLSCanonical and CCA as MEMBERS (core/member_abi.mojo).

One entry, `pls_member_v2`, published by each `_mojolearn_x_decomp` image
(`pls_member_publish`, beside its decomposition table). It runs the family's
existing typed RUN (bindings/pls_run.mojo over the typed kit,
bindings/decomp_kit_run.mojo: the centering and scaling, the one
XD_PLS_FIT component loop, the rotations, the coefficients, the
projections) through the route's entry, the x_decomp dispatch of the
estimator's own device route. No kernel is touched.

ROUTE SLOTS (filled by providers/python/member_pls_boundary_door.mojo):
`entry` the x_decomp dispatch of the route's image; `flags` = bit 0 the image
published its table (keyed), bit 1 it is resident (a GPU image), bits 2.. the
kit's `DK_HAS_*` optional-entry mask. A kit is built per member call from
these words, as the estimator's own door builds one per call.

MEMBER_OP_NEW parameters (Float64 words, 8): [0] kind (0 PLSRegression,
1 PLSCanonical, 2 CCA); [1] n_components; [2] max_iter; [3] mode B (0 / 1:
CCA); [4] canonical deflation (0 / 1); [5] svd algorithm (0 nipals, 1 svd);
[6] scale (0 / 1); [7] tol.

VIEWS: X dense row-major host float32; y host float32 rows x targets
(MEMBER_CAN_MULTI_OUTPUT). predict -> dst rows x targets; transform -> dst
rows x components (X only); inverse_transform: x = rows x components ->
dst rows x features. No sample weights; no score (the R^2 runs on the
metrics owner, not split from Python: the caller predicts and runs its own
metric). A device-home view, a row or column index and a nonzero
`fit_output` are refused in words.

FITTED-STATE BLOCKS, in index order: 0 `MEMBER_BLOCK_SCALARS` float64
[components fitted, features, targets, training rows]; 1 `MEMBER_BLOCK_COEF`
float32 targets x features; 2 `MEMBER_BLOCK_INTERCEPT` float32 1 x targets
(the target mean); then float32 family blocks `MEMBER_BLOCK_FAMILY + j`:
j 0 x mean (1 x p), 1 x std (1 x p), 2 y std (1 x q), 3 x_weights (p x c),
4 y_weights (q x c), 5 x_loadings (p x c), 6 y_loadings (q x c),
7 x_rotations (p x c), 8 y_rotations (q x c), 9 x scores (n x c), 10 y scores
(n x c), 11 the per-component iteration counts (1 x c'). An import needs
blocks 0, 1, 2 and j 0, 1; an operation whose block was not imported
refuses in words. FIT NOTICES (`MEMBER_BLOCK_NOTICES` by name): the door's
"y residual is constant" UserWarning (core/member_notices.mojo).

The fitted state is OWNED here, in a process-wide store of fixed chunks
(lock only for mint, free and lookup). RESIDENT OPERANDS (lane fx-audit 8): on
a GPU image a query operand's device copy (x mean, x std, y mean, coef,
x rotations, x loadings) is uploaded at the first query that reads it and kept
in the slot, as the door's resident record keeps it
(bindings/pls_boundary_unpack.mojo `_fitted_matrix`), and freed at the next
fit, import or free; it is added to each query's kit as the door adds it
(`add_resident`: same routes, same words). After the first query: one
acquire load per operand. SERIAL BY NATURE: the store's
bookkeeping and the per-block driver loops; the numerical work is the
image's own kernels.
"""
from std.sys.info import size_of
from std.memory import unsafe_memcpy
from std.atomic import Atomic, Ordering
from bindings.decomp_kit_run import DKit
from core.entry_table import EntryRef, entry_ref_call, entry_ref_unresolved
from x_decomp.descriptor_schema import XD_DEV_FREE
from bindings.pls_run import PlsFitWords, pls_fit_run, pls_project, pls_rebuild, pls_predict_run
from core.device_route import DeviceRoute
from core.member_abi import (
    MemberCall, MemberMatrix, MEMBER_F32, MEMBER_F64,
    MEMBER_OP_CAPABILITIES, MEMBER_OP_NEW, MEMBER_OP_FREE, MEMBER_OP_FIT, MEMBER_OP_PREDICT,
    MEMBER_OP_TRANSFORM, MEMBER_OP_INVERSE_TRANSFORM, MEMBER_OP_OUTPUTS, MEMBER_OP_EXPORT, MEMBER_OP_IMPORT,
    MEMBER_CAN_FIT, MEMBER_CAN_PREDICT, MEMBER_CAN_TRANSFORM, MEMBER_CAN_INVERSE_TRANSFORM,
    MEMBER_CAN_EXPORT, MEMBER_CAN_IMPORT, MEMBER_CAN_COEF, MEMBER_CAN_MULTI_OUTPUT, MEMBER_CAN_NOTICES,
    MEMBER_BLOCK_COEF, MEMBER_BLOCK_INTERCEPT, MEMBER_BLOCK_SCALARS, MEMBER_BLOCK_FAMILY, MEMBER_BLOCK_BY_INDEX,
    MEMBER_BLOCK_NOTICES, MEMBER_IMPORT_DONE,
    member_publish, member_call_from_spans, member_unsupported,
)
from core.member_families import MEMBER_FAMILY_PLS
from core.member_notices import member_notice_add, MEMBER_NOTICE_PLS_RESIDUAL, MEMBER_NOTICE_WORDS
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.shared_slot import SharedSlot, shared_slot_lock, shared_slot_unlock

comptime _CHUNK = 256
comptime _INDEX_SPAN = 16777216
comptime _BLOCKS = 15          # index 0 is the float64 scalars; 1 .. 14 float32
comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _F64P = MutPointer[Float64, MutUntrackedOrigin]
# Index of each block in `PlsMemberSlot.blocks` (and in the by-index walk).
comptime _I_COEF = 1
comptime _I_YMEAN = 2
comptime _I_XMEAN = 3
comptime _I_XSTD = 4
comptime _I_YSTD = 5
comptime _I_XLOAD = 8
comptime _I_XROT = 10


struct PlsMemberSlot(Copyable, Movable, Defaultable):
    var generation: Int
    var live: Bool
    var fitted: Bool
    var kind: Int
    var words: List[Float64]
    var scalars: List[Float64]
    var blocks: List[List[Float32]]
    var rows: List[Int]
    var cols: List[Int]
    # FIT NOTICES (core/member_notices.mojo): "y residual is constant", set by
    # the fit, cleared with the blocks.
    var notices: List[Float64]
    # RESIDENT OPERANDS (module note): per block its device copy (-1: none),
    # the bit mask of blocks whose copy is published (acquire / release), the
    # lock word of a first upload and the x_decomp entry that frees them.
    var dev: List[Int]
    var dev_ready: Int64
    var dev_lock: Int64
    var dev_entry: EntryRef

    def __init__(out self):
        self.generation = 0
        self.live = False
        self.fitted = False
        self.kind = 0
        self.words = List[Float64]()
        self.scalars = List[Float64]()
        self.blocks = List[List[Float32]]()
        self.rows = List[Int]()
        self.cols = List[Int]()
        self.notices = List[Float64]()
        self.dev = List[Int]()
        self.dev_ready = 0
        self.dev_lock = 0
        self.dev_entry = entry_ref_unresolved()


struct PlsMemberStore(Defaultable, Movable):
    var synchronization: List[Int64]
    var chunks: List[List[PlsMemberSlot]]
    var free: List[Int]
    var count: Int

    def __init__(out self):
        # Side-effect-free even when the process slot's initializers race.
        self.synchronization = List[Int64](length=2, fill=0)
        self.chunks = List[List[PlsMemberSlot]]()
        self.free = List[Int]()
        self.count = 0


comptime _STORE = SharedSlot[PlsMemberStore, "MojolearnPlsMemberStoreV2", 2]
comptime _SlotP = MutPointer[PlsMemberSlot, MutUntrackedOrigin]
comptime _CallP = MutPointer[MemberCall, MutUntrackedOrigin]


def _slot_address(index: Int) raises -> Int:
    # Under the store's lock; a chunk's buffer never moves.
    var store = _STORE.get_or_create_ptr()
    return Int(store[].chunks[index // _CHUNK].unsafe_ptr()) + (index % _CHUNK) * size_of[PlsMemberSlot]()


def _empty_blocks(mut slot: PlsMemberSlot):
    slot.blocks = List[List[Float32]](length=_BLOCKS, fill=List[Float32]())
    slot.rows = List[Int](length=_BLOCKS, fill=0)
    slot.cols = List[Int](length=_BLOCKS, fill=0)
    slot.scalars = List[Float64]()
    slot.notices = List[Float64]()


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
                raise Error("RuntimeError: mojolearn: too many live PLS member states")
            if store[].count == len(store[].chunks) * _CHUNK:
                store[].chunks.append(List[PlsMemberSlot](length=_CHUNK, fill=PlsMemberSlot()))
            index = store[].count
            store[].count += 1
        var slot = _SlotP(unsafe_from_address=_slot_address(index))
        slot[].generation += 1
        slot[].live = True
        slot[].fitted = False
        slot[].kind = Int(words[0])
        slot[].words = words^
        _empty_blocks(slot[])
        slot[].dev = List[Int](length=_BLOCKS, fill=-1)
        slot[].dev_ready = 0
        return slot[].generation * _INDEX_SPAN + index + 1
    finally:
        shared_slot_unlock(lock)


def _lookup(handle: Int, release: Bool) raises -> Int:
    """The slot address of a live handle; a released, foreign or made-up
    handle is refused: never a stale read."""
    var store = _STORE.get_or_create_ptr()
    var lock = rebind[MutPointer[Int64, MutUntrackedOrigin]](store[].synchronization.unsafe_ptr())
    var held = List[Int]()
    var held_entry = entry_ref_unresolved()
    shared_slot_lock(lock)
    try:
        var index = handle % _INDEX_SPAN - 1
        if handle <= 0 or index < 0 or index >= store[].count:
            raise Error("NotFittedError: this member state was released or never created")
        var address = _slot_address(index)
        var slot = _SlotP(unsafe_from_address=address)
        if not slot[].live or slot[].generation != handle // _INDEX_SPAN:
            raise Error("NotFittedError: this member state was released or never created")
        if release:
            slot[].live = False
            slot[].fitted = False
            slot[].words = List[Float64]()
            _empty_blocks(slot[])
            # The resident copies leave under the lock and are freed after it.
            held = slot[].dev.copy()
            held_entry = slot[].dev_entry
            slot[].dev = List[Int]()
            slot[].dev_ready = 0
            slot[].dev_entry = entry_ref_unresolved()
            store[].free.append(index)
        return address
    finally:
        shared_slot_unlock(lock)
        _free_device(held, held_entry)


def _free_device(handles: List[Int], entry: EntryRef):
    """Return resident operand copies to the x_decomp pool (best effort: a
    failing free leaks one small buffer, never a refusal of the operation)."""
    for i in range(len(handles)):  # driver-loop: over the fitted blocks (fifteen)
        if handles[i] >= 0:
            try:
                _ = entry_ref_call(entry, XD_DEV_FREE, List[Int](), [handles[i]], List[Float64]())
            except:
                pass


def _drop_device(slot: _SlotP):
    """Free this state's resident operand copies before its blocks change (a
    fit, an import). The state is owned by its caller there: no query runs."""
    if Atomic.load[ordering=Ordering.ACQUIRE](
            MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(UnsafePointer(to=slot[].dev_ready)))) == 0:
        return
    var handles = slot[].dev.copy()
    var entry = slot[].dev_entry
    slot[].dev = List[Int](length=_BLOCKS, fill=-1)
    slot[].dev_ready = 0
    slot[].dev_entry = entry_ref_unresolved()
    _free_device(handles, entry)


def _name(kind: Int) -> String:
    return String("PLSRegression") if kind == 0 else (String("PLSCanonical") if kind == 1 else String("CCA"))


def _dense(view: MemberMatrix, what: String, estimator: String) raises:
    if not view.dense_host(MEMBER_F32):
        raise Error("TypeError: a " + estimator + " member reads " + what + " as a dense host float32 view")


def _whole(call: _CallP, estimator: String) raises:
    if call[].row_index.rows > 0 or call[].column_index.rows > 0:
        raise Error("TypeError: a " + estimator + " member reads whole host views (no row or column index)")


def _kit(route: DeviceRoute) -> DKit:
    """The typed kit of the route's x_decomp image, as the door's
    `decomp_kit_unpack` builds it (one per operation)."""
    return DKit(route.entry, (route.flags & 1) != 0, (route.flags & 2) != 0, route.flags >> 2)


def _keep(mut kit: DKit, slot: _SlotP, index: Int, a: Int) raises:
    """Copy kit matrix `a` into the slot's block `index` (host words)."""
    kit.host(a)
    var count = kit.size(a)
    var words = List[Float32](length=count, fill=0.0)
    if count:
        unsafe_memcpy(dest=words.unsafe_ptr(), src=kit.ptr(a), count=count)
    slot[].blocks[index] = words^
    slot[].rows[index] = kit.nrows(a)
    slot[].cols[index] = kit.ncols(a)


def _block(mut kit: DKit, slot: _SlotP, index: Int, estimator: String) raises -> Int:
    """A fitted query operand. On a GPU image its device copy is the slot's
    (uploaded at the first query that reads it: module note), added as the
    door adds its resident record's (`add_resident`: host words first, so no
    route or bit changes); elsewhere the host words borrowed in place."""
    if len(slot[].blocks[index]) == 0:
        raise Error("NotFittedError: this " + estimator + " member state lacks the fitted block this operation reads")
    var address = Int(slot[].blocks[index].unsafe_ptr())
    var rows = slot[].rows[index]
    var cols = slot[].cols[index]
    var n = rows * cols
    if not kit.resident or not kit.keyed or n < 1:
        return kit.borrow(address, rows, cols)
    var bit = Int64(1) << Int64(index)
    var ready = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(UnsafePointer(to=slot[].dev_ready)))
    if (Atomic.load[ordering=Ordering.ACQUIRE](ready) & bit) == 0:
        var lock = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(UnsafePointer(to=slot[].dev_lock)))
        shared_slot_lock(lock)
        try:
            var mask = Atomic.load[ordering=Ordering.ACQUIRE](ready)
            if (mask & bit) == 0:
                var handle = kit.dev_alloc(n)
                try:
                    kit.dev_upload(handle, address, n)
                except error:
                    kit.dev_free(handle)
                    raise error^
                slot[].dev[index] = handle
                slot[].dev_entry = kit.entry
                Atomic.store[ordering=Ordering.RELEASE](ready, mask | bit)
        finally:
            shared_slot_unlock(lock)
    return kit.add_resident(address, slot[].dev[index], rows, cols)


def _write(mut kit: DKit, a: Int, dst: MemberMatrix, estimator: String) raises:
    _dense(dst, "the output buffer", estimator)
    if dst.rows != kit.nrows(a) or dst.cols != kit.ncols(a):
        raise Error("ValueError: the output buffer of a " + estimator + " member holds " + String(kit.nrows(a))
            + " x " + String(kit.ncols(a)) + " values")
    kit.host(a)
    var count = kit.size(a)
    if count:
        unsafe_memcpy(dest=_F32P(unsafe_from_address=dst.data), src=kit.ptr(a), count=count)


def _fit(slot: _SlotP, call: _CallP) raises:
    var estimator = _name(slot[].kind)
    _whole(call, estimator)
    if call[].fit_output != 0:
        member_unsupported(estimator, call[].fit_output)
    if call[].weights.rows > 0:
        raise Error("TypeError: a " + estimator + " member fits without sample weights")
    _dense(call[].x, "X", estimator)
    _dense(call[].y, "y", estimator)
    var n = call[].x.rows
    var p = call[].x.cols
    var q = call[].y.cols
    if call[].y.rows != n:
        raise Error("Found input variables with inconsistent numbers of samples: [" + String(n) + ", "
            + String(call[].y.rows) + "]")
    if n < 2:
        raise Error("Found array with " + String(n) + " sample(s) (shape=(" + String(n) + ", " + String(p)
            + ")) while a minimum of 2 is required.")
    var w = _F64P(unsafe_from_address=Int(slot[].words.unsafe_ptr()))
    var nc = Int(w[1])
    var canonical = w[4] != 0.0
    var bound = min(n, min(p, q)) if canonical else p
    if not (1 <= nc and nc <= bound):
        raise Error("n_components == " + String(nc) + ", while 1 <= n_components <= " + String(bound) + " is required")
    var words = PlsFitWords(nc, Int(w[2]), w[3] != 0.0, canonical, w[5] != 0.0, w[6] != 0.0, w[7])
    slot[].fitted = False
    _drop_device(slot)
    _empty_blocks(slot[])
    var kit = _kit(call[].route)
    try:
        var x = kit.borrow(call[].x.data, n, p)
        var ym = kit.borrow(call[].y.data, n, q)
        var fit = pls_fit_run(kit, x, ym, words, False)
        _keep(kit, slot, _I_COEF, fit.coef)
        _keep(kit, slot, _I_YMEAN, fit.y_mean)
        _keep(kit, slot, _I_XMEAN, fit.x_mean)
        _keep(kit, slot, _I_XSTD, fit.x_std)
        _keep(kit, slot, _I_YSTD, fit.y_std)
        # x_weights, y_weights, x_scores, y_scores, x_loadings, y_loadings
        var order: List[Int] = [6, 7, 12, 13, 8, 9]
        for k in range(6):  # driver-loop: over the six fitted blocks
            _keep(kit, slot, order[k], fit.outs[k])
        _keep(kit, slot, _I_XROT, fit.x_rotations)
        _keep(kit, slot, 11, fit.y_rotations)
        var counts = List[Float32](length=len(fit.iterations), fill=0.0)
        for k in range(len(fit.iterations)):  # driver-loop: over the fitted components
            counts[k] = Float32(Int(fit.iterations[k]))
        slot[].rows[14] = 1
        slot[].cols[14] = len(counts)
        slot[].blocks[14] = counts^
        slot[].scalars = List[Float64](length=4, fill=0.0)
        slot[].scalars[0] = Float64(fit.done)
        slot[].scalars[1] = Float64(p)
        slot[].scalars[2] = Float64(q)
        slot[].scalars[3] = Float64(n)
        # The door's "y residual is constant at iteration k" (`pls_fit_pack`).
        if fit.done < nc:
            member_notice_add(slot[].notices, MEMBER_NOTICE_PLS_RESIDUAL, Float64(fit.done))
    finally:
        kit.close()
    slot[].fitted = True


def _query(slot: _SlotP, call: _CallP, operation: Int) raises:
    var estimator = _name(slot[].kind)
    if not slot[].fitted:
        raise Error("NotFittedError: this " + estimator + " member is not fitted yet")
    _whole(call, estimator)
    _dense(call[].x, "X", estimator)
    var p = Int(slot[].scalars[1])
    var kit = _kit(call[].route)
    try:
        var rows = call[].x.rows
        var result: Int
        if operation == MEMBER_OP_INVERSE_TRANSFORM:
            var z = kit.borrow(call[].x.data, rows, call[].x.cols)
            if call[].x.cols != slot[].cols[_I_XLOAD]:
                raise Error("ValueError: X has " + String(call[].x.cols) + " components, the model has "
                    + String(slot[].cols[_I_XLOAD]))
            result = pls_rebuild(kit, z, _block(kit, slot, _I_XLOAD, estimator), _block(kit, slot, _I_XSTD, estimator),
                _block(kit, slot, _I_XMEAN, estimator))
        else:
            if call[].x.cols != p:
                raise Error("X has " + String(call[].x.cols) + " features, but " + estimator + " is expecting "
                    + String(p) + " features as input.")
            var x = kit.borrow(call[].x.data, rows, p)
            if operation == MEMBER_OP_PREDICT:
                result = pls_predict_run(kit, x, _block(kit, slot, _I_XMEAN, estimator),
                    _block(kit, slot, _I_COEF, estimator), _block(kit, slot, _I_YMEAN, estimator))
            else:
                result = pls_project(kit, x, _block(kit, slot, _I_XMEAN, estimator),
                    _block(kit, slot, _I_XSTD, estimator), _block(kit, slot, _I_XROT, estimator))
        _write(kit, result, call[].dst, estimator)
    finally:
        kit.close()


def _index_of(name: Int) -> Int:
    if name == MEMBER_BLOCK_SCALARS:
        return 0
    if name == MEMBER_BLOCK_COEF:
        return 1
    if name == MEMBER_BLOCK_INTERCEPT:
        return 2
    if name >= MEMBER_BLOCK_FAMILY and name < MEMBER_BLOCK_FAMILY + _BLOCKS - 3:
        return 3 + name - MEMBER_BLOCK_FAMILY
    return -1


def _name_of(index: Int) -> Int:
    if index == 0:
        return MEMBER_BLOCK_SCALARS
    if index == 1:
        return MEMBER_BLOCK_COEF
    if index == 2:
        return MEMBER_BLOCK_INTERCEPT
    return MEMBER_BLOCK_FAMILY + index - 3


def _export(slot: _SlotP, call: _CallP) raises:
    """A view of one fitted block (the slot owns the words until the state's
    next fit, import or free)."""
    if not slot[].fitted:
        raise Error("NotFittedError: this " + _name(slot[].kind) + " member is not fitted yet")
    var index: Int
    if (call[].flags & MEMBER_BLOCK_BY_INDEX) != 0:
        index = call[].block
    else:
        index = _index_of(call[].block)
    if index < 0 or index >= _BLOCKS:
        raise Error("member ABI: a PLS member has no fitted block " + String(call[].block))
    if index == 0:
        call[].dst = MemberMatrix(Int(slot[].scalars.unsafe_ptr()), 1, len(slot[].scalars), MEMBER_F64)
    else:
        call[].dst = MemberMatrix(Int(slot[].blocks[index].unsafe_ptr()), slot[].rows[index], slot[].cols[index])
    call[].block = _name_of(index)
    call[].out_cols = _BLOCKS


def _import(slot: _SlotP, call: _CallP) raises:
    """Copy one fitted block in; MEMBER_IMPORT_DONE completes the state."""
    var estimator = _name(slot[].kind)
    var index = _index_of(call[].block)
    _drop_device(slot)
    if index == 0:
        if not call[].x.dense_host(MEMBER_F64) or call[].x.rows * call[].x.cols != 4:
            raise Error("member ABI: the PLS scalars block is four host float64 words")
        slot[].fitted = False
        slot[].scalars = List[Float64](length=4, fill=0.0)
        unsafe_memcpy(dest=slot[].scalars.unsafe_ptr(), src=_F64P(unsafe_from_address=call[].x.data), count=4)
    elif index > 0:
        _dense(call[].x, "a fitted block", estimator)
        var count = call[].x.rows * call[].x.cols
        var words = List[Float32](length=count, fill=0.0)
        unsafe_memcpy(dest=words.unsafe_ptr(), src=_F32P(unsafe_from_address=call[].x.data), count=count)
        slot[].fitted = False
        slot[].blocks[index] = words^
        slot[].rows[index] = call[].x.rows
        slot[].cols[index] = call[].x.cols
    elif call[].block != 0 or (call[].flags & MEMBER_IMPORT_DONE) == 0:
        raise Error("member ABI: a PLS member has no fitted block " + String(call[].block))
    if (call[].flags & MEMBER_IMPORT_DONE) == 0:
        return
    if len(slot[].scalars) != 4:
        raise Error("member ABI: a fitted PLS state needs its scalars block")
    var p = Int(slot[].scalars[1])
    var q = Int(slot[].scalars[2])
    if (slot[].rows[_I_COEF] != q or slot[].cols[_I_COEF] != p or len(slot[].blocks[_I_YMEAN]) != q
            or len(slot[].blocks[_I_XMEAN]) != p or len(slot[].blocks[_I_XSTD]) != p):
        raise Error("member ABI: a fitted PLS state needs coef (targets x features), the target mean and the X "
            + "mean and std")
    slot[].fitted = True


def _outputs(call: _CallP) raises:
    """OUTPUTS with or without a state (item 10), float32 for every
    operation: predict = targets, transform = components, inverse_transform
    = features. A fitted state answers its own; otherwise the NEW words'
    n_components and the caller's shape words (`y.cols`, `x.cols`) do.
    Another operation is refused, never sized."""
    var asked = call[].flags
    var targets = max(1, call[].y.cols)
    var features = call[].x.cols
    var components = 0
    if call[].state != 0:
        var slot = _SlotP(unsafe_from_address=_lookup(call[].state, False))
        components = Int(slot[].words[1])
        if slot[].fitted:
            targets = Int(slot[].scalars[2])
            features = Int(slot[].scalars[1])
            components = slot[].cols[_I_XROT]
    elif call[].param_count == 8 and call[].params:
        components = Int(_F64P(unsafe_from_address=call[].params)[1])
    else:
        raise Error("member ABI: a PLS member's OUTPUTS without a state needs its NEW words")
    call[].classes = 0
    call[].out_dtype = MEMBER_F32
    if asked == MEMBER_OP_PREDICT:
        call[].out_cols = targets
    elif asked == MEMBER_OP_TRANSFORM:
        call[].out_cols = components
    elif asked == MEMBER_OP_INVERSE_TRANSFORM:
        call[].out_cols = features
    else:
        member_unsupported("a PLS member", asked)


def _new(call: _CallP) raises:
    if call[].param_count != 8 or not call[].params:
        raise Error("member ABI: a PLS member takes eight parameters")
    var w = _F64P(unsafe_from_address=call[].params)
    if w[0] != 0.0 and w[0] != 1.0 and w[0] != 2.0:
        raise Error("member ABI: unknown PLS member kind")
    var words = List[Float64](length=8, fill=0.0)
    unsafe_memcpy(dest=words.unsafe_ptr(), src=w, count=8)
    call[].state = _mint(words^)


def _pls_member(operation: Int, address: Int) raises -> Int:
    var call = _CallP(unsafe_from_address=address)
    if operation == MEMBER_OP_CAPABILITIES:
        return (MEMBER_CAN_FIT | MEMBER_CAN_PREDICT | MEMBER_CAN_TRANSFORM | MEMBER_CAN_INVERSE_TRANSFORM
            | MEMBER_CAN_EXPORT | MEMBER_CAN_IMPORT | MEMBER_CAN_COEF | MEMBER_CAN_MULTI_OUTPUT | MEMBER_CAN_NOTICES)
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
    var slot = _SlotP(unsafe_from_address=_lookup(call[].state, False))
    if operation == MEMBER_OP_FIT:
        _fit(slot, call)
    elif operation == MEMBER_OP_PREDICT or operation == MEMBER_OP_TRANSFORM or operation == MEMBER_OP_INVERSE_TRANSFORM:
        _query(slot, call, operation)
    elif operation == MEMBER_OP_EXPORT and call[].block == MEMBER_BLOCK_NOTICES and (
            call[].flags & MEMBER_BLOCK_BY_INDEX) == 0:
        call[].dst = MemberMatrix(Int(slot[].notices.unsafe_ptr()), len(slot[].notices) // MEMBER_NOTICE_WORDS,
                                  MEMBER_NOTICE_WORDS, MEMBER_F64)
    elif operation == MEMBER_OP_EXPORT:
        _export(slot, call)
    elif operation == MEMBER_OP_IMPORT:
        _import(slot, call)
    else:
        member_unsupported(_name(slot[].kind), operation)
    return 0


def pls_member_v2(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
    result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """The member entry of PLSRegression, PLSCanonical and CCA (schema `MEMBER_ABI`)."""
    try:
        var word = _pls_member(operation, member_call_from_spans(operation, aa, na, np, nf))
        if result:
            MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = word
        return 0
    except error:
        native_descriptor_report(error, context, sink)
        return 1


def pls_member_publish(mode: Int, vendor: Int, publisher: Int) raises:
    """Publish this image's PLS member entry. One line in each
    `_mojolearn_x_decomp` image's module initialization, after its
    decomposition table is published. A build with no GPU vendor
    (`vendor` < 0) publishes nothing."""
    if vendor < 0:
        return
    member_publish[MEMBER_FAMILY_PLS](mode, vendor, publisher, pls_member_v2)
