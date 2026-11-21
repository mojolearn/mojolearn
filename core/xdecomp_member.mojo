# SPDX-License-Identifier: Apache-2.0
"""The XDecomp family as MEMBERS (core/member_abi.mojo): IncrementalPCA
(kind 0) so far; the family's other classes are not published yet.

One entry, `xdecomp_member_v2`, published by each x_decomp image
(`xdecomp_member_publish`, beside its linear entry table publication:
x_decomp/publish_device_run.mojo, x_decomp/publish_host_run.mojo). It runs
the family's existing typed RUN, the IncrementalPCA native driver
(core/ipca_native_api.mojo `IPCANativeDriver`: IPCA_FIT, IPCA_QUERY,
IPCA_PREPARE, IPCA_FREE) over the entry of the `DeviceRoute` it is handed,
the entry the estimator's own door calls (bindings/ipca_native_door.mojo).
No kernel is touched. No score (the class has none).

Route slots read (IncrementalPCA): `entry` the x_decomp image's
IncrementalPCA entry; `flags` bit XDECOMP_ROUTE_RESIDENT (the entry keeps
resident device state, IPCA_CAP_RESIDENT): the fit keeps its device copies
(the door's `keep`), every transform / inverse_transform reads them (the
door's resident owners), a state imported from a fitted estimator prepares
them at its first query (the door's `ipca_owner_operand`). Released at
refit, import and free.

MEMBER_OP_NEW parameters (Float64 words), IncrementalPCA (4):
  [0] kind: 0 IncrementalPCA
  [1] n_components: -1 None, else the count
  [2] whiten: 0 / 1
  [3] batch_size: -1 None (5 * n_features at fit, as the door), else the size

Views: X dense row-major host float32, admitted finite by the caller's
UNPACK as the door's admission does; no y (ignored), no sample weights
(refused), no row or column index, no device-home view.
`call.fit_output = MEMBER_OP_TRANSFORM` writes the training rows'
projection to `dst` (rows x n_components_) in the fit call; with
n_components None the width is known after the fit (min(rows, features)):
the caller sizes `dst` from OUTPUTS or passes the count.

Fitted-state blocks (MEMBER_OP_EXPORT / MEMBER_OP_IMPORT), IncrementalPCA,
in index order:
  0 `MEMBER_BLOCK_SCALARS` float64 [n_components_, n_features_in_,
    n_samples_seen_, noise_variance_, batch_size_]
  1 `XDECOMP_BLOCK_COMPONENTS` (MEMBER_BLOCK_FAMILY) float32 n_components_ x n_features_in_
  2 `XDECOMP_BLOCK_SINGULAR` 1 x n_components_   (singular_values_)
  3 `XDECOMP_BLOCK_MEAN` 1 x n_features_in_       (mean_)
  4 `XDECOMP_BLOCK_VAR` 1 x n_features_in_        (var_)
  5 `XDECOMP_BLOCK_VARIANCE` 1 x n_components_   (explained_variance_)
  6 `XDECOMP_BLOCK_RATIO` 1 x n_components_      (explained_variance_ratio_)
An import needs every block.

The fitted state is OWNED here, in a process-wide store of fixed chunks
(the linear member's store): a slot never moves once minted, the lock covers
only mint, free and the handle lookup; resident handles are released after
the lock. Serial by nature here: the store's bookkeeping; the kernels are
the family's.
"""
from std.sys.info import size_of
from std.memory import unsafe_memcpy
from core.ipca_native_api import (
    IPCANativeDriver, IPCAFitRequest, IPCAQueryRequest, IPCAOutputs, IPCAStateIds,
)
from core.member_abi import (
    MemberCall, MemberMatrix, MEMBER_F32, MEMBER_F64,
    MEMBER_OP_CAPABILITIES, MEMBER_OP_NEW, MEMBER_OP_FREE, MEMBER_OP_FIT, MEMBER_OP_OUTPUTS,
    MEMBER_OP_TRANSFORM, MEMBER_OP_INVERSE_TRANSFORM, MEMBER_OP_EXPORT, MEMBER_OP_IMPORT,
    MEMBER_CAN_FIT, MEMBER_CAN_TRANSFORM, MEMBER_CAN_INVERSE_TRANSFORM, MEMBER_CAN_EXPORT, MEMBER_CAN_IMPORT,
    MEMBER_BLOCK_SCALARS, MEMBER_BLOCK_FAMILY, MEMBER_BLOCK_BY_INDEX, MEMBER_IMPORT_DONE,
    member_publish, member_call_from_spans, member_unsupported,
)
from core.member_families import MEMBER_FAMILY_XDECOMP
from core.entry_table import EntryRef, entry_ref_unresolved
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.shared_slot import SharedSlot, shared_slot_lock, shared_slot_unlock

comptime XDECOMP_KIND_IPCA = 0
comptime XDECOMP_IPCA_PARAM_COUNT = 4

comptime XDECOMP_BLOCK_COMPONENTS = MEMBER_BLOCK_FAMILY
comptime XDECOMP_BLOCK_SINGULAR = MEMBER_BLOCK_FAMILY + 1
comptime XDECOMP_BLOCK_MEAN = MEMBER_BLOCK_FAMILY + 2
comptime XDECOMP_BLOCK_VAR = MEMBER_BLOCK_FAMILY + 3
comptime XDECOMP_BLOCK_VARIANCE = MEMBER_BLOCK_FAMILY + 4
comptime XDECOMP_BLOCK_RATIO = MEMBER_BLOCK_FAMILY + 5

comptime XDECOMP_ROUTE_RESIDENT = 1

comptime _IPCA_SCALARS = 5
comptime _HANDLES = 5
comptime _CHUNK = 256
comptime _INDEX_SPAN = 16777216


struct XDecompMemberSlot(Copyable, Movable, Defaultable):
    var generation: Int
    var live: Bool
    var words: List[Float64]
    var fitted: Bool
    var count: Int
    var features: Int
    # IncrementalPCA: components, singular, mean, var, explained, ratio (the
    # door's six output rooms; a fit's rows are their prefixes).
    var blocks: List[List[Float32]]
    var scalars: List[Float64]
    # Resident device state (IPCAStateIds order: components, singular, mean,
    # var, explained variance), -1 none, and the entry that frees it.
    var handles: List[Int]
    var handle_entry: EntryRef

    def __init__(out self):
        self.generation = 0
        self.live = False
        self.words = List[Float64]()
        self.fitted = False
        self.count = 0
        self.features = 0
        self.blocks = List[List[Float32]]()
        self.scalars = List[Float64]()
        self.handles = List[Int](length=_HANDLES, fill=-1)
        self.handle_entry = entry_ref_unresolved()

    def kind(self) -> Int:
        return Int(self.words[0])

    def clear(mut self):
        """The fitted words. Resident handles are released by `_release`
        first (never dropped here: they are device memory)."""
        self.fitted = False
        self.count = 0
        self.features = 0
        self.blocks = List[List[Float32]]()
        self.scalars = List[Float64]()


struct XDecompMemberStore(Defaultable, Movable):
    var synchronization: List[Int64]
    var chunks: List[List[XDecompMemberSlot]]
    var free: List[Int]
    var count: Int

    def __init__(out self):
        # Side-effect-free even when the process slot's initializers race.
        self.synchronization = List[Int64](length=2, fill=0)
        self.chunks = List[List[XDecompMemberSlot]]()
        self.free = List[Int]()
        self.count = 0


comptime _STORE = SharedSlot[XDecompMemberStore, "MojolearnXDecompMemberStoreV2", 2]
comptime _Slot = MutPointer[XDecompMemberSlot, MutUntrackedOrigin]
comptime _Call = MutPointer[MemberCall, MutUntrackedOrigin]
comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]


def _slot_address(index: Int) raises -> Int:
    # Under the store's lock. A chunk's buffer is allocated once and never
    # grows, so the address stays valid after the lock is released.
    var store = _STORE.get_or_create_ptr()
    return Int(store[].chunks[index // _CHUNK].unsafe_ptr()) + (index % _CHUNK) * size_of[XDecompMemberSlot]()


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
                raise Error("RuntimeError: mojolearn: too many live XDecomp member states")
            if store[].count == len(store[].chunks) * _CHUNK:
                store[].chunks.append(List[XDecompMemberSlot](length=_CHUNK, fill=XDecompMemberSlot()))
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


def _free_handles(entry: EntryRef, handles: List[Int]) raises:
    """Release resident device state through the entry that made it. Every
    handle is attempted; the first failure is raised after the rest."""
    var failure = String("")
    for k in range(len(handles)):  # driver-loop: five resident state handles
        if handles[k] >= 0:
            try:
                IPCANativeDriver(entry.call).free(handles[k])
            except error:
                if len(failure) == 0:
                    failure = String(error)
    if len(failure) > 0:
        raise Error(failure)


def _lookup(handle: Int, release: Bool) raises -> Int:
    """The slot address of a live handle. A released handle, a handle of
    another store or a made-up integer is refused: never a stale read."""
    var store = _STORE.get_or_create_ptr()
    var lock = rebind[MutPointer[Int64, MutUntrackedOrigin]](store[].synchronization.unsafe_ptr())
    var address = 0
    var held = List[Int]()
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
            # Taken out under the lock, released after it (the device free
            # is the kernel's, never under this store's lock).
            held = slot[].handles.copy()
            entry = slot[].handle_entry
            slot[].handles = List[Int](length=_HANDLES, fill=-1)
            slot[].handle_entry = entry_ref_unresolved()
            slot[].live = False
            slot[].clear()
            slot[].words = List[Float64]()
            store[].free.append(index)
    finally:
        shared_slot_unlock(lock)
    if len(held) > 0:
        _free_handles(entry, held)
    return address


def _release(slot: _Slot) raises:
    """Release this state's resident device state before its words change."""
    var held = slot[].handles.copy()
    var entry = slot[].handle_entry
    slot[].handles = List[Int](length=_HANDLES, fill=-1)
    slot[].handle_entry = entry_ref_unresolved()
    _free_handles(entry, held)


def _name(slot: _Slot) -> String:
    return String("IncrementalPCA")


def _dense(view: MemberMatrix, what: String, estimator: String) raises:
    if not view.dense_host(MEMBER_F32):
        raise Error("TypeError: a " + estimator + " member reads " + what + " as a dense host float32 view")


def _whole(call: _Call, estimator: String) raises:
    if call[].row_index.rows > 0 or call[].column_index.rows > 0:
        raise Error("TypeError: a " + estimator + " member reads whole host views (no row or column index)")


def _address(slot: _Slot, block: Int) -> Int:
    return Int(slot[].blocks[block].unsafe_ptr())


# ---------------------------------------------------------------- IncrementalPCA


def _fit_ipca(slot: _Slot, call: _Call) raises:
    var estimator = _name(slot)
    _whole(call, estimator)
    _dense(call[].x, "X", estimator)
    if call[].weights.rows > 0:
        raise Error("TypeError: a " + estimator + " member takes no sample_weight")
    if call[].fit_output != 0 and call[].fit_output != MEMBER_OP_TRANSFORM:
        member_unsupported(estimator, call[].fit_output)
    var rows = call[].x.rows
    var cols = call[].x.cols
    var batch = 5 * cols if slot[].words[3] < 0.0 else Int(slot[].words[3])
    var requested = -1 if slot[].words[1] < 0.0 else Int(slot[].words[1])
    var route = call[].route
    var keep = (route.flags & XDECOMP_ROUTE_RESIDENT) != 0
    _release(slot)
    slot[].clear()
    # The door's six rooms (core/ipca_native_state.mojo IPCAOutputRoom).
    slot[].blocks.append(List[Float32](length=max(cols * cols, 1), fill=Float32(0)))
    for _ in range(5):  # driver-loop: five vector outputs
        slot[].blocks.append(List[Float32](length=max(cols, 1), fill=Float32(0)))
    var outputs = IPCAOutputs(_address(slot, 0), _address(slot, 1), _address(slot, 2), _address(slot, 3),
        _address(slot, 4), _address(slot, 5))
    var driver = IPCANativeDriver(route.entry.call)
    var result = driver.fit(IPCAFitRequest(call[].x.data, rows, cols, requested, batch, outputs, keep))
    if result.kept:
        for k in range(_HANDLES):  # driver-loop: five resident state handles
            slot[].handles[k] = result.state.values[k]
        slot[].handle_entry = route.entry
    slot[].count = result.count
    slot[].features = cols
    slot[].scalars = [Float64(result.count), Float64(cols), Float64(rows), result.noise, Float64(batch)]
    slot[].fitted = True
    if call[].fit_output == MEMBER_OP_TRANSFORM:
        _query(slot, call, False)


def _prepared(slot: _Slot, call: _Call, k: Int, block: Int, count: Int) raises -> Int:
    """The resident handle of one fitted block: the fit's kept copy, else
    prepared once (the door's `ipca_owner_operand`), else -1 (per call)."""
    if slot[].handles[k] >= 0 or (call[].route.flags & XDECOMP_ROUTE_RESIDENT) == 0:
        return slot[].handles[k]
    # One state is driven through one route (its plan's): the entry that
    # prepares is the entry that frees.
    var handle = IPCANativeDriver(call[].route.entry.call).prepare(_address(slot, block), count)
    slot[].handles[k] = handle
    slot[].handle_entry = call[].route.entry
    return handle


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
    var count = slot[].count
    var features = slot[].features
    if not inverse and call[].x.cols != features:
        raise Error("X has " + String(call[].x.cols) + " features, but IncrementalPCA is expecting "
                    + String(features))
    if inverse and call[].x.cols != count:
        raise Error("x_decomp: gemm inner dimensions " + String(call[].x.cols) + " and " + String(count) + " differ")
    var width = features if inverse else count
    if call[].dst.rows != rows or call[].dst.cols != width:
        raise Error("ValueError: the output buffer holds " + String(width) + " values per row of X")
    var whiten = slot[].words[2] != 0.0
    var resident = (call[].route.flags & XDECOMP_ROUTE_RESIDENT) != 0
    var ids = IPCAStateIds()
    if resident:
        # Only these three blocks are read by a query (the door's rule).
        ids.values[0] = _prepared(slot, call, 0, 0, count * features)
        ids.values[2] = _prepared(slot, call, 2, 2, features)
        if whiten:
            ids.values[4] = _prepared(slot, call, 4, 4, count)
    var request = IPCAQueryRequest(call[].x.data, rows, features, count, _address(slot, 2), _address(slot, 0),
        _address(slot, 4), call[].dst.data, inverse, whiten, resident, ids^)
    IPCANativeDriver(call[].route.entry.call).query(request)


def _block_index(index: Int) raises -> Int:
    if index == 0:
        return MEMBER_BLOCK_SCALARS
    if index >= 1 and index <= 6:
        return XDECOMP_BLOCK_COMPONENTS + index - 1
    raise Error("member ABI: an IncrementalPCA member has no fitted block index " + String(index))


def _export(slot: _Slot, call: _Call) raises:
    """A view of one fitted block. The slot owns the words; they stay valid
    until this state's next fit, import or free."""
    if not slot[].fitted:
        raise Error("NotFittedError: this " + _name(slot) + " member is not fitted yet")
    var block = call[].block
    if (call[].flags & MEMBER_BLOCK_BY_INDEX) != 0:
        block = _block_index(block)
    var count = slot[].count
    var features = slot[].features
    if block == MEMBER_BLOCK_SCALARS:
        call[].dst = MemberMatrix(Int(slot[].scalars.unsafe_ptr()), 1, len(slot[].scalars), MEMBER_F64)
    elif block == XDECOMP_BLOCK_COMPONENTS:
        call[].dst = MemberMatrix(_address(slot, 0), count, features)
    elif block >= XDECOMP_BLOCK_SINGULAR and block <= XDECOMP_BLOCK_RATIO:
        var k = block - XDECOMP_BLOCK_COMPONENTS
        var width = features if (block == XDECOMP_BLOCK_MEAN or block == XDECOMP_BLOCK_VAR) else count
        call[].dst = MemberMatrix(_address(slot, k), 1, width)
    else:
        raise Error("member ABI: an IncrementalPCA member has no fitted block " + String(block))
    call[].block = block
    call[].out_cols = 7


def _import(slot: _Slot, call: _Call) raises:
    """Copy one fitted block in; MEMBER_IMPORT_DONE completes the state."""
    var estimator = _name(slot)
    var block = call[].block
    _release(slot)
    if slot[].fitted or len(slot[].blocks) != 6:
        slot[].clear()
        for _ in range(6):  # driver-loop: six fitted blocks
            slot[].blocks.append(List[Float32]())
    if block == MEMBER_BLOCK_SCALARS:
        if not call[].x.dense_host(MEMBER_F64) or call[].x.rows * call[].x.cols != _IPCA_SCALARS:
            raise Error("member ABI: the " + estimator + " scalars block is five host float64 words")
        var source = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call[].x.data)
        slot[].scalars = List[Float64](length=_IPCA_SCALARS, fill=0.0)
        for index in range(_IPCA_SCALARS):  # driver-loop: five fitted scalars
            slot[].scalars[index] = source[index]
    elif block >= XDECOMP_BLOCK_COMPONENTS and block <= XDECOMP_BLOCK_RATIO:
        var size = call[].x.rows * call[].x.cols
        var words = List[Float32](length=size, fill=0.0)
        if size > 0:
            _dense(call[].x, "a fitted block", estimator)
            unsafe_memcpy(dest=words.unsafe_ptr(), src=_F32P(unsafe_from_address=call[].x.data), count=size)
        slot[].blocks[block - XDECOMP_BLOCK_COMPONENTS] = words^
    elif block != 0 or (call[].flags & MEMBER_IMPORT_DONE) == 0:
        raise Error("member ABI: an " + estimator + " member has no fitted block " + String(block))
    if (call[].flags & MEMBER_IMPORT_DONE) == 0:
        return
    if len(slot[].scalars) != _IPCA_SCALARS:
        raise Error("member ABI: a fitted " + estimator + " state needs its scalars block")
    var count = Int(slot[].scalars[0])
    var features = Int(slot[].scalars[1])
    if (count < 1 or features < 1 or len(slot[].blocks[0]) != count * features or len(slot[].blocks[1]) != count
            or len(slot[].blocks[2]) != features or len(slot[].blocks[3]) != features
            or len(slot[].blocks[4]) != count or len(slot[].blocks[5]) != count):
        raise Error("member ABI: a fitted " + estimator + " state needs every fitted block, of the fitted sizes")
    slot[].count = count
    slot[].features = features
    slot[].fitted = True


def _outputs(call: _Call) raises:
    """`call.out_cols` of transform / inverse_transform. Before a fit the
    transform width is known only for an integer n_components."""
    call[].classes = 0
    if call[].state == 0:
        raise Error("member ABI: an XDecomp member's output width needs its state")
    var slot = _Slot(unsafe_from_address=_lookup(call[].state, False))
    var asked = call[].flags
    if asked != MEMBER_OP_TRANSFORM and asked != MEMBER_OP_INVERSE_TRANSFORM:
        member_unsupported(_name(slot), asked)
    if slot[].fitted:
        call[].out_cols = slot[].count if asked == MEMBER_OP_TRANSFORM else slot[].features
        return
    if asked == MEMBER_OP_TRANSFORM and slot[].words[1] >= 0.0:
        call[].out_cols = Int(slot[].words[1])
        return
    raise Error("NotFittedError: this " + _name(slot) + " member's output width is known after its fit")


def _xdecomp_member(operation: Int, address: Int) raises -> Int:
    var call = _Call(unsafe_from_address=address)
    if operation == MEMBER_OP_CAPABILITIES:
        return MEMBER_CAN_FIT | MEMBER_CAN_TRANSFORM | MEMBER_CAN_INVERSE_TRANSFORM | MEMBER_CAN_EXPORT | MEMBER_CAN_IMPORT
    if operation == MEMBER_OP_NEW:
        if call[].param_count < 1 or not call[].params:
            raise Error("member ABI: an XDecomp member takes its parameter words")
        var source = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call[].params)
        if source[0] != Float64(XDECOMP_KIND_IPCA):
            raise Error("member ABI: unknown or unpublished XDecomp member kind")
        if call[].param_count != XDECOMP_IPCA_PARAM_COUNT:
            raise Error("member ABI: an IncrementalPCA member takes four parameters")
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
        _fit_ipca(slot, call)
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


def xdecomp_member_v2(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
    result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """The member entry of the XDecomp family (schema `MEMBER_ABI`)."""
    try:
        var word = _xdecomp_member(operation, member_call_from_spans(operation, aa, na, np, nf))
        if result:
            MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = word
        return 0
    except error:
        native_descriptor_report(error, context, sink)
        return 1


def xdecomp_member_publish(mode: Int, vendor: Int, publisher: Int) raises:
    """Publish this image's XDecomp member entry. One line in the image's
    module initialization, after its kernel tables are published."""
    member_publish[MEMBER_FAMILY_XDECOMP](mode, vendor, publisher, xdecomp_member_v2)
