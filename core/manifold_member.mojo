# SPDX-License-Identifier: Apache-2.0
"""UMAP as a MEMBER of family 33 Manifold (core/member_abi.mojo).

One entry, `manifold_member_v2`, published by the two METRICS images (the
images that hold UMAP's typed RUN: `_mojolearn_metrics` beside its UMAP
operations table, `_mojolearn_metrics_host` vendor 0). It runs the same
typed calls the estimator's own door runs (bindings/mani_umap_native_door.mojo
`_umap_fit` / `_umap_transform`): the checks of x_ann/mani_umap_run.mojo in
the door's order, then `umap_fit_call` / `umap_fit_ex_call` /
`umap_transform_call` (core/neighbors_entry_table_umap.mojo) on the route's
`entry`, the UMAP operations entry of the image the estimator's binding
route resolved (its (mode, vendor) key). No other route slot is read. The
same entries on the same words: the bits are the door's.

Kinds: 0 UMAP (the only one; TSNE is family 42 and not a member).

MEMBER_OP_NEW parameters (20 Float64 words):
  [0] kind (0)
  [1] n_neighbors  [2] n_components  [3] n_epochs (0: None)  [4] min_dist
  [5] spread  [6] set_op_mix_ratio  [7] local_connectivity  [8] seed
  [9] learning_rate  [10] repulsion_strength  [11] negative_sample_rate
      (words 1 .. 11: the door's validated `config`, in its order)
  [12] metric code (-1 euclidean / l2, 0 sqeuclidean, 2 cosine,
      3 manhattan, 7 chebyshev, 9 minkowski)  [13] metric argument (p)
  [14] a  [15] b  (words 12 .. 15: the door's `extras`)
  [16] target metric (0 categorical, 1 l2 / euclidean, -1 a target setting
      the door refuses or a target metric it does not implement)
  [17] target_n_neighbors as kept (`umap_target_neighbors`)
  [18] target_weight
  [19] the numeric-mode word of the settings (0 fast, 1 identical; the
      fitted mode the transform compares, written by the PACK)
The admission (providers/python/member_manifold_boundary_door.mojo) answers
False for init 'random' / 'pca' / an array (their starts need the x_decomp
kit images and, for an array, an n-shaped input per fit) and for integer
settings that a Float64 word does not hold exactly.

Views (all host, dense row-major; a device-home view, a row or column index
and a sample weight are refused in words): X float32 rows x features. y
(optional): an l2 target, float32 rows x dims; a categorical target is
refused at FIT in "member ABI" words (its codes and the unknown label -1
are not member words) so a caller falls back to the estimator's door.
`call.fit_output = MEMBER_OP_TRANSFORM` (fit_transform) writes the training
embedding (rows x n_components float32) to `dst` in the fit call.
TRANSFORM writes float32 rows x n_components. MEMBER_CAN_TEST_ROWS: every
operation that reads X or y refuses non-finite words itself, in the door's
words (x_ann/mani_umap_run.mojo:34 `umap_check_input` at fit, :101
`umap_continuous_target_dims` for y, :128 `umap_check_transform_input` at
transform), so a caller makes no finite scan of its own.

Fitted-state blocks (MEMBER_OP_EXPORT / MEMBER_OP_IMPORT), in index order:
`MEMBER_BLOCK_FAMILY` (64) float32 rows x features, the training rows;
`MEMBER_BLOCK_FAMILY + 1` (65) float32 rows x n_components, the fitted
embedding. An import needs both.

LIFETIME: a fit BORROWS the caller's X as its training rows (the transform
reads them for its neighbor graph and for the door's same-input reuse);
the caller keeps the words of X alive and unchanged until this state's next
fit, import or free. No copy is made per fit (the door copied X at every
fit; a copy per fit is a copy per fold). The first MEMBER_OP_EXPORT of the
rows copies them once into the slot, and the state reads that copy from
then on. The embedding is owned by the slot (the entry writes it there).

The fitted state is OWNED here, in a process-wide store of fixed chunks
(the core/linear_member.mojo form): a slot never moves once minted; the
store's lock covers only mint, free and the handle lookup. SERIAL BY
NATURE here: the store's bookkeeping and the two block copies (memcpy); the
layout and the transform are the entry's own parallel work.
"""
from std.sys.info import size_of
from std.memory import unsafe_memcpy
from core.member_abi import (
    MemberCall, MemberMatrix, MEMBER_F32,
    MEMBER_OP_CAPABILITIES, MEMBER_OP_NEW, MEMBER_OP_FREE, MEMBER_OP_FIT, MEMBER_OP_TRANSFORM,
    MEMBER_OP_OUTPUTS, MEMBER_OP_EXPORT, MEMBER_OP_IMPORT,
    MEMBER_CAN_FIT, MEMBER_CAN_TRANSFORM, MEMBER_CAN_EXPORT, MEMBER_CAN_IMPORT, MEMBER_CAN_TEST_ROWS,
    MEMBER_BLOCK_FAMILY, MEMBER_BLOCK_BY_INDEX, MEMBER_IMPORT_DONE,
    member_publish, member_call_from_spans, member_unsupported,
)
from core.member_abi import MEMBER_BORROWS_FIT, MEMBER_EXPORT_OWN
from core.member_families import MEMBER_FAMILY_MANIFOLD
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.shared_slot import SharedSlot, shared_slot_lock, shared_slot_unlock
from core.neighbors_entry_table_umap import (
    UmapConfigWords, UmapTargetWords, umap_fit_call, umap_fit_ex_call, umap_transform_call,
    UMAP_TAIL_LEGACY, UMAP_TAIL_RATES, UMAP_TAIL_EXTRAS,
)
from x_ann.mani_umap_run import (
    umap_check_input, umap_check_neighbors, umap_check_spectral_samples, umap_continuous_target_dims,
    umap_legacy_abi, umap_check_embedding, umap_check_transform_state, umap_check_transform_input,
    umap_transform_reuses_fit, umap_check_transform_output,
)

comptime MANIFOLD_KIND_UMAP = 0
comptime MANIFOLD_PARAM_COUNT = 20
comptime MF_BLOCK_ROWS = MEMBER_BLOCK_FAMILY
comptime MF_BLOCK_EMBEDDING = MEMBER_BLOCK_FAMILY + 1

comptime _CHUNK = 256
comptime _INDEX_SPAN = 16777216
comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _F64P = MutPointer[Float64, MutUntrackedOrigin]
comptime _CAPABILITIES = (MEMBER_CAN_FIT | MEMBER_CAN_TRANSFORM | MEMBER_CAN_EXPORT | MEMBER_CAN_IMPORT
    | MEMBER_CAN_TEST_ROWS)


struct ManifoldMemberSlot(Copyable, Movable, Defaultable):
    var generation: Int
    var live: Bool
    var fitted: Bool
    var params: List[Float64]
    var rows_addr: Int          # the training rows: borrowed X, or `rows` once owned
    var rows: List[Float32]
    var embedding: List[Float32]
    var n: Int
    var features: Int
    var embedding_rows: Int     # rows of the embedding block an import gave (checked at DONE)

    def __init__(out self):
        self.generation = 0
        self.live = False
        self.fitted = False
        self.params = List[Float64]()
        self.rows_addr = 0
        self.rows = List[Float32]()
        self.embedding = List[Float32]()
        self.n = 0
        self.features = 0
        self.embedding_rows = 0


struct ManifoldMemberStore(Defaultable, Movable):
    var synchronization: List[Int64]
    var chunks: List[List[ManifoldMemberSlot]]
    var free: List[Int]
    var count: Int

    def __init__(out self):
        # Side-effect-free even when the process slot's initializers race.
        self.synchronization = List[Int64](length=2, fill=0)
        self.chunks = List[List[ManifoldMemberSlot]]()
        self.free = List[Int]()
        self.count = 0


comptime _STORE = SharedSlot[ManifoldMemberStore, "MojolearnManifoldMemberStoreV2", 2]
comptime _SlotPtr = MutPointer[ManifoldMemberSlot, MutUntrackedOrigin]
comptime _CallPtr = MutPointer[MemberCall, MutUntrackedOrigin]


def _slot_address(index: Int) raises -> Int:
    # Under the store's lock; a chunk's buffer never grows.
    var store = _STORE.get_or_create_ptr()
    return Int(store[].chunks[index // _CHUNK].unsafe_ptr()) + (index % _CHUNK) * size_of[ManifoldMemberSlot]()


def _mint(words: _F64P, count: Int) raises -> Int:
    var store = _STORE.get_or_create_ptr()
    var lock = rebind[MutPointer[Int64, MutUntrackedOrigin]](store[].synchronization.unsafe_ptr())
    shared_slot_lock(lock)
    try:
        var index = -1
        if len(store[].free) > 0:
            index = store[].free.pop()
        else:
            if store[].count >= _INDEX_SPAN - 1:
                raise Error("RuntimeError: mojolearn: too many live UMAP member states")
            if store[].count == len(store[].chunks) * _CHUNK:
                store[].chunks.append(List[ManifoldMemberSlot](length=_CHUNK, fill=ManifoldMemberSlot()))
            index = store[].count
            store[].count += 1
        var slot = _SlotPtr(unsafe_from_address=_slot_address(index))
        slot[].generation += 1
        slot[].live = True
        slot[].fitted = False
        var params = List[Float64](length=count, fill=0.0)
        for i in range(count):  # small-loop(params: the twenty NEW words)
            params[i] = words[i]
        slot[].params = params^
        slot[].rows_addr = 0
        slot[].n = 0
        slot[].features = 0
        slot[].embedding_rows = 0
        return slot[].generation * _INDEX_SPAN + index + 1
    finally:
        shared_slot_unlock(lock)


def _lookup(handle: Int, release: Bool) raises -> Int:
    """The slot address of a live handle; a released, foreign or made-up
    handle is refused (never a stale read)."""
    var store = _STORE.get_or_create_ptr()
    var lock = rebind[MutPointer[Int64, MutUntrackedOrigin]](store[].synchronization.unsafe_ptr())
    shared_slot_lock(lock)
    try:
        var index = handle % _INDEX_SPAN - 1
        if handle <= 0 or index < 0 or index >= store[].count:
            raise Error("NotFittedError: this member state was released or never created")
        var address = _slot_address(index)
        var slot = _SlotPtr(unsafe_from_address=address)
        if not slot[].live or slot[].generation != handle // _INDEX_SPAN:
            raise Error("NotFittedError: this member state was released or never created")
        if release:
            slot[].live = False
            slot[].fitted = False
            slot[].rows_addr = 0
            slot[].rows = List[Float32]()
            slot[].embedding = List[Float32]()
            slot[].params = List[Float64]()
            store[].free.append(index)
        return address
    finally:
        shared_slot_unlock(lock)


def _whole(call: _CallPtr) raises:
    if call[].row_index.rows > 0 or call[].column_index.rows > 0:
        raise Error("TypeError: a UMAP member reads whole host views (no row or column index)")


def _dense(view: MemberMatrix, what: String) raises:
    if not view.dense_host(MEMBER_F32):
        raise Error("TypeError: a UMAP member reads " + what + " as a dense host float32 view")


def _config(slot: _SlotPtr) -> UmapConfigWords:
    """The door's `_umap_config_words`, from the NEW words 1 .. 15."""
    ref w = slot[].params
    return UmapConfigWords(Int(w[1]), Int(w[2]), Int(w[3]), w[4], w[5], w[6], w[7], Int(w[8]), w[9], w[10],
        Int(w[11]), Int(w[12]), w[13], w[14], w[15])


def _fit(slot: _SlotPtr, call: _CallPtr) raises:
    """`_umap_fit` with a spectral start: the door's checks in its order,
    then the legacy or extended fit entry into the slot's embedding."""
    _whole(call)
    if call[].fit_output != 0 and call[].fit_output != MEMBER_OP_TRANSFORM:
        member_unsupported("UMAP", call[].fit_output)
    if call[].weights.rows > 0:
        raise Error("TypeError: a UMAP member fits no sample_weight")
    _dense(call[].x, "X")
    var n = call[].x.rows
    var d = call[].x.cols
    var cfg = _config(slot)
    var nc = cfg.n_components
    if call[].fit_output == MEMBER_OP_TRANSFORM:
        _dense(call[].dst, "the embedding buffer")
        if call[].dst.rows != n or call[].dst.cols != nc:
            raise Error("ValueError: the embedding buffer holds " + String(nc) + " components per row of X")
    umap_check_input(call[].x.data, n * d)
    umap_check_neighbors(cfg.n_neighbors, n)
    umap_check_spectral_samples(False, n, nc)
    var has_target = call[].y.rows > 0
    var target = UmapTargetWords.none()
    if has_target:
        if slot[].params[16] != 1.0:
            raise Error("member ABI: a UMAP member reads an l2 / euclidean target only; a categorical target "
                + "(its label codes and the unknown label -1) or a refused target setting stays with the "
                + "estimator's own fit")
        _dense(call[].y, "y")
        var dims = umap_continuous_target_dims(2, call[].y.rows, call[].y.cols, n, call[].y.data,
            call[].y.rows * call[].y.cols)
        target = UmapTargetWords(2, dims, Int(slot[].params[17]), slot[].params[18], call[].y.data)
    slot[].fitted = False
    # Zero-filled, as the door's `family_array` output.
    var embedding = List[Float32](length=n * nc, fill=Float32(0.0))
    var embedding_address = Int(embedding.unsafe_ptr())
    var entry = call[].route.entry.call
    var columns = 0
    if umap_legacy_abi(False, has_target, cfg.default_extras(), nc):
        columns = umap_fit_call(entry, call[].x.data, embedding_address, n, d, cfg, not cfg.legacy_rates())
    else:
        columns = umap_fit_ex_call(entry, call[].x.data, embedding_address, 0, n, d, cfg, target)
    umap_check_embedding(columns == nc, embedding_address, n * nc)
    # Borrow, never copy (LIFETIME in the module docstring). A copy a
    # previous import or export left in the slot is dropped.
    slot[].rows = List[Float32]()
    slot[].rows_addr = call[].x.data
    slot[].embedding = embedding^
    slot[].n = n
    slot[].features = d
    slot[].fitted = True
    if call[].fit_output == MEMBER_OP_TRANSFORM and n * nc > 0:
        unsafe_memcpy(dest=_F32P(unsafe_from_address=call[].dst.data), src=slot[].embedding.unsafe_ptr(),
            count=n * nc)


def _transform(slot: _SlotPtr, call: _CallPtr) raises:
    """`_umap_transform` after the door's refit refusal (the PACK / state
    path compares settings): the door's checks, the same-input reuse of the
    fitted embedding, else the transform entry with the door's tail."""
    if not slot[].fitted:
        raise Error("UMAP transform requires a successful fit")
    _whole(call)
    _dense(call[].x, "X")
    var cfg = _config(slot)
    var nc = cfg.n_components
    var nq = call[].x.rows
    var d = call[].x.cols
    _dense(call[].dst, "the transform buffer")
    if call[].dst.rows != nq or call[].dst.cols != nc:
        raise Error("ValueError: the transform buffer holds " + String(nc) + " components per row of X")
    umap_check_transform_state(slot[].n, nc, slot[].n, nc, d, slot[].features)
    umap_check_transform_input(call[].x.data, nq * d)
    var count = nq * nc
    if umap_transform_reuses_fit(call[].x.data, slot[].rows_addr, nq, slot[].n, d):
        if count > 0:
            unsafe_memcpy(dest=_F32P(unsafe_from_address=call[].dst.data), src=slot[].embedding.unsafe_ptr(),
                count=count)
        return
    var tail = UMAP_TAIL_LEGACY if cfg.legacy_rates() else UMAP_TAIL_RATES
    if not cfg.default_extras():
        tail = UMAP_TAIL_EXTRAS
    # The door hands the entry a zero-filled output (`family_array`): the
    # same one zero-filled allocation here, copied over the caller's buffer.
    if count > 0:
        var zeros = List[Float32](length=count, fill=Float32(0.0))
        unsafe_memcpy(dest=_F32P(unsafe_from_address=call[].dst.data), src=zeros.unsafe_ptr(), count=count)
        _ = zeros^
    var columns = umap_transform_call(call[].route.entry.call, slot[].rows_addr, Int(slot[].embedding.unsafe_ptr()),
        call[].x.data, call[].dst.data, slot[].n, nq, d, cfg, tail)
    umap_check_transform_output(columns == nc, call[].dst.data, count)


def _own_rows(slot: _SlotPtr):
    """Copy the borrowed training rows into the slot once (first export)."""
    var count = slot[].n * slot[].features
    if slot[].rows_addr == Int(slot[].rows.unsafe_ptr()) and len(slot[].rows) == count:
        return
    var owned = List[Float32](length=count, fill=Float32(0.0))
    if count > 0:
        unsafe_memcpy(dest=owned.unsafe_ptr(), src=_F32P(unsafe_from_address=slot[].rows_addr), count=count)
    slot[].rows = owned^
    slot[].rows_addr = Int(slot[].rows.unsafe_ptr())


def _export(slot: _SlotPtr, call: _CallPtr) raises:
    """A view of one fitted block (the borrowed rows are copied here once);
    the slot owns the words until this state's next fit, import or free."""
    if not slot[].fitted:
        raise Error("NotFittedError: this UMAP member is not fitted yet")
    if (call[].flags & MEMBER_EXPORT_OWN) != 0:
        # THE LIFETIME RULE: the training rows, once; no view is answered.
        _own_rows(slot)
        return
    var block = call[].block
    if (call[].flags & MEMBER_BLOCK_BY_INDEX) != 0:
        if block < 0 or block > 1:
            raise Error("member ABI: a UMAP member has two fitted blocks")
        block = MF_BLOCK_ROWS + block
    if block == MF_BLOCK_ROWS:
        _own_rows(slot)
        call[].dst = MemberMatrix(Int(slot[].rows.unsafe_ptr()), slot[].n, slot[].features)
    elif block == MF_BLOCK_EMBEDDING:
        call[].dst = MemberMatrix(Int(slot[].embedding.unsafe_ptr()), slot[].n, Int(slot[].params[2]))
    else:
        raise Error("member ABI: a UMAP member has no fitted block " + String(block))
    call[].block = block
    call[].out_cols = 2


def _import(slot: _SlotPtr, call: _CallPtr) raises:
    """Copy one fitted block in; MEMBER_IMPORT_DONE completes the state."""
    if call[].block == MF_BLOCK_ROWS or call[].block == MF_BLOCK_EMBEDDING:
        _dense(call[].x, "a fitted block")
        var count = call[].x.rows * call[].x.cols
        var words = List[Float32](length=count, fill=Float32(0.0))
        if count > 0:
            unsafe_memcpy(dest=words.unsafe_ptr(), src=_F32P(unsafe_from_address=call[].x.data), count=count)
        slot[].fitted = False
        if call[].block == MF_BLOCK_ROWS:
            slot[].rows = words^
            slot[].rows_addr = Int(slot[].rows.unsafe_ptr())
            slot[].n = call[].x.rows
            slot[].features = call[].x.cols
        else:
            # Kept with its row count until DONE checks it against the rows.
            slot[].embedding = words^
            slot[].embedding_rows = call[].x.rows
    elif call[].block != 0 or (call[].flags & MEMBER_IMPORT_DONE) == 0:
        raise Error("member ABI: a UMAP member has no fitted block " + String(call[].block))
    if (call[].flags & MEMBER_IMPORT_DONE) == 0:
        return
    var nc = Int(slot[].params[2])
    var embedding_rows = slot[].embedding_rows
    if slot[].n == 0 or embedding_rows == 0:
        raise Error("member ABI: a fitted UMAP state needs its training rows and its embedding")
    var embedding_cols = len(slot[].embedding) // embedding_rows if embedding_rows > 0 else 0
    if embedding_rows * embedding_cols != len(slot[].embedding):
        raise Error("UMAP fitted embedding shape differs from fitted configuration")
    umap_check_transform_state(embedding_rows, embedding_cols, slot[].n, nc, slot[].features, slot[].features)
    slot[].fitted = True


def _manifold_member(operation: Int, address: Int) raises -> Int:
    var call = _CallPtr(unsafe_from_address=address)
    if operation == MEMBER_OP_CAPABILITIES:
        if call[].state != 0:
            _ = _lookup(call[].state, False)
        # MEMBER_BORROWS_FIT: a fitted UMAP state reads the caller's X (its
        # training rows) until an export of the rows block, or
        # MEMBER_EXPORT_OWN, copied them (LIFETIME in the module docstring).
        return _CAPABILITIES | MEMBER_BORROWS_FIT
    if operation == MEMBER_OP_NEW:
        if call[].param_count != MANIFOLD_PARAM_COUNT or not call[].params:
            raise Error("member ABI: a UMAP member takes twenty parameters (kind, the eleven config words, "
                + "the four extras words, target metric, target_n_neighbors, target_weight, mode)")
        var words = _F64P(unsafe_from_address=call[].params)
        if words[0] != Float64(MANIFOLD_KIND_UMAP):
            raise Error("member ABI: unknown Manifold member kind")
        call[].state = _mint(words, MANIFOLD_PARAM_COUNT)
        return 0
    if operation == MEMBER_OP_FREE:
        _ = _lookup(call[].state, True)
        call[].state = 0
        return 0
    if operation == MEMBER_OP_OUTPUTS:
        # TRANSFORM (and fit_transform's dst): n_components float32 columns,
        # with or without a state (the NEW words or the state's words).
        var components = 0
        if call[].state != 0:
            components = Int(_SlotPtr(unsafe_from_address=_lookup(call[].state, False))[].params[2])
        else:
            if call[].param_count != MANIFOLD_PARAM_COUNT or not call[].params:
                raise Error("member ABI: a UMAP member takes twenty parameters")
            components = Int(_F64P(unsafe_from_address=call[].params)[2])
        if call[].flags != 0 and call[].flags != MEMBER_OP_TRANSFORM:
            member_unsupported("UMAP", call[].flags)
        call[].classes = 0
        call[].out_cols = components
        call[].out_dtype = MEMBER_F32
        return 0
    var slot = _SlotPtr(unsafe_from_address=_lookup(call[].state, False))
    if operation == MEMBER_OP_FIT:
        _fit(slot, call)
    elif operation == MEMBER_OP_TRANSFORM:
        _transform(slot, call)
    elif operation == MEMBER_OP_EXPORT:
        _export(slot, call)
    elif operation == MEMBER_OP_IMPORT:
        _import(slot, call)
    else:
        member_unsupported("UMAP", operation)
    return 0


def manifold_member_v2(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
    result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """The member entry of UMAP (schema `MEMBER_ABI`)."""
    try:
        var word = _manifold_member(operation, member_call_from_spans(operation, aa, na, np, nf))
        if result:
            MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = word
        return 0
    except error:
        native_descriptor_report(error, context, sink)
        return 1


def manifold_member_publish(mode: Int, vendor: Int, publisher: Int) raises:
    """Publish this image's UMAP member entry. One line in each metrics
    image's module initialization, after its UMAP operations table."""
    if vendor < 0:
        return
    member_publish[MEMBER_FAMILY_MANIFOLD](mode, vendor, publisher, manifold_member_v2)
