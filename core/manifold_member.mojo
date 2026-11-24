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

MEMBER_OP_NEW parameters (33 Float64 words):
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
  [20] init (lane mf-nbr6): 0 'spectral', 1 'random', 2 'pca', 3 an array
  [21] the array: the address of the admitted float32 matrix (pinned by the
      plan), [22] its rows, [23] its columns (checked against (n,
      n_components) at fit in the door's words, then its finite test)
  [24 .. 26] 'random' / 'pca': the 'identical' x_decomp image's capability,
      ordered-LLE and key words (its entry is `route.second`); [27 .. 29]
      the default mode's x_decomp image (its entry is `route.third`), the
      two images the door's `_umap_initial` runs the start on
  [30] n_neighbors, [31] n_epochs, [32] the seed as their exact int64 bits
      (`bitcast`; words 1, 3 and 8 hold the same values rounded to Float64,
      so an integer at or above 2**53 is carried exactly)

Views (all host, dense row-major; a device-home view, a column index and a
sample weight are refused in words): X float32 rows x features, whole or
under a host int32 / int64 ROW INDEX (MEMBER_CAN_ROW_INDEX, lane
vdf-scope-cluster7: a fold's rows; y is read through the same index).
THE RESIDENT-STORE ROUTE (MEMBER_CAN_SCOPE on a GPU metrics image, which
publishes its store, core/metrics_store_ref.mojo): inside a `resident()`
scope with X's pin, a fold's rows are taken ON THE DEVICE out of the scope's
one copy of X in the metrics store (core/member_scope_rows.mojo) and the
entries read that slot (the slot form of core/neighbors_entry_table_umap.mojo:
the same bodies on the same words, so the same bits). A fold fit KEEPS its
slot as THE FITTED-ROWS SLOT, which a fold transform reads device to device;
it is freed at the next fit, an import of the rows, the free, or the first
export of the rows block (read home once, UMAP_OP_ROWS_READ). Otherwise (no
scope, no pin, a host image, init='pca') the rows are gathered on the host
cores into rows the state owns. y
(optional): an l2 target, float32 rows x dims; a categorical target is
refused at FIT in "member ABI" words (its codes and the unknown label -1
are not member words) so a caller falls back to the estimator's door.
`call.fit_output = MEMBER_OP_TRANSFORM` (fit_transform) writes the training
embedding (rows x n_components float32) to `dst` in the fit call.
TRANSFORM writes float32 rows x n_components. MEMBER_CAN_TEST_ROWS: every
operation that reads X or y refuses non-finite words itself, in the door's
words (x_ann/mani_umap_run.mojo:34 `umap_check_input` at fit, :101
`umap_continuous_target_dims` for y, :128 `umap_check_transform_input` at
transform), so a caller makes no finite scan of its own. On a slot the device
entry makes the same test on the device in the same words (after the
neighbor-count check rather than before it: a fold that fails both refuses
with the neighbor words).

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
from core.member_abi import MEMBER_BORROWS_FIT, MEMBER_EXPORT_OWN, MEMBER_GIL_DOOR
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
    umap_init_shape_matches, umap_init_shape_refusal, umap_check_init_finite,
)
from std.memory import bitcast
from bindings.mani_kit_run import MRun, MImage
from bindings.mani_umap_init_run import umap_random_start_run, umap_pca_start_run
from core.member_abi import (
    MEMBER_CAN_ROW_INDEX, MEMBER_CAN_SCOPE, MEMBER_SCOPE_NONE, MEMBER_HOME_DEVICE, MEMBER_I32, MEMBER_I64,
)
from core.entry_table import EntryRef, entry_ref_unresolved, entry_ref_mode, entry_ref_vendor
from core.metrics_store_ref import metrics_store_find, METRICS_STORE_PREFIX
from core.member_scope_rows import MemberScopeRows, member_scope_rows_in, member_scope_rows_done_in
from core.admitted_rows_gather import admitted_gather_rows_f32, ARG_OUT_OF_RANGE
from core.fold_x_word import fold_x_word
from bindings.arena_cache_run import arena_cache_free
from core.neighbors_entry_table_umap import umap_rows_read_call

comptime MANIFOLD_KIND_UMAP = 0
comptime MANIFOLD_PARAM_COUNT = 33
comptime MANIFOLD_INIT_SPECTRAL = 0
comptime MANIFOLD_INIT_RANDOM = 1
comptime MANIFOLD_INIT_PCA = 2
comptime MANIFOLD_INIT_ARRAY = 3
comptime MF_INIT_WORD = 20
comptime MF_EXACT_WORD = 30
comptime MF_BLOCK_ROWS = MEMBER_BLOCK_FAMILY
comptime MF_BLOCK_EMBEDDING = MEMBER_BLOCK_FAMILY + 1

comptime _CHUNK = 256
comptime _INDEX_SPAN = 16777216
comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _F64P = MutPointer[Float64, MutUntrackedOrigin]
comptime _CAPABILITIES = (MEMBER_CAN_FIT | MEMBER_CAN_TRANSFORM | MEMBER_CAN_EXPORT | MEMBER_CAN_IMPORT
    | MEMBER_CAN_TEST_ROWS | MEMBER_CAN_ROW_INDEX)
comptime _EST = "UMAP"


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
    var rows_slot: Int          # THE FITTED-ROWS SLOT of a fold fit on the metrics store (-1: none)
    var rows_store: EntryRef    # the store that slot is in

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
        self.rows_slot = -1
        self.rows_store = entry_ref_unresolved()


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
        for i in range(count):  # small-loop(params: the thirty-three NEW words)
            params[i] = words[i]
        slot[].params = params^
        slot[].rows_addr = 0
        slot[].n = 0
        slot[].features = 0
        slot[].embedding_rows = 0
        slot[].rows_slot = -1
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
            slot[].rows_slot = -1
            slot[].rows = List[Float32]()
            slot[].embedding = List[Float32]()
            slot[].params = List[Float64]()
            store[].free.append(index)
        return address
    finally:
        shared_slot_unlock(lock)


def _no_columns(call: _CallPtr) raises:
    if call[].column_index.rows > 0 or (call[].row_index.rows > 0 and call[].x.home == MEMBER_HOME_DEVICE):
        raise Error("TypeError: a UMAP member reads host views under a host row index (no column index)")


def _store(slot: _SlotPtr, call: _CallPtr, fit: Bool) -> Optional[EntryRef]:
    """THE RESIDENT-STORE ROUTE (lane vdf-scope-cluster7): the metrics image's store published for the route's
    (mode, vendor) (core/metrics_store_ref.mojo; a host image publishes none), under the caller's scope with X's
    pin. A fit with init='pca' keeps the gathered form (its start reads host rows)."""
    if call[].scope == MEMBER_SCOPE_NONE or call[].pin == 0:
        return Optional[EntryRef]()
    if fit and Int(slot[].params[MF_INIT_WORD]) == MANIFOLD_INIT_PCA:
        return Optional[EntryRef]()
    return metrics_store_find(entry_ref_mode(call[].route.entry), entry_ref_vendor(call[].route.entry))


def _fold_y(call: _CallPtr, mut words: List[Float32]) raises -> MemberMatrix:
    """The target's rows under the call's row index, gathered on the host cores (y is a parameter-sized block
    the entry uploads itself; X never crosses here)."""
    var y = call[].y
    var index = call[].row_index
    _dense(y, "y")
    var n = index.rows
    words = List[Float32](unsafe_uninit_length=max(n * y.cols, 1))
    var status: Int
    if index.dtype == MEMBER_I64:
        status = admitted_gather_rows_f32[DType.int64](y.data, y.rows, y.cols, index.data, n,
            Int(words.unsafe_ptr()))
    elif index.dtype == MEMBER_I32:
        status = admitted_gather_rows_f32[DType.int32](y.data, y.rows, y.cols, index.data, n,
            Int(words.unsafe_ptr()))
    else:
        raise Error("TypeError: a UMAP member reads a host int32 or int64 row index")
    if status == ARG_OUT_OF_RANGE:
        raise Error("UMAP continuous y must be finite with one row per sample")
    return MemberMatrix(Int(words.unsafe_ptr()), n, y.cols)


def _drop_rows_slot(slot: _SlotPtr) raises:
    """Free THE FITTED-ROWS SLOT (a new fit's rows, an import, a free or the first export replace it)."""
    if slot[].rows_slot >= 0:
        var held = slot[].rows_slot
        slot[].rows_slot = -1
        arena_cache_free(slot[].rows_store, held)


def _dense(view: MemberMatrix, what: String) raises:
    if not view.dense_host(MEMBER_F32):
        raise Error("TypeError: a UMAP member reads " + what + " as a dense host float32 view")


def _exact(slot: _SlotPtr, index: Int) -> Int:
    """An integer setting's exact int64 bits (words 30 .. 32)."""
    return Int(bitcast[DType.int64](slot[].params[MF_EXACT_WORD + index]))


def _config(slot: _SlotPtr) -> UmapConfigWords:
    """The door's `_umap_config_words`, from the NEW words 1 .. 15 (n_neighbors,
    n_epochs and the seed from their exact words 30 .. 32)."""
    ref w = slot[].params
    return UmapConfigWords(_exact(slot, 0), Int(w[2]), _exact(slot, 1), w[4], w[5], w[6], w[7], _exact(slot, 2),
        w[9], w[10], Int(w[11]), Int(w[12]), w[13], w[14], w[15])


def _start(slot: _SlotPtr, call: _CallPtr, x_word: Int, d: Int, n: Int, nc: Int, mut words: List[Float32]) raises -> Int:
    """The door's `_umap_initial` (bindings/mani_umap_native_door.mojo): 0 for
    'spectral'; 'random' / 'pca' run the start on the two x_decomp images
    (`route.second` the 'identical' one, `route.third` the default mode's)
    into `words` (the door's `mk_out2` host copy) and answer its address; an
    array is the pinned matrix, after the door's shape refusal and finite
    test (lane mf-nbr6). `x_word`: the host rows of the fit (pca reads them;
    a resident fit never runs pca, `_store`)."""
    var code = Int(slot[].params[MF_INIT_WORD])
    if code == MANIFOLD_INIT_SPECTRAL:
        return 0
    if code == MANIFOLD_INIT_ARRAY:
        var address = Int(slot[].params[MF_INIT_WORD + 1])
        var rows = Int(slot[].params[MF_INIT_WORD + 2])
        var cols = Int(slot[].params[MF_INIT_WORD + 3])
        if not umap_init_shape_matches(rows, cols, n, nc):
            var refusal = umap_init_shape_refusal("(" + String(rows) + ", " + String(cols) + ")", n, nc)
            raise refusal^
        umap_check_init_finite(address, rows * cols)
        return address
    if not call[].route.has_second or not call[].route.has_third:
        raise Error("member ABI: a UMAP member's random or pca start needs the route's x_decomp images "
            + "(second, third)")
    ref w = slot[].params
    var seed_word = _exact(slot, 2) & 0xFFFFFFFF
    var run = MRun()
    try:
        var k = run.image(MImage(call[].route.second, Int(w[MF_INIT_WORD + 6]), Int(w[MF_INIT_WORD + 4]),
            w[MF_INIT_WORD + 5] != 0.0, -1))
        var start = -1
        if code == MANIFOLD_INIT_RANDOM:
            start = umap_random_start_run(run, k, n, nc, seed_word)
        else:
            var hk = run.image(MImage(call[].route.third, Int(w[MF_INIT_WORD + 9]), Int(w[MF_INIT_WORD + 7]),
                w[MF_INIT_WORD + 8] != 0.0, -1))
            start = umap_pca_start_run(run, k, hk, x_word, n, d, 2, n, nc, seed_word)
        var count = n * nc
        words = List[Float32](length=max(count, 1), fill=Float32(0.0))
        if count > 0:
            unsafe_memcpy(dest=words.unsafe_ptr(), src=_F32P(unsafe_from_address=run.addr(start)), count=count)
    finally:
        run.close()
    return Int(words.unsafe_ptr())


def _fit(slot: _SlotPtr, call: _CallPtr) raises:
    """`_umap_fit` over the whole X, or (MEMBER_CAN_ROW_INDEX, lane vdf-scope-cluster7) over a fold's rows: THE
    RESIDENT-STORE ROUTE takes them on the device out of the scope's one copy of X in the metrics store and the
    fit entry reads that slot (the slot form, core/neighbors_entry_table_umap.mojo), which the state then keeps
    as its fitted rows; otherwise they are gathered on the host cores into rows the state owns."""
    _no_columns(call)
    if call[].row_index.rows == 0:
        _dense(call[].x, "X")
        _fit_rows(slot, call, call[].x.data, call[].x.rows, call[].x.cols, -1, entry_ref_unresolved(),
            List[Float32](), call[].y)
        return
    var y = call[].y
    var y_words = List[Float32]()
    if y.rows > 0 and slot[].params[16] == 1.0:
        y = _fold_y(call, y_words)
    var store = _store(slot, call, True)
    if store:
        var taken = member_scope_rows_in(call[], store.value(), METRICS_STORE_PREFIX, True, _EST)
        try:
            _fit_rows(slot, call, fold_x_word(taken.source), taken.rows, taken.cols, taken.source, store.value(),
                List[Float32](), y)
        except problem:
            member_scope_rows_done_in(store.value(), taken)
            raise problem^
        _ = y_words^
        return
    var gathered = member_scope_rows_in(call[], call[].route.entry, METRICS_STORE_PREFIX, False, _EST)
    # The gathered words move to the state (a swap: the aggregate stays whole); their address is unchanged.
    var owned = List[Float32]()
    swap(owned, gathered.gathered)
    _fit_rows(slot, call, gathered.source, gathered.rows, gathered.cols, -1, entry_ref_unresolved(), owned^, y)
    _ = y_words^


def _fit_rows(slot: _SlotPtr, call: _CallPtr, x_word: Int, n: Int, d: Int, rows_slot: Int, rows_store: EntryRef,
              var owned: List[Float32], y: MemberMatrix) raises:
    """The door's checks in its order, its start (`_start`), then the legacy or extended fit entry into the
    slot's embedding. `x_word`: host rows (the caller's whole X, or `owned`'s gathered words), or a negative slot
    word of the metrics store (`rows_slot`, kept by the state on success): the entry then makes the finite test
    on the device, in the same words."""
    if call[].fit_output != 0 and call[].fit_output != MEMBER_OP_TRANSFORM:
        member_unsupported("UMAP", call[].fit_output)
    if call[].weights.rows > 0:
        raise Error("TypeError: a UMAP member fits no sample_weight")
    var cfg = _config(slot)
    var nc = cfg.n_components
    if call[].fit_output == MEMBER_OP_TRANSFORM:
        _dense(call[].dst, "the embedding buffer")
        if call[].dst.rows != n or call[].dst.cols != nc:
            raise Error("ValueError: the embedding buffer holds " + String(nc) + " components per row of X")
    if x_word >= 0:
        umap_check_input(x_word, n * d)
    umap_check_neighbors(cfg.n_neighbors, n)
    # The start (lane mf-nbr6): `_umap_initial`, then the spectral-sample check with its answer.
    var start_words = List[Float32]()
    var start = _start(slot, call, x_word, d, n, nc, start_words)
    var initial = Int(slot[].params[MF_INIT_WORD]) != MANIFOLD_INIT_SPECTRAL
    umap_check_spectral_samples(initial, n, nc)
    var has_target = y.rows > 0
    var target = UmapTargetWords.none()
    if has_target:
        if slot[].params[16] != 1.0:
            raise Error("member ABI: a UMAP member reads an l2 / euclidean target only; a categorical target "
                + "(its label codes and the unknown label -1) or a refused target setting stays with the "
                + "estimator's own fit")
        _dense(y, "y")
        var dims = umap_continuous_target_dims(2, y.rows, y.cols, n, y.data, y.rows * y.cols)
        target = UmapTargetWords(2, dims, Int(slot[].params[17]), slot[].params[18], y.data)
    slot[].fitted = False
    # Zero-filled, as the door's `family_array` output.
    var embedding = List[Float32](length=n * nc, fill=Float32(0.0))
    var embedding_address = Int(embedding.unsafe_ptr())
    var entry = call[].route.entry.call
    var columns = 0
    if umap_legacy_abi(initial, has_target, cfg.default_extras(), nc):
        columns = umap_fit_call(entry, x_word, embedding_address, n, d, cfg, not cfg.legacy_rates())
    else:
        columns = umap_fit_ex_call(entry, x_word, embedding_address, start, n, d, cfg, target)
    _ = len(start_words)   # the start's words stay alive through the fit call above
    umap_check_embedding(columns == nc, embedding_address, n * nc)
    # The whole X is borrowed, never copied (LIFETIME in the module docstring); a fold's rows are the state's
    # own (gathered words, or THE FITTED-ROWS SLOT). A copy or slot a previous fit, import or export left is
    # dropped.
    _drop_rows_slot(slot)
    slot[].rows = owned^
    if len(slot[].rows) > 0:
        slot[].rows_addr = Int(slot[].rows.unsafe_ptr())
    else:
        slot[].rows_addr = x_word
    if rows_slot >= 0:
        slot[].rows_slot = rows_slot
        slot[].rows_store = rows_store
    slot[].embedding = embedding^
    slot[].n = n
    slot[].features = d
    slot[].fitted = True
    if call[].fit_output == MEMBER_OP_TRANSFORM and n * nc > 0:
        unsafe_memcpy(dest=_F32P(unsafe_from_address=call[].dst.data), src=slot[].embedding.unsafe_ptr(),
            count=n * nc)


def _transform(slot: _SlotPtr, call: _CallPtr) raises:
    """`_umap_transform` after the door's refit refusal (the PACK / state path compares settings), over the
    whole X or (MEMBER_CAN_ROW_INDEX) a fold's rows: taken on the device from the metrics store when the state
    holds THE FITTED-ROWS SLOT and the call carries the scope with X's pin, else gathered on the host cores."""
    if not slot[].fitted:
        raise Error("UMAP transform requires a successful fit")
    _no_columns(call)
    if call[].row_index.rows == 0:
        _dense(call[].x, "X")
        _transform_rows(slot, call, call[].x.data, call[].x.rows, call[].x.cols)
        return
    if slot[].rows_slot >= 0 and call[].scope != MEMBER_SCOPE_NONE and call[].pin != 0:
        var store = slot[].rows_store
        var taken = member_scope_rows_in(call[], store, METRICS_STORE_PREFIX, True, _EST)
        try:
            _transform_rows(slot, call, fold_x_word(taken.source), taken.rows, taken.cols)
        finally:
            member_scope_rows_done_in(store, taken)
        return
    var gathered = member_scope_rows_in(call[], call[].route.entry, METRICS_STORE_PREFIX, False, _EST)
    _transform_rows(slot, call, gathered.source, gathered.rows, gathered.cols)
    _ = gathered.rows  # the gathered queries stay alive through the transform call above


def _transform_rows(slot: _SlotPtr, call: _CallPtr, q_word: Int, nq: Int, d: Int) raises:
    """The door's checks, the same-input reuse of the fitted embedding, else the transform entry with the
    door's tail. With the training rows (or the queries) in the metrics store the entry takes the slot form and
    makes the finite and reuse tests on the device words itself."""
    var cfg = _config(slot)
    var nc = cfg.n_components
    _dense(call[].dst, "the transform buffer")
    if call[].dst.rows != nq or call[].dst.cols != nc:
        raise Error("ValueError: the transform buffer holds " + String(nc) + " components per row of X")
    umap_check_transform_state(slot[].n, nc, slot[].n, nc, d, slot[].features)
    var count = nq * nc
    var host_rows = slot[].rows_addr >= 0 and q_word >= 0
    if q_word >= 0:
        umap_check_transform_input(q_word, nq * d)
    if host_rows and umap_transform_reuses_fit(q_word, slot[].rows_addr, nq, slot[].n, d):
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
        q_word, call[].dst.data, slot[].n, nq, d, cfg, tail)
    umap_check_transform_output(columns == nc, call[].dst.data, count)


def _own_rows(slot: _SlotPtr, call: _CallPtr) raises:
    """Copy the borrowed training rows into the slot once (first export); THE FITTED-ROWS SLOT is read home
    once through the device entry (UMAP_OP_ROWS_READ) and freed, and the state reads its host copy from then
    on."""
    var count = slot[].n * slot[].features
    if slot[].rows_addr == Int(slot[].rows.unsafe_ptr()) and len(slot[].rows) == count:
        return
    var owned = List[Float32](length=count, fill=Float32(0.0))
    if slot[].rows_slot >= 0:
        if count > 0:
            _ = umap_rows_read_call(call[].route.entry.call, slot[].rows_addr, Int(owned.unsafe_ptr()), count)
        _drop_rows_slot(slot)
    elif count > 0:
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
        _own_rows(slot, call)
        return
    var block = call[].block
    if (call[].flags & MEMBER_BLOCK_BY_INDEX) != 0:
        if block < 0 or block > 1:
            raise Error("member ABI: a UMAP member has two fitted blocks")
        block = MF_BLOCK_ROWS + block
    if block == MF_BLOCK_ROWS:
        _own_rows(slot, call)
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
            _drop_rows_slot(slot)
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
        var borrows = MEMBER_BORROWS_FIT
        if call[].state != 0:
            var held = _SlotPtr(unsafe_from_address=_lookup(call[].state, False))
            if held[].fitted and (held[].rows_slot >= 0 or len(held[].rows) > 0):
                borrows = 0
        # MEMBER_BORROWS_FIT: a fitted UMAP state reads the caller's X (its
        # training rows) until an export of the rows block, or
        # MEMBER_EXPORT_OWN, copied them (LIFETIME in the module docstring).
        # A fold-fitted state owns its rows (gathered, or THE FITTED-ROWS
        # SLOT) and borrows nothing.
        # MEMBER_CAN_ROW_INDEX / MEMBER_CAN_SCOPE (lane vdf-scope-cluster7):
        # a fold's rows under a host row index; SCOPE when the route's image
        # published the metrics store (a GPU metrics image), so a search's
        # folds are taken on the device out of one copy of X.
        # MEMBER_GIL_DOOR (THE GIL RULE; fx-gil-RESULT.md, round 4): one
        # operation at a time on the public thread, the caller releasing the
        # GIL around it as the UMAP door does around the same calls
        # (`umap_fit_call`, `umap_fit_ex_call`, `umap_transform_call`); the
        # `umap_check_*` helpers read the call's own words; the store has
        # its own lock. NOT `MEMBER_GIL_FREE` (the image is not traced).
        var caps = _CAPABILITIES | borrows | MEMBER_GIL_DOOR
        if metrics_store_find(entry_ref_mode(call[].route.entry), entry_ref_vendor(call[].route.entry)):
            caps |= MEMBER_CAN_SCOPE
        return caps
    if operation == MEMBER_OP_NEW:
        if call[].param_count != MANIFOLD_PARAM_COUNT or not call[].params:
            raise Error("member ABI: a UMAP member takes thirty-three parameters (kind, the eleven config words, "
                + "the four extras words, target metric, target_n_neighbors, target_weight, mode, the init words, "
                + "the exact integer words)")
        var words = _F64P(unsafe_from_address=call[].params)
        if words[0] != Float64(MANIFOLD_KIND_UMAP):
            raise Error("member ABI: unknown Manifold member kind")
        call[].state = _mint(words, MANIFOLD_PARAM_COUNT)
        return 0
    if operation == MEMBER_OP_FREE:
        _drop_rows_slot(_SlotPtr(unsafe_from_address=_lookup(call[].state, False)))
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
                raise Error("member ABI: a UMAP member takes thirty-three parameters")
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
