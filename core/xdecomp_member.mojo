# SPDX-License-Identifier: Apache-2.0
"""The XDecomp family as MEMBERS (core/member_abi.mojo): IncrementalPCA
(kind 0), NMF (kind 3), GaussianRandomProjection (kind 5) and
SparseRandomProjection (kind 6) so far; the family's other classes are not
published yet.

Random projections (kinds 5, 6): route `entry` the x_decomp operations
table of the estimator's image (core/decomp_entry_table.mojo, the entry
bindings/rp_native_door.mojo resolves through `decomp_entries_routed`);
`flags` bit XDECOMP_ROUTE_POOL (a GPU binding with the device pool
operations, the door's `decomp_resident_binding`): the first projection
of a fitted state uploads its matrix once (XD_DEV_ALLOC / XD_DEV_UPLOAD)
and every later one binds that pool id, as the door's resident record does;
freed at refit, import and free (XD_DEV_FREE through the same entry).
The fit reads only X's shape (and a FUSED build scans X there); a transform
is the tiled projection of an X its caller admitted (`tiled` 1), an
inverse_transform the plain product with the pseudo-inverse. No
MEMBER_CAN_TEST_ROWS: the unfused fit never reads X's values
(x_decomp/rp_owner.mojo:91-108 draws from the shape only) and the inverse
product tests nothing (rp_owner.mojo:113-140), so the caller's admission
scans X, as the door's admission does.

MEMBER_OP_NEW parameters, random projections (6):
  [0] kind: 5 Gaussian, 6 Sparse
  [1] n_components: -1 'auto' (johnson_lindenstrauss_min_dim(rows, eps) at
      fit: the door's bound, its two refusals), else the count (> 0)
  [2] eps (read only for 'auto', checked at admission as the bound checks it)
  [3] density: -1 'auto' (1 / sqrt(n_features) at fit), else in (0, 1]
      (Sparse only; 0 for Gaussian)
  [4] compute_inverse_components 0 / 1
  [5] seed: random_state & 0xFFFFFFFF, 0 for None
FastICA (kind 7; bindings/ica_native_unpack.mojo `ica_fit` / `ica_project`,
the same x_decomp table entry): MEMBER_OP_NEW parameters (9):
  [0] kind 7   [1] n_components: -1 None, else the count (min(count, rows,
  features) when whitening; ignored without whitening, as the door)
  [2] whiten: 0 False, 1 'unit-variance', 2 'arbitrary-variance'
  [3] algorithm: 0 'parallel', 1 'deflation'   [4] fun: 0 logcosh, 1 exp, 2 cube
  [5] alpha (fun_args['alpha'], 1.0 by default)   [6] max_iter   [7] tol
  [8] seed: random_state & 0xFFFFFFFF, 0 for None (a given w_init is not a
      member: admission answers False and the caller keeps the door)
Blocks: 0 scalars float64 [n_components_, n_features_in_, n_iter_, whiten
code, cols (n_features_in_ when whitening, else n_components_)]; 1
components_ (n_components_ x cols); 2 mixing_ (cols x n_components_); 3
whitening_ (n_components_ x n_features_in_, whitening only); 4 mean_ (1 x
n_features_in_, whitening only). `fit_output = MEMBER_OP_TRANSFORM` writes
the sources (rows x n_components_) into `dst` in the fit call. No
MEMBER_CAN_TEST_ROWS: x_decomp/ica_owner.mojo makes no finite test (the
door's `decomp_input` admission does).

FactorAnalysis (kind 8; bindings/fa_native_unpack.mojo `fa_fit` / `fa_query`,
the same table entry): MEMBER_OP_NEW parameters (6 + noise_variance_init):
  [0] kind 8   [1] n_components: -1 falsy (n_features at fit), else the count
  [2] max_iter   [3] rotation: 0 None, 1 varimax, 2 quartimax   [4] tol
  [5] noise_variance_init: -1 None (ones), else its length c, followed by
      its c values [6 .. 6 + c) (checked against n_features at fit)
Blocks: 0 scalars float64 [n_components_, n_features_in_, n_iter_, then
loglike_ (one word per EM iteration of a member fit; none after an
import)]; 1 components_ (n_components_ x n_features_in_); 2
noise_variance_ (1 x n_features_in_); 3 mean_ (1 x n_features_in_).
Operations: transform (rows x n_components_), score_samples (rows x 1),
score (the mean log-likelihood into `call.value`; no y, no weights). No
MEMBER_CAN_TEST_ROWS: x_decomp/fa_owner.mojo has no finite test.

Dictionary learning (the typed kit of bindings/dict_learning_run.mojo, the
door's own RUN): DictionaryLearning 9, MiniBatchDictionaryLearning 10,
SparseCoder 11, SparsePCA 12, MiniBatchSparsePCA 13. Route: `entry` the
image's x_decomp operations entry; `flags` XDECOMP_ROUTE_POOL (fitted
operands kept on the device across queries, as the door's resident
record), XDECOMP_ROUTE_KIT_RESIDENT (the kit's `resident`) and the image's
DK_HAS_* bits shifted left by XDECOMP_ROUTE_KIT_SHIFT. Operations: fit
(fit_output TRANSFORM = the door's fit_transform), transform,
inverse_transform. No MEMBER_CAN_TEST_ROWS: the kit borrows X untested
(bindings/decomp_kit_run.mojo:246-252 `borrow`; the door's `decomp_input`
admission tests it, bindings/dict_learning_native_unpack.mojo:127-129).
Coding words of kinds 9, 10, 11 ([1 .. 7], `_coding_words`):
  [1] transform_algorithm 0 lasso_lars, 1 lasso_cd, 2 lars, 3 omp, 4 threshold
  [2] transform_n_nonzero_coefs: -1 None (int(min(max(m / 10, 1), atoms)))
  [3] the lasso / threshold alpha (transform_alpha, else alpha, else 1)
  [4] transform_max_iter (lasso only, else 0)   [5] positive_code (lasso)
  [6] positive_code (threshold)   [7] split_sign 0 / 1
DictionaryLearning (17): [8] n_components (-1 None) [9] alpha [10] max_iter
  [11] tol [12] fit_algorithm 0 lars, 1 cd [13] seed (random_state &
  0xFFFFFFFF) [14] positive_dict [15] positive_code [16] transform_max_iter
MiniBatchDictionaryLearning (20): [8] n_components [9] batch_size [10] seed
  [11] shuffle [12] max_iter [13] max_no_improvement (-1 None) [14]
  fit_algorithm [15] transform_max_iter [16] positive_dict [17]
  positive_code [18] alpha [19] tol
SparseCoder (11): [8] the address of the admitted float32 dictionary (the
  admit pins it; the fit copies it into components_) [9] its rows [10] its
  columns
SparsePCA (8): [1] n_components [2] alpha [3] max_iter [4] tol [5] method
  0 lars, 1 cd [6] seed [7] ridge_alpha
MiniBatchSparsePCA (11): [1] n_components [2] alpha [3] batch_size [4] seed
  [5] shuffle [6] max_iter [7] max_no_improvement [8] method [9] tol
  [10] ridge_alpha
Blocks: 0 scalars float64 [n_components_, n_features_in_, n_iter_, n_steps_
(0 where the class has none), then error_ (DictionaryLearning, SparsePCA)];
1 components_ (n_components_ x n_features_in_); 2 mean_ (1 x
n_features_in_, SparsePCA kinds only).

Random projection blocks: 0 `MEMBER_BLOCK_SCALARS` float64 [n_components_,
n_features_in_, density_ (0 Gaussian), inverse 0 / 1]; 1
`XDECOMP_BLOCK_COMPONENTS` float32 n_components_ x n_features_in_
(components_); 2 `XDECOMP_BLOCK_RP_INVERSE` float32 n_features_in_ x
n_components_ (inverse_components_; only when the scalars say 1).

One entry, `xdecomp_member_v2`, published by each x_decomp image
(`xdecomp_member_publish`, beside its linear entry table publication:
x_decomp/publish_device_run.mojo, x_decomp/publish_host_run.mojo). It runs
the family's existing typed RUN, the IncrementalPCA native driver
(core/ipca_native_api.mojo `IPCANativeDriver`: IPCA_FIT, IPCA_QUERY,
IPCA_PREPARE, IPCA_FREE) and the NMF native driver (core/nmf_native_api.mojo
`NMFNativeDriver`: NMF_FIT, NMF_INVERSE) over the entry of the `DeviceRoute`
it is handed, the entry the estimator's own door calls
(bindings/ipca_native_door.mojo, bindings/nmf_native_door.mojo). No kernel
is touched. No score (neither class has one).

NMF: route `entry` the x_decomp image's NMF entry; `flags` bit
XDECOMP_ROUTE_NMF_INVERSE (the entry exports the inverse, NMF_CAP_INVERSE;
an inverse_transform without it is refused in the door's words). The fit
with `fit_output = MEMBER_OP_TRANSFORM` writes W into `dst` in the ONE fit
call (the door's fit_transform returns the fit's own W); a transform is the
door's solver run with H fixed (NMF_TRANSFORM), the regularization scaled by
the transform's rows as the door scales it. X is tested finite in the kernel
over its own upload, as the door's deferred admission does.

MEMBER_OP_NEW parameters, NMF (12):
  [0] kind: 3 NMF
  [1] n_components: -1 None or 'auto' (n_features at fit), else the count
  [2] init: -1 None (nndsvda when n_components <= min(rows, features), else
      random, at fit), 0 random, 1 nndsvd, 2 nndsvda, 3 nndsvdar ('custom'
      takes W and H: refused at admission, not a member)
  [3] solver: 0 'cd', 1 'mu'     [4] beta: 2 frobenius, 1 kullback-leibler, 0 itakura-saito
  [5] max_iter   [6] tol   [7] alpha_W   [8] alpha_H (alpha_W for 'same')
  [9] l1_ratio   [10] shuffle 0 / 1   [11] seed: random_state & 0xFFFFFFFF, 0 for None
NMF blocks: 0 `MEMBER_BLOCK_SCALARS` float64 [n_components_, n_features_in_,
n_iter_, reconstruction_err_]; 1 `XDECOMP_BLOCK_COMPONENTS` float32
n_components_ x n_features_in_ (components_).

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
    MEMBER_OP_SCORE, MEMBER_OP_SCORE_SAMPLES, MEMBER_CAN_SCORE, MEMBER_CAN_SCORE_SAMPLES,
    MEMBER_OP_PREDICT, MEMBER_OP_DECISION, MEMBER_CAN_PREDICT, MEMBER_CAN_DECISION, MEMBER_I32,
    MEMBER_CAN_FIT, MEMBER_CAN_TRANSFORM, MEMBER_CAN_INVERSE_TRANSFORM, MEMBER_CAN_EXPORT, MEMBER_CAN_IMPORT,
    MEMBER_BLOCK_SCALARS, MEMBER_BLOCK_FAMILY, MEMBER_BLOCK_BY_INDEX, MEMBER_IMPORT_DONE,
    member_publish, member_call_from_spans, member_unsupported,
)
from core.nmf_native_api import (
    NMFNativeDriver, NMFFitRequest, NMFInverseRequest, nmf_regularization,
)
from core.admitted_refusals import admitted_finite_refuse
from std.math import sqrt
from core.decomp_factor_api import (
    RPFitRequest, RPProjectRequest, rp_fit_call, rp_project_call,
    ICAFitRequest, ICAProjectRequest, ica_fit_call, ica_project_call,
    FAFitRequest, FAQueryRequest, fa_fit_call, fa_query_call,
)
from core.decomp_entry_table import decomp_table_call
from x_decomp.descriptor_schema import XD_DEV_ALLOC, XD_DEV_FREE, XD_DEV_UPLOAD
from x_decomp.jl_run import _cube_rounded, _pinned_log
from x_decomp.nmf import NmfArgs
from x_decomp.nmf_owner import NMF_INIT_RANDOM, NMF_INIT_NNDSVDA, NMF_INIT_NNDSVDAR, NMF_TRANSFORM
from x_decomp.decomp_owner_common_run import decomp_gemm_check, decomp_min_samples
from core.member_families import MEMBER_FAMILY_XDECOMP
from core.entry_table import EntryRef, entry_ref_unresolved
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.shared_slot import SharedSlot, shared_slot_lock, shared_slot_unlock
from bindings.decomp_kit_run import DKit, DK_SCALE, DK_ADDS
from bindings.mcd_run import mcd_fit_run, mcd_labels_run, mcd_mahal
from core.decomp_native_impl import decomp_f32_to_i32, decomp_u8_to_f32
from bindings.dict_learning_run import (
    EncodeWords, Gram, DictWords, MiniBatchWords, SpcaFit, dl_split_sign, dl_encode_run, dl_inverse_run,
    dl_dict_learning, dl_minibatch_fit, dl_spca_project, dl_spca_inverse, dl_spca_fit_run,
)

comptime XDECOMP_KIND_IPCA = 0
comptime XDECOMP_IPCA_PARAM_COUNT = 4
comptime XDECOMP_KIND_NMF = 3
comptime XDECOMP_NMF_PARAM_COUNT = 12
comptime XDECOMP_KIND_GRP = 5
comptime XDECOMP_KIND_SRP = 6
comptime XDECOMP_RP_PARAM_COUNT = 6
comptime XDECOMP_KIND_ICA = 7
comptime XDECOMP_ICA_PARAM_COUNT = 9
comptime XDECOMP_KIND_FA = 8
comptime XDECOMP_FA_PARAM_COUNT = 6  # plus the noise_variance_init values
# The dictionary-learning kinds run the typed kit (bindings/decomp_kit_run.mojo
# `DKit` over the image's x_decomp operations entry).
comptime XDECOMP_KIND_DL = 9
comptime XDECOMP_DL_PARAM_COUNT = 17
comptime XDECOMP_KIND_MBDL = 10
comptime XDECOMP_MBDL_PARAM_COUNT = 20
comptime XDECOMP_KIND_CODER = 11
comptime XDECOMP_CODER_PARAM_COUNT = 11
comptime XDECOMP_KIND_SPCA = 12
comptime XDECOMP_SPCA_PARAM_COUNT = 8
comptime XDECOMP_KIND_MBSPCA = 13
comptime XDECOMP_MBSPCA_PARAM_COUNT = 11
comptime XDECOMP_KIND_ENVELOPE = 14
comptime XDECOMP_ENVELOPE_PARAM_COUNT = 6

comptime XDECOMP_BLOCK_COMPONENTS = MEMBER_BLOCK_FAMILY
comptime XDECOMP_BLOCK_SINGULAR = MEMBER_BLOCK_FAMILY + 1
# Random projections: the pseudo-inverse in the second family block.
comptime XDECOMP_BLOCK_RP_INVERSE = MEMBER_BLOCK_FAMILY + 1
comptime XDECOMP_BLOCK_MEAN = MEMBER_BLOCK_FAMILY + 2
comptime XDECOMP_BLOCK_VAR = MEMBER_BLOCK_FAMILY + 3
comptime XDECOMP_BLOCK_VARIANCE = MEMBER_BLOCK_FAMILY + 4
comptime XDECOMP_BLOCK_RATIO = MEMBER_BLOCK_FAMILY + 5

comptime XDECOMP_ROUTE_RESIDENT = 1
comptime XDECOMP_ROUTE_NMF_INVERSE = 2
comptime XDECOMP_ROUTE_POOL = 4
# The kit kinds: bit 8 = the image is resident (`x_decomp_dev_ew`, the
# kit's `resident`); the image's DK_HAS_* bits sit from bit 8 up.
comptime XDECOMP_ROUTE_KIT_RESIDENT = 8
comptime XDECOMP_ROUTE_KIT_SHIFT = 8

comptime _IPCA_SCALARS = 5
comptime _NMF_SCALARS = 4
comptime _RP_SCALARS = 4
comptime _ICA_SCALARS = 5
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


def _is_rp(kind: Int) -> Bool:
    return kind == XDECOMP_KIND_GRP or kind == XDECOMP_KIND_SRP


def _is_kit(kind: Int) -> Bool:
    return kind >= XDECOMP_KIND_DL and kind <= XDECOMP_KIND_MBSPCA


def _is_spca(kind: Int) -> Bool:
    return kind == XDECOMP_KIND_SPCA or kind == XDECOMP_KIND_MBSPCA


def _pooled(kind: Int) -> Bool:
    """Handles of this kind are x_decomp pool ids (XD_DEV_FREE)."""
    return _is_rp(kind) or _is_kit(kind) or kind == XDECOMP_KIND_ENVELOPE


def _free_handles(entry: EntryRef, handles: List[Int], pool: Bool) raises:
    """Release resident device state through the entry that made it
    (`pool`: x_decomp pool ids, XD_DEV_FREE; else IncrementalPCA state ids).
    Every handle is attempted; the first failure is raised after the rest."""
    var failure = String("")
    for k in range(len(handles)):  # driver-loop: five resident state handles
        if handles[k] >= 0:
            try:
                if pool:
                    var ids: List[Int] = [handles[k]]
                    _ = decomp_table_call(entry, XD_DEV_FREE, List[Int](), ids, List[Float64]())
                else:
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
    var pool = False
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
            pool = len(slot[].words) > 0 and _pooled(slot[].kind())
            slot[].handles = List[Int](length=_HANDLES, fill=-1)
            slot[].handle_entry = entry_ref_unresolved()
            slot[].live = False
            slot[].clear()
            slot[].words = List[Float64]()
            store[].free.append(index)
    finally:
        shared_slot_unlock(lock)
    if len(held) > 0:
        _free_handles(entry, held, pool)
    return address


def _release(slot: _Slot) raises:
    """Release this state's resident device state before its words change."""
    var held = slot[].handles.copy()
    var entry = slot[].handle_entry
    slot[].handles = List[Int](length=_HANDLES, fill=-1)
    slot[].handle_entry = entry_ref_unresolved()
    _free_handles(entry, held, _pooled(slot[].kind()))


def _kind_name(kind: Int) -> String:
    if kind == XDECOMP_KIND_NMF:
        return String("NMF")
    if kind == XDECOMP_KIND_GRP:
        return String("GaussianRandomProjection")
    if kind == XDECOMP_KIND_SRP:
        return String("SparseRandomProjection")
    if kind == XDECOMP_KIND_ICA:
        return String("FastICA")
    if kind == XDECOMP_KIND_FA:
        return String("FactorAnalysis")
    if kind == XDECOMP_KIND_DL:
        return String("DictionaryLearning")
    if kind == XDECOMP_KIND_MBDL:
        return String("MiniBatchDictionaryLearning")
    if kind == XDECOMP_KIND_CODER:
        return String("SparseCoder")
    if kind == XDECOMP_KIND_SPCA:
        return String("SparsePCA")
    if kind == XDECOMP_KIND_MBSPCA:
        return String("MiniBatchSparsePCA")
    if kind == XDECOMP_KIND_ENVELOPE:
        return String("EllipticEnvelope")
    return String("IncrementalPCA")


def _name(slot: _Slot) -> String:
    return _kind_name(slot[].kind())


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


# ---------------------------------------------------------------- NMF


def _nmf_args(slot: _Slot, rows: Int, features: Int, update_h: Bool) -> NmfArgs:
    # The door's `_regs` (scaled by this call's rows) and `NmfArgs`.
    var regs = nmf_regularization(rows, features, slot[].words[7], slot[].words[8], slot[].words[9])
    return NmfArgs(slot[].words[4], update_h, Int(slot[].words[5]), slot[].words[6], regs.l1w, regs.l1h,
        regs.l2w, regs.l2h, slot[].words[10] != 0.0, Int(slot[].words[11]))


def _fit_nmf(slot: _Slot, call: _Call) raises:
    var estimator = _name(slot)
    _whole(call, estimator)
    _dense(call[].x, "X", estimator)
    if call[].weights.rows > 0:
        raise Error("TypeError: a " + estimator + " member takes no sample_weight")
    if call[].fit_output != 0 and call[].fit_output != MEMBER_OP_TRANSFORM:
        member_unsupported(estimator, call[].fit_output)
    var n = call[].x.rows
    var d = call[].x.cols
    var nc = d if slot[].words[1] < 0.0 else Int(slot[].words[1])
    var smaller = min(n, d)
    var init = Int(slot[].words[2])
    if init < 0:
        init = NMF_INIT_NNDSVDA if nc <= smaller else NMF_INIT_RANDOM
    if init > NMF_INIT_NNDSVDAR:
        raise Error("member ABI: unknown NMF init code " + String(init))
    if init != NMF_INIT_RANDOM and nc > smaller:
        raise Error("init = 'nndsvd' can only be used when n_components <= min(n_samples, n_features)")
    var w_output = 0
    var w_scratch = List[Float32]()
    if call[].fit_output == MEMBER_OP_TRANSFORM:
        # W of the fit itself, written in the fit call (the door's fit_transform).
        _dense(call[].dst, "the output buffer", estimator)
        if call[].dst.rows != n or call[].dst.cols != nc:
            raise Error("ValueError: the output buffer holds n_components values per row of X")
        w_output = call[].dst.data
    else:
        w_scratch = List[Float32](length=max(n * nc, 1), fill=Float32(0))
        w_output = Int(w_scratch.unsafe_ptr())
    _release(slot)
    slot[].clear()
    slot[].blocks.append(List[Float32](length=max(nc * d, 1), fill=Float32(0)))
    var h_output = _address(slot, 0)
    var request = NMFFitRequest(call[].x.data, h_output, h_output, w_output, h_output, n, d, nc, init,
        Int(slot[].words[3]), _nmf_args(slot, n, d, True), True)
    var result = NMFNativeDriver(call[].route.entry.call).fit(request)
    _ = w_scratch^
    admitted_finite_refuse(result.finite, estimator)
    slot[].count = nc
    slot[].features = d
    slot[].scalars = [Float64(nc), Float64(d), Float64(result.iterations), result.error]
    slot[].fitted = True


def _query_nmf(slot: _Slot, call: _Call, inverse: Bool) raises:
    """transform (the solver with H fixed: X -> W, rows x n_components_) or
    inverse_transform (W H: rows x n_components_ -> rows x n_features_in_)."""
    var estimator = _name(slot)
    if not slot[].fitted:
        raise Error("NotFittedError: this " + estimator + " member is not fitted yet")
    _whole(call, estimator)
    _dense(call[].x, "X", estimator)
    _dense(call[].dst, "the output buffer", estimator)
    var rows = call[].x.rows
    var nc = slot[].count
    var d = slot[].features
    var width = d if inverse else nc
    if inverse:
        if (call[].route.flags & XDECOMP_ROUTE_NMF_INVERSE) == 0:
            raise Error("NotImplementedError: mojolearn NMF: this build of _mojolearn_x_decomp does not export "
                        + "x_decomp_nmf_inverse; rebuild it with bindings/build_x_decomp.sh")
        decomp_gemm_check(call[].x.cols, nc)
    elif call[].x.cols != d:
        raise Error("X has " + String(call[].x.cols) + " features, but NMF is expecting " + String(d))
    if call[].dst.rows != rows or call[].dst.cols != width:
        raise Error("ValueError: the output buffer holds " + String(width) + " values per row of X")
    var driver = NMFNativeDriver(call[].route.entry.call)
    if inverse:
        driver.inverse(NMFInverseRequest(call[].x.data, _address(slot, 0), call[].dst.data, rows, d, nc))
        return
    # The H output of a transform is the door's scratch (NMFScratch): kept
    # in the slot, sized once per fitted state.
    if len(slot[].blocks) < 2:
        slot[].blocks.append(List[Float32](length=max(nc * d, 1), fill=Float32(0)))
    var scratch = _address(slot, 1)
    var request = NMFFitRequest(call[].x.data, scratch, _address(slot, 0), call[].dst.data, scratch, rows, d, nc,
        NMF_TRANSFORM, Int(slot[].words[3]), _nmf_args(slot, rows, d, False), True)
    var result = driver.fit(request)
    admitted_finite_refuse(result.finite, estimator)


def _export_nmf(slot: _Slot, call: _Call) raises:
    var block = call[].block
    if (call[].flags & MEMBER_BLOCK_BY_INDEX) != 0:
        if block < 0 or block > 1:
            raise Error("member ABI: an NMF member has no fitted block index " + String(block))
        block = MEMBER_BLOCK_SCALARS if block == 0 else XDECOMP_BLOCK_COMPONENTS
    if block == MEMBER_BLOCK_SCALARS:
        call[].dst = MemberMatrix(Int(slot[].scalars.unsafe_ptr()), 1, len(slot[].scalars), MEMBER_F64)
    elif block == XDECOMP_BLOCK_COMPONENTS:
        call[].dst = MemberMatrix(_address(slot, 0), slot[].count, slot[].features)
    else:
        raise Error("member ABI: an NMF member has no fitted block " + String(block))
    call[].block = block
    call[].out_cols = 2


def _import_nmf(slot: _Slot, call: _Call) raises:
    var estimator = _name(slot)
    var block = call[].block
    if slot[].fitted or len(slot[].blocks) != 1:
        slot[].clear()
        slot[].blocks.append(List[Float32]())
    if block == MEMBER_BLOCK_SCALARS:
        if not call[].x.dense_host(MEMBER_F64) or call[].x.rows * call[].x.cols != _NMF_SCALARS:
            raise Error("member ABI: the NMF scalars block is four host float64 words")
        var source = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call[].x.data)
        slot[].scalars = List[Float64](length=_NMF_SCALARS, fill=0.0)
        for index in range(_NMF_SCALARS):  # driver-loop: four fitted scalars
            slot[].scalars[index] = source[index]
    elif block == XDECOMP_BLOCK_COMPONENTS:
        var size = call[].x.rows * call[].x.cols
        var words = List[Float32](length=size, fill=0.0)
        if size > 0:
            _dense(call[].x, "a fitted block", estimator)
            unsafe_memcpy(dest=words.unsafe_ptr(), src=_F32P(unsafe_from_address=call[].x.data), count=size)
        slot[].blocks[0] = words^
    elif block != 0 or (call[].flags & MEMBER_IMPORT_DONE) == 0:
        raise Error("member ABI: an NMF member has no fitted block " + String(block))
    if (call[].flags & MEMBER_IMPORT_DONE) == 0:
        return
    if len(slot[].scalars) != _NMF_SCALARS:
        raise Error("member ABI: a fitted NMF state needs its scalars block")
    var nc = Int(slot[].scalars[0])
    var d = Int(slot[].scalars[1])
    if nc < 1 or d < 1 or len(slot[].blocks[0]) != nc * d:
        raise Error("member ABI: a fitted NMF state needs every fitted block, of the fitted sizes")
    slot[].count = nc
    slot[].features = d
    slot[].fitted = True


# ---------------------------------------------------------------- FastICA


def _fit_ica(slot: _Slot, call: _Call) raises:
    """bindings/ica_native_unpack.mojo `ica_fit` (`FastICA._fit`): one
    ICAFitRequest through the route's operations table; with
    `fit_output = MEMBER_OP_TRANSFORM` the sources go to `dst` in the fit
    call (the door's `project`)."""
    var estimator = _name(slot)
    _whole(call, estimator)
    _dense(call[].x, "X", estimator)
    if call[].weights.rows > 0:
        raise Error("TypeError: a " + estimator + " member takes no sample_weight")
    if call[].fit_output != 0 and call[].fit_output != MEMBER_OP_TRANSFORM:
        member_unsupported(estimator, call[].fit_output)
    var n = call[].x.rows
    var d = call[].x.cols
    decomp_min_samples(n, d, "FastICA")
    var code = Int(slot[].words[2])
    var smaller = min(n, d)
    var nc = smaller
    if code != 0 and slot[].words[1] >= -0.5:
        nc = min(Int(slot[].words[1]), smaller)
    if nc < 0:
        raise Error("ValueError: negative dimensions are not allowed")
    var cols = d if code != 0 else nc
    var project = call[].fit_output == MEMBER_OP_TRANSFORM
    var sources = 0
    if project:
        _dense(call[].dst, "the output buffer", estimator)
        if call[].dst.rows != n or call[].dst.cols != nc:
            raise Error("ValueError: the output buffer holds n_components values per row of X")
        sources = call[].dst.data
    slot[].clear()
    # components, mixing, whitening, mean: the door's four output arrays.
    slot[].blocks.append(List[Float32](length=max(nc * cols, 1), fill=Float32(0)))
    slot[].blocks.append(List[Float32](length=max(cols * nc, 1), fill=Float32(0)))
    slot[].blocks.append(List[Float32](length=(nc * d) if code != 0 else 0, fill=Float32(0)))
    slot[].blocks.append(List[Float32](length=d if code != 0 else 0, fill=Float32(0)))
    var spare = _address(slot, 0)
    var request = ICAFitRequest(call[].x.data, 0, spare, _address(slot, 1),
        _address(slot, 2) if code != 0 else spare, _address(slot, 3) if code != 0 else spare,
        sources if project else spare,
        n, d, nc, code, Int(slot[].words[8]), Int(slot[].words[3]), Int(slot[].words[4]),
        Int(slot[].words[6]), project, slot[].words[5], slot[].words[7])
    var iterations = ica_fit_call(call[].route.entry, request)
    slot[].count = nc
    slot[].features = d
    slot[].scalars = [Float64(nc), Float64(d), Float64(iterations), Float64(code), Float64(cols)]
    slot[].fitted = True


def _query_ica(slot: _Slot, call: _Call, inverse: Bool) raises:
    """`ica_project`: transform (X - mean_) components_^T, inverse_transform
    X mixing_^T (+ mean_), with the door's inner-dimension refusal."""
    var estimator = _name(slot)
    if not slot[].fitted:
        raise Error("NotFittedError: this " + estimator + " member is not fitted yet")
    _whole(call, estimator)
    _dense(call[].x, "X", estimator)
    _dense(call[].dst, "the output buffer", estimator)
    var rows = call[].x.rows
    var width = call[].x.cols
    var nc = slot[].count
    var cols = Int(slot[].scalars[4])
    var whiten = slot[].scalars[3] != 0.0
    # components_ is nc x cols, mixing_ cols x nc.
    var matrix_rows = cols if inverse else nc
    decomp_gemm_check(width, nc if inverse else cols)
    if call[].dst.rows != rows or call[].dst.cols != matrix_rows:
        raise Error("ValueError: the output buffer holds " + String(matrix_rows) + " values per row of X")
    var matrix = _address(slot, 1 if inverse else 0)
    var request = ICAProjectRequest(call[].x.data, matrix, _address(slot, 3) if whiten else matrix,
        call[].dst.data, inverse, rows, width, matrix_rows, whiten)
    ica_project_call(call[].route.entry, request)


def _export_ica(slot: _Slot, call: _Call) raises:
    var block = call[].block
    if (call[].flags & MEMBER_BLOCK_BY_INDEX) != 0:
        if block < 0 or block > 4:
            raise Error("member ABI: a FastICA member has no fitted block index " + String(block))
        block = MEMBER_BLOCK_SCALARS if block == 0 else XDECOMP_BLOCK_COMPONENTS + block - 1
    var nc = slot[].count
    var d = slot[].features
    var cols = Int(slot[].scalars[4])
    var whiten = slot[].scalars[3] != 0.0
    if block == MEMBER_BLOCK_SCALARS:
        call[].dst = MemberMatrix(Int(slot[].scalars.unsafe_ptr()), 1, len(slot[].scalars), MEMBER_F64)
    elif block == XDECOMP_BLOCK_COMPONENTS:
        call[].dst = MemberMatrix(_address(slot, 0), nc, cols)
    elif block == XDECOMP_BLOCK_COMPONENTS + 1:
        call[].dst = MemberMatrix(_address(slot, 1), cols, nc)
    elif block == XDECOMP_BLOCK_COMPONENTS + 2 and whiten and len(slot[].blocks[2]) == nc * d:
        call[].dst = MemberMatrix(_address(slot, 2), nc, d)
    elif block == XDECOMP_BLOCK_COMPONENTS + 3 and whiten:
        call[].dst = MemberMatrix(_address(slot, 3), 1, d)
    else:
        raise Error("member ABI: a FastICA member has no fitted block " + String(block))
    call[].block = block
    call[].out_cols = 5


def _import_ica(slot: _Slot, call: _Call) raises:
    var estimator = _name(slot)
    var block = call[].block
    if slot[].fitted or len(slot[].blocks) != 4:
        slot[].clear()
        for _ in range(4):  # driver-loop: four fitted blocks
            slot[].blocks.append(List[Float32]())
    if block == MEMBER_BLOCK_SCALARS:
        if not call[].x.dense_host(MEMBER_F64) or call[].x.rows * call[].x.cols != _ICA_SCALARS:
            raise Error("member ABI: the FastICA scalars block is five host float64 words")
        var source = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call[].x.data)
        slot[].scalars = List[Float64](length=_ICA_SCALARS, fill=0.0)
        for index in range(_ICA_SCALARS):  # driver-loop: five fitted scalars
            slot[].scalars[index] = source[index]
    elif block >= XDECOMP_BLOCK_COMPONENTS and block <= XDECOMP_BLOCK_COMPONENTS + 3:
        var size = call[].x.rows * call[].x.cols
        var words = List[Float32](length=size, fill=0.0)
        if size > 0:
            _dense(call[].x, "a fitted block", estimator)
            unsafe_memcpy(dest=words.unsafe_ptr(), src=_F32P(unsafe_from_address=call[].x.data), count=size)
        slot[].blocks[block - XDECOMP_BLOCK_COMPONENTS] = words^
    elif block != 0 or (call[].flags & MEMBER_IMPORT_DONE) == 0:
        raise Error("member ABI: a FastICA member has no fitted block " + String(block))
    if (call[].flags & MEMBER_IMPORT_DONE) == 0:
        return
    if len(slot[].scalars) != _ICA_SCALARS:
        raise Error("member ABI: a fitted FastICA state needs its scalars block")
    var nc = Int(slot[].scalars[0])
    var d = Int(slot[].scalars[1])
    var cols = Int(slot[].scalars[4])
    var whiten = slot[].scalars[3] != 0.0
    if (nc < 1 or d < 1 or cols < 1 or len(slot[].blocks[0]) != nc * cols or len(slot[].blocks[1]) != cols * nc
            or (whiten and len(slot[].blocks[3]) != d)):
        raise Error("member ABI: a fitted FastICA state needs every fitted block, of the fitted sizes")
    slot[].count = nc
    slot[].features = d
    slot[].fitted = True


# ---------------------------------------------------------------- FactorAnalysis

comptime _FA_TRANSFORM = 0
comptime _FA_SCORE_SAMPLES = 3
comptime _FA_SCORE = 4


def _fit_fa(slot: _Slot, call: _Call) raises:
    """bindings/fa_native_unpack.mojo `fa_fit`: psi0 (ones, or the
    noise_variance_init words), one FAFitRequest through the route's table,
    the EM log-likelihoods kept after the scalars."""
    var estimator = _name(slot)
    _whole(call, estimator)
    _dense(call[].x, "X", estimator)
    if call[].weights.rows > 0:
        raise Error("TypeError: a " + estimator + " member takes no sample_weight")
    if call[].fit_output != 0 and call[].fit_output != MEMBER_OP_TRANSFORM:
        member_unsupported(estimator, call[].fit_output)
    var n = call[].x.rows
    var d = call[].x.cols
    var nc = Int(slot[].words[1]) if slot[].words[1] >= 0.5 else d
    if nc < 0:
        raise Error("ValueError: negative dimensions are not allowed")
    var max_iter = Int(slot[].words[2])
    var given = Int(slot[].words[5])
    var psi0 = List[Float32](length=max(d, 1), fill=Float32(1.0))
    if given >= 0:
        if given != d:
            raise Error("noise_variance_init dimension does not match the number of features : " + String(given)
                        + " != " + String(d))
        for j in range(d):  # driver-loop: over the features' initial noise variances (model structure)
            psi0[j] = Float32(slot[].words[XDECOMP_FA_PARAM_COUNT + j])
    slot[].clear()
    slot[].blocks.append(List[Float32](length=max(nc * d, 1), fill=Float32(0)))
    slot[].blocks.append(List[Float32](length=max(d, 1), fill=Float32(0)))
    slot[].blocks.append(List[Float32](length=max(d, 1), fill=Float32(0)))
    var ll = List[Float64](length=max(max_iter, 1), fill=Float64(0))
    var rotation = Int(slot[].words[3])
    var request = FAFitRequest(call[].x.data, Int(psi0.unsafe_ptr()), _address(slot, 0), _address(slot, 1),
        _address(slot, 2), Int(ll.unsafe_ptr()), n, d, nc, max_iter, rotation, slot[].words[4])
    var iterations = fa_fit_call(call[].route.entry, request)
    _ = psi0^
    if iterations == 0:
        # The door's `_fit_store` refusal with no fitted W.
        slot[].clear()
        raise Error("AttributeError: 'NoneType' object has no attribute '"
                    + (String("T") if rotation != 0 else String("out")) + "'")
    var scalars = List[Float64](capacity=3 + iterations)
    scalars.append(Float64(nc))
    scalars.append(Float64(d))
    scalars.append(Float64(iterations))
    for k in range(iterations):  # driver-loop: over the EM iterations' log-likelihoods
        scalars.append(ll[k])
    _ = ll^
    slot[].scalars = scalars^
    slot[].count = nc
    slot[].features = d
    slot[].fitted = True
    if call[].fit_output == MEMBER_OP_TRANSFORM:
        _query_fa(slot, call, _FA_TRANSFORM)


def _query_fa(slot: _Slot, call: _Call, op: Int) raises:
    """`fa_query`: transform (rows x n_components_), score_samples (rows x
    1) or score (`call.value`), with the door's broadcast refusal."""
    var estimator = _name(slot)
    if not slot[].fitted:
        raise Error("NotFittedError: this " + estimator + " member is not fitted yet")
    _whole(call, estimator)
    _dense(call[].x, "X", estimator)
    if op == _FA_SCORE and (call[].weights.rows > 0 or call[].y.rows > 0):
        raise Error("TypeError: FactorAnalysis.score takes X only (no y, no sample_weight)")
    var n = call[].x.rows
    var width = call[].x.cols
    var nc = slot[].count
    var d = slot[].features
    if width != d:
        raise Error("x_decomp: cannot broadcast 1x" + String(d) + " against " + String(n) + "x" + String(width))
    var dst = _address(slot, 0)
    if op != _FA_SCORE:
        _dense(call[].dst, "the output buffer", estimator)
        var wanted = nc if op == _FA_TRANSFORM else 1
        if call[].dst.rows != n or call[].dst.cols != wanted:
            raise Error("ValueError: the output buffer holds " + String(wanted) + " values per row of X")
        dst = call[].dst.data
    var request = FAQueryRequest(op, call[].x.data, _address(slot, 0), _address(slot, 1), _address(slot, 2), dst,
        n, d, nc)
    var value = fa_query_call(call[].route.entry, request)
    if op == _FA_SCORE:
        call[].value = value


def _export_fa(slot: _Slot, call: _Call) raises:
    var block = call[].block
    if (call[].flags & MEMBER_BLOCK_BY_INDEX) != 0:
        if block < 0 or block > 3:
            raise Error("member ABI: a FactorAnalysis member has no fitted block index " + String(block))
        block = MEMBER_BLOCK_SCALARS if block == 0 else XDECOMP_BLOCK_COMPONENTS + block - 1
    if block == MEMBER_BLOCK_SCALARS:
        call[].dst = MemberMatrix(Int(slot[].scalars.unsafe_ptr()), 1, len(slot[].scalars), MEMBER_F64)
    elif block == XDECOMP_BLOCK_COMPONENTS:
        call[].dst = MemberMatrix(_address(slot, 0), slot[].count, slot[].features)
    elif block == XDECOMP_BLOCK_COMPONENTS + 1 or block == XDECOMP_BLOCK_COMPONENTS + 2:
        call[].dst = MemberMatrix(_address(slot, block - XDECOMP_BLOCK_COMPONENTS), 1, slot[].features)
    else:
        raise Error("member ABI: a FactorAnalysis member has no fitted block " + String(block))
    call[].block = block
    call[].out_cols = 4


def _import_fa(slot: _Slot, call: _Call) raises:
    var estimator = _name(slot)
    var block = call[].block
    if slot[].fitted or len(slot[].blocks) != 3:
        slot[].clear()
        for _ in range(3):  # driver-loop: three fitted blocks
            slot[].blocks.append(List[Float32]())
    if block == MEMBER_BLOCK_SCALARS:
        var count = call[].x.rows * call[].x.cols
        if not call[].x.dense_host(MEMBER_F64) or count < 3:
            raise Error("member ABI: the FactorAnalysis scalars block is three or more host float64 words")
        var source = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call[].x.data)
        slot[].scalars = List[Float64](length=count, fill=0.0)
        for index in range(count):  # driver-loop: the fitted scalars and log-likelihoods
            slot[].scalars[index] = source[index]
    elif block >= XDECOMP_BLOCK_COMPONENTS and block <= XDECOMP_BLOCK_COMPONENTS + 2:
        var size = call[].x.rows * call[].x.cols
        var words = List[Float32](length=size, fill=0.0)
        if size > 0:
            _dense(call[].x, "a fitted block", estimator)
            unsafe_memcpy(dest=words.unsafe_ptr(), src=_F32P(unsafe_from_address=call[].x.data), count=size)
        slot[].blocks[block - XDECOMP_BLOCK_COMPONENTS] = words^
    elif block != 0 or (call[].flags & MEMBER_IMPORT_DONE) == 0:
        raise Error("member ABI: a FactorAnalysis member has no fitted block " + String(block))
    if (call[].flags & MEMBER_IMPORT_DONE) == 0:
        return
    if len(slot[].scalars) < 3:
        raise Error("member ABI: a fitted FactorAnalysis state needs its scalars block")
    var nc = Int(slot[].scalars[0])
    var d = Int(slot[].scalars[1])
    if (nc < 1 or d < 1 or len(slot[].blocks[0]) != nc * d or len(slot[].blocks[1]) != d
            or len(slot[].blocks[2]) != d):
        raise Error("member ABI: a fitted FactorAnalysis state needs every fitted block, of the fitted sizes")
    slot[].count = nc
    slot[].features = d
    slot[].fitted = True


# ---------------------------------------------------------------- the dictionary-learning kit
# DictionaryLearning (9), MiniBatchDictionaryLearning (10), SparseCoder (11),
# SparsePCA (12), MiniBatchSparsePCA (13): the door's typed kit RUN
# (bindings/dict_learning_run.mojo over bindings/decomp_kit_run.mojo `DKit`),
# the kit built from the route words the admit read from the image's entries
# (`decomp_kit_unpack`). Fitted operands stay on the device across queries
# as the door's resident record keeps them (pool ids in `handles`: 0
# components_, 1 mean_, 2 the Gram of components_), freed at refit, import
# and free through the same entry.

comptime _KIT_GRAM = 2


def _kit(call: _Call) -> DKit:
    """`decomp_kit_unpack(entries)` from the route words: the image's x_decomp
    entry (resolved at admit), its residency and its DK_HAS_* bits."""
    var flags = call[].route.flags
    return DKit(call[].route.entry, True, (flags & XDECOMP_ROUTE_KIT_RESIDENT) != 0, flags >> XDECOMP_ROUTE_KIT_SHIFT)


def _four(a: Int, b: Int, c: Int, d: Int) -> List[Float64]:
    var words = List[Float64](capacity=4)
    words.append(Float64(a))
    words.append(Float64(b))
    words.append(Float64(c))
    words.append(Float64(d))
    return words^


def _kit_copy(mut kit: DKit, a: Int, dst: Int) raises:
    """Matrix a's words into `dst`: a device-only result comes down once,
    straight into it (the door's `decomp_kit_out_shape`), else one copy."""
    var n = kit.size(a)
    if n < 1:
        return
    if kit.matrices[a].device >= 0 and kit.matrices[a].store < 0 and kit.matrices[a].borrowed == 0:
        kit.dev_download(kit.matrices[a].device, dst, n)
        kit.matrices[a].on_device = False
    else:
        unsafe_memcpy(dest=_F32P(unsafe_from_address=dst), src=_F32P(unsafe_from_address=kit.at(a)), count=n)


def _kit_take(mut kit: DKit, a: Int) raises -> List[Float32]:
    var words = List[Float32](length=kit.size(a), fill=Float32(0))
    if len(words) > 0:
        _kit_copy(kit, a, Int(words.unsafe_ptr()))
    return words^


def _kit_out(call: _Call, mut kit: DKit, a: Int, estimator: String) raises:
    _dense(call[].dst, "the output buffer", estimator)
    if call[].dst.rows != kit.nrows(a) or call[].dst.cols != kit.ncols(a):
        raise Error("ValueError: the output buffer holds " + String(kit.ncols(a)) + " values per row of X")
    _kit_copy(kit, a, call[].dst.data)


def _kit_operand(slot: _Slot, call: _Call, mut kit: DKit, k: Int, rows: Int, cols: Int) raises -> Int:
    """Fitted block k: its device copy kept in the pool (uploaded at the
    first query, the door's `decomp_door_resident_input`) on a resident
    image, else the host words borrowed."""
    if kit.resident:
        var handle = _rp_pool(slot, call, k, rows * cols)
        if handle >= 0:
            return kit.add_resident(_address(slot, k), handle, rows, cols)
    return kit.borrow(_address(slot, k), rows, cols)


def _kit_gram(slot: _Slot, mut kit: DKit, wanted: Bool) -> Gram:
    # The door's `_gram`: the kept device Gram of components_, else -1.
    if not wanted or slot[].handles[_KIT_GRAM] < 0:
        return Gram(-1)
    return Gram(kit.add_device(slot[].handles[_KIT_GRAM], slot[].count, slot[].count))


def _kit_keep_gram(slot: _Slot, call: _Call, mut kit: DKit, gram: Gram):
    # The door's `_keep_gram`: a Gram RUN left on the device is kept.
    if gram.computed < 0 or slot[].handles[_KIT_GRAM] >= 0 or not kit.resident:
        return
    if (call[].route.flags & XDECOMP_ROUTE_POOL) == 0:
        return
    var handle = kit.release_device(gram.computed)
    if handle >= 0:
        slot[].handles[_KIT_GRAM] = handle
        slot[].handle_entry = call[].route.entry


def _encode(slot: _Slot, m: Int, kc: Int) -> EncodeWords:
    """bindings/dict_learning_native_unpack.mojo `_encode_words` from the
    coding words [1 .. 7]; the lars / omp budget defaults to
    int(min(max(m / 10, 1), kc)) for the input width m and kc atoms."""
    var index = Int(slot[].words[1])
    var count = 0
    if index == 2 or index == 3:
        if slot[].words[2] >= 0.0:
            count = Int(slot[].words[2])
        else:
            count = Int(min(max(Float64(m) / 10.0, Float64(1.0)), Float64(kc)))
    return EncodeWords(index, count, slot[].words[3], Int(slot[].words[4]), Int(slot[].words[5]),
                       slot[].words[6] != 0.0, slot[].words[7] != 0.0)


def _fit_kit_begin(slot: _Slot, call: _Call, estimator: String) raises:
    _whole(call, estimator)
    _dense(call[].x, "X", estimator)
    if call[].weights.rows > 0:
        raise Error("TypeError: a " + estimator + " member takes no sample_weight")
    if call[].fit_output != 0 and call[].fit_output != MEMBER_OP_TRANSFORM:
        member_unsupported(estimator, call[].fit_output)
    # New components replace the old: their device copies go first.
    _release(slot)
    slot[].clear()


def _kit_keep(slot: _Slot, mut kit: DKit, comps: Int, mean: Int, var scalars: List[Float64]) raises:
    slot[].blocks.append(_kit_take(kit, comps))
    if mean >= 0:
        slot[].blocks.append(_kit_take(kit, mean))
    slot[].count = kit.nrows(comps)
    slot[].features = kit.ncols(comps)
    slot[].scalars = scalars^
    slot[].fitted = True


def _fit_dl(slot: _Slot, call: _Call) raises:
    """bindings/dict_learning_native_pack.mojo `dictionary_learning_fit` /
    `minibatch_dictionary_fit` over the borrowed X (no code_init, dict_init
    or callback: those settings keep the door). fit_output TRANSFORM: the
    DictionaryLearning fit's own code (split by sign when asked), the
    mini-batch fit's encode of X, as the doors' fit_transform."""
    var estimator = _name(slot)
    _fit_kit_begin(slot, call, estimator)
    var kind = slot[].kind()
    var project = call[].fit_output == MEMBER_OP_TRANSFORM
    var n = call[].x.rows
    var m = call[].x.cols
    var kit = _kit(call)
    try:
        var x = kit.borrow(call[].x.data, n, m)
        var nc = m if slot[].words[8] < 0.0 else Int(slot[].words[8])
        var d = -1
        var code = -1
        var scalars = List[Float64]()
        if kind == XDECOMP_KIND_DL:
            var words = DictWords(nc, slot[].words[9], Int(slot[].words[10]), slot[].words[11], Int(slot[].words[12]),
                                  Int(slot[].words[13]), slot[].words[14] != 0.0, slot[].words[15] != 0.0,
                                  Int(slot[].words[16]))
            var fit = dl_dict_learning(kit, x, words, -1, -1)
            d = fit.d
            if project:
                code = dl_split_sign(kit, fit.code) if slot[].words[7] != 0.0 else fit.code
            scalars = _four(kit.nrows(d), m, fit.done, 0)
            for k in range(len(fit.errors)):  # driver-loop: over the fit's recorded errors (model structure)
                scalars.append(fit.errors[k])
        else:
            var batch = min(Int(slot[].words[9]), n)
            if batch <= 0:
                raise Error("integer division or modulo by zero")
            var per_pass = -((-n) // batch)
            var mb = MiniBatchWords(batch, Int(slot[].words[10]), slot[].words[11] != 0.0,
                                    Int(slot[].words[12] * Float64(per_pass)), Int(slot[].words[13]),
                                    Int(slot[].words[14]), Int(slot[].words[15]), slot[].words[16] != 0.0,
                                    slot[].words[17] != 0.0, slot[].words[18], slot[].words[19])
            var fit = dl_minibatch_fit(kit, x, nc, -1, mb)
            d = fit[0]
            var steps = fit[1]
            if project:
                var gram = Gram(-1)
                code = dl_encode_run(kit, x, d, _encode(slot, m, nc), gram)
            scalars = _four(kit.nrows(d), m, -((-steps) // per_pass), steps)
        _kit_keep(slot, kit, d, -1, scalars^)
        if project:
            _kit_out(call, kit, code, estimator)
    finally:
        kit.close()


def _fit_coder(slot: _Slot, call: _Call) raises:
    """bindings/dict_learning_native_pack.mojo `sparse_coder` fit: the
    dictionary (the admitted float32 parameter the plan pins, words [8 ..
    10]) checked against X's width and kept as components_."""
    var estimator = _name(slot)
    _fit_kit_begin(slot, call, estimator)
    var rows = Int(slot[].words[9])
    var cols = Int(slot[].words[10])
    if call[].x.cols != cols:
        raise Error("Dictionary and X have different numbers of features:dictionary.shape: (" + String(rows) + ", "
                    + String(cols) + ") X.shape: (" + String(call[].x.rows) + ", " + String(call[].x.cols) + ")")
    var words = List[Float32](length=rows * cols, fill=Float32(0))
    if rows * cols > 0:
        unsafe_memcpy(dest=words.unsafe_ptr(), src=_F32P(unsafe_from_address=Int(slot[].words[8])), count=rows * cols)
    slot[].blocks.append(words^)
    slot[].count = rows
    slot[].features = cols
    slot[].scalars = _four(rows, cols, 0, 0)
    slot[].fitted = True
    if call[].fit_output == MEMBER_OP_TRANSFORM:
        _query_kit(slot, call, False)


def _fit_spca(slot: _Slot, call: _Call) raises:
    """bindings/dict_learning_native_pack.mojo `sparse_pca_fit` over the
    borrowed X (no V_init / U_init: those keep the door); fit_output
    TRANSFORM = its fit_transform's ridge projection."""
    var estimator = _name(slot)
    _fit_kit_begin(slot, call, estimator)
    var minibatch = slot[].kind() == XDECOMP_KIND_MBSPCA
    var project = call[].fit_output == MEMBER_OP_TRANSFORM
    var n = call[].x.rows
    var m = call[].x.cols
    var kit = _kit(call)
    try:
        var x = kit.borrow(call[].x.data, n, m)
        var nc = m if slot[].words[1] < 0.0 else Int(slot[].words[1])
        var dict_words = DictWords(nc, 0.0, 0, 0.0, 0, 0, False, False, 0)
        var mb = MiniBatchWords(1, 0, False, 0, -1, 0, 0, False, False, 0.0, 0.0)
        var encode = EncodeWords(0, 0, 1.0, 0, 0, False, False)
        var per_pass = 1
        var ridge = Float32(0.0)
        if not minibatch:
            dict_words = DictWords(nc, slot[].words[2], Int(slot[].words[3]), slot[].words[4], Int(slot[].words[5]),
                                   Int(slot[].words[6]), False, False, 1000)
            if project:
                ridge = Float32(slot[].words[7])
        else:
            # The fit runs on X^T: its rows are the m features.
            var batch = min(Int(slot[].words[3]), m)
            if batch <= 0:
                raise Error("integer division or modulo by zero")
            per_pass = -((-m) // batch)
            mb = MiniBatchWords(batch, Int(slot[].words[4]), slot[].words[5] != 0.0,
                                Int(slot[].words[6] * Float64(per_pass)), Int(slot[].words[7]), Int(slot[].words[8]),
                                1000, False, False, slot[].words[2], slot[].words[9])
            encode = EncodeWords(0, 0, slot[].words[2], 1000, 0, False, False)
            if project:
                ridge = Float32(slot[].words[10])
        var fit = dl_spca_fit_run(kit, x, -1, -1, minibatch, dict_words, mb, encode, project, ridge)
        var iterations = -((-fit.steps) // per_pass) if minibatch else fit.done
        var scalars = _four(kit.nrows(fit.comps), m, iterations, fit.steps)
        for k in range(len(fit.errors)):  # driver-loop: over the fit's recorded errors (model structure)
            scalars.append(fit.errors[k])
        _kit_keep(slot, kit, fit.comps, fit.mean, scalars^)
        if project:
            _kit_out(call, kit, fit.projected, estimator)
    finally:
        kit.close()


def _query_kit(slot: _Slot, call: _Call, inverse: Bool) raises:
    """The doors' transform / inverse_transform (`coding_transform`,
    `sparse_coder`, `sparse_pca_transform`) over the kept operands."""
    var estimator = _name(slot)
    if not slot[].fitted:
        raise Error("NotFittedError: this " + estimator + " member is not fitted yet")
    _whole(call, estimator)
    _dense(call[].x, "X", estimator)
    var kind = slot[].kind()
    var kit = _kit(call)
    try:
        var x = kit.borrow(call[].x.data, call[].x.rows, call[].x.cols)
        var comps = _kit_operand(slot, call, kit, 0, slot[].count, slot[].features)
        var out = -1
        if _is_spca(kind):
            var mean = _kit_operand(slot, call, kit, 1, 1, slot[].features)
            if inverse:
                out = dl_spca_inverse(kit, x, mean, comps)
            else:
                var ridge = Float32(slot[].words[7] if kind == XDECOMP_KIND_SPCA else slot[].words[10])
                var gram = _kit_gram(slot, kit, True)
                out = dl_spca_project(kit, x, mean, comps, ridge, gram)
                _kit_keep_gram(slot, call, kit, gram)
        elif inverse:
            out = dl_inverse_run(kit, x, comps, slot[].words[7] != 0.0)
        else:
            var words = _encode(slot, call[].x.cols, slot[].count)
            var gram = _kit_gram(slot, kit, words.algorithm != 4)
            out = dl_encode_run(kit, x, comps, words, gram)
            _kit_keep_gram(slot, call, kit, gram)
        _kit_out(call, kit, out, estimator)
    finally:
        kit.close()


def _export_kit(slot: _Slot, call: _Call) raises:
    var spca = _is_spca(slot[].kind())
    var last = 2 if spca else 1
    var block = call[].block
    if (call[].flags & MEMBER_BLOCK_BY_INDEX) != 0:
        if block < 0 or block > last:
            raise Error("member ABI: a " + _name(slot) + " member has no fitted block index " + String(block))
        block = MEMBER_BLOCK_SCALARS if block == 0 else XDECOMP_BLOCK_COMPONENTS + block - 1
    if block == MEMBER_BLOCK_SCALARS:
        call[].dst = MemberMatrix(Int(slot[].scalars.unsafe_ptr()), 1, len(slot[].scalars), MEMBER_F64)
    elif block == XDECOMP_BLOCK_COMPONENTS:
        call[].dst = MemberMatrix(_address(slot, 0), slot[].count, slot[].features)
    elif spca and block == XDECOMP_BLOCK_COMPONENTS + 1:
        call[].dst = MemberMatrix(_address(slot, 1), 1, slot[].features)
    else:
        raise Error("member ABI: a " + _name(slot) + " member has no fitted block " + String(block))
    call[].block = block
    call[].out_cols = last + 1


def _import_kit(slot: _Slot, call: _Call) raises:
    var estimator = _name(slot)
    var parts = 2 if _is_spca(slot[].kind()) else 1
    var block = call[].block
    if slot[].fitted or len(slot[].blocks) != parts:
        slot[].clear()
        for _ in range(parts):  # driver-loop: the fitted blocks
            slot[].blocks.append(List[Float32]())
    if block == MEMBER_BLOCK_SCALARS:
        var count = call[].x.rows * call[].x.cols
        if not call[].x.dense_host(MEMBER_F64) or count < 4:
            raise Error("member ABI: the " + estimator + " scalars block is four or more host float64 words")
        var source = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call[].x.data)
        slot[].scalars = List[Float64](length=count, fill=0.0)
        for index in range(count):  # driver-loop: the fitted scalars and errors
            slot[].scalars[index] = source[index]
    elif block >= XDECOMP_BLOCK_COMPONENTS and block < XDECOMP_BLOCK_COMPONENTS + parts:
        var size = call[].x.rows * call[].x.cols
        var words = List[Float32](length=size, fill=0.0)
        if size > 0:
            _dense(call[].x, "a fitted block", estimator)
            unsafe_memcpy(dest=words.unsafe_ptr(), src=_F32P(unsafe_from_address=call[].x.data), count=size)
        slot[].blocks[block - XDECOMP_BLOCK_COMPONENTS] = words^
    elif block != 0 or (call[].flags & MEMBER_IMPORT_DONE) == 0:
        raise Error("member ABI: a " + estimator + " member has no fitted block " + String(block))
    if (call[].flags & MEMBER_IMPORT_DONE) == 0:
        return
    if len(slot[].scalars) < 4:
        raise Error("member ABI: a fitted " + estimator + " state needs its scalars block")
    var nc = Int(slot[].scalars[0])
    var d = Int(slot[].scalars[1])
    if nc < 1 or d < 1 or len(slot[].blocks[0]) != nc * d or (parts == 2 and len(slot[].blocks[1]) != d):
        raise Error("member ABI: a fitted " + estimator + " state needs every fitted block, of the fitted sizes")
    slot[].count = nc
    slot[].features = d
    slot[].fitted = True


def _kit_width(words: MutPointer[Float64, MutUntrackedOrigin], asked: Int, features: Int,
               fitted: Int) raises -> Int:
    """transform: the atoms (n_components, n_features when None; the
    dictionary's rows for SparseCoder), twice that with split_sign;
    inverse_transform: n_features."""
    var kind = Int(words[0])
    if asked == MEMBER_OP_INVERSE_TRANSFORM:
        return features
    if asked != MEMBER_OP_TRANSFORM:
        member_unsupported(_kind_name(kind), asked)
    if _is_spca(kind):
        if fitted >= 0:
            return fitted
        return Int(words[1]) if words[1] >= 0.0 else features
    var atoms = fitted
    if atoms < 0:
        if kind == XDECOMP_KIND_CODER:
            atoms = Int(words[9])
        else:
            atoms = Int(words[8]) if words[8] >= 0.0 else features
    return 2 * atoms if words[7] != 0.0 else atoms


# ---------------------------------------------------------------- EllipticEnvelope
# Kind 14: bindings/mcd_native_door.mojo `mcd_fit` / `mcd_operation` over
# the kit (bindings/mcd_run.mojo). Blocks after the scalars [n_features_in_,
# training rows, offset_]: 1 location_ (1 x p), 2 _precision (p x p), 3
# raw_location_, 4 raw_covariance_, 5 covariance_, 6 dist_ (1 x n), 7
# raw_support_, 8 support_ (1 x n, 0 / 1 words). An imported state needs
# only 1 and 2 (what the queries read). location_ and _precision stay on the
# device across queries (pool ids handles[0], [1]), as the door's record.

comptime _MCD_BLOCKS = 8


def _u8_block(support: List[UInt8], n: Int) -> List[Float32]:
    var words = List[Float32](length=n, fill=Float32(0))
    if n > 0:
        decomp_u8_to_f32(MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(support.unsafe_ptr())),
                         _F32P(unsafe_from_address=Int(words.unsafe_ptr())), n)
    return words^


def _mcd_labels(call: _Call, mut kit: DKit, chosen: Int, rows: Int, estimator: String) raises:
    # The door's `_labels_pack`: the +1 / -1 float32 words as int32 labels.
    if not call[].dst.dense_host(MEMBER_I32) or call[].dst.rows != rows or call[].dst.cols != 1:
        raise Error("TypeError: a " + estimator + " member writes predict into a dense host int32 view, one per row")
    if chosen >= 0:
        decomp_f32_to_i32(kit.ptr(chosen), MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=call[].dst.data),
                          rows)


def _fit_mcd(slot: _Slot, call: _Call) raises:
    """`mcd_fit` (envelope) over the borrowed X; fit_output PREDICT =
    fit_predict's labels over the fit's own distances."""
    var estimator = _name(slot)
    _whole(call, estimator)
    _dense(call[].x, "X", estimator)
    if call[].weights.rows > 0:
        raise Error("TypeError: a " + estimator + " member takes no sample_weight")
    if call[].fit_output != 0 and call[].fit_output != MEMBER_OP_PREDICT:
        member_unsupported(estimator, call[].fit_output)
    _release(slot)
    slot[].clear()
    var n = call[].x.rows
    var p = call[].x.cols
    if n < 2:
        raise Error("Found array with " + String(n) + " sample(s) (shape=(" + String(n) + ", " + String(p)
                    + ")) while a minimum of 2 is required by MinCovDet.")
    var h = min((n + p + 2) // 2, n) if slot[].words[1] < 0.0 else Int(slot[].words[1] * Float64(n))
    var kit = _kit(call)
    try:
        var x = kit.borrow(call[].x.data, n, p)
        var fit = mcd_fit_run(kit, x, h, Int(slot[].words[2]), slot[].words[3] != 0.0, True, slot[].words[4])
        slot[].blocks.append(_kit_take(kit, fit.loc))
        slot[].blocks.append(_kit_take(kit, fit.precision))
        slot[].blocks.append(_kit_take(kit, fit.raw_loc))
        slot[].blocks.append(_kit_take(kit, fit.raw_cov))
        slot[].blocks.append(_kit_take(kit, fit.cov))
        slot[].blocks.append(_kit_take(kit, fit.dist))
        slot[].blocks.append(_u8_block(fit.raw_support, n))
        slot[].blocks.append(_u8_block(fit.support, n))
        var scalars = List[Float64](capacity=3)
        scalars.append(Float64(p))
        scalars.append(Float64(n))
        scalars.append(fit.offset)
        slot[].scalars = scalars^
        slot[].count = n
        slot[].features = p
        slot[].fitted = True
        if call[].fit_output == MEMBER_OP_PREDICT:
            _mcd_labels(call, kit, mcd_labels_run(kit, fit.dist, fit.offset), n, estimator)
    finally:
        kit.close()


def _query_mcd(slot: _Slot, call: _Call, operation: Int) raises:
    """`mcd_operation`: predict (int32 +1 / -1), decision_function
    (-dist - offset_) or score_samples (-dist), one value per row."""
    var estimator = _name(slot)
    if not slot[].fitted:
        raise Error("NotFittedError: this " + estimator + " member is not fitted yet")
    _whole(call, estimator)
    _dense(call[].x, "X", estimator)
    var p = slot[].features
    var offset = slot[].scalars[2]
    var kit = _kit(call)
    try:
        var x = kit.borrow(call[].x.data, call[].x.rows, call[].x.cols)
        var loc = _kit_operand(slot, call, kit, 0, 1, p)
        var precision = _kit_operand(slot, call, kit, 1, p, p)
        var d = mcd_mahal(kit, x, loc, precision)
        if operation == MEMBER_OP_PREDICT:
            _mcd_labels(call, kit, mcd_labels_run(kit, d, offset), kit.nrows(d), estimator)
            return
        var negated = kit.ew(DK_SCALE, d, scalar=-1.0)
        if operation == MEMBER_OP_DECISION:
            _kit_out(call, kit, kit.ew(DK_ADDS, negated, scalar=-offset), estimator)
        else:
            _kit_out(call, kit, negated, estimator)
    finally:
        kit.close()


def _export_mcd(slot: _Slot, call: _Call) raises:
    var block = call[].block
    if (call[].flags & MEMBER_BLOCK_BY_INDEX) != 0:
        if block < 0 or block > _MCD_BLOCKS:
            raise Error("member ABI: an EllipticEnvelope member has no fitted block index " + String(block))
        block = MEMBER_BLOCK_SCALARS if block == 0 else XDECOMP_BLOCK_COMPONENTS + block - 1
    var p = slot[].features
    var k = block - XDECOMP_BLOCK_COMPONENTS
    if block == MEMBER_BLOCK_SCALARS:
        call[].dst = MemberMatrix(Int(slot[].scalars.unsafe_ptr()), 1, len(slot[].scalars), MEMBER_F64)
    elif k >= 0 and k < len(slot[].blocks):
        var size = len(slot[].blocks[k])
        var rows = p if (k == 1 or k == 3 or k == 4) else 1
        call[].dst = MemberMatrix(_address(slot, k), rows, size // max(rows, 1))
    else:
        raise Error("member ABI: an EllipticEnvelope member has no fitted block " + String(block))
    call[].block = block
    call[].out_cols = _MCD_BLOCKS + 1


def _import_mcd(slot: _Slot, call: _Call) raises:
    var estimator = _name(slot)
    var block = call[].block
    if slot[].fitted or len(slot[].blocks) != _MCD_BLOCKS:
        slot[].clear()
        for _ in range(_MCD_BLOCKS):  # driver-loop: the fitted blocks
            slot[].blocks.append(List[Float32]())
    var k = block - XDECOMP_BLOCK_COMPONENTS
    if block == MEMBER_BLOCK_SCALARS:
        if not call[].x.dense_host(MEMBER_F64) or call[].x.rows * call[].x.cols != 3:
            raise Error("member ABI: the EllipticEnvelope scalars block is three host float64 words")
        var source = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call[].x.data)
        slot[].scalars = List[Float64](length=3, fill=0.0)
        for index in range(3):  # driver-loop: three fitted scalars
            slot[].scalars[index] = source[index]
    elif k >= 0 and k < _MCD_BLOCKS:
        var size = call[].x.rows * call[].x.cols
        var words = List[Float32](length=size, fill=0.0)
        if size > 0:
            _dense(call[].x, "a fitted block", estimator)
            unsafe_memcpy(dest=words.unsafe_ptr(), src=_F32P(unsafe_from_address=call[].x.data), count=size)
        slot[].blocks[k] = words^
    elif block != 0 or (call[].flags & MEMBER_IMPORT_DONE) == 0:
        raise Error("member ABI: an EllipticEnvelope member has no fitted block " + String(block))
    if (call[].flags & MEMBER_IMPORT_DONE) == 0:
        return
    if len(slot[].scalars) != 3:
        raise Error("member ABI: a fitted EllipticEnvelope state needs its scalars block")
    var p = Int(slot[].scalars[0])
    if p < 1 or len(slot[].blocks[0]) != p or len(slot[].blocks[1]) != p * p:
        raise Error("member ABI: a fitted EllipticEnvelope state needs location_ and _precision, of the fitted sizes")
    slot[].count = Int(slot[].scalars[1])
    slot[].features = p
    slot[].fitted = True


def _mcd_width(asked: Int, mut dtype: Int) raises -> Int:
    # predict: one int32 label per row; decision_function and
    # score_samples: one float32 value per row.
    if asked == MEMBER_OP_PREDICT:
        dtype = MEMBER_I32
        return 1
    if asked == MEMBER_OP_DECISION or asked == MEMBER_OP_SCORE_SAMPLES:
        return 1
    member_unsupported("EllipticEnvelope", asked)
    return 0


# ---------------------------------------------------------------- random projections


def _jl_bound(e: Float64, n: Float64) -> Float64:
    # bindings/jl_native_pack.mojo `_jl_bound`: the same binary64 operations
    # (eps^3 correctly rounded, the pinned log), so the same integer.
    var denominator = (e * e) / 2.0 - _cube_rounded(e) / 3.0
    return 4.0 * _pinned_log(n) / denominator


def _fit_rp(slot: _Slot, call: _Call) raises:
    """bindings/rp_native_door.mojo `rp_fit`: the target dimension, the
    density, one RPFitRequest through the route's operations table."""
    var estimator = _name(slot)
    _whole(call, estimator)
    _dense(call[].x, "X", estimator)
    if call[].weights.rows > 0:
        raise Error("TypeError: a " + estimator + " member takes no sample_weight")
    if call[].fit_output != 0 and call[].fit_output != MEMBER_OP_TRANSFORM:
        member_unsupported(estimator, call[].fit_output)
    var n = call[].x.rows
    var d = call[].x.cols
    var sparse = slot[].kind() == XDECOMP_KIND_SRP
    var auto = slot[].words[1] < 0.0
    var kc = 0
    if auto:
        var eps = slot[].words[2]
        var bound = _jl_bound(eps, Float64(n))
        # The door compares the published (truncated) integer.
        var too_large = not (bound < 9.0e18)
        var whole = 0 if too_large else Int(bound)
        var shown = String(bound) if too_large else String(whole)
        if not too_large and whole <= 0:
            raise Error("eps=" + String(eps) + " and n_samples=" + String(n) + " lead to a target dimension of "
                        + shown + " which is invalid")
        if too_large or d < whole:
            raise Error("eps=" + String(eps) + " and n_samples=" + String(n) + " lead to a target dimension of "
                        + shown + " which is larger than the original space with n_features=" + String(d))
        kc = whole
    else:
        kc = Int(slot[].words[1])
    var density = Float64(0.0)
    if sparse:
        density = 1.0 / sqrt(Float64(d)) if slot[].words[3] < 0.0 else slot[].words[3]
        if not (density > 0 and density <= 1):
            raise Error("Expected density in range ]0, 1], got: " + String(density))
    var inverse = slot[].words[4] != 0.0
    _release(slot)
    slot[].clear()
    slot[].blocks.append(List[Float32](length=max(kc * d, 1), fill=Float32(0)))
    slot[].blocks.append(List[Float32](length=(d * kc) if inverse else 0, fill=Float32(0)))
    var components = _address(slot, 0)
    var request = RPFitRequest(call[].x.data, components, _address(slot, 1) if inverse else components, n, d, kc,
        sparse, Int(slot[].words[5]), inverse, not auto and not inverse, density)
    var bad = rp_fit_call(call[].route.entry, request)
    if bad >= 0:
        slot[].clear()
        raise Error("X: input must be finite; NaN/inf are unsupported")
    slot[].count = kc
    slot[].features = d
    slot[].scalars = [Float64(kc), Float64(d), density, 1.0 if inverse else 0.0]
    slot[].fitted = True
    if call[].fit_output == MEMBER_OP_TRANSFORM:
        _query_rp(slot, call, False)


def _rp_pool(slot: _Slot, call: _Call, k: Int, words: Int) raises -> Int:
    """The pool id of fitted block `k` on the route's device: kept, else
    uploaded once (the door's `decomp_resident_operand`), else -1 (a host
    binding: the projection reads the host words)."""
    if slot[].handles[k] >= 0 or (call[].route.flags & XDECOMP_ROUTE_POOL) == 0 or words < 1:
        return slot[].handles[k]
    var entry = call[].route.entry
    var sizes: List[Int] = [words]
    var handle = decomp_table_call(entry, XD_DEV_ALLOC, List[Int](), sizes, List[Float64]())
    try:
        var source: List[Int] = [_address(slot, k)]
        var upload: List[Int] = [handle, words]
        _ = decomp_table_call(entry, XD_DEV_UPLOAD, source, upload, List[Float64]())
    except error:
        var ids: List[Int] = [handle]
        _ = decomp_table_call(entry, XD_DEV_FREE, List[Int](), ids, List[Float64]())
        raise error^
    # One state is driven through one route (its plan's): the entry that
    # uploads is the entry that frees.
    slot[].handles[k] = handle
    slot[].handle_entry = entry
    return handle


def _query_rp(slot: _Slot, call: _Call, inverse: Bool) raises:
    """`rp_project`: transform (X components_^T, the tiled projection of an
    admitted X) or inverse_transform (X inverse_components_^T)."""
    var estimator = _name(slot)
    if not slot[].fitted:
        raise Error("NotFittedError: this " + estimator + " member is not fitted yet")
    _whole(call, estimator)
    _dense(call[].x, "X", estimator)
    _dense(call[].dst, "the output buffer", estimator)
    var rows = call[].x.rows
    var kc = slot[].count
    var d = slot[].features
    if inverse:
        if slot[].scalars[3] == 0.0:
            raise Error("inverse_transform needs compute_inverse_components=True")
        decomp_gemm_check(call[].x.cols, kc)
    elif call[].x.cols != d:
        raise Error("X has " + String(call[].x.cols) + " features, but " + estimator + " is expecting " + String(d))
    var width = d if inverse else kc
    if call[].dst.rows != rows or call[].dst.cols != width:
        raise Error("ValueError: the output buffer holds " + String(width) + " values per row of X")
    var k = 1 if inverse else 0
    var pool = _rp_pool(slot, call, k, kc * d)
    var request = RPProjectRequest(call[].x.data, _address(slot, k), call[].dst.data, rows, call[].x.cols, width,
        0 if inverse else 1, pool)
    if rp_project_call(call[].route.entry, request) != 0:
        raise Error("X: input must be finite; NaN/inf are unsupported")


def _export_rp(slot: _Slot, call: _Call) raises:
    var block = call[].block
    if (call[].flags & MEMBER_BLOCK_BY_INDEX) != 0:
        if block < 0 or block > 2:
            raise Error("member ABI: a random projection member has no fitted block index " + String(block))
        block = MEMBER_BLOCK_SCALARS if block == 0 else XDECOMP_BLOCK_COMPONENTS + block - 1
    if block == MEMBER_BLOCK_SCALARS:
        call[].dst = MemberMatrix(Int(slot[].scalars.unsafe_ptr()), 1, len(slot[].scalars), MEMBER_F64)
    elif block == XDECOMP_BLOCK_COMPONENTS:
        call[].dst = MemberMatrix(_address(slot, 0), slot[].count, slot[].features)
    elif block == XDECOMP_BLOCK_RP_INVERSE and slot[].scalars[3] != 0.0:
        call[].dst = MemberMatrix(_address(slot, 1), slot[].features, slot[].count)
    else:
        raise Error("member ABI: a random projection member has no fitted block " + String(block))
    call[].block = block
    call[].out_cols = 3


def _import_rp(slot: _Slot, call: _Call) raises:
    var estimator = _name(slot)
    var block = call[].block
    if slot[].fitted or len(slot[].blocks) != 2:
        slot[].clear()
        slot[].blocks.append(List[Float32]())
        slot[].blocks.append(List[Float32]())
    if block == MEMBER_BLOCK_SCALARS:
        if not call[].x.dense_host(MEMBER_F64) or call[].x.rows * call[].x.cols != _RP_SCALARS:
            raise Error("member ABI: the " + estimator + " scalars block is four host float64 words")
        var source = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call[].x.data)
        slot[].scalars = List[Float64](length=_RP_SCALARS, fill=0.0)
        for index in range(_RP_SCALARS):  # driver-loop: four fitted scalars
            slot[].scalars[index] = source[index]
    elif block == XDECOMP_BLOCK_COMPONENTS or block == XDECOMP_BLOCK_RP_INVERSE:
        var size = call[].x.rows * call[].x.cols
        var words = List[Float32](length=size, fill=0.0)
        if size > 0:
            _dense(call[].x, "a fitted block", estimator)
            unsafe_memcpy(dest=words.unsafe_ptr(), src=_F32P(unsafe_from_address=call[].x.data), count=size)
        slot[].blocks[block - XDECOMP_BLOCK_COMPONENTS] = words^
    elif block != 0 or (call[].flags & MEMBER_IMPORT_DONE) == 0:
        raise Error("member ABI: a " + estimator + " member has no fitted block " + String(block))
    if (call[].flags & MEMBER_IMPORT_DONE) == 0:
        return
    if len(slot[].scalars) != _RP_SCALARS:
        raise Error("member ABI: a fitted " + estimator + " state needs its scalars block")
    var kc = Int(slot[].scalars[0])
    var d = Int(slot[].scalars[1])
    var inverse_size = len(slot[].blocks[1])
    if (kc < 1 or d < 1 or len(slot[].blocks[0]) != kc * d
            or (slot[].scalars[3] != 0.0 and inverse_size != d * kc)
            or (slot[].scalars[3] == 0.0 and inverse_size != 0)):
        raise Error("member ABI: a fitted " + estimator + " state needs every fitted block, of the fitted sizes")
    slot[].count = kc
    slot[].features = d
    slot[].fitted = True


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
    if slot[].kind() == XDECOMP_KIND_NMF:
        _export_nmf(slot, call)
        return
    if _is_rp(slot[].kind()):
        _export_rp(slot, call)
        return
    if slot[].kind() == XDECOMP_KIND_ICA:
        _export_ica(slot, call)
        return
    if slot[].kind() == XDECOMP_KIND_FA:
        _export_fa(slot, call)
        return
    if _is_kit(slot[].kind()):
        _export_kit(slot, call)
        return
    if slot[].kind() == XDECOMP_KIND_ENVELOPE:
        _export_mcd(slot, call)
        return
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
    if slot[].kind() == XDECOMP_KIND_NMF:
        _import_nmf(slot, call)
        return
    if _is_rp(slot[].kind()):
        _import_rp(slot, call)
        return
    if slot[].kind() == XDECOMP_KIND_ICA:
        _import_ica(slot, call)
        return
    if slot[].kind() == XDECOMP_KIND_FA:
        _import_fa(slot, call)
        return
    if _is_kit(slot[].kind()):
        _import_kit(slot, call)
        return
    if slot[].kind() == XDECOMP_KIND_ENVELOPE:
        _import_mcd(slot, call)
        return
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


def _ica_width(requested: Float64, whiten: Float64, features: Int) raises -> Int:
    """FastICA's transform width before a fit: min(n_components, n_features)
    when whitening with an integer n_components (a fit on fewer rows than
    that is narrower: the fit refuses the buffer then); otherwise the width
    is min(rows, n_features), chosen by the fit."""
    if whiten != 0.0 and requested >= 0.0 and features > 0:
        return min(Int(requested), features)
    raise Error("NotFittedError: a FastICA member with this n_components or whiten knows its output width after its fit")


def _fa_width(asked: Int, requested: Float64, features: Int, fitted: Int) raises -> Int:
    """FactorAnalysis: transform n_components_ (n_features when n_components
    is falsy), score_samples one value per row; no inverse."""
    if asked == MEMBER_OP_SCORE_SAMPLES:
        return 1
    if asked != MEMBER_OP_TRANSFORM:
        member_unsupported("FactorAnalysis", asked)
    if fitted >= 0:
        return fitted
    return Int(requested) if requested >= 0.5 else features


def _outputs(call: _Call) raises:
    """`call.out_cols` of transform / inverse_transform. Before a fit the
    transform width is known only for an integer n_components."""
    call[].classes = 0
    # Every transform / inverse_transform of the family writes float32.
    call[].out_dtype = MEMBER_F32
    if call[].state == 0:
        # Asked before any state (`member_outputs`): from the parameter words.
        if call[].param_count < 2 or not call[].params:
            raise Error("member ABI: an XDecomp member's output width needs its parameter words")
        var words = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call[].params)
        var kind = Int(words[0])
        var asked_first = call[].flags
        if kind == XDECOMP_KIND_FA:
            call[].out_cols = _fa_width(asked_first, words[1], call[].x.cols, -1)
            return
        if _is_kit(kind):
            call[].out_cols = _kit_width(words, asked_first, call[].x.cols, -1)
            return
        if kind == XDECOMP_KIND_ENVELOPE:
            call[].out_cols = _mcd_width(asked_first, call[].out_dtype)
            return
        if asked_first == MEMBER_OP_INVERSE_TRANSFORM:
            call[].out_cols = call[].x.cols
            return
        if asked_first != MEMBER_OP_TRANSFORM:
            member_unsupported(_kind_name(kind), asked_first)
        if kind == XDECOMP_KIND_ICA:
            call[].out_cols = _ica_width(words[1], words[2], call[].x.cols)
            return
        if words[1] >= 0.0:
            call[].out_cols = Int(words[1])
            return
        if kind == XDECOMP_KIND_NMF:
            call[].out_cols = call[].x.cols
            return
        raise Error("NotFittedError: a " + _kind_name(kind) + " member with this n_components knows its output width "
                    + "after its fit")
    var slot = _Slot(unsafe_from_address=_lookup(call[].state, False))
    var asked = call[].flags
    if slot[].kind() == XDECOMP_KIND_FA:
        call[].out_cols = _fa_width(asked, slot[].words[1], call[].x.cols, slot[].count if slot[].fitted else -1)
        return
    if slot[].kind() == XDECOMP_KIND_ENVELOPE:
        call[].out_cols = _mcd_width(asked, call[].out_dtype)
        return
    if _is_kit(slot[].kind()):
        var held_words = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(slot[].words.unsafe_ptr()))
        call[].out_cols = _kit_width(held_words, asked, slot[].features if slot[].fitted else call[].x.cols,
                                     slot[].count if slot[].fitted else -1)
        return
    if asked != MEMBER_OP_TRANSFORM and asked != MEMBER_OP_INVERSE_TRANSFORM:
        member_unsupported(_name(slot), asked)
    if slot[].fitted:
        call[].out_cols = slot[].count if asked == MEMBER_OP_TRANSFORM else slot[].features
        return
    if asked == MEMBER_OP_INVERSE_TRANSFORM and slot[].kind() == XDECOMP_KIND_NMF:
        raise Error("NotFittedError: this NMF member's output width is known after its fit")
    if asked == MEMBER_OP_TRANSFORM and slot[].kind() == XDECOMP_KIND_ICA:
        call[].out_cols = _ica_width(slot[].words[1], slot[].words[2], call[].x.cols)
        return
    if asked == MEMBER_OP_TRANSFORM and slot[].words[1] >= 0.0:
        call[].out_cols = Int(slot[].words[1])
        return
    raise Error("NotFittedError: this " + _name(slot) + " member's output width is known after its fit")


def _xdecomp_member(operation: Int, address: Int) raises -> Int:
    var call = _Call(unsafe_from_address=address)
    if operation == MEMBER_OP_CAPABILITIES:
        var mask = MEMBER_CAN_FIT | MEMBER_CAN_TRANSFORM | MEMBER_CAN_EXPORT | MEMBER_CAN_IMPORT
        if call[].state == 0:
            # The family's union.
            return (mask | MEMBER_CAN_INVERSE_TRANSFORM | MEMBER_CAN_SCORE | MEMBER_CAN_SCORE_SAMPLES
                    | MEMBER_CAN_PREDICT | MEMBER_CAN_DECISION)
        # Kind-exact (`member_kind_capabilities`): an NMF state answers inverse
        # only when its route's image exports it (NMF_CAP_INVERSE at admit).
        # A random projection answers inverse only with the pseudo-inverse
        # (compute_inverse_components, or an imported inverse_components_).
        var held = _Slot(unsafe_from_address=_lookup(call[].state, False))
        var kind = held[].kind()
        if kind == XDECOMP_KIND_FA:
            return mask | MEMBER_CAN_SCORE | MEMBER_CAN_SCORE_SAMPLES
        if kind == XDECOMP_KIND_ENVELOPE:
            return (MEMBER_CAN_FIT | MEMBER_CAN_PREDICT | MEMBER_CAN_DECISION | MEMBER_CAN_SCORE_SAMPLES
                    | MEMBER_CAN_EXPORT | MEMBER_CAN_IMPORT)
        if _is_rp(kind):
            if held[].words[4] != 0.0 or (held[].fitted and held[].scalars[3] != 0.0):
                mask |= MEMBER_CAN_INVERSE_TRANSFORM
        elif kind != XDECOMP_KIND_NMF or (call[].route.flags & XDECOMP_ROUTE_NMF_INVERSE) != 0:
            mask |= MEMBER_CAN_INVERSE_TRANSFORM
        return mask
    if operation == MEMBER_OP_NEW:
        if call[].param_count < 1 or not call[].params:
            raise Error("member ABI: an XDecomp member takes its parameter words")
        var source = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call[].params)
        if source[0] == Float64(XDECOMP_KIND_IPCA):
            if call[].param_count != XDECOMP_IPCA_PARAM_COUNT:
                raise Error("member ABI: an IncrementalPCA member takes four parameters")
        elif source[0] == Float64(XDECOMP_KIND_NMF):
            if call[].param_count != XDECOMP_NMF_PARAM_COUNT:
                raise Error("member ABI: an NMF member takes twelve parameters")
        elif source[0] == Float64(XDECOMP_KIND_GRP) or source[0] == Float64(XDECOMP_KIND_SRP):
            if call[].param_count != XDECOMP_RP_PARAM_COUNT:
                raise Error("member ABI: a random projection member takes six parameters")
        elif source[0] == Float64(XDECOMP_KIND_ICA):
            if call[].param_count != XDECOMP_ICA_PARAM_COUNT:
                raise Error("member ABI: a FastICA member takes nine parameters")
        elif source[0] == Float64(XDECOMP_KIND_FA):
            if (call[].param_count < XDECOMP_FA_PARAM_COUNT
                    or call[].param_count != XDECOMP_FA_PARAM_COUNT + max(Int(source[5]), 0)):
                raise Error("member ABI: a FactorAnalysis member takes six parameters and its noise_variance_init")
        elif _is_kit(Int(source[0])) and source[0] == Float64(Int(source[0])):
            var kind = Int(source[0])
            var wanted = XDECOMP_DL_PARAM_COUNT
            if kind == XDECOMP_KIND_MBDL:
                wanted = XDECOMP_MBDL_PARAM_COUNT
            elif kind == XDECOMP_KIND_CODER:
                wanted = XDECOMP_CODER_PARAM_COUNT
            elif kind == XDECOMP_KIND_SPCA:
                wanted = XDECOMP_SPCA_PARAM_COUNT
            elif kind == XDECOMP_KIND_MBSPCA:
                wanted = XDECOMP_MBSPCA_PARAM_COUNT
            if call[].param_count != wanted:
                raise Error("member ABI: a " + _kind_name(kind) + " member takes " + String(wanted) + " parameters")
        elif source[0] == Float64(XDECOMP_KIND_ENVELOPE):
            if call[].param_count != XDECOMP_ENVELOPE_PARAM_COUNT:
                raise Error("member ABI: an EllipticEnvelope member takes six parameters")
        else:
            raise Error("member ABI: unknown or unpublished XDecomp member kind")
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
    var kind = slot[].kind()
    if operation == MEMBER_OP_FIT:
        if kind == XDECOMP_KIND_NMF:
            _fit_nmf(slot, call)
        elif _is_rp(kind):
            _fit_rp(slot, call)
        elif kind == XDECOMP_KIND_ICA:
            _fit_ica(slot, call)
        elif kind == XDECOMP_KIND_FA:
            _fit_fa(slot, call)
        elif kind == XDECOMP_KIND_DL or kind == XDECOMP_KIND_MBDL:
            _fit_dl(slot, call)
        elif kind == XDECOMP_KIND_CODER:
            _fit_coder(slot, call)
        elif _is_spca(kind):
            _fit_spca(slot, call)
        elif kind == XDECOMP_KIND_ENVELOPE:
            _fit_mcd(slot, call)
        else:
            _fit_ipca(slot, call)
    elif kind == XDECOMP_KIND_ENVELOPE and (operation == MEMBER_OP_PREDICT or operation == MEMBER_OP_DECISION
            or operation == MEMBER_OP_SCORE_SAMPLES):
        _query_mcd(slot, call, operation)
    elif kind == XDECOMP_KIND_ENVELOPE and (operation == MEMBER_OP_TRANSFORM
            or operation == MEMBER_OP_INVERSE_TRANSFORM):
        member_unsupported(_name(slot), operation)
    elif kind == XDECOMP_KIND_FA and operation == MEMBER_OP_TRANSFORM:
        _query_fa(slot, call, _FA_TRANSFORM)
    elif kind == XDECOMP_KIND_FA and operation == MEMBER_OP_SCORE_SAMPLES:
        _query_fa(slot, call, _FA_SCORE_SAMPLES)
    elif kind == XDECOMP_KIND_FA and operation == MEMBER_OP_SCORE:
        _query_fa(slot, call, _FA_SCORE)
    elif kind != XDECOMP_KIND_FA and (operation == MEMBER_OP_TRANSFORM or operation == MEMBER_OP_INVERSE_TRANSFORM):
        var inverse = operation == MEMBER_OP_INVERSE_TRANSFORM
        if kind == XDECOMP_KIND_NMF:
            _query_nmf(slot, call, inverse)
        elif _is_rp(kind):
            _query_rp(slot, call, inverse)
        elif kind == XDECOMP_KIND_ICA:
            _query_ica(slot, call, inverse)
        elif _is_kit(kind):
            _query_kit(slot, call, inverse)
        else:
            _query(slot, call, inverse)
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
