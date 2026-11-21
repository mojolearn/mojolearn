# SPDX-License-Identifier: Apache-2.0
"""The Prep family as MEMBERS (core/member_abi.mojo): RobustScaler (kind 0),
MaxAbsScaler (kind 1), Binarizer (kind 2), Normalizer (kind 3),
PolynomialFeatures (kind 4), QuantileTransformer (kind 5) and
PowerTransformer (kind 6) and SimpleImputer (kind 7); the encoders,
IterativeImputer, KBinsDiscretizer and SplineTransformer are not published
(see the boundary file for each blocker).

One entry, `prep_member_v2`, published by each preprocessing image beside
the scaler member (`prep_member_publish`, bindings/_mojolearn_preprocessing.mojo
and bindings/_mojolearn_preprocessing_host.mojo). RobustScaler runs the
door's own x_prep programs (x_prep/robust_native_run.mojo
`_robust_fit_program`, the affine stage 3) through the typed RUN of the
x_prep transport (bindings/x_prep_native_owner_run.mojo `xprep_execute`,
`xprep_release`) over the transport entry of the `DeviceRoute` it is handed
(`second`): the same arena layout, input records, range runner, device
scratch and output region the door (bindings/robust_native_pack.mojo
`robust_fit` / `robust_transform`) builds, so no bit moves. No kernel is
touched. MaxAbsScaler runs its door's x_prep programs the same way
(x_prep/maxabs_native_run.mojo `_maxabs_fit_program`, the affine stage 3;
door bindings/maxabs_native_door.mojo `maxabs_fit` / `_maxabs_apply`). No
score; no inverse_transform for kinds 0 to 4 (their classes have none). Binarizer and
Normalizer are stateless: their fit records X's width only (X is not
uploaded), their transform is the door's one-stage program
(bindings/prep_stateless_native_door.mojo `_transform` binarizer branch,
bindings/prep_normalizer_lifecycle_pack.mojo `_normalizer_apply`).
PolynomialFeatures, QuantileTransformer and PowerTransformer run the stateless
door's programs (bindings/prep_stateless_native_door.mojo `_fit` / `_transform`
polynomial branch, `_quantile_fit` / `_quantile_apply`, `_power_fit` /
`_power_apply`; programs in x_prep/prep_stateless_native_run.mojo) with the
same layouts, inputs and stages. PolynomialFeatures writes the caller's
row-major dst: `order='F'` changes only the public array's memory order, so
the member runs no p2m_transpose stage (the values are the door's).
QuantileTransformer and PowerTransformer publish inverse_transform.
SimpleImputer runs the encode door's typed stage builders
(bindings/prep_encode_imputer_run.mojo `imputer_fit_stage` /
`imputer_transform_stage` on an `EncProgram`, door
bindings/prep_encode_native_door.mojo `_imputer_fit` / `_imputer_transform`):
the member resolves the program's input records itself (owner 0 X, owner 1
the fitted fill vector, -1 the program's own words) and runs it through the
same transport, inout ranges and output region.

Route slots read: `second` the x_prep transport entry; `flags`
PREP_ROUTE_OUT (x_prep_run_out), PREP_ROUTE_SCRATCH (x_prep_run_scratch),
PREP_ROUTE_RANGES (x_prep_run_ranges and MOJOLEARN_ARENA_RANGES not 0),
PREP_ROUTE_DEV_PUT (x_prep_dev_put and x_prep_dev_free). The switches the
door reads per call (MOJOLEARN_ARENA_RANGES, MOJOLEARN_XPREP_BLOCKED, the R3
`work` choice) are read once at the caller's UNPACK.

MEMBER_OP_NEW parameters, RobustScaler (Float64 words, 9):
  [0] kind: 0 RobustScaler
  [1] with_centering 0 / 1   [2] with_scaling 0 / 1
  [3] quantile_range lower, [4] upper (percent, checked at admission)
  [5] unit_variance 0 / 1
  [6] radix quantile selection 0 / 1 (the door's `select`: fast, Metal, robust_qselect)
  [7] 2048-row statistics blocks 0 / 1 (MOJOLEARN_XPREP_BLOCKED)
  [8] R3 work region 0 / 1 (`xprep_r3("work")`)

MEMBER_OP_NEW parameters, MaxAbsScaler (Float64 words, 3):
  [0] kind: 1 MaxAbsScaler
  [1] 2048-row statistics blocks 0 / 1 (MOJOLEARN_XPREP_BLOCKED)
  [2] R3 work region 0 / 1 (`xprep_r3("work")`, read only with [1])
MEMBER_OP_NEW parameters, Binarizer (Float64 words, 2): [0] kind 2,
[1] threshold (rounded to float32 at transform, as the door).
MEMBER_OP_NEW parameters, Normalizer (Float64 words, 2): [0] kind 3,
[1] norm: 0 'l1', 1 'l2', 2 'max'.
MEMBER_OP_NEW parameters, PolynomialFeatures (Float64 words, 5): [0] kind 4,
[1] min_degree, [2] max_degree, [3] interaction_only 0 / 1,
[4] include_bias 0 / 1 (order is checked at admission; see above).
MEMBER_OP_NEW parameters, QuantileTransformer (Float64 words, 8): [0] kind 5,
[1] output_distribution: 0 'uniform', 1 'normal', [2] n_quantiles,
[3] subsample (-1: None), [4] seed low 32 bits, [5] seed high 32 bits
(random_state, None = 0), [6] the identical binding's family word
(`x_prep_idn_fam`, 0 on another tier), [7] R3 work region 0 / 1.
MEMBER_OP_NEW parameters, PowerTransformer (Float64 words, 8): [0] kind 6,
[1] method: 0 'yeo-johnson', 1 'box-cox', [2] standardize 0 / 1, [3] the
family word (as QuantileTransformer [6]), [4] the fast binding's
`x_prep_ptimpute_flags` (0 on another tier), [5] host-column binding 0 / 1,
[6] identical tier 0 / 1, [7] R3 work region 0 / 1.
MEMBER_OP_NEW parameters, SimpleImputer (Float64 words, 10): [0] kind 7,
[1] strategy: 0 'mean', 1 'median', 2 'most_frequent', 3 'constant',
[2] a numeric missing_values is marked NaN first 0 / 1, [3] that value,
[4] median by radix select 0 / 1 (the door's `_qselect`), [5] blocked column
statistics 0 / 1 (`x_prep_idn_fam` & IDN_STATS_BLOCKED), [6] fill_value
(constant; 0 for None), [7] keep_empty_features 0 / 1, [8] add_indicator
0 / 1, [9] R3 work region 0 / 1. Output width = kept + indicator columns.
MEMBER_OP_NEW parameters, KBinsDiscretizer (Float64 words, 10 + c): [0] kind
8, [1] encode 0 'onehot', 1 'onehot-dense', 2 'ordinal', [2] strategy 0
'uniform', 3 'kmeans', quantile by quantile_method 1 averaged_inverted_cdf,
2 linear, 4 inverted_cdf, 5 closest_observation, 6
interpolated_inverted_cdf, 7 hazen, 8 weibull, 9 median_unbiased, 10
normal_unbiased, [3] subsample (-1 None), [4] / [5] the uint64 random_state
low / high 32-bit words, [6] blocked column statistics 0 / 1, [7] R3 work
region 0 / 1, [8] c: 0 = one n_bins for every column in [9], else c
per-column n_bins in [10 .. 10 + c). sample_weight (MEMBER_CAN_SAMPLE_WEIGHT,
this kind only): a dense host float32 vector, one weight per row of X (of
the whole X with a row index). Transform: ordinal codes (features columns)
or one-hot blocks (sum of n_bins_ columns), float32; inverse_transform: the
bin centers (features columns).

The Apple fast direct fit entry (`x_prep_maxabs_fit_direct`, opt-in
PREP3_MAXABS) is not a member route: the admit answers False there and the
caller keeps the door.

Views: X dense row-major host float32 (NaN allowed, as the estimator allows
it); no y (ignored), no sample weights (refused, except KBinsDiscretizer), no column index, no
device-home view. ROW INDEX (MEMBER_CAN_ROW_INDEX): `call.row_index` a
dense host int32 / int64 vector of rows of the WHOLE X in `call.x` (bounds
checked; repeats allowed); the operation reads those rows, `dst` has one
row per index. On a device route whose image has the range runner, the put
and the device row gather, the whole X is put (or found resident) and the
fold is taken on the device; elsewhere the fold's rows are gathered on the
host. SCOPE (MEMBER_CAN_SCOPE): with `call.scope` >= 0 and `call.pin` != 0
on such a route, the whole X (a fold's base at any size, a plain X from the
scope's floor up) is kept resident in the caller's scope cache: one scope
read, one cache lease and one typed lookup per operation, as the door.
When the cache publishes X it keeps `call.pin` as the entry's pin (one
reference the scope drops at its close) and the call answers
`call.value = 1.0`; otherwise `call.value = 0.0` and the pin stays the
caller's. A caller hands `pin` only for X's own buffer, never for a
gathered fold. `call.fit_output = MEMBER_OP_TRANSFORM` writes the
training rows' transform to `dst` (rows x features): in the fit's own
program when rows * features < 2^27 (the door's fused fit_transform), else
the fit then the transform program, as the door.

Fitted-state blocks (MEMBER_OP_EXPORT / MEMBER_OP_IMPORT), RobustScaler:
  0 `MEMBER_BLOCK_SCALARS` float64 [n_features_in_]
  1 `PREP_BLOCK_CENTER` (MEMBER_BLOCK_FAMILY) float32 1 x features (1 x 0 without centering)
  2 `PREP_BLOCK_SCALE` float32 1 x features (1 x 0 without scaling)
MaxAbsScaler:
  0 `MEMBER_BLOCK_SCALARS` float64 [n_features_in_, n_samples_seen_]
  1 `PREP_BLOCK_MAX_ABS` (= PREP_BLOCK_CENTER) float32 1 x features (max_abs_)
  2 `PREP_BLOCK_SCALE` float32 1 x features (scale_)
Binarizer, Normalizer: 0 `MEMBER_BLOCK_SCALARS` float64 [n_features_in_] only.
PolynomialFeatures:
  0 `MEMBER_BLOCK_SCALARS` float64 [n_features_in_, n_output_features_]
  1 `PREP_BLOCK_POLY_INDEX` (= PREP_BLOCK_CENTER) float32 term feature indices (`_poly_index`)
  2 `PREP_BLOCK_POLY_START` (= PREP_BLOCK_SCALE) float32 term starts (`_poly_start`)
QuantileTransformer:
  0 `MEMBER_BLOCK_SCALARS` float64 [n_features_in_, n_quantiles_]
  1 `PREP_BLOCK_QUANTILES` (= PREP_BLOCK_CENTER) float32 features x n_quantiles_ (`_q`)
  2 `PREP_BLOCK_REFERENCES` (= PREP_BLOCK_SCALE) float32 n_quantiles_ (references_)
  3 `PREP_BLOCK_EXTRA` float32 n_quantiles_ x features (quantiles_)
PowerTransformer:
  0 `MEMBER_BLOCK_SCALARS` float64 [n_features_in_]
  1 `PREP_BLOCK_CENTER` float32 features (`_mean`; 1 x 0 without standardize)
  2 `PREP_BLOCK_SCALE` float32 features (`_scale`; 1 x 0 without standardize)
  3 `PREP_BLOCK_EXTRA` float32 features (lambdas_)
  4 `PREP_BLOCK_ANCHOR` float32 features (`_pt_anchor`; 1 x 0 when not kept)
  5 `PREP_BLOCK_ANCHOR_KIND` float32 features (`_pt_anchor_kind`; 1 x 0 when not kept)
SimpleImputer:
  0 `MEMBER_BLOCK_SCALARS` float64 [n_features_in_, statistics_ is _fill 0 / 1]
  1 `PREP_BLOCK_CENTER` float32 features (statistics_)
  2 `PREP_BLOCK_SCALE` float32 features (`_fill`)
  3 `PREP_BLOCK_EXTRA` float32 kept column indices (`_keep`)
  4 `PREP_BLOCK_ANCHOR` float32 indicator column indices (`_indicator`)
KBinsDiscretizer:
  0 `MEMBER_BLOCK_SCALARS` float64 [n_features_in_, `_stride` (max n_bins + 1)]
  1 `PREP_BLOCK_CENTER` float32 features x `_stride` (`_edges`, each row's first n_bins_ + 1 words used)
  2 `PREP_BLOCK_SCALE` float32 features (`_ne`, the edge counts = n_bins_ + 1)

The fitted state is OWNED here, in a process-wide store of fixed chunks (the
linear member's store): a slot never moves once minted, the lock covers only
mint, free and the handle lookup. Serial by nature here: the store's
bookkeeping; the programs are the family's.
"""
from std.sys.info import size_of
from std.memory import unsafe_memcpy
from bindings.x_prep_native_owner_run import XprepRunInput, XprepRoute, xprep_execute, xprep_release
from x_prep.robust_native_run import _robust_fit_program as robust_fit_program
from x_prep.maxabs_native_run import _maxabs_fit_program as maxabs_fit_program
from core.scaler_fit_impl import scaler_refuse_infinite
from x_prep.prep_stateless_native_run import (
    OP_BINARIZE, OP_NORMALIZE, OP_POLY, OP_QT_APPLY, OP_QT_INVERSE, OP_RGATHER, OP_PERM_ROWS,
    OP_PT_APPLY, OP_PT_INVERSE, OUT_MIN_WORDS, PrepLayout, _place_scratch, _poly_terms,
    _quantile_program, _power_program,
)
from core.preprocessing_program import prep_program_stage as _stage
from bindings.prep_encode_prog_run import EncProgram
from bindings.prep_encode_kbins_run import (
    KBinsFit, KBinsApply, KBinsInverse, kbins_fit_stage, kbins_transform_stage, kbins_inverse_stage,
    enc_weights_stage, KB_DRAW_NONE, KB_DRAW_PLAIN, KB_DRAW_WEIGHTED,
)
from bindings.prep_encode_imputer_run import (
    ImputerFit, ImputerApply, imputer_fit_stage, imputer_transform_stage, IMP_CONSTANT,
)
from core.member_abi import (
    MemberCall, MemberMatrix, MEMBER_F32, MEMBER_F64, MEMBER_I32, MEMBER_I64,
    MEMBER_CAN_SCOPE, MEMBER_CAN_ROW_INDEX, MEMBER_CAN_SAMPLE_WEIGHT,
    MEMBER_OP_CAPABILITIES, MEMBER_OP_NEW, MEMBER_OP_FREE, MEMBER_OP_FIT, MEMBER_OP_OUTPUTS,
    MEMBER_OP_TRANSFORM, MEMBER_OP_INVERSE_TRANSFORM, MEMBER_OP_EXPORT, MEMBER_OP_IMPORT,
    MEMBER_CAN_FIT, MEMBER_CAN_TRANSFORM, MEMBER_CAN_INVERSE_TRANSFORM, MEMBER_CAN_EXPORT, MEMBER_CAN_IMPORT,
    MEMBER_BLOCK_SCALARS, MEMBER_BLOCK_FAMILY, MEMBER_BLOCK_BY_INDEX, MEMBER_IMPORT_DONE,
    member_publish, member_call_from_spans, member_unsupported,
)
from core.member_families import MEMBER_FAMILY_PREP
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.shared_slot import SharedSlot, shared_slot_lock, shared_slot_unlock
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk
from bindings.arena_cache_run import arena_scope_floor, arena_scope_cache, arena_cache_find_or_put

comptime PREP_KIND_ROBUST = 0
comptime PREP_KIND_MAXABS = 1
comptime PREP_ROBUST_PARAM_COUNT = 9
comptime PREP_MAXABS_PARAM_COUNT = 3
comptime PREP_KIND_BINARIZER = 2
comptime PREP_KIND_NORMALIZER = 3
comptime PREP_STATELESS_PARAM_COUNT = 2
comptime PREP_KIND_POLYNOMIAL = 4
comptime PREP_KIND_QUANTILE = 5
comptime PREP_KIND_POWER = 6
comptime PREP_POLY_PARAM_COUNT = 5
comptime PREP_QUANTILE_PARAM_COUNT = 8
comptime PREP_POWER_PARAM_COUNT = 8
comptime PREP_KIND_IMPUTER = 7
comptime PREP_IMPUTER_PARAM_COUNT = 10
comptime PREP_KIND_KBINS = 8
#: KBinsDiscretizer's fixed words; `words[8]` per-column n_bins words follow.
comptime PREP_KBINS_HEADER = 10
#: KBinsDiscretizer's encode codes.
comptime PREP_KBINS_ONEHOT = 0
comptime PREP_KBINS_ONEHOT_DENSE = 1
comptime PREP_KBINS_ORDINAL = 2

comptime PREP_BLOCK_CENTER = MEMBER_BLOCK_FAMILY
comptime PREP_BLOCK_SCALE = MEMBER_BLOCK_FAMILY + 1
#: MaxAbsScaler's max_abs_ sits in the block RobustScaler's center_ uses.
comptime PREP_BLOCK_MAX_ABS = PREP_BLOCK_CENTER
comptime PREP_BLOCK_EXTRA = MEMBER_BLOCK_FAMILY + 2
comptime PREP_BLOCK_ANCHOR = MEMBER_BLOCK_FAMILY + 3
comptime PREP_BLOCK_ANCHOR_KIND = MEMBER_BLOCK_FAMILY + 4
#: PolynomialFeatures' term table, QuantileTransformer's fitted quantiles.
comptime PREP_BLOCK_POLY_INDEX = PREP_BLOCK_CENTER
comptime PREP_BLOCK_POLY_START = PREP_BLOCK_SCALE
comptime PREP_BLOCK_QUANTILES = PREP_BLOCK_CENTER
comptime PREP_BLOCK_REFERENCES = PREP_BLOCK_SCALE

comptime PREP_ROUTE_OUT = 1
comptime PREP_ROUTE_SCRATCH = 2
comptime PREP_ROUTE_RANGES = 4
comptime PREP_ROUTE_DEV_PUT = 8
#: The image exports the device row gather (`x_prep_dev_take_rows`).
comptime PREP_ROUTE_TAKE_ROWS = 16

#: The door's limits: the fused fit_transform and the output region
#: (bindings/robust_native_pack.mojo), the direct put (x_prep_native_owner_door).
comptime _FUSED_LIMIT = 134217728
comptime _PUT_WORDS = 1048576
comptime _INT32_BOUND = 2147483647

comptime _CHUNK = 256
comptime _INDEX_SPAN = 16777216


struct PrepMemberSlot(Copyable, Movable, Defaultable):
    var generation: Int
    var live: Bool
    var words: List[Float64]
    var fitted: Bool
    var features: Int
    var center: List[Float32]
    var scale: List[Float32]
    var scalars: List[Float64]
    var extra: List[Float32]
    var anchor: List[Float32]
    var anchor_kind: List[Float32]

    def __init__(out self):
        self.generation = 0
        self.live = False
        self.words = List[Float64]()
        self.fitted = False
        self.features = 0
        self.center = List[Float32]()
        self.scale = List[Float32]()
        self.scalars = List[Float64]()
        self.extra = List[Float32]()
        self.anchor = List[Float32]()
        self.anchor_kind = List[Float32]()

    def kind(self) -> Int:
        return Int(self.words[0])

    def clear(mut self):
        self.fitted = False
        self.features = 0
        self.center = List[Float32]()
        self.scale = List[Float32]()
        self.scalars = List[Float64]()
        self.extra = List[Float32]()
        self.anchor = List[Float32]()
        self.anchor_kind = List[Float32]()


struct PrepMemberStore(Defaultable, Movable):
    var synchronization: List[Int64]
    var chunks: List[List[PrepMemberSlot]]
    var free: List[Int]
    var count: Int

    def __init__(out self):
        # Side-effect-free even when the process slot's initializers race.
        self.synchronization = List[Int64](length=2, fill=0)
        self.chunks = List[List[PrepMemberSlot]]()
        self.free = List[Int]()
        self.count = 0


comptime _STORE = SharedSlot[PrepMemberStore, "MojolearnPrepMemberStoreV2", 2]
comptime _Slot = MutPointer[PrepMemberSlot, MutUntrackedOrigin]
comptime _Call = MutPointer[MemberCall, MutUntrackedOrigin]
comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]


def _slot_address(index: Int) raises -> Int:
    # Under the store's lock. A chunk's buffer is allocated once and never
    # grows, so the address stays valid after the lock is released.
    var store = _STORE.get_or_create_ptr()
    return Int(store[].chunks[index // _CHUNK].unsafe_ptr()) + (index % _CHUNK) * size_of[PrepMemberSlot]()


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
                raise Error("RuntimeError: mojolearn: too many live Prep member states")
            if store[].count == len(store[].chunks) * _CHUNK:
                store[].chunks.append(List[PrepMemberSlot](length=_CHUNK, fill=PrepMemberSlot()))
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
    if kind == PREP_KIND_MAXABS:
        return String("MaxAbsScaler")
    if kind == PREP_KIND_BINARIZER:
        return String("Binarizer")
    if kind == PREP_KIND_NORMALIZER:
        return String("Normalizer")
    if kind == PREP_KIND_POLYNOMIAL:
        return String("PolynomialFeatures")
    if kind == PREP_KIND_QUANTILE:
        return String("QuantileTransformer")
    if kind == PREP_KIND_POWER:
        return String("PowerTransformer")
    if kind == PREP_KIND_IMPUTER:
        return String("SimpleImputer")
    if kind == PREP_KIND_KBINS:
        return String("KBinsDiscretizer")
    return String("RobustScaler")


def _block_count(kind: Int) -> Int:
    # Fitted blocks, the scalars block included (the docstring's table).
    if kind == PREP_KIND_BINARIZER or kind == PREP_KIND_NORMALIZER:
        return 1
    if kind == PREP_KIND_QUANTILE:
        return 4
    if kind == PREP_KIND_POWER:
        return 6
    if kind == PREP_KIND_IMPUTER:
        return 5
    return 3


def _scalar_count(kind: Int) -> Int:
    if (kind == PREP_KIND_MAXABS or kind == PREP_KIND_POLYNOMIAL or kind == PREP_KIND_QUANTILE
            or kind == PREP_KIND_IMPUTER or kind == PREP_KIND_KBINS):
        return 2
    return 1


def _invertible(kind: Int) -> Bool:
    return kind == PREP_KIND_QUANTILE or kind == PREP_KIND_POWER or kind == PREP_KIND_KBINS


def _name(slot: _Slot) -> String:
    return _kind_name(slot[].kind())


def _dense(view: MemberMatrix, what: String, estimator: String) raises:
    if not view.dense_host(MEMBER_F32):
        raise Error("TypeError: a " + estimator + " member reads " + what + " as a dense host float32 view")


def _whole(call: _Call, estimator: String) raises:
    """Every column of X; a row index is a dense host int32 / int64 vector
    (the fold's rows of the WHOLE X the caller hands, MEMBER_CAN_ROW_INDEX)."""
    if call[].column_index.rows > 0:
        raise Error("member ABI: a " + estimator + " member reads every column of X (no column index)")
    var index = call[].row_index
    if index.rows > 0 and not ((index.dtype == MEMBER_I32 or index.dtype == MEMBER_I64) and index.cols == 1
                               and index.dense_host(index.dtype)):
        raise Error("member ABI: a " + estimator + " member reads a dense host int32 / int64 row index vector")


def _rows(call: _Call) -> Int:
    """The rows an operation reads: the row index's length, else X's."""
    return call[].row_index.rows if call[].row_index.rows > 0 else call[].x.rows


struct PrepXHeld(Movable):
    """Words X's input record addresses beyond the caller's views, owned by
    one operation and named again after its program ran: the fold's rows
    gathered on the host (a route without the device row gather) and an
    int32 row index widened to the int64 numbers the device gather reads."""
    var gathered: List[Float32]
    var indices: List[Int64]

    def __init__(out self):
        self.gathered = List[Float32]()
        self.indices = List[Int64]()


def _fold_rows(call: _Call, mut held: PrepXHeld, gather: Bool) raises -> Int:
    """The fold's row numbers, bounds-checked against X's rows, split across
    the host's workers: with `gather`, its rows copied into `held.gathered`
    (0 returned); else the address of int64 numbers (the caller's, or an
    int32 index widened into `held.indices`). Once per operation."""
    var n = call[].row_index.rows
    var base = call[].x.rows
    var d = call[].x.cols
    var wide = call[].row_index.dtype == MEMBER_I64
    var widen = not gather and not wide
    if gather and len(held.gathered) == 0:
        held.gathered = List[Float32](length=max(n * d, 1), fill=Float32(0))
    if widen and len(held.indices) == 0:
        held.indices = List[Int64](length=n, fill=Int64(0))
    var tasks = host_predict_task_count_work(n, n * d if gather else n)
    var chunk = host_predict_chunk(n, tasks)
    var bad = List[Int](length=tasks, fill=0)
    var flags = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=Int(bad.unsafe_ptr()))
    var source = call[].row_index.data
    var x = _F32P(unsafe_from_address=call[].x.data)
    var into = _F32P(unsafe_from_address=Int(held.gathered.unsafe_ptr()))
    var wider = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(held.indices.unsafe_ptr()))

    def task(worker: Int) {imm chunk, imm n, imm base, imm d, imm wide, imm gather, imm widen, imm source, imm flags,
                           imm x, imm into, imm wider}:
        var begin = worker * chunk
        var end = min(begin + chunk, n)
        var out_of_range = 0
        for r in range(begin, end):
            var row = Int(MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=source)[r]) if wide else Int(
                MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=source)[r])
            if row < 0 or row >= base:
                out_of_range = 1
                continue
            if gather:
                unsafe_memcpy(dest=into + r * d, src=x + row * d, count=d)
            elif widen:
                wider[r] = Int64(row)
        flags[worker] = out_of_range

    host_parallelize(task, tasks)
    for w in range(tasks):  # driver-loop: over the workers' range flags
        if bad[w] != 0:
            raise Error("IndexError: a row index is out of range for X's " + String(base) + " rows")
    if gather:
        return 0
    return Int(held.indices.unsafe_ptr()) if widen else source


def _resident(call: _Call, words: Int, rows: Bool) raises -> Int:
    """The store id of X's resident copy in the caller's scope (found, or put
    once and published with `call.pin` as the entry's pin), -1 when the call
    keeps nothing resident: no scope, nothing pinned, or below the scope's
    floor (a row-gathered base at any size, as the door). `call.value` = 1
    when the cache KEPT `call.pin` (the scope drops that reference at its
    close); 0 left as is otherwise (found, lost a race, or not kept)."""
    if call[].scope < 0 or call[].pin == 0:
        return -1
    if not rows and words < arena_scope_floor(call[].scope):
        return -1
    var store = call[].route.second
    var cache = arena_scope_cache(call[].scope, store, "x_prep")
    var put = arena_cache_find_or_put(cache, store, call[].x.data, words, call[].pin)
    if put.pinned:
        call[].value = 1.0
    return put.store_id


def _x_record(call: _Call, offset: Int, words: Int, mut held: PrepXHeld, inout: Bool,
    scoped: Bool = True) raises -> XprepRunInput:
    """X as a program's input record at `offset` (`words` = the rows the
    operation reads times X's width). The door's rules
    (bindings/x_prep_native_owner_door.mojo `xprep_run_records`), from the
    route words read at admit: on a device route with the range runner a
    whole X stays resident in the caller's scope when it is pinned, and a
    fold (`call.row_index`) is taken on the device out of the resident (or
    put) whole X; without the device gather the fold's rows are gathered on
    the host; otherwise X is copied into the arena, or put on the device
    directly when the image has the put and X is large. An in-out input is
    never resident nor device-gathered."""
    var flags = call[].route.flags
    # The range runner is certain for any scratch only with an output or a
    # scratch entry (`_execute_ranges`): a resident or gathered record needs it.
    var device = ((flags & PREP_ROUTE_DEV_PUT) != 0 and (flags & PREP_ROUTE_RANGES) != 0
                  and (flags & (PREP_ROUTE_OUT | PREP_ROUTE_SCRATCH)) != 0 and not inout)
    var d = call[].x.cols
    var base_words = call[].x.rows * d
    if call[].row_index.rows > 0:
        var n = call[].row_index.rows
        if device and (flags & PREP_ROUTE_TAKE_ROWS) != 0:
            var indices = _fold_rows(call, held, False)
            var source = _resident(call, base_words, True) if scoped else -1
            return XprepRunInput(offset, words, call[].x.data, source, source < 0, base_words, True, indices, n, d)
        _ = _fold_rows(call, held, True)
        var put_rows = (flags & PREP_ROUTE_DEV_PUT) != 0 and words >= _PUT_WORDS
        return XprepRunInput(offset, words, Int(held.gathered.unsafe_ptr()), -1, put_rows, words, False, 0, 0, 0)
    if device and scoped:
        var source = _resident(call, base_words, False)
        if source >= 0:
            return XprepRunInput(offset, words, 0, source, False, 0, False, 0, 0, 0)
    var put = (flags & PREP_ROUTE_DEV_PUT) != 0 and words >= _PUT_WORDS
    return XprepRunInput(offset, words, call[].x.data, -1, put, words, False, 0, 0, 0)


def _x_input(call: _Call, words: Int, mut held: PrepXHeld) raises -> XprepRunInput:
    return _x_record(call, 0, words, held, False)


def _x_input(call: _Call, words: Int) -> XprepRunInput:
    # A whole X of a member that reads neither a scope nor a row index (the
    # Select family): copied into the arena by the transport, or put on the
    # device directly when the image has the put and X is large.
    var put = (call[].route.flags & PREP_ROUTE_DEV_PUT) != 0 and words >= _PUT_WORDS
    return XprepRunInput(0, words, call[].x.data, -1, put, words, False, 0, 0, 0)


def _execute(call: _Call, mut arena: List[Float32], host: Int, scratch: Int, output: Int, out_words: Int,
    program: List[Int32], inputs: List[XprepRunInput]) raises:
    """One program run through the route's transport (the door's
    `xprep_run_records` RUN and cleanup, without its Python records)."""
    _execute_ranges(call, arena, host, scratch, output, out_words, program, inputs, List[Int]())


def _execute_ranges(call: _Call, mut arena: List[Float32], host: Int, scratch: Int, output: Int, out_words: Int,
    program: List[Int32], inputs: List[XprepRunInput], inout_ranges: List[Int]) raises:
    # `_execute` with the program's inout ranges (an `EncProgram`'s).
    var flags = call[].route.flags
    var has_out = (flags & PREP_ROUTE_OUT) != 0
    var has_scratch = (flags & PREP_ROUTE_SCRATCH) != 0
    var device_scratch = scratch > 0 and (has_out or has_scratch)
    var ranges = (scratch == 0 or has_out or has_scratch) and (flags & PREP_ROUTE_RANGES) != 0
    if not ranges:
        for i in range(len(inputs)):  # driver-loop: over the program's input records
            if inputs[i].source >= 0 or inputs[i].rows:
                raise Error("RuntimeError: x_prep member: a resident or row-gathered input needs the range runner")
    var key = call[].route.second
    var owned = List[Int](capacity=2 * len(inputs))
    try:
        xprep_execute(key, Int(arena.unsafe_ptr()), host, scratch, output, out_words, program, inputs, inout_ranges,
            XprepRoute(ranges, device_scratch, has_out), owned)
    except error:
        var cleanup = xprep_release(key, owned)
        if cleanup.byte_length() > 0:
            raise Error("RuntimeError: x_prep operation failed: " + String(error) + "; cleanup also failed: " + cleanup)
        raise error^
    var cleanup = xprep_release(key, owned)
    if cleanup.byte_length() > 0:
        raise Error("RuntimeError: x_prep slot cleanup failed: " + cleanup)
    _ = arena


def _fit_robust(slot: _Slot, call: _Call) raises:
    var estimator = _name(slot)
    _whole(call, estimator)
    _dense(call[].x, "X", estimator)
    if call[].weights.rows > 0:
        raise Error("TypeError: a " + estimator + " member takes no sample_weight")
    if call[].fit_output != 0 and call[].fit_output != MEMBER_OP_TRANSFORM:
        member_unsupported(estimator, call[].fit_output)
    var n = _rows(call)
    var d = call[].x.cols
    var project = call[].fit_output == MEMBER_OP_TRANSFORM
    if project:
        _dense(call[].dst, "the output buffer", estimator)
        if call[].dst.rows != n or call[].dst.cols != d:
            raise Error("ValueError: the output buffer holds one value per element of X")
    var centering = slot[].words[1] != 0.0
    var scaling = slot[].words[2] != 0.0
    var lo = slot[].words[3]
    var hi = slot[].words[4]
    var unit = slot[].words[5] != 0.0
    var select = slot[].words[6] != 0.0
    var blocked = slot[].words[7] != 0.0
    var work = slot[].words[8] != 0.0
    # The door's arena layout (bindings/robust_native_pack.mojo robust_fit).
    var blocks = (n + 2047) // 2048
    var sort_words = 0 if select else n * d
    var part_words = 5 * blocks * d if blocked else 0
    var scratch = sort_words + part_words if work else 0
    var sorted = n * d
    var stats = n * d + (0 if work else sort_words)
    var fractions = stats + 6 * d
    var q = fractions + 3
    var center = q + 3 * d
    var scale = center + d
    var host = scale + d + (0 if work else part_words)
    var fused = project and n * d < _FUSED_LIMIT
    var dest = host
    if fused:
        host += n * d
    var part = scale + d
    if work:
        sorted = host
        part = host + sort_words
    if host + scratch > _INT32_BOUND:
        raise Error("x_prep: the program exceeds the native Int32 indexing bound")
    var flags = call[].route.flags
    var device_scratch = scratch > 0 and (flags & (PREP_ROUTE_OUT | PREP_ROUTE_SCRATCH)) != 0
    var arena = List[Float32](length=max(host + (0 if device_scratch else scratch), 1), fill=Float32(0))
    var fraction_values: List[Float32] = [Float32(lo / 100.0), Float32(0.5), Float32(hi / 100.0)]
    var inputs = List[XprepRunInput](capacity=2)
    var held = PrepXHeld()
    inputs.append(_x_input(call, n * d, held))
    inputs.append(XprepRunInput(fractions, 3, Int(fraction_values.unsafe_ptr()), -1, False, 0, False, 0, 0, 0))
    var program = List[Int32]()
    robust_fit_program(program, select, blocked, unit, fused, centering, scaling, n, d, blocks, sorted, part, stats,
        fractions, q, center, scale, dest)
    _execute(call, arena, host, scratch, 0, 0, program, inputs)
    _ = held^  # the fold words X's record addresses live past the run
    _ = fraction_values^
    slot[].clear()
    var ap = _F32P(unsafe_from_address=Int(arena.unsafe_ptr()))
    if centering:
        slot[].center = List[Float32](length=d, fill=Float32(0))
        unsafe_memcpy(dest=slot[].center.unsafe_ptr(), src=ap + center, count=d)
    if scaling:
        slot[].scale = List[Float32](length=d, fill=Float32(0))
        unsafe_memcpy(dest=slot[].scale.unsafe_ptr(), src=ap + scale, count=d)
    slot[].features = d
    slot[].scalars = [Float64(d)]
    slot[].fitted = True
    if fused:
        unsafe_memcpy(dest=_F32P(unsafe_from_address=call[].dst.data), src=ap + dest, count=n * d)
    _ = arena^
    if project and not fused:
        _transform_robust(slot, call)


def _transform_robust(slot: _Slot, call: _Call) raises:
    var estimator = _name(slot)
    if not slot[].fitted:
        raise Error("NotFittedError: This RobustScaler instance is not fitted yet. Call 'fit' with appropriate "
                    + "arguments before using transform.")
    _whole(call, estimator)
    _dense(call[].x, "X", estimator)
    _dense(call[].dst, "the output buffer", estimator)
    var n = _rows(call)
    var d = slot[].features
    if call[].x.cols != d:
        raise Error("RobustScaler input feature count differs from fit")
    if call[].dst.rows != n or call[].dst.cols != d:
        raise Error("ValueError: the output buffer holds one value per element of X")
    var has_center = len(slot[].center) > 0
    var has_scale = len(slot[].scale) > 0
    # The door's layout (robust_transform).
    var center_at = n * d if has_center else -1
    var scale_at = n * d + (d if has_center else 0) if has_scale else -1
    var dest = n * d + (d if has_center else 0) + (d if has_scale else 0)
    if dest + n * d > _INT32_BOUND:
        raise Error("x_prep: the program exceeds the native Int32 indexing bound")
    var dev_out = n * d >= _FUSED_LIMIT and (call[].route.flags & PREP_ROUTE_OUT) != 0
    var host = dest + (0 if dev_out else n * d)
    var arena = List[Float32](length=max(host, 1), fill=Float32(0))
    var inputs = List[XprepRunInput](capacity=3)
    var held = PrepXHeld()
    inputs.append(_x_input(call, n * d, held))
    if has_center:
        inputs.append(XprepRunInput(center_at, d, Int(slot[].center.unsafe_ptr()), -1, False, 0, False, 0, 0, 0))
    if has_scale:
        inputs.append(XprepRunInput(scale_at, d, Int(slot[].scale.unsafe_ptr()), -1, False, 0, False, 0, 0, 0))
    var program = List[Int32]()
    _stage(program, 3, n * d, 0, n * d, d, center_at, scale_at, dest)
    # A large output comes back through the image's output region straight
    # into the caller's buffer (the door's output store).
    _execute(call, arena, host, 0, call[].dst.data if dev_out else 0, n * d if dev_out else 0, program, inputs)
    _ = held^  # the fold words X's record addresses live past the run
    if not dev_out:
        unsafe_memcpy(dest=_F32P(unsafe_from_address=call[].dst.data),
            src=_F32P(unsafe_from_address=Int(arena.unsafe_ptr())) + dest, count=n * d)
    _ = arena^


def _fit_maxabs(slot: _Slot, call: _Call) raises:
    """The door's `maxabs_fit` (bindings/maxabs_native_door.mojo) over the
    caller's X: the same arena layout, program and fused fit_transform."""
    var estimator = _name(slot)
    _whole(call, estimator)
    _dense(call[].x, "X", estimator)
    if call[].weights.rows > 0:
        raise Error("TypeError: a " + estimator + " member takes no sample_weight")
    if call[].fit_output != 0 and call[].fit_output != MEMBER_OP_TRANSFORM:
        member_unsupported(estimator, call[].fit_output)
    var n = _rows(call)
    var d = call[].x.cols
    var project = call[].fit_output == MEMBER_OP_TRANSFORM
    if project:
        _dense(call[].dst, "the output buffer", estimator)
        if call[].dst.rows != n or call[].dst.cols != d:
            raise Error("ValueError: the output buffer holds one value per element of X")
    # Existing row blocks are 2048 rows, independent of benchmark shapes.
    var blocked = slot[].words[1] != 0.0
    var blocks = (n + 2047) // 2048
    var work = blocked and slot[].words[2] != 0.0
    var scratch = 5 * blocks * d if work else 0
    var host = n * d + ((2 * d + (0 if work else 5 * blocks * d)) if blocked else 7 * d)
    # fit_transform: the scaled X takes n*d words after the fit's host words
    # (the transform program's affine stage, operands and order).
    var fused = project and n * d < _FUSED_LIMIT
    var dest = host
    if fused:
        host += n * d
    if host + scratch > _INT32_BOUND:
        raise Error("x_prep: the program exceeds the native Int32 indexing bound")
    var flags = call[].route.flags
    var device_scratch = scratch > 0 and (flags & (PREP_ROUTE_OUT | PREP_ROUTE_SCRATCH)) != 0
    var arena = List[Float32](length=max(host + (0 if device_scratch else scratch), 1), fill=Float32(0))
    var sc = n * d
    var program = List[Int32]()
    var ma = maxabs_fit_program(program, blocked, work, fused, n, d, blocks, host, sc, dest)
    var inputs = List[XprepRunInput](capacity=1)
    var held = PrepXHeld()
    inputs.append(_x_input(call, n * d, held))
    _execute(call, arena, host, scratch, 0, 0, program, inputs)
    _ = held^  # the fold words X's record addresses live past the run
    var ap = _F32P(unsafe_from_address=Int(arena.unsafe_ptr()))
    # scikit-learn refuses an infinite input (NaN alone is allowed); the
    # column maxima of |X| are infinite exactly for those columns.
    scaler_refuse_infinite(ap + ma, d, "ValueError: Input X contains infinity or a value too large for dtype('float64').")
    slot[].clear()
    slot[].center = List[Float32](length=d, fill=Float32(0))
    unsafe_memcpy(dest=slot[].center.unsafe_ptr(), src=ap + ma, count=d)
    slot[].scale = List[Float32](length=d, fill=Float32(0))
    unsafe_memcpy(dest=slot[].scale.unsafe_ptr(), src=ap + sc, count=d)
    slot[].features = d
    slot[].scalars = [Float64(d), Float64(n)]
    slot[].fitted = True
    if fused:
        unsafe_memcpy(dest=_F32P(unsafe_from_address=call[].dst.data), src=ap + dest, count=n * d)
    _ = arena^
    if project and not fused:
        _transform_maxabs(slot, call)


def _transform_maxabs(slot: _Slot, call: _Call) raises:
    """The door's `_maxabs_apply`: stage 3 with scale_ after X."""
    var estimator = _name(slot)
    if not slot[].fitted:
        raise Error("NotFittedError: This MaxAbsScaler instance is not fitted yet. Call 'fit' with appropriate "
                    + "arguments before using transform.")
    _whole(call, estimator)
    _dense(call[].x, "X", estimator)
    _dense(call[].dst, "the output buffer", estimator)
    var n = _rows(call)
    var d = slot[].features
    if call[].x.cols != d:
        raise Error("MaxAbsScaler input feature count differs from fit")
    if call[].dst.rows != n or call[].dst.cols != d:
        raise Error("ValueError: the output buffer holds one value per element of X")
    var dest = n * d + d
    if dest + n * d > _INT32_BOUND:
        raise Error("x_prep: the program exceeds the native Int32 indexing bound")
    var dev_out = n * d >= _FUSED_LIMIT and (call[].route.flags & PREP_ROUTE_OUT) != 0
    var host = dest + (0 if dev_out else n * d)
    var arena = List[Float32](length=max(host, 1), fill=Float32(0))
    var inputs = List[XprepRunInput](capacity=2)
    var held = PrepXHeld()
    inputs.append(_x_input(call, n * d, held))
    inputs.append(XprepRunInput(n * d, d, Int(slot[].scale.unsafe_ptr()), -1, False, 0, False, 0, 0, 0))
    var program = List[Int32]()
    _stage(program, 3, n * d, 0, n * d, d, -1, n * d, dest)
    # A large output comes back through the image's output region straight
    # into the caller's buffer (the door's output store).
    _execute(call, arena, host, 0, call[].dst.data if dev_out else 0, n * d if dev_out else 0, program, inputs)
    _ = held^  # the fold words X's record addresses live past the run
    if not dev_out:
        unsafe_memcpy(dest=_F32P(unsafe_from_address=call[].dst.data),
            src=_F32P(unsafe_from_address=Int(arena.unsafe_ptr())) + dest, count=n * d)
    _ = arena^


def _fit_stateless(slot: _Slot, call: _Call) raises:
    """Binarizer / Normalizer fit: X's width, as the doors' fit records it
    (nothing is uploaded or computed); fit_transform then transforms."""
    var estimator = _name(slot)
    _whole(call, estimator)
    _dense(call[].x, "X", estimator)
    if call[].weights.rows > 0:
        raise Error("TypeError: a " + estimator + " member takes no sample_weight")
    if call[].fit_output != 0 and call[].fit_output != MEMBER_OP_TRANSFORM:
        member_unsupported(estimator, call[].fit_output)
    slot[].clear()
    slot[].features = call[].x.cols
    slot[].scalars = [Float64(call[].x.cols)]
    slot[].fitted = True
    if call[].fit_output == MEMBER_OP_TRANSFORM:
        _transform_stateless(slot, call)


def _transform_stateless(slot: _Slot, call: _Call) raises:
    var estimator = _name(slot)
    if not slot[].fitted:
        raise Error("RuntimeError: mojolearn: this " + estimator + " instance is not fitted yet")
    _whole(call, estimator)
    _dense(call[].x, "X", estimator)
    _dense(call[].dst, "the output buffer", estimator)
    var n = _rows(call)
    var d = slot[].features
    if call[].x.cols != d:
        raise Error("mojolearn: X has " + String(call[].x.cols) + " features, " + estimator + " was fitted with "
                    + String(d))
    if call[].dst.rows != n or call[].dst.cols != d:
        raise Error("ValueError: the output buffer holds one value per element of X")
    var words = n * d
    var program = List[Int32]()
    var inputs = List[XprepRunInput](capacity=2)
    var held = PrepXHeld()
    inputs.append(_x_input(call, words, held))
    if slot[].kind() == PREP_KIND_BINARIZER:
        # X, the threshold word, then the output as arena words (the door's
        # binarizer branch; no output region).
        var threshold: List[Float32] = [Float32(slot[].words[1])]
        var out = words + 1
        var host = out + words
        if host > _INT32_BOUND:
            raise Error("ValueError: x_prep: the program exceeds the native Int32 indexing bound")
        inputs.append(XprepRunInput(words, 1, Int(threshold.unsafe_ptr()), -1, False, 0, False, 0, 0, 0))
        _stage(program, OP_BINARIZE, words, 0, words, words, out)
        var arena = List[Float32](length=max(host, 1), fill=Float32(0))
        _execute(call, arena, host, 0, 0, 0, program, inputs)
        _ = held^  # the fold words X's record addresses live past the run
        _ = threshold^
        unsafe_memcpy(dest=_F32P(unsafe_from_address=call[].dst.data),
            src=_F32P(unsafe_from_address=Int(arena.unsafe_ptr())) + out, count=words)
        _ = arena^
        return
    # Normalizer (the door's _normalizer_apply): a large output comes back
    # through the image's output region straight into the caller's buffer.
    var dev_out = words >= OUT_MIN_WORDS and (call[].route.flags & PREP_ROUTE_OUT) != 0
    var host = words + (0 if dev_out else words)
    if host + (words if dev_out else 0) > _INT32_BOUND:
        raise Error("ValueError: x_prep: the program exceeds the native Int32 indexing bound")
    _stage(program, OP_NORMALIZE, n, 0, n, d, Int(slot[].words[1]), host if dev_out else words)
    var arena = List[Float32](length=max(host, 1), fill=Float32(0))
    _execute(call, arena, host, 0, call[].dst.data if dev_out else 0, words if dev_out else 0, program, inputs)
    _ = held^  # the fold words X's record addresses live past the run
    if not dev_out:
        unsafe_memcpy(dest=_F32P(unsafe_from_address=call[].dst.data),
            src=_F32P(unsafe_from_address=Int(arena.unsafe_ptr())) + words, count=words)
    _ = arena^


def _fit_begin(slot: _Slot, call: _Call) raises:
    # The checks every fit shares: whole dense host X, no weights, the one
    # fit output a transformer has.
    var estimator = _name(slot)
    _whole(call, estimator)
    _dense(call[].x, "X", estimator)
    if call[].weights.rows > 0:
        raise Error("TypeError: a " + estimator + " member takes no sample_weight")
    if call[].fit_output != 0 and call[].fit_output != MEMBER_OP_TRANSFORM:
        member_unsupported(estimator, call[].fit_output)


def _apply_begin(slot: _Slot, call: _Call, out_cols: Int) raises:
    # The checks every transform shares (the stateless door's words).
    var estimator = _name(slot)
    if not slot[].fitted:
        raise Error("RuntimeError: mojolearn: this " + estimator + " instance is not fitted yet")
    _whole(call, estimator)
    _dense(call[].x, "X", estimator)
    _dense(call[].dst, "the output buffer", estimator)
    var d = slot[].features
    if call[].x.cols != d:
        raise Error("mojolearn: X has " + String(call[].x.cols) + " features, " + estimator + " was fitted with "
                    + String(d))
    if call[].dst.rows != _rows(call) or call[].dst.cols != out_cols:
        raise Error("ValueError: the output buffer holds " + String(out_cols) + " values per row of X")


def _run_layout(call: _Call, mut program: List[Int32], layout: PrepLayout,
    inputs: List[XprepRunInput]) raises -> List[Float32]:
    """The door's `_run_layout`: scratch words placed past the host words,
    device scratch when the image has it; returns the arena."""
    if layout.host + layout.scratch > _INT32_BOUND:
        raise Error("ValueError: x_prep: the program exceeds the native Int32 indexing bound")
    _place_scratch(program, layout.host)
    var device_scratch = layout.scratch > 0 and (call[].route.flags & (PREP_ROUTE_OUT | PREP_ROUTE_SCRATCH)) != 0
    var arena = List[Float32](length=max(layout.host + (0 if device_scratch else layout.scratch), 1), fill=Float32(0))
    _execute(call, arena, layout.host, layout.scratch, 0, 0, program, inputs)
    return arena^


def _take(arena: List[Float32], offset: Int, count: Int) -> List[Float32]:
    var words = List[Float32](length=count, fill=Float32(0))
    if count > 0:
        unsafe_memcpy(dest=words.unsafe_ptr(), src=_F32P(unsafe_from_address=Int(arena.unsafe_ptr())) + offset,
            count=count)
    return words^


def _poly_table(words: List[Float64], d: Int, mut index: List[Int], mut start: List[Int]):
    # The door's `_fit` term table from the parameter words.
    _poly_terms(d, Int(words[1]), Int(words[2]), words[3] == 0.0, words[4] != 0.0, index, start)


def _fit_poly(slot: _Slot, call: _Call) raises:
    """The door's `_fit` polynomial branch: the term table of X's width, as
    float32 index words (`_words`: count max(len, 1), exact below 2^24)."""
    _fit_begin(slot, call)
    var d = call[].x.cols
    var index = List[Int]()
    var start = List[Int]()
    _poly_table(slot[].words, d, index, start)
    var nout = len(start) - 1
    slot[].clear()
    slot[].center = List[Float32](length=max(len(index), 1), fill=Float32(0))
    for i in range(len(index)):  # driver-loop: over polynomial term table words (feature metadata)
        slot[].center[i] = Float32(index[i])
    slot[].scale = List[Float32](length=max(len(start), 1), fill=Float32(0))
    for i in range(len(start)):  # driver-loop: over polynomial term starts (feature metadata)
        slot[].scale[i] = Float32(start[i])
    slot[].features = d
    slot[].scalars = [Float64(d), Float64(nout)]
    slot[].fitted = True
    if call[].fit_output == MEMBER_OP_TRANSFORM:
        _transform_poly(slot, call)


def _transform_poly(slot: _Slot, call: _Call) raises:
    """The door's `_transform` polynomial branch (X, the term indices, the
    term starts, the products) into the caller's row-major dst."""
    var nout = Int(slot[].scalars[1]) if slot[].fitted else 0
    _apply_begin(slot, call, nout)
    var n = _rows(call)
    var d = slot[].features
    if len(slot[].scale) != nout + 1:
        raise Error("RuntimeError: PolynomialFeatures term table does not match n_output_features_")
    var words = n * nout
    if words == 0:
        return
    var io = n * d
    var so = io + len(slot[].center)
    var out = so + len(slot[].scale)
    var host = out + words
    if host > _INT32_BOUND:
        raise Error("ValueError: x_prep: the program exceeds the native Int32 indexing bound")
    var inputs = List[XprepRunInput](capacity=3)
    var held = PrepXHeld()
    inputs.append(_x_input(call, n * d, held))
    inputs.append(XprepRunInput(io, len(slot[].center), Int(slot[].center.unsafe_ptr()), -1, False, 0, False, 0, 0, 0))
    inputs.append(XprepRunInput(so, len(slot[].scale), Int(slot[].scale.unsafe_ptr()), -1, False, 0, False, 0, 0, 0))
    var program = List[Int32]()
    _stage(program, OP_POLY, words, 0, n, d, io, so, nout, out)
    var arena = List[Float32](length=host, fill=Float32(0))
    _execute(call, arena, host, 0, 0, 0, program, inputs)
    _ = held^  # the fold words X's record addresses live past the run
    unsafe_memcpy(dest=_F32P(unsafe_from_address=call[].dst.data),
        src=_F32P(unsafe_from_address=Int(arena.unsafe_ptr())) + out, count=words)
    _ = arena^


def _fit_quantile(slot: _Slot, call: _Call) raises:
    """The door's `_quantile_fit`: the optional keyed row subsample
    (f2_perm_rows, p2m_rgather), then `_quantile_program`."""
    _fit_begin(slot, call)
    var n_all = _rows(call)
    var d = call[].x.cols
    var n = n_all
    var subsample = Int(slot[].words[3])
    var permute = subsample >= 0 and n_all > subsample
    if permute:
        n = subsample
    var nq = max(1, min(Int(slot[].words[2]), n))
    var layout = PrepLayout(n_all * d, slot[].words[7] != 0.0)
    var program = List[Int32]()
    var inputs = List[XprepRunInput](capacity=3)
    var held = PrepXHeld()
    inputs.append(_x_input(call, n_all * d, held))
    var xo = 0
    # The uint64 seed's (low, high) 32-bit words (`_seed_words`).
    var seed_words = List[Float32](length=2, fill=Float32(0))
    if permute:
        var rows = layout.alloc(n)
        var sample = layout.alloc(n * d)
        var so = layout.alloc(2)
        var bits = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=Int(seed_words.unsafe_ptr()))
        bits[0] = UInt32(Int(slot[].words[4]))
        bits[1] = UInt32(Int(slot[].words[5]))
        inputs.append(XprepRunInput(so, 2, Int(seed_words.unsafe_ptr()), -1, False, 0, False, 0, 0, 0))
        _stage(program, OP_PERM_ROWS, n, so, n_all, rows)
        _stage(program, OP_RGATHER, n * d, xo, d, rows, sample)
        xo = sample
    var placed = _quantile_program(program, layout, Int(slot[].words[6]), xo, n, d, nq)
    var grid = placed[0]
    var nb = placed[1]
    var qo = placed[2]
    var qt = placed[3]
    var level: List[Float32] = [Float32(nq - 1)]
    inputs.append(XprepRunInput(nb, 1, Int(level.unsafe_ptr()), -1, False, 0, False, 0, 0, 0))
    var arena = _run_layout(call, program, layout, inputs)
    _ = held^  # the fold words X's record addresses live past the run
    _ = seed_words^
    _ = level^
    slot[].clear()
    slot[].center = _take(arena, qo, nq * d)
    slot[].scale = _take(arena, grid, nq)
    slot[].extra = _take(arena, qt, nq * d)
    _ = arena^
    slot[].features = d
    slot[].scalars = [Float64(d), Float64(nq)]
    slot[].fitted = True
    if call[].fit_output == MEMBER_OP_TRANSFORM:
        _apply_quantile(slot, call, False)


def _apply_quantile(slot: _Slot, call: _Call, inverse: Bool) raises:
    """The door's `_quantile_apply`: one qt_apply / qt_inverse stage."""
    _apply_begin(slot, call, slot[].features)
    var n = _rows(call)
    var d = slot[].features
    var nq = Int(slot[].scalars[1])
    if len(slot[].center) != nq * d or len(slot[].scale) != nq:
        raise Error("RuntimeError: QuantileTransformer fitted quantiles do not match n_quantiles_")
    var normal = slot[].words[1] != 0.0
    var words = n * d
    var qo = words
    var ro = qo + nq * d
    var out = ro + nq
    var dev_out = words >= OUT_MIN_WORDS and (call[].route.flags & PREP_ROUTE_OUT) != 0
    var host = out + (0 if dev_out else words)
    if host + (words if dev_out else 0) > _INT32_BOUND:
        raise Error("ValueError: x_prep: the program exceeds the native Int32 indexing bound")
    var inputs = List[XprepRunInput](capacity=3)
    var held = PrepXHeld()
    inputs.append(_x_input(call, words, held))
    inputs.append(XprepRunInput(qo, nq * d, Int(slot[].center.unsafe_ptr()), -1, False, 0, False, 0, 0, 0))
    inputs.append(XprepRunInput(ro, nq, Int(slot[].scale.unsafe_ptr()), -1, False, 0, False, 0, 0, 0))
    var program = List[Int32]()
    _stage(program, OP_QT_INVERSE if inverse else OP_QT_APPLY, words, 0, n, d, qo, nq, ro, 1 if normal else 0, out)
    var arena = List[Float32](length=max(host, 1), fill=Float32(0))
    # A large output comes back through the image's output region straight
    # into the caller's buffer (the door's output store).
    _execute(call, arena, host, 0, call[].dst.data if dev_out else 0, words if dev_out else 0, program, inputs)
    _ = held^  # the fold words X's record addresses live past the run
    if not dev_out:
        unsafe_memcpy(dest=_F32P(unsafe_from_address=call[].dst.data),
            src=_F32P(unsafe_from_address=Int(arena.unsafe_ptr())) + out, count=words)
    _ = arena^


def _fit_power(slot: _Slot, call: _Call) raises:
    """The door's `_power_fit`: `_power_program` with the words its door
    reads per fit (family word, fast flags, host-column binding, tier),
    then the Box-Cox refusal on the column minima, in the door's words."""
    _fit_begin(slot, call)
    var n = _rows(call)
    var d = call[].x.cols
    var method = Int(slot[].words[1])
    var standardize = slot[].words[2] != 0.0
    var layout = PrepLayout(n * d, slot[].words[7] != 0.0)
    var program = List[Int32]()
    var placed = _power_program(program, layout, Int(slot[].words[3]), Int(slot[].words[4]), slot[].words[5] != 0.0,
        slot[].words[6] != 0.0, n, d, method, standardize)
    var st = placed[0]
    var lam = placed[1]
    var anchor = placed[2]
    var anchor_kind = placed[3]
    var mean = placed[4]
    var scale = placed[5]
    var centered = placed[6]
    var inputs = List[XprepRunInput](capacity=1)
    var held = PrepXHeld()
    inputs.append(_x_input(call, n * d, held))
    var arena = _run_layout(call, program, layout, inputs)
    _ = held^  # the fold words X's record addresses live past the run
    if method == 1:
        var base = _F32P(unsafe_from_address=Int(arena.unsafe_ptr()))
        for j in range(d):  # driver-loop: over the per-column minima (feature metadata)
            if base[st + 3 * d + j] <= 0:
                raise Error("mojolearn: The Box-Cox transformation can only be applied to strictly positive data")
    slot[].clear()
    slot[].extra = _take(arena, lam, d)
    if centered and standardize:
        slot[].anchor = _take(arena, anchor, d)
        slot[].anchor_kind = _take(arena, anchor_kind, d)
    if standardize:
        slot[].center = _take(arena, mean, d)
        slot[].scale = _take(arena, scale, d)
    _ = arena^
    slot[].features = d
    slot[].scalars = [Float64(d)]
    slot[].fitted = True
    if call[].fit_output == MEMBER_OP_TRANSFORM:
        _apply_power(slot, call, False)


def _apply_power(slot: _Slot, call: _Call, inverse: Bool) raises:
    """The door's `_power_apply`: one pt_apply / pt_inverse stage over the
    fitted lambdas, standardization and anchors, in the door's layout."""
    _apply_begin(slot, call, slot[].features)
    var n = _rows(call)
    var d = slot[].features
    var standardize = len(slot[].center) > 0
    var anchored = len(slot[].anchor) > 0
    var words = n * d
    var inputs = List[XprepRunInput](capacity=6)
    var held = PrepXHeld()
    inputs.append(_x_input(call, words, held))
    var host = words
    var lo = host
    host += d
    inputs.append(XprepRunInput(lo, d, Int(slot[].extra.unsafe_ptr()), -1, False, 0, False, 0, 0, 0))
    var mo = -1
    var so = -1
    if standardize:
        mo = host
        host += d
        so = host
        host += d
        inputs.append(XprepRunInput(mo, d, Int(slot[].center.unsafe_ptr()), -1, False, 0, False, 0, 0, 0))
        inputs.append(XprepRunInput(so, d, Int(slot[].scale.unsafe_ptr()), -1, False, 0, False, 0, 0, 0))
    var dev_out = words >= OUT_MIN_WORDS and (call[].route.flags & PREP_ROUTE_OUT) != 0
    var out = host
    if not dev_out:
        host += words
    var ao = -1
    var ko = -1
    if anchored:
        ao = host
        host += d
        ko = host
        host += d
        inputs.append(XprepRunInput(ao, d, Int(slot[].anchor.unsafe_ptr()), -1, False, 0, False, 0, 0, 0))
        inputs.append(XprepRunInput(ko, d, Int(slot[].anchor_kind.unsafe_ptr()), -1, False, 0, False, 0, 0, 0))
    if host + (words if dev_out else 0) > _INT32_BOUND:
        raise Error("ValueError: x_prep: the program exceeds the native Int32 indexing bound")
    if dev_out:
        out = host
    var program = List[Int32]()
    _stage(program, OP_PT_INVERSE if inverse else OP_PT_APPLY, words, 0, n, d, lo, Int(slot[].words[1]), mo, so, out,
        ao, ko)
    var arena = List[Float32](length=max(host, 1), fill=Float32(0))
    _execute(call, arena, host, 0, call[].dst.data if dev_out else 0, words if dev_out else 0, program, inputs)
    _ = held^  # the fold words X's record addresses live past the run
    if not dev_out:
        unsafe_memcpy(dest=_F32P(unsafe_from_address=call[].dst.data),
            src=_F32P(unsafe_from_address=Int(arena.unsafe_ptr())) + out, count=words)
    _ = arena^


def _enc_run(call: _Call, mut prog: EncProgram, second: Int, output: Int, scoped: Bool = True) raises -> List[Float32]:
    """`_enc_run` with one second buffer (owner 1)."""
    var seconds: List[Int] = [second]
    return _enc_run(call, prog, seconds, output, scoped)


def _enc_run(call: _Call, mut prog: EncProgram, seconds: List[Int], output: Int, scoped: Bool = True) raises -> List[Float32]:
    """The encode door's `EncProg.run_caps` over the member's own input
    records: owner 0 is the caller's X (`_x_record`: resident in the caller's
    scope, gathered by the fold's row index, or put on the device when the
    image has the put and X is large), owner k >= 1 the buffer at
    `seconds[k - 1]` (a fitted vector, y, the weights), -1 the program's own words. `output`: the caller's buffer for the
    program's output region (0: none). `scoped` False: a member without
    MEMBER_CAN_SCOPE (the Select family) never reads the caller's scope or
    pin. Returns the host arena."""
    prog.check_bound()
    var held = PrepXHeld()
    var inputs = List[XprepRunInput](capacity=len(prog.inputs))
    for i in range(len(prog.inputs)):  # driver-loop: over the program's input records
        var record = prog.inputs[i]
        if record.owner == 0:
            # X: a resident or gathered record never for an input a stage
            # rewrites (the door's rule: in-out inputs are copied up).
            var inout = False
            for j in range(len(prog.inout_ranges) // 2):  # driver-loop: over the program's in-out input ranges
                if prog.inout_ranges[2 * j] == record.offset and prog.inout_ranges[2 * j + 1] == record.offset + record.words:
                    inout = True
            inputs.append(_x_record(call, record.offset, record.words, held, inout, scoped))
        else:
            var address = record.address if record.owner < 0 else seconds[record.owner - 1]
            inputs.append(XprepRunInput(record.offset, record.words, address, -1, False, 0, False, 0, 0, 0))
    var host = prog.host
    var scratch = prog.scratch_size
    var device_scratch = scratch > 0 and (call[].route.flags & (PREP_ROUTE_OUT | PREP_ROUTE_SCRATCH)) != 0
    var arena = List[Float32](length=max(host + (0 if device_scratch else scratch), 1), fill=Float32(0))
    _execute_ranges(call, arena, host, scratch, output if prog.out_words > 0 else 0, prog.out_words,
        prog.stage_words(), inputs, prog.inout_ranges)
    _ = held^  # the fold words X's record addresses live past the run
    return arena^


def _iword(arena: List[Float32], offset: Int) -> Int:
    # An int32 word a stage wrote into the arena (the door's `_iword`).
    return Int(MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(arena.unsafe_ptr()))[offset])


def _fold_weights(call: _Call) raises -> List[Float32]:
    """The weights of the fold's rows (`call.row_index` over the whole X's
    weights), bounds-checked, split across the host's workers."""
    var n = call[].row_index.rows
    var base = call[].weights.rows
    var out = List[Float32](length=n, fill=Float32(0))
    var tasks = host_predict_task_count_work(n, n)
    var chunk = host_predict_chunk(n, tasks)
    var bad = List[Int](length=tasks, fill=0)
    var flags = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=Int(bad.unsafe_ptr()))
    var wide = call[].row_index.dtype == MEMBER_I64
    var source = call[].row_index.data
    var weights = _F32P(unsafe_from_address=call[].weights.data)
    var into = _F32P(unsafe_from_address=Int(out.unsafe_ptr()))

    def task(worker: Int) {imm chunk, imm n, imm base, imm wide, imm source, imm flags, imm weights, imm into}:
        var begin = worker * chunk
        var end = min(begin + chunk, n)
        var out_of_range = 0
        for r in range(begin, end):
            var row = Int(MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=source)[r]) if wide else Int(
                MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=source)[r])
            if row < 0 or row >= base:
                out_of_range = 1
                continue
            into[r] = weights[row]
        flags[worker] = out_of_range

    host_parallelize(task, tasks)
    for w in range(tasks):  # driver-loop: over the workers' range flags
        if bad[w] != 0:
            raise Error("IndexError: a row index is out of range for X's " + String(base) + " rows")
    return out^


def _kbins_method(strategy: Int) -> String:
    # The quantile_method a strategy code stands for (the door's `_kbins_strategy`).
    if strategy == 1:
        return String("averaged_inverted_cdf")
    if strategy == 2:
        return String("linear")
    if strategy == 4:
        return String("inverted_cdf")
    if strategy == 5:
        return String("closest_observation")
    if strategy == 6:
        return String("interpolated_inverted_cdf")
    if strategy == 7:
        return String("hazen")
    if strategy == 8:
        return String("weibull")
    if strategy == 9:
        return String("median_unbiased")
    return String("normal_unbiased")


def _kbins_widths(slot: _Slot) -> List[Int]:
    # n_bins_: each column's edge count less one.
    var widths = List[Int](capacity=len(slot[].scale))
    for j in range(len(slot[].scale)):  # driver-loop: over the fitted per-column edge counts (feature metadata)
        widths.append(Int(slot[].scale[j]) - 1)
    return widths^


def _kbins_cols(slot: _Slot) -> Int:
    # transform's width: the codes (ordinal) or the one-hot blocks.
    if slot[].words[1] == Float64(PREP_KBINS_ORDINAL):
        return slot[].features
    var total = 0
    for j in range(len(slot[].scale)):  # driver-loop: over the fitted per-column edge counts (feature metadata)
        total += Int(slot[].scale[j]) - 1
    return total


def _fit_kbins(slot: _Slot, call: _Call) raises:
    """The encode door's `_kbins_fit` (bindings/prep_encode_native_door.mojo):
    `_check_weights`' program on the weights when given, then
    `kbins_fit_stage` over X (subsample drawn in the program), in the door's
    order and words. The weights of a fold are its rows of the whole X's."""
    var estimator = _name(slot)
    _whole(call, estimator)
    _dense(call[].x, "X", estimator)
    if call[].fit_output != 0 and call[].fit_output != MEMBER_OP_TRANSFORM:
        member_unsupported(estimator, call[].fit_output)
    var n = _rows(call)
    var d = call[].x.cols
    var w = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(slot[].words.unsafe_ptr()))
    var strat = Int(w[2])
    var blocked = w[6] != 0.0
    var work = w[7] != 0.0
    var given = call[].weights.rows > 0
    var fold_weights = List[Float32]()
    var weights_at = 0
    var positive = 0
    if given:
        if not call[].weights.dense_host(MEMBER_F32) or call[].weights.cols != 1:
            raise Error("member ABI: a KBinsDiscretizer member reads sample_weight as a dense host float32 vector")
        if call[].weights.rows != call[].x.rows:
            raise Error("mojolearn: sample_weight has " + String(call[].weights.rows) + " entries; X has "
                        + String(call[].x.rows) + " rows")
        weights_at = call[].weights.data
        if call[].row_index.rows > 0:
            fold_weights = _fold_weights(call)
            weights_at = Int(fold_weights.unsafe_ptr())
        var check = EncProgram(work)
        var plan = enc_weights_stage(check, 1, n, blocked)
        var ran = _enc_run(call, check, weights_at, 0)
        var ap = _F32P(unsafe_from_address=Int(ran.unsafe_ptr()))
        var count = ap[plan.stats]
        var low = ap[plan.stats + 3]
        positive = _iword(ran, plan.positives)
        _ = ran^
        _ = len(check.consts_f)  # the program's own input words live past the run
        # a NaN is not counted by col_stats
        if Int(count) != n or low < 0:
            raise Error("mojolearn: KBinsDiscretizer sample_weight must be nonnegative")
        if positive == 0:
            raise Error("mojolearn: KBinsDiscretizer sample_weight is all zero")
    var subsample = Int(w[3])
    var resample = subsample >= 0 and n > subsample
    var draw = KB_DRAW_NONE
    var draws = 0
    if resample:
        draws = subsample
        # a weighted resample with replacement is drawn in the fit's program,
        # and the edges of its rows are then unweighted
        draw = KB_DRAW_WEIGHTED if given else KB_DRAW_PLAIN
    var weighted = given and not resample
    var per_column = Int(w[8])
    var bins = List[Int](capacity=d)
    if per_column == 0:
        for _ in range(d):  # driver-loop: over the per-column n_bins argument
            bins.append(Int(w[9]))
    else:
        for j in range(per_column):  # driver-loop: over the per-column n_bins argument
            bins.append(Int(w[PREP_KBINS_HEADER + j]))
    var smallest = bins[0] if len(bins) > 0 else 0
    var nbmax = 0
    for j in range(len(bins)):  # driver-loop: over the per-column n_bins argument
        smallest = min(smallest, bins[j])
        nbmax = max(nbmax, bins[j])
    if len(bins) != d or smallest < 2:
        raise Error("mojolearn: n_bins must be >= 2 per feature")
    if weighted and strat != 0 and strat != 1 and strat != 3 and strat != 4:
        raise Error("mojolearn: When fitting with strategy='quantile' and sample weights, quantile_method "
                    + "should either be set to 'averaged_inverted_cdf' or 'inverted_cdf', got "
                    + "quantile_method='" + _kbins_method(strat) + "' instead.")
    var prog = EncProgram(work)
    var request = KBinsFit(n, d, 0, strat, bins^, nbmax, draw, draws, Int(w[4]), Int(w[5]), weighted,
                           1 if given else -1, positive, blocked)
    var plan = kbins_fit_stage(prog, request)
    var arena = _enc_run(call, prog, weights_at, 0)
    _ = fold_weights^  # the fold's weights live past the run
    _ = len(prog.consts_f)  # the program's own input words live past the run
    var stride = nbmax + 1
    slot[].clear()
    slot[].center = _take(arena, plan.edges, d * stride)
    slot[].scale = _take(arena, plan.counts, d)
    _ = arena^
    slot[].features = d
    slot[].scalars = [Float64(d), Float64(stride)]
    slot[].fitted = True
    if call[].fit_output == MEMBER_OP_TRANSFORM:
        _transform_kbins(slot, call)


def _transform_kbins(slot: _Slot, call: _Call) raises:
    """The encode door's `_kbins_transform`: `kbins_transform_stage` (the
    codes, then the one-hot blocks), its output region straight into the
    caller's dst when the image has one and the output is large."""
    var cols = _kbins_cols(slot) if slot[].fitted else 0
    _apply_begin(slot, call, cols)
    var n = _rows(call)
    var d = slot[].features
    if n * cols == 0:
        return
    var onehot = slot[].words[1] != Float64(PREP_KBINS_ORDINAL)
    var widths = _kbins_widths(slot) if onehot else List[Int]()
    var prog = EncProgram(slot[].words[7] != 0.0)
    var request = KBinsApply(n, d, 0, 1, len(slot[].center), 2, len(slot[].scale), Int(slot[].scalars[1]), onehot,
                             widths^)
    var result = kbins_transform_stage(prog, request, (call[].route.flags & PREP_ROUTE_OUT) != 0)
    var seconds: List[Int] = [Int(slot[].center.unsafe_ptr()), Int(slot[].scale.unsafe_ptr())]
    var arena = _enc_run(call, prog, seconds, call[].dst.data)
    if prog.out_words == 0:
        unsafe_memcpy(dest=_F32P(unsafe_from_address=call[].dst.data),
            src=_F32P(unsafe_from_address=Int(arena.unsafe_ptr())) + result, count=n * cols)
    _ = arena^
    _ = len(prog.consts_f)  # the program's own input words live past the run


def _inverse_kbins(slot: _Slot, call: _Call) raises:
    """The encode door's `_kbins_inverse`: codes (ordinal) or one-hot blocks
    back to the bin centers; rows naming no bin refused in the door's words
    (the first ten)."""
    var estimator = _name(slot)
    if not slot[].fitted:
        raise Error("RuntimeError: mojolearn: this " + estimator + " instance is not fitted yet")
    _whole(call, estimator)
    _dense(call[].x, "X", estimator)
    _dense(call[].dst, "the output buffer", estimator)
    var d = slot[].features
    var n = _rows(call)
    var cols = call[].x.cols
    var ordinal = slot[].words[1] == Float64(PREP_KBINS_ORDINAL)
    var why = String("hold codes that name no bin")
    if ordinal:
        if cols != d:
            raise Error("mojolearn: X has " + String(cols) + " features, " + estimator + " was fitted with "
                        + String(d))
    else:
        var total = _kbins_cols(slot)
        if cols != total:
            raise Error("mojolearn: X has " + String(cols) + " columns, expected " + String(total))
        why = "can not be inverted because they contain all zeros"
    if call[].dst.rows != n or call[].dst.cols != d:
        raise Error("ValueError: the output buffer holds " + String(d) + " values per row of X")
    if n == 0:
        return
    var prog = EncProgram(slot[].words[7] != 0.0)
    var request = KBinsInverse(n, cols, 0, ordinal, _kbins_widths(slot), 1, len(slot[].center), Int(slot[].scalars[1]))
    var plan = kbins_inverse_stage(prog, request, (call[].route.flags & PREP_ROUTE_OUT) != 0)
    var arena = _enc_run(call, prog, Int(slot[].center.unsafe_ptr()), call[].dst.data)
    _ = len(prog.consts_f)  # the program's own input words live past the run
    var k = _iword(arena, plan.bad_count)
    if k > 0:
        var text = String("[")
        for i in range(min(k, 10)):  # driver-loop: over at most ten reported rows
            text += (", " if i > 0 else "") + String(_iword(arena, plan.bad_rows + i))
        raise Error("mojolearn: samples " + text + "] " + why)
    if prog.out_words == 0:
        unsafe_memcpy(dest=_F32P(unsafe_from_address=call[].dst.data),
            src=_F32P(unsafe_from_address=Int(arena.unsafe_ptr())) + plan.centers, count=n * d)
    _ = arena^


def _index_words(values: List[Float32]) -> List[Int]:
    var out = List[Int](capacity=len(values))
    for i in range(len(values)):  # driver-loop: over fitted column indices (feature metadata)
        out.append(Int(values[i]))
    return out^


def _fit_imputer(slot: _Slot, call: _Call) raises:
    """The door's `_imputer_fit` over the caller's X: `imputer_fit_stage`,
    then the kept and indicator columns from the per-column counts."""
    _fit_begin(slot, call)
    var n = _rows(call)
    var d = call[].x.cols
    var w = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(slot[].words.unsafe_ptr()))
    var strategy = Int(w[1])
    var konst = strategy == IMP_CONSTANT
    var keep_empty = w[7] != 0.0
    var indicators = w[8] != 0.0
    var prog = EncProgram(w[9] != 0.0)
    var request = ImputerFit(n, d, 0, w[2] != 0.0, w[3], strategy, w[4] != 0.0, w[5] != 0.0, w[6], keep_empty)
    var plan = imputer_fit_stage(prog, request)
    var arena = _enc_run(call, prog, 0, 0)
    var ap = _F32P(unsafe_from_address=Int(arena.unsafe_ptr()))
    var any_empty = False
    for j in range(d):  # driver-loop: over the per-column non-missing counts (feature metadata)
        any_empty = any_empty or Int(ap[plan.stats + j]) == 0
    slot[].clear()
    var shared = not (konst or any_empty)
    if shared:
        slot[].center = _take(arena, plan.source, d)
        slot[].scale = slot[].center.copy()
    else:
        slot[].center = _take(arena, plan.statistics, d)
        slot[].scale = _take(arena, plan.fill, d)
    for j in range(d):  # driver-loop: over the kept columns (feature metadata)
        if keep_empty or Int(ap[plan.stats + j]) != 0:
            slot[].extra.append(Float32(j))
    if indicators:
        for j in range(d):  # driver-loop: over the indicator columns (feature metadata)
            if Int(ap[plan.stats + j]) < n:
                slot[].anchor.append(Float32(j))
    _ = arena^
    _ = len(prog.consts_f)  # the program's own input words live past the run
    slot[].features = d
    slot[].scalars = [Float64(d), 1.0 if shared else 0.0]
    slot[].fitted = True
    if call[].fit_output == MEMBER_OP_TRANSFORM:
        _transform_imputer(slot, call)


def _transform_imputer(slot: _Slot, call: _Call) raises:
    """The door's `_imputer_transform`: `imputer_transform_stage` (the filled
    kept columns, then the indicator columns), its output region straight
    into the caller's dst when the image has one and the output is large."""
    var cols = len(slot[].extra) + len(slot[].anchor)
    _apply_begin(slot, call, cols)
    var n = _rows(call)
    var d = slot[].features
    if n * cols == 0:
        return
    var prog = EncProgram(slot[].words[9] != 0.0)
    var request = ImputerApply(n, d, 0, slot[].words[2] != 0.0, slot[].words[3], 1, len(slot[].scale),
        _index_words(slot[].extra), _index_words(slot[].anchor))
    var region = (call[].route.flags & PREP_ROUTE_OUT) != 0
    var out = imputer_transform_stage(prog, request, region)
    var arena = _enc_run(call, prog, Int(slot[].scale.unsafe_ptr()), call[].dst.data)
    if prog.out_words == 0:
        unsafe_memcpy(dest=_F32P(unsafe_from_address=call[].dst.data),
            src=_F32P(unsafe_from_address=Int(arena.unsafe_ptr())) + out, count=n * cols)
    _ = arena^
    _ = len(prog.consts_f)  # the program's own input words live past the run


def _block_view(slot: _Slot, block: Int) raises -> MemberMatrix:
    if block == PREP_BLOCK_CENTER:
        return MemberMatrix(Int(slot[].center.unsafe_ptr()), 1, len(slot[].center))
    if block == PREP_BLOCK_SCALE:
        return MemberMatrix(Int(slot[].scale.unsafe_ptr()), 1, len(slot[].scale))
    if block == PREP_BLOCK_EXTRA:
        return MemberMatrix(Int(slot[].extra.unsafe_ptr()), 1, len(slot[].extra))
    if block == PREP_BLOCK_ANCHOR:
        return MemberMatrix(Int(slot[].anchor.unsafe_ptr()), 1, len(slot[].anchor))
    return MemberMatrix(Int(slot[].anchor_kind.unsafe_ptr()), 1, len(slot[].anchor_kind))


def _export(slot: _Slot, call: _Call) raises:
    """A view of one fitted block. The slot owns the words; they stay valid
    until this state's next fit, import or free."""
    if not slot[].fitted:
        raise Error("NotFittedError: this " + _name(slot) + " member is not fitted yet")
    var count = _block_count(slot[].kind())
    var block = call[].block
    if (call[].flags & MEMBER_BLOCK_BY_INDEX) != 0:
        if block < 0 or block >= count:
            raise Error("member ABI: a " + _name(slot) + " member has no fitted block index " + String(block))
        block = MEMBER_BLOCK_SCALARS if block == 0 else PREP_BLOCK_CENTER + block - 1
    if block == MEMBER_BLOCK_SCALARS:
        call[].dst = MemberMatrix(Int(slot[].scalars.unsafe_ptr()), 1, len(slot[].scalars), MEMBER_F64)
    elif block >= PREP_BLOCK_CENTER and block < PREP_BLOCK_CENTER + count - 1:
        call[].dst = _block_view(slot, block)
    else:
        raise Error("member ABI: a " + _name(slot) + " member has no fitted block " + String(block))
    call[].block = block
    call[].out_cols = count


def _import(slot: _Slot, call: _Call) raises:
    """Copy one fitted block in; MEMBER_IMPORT_DONE completes the state."""
    var estimator = _name(slot)
    var kind = slot[].kind()
    var block = call[].block
    if slot[].fitted:
        slot[].clear()
    var count = _block_count(kind)
    if block == MEMBER_BLOCK_SCALARS:
        var scalar_count = _scalar_count(kind)
        if not call[].x.dense_host(MEMBER_F64) or call[].x.rows * call[].x.cols != scalar_count:
            raise Error("member ABI: the " + estimator + " scalars block is " + String(scalar_count)
                        + " host float64 words")
        var given = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call[].x.data)
        slot[].scalars = List[Float64](length=scalar_count, fill=0.0)
        for index in range(scalar_count):  # driver-loop: one or two scalar words
            slot[].scalars[index] = given[index]
    elif block >= PREP_BLOCK_CENTER and block < PREP_BLOCK_CENTER + count - 1:
        var size = call[].x.rows * call[].x.cols
        var words = List[Float32](length=size, fill=Float32(0))
        if size > 0:
            _dense(call[].x, "a fitted block", estimator)
            unsafe_memcpy(dest=words.unsafe_ptr(), src=_F32P(unsafe_from_address=call[].x.data), count=size)
        if block == PREP_BLOCK_CENTER:
            slot[].center = words^
        elif block == PREP_BLOCK_SCALE:
            slot[].scale = words^
        elif block == PREP_BLOCK_EXTRA:
            slot[].extra = words^
        elif block == PREP_BLOCK_ANCHOR:
            slot[].anchor = words^
        else:
            slot[].anchor_kind = words^
    elif block != 0 or (call[].flags & MEMBER_IMPORT_DONE) == 0:
        raise Error("member ABI: a " + estimator + " member has no fitted block " + String(block))
    if (call[].flags & MEMBER_IMPORT_DONE) == 0:
        return
    if len(slot[].scalars) != _scalar_count(kind):
        raise Error("member ABI: a fitted " + estimator + " state needs its scalars block")
    var d = Int(slot[].scalars[0])
    var complete = d >= 1
    if kind == PREP_KIND_MAXABS:
        complete = complete and len(slot[].center) == d and len(slot[].scale) == d
    elif kind == PREP_KIND_POLYNOMIAL:
        var nout = Int(slot[].scalars[1])
        complete = complete and nout >= 0 and len(slot[].scale) == nout + 1 and len(slot[].center) >= 1
    elif kind == PREP_KIND_QUANTILE:
        var nq = Int(slot[].scalars[1])
        complete = (complete and nq >= 1 and len(slot[].center) == nq * d and len(slot[].scale) == nq
                    and len(slot[].extra) == nq * d)
    elif kind == PREP_KIND_IMPUTER:
        complete = (complete and len(slot[].center) == d and len(slot[].scale) == d and len(slot[].extra) <= d
                    and len(slot[].anchor) <= d)
    elif kind == PREP_KIND_KBINS:
        var stride = Int(slot[].scalars[1])
        complete = complete and stride >= 2 and len(slot[].center) == d * stride and len(slot[].scale) == d
        for j in range(len(slot[].scale)):  # driver-loop: over the fitted per-column edge counts
            complete = complete and Int(slot[].scale[j]) >= 2 and Int(slot[].scale[j]) <= stride
    elif kind == PREP_KIND_POWER:
        complete = (complete and len(slot[].extra) == d and len(slot[].center) == len(slot[].scale)
                    and (len(slot[].center) == 0 or len(slot[].center) == d)
                    and len(slot[].anchor) == len(slot[].anchor_kind)
                    and (len(slot[].anchor) == 0 or len(slot[].anchor) == d))
    else:
        complete = (complete and (len(slot[].center) == 0 or len(slot[].center) == d)
                    and (len(slot[].scale) == 0 or len(slot[].scale) == d))
    if not complete:
        raise Error("member ABI: a fitted " + estimator + " state needs every fitted block, of the fitted sizes")
    slot[].features = d
    slot[].fitted = True


def _outputs(call: _Call) raises:
    """`call.out_cols` of transform (and inverse_transform where the kind has
    it): X's width, or PolynomialFeatures' term count."""
    call[].classes = 0
    # Every transform / inverse_transform of the family writes float32.
    call[].out_dtype = MEMBER_F32
    if call[].state == 0:
        # Asked before any state (`member_outputs`): answered from the words.
        var kind = PREP_KIND_ROBUST
        var source = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call[].params)
        if call[].param_count >= 1 and call[].params:
            kind = Int(source[0])
        if call[].flags != MEMBER_OP_TRANSFORM and not (call[].flags == MEMBER_OP_INVERSE_TRANSFORM and _invertible(kind)):
            member_unsupported(_kind_name(kind), call[].flags)
        if kind == PREP_KIND_POLYNOMIAL:
            if call[].param_count != PREP_POLY_PARAM_COUNT:
                raise Error("member ABI: a PolynomialFeatures member takes five parameters")
            var words = List[Float64](length=PREP_POLY_PARAM_COUNT, fill=0.0)
            for w in range(PREP_POLY_PARAM_COUNT):  # driver-loop: the parameter words
                words[w] = source[w]
            var index = List[Int]()
            var start = List[Int]()
            _poly_table(words, call[].x.cols, index, start)
            call[].out_cols = len(start) - 1
            return
        if kind == PREP_KIND_KBINS and call[].flags == MEMBER_OP_TRANSFORM:
            # ordinal codes keep X's width; one-hot blocks are as wide as the
            # fitted bins (duplicate edges are dropped by the fit).
            if call[].param_count < PREP_KBINS_HEADER or source[1] != Float64(PREP_KBINS_ORDINAL):
                raise Error("member ABI: a one-hot KBinsDiscretizer member's output width is known after its fit")
        if kind == PREP_KIND_IMPUTER:
            # keep_empty_features without add_indicator keeps every column;
            # otherwise the width is chosen by the fit.
            if call[].param_count != PREP_IMPUTER_PARAM_COUNT or source[7] == 0.0 or source[8] != 0.0:
                raise Error("member ABI: a SimpleImputer member's output width is known after its fit")
        call[].out_cols = call[].x.cols
        return
    var slot = _Slot(unsafe_from_address=_lookup(call[].state, False))
    if call[].flags != MEMBER_OP_TRANSFORM and not (call[].flags == MEMBER_OP_INVERSE_TRANSFORM
                                                    and _invertible(slot[].kind())):
        member_unsupported(_name(slot), call[].flags)
    if not slot[].fitted:
        raise Error("NotFittedError: this " + _name(slot) + " member's output width is known after its fit")
    if slot[].kind() == PREP_KIND_POLYNOMIAL and call[].flags == MEMBER_OP_TRANSFORM:
        call[].out_cols = Int(slot[].scalars[1])
        return
    if slot[].kind() == PREP_KIND_IMPUTER:
        call[].out_cols = len(slot[].extra) + len(slot[].anchor)
        return
    if slot[].kind() == PREP_KIND_KBINS and call[].flags == MEMBER_OP_TRANSFORM:
        call[].out_cols = _kbins_cols(slot)
        return
    call[].out_cols = slot[].features


def _new_count(kind: Int) raises -> Int:
    if kind == PREP_KIND_ROBUST:
        return PREP_ROBUST_PARAM_COUNT
    if kind == PREP_KIND_MAXABS:
        return PREP_MAXABS_PARAM_COUNT
    if kind == PREP_KIND_BINARIZER or kind == PREP_KIND_NORMALIZER:
        return PREP_STATELESS_PARAM_COUNT
    if kind == PREP_KIND_POLYNOMIAL:
        return PREP_POLY_PARAM_COUNT
    if kind == PREP_KIND_QUANTILE:
        return PREP_QUANTILE_PARAM_COUNT
    if kind == PREP_KIND_POWER:
        return PREP_POWER_PARAM_COUNT
    if kind == PREP_KIND_IMPUTER:
        return PREP_IMPUTER_PARAM_COUNT
    if kind == PREP_KIND_KBINS:
        return PREP_KBINS_HEADER  # plus words[8] per-column n_bins words
    raise Error("member ABI: unknown or unpublished Prep member kind")


def _prep_member(operation: Int, address: Int) raises -> Int:
    var call = _Call(unsafe_from_address=address)
    if operation == MEMBER_OP_CAPABILITIES:
        # Every kind reads a fold's row index of the whole X and keeps a
        # pinned X resident in the caller's scope (`_x_record`).
        var mask = (MEMBER_CAN_FIT | MEMBER_CAN_TRANSFORM | MEMBER_CAN_EXPORT | MEMBER_CAN_IMPORT
                    | MEMBER_CAN_ROW_INDEX | MEMBER_CAN_SCOPE)
        if call[].state == 0:
            return mask | MEMBER_CAN_INVERSE_TRANSFORM | MEMBER_CAN_SAMPLE_WEIGHT  # the family's union
        # Kind-exact: QuantileTransformer, PowerTransformer and
        # KBinsDiscretizer invert; only KBinsDiscretizer reads weights; no
        # kind reads a column index.
        var held = _Slot(unsafe_from_address=_lookup(call[].state, False))
        if _invertible(held[].kind()):
            mask |= MEMBER_CAN_INVERSE_TRANSFORM
        if held[].kind() == PREP_KIND_KBINS:
            mask |= MEMBER_CAN_SAMPLE_WEIGHT
        return mask
    if operation == MEMBER_OP_NEW:
        if call[].param_count < 1 or not call[].params:
            raise Error("member ABI: a Prep member takes its parameter words")
        var source = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call[].params)
        var kind = Int(source[0])
        var expected = _new_count(kind)
        if kind == PREP_KIND_KBINS and call[].param_count >= PREP_KBINS_HEADER and source[8] >= 0.0:
            expected += Int(source[8])
        if Float64(kind) != source[0] or call[].param_count != expected:
            raise Error("member ABI: a " + _kind_name(kind) + " member takes " + String(_new_count(kind))
                        + " parameter words")
        if kind == PREP_KIND_NORMALIZER and source[1] != 0.0 and source[1] != 1.0 and source[1] != 2.0:
            raise Error("member ABI: unknown Normalizer norm code")
        if kind == PREP_KIND_POLYNOMIAL and (source[1] < 0.0 or source[1] > source[2]):
            raise Error("member ABI: invalid PolynomialFeatures degree words")
        if kind == PREP_KIND_QUANTILE and (source[2] < 1.0 or (source[1] != 0.0 and source[1] != 1.0)):
            raise Error("member ABI: invalid QuantileTransformer parameter words")
        if kind == PREP_KIND_POWER and source[1] != 0.0 and source[1] != 1.0:
            raise Error("member ABI: unknown PowerTransformer method code")
        if kind == PREP_KIND_IMPUTER and (source[1] < 0.0 or source[1] > 3.0):
            raise Error("member ABI: unknown SimpleImputer strategy code")
        if kind == PREP_KIND_KBINS and (source[1] < 0.0 or source[1] > 2.0 or source[2] < 0.0 or source[2] > 10.0):
            raise Error("member ABI: invalid KBinsDiscretizer encode or strategy code")
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
    if operation == MEMBER_OP_FIT or operation == MEMBER_OP_TRANSFORM or operation == MEMBER_OP_INVERSE_TRANSFORM:
        call[].value = 0.0  # 1 when the caller's scope kept `call.pin` (`_resident`)
    if operation == MEMBER_OP_FIT:
        if kind == PREP_KIND_BINARIZER or kind == PREP_KIND_NORMALIZER:
            _fit_stateless(slot, call)
        elif kind == PREP_KIND_MAXABS:
            _fit_maxabs(slot, call)
        elif kind == PREP_KIND_POLYNOMIAL:
            _fit_poly(slot, call)
        elif kind == PREP_KIND_QUANTILE:
            _fit_quantile(slot, call)
        elif kind == PREP_KIND_POWER:
            _fit_power(slot, call)
        elif kind == PREP_KIND_IMPUTER:
            _fit_imputer(slot, call)
        elif kind == PREP_KIND_KBINS:
            _fit_kbins(slot, call)
        else:
            _fit_robust(slot, call)
    elif operation == MEMBER_OP_TRANSFORM:
        if kind == PREP_KIND_BINARIZER or kind == PREP_KIND_NORMALIZER:
            _transform_stateless(slot, call)
        elif kind == PREP_KIND_MAXABS:
            _transform_maxabs(slot, call)
        elif kind == PREP_KIND_POLYNOMIAL:
            _transform_poly(slot, call)
        elif kind == PREP_KIND_QUANTILE:
            _apply_quantile(slot, call, False)
        elif kind == PREP_KIND_POWER:
            _apply_power(slot, call, False)
        elif kind == PREP_KIND_IMPUTER:
            _transform_imputer(slot, call)
        elif kind == PREP_KIND_KBINS:
            _transform_kbins(slot, call)
        else:
            _transform_robust(slot, call)
    elif operation == MEMBER_OP_INVERSE_TRANSFORM and kind == PREP_KIND_QUANTILE:
        _apply_quantile(slot, call, True)
    elif operation == MEMBER_OP_INVERSE_TRANSFORM and kind == PREP_KIND_POWER:
        _apply_power(slot, call, True)
    elif operation == MEMBER_OP_INVERSE_TRANSFORM and kind == PREP_KIND_KBINS:
        _inverse_kbins(slot, call)
    elif operation == MEMBER_OP_EXPORT:
        _export(slot, call)
    elif operation == MEMBER_OP_IMPORT:
        _import(slot, call)
    else:
        member_unsupported(_name(slot), operation)
    return 0


def prep_member_v2(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
    result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """The member entry of the Prep family (schema `MEMBER_ABI`)."""
    try:
        var word = _prep_member(operation, member_call_from_spans(operation, aa, na, np, nf))
        if result:
            MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = word
        return 0
    except error:
        native_descriptor_report(error, context, sink)
        return 1


def prep_member_publish(mode: Int, vendor: Int, publisher: Int) raises:
    """Publish this image's Prep member entry. One line in the image's
    module initialization, after its kernel tables are published."""
    member_publish[MEMBER_FAMILY_PREP](mode, vendor, publisher, prep_member_v2)
