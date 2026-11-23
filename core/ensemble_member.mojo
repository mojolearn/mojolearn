# SPDX-License-Identifier: Apache-2.0
"""THE META MEMBER KIND (lane mf-meta3, 2026-11-23; core/member_abi.mojo,
MEMBER-ABI-V2.md): VotingClassifier, VotingRegressor, StackingClassifier and
StackingRegressor as MEMBERS of another meta-estimator, a search, a
calibration or an imputer, through the one member ABI. Family
`MEMBER_FAMILY_ENSEMBLE`, published by both x_trees images
(bindings/_mojolearn_x_trees.mojo, bindings/_mojolearn_x_trees_host.mojo)
beside their glue table. No Python: no Python object parameter, field, local
or import.

A META STATE OWNS A NESTED SET. Its nested members are ordinary native
members (any family but this one: a meta of metas keeps its doors), each
with its own typed `MemberPlan`. The plans cannot travel as Float64 words,
so the boundary's UNPACK (providers/python/member_ensemble_boundary_door.mojo)
MINTS them once per admission into the process-wide META PLAN STORE
(`meta_plan_mint`) and puts the handle in the META plan's words; the plan
entry is released by the capsule the admission keeps in the owner's pins
(bindings/meta_plan_capsule_pack.mojo), so it lives exactly as long as any
copy of the META plan can still make a state. `MEMBER_OP_NEW` COPIES the
nested plans into the new state: a state never reads the plan store again.

MEMBER_OP_NEW parameters (Float64 words, count `META_PARAM_COUNT`):
[0] kind (`META_VOTE_CLASSIFIER`, `META_VOTE_REGRESSOR`,
`META_STACK_CLASSIFIER`, `META_STACK_REGRESSOR`), [1] the plan store handle,
[2] soft voting (vote) / passthrough (stack), 1.0 or 0.0, [3] the stacking
cv fold count (0 for a vote), [4] the worker bound (the estimator's
`n_jobs`, 0: none), [5] reserved (0).

The plan entry holds, in member order: the nested plans (a stack's final
estimator LAST), the vote weights (one per voting member), the stack kinds
(`STACK_PREDICT` / `_PROBA` / `_DECISION`, one per stacked member) and each
nested KIND's capability mask (asked once by the admission).

FIT (`MEMBER_OP_FIT`, x float32 dense host rows x features, y one column:
int32 class codes 0 .. classes-1 for a classifier kind, float32 targets for
a regressor kind, optional float32 weights handed to every nested fit):
  * vote: the voting members as ONE native member set over the META's X
    (core/member_set_run.mojo `member_set_fit`: the GIL-free kinds on host
    workers), exactly the set VotingClassifier / VotingRegressor.fit makes
    (bindings/trees_public_meta_door.mojo `_voting_fit`);
  * stack (`_stacking_fit`, same steps, same order): the members on every
    row (one set); the width probe over the first rows
    (xtrees/trees_meta_vote_run.mojo `stack_place_run`, no meta block); the
    cv folds of `xt_device_folds` (stratified over the codes for a
    classifier, plain for a regressor) as waves of fold units (one set per
    wave, the train rows as each unit's row draw), each fold's test outputs
    placed by `stack_place_run`; passthrough X scattered after the members'
    columns; the meta features rounded to float32 and the final estimator
    fitted on them (a set of one). The fold states are released after their
    test outputs (scikit-learn keeps no fold estimator).
A fold whose train rows lack a class fails the fold set
(`MEMBER_SET_LACKING`, raised before any fit) and so this fit: the caller
takes its own path. A stratified fold of `xt_device_folds` never lacks one
(each class has at least `cv` rows, or the folds refuse).

PREDICT / PREDICT_PROBA / DECISION: the meta-estimator's own predict RUNs
(xtrees/trees_meta_vote_run.mojo, one body per operation): `vote_codes_run`
(int32 codes), `vote_soft_run` (float64 probabilities, soft only),
`vote_regress_run` (float64 predictions); stacking `stack_place_run` over
the members, passthrough, float32 meta features, then ONE call of the final
member, whose output (its own width and element type) is the META's. A
nested member that refuses raises its own words.

OUTPUTS: predict one column (int32 codes for the voting classifier,
float64 for the voting regressor, the final member's type for a stack);
predict_proba `classes` columns (float64 for a vote; the final's for a
stack); decision (stack only) the final's. Without a state the plan entry's
final plan is asked (`member_outputs` / `member_output_dtype`).

FITTED-STATE BLOCKS: `META_BLOCK_NESTED` (64), int64 1 x members: the nested
state handles in member order (the final last), owned by this state;
`META_BLOCK_WIDTHS` (65), int64 1 x stacked members: each member's meta
feature columns; `MEMBER_BLOCK_SCALARS` (4), float64 [features, classes]
(export only). EXPORT hands out views; IMPORT takes the handles over (the
state frees them) and `MEMBER_IMPORT_DONE` reads `call.classes`.
`MEMBER_EXPORT_OWN` is forwarded to every nested state (`member_own`).
The boundary's PACK reads the nested plans of a state with
`meta_state_plans` and its nested fit notices with `meta_state_notices`
(direct reads of this process-wide store: the ABI has no word for a plan).

KIND-EXACT BITS: `MEMBER_GIL_FREE` when every nested kind has it,
`MEMBER_GIL_DOOR` when every nested kind has one of the two (the nested
turns then run on the calling thread, as `member_set_fit` runs them);
`MEMBER_CAN_SAMPLE_WEIGHT` when every nested kind has it;
`MEMBER_BORROWS_FIT` always (the nested states read the META's X by view).
No `MEMBER_CAN_TEST_ROWS`: the caller proves X finite.

THE LIFETIME RULE: a state owns its nested states, the stack's float32 meta
features (the final member's fit buffer) and the folds' row numbers until
its FREE. Parallel: the nested sets run their GIL-free members on host
workers; the fold waves hold at most one gathered train block per worker.

THE UNIT KINDS (lane mf-meta3b, 2026-11-23): OneVsRestClassifier
(`META_OVR`), CalibratedClassifierCV (`META_CALIBRATED`), BaggingClassifier
and BaggingRegressor (`META_BAG_CLASSIFIER` / `_REGRESSOR`). Their plan
entry holds ONE nested plan, the base estimator; a fit makes as many units
of it as the door's fit makes clones (`_expand`), each unit its own nested
state, fitted as one member set. The kind's settings are the entry's
`words` (`META_OVR_*`, `META_CAL_*`, `META_BAG_*`), read by the boundary's
UNPACK as the door's fit reads them:
  * OneVsRest (`_ovr_fit`): k == 2 one unit on the codes, else k units on
    the class indicators (`xt_indicator_codes`), two classes each. predict:
    k == 2 the unit's codes, else `ovr_assemble_run` over each unit's
    decision column (the base has decision_function) or the probability
    column of code 1; predict_proba: k == 2 `xt_binary_proba` of the unit's
    code-1 probabilities, else `ovr_assemble_run` (finish 1).
  * CalibratedClassifierCV (`_cal_native`, an int cv: the stratified folds
    of `xt_device_folds`): one unit per fold over its train rows (plus one
    over every row for ensemble=False), the whole fit as
    xtrees/trees_calibrated_fit_run.mojo `calibrated_fit_run`; the kept
    pairs (each fold's unit, or only the full unit for ensemble=False) with
    their held classes and calibrators. predict_proba / predict:
    `calibrated_proba_run` over the pairs, by the door's pair words
    (`calibrated_pair_words`: the code-1 column for two classes, THE
    PLACEMENT RULE `cal_fold_place` for a pair that lacks a class), then the
    row argmax for predict.
  * Bagging (`_bagging_fit_bags`, the per-member route): unit k draws its
    rows (`xt_sample_indices`, stream 2k) and features (stream 2k + 1) and
    holds the seed `_sub_seed(seed, k)` in its seed words
    (xtrees/trees_ada_fit_run.mojo `ADA_MEMBER_SEED_WORD`, the RF and ET
    layouts' `random_state` slot). A classifier unit whose draw lacks a
    class is fitted on the classes it drew, re-coded (the door's member fit
    on its drawn codes), its probability columns placed at their ensemble
    codes. oob_score: each unit's out-of-bag rows (`xt_unseen_rows`) and
    `bag_oob_run`, then `bag_oob_classifier_finish` / `bag_oob_r2_run`.
    predict / predict_proba: `bag_predict_run` over each unit's feature
    subset of X, then the row argmax for a classifier's predict. The
    batched route (a bag of best-splitter trees as one forest fit, a
    measured winner) stays the door's: the boundary admits no configuration
    that may take it.
A unit kind keeps each kept unit's draws and re-coded targets until FREE,
and every unit fitted on a drawn block owns its rows (`member_own`) before
the set's gathered blocks go.

STACKING, lane mf-meta3b: a user's cv INDEX LIST is read by the UNPACK into
the entry's typed folds (`fold_train` / `fold_test`, int32 row numbers) and
checked against the fit's rows as the door's `_splits` checks them; a fold
whose train rows lack a class is fitted on its own classes, re-coded, and
placed by `stack_place_run`'s lacking-class rule. A classifier stacked with
stack_method='predict' writes its predicted codes as float32 meta features
(`_label_block`: the label scikit-learn's encoded fit predicts).
"""
from std.memory import bitcast
from std.sys.info import size_of
from std.sys.compile import is_defined
from bindings.native_unit_os_threads_run import native_unit_cores
from core.host_parallel import host_parallelize
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.device_route import DEVICE_CPU
from core.entry_table import EntryRef, entry_ref_unresolved
from core.member_abi import (
    MemberPlan, MemberCall, MemberMatrix, member_call, member_call_of, member_free, member_own,
    member_outputs, member_output_dtype, member_outputs_of,
    MEMBER_F32, MEMBER_F64, MEMBER_I32, MEMBER_I64, MEMBER_DTYPE_UNSAID,
    MEMBER_OP_CAPABILITIES, MEMBER_OP_NEW, MEMBER_OP_FREE, MEMBER_OP_FIT, MEMBER_OP_PREDICT,
    MEMBER_OP_PREDICT_PROBA, MEMBER_OP_DECISION, MEMBER_OP_OUTPUTS, MEMBER_OP_EXPORT, MEMBER_OP_IMPORT,
    MEMBER_CAN_FIT, MEMBER_CAN_PREDICT, MEMBER_CAN_PREDICT_PROBA, MEMBER_CAN_DECISION, MEMBER_CAN_EXPORT,
    MEMBER_CAN_IMPORT, MEMBER_CAN_CLASSIFIER, MEMBER_CAN_SAMPLE_WEIGHT, MEMBER_CAN_TEST_ROWS,
    MEMBER_BORROWS_FIT, MEMBER_GIL_FREE, MEMBER_GIL_DOOR,
    MEMBER_BLOCK_FAMILY, MEMBER_BLOCK_SCALARS, MEMBER_BLOCK_BY_INDEX, MEMBER_EXPORT_OWN, MEMBER_IMPORT_DONE,
    member_publish, member_call_from_spans, member_unsupported,
)
from core.member_families import MEMBER_FAMILY_ENSEMBLE, MEMBER_FAMILY_GP
from core.member_notices import member_notices_take
from core.member_set_run import (
    MemberSetRun, member_set_fit, member_set_run_release, member_set_first_failure,
    MEMBER_SET_WORKER, MEMBER_SET_TURN_RELEASED, MEMBER_SET_TURN_HELD,
)
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.shared_slot import SharedSlot, shared_slot_lock, shared_slot_unlock
from core.trees_entry_table import TreesEntryKey
from xtrees.glue_calls import (
    xt_ref, xt_device_folds, xt_gather_f32, xt_scatter, xt_indicator_codes, xt_binary_proba, xt_argmax_rows,
    xt_sample_indices, xt_unseen_rows,
)
from xtrees.trees_ada_fit_run import ADA_MEMBER_SEED_WORD
from xtrees.trees_boost_run import _sub_seed, boost_count_check
from xtrees.trees_calibrated_fit_run import (
    CalFitPlan, CalFitOut, calibrated_fit_run, cal_fold_place, CAL_ALL_ANSWERED,
)
from xtrees.trees_meta_run import meta_binary_column_check
from xtrees.trees_meta_vote_run import (
    VoteMember, VoteRows, vote_codes_run, vote_soft_run, vote_regress_run, stack_place_run, member_f32_block,
    ovr_assemble_run, bag_predict_run, bag_oob_run, bag_oob_classifier_finish, bag_oob_r2_run,
    calibrated_proba_run, CalPair, CalFit,
    VOTE_ALL_ANSWERED, STACK_PREDICT, STACK_PROBA, STACK_DECISION,
)

comptime META_VOTE_CLASSIFIER = 0
comptime META_VOTE_REGRESSOR = 1
comptime META_STACK_CLASSIFIER = 2
comptime META_STACK_REGRESSOR = 3
comptime META_OVR = 4
comptime META_CALIBRATED = 5
comptime META_BAG_CLASSIFIER = 6
comptime META_BAG_REGRESSOR = 7
comptime META_KIND_LAST = META_BAG_REGRESSOR
comptime META_PARAM_COUNT = 6
comptime META_BLOCK_NESTED = MEMBER_BLOCK_FAMILY
comptime META_BLOCK_WIDTHS = MEMBER_BLOCK_FAMILY + 1
#: Calibrated: the pairs' words (`meta_cal_words`), float64 1 x count.
comptime META_BLOCK_CAL = MEMBER_BLOCK_FAMILY + 2
#: Bagging: the units' draws (`meta_bag_words`), int32 1 x count.
comptime META_BLOCK_DRAWS = MEMBER_BLOCK_FAMILY + 3
#: Bagging oob_score: the out-of-bag outputs, float64 1 x (n * width + 1), the score last.
comptime META_BLOCK_OOB = MEMBER_BLOCK_FAMILY + 4

#: The entry's settings words of each unit kind.
comptime META_OVR_DECISION = 0          # the base has decision_function (1.0)
comptime META_OVR_WORDS = 1
comptime META_CAL_SIGMOID = 0           # method='sigmoid' (1.0), else isotonic
comptime META_CAL_ENSEMBLE = 1          # ensemble=True (1.0)
comptime META_CAL_DECISION = 2          # the base has decision_function (1.0)
comptime META_CAL_WORDS = 3
comptime META_BAG_COUNT = 0             # n_estimators
comptime META_BAG_SEED = 1              # `_seed(random_state)`
comptime META_BAG_ROWS_INT = 2          # max_samples is an int (1.0) or a fraction (0.0)
comptime META_BAG_ROWS = 3              # its value
comptime META_BAG_COLS_INT = 4          # max_features is an int (1.0) or a fraction (0.0)
comptime META_BAG_COLS = 5              # its value
comptime META_BAG_BOOTSTRAP = 6
comptime META_BAG_BOOTSTRAP_FEATURES = 7
comptime META_BAG_OOB = 8               # oob_score
comptime META_BAG_WORDS = 9

comptime _CALIB_FAST_HOST_FOLDS = is_defined["MOJOLEARN_CALIB_FAST_HOST_FOLDS"]()

comptime _CHUNK = 64
comptime _INDEX_SPAN = 16777216
comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _F64P = MutPointer[Float64, MutUntrackedOrigin]
comptime _I32P = MutPointer[Int32, MutUntrackedOrigin]
comptime _I64P = MutPointer[Int64, MutUntrackedOrigin]
comptime _CallPtr = MutPointer[MemberCall, MutUntrackedOrigin]


def _stack_kind_code(kind: Int) -> Bool:
    return kind == META_STACK_CLASSIFIER or kind == META_STACK_REGRESSOR


def _classifier_kind(kind: Int) -> Bool:
    return (kind == META_VOTE_CLASSIFIER or kind == META_STACK_CLASSIFIER or kind == META_OVR
            or kind == META_CALIBRATED or kind == META_BAG_CLASSIFIER)


def _unit_kind(kind: Int) -> Bool:
    """One nested base estimator, as many units as the fit makes (module note, THE UNIT KINDS)."""
    return kind == META_OVR or kind == META_CALIBRATED or kind == META_BAG_CLASSIFIER or kind == META_BAG_REGRESSOR


def _bag_kind(kind: Int) -> Bool:
    return kind == META_BAG_CLASSIFIER or kind == META_BAG_REGRESSOR


# ------------------------------------------------------------ the plan store


struct MetaPlanSlot(Copyable, Movable, Defaultable):
    var generation: Int
    var live: Bool
    var plans: List[MemberPlan]
    var weights: List[Float64]
    var stack_kinds: List[Int]
    var masks: List[Int]
    # Lane mf-meta3b: the kind's settings words and a stack's user cv folds.
    var words: List[Float64]
    var fold_train: List[List[Int32]]
    var fold_test: List[List[Int32]]

    def __init__(out self):
        self.generation = 0
        self.live = False
        self.plans = List[MemberPlan]()
        self.weights = List[Float64]()
        self.stack_kinds = List[Int]()
        self.masks = List[Int]()
        self.words = List[Float64]()
        self.fold_train = List[List[Int32]]()
        self.fold_test = List[List[Int32]]()


struct MetaPlanStore(Defaultable, Movable):
    var synchronization: List[Int64]
    var chunks: List[List[MetaPlanSlot]]
    var free: List[Int]
    var count: Int

    def __init__(out self):
        self.synchronization = List[Int64](length=2, fill=0)
        self.chunks = List[List[MetaPlanSlot]]()
        self.free = List[Int]()
        self.count = 0


comptime _PLANS = SharedSlot[MetaPlanStore, "MojolearnEnsembleMemberPlanStoreV1", 1]
comptime _PlanSlotPtr = MutPointer[MetaPlanSlot, MutUntrackedOrigin]


def _plan_slot_address(index: Int) raises -> Int:
    var store = _PLANS.get_or_create_ptr()
    return Int(store[].chunks[index // _CHUNK].unsafe_ptr()) + (index % _CHUNK) * size_of[MetaPlanSlot]()


def meta_plan_mint(var plans: List[MemberPlan], var weights: List[Float64], var stack_kinds: List[Int],
                   var masks: List[Int]) raises -> Int:
    """UNPACK (the boundary's admission): keep the nested plans of one META
    plan; the handle goes into the META plan's words. Released by
    `meta_plan_release` (the admission's capsule)."""
    return meta_plan_mint_with(plans^, weights^, stack_kinds^, masks^, List[Float64](), List[List[Int32]](),
                               List[List[Int32]]())


def meta_plan_mint_with(var plans: List[MemberPlan], var weights: List[Float64], var stack_kinds: List[Int],
                        var masks: List[Int], var words: List[Float64], var fold_train: List[List[Int32]],
                        var fold_test: List[List[Int32]]) raises -> Int:
    """`meta_plan_mint` with the kind's settings words (THE UNIT KINDS) and
    a stack's user cv folds (int32 train / test row numbers per fold, in
    the order the user's list gives them; empty: the int cv of the
    parameter words)."""
    var store = _PLANS.get_or_create_ptr()
    var lock = rebind[MutPointer[Int64, MutUntrackedOrigin]](store[].synchronization.unsafe_ptr())
    shared_slot_lock(lock)
    try:
        var index = -1
        if len(store[].free) > 0:
            index = store[].free.pop()
        else:
            if store[].count >= _INDEX_SPAN - 1:
                raise Error("RuntimeError: mojolearn: too many live meta member plans")
            if store[].count == len(store[].chunks) * _CHUNK:
                store[].chunks.append(List[MetaPlanSlot](length=_CHUNK, fill=MetaPlanSlot()))
            index = store[].count
            store[].count += 1
        var slot = _PlanSlotPtr(unsafe_from_address=_plan_slot_address(index))
        slot[].generation += 1
        slot[].live = True
        slot[].plans = plans^
        slot[].weights = weights^
        slot[].stack_kinds = stack_kinds^
        slot[].masks = masks^
        slot[].words = words^
        slot[].fold_train = fold_train^
        slot[].fold_test = fold_test^
        return slot[].generation * _INDEX_SPAN + index + 1
    finally:
        shared_slot_unlock(lock)


def _plan_lookup(handle: Int, release: Bool) raises -> Int:
    var store = _PLANS.get_or_create_ptr()
    var lock = rebind[MutPointer[Int64, MutUntrackedOrigin]](store[].synchronization.unsafe_ptr())
    shared_slot_lock(lock)
    try:
        var index = handle % _INDEX_SPAN - 1
        if handle <= 0 or index < 0 or index >= store[].count:
            raise Error("RuntimeError: mojolearn: this meta member plan was released")
        var address = _plan_slot_address(index)
        var slot = _PlanSlotPtr(unsafe_from_address=address)
        if not slot[].live or slot[].generation != handle // _INDEX_SPAN:
            raise Error("RuntimeError: mojolearn: this meta member plan was released")
        if release:
            slot[].live = False
            slot[].plans = List[MemberPlan]()
            slot[].weights = List[Float64]()
            slot[].stack_kinds = List[Int]()
            slot[].masks = List[Int]()
            slot[].words = List[Float64]()
            slot[].fold_train = List[List[Int32]]()
            slot[].fold_test = List[List[Int32]]()
            store[].free.append(index)
        return address
    finally:
        shared_slot_unlock(lock)


def meta_plan_release(handle: Int):
    """The admission's capsule destructor: the plan entry goes. A state made
    from it copied what it reads. Never raises."""
    try:
        _ = _plan_lookup(handle, True)
    except:
        pass


def _plan_final(handle: Int) raises -> MemberPlan:
    """The LAST nested plan of a plan entry (a stack's final estimator), copied
    under the store's lock."""
    var store = _PLANS.get_or_create_ptr()
    var address = _plan_lookup(handle, False)
    var lock = rebind[MutPointer[Int64, MutUntrackedOrigin]](store[].synchronization.unsafe_ptr())
    shared_slot_lock(lock)
    try:
        var slot = _PlanSlotPtr(unsafe_from_address=address)
        if len(slot[].plans) == 0:
            raise Error("member ABI: a meta member plan holds no members")
        return slot[].plans[len(slot[].plans) - 1].copy()
    finally:
        shared_slot_unlock(lock)


# ------------------------------------------------------------ the state store


struct EnsembleMemberSlot(Copyable, Movable, Defaultable):
    var generation: Int
    var live: Bool
    var kind: Int
    var flag: Bool        # soft (vote) / passthrough (stack)
    var folds: Int
    var cap: Int
    var plans: List[MemberPlan]
    var weights: List[Float64]
    var stack_kinds: List[Int]
    var masks: List[Int]
    var glue: EntryRef
    var fitted: Bool
    var classes: Int
    # The fitted X's feature count (0 after an import) and the scalar block
    # [features, classes] PACK reads (MEMBER_BLOCK_SCALARS).
    var features: Int
    var scalar_words: List[Float64]
    # The nested states (the final last), owned; their int64 export view.
    var states: List[Int]
    var state_words: List[Int64]
    # Per nested member, per operation: columns and element type (after fit).
    var pred_cols: List[Int]
    var pred_dtype: List[Int]
    var proba_cols: List[Int]
    var proba_dtype: List[Int]
    var dec_cols: List[Int]
    var dec_dtype: List[Int]
    # Stack: each stacked member's columns, their sum, the float32 meta
    # features the final member was fitted on (it may borrow them).
    var widths: List[Int]
    var width_words: List[Int64]
    var total: Int
    var meta32: List[Float32]
    # The class columns 0 .. classes-1 every nested classifier answers in.
    var iota: List[Int32]
    # The nested fit notices, in fit order: the nested member index and its words.
    var notice_owner: List[Int]
    var notice_words: List[List[Float64]]
    # Lane mf-meta3b. The entry's settings words and a stack's user folds;
    # THE UNIT KINDS' base plan and mask (the entry's; `plans` / `masks` hold
    # one copy per unit after a fit or an import).
    var words: List[Float64]
    var fold_train: List[List[Int32]]
    var fold_test: List[List[Int32]]
    var base_plans: List[MemberPlan]
    var base_masks: List[Int]
    # Per kept unit: its classes as ensemble codes when it lacks some (empty:
    # every class), its re-coded targets (n words), its row draw and its
    # feature subset (Bagging; empty: every row / column), its out-of-bag rows.
    var unit_held: List[List[Int32]]
    var unit_codes: List[List[Int32]]
    var unit_rows: List[List[Int32]]
    var unit_cols: List[List[Int32]]
    var unit_oob: List[List[Int32]]
    # OneVsRest: the class indicators its units were fitted on.
    var targets: List[List[Int32]]
    # Calibrated: the calibrators (CalFitOut's lists) and each pair's first and count.
    var cal_iso: List[Bool]
    var cal_a: List[Float64]
    var cal_b: List[Float64]
    var cal_m: List[Int]
    var cal_kx: List[List[Float64]]
    var cal_ky: List[List[Float64]]
    var pair_first: List[Int]
    var pair_count: List[Int]
    # The export views: the calibrated pairs' words, the bags' draws, the oob outputs.
    var cal_words: List[Float64]
    var draw_words: List[Int32]
    var oob_words: List[Float64]

    def __init__(out self):
        self.generation = 0
        self.live = False
        self.kind = 0
        self.flag = False
        self.folds = 0
        self.cap = 0
        self.plans = List[MemberPlan]()
        self.weights = List[Float64]()
        self.stack_kinds = List[Int]()
        self.masks = List[Int]()
        self.glue = entry_ref_unresolved()
        self.fitted = False
        self.classes = 0
        self.features = 0
        self.scalar_words = List[Float64]()
        self.states = List[Int]()
        self.state_words = List[Int64]()
        self.pred_cols = List[Int]()
        self.pred_dtype = List[Int]()
        self.proba_cols = List[Int]()
        self.proba_dtype = List[Int]()
        self.dec_cols = List[Int]()
        self.dec_dtype = List[Int]()
        self.widths = List[Int]()
        self.width_words = List[Int64]()
        self.total = 0
        self.meta32 = List[Float32]()
        self.iota = List[Int32]()
        self.notice_owner = List[Int]()
        self.notice_words = List[List[Float64]]()
        self.words = List[Float64]()
        self.fold_train = List[List[Int32]]()
        self.fold_test = List[List[Int32]]()
        self.base_plans = List[MemberPlan]()
        self.base_masks = List[Int]()
        self.unit_held = List[List[Int32]]()
        self.unit_codes = List[List[Int32]]()
        self.unit_rows = List[List[Int32]]()
        self.unit_cols = List[List[Int32]]()
        self.unit_oob = List[List[Int32]]()
        self.targets = List[List[Int32]]()
        self.cal_iso = List[Bool]()
        self.cal_a = List[Float64]()
        self.cal_b = List[Float64]()
        self.cal_m = List[Int]()
        self.cal_kx = List[List[Float64]]()
        self.cal_ky = List[List[Float64]]()
        self.pair_first = List[Int]()
        self.pair_count = List[Int]()
        self.cal_words = List[Float64]()
        self.draw_words = List[Int32]()
        self.oob_words = List[Float64]()


struct EnsembleMemberStore(Defaultable, Movable):
    var synchronization: List[Int64]
    var chunks: List[List[EnsembleMemberSlot]]
    var free: List[Int]
    var count: Int

    def __init__(out self):
        self.synchronization = List[Int64](length=2, fill=0)
        self.chunks = List[List[EnsembleMemberSlot]]()
        self.free = List[Int]()
        self.count = 0


comptime _STORE = SharedSlot[EnsembleMemberStore, "MojolearnEnsembleMemberStoreV1", 1]
comptime _SlotPtr = MutPointer[EnsembleMemberSlot, MutUntrackedOrigin]


def _slot_address(index: Int) raises -> Int:
    var store = _STORE.get_or_create_ptr()
    return Int(store[].chunks[index // _CHUNK].unsafe_ptr()) + (index % _CHUNK) * size_of[EnsembleMemberSlot]()


def _mint() raises -> Int:
    var store = _STORE.get_or_create_ptr()
    var lock = rebind[MutPointer[Int64, MutUntrackedOrigin]](store[].synchronization.unsafe_ptr())
    shared_slot_lock(lock)
    try:
        var index = -1
        if len(store[].free) > 0:
            index = store[].free.pop()
        else:
            if store[].count >= _INDEX_SPAN - 1:
                raise Error("RuntimeError: mojolearn: too many live meta member states")
            if store[].count == len(store[].chunks) * _CHUNK:
                store[].chunks.append(List[EnsembleMemberSlot](length=_CHUNK, fill=EnsembleMemberSlot()))
            index = store[].count
            store[].count += 1
        var slot = _SlotPtr(unsafe_from_address=_slot_address(index))
        slot[].generation += 1
        slot[].live = True
        slot[].fitted = False
        return slot[].generation * _INDEX_SPAN + index + 1
    finally:
        shared_slot_unlock(lock)


def _lookup(handle: Int, release: Bool) raises -> Int:
    """The slot address of a live handle; a released, foreign or made-up
    handle is refused (never a stale read). `release`: the slot's words are
    dropped (the caller freed its nested states first)."""
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
            slot[] = EnsembleMemberSlot()
            slot[].generation = handle // _INDEX_SPAN
            store[].free.append(index)
        return address
    finally:
        shared_slot_unlock(lock)


def _name(kind: Int) -> String:
    if kind == META_VOTE_CLASSIFIER:
        return String("VotingClassifier")
    if kind == META_VOTE_REGRESSOR:
        return String("VotingRegressor")
    if kind == META_STACK_CLASSIFIER:
        return String("StackingClassifier")
    if kind == META_OVR:
        return String("OneVsRestClassifier")
    if kind == META_CALIBRATED:
        return String("CalibratedClassifierCV")
    if kind == META_BAG_CLASSIFIER:
        return String("BaggingClassifier")
    if kind == META_BAG_REGRESSOR:
        return String("BaggingRegressor")
    return String("StackingRegressor")


def _drop_states(slot: _SlotPtr):
    """Free every nested state the slot owns. Never raises."""
    for u in range(len(slot[].states)):  # driver-loop: over the nested members (their states)
        if slot[].states[u] != 0 and u < len(slot[].plans):
            var call = MemberCall(slot[].plans[u].route, slot[].states[u])
            member_free(slot[].plans[u], call)
        slot[].states[u] = 0
    slot[].states = List[Int]()
    slot[].state_words = List[Int64]()
    slot[].fitted = False
    slot[].meta32 = List[Float32]()
    slot[].notice_owner = List[Int]()
    slot[].notice_words = List[List[Float64]]()
    slot[].unit_held = List[List[Int32]]()
    slot[].unit_codes = List[List[Int32]]()
    slot[].unit_rows = List[List[Int32]]()
    slot[].unit_cols = List[List[Int32]]()
    slot[].unit_oob = List[List[Int32]]()
    slot[].targets = List[List[Int32]]()
    slot[].cal_iso = List[Bool]()
    slot[].cal_a = List[Float64]()
    slot[].cal_b = List[Float64]()
    slot[].cal_m = List[Int]()
    slot[].cal_kx = List[List[Float64]]()
    slot[].cal_ky = List[List[Float64]]()
    slot[].pair_first = List[Int]()
    slot[].pair_count = List[Int]()
    slot[].cal_words = List[Float64]()
    slot[].draw_words = List[Int32]()
    slot[].oob_words = List[Float64]()


# --------------------------------------------------------------- the masks


def _all(masks: List[Int], bit: Int) -> Bool:
    for u in range(len(masks)):  # driver-loop: over the nested kinds (one mask each)
        if (masks[u] & bit) == 0:
            return False
    return True


def _kind_mask(kind: Int, flag: Bool, masks: List[Int]) -> Int:
    """The kind-exact capability mask of a META state over nested kinds
    `masks` (module note, KIND-EXACT BITS; a unit kind: its base's one
    mask). A unit kind answers what the door's operations of its class
    answer: OneVsRest predict_proba when its base has it, Calibrated and a
    Bagging classifier predict_proba always (their admission requires the
    base's scores), sample weights for Bagging only (the OneVsRest and
    Calibrated fits take none), no decision_function."""
    var mask = MEMBER_CAN_FIT | MEMBER_CAN_PREDICT | MEMBER_CAN_EXPORT | MEMBER_CAN_IMPORT | MEMBER_BORROWS_FIT
    if _classifier_kind(kind):
        mask |= MEMBER_CAN_CLASSIFIER
    if _unit_kind(kind):
        var base = masks[0] if len(masks) > 0 else 0
        if kind == META_CALIBRATED or kind == META_BAG_CLASSIFIER:
            mask |= MEMBER_CAN_PREDICT_PROBA
        elif kind == META_OVR and (base & MEMBER_CAN_PREDICT_PROBA) != 0:
            mask |= MEMBER_CAN_PREDICT_PROBA
        if _bag_kind(kind) and (base & MEMBER_CAN_SAMPLE_WEIGHT) != 0:
            mask |= MEMBER_CAN_SAMPLE_WEIGHT
        if (base & MEMBER_GIL_FREE) != 0:
            mask |= MEMBER_GIL_FREE
        elif (base & MEMBER_GIL_DOOR) != 0:
            mask |= MEMBER_GIL_DOOR
        return mask
    if len(masks) > 0:
        var final = masks[len(masks) - 1]
        if kind == META_VOTE_CLASSIFIER and flag and _all(masks, MEMBER_CAN_PREDICT_PROBA):
            mask |= MEMBER_CAN_PREDICT_PROBA
        if kind == META_STACK_CLASSIFIER:
            if (final & MEMBER_CAN_PREDICT_PROBA) != 0:
                mask |= MEMBER_CAN_PREDICT_PROBA
            if (final & MEMBER_CAN_DECISION) != 0:
                mask |= MEMBER_CAN_DECISION
    if _all(masks, MEMBER_CAN_SAMPLE_WEIGHT):
        mask |= MEMBER_CAN_SAMPLE_WEIGHT
    if _all(masks, MEMBER_GIL_FREE):
        mask |= MEMBER_GIL_FREE
    elif _all_either(masks):
        mask |= MEMBER_GIL_DOOR
    return mask


def _all_either(masks: List[Int]) -> Bool:
    for u in range(len(masks)):  # driver-loop: over the nested kinds (one mask each)
        if (masks[u] & (MEMBER_GIL_FREE | MEMBER_GIL_DOOR)) == 0:
            return False
    return True


comptime _UNION = (MEMBER_CAN_FIT | MEMBER_CAN_PREDICT | MEMBER_CAN_PREDICT_PROBA | MEMBER_CAN_DECISION
    | MEMBER_CAN_EXPORT | MEMBER_CAN_IMPORT | MEMBER_CAN_CLASSIFIER | MEMBER_CAN_SAMPLE_WEIGHT | MEMBER_BORROWS_FIT
    | MEMBER_GIL_FREE | MEMBER_GIL_DOOR)


def _set_word(mask: Int) -> UInt8:
    """THE GIL RULE word of a nested member in its set."""
    if (mask & MEMBER_GIL_FREE) != 0:
        return MEMBER_SET_WORKER
    if (mask & MEMBER_GIL_DOOR) != 0:
        return MEMBER_SET_TURN_RELEASED
    return MEMBER_SET_TURN_HELD


# ------------------------------------------------------------------ NEW


def _new(call: _CallPtr) raises:
    if call[].param_count != META_PARAM_COUNT or not call[].params:
        raise Error("member ABI: a meta member takes six parameters (kind, plan, soft or passthrough, folds, "
            + "worker bound, reserved)")
    var words = _F64P(unsafe_from_address=call[].params)
    var kind = Int(words[0])
    if kind < META_VOTE_CLASSIFIER or kind > META_KIND_LAST:
        raise Error("member ABI: unknown meta member kind")
    var plan_address = _plan_lookup(Int(words[1]), False)
    var state = _mint()
    var store = _PLANS.get_or_create_ptr()
    var lock = rebind[MutPointer[Int64, MutUntrackedOrigin]](store[].synchronization.unsafe_ptr())
    var slot = _SlotPtr(unsafe_from_address=_lookup(state, False))
    slot[].kind = kind
    slot[].flag = words[2] != 0.0
    slot[].folds = Int(words[3])
    slot[].cap = Int(words[4])
    shared_slot_lock(lock)
    try:
        var entry = _PlanSlotPtr(unsafe_from_address=plan_address)
        slot[].plans = entry[].plans.copy()
        slot[].weights = entry[].weights.copy()
        slot[].stack_kinds = entry[].stack_kinds.copy()
        slot[].masks = entry[].masks.copy()
        slot[].words = entry[].words.copy()
        slot[].fold_train = entry[].fold_train.copy()
        slot[].fold_test = entry[].fold_test.copy()
    finally:
        shared_slot_unlock(lock)
    try:
        if len(slot[].plans) == 0:
            raise Error("member ABI: a meta member plan holds no members")
        var user_folds = len(slot[].fold_test)
        if len(slot[].fold_train) != user_folds:
            raise Error("member ABI: a meta member's user folds need a train and a test side each")
        if _stack_kind_code(kind) and (len(slot[].stack_kinds) + 1 != len(slot[].plans)
                                       or (slot[].folds < 2 and user_folds == 0)):
            raise Error("member ABI: a stacking member needs its members, its final estimator and cv >= 2")
        if _unit_kind(kind):
            var need = META_OVR_WORDS
            if kind == META_CALIBRATED:
                need = META_CAL_WORDS
            elif _bag_kind(kind):
                need = META_BAG_WORDS
            if len(slot[].plans) != 1 or len(slot[].masks) != 1 or len(slot[].words) != need:
                raise Error("member ABI: a " + _name(kind) + " member needs its one base estimator and its "
                            + String(need) + " settings words")
            if kind == META_CALIBRATED and slot[].folds < 2:
                raise Error("member ABI: a CalibratedClassifierCV member needs cv >= 2")
            slot[].base_plans = slot[].plans.copy()
            slot[].base_masks = slot[].masks.copy()
        elif not _stack_kind_code(kind) and len(slot[].weights) != len(slot[].plans):
            raise Error("member ABI: a voting member needs one weight per member")
        # The glue entry of the x_trees image this META entry runs on,
        # resolved once per state (never per call).
        slot[].glue = xt_ref(TreesEntryKey(call[].route.mode, call[].route.vendor))
    except error:
        _ = _lookup(state, True)
        raise error^
    call[].state = state


# ------------------------------------------------------------------ helpers


def _nested_outputs(slot: _SlotPtr) raises:
    """After a fit or import: each nested member's columns and element type
    per operation it answers (asked of its fitted state once)."""
    var count = len(slot[].plans)
    slot[].pred_cols = List[Int](length=count, fill=0)
    slot[].pred_dtype = List[Int](length=count, fill=MEMBER_F32)
    slot[].proba_cols = List[Int](length=count, fill=0)
    slot[].proba_dtype = List[Int](length=count, fill=MEMBER_F32)
    slot[].dec_cols = List[Int](length=count, fill=0)
    slot[].dec_dtype = List[Int](length=count, fill=MEMBER_F32)
    for u in range(count):  # driver-loop: over the nested members (one OUTPUTS question per operation)
        if slot[].states[u] == 0:
            continue  # not fitted yet (a stack's final, during its fit)
        var plan = slot[].plans[u].copy()
        var call = member_call_of(plan, slot[].states[u])
        slot[].pred_cols[u] = member_outputs_of(plan, call, MEMBER_OP_PREDICT)
        slot[].pred_dtype[u] = call.out_dtype if call.out_dtype != MEMBER_DTYPE_UNSAID else MEMBER_F32
        if plan.member.can(MEMBER_CAN_PREDICT_PROBA) and (slot[].masks[u] & MEMBER_CAN_PREDICT_PROBA) != 0:
            slot[].proba_cols[u] = member_outputs_of(plan, call, MEMBER_OP_PREDICT_PROBA)
            slot[].proba_dtype[u] = call.out_dtype if call.out_dtype != MEMBER_DTYPE_UNSAID else MEMBER_F32
        if plan.member.can(MEMBER_CAN_DECISION) and (slot[].masks[u] & MEMBER_CAN_DECISION) != 0:
            slot[].dec_cols[u] = member_outputs_of(plan, call, MEMBER_OP_DECISION)
            slot[].dec_dtype[u] = call.out_dtype if call.out_dtype != MEMBER_DTYPE_UNSAID else MEMBER_F32
    slot[].iota = List[Int32](length=max(slot[].classes, 1), fill=Int32(0))
    for c in range(slot[].classes):  # small-loop(the class columns 0 .. classes-1, once per fit)
        slot[].iota[c] = Int32(c)
    slot[].state_words = List[Int64](length=count, fill=Int64(0))
    for u in range(count):  # driver-loop: over the nested members (the export view)
        slot[].state_words[u] = Int64(slot[].states[u])
    slot[].scalar_words = [Float64(slot[].features), Float64(slot[].classes)]
    slot[].width_words = List[Int64](length=len(slot[].widths), fill=Int64(0))
    for j in range(len(slot[].widths)):  # driver-loop: over the stacked members (the export view)
        slot[].width_words[j] = Int64(slot[].widths[j])


def _member_of(slot: _SlotPtr, u: Int, state: Int, operation: Int, weight: Float64) -> VoteMember:
    """Nested member `u` over `state` as a RUN's `VoteMember` (its plan read
    in place: the slot's list never moves while the state lives)."""
    var cols = slot[].pred_cols[u]
    var dtype = slot[].pred_dtype[u]
    if operation == MEMBER_OP_PREDICT_PROBA:
        cols = slot[].proba_cols[u]
        dtype = slot[].proba_dtype[u]
    elif operation == MEMBER_OP_DECISION:
        cols = slot[].dec_cols[u]
        dtype = slot[].dec_dtype[u]
    return VoteMember(Int(UnsafePointer(to=slot[].plans[u])), state, operation, cols, dtype,
                      Int(slot[].iota.unsafe_ptr()), weight)


def _stack_operation(kind: Int) -> Int:
    if kind == STACK_PROBA:
        return MEMBER_OP_PREDICT_PROBA
    if kind == STACK_DECISION:
        return MEMBER_OP_DECISION
    return MEMBER_OP_PREDICT


def _raise_member(m: VoteMember, rows: VoteRows) raises:
    """The words of a nested member whose call a RUN reported refused: the
    same call again, raising."""
    var plan = MutPointer[MemberPlan, MutUntrackedOrigin](unsafe_from_address=m.plan)
    var call = member_call_of(plan[], m.state)
    call.x = MemberMatrix(rows.x, rows.n, rows.d)
    var scratch = List[Int64](length=max(rows.n * m.cols, 1), fill=Int64(0))
    call.dst = MemberMatrix(Int(scratch.unsafe_ptr()), rows.n, m.cols, m.dtype)
    _ = member_call(plan[].member, m.operation, call)
    _ = len(scratch)  # written through the call above
    raise Error("RuntimeError: mojolearn: a meta member's member refused its call")


def _take_notices(slot: _SlotPtr, u: Int, plan: MemberPlan, state: Int):
    """Keep nested member `u`'s fit notices (raised by the boundary's PACK
    with that member's own context, in fit order). Never raises."""
    try:
        var words = List[Float64]()
        member_notices_take(plan, state, words)
        if len(words) > 0:
            slot[].notice_owner.append(u)
            slot[].notice_words.append(words^)
    except:
        pass


def _set_of(slot: _SlotPtr, x: Int, rows: Int, cols: Int, y_dtype: Int) -> MemberSetRun:
    var run = MemberSetRun(x, rows, cols, y_dtype, MEMBER_I32)
    run.cap = slot[].cap
    return run^


def _add(slot: _SlotPtr, mut run: MemberSetRun, u: Int, classes: Int, target: Int, weights: Int, draw: Int,
         count: Int):
    _add_drawn(slot, run, u, classes, target, weights, draw, count, 0, 0)


def _add_drawn(slot: _SlotPtr, mut run: MemberSetRun, u: Int, classes: Int, target: Int, weights: Int, draw: Int,
               count: Int, column_draw: Int, column_count: Int):
    """Nested member `u` into `run`, reading its row draw and its feature
    subset (counts 0: every row / column)."""
    # The slot's own plan carries the class count too: the RUNs' member calls
    # read it in place (`_member_of`).
    slot[].plans[u].classes = classes
    var plan = slot[].plans[u].copy()
    var mask = slot[].masks[u]
    run.add(plan^, target, weights, draw, count, column_draw, column_count,
            UInt8(1) if (mask & MEMBER_CAN_TEST_ROWS) != 0 else UInt8(0),
            UInt8(1) if (mask & MEMBER_BORROWS_FIT) != 0 else UInt8(0), _set_word(mask))


def _expand(slot: _SlotPtr, units: Int):
    """A unit kind: one copy of the base plan and mask per unit (module
    note, THE UNIT KINDS)."""
    slot[].plans = List[MemberPlan](capacity=units)
    slot[].masks = List[Int](capacity=units)
    for _ in range(units):  # driver-loop: over the units (one plan copy each)
        slot[].plans.append(slot[].base_plans[0].copy())
        slot[].masks.append(slot[].base_masks[0])


def _held_and_codes(codes: Int, k: Int, n: Int, draws: List[Int], counts: List[Int],
                    mut held: List[List[Int32]], mut recoded: List[List[Int32]]):
    """Per unit (its rows: the int32 row numbers at `draws[u]`, `counts[u]`
    of them; 0: every row): the ensemble codes its rows hold, ascending,
    when it lacks some (else empty), and its re-coded targets (`n` words:
    the unit's own code at every row of a held class; else empty), the y
    the door's member fit on those rows encodes. Both passes on the host
    cores, one task per unit."""
    var units = len(draws)
    var seen = List[UInt8](length=max(units * k, 1), fill=UInt8(0))
    var seen_at = Int(seen.unsafe_ptr())
    var draws_at = Int(draws.unsafe_ptr())
    var counts_at = Int(counts.unsafe_ptr())

    def _mark(u: Int) {imm seen_at, imm draws_at, imm counts_at, imm codes, imm k, imm n}:
        var draw = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=draws_at)[u]
        var count = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=counts_at)[u]
        var rows = count if draw != 0 else n
        var picks = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=draw)
        var y = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=codes)
        var marks = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=seen_at + u * k)
        for i in range(rows):  # small-loop(one pass over one unit's rows)
            var c = Int(y[Int(picks[i]) if draw != 0 else i])
            if c >= 0 and c < k:
                marks[c] = UInt8(1)

    host_parallelize(_mark, units)
    held = List[List[Int32]](capacity=units)
    recoded = List[List[Int32]](capacity=units)
    var maps = List[Int32](length=max(units * k, 1), fill=Int32(0))
    var lacking = List[Int]()
    for u in range(units):  # driver-loop: over the units (their held classes)
        var present = List[Int32]()
        for c in range(k):  # small-loop(the classes of one unit)
            if seen[u * k + c] != UInt8(0):
                maps[u * k + c] = Int32(len(present))
                present.append(Int32(c))
        if len(present) == k:
            held.append(List[Int32]())
            recoded.append(List[Int32]())
        else:
            held.append(present^)
            recoded.append(List[Int32](length=max(n, 1), fill=Int32(0)))
            lacking.append(u)
    _ = len(seen)  # read through the tasks above
    if len(lacking) == 0:
        return
    var targets = List[Int](capacity=len(lacking))
    for i in range(len(lacking)):  # driver-loop: over the lacking units (their target addresses)
        targets.append(Int(recoded[lacking[i]].unsafe_ptr()))
    var lacking_at = Int(lacking.unsafe_ptr())
    var targets_at = Int(targets.unsafe_ptr())
    var maps_at = Int(maps.unsafe_ptr())

    def _recode(i: Int) {imm lacking_at, imm targets_at, imm maps_at, imm codes, imm k, imm n}:
        var u = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=lacking_at)[i]
        var dst = MutPointer[Int32, MutUntrackedOrigin](
            unsafe_from_address=MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=targets_at)[i])
        var to = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=maps_at + 4 * u * k)
        var y = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=codes)
        for r in range(n):  # small-loop(one pass over the rows: the unit's code at each)
            var c = Int(y[r])
            dst[r] = to[c] if c >= 0 and c < k else Int32(0)

    host_parallelize(_recode, len(lacking))
    _ = len(maps)  # read through the tasks above
    _ = len(targets)
    _ = len(lacking)


def _held_classes(held: List[Int32], k: Int) -> Int:
    return len(held) if len(held) > 0 else k


def _index_of_code(held: List[Int32], code: Int) -> Int:
    """`list.index(code)` of a unit's classes (`held`; empty: 0 .. k-1, so
    any code below k is its own index), -1 when absent."""
    if len(held) == 0:
        return code
    for c in range(len(held)):  # small-loop(a unit's classes)
        if Int(held[c]) == code:
            return c
    return -1


def _codes_into(slot: _SlotPtr, u: Int, state: Int, rows: VoteRows, dst: Int) raises:
    """Nested member `u`'s int32 class codes over `rows` at `dst` (int64
    codes narrowed once, as the door's int32 cast narrowed them); its call
    refusing raises its own words."""
    var call = member_call_of(slot[].plans[u], state)
    call.x = MemberMatrix(rows.x, rows.n, rows.d)
    if slot[].pred_dtype[u] == MEMBER_I64:
        var wide = List[Int64](length=max(rows.n, 1), fill=Int64(0))
        call.dst = MemberMatrix(Int(wide.unsafe_ptr()), rows.n, 1, MEMBER_I64)
        _ = member_call(slot[].plans[u].member, MEMBER_OP_PREDICT, call)
        var codes = _I32P(unsafe_from_address=dst)
        for i in range(rows.n):  # small-loop(one narrowing pass over one member's codes, as the door's cast)
            codes[i] = Int32(wide[i])
        _ = len(wide)
    else:
        call.dst = MemberMatrix(dst, rows.n, 1, MEMBER_I32)
        _ = member_call(slot[].plans[u].member, MEMBER_OP_PREDICT, call)


def _label_block(slot: _SlotPtr, u: Int, state: Int, rows: VoteRows, held: List[Int32],
                 mut keep: List[List[Float32]]) raises -> VoteMember:
    """A classifier stacked with stack_method='predict' (lane mf-meta3b):
    its predicted codes over `rows` as one float32 column, the meta feature
    of scikit-learn's encoded fit (`held` non-empty: a re-coded fold's
    member, whose code c is the ensemble code held[c]). The block is
    answered here, so the RUN reads it as a door block."""
    var n = rows.n
    var codes = List[Int32](length=max(n, 1), fill=Int32(0))
    _codes_into(slot, u, state, rows, Int(codes.unsafe_ptr()))
    var words = List[Float32](length=max(n, 1), fill=0.0)
    for i in range(n):  # small-loop(one widening pass over one member's codes)
        var c = Int(codes[i])
        if len(held) > 0 and c >= 0 and c < len(held):
            c = Int(held[c])
        words[i] = Float32(c)
    _ = len(codes)
    keep.append(words^)
    return VoteMember(Int(keep[len(keep) - 1].unsafe_ptr()), 1, 0, 1.0)


def _stack_members(slot: _SlotPtr, states: List[Int], offset: Int, rows: VoteRows, held: List[Int32],
                   mut keep: List[List[Float32]]) raises -> List[VoteMember]:
    """The stacked members (member u over `states[offset + u]`) as the
    RUN's VoteMembers over `rows`: a classifier's predict as its label block
    (`_label_block`), every other operation a native call; `held`
    non-empty: the fold's classes (each member's class columns)."""
    var bases = len(slot[].stack_kinds)
    var classifier = slot[].kind == META_STACK_CLASSIFIER
    var members = List[VoteMember](capacity=bases)
    for u in range(bases):  # driver-loop: over the stacked members
        var state = states[offset + u]
        if classifier and slot[].stack_kinds[u] == STACK_PREDICT:
            members.append(_label_block(slot, u, state, rows, held, keep))
            continue
        var m = _member_of(slot, u, state, _stack_operation(slot[].stack_kinds[u]), 1.0)
        if len(held) > 0:
            m.class_cols = Int(held.unsafe_ptr())
        members.append(m)
    return members^


def _stack_counts(slot: _SlotPtr, k: Int, held: List[Int32]) -> List[Int]:
    """Each stacked member's fitted class count (0 for STACK_PREDICT)."""
    var bases = len(slot[].stack_kinds)
    var counts = List[Int](length=bases, fill=0)
    for u in range(bases):  # driver-loop: over the stacked members (their class counts)
        if slot[].stack_kinds[u] != STACK_PREDICT:
            counts[u] = _held_classes(held, k)
    return counts^


def _user_folds(slot: _SlotPtr, n: Int, mut tr_at: List[Int], mut tr_n: List[Int], mut te_at: List[Int],
                mut te_n: List[Int]) raises:
    """A user's cv index list (the entry's typed folds) against the fit's
    `n` rows, refused as the door's `_splits` refuses it, in its words and
    order: per fold its empty side, its train then test rows outside
    [0, n); no fold; then a test side that is not a partition (stacking's
    cross_val_predict). The bounds of every fold on the host cores."""
    var folds = len(slot[].fold_test)
    var bad = List[Int](length=max(folds, 1), fill=0)
    var bad_at = Int(bad.unsafe_ptr())
    var tr_ptrs = List[Int](length=max(folds, 1), fill=0)
    var tr_counts = List[Int](length=max(folds, 1), fill=0)
    var te_ptrs = List[Int](length=max(folds, 1), fill=0)
    var te_counts = List[Int](length=max(folds, 1), fill=0)
    for f in range(folds):  # driver-loop: over the user's folds (their addresses)
        tr_ptrs[f] = Int(slot[].fold_train[f].unsafe_ptr())
        tr_counts[f] = len(slot[].fold_train[f])
        te_ptrs[f] = Int(slot[].fold_test[f].unsafe_ptr())
        te_counts[f] = len(slot[].fold_test[f])
    var trp = Int(tr_ptrs.unsafe_ptr())
    var trc = Int(tr_counts.unsafe_ptr())
    var tep = Int(te_ptrs.unsafe_ptr())
    var tec = Int(te_counts.unsafe_ptr())

    def _check(f: Int) {imm bad_at, imm trp, imm trc, imm tep, imm tec, imm n}:
        var flag = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=bad_at)
        var ntr = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=trc)[f]
        var nte = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=tec)[f]
        if ntr == 0 or nte == 0:
            flag[f] = 1
            return
        var tr = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=MutPointer[Int, MutUntrackedOrigin](
            unsafe_from_address=trp)[f])
        var te = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=MutPointer[Int, MutUntrackedOrigin](
            unsafe_from_address=tep)[f])
        for i in range(ntr):  # small-loop(one bounds pass over one fold's train rows)
            if Int(tr[i]) < 0 or Int(tr[i]) >= n:
                flag[f] = 2
                return
        for i in range(nte):  # small-loop(one bounds pass over one fold's test rows)
            if Int(te[i]) < 0 or Int(te[i]) >= n:
                flag[f] = 2
                return

    host_parallelize(_check, folds)
    for f in range(folds):  # driver-loop: over the user's folds (the first refusal in fold order)
        if bad[f] == 1:
            raise Error("every cv split needs train and test rows")
        if bad[f] == 2:
            raise Error("a cv index is outside [0, " + String(n) + ")")
    if folds == 0:
        raise Error("cv produced no splits")
    var seen = List[Int32](length=max(n, 1), fill=Int32(0))
    for f in range(folds):  # driver-loop: over the user's folds (the partition count)
        for i in range(te_counts[f]):  # small-loop(one count per test row: n rows in all over a partition)
            seen[Int(slot[].fold_test[f][i])] += 1
    for i in range(n):  # small-loop(the partition check: one word per row, once per fit)
        if seen[i] != 1:
            raise Error("cross_val_predict only works for partitions: every row must be in exactly one test set")
    for f in range(folds):  # driver-loop: over the user's folds (the RUN's addresses)
        tr_at.append(tr_ptrs[f])
        tr_n.append(tr_counts[f])
        te_at.append(te_ptrs[f])
        te_n.append(te_counts[f])
    _ = len(bad)
    _ = len(tr_ptrs)
    _ = len(tr_counts)
    _ = len(te_ptrs)
    _ = len(te_counts)


def _keep_states(slot: _SlotPtr, mut run: MemberSetRun, first: Int):
    """The fitted states of `run` become the slot's nested members `first`,
    `first + 1`, ... (their fit notices kept)."""
    for v in range(len(run.states)):  # driver-loop: over the set's members (their states change owner)
        slot[].states[first + v] = run.states[v]
        _take_notices(slot, first + v, run.plans[v], run.states[v])
        run.states[v] = 0


# ------------------------------------------------------------------ FIT


def _fit_vote(slot: _SlotPtr, call: _CallPtr) raises:
    var classes = call[].classes if slot[].kind == META_VOTE_CLASSIFIER else 0
    var y_dtype = MEMBER_I32 if slot[].kind == META_VOTE_CLASSIFIER else MEMBER_F32
    var weights = call[].weights.data if call[].weights.given() else 0
    var m = len(slot[].plans)
    var run = _set_of(slot, call[].x.data, call[].x.rows, call[].x.cols, y_dtype)
    for u in range(m):  # driver-loop: over the voting members (admission into the set, once each)
        _add(slot, run, u, classes, call[].y.data, weights, 0, 0)
    member_set_fit(run)
    slot[].states = List[Int](length=m, fill=0)
    _keep_states(slot, run, 0)
    slot[].classes = classes
    slot[].widths = List[Int]()


def _fit_stack(slot: _SlotPtr, call: _CallPtr) raises:
    var classifier = slot[].kind == META_STACK_CLASSIFIER
    var k = call[].classes if classifier else 0
    var y_dtype = MEMBER_I32 if classifier else MEMBER_F32
    var n = call[].x.rows
    var d = call[].x.cols
    var x = call[].x.data
    var y = call[].y.data
    var weights = call[].weights.data if call[].weights.given() else 0
    var bases = len(slot[].stack_kinds)
    var glue = slot[].glue
    var binary = k == 2
    var every = List[Int32]()  # a member that holds every class
    slot[].classes = k
    slot[].states = List[Int](length=bases + 1, fill=0)
    # 1. The members on every row (one set).
    var full = _set_of(slot, x, n, d, y_dtype)
    for u in range(bases):  # driver-loop: over the stacked members (admission, once each)
        _add(slot, full, u, k, y, weights, 0, 0)
    member_set_fit(full)
    _keep_states(slot, full, 0)
    _nested_outputs(slot)
    var counts = _stack_counts(slot, k, every)
    # 2. The width probe over the first rows.
    var head_keep = List[List[Float32]]()
    var head = VoteRows(x, min(n, 2), d)
    var members = _stack_members(slot, slot[].states, 0, head, every, head_keep)
    var widths = List[Int]()
    var refused = stack_place_run(glue, head, members, slot[].stack_kinds, counts, k, binary, 0, 0, 0, 0, 0,
                                  widths, 0)
    _ = len(head_keep)  # the label blocks are read through the RUN above
    if refused != VOTE_ALL_ANSWERED:
        _raise_member(members[refused], head)
    var total = 0
    for j in range(len(widths)):  # driver-loop: over the stacked members
        total += widths[j]
    var width = total + (d if slot[].flag else 0)
    slot[].widths = widths.copy()
    slot[].total = total
    var meta = List[Float64](length=max(n * width, 1), fill=0.0)
    var meta_at = Int(meta.unsafe_ptr())
    # 3. The cv folds: a user's index list (the entry's typed folds, checked
    # here), else stratified over the codes (classifier) or plain (regressor).
    var tr_at = List[Int]()
    var tr_n = List[Int]()
    var te_at = List[Int]()
    var te_n = List[Int]()
    var n_splits = slot[].folds
    var fold_counts = List[Int32](length=n_splits + 1, fill=Int32(0))
    var fold_rows = List[Int32](length=max(n_splits * n, 1), fill=Int32(0))
    var rows_at = Int(fold_rows.unsafe_ptr())
    if len(slot[].fold_test) > 0:
        _user_folds(slot, n, tr_at, tr_n, te_at, te_n)
    else:
        var codes_at = y if classifier else Int(fold_counts.unsafe_ptr())
        var status = xt_device_folds(glue, codes_at, rows_at, Int(fold_counts.unsafe_ptr()), n, n_splits, k)
        if status == 1:
            raise Error("ValueError: n_splits=" + String(n_splits)
                        + " cannot be greater than the number of members in each class")
        if status != 0:
            raise Error("ValueError: y codes outside [0, n_classes)")
        var last = 0
        for f in range(n_splits):  # driver-loop: over the cv folds
            if Int(fold_counts[f]) > 0:
                last = f
        for f in range(last + 1):  # driver-loop: over the cv folds (their train and test rows)
            var c = Int(fold_counts[f])
            tr_at.append(rows_at + 4 * (f * n))
            tr_n.append(n - c)
            te_at.append(rows_at + 4 * (f * n + n - c))
            te_n.append(c)
    var used = len(te_at)
    # A classifier fold whose train rows lack a class (a user's folds can) is
    # fitted on its own classes, re-coded, as the door's member fit on that
    # fold's codes; its members' columns are then placed at their classes.
    var held = List[List[Int32]]()
    var recoded = List[List[Int32]]()
    if classifier:
        _held_and_codes(y, k, n, tr_at, tr_n, held, recoded)
    var columns = List[Int32](length=max(d, 1), fill=Int32(0))
    for j in range(d):  # small-loop(the column numbers 0 .. d-1 of a fold's gather, once per fit)
        columns[j] = Int32(j)
    # Waves of folds: as many folds' train blocks as the workers use at once
    # (ceil(workers / members) folds on the CPU, one fold when a member runs on
    # a GPU), the wave rule of `_stacking_fit`.
    var host = True
    for u in range(bases):  # driver-loop: over the stacked members (their routes)
        if slot[].plans[u].route.device != DEVICE_CPU:
            host = False
    var workers = 1
    if host:
        workers = native_unit_cores()
        if slot[].cap > 0:
            workers = min(workers, slot[].cap)
    var wave = max(1, (workers + bases - 1) // max(bases, 1))
    var f0 = 0
    while f0 < used:  # driver-loop: over waves of cv folds
        var f1 = min(used, f0 + wave)
        var run = _set_of(slot, x, n, d, y_dtype)
        # The held classes were read above: the set makes no second cover pass.
        run.skip_lacking = classifier
        for f in range(f0, f1):  # driver-loop: over the wave's cv folds (their units)
            var target = y
            var classes = k
            if classifier and len(held[f]) > 0:
                target = Int(recoded[f].unsafe_ptr())
                classes = len(held[f])
            for u in range(bases):  # driver-loop: over the stacked members
                _add(slot, run, u, classes, target, weights, tr_at[f], tr_n[f])
        try:
            member_set_fit(run)
            for v in range(len(run.states)):  # driver-loop: over the wave's units (their fit notices)
                _take_notices(slot, v % bases, run.plans[v], run.states[v])
            for f in range(f0, f1):  # driver-loop: over the wave's cv folds (test outputs)
                var c = te_n[f]
                var test_at = te_at[f]
                var fold_held = held[f].copy() if classifier else List[Int32]()
                # The members' calls read their class count from the slot's plans in place.
                for u in range(bases):  # driver-loop: over the stacked members (the fold's class count)
                    slot[].plans[u].classes = _held_classes(fold_held, k) if classifier else 0
                var xte = List[Float32](length=max(c * d, 1), fill=0.0)
                _ = xt_gather_f32(glue, x, test_at, Int(columns.unsafe_ptr()), Int(xte.unsafe_ptr()), n, d, c, d)
                var rows = VoteRows(Int(xte.unsafe_ptr()), c, d)
                var fold_keep = List[List[Float32]]()
                var fold_members = _stack_members(slot, run.states, (f - f0) * bases, rows, fold_held, fold_keep)
                var fold_counts_of = _stack_counts(slot, k, fold_held)
                var got = List[Int]()
                var missed = stack_place_run(glue, rows, fold_members, slot[].stack_kinds, fold_counts_of, k, binary,
                                             meta_at, n, width, test_at, 0, got, 0)
                _ = len(fold_keep)  # the label blocks are read through the RUN above
                _ = len(fold_held)  # the class columns are read through the RUN above
                if missed != VOTE_ALL_ANSWERED:
                    _raise_member(fold_members[missed], rows)
                for j in range(bases):  # driver-loop: over the stacked members
                    if got[j] != widths[j]:
                        raise Error("ValueError: a member's output width changed between folds")
                _ = len(xte)  # read through the RUN above
        finally:
            # scikit-learn keeps no fold estimator: the wave's states and
            # gathered train blocks go here; the full members' class count
            # comes back to the slot's plans.
            member_set_run_release(run)
            for u in range(bases):  # driver-loop: over the stacked members (their class count)
                slot[].plans[u].classes = k
        f0 = f1
    _ = len(recoded)  # the re-coded targets are read through the fold sets above
    # 4. Passthrough: X after the members' columns.
    if slot[].flag:
        var order = List[Int32](length=max(n, 1), fill=Int32(0))
        for i in range(n):  # small-loop(the row numbers 0 .. n-1 the scatter places, once per fit)
            order[i] = Int32(i)
        _ = xt_scatter(glue, meta_at, x, Int(order.unsafe_ptr()), n, width, n, d, total)
        _ = len(order)
    # 5. The final estimator on the float32 meta features.
    slot[].meta32 = List[Float32](length=max(n * width, 1), fill=0.0)
    for i in range(n * width):  # small-loop(one rounding pass of the meta features, as the door's float32 admission)
        slot[].meta32[i] = meta[i].cast[DType.float32]()
    _ = len(meta)
    _ = len(fold_rows)  # the fold draws are read through the fold sets above
    _ = len(fold_counts)
    _ = len(columns)
    var last_set = _set_of(slot, Int(slot[].meta32.unsafe_ptr()), n, width, y_dtype)
    _add(slot, last_set, bases, k, y, weights, 0, 0)
    member_set_fit(last_set)
    _keep_states(slot, last_set, bases)


def _fit_ovr(slot: _SlotPtr, call: _CallPtr) raises:
    """`_ovr_fit`: one binary unit on the codes (k == 2), else one per class
    on its indicator (`xt_indicator_codes`, the door's `_indicator`), every
    unit two classes, fitted as ONE set."""
    var k = call[].classes
    if k < 2:
        raise Error("y has fewer than 2 classes")
    var n = call[].x.rows
    var d = call[].x.cols
    var x = call[].x.data
    var y = call[].y.data
    var units = 1 if k == 2 else k
    _expand(slot, units)
    slot[].targets = List[List[Int32]]()
    if k > 2:
        for j in range(k):  # driver-loop: over the classes (one native indicator each)
            var yj = List[Int32](length=max(n, 1), fill=Int32(0))
            _ = xt_indicator_codes(slot[].glue, y, Int(yj.unsafe_ptr()), 0, n, j)
            slot[].targets.append(yj^)
    var run = _set_of(slot, x, n, d, MEMBER_I32)
    for u in range(units):  # driver-loop: over the units (admission into the set, once each)
        var target = y if k == 2 else Int(slot[].targets[u].unsafe_ptr())
        _add(slot, run, u, 2, target, 0, 0, 0)
    member_set_fit(run)
    slot[].states = List[Int](length=units, fill=0)
    _keep_states(slot, run, 0)
    slot[].classes = k


def _own_drawn(slot: _SlotPtr, drawn: List[Bool]) raises:
    """THE LIFETIME RULE: every kept unit fitted on a drawn block (a fold's
    train rows, a bag's draw) owns its rows before the set's gathered
    blocks, the draws and the re-coded targets go."""
    for u in range(len(slot[].states)):  # driver-loop: over the kept units (one ownership question each)
        if u < len(drawn) and drawn[u] and slot[].states[u] != 0:
            member_own(slot[].plans[u], slot[].states[u])


def _fit_cal(slot: _SlotPtr, call: _CallPtr) raises:
    """`_cal_native` over an int cv: the stratified folds of
    `xt_device_folds`, one unit per fold over its train rows (and one over
    every row for ensemble=False), `calibrated_fit_run`, then the kept pairs
    and their calibrators. A failing fit raises the first failing fold's
    words in fold order (GaussianProcessClassifier's one-class words with
    the label suffix its fit door adds, `_cal_fit_words`)."""
    var k = call[].classes
    if k < 2:
        raise Error("y has fewer than 2 classes")
    var n = call[].x.rows
    var d = call[].x.cols
    var x = call[].x.data
    var y = call[].y.data
    var glue = slot[].glue
    var sigmoid = slot[].words[META_CAL_SIGMOID] != 0.0
    var ensemble = slot[].words[META_CAL_ENSEMBLE] != 0.0
    var decision = slot[].words[META_CAL_DECISION] != 0.0
    var n_splits = slot[].folds
    var fold_counts = List[Int32](length=n_splits + 1, fill=Int32(0))
    var fold_rows = List[Int32](length=max(n_splits * n, 1), fill=Int32(0))
    var rows_at = Int(fold_rows.unsafe_ptr())
    var status = xt_device_folds(glue, y, rows_at, Int(fold_counts.unsafe_ptr()), n, n_splits, k)
    if status == 1:
        raise Error("n_splits=" + String(n_splits) + " cannot be greater than the number of members in each class")
    if status != 0:
        raise Error("y codes outside [0, n_classes)")
    var last = 0
    for f in range(n_splits):  # driver-loop: over the cv folds
        if Int(fold_counts[f]) > 0:
            last = f
    var used = last + 1
    var units = used if ensemble else used + 1
    _expand(slot, units)
    var run = _set_of(slot, x, n, d, MEMBER_I32)
    for f in range(used):  # driver-loop: over the cv folds (admission, once each)
        var c = Int(fold_counts[f])
        _add(slot, run, f, k, y, 0, rows_at + 4 * (f * n), n - c)
    if not ensemble:
        _add(slot, run, used, k, y, 0, 0, 0)
    var operation = MEMBER_OP_DECISION if decision else MEMBER_OP_PREDICT_PROBA
    var plan = CalFitPlan(x, n, d, y, k, operation, sigmoid, ensemble, _CALIB_FAST_HOST_FOLDS)
    for f in range(used):  # driver-loop: over the cv folds (their test rows)
        var c = Int(fold_counts[f])
        plan.fold(rows_at + 4 * (f * n + n - c), c)
    var res = CalFitOut()
    # A refused scoring call raises its member's words (the RUN released every state).
    var answered = calibrated_fit_run(glue, run, plan, res)
    if answered != CAL_ALL_ANSWERED:
        var first = member_set_first_failure(run)
        var words = String("RuntimeError: mojolearn: a CalibratedClassifierCV member's fit failed")
        if first >= 0:
            words = run.errors[first].copy()
            if (first < plan.folds() and len(plan.fold_classes[first]) == 1
                    and run.plans[first].family == MEMBER_FAMILY_GP and words.endswith("got 1 class")):
                words += " (only class " + String(Int(plan.fold_classes[first][0])) + " is present)"
        member_set_run_release(run)
        raise Error(words)
    # The kept pairs: every fold's unit (ensemble), else only the unit over every row.
    var kept = List[Int]()
    if ensemble:
        for f in range(used):  # driver-loop: over the cv folds
            kept.append(f)
    else:
        kept.append(used)
    var pairs = len(kept)
    var plans = List[MemberPlan](capacity=pairs)
    var masks = List[Int](capacity=pairs)
    var drawn = List[Bool](capacity=pairs)
    slot[].states = List[Int](length=pairs, fill=0)
    slot[].unit_held = List[List[Int32]](capacity=pairs)
    for i in range(pairs):  # driver-loop: over the kept pairs (their states change owner)
        var u = kept[i]
        plans.append(run.plans[u].copy())
        masks.append(slot[].masks[u])
        drawn.append(u < used)
        slot[].states[i] = run.states[u]
        run.states[u] = 0
        slot[].unit_held.append(plan.fold_classes[u].copy() if u < plan.folds() else List[Int32]())
    # Every unit's fit notices, in unit order (a discarded fold's with the first pair's context: one base class).
    for u in range(len(run.states)):  # driver-loop: over the units (their fit notices)
        var owner = 0
        for i in range(pairs):  # small-loop(the kept pairs)
            if kept[i] == u:
                owner = i
        var state = slot[].states[owner] if run.states[u] == 0 else run.states[u]
        _take_notices(slot, owner, run.plans[u], state)
    slot[].plans = plans^
    slot[].masks = masks^
    try:
        _own_drawn(slot, drawn)
    finally:
        # The discarded fold units (ensemble=False) and the gathered blocks go.
        member_set_run_release(run)
    _ = plan.folds()  # its re-coded targets were read through the fits and the ownership above
    _ = len(fold_rows)  # the fold rows are read through the fits and the ownership above
    _ = len(fold_counts)
    slot[].cal_iso = res.isotonic.copy()
    slot[].cal_a = res.a.copy()
    slot[].cal_b = res.b.copy()
    slot[].cal_m = res.m.copy()
    slot[].cal_kx = res.knots_x.copy()
    slot[].cal_ky = res.knots_y.copy()
    slot[].pair_first = List[Int](capacity=pairs)
    slot[].pair_count = List[Int](capacity=pairs)
    if ensemble:
        for f in range(used):  # driver-loop: over the cv folds (their calibrators)
            slot[].pair_first.append(res.first[f])
            slot[].pair_count.append(res.count[f])
    else:
        slot[].pair_first.append(res.final_first)
        slot[].pair_count.append(res.final_count)
    slot[].classes = k


def _bag_count(is_int: Bool, value: Float64, total: Int, name: String) raises -> Int:
    """`_BaggingBase._count`: an int as it is, a fraction of `total`
    truncated (its domain (0, 1] proved by the UNPACK); the resolved count's
    refusal in the door's words."""
    var k = Int(value) if is_int else Int(value * Float64(total))
    return boost_count_check(name, k, total)


def _seed_words(mut plan: MemberPlan, seed: Int) raises:
    """A Bagging unit's member seed in its forest seed slot words (an exact
    nonnegative int: both readings, no flag; the door's `_clone_from`)."""
    if len(plan.params) < ADA_MEMBER_SEED_WORD + 3:
        raise Error("member ABI: a Bagging member's base has no seed slot")
    plan.params[ADA_MEMBER_SEED_WORD] = bitcast[DType.float64](Int64(seed))
    plan.params[ADA_MEMBER_SEED_WORD + 1] = Float64(seed)
    plan.params[ADA_MEMBER_SEED_WORD + 2] = 0.0


def _fit_bag(slot: _SlotPtr, call: _CallPtr) raises:
    """`_bagging_fit_bags`' per-member route as ONE set: unit u's draws
    (`xt_sample_indices`, streams 2u and 2u + 1) and seed `_sub_seed(seed,
    u)`, the door's order and words; then the out-of-bag outputs
    (oob_score)."""
    var classifier = slot[].kind == META_BAG_CLASSIFIER
    var k = call[].classes if classifier else 0
    if classifier and k < 2:
        raise Error("y has fewer than 2 classes")
    var n = call[].x.rows
    var d = call[].x.cols
    var x = call[].x.data
    var y = call[].y.data
    var weights = call[].weights.data if call[].weights.given() else 0
    var glue = slot[].glue
    var count = Int(slot[].words[META_BAG_COUNT])
    var seed = Int(slot[].words[META_BAG_SEED])
    var n_rows = _bag_count(slot[].words[META_BAG_ROWS_INT] != 0.0, slot[].words[META_BAG_ROWS], n, "max_samples")
    var n_feat = _bag_count(slot[].words[META_BAG_COLS_INT] != 0.0, slot[].words[META_BAG_COLS], d, "max_features")
    var bootstrap = slot[].words[META_BAG_BOOTSTRAP] != 0.0
    var bootstrap_features = slot[].words[META_BAG_BOOTSTRAP_FEATURES] != 0.0
    var all_rows = n_rows == n and not bootstrap
    var all_cols = n_feat == d and not bootstrap_features
    _expand(slot, count)
    slot[].unit_rows = List[List[Int32]](capacity=count)
    slot[].unit_cols = List[List[Int32]](capacity=count)
    var draw_at = List[Int](capacity=count)
    var draw_n = List[Int](capacity=count)
    for u in range(count):  # driver-loop: over the member estimators (their draws and seeds, the door's order)
        var rows = List[Int32]()
        if not all_rows:
            rows = List[Int32](length=max(n_rows, 1), fill=Int32(0))
            _ = xt_sample_indices(glue, Int(rows.unsafe_ptr()), n, n_rows, 1 if bootstrap else 0, seed, 2 * u)
        var cols = List[Int32]()
        if not all_cols:
            cols = List[Int32](length=max(n_feat, 1), fill=Int32(0))
            _ = xt_sample_indices(glue, Int(cols.unsafe_ptr()), d, n_feat, 1 if bootstrap_features else 0, seed,
                                  2 * u + 1)
        _seed_words(slot[].plans[u], _sub_seed(seed, u))
        slot[].unit_rows.append(rows^)
        slot[].unit_cols.append(cols^)
    for u in range(count):  # driver-loop: over the member estimators (their draws' addresses, now final)
        draw_at.append(0 if all_rows else Int(slot[].unit_rows[u].unsafe_ptr()))
        draw_n.append(0 if all_rows else n_rows)
    var held = List[List[Int32]]()
    var recoded = List[List[Int32]]()
    if classifier:
        _held_and_codes(y, k, n, draw_at, draw_n, held, recoded)
    var run = _set_of(slot, x, n, d, MEMBER_I32 if classifier else MEMBER_F32)
    # A classifier's held classes were read above: no second cover pass.
    run.skip_lacking = classifier
    for u in range(count):  # driver-loop: over the member estimators (admission, once each)
        var target = y
        var classes = 0
        if classifier:
            classes = k
            if len(held[u]) > 0:
                target = Int(recoded[u].unsafe_ptr())
                classes = len(held[u])
        _add_drawn(slot, run, u, classes, target, weights, draw_at[u], draw_n[u],
                   0 if all_cols else Int(slot[].unit_cols[u].unsafe_ptr()), 0 if all_cols else n_feat)
    member_set_fit(run)
    slot[].states = List[Int](length=count, fill=0)
    _keep_states(slot, run, 0)
    var drawn = List[Bool](length=count, fill=not (all_rows and all_cols))
    try:
        _own_drawn(slot, drawn)
    finally:
        member_set_run_release(run)
    _ = len(recoded)  # the re-coded targets are read through the fits and the ownership above
    slot[].unit_held = held^
    slot[].classes = k
    _nested_outputs(slot)
    if slot[].words[META_BAG_OOB] != 0.0:
        _bag_oob(slot, call, draw_at, draw_n, n_rows)
    _ = len(draw_at)
    # Each unit owns its rows now and the out-of-bag rows are taken: the row
    # draws are not kept (the door keeps none).
    slot[].unit_rows = List[List[Int32]]()


def _bag_oob(slot: _SlotPtr, call: _CallPtr, draw_at: List[Int], draw_n: List[Int], n_rows: Int) raises:
    """`_set_oob_score`: each unit's output on the rows it never drew (its
    feature subset), summed with `bag_oob_run`; a classifier's
    `oob_decision_function_` and `oob_score_` (`bag_oob_classifier_finish`),
    a regressor's `oob_prediction_` and R2 (`bag_oob_r2_run`). Into
    `oob_words`, the score last. The out-of-bag rows of every unit are kept
    (PACK writes them, `_oob_rows`)."""
    var classifier = slot[].kind == META_BAG_CLASSIFIER
    var n = call[].x.rows
    var d = call[].x.cols
    var x = call[].x.data
    var y = call[].y.data
    var k = slot[].classes if classifier else 1
    var glue = slot[].glue
    var count = len(slot[].states)
    var every = List[Int32](length=max(d, 1), fill=Int32(0))
    for j in range(d):  # small-loop(the column numbers 0 .. d-1, once per fit)
        every[j] = Int32(j)
    slot[].unit_oob = List[List[Int32]](capacity=count)
    var blocks = List[List[Float32]](capacity=count)
    var xs = List[Int](capacity=count)
    var ns = List[Int](capacity=count)
    var ds = List[Int](capacity=count)
    var oobs = List[Int](capacity=count)
    var members = List[VoteMember](capacity=count)
    for u in range(count):  # driver-loop: over the member estimators (their out-of-bag rows and blocks)
        var unseen = List[Int32](length=max(n, 1), fill=Int32(0))
        # oob_score needs bootstrap=True (the UNPACK refuses it otherwise), so every unit drew its rows.
        var m = xt_unseen_rows(glue, draw_at[u], Int(unseen.unsafe_ptr()), draw_n[u], n) if draw_at[u] != 0 else 0
        var oob = List[Int32](length=m, fill=Int32(0))
        for i in range(m):  # small-loop(one copy of a member's out-of-bag rows)
            oob[i] = unseen[i]
        _ = len(unseen)
        var width = d if len(slot[].unit_cols[u]) == 0 else len(slot[].unit_cols[u])
        var block = List[Float32]()
        if m > 0:
            block = List[Float32](length=m * width, fill=0.0)
            var cols_at = Int(every.unsafe_ptr()) if len(slot[].unit_cols[u]) == 0 else Int(
                slot[].unit_cols[u].unsafe_ptr())
            _ = xt_gather_f32(glue, x, Int(oob.unsafe_ptr()), cols_at, Int(block.unsafe_ptr()), n, d, m, width)
        slot[].unit_oob.append(oob^)
        blocks.append(block^)
        ns.append(m)
        ds.append(width)
        if m == 0:
            members.append(VoteMember(0, 0, 0, 1.0))
            continue
        var operation = MEMBER_OP_PREDICT_PROBA if classifier else MEMBER_OP_PREDICT
        var member = _member_of(slot, u, slot[].states[u], operation, 1.0)
        if classifier and len(slot[].unit_held[u]) > 0:
            member.class_cols = Int(slot[].unit_held[u].unsafe_ptr())
        members.append(member)
    for u in range(count):  # driver-loop: over the member estimators (their blocks' addresses, now final)
        xs.append(Int(blocks[u].unsafe_ptr()))
        oobs.append(Int(slot[].unit_oob[u].unsafe_ptr()))
    var acc = List[Float64](length=max(n * k, 1), fill=0.0)
    var counts = List[Int32](length=max(n, 1), fill=Int32(0))
    var refused = bag_oob_run(glue, members, xs, ns, ds, oobs, n, k, classifier, Int(acc.unsafe_ptr()),
                              Int(counts.unsafe_ptr()), 0)
    if refused != VOTE_ALL_ANSWERED:
        _raise_member(members[refused], VoteRows(xs[refused], ns[refused], ds[refused]))
    _ = len(blocks)  # read through the RUN above
    _ = len(every)
    if classifier:
        var pred = List[Int32](length=max(n, 1), fill=Int32(0))
        var hits = bag_oob_classifier_finish(glue, Int(acc.unsafe_ptr()), y, Int(pred.unsafe_ptr()), n, k)
        _ = len(pred)
        slot[].oob_words = List[Float64](capacity=n * k + 1)
        for i in range(n * k):  # small-loop(one copy of the out-of-bag decision function into the export view)
            slot[].oob_words.append(acc[i])
        slot[].oob_words.append(Float64(hits) / Float64(n))
    else:
        var pred = List[Float64](length=max(n, 1), fill=0.0)
        var r2 = bag_oob_r2_run(glue, Int(acc.unsafe_ptr()), Int(counts.unsafe_ptr()), y, Int(pred.unsafe_ptr()), n)
        if not r2.finite:
            raise Error("BaggingRegressor oob_score: a non-finite target or out-of-bag prediction, "
                        + "or a sum that overflows float64")
        var score = 1.0 - r2.ss_res / r2.ss_tot if r2.ss_tot > 0 else (1.0 if r2.ss_res == 0 else 0.0)
        slot[].oob_words = List[Float64](capacity=n + 1)
        for i in range(n):  # small-loop(one copy of the out-of-bag predictions into the export view)
            slot[].oob_words.append(pred[i])
        slot[].oob_words.append(score)
    _ = len(acc)
    _ = len(counts)


def _unit_views(slot: _SlotPtr):
    """A unit kind's export views after a fit or an import: the calibrated
    pairs' words (`META_BLOCK_CAL`: pairs, then per pair its held count and
    codes, its calibrator count and per calibrator isotonic, a, b, m, the m
    knots x and the m knots y) and the bags' words (`META_BLOCK_DRAWS`:
    units, then per unit its feature count and columns, its held count and
    codes, its out-of-bag count and rows)."""
    if slot[].kind == META_CALIBRATED:
        var w = List[Float64]()
        w.append(Float64(len(slot[].states)))
        for i in range(len(slot[].states)):  # driver-loop: over the pairs
            var held = slot[].unit_held[i].copy() if i < len(slot[].unit_held) else List[Int32]()
            w.append(Float64(len(held)))
            for c in range(len(held)):  # small-loop(a pair's held classes)
                w.append(Float64(Int(held[c])))
            w.append(Float64(slot[].pair_count[i]))
            for j in range(slot[].pair_first[i], slot[].pair_first[i] + slot[].pair_count[i]):  # driver-loop: over the pair's calibrators
                w.append(1.0 if slot[].cal_iso[j] else 0.0)
                w.append(slot[].cal_a[j])
                w.append(slot[].cal_b[j])
                w.append(Float64(slot[].cal_m[j]))
                for q in range(slot[].cal_m[j]):  # small-loop(one calibrator's knots)
                    w.append(slot[].cal_kx[j][q])
                for q in range(slot[].cal_m[j]):  # small-loop(one calibrator's knots)
                    w.append(slot[].cal_ky[j][q])
        slot[].cal_words = w^
    elif _bag_kind(slot[].kind):
        var w = List[Int32]()
        var units = len(slot[].states)
        w.append(Int32(units))
        for u in range(units):  # driver-loop: over the member estimators
            var cols = slot[].unit_cols[u].copy() if u < len(slot[].unit_cols) else List[Int32]()
            w.append(Int32(len(cols)))
            for j in range(len(cols)):  # small-loop(a member's feature subset)
                w.append(cols[j])
            var held = slot[].unit_held[u].copy() if u < len(slot[].unit_held) else List[Int32]()
            w.append(Int32(len(held)))
            for c in range(len(held)):  # small-loop(a member's held classes)
                w.append(held[c])
            var oob = slot[].unit_oob[u].copy() if u < len(slot[].unit_oob) else List[Int32]()
            w.append(Int32(len(oob)))
            for i in range(len(oob)):  # small-loop(a member's out-of-bag rows, once per fit)
                w.append(oob[i])
        slot[].draw_words = w^


def _fit(slot: _SlotPtr, call: _CallPtr) raises:
    var name = _name(slot[].kind)
    if not call[].x.dense_host(MEMBER_F32):
        raise Error("TypeError: a " + name + " member reads X as a dense host float32 view")
    if call[].row_index.rows > 0 or call[].column_index.rows > 0:
        raise Error("TypeError: a " + name + " member reads whole host views (no row or column index)")
    var y_dtype = MEMBER_I32 if _classifier_kind(slot[].kind) else MEMBER_F32
    if not call[].y.dense_host(y_dtype) or call[].y.rows != call[].x.rows or call[].y.cols != 1:
        raise Error("TypeError: a " + name + " member reads y as one dense host column of "
                    + (String("int32 class codes") if y_dtype == MEMBER_I32 else String("float32 targets")))
    if call[].weights.given() and (not call[].weights.dense_host(MEMBER_F32) or call[].weights.rows != call[].x.rows):
        raise Error("TypeError: a " + name + " member reads the sample weights as one dense host float32 column")
    if _classifier_kind(slot[].kind) and call[].classes < 1:
        raise Error("member ABI: a " + name + " member needs its class count at fit")
    if call[].weights.given() and (slot[].kind == META_OVR or slot[].kind == META_CALIBRATED):
        raise Error("TypeError: a " + name + " member's fit takes no sample weights")
    _drop_states(slot)
    slot[].features = call[].x.cols
    try:
        if _stack_kind_code(slot[].kind):
            _fit_stack(slot, call)
        elif slot[].kind == META_OVR:
            _fit_ovr(slot, call)
        elif slot[].kind == META_CALIBRATED:
            _fit_cal(slot, call)
        elif _bag_kind(slot[].kind):
            _fit_bag(slot, call)   # its own `_nested_outputs` (the out-of-bag outputs read them)
        else:
            _fit_vote(slot, call)
        if not _bag_kind(slot[].kind):
            _nested_outputs(slot)
        _unit_views(slot)
        slot[].fitted = True
    except error:
        _drop_states(slot)
        raise error^


# ------------------------------------------------------------------ PREDICT


def _fitted(slot: _SlotPtr) raises:
    if not slot[].fitted:
        raise Error("NotFittedError: this " + _name(slot[].kind) + " member is not fitted yet")


def _rows(call: _CallPtr, name: String) raises -> VoteRows:
    if not call[].x.dense_host(MEMBER_F32):
        raise Error("TypeError: a " + name + " member reads X as a dense host float32 view")
    return VoteRows(call[].x.data, call[].x.rows, call[].x.cols)


def _zero_dst(call: _CallPtr, count: Int):
    var p = _F64P(unsafe_from_address=call[].dst.data)
    for i in range(count):  # small-loop(the float64 accumulator the vote RUN sums into, zeroed once per call)
        p[i] = 0.0


def _predict_vote(slot: _SlotPtr, call: _CallPtr, operation: Int) raises:
    var name = _name(slot[].kind)
    var rows = _rows(call, name)
    var n = rows.n
    var m = len(slot[].plans)
    var soft = slot[].flag
    var k = slot[].classes
    var members = List[VoteMember](capacity=m)
    var op = MEMBER_OP_PREDICT
    if slot[].kind == META_VOTE_CLASSIFIER and soft:
        op = MEMBER_OP_PREDICT_PROBA
    for u in range(m):  # driver-loop: over the voting members
        members.append(_member_of(slot, u, slot[].states[u], op, slot[].weights[u]))
    var refused = VOTE_ALL_ANSWERED
    if slot[].kind == META_VOTE_REGRESSOR:
        if operation != MEMBER_OP_PREDICT:
            member_unsupported(name, operation)
        if not call[].dst.dense_host(MEMBER_F64) or call[].dst.rows != n:
            raise Error("TypeError: a VotingRegressor member writes predict as one dense host float64 column")
        _zero_dst(call, n)
        refused = vote_regress_run(slot[].glue, rows, members, call[].dst.data)
    elif operation == MEMBER_OP_PREDICT:
        if not call[].dst.dense_host(MEMBER_I32) or call[].dst.rows != n:
            raise Error("TypeError: a VotingClassifier member writes predict as one dense host int32 column")
        var acc = List[Float64](length=max(n * k, 1), fill=0.0)
        refused = vote_codes_run(slot[].glue, rows, members, k, soft, Int(acc.unsafe_ptr()), call[].dst.data)
        _ = len(acc)  # read through the RUN above
    elif operation == MEMBER_OP_PREDICT_PROBA and soft:
        if not call[].dst.dense_host(MEMBER_F64) or call[].dst.rows != n or call[].dst.cols != k:
            raise Error("TypeError: a VotingClassifier member writes predict_proba as dense host float64 rows")
        _zero_dst(call, n * k)
        refused = vote_soft_run(slot[].glue, rows, members, k, call[].dst.data)
    else:
        member_unsupported(name, operation)
    if refused != VOTE_ALL_ANSWERED:
        _raise_member(members[refused], rows)


def _predict_stack(slot: _SlotPtr, call: _CallPtr, operation: Int) raises:
    var name = _name(slot[].kind)
    var rows = _rows(call, name)
    var n = rows.n
    var d = rows.d
    var bases = len(slot[].stack_kinds)
    var k = slot[].classes
    var width = slot[].total + (d if slot[].flag else 0)
    var every = List[Int32]()
    var counts = _stack_counts(slot, k, every)
    var keep = List[List[Float32]]()
    var members = _stack_members(slot, slot[].states, 0, rows, every, keep)
    var meta = List[Float64](length=max(n * width, 1), fill=0.0)
    var order = List[Int32](length=max(n, 1), fill=Int32(0))
    for i in range(n):  # small-loop(the row numbers 0 .. n-1 the scatter places, once per call)
        order[i] = Int32(i)
    var got = List[Int]()
    var refused = stack_place_run(slot[].glue, rows, members, slot[].stack_kinds, counts, k, k == 2,
                                  Int(meta.unsafe_ptr()), n, width, Int(order.unsafe_ptr()), 0, got, 0)
    _ = len(keep)  # the label blocks are read through the RUN above
    if refused != VOTE_ALL_ANSWERED:
        _raise_member(members[refused], rows)
    if slot[].flag:
        _ = xt_scatter(slot[].glue, Int(meta.unsafe_ptr()), rows.x, Int(order.unsafe_ptr()), n, width, n, d,
                       slot[].total)
    var meta32 = List[Float32](length=max(n * width, 1), fill=0.0)
    for i in range(n * width):  # small-loop(one rounding pass of the meta features, as the door's float32 admission)
        meta32[i] = meta[i].cast[DType.float32]()
    _ = len(meta)
    _ = len(order)
    var plan = slot[].plans[bases].copy()
    var final = member_call_of(plan, slot[].states[bases])
    final.x = MemberMatrix(Int(meta32.unsafe_ptr()), n, width)
    final.dst = call[].dst
    _ = member_call(plan.member, operation, final)
    _ = len(meta32)  # read through the final member's call above


def _codes_dst(call: _CallPtr, n: Int, name: String) raises:
    if not call[].dst.dense_host(MEMBER_I32) or call[].dst.rows != n:
        raise Error("TypeError: a " + name + " member writes predict as one dense host int32 column")


def _proba_dst(call: _CallPtr, n: Int, k: Int, name: String) raises:
    if not call[].dst.dense_host(MEMBER_F64) or call[].dst.rows != n or call[].dst.cols != k:
        raise Error("TypeError: a " + name + " member writes predict_proba as dense host float64 rows")


def _predict_ovr(slot: _SlotPtr, call: _CallPtr, operation: Int) raises:
    """`_ovr_codes` / `_ovr_predict_proba` (module note)."""
    var name = _name(slot[].kind)
    var rows = _rows(call, name)
    var n = rows.n
    var k = slot[].classes
    var glue = slot[].glue
    if operation == MEMBER_OP_PREDICT:
        _codes_dst(call, n, name)
        if k == 2:
            _codes_into(slot, 0, slot[].states[0], rows, call[].dst.data)
            return
    elif operation == MEMBER_OP_PREDICT_PROBA:
        _proba_dst(call, n, k, name)
    else:
        member_unsupported(name, operation)
    if operation == MEMBER_OP_PREDICT_PROBA and k == 2:
        # `_proba_positive` then `_trees_binary_proba`: the unit's code-1
        # probabilities (float32, as the door's admission rounds them).
        var keep = List[List[Float32]]()
        var m = _member_of(slot, 0, slot[].states[0], MEMBER_OP_PREDICT_PROBA, 1.0)
        var block = member_f32_block(m, rows, keep)
        var col = _index_of_code(List[Int32](), 1)
        if col >= m.cols:
            col = -1
        var p = List[Float32](length=max(n, 1), fill=0.0)
        if col >= 0:
            var b = _F32P(unsafe_from_address=block)
            for i in range(n):  # small-loop(one column copy of one member's block)
                p[i] = b[i * m.cols + col]
        _ = len(keep)
        _ = xt_binary_proba(glue, Int(p.unsafe_ptr()), call[].dst.data, n)
        _ = len(p)
        return
    # k > 2: each unit's decision column (the base has decision_function and
    # predict asks), else its probability column of code 1.
    var decision = operation == MEMBER_OP_PREDICT and slot[].words[META_OVR_DECISION] != 0.0
    var op = MEMBER_OP_DECISION if decision else MEMBER_OP_PREDICT_PROBA
    var members = List[VoteMember](capacity=k)
    var at = List[Int](capacity=k)
    for u in range(k):  # driver-loop: over the units
        members.append(_member_of(slot, u, slot[].states[u], op, 1.0))
        at.append(0 if decision else 1)
    if operation == MEMBER_OP_PREDICT_PROBA:
        _zero_dst(call, n * k)
    var refused = ovr_assemble_run(glue, rows, members, at, k, 2 if operation == MEMBER_OP_PREDICT else 1,
                                   call[].dst.data)
    if refused != VOTE_ALL_ANSWERED:
        _raise_member(members[refused], rows)


def _predict_cal(slot: _SlotPtr, call: _CallPtr, operation: Int) raises:
    """`_calibrated_fused` over the kept pairs (`calibrated_pair_words`),
    then the row argmax for predict (`_calibrated_codes`)."""
    var name = _name(slot[].kind)
    var rows = _rows(call, name)
    var n = rows.n
    var k = slot[].classes
    var glue = slot[].glue
    if operation == MEMBER_OP_PREDICT:
        _codes_dst(call, n, name)
    elif operation == MEMBER_OP_PREDICT_PROBA:
        _proba_dst(call, n, k, name)
    else:
        member_unsupported(name, operation)
    var op = MEMBER_OP_DECISION if slot[].words[META_CAL_DECISION] != 0.0 else MEMBER_OP_PREDICT_PROBA
    var pairs = len(slot[].states)
    var members = List[VoteMember](capacity=pairs)
    var cal_pairs = List[CalPair](capacity=pairs)
    var cals = List[CalFit]()
    var placed = List[List[Int32]]()
    for i in range(pairs):  # driver-loop: over the (member, calibrators) pairs (their words, once per call)
        var m = _member_of(slot, i, slot[].states[i], op, 1.0)
        var cols = m.cols
        var at = 0
        var cols_at = 0
        var ks = 0
        if k == 2:
            if cols != 1:
                at = _index_of_code(slot[].unit_held[i], 1)
                meta_binary_column_check(at)
        elif len(slot[].unit_held[i]) == 0:
            cols_at = Int(slot[].iota.unsafe_ptr())
            ks = k
        else:
            # THE PLACEMENT RULE of a pair whose member lacks a class (lane mf-cal3b).
            var place = cal_fold_place(k, slot[].unit_held[i], op, cols, True, 0)
            if len(place.calibrate) != slot[].pair_count[i]:
                raise Error("ValueError: mojolearn: CalibratedClassifierCV: a pair holds " + String(slot[].pair_count[i])
                            + " calibrators for a member placed at " + String(len(place.calibrate)) + " columns")
            placed.append(place.place.copy())
            cols_at = Int(placed[len(placed) - 1].unsafe_ptr())
            ks = len(place.place)
        m.class_cols = cols_at
        members.append(m)
        cal_pairs.append(CalPair(at, cols_at, ks, len(cals), slot[].pair_count[i]))
        for j in range(slot[].pair_first[i], slot[].pair_first[i] + slot[].pair_count[i]):  # driver-loop: over the pair's calibrators
            cals.append(CalFit(slot[].cal_iso[j], slot[].cal_a[j], slot[].cal_b[j],
                               Int(slot[].cal_kx[j].unsafe_ptr()), Int(slot[].cal_ky[j].unsafe_ptr()), slot[].cal_m[j]))
    var scratch = List[Float64]()
    var acc = call[].dst.data
    if operation == MEMBER_OP_PREDICT:
        scratch = List[Float64](length=max(n * k, 1), fill=0.0)
        acc = Int(scratch.unsafe_ptr())
    else:
        _zero_dst(call, n * k)
    var refused = calibrated_proba_run(glue, rows, members, cal_pairs, cals, k, acc)
    _ = len(placed)  # the placed class codes are read through the RUN above
    if refused != VOTE_ALL_ANSWERED:
        _raise_member(members[refused], rows)
    if operation == MEMBER_OP_PREDICT:
        _ = xt_argmax_rows(glue, acc, call[].dst.data, n, k)
    _ = len(scratch)


def _predict_bag(slot: _SlotPtr, call: _CallPtr, operation: Int) raises:
    """`_bagging_proba` / `bagging_predict`: each unit over its feature
    subset of X (`_sub_x`), `bag_predict_run`, then the row argmax for a
    classifier's predict (`bagging_codes`)."""
    var name = _name(slot[].kind)
    var classifier = slot[].kind == META_BAG_CLASSIFIER
    var rows = _rows(call, name)
    var n = rows.n
    var d = rows.d
    var k = slot[].classes if classifier else 1
    var glue = slot[].glue
    if operation == MEMBER_OP_PREDICT:
        if classifier:
            _codes_dst(call, n, name)
        elif not call[].dst.dense_host(MEMBER_F64) or call[].dst.rows != n:
            raise Error("TypeError: a BaggingRegressor member writes predict as one dense host float64 column")
    elif operation == MEMBER_OP_PREDICT_PROBA and classifier:
        _proba_dst(call, n, k, name)
    else:
        member_unsupported(name, operation)
    var count = len(slot[].states)
    var order = List[Int32](length=max(n, 1), fill=Int32(0))
    for i in range(n):  # small-loop(the row numbers 0 .. n-1 of the feature gathers, once per call)
        order[i] = Int32(i)
    var blocks = List[List[Float32]](capacity=count)
    var xs = List[Int](capacity=count)
    var ds = List[Int](capacity=count)
    var members = List[VoteMember](capacity=count)
    var unit_op = MEMBER_OP_PREDICT_PROBA if classifier else MEMBER_OP_PREDICT
    for u in range(count):  # driver-loop: over the member estimators (their feature subsets, `_sub_x`)
        var c = len(slot[].unit_cols[u])
        var block = List[Float32]()
        if c > 0 and not _all_columns(slot[].unit_cols[u], d):
            block = List[Float32](length=n * c, fill=0.0)
            _ = xt_gather_f32(glue, rows.x, Int(order.unsafe_ptr()), Int(slot[].unit_cols[u].unsafe_ptr()),
                              Int(block.unsafe_ptr()), n, d, n, c)
        blocks.append(block^)
        var m = _member_of(slot, u, slot[].states[u], unit_op, 1.0)
        if classifier and len(slot[].unit_held[u]) > 0:
            m.class_cols = Int(slot[].unit_held[u].unsafe_ptr())
        members.append(m)
    for u in range(count):  # driver-loop: over the member estimators (their rows' addresses, now final)
        if len(blocks[u]) > 0:
            xs.append(Int(blocks[u].unsafe_ptr()))
            ds.append(len(slot[].unit_cols[u]))
        else:
            xs.append(rows.x)
            ds.append(d)
    var scratch = List[Float64]()
    var acc = call[].dst.data
    if classifier and operation == MEMBER_OP_PREDICT:
        scratch = List[Float64](length=max(n * k, 1), fill=0.0)
        acc = Int(scratch.unsafe_ptr())
    else:
        _zero_dst(call, n * k)
    var refused = bag_predict_run(glue, members, xs, ds, n, k, classifier, acc, 0)
    _ = len(blocks)  # read through the RUN above
    _ = len(order)
    if refused != VOTE_ALL_ANSWERED:
        _raise_member(members[refused], VoteRows(xs[refused], n, ds[refused]))
    if classifier and operation == MEMBER_OP_PREDICT:
        _ = xt_argmax_rows(glue, acc, call[].dst.data, n, k)
    _ = len(scratch)


def _all_columns(cols: List[Int32], d: Int) -> Bool:
    """`cols.tolist() == list(range(d))` (`_BaggingBase._sub_X`'s rule)."""
    if len(cols) != d:
        return False
    for j in range(d):  # small-loop(a member's feature subset)
        if Int(cols[j]) != j:
            return False
    return True


def _predict(slot: _SlotPtr, call: _CallPtr, operation: Int) raises:
    _fitted(slot)
    if _stack_kind_code(slot[].kind):
        if operation == MEMBER_OP_PREDICT_PROBA or operation == MEMBER_OP_DECISION:
            if slot[].kind != META_STACK_CLASSIFIER:
                member_unsupported(_name(slot[].kind), operation)
        _predict_stack(slot, call, operation)
    elif slot[].kind == META_OVR:
        _predict_ovr(slot, call, operation)
    elif slot[].kind == META_CALIBRATED:
        _predict_cal(slot, call, operation)
    elif _bag_kind(slot[].kind):
        _predict_bag(slot, call, operation)
    else:
        _predict_vote(slot, call, operation)


# ------------------------------------------------------------------ OUTPUTS


def _outputs_vote(call: _CallPtr, kind: Int, soft: Bool, classes: Int) raises:
    if call[].flags == MEMBER_OP_PREDICT:
        call[].out_cols = 1
        call[].out_dtype = MEMBER_I32 if kind == META_VOTE_CLASSIFIER else MEMBER_F64
    elif call[].flags == MEMBER_OP_PREDICT_PROBA and kind == META_VOTE_CLASSIFIER and soft:
        call[].out_cols = classes
        call[].out_dtype = MEMBER_F64
    else:
        member_unsupported(_name(kind), call[].flags)
    call[].classes = classes if kind == META_VOTE_CLASSIFIER else 0


def _outputs_unit(call: _CallPtr, kind: Int, classes: Int) raises:
    """A unit kind's outputs: predict one column (int32 codes for a
    classifier, float64 for BaggingRegressor), predict_proba `classes`
    float64 columns (classifiers)."""
    var classifier = _classifier_kind(kind)
    if call[].flags == MEMBER_OP_PREDICT:
        call[].out_cols = 1
        call[].out_dtype = MEMBER_I32 if classifier else MEMBER_F64
    elif call[].flags == MEMBER_OP_PREDICT_PROBA and classifier:
        call[].out_cols = classes
        call[].out_dtype = MEMBER_F64
    else:
        member_unsupported(_name(kind), call[].flags)
    call[].classes = classes if classifier else 0


def _outputs(call: _CallPtr) raises:
    if call[].state == 0:
        if call[].param_count != META_PARAM_COUNT or not call[].params:
            raise Error("member ABI: a meta member takes six parameters")
        var words = _F64P(unsafe_from_address=call[].params)
        var kind = Int(words[0])
        if _unit_kind(kind):
            _outputs_unit(call, kind, call[].classes)
            return
        if not _stack_kind_code(kind):
            _outputs_vote(call, kind, words[2] != 0.0, call[].classes)
            return
        var final = _plan_final(Int(words[1]))
        final.classes = call[].classes if kind == META_STACK_CLASSIFIER else 0
        var asked = call[].flags
        if asked != MEMBER_OP_PREDICT and kind != META_STACK_CLASSIFIER:
            member_unsupported(_name(kind), asked)
        # The final member's width does not depend on its input width.
        call[].out_cols = member_outputs(final, asked, max(call[].x.cols, 1), max(call[].y.cols, 1))
        call[].out_dtype = member_output_dtype(final, asked, max(call[].x.cols, 1), max(call[].y.cols, 1),
                                               MEMBER_DTYPE_UNSAID)
        if kind != META_STACK_CLASSIFIER:
            call[].classes = 0
        return
    var slot = _SlotPtr(unsafe_from_address=_lookup(call[].state, False))
    if _unit_kind(slot[].kind):
        _outputs_unit(call, slot[].kind, slot[].classes if slot[].fitted else call[].classes)
        return
    if not _stack_kind_code(slot[].kind):
        _outputs_vote(call, slot[].kind, slot[].flag, slot[].classes if slot[].fitted else call[].classes)
        return
    if not slot[].fitted:
        raise Error("NotFittedError: this " + _name(slot[].kind) + " member is not fitted yet")
    var asked = call[].flags
    if asked != MEMBER_OP_PREDICT and slot[].kind != META_STACK_CLASSIFIER:
        member_unsupported(_name(slot[].kind), asked)
    var bases = len(slot[].stack_kinds)
    var plan = slot[].plans[bases].copy()
    var final = member_call_of(plan, slot[].states[bases])
    call[].out_cols = member_outputs_of(plan, final, asked)
    call[].out_dtype = final.out_dtype
    call[].classes = slot[].classes


# ------------------------------------------------------------------ EXPORT / IMPORT


def _export(slot: _SlotPtr, call: _CallPtr) raises:
    if call[].flags == MEMBER_EXPORT_OWN:
        for u in range(len(slot[].states)):  # driver-loop: over the nested members (each owns its rows)
            if slot[].states[u] != 0:
                member_own(slot[].plans[u], slot[].states[u])
        return
    _fitted(slot)
    var block = call[].block
    if call[].flags == MEMBER_BLOCK_BY_INDEX:
        # vote / stack: nested, widths; a unit kind: nested, then its own words (calibrated pairs, bag draws),
        # then a bag's out-of-bag outputs.
        var blocks = 2
        if slot[].kind == META_OVR:
            blocks = 1
        elif _bag_kind(slot[].kind):
            blocks = 3
        if block < 0 or block >= blocks:
            raise Error("member ABI: a meta member has " + String(blocks) + " blocks")
        if block == 0:
            block = META_BLOCK_NESTED
        elif not _unit_kind(slot[].kind):
            block = META_BLOCK_WIDTHS
        elif block == 1:
            block = META_BLOCK_CAL if slot[].kind == META_CALIBRATED else META_BLOCK_DRAWS
        else:
            block = META_BLOCK_OOB
        call[].block = block
    call[].out_cols = 2
    if block == META_BLOCK_NESTED:
        call[].dst = MemberMatrix(Int(slot[].state_words.unsafe_ptr()), 1, len(slot[].state_words), MEMBER_I64)
    elif block == META_BLOCK_WIDTHS and not _unit_kind(slot[].kind):
        call[].dst = MemberMatrix(Int(slot[].width_words.unsafe_ptr()), 1 if len(slot[].width_words) > 0 else 0,
                                  len(slot[].width_words), MEMBER_I64)
    elif block == META_BLOCK_CAL and slot[].kind == META_CALIBRATED:
        call[].dst = MemberMatrix(Int(slot[].cal_words.unsafe_ptr()), 1, len(slot[].cal_words), MEMBER_F64)
    elif block == META_BLOCK_DRAWS and _bag_kind(slot[].kind):
        call[].dst = MemberMatrix(Int(slot[].draw_words.unsafe_ptr()), 1, len(slot[].draw_words), MEMBER_I32)
    elif block == META_BLOCK_OOB and _bag_kind(slot[].kind):
        call[].dst = MemberMatrix(Int(slot[].oob_words.unsafe_ptr()), 1 if len(slot[].oob_words) > 0 else 0,
                                  len(slot[].oob_words), MEMBER_F64)
    elif block == MEMBER_BLOCK_SCALARS:
        call[].dst = MemberMatrix(Int(slot[].scalar_words.unsafe_ptr()), 1, len(slot[].scalar_words), MEMBER_F64)
    else:
        raise Error("member ABI: a meta member has no fitted-state block " + String(block))


def _import_cal(slot: _SlotPtr, words: _F64P, count: Int) raises:
    """IMPORT `META_BLOCK_CAL` (`_unit_views`' layout): each pair's held
    classes and calibrators."""
    var bad = String("member ABI: a CalibratedClassifierCV member's pair words are malformed")
    if count < 1:
        raise Error(bad)
    var pairs = Int(words[0])
    var at = 1
    slot[].unit_held = List[List[Int32]](capacity=pairs)
    slot[].cal_iso = List[Bool]()
    slot[].cal_a = List[Float64]()
    slot[].cal_b = List[Float64]()
    slot[].cal_m = List[Int]()
    slot[].cal_kx = List[List[Float64]]()
    slot[].cal_ky = List[List[Float64]]()
    slot[].pair_first = List[Int](capacity=pairs)
    slot[].pair_count = List[Int](capacity=pairs)
    for _ in range(pairs):  # driver-loop: over the pairs
        if at >= count:
            raise Error(bad)
        var h = Int(words[at])
        at += 1
        if at + h + 1 > count:
            raise Error(bad)
        var held = List[Int32](capacity=h)
        for c in range(h):  # small-loop(a pair's held classes)
            held.append(Int32(Int(words[at + c])))
        at += h
        slot[].unit_held.append(held^)
        var cals = Int(words[at])
        at += 1
        slot[].pair_first.append(len(slot[].cal_iso))
        slot[].pair_count.append(cals)
        for _ in range(cals):  # driver-loop: over the pair's calibrators
            if at + 4 > count:
                raise Error(bad)
            slot[].cal_iso.append(words[at] != 0.0)
            slot[].cal_a.append(words[at + 1])
            slot[].cal_b.append(words[at + 2])
            var m = Int(words[at + 3])
            at += 4
            if m < 0 or at + 2 * m > count:
                raise Error(bad)
            slot[].cal_m.append(m)
            var kx = List[Float64](capacity=m)
            var ky = List[Float64](capacity=m)
            for q in range(m):  # small-loop(one calibrator's knots)
                kx.append(words[at + q])
            for q in range(m):  # small-loop(one calibrator's knots)
                ky.append(words[at + m + q])
            at += 2 * m
            slot[].cal_kx.append(kx^)
            slot[].cal_ky.append(ky^)


def _import_draws(slot: _SlotPtr, words: _I32P, count: Int) raises:
    """IMPORT `META_BLOCK_DRAWS` (`_unit_views`' layout): each member's
    feature subset, held classes and out-of-bag rows."""
    var bad = String("member ABI: a Bagging member's draw words are malformed")
    if count < 1:
        raise Error(bad)
    var units = Int(words[0])
    var at = 1
    slot[].unit_cols = List[List[Int32]](capacity=units)
    slot[].unit_held = List[List[Int32]](capacity=units)
    slot[].unit_oob = List[List[Int32]](capacity=units)
    for _ in range(units):  # driver-loop: over the member estimators
        for part in range(3):  # small-loop(a member's three runs: columns, held classes, out-of-bag rows)
            if at >= count:
                raise Error(bad)
            var c = Int(words[at])
            at += 1
            if c < 0 or at + c > count:
                raise Error(bad)
            var run = List[Int32](capacity=c)
            for j in range(c):  # small-loop(one run of words)
                run.append(words[at + j])
            at += c
            if part == 0:
                slot[].unit_cols.append(run^)
            elif part == 1:
                slot[].unit_held.append(run^)
            else:
                slot[].unit_oob.append(run^)


def _import(slot: _SlotPtr, call: _CallPtr) raises:
    if call[].block == META_BLOCK_NESTED:
        var count = call[].x.rows * call[].x.cols
        var unit = _unit_kind(slot[].kind)
        if (count < 1 if unit else count != len(slot[].plans)) or call[].x.dtype != MEMBER_I64 or call[].x.data == 0:
            raise Error("member ABI: a meta member imports one int64 state handle per member")
        _drop_states(slot)
        if unit:
            _expand(slot, count)
        var words = _I64P(unsafe_from_address=call[].x.data)
        slot[].states = List[Int](length=count, fill=0)
        for u in range(count):  # driver-loop: over the nested members (their imported states change owner)
            slot[].states[u] = Int(words[u])
    elif call[].block == META_BLOCK_WIDTHS and not _unit_kind(slot[].kind):
        var count = call[].x.rows * call[].x.cols
        if count != len(slot[].stack_kinds) or (count > 0 and (call[].x.dtype != MEMBER_I64 or call[].x.data == 0)):
            raise Error("member ABI: a stacking member imports one int64 width per stacked member")
        var words = _I64P(unsafe_from_address=call[].x.data)
        slot[].widths = List[Int](length=count, fill=0)
        slot[].total = 0
        for j in range(count):  # driver-loop: over the stacked members
            slot[].widths[j] = Int(words[j])
            slot[].total += slot[].widths[j]
    elif call[].block == META_BLOCK_CAL and slot[].kind == META_CALIBRATED:
        if call[].x.dtype != MEMBER_F64 or call[].x.data == 0:
            raise Error("member ABI: a CalibratedClassifierCV member imports its pairs as float64 words")
        _import_cal(slot, _F64P(unsafe_from_address=call[].x.data), call[].x.rows * call[].x.cols)
    elif call[].block == META_BLOCK_DRAWS and _bag_kind(slot[].kind):
        if call[].x.dtype != MEMBER_I32 or call[].x.data == 0:
            raise Error("member ABI: a Bagging member imports its draws as int32 words")
        _import_draws(slot, _I32P(unsafe_from_address=call[].x.data), call[].x.rows * call[].x.cols)
    elif call[].block != 0 or call[].flags != MEMBER_IMPORT_DONE:
        raise Error("member ABI: a meta member has no fitted-state block " + String(call[].block))
    if call[].flags == MEMBER_IMPORT_DONE:
        var kind = slot[].kind
        if len(slot[].states) != len(slot[].plans) or len(slot[].states) == 0:
            raise Error("member ABI: a meta member import needs its nested states")
        if _stack_kind_code(kind) and len(slot[].widths) != len(slot[].stack_kinds):
            raise Error("member ABI: a stacking member import needs its widths")
        if kind == META_CALIBRATED and len(slot[].pair_first) != len(slot[].states):
            raise Error("member ABI: a CalibratedClassifierCV member import needs one pair per member")
        if _bag_kind(kind) and len(slot[].unit_cols) != len(slot[].states):
            raise Error("member ABI: a Bagging member import needs one draw per member")
        if kind == META_OVR and len(slot[].states) != (1 if call[].classes == 2 else call[].classes):
            raise Error("member ABI: a OneVsRestClassifier member import needs one member per class (one for two)")
        slot[].classes = call[].classes if _classifier_kind(kind) else 0
        for u in range(len(slot[].plans)):  # driver-loop: over the nested members (their class count)
            var classes = slot[].classes
            if kind == META_OVR:
                classes = 2
            elif (kind == META_CALIBRATED or kind == META_BAG_CLASSIFIER) and u < len(slot[].unit_held):
                classes = _held_classes(slot[].unit_held[u], slot[].classes)
            slot[].plans[u].classes = classes
        _nested_outputs(slot)
        _unit_views(slot)
        slot[].fitted = True


# ------------------------------------------------------------------ PACK reads


def meta_state_plans(state: Int) raises -> List[MemberPlan]:
    """PACK (the boundary): the nested plans of a META state, in member
    order (a stack's final last), copied. The state owns their states
    (`META_BLOCK_NESTED`)."""
    var slot = _SlotPtr(unsafe_from_address=_lookup(state, False))
    return slot[].plans.copy()


def meta_state_notices(state: Int, mut owners: List[Int], mut words: List[List[Float64]]) raises:
    """PACK: the nested fit notices of a META state, in fit order (each with
    the nested member it belongs to), handed over once."""
    var slot = _SlotPtr(unsafe_from_address=_lookup(state, False))
    owners = slot[].notice_owner.copy()
    words = slot[].notice_words.copy()
    slot[].notice_owner = List[Int]()
    slot[].notice_words = List[List[Float64]]()


# ------------------------------------------------------------------ the entry


def _ensemble_member(operation: Int, address: Int) raises -> Int:
    var call = _CallPtr(unsafe_from_address=address)
    if operation == MEMBER_OP_CAPABILITIES:
        if call[].state == 0:
            return _UNION
        var kind = _SlotPtr(unsafe_from_address=_lookup(call[].state, False))
        if _unit_kind(kind[].kind):
            return _kind_mask(kind[].kind, kind[].flag, kind[].base_masks)
        return _kind_mask(kind[].kind, kind[].flag, kind[].masks)
    if operation == MEMBER_OP_NEW:
        _new(call)
        return 0
    if operation == MEMBER_OP_FREE:
        var held = _SlotPtr(unsafe_from_address=_lookup(call[].state, False))
        _drop_states(held)
        _ = _lookup(call[].state, True)
        call[].state = 0
        return 0
    if operation == MEMBER_OP_OUTPUTS:
        _outputs(call)
        return 0
    var slot = _SlotPtr(unsafe_from_address=_lookup(call[].state, False))
    if operation == MEMBER_OP_FIT:
        _fit(slot, call)
    elif operation == MEMBER_OP_PREDICT or operation == MEMBER_OP_PREDICT_PROBA or operation == MEMBER_OP_DECISION:
        _predict(slot, call, operation)
    elif operation == MEMBER_OP_EXPORT:
        _export(slot, call)
    elif operation == MEMBER_OP_IMPORT:
        _import(slot, call)
    else:
        member_unsupported(_name(slot[].kind), operation)
    return 0


def ensemble_member_v2(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
    result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """The member entry of the META kind (schema `MEMBER_ABI`)."""
    try:
        var word = _ensemble_member(operation, member_call_from_spans(operation, aa, na, np, nf))
        if result:
            MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = word
        return 0
    except error:
        native_descriptor_report(error, context, sink)
        return 1


def ensemble_member_publish(mode: Int, vendor: Int, publisher: Int) raises:
    """Publish this image's META member entry: one line in each x_trees
    image's module initialization, after its glue table (`vendor` -1: a build
    with no GPU vendor publishes nothing)."""
    if vendor < 0:
        return
    member_publish[MEMBER_FAMILY_ENSEMBLE](mode, vendor, publisher, ensemble_member_v2)


def ensemble_member_publish_image(vendor_word: Int) raises:
    """The one publication line of an x_trees image's module initialization,
    after its glue table: the image's tier and the vendor word of its glue
    table key (`x_trees_glue_vendor`; -1 publishes nothing)."""
    if vendor_word < 0:
        return
    var store = _STORE.get_or_create_ptr()
    ensemble_member_publish(Int(GLOBAL_NUMERIC_MODE), vendor_word, Int(store))
