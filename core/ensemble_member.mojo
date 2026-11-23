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
"""
from std.sys.info import size_of
from bindings.native_unit_os_threads_run import native_unit_cores
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
from core.member_families import MEMBER_FAMILY_ENSEMBLE
from core.member_notices import member_notices_take
from core.member_set_run import (
    MemberSetRun, member_set_fit, member_set_run_release,
    MEMBER_SET_WORKER, MEMBER_SET_TURN_RELEASED, MEMBER_SET_TURN_HELD,
)
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.shared_slot import SharedSlot, shared_slot_lock, shared_slot_unlock
from core.trees_entry_table import TreesEntryKey
from xtrees.glue_calls import xt_ref, xt_device_folds, xt_gather_f32, xt_scatter
from xtrees.trees_meta_vote_run import (
    VoteMember, VoteRows, vote_codes_run, vote_soft_run, vote_regress_run, stack_place_run,
    VOTE_ALL_ANSWERED, STACK_PREDICT, STACK_PROBA, STACK_DECISION,
)

comptime META_VOTE_CLASSIFIER = 0
comptime META_VOTE_REGRESSOR = 1
comptime META_STACK_CLASSIFIER = 2
comptime META_STACK_REGRESSOR = 3
comptime META_PARAM_COUNT = 6
comptime META_BLOCK_NESTED = MEMBER_BLOCK_FAMILY
comptime META_BLOCK_WIDTHS = MEMBER_BLOCK_FAMILY + 1

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
    return kind == META_VOTE_CLASSIFIER or kind == META_STACK_CLASSIFIER


# ------------------------------------------------------------ the plan store


struct MetaPlanSlot(Copyable, Movable, Defaultable):
    var generation: Int
    var live: Bool
    var plans: List[MemberPlan]
    var weights: List[Float64]
    var stack_kinds: List[Int]
    var masks: List[Int]

    def __init__(out self):
        self.generation = 0
        self.live = False
        self.plans = List[MemberPlan]()
        self.weights = List[Float64]()
        self.stack_kinds = List[Int]()
        self.masks = List[Int]()


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


# --------------------------------------------------------------- the masks


def _all(masks: List[Int], bit: Int) -> Bool:
    for u in range(len(masks)):  # driver-loop: over the nested kinds (one mask each)
        if (masks[u] & bit) == 0:
            return False
    return True


def _kind_mask(kind: Int, flag: Bool, masks: List[Int]) -> Int:
    """The kind-exact capability mask of a META state over nested kinds
    `masks` (module note, KIND-EXACT BITS)."""
    var mask = MEMBER_CAN_FIT | MEMBER_CAN_PREDICT | MEMBER_CAN_EXPORT | MEMBER_CAN_IMPORT | MEMBER_BORROWS_FIT
    if _classifier_kind(kind):
        mask |= MEMBER_CAN_CLASSIFIER
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
    if kind < META_VOTE_CLASSIFIER or kind > META_STACK_REGRESSOR:
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
    finally:
        shared_slot_unlock(lock)
    try:
        if len(slot[].plans) == 0:
            raise Error("member ABI: a meta member plan holds no members")
        if _stack_kind_code(kind) and (len(slot[].stack_kinds) + 1 != len(slot[].plans) or slot[].folds < 2):
            raise Error("member ABI: a stacking member needs its members, its final estimator and cv >= 2")
        if not _stack_kind_code(kind) and len(slot[].weights) != len(slot[].plans):
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
    # The slot's own plan carries the class count too: the RUNs' member calls
    # read it in place (`_member_of`).
    slot[].plans[u].classes = classes
    var plan = slot[].plans[u].copy()
    var mask = slot[].masks[u]
    run.add(plan^, target, weights, draw, count, 0, 0, UInt8(1) if (mask & MEMBER_CAN_TEST_ROWS) != 0 else UInt8(0),
            UInt8(1) if (mask & MEMBER_BORROWS_FIT) != 0 else UInt8(0), _set_word(mask))


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
    slot[].classes = k
    slot[].states = List[Int](length=bases + 1, fill=0)
    # 1. The members on every row (one set).
    var full = _set_of(slot, x, n, d, y_dtype)
    for u in range(bases):  # driver-loop: over the stacked members (admission, once each)
        _add(slot, full, u, k, y, weights, 0, 0)
    member_set_fit(full)
    _keep_states(slot, full, 0)
    _nested_outputs(slot)
    var counts = List[Int](length=bases, fill=0)
    for u in range(bases):  # driver-loop: over the stacked members (their class counts)
        if slot[].stack_kinds[u] != STACK_PREDICT:
            counts[u] = k
    # 2. The width probe over the first rows.
    var members = List[VoteMember](capacity=bases)
    for u in range(bases):  # driver-loop: over the stacked members
        members.append(_member_of(slot, u, slot[].states[u], _stack_operation(slot[].stack_kinds[u]), 1.0))
    var head = VoteRows(x, min(n, 2), d)
    var widths = List[Int]()
    var refused = stack_place_run(glue, head, members, slot[].stack_kinds, counts, k, binary, 0, 0, 0, 0, 0,
                                  widths, 0)
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
    # 3. The cv folds: stratified over the codes (classifier), plain (regressor).
    var n_splits = slot[].folds
    var fold_counts = List[Int32](length=n_splits + 1, fill=Int32(0))
    var fold_rows = List[Int32](length=max(n_splits * n, 1), fill=Int32(0))
    var codes_at = y if classifier else Int(fold_counts.unsafe_ptr())
    var status = xt_device_folds(glue, codes_at, Int(fold_rows.unsafe_ptr()), Int(fold_counts.unsafe_ptr()), n,
                                 n_splits, k)
    if status == 1:
        raise Error("ValueError: n_splits=" + String(n_splits)
                    + " cannot be greater than the number of members in each class")
    if status != 0:
        raise Error("ValueError: y codes outside [0, n_classes)")
    var used = 0
    for f in range(n_splits):  # driver-loop: over the cv folds
        if Int(fold_counts[f]) > 0:
            used = f
    used += 1
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
    var rows_at = Int(fold_rows.unsafe_ptr())
    var f0 = 0
    while f0 < used:  # driver-loop: over waves of cv folds
        var f1 = min(used, f0 + wave)
        var run = _set_of(slot, x, n, d, y_dtype)
        for f in range(f0, f1):  # driver-loop: over the wave's cv folds (their units)
            var c = Int(fold_counts[f])
            for u in range(bases):  # driver-loop: over the stacked members
                _add(slot, run, u, k, y, weights, rows_at + 4 * (f * n), n - c)
        try:
            member_set_fit(run)
            for v in range(len(run.states)):  # driver-loop: over the wave's units (their fit notices)
                _take_notices(slot, v % bases, run.plans[v], run.states[v])
            for f in range(f0, f1):  # driver-loop: over the wave's cv folds (test outputs)
                var c = Int(fold_counts[f])
                var test_at = rows_at + 4 * (f * n + n - c)
                var xte = List[Float32](length=max(c * d, 1), fill=0.0)
                _ = xt_gather_f32(glue, x, test_at, Int(columns.unsafe_ptr()), Int(xte.unsafe_ptr()), n, d, c, d)
                var fold_members = List[VoteMember](capacity=bases)
                for u in range(bases):  # driver-loop: over the stacked members
                    fold_members.append(_member_of(slot, u, run.states[(f - f0) * bases + u],
                                                   _stack_operation(slot[].stack_kinds[u]), 1.0))
                var rows = VoteRows(Int(xte.unsafe_ptr()), c, d)
                var got = List[Int]()
                var missed = stack_place_run(glue, rows, fold_members, slot[].stack_kinds, counts, k, binary,
                                             meta_at, n, width, test_at, 0, got, 0)
                if missed != VOTE_ALL_ANSWERED:
                    _raise_member(fold_members[missed], rows)
                for j in range(bases):  # driver-loop: over the stacked members
                    if got[j] != widths[j]:
                        raise Error("ValueError: a member's output width changed between folds")
                _ = len(xte)  # read through the RUN above
        finally:
            # scikit-learn keeps no fold estimator: the wave's states and
            # gathered train blocks go here.
            member_set_run_release(run)
        f0 = f1
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
    var last = _set_of(slot, Int(slot[].meta32.unsafe_ptr()), n, width, y_dtype)
    _add(slot, last, bases, k, y, weights, 0, 0)
    member_set_fit(last)
    _keep_states(slot, last, bases)


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
    _drop_states(slot)
    slot[].features = call[].x.cols
    try:
        if _stack_kind_code(slot[].kind):
            _fit_stack(slot, call)
        else:
            _fit_vote(slot, call)
        _nested_outputs(slot)
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
    var counts = List[Int](length=bases, fill=0)
    var members = List[VoteMember](capacity=bases)
    for u in range(bases):  # driver-loop: over the stacked members
        if slot[].stack_kinds[u] != STACK_PREDICT:
            counts[u] = k
        members.append(_member_of(slot, u, slot[].states[u], _stack_operation(slot[].stack_kinds[u]), 1.0))
    var meta = List[Float64](length=max(n * width, 1), fill=0.0)
    var order = List[Int32](length=max(n, 1), fill=Int32(0))
    for i in range(n):  # small-loop(the row numbers 0 .. n-1 the scatter places, once per call)
        order[i] = Int32(i)
    var got = List[Int]()
    var refused = stack_place_run(slot[].glue, rows, members, slot[].stack_kinds, counts, k, k == 2,
                                  Int(meta.unsafe_ptr()), n, width, Int(order.unsafe_ptr()), 0, got, 0)
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


def _predict(slot: _SlotPtr, call: _CallPtr, operation: Int) raises:
    _fitted(slot)
    if _stack_kind_code(slot[].kind):
        if operation == MEMBER_OP_PREDICT_PROBA or operation == MEMBER_OP_DECISION:
            if slot[].kind != META_STACK_CLASSIFIER:
                member_unsupported(_name(slot[].kind), operation)
        _predict_stack(slot, call, operation)
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


def _outputs(call: _CallPtr) raises:
    if call[].state == 0:
        if call[].param_count != META_PARAM_COUNT or not call[].params:
            raise Error("member ABI: a meta member takes six parameters")
        var words = _F64P(unsafe_from_address=call[].params)
        var kind = Int(words[0])
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
        if block < 0 or block > 1:
            raise Error("member ABI: a meta member has two blocks")
        block = META_BLOCK_NESTED if block == 0 else META_BLOCK_WIDTHS
        call[].block = block
    call[].out_cols = 2
    if block == META_BLOCK_NESTED:
        call[].dst = MemberMatrix(Int(slot[].state_words.unsafe_ptr()), 1, len(slot[].state_words), MEMBER_I64)
    elif block == META_BLOCK_WIDTHS:
        call[].dst = MemberMatrix(Int(slot[].width_words.unsafe_ptr()), 1 if len(slot[].width_words) > 0 else 0,
                                  len(slot[].width_words), MEMBER_I64)
    elif block == MEMBER_BLOCK_SCALARS:
        call[].dst = MemberMatrix(Int(slot[].scalar_words.unsafe_ptr()), 1, len(slot[].scalar_words), MEMBER_F64)
    else:
        raise Error("member ABI: a meta member has no fitted-state block " + String(block))


def _import(slot: _SlotPtr, call: _CallPtr) raises:
    if call[].block == META_BLOCK_NESTED:
        var count = call[].x.rows * call[].x.cols
        if count != len(slot[].plans) or call[].x.dtype != MEMBER_I64 or call[].x.data == 0:
            raise Error("member ABI: a meta member imports one int64 state handle per member")
        _drop_states(slot)
        var words = _I64P(unsafe_from_address=call[].x.data)
        slot[].states = List[Int](length=count, fill=0)
        for u in range(count):  # driver-loop: over the nested members (their imported states change owner)
            slot[].states[u] = Int(words[u])
    elif call[].block == META_BLOCK_WIDTHS:
        var count = call[].x.rows * call[].x.cols
        if count != len(slot[].stack_kinds) or (count > 0 and (call[].x.dtype != MEMBER_I64 or call[].x.data == 0)):
            raise Error("member ABI: a stacking member imports one int64 width per stacked member")
        var words = _I64P(unsafe_from_address=call[].x.data)
        slot[].widths = List[Int](length=count, fill=0)
        slot[].total = 0
        for j in range(count):  # driver-loop: over the stacked members
            slot[].widths[j] = Int(words[j])
            slot[].total += slot[].widths[j]
    elif call[].block != 0 or call[].flags != MEMBER_IMPORT_DONE:
        raise Error("member ABI: a meta member has no fitted-state block " + String(call[].block))
    if call[].flags == MEMBER_IMPORT_DONE:
        if len(slot[].states) != len(slot[].plans):
            raise Error("member ABI: a meta member import needs its nested states")
        if _stack_kind_code(slot[].kind) and len(slot[].widths) != len(slot[].stack_kinds):
            raise Error("member ABI: a stacking member import needs its widths")
        slot[].classes = call[].classes if _classifier_kind(slot[].kind) else 0
        for u in range(len(slot[].plans)):  # driver-loop: over the nested members (their class count)
            slot[].plans[u].classes = slot[].classes
        _nested_outputs(slot)
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
