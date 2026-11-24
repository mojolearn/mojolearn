# SPDX-License-Identifier: Apache-2.0
"""A SET OF NATIVE MEMBERS fitted together (core/member_abi.mojo): the member
fit loop of an ensemble or meta-estimator (bagging, voting, the multi-output
and one-vs-rest wrappers) without a call of a member's own fit door.

WHAT THIS REPLACES for a member whose family has published its member entry.
The callers' loops fitted one member after the other with the GIL held: each
member's fit door re-admitted X and y, read its parameters and published its
state through Python objects. Here:
  * UNPACK (GIL held, once per member): `member_set_admit` admits the
    member's estimator (`member_admit`: route, parameter words, resolved
    entry). X, the targets, the weights and the row / column draws are read
    once as addresses.
  * RUN (`member_set_run`), TWO MODES per member (THE GIL RULE,
    core/member_abi.mojo `MEMBER_GIL_FREE`). A member whose KIND has the bit
    is fitted on `workers` host tasks with the GIL released
    (`host_parallelize` over lanes; a lane takes every `workers`-th such
    member). A member whose kind lacks it is fitted on the public thread
    with the GIL held, one after the other, after the workers joined (its
    route's entries release inside as they do for the family's own door and
    may rely on the GIL as their lock): what the callers' loops did. Each
    member has its own `MemberCall` and state. A member that draws rows or columns hands the
    draw to the family as an index view when the family reads one
    (`MEMBER_CAN_ROW_INDEX` / `MEMBER_CAN_COLUMN_INDEX`); otherwise the
    worker gathers the drawn block into its own buffers. A member over every
    row and column reads the caller's X by view: no copy.
  * PACK (`member_set_pack`, GIL held): the fitted state is written into the
    member's Python estimator (`member_pack`: the attributes its own fit
    writes, so the caller's predict paths and `estimators_` read it as
    before) and the state is released. Its FIT NOTICES (core/member_notices.mojo)
    are taken there and raised by `member_set_notices`, which the caller calls
    once its whole publication succeeded (a fallback to its own loop, whose
    doors raise them, then raises nothing twice).

THE RUN IS TYPED (lane bm-meta-2, 2026-11-22): `MemberSetRun` and every
step that fits (the class-cover pass, the gather, the workers, the turns)
are core/member_set_run.mojo, with no Python; this door holds UNPACK, the
pin references, the GIL lock around the run's steps and PACK.

ALL OR NOTHING: a caller admits every member first; when one is not a native
member (`member_set_admit` answers False) nothing has run and the caller
keeps its own loop for the whole set.

WORKERS (core/member_set_run.mojo `member_set_run_workers`): the host cores for members whose route is the
CPU, bounded by the member count; one when a member's route is a GPU (the
members share one device and its memory). Memory per worker: at most one
gathered block (drawn rows x drawn columns of float32, the targets and the
weights of those rows), which the serial loops held one at a time; nothing
for a member that reads the whole X or an index view. With one worker the
gather itself is split across the host cores.

A MEMBER ROUTE NEVER TURNS A REFUSAL INTO A RESULT. A classifier member is
fitted with the ensemble's class count, so a member whose own rows (a cv
fold, a bootstrap draw) lack a class would fit where its own fit door, which
encodes the classes it is given, fits another model or refuses (too few
classes), and where the caller then reads a narrower output (stacking's
"output width changed between folds"). `member_set_run` therefore reads every
classifier unit's codes first (core/member_set_run.mojo: one pass over its int32
codes, the units across the host cores, a lone unit's rows split across
them) and raises before any fit when a unit lacks a class or holds a code
outside 0 .. classes-1: nothing is published and the caller's own loop runs,
whose doors decide as they always did.

WHO REFUSES A NON-FINITE ROW. A member kind with `MEMBER_CAN_TEST_ROWS`
refuses non-finite X, y and weights itself, from the pass its kernel already
makes: the caller makes NO host scan for it (`member_set_tests`,
`MemberHeldCall.tests`); its refusal fails the set or the call, the caller's
door runs and raises the estimator's own words. A kind without the bit reads
admitted words: the caller proves the rows finite first (one host scan per
distinct block) and takes its doors when they are not.

THE RESIDENT SCOPE (`member_set_resident`; THE PIN RULE at core/member_abi.mojo
`MemberCall.pin`). A caller that fits its set inside `resident()` names, per
member, the Python object that owns the member's X words. For a member whose
family reads a scope (`MEMBER_CAN_SCOPE`) the thread's scope is read once per
set; `member_set_run` then takes one reference per such member with the
interpreter held, the member's fit carries the scope word and offers the
reference (never for a block the worker gathered itself), the member's
cache keeps X on the device for the set's other members and for the doors
that follow, and the references no cache took are dropped after the workers
joined. Outside `resident()`, or for a member without the bit, nothing is
read, pinned or offered: the member puts X per fit, as its own door does
there.

THE PREDICT SIDE (`member_held_open` / `member_held_run` /
`member_held_close`): one member call on the
fitted state the estimator holds (bindings/member_held_door.mojo), written
into a new Array of the element type the member's OUTPUTS answers
(`call.out_dtype`); not served = the caller's door.
"""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.member_boundary_door import member_admit_serving_labeled, member_pack, member_pack_labeled
from bindings.member_class_weight_unpack import member_cw_fill
from core.member_class_weight import member_cw_start, member_cw_pending
from bindings.member_held_door import MemberHeld, member_held, member_held_release, member_held_adopt
from bindings.member_fit_notices_pack import member_fit_notices_raise
from bindings.admitted_input_unpack import admitted_unfilled
from bindings.prep_native_estimator_door import prep_estimator_plan
from bindings.prep_native_estimator_door import prep_native_context
from bindings.member_scope_door import member_scope_word, member_pin_take, member_pin_return
from bindings.py_balanced import py_call, py_type, py_get, py_setitem
from core.device_route import DEVICE_CPU
from core.member_set_run import (
    MemberSetRun, member_set_run_tests, member_set_run_host, member_set_run_width, member_set_prepare,
    member_set_fit_free, member_set_turn, member_set_first_failure, member_set_run_release, MEMBER_SET_LACKING,
    MEMBER_SET_WORKER, MEMBER_SET_TURN_RELEASED, MEMBER_SET_TURN_HELD,
)
from core.member_abi import (
    MemberPlan, MemberCall, MemberMatrix, member_call, member_call_of, member_free, member_outputs_of,
    member_kind_capabilities,
    MEMBER_OP_CAPABILITIES, MEMBER_OP_FIT, MEMBER_OP_PREDICT, MEMBER_OP_PREDICT_PROBA, MEMBER_OP_DECISION,
    MEMBER_F32, MEMBER_F64, MEMBER_I32, MEMBER_I64, MEMBER_DTYPE_UNSAID,
    MEMBER_CAN_FIT, MEMBER_CAN_PREDICT, MEMBER_CAN_PREDICT_PROBA, MEMBER_CAN_DECISION,
    MEMBER_CAN_SAMPLE_WEIGHT, MEMBER_CAN_CLASSIFIER, MEMBER_IGNORES_Y,
    MEMBER_CAN_TEST_ROWS,
    MEMBER_CAN_SCOPE, MEMBER_BORROWS_FIT, MEMBER_GIL_FREE,
    MEMBER_GIL_DOOR, MEMBER_READS_Y_GIVEN,
)
from core.member_families import MEMBER_FAMILY_TREE, MEMBER_FAMILY_FOREST, MEMBER_FAMILY_EXTRA_TREES, MEMBER_FAMILY_ENSEMBLE
from bindings.member_owner_unpack import member_mark
from core.member_notices import member_notices_take

# TOMBSTONE: MOJOLEARN_MEMBER_SET_OFF (removed without measurement: plumbing, rule 2026-11-22) deleted 2026-11-23 by lane/ab-prep-classical; code recoverable at 4d76a9e16e.
# Restore: git apply ~/mojolearn-evidence/experiments-removed/MOJOLEARN_MEMBER_SET_OFF.patch; record in docs/TOMBSTONES.md.


struct MemberSet(Movable):
    """A caller's hold on its member set: the typed run plus, for UNPACK and
    PACK only, each member's Python estimator, its family context and the
    pins that keep its images loaded. Keep it alive past `member_set_pack`."""
    var run: MemberSetRun
    var estimators: PythonObject
    var contexts: PythonObject
    var fields: PythonObject
    var pins: PythonObject
    # Each member's import-time class plan (the held state's key at PACK).
    var class_plans: PythonObject
    # PACK leaves each fitted state with its estimator as its held member
    # state. Set by a caller whose predict side reads held states; else the
    # states are freed at PACK (a second copy of the model nobody reads).
    var adopt: Bool
    # With `adopt`: also the states of tree, forest and extra-trees members
    # whose route is the CPU. Set by a caller that predicts those through
    # their held states when no device copy of X is staged for them (bagging).
    var adopt_trees: Bool
    # After PACK: every packed member's state stayed with its estimator (a
    # later `member_held_open` of it is a hit: no admission, no IMPORT).
    var kept_all: Bool
    # THE RESIDENT SCOPE: per member, the object that owns its X words (None:
    # nothing is offered), and whether the thread's scope was read.
    var sources: PythonObject
    var scope_read: Bool
    # The caller's class labels in code order (None: it hands none over).
    # Set before `member_set_admit` by a caller whose code targets have
    # labels it holds: a classifier member whose class_weight is a dict
    # then reads its table over them at admission (lane mf-adm3,
    # bindings/member_class_weight_unpack.mojo). None: the codes 0 ..
    # classes-1 themselves (lane mf-left4: such a caller's own loop fits the
    # member on its codes, so they are the labels its fit door reads).
    var labels: PythonObject
    # THE GIVEN y (lane mf-tail5; core/member_abi.mojo `MemberCall.y_given`):
    # per member, 1 when its KIND reads it (MEMBER_READS_Y_GIVEN), answered
    # by the same one kind question as `tests` / `borrows`.
    var reads: List[UInt8]

    def __init__(out self, x: Int, rows: Int, cols: Int, y_dtype: Int, index_dtype: Int) raises:
        self.run = MemberSetRun(x, rows, cols, y_dtype, index_dtype)
        self.estimators = Python.list()
        self.contexts = Python.list()
        self.fields = Python.list()
        self.pins = Python.list()
        self.class_plans = Python.list()
        self.adopt = False
        self.adopt_trees = False
        self.kept_all = True
        self.sources = Python.list()
        self.scope_read = False
        self.labels = PythonObject(None)
        self.reads = List[UInt8]()

    def count(self) -> Int:
        return len(self.run.plans)


def member_set_admit(mut set: MemberSet, estimator: PythonObject, target: Int, weights: Int, row_draw: Int,
    row_count: Int, column_draw: Int, column_count: Int, classes: Int, context: PythonObject,
    b: PythonObject) raises -> Bool:
    """UNPACK: add `estimator` (an unfitted clone the caller made; PACK writes
    its fitted state into it) to the set, or answer False having added
    nothing when it is not a native member this set can fit: its class has no
    plan or no member mark, its family's admit keeps the estimator's own
    doors or refuses (`member_admit_serving`: never raises, the fallback
    rule), it has no fit, it is given weights it does not read, or the
    targets are class codes and its family has no classifier.
    `context` is the caller's context (it holds the model-selection plans),
    `b` the builtins module. `classes`: the class count of code targets, 0
    for a regressor. Raises only for a class_weight dict the member's
    estimator refuses over `set.labels` (its own words; lane mf-adm3)."""
    # TOMBSTONE: MOJOLEARN_MEMBER_SET_OFF (plumbing) deleted 2026-11-23 by lane/ab-prep-classical; code recoverable at 4d76a9e16e.
    var plan = prep_estimator_plan(estimator, context)
    if plan is None:
        return False
    if py_get(plan, "member") is None:
        return False
    var member_context = plan["context"]
    var fields = py_call(plan["fields_get"], estimator, py_type(estimator))
    # The family's union mask: a family of regressors and classifiers answers
    # the kind-exact refusal itself at NEW / FIT, and the caller then runs
    # its own loop (a failing `member_set_run` publishes nothing).
    var needs = MEMBER_CAN_FIT
    if weights != 0:
        needs = needs | MEMBER_CAN_SAMPLE_WEIGHT
    if classes > 0:
        needs = needs | MEMBER_CAN_CLASSIFIER
    # A caller that hands no labels fits the member on its codes 0 ..
    # classes-1, and its own loop fits the member's fit door on those same
    # codes: the codes ARE the labels that fit sees (lane mf-left4), so a
    # class_weight dict is read over them (keyed by codes, as that door
    # reads it) and no classifier admission of a set falls back for it.
    var deferred = Python.list() if set.labels is None else set.labels
    # A META member (lane mf-split4) is handed the labels of the codes it is
    # fitted on (the caller's, else the codes 0 .. classes-1 themselves): its
    # admission hands each nested estimator the labels ITS fit sees, so a
    # nested class_weight table is filled as this function fills a leaf's.
    # Read only for a classifier set whose caller holds no labels: one mark
    # read and one small list per such META admission.
    if classes > 0 and set.labels is None and member_mark(plan, b).family == MEMBER_FAMILY_ENSEMBLE:
        deferred = py_call(b.list, py_call(b.range, PythonObject(classes)))
    var owner = member_admit_serving_labeled(estimator, fields, plan, member_context, b, needs,
                                             deferred if classes > 0 else PythonObject(None))
    if not owner.published and classes > 0:
        # lane mf-member8 (round b): a kind whose fit reads no target (MEMBER_IGNORES_Y: NearestNeighbors,
        # RadiusNeighbors, OPTICS, MDS, SpectralEmbedding) is fitted on the codes as scikit-learn's meta fits it
        # on y, which it ignores (its predict then fails in scikit-learn's words, as there). Asked only on this
        # not-served path: one kind question, nothing on a served admission.
        var plain = member_admit_serving_labeled(estimator, fields, plan, member_context, b,
                                                 needs & ~MEMBER_CAN_CLASSIFIER, PythonObject(None))
        var unread = False
        if plain.published:
            try:
                unread = (member_kind_capabilities(plain.plan) & MEMBER_IGNORES_Y) != 0
            except:
                unread = False  # the set's rule: a member it cannot ask is not served (never raises)
        if unread:
            owner = plain^
    if not owner.published:
        return False
    # A class_weight dict's table over the caller's labels (lane mf-adm3);
    # a dict the estimator refuses, or a value it refuses (lane mf-left4),
    # raises its own words here, before any fit.
    var labels = PythonObject(None)
    if classes > 0:
        labels = set.labels
        if labels is None and member_cw_pending(owner.plan.params,
                                                member_cw_start(owner.plan.family, owner.plan.kind, owner.plan.params)):
            labels = py_call(b.list, py_call(b.range, PythonObject(classes)))
    if not member_cw_fill(owner, estimator, fields, labels, member_context, b):
        return False
    owner.plan.classes = classes
    # Whether this member's KIND tests the rows it reads: the answer of an
    # earlier member of the same family, kind and route (clones of one
    # estimator: the member before it), else asked once. A family whose union
    # mask lacks the bit is never asked (no state is minted for the answer).
    # THE LIFETIME RULE: whether this member's KIND keeps a pointer into its
    # fit buffers (MEMBER_BORROWS_FIT), answered with the same one question.
    # THE GIL RULE: whether this member's KIND may run on a worker with no GIL
    # (MEMBER_GIL_FREE), the same one question; no answer = it may not.
    var tests = UInt8(0)
    var borrows = UInt8(0)
    var reads = UInt8(0)
    var gil_free = MEMBER_SET_TURN_HELD
    if owner.plan.member.can(MEMBER_CAN_TEST_ROWS | MEMBER_BORROWS_FIT | MEMBER_GIL_FREE | MEMBER_GIL_DOOR
                             | MEMBER_READS_Y_GIVEN):
        var asked = False
        var v = len(set.run.plans) - 1
        while v >= 0:  # driver-loop: over the members admitted before (one compare when they are clones)
            if (set.run.plans[v].family == owner.plan.family and set.run.plans[v].kind == owner.plan.kind
                    and set.run.plans[v].route.device == owner.plan.route.device
                    and set.run.plans[v].route.vendor == owner.plan.route.vendor
                    and set.run.plans[v].route.mode == owner.plan.route.mode):
                tests = set.run.tests[v]
                borrows = set.run.borrows[v]
                gil_free = set.run.gil_free[v]
                reads = set.reads[v]
                asked = True
                break
            v -= 1
        if not asked:
            try:
                var kind_mask = member_kind_capabilities(owner.plan)
                if (kind_mask & MEMBER_CAN_TEST_ROWS) != 0:
                    tests = UInt8(1)
                if (kind_mask & MEMBER_BORROWS_FIT) != 0:
                    borrows = UInt8(1)
                if (kind_mask & MEMBER_READS_Y_GIVEN) != 0:
                    reads = UInt8(1)
                if (kind_mask & MEMBER_GIL_FREE) != 0:
                    gil_free = MEMBER_SET_WORKER
                elif (kind_mask & MEMBER_GIL_DOOR) != 0:
                    # serial on the public thread, the GIL released around
                    # the member's turn as its own door releases it
                    gil_free = MEMBER_SET_TURN_RELEASED
            except:
                # No answer: the caller scans the rows, the gathered block is
                # held as for a borrowing kind and the member is fitted on the
                # public thread with the GIL held (the safe readings).
                tests = UInt8(0)
                borrows = UInt8(1)
                reads = UInt8(0)
                gil_free = MEMBER_SET_TURN_HELD
    set.run.add(owner.plan.copy(), target, weights, row_draw, row_count, column_draw, column_count, tests, borrows,
                gil_free)
    set.reads.append(reads)
    _ = py_call(set.sources.append, PythonObject(None))
    _ = py_call(set.estimators.append, estimator)
    _ = py_call(set.contexts.append, member_context)
    _ = py_call(set.fields.append, fields)
    _ = py_call(set.pins.append, owner.pins)
    _ = py_call(set.class_plans.append, plan)
    _ = owner.published
    return True


def member_set_reads_given(set: MemberSet) -> Bool:
    """The member admitted last reads THE GIVEN y (its kind has
    MEMBER_READS_Y_GIVEN, lane mf-tail5): its caller then hands it with
    `member_set_given`; for any other member it makes none."""
    var last = len(set.reads) - 1
    return last >= 0 and set.reads[last] != UInt8(0)


def member_set_given(mut set: MemberSet, address: Int, dtype: Int):
    """UNPACK: THE GIVEN y of the member admitted last (core/member_abi.mojo
    `MemberCall.y_given`; lane mf-tail5): its targets' words as the caller
    was given them (MEMBER_F64 / _I64 / _I32), addressed exactly as its
    target (its row draw selects both). Ignored for a member whose kind
    does not read it. The caller keeps the words alive until the run's fits
    returned."""
    if member_set_reads_given(set):
        set.run.given_target(address, dtype)


def member_set_own_block(mut set: MemberSet, x: Int, rows: Int):
    """UNPACK: the member admitted last reads its own float32 C-order X
    block (`rows` x the set's columns, at address `x`; its target and
    weights addresses then hold `rows` values) and not the set's X: a cv
    fold's gathered rows, shared read-only by that fold's members. THE
    LIFETIME RULE: the caller keeps the block (and every target, weight and
    draw address it gave `member_set_admit`) alive until the LAST
    `member_set_pack` / `member_set_release` of the set has returned: a
    member that borrows its fit buffers reads them again at its export."""
    set.run.own_block(x, rows)


def member_set_resident(mut set: MemberSet, source: PythonObject, context: PythonObject) raises:
    """UNPACK, after the admit (and `member_set_own_block`) of the member
    admitted last: `source` is the Python object whose buffer holds that
    member's X words (the set's X, or its own block). Inside the calling
    thread's `resident()` scope a member whose family reads a scope then
    keeps that X on the device across the set's fits (module note, THE
    RESIDENT SCOPE). Nothing is read or recorded for a member without
    `MEMBER_CAN_SCOPE`; the scope is read once per set, at the first member
    that has it. `context` is the caller's context, as `member_set_admit`
    takes it. Raises what the scope's own doors raise for a thread-local
    that holds something else: the caller then keeps its own loop, whose
    doors refuse in their own words."""
    var last = len(set.run.plans) - 1
    if last < 0 or not set.run.plans[last].member.can(MEMBER_CAN_SCOPE):
        return
    if not set.scope_read:
        var rows_context = prep_native_context(context)["family_msel_rows_context"]
        set.run.scope = member_scope_word(rows_context)
        set.scope_read = True
    if set.run.scope < 0:
        return
    py_setitem(set.sources, PythonObject(last), source)


def member_set_tests(set: MemberSet, u: Int) -> Bool:
    """Member `u`'s kind refuses non-finite X, y and weights itself
    (`MEMBER_CAN_TEST_ROWS`): the caller makes no finite scan of what that
    member reads. False: the member reads admitted words and the caller
    proves them finite before `member_set_run`."""
    return member_set_run_tests(set.run, u)


def member_set_host(set: MemberSet) -> Bool:
    """Every member admitted so far runs on the CPU route."""
    return member_set_run_host(set.run)


def member_set_width(set: MemberSet) -> Int:
    """How many members a caller that fits in waves puts in one set: the
    host cores when every member admitted so far runs on the CPU, else one.
    A wave bounds what is held at once (the draws and the gathered blocks of
    one wave) to the worker count."""
    return member_set_run_width(set.run)


def member_set_release(mut set: MemberSet):
    """Release every state the set still holds. Never raises."""
    member_set_run_release(set.run)


def member_set_pins_take(mut set: MemberSet) raises:
    """THE PIN RULE, with the interpreter held: one reference per member that
    offers its X object (`member_set_resident`). A caller that runs the set's
    RUN itself (CalibratedClassifierCV: one GIL release around its whole
    typed RUN) takes them here, before the region, and returns them with
    `member_set_pins_return` after it, in a `finally`; `member_set_run` does
    the same."""
    for u in range(len(set.run.plans)):  # driver-loop: over the members (one reference each)
        set.run.pin_tokens[u] = 0
        var source = set.sources[u]
        if source is not None:
            set.run.pin_tokens[u] = member_pin_take(source, 1)


def member_set_pins_return(mut set: MemberSet) raises:
    """THE PIN RULE, with the interpreter held: drop the references no
    member's cache took (`member_set_pins_take`)."""
    for u in range(len(set.run.plans)):  # driver-loop: over the members (the references no cache took)
        if set.run.pin_tokens[u] != 0:
            member_pin_return(set.run.pin_tokens[u], 1)
            set.run.pin_tokens[u] = 0
    _ = set.pins


def member_set_run(mut set: MemberSet) raises:
    """RUN (core/member_set_run.mojo) with this door's two CPython duties:
    THE PIN RULE (one reference per member that offers its X object, taken
    with the interpreter held, dropped when no cache took it) and THE GIL
    RULE's lock: `member_set_fit_free` (the class-cover pass, then the
    members whose kind has `MEMBER_GIL_FREE` on host workers) runs with the
    GIL released; every other member then takes its turn on this thread,
    the GIL released around a `MEMBER_GIL_DOOR` member's turn and held for
    any other. After it every member holds a fitted state for
    `member_set_pack`; on a failure every state is released and the first
    failing member's words (in member order) are raised.

    Classifier sets are read first (module note: the class-cover pass): when
    a member's own rows lack a class, or hold a code out of range, nothing
    is fitted and the set raises, so the caller's own loop and its doors
    decide."""
    var count = len(set.run.plans)
    if count == 0:
        return
    var turns = member_set_prepare(set.run)
    var covered = True
    var py = Python()
    try:
        member_set_pins_take(set)
        with GILReleased(py):
            covered = member_set_fit_free(set.run)
        # GIL held again: the members whose kind may not run without it.
        if covered and turns:
            for u in range(count):  # driver-loop: over the members whose kind lacks MEMBER_GIL_FREE
                if set.run.gil_free[u] == MEMBER_SET_WORKER:
                    continue
                if set.run.gil_free[u] == MEMBER_SET_TURN_RELEASED:
                    # MEMBER_GIL_DOOR: still this thread, still one member at
                    # a time; other Python threads run during the fit, as
                    # during the family's own door.
                    with GILReleased(py):
                        member_set_turn(set.run, u)
                else:
                    member_set_turn(set.run, u)
    finally:
        member_set_pins_return(set)
    if not covered:
        raise Error(MEMBER_SET_LACKING)
    var first = member_set_first_failure(set.run)
    if first >= 0:
        var words = set.run.errors[first].copy()
        member_set_release(set)
        raise Error(words)


def member_set_pack(mut set: MemberSet, u: Int, classes: PythonObject, b: PythonObject) raises -> PythonObject:
    """PACK: member `u`'s fitted estimator (the clone given at admission,
    now holding the attributes its own fit writes); its state is released.
    `classes`: the labels the member's codes stand for (code c is
    `classes[c]`; a caller whose members see the ensemble's codes passes the
    codes themselves), or None for a regressor. Labels never cross the ABI:
    the family's pack takes them (`member_pack_labeled`) and writes its own
    class attributes, as its fit door does."""
    var estimator = set.estimators[u]
    var member_context = set.contexts[u]
    var fields = set.fields[u]
    try:
        set.run.notices[u].clear()
        member_notices_take(set.run.plans[u], set.run.states[u], set.run.notices[u])
        if classes is None:
            member_pack(estimator, fields, member_context, b, set.run.plans[u], set.run.states[u])
        else:
            member_pack_labeled(estimator, fields, member_context, b, set.run.plans[u], set.run.states[u], classes)
    except error:
        var failed = MemberCall(set.run.plans[u].route, set.run.states[u])
        member_free(set.run.plans[u], failed)
        set.run.states[u] = 0
        set.run.part_free(u)
        raise error^
    # `adopt`: the fitted state stays with the estimator as its held member
    # state (bindings/member_held_door.mojo), so the caller's predict side
    # reads it without an import. Not kept: freed here, as before.
    var kept = False
    var family = set.run.plans[u].family
    # A tree, forest or extra-trees member is predicted by its own door (from
    # the one device copy of X the caller stages): its state would be a
    # second copy of the model nobody reads, except where the caller says it
    # reads them (`adopt_trees`, CPU routes: nothing is staged there).
    var read = (not (family == MEMBER_FAMILY_TREE or family == MEMBER_FAMILY_FOREST
        or family == MEMBER_FAMILY_EXTRA_TREES)
        or (set.adopt_trees and set.run.plans[u].route.device == DEVICE_CPU))
    if set.adopt and read:
        try:
            kept = member_held_adopt(estimator, set.class_plans[u], member_context, set.pins[u], set.run.plans[u],
                                     set.run.states[u])
        except:
            kept = False
    if not kept:
        set.kept_all = False
        var call = MemberCall(set.run.plans[u].route, set.run.states[u])
        member_free(set.run.plans[u], call)
    set.run.states[u] = 0
    # THE LIFETIME RULE: the state is freed, or adopted (`member_held_adopt`
    # made a borrowing kind own its rows first): the held block goes now.
    set.run.part_free(u)
    return estimator


def member_set_notices(mut set: MemberSet) raises:
    """PACK: every packed member's fit notices, in member order, with its own
    context, as each member's own fit door raised them (GIL held). The caller
    calls it once its publication loop succeeded; a second call raises
    nothing."""
    for u in range(len(set.run.notices)):  # driver-loop: over the members (a few warnings each)
        if len(set.run.notices[u]) > 0:
            member_fit_notices_raise(set.run.notices[u], set.contexts[u])
            set.run.notices[u].clear()


def member_marked(estimator: PythonObject, context: PythonObject) -> Bool:
    """True when `estimator`'s class carries a member mark (its family may
    serve it; `member_held_open` still answers per call). One class-plan
    read: a caller asks once per public operation for members of one class,
    before work only the native route needs (its finite proof of X). Never
    raises."""
    # TOMBSTONE: MOJOLEARN_MEMBER_SET_OFF (plumbing) deleted 2026-11-23 by lane/ab-prep-classical; code recoverable at 4d76a9e16e.
    try:
        var plan = prep_estimator_plan(estimator, context)
        if plan is None:
            return False
        return py_get(plan, "member") is not None
    except:
        return False


struct MemberHeldCall(Movable):
    """One public operation's opened call on a fitted member's held state
    (`member_held_open`). `served` False: nothing is held, the caller's door
    runs. `tests`: the member's kind refuses non-finite rows itself
    (`MEMBER_CAN_TEST_ROWS`): the caller scans nothing; False: the caller
    proves the rows finite before `member_held_run`, or closes and takes its
    door. `cols` / `dtype`: the columns per row and the element type
    (MEMBER_F32 / F64 / I32 / I64) the member writes for `operation`.
    `host`: the member's route is the CPU. `gil_free`: the member's kind
    has `MEMBER_GIL_FREE` (THE GIL RULE): `member_held_run` releases the GIL
    around its call; False: the call is made with the GIL held."""
    var held: MemberHeld
    var served: Bool
    var tests: Bool
    var gil_free: Bool
    var host: Bool
    var operation: Int
    var cols: Int
    var dtype: Int

    def __init__(out self):
        self.held = MemberHeld()
        self.served = False
        self.tests = False
        self.gil_free = False
        self.host = False
        self.operation = MEMBER_OP_PREDICT
        self.cols = 0
        self.dtype = MEMBER_F32


def _out_typestr(dtype: Int) -> String:
    if dtype == MEMBER_F64:
        return "<f8"
    if dtype == MEMBER_I32:
        return "<i4"
    if dtype == MEMBER_I64:
        return "<i8"
    return "<f4"


def _member_held_open(mut opened: MemberHeldCall, member: PythonObject, operation: Int, classifier: Bool,
    width: Int, skip_trees: Bool, context: PythonObject, b: PythonObject) raises:
    var plan = prep_estimator_plan(member, context)
    if plan is None:
        return
    var mark = py_get(plan, "member")
    if mark is None or Bool(py=plan["classifier"]) != classifier:
        return
    if skip_trees:
        var family = Int(py=mark[0])
        if family == MEMBER_FAMILY_TREE or family == MEMBER_FAMILY_FOREST or family == MEMBER_FAMILY_EXTRA_TREES:
            return
    var needs = MEMBER_CAN_PREDICT
    if operation == MEMBER_OP_PREDICT_PROBA:
        needs = MEMBER_CAN_PREDICT_PROBA
    elif operation == MEMBER_OP_DECISION:
        needs = MEMBER_CAN_DECISION
    var member_context = plan["context"]
    var fields = py_call(plan["fields_get"], member, py_type(member))
    opened.held = member_held(member, fields, plan, member_context, b, needs)
    if not opened.held.published:
        return
    opened.operation = operation
    var call = member_call_of(opened.held.plan, opened.held.state)
    var cols = member_outputs_of(opened.held.plan, call, operation)
    if cols < 1 or (width != 0 and cols != width):
        return
    # The element type is the member's own answer (`call.out_dtype`, written
    # by the OUTPUTS call above): codes int32 or int64, scores and
    # probabilities float32 or float64. A member that does not say it
    # (published before the word existed): a regressor writes float32 (every
    # published regressor documents it); a classifier is not served and its
    # door runs. Nothing is learned from a refused call.
    var dtype = call.out_dtype
    var codes = classifier and operation == MEMBER_OP_PREDICT
    if dtype == MEMBER_DTYPE_UNSAID:
        if classifier:
            return
        dtype = MEMBER_F32
    if codes:
        if dtype != MEMBER_I32 and dtype != MEMBER_I64:
            return
    elif dtype != MEMBER_F32 and dtype != MEMBER_F64:
        return
    # Kind-exact, asked of the held state (no state is minted); a family
    # whose union mask lacks the bit is not asked.
    # THE GIL RULE: the same answer says whether the call may run released.
    if opened.held.plan.member.can(MEMBER_CAN_TEST_ROWS | MEMBER_GIL_FREE | MEMBER_GIL_DOOR):
        var kind_mask = member_call(opened.held.plan.member, MEMBER_OP_CAPABILITIES, call)
        opened.tests = (kind_mask & MEMBER_CAN_TEST_ROWS) != 0
        # One call on this thread either way: released for a kind that may
        # run with no GIL, and for a kind whose door releases around the
        # same work (MEMBER_GIL_DOOR).
        opened.gil_free = (kind_mask & (MEMBER_GIL_FREE | MEMBER_GIL_DOOR)) != 0
    opened.host = opened.held.plan.route.device == DEVICE_CPU
    opened.cols = cols
    opened.dtype = dtype
    opened.served = True


def member_held_open(member: PythonObject, operation: Int, classifier: Bool, width: Int, skip_trees: Bool,
    context: PythonObject, b: PythonObject) -> MemberHeldCall:
    """PREDICT SIDE, UNPACK: open the fitted estimator `member`'s `operation`
    (MEMBER_OP_PREDICT, _PREDICT_PROBA or _DECISION) on the member state it
    holds (bindings/member_held_door.mojo: a hit after the first use, and the
    state a native fit left with it). Once per member per public operation.
    The answer says whether the member is served, how wide and of which
    element type its output is, and whether its kind tests the rows itself
    (`tests`): the caller makes its finite scan of X only when it does not.
    Close it with `member_held_close` on every path.

    `served` False, with nothing changed, when: the class has no member
    mark, it is not of the asked kind (`classifier`), its output has not
    `width` columns (0: any), its family lacks the operation, admits or
    imports nothing, or a classifier's member does not say its element type.
    The caller then runs the member's own door, which raises the estimator's
    own words (the fallback rule). Never raises. `skip_trees`: a tree,
    forest or extra-trees member is not served (the caller's door predicts
    those from the one device copy of X it staged for all of them)."""
    var opened = MemberHeldCall()
    # TOMBSTONE: MOJOLEARN_MEMBER_SET_OFF (plumbing) deleted 2026-11-23 by lane/ab-prep-classical; code recoverable at 4d76a9e16e.
    try:
        _member_held_open(opened, member, operation, classifier, width, skip_trees, context, b)
    except:
        opened.served = False
    return opened^


def member_held_run(mut opened: MemberHeldCall, x: Int, n: Int, d: Int, context: PythonObject) -> PythonObject:
    """PREDICT SIDE, RUN: the opened member's operation over the `n` x `d`
    float32 C-order host rows at `x`: ONE member call (the GIL released
    around it for a kind with `MEMBER_GIL_FREE`; held for any other kind,
    whose entries release inside as for the family's own door), written into a new Array of the member's element type, (n,) for one
    column per row, else (n, columns). A classifier's predict writes its
    class CODES (code c is the member's c-th class); the caller maps them.
    None when the call refuses (a member that tests its rows refuses a
    non-finite one here): the caller's door runs and raises the estimator's
    own words. Never raises. THE CALLER keeps the rows alive through the
    call and, for a member that does not test them (`tests` False), has
    proven them finite. Serial by nature: one state, one call at a time (the
    member splits its own rows)."""
    if not opened.served:
        return PythonObject(None)
    try:
        var call = member_call_of(opened.held.plan, opened.held.state)
        var shape = List[Int](length=1, fill=n)
        if opened.cols > 1:
            shape.append(opened.cols)
        # Unfilled: the member writes every cell before anything reads it.
        var dst = admitted_unfilled(shape, _out_typestr(opened.dtype), context)
        call.x = MemberMatrix(x, n, d)
        call.dst = MemberMatrix(dst.addr, n, opened.cols, opened.dtype)
        var plan_member = opened.held.plan.member
        var operation = opened.operation
        if opened.gil_free:
            with GILReleased(Python()):
                _ = member_call(plan_member, operation, call)
        else:
            _ = member_call(plan_member, operation, call)
        _ = opened.served   # the hold (its plan words, its owner pin) outlives the call
        return dst.keep
    except:
        return PythonObject(None)


def member_held_close(mut opened: MemberHeldCall):
    """End of the opened call, every path, after `member_held_run` returned:
    a kept state stays with the estimator. Never raises."""
    member_held_release(opened.held)
    opened.served = False
