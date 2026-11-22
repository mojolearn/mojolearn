# SPDX-License-Identifier: Apache-2.0
"""A fitted estimator's HELD member state (MEMBER-ABI-V2.md item 19): the
calls a predict-side caller makes instead of `member_admit` + `member_state`
(an IMPORT of the fitted state) at every public operation.

UNPACK, GIL held, ONCE PER PUBLIC OPERATION and per member estimator (never
per row chunk, per fold or per worker):

    var held = member_held(model, fields, plan, context, b, needs)
    if not held.published: <the member's own doors, as before the mark>
    ... RUN: MemberCall(held.plan.route, held.state); PREDICT / TRANSFORM ...
    member_held_release(held)        # every exit; after the RUN has joined

A HIT is one generation read of the estimator (bindings/py_model_records.mojo
`model_generation`: the instance dictionary's watch epoch, its address and
the class address), one record read and three compares. No admission, no
parameter read, no IMPORT, no copy of the fitted state. A MISS (first use,
a refit, `set_params`, any write to the instance dictionary, another class
plan or context) admits the member, imports the fitted attributes once
(`member_state`) and keeps the new state, which replaces and frees the old.

WHAT IT REPLACES: the member's own predict door through the Python model
(parameter and fitted-state checks at every call), or one `member_state`
IMPORT per public call. A hit does strictly less than either.

WHEN NOTHING IS KEPT (`held.loose` True): the estimator's generation word is
0 (no dictionary watcher: Python 3.10 / 3.11, an instance without a plain
`__dict__`), the estimator takes no weak reference, or the owner type's image
(the runtime classical host) is not loaded. The state is then this operation's own and
`member_held_release` frees it: one IMPORT per public operation, as before.

THE FALLBACK RULE holds: `member_held` never raises for a member it cannot
serve (unpublished family, a missing capability bit, an admission or an
IMPORT the family refuses, an estimator that is not fitted): `published` is
False and the caller's old path runs and raises its own words.

ONE CALL AT A TIME PER HELD STATE: a held state is the estimator's fitted
state. A caller makes one member call at a time on it (the member splits its
own rows across the cores); it never hands one state to several workers
unless the family's member file documents its predict as reentrant. Workers
that need their own states are fits (`member_new`), not this.

AT FIT (`member_held_adopt`): a caller that fitted a member natively and
packed it into a Python estimator (`member_pack`) hands the fitted state to
that estimator instead of freeing it, so its first predict is already a hit.

SERIAL BY NATURE: one record read per member estimator per operation.
"""
from std.python import PythonObject
from bindings.py_balanced import py_get
from bindings.member_boundary_door import member_admit_serving, member_state
from bindings.member_held_owner_unpack import MemberHeldOwner, MEMBER_HELD_SLOT, member_held_type_ready
from bindings.py_model_records import (
    model_generation, records_get, records_put, records_drop, records_stampable, RECORDS_PREPARED,
)
from core.member_abi import MemberPlan, MemberCall, member_free, member_own, MEMBER_CAN_IMPORT


struct MemberHeld(Movable):
    """One operation's hold on a fitted member state. `published` False:
    nothing is held, take the old path. `keeper` is the estimator's owner
    object: keep `held` alive until the RUN has joined. `loose`: no owner
    could be kept and `member_held_release` frees the state."""
    var published: Bool
    var plan: MemberPlan
    var state: Int
    var keeper: PythonObject
    var loose: Bool

    def __init__(out self):
        self.published = False
        self.plan = MemberPlan()
        self.state = 0
        self.keeper = PythonObject(None)
        self.loose = False


def _member_held_keep(model: PythonObject, plan: PythonObject, context: PythonObject, pins: PythonObject,
    mut held: MemberHeld) raises:
    """Give `held.state` to an owner object and keep it for `model`. The
    generation is read HERE, after the admission and the import (either may
    have written the instance dictionary). Leaves `held.loose` True when no
    owner type or no generation exists."""
    held.loose = True
    if not member_held_type_ready():
        return
    # Opens the estimator's record entry (and its dictionary watch) when this
    # is its first record; False when no generation can be kept for it.
    if not records_stampable(model):
        return
    var generation = model_generation(model)
    var owner = MemberHeldOwner(held.plan.copy(), held.state, generation, plan, context, pins)
    # From here the owner object frees the state (its `__del__`), kept or not.
    var record = PythonObject(alloc=owner^)
    held.loose = False
    held.keeper = record
    _ = records_put(RECORDS_PREPARED, model, MEMBER_HELD_SLOT, record)


def member_held(model: PythonObject, fields: PythonObject, plan: PythonObject, context: PythonObject,
    b: PythonObject, needs: Int) raises -> MemberHeld:
    """The fitted member state of the fitted estimator `model`: the one it
    holds when nothing changed since it was made, else a new one imported
    from its fitted attributes and kept. `plan` is the class's import-time
    plan, `context` the family context and `needs` every capability bit the
    caller will ask (as for `member_admit_serving`). Raises only for a
    failure of the record table itself."""
    var held = MemberHeld()
    var generation = model_generation(model)
    if generation != 0:
        var record = records_get(RECORDS_PREPARED, model, MEMBER_HELD_SLOT)
        if record is not None:
            var native = record.downcast_value_ptr[MemberHeldOwner]()
            if (native[].generation == generation and native[].plan_key is plan and native[].context is context
                    and native[].plan.member.has(needs)):
                held.plan = native[].plan.copy()
                held.state = native[].state
                held.keeper = record
                held.published = True
                return held^
    var owner = member_admit_serving(model, fields, plan, context, b, needs | MEMBER_CAN_IMPORT)
    if not owner.published:
        return held^
    var state = 0
    try:
        # A prefit classifier's class count is its public `classes_`
        # (member_state's rule: the caller sets `plan.classes` first).
        var labels = py_get(fields, "classes_")
        if labels is not None and owner.plan.classes == 0:
            owner.plan.classes = len(labels)
        state = member_state(model, fields, context, b, owner.plan)
    except:
        return held^
    if state == 0:
        return held^
    held.plan = owner.plan.copy()
    held.state = state
    held.published = True
    try:
        _member_held_keep(model, plan, context, owner.pins, held)
    except:
        # Nothing kept: the state is this operation's (`loose`), unless the
        # owner object was already made, which then frees it itself.
        pass
    return held^


def member_held_adopt(model: PythonObject, plan: PythonObject, context: PythonObject, pins: PythonObject,
    member: MemberPlan, state: Int) raises -> Bool:
    """AT FIT: make the fitted member `state` the held state of `model`, the
    Python estimator it was just packed into (call after `member_pack`: the
    generation is read after the pack's writes). True: the estimator owns
    the state now, do NOT free it. False: nothing was kept and the state is
    still the caller's to free.

    THE LIFETIME RULE (core/member_abi.mojo): a held state outlives the
    operation and so every buffer its fit was handed (the caller's X, a
    gathered fold, a draw). A kind that borrows its fit buffers
    (`MEMBER_BORROWS_FIT`) is therefore told to own its rows HERE, once per
    adopted state (`member_own`: one capability question; a kind that owns
    its state, or a state its export already copied, copies nothing), and
    only when the state will be kept. A member that cannot own is not
    adopted (False): the caller frees it and the estimator's first predict
    imports its fitted attributes instead. The caller's fit buffers must
    still be valid when this is called."""
    var held = MemberHeld()
    held.plan = member.copy()
    held.state = state
    try:
        # The two tests `_member_held_keep` makes first: no copy is made for
        # a state nothing would keep.
        if not member_held_type_ready() or not records_stampable(model):
            return False
        member_own(member, state)
    except:
        return False
    try:
        _member_held_keep(model, plan, context, pins, held)
    except:
        pass
    if held.loose:
        return False
    # Kept in the table, or the owner object dies here and frees the state:
    # either way it is no longer the caller's.
    return True


def member_held_release(mut held: MemberHeld):
    """End of the operation, every exit, after the RUN has joined. Frees a
    state no owner keeps; a kept state stays with its estimator. Never raises."""
    if held.loose and held.state != 0:
        var call = MemberCall(held.plan.route, held.state)
        member_free(held.plan, call)
    held.state = 0
    held.loose = False
    held.published = False

