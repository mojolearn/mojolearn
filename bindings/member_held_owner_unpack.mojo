# SPDX-License-Identifier: Apache-2.0
"""The ONE held fitted member state of a fitted estimator: the native owner
type and its registration (bindings/member_held_door.mojo holds the calls).

A predict-side caller (a meta-estimator's predict, bagging / AdaBoost predict,
an explainer) needs the FITTED state of a member estimator as a member handle
(core/member_abi.mojo). Importing the fitted attributes into a new state at
every public call is a copy of the fitted state per call. Instead the
estimator keeps one `MemberHeldOwner` in its native record table
(bindings/py_model_records.mojo, the table that keeps its device route): the
typed plan, the state handle, the generation word of the estimator the state
was imported from, and the pins that keep the member's images loaded.

Lifetime: the record table releases the owner when the estimator dies (its
weak reference's finalizer) or when a newer owner replaces it; `__del__`
frees the member state. A public operation that reads the owner keeps the
Python object named until its RUN has joined, so a refit on another thread
releases the old state only after the operations in flight on it return.

The type is registered ONCE, by the runtime classical host (the module that
owns the device route owner type); every other image allocates and downcasts
through the registered type and registers nothing.

SERIAL BY NATURE: one type registration per process.
"""
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder, lookup_py_type_object
from bindings.py_type_registry import py_type_absent, py_type_attach
from core.member_abi import MemberPlan, MemberCall, member_free

#: The record name under `RECORDS_PREPARED` (one per estimator).
comptime MEMBER_HELD_SLOT = "member_held_state_v1"


struct MemberHeldOwner(Movable, Writable):
    """One fitted member state and what it was made for."""
    var plan: MemberPlan
    var state: Int
    # The estimator's generation word when the state was imported (or
    # adopted at fit): any later change of its instance dictionary, its
    # `__dict__` object or its class is another word.
    var generation: Int
    # The import-time class plan and the family context the state was
    # admitted with. Held, so their addresses stay theirs; a hit compares
    # identities only.
    var plan_key: PythonObject
    var context: PythonObject
    # What keeps the member's images loaded (the admission's pins). Never the
    # estimator itself: a record must not retain its owner.
    var pins: PythonObject

    def __init__(out self, var plan: MemberPlan, state: Int, generation: Int, plan_key: PythonObject,
        context: PythonObject, pins: PythonObject):
        self.plan = plan^
        self.state = state
        self.generation = generation
        self.plan_key = plan_key
        self.context = context
        self.pins = pins

    def __del__(deinit self):
        # The images are pinned until the fields below are released, so the
        # member's entry is still loaded for this FREE. Never raises.
        if self.state != 0:
            var call = MemberCall(self.plan.route, self.state)
            member_free(self.plan, call)
        _ = self.pins
        _ = self.context
        _ = self.plan_key

    def write_to(self, mut writer: Some[Writer]):
        writer.write("MemberHeldOwner(family=", self.plan.family, ", kind=", self.plan.kind, ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


def register_member_held_owner(mut module: PythonModuleBuilder) raises:
    """Module setup of the ONE registering image (before `finalize()`)."""
    var existing = PythonObject(None)
    try:
        existing = lookup_py_type_object[MemberHeldOwner]()
    except error:
        if not py_type_absent(error):
            raise error
        _ = module.add_type[MemberHeldOwner]("_MemberHeldOwner")
        return
    py_type_attach(module.module, "_MemberHeldOwner", existing)


def member_held_type_ready() -> Bool:
    """True when the owner type exists in this interpreter (the registering
    image is loaded). False: nothing can be held and a caller's state is its
    own to free."""
    try:
        _ = lookup_py_type_object[MemberHeldOwner]()
        return True
    except:
        return False
