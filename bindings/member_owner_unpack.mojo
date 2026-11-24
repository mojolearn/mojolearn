# SPDX-License-Identifier: Apache-2.0
"""A caller's hold on one native member (core/member_abi.mojo): the typed
plan its RUN reads, and the Python pins that keep the member's images loaded
while the caller's operation runs. UNPACK only: the owner never enters RUN;
workers get `owner.plan` (typed words) and nothing else.

SERIAL BY NATURE: one record read per member admission.
"""
from std.python import PythonObject
from bindings.py_balanced import py_call, py_contains, py_get, py_item, py_setitem, py_type
from core.member_abi import MemberPlan
from core.member_families import MEMBER_FAMILY_COUNT


struct MemberOwner(Movable):
    """`published` False: the member's class has no member entry and the
    caller keeps the member's own doors. True: `plan` is complete (family,
    kind, resolved entry and capability mask, route, parameter words) and
    `pins` holds what keeps its images loaded; keep the owner alive past the
    last member call (`_ = owner.published` after the RUN block)."""
    var plan: MemberPlan
    var pins: PythonObject
    var published: Bool
    # The FIT caller's class labels in code order, set before a META
    # family's admission reads them (lane mf-split4,
    # bindings/member_boundary_door.mojo `member_admit_labeled`); None (or
    # an empty list): none known. A leaf family's admission never reads it.
    var labels: PythonObject
    # A SERVING admission (lane mf-door7, 2026-11-23): the caller only
    # predicts on fitted state (`member_admit_serving` with `needs` without
    # MEMBER_CAN_FIT, a META import's nested admission), so a META family
    # skips the settings only its FIT reads (cv, scorers, a search's
    # candidates, the device route of a fit). False: a fit may follow.
    var serving: Bool

    def __init__(out self):
        self.plan = MemberPlan()
        self.pins = PythonObject(None)
        self.published = False
        self.labels = PythonObject(None)
        self.serving = False


@fieldwise_init
struct MemberMark(ImplicitlyCopyable, Movable):
    """A class's member mark: its family number (-1: none) and kind word."""
    var family: Int
    var kind: Int


def member_mark(plan: PythonObject, b: PythonObject) raises -> MemberMark:
    """The mark model selection's import-time plan holds for the class
    (`plan["member"]`: the class's own `_mojolearn_member = (family, kind)`,
    never an inherited one), checked: two exact integers, a known family."""
    var mark = py_get(plan, "member")
    if mark is None:
        return MemberMark(-1, 0)
    if py_type(mark) is not b.tuple or len(mark) != 2:
        raise Error("TypeError: mojolearn: a class's member mark is (family, kind)")
    var family = mark[0]
    var kind = mark[1]
    if py_type(family) is not b.int or py_type(kind) is not b.int:
        raise Error("TypeError: mojolearn: a class's member mark is two exact integers")
    var number = Int(py=family)
    if number < 0 or number >= MEMBER_FAMILY_COUNT:
        raise Error("ValueError: mojolearn: unknown member family number " + String(number))
    return MemberMark(number, Int(py=kind))


def member_host_only(model: PythonObject) raises -> Bool:
    """UNPACK: the class's OWN host-only word, `_mojolearn_member_host = True`
    (lane mf-cal3b, 2026-11-23): an explicit Host* class of
    python/mojolearn/_classical_host.py, whose own fit and predict bind the
    host image of its family (`_bind`) whatever its `device`. A family's
    admit that reads it binds that host image (the kernel SVM, neighbors and
    GP admits), never the device route. One read of the class's own
    dictionary per member admission (a registered class's metaclass is
    `type`: no hook runs); an inherited word does not count."""
    var own = py_type(model).__dict__
    var key = PythonObject("_mojolearn_member_host")
    if not py_contains(own, key):
        return False
    return py_item(own, key) is PythonObject(True)


def member_labels_hold(fields: PythonObject, labels: PythonObject) raises:
    """PACK: hold a classifier member's class labels in the instance
    dictionary `fields` under the one reserved key, for the family's pack
    (bindings/member_boundary_door.mojo `member_pack_labeled`)."""
    py_setitem(fields, PythonObject("_mojolearn_member_labels"), labels)


def member_labels_take(fields: PythonObject, b: PythonObject) raises -> PythonObject:
    """PACK: the class labels the caller handed to `member_pack_labeled`
    (None: the caller gave none), removed from the instance dictionary. A
    family's pack that writes the labels in its own form calls this once;
    code `c` of the member's `y` and predictions is `labels[c]`. `fields` is
    the exact instance dictionary, so the unbound `dict.pop` runs no hook."""
    var key = PythonObject("_mojolearn_member_labels")
    var labels = py_call(b.dict.pop, fields, key, PythonObject(None))
    return labels
