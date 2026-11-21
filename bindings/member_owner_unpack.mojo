# SPDX-License-Identifier: Apache-2.0
"""A caller's hold on one native member (core/member_abi.mojo): the typed
plan its RUN reads, and the Python pins that keep the member's images loaded
while the caller's operation runs. UNPACK only: the owner never enters RUN;
workers get `owner.plan` (typed words) and nothing else.

SERIAL BY NATURE: one record read per member admission.
"""
from std.python import PythonObject
from bindings.py_balanced import py_get, py_type
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

    def __init__(out self):
        self.plan = MemberPlan()
        self.pins = PythonObject(None)
        self.published = False


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
