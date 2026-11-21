# SPDX-License-Identifier: Apache-2.0
"""A member caller's UNPACK / PACK for the resident scope (core/member_abi.mojo
`MemberCall.scope` / `.pin`, THE PIN RULE there): the calling thread's scope
handle word, read once per public operation, and the references a caller
offers with a pinned X.

Python objects and the interpreter lock are needed for every function here:
call them at UNPACK (before the run releases the lock) and at PACK (after
its workers joined), never from a worker.

HOW A CALLER USES THEM. Once per public operation, and only for a member
whose kind has `MEMBER_CAN_SCOPE`: `member_scope_word(rows_context)`. Outside
`resident()` it answers `MEMBER_SCOPE_NONE`: the caller writes no scope and no
pin, and the member puts X on the device per call, as its own door does
outside a scope. Inside one: `member_pin_take(owner of X, calls)` with the
number of member calls that will read X as words of that object, the token
into `call.pin` before each such call, `member_pin_settle` after it, and
`member_pin_return(token, calls - taken)` at PACK. One reference backs every
offer, so a member's cache can never hold a reference nobody took.
"""
from std.python import PythonObject
from bindings.arena_owner_door import arena_pin, arena_unpin
from bindings.arena_scope_door import arena_scope_active


def member_scope_word(rows_context: PythonObject) raises -> Int:
    """The handle word of the calling thread's active resident scope
    (core/arena_state.mojo), or `MEMBER_SCOPE_NONE` (the same value as
    `ARENA_HANDLE_NONE`) outside `resident()`. `rows_context`: the arena
    rows context (`_arena_io._ROWS_CONTEXT`; model selection's context holds
    it as `family_msel_rows_context`). One thread-local slot read and, inside
    a scope, one downcast: once per public operation, never per member call.
    Raises what the scope's own doors raise for a thread-local that holds
    something else: the caller then keeps the path it had before the member
    (whose doors refuse in their own words), never a silent run without the
    scope."""
    var strings = rows_context["arena_native_types"]["str"]
    var local = rows_context["cache_local"]
    var local_type = rows_context["cache_local_type"]
    var getter = rows_context["cache_scope_owner_get"]
    var cls = rows_context["cache_scope_class"]
    return arena_scope_active(local, local_type, strings, getter, cls)


def member_pin_take(source: PythonObject, calls: Int) -> Int:
    """`calls` references on `source` (the object that owns X's host words),
    one per member call that will offer it; the pin token (0 when `calls` is
    not positive). A reference is one counter increment."""
    var token = 0
    for _ in range(calls):  # driver-loop: over the offering calls (one counter increment each)
        token = arena_pin(source)
    return token


def member_pin_return(token: Int, count: Int):
    """Drop `count` references of `member_pin_take` that no cache took. After
    every worker joined and every native release; the object's own finalizer
    may run here."""
    if token == 0:
        return
    for _ in range(count):  # driver-loop: over the references no cache took (one counter decrement each)
        arena_unpin(token)
