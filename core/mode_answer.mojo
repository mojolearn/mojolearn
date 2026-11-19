# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of Mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The set that answers a numeric-mode word: the one shared step of every
door's mode admission (lane fx-fast-selector).

`fast` is `identical` with named relaxations switched on. Off Apple no fast
set is built yet, and the selector decides once, at import, that the identical
set answers the name `fast` (python/mojolearn/_backend.py, `_FAST_ANSWER`).
A door admits the user's word as before (`fast`, `identical`, or its refusal
with its own wording) and then calls `answer_mode_word` (here) or
`answer_mode_object` (bindings/mode_answer_door.mojo, for a door that keys its
tables by the Python word), which replace an admitted `fast` by `identical`
where that decision holds. Everything after the admission (the mode-keyed
tables, the compiled-mode checks, the fitted-mode comparisons, the branches)
then sees the word of the set that runs. The estimator's own `numeric_mode`
attribute is never written: it reads back what the user passed.

Cost. On a macOS build the helpers return at compile time, so Apple is
unchanged. Elsewhere an admitted `identical` pays one comparison of the
admitted word; only the word `fast` reads the process-wide answer, which
`connect_mode_answer` stored once at import. No Python lookup, lock,
allocation or string build.

Not connected (a native test, or an install that carries a fast set): the
answer is False and `fast` names the fast set as before.

A leaf module with no Python in it (std and core/shared_slot only), so that
core/linear_metadata.mojo and the doors can all import it.
"""

from std.sys.info import CompilationTarget
from core.shared_slot import SharedSlot


struct ModeAnswer(Defaultable, Movable):
    # True when the identical set answers the name `fast` in this process.
    var fast_is_identical: Bool

    def __init__(out self):
        self.fast_is_identical = False


# The named global may be shared by every Mojo extension in the process:
# bump the name's version whenever ModeAnswer's layout changes.
comptime MODE_ANSWER = SharedSlot[ModeAnswer, "MojolearnModeAnswerV1"]


def connect_mode_answer(fast_is_identical: Bool) raises:
    """Store the selector's decision, once, at package import."""
    MODE_ANSWER.get_or_create_ptr()[].fast_is_identical = fast_is_identical


@always_inline
def fast_is_answered_by_identical() raises -> Bool:
    """Whether the identical set answers the word `fast` in this process."""
    comptime if CompilationTarget.is_macos():
        return False
    return MODE_ANSWER.get_or_create_ptr()[].fast_is_identical


@always_inline
def answer_mode_word(mut word: String) raises:
    """`word` is an admitted mode word. Replace `fast` by `identical` where
    the identical set answers it."""
    comptime if CompilationTarget.is_macos():
        return
    if word == "fast":
        if MODE_ANSWER.get_or_create_ptr()[].fast_is_identical:
            word = "identical"
