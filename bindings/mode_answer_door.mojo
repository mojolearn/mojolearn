# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of Mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python-word form of the shared mode-admission step
(core/mode_answer.mojo, lane fx-fast-selector), for a door that keys its
import-connected tables by the Python mode word. The Python str "identical"
is retained once at import, so answering an admitted `fast` allocates
nothing per call."""

from std.python import PythonObject
from std.sys.info import CompilationTarget
from core.mode_answer import connect_mode_answer, fast_is_answered_by_identical
from core.shared_slot import SharedSlot


struct ModeAnswerWords(Defaultable, Movable):
    # The retained Python str "identical" (one entry once connected).
    var words: List[PythonObject]

    def __init__(out self):
        self.words = List[PythonObject]()


# The named global may be shared by every Mojo extension in the process:
# bump the name's version whenever ModeAnswerWords' layout changes.
comptime MODE_ANSWER_WORDS = SharedSlot[ModeAnswerWords, "MojolearnModeAnswerWordsV1"]


def connect_mode_answer_words(fast_is_identical: Bool, identical: PythonObject) raises:
    """Store the selector's decision and the retained word, once, at import."""
    var held = MODE_ANSWER_WORDS.get_or_create_ptr()
    if len(held[].words) == 0:
        held[].words.append(identical)
    connect_mode_answer(fast_is_identical)


@always_inline
def answer_mode_object(mut mode: PythonObject, word: String) raises:
    """`mode` is an admitted Python mode word and `word` its text. Replace
    `fast` by the retained `identical` object where the identical set
    answers it."""
    comptime if CompilationTarget.is_macos():
        return
    if word == "fast":
        if fast_is_answered_by_identical():
            var held = MODE_ANSWER_WORDS.get_or_create_ptr()
            if len(held[].words) > 0:
                mode = held[].words[0]
