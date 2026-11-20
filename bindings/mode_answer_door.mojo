# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of Mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python-word form of the shared mode-admission step
(core/mode_answer.mojo, lane fx-fast-selector), for a door that keys its
import-connected tables by the Python mode word. The Python str "identical"
is retained once at import, so answering an admitted `fast` allocates
nothing per call."""

from std.python import PythonObject
from bindings.py_balanced import py_type, py_get
from std.sys.info import CompilationTarget
from core.mode_answer import MODE_FAMILIES_ALL, MODE_FAMILIES_ANSWERED, connect_mode_answer, fast_is_answered_by_identical
from core.shared_slot import SharedSlot


struct ModeAnswerWords(Defaultable, Movable):
    # The retained Python str "identical" (one entry once connected).
    var words: List[PythonObject]

    def __init__(out self):
        self.words = List[PythonObject]()


# The named global may be shared by every Mojo extension in the process:
# bump the name's version whenever ModeAnswerWords' layout changes.
comptime MODE_ANSWER_WORDS = SharedSlot[ModeAnswerWords, "MojolearnModeAnswerWordsV1"]


def connect_mode_answer_words(fast_is_identical: Bool, identical: PythonObject, trial_families: Int = -1) raises:
    """Store the selector's decision and the retained word, once, at import.
    `trial_families` (0 or more) is the built-install test's switch: that
    family mask replaces MODE_FAMILIES_ANSWERED, so one built install can
    show which families are whole under any candidate set."""
    var held = MODE_ANSWER_WORDS.get_or_create_ptr()
    if len(held[].words) == 0:
        held[].words.append(identical)
    var families = 0
    if fast_is_identical:
        families = MODE_FAMILIES_ANSWERED if trial_families < 0 else (trial_families & MODE_FAMILIES_ALL)
    connect_mode_answer(families)


@always_inline
def answer_mode_object(mut mode: PythonObject, word: String, family: Int) raises:
    """`mode` is an admitted Python mode word of a door of `family` and `word`
    its text. Replace `fast` by the retained `identical` object where the
    identical set answers it for that family."""
    comptime if CompilationTarget.is_macos():
        return
    if word == "fast":
        if fast_is_answered_by_identical(family):
            var held = MODE_ANSWER_WORDS.get_or_create_ptr()
            if len(held[].words) > 0:
                mode = held[].words[0]


@always_inline
def answer_mode_python(mut mode: PythonObject, family: Int) raises:
    """`mode` is an admitted exact Python str mode word of a door of `family`
    whose text the door has not built. Replace `fast` by the retained
    `identical` object where the identical set answers it for that family.
    The retained object is tested by identity first (the interned word the
    package and a user's literal pass), so an `identical` call builds no
    String; only another word is read as text."""
    comptime if CompilationTarget.is_macos():
        return
    if not fast_is_answered_by_identical(family):
        return
    var held = MODE_ANSWER_WORDS.get_or_create_ptr()
    if len(held[].words) == 0 or mode is held[].words[0]:
        return
    if String(py=mode) == "fast":
        mode = held[].words[0]


def fast_answers_compiled_code(want: PythonObject, codes: PythonObject, got: Int, family: Int) raises -> Bool:
    """The MISMATCH branch of a door's "requested word against the binary's
    compiled mode code" check, for a door of `family` whose check reads the
    raw word: whether `want` is the word `fast`, the identical set answers
    it for that family, and `got` is the code `codes` holds for `identical`.
    Called only after the door's own comparison failed, so a call whose word
    matches its binary pays nothing; the refusal that follows a False is the
    door's own, unchanged."""
    comptime if CompilationTarget.is_macos():
        return False
    if not fast_is_answered_by_identical(family):
        return False
    var held = MODE_ANSWER_WORDS.get_or_create_ptr()
    if len(held[].words) == 0:
        return False
    var identical = held[].words[0]
    if py_type(want) is not py_type(identical):
        return False
    if String(py=want) != "fast":
        return False
    var code = py_get(codes, identical, PythonObject(None))
    if code is None:
        return False
    return Int(py=code) == got
