# SPDX-License-Identifier: Apache-2.0
"""RUN helpers of the tree ensembles' `score` door
(bindings/trees_public_score.mojo). No Python import:
`tools/python_free_dirs.sh` covers this directory.

Lane loops-trees-ab (2026-11-11): the door's private COUNT program
(ScoreProgram, AccuracyPlan, score_accuracy_check / _plan / _answer and the
SCORE_* stage constants) is deleted with the door's private accuracy copy;
the door uses the shared `labels_accuracy` (core/labels_accuracy_plan.mojo,
the same stage words). Code recoverable at bf33328698.
"""


def score_shape_text(shape: List[Int]) -> String:
    """The Python repr of a shape tuple of ints: `()`, `(3,)`, `(3, 4)`."""
    var text = String("(")
    for k in range(len(shape)):  # driver-loop: over array rank metadata
        if k > 0:
            text += ", "
        text += String(shape[k])
    if len(shape) == 1:
        text += ","
    return text + ")"
