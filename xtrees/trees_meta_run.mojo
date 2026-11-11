# SPDX-License-Identifier: Apache-2.0
"""RUN of the meta-estimator door (bindings/trees_public_meta.mojo): the
calibrated-probability program `x_trees_calibrated_proba` reads
(xtrees/calibrated_apply.mojo), built from typed values. The door's UNPACK
calls each member's `decision_function` / `predict_proba` (a public Python
method of the member, the boundary) and reads its calibrators' parameters;
this module lays the words out. No Python import:
`tools/python_free_dirs.sh` covers this directory.

Lane doorsplit-trees (2026-11-11): moved from `_calibrated_fused` word for
word (same addresses, integer and float words in the same order: no bit
change). Building the program is a few words per calibrator; the program
itself runs as one device launch.
"""


struct CalibratedProgram(Movable):
    """The address, integer and float words of one calibrated predict:
    a header (n, k, pairs), then per pair four words (score columns, the
    positive column, member classes, calibrators) and per calibrator four
    more."""
    var addrs: List[Int]
    var ints: List[Int]
    var floats: List[Float64]

    def __init__(out self, acc_address: Int, n: Int, k: Int, pairs: Int):
        self.addrs = [acc_address]
        self.ints = [n, k, pairs]
        self.floats = [Float64(n), Float64(k), Float64(pairs)]

    def _word(mut self, i: Int, f: Float64):
        self.ints.append(i)
        self.floats.append(f)

    def pair(mut self, scores_address: Int, classes_address: Int, score_cols: Int, at: Int, member_classes: Int,
             calibrators: Int):
        """A (member, calibrators) pair: its score block, its class codes
        (0 for a binary pair) and the four integer words."""
        self.addrs.append(scores_address)
        self.addrs.append(classes_address)
        var words: List[Int] = [score_cols, at, member_classes, calibrators]
        for w in range(4):  # small-loop(the four integer words of a pair)
            self._word(words[w], Float64(words[w]))

    def sigmoid(mut self, a: Float64, b: Float64):
        """A Platt calibrator `1 / (1 + exp(a * s + b))`."""
        self.addrs.append(0)
        self.addrs.append(0)
        self._word(0, 0.0)
        self._word(0, a)
        self._word(0, b)
        self._word(0, 0.0)

    def isotonic(mut self, knots_x: Int, knots_y: Int, m: Int):
        """An isotonic calibrator over `m` knots."""
        self.addrs.append(knots_x)
        self.addrs.append(knots_y)
        self._word(1, 1.0)
        self._word(0, 0.0)
        self._word(0, 0.0)
        self._word(m, Float64(m))


def meta_binary_column_check(at: Int) raises:
    """`list.index(1)` of a binary member's classes: refused when absent."""
    if at < 0:
        raise Error("1 is not in list")
