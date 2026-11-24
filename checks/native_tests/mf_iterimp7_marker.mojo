# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""Lane mf-iterimp7: the unlabeled-marker code an IterativeImputer classifier step writes for LabelPropagation /
LabelSpreading (x_prep/iterimp_run.mojo `iterimp_marker_code`), and the member's default (no marker word), with
no Python. Oracle from scikit-learn's own reading of the step's target (sklearn/semi_supervised/_label_propagation.py
`fit`: `unlabeled = y == -1`, `classes = np.unique(y)`, `classes = classes[classes != -1]`), written from that
source, not from the module under test: the marker's code is the position of -1 among the step's ascending classes
(`iterimp_class_values` makes them ascending, np.unique's order).

 1. Classes [-1, 0, 2]: code 0; [-3, -1, 4]: code 1; [-2, -1]: code 1 (the marker is the last class).
 2. No -1: [0, 1, 2] and [-2, 3]: -1 (the step writes nothing; the plan keeps its -1.0 "no marker" word); an empty
    class list: -1.
 3. Only -1 itself counts: [-0.99999994 (the float32 next to -1), 0]: -1; and IterImpMember's default marker word
    is -1 (a kind without a marker writes nothing).

FAIL-PROOF EDITS (each must make the named case fail): in `iterimp_marker_code` compare `classes[c] == Float32(1.0)`
(case 1); return 0 instead of -1 after the loop (case 2); compare `classes[c] <= Float32(-1.0)` (case 1, [-3, -1, 4]
gives 0); initialise `self.marker_word = 0` in IterImpMember (case 3).
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, require
from core.member_abi import MemberPlan
from x_prep.iterimp_run import iterimp_marker_code, IterImpMember

comptime CASES = 3


def _code(values: List[Float32], want: Int, label: String) raises:
    var got = iterimp_marker_code(values)
    require(got == want, label + ": got " + String(got) + " want " + String(want))


def check_marker_found() raises:
    var a: List[Float32] = [-1.0, 0.0, 2.0]
    _code(a, 0, "[-1, 0, 2]")
    var b: List[Float32] = [-3.0, -1.0, 4.0]
    _code(b, 1, "[-3, -1, 4]")
    var c: List[Float32] = [-2.0, -1.0]
    _code(c, 1, "[-2, -1]")


def check_no_marker() raises:
    var a: List[Float32] = [0.0, 1.0, 2.0]
    _code(a, -1, "[0, 1, 2]")
    var b: List[Float32] = [-2.0, 3.0]
    _code(b, -1, "[-2, 3]")
    _code(List[Float32](), -1, "[]")


def check_near_and_default() raises:
    var a: List[Float32] = [-0.99999994, 0.0]
    _code(a, -1, "[-0.99999994, 0]")
    var member = IterImpMember(MemberPlan(), False, False)
    require(member.marker_word == -1, "IterImpMember default marker word: got " + String(member.marker_word))


def test_mf_iterimp7_marker() raises:
    def run_case(which: Int) raises:
        if which == 0:
            check_marker_found()
        elif which == 1:
            check_no_marker()
        else:
            check_near_and_default()

    check_cases("mf_iterimp7_marker", CASES, run_case)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
