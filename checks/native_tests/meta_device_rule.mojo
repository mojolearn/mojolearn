# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""A META-ESTIMATOR'S MEMBER DEVICE (lane vdf-meta; core/meta_device_rule.mojo), in a process with no Python.

1. the meta-estimator's own word: the shared rule and words (`function_device_member`): handed to a member naming
   none, equal words admitted, a member's own kept under None, another word refused naming the meta-estimator;
2. a word HANDED DOWN by an outer meta-estimator: handed on to a member naming none, equal words admitted;
3. the same with another explicit word on the member: refused naming the outer one ("handed to <meta>").

FAIL-PROOF EDITS (each must make the named case fail): drop the `origin` branch (case 2's words are the shared
ones, case 2 fails); drop the final raise (case 2); return `own` where a member names none (case 1).
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, require
from core.device_route import DEVICE_CPU, DEVICE_GPU, DEVICE_INSTALL
from core.meta_device_rule import meta_member_device

comptime CASES = 3


def _words(request: Int, own: Int, origin: String) -> String:
    try:
        _ = meta_member_device(request, own, "StackingClassifier", origin, "DecisionTreeClassifier")
    except problem:
        return String(problem)
    return String("")


def run_case(which: Int) raises:
    if which == 0:
        require(meta_member_device(DEVICE_CPU, DEVICE_INSTALL, "M", "", "C") == DEVICE_CPU, "handed to a member")
        require(meta_member_device(DEVICE_CPU, DEVICE_CPU, "M", "M", "C") == DEVICE_CPU, "equal words")
        require(meta_member_device(DEVICE_INSTALL, DEVICE_GPU, "M", "", "C") == DEVICE_GPU, "None keeps the member's")
        var got = _words(DEVICE_CPU, DEVICE_GPU, "")
        require(got == "ValueError: StackingClassifier was given device='cpu' but its estimator DecisionTreeClassifier "
            + "was constructed with device='gpu'; give both the same device, or leave one of them None",
            "own word: " + got)
    elif which == 1:
        require(meta_member_device(DEVICE_CPU, DEVICE_INSTALL, "M", "Outer", "C") == DEVICE_CPU, "handed on")
        require(meta_member_device(DEVICE_GPU, DEVICE_GPU, "M", "Outer", "C") == DEVICE_GPU, "equal words, handed")
    else:
        var got = _words(DEVICE_CPU, DEVICE_GPU, "VotingClassifier")
        require(got == "ValueError: VotingClassifier was given device='cpu', handed to StackingClassifier, but its "
            + "estimator DecisionTreeClassifier was constructed with device='gpu'; give both the same device, or "
            + "leave one of them None", "handed word: " + got)


def test_meta_device_rule() raises:
    check_cases("meta_device_rule", CASES, run_case)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
