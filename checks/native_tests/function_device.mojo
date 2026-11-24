# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""A PUBLIC FUNCTION'S DEVICE (lane vd-functions; core/function_device.mojo), in a process with no Python.

The route selection of `device=` on a public function, and the words of each refusal:

1. None (DEVICE_INSTALL) keeps today's route on every installation and for every function: `host_beside` False,
   whatever the implementations;
2. "gpu" on a GPU installation for a function with a GPU implementation: today's (device) route;
3. "gpu" for a function with no GPU implementation is refused in words naming the function, on both installations
   (the CPU route never runs in its place);
4. "gpu" on a CPU-only installation is refused in the estimators' words (`device_gpu_absent`);
5. "cpu" on a CPU-only installation: today's route (its one set is the host one);
6. "cpu" on a GPU installation for a function with both implementations: the CPU images beside the GPU set;
7. "cpu" on a GPU installation for a host-only function: today's route (it already runs there);
8. "cpu" for a function with no CPU implementation on a GPU installation: the estimators' `device_cpu_absent` words;
9. a model-selection tool's device handed to a member that names none; the member's own kept when the tool says
   None; equal words admitted; different explicit words refused naming the tool and the member's class.

FAIL-PROOF EDITS (each must make the named case fail): return `FunctionDevice(DEVICE_CPU, True)` for
`cpu_only` (case 5); drop the `has_gpu` refusal (case 3); return `True` for `host_beside` of a host-only function
(case 7); drop the final raise of `function_device_member` (case 9).
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, require
from core.device_route import DEVICE_CPU, DEVICE_GPU, DEVICE_INSTALL
from core.function_device import FunctionDevice, function_device_decide, function_device_member

comptime CASES = 9
comptime NAME = "example_score"


def _words(request: Int, cpu_only: Bool, has_gpu: Bool, has_cpu: Bool) -> String:
    """The words `function_device_decide` raised, or "" when it admitted."""
    try:
        _ = function_device_decide(request, cpu_only, has_gpu, has_cpu, NAME)
    except problem:
        return String(problem)
    return String("")


def _member_words(request: Int, member: Int) -> String:
    try:
        _ = function_device_member(request, member, "cross_val_score", "Ridge")
    except problem:
        return String(problem)
    return String("")


def run_case(which: Int) raises:
    if which == 0:
        for cpu_only in range(2):  # small-loop(fixture)
            for has_gpu in range(2):  # small-loop(fixture)
                var d = function_device_decide(DEVICE_INSTALL, cpu_only == 1, has_gpu == 1, True, NAME)
                require(d.request == DEVICE_INSTALL and not d.host_beside, "None keeps today's route")
    elif which == 1:
        var d = function_device_decide(DEVICE_GPU, False, True, True, NAME)
        require(d.request == DEVICE_GPU and not d.host_beside, "gpu on a GPU installation runs today's route")
    elif which == 2:
        var want = String("ValueError: device='gpu' is not available for example_score: it has no GPU "
            + "implementation. Use device='cpu', or leave device=None")
        require(_words(DEVICE_GPU, False, False, True) == want, "no GPU implementation, GPU installation: " +
            _words(DEVICE_GPU, False, False, True))
        require(_words(DEVICE_GPU, True, False, True) == want, "no GPU implementation, CPU-only installation")
    elif which == 3:
        var got = _words(DEVICE_GPU, True, True, True)
        require(got.startswith("ValueError: device='gpu' needs a GPU build of mojolearn"),
            "gpu on a CPU-only installation: " + got)
    elif which == 4:
        var d = function_device_decide(DEVICE_CPU, True, True, True, NAME)
        require(d.request == DEVICE_CPU and not d.host_beside, "cpu on a CPU-only installation runs today's route")
    elif which == 5:
        var d = function_device_decide(DEVICE_CPU, False, True, True, NAME)
        require(d.request == DEVICE_CPU and d.host_beside, "cpu beside a GPU set runs the CPU images")
    elif which == 6:
        var d = function_device_decide(DEVICE_CPU, False, False, True, NAME)
        require(d.request == DEVICE_CPU and not d.host_beside, "a host-only function runs today's route")
    elif which == 7:
        var got = _words(DEVICE_CPU, False, True, False)
        require(got == "ValueError: device='cpu' is not available for example_score: it has no CPU implementation. "
            + "Leave device=None to run on this installation's device", "no CPU implementation: " + got)
    else:
        require(function_device_member(DEVICE_CPU, DEVICE_INSTALL, "t", "M") == DEVICE_CPU, "tool hands its device")
        require(function_device_member(DEVICE_INSTALL, DEVICE_GPU, "t", "M") == DEVICE_GPU, "member keeps its own")
        require(function_device_member(DEVICE_INSTALL, DEVICE_INSTALL, "t", "M") == DEVICE_INSTALL, "both None")
        require(function_device_member(DEVICE_GPU, DEVICE_GPU, "t", "M") == DEVICE_GPU, "equal words admitted")
        var got = _member_words(DEVICE_CPU, DEVICE_GPU)
        require(got == "ValueError: cross_val_score was given device='cpu' but its estimator Ridge was constructed "
            + "with device='gpu'; give both the same device, or leave one of them None", "conflict: " + got)


def test_function_device() raises:
    check_cases("function_device", CASES, run_case)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
