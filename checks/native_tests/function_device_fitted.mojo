# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""A QUERY FUNCTION'S DEVICE (lane vdf-misc; core/function_device_fitted.mojo), in a process with no Python.

The device of a public function that queries a fitted model (the HDBSCAN query functions) against the model's own:

1. None follows the model, whatever the model says;
2. an explicit word equal to the model's explicit device is admitted;
3. an explicit word different from the model's explicit device is refused in `function_device_member`'s words;
4. "cpu" beside a GPU set with a model constructed with device=None is refused, naming the function and the class;
5. "gpu" with a model constructed with device=None (GPU installation), and "cpu" on a CPU-only installation
   (host_beside False) with a model constructed with device=None, are admitted.

FAIL-PROOF EDITS (each must make the named case fail): drop the final raise (case 4); return before
`function_device_member` for every explicit word (case 3).
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, require
from core.device_route import DEVICE_CPU, DEVICE_GPU, DEVICE_INSTALL
from core.function_device import FunctionDevice
from core.function_device_fitted import function_device_fitted

comptime CASES = 5
comptime NAME = "approximate_predict"
comptime CLS = "HDBSCAN"


def _words(device: FunctionDevice, member: Int) -> String:
    """The words `function_device_fitted` raised, or "" when it admitted."""
    try:
        function_device_fitted(device, member, NAME, CLS)
    except problem:
        return String(problem)
    return String("")


def run_case(which: Int) raises:
    if which == 0:
        for member in range(3):  # small-loop(fixture: the three device codes)
            var code = DEVICE_INSTALL if member == 0 else (DEVICE_GPU if member == 1 else DEVICE_CPU)
            require(_words(FunctionDevice(DEVICE_INSTALL, False), code) == "", "None follows the model")
    elif which == 1:
        require(_words(FunctionDevice(DEVICE_CPU, True), DEVICE_CPU) == "", "cpu with a cpu model")
        require(_words(FunctionDevice(DEVICE_GPU, False), DEVICE_GPU) == "", "gpu with a gpu model")
    elif which == 2:
        var got = _words(FunctionDevice(DEVICE_CPU, True), DEVICE_GPU)
        require(got == "ValueError: approximate_predict was given device='cpu' but its estimator HDBSCAN was "
            + "constructed with device='gpu'; give both the same device, or leave one of them None",
            "explicit conflict: " + got)
    elif which == 3:
        var got = _words(FunctionDevice(DEVICE_CPU, True), DEVICE_INSTALL)
        require(got == "ValueError: approximate_predict was given device='cpu' but its HDBSCAN was constructed "
            + "with device=None and runs on this installation's GPU; a fitted model is queried where it was "
            + "fitted: construct it with device='cpu', or leave device=None", "cpu beside a None model: " + got)
    else:
        require(_words(FunctionDevice(DEVICE_GPU, False), DEVICE_INSTALL) == "", "gpu with a None model")
        require(_words(FunctionDevice(DEVICE_CPU, False), DEVICE_INSTALL) == "", "cpu-only with a None model")


def test_function_device_fitted() raises:
    check_cases("function_device_fitted", CASES, run_case)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
