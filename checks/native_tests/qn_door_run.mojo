# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""Native cases for QNRegressor's Python-free RUN (glm/qn_door_run.mojo):
the typed checks with their public wording, the change tolerance and the
publication of a fitted [coef | intercept] vector. Analytic answers, Mojo
buffers only; independent fixtures run through the shared host worker pool.
RUN calls through the route's resolved entry and reads no table: a request
that carries an unresolved entry is refused by that entry, in its words. The
device rule of the route (core/device_route.mojo): None is the
installation's device, an explicit GPU without a GPU set and a CPU route in
fast mode are refused, the CPU table is named only beside a device set, and
the resolved image must report the planned vendor and the selected tier.
A fit on a loaded estimators image is exercised by the installed-wheel QN
cases (python/mojolearn/tests/test_family_device_boundary.py)."""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, require
from core.logistic_native_api import QnFitRequest, QnQueryRequest
from core.entry_table import entry_ref_unresolved
from core.device_route import (
    DEVICE_GPU, DEVICE_CPU, DEVICE_INSTALL, DeviceRoute, device_mode_code, device_mode_word, device_vendor_word, device_route_plan, device_route_admit,
)
from glm.qn_door_run import (
    QnDoorPredictRequest, qn_door_predict_run,
    qn_check_lengths, qn_check_features, qn_check_coefficients, qn_change_tolerance, qn_door_fit_publish,
)

comptime QN_DOOR_FIXTURES = 22


def refusal(index: Int) raises -> String:
    """The text the checked call raises, or "" when it passes."""
    try:
        if index == 0:
            qn_check_lengths(5, 5)
        elif index == 1:
            qn_check_lengths(5, 4)
        elif index == 2:
            qn_check_features(3, 3)
        elif index == 3:
            qn_check_features(3, 2)
        elif index == 4:
            qn_check_coefficients(True, 4, 3, True)
        elif index == 5:
            qn_check_coefficients(True, 3, 3, True)
        elif index == 6:
            qn_check_coefficients(False, 3, 3, False)
    except error:
        return String(error)
    return String("")


def check_fixture(index: Int) raises:
    var text = refusal(index)
    if index == 0 or index == 2 or index == 4:
        require(text == "", "a valid QN input was refused: " + text)
    elif index == 1:
        require(text == "mojolearn QNRegressor X and y lengths differ", "length refusal changed")
    elif index == 3:
        require(text == "mojolearn QNRegressor feature count differs from fit", "width refusal changed")
    else:
        require(text == "TypeError: QN fitted coefficients require contiguous float32 storage matching the fitted width",
                "coefficient refusal changed")


def publish_fixture(intercept: Bool) raises:
    var w: List[Float32] = [1.5, -2.25, 0.125, 7.0]
    var info: List[Float32] = [0.75, 3.0]
    var coef = List[Float32](length=3, fill=0.0)
    var iterations = List[Int64](length=1, fill=0)
    var request = QnFitRequest(0, 0, Int(w.unsafe_ptr()), Int(info.unsafe_ptr()), 0, 0, 2, 3, 1, 10, 5, 4,
        intercept, False, 0, 0.0, 0.0, 1e-4, 1e-6, 0.0)
    var result = qn_door_fit_publish(request, 17, Int(coef.unsafe_ptr()), Int(iterations.unsafe_ptr()))
    require(coef[0] == 1.5 and coef[1] == -2.25 and coef[2] == 0.125, "coefficients not copied")
    require(iterations[0] == 17, "iteration count not written")
    require(result.intercept == (7.0 if intercept else 0.0), "intercept changed")
    require(result.objective == 0.75 and result.retcode == 3, "solver words changed")
    _ = w^
    _ = info^
    _ = coef^
    _ = iterations^


def plan_refusal(device: Int, process_vendor: Int, mode: Int) -> String:
    try:
        _ = device_route_plan(device, process_vendor, mode)
    except error:
        return String(error)
    return String("")


def admit_refusal(device: Int, process_vendor: Int, image_mode: Int, image_vendor: Int, mode: Int) raises -> String:
    var plan = device_route_plan(device, process_vendor, mode)
    try:
        device_route_admit("QN", plan, image_mode, image_vendor, mode)
    except error:
        return String(error)
    return String("")


def route_fixture(index: Int) raises:
    """Units 12 to 21: the device rule and the route words."""
    if index == 12:
        # None is the installation's device: the GPU set where there is one.
        for vendor in range(1, 4):  # driver-loop: over the three GPU vendor words
            var plan = device_route_plan(DEVICE_INSTALL, vendor, 1)
            require(plan.device == DEVICE_GPU and not plan.cpu_table and plan.expected_vendor == vendor,
                    "None did not resolve to the GPU set")
    elif index == 13:
        # None in a CPU-only process: the process's own (CPU) table, in either tier.
        var plan = device_route_plan(DEVICE_INSTALL, 0, 1)
        require(plan.device == DEVICE_CPU and not plan.cpu_table and plan.expected_vendor == 0,
                "None did not resolve to the CPU in a CPU-only process")
        var explicit = device_route_plan(DEVICE_CPU, 0, 0)
        require(explicit.device == DEVICE_CPU and not explicit.cpu_table, "cpu in a CPU-only process changed")
    elif index == 14:
        # cpu beside a device set: the CPU table, vendor word 0.
        for vendor in range(1, 4):  # driver-loop: over the three GPU vendor words
            var plan = device_route_plan(DEVICE_CPU, vendor, 1)
            require(plan.device == DEVICE_CPU and plan.cpu_table and plan.expected_vendor == 0,
                    "cpu beside a device set did not name the CPU table")
    elif index == 15:
        require(plan_refusal(DEVICE_CPU, 2, 0)
                == "ValueError: device='cpu' runs in identical mode only; numeric_mode='fast' is for Apple GPUs",
                "cpu with fast was not refused in words")
    elif index == 16:
        var text = plan_refusal(DEVICE_GPU, 0, 1)
        require(text.startswith("ValueError: device='gpu' needs a GPU build of mojolearn") and "device='cpu'" in text,
                "an explicit gpu without a GPU set was not refused in words: " + text)
    elif index == 17:
        require(admit_refusal(DEVICE_INSTALL, 2, 1, 2, 1) == "", "the selected device image was refused")
        require(admit_refusal(DEVICE_CPU, 2, 1, 0, 1) == "", "the CPU image was refused for device='cpu'")
        require(admit_refusal(DEVICE_INSTALL, 0, 0, 0, 0) == "", "the CPU-only process's image was refused")
    elif index == 18:
        require(admit_refusal(DEVICE_INSTALL, 2, 1, 3, 1) == "RuntimeError: QN native vendor differs from selected execution",
                "a foreign device image was admitted")
        require(admit_refusal(DEVICE_CPU, 2, 1, 2, 1) == "RuntimeError: device='cpu' requires the CPU identical binding",
                "a device image was admitted for device='cpu'")
    elif index == 19:
        require(admit_refusal(DEVICE_INSTALL, 2, 0, 2, 1) == "RuntimeError: QN native mode differs from selected execution",
                "an image of the other tier was admitted")
    elif index == 20:
        require(device_mode_code("identical") == 1 and device_mode_code("fast") == 0 and device_mode_code("Identical") == -1,
                "mode codes changed")
        require(device_mode_word(1) == "identical" and device_mode_word(0) == "fast", "mode words changed")
        require(device_vendor_word(0) == "cpu" and device_vendor_word(1) == "metal" and device_vendor_word(2) == "cuda"
                and device_vendor_word(3) == "hip", "vendor words changed")
        var route = DeviceRoute(DEVICE_CPU)
        require(route.device == DEVICE_CPU and route.mode == -1 and route.vendor == -1 and not route.has_second
                and route.entry.word == 0 and route.second.word == 0, "a new route is not unresolved")
    else:
        # RUN calls through the request's entry: an unresolved one refuses
        # in its own words and no table is read.
        var route = DeviceRoute(DEVICE_GPU)
        var request = QnQueryRequest(0, 0, 0, 0, 0, 0, 1, False, False)
        var text = String("")
        try:
            qn_door_predict_run(QnDoorPredictRequest(route.entry, request^))
        except error:
            text = String(error)
        require("not resolved for this owner" in text, "RUN did not call through the request's entry: " + text)


def fixture(index: Int) raises:
    if index >= 12:
        route_fixture(index)
    elif index < 7:
        check_fixture(index)
    elif index == 7:
        require(qn_change_tolerance(None, 0.5) == 0.005, "default change tolerance changed")
    elif index == 8:
        require(qn_change_tolerance(Optional[Float64](0.25), 0.5) == 0.25, "delta not used")
    elif index == 9:
        require(qn_change_tolerance(Optional[Float64](0.0), 0.5) == 0.0, "a zero delta must be kept")
    elif index == 10:
        publish_fixture(True)
    else:
        publish_fixture(False)


def test_fixtures() raises:
    check_cases("fixture", QN_DOOR_FIXTURES, fixture)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
