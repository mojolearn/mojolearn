# SPDX-License-Identifier: Apache-2.0
"""Native cases for QNRegressor's Python-free RUN (glm/qn_door_run.mojo):
the typed checks with their public wording, the change tolerance and the
publication of a fitted [coef | intercept] vector. Analytic answers, Mojo
buffers only; independent fixtures run through the shared host worker pool.
The table-calling RUN entries (qn_door_fit_run, qn_door_predict_run) need a
loaded estimators image and are exercised by the installed-wheel QN cases."""
from core.host_parallel import host_parallelize
from core.logistic_native_api import QnFitRequest
from glm.qn_door_run import (
    qn_check_lengths, qn_check_features, qn_check_coefficients, qn_change_tolerance, qn_door_fit_publish,
)

comptime QN_DOOR_FIXTURES = 12


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


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


def fixture(index: Int) raises:
    if index < 7:
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


def run_checks() raises:
    var failures = List[Int](length=QN_DOOR_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(index: Int) {imm fp}:
        try:
            fixture(index)
        except error:
            fp.unsafe_store(index, 1)
            print("FAIL qn_door_run fixture", index, String(error))
    host_parallelize(task, QN_DOOR_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    if failed != 0:
        raise Error("qn_door_run: " + String(failed) + " fixture(s) failed")
    print("qn_door_run:", QN_DOOR_FIXTURES, "fixtures passed")
