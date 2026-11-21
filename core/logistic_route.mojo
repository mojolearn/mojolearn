# SPDX-License-Identifier: Apache-2.0
"""The device route of one LogisticRegression estimator, as native words.

The sibling of core/linear_route.mojo for the quasi-Newton (QN) entry: the
device words (`device=None | "gpu" | "cpu"`, their codes and refusals) are
that module's, used here unchanged; this module adds only what the QN family
resolves. A route holds the thin C entry of ONE compiled estimators image
(the device image of the selected tier, or the CPU identical image), and the
answers that image gave about itself when the route was resolved: its tier,
whether it has the row-argmax entry, whether a cross-validation fold resident
on the device may be read by it.

A route is resolved ONCE per estimator
(providers/python/logistic_route_boundary_door.mojo). Every later fit,
decision_function, predict, predict_proba, predict_log_proba and score calls
through `route.driver`; none of them tests the device, builds a mode word,
reads a table or asks the image for its tier again.

RUN of this family is the two functions below: typed requests in, the
iteration count or nothing out, no Python object in reach.

Resolution is a metadata operation with no independent numerical work to
split; it runs at an estimator's first operation, never per row.
"""
from core.linear_route import LINEAR_DEVICE_CPU, LINEAR_DEVICE_GPU
from core.logistic_native_api import QnNativeDriver, QnFitRequest, QnQueryRequest


@fieldwise_init
struct LogisticRoute(Copyable, Movable):
    """Resolved entry of one estimator. `mode` is the word the compiled
    image itself reports (1 identical / 0 fast); `vendor` is 0 for the CPU
    image, else the process's device vendor word (1 Metal, 2 CUDA, 3 HIP)."""
    var device: Int
    var mode: Int
    var vendor: Int
    # The image has the row-argmax entry (QN_NATIVE_MULTICLASS); without it
    # predict of more than two classes takes the scores and the host argmax.
    var multiclass: Bool
    # A cross-validation fold resident on the device is read on the device
    # (the fold entry of the identical device image). False on every CPU route.
    var fold: Bool
    var driver: QnNativeDriver


def _logistic_mode_word(mode: Int) -> String:
    return String("identical") if mode == 1 else (String("fast") if mode == 0 else String("unknown"))


def logistic_route_admit(device: Int, driver: QnNativeDriver, compiled_mode: Int, compiled_host: Bool,
    multiclass: Bool, expected_mode: Int, process_vendor: Int, cpu_table: Bool, explicit_host: Bool,
    fold_published: Bool) raises -> LogisticRoute:
    """Admit the image a route resolved to: it must be compiled for the tier
    the estimator selected and, where the route names the CPU table beside a
    device set (`cpu_table`) or belongs to an explicit Host* class
    (`explicit_host`), it must be the CPU image. A device route in a process
    with a device set must not be the CPU image. `device` is the resolved
    device (LINEAR_DEVICE_GPU or LINEAR_DEVICE_CPU, never the request None)."""
    if compiled_mode != expected_mode:
        var wanted = _logistic_mode_word(expected_mode)
        raise Error("RuntimeError: mojolearn: _mojolearn_estimators was compiled for "
            + _logistic_mode_word(compiled_mode) + ", but this call requested " + wanted
            + "; rebuild the " + wanted + " binding")
    if explicit_host:
        if not compiled_host or compiled_mode != 1:
            raise Error("RuntimeError: the explicit host model requires its CPU identical binding")
    elif cpu_table:
        if not compiled_host or compiled_mode != 1:
            raise Error("RuntimeError: device='cpu' requires the CPU identical binding")
    elif device == LINEAR_DEVICE_GPU and compiled_host:
        raise Error("RuntimeError: LogisticRegression native vendor differs from selected execution")
    var on_cpu = device == LINEAR_DEVICE_CPU
    return LogisticRoute(device, compiled_mode, 0 if on_cpu else process_vendor, multiclass,
        fold_published and not on_cpu and compiled_mode == 1, driver.copy())


def logistic_fit_run(route: LogisticRoute, request: QnFitRequest) raises -> Int:
    """RUN of fit: the solver of the route's image over the admitted words.
    Returns the iteration count; the coefficients, the two info words and the
    finite status are written through the request's addresses."""
    return route.driver.fit(request)


def logistic_query_run(route: LogisticRoute, operation: Int, request: QnQueryRequest) raises:
    """RUN of decision_function / predict / predict_proba / predict_log_proba:
    one entry of the route's image (QN_NATIVE_DECISION, _BINARY, _MULTICLASS
    or _PROBA) writing through the request's destination address."""
    route.driver.query(operation, request)
