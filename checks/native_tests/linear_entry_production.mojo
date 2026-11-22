# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""NET-P01: production handle -> resolved device route -> actual host prediction.

The route (core/linear_route.mojo) is what an estimator resolves once for its
`device`: it is refused before its table is published, refused for a vendor
the compiled entry does not report, reports the CPU words of the host
entries, and an unresolved route's entry refuses instead of calling anything.

Run in a fresh standalone process: the initial missing-table assertion and
publication are ordered because they share one production table. After that,
independent lifetime graphs (one per worker) use the full native host worker pool.
No Python module, synthetic dispatcher or production call counter participates.
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.linear_boundary_types import LINEAR_PREDICT_ABI
from core.linear_entry_table import LinearEntryKey, LINEAR_TABLE_ESTIMATORS, linear_table_publish
from core.linear_fit_types import LINEAR_FIT_ABI, LINEAR_FIT_OPERATION_COUNT
from core.linear_model_state import LinearModelState
from core.linear_registry import LINEAR_STATES, linear_lease_predict
from core.device_route import (
    DeviceRoute, DEVICE_CPU, DEVICE_GPU, DEVICE_INSTALL, device_code, device_installed,
)
from core.linear_route import linear_route_resolve
from core.native_descriptor_call import native_descriptor_call
from glm.host.linear_fit_native import LINEAR_FIT_HOST_API, linear_fit_host_v1
from glm.host.linear_predict_native import linear_predict_host_v1

comptime LINEAR_ENTRY_PRODUCTION_CASES = 2


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def model_state(address: Int) -> LinearModelState:
    var state = LinearModelState()
    state.coef = address
    state.coef_count = 3
    state.features = 3
    state.intercept = 0.25
    state.has_coef = True
    state.fitted = True
    return state^


def device_rules() raises:
    # `device=None` is the installation's device; an explicit gpu without a
    # GPU set is refused in words that name the CPU; cpu always runs.
    require(device_code("gpu") == DEVICE_GPU and device_code("cpu") == DEVICE_CPU
        and device_code("GPU") < 0 and device_code("") < 0, "device words are not exact")
    require(device_installed(DEVICE_INSTALL, 0) == DEVICE_CPU,
        "the default device of a CPU-only process is not the CPU")
    for vendor in range(1, 4):  # the three GPU vendor words
        require(device_installed(DEVICE_INSTALL, vendor) == DEVICE_GPU,
            "the default device of a GPU process is not the GPU")
        require(device_installed(DEVICE_GPU, vendor) == DEVICE_GPU
            and device_installed(DEVICE_CPU, vendor) == DEVICE_CPU,
            "an explicit device did not keep its word on a GPU process")
    require(device_installed(DEVICE_CPU, 0) == DEVICE_CPU, "cpu did not run on a CPU-only process")
    var refused = False
    try:
        _ = device_installed(DEVICE_GPU, 0)
    except problem:
        require(String(problem).find("device='cpu'") >= 0, "the gpu refusal does not name device='cpu'")
        refused = True
    require(refused, "an explicit gpu ran on a CPU-only process")


def route_metrics_default() raises:
    # Only a CPU route resolved beside a GPU set is given a metrics entry
    # (providers/python/metrics_route_boundary_door.mojo). Every other route
    # scores on the process's table: it claims none, and its metrics ref
    # refuses a call instead of calling anything.
    for device in range(DEVICE_GPU, DEVICE_INSTALL + 1):  # the three device words
        var route = DeviceRoute(device)
        require(not route.has_fifth, "a new route claims a metrics entry")
        var refused = False
        try:
            _ = native_descriptor_call(route.fifth.call, 0, List[Int](), List[Int](), List[Float64]())
        except problem:
            require(String(problem).find("not resolved") >= 0, "wrong unresolved metrics entry refusal")
            refused = True
        require(refused, "an unresolved metrics entry answered")


def missing_before_publication() raises:
    var coefficients: List[Float32] = [2, -1, 3]
    var source: List[Float32] = [1, 2, 1]
    var destination = List[Float32](length=1, fill=-37)
    var handle = LINEAR_STATES.insert(model_state(Int(coefficients.unsafe_ptr())))
    try:
        var lease = LINEAR_STATES.acquire(handle)
        var refused = False
        try:
            var early = linear_route_resolve(DEVICE_CPU, LinearEntryKey(GLOBAL_NUMERIC_MODE, 0),
                0, GLOBAL_NUMERIC_MODE, False)
            _ = linear_lease_predict(early.second.call, lease,
                Int(source.unsafe_ptr()), Int(destination.unsafe_ptr()), 1, 3, String("Ridge"))
        except problem:
            require(String(problem).find("not published") >= 0, "wrong missing production table refusal")
            refused = True
        require(refused and destination[0] == Float32(-37), "missing production table wrote a prediction")
        # A route that was never resolved holds a refusing entry, not a null
        # pointer: a call through it raises and writes nothing.
        var unresolved = DeviceRoute(DEVICE_GPU)
        refused = False
        try:
            _ = linear_lease_predict(unresolved.second.call, lease,
                Int(source.unsafe_ptr()), Int(destination.unsafe_ptr()), 1, 3, String("Ridge"))
        except problem:
            require(String(problem).find("not resolved") >= 0, "wrong unresolved route refusal")
            refused = True
        require(refused and destination[0] == Float32(-37), "unresolved route wrote a prediction")
        _ = lease
    finally:
        LINEAR_STATES.release(handle)
        _ = coefficients
        _ = source
        _ = destination


def fixture() raises:
    # These explicit native owners remain alive through both snapshots and
    # release. LinearModelState itself owns descriptors, not their buffers.
    var coefficients: List[Float32] = [2, -1, 3]
    var replacement: List[Float32] = [4, 5, 6]
    var source: List[Float32] = [1, 2, 1, 2, 0, -1]
    var narrow = List[Float32](length=2, fill=-37)
    var destination = Int(narrow.unsafe_ptr())
    # Each fixture resolves its own route, as each estimator does: concurrent
    # resolutions of one published table give the same entries.
    var route = linear_route_resolve(DEVICE_CPU, LinearEntryKey(GLOBAL_NUMERIC_MODE, 0),
        0, GLOBAL_NUMERIC_MODE, False)
    require(route.device == DEVICE_CPU and route.vendor == 0 and route.mode == GLOBAL_NUMERIC_MODE,
        "CPU route does not report the host entries' vendor and tier")
    var foreign = False
    try:
        _ = linear_route_resolve(DEVICE_GPU, LinearEntryKey(GLOBAL_NUMERIC_MODE, 0),
            2, GLOBAL_NUMERIC_MODE, False)
    except problem:
        require(String(problem).find("vendor differs") >= 0, "wrong foreign vendor refusal")
        foreign = True
    require(foreign, "a route resolved for a vendor its compiled entry does not report")
    var entry = route.second.call
    var handle = LINEAR_STATES.insert(model_state(Int(coefficients.unsafe_ptr())))
    var released = False
    try:
        var original = LINEAR_STATES.acquire(handle)
        LINEAR_STATES.replace(handle, model_state(Int(replacement.unsafe_ptr())))
        var current = LINEAR_STATES.acquire(handle)
        _ = linear_lease_predict(entry, current, Int(source.unsafe_ptr()), destination, 2, 3, String("Ridge"))
        require(narrow[0] == Float32(20.25) and narrow[1] == Float32(2.25),
            "production handle/table prediction did not use replacement coefficients")
        LINEAR_STATES.release(handle)
        released = True
        # Releasing/refitting the registry entry must not change an acquired
        # old snapshot; the original buffer is still explicitly pinned here.
        _ = linear_lease_predict(entry, original, Int(source.unsafe_ptr()), destination, 2, 3, String("Ridge"))
        require(narrow[0] == Float32(3.25) and narrow[1] == Float32(1.25),
            "production lease lost its original prediction after refit/release")
        var refused = False
        try:
            _ = LINEAR_STATES.acquire(handle)
        except:
            refused = True
        require(refused, "released production linear handle remained usable")
        _ = original
        _ = current
    finally:
        if not released:
            LINEAR_STATES.release(handle)
        _ = coefficients
        _ = replacement
        _ = source
        _ = narrow


def _fixture_case(unit: Int) raises:
    # Each unit runs its own lifetime graph (unit is the case index only).
    _ = unit
    fixture()


def test_rules_then_published_fixtures() raises:
    """One test, in this order: the preconditions and the publication come first (process-global), then the case table."""
    device_rules()
    route_metrics_default()
    missing_before_publication()
    # Same native publication, schema, endpoint order and publisher identity
    # as _mojolearn_estimators_host initialization, with no Python import.
    var api = LINEAR_FIT_HOST_API.get_or_create_ptr()
    linear_table_publish[LINEAR_TABLE_ESTIMATORS](LinearEntryKey(GLOBAL_NUMERIC_MODE, 0),
        Int(api[].entries.unsafe_ptr()), [String(LINEAR_FIT_ABI), String(LINEAR_PREDICT_ABI)],
        [LINEAR_FIT_OPERATION_COUNT, 1], [linear_fit_host_v1, linear_predict_host_v1])
    check_cases("fixture", LINEAR_ENTRY_PRODUCTION_CASES, _fixture_case)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
