# SPDX-License-Identifier: Apache-2.0
"""NET-P01: production handle -> resolved device route -> actual host prediction.

The route (core/linear_route.mojo) is what an estimator resolves once for its
`device`: it is refused before its table is published, refused for a vendor
the compiled entry does not report, reports the CPU words of the host
entries, and an unresolved route's entry refuses instead of calling anything.

Run in a fresh standalone process: the initial missing-table assertion and
publication are ordered because they share one production table. After that,
independent narrow/wide lifetime graphs use the full native host worker pool.
No Python module, synthetic dispatcher or production call counter participates.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.host_parallel import host_parallelize
from core.linear_boundary_types import LINEAR_PREDICT_ABI
from core.linear_entry_table import LinearEntryKey, LINEAR_TABLE_ESTIMATORS, linear_table_publish
from core.linear_fit_types import LINEAR_FIT_ABI, LINEAR_FIT_OPERATION_COUNT
from core.linear_model_state import LinearModelState
from core.linear_registry import LINEAR_STATES, linear_lease_predict
from core.linear_route import (
    LinearRoute, LINEAR_DEVICE_CPU, LINEAR_DEVICE_GPU, LINEAR_DEVICE_INSTALL,
    linear_device_code, linear_device_installed, linear_route_resolve,
)
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
    require(linear_device_code("gpu") == LINEAR_DEVICE_GPU and linear_device_code("cpu") == LINEAR_DEVICE_CPU
        and linear_device_code("GPU") < 0 and linear_device_code("") < 0, "device words are not exact")
    require(linear_device_installed(LINEAR_DEVICE_INSTALL, 0) == LINEAR_DEVICE_CPU,
        "the default device of a CPU-only process is not the CPU")
    for vendor in range(1, 4):  # the three GPU vendor words
        require(linear_device_installed(LINEAR_DEVICE_INSTALL, vendor) == LINEAR_DEVICE_GPU,
            "the default device of a GPU process is not the GPU")
        require(linear_device_installed(LINEAR_DEVICE_GPU, vendor) == LINEAR_DEVICE_GPU
            and linear_device_installed(LINEAR_DEVICE_CPU, vendor) == LINEAR_DEVICE_CPU,
            "an explicit device did not keep its word on a GPU process")
    require(linear_device_installed(LINEAR_DEVICE_CPU, 0) == LINEAR_DEVICE_CPU, "cpu did not run on a CPU-only process")
    var refused = False
    try:
        _ = linear_device_installed(LINEAR_DEVICE_GPU, 0)
    except problem:
        require(String(problem).find("device='cpu'") >= 0, "the gpu refusal does not name device='cpu'")
        refused = True
    require(refused, "an explicit gpu ran on a CPU-only process")


def missing_before_publication() raises:
    var coefficients: List[Float32] = [2, -1, 3]
    var source: List[Float32] = [1, 2, 1]
    var destination = List[Float32](length=1, fill=-37)
    var handle = LINEAR_STATES.insert(model_state(Int(coefficients.unsafe_ptr())))
    try:
        var lease = LINEAR_STATES.acquire(handle)
        var refused = False
        try:
            var early = linear_route_resolve(LINEAR_DEVICE_CPU, LinearEntryKey(GLOBAL_NUMERIC_MODE, 0),
                0, GLOBAL_NUMERIC_MODE, False)
            _ = linear_lease_predict(early.predict.call, lease,
                Int(source.unsafe_ptr()), Int(destination.unsafe_ptr()), 1, 3, False, String("Ridge"))
        except problem:
            require(String(problem).find("not published") >= 0, "wrong missing production table refusal")
            refused = True
        require(refused and destination[0] == Float32(-37), "missing production table wrote a prediction")
        # A route that was never resolved holds a refusing entry, not a null
        # pointer: a call through it raises and writes nothing.
        var unresolved = LinearRoute(LINEAR_DEVICE_GPU)
        refused = False
        try:
            _ = linear_lease_predict(unresolved.predict.call, lease,
                Int(source.unsafe_ptr()), Int(destination.unsafe_ptr()), 1, 3, False, String("Ridge"))
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


def fixture(wide: Bool) raises:
    # These explicit native owners remain alive through both snapshots and
    # release. LinearModelState itself owns descriptors, not their buffers.
    var coefficients: List[Float32] = [2, -1, 3]
    var replacement: List[Float32] = [4, 5, 6]
    var source: List[Float32] = [1, 2, 1, 2, 0, -1]
    var narrow = List[Float32](length=2, fill=-37)
    var wider = List[Float64](length=2, fill=-37)
    var destination = Int(wider.unsafe_ptr()) if wide else Int(narrow.unsafe_ptr())
    # Each fixture resolves its own route, as each estimator does: concurrent
    # resolutions of one published table give the same entries.
    var route = linear_route_resolve(LINEAR_DEVICE_CPU, LinearEntryKey(GLOBAL_NUMERIC_MODE, 0),
        0, GLOBAL_NUMERIC_MODE, False)
    require(route.device == LINEAR_DEVICE_CPU and route.vendor == 0 and route.mode == GLOBAL_NUMERIC_MODE,
        "CPU route does not report the host entries' vendor and tier")
    var foreign = False
    try:
        _ = linear_route_resolve(LINEAR_DEVICE_GPU, LinearEntryKey(GLOBAL_NUMERIC_MODE, 0),
            2, GLOBAL_NUMERIC_MODE, False)
    except problem:
        require(String(problem).find("vendor differs") >= 0, "wrong foreign vendor refusal")
        foreign = True
    require(foreign, "a route resolved for a vendor its compiled entry does not report")
    var entry = route.predict.call
    var handle = LINEAR_STATES.insert(model_state(Int(coefficients.unsafe_ptr())))
    var released = False
    try:
        var original = LINEAR_STATES.acquire(handle)
        LINEAR_STATES.replace(handle, model_state(Int(replacement.unsafe_ptr())))
        var current = LINEAR_STATES.acquire(handle)
        _ = linear_lease_predict(entry, current, Int(source.unsafe_ptr()), destination, 2, 3, wide, String("Ridge"))
        require((wider[0] == 20.25 and wider[1] == 2.25) if wide else
            (narrow[0] == Float32(20.25) and narrow[1] == Float32(2.25)),
            "production handle/table prediction did not use replacement coefficients")
        LINEAR_STATES.release(handle)
        released = True
        # Releasing/refitting the registry entry must not change an acquired
        # old snapshot; the original buffer is still explicitly pinned here.
        _ = linear_lease_predict(entry, original, Int(source.unsafe_ptr()), destination, 2, 3, wide, String("Ridge"))
        require((wider[0] == 3.25 and wider[1] == 1.25) if wide else
            (narrow[0] == Float32(3.25) and narrow[1] == Float32(1.25)),
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
        _ = wider


def run_checks() raises:
    device_rules()
    missing_before_publication()
    # Same native publication, schema, endpoint order and publisher identity
    # as _mojolearn_estimators_host initialization, with no Python import.
    var api = LINEAR_FIT_HOST_API.get_or_create_ptr()
    linear_table_publish[LINEAR_TABLE_ESTIMATORS](LinearEntryKey(GLOBAL_NUMERIC_MODE, 0),
        Int(api[].entries.unsafe_ptr()), [String(LINEAR_FIT_ABI), String(LINEAR_PREDICT_ABI)],
        [LINEAR_FIT_OPERATION_COUNT, 1], [linear_fit_host_v1, linear_predict_host_v1])
    var failures = List[Int](length=LINEAR_ENTRY_PRODUCTION_CASES, fill=0)
    var flags = failures.unsafe_ptr()
    def task(unit: Int) {imm flags}:
        try:
            fixture(unit == 1)
        except problem:
            flags[unit] = 1
            print("FAIL production linear composition", unit, String(problem))
    host_parallelize(task, LINEAR_ENTRY_PRODUCTION_CASES)
    var failed = 0
    for failure in failures:  # collect independent fixture status words
        failed += failure
    _ = failures
    require(failed == 0, "production linear handle/table/predict composition failed")
    print("PASS production linear handle/table/predict", LINEAR_ENTRY_PRODUCTION_CASES)
