# SPDX-License-Identifier: Apache-2.0
"""NET-P01: production handle -> published table -> actual host prediction.

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


def missing_before_publication() raises:
    var coefficients: List[Float32] = [2, -1, 3]
    var source: List[Float32] = [1, 2, 1]
    var destination = List[Float32](length=1, fill=-37)
    var handle = LINEAR_STATES.insert(model_state(Int(coefficients.unsafe_ptr())))
    try:
        var lease = LINEAR_STATES.acquire(handle)
        var refused = False
        try:
            linear_lease_predict(LinearEntryKey(GLOBAL_NUMERIC_MODE, 0), lease,
                Int(source.unsafe_ptr()), Int(destination.unsafe_ptr()), 1, 3, False, String("Ridge"))
        except problem:
            require(String(problem).find("not published") >= 0, "wrong missing production table refusal")
            refused = True
        require(refused and destination[0] == Float32(-37), "missing production table wrote a prediction")
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
    var key = LinearEntryKey(GLOBAL_NUMERIC_MODE, 0)
    var handle = LINEAR_STATES.insert(model_state(Int(coefficients.unsafe_ptr())))
    var released = False
    try:
        var original = LINEAR_STATES.acquire(handle)
        LINEAR_STATES.replace(handle, model_state(Int(replacement.unsafe_ptr())))
        var current = LINEAR_STATES.acquire(handle)
        linear_lease_predict(key, current, Int(source.unsafe_ptr()), destination, 2, 3, wide, String("Ridge"))
        require((wider[0] == 20.25 and wider[1] == 2.25) if wide else
            (narrow[0] == Float32(20.25) and narrow[1] == Float32(2.25)),
            "production handle/table prediction did not use replacement coefficients")
        LINEAR_STATES.release(handle)
        released = True
        # Releasing/refitting the registry entry must not change an acquired
        # old snapshot; the original buffer is still explicitly pinned here.
        linear_lease_predict(key, original, Int(source.unsafe_ptr()), destination, 2, 3, wide, String("Ridge"))
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
