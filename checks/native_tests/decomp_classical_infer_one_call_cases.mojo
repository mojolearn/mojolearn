# SPDX-License-Identifier: Apache-2.0
"""PCA transform writes every output word (ported from
python/mojolearn/tests/test_classical_infer_one_call.py
`test_pca_transform_output_is_fully_written`).

The public transform hands the provider an uninitialised output. Here the
same fitted state is queried twice into two outputs pre-filled with
different words (a quiet NaN and a finite sentinel): every word must be
finite and the two outputs must carry the same bits, so no word of either
can be the pre-fill. The shape is the original 257 x 5 fit with four
components through the covariance route (`svd_solver='covariance_eigh'`).
The data are a deterministic Mojo-generated matrix: the property does not
depend on NumPy's stream. Guard words around each output stay untouched.
"""
from std.math import isfinite, sin
from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.pca_native_api import PCANativeDriver, PCAFoldStore
from checks.native_tests.pca_native_cases import PCATestState, PCA_TEST_GUARD, require, guards, unchanged

comptime INFER_ONE_CALL_FIXTURES = 1
comptime INFER_ROWS = 257
comptime INFER_FEATURES = 5
comptime INFER_COMPONENTS = 4


def infer_data() -> List[Float32]:
    var count = INFER_ROWS * INFER_FEATURES
    var result = List[Float32](unsafe_uninit_length=count)
    var rp = result.unsafe_ptr()
    def cell(index: Int) {imm rp}:
        var row = index // INFER_FEATURES
        var column = index % INFER_FEATURES
        var value = sin(0.37 * Float64(row) * Float64(column + 1) + Float64(column)) * Float64(column + 1)
        rp.unsafe_store(index, Float32(value + 0.01 * Float64(row)))
    host_parallelize(cell, count)
    return result^


def prefilled(count: Int, word: Float32) -> List[Float32]:
    var result = List[Float32](length=count + 2, fill=word)
    result[0] = PCA_TEST_GUARD
    result[count + 1] = PCA_TEST_GUARD
    return result^


def fully_written(first: List[Float32], second: List[Float32], count: Int) raises:
    var failures = List[Int](length=count, fill=0)
    var fp = failures.unsafe_ptr()
    var ap = first.unsafe_ptr() + 1
    var bp = second.unsafe_ptr() + 1
    def cell(index: Int) {imm fp, imm ap, imm bp}:
        var finite = isfinite(ap[index])
        var same = bitcast[DType.uint32](ap[index]) == bitcast[DType.uint32](bp[index])
        fp.unsafe_store(index, Int(not (finite and same)))
    host_parallelize(cell, count)
    for failure in failures:
        require(failure == 0, "PCA transform left an output word unwritten")
    _ = first
    _ = second


def fixture(driver: PCANativeDriver, unit: Int) raises:
    require(unit == 0, "PCA infer-one-call fixture id")
    var source = infer_data()
    var original = source.copy()
    var state = PCATestState(INFER_ROWS, INFER_FEATURES, INFER_COMPONENTS)
    var answer = driver.fit(state.fit_request(Int(source.unsafe_ptr()), INFER_ROWS, INFER_FEATURES,
        INFER_COMPONENTS, False, 0, False, False, PCAFoldStore()))
    require(answer.finite == 0 and answer.in_range == 0, "PCA covariance fit status")
    require(answer.keep == INFER_COMPONENTS, "PCA kept the requested components")
    var count = INFER_ROWS * INFER_COMPONENTS
    var first = prefilled(count, bitcast[DType.float32](UInt32(0x7FC00001)))
    var second = prefilled(count, Float32(-12345.5))
    var status = driver.query(state.query_request(Int(source.unsafe_ptr()), Int(first.unsafe_ptr()) + 4,
        INFER_ROWS, INFER_FEATURES, INFER_COMPONENTS, INFER_ROWS, False, PCAFoldStore()))
    require(status.finite == 0 and status.in_range == 0, "first PCA transform status")
    status = driver.query(state.query_request(Int(source.unsafe_ptr()), Int(second.unsafe_ptr()) + 4,
        INFER_ROWS, INFER_FEATURES, INFER_COMPONENTS, INFER_ROWS, False, PCAFoldStore()))
    require(status.finite == 0 and status.in_range == 0, "second PCA transform status")
    guards(first, "first PCA transform output")
    guards(second, "second PCA transform output")
    fully_written(first, second, count)
    unchanged(source, original, "PCA transform input")
    _ = first^
    _ = second^
    _ = state^
