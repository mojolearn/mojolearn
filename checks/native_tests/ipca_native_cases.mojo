# SPDX-License-Identifier: Apache-2.0
"""Actual IncrementalPCA fit/update/query graphs and independent covariance oracles.

Original public input is 6x3 with rank two and fit batches of three. The
repeated-batch reference retains rank-two covariance between updates, so it
checks the incremental approximation rather than substituting full PCA.
"""
from std.math import sqrt, isfinite
from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.ipca_native_api import IPCANativeDriver, IPCAOutputs, IPCAPrevious, IPCAStateIds, IPCAFitRequest, IPCAPartialRequest, IPCAQueryRequest, IPCAFitResult
from checks.native_tests.pca_native_cases import require, close_values, guards, unchanged, write_guarded, PCA_TEST_GUARD

comptime IPCA_NATIVE_FIXTURES = 15
comptime IPCA_RESIDENT_FIXTURES = 2


def source_values(name: String, count: Int) raises -> List[Float32]:
    var result = List[Float32](unsafe_uninit_length=count)
    with open("checks/native_tests/fixtures/ipca/" + name + ".f32", "r") as file:
        var bytes = file.read_bytes()
        require(len(bytes) == 4 * count, "IPCA source extent")
        var bp = bytes.unsafe_ptr()
        var rp = result.unsafe_ptr()
        def cell(index: Int) {imm bp, imm rp}:
            var at = 4 * index
            var bits = UInt32(bp[at]) | (UInt32(bp[at + 1]) << 8) | (UInt32(bp[at + 2]) << 16) | (UInt32(bp[at + 3]) << 24)
            rp.unsafe_store(index, bitcast[DType.float32](bits))
        host_parallelize(cell, count)
        _ = bytes^
    return result^


def reference_values(name: String, count: Int) raises -> List[Float64]:
    var result = List[Float64](unsafe_uninit_length=count)
    with open("checks/native_tests/fixtures/ipca/" + name + ".f64", "r") as file:
        var bytes = file.read_bytes()
        require(len(bytes) == 8 * count, "IPCA reference extent")
        var bp = bytes.unsafe_ptr()
        var rp = result.unsafe_ptr()
        def cell(index: Int) {imm bp, imm rp}:
            var bits = UInt64(0)
            for byte in range(8):  # Fixed-width file decoding, not data work.
                bits |= UInt64(bp[index * 8 + byte]) << UInt64(byte * 8)
            rp.unsafe_store(index, bitcast[DType.float64](bits))
        host_parallelize(cell, count)
        _ = bytes^
    return result^


struct IPCATestRoom(Movable):
    var components: List[Float32]
    var singular: List[Float32]
    var mean: List[Float32]
    var population_variance: List[Float32]
    var explained_variance: List[Float32]
    var explained_ratio: List[Float32]

    def __init__(out self):
        self.components = List[Float32](length=11, fill=PCA_TEST_GUARD)
        self.singular = List[Float32](length=5, fill=PCA_TEST_GUARD)
        self.mean = List[Float32](length=5, fill=PCA_TEST_GUARD)
        self.population_variance = List[Float32](length=5, fill=PCA_TEST_GUARD)
        self.explained_variance = List[Float32](length=5, fill=PCA_TEST_GUARD)
        self.explained_ratio = List[Float32](length=5, fill=PCA_TEST_GUARD)

    def outputs(self) -> IPCAOutputs:
        return IPCAOutputs(Int(self.components.unsafe_ptr()) + 4, Int(self.singular.unsafe_ptr()) + 4,
            Int(self.mean.unsafe_ptr()) + 4, Int(self.population_variance.unsafe_ptr()) + 4,
            Int(self.explained_variance.unsafe_ptr()) + 4, Int(self.explained_ratio.unsafe_ptr()) + 4)

    def previous(self) -> IPCAPrevious:
        var output = self.outputs()
        return IPCAPrevious(output.components, output.singular, output.mean, output.population_variance)

    def check_guards(self) raises:
        guards(self.components, "IPCA output field 0")
        guards(self.singular, "IPCA output field 1")
        guards(self.mean, "IPCA output field 2")
        guards(self.population_variance, "IPCA output field 3")
        guards(self.explained_variance, "IPCA output field 4")
        guards(self.explained_ratio, "IPCA output field 5")


def release(driver: IPCANativeDriver, mut result: IPCAFitResult) raises:
    var failed = False
    for index in range(5):  # Release each separately owned resident field once.
        var handle = result.state.values[index]
        result.state.values[index] = -1
        if handle >= 0:
            try:
                driver.free(handle)
            except error:
                print("FAIL IPCA resident release", index, String(error))
                failed = True
    result.kept = False
    require(not failed, "IPCA resident cleanup failure")


def state_oracle(room: IPCATestRoom, result: IPCAFitResult, name: String, count: Int, resident: Bool) raises:
    require(result.count == count and result.kept == resident, "IPCA resulting count/resident status")
    var mean = reference_values(name + "-mean", 3)
    var population = reference_values(name + "-population-variance", 3)
    var singular = reference_values(name + "-singular", 3)
    var explained = reference_values(name + "-explained", 3)
    var ratio = reference_values(name + "-ratio", 3)
    var noise = reference_values(name + "-noise", 1)
    var vectors = reference_values(name + "-vectors", 9)
    close_values(room.mean, mean, 3, 1e-4, 1e-5, "IPCA independent full-history mean")
    close_values(room.population_variance, population, 3, 1e-4, 1e-5, "IPCA independent full-history population variance")
    close_values(room.singular, singular, count, 1e-3, 1e-4, "IPCA independent incremental singular values")
    close_values(room.explained_variance, explained, count, 1e-3, 1e-4, "IPCA independent explained variance")
    close_values(room.explained_ratio, ratio, count, 1e-3, 1e-5, "IPCA independent explained ratio")
    var expected_noise = noise[0] if count < 3 else Float64(0)
    require(isfinite(result.noise) and abs(result.noise - expected_noise) <= Float64(1e-4) + Float64(1e-3) * abs(expected_noise),
            "IPCA independent trailing covariance noise")
    var failures = List[Int](length=count * 9, fill=0)
    var fp = failures.unsafe_ptr()
    var cp = room.components.unsafe_ptr() + 1
    var vp = vectors.unsafe_ptr()
    def cell(index: Int) {imm fp, imm cp, imm vp}:
        var component = index // 9
        var row = (index % 9) // 3
        var col = index % 3
        var actual = Float64(cp[component * 3 + row]) * Float64(cp[component * 3 + col])
        var expected = vp[component * 3 + row] * vp[component * 3 + col]
        fp.unsafe_store(index, Int(not (abs(actual - expected) <= Float64(2e-4) + Float64(2e-3) * abs(expected))))
    host_parallelize(cell, count * 9)
    for failed in failures:
        require(failed == 0, "IPCA independent component eigenspace, sign invariant")
    for index in range(5):
        require(result.state.values[index] >= 0 if resident else result.state.values[index] == -1,
                "IPCA returned resident field identities or empty sentinels")
    room.check_guards()
    _ = vectors^
    _ = room


def query_oracle(driver: IPCANativeDriver, source: List[Float32], room: IPCATestRoom,
                 count: Int, whiten: Bool, resident: Bool, state: IPCAStateIds) raises:
    var scores = List[Float32](length=6 * count + 2, fill=PCA_TEST_GUARD)
    var inverse = List[Float32](length=20, fill=PCA_TEST_GUARD)
    var expected = List[Float64](unsafe_uninit_length=6 * count)
    var reconstructed = List[Float64](unsafe_uninit_length=18)
    var xp = source.unsafe_ptr()
    var cp = room.components.unsafe_ptr() + 1
    var mp = room.mean.unsafe_ptr() + 1
    var vp = room.explained_variance.unsafe_ptr() + 1
    var ep = expected.unsafe_ptr()
    def forward(index: Int) {imm xp, imm cp, imm mp, imm vp, imm ep, imm count, imm whiten}:
        var row = index // count
        var component = index % count
        var value = Float64(0)
        for feature in range(3):  # Three-term mathematical dot product.
            value += (Float64(xp[row * 3 + feature]) - Float64(mp[feature])) * Float64(cp[component * 3 + feature])
        if whiten:
            value /= sqrt(Float64(vp[component]))
        ep.unsafe_store(index, value)
    host_parallelize(forward, 6 * count)
    var rp = reconstructed.unsafe_ptr()
    def backward(index: Int) {imm ep, imm cp, imm mp, imm vp, imm rp, imm count, imm whiten}:
        var row = index // 3
        var feature = index % 3
        var value = Float64(mp[feature])
        for component in range(count):
            var term = ep[row * count + component]
            if whiten:
                term *= sqrt(Float64(vp[component]))
            value += term * Float64(cp[component * 3 + feature])
        rp.unsafe_store(index, value)
    host_parallelize(backward, 18)
    var output = room.outputs()
    try:
        driver.query(IPCAQueryRequest(Int(source.unsafe_ptr()), 6, 3, count, output.mean, output.components,
            output.explained_variance, Int(scores.unsafe_ptr()) + 4, False, whiten, resident, state.copy()))
        close_values(scores, expected, 6 * count, 2e-4, 2e-4, "IPCA mathematical transform")
        driver.query(IPCAQueryRequest(Int(scores.unsafe_ptr()) + 4, 6, 3, count, output.mean, output.components,
            output.explained_variance, Int(inverse.unsafe_ptr()) + 4, True, whiten, resident, state.copy()))
        close_values(inverse, reconstructed, 18, 2e-4, 2e-4, "IPCA mathematical inverse")
        if count == 3:
            var original = List[Float64](unsafe_uninit_length=18)
            var op = original.unsafe_ptr()
            def copy_cell(index: Int) {imm op, imm xp}:
                op.unsafe_store(index, Float64(xp[index]))
            host_parallelize(copy_cell, 18)
            close_values(inverse, original, 18, 2e-4, 2e-4, "IPCA full-rank original-data reconstruction")
        guards(scores, "IPCA query scores")
        guards(inverse, "IPCA inverse result")
        room.check_guards()
    finally:
        _ = scores^
        _ = inverse^
        _ = expected^
        _ = reconstructed^
        _ = source
        _ = room


def fitted_case(driver: IPCANativeDriver, unit: Int, host: Bool) raises:
    var source = source_values("source", 18)
    var before = source.copy()
    var room = IPCATestRoom()
    var resident = unit == 15
    var count = 3 if unit == 3 or unit == 4 else 2
    var requested = -1 if unit == 4 else count
    var batch = 6 if unit == 3 else 3
    var result = IPCAFitResult(0, 0, False, IPCAStateIds())
    try:
        result = driver.fit(IPCAFitRequest(Int(source.unsafe_ptr()), 6, 3, requested, batch, room.outputs(), resident))
        state_oracle(room, result, "all6", count, resident)
        query_oracle(driver, source, room, count, unit == 2 or resident, resident, result.state)
        unchanged(source, before, "IPCA fitted source")
    finally:
        release(driver, result)
        _ = room^
        _ = source^


def partial_case(driver: IPCANativeDriver, unit: Int, host: Bool) raises:
    var source = source_values("source", 18)
    var before = source.copy()
    var dummy = IPCATestRoom()
    var first = IPCATestRoom()
    var second = IPCATestRoom()
    var resident = unit == 16
    var repeated = unit == 7
    var rows = 6 if repeated else 3
    var first_result = IPCAFitResult(0, 0, False, IPCAStateIds())
    var second_result = IPCAFitResult(0, 0, False, IPCAStateIds())
    try:
        first_result = driver.partial(IPCAPartialRequest(Int(source.unsafe_ptr()), rows, 3, 2, 0, 0, 0,
            dummy.previous(), first.outputs(), False, IPCAStateIds(), resident))
        state_oracle(first, first_result, "all6" if repeated else "first3", 2, resident)
        var source_address = Int(source.unsafe_ptr()) + (0 if repeated else 36)
        second_result = driver.partial(IPCAPartialRequest(source_address, rows, 3, 2, rows, 2, 3,
            first.previous(), second.outputs(), resident, first_result.state.copy(), resident))
        state_oracle(second, second_result, "repeated12" if repeated else "all6", 2, resident)
        query_oracle(driver, source, second, 2, unit == 6 or resident, resident, second_result.state)
        # Both generations remain live: query the old resident IDs after the
        # update as well, so host publication alone cannot hide pool aliasing.
        state_oracle(first, first_result, "all6" if repeated else "first3", 2, resident)
        if resident:
            query_oracle(driver, source, first, 2, False, True, first_result.state)
        unchanged(source, before, "IPCA partial-fit source")
    finally:
        try:
            release(driver, second_result)
        finally:
            try:
                release(driver, first_result)
            finally:
                _ = dummy^
                _ = first^
                _ = second^
                _ = source^


def refusal_case(driver: IPCANativeDriver, unit: Int) raises:
    var source = source_values("source", 18)
    var original = source.copy()
    var previous = IPCATestRoom()
    var output = IPCATestRoom()
    var untouched = IPCATestRoom()
    var dst = List[Float32](length=14, fill=PCA_TEST_GUARD)
    var dst_before = dst.copy()
    var expected = String("integer division or modulo by zero")
    if unit == 9:
        expected = "n_components=4 must be between 1 and n_features=3"
    elif unit == 10:
        expected = "Number of input features has changed from the previous batch"
    elif unit == 11:
        expected = "Found array with 1 sample(s) (shape=(1, 3)) while a minimum of 2 is required."
    elif unit == 12:
        expected = "x_decomp: a resident state id must be -1 or a pool id"
    elif unit == 13:
        expected = "x_decomp: dimension out of range"
    var refused = False
    try:
        try:
            if unit == 8 or unit == 13:
                _ = driver.fit(IPCAFitRequest(Int(source.unsafe_ptr()), 2147483648 if unit == 13 else 6,
                    3, 2, 0 if unit == 8 else 3, output.outputs(), False))
            elif unit == 12:
                var fields = output.outputs()
                var ids = IPCAStateIds()
                ids.values[0] = -2
                driver.query(IPCAQueryRequest(Int(source.unsafe_ptr()), 6, 3, 2, fields.mean, fields.components,
                    fields.explained_variance, Int(dst.unsafe_ptr()) + 4, False, False, True, ids^))
            else:
                _ = driver.partial(IPCAPartialRequest(Int(source.unsafe_ptr()), 1 if unit == 11 else 3,
                    3, 4 if unit == 9 else (1 if unit == 11 else 2), 3 if unit == 10 else 0,
                    2 if unit == 10 else 0, 4 if unit == 10 else 0, previous.previous(), output.outputs(),
                    False, IPCAStateIds(), False))
        except error:
            refused = String(error) == expected
        require(refused, "IPCA actual provider preserves native refusal")
        unchanged(source, original, "IPCA refused source")
        unchanged(dst, dst_before, "IPCA refused query destination")
        unchanged(output.components, untouched.components, "IPCA refused output field 0")
        unchanged(output.singular, untouched.singular, "IPCA refused output field 1")
        unchanged(output.mean, untouched.mean, "IPCA refused output field 2")
        unchanged(output.population_variance, untouched.population_variance, "IPCA refused output field 3")
        unchanged(output.explained_variance, untouched.explained_variance, "IPCA refused output field 4")
        unchanged(output.explained_ratio, untouched.explained_ratio, "IPCA refused output field 5")
    finally:
        _ = source^
        _ = previous^
        _ = output^
        _ = dst^


def prepare_case(driver: IPCANativeDriver, host: Bool) raises:
    var source = source_values("source", 18)
    var room = IPCATestRoom()
    var components: List[Float32] = [1, 0, 0, 0, 1, 0]
    var mean: List[Float32] = [1, 2, 3]
    var variance: List[Float32] = [4, 9]
    write_guarded(room.components, components)
    write_guarded(room.mean, mean)
    write_guarded(room.explained_variance, variance)
    var handle = -1
    try:
        if host:
            var refused = False
            try:
                handle = driver.prepare(room.outputs().mean, 3)
            except error:
                refused = String(error) == "IncrementalPCA host provider does not keep resident state"
            require(refused, "IPCA host prepare refuses unsupported resident state")
        else:
            handle = driver.prepare(room.outputs().mean, 3)
            require(handle >= 0, "IPCA prepared resident mean returns valid handle including zero")
            var ids = IPCAStateIds()
            ids.values[2] = handle
            query_oracle(driver, source, room, 2, True, True, ids)
    finally:
        try:
            if handle >= 0:
                driver.free(handle)
        finally:
            _ = room^
            _ = source^


def fixture(driver: IPCANativeDriver, unit: Int, host: Bool) raises:
    if unit == 0:
        require(driver.metadata(0) == 1 and driver.metadata(2) == (7 if host else 15),
                "IPCA actual native mode and capability metadata")
        var vendor = driver.metadata(1)
        require(vendor == 1 if host else vendor >= 2 and vendor <= 4, "IPCA actual native vendor")
    elif unit <= 4 or unit == 15:
        fitted_case(driver, unit, host)
    elif unit <= 7 or unit == 16:
        partial_case(driver, unit, host)
    elif unit <= 13:
        refusal_case(driver, unit)
    else:
        prepare_case(driver, host)
