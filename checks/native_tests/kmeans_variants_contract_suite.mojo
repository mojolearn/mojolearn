# SPDX-License-Identifier: Apache-2.0
"""Typed request/result transport contracts, independent of numerical kernels.

The witness checks the documented legacy entry words and returns distinct
markers. It is not a replacement algorithm. Every fixture owns its result
slot; independent fixtures execute concurrently with no shared mutable state.
"""
from core.host_parallel import host_parallelize
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_fixed_call import native_descriptor_call_fixed
from core.kmeans_variants_api import (
    KMeansVariantsDriver, VariantsSpan, VariantsResultSlot,
    MiniBatchFitRequest, MiniBatchPartialRequest, BisectFitRequest,
    BisectPredictRequest, VariantsQueryRequest,
    VARIANTS_MODE, VARIANTS_CAPABILITIES, VARIANTS_MINIBATCH_FIT,
)
from x_cluster.variants_dispatch import KMeansVariantsBackend, kmeans_variants_dispatch
from x_cluster.entries import ENTRY_MINIBATCH, ENTRY_MINIBATCH_PARTIAL, ENTRY_BISECT, ENTRY_NEAREST, ENTRY_DISTANCES
from x_cluster.out import ClusterOut
from core.kmeans_variants_plan import VariantsAuxiliary

comptime KMEANS_VARIANTS_CONTRACT_FIXTURES = 23


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def words(actual: List[Int], expected: List[Int]) raises:
    require(len(actual) == len(expected), "legacy entry word count")
    for i in range(len(expected)):  # metadata: bounded ABI option words
        require(actual[i] == expected[i], "legacy entry word " + String(i))


struct Witness(KMeansVariantsBackend):
    @staticmethod
    def mode() -> Int:
        return 27

    @staticmethod
    def capabilities() -> Int:
        return 3

    @staticmethod
    def execute(which: Int, source: Int, count: Int, auxiliary: Int,
                auxiliary_count: Int, ints: List[Int], floats: List[Float64]) raises -> ClusterOut:
        require(source == 101 and count == 107 and auxiliary == 103 and auxiliary_count == 109,
                "source and auxiliary spans preserve independent addresses/counts")
        var value = ClusterOut()
        if which == ENTRY_MINIBATCH or which == ENTRY_MINIBATCH_PARTIAL:
            var centers: List[Float32] = [11, 13]
            value.f.append(centers^)
            var counts: List[Float32] = [17, 19]
            value.f.append(counts^)
            var labels: List[Int32] = [1, 0]
            value.i.append(labels^)
            value.s.append(23.5)
            if which == ENTRY_MINIBATCH:
                var expected: List[Int] = [2, 3, 5, 7, 11, -1, 13, 17, 1, 19, 0, 1, 1]
                words(ints, expected)
                require(len(floats) == 2 and floats[0] == 0.125 and floats[1] == 0.25,
                        "fit tolerance/reassignment order")
                value.s.append(29)
                value.s.append(31)
            else:
                var expected: List[Int] = [2, 3, 5, 0, 2, 7, 11, 13, 2147483649, 4294967295, 1]
                words(ints, expected)
                require(len(floats) == 1 and floats[0] == 0.25, "partial reassignment ratio")
                value.s.append(37)
                value.s.append(2147483651)
                value.s.append(4294967293)
        elif which == ENTRY_BISECT:
            var expected: List[Int] = [2, 3, 5, 7, 1, 11, 13, 0]
            words(ints, expected)
            require(len(floats) == 1 and floats[0] == 0.125, "bisect tolerance")
            var centers: List[Float32] = [11, 13]
            value.f.append(centers^)
            var tree_centers: List[Float32] = [17, 19, 23]
            value.f.append(tree_centers^)
            var labels: List[Int32] = [1, 0]
            value.i.append(labels^)
            var tree_nodes: List[Int32] = [1, 2, -1, -1, -1, 0, -1, -1, 1]
            value.i.append(tree_nodes^)
            value.s.append(29.5)
        else:
            require(which == ENTRY_NEAREST or which == ENTRY_DISTANCES, "query selects existing entry")
            require(len(ints) == 4 and ints[0] == 2 and ints[1] == 5 and ints[2] == 3,
                    "query rows/clusters/features")
            require(len(floats) == 0, "query has no float options")
            var distances: List[Float32] = [2, 3]
            value.f.append(distances^)
            if which == ENTRY_NEAREST:
                var labels: List[Int32] = [1, 0]
                value.i.append(labels^)
                if ints[3] != 0:
                    value.s.append(13)
        return value^

    @staticmethod
    def tree_predict(source: Int, count: Int, centers: Int, center_count: Int,
                     nodes: Int, node_count: Int, rows: Int, features: Int) raises -> List[Int32]:
        require(source == 101 and count == 107 and centers == 103 and center_count == 109,
                "tree query numeric spans")
        require(nodes == 113 and node_count == 127 and rows == 2 and features == 3,
                "tree nodes remain a distinct native Int32 span")
        var labels: List[Int32] = [3, 1]
        return labels^

    @staticmethod
    def prepare(centers: Int, clusters: Int, features: Int) raises -> Int:
        require(centers == 103 and clusters == 5 and features == 3, "resident prepare words")
        return 137

    @staticmethod
    def release(handle: Int) raises:
        require(handle == 137, "resident release handle")

    @staticmethod
    def resident_query(handle: Int, kind: Int, source: Int, rows: Int,
                       clusters: Int, features: Int, destination: Int) raises -> Float64:
        require(handle == 137 and kind >= 0 and kind <= 2 and source == 101,
                "resident handle/kind/source")
        require(rows == 2 and clusters == 5 and features == 3 and destination == 139,
                "resident dimensions/output")
        return Float64(kind) + 0.5


def entry(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
          result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    return kmeans_variants_dispatch[Witness](operation, aa, na, pp, np, ff, nf, result, context, sink)


def fit_request() -> MiniBatchFitRequest:
    return MiniBatchFitRequest(VariantsSpan(101, 107), VariantsSpan(103, 109),
        2, 3, 5, 7, 11, -1, 13, 17, True, 19, False, True, True, 0.125, 0.25)


def auxiliary_case(unit: Int) raises:
    if unit == 17:
        var auxiliary = VariantsAuxiliary(0)
        var parts = List[VariantsSpan]()
        auxiliary.copy_parts(parts)
        require(auxiliary.span().address == 0 and auxiliary.span().count == 0, "empty auxiliary has no address")
    elif unit == 18 or unit == 19:
        var first: List[Float32] = [3, -5]
        var last: List[Float32] = [7, 11, 13]
        var parts: List[VariantsSpan] = [VariantsSpan(Int(first.unsafe_ptr()), 2),
            VariantsSpan(0, 0), VariantsSpan(Int(last.unsafe_ptr()), 3)]
        var auxiliary = VariantsAuxiliary(5)
        auxiliary.copy_parts(parts)
        first[0] = 99
        last[2] = 101
        require(auxiliary.values[0] == 3 and auxiliary.values[1] == -5 and auxiliary.values[2] == 7
                and auxiliary.values[3] == 11 and auxiliary.values[4] == 13,
                "ordered disjoint copies skip empty span and own detached values")
        if unit == 19:
            var moved = auxiliary^
            require(moved.span().count == 5 and moved.values[0] == 3 and moved.values[4] == 13,
                    "auxiliary move preserves owned storage")
        _ = first^
        _ = last^
        _ = parts^
    else:
        var caught = False
        try:
            if unit == 20:
                var invalid = VariantsAuxiliary(-1)
            elif unit == 21:
                var invalid = VariantsAuxiliary(0)
                var parts: List[VariantsSpan] = [VariantsSpan(0, 1)]
                invalid.copy_parts(parts)
            else:
                var invalid = VariantsAuxiliary(9223372036854775807 // 4 + 1)
        except problem:
            caught = String(problem).find("cluster auxiliary") >= 0
        require(caught, "invalid auxiliary refuses before pointer access/allocation")


def fixture(unit: Int) raises:
    if unit >= 17:
        auxiliary_case(unit)
        return
    var driver = KMeansVariantsDriver(entry)
    var output = VariantsResultSlot()
    if unit == 0:
        require(driver.metadata(VARIANTS_MODE) == 27, "mode metadata")
        require(driver.metadata(VARIANTS_CAPABILITIES) == 3, "capability metadata")
    elif unit == 1 or unit == 11:
        driver.minibatch_fit(fit_request(), output)
        require(output.storage[0].centers[1] == 13 and output.storage[0].counts[1] == 19,
                "fit independently owns centers and counts")
        require(output.storage[0].labels[0] == 1 and output.storage[0].inertia == 23.5,
                "fit label/scalar roles")
        require(output.storage[0].steps == 29 and output.storage[0].iterations == 31,
                "fit counters retain their legacy roles")
        if unit == 11:
            var moved = output^
            var original_address = moved.address()
            driver.minibatch_fit(fit_request(), moved)
            require(moved.address() == original_address and moved.storage[0].centers[0] == 11,
                    "move/reuse keeps a stable writable result slot")
    elif unit == 2:
        var request = MiniBatchPartialRequest(VariantsSpan(101, 107), VariantsSpan(103, 109),
            2, 3, 5, False, 2, 7, 11, 13, 2147483649, 4294967295, True, 0.25)
        driver.minibatch_partial(request, output)
        require(output.storage[0].since_reassignment == 37 and output.storage[0].state_hi == 2147483651
                and output.storage[0].state_lo == 4294967293, "partial unsigned RNG halves and counter")
        require(output.storage[0].steps == 0 and output.storage[0].iterations == 0,
                "partial counters are not fit counters")
    elif unit == 3:
        driver.bisect_fit(BisectFitRequest(VariantsSpan(101, 107), VariantsSpan(103, 109),
            2, 3, 5, 7, True, 11, 13, False, 0.125), output)
        require(output.storage[0].centers[1] == 13 and output.storage[0].tree_centers[2] == 23,
                "bisect public and tree centers retain distinct ownership")
        require(output.storage[0].tree_nodes[2] == -1 and output.storage[0].tree_nodes[8] == 1
                and output.storage[0].labels[0] == 1 and output.storage[0].inertia == 29.5,
                "bisect node/label/scalar roles")
    elif unit == 4:
        driver.bisect_predict(BisectPredictRequest(VariantsSpan(101, 107), VariantsSpan(103, 109),
            VariantsSpan(113, 127), 2, 3), output)
        require(output.storage[0].labels[0] == 3 and output.storage[0].labels[1] == 1, "tree prediction output")
    elif unit >= 5 and unit <= 7:
        var score = unit == 6
        driver.query(VariantsQueryRequest(VariantsSpan(101, 107), VariantsSpan(103, 109),
            2, 5, 3, score), unit == 7, output)
        require(output.storage[0].distances[0] == 2 and output.storage[0].distances[1] == 3,
                "query distances preserved without conversion")
        require(len(output.storage[0].labels) == (0 if unit == 7 else 2), "query label presence")
        require(output.storage[0].inertia == (Float64(13) if score else Float64(0)), "optional query score")
    elif unit == 8:
        var handle = driver.prepare(103, 5, 3)
        require(handle == 137, "prepared handle returned as native integer")
        driver.release(handle)
    elif unit == 9 or unit == 10 or unit == 12:
        var kind = 0 if unit == 9 else (1 if unit == 10 else 2)
        var score = driver.resident_query(137, kind, 101, 2, 5, 3, 139, output)
        require(score == Float64(kind) + 0.5, "resident query score retained in heap-owned result")
    else:
        var caught = False
        var expected = String("unknown operation")
        try:
            if unit == 13:
                _ = driver.metadata(999)
            elif unit == 14:
                expected = "metadata takes no arguments"
                var a: InlineArray[Int, 1] = [1]
                var p: InlineArray[Int, 0] = []
                var f: InlineArray[Float64, 0] = []
                _ = native_descriptor_call_fixed(entry, VARIANTS_MODE, a, p, f)
            elif unit == 15:
                expected = "invalid numerical request"
                _ = driver.metadata(VARIANTS_MINIBATCH_FIT)
            else:
                expected = "invalid numerical request"
                var a: InlineArray[Int, 3] = [101, 103, 0]
                var p: InlineArray[Int, 15] = [107, 109, 2, 3, 5, 7, 11, -1, 13, 17, 1, 19, 0, 1, 1]
                var f: InlineArray[Float64, 2] = [0.125, 0.25]
                _ = native_descriptor_call_fixed(entry, VARIANTS_MINIBATCH_FIT, a, p, f)
        except problem:
            caught = String(problem).find(expected) >= 0
        require(caught, "malformed request must refuse before reaching backend/output storage")


def run_checks() raises:
    var failures = List[Int](length=KMEANS_VARIANTS_CONTRACT_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def run(unit: Int) {imm fp}:
        try:
            fixture(unit)
        except problem:
            fp[unit] = 1
            print("FAIL kmeans_variants_contract fixture", unit, String(problem))
    host_parallelize(run, KMEANS_VARIANTS_CONTRACT_FIXTURES)
    var failed = 0
    for value in failures:  # metadata: fixture summaries
        failed += value
    require(failed == 0, "native KMeans variant contracts failed " + String(failed) + " fixtures")
    print("PASS kmeans_variants_contract:", KMEANS_VARIANTS_CONTRACT_FIXTURES, "native transport/ownership/refusal fixtures")
