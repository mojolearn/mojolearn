# SPDX-License-Identifier: Apache-2.0
"""Native descriptor adapter around the unchanged cluster execution owner."""
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.kmeans_variants_api import (
    VariantsResult, VARIANTS_MODE, VARIANTS_CAPABILITIES, VARIANTS_MINIBATCH_FIT,
    VARIANTS_MINIBATCH_PARTIAL, VARIANTS_BISECT_FIT, VARIANTS_BISECT_PREDICT,
    VARIANTS_NEAREST, VARIANTS_DISTANCES, VARIANTS_PREPARE, VARIANTS_RELEASE, VARIANTS_RESIDENT_QUERY,
)
from x_cluster.entries import ENTRY_MINIBATCH, ENTRY_MINIBATCH_PARTIAL, ENTRY_BISECT, ENTRY_NEAREST, ENTRY_DISTANCES
from x_cluster.out import ClusterOut


trait KMeansVariantsBackend:
    @staticmethod
    def mode() -> Int: ...
    @staticmethod
    def capabilities() -> Int: ...
    @staticmethod
    def execute(which: Int, source: Int, count: Int, auxiliary: Int, auxiliary_count: Int,
                ints: List[Int], floats: List[Float64]) raises -> ClusterOut: ...
    @staticmethod
    def tree_predict(source: Int, count: Int, centers: Int, center_count: Int,
                     nodes: Int, node_count: Int, rows: Int, features: Int) raises -> List[Int32]: ...
    @staticmethod
    def prepare(centers: Int, clusters: Int, features: Int) raises -> Int: ...
    @staticmethod
    def release(handle: Int) raises: ...
    @staticmethod
    def resident_query(handle: Int, kind: Int, source: Int, rows: Int, clusters: Int,
                       features: Int, destination: Int) raises -> Float64: ...


def _word(address: Int, at: Int) -> Int:
    return Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=address)[at]


def _take(operation: Int, var result: ClusterOut) raises -> VariantsResult:
    var value = VariantsResult()
    if operation == VARIANTS_MINIBATCH_FIT or operation == VARIANTS_MINIBATCH_PARTIAL:
        var scalar_count = 3 if operation == VARIANTS_MINIBATCH_FIT else 4
        if len(result.f) != 2 or len(result.i) != 1 or len(result.s) != scalar_count:
            raise Error("KMeans variants native ABI: invalid MiniBatch result slots")
        value.centers = result.f.pop(0)
        value.counts = result.f.pop(0)
        value.labels = result.i.pop(0)
        value.inertia = result.s[0]
        if operation == VARIANTS_MINIBATCH_FIT:
            value.steps = Int(result.s[1])
            value.iterations = Int(result.s[2])
        else:
            value.since_reassignment = Int(result.s[1])
            value.state_hi = Int(result.s[2])
            value.state_lo = Int(result.s[3])
    elif operation == VARIANTS_BISECT_FIT:
        if len(result.f) != 2 or len(result.i) != 2 or len(result.s) != 1:
            raise Error("KMeans variants native ABI: invalid Bisect result slots")
        value.centers = result.f.pop(0)
        value.tree_centers = result.f.pop(0)
        value.labels = result.i.pop(0)
        value.tree_nodes = result.i.pop(0)
        value.inertia = result.s[0]
    elif operation == VARIANTS_NEAREST:
        if len(result.f) != 1 or len(result.i) != 1 or len(result.s) > 1:
            raise Error("KMeans variants native ABI: invalid nearest result slots")
        value.distances = result.f.pop(0)
        value.labels = result.i.pop(0)
        if len(result.s):
            value.inertia = result.s[0]
    else:
        if len(result.f) != 1 or len(result.i) != 0 or len(result.s) != 0:
            raise Error("KMeans variants native ABI: invalid distance result slots")
        value.distances = result.f.pop(0)
    return value^


def kmeans_variants_dispatch[K: KMeansVariantsBackend](operation: Int,
    aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
    result: Int, context: Int, sink: NativeDescriptorError) -> Int:
    try:
        if result == 0 or (na > 0 and aa == 0) or (np > 0 and pp == 0) or (nf > 0 and ff == 0):
            raise Error("KMeans variants native ABI: missing request or result span")
        var answer = 0
        if operation == VARIANTS_MODE or operation == VARIANTS_CAPABILITIES:
            if na != 0 or np != 0 or nf != 0:
                raise Error("KMeans variants native ABI: metadata takes no arguments")
            answer = K.mode() if operation == VARIANTS_MODE else K.capabilities()
        elif operation == VARIANTS_PREPARE:
            if na != 1 or np != 2 or nf != 0:
                raise Error("KMeans variants native ABI: invalid prepare request")
            answer = K.prepare(_word(aa, 0), _word(pp, 0), _word(pp, 1))
        elif operation == VARIANTS_RELEASE:
            if na != 0 or np != 1 or nf != 0:
                raise Error("KMeans variants native ABI: invalid release request")
            K.release(_word(pp, 0))
        elif operation == VARIANTS_RESIDENT_QUERY:
            if na != 3 or np != 5 or nf != 0 or _word(aa, 2) == 0:
                raise Error("KMeans variants native ABI: invalid resident query request")
            var dst = MutPointer[VariantsResult, MutUntrackedOrigin](unsafe_from_address=_word(aa, 2))
            dst[].inertia = K.resident_query(_word(pp, 0), _word(pp, 1), _word(aa, 0),
                _word(pp, 2), _word(pp, 3), _word(pp, 4), _word(aa, 1))
        elif operation == VARIANTS_BISECT_PREDICT:
            if na != 4 or np != 5 or nf != 0 or _word(aa, 3) == 0:
                raise Error("KMeans variants native ABI: invalid tree query request")
            var value = VariantsResult()
            value.labels = K.tree_predict(_word(aa, 0), _word(pp, 0), _word(aa, 1),
                _word(pp, 1), _word(aa, 2), _word(pp, 2), _word(pp, 3), _word(pp, 4))
            var dst = MutPointer[VariantsResult, MutUntrackedOrigin](unsafe_from_address=_word(aa, 3))
            dst[] = value^
        else:
            var integers = 0
            var scalars = 0
            var which = -1
            if operation == VARIANTS_MINIBATCH_FIT:
                integers = 15
                scalars = 2
                which = ENTRY_MINIBATCH
            elif operation == VARIANTS_MINIBATCH_PARTIAL:
                integers = 13
                scalars = 1
                which = ENTRY_MINIBATCH_PARTIAL
            elif operation == VARIANTS_BISECT_FIT:
                integers = 10
                scalars = 1
                which = ENTRY_BISECT
            elif operation == VARIANTS_NEAREST or operation == VARIANTS_DISTANCES:
                integers = 6
                which = ENTRY_NEAREST if operation == VARIANTS_NEAREST else ENTRY_DISTANCES
            else:
                raise Error("KMeans variants native ABI: unknown operation")
            if na != 3 or np != integers or nf != scalars or _word(aa, 2) == 0:
                raise Error("KMeans variants native ABI: invalid numerical request")
            var ip = List[Int]()
            for i in range(2, integers):  # driver-loop: fixed native entry option words
                ip.append(_word(pp, i))
            var fp = List[Float64]()
            for i in range(scalars):  # driver-loop: at most two native float options
                fp.append(Pointer[Float64, ImmUntrackedOrigin](unsafe_from_address=ff)[i])
            var native = K.execute(which, _word(aa, 0), _word(pp, 0), _word(aa, 1), _word(pp, 1), ip, fp)
            var value = _take(operation, native^)
            var dst = MutPointer[VariantsResult, MutUntrackedOrigin](unsafe_from_address=_word(aa, 2))
            dst[] = value^
            _ = ip^
            _ = fp^
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[0] = answer
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1
