# SPDX-License-Identifier: Apache-2.0
"""Native handle leases survive cache retirement and release exactly once.

The descriptor records releases only; these are CPU ownership contracts,
not GPU concurrent-submit tests or a substitute numerical provider.
"""
from std.memory import ArcPointer
from core.host_parallel import host_parallelize
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.kmeans_variants_api import KMeansVariantsDriver, VARIANTS_RELEASE
from core.kmeans_variants_resident_state import VariantsResidentState

comptime KMEANS_VARIANTS_LEASE_FIXTURES = 4


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def entry(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
          result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        require(operation == VARIANTS_RELEASE and na == 0 and np == 1 and nf == 0,
                "lease uses the typed release operation")
        var handle = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=pp)[0]
        var counter = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=handle)
        counter[0] += 1
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[0] = 0
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


def fixture(unit: Int) raises:
    var counter = List[Int](length=1, fill=0)
    var handle = Int(counter.unsafe_ptr())
    var resident = ArcPointer[VariantsResidentState](VariantsResidentState(KMeansVariantsDriver(entry), handle))
    if unit == 0:
        _ = resident^
    elif unit == 1:
        var query = resident.copy()
        _ = resident^
        require(counter[0] == 0 and query[].handle == handle, "retiring cache keeps active handle")
        _ = query^
    elif unit == 2:
        var first = resident.copy()
        var second = resident.copy()
        _ = resident^
        _ = first^
        require(counter[0] == 0 and second[].handle == handle, "last of two active queries still owns handle")
        _ = second^
    else:
        var slots = List[ArcPointer[VariantsResidentState]]()
        slots.append(resident^)
        var query = slots[0].copy()
        var retired = slots.pop()
        _ = retired^
        require(counter[0] == 0 and query[].handle == handle, "table removal does not invalidate a copied lease")
        _ = query^
        _ = slots^
    require(counter[0] == 1, "last native lease releases exactly once")
    _ = counter^


def run_checks() raises:
    var failures = List[Int](length=KMEANS_VARIANTS_LEASE_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def run(unit: Int) {imm fp}:
        try:
            fixture(unit)
        except problem:
            fp[unit] = 1
            print("FAIL kmeans_variants_lease fixture", unit, String(problem))
    host_parallelize(run, KMEANS_VARIANTS_LEASE_FIXTURES)
    var failed = 0
    for value in failures:  # metadata: fixture summaries
        failed += value
    require(failed == 0, "native KMeans variant leases failed " + String(failed) + " fixtures")
    print("PASS kmeans_variants_lease:", KMEANS_VARIANTS_LEASE_FIXTURES, "native ownership fixtures")
