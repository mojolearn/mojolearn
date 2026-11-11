# SPDX-License-Identifier: Apache-2.0
"""Last native Arc releases the Spectral handle once, including quiet errors.

Counters are real owned native storage retained beyond every release callback.
This suite makes no GPU concurrency or public Python lifetime claim.
"""
from std.memory import ArcPointer
from core.host_parallel import host_parallelize
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.spectral_native_api import SpectralDriver, SPECTRAL_RELEASE
from core.spectral_native_state import SpectralResidentState

comptime SPECTRAL_STATE_FIXTURES = 6


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def release_entry(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
                  result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        require(operation == SPECTRAL_RELEASE and na == 0 and np == 1 and nf == 0,
            "Spectral owner uses the native release operation")
        var handle = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=pp)[0]
        var counter = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=handle)
        counter[0] += 1
        if counter[1] != 0:
            raise Error("Spectral prescribed cleanup failure")
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[0] = 0
        return 0
    except error:
        native_descriptor_report(error,context,sink)
        return 1


def unwind(handle: Int) raises:
    var cache = ArcPointer[SpectralResidentState](SpectralResidentState(SpectralDriver(release_entry),handle))
    var query = cache.copy()
    _ = cache^
    try:
        raise Error("Spectral original query failure")
    finally:
        _ = query^


def fixture(unit: Int) raises:
    var counter = List[Int](length=2,fill=0)
    var handle = Int(counter.unsafe_ptr())
    if unit == 5:
        counter[1] = 1
        var refused = False
        try:
            unwind(handle)
        except error:
            require(String(error) == "Spectral original query failure", "Quiet cleanup preserves original unwinding error")
            refused = True
        require(refused,"Spectral ownership unwinding fixture did not raise")
    else:
        var cache = ArcPointer[SpectralResidentState](SpectralResidentState(SpectralDriver(release_entry),handle))
        if unit == 0:
            _ = cache^
        elif unit == 1:
            var query = cache.copy()
            _ = cache^
            require(counter[0] == 0 and query[].handle == handle,"Retired Spectral cache retains active query handle")
            _ = query^
        elif unit == 2:
            var first = cache.copy()
            var second = cache.copy()
            _ = cache^
            _ = first^
            require(counter[0] == 0 and second[].handle == handle,"Last of two Spectral queries retains the handle")
            _ = second^
        elif unit == 3:
            var slots = List[ArcPointer[SpectralResidentState]]()
            slots.append(cache^)
            var query = slots[0].copy()
            var retired = slots.pop()
            _ = retired^
            require(counter[0] == 0 and query[].handle == handle,"Native table removal preserves copied Spectral lease")
            _ = query^
            _ = slots^
        else:
            counter[1] = 1
            _ = cache^
    require(counter[0] == 1,"Last Spectral Arc releases exactly once")
    _ = counter^


def run_checks() raises:
    var failures = List[Int](length=SPECTRAL_STATE_FIXTURES,fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            fixture(unit)
        except error:
            fp.unsafe_store(unit,1)
            print("FAIL spectral_state fixture",unit,String(error))
    host_parallelize(task,SPECTRAL_STATE_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0,"spectral_state failed " + String(failed) + " fixtures")
    print("PASS spectral_state:",SPECTRAL_STATE_FIXTURES,"native last-Arc ownership fixtures")
