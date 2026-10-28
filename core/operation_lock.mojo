# SPDX-License-Identifier: Apache-2.0
# Shared host recursive lock for native operation owners. The two stable words
# are thread id and recursion depth; only the owning thread accesses depth.
# This is not a GPU mutex and never inspects a Python/thread-library object.
from std.atomic import Atomic, Ordering
from std.ffi import external_call, c_int
from std.sys.info import CompilationTarget
from std.time import perf_counter_ns
from std.math import isfinite


def operation_lock_thread() -> Int64:
    # Supported LP64 Darwin/Linux pthread_t is a pointer/unsigned-long word,
    # as in the existing core.device_plan owner registry.
    return Int64(Int(external_call["pthread_self", UInt]()))


def _operation_lock_wait() raises:
    # A contended operation may wait for GPU work. Sleeping 100 microseconds
    # avoids consuming a CPU core while preserving an immediate uncontended
    # path. This is a general scheduling policy, independent of model shapes.
    # POSIX timespec on supported LP64 hosts is two signed 64-bit words.
    var request = List[Int64](length=2, fill=0)
    # Mojo Pointer is non-nullable. Supply ordinary POSIX remainder storage;
    # interrupted waits still return the same cancellation below.
    var remaining = List[Int64](length=2, fill=0)
    request[1] = 100000
    if external_call["nanosleep", c_int](request.unsafe_ptr(), remaining.unsafe_ptr()) != 0:
        var number = 0
        comptime if CompilationTarget.is_macos():
            number = Int(external_call["__error", MutPointer[c_int, MutUntrackedOrigin]]()[])
        else:
            number = Int(external_call["__errno_location", MutPointer[c_int, MutUntrackedOrigin]]()[])
        if number == 4:
            # Never invoke Python signal handlers while executing native work.
            # Return cancellation so the binding can release owners and resume
            # the interpreter's pending signal handling after this operation.
            raise Error("InterruptedError: native operation lock interrupted by an OS signal")
        raise Error("native operation lock: wait failed (errno " + String(number) + ")")


def operation_lock_acquire(handle: Int, blocking: Bool = True, timeout: Float64 = -1.0) raises -> Bool:
    if handle == 0:
        raise Error("native operation lock: null owner")
    if not isfinite(timeout) or (timeout < 0 and timeout != -1.0):
        raise Error("ValueError: timeout must be non-negative or -1")
    if not blocking and timeout != -1.0:
        raise Error("ValueError: can't specify a timeout for a non-blocking call")
    # Keep the same bounded timeout range as the native LP64 thread primitive.
    if timeout > 9223372036.0:
        raise Error("OverflowError: timeout value is too large")
    var words = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=handle)
    var thread = operation_lock_thread()
    var start = Int(perf_counter_ns()) if timeout >= 0 else 0
    while True:  # driver-loop: over native ownership retries, no data computation
        var owner = Atomic.load[ordering=Ordering.ACQUIRE](words)
        if owner == thread:
            if words[1] == 9223372036854775807:
                raise Error("OverflowError: native operation lock recursion limit")
            words[1] += 1
            return True
        if owner == 0:
            var expected = Int64(0)
            if Atomic.compare_exchange[success_ordering=Ordering.SEQUENTIAL,
                                       failure_ordering=Ordering.RELAXED](words, expected, thread):
                words[1] = 1
                return True
        if not blocking:
            return False
        if timeout >= 0 and Float64(Int(perf_counter_ns()) - start) >= timeout * 1e9:
            return False
        _operation_lock_wait()


def operation_lock_release(handle: Int) raises:
    if handle == 0:
        raise Error("native operation lock: null owner")
    var words = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=handle)
    if Atomic.load[ordering=Ordering.ACQUIRE](words) != operation_lock_thread():
        raise Error("RuntimeError: cannot release un-acquired lock")
    if words[1] > 1:
        words[1] -= 1
    else:
        words[1] = 0
        Atomic.store[ordering=Ordering.RELEASE](words, Int64(0))


def operation_lock_owned(handle: Int) -> Bool:
    if handle == 0:
        return False
    return Atomic.load[ordering=Ordering.ACQUIRE](
        MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=handle)) == operation_lock_thread()


struct NativeOperationLock(Movable):
    var words: List[Int64]

    def __init__(out self):
        self.words = List[Int64](length=2, fill=0)

    def handle(mut self) -> Int:
        return Int(self.words.unsafe_ptr())
