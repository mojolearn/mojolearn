# SPDX-License-Identifier: Apache-2.0
# RUN part of bindings/native_unit_threads.mojo (named apart from its public
# native_unit_threads_run function): typed Mojo only (no Python
# imports). The thread start and join, the unit gate words and the core
# count; the door (native_unit_threads.mojo) holds the GIL protocol: it
# releases the GIL around the gate's yield and around the joins, exactly
# where it did before the split.
from std.atomic import Atomic, Ordering
from std.ffi import c_int, c_size_t, external_call
from std.sys.info import num_physical_cores

#: The thread entry: `void *(*)(void *)` with the argument and result as
#: address words (same register class under the C ABI on every target).
comptime NativeUnitEntry = def(Int) thin abi("C") -> Int

#: Words reserved for a `pthread_attr_t` (56 bytes on glibc x86_64/aarch64,
#: 64 bytes on macOS): 128 bytes.
comptime _ATTR_WORDS = 16
#: Stack of every unit thread: owners recurse over model structure (nested
#: estimators, tree builders); a macOS secondary thread gets 512 KiB by
#: default, the main thread 8 MiB. Reserved address space, not resident.
comptime _UNIT_THREAD_STACK_BYTES = 64 << 20


def native_unit_cores() -> Int:
    """Physical cores of this host (at least 1)."""
    return max(num_physical_cores(), 1)


def native_unit_gate() -> List[Int64]:
    """A gate one unit holds at a time (a unit's gathers or scores that share
    an owner's program state): one zeroed word on the heap."""
    return List[Int64](length=1, fill=0)


def native_unit_gate_try(mut gate: List[Int64]) -> Bool:
    """One attempt to take the gate: an atomic compare-and-swap of its word
    from 0 to 1 (never a plain load: see `native_unit_gate_enter`)."""
    var word = rebind[MutPointer[Int64, MutUntrackedOrigin]](gate.unsafe_ptr())
    var expected = Int64(0)
    return Atomic.compare_exchange[
        success_ordering=Ordering.SEQUENTIAL,
        failure_ordering=Ordering.RELAXED,
    ](word, expected, Int64(1))


def native_unit_yield():
    """Give the CPU to another thread (the caller has released the GIL)."""
    _ = external_call["sched_yield", c_int]()


def native_unit_gate_leave(mut gate: List[Int64]):
    """Open the gate `native_unit_gate_enter` took."""
    var word = rebind[MutPointer[Int64, MutUntrackedOrigin]](gate.unsafe_ptr())
    Atomic.store[ordering=Ordering.RELEASE](word, Int64(0))


def native_unit_threads_start(entry: NativeUnitEntry, arg: Int, extra: Int, mut handles: List[Int]) -> Int:
    """Start up to `extra` OS threads running `entry(arg)`, their handles
    written into `handles` (length at least `extra`); returns how many
    started. A thread that cannot be started stops the starts, it is not an
    error. The caller keeps `handles` alive until `native_unit_threads_join`."""
    var attr = List[Int64](length=_ATTR_WORDS, fill=Int64(0))
    var started = 0
    if extra > 0 and external_call["pthread_attr_init", c_int](Int(attr.unsafe_ptr())) == 0:
        _ = external_call["pthread_attr_setstacksize", c_int](Int(attr.unsafe_ptr()), c_size_t(_UNIT_THREAD_STACK_BYTES))
        for i in range(extra):  # driver-loop: over unit threads (at most one per core)
            var slot = Int(handles.unsafe_ptr()) + i * 8
            if external_call["pthread_create", c_int](slot, Int(attr.unsafe_ptr()), entry, arg) != 0:
                break
            started += 1
        _ = external_call["pthread_attr_destroy", c_int](Int(attr.unsafe_ptr()))
    _ = attr^
    return started


def native_unit_threads_join(handles: List[Int], started: Int):
    """Join the first `started` threads of `handles` (the caller has released
    the GIL: the workers need it to finish their units)."""
    for i in range(started):  # driver-loop: over started unit threads
        _ = external_call["pthread_join", c_int](handles[i], Int(0))
