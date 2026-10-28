# SPDX-License-Identifier: Apache-2.0
# The common lifecycle boundary for native orchestration. Admission pins the
# provider/image, session and native lock once. Invocation reads only native
# lease fields: no Python lookup, descriptor hook or callable dispatch occurs.
# The outer engine enters with the GIL held. Each numerical call/native lock
# wait releases it; raw session busy reservations remain serialized with the
# original public bindings. Lock order is the engine's per-layer order.
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_call, native_descriptor_words, native_descriptor_report
from core.native_execution import NativeExecutionEndpoint
from core.operation_lock import operation_lock_acquire, operation_lock_release
from providers.python.neural_descriptor_pyapi import NeuralDescriptorProvider


struct NativeExecutionLease(Copyable, Movable):
    var entry: NativeDescriptorCall
    var calls: Int
    var count: Int
    var session: Int
    var reserve: Int
    var release: Int
    var lock: Int
    var release_gil: Bool

    def __init__(out self, entry: NativeDescriptorCall, calls: Int, count: Int,
                 session: Int, reserve: Int, release: Int, lock: Int, release_gil: Bool):
        self.entry = entry
        self.calls = calls
        self.count = count
        self.session = session
        self.reserve = reserve
        self.release = release
        self.lock = lock
        self.release_gil = release_gil

    def counted(self, operation: Int, a: List[Int], p: List[Int], f: List[Float64]) raises -> Int:
        if operation < 0 or operation >= self.count:
            raise Error("native execution: operation is outside the admitted table")
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=self.calls)[operation] += 1
        return native_descriptor_call(self.entry, operation, a, p, f)



def _native_execution_finish(lease: NativeExecutionLease, reserved: Bool, acquired: Bool) raises:
    try:
        if reserved:
            _ = lease.counted(lease.release, [], [lease.session], [])
    finally:
        if acquired:
            operation_lock_release(lease.lock)

def _native_execution_run(lease: NativeExecutionLease, operation: Int,
                          a: List[Int], p: List[Int], f: List[Float64]) raises -> Int:
    var acquired = False
    var reserved = False
    var result = 0
    try:
        if lease.lock != 0:
            with GILReleased(Python()):
                acquired = operation_lock_acquire(lease.lock)
        if lease.reserve >= 0:
            _ = lease.counted(lease.reserve, [], [lease.session], [])
            reserved = True
        var words = List[Int]()
        if lease.session != 0:
            words.append(lease.session)
        for value in p:  # small-loop(admitted operation scalar words)
            words.append(value)
        if operation < 0 or operation >= lease.count:
            raise Error("native execution: operation is outside the admitted table")
        # Diagnostic counters are modified before releasing the admission GIL.
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=lease.calls)[operation] += 1
        if lease.release_gil:
            with GILReleased(Python()):
                result = native_descriptor_call(lease.entry, operation, a, words, f)
        else:
            result = native_descriptor_call(lease.entry, operation, a, words, f)
    except problem:
        _native_execution_finish(lease, reserved, acquired)
        raise problem
    _native_execution_finish(lease, reserved, acquired)
    return result


def native_execution_invoke(context: Int, operation: Int, aa: Int, na: Int,
                            pp: Int, np: Int, ff: Int, nf: Int, result: Int,
                            error_context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        if context == 0 or result == 0 or nf < 0 or (nf != 0 and ff == 0):
            raise Error("native execution: invalid admitted context or scalar span")
        var lease = MutPointer[NativeExecutionLease, MutUntrackedOrigin](unsafe_from_address=context)
        var a = native_descriptor_words(aa, na)
        var p = native_descriptor_words(pp, np)
        var f = List[Float64]()
        var values = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=ff)
        for i in range(nf):  # small-loop(admitted operation float words)
            f.append(values[i])
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[0] = _native_execution_run(lease[], operation, a, p, f)
        return 0
    except problem:
        native_descriptor_report(problem, error_context, sink)
        return 1


struct AdmittedNativeExecution(Movable):
    var provider: PythonObject
    var session_owner: PythonObject
    var lock_owner: PythonObject
    var lease: List[NativeExecutionLease]

    def __init__(out self, provider: PythonObject, session_owner: PythonObject, session: Int,
                 reserve: Int, release: Int, lock_owner: PythonObject, lock: Int,
                 release_gil: Bool = True) raises:
        var owner = provider.downcast_value_ptr[NeuralDescriptorProvider]()
        if (reserve < 0) != (release < 0) or (reserve >= 0 and session == 0):
            raise Error("native execution: reservation needs an admitted session and both operations")
        if reserve >= len(owner[].calls) or release >= len(owner[].calls):
            raise Error("native execution: reservation is outside the admitted table")
        self.provider = provider
        self.session_owner = session_owner
        self.lock_owner = lock_owner
        self.lease = List[NativeExecutionLease]()
        self.lease.append(NativeExecutionLease(owner[].entry, Int(owner[].calls.unsafe_ptr()), len(owner[].calls),
            session, reserve, release, lock, release_gil))

    def endpoint(mut self) -> NativeExecutionEndpoint:
        # Inner List storage stays stable even when the admission owner moves
        # into a model's retained owner vector. The lease is never resized.
        return NativeExecutionEndpoint(native_execution_invoke, Int(self.lease.unsafe_ptr()))
