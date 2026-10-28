# SPDX-License-Identifier: Apache-2.0
# The common lifecycle boundary for native orchestration. Admission pins the
# provider/image, session and native lock once. Invocation reads only native
# lease fields: no Python lookup, descriptor hook or callable dispatch occurs.
# The outer engine enters with the GIL held. Each numerical call/native lock
# wait releases it; raw session busy reservations remain serialized with the
# original public bindings. Lock order is the engine's per-layer order.
from std.memory import memcpy
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_call_spans, native_descriptor_report
from core.native_call_output import NativeCallWorkspace, NativeCallWorkspaceLease
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
    var workspace: Int
    var session_word: Int

    def __init__(out self, entry: NativeDescriptorCall, calls: Int, count: Int,
                 session: Int, reserve: Int, release: Int, lock: Int, release_gil: Bool, workspace: Int, session_word: Int):
        self.entry = entry
        self.calls = calls
        self.count = count
        self.session = session
        self.reserve = reserve
        self.release = release
        self.lock = lock
        self.release_gil = release_gil
        self.workspace = workspace
        self.session_word = session_word

    def counted(self, operation: Int, mut scratch: NativeCallWorkspace) raises -> Int:
        if operation < 0 or operation >= self.count:
            raise Error("native execution: operation is outside the admitted table")
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=self.calls)[operation] += 1
        return native_descriptor_call_spans(self.entry, operation, 0, 0,
            self.session_word, 1, 0, 0, scratch.output)


def _native_execution_finish(lease: NativeExecutionLease, reserved: Bool, acquired: Bool,
                             mut scratch: NativeCallWorkspace) raises:
    try:
        if reserved:
            _ = lease.counted(lease.release, scratch)
    finally:
        if acquired:
            operation_lock_release(lease.lock)


def _native_execution_run(lease: NativeExecutionLease, operation: Int,
                          aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
                          mut scratch: NativeCallWorkspace) raises -> Int:
    var acquired = False
    var reserved = False
    var result = 0
    try:
        if lease.lock != 0:
            with GILReleased(Python()):
                acquired = operation_lock_acquire(lease.lock)
        if lease.reserve >= 0:
            _ = lease.counted(lease.reserve, scratch)
            reserved = True
        # Ordinary calls borrow their admitted metadata directly. Only session
        # calls need an ABI prefix; reuse the exclusive workspace for that one
        # copy, preserving argument order and the caller's original vector.
        var words = pp
        var word_count = np
        if lease.session != 0:
            scratch.words.resize(np + 1, 0)
            scratch.words[0] = lease.session
            if np > 0:
                memcpy(dest=scratch.words.unsafe_ptr() + 1,
                    src=MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=pp), count=np)
            words = Int(scratch.words.unsafe_ptr())
            word_count += 1
        if operation < 0 or operation >= lease.count:
            raise Error("native execution: operation is outside the admitted table")
        # Diagnostic counters are modified before releasing the admission GIL.
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=lease.calls)[operation] += 1
        if lease.release_gil:
            with GILReleased(Python()):
                result = native_descriptor_call_spans(lease.entry, operation, aa, na,
                    words, word_count, ff, nf, scratch.output)
        else:
            result = native_descriptor_call_spans(lease.entry, operation, aa, na,
                words, word_count, ff, nf, scratch.output)
    except problem:
        _native_execution_finish(lease, reserved, acquired, scratch)
        raise problem
    _native_execution_finish(lease, reserved, acquired, scratch)
    return result


def native_execution_invoke(context: Int, operation: Int, aa: Int, na: Int,
                            pp: Int, np: Int, ff: Int, nf: Int, result: Int,
                            error_context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        if context == 0 or result == 0 or nf < 0 or (nf != 0 and ff == 0):
            raise Error("native execution: invalid admitted context or scalar span")
        var lease = MutPointer[NativeExecutionLease, MutUntrackedOrigin](unsafe_from_address=context)
        # Preserve the old copying helper's admission and error text before
        # lock/reservation. The ABI already lends these spans through return.
        if na < 0 or (na > 0 and aa == 0):
            raise Error("native descriptor: null or invalid word span")
        if np < 0 or (np > 0 and pp == 0):
            raise Error("native descriptor: null or invalid word span")
        var loan = NativeCallWorkspaceLease(lease[].workspace)
        var scratch = loan.pointer()
        try:
            MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[0] = _native_execution_run(
                lease[], operation, aa, na, pp, np, ff, nf, scratch[])
        finally:
            loan.release()
        return 0
    except problem:
        native_descriptor_report(problem, error_context, sink)
        return 1


struct AdmittedNativeExecution(Movable):
    var provider: PythonObject
    var session_owner: PythonObject
    var lock_owner: PythonObject
    var lease: List[NativeExecutionLease]
    var call_workspace: List[NativeCallWorkspace]
    var endpoint_workspace: List[NativeCallWorkspace]
    var session_word: List[Int]

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
        # Each admission owns stable result/scratch storage. Moving this owner
        # keeps inner List allocations stable; none is resized after publishing
        # its address. Nonblocking claims protect reentry and concurrent calls.
        self.call_workspace = List[NativeCallWorkspace]()
        self.call_workspace.append(NativeCallWorkspace())
        self.endpoint_workspace = List[NativeCallWorkspace]()
        self.endpoint_workspace.append(NativeCallWorkspace())
        self.session_word = List[Int](length=1, fill=session)
        self.lease = List[NativeExecutionLease]()
        self.lease.append(NativeExecutionLease(owner[].entry, Int(owner[].calls.unsafe_ptr()), len(owner[].calls),
            session, reserve, release, lock, release_gil, Int(self.call_workspace.unsafe_ptr()), Int(self.session_word.unsafe_ptr())))

    def endpoint(mut self) -> NativeExecutionEndpoint:
        # Inner List storage stays stable even when the admission owner moves
        # into a model's retained owner vector. The lease is never resized.
        return NativeExecutionEndpoint(native_execution_invoke, Int(self.lease.unsafe_ptr()),
            Int(self.endpoint_workspace.unsafe_ptr()))
