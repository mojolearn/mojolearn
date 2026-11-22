# SPDX-License-Identifier: Apache-2.0
# Shared native operation endpoint. This transport adds an admitted lifetime
# context to the existing descriptor ABI; it never exposes Python objects.
from core.native_string import unsafe_string_from_utf8_cstr
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_error
from core.native_call_output import NativeCallWorkspaceLease
from core.native_descriptor_abi import NativeDescriptorCall
from core.native_descriptor_call import native_descriptor_call_spans
from core.native_call_output import NativeCallWorkspace
from core.operation_lock import operation_lock_release

comptime NativeExecutionInvoke = def(
    Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, NativeDescriptorError,
) thin abi("C") -> Int


struct NativeExecutionEndpoint(Copyable, Movable):
    var invoke: NativeExecutionInvoke
    var context: Int
    var workspace: Int

    def __init__(out self, invoke: NativeExecutionInvoke, context: Int, workspace: Int = 0):
        self.invoke = invoke
        self.context = context
        self.workspace = workspace

    def call(self, operation: Int, addresses: List[Int], ints: List[Int], floats: List[Float64]) raises -> Int:
        # The endpoint's admission owner must remain alive through this call.
        # Its context points to native storage, not an extension object layout.
        var loan = NativeCallWorkspaceLease(self.workspace)
        var scratch = loan.pointer()
        try:
            scratch[].output.result = 0
            scratch[].output.error.clear()
            var status = self.invoke(self.context, operation,
                Int(addresses.unsafe_ptr()), len(addresses), Int(ints.unsafe_ptr()), len(ints),
                Int(floats.unsafe_ptr()), len(floats), Int(MutPointer(to=scratch[].output.result)),
                Int(MutPointer(to=scratch[].output.error)), native_descriptor_error)
            if status != 0:
                scratch[].output.error.append(0)
                raise Error(unsafe_string_from_utf8_cstr(scratch[].output.error.unsafe_ptr()))
            return scratch[].output.result
        finally:
            # Keep the heap owner and exclusive claim through error conversion.
            loan.release()
            # The provider now borrows the original spans through its GIL
            # release window; pin even temporary call-site vectors until return.
            _ = addresses
            _ = ints
            _ = floats


# Moved verbatim from bindings/native_execution_pyapi_run.mojo (lane bindings-inline, 2026-11-22).
# The typed native execution lease (RUN): the admitted entry, its call counters, session and
# reservation words, and the counted call and finish used by the GIL-releasing endpoint in
# `providers/python/native_execution_pyapi_door.mojo`.

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
