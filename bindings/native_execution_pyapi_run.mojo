# SPDX-License-Identifier: Apache-2.0
"""The typed native execution lease (RUN): the admitted entry, its call counters, session and
reservation words, and the counted call and finish used by the GIL-releasing endpoint in
`providers/python/native_execution_pyapi_door.mojo`."""
from core.native_descriptor_abi import NativeDescriptorCall
from core.native_descriptor_call import native_descriptor_call_spans
from core.native_call_output import NativeCallWorkspace
from core.operation_lock import operation_lock_release


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
