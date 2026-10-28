# SPDX-License-Identifier: Apache-2.0
# Caller-owned native ABI output. Place this value in a one-element List before
# exposing its field addresses: the foreign call needs stable writable storage,
# not an inline value whose address crosses the ABI as an integer.
from std.atomic import Atomic, Ordering


struct NativeCallOutput(Movable):
    var result: Int
    var error: List[UInt8]

    def __init__(out self):
        self.result = 0
        self.error = List[UInt8]()


struct NativeCallWorkspace(Movable):
    # Heap-own this value before exposing its output fields across the C ABI.
    # The nonblocking claim gives nested/concurrent calls private scratch rather
    # than sharing output/error storage or waiting while the GIL is held.
    var busy: Int64
    var output: NativeCallOutput
    var words: List[Int]

    def __init__(out self):
        self.busy = 0
        self.output = NativeCallOutput()
        self.words = List[Int]()


struct NativeCallWorkspaceLease(Movable):
    var retained: Int
    var handle: Int
    var temporary: List[NativeCallWorkspace]

    def __init__(out self, handle: Int = 0):
        self.retained = 0
        self.handle = 0
        self.temporary = List[NativeCallWorkspace]()
        if handle != 0:
            var slot = MutPointer[NativeCallWorkspace, MutUntrackedOrigin](unsafe_from_address=handle)
            var expected = Int64(0)
            if Atomic.compare_exchange[success_ordering=Ordering.SEQUENTIAL,
                                       failure_ordering=Ordering.RELAXED](
                MutPointer(to=slot[].busy), expected, Int64(1)):
                self.retained = handle
                self.handle = handle
        if self.handle == 0:
            self.temporary.append(NativeCallWorkspace())
            self.handle = Int(self.temporary.unsafe_ptr())

    def pointer(self) -> MutPointer[NativeCallWorkspace, MutUntrackedOrigin]:
        return MutPointer[NativeCallWorkspace, MutUntrackedOrigin](unsafe_from_address=self.handle)

    def release(mut self):
        if self.retained != 0:
            var slot = MutPointer[NativeCallWorkspace, MutUntrackedOrigin](unsafe_from_address=self.retained)
            Atomic.store[ordering=Ordering.RELEASE](MutPointer(to=slot[].busy), Int64(0))
            self.retained = 0

    def __del__(deinit self):
        self.release()
