# SPDX-License-Identifier: Apache-2.0
# Once-admitted PEP3118 storage layout. Metadata is native and reusable;
# transfers occur at the caller's original execution position so aliases see
# intervening writes. Byte copies never convert/reassociate numeric values.
from std.memory import memcpy


struct BorrowedBufferLayout(Copyable, Movable):
    var address: Int
    var itemsize: Int
    var count: Int
    var readonly: Bool
    var shape: List[Int]
    var strides: List[Int]
    var suboffsets: List[Int]

    def __init__(out self, address: Int, itemsize: Int, readonly: Bool,
                 var shape: List[Int], var strides: List[Int], var suboffsets: List[Int]) raises:
        if itemsize < 1 or len(shape) != len(strides) or len(shape) != len(suboffsets):
            raise Error("native buffer: invalid admitted storage layout")
        var count = 1
        for axis in range(len(shape)):  # small-loop(admitted buffer rank)
            if shape[axis] < 0 or (shape[axis] != 0 and count > 9223372036854775807 // shape[axis]):
                raise Error("native buffer: invalid shape or element-count overflow")
            count *= shape[axis]
        if count > 9223372036854775807 // itemsize or (count != 0 and address == 0):
            raise Error("native buffer: null storage or byte-count overflow")
        self.address = address
        self.itemsize = itemsize
        self.count = count
        self.readonly = readonly
        self.shape = shape^
        self.strides = strides^
        self.suboffsets = suboffsets^

    def item_address(self, coordinates: List[Int]) -> Int:
        var address = self.address
        for axis in range(len(self.shape)):  # small-loop(native direct/indirect buffer coordinate words)
            address += coordinates[axis] * self.strides[axis]
            if self.suboffsets[axis] >= 0:
                address = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=address)[0] + self.suboffsets[axis]
        return address

    def advance(self, mut coordinates: List[Int]):
        for reverse in range(len(self.shape)):  # small-loop(native buffer coordinate carry)
            var axis = len(self.shape) - reverse - 1
            coordinates[axis] += 1
            if coordinates[axis] < self.shape[axis]:
                break
            coordinates[axis] = 0

    def contiguous(self) -> Bool:
        var stride = self.itemsize
        for reverse in range(len(self.shape)):  # small-loop(admitted layout rank)
            var axis = len(self.shape) - reverse - 1
            if self.suboffsets[axis] >= 0 or (self.shape[axis] > 1 and self.strides[axis] != stride):
                return False
            stride *= self.shape[axis]
        return True

    def copy_to(self, destination: Int) raises:
        if self.count != 0 and destination == 0:
            raise Error("native buffer: null copy destination")
        if self.count == 0:
            return
        if self.contiguous():
            memcpy(dest=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=destination),
                src=Pointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=self.address), count=self.count * self.itemsize)
            return
        var coordinates = List[Int](length=len(self.shape), fill=0)
        for i in range(self.count):  # cpu-route: byte-preserving preparation of a borrowed host buffer
            memcpy(dest=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=destination + i * self.itemsize),
                src=Pointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=self.item_address(coordinates)), count=self.itemsize)
            self.advance(coordinates)

    def copy_from(self, source: Int) raises:
        if self.readonly:
            raise Error("TypeError: native buffer: writeback requires writable storage")
        if self.count != 0 and source == 0:
            raise Error("native buffer: null writeback source")
        if self.count == 0:
            return
        if self.contiguous():
            memcpy(dest=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=self.address),
                src=Pointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=source), count=self.count * self.itemsize)
            return
        var coordinates = List[Int](length=len(self.shape), fill=0)
        for i in range(self.count):  # cpu-route: byte-preserving publication into the admitted host buffer
            memcpy(dest=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=self.item_address(coordinates)),
                src=Pointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=source + i * self.itemsize), count=self.itemsize)
            self.advance(coordinates)
