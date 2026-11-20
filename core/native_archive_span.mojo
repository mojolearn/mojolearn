# SPDX-License-Identifier: Apache-2.0
"""Checked borrowed archive bytes. The caller retains the owning allocation."""


@fieldwise_init
struct ArchiveSpan(Copyable, Movable):
    var address: Int
    var size: Int

    def byte(self, at: Int) raises -> Int:
        if at < 0 or at >= self.size:
            raise Error("truncated checkpoint record")
        return Int(MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=self.address)[at])

    def uint(self, at: Int, width: Int) raises -> Int:
        if at < 0 or width < 1 or width > 8 or at > self.size - width:
            raise Error("truncated checkpoint record")
        var bytes = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=self.address)
        var value = UInt64(0)
        for k in range(width):
            value |= UInt64(bytes[at + k]) << UInt64(8 * k)
        if value > 0x7FFFFFFFFFFFFFFF:
            raise Error("checkpoint size exceeds the supported address range")
        return Int(value)

    def slice(self, at: Int, size: Int) raises -> ArchiveSpan:
        if at < 0 or size < 0 or at > self.size or size > self.size - at:
            raise Error("truncated checkpoint payload")
        return ArchiveSpan(self.address + at, size)

    def equal(self, other: ArchiveSpan) -> Bool:
        if self.size != other.size:
            return False
        var lhs = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=self.address)
        var rhs = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=other.address)
        for k in range(self.size):
            if lhs[k] != rhs[k]:
                return False
        return True
