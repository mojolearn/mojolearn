# SPDX-License-Identifier: Apache-2.0
"""Native descriptors and finite-verdict propagation for held input buffers."""
from core.host_finite_scan import host_all_finite_f32


struct HeldBufferDescriptor(Copyable, Movable):
    var identity: Int
    var parent: Int
    var address: Int
    var rows: Int
    var cols: Int
    var finite: Bool

    def __init__(out self, identity: Int, parent: Int, address: Int, rows: Int, cols: Int, finite: Bool):
        self.identity = identity
        self.parent = parent
        self.address = address
        self.rows = rows
        self.cols = cols
        self.finite = finite


struct HeldVerification(Movable):
    var finite: Bool
    var proved: List[Int]

    def __init__(out self, finite: Bool, var proved: List[Int]):
        self.finite = finite
        self.proved = proved^


def admitted_holdable(rank: Int, dtype: String, order: String, rows: Int, cols: Int) -> Bool:
    return rank == 2 and dtype == "<f4" and order == "C" and rows >= 1 and cols >= 1


def admitted_held_scan(address: Int, rows: Int, cols: Int) -> Bool:
    return host_all_finite_f32(MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=address), rows * cols)


def admitted_held_verify_native(buffers: List[HeldBufferDescriptor], index: Int) raises -> HeldVerification:
    if index < 0 or index >= len(buffers):
        raise Error("admitted input: invalid held-buffer snapshot")
    if buffers[index].finite:
        return HeldVerification(True, List[Int]())
    var chain: List[Int] = [index]
    var top = index
    while True:  # driver-loop: over held-buffer ancestor handles
        var parent = buffers[top].parent
        if not parent:
            break
        var found = -1
        for i in range(len(buffers)):  # driver-loop: resolve a native owner identity in this scope
            if buffers[i].identity == parent:
                found = i
                break
        if found < 0:
            break
        if buffers[found].finite:
            return HeldVerification(True, chain^)
        for i in range(len(chain)):  # driver-loop: refuse malformed ancestor cycles
            if chain[i] == found:
                raise Error("admitted input: cyclic held-buffer ancestry")
        chain.append(found)
        top = found
    if not admitted_held_scan(buffers[top].address, buffers[top].rows, buffers[top].cols):
        if top == index or not admitted_held_scan(buffers[index].address, buffers[index].rows, buffers[index].cols):
            return HeldVerification(False, List[Int]())
        return HeldVerification(True, [index])
    return HeldVerification(True, chain^)
