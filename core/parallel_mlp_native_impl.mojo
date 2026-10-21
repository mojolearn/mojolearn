# SPDX-License-Identifier: Apache-2.0
"""Typed SmallMLPTrainer buffer checks moved from bindings/parallel_mlp_native.mojo."""
from std.math import isfinite


def mlp_check_finite(data: Pointer[Float32, ImmUntrackedOrigin], size: Int, label: String) raises:
    for i in range(size):
        if not isfinite(data[i]):
            raise Error("RuntimeError: SmallMLPTrainer nonfinite " + label)


def mlp_check_optimizer_state(
    v: Pointer[Float32, ImmUntrackedOrigin], flags: Pointer[Int32, ImmUntrackedOrigin],
) raises:
    for i in range(195):
        if v[i] < 0:
            raise Error("SmallMLPTrainer state requires nonnegative v")
    for i in range(4):
        if flags[i] < 0 or flags[i] > 1:
            raise Error("SmallMLPTrainer state requires binary flags")


def mlp_targets_into(
    addr: Int, item: Int, signed: Bool, rows: Int, ids: MutPointer[Int32, MutUntrackedOrigin],
) raises:
    """Native little-endian integer targets (`item` bytes each) to int32 class ids.
    The range check runs on the full-width value before conversion, so unsigned
    64-bit invalid targets get the class-range refusal instead of signed overflow."""
    var bytes = Pointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=addr)
    for i in range(rows):
        var raw = UInt64(0)
        for k in range(item):
            raw |= UInt64(bytes[i * item + k]) << UInt64(8 * k)
        var negative = signed and ((raw >> UInt64(8 * item - 1)) & UInt64(1)) == UInt64(1)
        if negative or raw >= UInt64(3):
            raise Error("SmallMLPTrainer targets require classes 0..2")
        ids[i] = Int32(Int(raw))
