# SPDX-License-Identifier: Apache-2.0
"""Typed helpers of the user-objects door (RUN): the refusal word format, index reads from an
admitted int32/int64 buffer and the float integrality test."""


def _refusal(kind: String, message: String) -> Error:
    # the door raises it as `kind` (TypeError, ValueError, IndexError, OverflowError)
    return Error("__refuse__" + kind + ":" + message)


def _index_at(address: Int, k: Int, width: Int) -> Int:
    if width == 4:
        return Int(MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=address)[k])
    return Int(MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=address)[k])


def _float_is_integer(x: Float64) -> Bool:
    # math.isfinite(x) and x.is_integer()
    if not (x - x == 0.0):
        return False
    if x >= 9007199254740992.0 or x <= -9007199254740992.0:
        return True  # every float this large is an integer
    return Float64(Int(x)) == x
