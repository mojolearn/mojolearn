# SPDX-License-Identifier: Apache-2.0
"""The split descriptor's fold digest, written from int64 index buffers.

`split_descriptor` hashes `("folds", canonical_json(folds))` with 8-byte
little-endian length prefixes, where `folds` is the list of (train, test)
index lists. The canonical JSON of that value is `[[[a,b,...],[c,...]],...]`:
decimal integers, no spaces. This module writes exactly those bytes straight
from the int64 index buffers (no Python int lists, no per-index CPython call),
so the digest is byte-for-byte the one the JSON walk produced.

cpu-route: an integrity digest over host index bytes for the descriptor,
outside any GPU fit, transform or predict.
"""
from core.sha256 import Sha256

comptime _I64P = UnsafePointer[Int64, MutUntrackedOrigin]


def _put_i64(mut out: List[UInt8], value: Int64):
    """Append the decimal digits of `value` (Python `str(int)` spelling)."""
    var mag: UInt64
    if value < 0:
        out.append(UInt8(45))  # '-'
        mag = (-(value + 1)).cast[DType.uint64]() + 1
    else:
        mag = value.cast[DType.uint64]()
    var start = len(out)
    while True:  # the digits of one index, least significant first
        out.append(UInt8(48) + (mag % 10).cast[DType.uint8]())
        mag = mag // 10
        if mag == 0:
            break
    var i = start
    var j = len(out) - 1
    while i < j:  # reverse the digits in place
        var t = out[i]
        out[i] = out[j]
        out[j] = t
        i += 1
        j -= 1


def _put_indices(mut out: List[UInt8], address: Int, count: Int):
    """Append `[a,b,...]` for `count` int64 values at host `address`."""
    var p = _I64P(unsafe_from_address=address)
    out.append(UInt8(91))  # '['
    for i in range(count):  # the fold's indices
        if i:
            out.append(UInt8(44))  # ','
        _put_i64(out, p[i])
    out.append(UInt8(93))  # ']'


def _put_le8(mut hasher: Sha256, count: Int):
    var raw = List[UInt8](capacity=8)
    for i in range(8):  # the 8-byte little-endian length prefix
        raw.append(UInt8((count >> (8 * i)) & 0xFF))
    hasher.update_list(raw)


def msel_fold_digest_hex(
    train_addresses: List[Int], train_counts: List[Int], test_addresses: List[Int], test_counts: List[Int]
) -> String:
    """The hex digest of `("folds", canonical_json(folds))` with the 8-byte
    length prefixes. The caller keeps every index buffer alive."""
    var text = List[UInt8]()
    text.append(UInt8(91))  # '['
    for f in range(len(train_addresses)):  # over folds
        if f:
            text.append(UInt8(44))  # ','
        text.append(UInt8(91))  # '['
        _put_indices(text, train_addresses[f], train_counts[f])
        text.append(UInt8(44))  # ','
        _put_indices(text, test_addresses[f], test_counts[f])
        text.append(UInt8(93))  # ']'
    text.append(UInt8(93))  # ']'
    var tag = List[UInt8]()  # b"folds"
    tag.append(UInt8(102))
    tag.append(UInt8(111))
    tag.append(UInt8(108))
    tag.append(UInt8(100))
    tag.append(UInt8(115))
    var hasher = Sha256()
    _put_le8(hasher, len(tag))
    hasher.update_list(tag)
    _put_le8(hasher, len(text))
    hasher.update_list(text)
    return hasher.hexdigest()
