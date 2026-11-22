# SPDX-License-Identifier: Apache-2.0
"""The digest of a fitted model: one SHA-256 over its fitted fields (RUN, typed).

`fitted_digest_hex(class_name, members)` is the whole of `model.digest_` below
the boundary. The members are the described fitted fields the saved-model
writer already builds (`NativeArchiveMember`: name, dtype, shape, bytes), in
name order; nothing here reads a Python object.

THE FORMAT (`mojolearn.digest.v1`; every count is 8 bytes little-endian,
every text is its UTF-8 byte count then its bytes):

    top = SHA-256(
        text(format word) text(class name) count(fields)
        for each field, in the order given:
            text(name) text(dtype) count(rank) count(dimension)... count(bytes)
            SHA-256(chunk 0) SHA-256(chunk 1) ...        32 raw bytes each
    )

A field's bytes are its elements in C order, cut into chunks of
FITTED_DIGEST_CHUNK bytes (the last one shorter; no chunk for an empty
field). The chunk size is a constant of the format, not of the machine: the
same fields give the same digest on any core count, device or vendor.

PARALLEL PLAN. The independent units are the chunks of every member, run
across the host pool (`host_parallelize`: the worker count is the pool's).
A C-contiguous member's chunk is hashed in place (one SHA-256 state per
worker). A strided member (Fortran order, a sliced view, negative strides,
a device field downloaded column-major) is never copied whole: each of its
chunk units gathers its own byte range of the C-order stream into a chunk
buffer of at most FITTED_DIGEST_CHUNK bytes and hashes that, so the extra
memory is one chunk per worker in flight, and the gather and the hash of
different chunks run on different cores.
The top chain is serial by nature (SHA-256 chains each block on the one
before) and is fed only the field headers and 32 bytes per chunk, so it is
a few kilobytes for a model of gigabytes.

cpu-route: an integrity digest over host bytes, outside any GPU fit,
transform or predict; it runs only when `digest_` is read.
"""
from core.sha256 import Sha256
from core.host_parallel import host_parallelize
from core.native_archive_write import NativeArchiveMember, archive_uint, archive_append, archive_text_bytes

#: The version word of the digest: a change of the format changes it.
comptime FITTED_DIGEST_FORMAT = "mojolearn.digest.v1"
#: Bytes per independently hashed chunk of one field (4 MiB). Part of the format.
comptime FITTED_DIGEST_CHUNK = 4194304


def fitted_digest_text(mut dst: List[UInt8], text: String):
    """`text` as the format writes it: its UTF-8 byte count, then its bytes."""
    var raw = archive_text_bytes(text)
    archive_uint(dst, UInt64(len(raw)), 8)
    archive_append(dst, raw)


def fitted_digest_strided(member: NativeArchiveMember) -> Bool:
    """True when `member`'s bytes are not its elements in C order at
    `source` (the contiguity rule of `NativeArchiveMember.prepare`)."""
    var step = member.width
    for axis in range(len(member.shape) - 1, -1, -1):  # driver-loop: over the field's rank (layout metadata)
        if member.shape[axis] > 1 and member.strides[axis] != step:
            return True
        step *= member.shape[axis]
    return False


@always_inline
def _digest_gather(member: NativeArchiveMember, at: Int, count: Int, dst: Int):
    """Bytes `[at, at + count)` of `member`'s C-order stream into `dst`,
    read through its shape and byte strides (any sign). The element offset
    is stepped like an odometer, one add per element."""
    var into = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=dst)
    var width = member.width
    var rank = len(member.shape)
    var element = at // width
    var index = List[Int](length=rank, fill=0)
    var rest = element
    var offset = 0
    for axis in range(rank - 1, -1, -1):
        index[axis] = rest % member.shape[axis]
        rest //= member.shape[axis]
        offset += index[axis] * member.strides[axis]
    var position = element * width
    var end = at + count
    while position < end:
        var src = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=member.source + offset)
        var low = max(at - position, 0)
        var high = min(end - position, width)
        for byte in range(low, high):
            into[position + byte - at] = src[byte]
        position += width
        var axis = rank - 1
        while axis >= 0:
            index[axis] += 1
            offset += member.strides[axis]
            if index[axis] < member.shape[axis]:
                break
            offset -= member.shape[axis] * member.strides[axis]
            index[axis] = 0
            axis -= 1


def fitted_digest_hex(class_name: String, members: List[NativeArchiveMember]) -> String:
    """The digest of the fields `members` of a model of class `class_name`:
    64 lowercase hex digits. `members` are hashed in the order given (the
    caller sorts them by name); a strided member is read in C order chunk
    by chunk (`_digest_gather`), never copied whole and never modified."""
    var n = len(members)
    var mp = members.unsafe_ptr()
    # The chunk units of every member, in member order. A contiguous unit
    # holds its address; a strided unit holds its member (`owners`, -1 for
    # contiguous) and its byte offset in the member's C-order stream.
    var starts = List[Int]()
    var addresses = List[Int]()
    var counts = List[Int]()
    var owners = List[Int]()
    for i in range(n):  # driver-loop: over the model's fitted fields (chunk plan)
        starts.append(len(addresses))
        var strided = fitted_digest_strided(members[i])
        var at = 0
        while at < members[i].size:  # driver-loop: over one field's chunks (plan only, no bytes read)
            var size = min(members[i].size - at, FITTED_DIGEST_CHUNK)
            addresses.append(at if strided else members[i].source + at)
            counts.append(size)
            owners.append(i if strided else -1)
            at += size
    starts.append(len(addresses))
    var units = len(addresses)
    var digests = List[UInt8](length=32 * units, fill=0)
    var ap = addresses.unsafe_ptr()
    var cp = counts.unsafe_ptr()
    var op = owners.unsafe_ptr()
    var dp = digests.unsafe_ptr()

    def hash_chunk(unit: Int) {imm ap, imm cp, imm op, imm dp, imm mp}:
        var hasher = Sha256()
        if op[unit] < 0:
            hasher.update(ap[unit], cp[unit])
        else:
            # one chunk buffer per unit in flight, at most FITTED_DIGEST_CHUNK bytes
            var chunk = List[UInt8](length=cp[unit], fill=0)
            _digest_gather(mp[op[unit]], ap[unit], cp[unit], Int(chunk.unsafe_ptr()))
            hasher.update(Int(chunk.unsafe_ptr()), cp[unit])
            _ = chunk
        var words = hasher.digest_words()
        for w in range(8):
            for k in range(4):
                dp[unit * 32 + w * 4 + k] = UInt8((words[w] >> UInt32(24 - 8 * k)) & 255)

    host_parallelize(hash_chunk, units)
    # The top chain: serial by nature, over headers and chunk digests only.
    var top = Sha256()
    var head = List[UInt8]()
    fitted_digest_text(head, FITTED_DIGEST_FORMAT)
    fitted_digest_text(head, class_name)
    archive_uint(head, UInt64(n), 8)
    top.update_list(head)
    var base = Int(digests.unsafe_ptr())
    for i in range(n):  # driver-loop: over the model's fitted fields (headers and chunk digests)
        var record = List[UInt8]()
        fitted_digest_text(record, members[i].name)
        fitted_digest_text(record, members[i].dtype)
        archive_uint(record, UInt64(len(members[i].shape)), 8)
        for axis in range(len(members[i].shape)):
            archive_uint(record, UInt64(members[i].shape[axis]), 8)
        archive_uint(record, UInt64(members[i].size), 8)
        top.update_list(record)
        var held = starts[i + 1] - starts[i]
        if held > 0:
            top.update(base + 32 * starts[i], 32 * held)
    var result = top.hexdigest()
    # The chunk addresses point into the members: keep them alive to here.
    _ = digests
    _ = addresses
    _ = counts
    _ = owners
    _ = starts
    _ = members
    return result
