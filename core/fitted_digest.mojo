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

PARALLEL PLAN. The independent units are (1) the members that need their
bytes linearized to C order and (2) every chunk of every member; both run
across the host pool (`host_parallelize`: the worker count is the pool's,
the memory per worker is one SHA-256 state, the chunks are hashed in place).
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


def fitted_digest_hex(class_name: String, mut members: List[NativeArchiveMember]) -> String:
    """The digest of the fields `members` of a model of class `class_name`:
    64 lowercase hex digits. `members` are hashed in the order given (the
    caller sorts them by name); each is linearized to C order here."""
    var n = len(members)
    var mp = members.unsafe_ptr()

    def linearize(unit: Int) {imm mp}:
        mp[unit].prepare()

    host_parallelize(linearize, n)
    # The chunk units of every member, in member order.
    var starts = List[Int]()
    var addresses = List[Int]()
    var counts = List[Int]()
    for i in range(n):  # driver-loop: over the model's fitted fields (chunk plan)
        starts.append(len(addresses))
        var at = 0
        while at < members[i].size:  # driver-loop: over one field's chunks (plan only, no bytes read)
            var size = min(members[i].size - at, FITTED_DIGEST_CHUNK)
            addresses.append(members[i].source + at)
            counts.append(size)
            at += size
    starts.append(len(addresses))
    var units = len(addresses)
    var digests = List[UInt8](length=32 * units, fill=0)
    var ap = addresses.unsafe_ptr()
    var cp = counts.unsafe_ptr()
    var dp = digests.unsafe_ptr()

    def hash_chunk(unit: Int) {imm ap, imm cp, imm dp}:
        var hasher = Sha256()
        hasher.update(ap[unit], cp[unit])
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
    _ = starts
    _ = members
    return result
