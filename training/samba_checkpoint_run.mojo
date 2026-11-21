# SPDX-License-Identifier: Apache-2.0
"""RUN half of the Samba v2 checkpoint stream (lane rs-neural, wave 8 run
split, from bindings/samba_checkpoint_native.mojo): the array names, the
hash constructor and the bounded libc write that hashes each chunk as it is
written. Typed Mojo only, no Python import (tools/python_free_dirs.sh). Same
bodies, so the file bytes and digests are unchanged."""
from std.ffi import external_call
from core.sha256 import Sha256


comptime SAMBA_TRANSFER_CHUNK = 1048576


def _checkpoint_names() -> List[String]:
    return [String("parameters"), String("exp_avg"), String("exp_avg_sq"), String("buf_initialized")]


def _checkpoint_hash() -> Sha256:
    return Sha256()


def _checkpoint_write(fd: Int, address: Int, count: Int, mut written: Sha256, mut whole: Sha256) raises:
    """Write `count` bytes at `address` to `fd` in bounded chunks with the
    libc `write` (non-variadic), hashing each chunk into `written` and
    `whole` from the same bytes as it is written (row N168: formerly a slice
    object, a Python `stream.write` and two `array('B')` copies per MiB)."""
    var start = 0
    while start < count:  # driver-loop: over bounded file write chunks
        var size = min(SAMBA_TRANSFER_CHUNK, count - start)
        var done = 0
        while done < size:  # driver-loop: over partial writes of one chunk
            # The std declares libc `write` as (Int32 fd, byte pointer, Int count)
            # -> Int under Mojo 1.2 (probed 2026-11-11, r6 build: any other
            # lowering is "existing function with conflicting signature"; one C
            # symbol admits one signature per binary).
            var put = external_call["write", Int](Int32(fd), UnsafePointer[UInt8, MutUntrackedOrigin](unsafe_from_address=address + start + done), size - done)
            if put <= 0:
                raise Error("OSError: Samba checkpoint short write")
            done += put
        written.update(address + start, size)
        whole.update(address + start, size)
        start += size


# --------------------------------------------------------------------------
# RUN of the checkpoint doors (bindings/samba_checkpoint_native_unpack.mojo
# and _pack.mojo, door split 2026-11-21): the bounded libc read that hashes
# each chunk as it arrives, and the stream's checks in the doors' words.
#
# The four arrays of one file are written and read in file order through one
# descriptor (a stream), and the whole-file digest is one running hash over
# them: serial by nature.


def _checkpoint_read(fd: Int, address: Int, count: Int, mut actual: Sha256) raises:
    """Read `count` bytes from `fd` (at its current position) into `address`
    in bounded chunks with the libc `read`, hashing each chunk into `actual`
    from the bytes read (formerly, per MiB: two slice objects, a Python
    `readinto` and a hashed copy). The mirror of `_checkpoint_write`."""
    var start = 0
    while start < count:  # driver-loop: over bounded file read chunks
        var size = min(SAMBA_TRANSFER_CHUNK, count - start)
        var done = 0
        while done < size:  # driver-loop: over partial reads of one chunk
            # The lowering of libc `read` proven in
            # checks/native_tests/transport_frame_suite.mojo.
            var got = external_call["read", Int](Int32(fd), UnsafePointer[UInt8, MutUntrackedOrigin](unsafe_from_address=address + start + done), size - done)
            if got < 1:
                raise Error("Samba checkpoint truncated array")
            done += got
        actual.update(address + start, size)
        start += size


def samba_checkpoint_header_check(schema: String, profile: String) raises:
    if schema != "mojolearn.samba-stack-state.v1" or profile != "mojolearn.samba-stack.fp32.v1":
        raise Error("Samba checkpoint state schema mismatch")


def samba_checkpoint_total_check(total: Int) raises:
    if total < 1 or total > 2147483647:
        raise Error("Samba checkpoint registry exceeds signed 32-bit indexing")


def samba_checkpoint_mode_check(mode: String) raises:
    if mode != "identical" and mode != "fast":
        raise Error("Samba checkpoint numeric mode mismatch")


def samba_checkpoint_array_count(index: Int, tensors: Int, total: Int) -> Int:
    """The element count of stored array `index`: the three float32 vectors
    hold the registry's parameter count, the int32 flags one per tensor."""
    return tensors if index == 3 else total


def samba_checkpoint_array_dtype(index: Int) -> String:
    return String("<i4") if index == 3 else String("<f4")


def samba_checkpoint_descriptor_check(index: Int, name: String, dtype: String, nbytes: Int, count: Int) raises:
    if name != _checkpoint_names()[index] or dtype != samba_checkpoint_array_dtype(index) or nbytes != count * 4:
        raise Error("Samba checkpoint array descriptor mismatch")


def samba_checkpoint_digest_check(digest: String) raises:
    """Every byte of an admitted 64-character digest is lowercase hex."""
    var bytes = digest.as_bytes()
    for j in range(len(bytes)):  # driver-loop: over digest metadata characters
        var c = Int(bytes[j])
        if not ((c >= 48 and c <= 57) or (c >= 97 and c <= 102)):
            raise Error("Samba checkpoint digest is not lowercase SHA256 hex")


def samba_checkpoint_header_size_check(size: Int, limit: Int) raises:
    if size > limit:
        raise Error("Samba checkpoint metadata exceeds the header limit")


def samba_checkpoint_unchanged_check(written: String, recorded: String) raises:
    if written != recorded:
        raise Error("Samba checkpoint state changed during save")


def samba_checkpoint_array_check(actual: String, recorded: String, name: String) raises:
    if actual != recorded:
        raise Error("Samba checkpoint array integrity mismatch: " + name)
