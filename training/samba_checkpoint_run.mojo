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
