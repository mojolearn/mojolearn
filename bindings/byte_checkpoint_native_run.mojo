# SPDX-License-Identifier: Apache-2.0
"""Typed write and typed read of the streamed byte-LM archive (RUN, no Python).

Read (lane pools-frame, 2026-11-21): `byte_checkpoint_read_header` reads and
admits the header of the file `byte_checkpoint_write` writes and returns the
state's metadata text, the four element counts and the four array digests;
the caller allocates the four arrays; `byte_checkpoint_read_arrays` reads
them in place and checks each against its digest. The four arrays are
independent: each is read (`pread`, its own file range) and hashed on its own
host task. The header is read in the one canonical form the writers emit
(keys in code-point order, no white space); any other text is refused as a
schema mismatch.

`byte_checkpoint_write` writes the file `byte_checkpoint_save` of the boundary
door writes, from typed values: a path, the state's metadata as its JSON
object text, and the four arrays (parameters, m, v as float32; flags as
int32) by host address and element count. No Python object is read and the
GIL is not touched: the door that enters RUN releases it. The file is the
same bytes for the same state: magic, 8-byte little-endian header length,
the header (`{"arrays":[...],"schema":...,"state":<metadata>}`, keys in
code-point order), the header's SHA-256, the four arrays raw.

Parallel plan: the four array digests the header needs are independent and
are taken on four host tasks. The write itself is serial by nature: the file
is written in order through one descriptor and its digest is one SHA-256
chain over those bytes in that order.

File errors carry errno in `ByteCheckpointWriteFailure` so PACK can raise
the exact `OSError` subclass with its filename, as the door does.
"""
from std.ffi import c_int, external_call
from std.sys.info import CompilationTarget
from bindings.native_posix_transport_run import posix_errno, posix_clock
from core.host_parallel import host_parallelize
from core.sha256 import Sha256

comptime BYTE_CHECKPOINT_RUN_CHUNK = 1048576
comptime BYTE_CHECKPOINT_RUN_HEADER_LIMIT = 1048576
comptime _CheckpointBytes = UnsafePointer[UInt8, MutUntrackedOrigin]


struct ByteCheckpointWriteFailure(Movable):
    """The file call that failed (`number` is errno; 0 when no file call failed)."""
    var number: Int
    var operation: String
    var path: String

    def __init__(out self):
        self.number = 0
        self.operation = String("")
        self.path = String("")


def _checkpoint_fail(mut failure: ByteCheckpointWriteFailure, operation: String, path: String, number: Int) raises:
    failure.number = number
    failure.operation = operation
    failure.path = path
    raise Error("OSError: Byte-LM checkpoint " + operation + " failed (errno " + String(number) + "): " + path)


def _checkpoint_append(mut dst: List[UInt8], text: String):
    var view = text.as_bytes()
    for i in range(len(view)):  # small-loop(bytes of one short header token)
        dst.append(view[i])


def _checkpoint_hex(mut dst: List[UInt8], words: MutPointer[UInt32, MutUntrackedOrigin], at: Int):
    for w in range(8):  # small-loop(the eight digest words)
        var v = Int(words[at + w])
        for d in range(8):  # small-loop(hex digits of one word)
            var nib = (v >> (28 - 4 * d)) & 15
            dst.append(UInt8(48 + nib) if nib < 10 else UInt8(87 + nib))


def _checkpoint_put(fd: Int, address: Int, count: Int, mut failure: ByteCheckpointWriteFailure, path: String) raises:
    var done = 0
    while done < count:  # driver-loop: over partial writes of one chunk
        # libc `write` as the std declares it: (Int32 fd, byte pointer, Int count) -> Int.
        var put = external_call["write", Int](Int32(fd), _CheckpointBytes(unsafe_from_address=address + done), count - done)
        if put > 0:
            done += put
        elif put == 0:
            raise Error("Byte-LM checkpoint short write")
        else:
            var number = posix_errno()
            if number != 4:  # EINTR: the write is taken up again, as CPython does
                _checkpoint_fail(failure, "write", path, number)


def byte_checkpoint_write(var path: String, metadata_address: Int, metadata_bytes: Int,
                          addresses: List[Int], counts: List[Int],
                          mut failure: ByteCheckpointWriteFailure) raises -> String:
    """Write the archive to `path` (through a temporary beside it, then one
    rename) and return the SHA-256 of the whole file, 64 lowercase hex digits.

    `metadata_address`/`metadata_bytes`: the state's JSON object text without
    the four arrays (ASCII, as the transport header encoder writes it; its
    fields are the caller's and are not checked here). `addresses`/`counts`:
    the four arrays in the order parameters, m, v, flags; counts are elements
    of four bytes. The caller keeps every range alive and unchanged until this
    returns; a range that changes while it is written is refused."""
    if len(addresses) != 4 or len(counts) != 4:
        raise Error("Byte-LM checkpoint array registry mismatch")
    for i in range(4):  # small-loop(the four checkpoint arrays)
        if counts[i] < 0 or (counts[i] > 0 and addresses[i] == 0):
            raise Error("Byte-LM checkpoint arrays must be contiguous float32/int32 vectors")
    if path.byte_length() == 0 or path.find("\0") >= 0:
        raise Error("Byte-LM checkpoint path is empty or contains NUL")
    var metadata = _CheckpointBytes(unsafe_from_address=metadata_address)
    if metadata_bytes < 2 or metadata_bytes > BYTE_CHECKPOINT_RUN_HEADER_LIMIT or Int(metadata[0]) != 123 or Int(metadata[metadata_bytes - 1]) != 125:
        raise Error("Byte-LM checkpoint state metadata mismatch")

    # The four array digests, one host task each.
    var spans = List[Int](length=8, fill=0)
    for i in range(4):  # small-loop(the four checkpoint arrays)
        spans[2 * i] = addresses[i]
        spans[2 * i + 1] = counts[i] * 4
    var digests = List[UInt32](length=32, fill=UInt32(0))
    var span_ptr = rebind[MutPointer[Int, MutUntrackedOrigin]](spans.unsafe_ptr())
    var digest_ptr = rebind[MutPointer[UInt32, MutUntrackedOrigin]](digests.unsafe_ptr())

    def _digest(unit: Int) {imm span_ptr, imm digest_ptr}:
        var hasher = Sha256()
        hasher.update(span_ptr[2 * unit], span_ptr[2 * unit + 1])
        var words = hasher.digest_words()
        for w in range(8):  # small-loop(the eight digest words)
            digest_ptr.unsafe_store(8 * unit + w, words[w])

    host_parallelize(_digest, 4)

    var header = List[UInt8]()
    _checkpoint_append(header, "{\"arrays\":[")
    for i in range(4):  # small-loop(the four checkpoint arrays)
        if i != 0:
            header.append(UInt8(44))
        _checkpoint_append(header, "{\"dtype\":\"")
        _checkpoint_append(header, "<i4" if i == 3 else "<f4")
        _checkpoint_append(header, "\",\"name\":\"")
        _checkpoint_append(header, "parameters" if i == 0 else ("m" if i == 1 else ("v" if i == 2 else "flags")))
        _checkpoint_append(header, "\",\"nbytes\":" + String(counts[i] * 4) + ",\"sha256\":\"")
        _checkpoint_hex(header, digest_ptr, 8 * i)
        _checkpoint_append(header, "\",\"shape\":[" + String(counts[i]) + "]}")
    _checkpoint_append(header, "],\"schema\":\"mojolearn.byte-lm-stream.v1\",\"state\":")
    for i in range(metadata_bytes):  # small-loop(bytes of the state metadata text)
        var b = Int(metadata[i])
        if b < 32 or b > 126:
            raise Error("Byte-LM checkpoint state metadata mismatch")
        header.append(UInt8(b))
    header.append(UInt8(125))
    if len(header) > BYTE_CHECKPOINT_RUN_HEADER_LIMIT:
        raise Error("Byte-LM checkpoint metadata exceeds the header limit")

    var prefix = List[UInt8]()
    _checkpoint_append(prefix, "MOJOLEARN-BYTE-LM")
    prefix.append(UInt8(1))
    prefix.append(UInt8(10))
    for i in range(8):  # small-loop(the 8-byte header length word)
        prefix.append(UInt8((len(header) >> (8 * i)) & 255))
    for i in range(len(header)):  # small-loop(bytes of the header)
        prefix.append(header[i])
    var header_hash = Sha256()
    header_hash.update(Int(header.unsafe_ptr()), len(header))
    var header_words = header_hash.digest_words()
    for w in range(8):  # small-loop(the eight digest words, big-endian bytes)
        for b in range(4):  # small-loop(bytes of one digest word)
            prefix.append(UInt8((Int(header_words[w]) >> (24 - 8 * b)) & 255))

    # A fresh name beside the target, created exclusively with mode 0600.
    var cwd = -100  # AT_FDCWD
    var flags = 1 | 0x40 | 0x80 | 0x80000  # O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC
    comptime if CompilationTarget.is_macos():
        cwd = -2
        flags = 1 | 0x200 | 0x800 | 0x1000000
    var stamp = Int(posix_clock() * 1e9)
    var temporary = String("")
    var fd = -1
    for attempt in range(16):  # small-loop(temporary names until one is new)
        temporary = path + ".tmp-" + String(Int(external_call["getpid", c_int]())) + "-" + String(stamp + attempt)
        fd = Int(external_call["openat", c_int, num_fixed_args=3](c_int(cwd), temporary.as_c_string_slice().unsafe_ptr(), c_int(flags), c_int(384)))
        if fd >= 0:
            break
        var number = posix_errno()
        if number != 17 and number != 4:  # EEXIST: next name; EINTR: again
            _checkpoint_fail(failure, "open", temporary, number)
    if fd < 0:
        _checkpoint_fail(failure, "open", temporary, 17)

    var whole = Sha256()
    var opened = True
    try:
        _checkpoint_put(fd, Int(prefix.unsafe_ptr()), len(prefix), failure, temporary)
        whole.update(Int(prefix.unsafe_ptr()), len(prefix))
        for i in range(4):  # driver-loop: over the four checkpoint arrays
            var written = Sha256()
            var start = 0
            var total = counts[i] * 4
            while start < total:  # driver-loop: over bounded checkpoint I/O chunks, preserving raw bytes
                var size = min(BYTE_CHECKPOINT_RUN_CHUNK, total - start)
                _checkpoint_put(fd, addresses[i] + start, size, failure, temporary)
                written.update(addresses[i] + start, size)
                whole.update(addresses[i] + start, size)
                start += size
            var seen = written.digest_words()
            for w in range(8):  # small-loop(the eight digest words)
                if seen[w] != digests[8 * i + w]:
                    raise Error("Byte-LM checkpoint state changed during save")
        if external_call["fsync", c_int](c_int(fd)) != 0:
            _checkpoint_fail(failure, "fsync", temporary, posix_errno())
        opened = False
        if external_call["close", c_int](c_int(fd)) != 0:
            _checkpoint_fail(failure, "close", temporary, posix_errno())
        if external_call["renameat", c_int](c_int(cwd), temporary.as_c_string_slice().unsafe_ptr(), c_int(cwd), path.as_c_string_slice().unsafe_ptr()) != 0:
            _checkpoint_fail(failure, "replace", path, posix_errno())
    except error:
        if opened:
            _ = external_call["close", c_int](c_int(fd))
        _ = external_call["unlinkat", c_int](c_int(cwd), temporary.as_c_string_slice().unsafe_ptr(), c_int(0))
        raise error
    # Keep the byte lists the pointers above read alive to here.
    _ = len(spans)
    _ = len(digests)
    _ = len(prefix)
    _ = len(header)
    return whole.hexdigest()


struct ByteCheckpointArchive(Movable):
    """The admitted header of one archive: the state's metadata as its JSON
    object text, the element counts of parameters, m, v and flags, their
    SHA-256 digests (eight words each, 32 in all) and the file offset of the
    first array byte. `complete` says the file ends exactly with the last
    array (probed while the header was read, so the caller can refuse a
    truncated or extended file after it admitted the state and before it
    allocates anything)."""
    var metadata: String
    var counts: List[Int]
    var digests: List[UInt32]
    var data_offset: Int
    var complete: Bool

    def __init__(out self):
        self.metadata = String("")
        self.counts = List[Int]()
        self.digests = List[UInt32]()
        self.data_offset = 0
        self.complete = False


def _checkpoint_open_read(path: String, mut failure: ByteCheckpointWriteFailure) raises -> Int:
    if path.byte_length() == 0 or path.find("\0") >= 0:
        raise Error("Byte-LM checkpoint path is empty or contains NUL")
    var cwd = -100  # AT_FDCWD
    var flags = 0x80000  # O_RDONLY | O_CLOEXEC
    comptime if CompilationTarget.is_macos():
        cwd = -2
        flags = 0x1000000
    var fd = -1
    while fd < 0:  # driver-loop: an interrupted open is taken up again, as CPython does
        fd = Int(external_call["openat", c_int, num_fixed_args=3](c_int(cwd), path.as_c_string_slice().unsafe_ptr(), c_int(flags), c_int(0)))
        if fd < 0:
            var number = posix_errno()
            if number != 4:
                _checkpoint_fail(failure, "open", path, number)
    return fd


def _checkpoint_pread(fd: Int, address: Int, count: Int, offset: Int) -> Int:
    """Reads up to `count` bytes at file `offset` into `address`. Returns the
    bytes read (fewer than `count` only at the end of the file), or -errno."""
    var done = 0
    while done < count:  # driver-loop: over partial reads of one range
        var got = external_call["pread", Int](Int32(fd), _CheckpointBytes(unsafe_from_address=address + done), count - done, offset + done)
        if got > 0:
            done += got
        elif got == 0:
            return done
        else:
            var number = posix_errno()
            if number != 4:  # EINTR: the read is taken up again
                return -number
    return done


def _checkpoint_expect(text: List[UInt8], mut at: Int, literal: String) raises:
    var view = literal.as_bytes()
    if at + len(view) > len(text):
        raise Error("Byte-LM checkpoint stream schema mismatch")
    for i in range(len(view)):  # small-loop(bytes of one short header token)
        if text[at + i] != view[i]:
            raise Error("Byte-LM checkpoint stream schema mismatch")
    at += len(view)


def _checkpoint_number(text: List[UInt8], mut at: Int) raises -> Int:
    var value = 0
    var digits = 0
    while at < len(text) and Int(text[at]) >= 48 and Int(text[at]) <= 57:  # small-loop(decimal digits of one extent)
        if digits >= 18 or (digits == 1 and value == 0):
            raise Error("Byte-LM checkpoint array descriptor mismatch")
        value = value * 10 + Int(text[at]) - 48
        digits += 1
        at += 1
    if digits == 0:
        raise Error("Byte-LM checkpoint array descriptor mismatch")
    return value


def byte_checkpoint_read_header(path: String, mut failure: ByteCheckpointWriteFailure) raises -> ByteCheckpointArchive:
    """Read and admit the archive's header: the magic, the header length, the
    header against its SHA-256 and the header's canonical text. The state's
    fields are the caller's to admit (`metadata` is returned as written)."""
    var fd = _checkpoint_open_read(path, failure)
    var archive = ByteCheckpointArchive()
    try:
        var lead = List[UInt8](length=27, fill=UInt8(0))
        var got = _checkpoint_pread(fd, Int(lead.unsafe_ptr()), 27, 0)
        if got < 0:
            _checkpoint_fail(failure, "read", path, -got)
        var magic_text = String("MOJOLEARN-BYTE-LM")
        var magic = magic_text.as_bytes()
        var same = got >= 19 and Int(lead[17]) == 1 and Int(lead[18]) == 10
        for i in range(17):  # small-loop(the 17 magic letters)
            if lead[i] != magic[i]:
                same = False
        if not same:
            raise Error("Byte-LM checkpoint stream magic mismatch")
        if got != 27:
            raise Error("Byte-LM checkpoint truncated header length")
        var length = 0
        var oversized = False
        for i in range(8):  # small-loop(the 8-byte header length word)
            if i >= 4 and Int(lead[19 + i]) != 0:
                oversized = True
            elif i < 4:
                length |= Int(lead[19 + i]) << (8 * i)
        if oversized or length < 1 or length > BYTE_CHECKPOINT_RUN_HEADER_LIMIT:
            raise Error("Byte-LM checkpoint header length exceeds limit")
        var header = List[UInt8](length=length + 32, fill=UInt8(0))
        got = _checkpoint_pread(fd, Int(header.unsafe_ptr()), length + 32, 27)
        if got < 0:
            _checkpoint_fail(failure, "read", path, -got)
        var hasher = Sha256()
        hasher.update(Int(header.unsafe_ptr()), length)
        var words = hasher.digest_words()
        var intact = got == length + 32
        for w in range(8):  # small-loop(the eight digest words, big-endian bytes)
            for b in range(4):  # small-loop(bytes of one digest word)
                if Int(header[length + 4 * w + b]) != ((Int(words[w]) >> (24 - 8 * b)) & 255):
                    intact = False
        if not intact:
            raise Error("Byte-LM checkpoint header integrity mismatch")
        header.resize(length, UInt8(0))
        var at = 0
        _checkpoint_expect(header, at, "{\"arrays\":[")
        for i in range(4):  # small-loop(the four checkpoint arrays)
            if i != 0:
                _checkpoint_expect(header, at, ",")
            _checkpoint_expect(header, at, "{\"dtype\":\"")
            _checkpoint_expect(header, at, "<i4" if i == 3 else "<f4")
            _checkpoint_expect(header, at, "\",\"name\":\"")
            _checkpoint_expect(header, at, "parameters" if i == 0 else ("m" if i == 1 else ("v" if i == 2 else "flags")))
            _checkpoint_expect(header, at, "\",\"nbytes\":")
            var nbytes = _checkpoint_number(header, at)
            _checkpoint_expect(header, at, ",\"sha256\":\"")
            if at + 64 > len(header):
                raise Error("Byte-LM checkpoint array descriptor mismatch")
            for w in range(8):  # small-loop(the eight digest words)
                var word = 0
                for d in range(8):  # small-loop(hex digits of one word)
                    var ch = Int(header[at + 8 * w + d])
                    if ch >= 48 and ch <= 57:
                        word = (word << 4) | (ch - 48)
                    elif ch >= 97 and ch <= 102:
                        word = (word << 4) | (ch - 87)
                    else:
                        raise Error("Byte-LM checkpoint array descriptor mismatch")
                archive.digests.append(UInt32(word))
            at += 64
            _checkpoint_expect(header, at, "\",\"shape\":[")
            var count = _checkpoint_number(header, at)
            _checkpoint_expect(header, at, "]}")
            if nbytes != count * 4:
                raise Error("Byte-LM checkpoint array descriptor mismatch")
            archive.counts.append(count)
        _checkpoint_expect(header, at, "],\"schema\":\"mojolearn.byte-lm-stream.v1\",\"state\":")
        if at + 3 > len(header) or Int(header[at]) != 123 or Int(header[len(header) - 2]) != 125 or Int(header[len(header) - 1]) != 125:
            raise Error("Byte-LM checkpoint stream schema mismatch")
        var state = List[UInt8]()
        for i in range(at, len(header) - 1):  # small-loop(bytes of the state metadata text)
            if Int(header[i]) < 32 or Int(header[i]) > 126:
                raise Error("Byte-LM checkpoint state metadata mismatch")
            state.append(header[i])
        archive.metadata = String(from_utf8=Span(unsafe_ptr=Pointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=Int(state.unsafe_ptr())), length=len(state)))
        _ = len(state)
        archive.data_offset = 27 + length + 32
        _ = len(lead)
        # The extent: the last array byte exists and nothing follows it. Two
        # one-byte reads; counts no file could hold are not probed.
        var total = 0
        var bounded = True
        for i in range(4):  # small-loop(the four checkpoint arrays)
            if archive.counts[i] > (1 << 40):
                bounded = False
            else:
                total += archive.counts[i] * 4
        if bounded:
            var probe = List[UInt8](length=1, fill=UInt8(0))
            var last = 1
            if total > 0:
                last = _checkpoint_pread(fd, Int(probe.unsafe_ptr()), 1, archive.data_offset + total - 1)
            var beyond = _checkpoint_pread(fd, Int(probe.unsafe_ptr()), 1, archive.data_offset + total)
            _ = len(probe)
            if last < 0:
                _checkpoint_fail(failure, "read", path, -last)
            if beyond < 0:
                _checkpoint_fail(failure, "read", path, -beyond)
            archive.complete = last == 1 and beyond == 0
    except error:
        _ = external_call["close", c_int](c_int(fd))
        raise error
    if external_call["close", c_int](c_int(fd)) != 0:
        _checkpoint_fail(failure, "close", path, posix_errno())
    return archive^


def _checkpoint_read_unit(plan: MutPointer[Int, MutUntrackedOrigin], expected: MutPointer[UInt32, MutUntrackedOrigin], unit: Int) -> Int:
    """One array: its file range read into its storage in bounded chunks and
    hashed. `plan` holds [fd, then per array: address, bytes, file offset].
    Returns 0, 1 (the file ends inside the array), 2 (the bytes differ from
    the header's digest) or -errno."""
    var fd = plan[0]
    var address = plan[1 + 3 * unit]
    var total = plan[2 + 3 * unit]
    var offset = plan[3 + 3 * unit]
    var hasher = Sha256()
    var start = 0
    while start < total:  # driver-loop: over bounded checkpoint I/O chunks, preserving raw bytes
        var size = min(BYTE_CHECKPOINT_RUN_CHUNK, total - start)
        var got = _checkpoint_pread(fd, address + start, size, offset + start)
        if got < 0:
            return got
        if got != size:
            return 1
        hasher.update(address + start, size)
        start += size
    var seen = hasher.digest_words()
    for w in range(8):  # small-loop(the eight digest words)
        if seen[w] != expected[8 * unit + w]:
            return 2
    return 0


def byte_checkpoint_read_arrays(path: String, archive: ByteCheckpointArchive, addresses: List[Int],
                                mut failure: ByteCheckpointWriteFailure) raises:
    """Read the four arrays into the caller's storage (`addresses` in the
    order parameters, m, v, flags; `archive.counts[i]` elements of four bytes
    each, which the caller allocated) and check each against the header's
    digest. The file's extent was probed with the header
    (`archive.complete`); a file cut short since then is refused as a
    truncated array. One host task per array."""
    if len(addresses) != 4 or len(archive.counts) != 4 or len(archive.digests) != 32:
        raise Error("Byte-LM checkpoint array registry mismatch")
    var plan = List[Int](length=13, fill=0)
    var offset = archive.data_offset
    for i in range(4):  # small-loop(the four checkpoint arrays)
        if archive.counts[i] < 0 or (archive.counts[i] > 0 and addresses[i] == 0):
            raise Error("Byte-LM checkpoint arrays must be contiguous float32/int32 vectors")
        plan[1 + 3 * i] = addresses[i]
        plan[2 + 3 * i] = archive.counts[i] * 4
        plan[3 + 3 * i] = offset
        offset += archive.counts[i] * 4
    var fd = _checkpoint_open_read(path, failure)
    plan[0] = fd
    var status = List[Int](length=4, fill=0)
    var expected = archive.digests.copy()
    var plan_ptr = rebind[MutPointer[Int, MutUntrackedOrigin]](plan.unsafe_ptr())
    var status_ptr = rebind[MutPointer[Int, MutUntrackedOrigin]](status.unsafe_ptr())
    var expected_ptr = rebind[MutPointer[UInt32, MutUntrackedOrigin]](expected.unsafe_ptr())

    def _read(unit: Int) {imm plan_ptr, imm status_ptr, imm expected_ptr}:
        status_ptr.unsafe_store(unit, _checkpoint_read_unit(plan_ptr, expected_ptr, unit))

    host_parallelize(_read, 4)
    _ = external_call["close", c_int](c_int(fd))
    _ = len(plan)
    _ = len(expected)
    for i in range(4):  # small-loop(the four checkpoint arrays, refused in file order)
        if status[i] < 0:
            _checkpoint_fail(failure, "read", path, -status[i])
        if status[i] == 1:
            raise Error("Byte-LM checkpoint truncated array")
        if status[i] == 2:
            raise Error("Byte-LM checkpoint array integrity mismatch: " + String("parameters" if i == 0 else ("m" if i == 1 else ("v" if i == 2 else "flags"))))
