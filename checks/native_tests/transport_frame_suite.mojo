# SPDX-License-Identifier: Apache-2.0
"""The typed transport frame and the typed byte-LM archive write.

Three independent fixtures, one host task each: the header text, the header
readers, one frame through a socket pair. The archive fixture runs after them
on the calling thread, because the write splits its own array digests across
host tasks and a host split is not entered from inside another.
"""
from std.ffi import c_int, external_call
from std.sys.info import CompilationTarget
from core.host_parallel import host_parallelize
from core.sha256 import Sha256
from core.transport_header import (
    TransportHeader, transport_header_has, transport_header_int, transport_header_bool,
    transport_header_text, transport_header_words, transport_header_payload_bytes,
)
from bindings.native_posix_transport_run import posix_close, posix_clock, posix_socket_prepare
from bindings.native_transport_run import transport_send_words, transport_receive_into
from bindings.byte_checkpoint_native_run import byte_checkpoint_write, ByteCheckpointWriteFailure

comptime TRANSPORT_FRAME_FIXTURES = 3


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def _same(bytes: List[UInt8], expected: String) -> Bool:
    var view = expected.as_bytes()
    if len(view) != len(bytes):
        return False
    for i in range(len(view)):  # small-loop(bytes of one short expected text)
        if view[i] != bytes[i]:
            return False
    return True


def _header_text() raises:
    var header = TransportHeader()
    header.put_int("step", 7)
    header.put_text("cmd", "apply")
    header.put_bool("chained", True)
    var ids: List[Int] = [3, 4, -5]
    header.put_words("shards", ids)
    header.put_null("total")
    var text = header.encode(12)
    require(_same(text, "{\"chained\":true,\"cmd\":\"apply\",\"nbytes\":12,\"shards\":[3,4,-5],\"step\":7,\"total\":null}"),
            "transport header: fields are not in key order with nbytes in place")
    var empty = TransportHeader()
    require(_same(empty.encode(0), "{\"nbytes\":0}"), "transport header: empty header text")
    var refused = False
    try:
        header.put_int("step", 8)
    except:
        refused = True
    require(refused, "transport header: a field written twice was accepted")
    refused = False
    try:
        header.put_int("nbytes", 1)
    except:
        refused = True
    require(refused, "transport header: a caller-written nbytes was accepted")


def _header_read() raises:
    var header = TransportHeader()
    header.put_text("name", "a\"b\\c\né😀")
    header.put_int("completed", -42)
    header.put_bool("first", False)
    var bits: List[Int] = [4607182418800017408, 0, -1]
    header.put_words("losses", bits)
    var none = List[Int]()
    header.put_words("empty", none)
    var text = header.encode(1048576)
    var at = Int(text.unsafe_ptr())
    var n = len(text)
    require(transport_header_payload_bytes(at, n) == 1048576, "transport header: nbytes read back")
    require(transport_header_int(at, n, "completed") == -42, "transport header: integer read back")
    require(not transport_header_bool(at, n, "first"), "transport header: boolean read back")
    require(transport_header_text(at, n, "name") == "a\"b\\c\né😀", "transport header: text read back")
    var words = transport_header_words(at, n, "losses")
    require(len(words) == 3 and words[0] == 4607182418800017408 and words[1] == 0 and words[2] == -1,
            "transport header: integer list read back")
    require(len(transport_header_words(at, n, "empty")) == 0, "transport header: empty list read back")
    require(not transport_header_has(at, n, "missing") and transport_header_has(at, n, "name"),
            "transport header: field presence")
    # A nested object's field is not the frame's.
    var nested = String("{\"a\":{\"nbytes\":5,\"s\":\"},\"},\"nbytes\":9}")
    var view = nested.as_bytes()
    var raw = List[UInt8]()
    for i in range(len(view)):  # small-loop(bytes of one short fixture text)
        raw.append(view[i])
    require(transport_header_payload_bytes(Int(raw.unsafe_ptr()), len(raw)) == 9, "transport header: nested nbytes was read")
    _ = len(raw)
    _ = len(text)


def _frame_pair() raises:
    var fds = Array[c_int, 2](fill=c_int(-1))
    require(external_call["socketpair", c_int](c_int(1), c_int(1), c_int(0), fds.unsafe_ptr()) == 0,
            "transport frame: socketpair failed")
    var left = Int(fds[0])
    var right = Int(fds[1])
    try:
        posix_socket_prepare(left)
        posix_socket_prepare(right)
        var payload = List[Float32](length=1024, fill=Float32(0.0))
        for i in range(1024):  # small-loop(fixture payload values)
            payload[i] = Float32(i) * Float32(0.5)
        var header = TransportHeader()
        header.put_text("cmd", "fold")
        header.put_int("step", 3)
        var text = header.encode(4096)
        transport_send_words(left, 5.0, Int(text.unsafe_ptr()), len(text), Int(payload.unsafe_ptr()), 4096)
        var got_header = List[UInt8](length=256, fill=UInt8(0))
        var got = List[Float32](length=1024, fill=Float32(-1.0))
        var sizes = transport_receive_into(right, 5.0, Int(got_header.unsafe_ptr()), 256, Int(got.unsafe_ptr()), 4096)
        require(sizes[0] == len(text) and sizes[1] == 4096, "transport frame: received sizes")
        require(transport_header_int(Int(got_header.unsafe_ptr()), sizes[0], "step") == 3
                and transport_header_text(Int(got_header.unsafe_ptr()), sizes[0], "cmd") == "fold",
                "transport frame: received header fields")
        for i in range(1024):  # small-loop(fixture payload values)
            require(got[i] == payload[i], "transport frame: received payload differs")
        # A header whose nbytes is not the payload size is refused before a byte is sent.
        var refused = False
        try:
            transport_send_words(left, 5.0, Int(text.unsafe_ptr()), len(text), Int(payload.unsafe_ptr()), 8)
        except:
            refused = True
        require(refused, "transport frame: a wrong nbytes was sent")
        # A payload above the caller's bound is refused with nothing of it read.
        var none = TransportHeader()
        var small = none.encode(16)
        transport_send_words(left, 5.0, Int(small.unsafe_ptr()), len(small), Int(payload.unsafe_ptr()), 16)
        refused = False
        try:
            _ = transport_receive_into(right, 5.0, Int(got_header.unsafe_ptr()), 256, Int(got.unsafe_ptr()), 8)
        except:
            refused = True
        require(refused, "transport frame: a payload above the bound was received")
        _ = len(payload)
        _ = len(text)
        _ = len(small)
        _ = len(got_header)
        _ = len(got)
    finally:
        posix_close(left)
        posix_close(right)


def _archive() raises:
    var parameters = List[Float32](length=300000, fill=Float32(0.0))
    var first = List[Float32](length=300000, fill=Float32(0.25))
    var second = List[Float32](length=300000, fill=Float32(0.5))
    var flags = List[Int32](length=7, fill=Int32(1))
    for i in range(300000):  # small-loop(fixture parameter values)
        parameters[i] = Float32(i % 977) * Float32(0.125)
    var state = String("{\"completed_steps\":2,\"next_batch_index\":2}")
    var state_view = state.as_bytes()
    var metadata = List[UInt8]()
    for i in range(len(state_view)):  # small-loop(bytes of one short fixture text)
        metadata.append(state_view[i])
    var addresses: List[Int] = [Int(parameters.unsafe_ptr()), Int(first.unsafe_ptr()), Int(second.unsafe_ptr()), Int(flags.unsafe_ptr())]
    var counts: List[Int] = [300000, 300000, 300000, 7]
    var path = String("/tmp/mojolearn-byte-checkpoint-check-") + String(Int(external_call["getpid", c_int]())) + "-" + String(Int(posix_clock() * 1e9))
    var failure = ByteCheckpointWriteFailure()
    var digest = byte_checkpoint_write(path, Int(metadata.unsafe_ptr()), len(metadata), addresses, counts, failure)
    require(failure.number == 0 and digest.byte_length() == 64, "byte checkpoint: write reported a file failure")
    var cwd = -100
    comptime if CompilationTarget.is_macos():
        cwd = -2
    var fd = Int(external_call["openat", c_int, num_fixed_args=3](c_int(cwd), path.as_c_string_slice().unsafe_ptr(), c_int(0), c_int(0)))
    require(fd >= 0, "byte checkpoint: the written file does not open")
    var whole = Sha256()
    var total = 0
    var head = List[UInt8]()
    try:
        var chunk = List[UInt8](length=1048576, fill=UInt8(0))
        while True:  # driver-loop: over bounded file read chunks
            var got = external_call["read", Int](Int32(fd), UnsafePointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(chunk.unsafe_ptr())), 1048576)
            require(got >= 0, "byte checkpoint: reading the written file failed")
            if got == 0:
                break
            if total == 0:
                for i in range(min(got, 27)):  # small-loop(the magic and the header length word)
                    head.append(chunk[i])
            whole.update(Int(chunk.unsafe_ptr()), got)
            total += got
        _ = len(chunk)
    finally:
        posix_close(fd)
        _ = external_call["unlinkat", c_int](c_int(cwd), path.as_c_string_slice().unsafe_ptr(), c_int(0))
    require(whole.hexdigest() == digest, "byte checkpoint: the file digest is not the returned one")
    require(len(head) == 27 and Int(head[0]) == 77 and Int(head[16]) == 77 and Int(head[17]) == 1 and Int(head[18]) == 10,
            "byte checkpoint: stream magic")
    var header_bytes = 0
    for i in range(8):  # small-loop(the 8-byte header length word)
        header_bytes |= Int(head[19 + i]) << (8 * i)
    require(total == 27 + header_bytes + 32 + 4 * (3 * 300000 + 7), "byte checkpoint: file size")
    # A missing directory is a file failure with its errno, and leaves nothing behind.
    var missing = ByteCheckpointWriteFailure()
    var refused = False
    try:
        _ = byte_checkpoint_write(String("/nonexistent-mojolearn-directory/state.bin"), Int(metadata.unsafe_ptr()), len(metadata), addresses, counts, missing)
    except:
        refused = True
    require(refused and missing.number == 2 and missing.operation == "open", "byte checkpoint: a missing directory was not reported as ENOENT")
    _ = len(parameters)
    _ = len(first)
    _ = len(second)
    _ = len(flags)
    _ = len(metadata)


def _fixture(which: Int) raises:
    if which == 0:
        _header_text()
    elif which == 1:
        _header_read()
    else:
        _frame_pair()


def run_checks() raises:
    var flags = List[UInt8](length=TRANSPORT_FRAME_FIXTURES, fill=UInt8(0))
    var flag_ptr = rebind[MutPointer[UInt8, MutUntrackedOrigin]](flags.unsafe_ptr())

    def _one(unit: Int) {imm flag_ptr}:
        try:
            _fixture(unit)
        except:
            flag_ptr.unsafe_store(unit, UInt8(1))

    host_parallelize(_one, TRANSPORT_FRAME_FIXTURES)
    for unit in range(TRANSPORT_FRAME_FIXTURES):  # driver-loop: over the fixtures' flags
        if flags[unit] != UInt8(0):
            # Raise the failing fixture's own words.
            _fixture(unit)
            raise Error("transport frame: fixture " + String(unit) + " failed on its host task and passed alone")
    _ = len(flags)
    _archive()
