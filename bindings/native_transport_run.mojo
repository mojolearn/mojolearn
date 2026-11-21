# SPDX-License-Identifier: Apache-2.0
"""Typed transport helpers (RUN): the monotonic clock, the timeout check, the host name,
process termination and the typed frame (`transport_send_words`, `transport_receive_into`),
over the POSIX calls in `bindings/native_posix_transport_run.mojo`.

The typed frame works on a file descriptor and host addresses. It reads no Python object and
does not touch the GIL: the door that enters RUN releases it around the whole loop. The bytes
on the wire are the ones `transport_send` / `transport_receive` of the boundary door write
and read (4-byte big-endian header length, the JSON header with `nbytes`, the payload); the
header is built and read with `core/transport_header.mojo`."""
from std.ffi import c_int, c_size_t, external_call
from std.math import isfinite
from core.native_string import unsafe_string_from_utf8_cstr
from bindings.native_posix_transport_run import (
    PosixBytes, posix_clock, posix_terminate, posix_errno, posix_failure, posix_transfer_run,
)
from core.transport_header import TRANSPORT_HEADER_LIMIT, transport_header_payload_bytes


def transport_clock() raises -> Float64:
    return posix_clock()


def _transport_timeout(seconds: Float64) raises:
    if seconds <= 0 or not isfinite(seconds):
        raise Error("transport: timeout must be finite and positive")


def transport_hostname() raises -> String:
    var buffer = Array[Int8, 256](uninitialized=True)
    if external_call["gethostname", c_int](buffer.unsafe_ptr(), c_size_t(256)) != 0:
        posix_failure("gethostname", posix_errno())
    buffer[255] = 0
    return unsafe_string_from_utf8_cstr(buffer.unsafe_ptr())


def transport_terminate(pid: Int, signal: Int = 15) raises:
    if pid <= 0 or pid > 2147483647 or signal <= 0 or signal > 2147483647:
        raise Error("transport: invalid process ID or signal")
    posix_terminate(pid, signal)


def transport_send_words(fd: Int, timeout: Float64, header_address: Int, header_bytes: Int,
                         payload_address: Int, payload_bytes: Int) raises:
    """Send one frame. The header is the `header_bytes` bytes at `header_address`
    (`TransportHeader.encode(payload_bytes)`); the payload is read in place from
    `payload_address`. `timeout` is the socket's (seconds, or -1 for none) and
    bounds each of the three writes, as it does in the boundary door. The caller
    keeps both byte ranges alive and unmoved until this returns."""
    if fd < 0:
        raise Error("transport: socket is closed")
    if header_bytes < 2 or header_bytes > TRANSPORT_HEADER_LIMIT:
        raise Error("transport: oversized header")
    if payload_bytes < 0 or transport_header_payload_bytes(header_address, header_bytes) != payload_bytes:
        raise Error("transport: header nbytes differs from the payload size")
    var prefix = Array[UInt8, 4](fill=UInt8(0))
    for i in range(4):  # small-loop(the 4-byte frame length word)
        prefix[i] = UInt8((header_bytes >> (24 - i * 8)) & 255)
    posix_transfer_run(fd, PosixBytes(unsafe_from_address=Int(prefix.unsafe_ptr())), 4, timeout, True)
    posix_transfer_run(fd, PosixBytes(unsafe_from_address=header_address), header_bytes, timeout, True)
    if payload_bytes > 0:
        posix_transfer_run(fd, PosixBytes(unsafe_from_address=payload_address), payload_bytes, timeout, True)


def transport_receive_into(fd: Int, timeout: Float64, header_address: Int, header_capacity: Int,
                           payload_address: Int, payload_capacity: Int) raises -> Tuple[Int, Int]:
    """Receive one frame into the caller's two buffers and return (header bytes,
    payload bytes). A header longer than `header_capacity` or a payload longer
    than `payload_capacity` is refused before any of it is read, with the door's
    words; the connection is then out of step and the caller closes it."""
    if fd < 0:
        raise Error("transport: socket is closed")
    if header_capacity < 0 or payload_capacity < 0:
        raise Error("transport: receive requires a native socket and nonnegative bound")
    var prefix = Array[UInt8, 4](fill=UInt8(0))
    posix_transfer_run(fd, PosixBytes(unsafe_from_address=Int(prefix.unsafe_ptr())), 4, timeout, False)
    var count = 0
    for i in range(4):  # small-loop(the 4-byte frame length word)
        count = (count << 8) | Int(prefix[i])
    if count > TRANSPORT_HEADER_LIMIT or count > header_capacity:
        raise Error("transport: oversized header")
    posix_transfer_run(fd, PosixBytes(unsafe_from_address=header_address), count, timeout, False)
    var n = transport_header_payload_bytes(header_address, count)
    if n < 0 or n > payload_capacity:
        raise Error("transport: payload exceeds caller bound")
    if n > 0:
        posix_transfer_run(fd, PosixBytes(unsafe_from_address=payload_address), n, timeout, False)
    return (count, n)
