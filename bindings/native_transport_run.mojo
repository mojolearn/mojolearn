# SPDX-License-Identifier: Apache-2.0
"""Typed transport helpers (RUN): the monotonic clock, the timeout check, the host name and
process termination, over the POSIX calls in `bindings/native_posix_transport_run.mojo`."""
from std.ffi import c_int, c_size_t, external_call
from std.math import isfinite
from core.native_string import unsafe_string_from_utf8_cstr
from bindings.native_posix_transport_run import posix_clock, posix_terminate, posix_errno, posix_failure


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
