# SPDX-License-Identifier: Apache-2.0
# RUN part of bindings/native_posix_transport.mojo: typed Mojo only (no Python
# imports). Public libc ABI on the supported LP64 Linux and Darwin hosts; the
# calls that release the GIL around a blocking libc call (sleep, poll,
# getaddrinfo, posix_spawnp, waitpid) and their callers stay in the door file.
from std.ffi import c_int, c_uint, external_call
from std.sys.info import CompilationTarget

comptime PosixBytes = MutPointer[UInt8, MutUntrackedOrigin]
comptime PosixWords = MutPointer[Int, MutUntrackedOrigin]


def posix_errno() -> Int:
    comptime if CompilationTarget.is_macos():
        return Int(external_call["__error", MutPointer[c_int, MutUntrackedOrigin]]()[])
    else:
        return Int(external_call["__errno_location", MutPointer[c_int, MutUntrackedOrigin]]()[])


def posix_failure(operation: String, number: Int) raises:
    # CPython's low-level signal handler only marks pending application work.
    # Retrying EINTR here could swallow Ctrl-C forever; PyErr_CheckSignals
    # would call Python inside the product operation. Return native cancellation
    # instead, so cleanup runs and the interpreter resumes its handler after
    # the binding returns. Even non-raising handlers interrupt this operation.
    if number == 4:
        raise Error("InterruptedError: transport: " + operation + " interrupted by an OS signal")
    raise Error("transport: " + operation + " failed (errno " + String(number) + ")")


def posix_clock() raises -> Float64:
    var stamp = Array[Int64, 2](fill=Int64(0))
    var clock_id = 1  # CLOCK_MONOTONIC
    comptime if CompilationTarget.is_macos():
        clock_id = 6
    if external_call["clock_gettime", c_int](c_int(clock_id), stamp.unsafe_ptr()) != 0:
        posix_failure("clock_gettime", posix_errno())
    return Float64(stamp[0]) + Float64(stamp[1]) * 1e-9


def posix_close(fd: Int):
    if fd >= 0:
        # Never retry close after EINTR: Linux has already released the fd,
        # and retrying could close a descriptor reused by another thread.
        _ = external_call["close", c_int](c_int(fd))


def posix_again(number: Int) -> Bool:
    comptime if CompilationTarget.is_macos():
        return number == 35  # EAGAIN == EWOULDBLOCK
    else:
        return number == 11


def posix_deadline(timeout: Float64) raises -> Float64:
    return -1.0 if timeout < 0 else posix_clock() + timeout


def posix_socket_prepare(fd: Int) raises:
    # Preserve existing flags; all owned descriptors are nonblocking and
    # close-on-exec, including accepted connections before another spawn.
    var flags = external_call["fcntl", c_int, num_fixed_args=2](c_int(fd), c_int(1))
    if flags < 0 or external_call["fcntl", c_int, num_fixed_args=2](c_int(fd), c_int(2), c_int(flags | 1)) < 0:
        posix_failure("FD_CLOEXEC", posix_errno())
    var nonblock = 2048
    comptime if CompilationTarget.is_macos():
        nonblock = 4
    flags = external_call["fcntl", c_int, num_fixed_args=2](c_int(fd), c_int(3))
    if flags < 0 or external_call["fcntl", c_int, num_fixed_args=2](c_int(fd), c_int(4), flags | c_int(nonblock)) < 0:
        posix_failure("O_NONBLOCK", posix_errno())
    comptime if CompilationTarget.is_macos():
        # Darwin lacks MSG_NOSIGNAL. Set the socket's supported equivalent;
        # do not mutate the application's process-wide signal disposition.
        var enabled = Array[c_int, 1](fill=c_int(1))
        if external_call["setsockopt", c_int](c_int(fd), c_int(65535), c_int(0x1022), enabled.unsafe_ptr(), c_uint(4)) != 0:
            posix_failure("SO_NOSIGPIPE", posix_errno())


def posix_terminate(pid: Int, signal: Int) raises:
    if external_call["kill", c_int](c_int(pid), c_int(signal)) != 0:
        posix_failure("kill", posix_errno())
