# SPDX-License-Identifier: Apache-2.0
# Public libc ABI on the supported LP64 Linux and Darwin hosts. No CPython
# socket/posix wrappers: those wrappers emit application audit callbacks.
# addrinfo's field order comes from glibc resolv/netdb.h and Apple's
# Libinfo/lookup.subproj/netdb.h; sockaddr/poll/fcntl values come from the
# corresponding public system headers, not interpreter or compiler layouts.
from std.ffi import c_int, c_uint, c_size_t, external_call
from std.sys.info import CompilationTarget
from std.math import ceil, isfinite
from std.python import Python

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


def posix_sleep(seconds: Float64) raises:
    if seconds < 0 or not isfinite(seconds) or seconds >= 9223372036854775808.0:
        raise Error("transport: invalid sleep interval")
    var whole = Int64(seconds)
    var request = Array[Int64, 2](fill=Int64(0))
    request[0] = whole
    request[1] = Int64((seconds - Float64(whole)) * 1e9)
    var remaining = Array[Int64, 2](fill=Int64(0))
    ref cpy = Python().cpython()
    var release = cpy.lib.get_function[PosixBytes]("PyEval_SaveThread")
    var restore = cpy.lib.get_function[NoneType]("PyEval_RestoreThread")
    var thread = release()
    var status = external_call["nanosleep", c_int](request.unsafe_ptr(), remaining.unsafe_ptr())
    var number = posix_errno()
    restore(thread)
    if status != 0:
        posix_failure("nanosleep", number)


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


def posix_ready(fd: Int, writing: Bool, deadline: Float64) raises:
    # pollfd is {int fd; short events; short revents} on both public ABIs.
    var words = Array[Int32, 2](fill=Int32(0))
    words[0] = Int32(fd)
    var shorts = MutPointer[Int16, MutUntrackedOrigin](unsafe_from_address=Int(words.unsafe_ptr()))
    shorts[2] = Int16(4 if writing else 1)  # POLLOUT / POLLIN
    while True:  # driver-loop: bounded native readiness wait
        var milliseconds = -1
        if deadline >= 0:
            var remaining = deadline - posix_clock()
            if remaining <= 0:
                raise Error("transport: operation timed out")
            milliseconds = Int(min(ceil(remaining * 1000.0), 2147483647.0))
        shorts[3] = 0
        # Match the existing native socket GIL release while the OS waits.
        # Resolve both public C entries before release; the released section
        # touches only native storage and libc, never Python objects.
        ref cpy = Python().cpython()
        var release = cpy.lib.get_function[PosixBytes]("PyEval_SaveThread")
        var restore = cpy.lib.get_function[NoneType]("PyEval_RestoreThread")
        var thread = release()
        var status = c_int(0)
        comptime if CompilationTarget.is_macos():
            status = external_call["poll", c_int](words.unsafe_ptr(), c_uint(1), c_int(milliseconds))
        else:
            status = external_call["poll", c_int](words.unsafe_ptr(), c_size_t(1), c_int(milliseconds))
        var number = posix_errno()
        restore(thread)
        if status > 0:
            if (shorts[3] & 32) != 0:  # POLLNVAL
                raise Error("transport: invalid socket descriptor")
            return  # read/write/getsockopt reports EOF and other socket errors.
        if status == 0:
            raise Error("transport: operation timed out")
        posix_failure("poll", number)


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


def posix_resolve(var host: String, port: Int, passive: Bool) raises -> Int:
    if host.find("\0") >= 0:
        raise Error("transport: hostname contains NUL")
    # Public LP64 addrinfo: four int fields, socklen_t, alignment padding,
    # three pointers. ai_addr/ai_canonname order differs on Darwin.
    var hints = Array[Int, 6](fill=0)
    var fields = MutPointer[c_int, MutUntrackedOrigin](unsafe_from_address=Int(hints.unsafe_ptr()))
    fields[0] = c_int(1 if passive else 0)  # AI_PASSIVE on Linux and Darwin.
    fields[2] = 1  # SOCK_STREAM
    var service = String(port)
    var hostname = Optional[PosixBytes](None)
    if host != "":
        hostname = PosixBytes(unsafe_from_address=Int(host.as_c_string_slice().unsafe_ptr()))
    var result = Array[Int, 1](fill=0)
    var service_pointer = service.as_c_string_slice().unsafe_ptr()
    ref cpy = Python().cpython()
    var release = cpy.lib.get_function[PosixBytes]("PyEval_SaveThread")
    var restore = cpy.lib.get_function[NoneType]("PyEval_RestoreThread")
    var thread = release()
    var status = external_call["getaddrinfo", c_int](hostname, service_pointer, hints.unsafe_ptr(), result.unsafe_ptr())
    var number = posix_errno()
    restore(thread)
    _ = host
    _ = service^
    if status != 0:
        var system_error = -11  # EAI_SYSTEM
        comptime if CompilationTarget.is_macos():
            system_error = 11
        if Int(status) == system_error:
            posix_failure("getaddrinfo", number)
        raise Error("transport: address resolution failed (EAI " + String(status) + ")")
    return result[0]


def posix_open(host: String, port: Int, timeout: Float64, listening: Bool) raises -> Int:
    var first = posix_resolve(host, port, listening)
    var current = first
    var failure = String("transport: no resolved network endpoint")
    try:
        while current != 0:  # driver-loop: try each resolved IPv4/IPv6 endpoint
            var info = PosixWords(unsafe_from_address=current)
            var fields = MutPointer[c_int, MutUntrackedOrigin](unsafe_from_address=current)
            var address = info[3]
            comptime if CompilationTarget.is_macos():
                address = info[4]
            var kind = fields[2]
            comptime if CompilationTarget.is_linux():
                kind |= 0x80800  # SOCK_CLOEXEC | SOCK_NONBLOCK atomically.
            var fd = Int(external_call["socket", c_int](fields[1], kind, fields[3]))
            if fd >= 0:
                try:
                    posix_socket_prepare(fd)
                    if listening:
                        var enabled = Array[c_int, 1](fill=c_int(1))
                        var level = 1
                        var option = 2
                        comptime if CompilationTarget.is_macos():
                            level = 65535
                            option = 4
                        if external_call["setsockopt", c_int](c_int(fd), c_int(level), c_int(option), enabled.unsafe_ptr(), c_uint(4)) != 0:
                            posix_failure("SO_REUSEADDR", posix_errno())
                        if external_call["bind", c_int](c_int(fd), PosixBytes(unsafe_from_address=address), c_uint(fields[4])) != 0:
                            posix_failure("bind", posix_errno())
                        if external_call["listen", c_int](c_int(fd), c_int(128)) != 0:
                            posix_failure("listen", posix_errno())
                    else:
                        var status = external_call["connect", c_int](c_int(fd), PosixBytes(unsafe_from_address=address), c_uint(fields[4]))
                        if status != 0:
                            var number = posix_errno()
                            var pending = number == 115 or number == 114
                            comptime if CompilationTarget.is_macos():
                                pending = number == 36 or number == 37
                            if number == 4 or (not pending and not posix_again(number)):
                                posix_failure("connect", number)
                            posix_ready(fd, True, posix_deadline(timeout))
                            var error = Array[c_int, 1](fill=c_int(0))
                            var size = Array[c_uint, 1](fill=c_uint(4))
                            var level = 1
                            var option = 4
                            comptime if CompilationTarget.is_macos():
                                level = 65535
                                option = 0x1007
                            if external_call["getsockopt", c_int](c_int(fd), c_int(level), c_int(option), error.unsafe_ptr(), size.unsafe_ptr()) != 0:
                                posix_failure("SO_ERROR", posix_errno())
                            if error[0] != 0:
                                posix_failure("connect", Int(error[0]))
                    return fd
                except error:
                    failure = String(error)
                    posix_close(fd)
                    if failure.startswith("InterruptedError:"):
                        raise error
            else:
                failure = "transport: socket failed (errno " + String(posix_errno()) + ")"
            current = info[5]
    finally:
        if first != 0:
            external_call["freeaddrinfo", NoneType](PosixBytes(unsafe_from_address=first))
    raise Error(failure)


def posix_accept(fd: Int, timeout: Float64) raises -> Int:
    var deadline = posix_deadline(timeout)
    while True:  # driver-loop: accept readiness can race another OS event
        posix_ready(fd, False, deadline)
        var child = -1
        comptime if CompilationTarget.is_linux():
            child = Int(external_call["accept4", c_int](c_int(fd), Optional[PosixBytes](None), Optional[PosixBytes](None), c_int(0x80800)))
        else:
            child = Int(external_call["accept", c_int](c_int(fd), Optional[PosixBytes](None), Optional[PosixBytes](None)))
        if child >= 0:
            try:
                posix_socket_prepare(child)
            except:
                posix_close(child)
                raise
            return child
        var number = posix_errno()
        if not posix_again(number):
            posix_failure("accept", number)


def posix_transfer(fd: Int, pointer: PosixBytes, count: Int, timeout: Float64, writing: Bool) raises:
    var offset = 0
    var deadline = posix_deadline(timeout)
    while offset < count:  # driver-loop: retain owner until every byte transfers
        posix_ready(fd, writing, deadline)
        var done = Int(0)
        if writing:
            var flags = 0
            comptime if CompilationTarget.is_linux():
                flags = 0x4000  # MSG_NOSIGNAL
            done = external_call["send", Int](c_int(fd), pointer + offset, c_size_t(count - offset), c_int(flags))
        else:
            done = external_call["recv", Int](c_int(fd), pointer + offset, c_size_t(count - offset), c_int(0))
        if done > 0:
            offset += done
        elif done == 0:
            raise Error("transport: peer closed the connection")
        else:
            var number = posix_errno()
            if not posix_again(number):
                posix_failure("send" if writing else "receive", number)


def posix_spawn(var executable: String, var argv: List[String], var environment: List[String]) raises -> Int:
    # String.as_c_string_slice requires mutable ownership in Mojo 1.0.
    # Keep these owned strings/lists alive until the synchronous spawn returns.
    if executable.find("\0") >= 0:
        raise Error("transport: executable contains NUL")
    var arguments = List[Int](length=len(argv) + 1, fill=0)
    var variables = List[Int](length=len(environment) + 1, fill=0)
    for i in range(len(argv)):  # driver-loop: borrowed CString spans remain owned through spawn
        if argv[i].find("\0") >= 0:
            raise Error("transport: process argument contains NUL")
        arguments[i] = Int(argv[i].as_c_string_slice().unsafe_ptr())
    for i in range(len(environment)):  # driver-loop: same ownership for the environment
        if environment[i].find("\0") >= 0:
            raise Error("transport: environment entry contains NUL")
        variables[i] = Int(environment[i].as_c_string_slice().unsafe_ptr())
    var pid = Array[c_int, 1](fill=c_int(0))
    var executable_pointer = executable.as_c_string_slice().unsafe_ptr()
    ref cpy = Python().cpython()
    var release = cpy.lib.get_function[PosixBytes]("PyEval_SaveThread")
    var restore = cpy.lib.get_function[NoneType]("PyEval_RestoreThread")
    var thread = release()
    var status = external_call["posix_spawnp", c_int](pid.unsafe_ptr(), executable_pointer, Optional[PosixBytes](None), Optional[PosixBytes](None), arguments.unsafe_ptr(), variables.unsafe_ptr())
    restore(thread)
    _ = executable
    _ = argv
    _ = environment
    _ = arguments^
    _ = variables^
    if status != 0:
        posix_failure("posix_spawnp", Int(status))
    return Int(pid[0])


def posix_wait(pid: Int, nohang: Bool) raises -> Int:
    var status = Array[c_int, 1](fill=c_int(0))
    ref cpy = Python().cpython()
    var release = cpy.lib.get_function[PosixBytes]("PyEval_SaveThread")
    var restore = cpy.lib.get_function[NoneType]("PyEval_RestoreThread")
    var thread = release()
    var found = external_call["waitpid", c_int](c_int(pid), status.unsafe_ptr(), c_int(1 if nohang else 0))
    var number = posix_errno()
    restore(thread)
    if found >= 0:
        return -1 if found == 0 else Int(status[0])
    posix_failure("waitpid", number)
    return -1  # posix_failure always raises


def posix_terminate(pid: Int, signal: Int) raises:
    if external_call["kill", c_int](c_int(pid), c_int(signal)) != 0:
        posix_failure("kill", posix_errno())
