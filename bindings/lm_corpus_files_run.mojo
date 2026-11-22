# SPDX-License-Identifier: Apache-2.0
"""Native files and paths of the corpus operations (RUN, no Python).

Everything `lm_corpus.prepare`, the corpus and batches constructors and the
private tokenize step do to the filesystem: path expansion, normalisation
and symbolic-link resolution over strings; whole-file reads; writes from a
host address; the streamed digest with its byte count
(`core/sha256_file.mojo`); directory creation; the unpublished temporary
directory and its rename into place; the clock. A file is read once, into
native bytes.

A failed file call raises `OSError[<errno>]: <path>`; the door's PACK turns
that into the `OSError` subclass CPython picks for the number, with its
`errno` and `filename` (what the retained posix entries raised).

Serial by nature: each call here is one file or one directory entry; the
compute between them (BPE training and encoding) is parallel in its own
modules.
"""
from std.ffi import c_int, external_call
from std.os import getenv, getuid, listdir, mkdir, remove, rmdir
from std.os.path import getsize, isdir, isfile, islink
from std.pathlib import cwd
from std.pwd import getpwnam, getpwuid
from std.sys.info import CompilationTarget
from std.time import perf_counter_ns
from bindings.native_posix_transport_run import posix_errno
from core.sha256_file import sha256_file_sized
from core.shared_slot import SharedSlot

comptime _CorpusBytes = UnsafePointer[UInt8, MutUntrackedOrigin]
comptime CORPUS_OS_ERROR = "OSError["


struct CorpusEnvironment(Defaultable, Movable):
    """The corpus doors' user settings, read ONCE per process on first use
    (lane envgate-neural-lm, 2026-11-15: `HOME` was a getenv per
    `corpus_path` call and `MOJOLEARN_LM_CACHE` one per corpus call). Set
    them before the process starts; a change after the first corpus call is
    not seen."""
    var home: String
    var lm_cache: String

    def __init__(out self):
        self.home = String(getenv("HOME", ""))
        self.lm_cache = String(getenv("MOJOLEARN_LM_CACHE", ""))


comptime CORPUS_ENVIRONMENT = SharedSlot[CorpusEnvironment, "MojolearnCorpusEnvironmentV1"]


def corpus_env_home() raises -> String:
    """`HOME` as the process started (empty when unset)."""
    return CORPUS_ENVIRONMENT.get_or_create_ptr()[].home.copy()


def corpus_env_lm_cache() raises -> String:
    """`MOJOLEARN_LM_CACHE` as the process started (empty when unset)."""
    return CORPUS_ENVIRONMENT.get_or_create_ptr()[].lm_cache.copy()


def corpus_os_error(path: String, number: Int) -> Error:
    return Error(String(CORPUS_OS_ERROR) + String(number) + "]: " + path)


def corpus_run_seconds() -> Float64:
    return Float64(perf_counter_ns()) * 1e-9


def corpus_run_join(path: String, leaf: String) -> String:
    """`path` without trailing slashes, a slash, `leaf`."""
    var view = path.as_bytes()
    var keep = len(view)
    while keep > 0 and Int(view[keep - 1]) == 47:  # small-loop(trailing slashes of one path)
        keep -= 1
    var bytes = List[UInt8](capacity=keep + 1 + leaf.byte_length())
    for i in range(keep):  # small-loop(bytes of one path)
        bytes.append(view[i])
    bytes.append(UInt8(47))
    var tail = leaf.as_bytes()
    for i in range(len(tail)):  # small-loop(bytes of one file name)
        bytes.append(tail[i])
    var result = String(StringSlice(unsafe_from_utf8=Span(bytes)))
    _ = len(bytes)
    return result


def corpus_run_parent(path: String) -> String:
    """The text before the last slash (empty when the slash is first; the
    whole text when there is none), as `rsplit("/", 1)[0]`."""
    var view = path.as_bytes()
    var cut = len(view)
    for i in range(len(view)):  # small-loop(bytes of one path)
        if Int(view[i]) == 47:
            cut = i
    var bytes = List[UInt8](capacity=cut)
    for i in range(cut):  # small-loop(bytes of one path)
        bytes.append(view[i])
    var result = String(StringSlice(unsafe_from_utf8=Span(bytes)))
    _ = len(bytes)
    return result


def _path_parts(text: String) -> List[String]:
    """The components between slashes, without the empty ones and `.`."""
    var parts = List[String]()
    for part_slice in text.split("/"):  # small-loop(components of one path)
        var part = String(part_slice)
        if part.byte_length() == 0 or part == ".":
            continue
        parts.append(part)
    return parts^


def _path_joined(parts: List[String], prefix: String) -> String:
    var result = prefix.copy()
    for i in range(len(parts)):  # small-loop(components of one path)
        if i:
            result += "/"
        result += parts[i]
    return result


def corpus_run_normalize(text: String) -> String:
    """Drop empty components and `.`; keep one leading slash, or exactly two."""
    var prefix = String("")
    if text.startswith("/"):
        prefix = String("//") if text.startswith("//") and not text.startswith("///") else String("/")
    var result = _path_joined(_path_parts(text), prefix)
    return String(".") if result.byte_length() == 0 else result


def corpus_run_expand(text: String) raises -> String:
    """`~` and `~user` at the start of a path: `HOME` as the process
    started, else the user database."""
    if not text.startswith("~"):
        return text.copy()
    var head = String("")
    var rest = String("")
    var first = True
    var tail = False
    for part_slice in text.split("/"):  # small-loop(components of one path)
        if first:
            head = String(part_slice)
            first = False
        else:
            if tail:
                rest += "/"
            rest += String(part_slice)
            tail = True
    var name_bytes = List[UInt8]()
    var view = head.as_bytes()
    for i in range(1, len(view)):  # small-loop(bytes of one user name)
        name_bytes.append(view[i])
    var home = String("")
    if len(name_bytes) == 0:
        home = corpus_env_home()
        if home.byte_length() == 0:
            home = getpwuid(getuid()).pw_dir.copy()
    else:
        var name = String(StringSlice(unsafe_from_utf8=Span(name_bytes)))
        _ = len(name_bytes)
        try:
            home = getpwnam(name).pw_dir.copy()
        except:
            raise Error("KeyError: getpwnam(): name not found: '" + name + "'")
    return home + "/" + rest if tail else home


def _read_link(path: String) raises -> String:
    var target = path.copy()
    var buffer = List[UInt8](length=4097, fill=UInt8(0))
    var count = external_call["readlink", Int](target.as_c_string_slice().unsafe_ptr(), buffer.unsafe_ptr(), 4096)
    if count < 0:
        raise corpus_os_error(path, posix_errno())
    var bytes = List[UInt8](capacity=count)
    for i in range(count):  # small-loop(bytes of one link target)
        bytes.append(buffer[i])
    var result = String(StringSlice(unsafe_from_utf8=Span(bytes)))
    _ = len(bytes)
    _ = len(buffer)
    return result


def corpus_run_resolve(text: String, depth: Int = 0) raises -> String:
    """The absolute path with every symbolic link followed; a component that
    does not exist is kept (`Path.resolve(strict=False)`)."""
    if depth > 40:
        raise Error("RuntimeError: corpus path contains a symbolic-link loop")
    var source = text.copy()
    if not source.startswith("/"):
        source = cwd().path + "/" + source
    var walked = List[String]()
    for part_slice in source.split("/"):  # driver-loop: resolve admitted filesystem components and native symlinks
        var part = String(part_slice)
        if part.byte_length() == 0 or part == ".":
            continue
        if part == "..":
            if len(walked) > 0:
                _ = walked.pop()
            continue
        var current = _path_joined(walked, String("/"))
        var next = current + part if len(walked) == 0 else current + "/" + part
        if islink(next):
            var target = _read_link(next)
            if not target.startswith("/"):
                target = current + "/" + target
            walked = _path_parts(corpus_run_resolve(target, depth + 1))
        else:
            walked.append(part)
    return _path_joined(walked, String("/"))


def corpus_run_is_file(path: String) -> Bool:
    return isfile(path)


def corpus_run_file_size(path: String) raises -> Int:
    var size = 0
    var number = 0
    try:
        size = getsize(path)
    except:
        number = max(posix_errno(), 1)
    if number != 0:
        raise corpus_os_error(path, 2 if not isfile(path) and not isdir(path) else number)
    return size


def corpus_run_read(path: String) raises -> List[UInt8]:
    """The whole file, read once."""
    var data = List[UInt8]()
    var number = 0
    try:
        with open(path, "r") as fh:
            data = fh.read_bytes()
    except:
        number = max(posix_errno(), 1)
    if number != 0:
        raise corpus_os_error(path, 2 if not isfile(path) and not isdir(path) else number)
    return data^


def corpus_run_hash_file(path: String, mut size: Int) raises -> String:
    """The streamed SHA-256 of the file and its byte count, in one pass."""
    var digest = String("")
    var number = 0
    try:
        digest = sha256_file_sized(path, size)
    except:
        number = max(posix_errno(), 1)
    if number != 0:
        raise corpus_os_error(path, 2 if not isfile(path) and not isdir(path) else number)
    return digest


def corpus_run_write(path: String, address: Int, count: Int) raises:
    """Create or truncate `path` (mode 0666 before the umask) and write the
    `count` host bytes at `address`."""
    var target = path.copy()
    var cwd_fd = -100  # AT_FDCWD
    var flags = 1 | 0x40 | 0x200 | 0x80000  # O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC
    comptime if CompilationTarget.is_macos():
        cwd_fd = -2
        flags = 1 | 0x200 | 0x400 | 0x1000000
    var fd = -1
    while fd < 0:  # small-loop(an interrupted open is taken up again)
        fd = Int(external_call["openat", c_int, num_fixed_args=3](c_int(cwd_fd), target.as_c_string_slice().unsafe_ptr(), c_int(flags), c_int(438)))
        if fd < 0:
            var number = posix_errno()
            if number != 4:
                raise corpus_os_error(path, number)
    var done = 0
    var failed = 0
    while done < count:  # driver-loop: over partial writes of one file
        # libc `write` as the std declares it: (Int32 fd, byte pointer, Int count) -> Int.
        var put = external_call["write", Int](Int32(fd), _CorpusBytes(unsafe_from_address=address + done), count - done)
        if put > 0:
            done += put
        elif put == 0:
            failed = 5
            break
        else:
            var number = posix_errno()
            if number != 4:
                failed = number
                break
    if external_call["close", c_int](c_int(fd)) != 0 and failed == 0:
        failed = posix_errno()
    if failed != 0:
        raise corpus_os_error(path, failed)


def corpus_run_write_bytes(path: String, data: List[UInt8]) raises:
    corpus_run_write(path, Int(data.unsafe_ptr()), len(data))
    _ = len(data)


def corpus_run_mkdirs(path: String) raises:
    """The directory and its missing parents (mode 0777 before the umask)."""
    if isdir(path):
        return
    var parent = corpus_run_parent(path)
    if parent.byte_length() > 0 and parent != path:
        corpus_run_mkdirs(parent)
    var number = 0
    try:
        mkdir(path, 0o777)
    except:
        number = max(posix_errno(), 1)
    if number != 0 and not isdir(path):
        raise corpus_os_error(path, number)


def _corpus_remove(path: String) raises:
    if islink(path) or not isdir(path):
        remove(path)
        return
    var entries = listdir(path)
    for i in range(len(entries)):  # driver-loop: remove only the unpublished corpus temporary tree
        _corpus_remove(path + "/" + entries[i])
    rmdir(path)


def corpus_run_atomic_begin(final: String) raises -> String:
    """A new private directory beside `final` (its parents made first)."""
    var parent = corpus_run_parent(final)
    if parent == final:
        parent = String(".")
    elif parent.byte_length() == 0:
        parent = String("/")
    corpus_run_mkdirs(parent)
    var stamp = Int(perf_counter_ns())
    var temporary = String("")
    var number = 17
    for attempt in range(16):  # small-loop(temporary names until one is new)
        temporary = final + ".partial-" + String(stamp + attempt)
        number = 0
        try:
            mkdir(temporary, 0o700)
        except:
            number = max(posix_errno(), 1)
        if number == 0:
            return temporary
        if not isdir(temporary) and not isfile(temporary):
            break
    raise corpus_os_error(temporary, number)


def corpus_run_atomic_commit(temporary: String, final: String) raises:
    """Publish the finished directory; when another writer published first,
    drop this one."""
    if isdir(final):
        _corpus_remove(temporary)
        return
    var source = temporary.copy()
    var target = final.copy()
    var cwd_fd = -100  # AT_FDCWD
    comptime if CompilationTarget.is_macos():
        cwd_fd = -2
    if external_call["renameat", c_int](c_int(cwd_fd), source.as_c_string_slice().unsafe_ptr(), c_int(cwd_fd), target.as_c_string_slice().unsafe_ptr()) != 0:
        raise corpus_os_error(final, posix_errno())


def corpus_run_atomic_abort(temporary: String):
    try:
        _corpus_remove(temporary)
    except:
        pass
