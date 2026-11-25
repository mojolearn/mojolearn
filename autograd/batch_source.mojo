# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE BATCH SOURCE: streamed, variable-sized training batches read from sharded files on disk, in Mojo
(lane batch-source, Andrew 2026-11-25: "train any custom model on streamed, variable-sized batches").

What existed: `lm_corpus.TokenBatches` (core/lm_batch_run.mojo) serves FIXED-length rows of one mapped token
file to the byte LM trainer, and `Program.fit` batches arrays already in memory. Neither streams variable-length
records from disk, buckets them, pads each batch to its own length or keeps a resumable position. This module is
that source; `Program.fit(source=)` (autograd/program_runtime.mojo, the streamed instance of the fit loop) is its
consumer. Fits that pass arrays run the loop instance they ran before (the streamed one is a separate comptime
instance).

FORMAT (`mojolearn-batch-shards v1`). A directory holding `index.txt` and plain NumPy `.npy` files, so a shard is
written with `numpy.save` and nothing else:

    index.txt                 line 1: `mojolearn-batch-shards v1`; then one line `shard <stem>` per shard, in order
    <stem>.steps.npy          float32 (total_steps, F), C order: every record's steps, record after record
    <stem>.offsets.npy        int64 (n + 1,): record r is steps[offsets[r]:offsets[r + 1]] (length >= 1)
    <stem>.targets.npy        optional, float32 (n, W), C order: one target row per record

Every shard has the same F, the same W and the same set of files. The records are numbered over the shards in index
order. The index's first line is the format word: any other word is refused by name.

FIELDS a Program input (or the fit's target) reads, each batch with B rows and padded length L (the longest record
of THIS batch): "steps" (B, L, F) with zeros past each record's length; "lengths" (B, 1) float32; "mask" (B, L)
float32, 1 at a real step, 0 at padding; "targets" (B, W).

THE PLAN of an epoch e (host integers only, the same on every vendor and the host column): the record order is the
identity, or, with shuffle, a Fisher-Yates permutation drawn from splitmix64 keyed by (seed, e, 0); the order is cut
into windows of batch * bucket records; each window is sorted by length (stable: ties keep the order) and cut into
batches of `batch` records, so a batch holds records of similar length; with shuffle the batch order is then
permuted by a second stream keyed by (seed, e, 1). bucket = 1 is no bucketing. Every epoch has the same number of
batches (n_batches below). Parallel plan: the windows are sorted on host tasks (independent units, one per
window); the two Fisher-Yates walks are serial by nature (each swap reads the previous state) and are integer work
over the record ids, once an epoch.

THE READ of one batch: each record is one `pread` of its steps (and its target row) into its own row of the host
buffers, the padding zero-filled, on host tasks (one record a task: the reads are independent). PREFETCH: the fit
loop reads the NEXT batch into the host buffers right after it has queued the current batch's training, so on a
GPU the disk reads run while the device trains (the upload copies the host words into the executor's staging at
once, autograd/program_runtime.mojo).

THE POSITION is (epoch, consumed): `consumed` batches of `epoch` have been trained (their optimizer step is
queued). The read-ahead batch is NOT consumed: a checkpoint records the consumed position, so a resume continues
with the batch that was read ahead but never trained. `held_*` below is the read-ahead; `epoch` / `consumed` the
consumed position. A source also has a FINGERPRINT (FNV-1a over the record count, F, W and every record's length),
so a position is never applied to other data.
"""
from std.ffi import c_int, external_call
from std.sys.info import CompilationTarget
from std.sys.compile import is_defined
from std.builtin.sort import sort
from sequence.ops import FP
from bindings.native_posix_transport_run import posix_errno
from core.host_parallel import host_parallelize
from core.native_archive_span import ArchiveSpan
from core.native_archive_npy import NativeNpyLayout, ArchiveLiteralWarning, native_npy_layout_collect

comptime BATCH_SOURCE_FORMAT = "mojolearn-batch-shards v1"
comptime BS_FIELD_STEPS = 1
comptime BS_FIELD_LENGTHS = 2
comptime BS_FIELD_MASK = 3
comptime BS_FIELD_TARGETS = 4
comptime _BS_HEADER_LIMIT = 65536
comptime _BS_INDEX_LIMIT = 16777216
comptime _Bytes = UnsafePointer[UInt8, MutUntrackedOrigin]

# Sabotage arms (tools/identity_lanes/coverage_sabotage.tsv coverage-autograd-batch_source_*; lane
# ops-batch-source-resume). Off in every build; an arm's patch switches one on. The plain name faults the device
# executor only, _HOST the host executor only (comptime E.IS_DEVICE at the one use, `batch_source_saved`).
#   READ_AHEAD: the position a fit hands back for a checkpoint is the read-ahead batch's, not the consumed one, so a
#   resume skips the batch that was read ahead but never trained.
comptime BATCH_SOURCE_SABOTAGE_READ_AHEAD = is_defined["MOJOLEARN_BATCH_SOURCE_SABOTAGE_READ_AHEAD"]()
comptime BATCH_SOURCE_SABOTAGE_READ_AHEAD_HOST = is_defined["MOJOLEARN_BATCH_SOURCE_SABOTAGE_READ_AHEAD_HOST"]()


def batch_source_field(name: String) raises -> Int:
    """The field code of a field name; anything else is refused by name."""
    if name == "steps":
        return BS_FIELD_STEPS
    if name == "lengths":
        return BS_FIELD_LENGTHS
    if name == "mask":
        return BS_FIELD_MASK
    if name == "targets":
        return BS_FIELD_TARGETS
    raise Error("ValueError: BatchSource: unknown field '" + name + "' (fields: steps, lengths, mask, targets)")


def batch_source_field_name(code: Int) -> String:
    if code == BS_FIELD_STEPS:
        return "steps"
    if code == BS_FIELD_LENGTHS:
        return "lengths"
    if code == BS_FIELD_MASK:
        return "mask"
    return "targets"


def batch_source_field_rank(code: Int) -> Int:
    """The rank of a field's batch array: steps (B, L, F) 3, every other field 2."""
    return 3 if code == BS_FIELD_STEPS else 2


struct BatchSourceSpec(Copyable, Movable):
    """The typed words of a `BatchSource` handle, read once at UNPACK: the directory, rows per batch, the Program
    input fields in capture order, the fit target's field (0: none), shuffle, seed, batches per bucket window, the
    fingerprint the handle was opened with, and the consumed position."""

    var path: String
    var batch: Int
    var inputs: List[Int]
    var target: Int
    var shuffle: Bool
    var seed: UInt64
    var bucket: Int
    var fingerprint: UInt64
    var epoch: Int
    var consumed: Int

    def __init__(out self):
        self.path = String("")
        self.batch = 1
        self.inputs = List[Int]()
        self.target = 0
        self.shuffle = True
        self.seed = 0
        self.bucket = 1
        self.fingerprint = 0
        self.epoch = 0
        self.consumed = 0


comptime _IP = MutPointer[Int, MutUntrackedOrigin]


@always_inline
def _ip(address: Int) -> _IP:
    """An untracked view of an Int list's storage (host tasks write disjoint cells; the owner outlives the split)."""
    return _IP(unsafe_from_address=address)


@always_inline
def _mix(x: UInt64) -> UInt64:
    var z = x + UInt64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


def _stream_key(seed: UInt64, epoch: Int, salt: Int) -> UInt64:
    return _mix(seed ^ _mix(UInt64(epoch) * UInt64(0x100000001B3) + UInt64(salt)))


def _fisher_yates(mut values: List[Int], key: UInt64):
    """Permute `values` in place with the splitmix64 stream from `key`. Serial by nature: each swap reads the
    state the previous one left."""
    var state = key
    var i = len(values) - 1
    while i > 0:  # driver-loop: over the permutation's swaps (host integers, once an epoch)
        state = _mix(state)
        var j = Int(state % UInt64(i + 1))
        var t = values[i]
        values[i] = values[j]
        values[j] = t
        i -= 1


def _open_read(path: String) raises -> Int:
    if path.byte_length() == 0 or path.find("\0") >= 0:
        raise Error("ValueError: BatchSource: a shard path is empty or contains NUL")
    var cwd = -100  # AT_FDCWD
    var flags = 0x80000  # O_RDONLY | O_CLOEXEC
    comptime if CompilationTarget.is_macos():
        cwd = -2
        flags = 0x1000000
    var fd = -1
    while fd < 0:  # driver-loop: an interrupted open is taken up again
        fd = Int(external_call["openat", c_int, num_fixed_args=3](
            c_int(cwd), path.as_c_string_slice().unsafe_ptr(), c_int(flags), c_int(0)))
        if fd < 0:
            var number = posix_errno()
            if number != 4:
                raise Error("FileNotFoundError: BatchSource: cannot open " + path + " (errno " + String(number) + ")")
    return fd


def _pread(fd: Int, address: Int, count: Int, offset: Int) -> Int:
    """Read `count` bytes at file `offset` into `address`: the bytes read (fewer only at the end of the file),
    or -errno. Safe from several host tasks at once (pread has no shared file position)."""
    var done = 0
    while done < count:  # driver-loop: over partial reads of one range
        var got = external_call["pread", Int](Int32(fd), _Bytes(unsafe_from_address=address + done), count - done,
                                              offset + done)
        if got > 0:
            done += got
        elif got == 0:
            return done
        else:
            var number = posix_errno()
            if number != 4:
                return -number
    return done


def _file_size(fd: Int) -> Int:
    return Int(external_call["lseek", Int](Int32(fd), 0, Int32(2)))


def _close(fd: Int):
    if fd >= 0:
        _ = external_call["close", c_int](c_int(fd))


struct _Npy(Copyable, Movable):
    """One opened .npy file: its descriptor, dtype, shape and the payload's byte offset."""

    var fd: Int
    var dtype: String
    var shape: List[Int]
    var offset: Int

    def __init__(out self):
        self.fd = -1
        self.dtype = String("")
        self.shape = List[Int]()
        self.offset = 0


def _open_npy(path: String) raises -> _Npy:
    """Open `path` and admit its .npy header (core/native_archive_npy.mojo's parser, the one the archive reader
    uses); only C order is accepted."""
    var out = _Npy()
    out.fd = _open_read(path)
    try:
        var size = _file_size(out.fd)
        var head = List[UInt8](length=12, fill=0)
        if size < 12 or _pread(out.fd, Int(head.unsafe_ptr()), 12, 0) != 12:
            raise Error("ValueError: BatchSource: " + path + " is not an .npy file")
        var begin = 10 if head[6] == 1 else 12
        var header = Int(head[8]) | (Int(head[9]) << 8)
        if head[6] != 1:
            header = header | (Int(head[10]) << 16) | (Int(head[11]) << 24)
        if header < 0 or header > _BS_HEADER_LIMIT or begin + header > size:
            raise Error("ValueError: BatchSource: " + path + " has a bad .npy header")
        var raw = List[UInt8](length=begin + header, fill=0)
        if _pread(out.fd, Int(raw.unsafe_ptr()), begin + header, 0) != begin + header:
            raise Error("ValueError: BatchSource: " + path + " is truncated")
        # The span claims the file's size so the parser's payload check sees the whole file; it reads only the
        # header bytes held in `raw`.
        var warnings = List[ArchiveLiteralWarning]()
        var layout = native_npy_layout_collect(ArchiveSpan(Int(raw.unsafe_ptr()), size), 12, 0, warnings)
        _ = raw^  # the parser read through the raw address
        if layout.fortran:
            raise Error("ValueError: BatchSource: " + path + " is in Fortran order; save it in C order")
        out.dtype = layout.dtype
        out.shape = layout.shape.copy()
        out.offset = begin + header
    except error:
        _close(out.fd)
        raise error
    return out^


def _ascii(bytes: List[UInt8], start: Int, stop: Int) raises -> String:
    """Bytes [start, stop) as text; a byte above 127 is refused (index lines are ASCII)."""
    var out = String("")
    for k in range(start, stop):  # driver-loop: over one index line's bytes
        if bytes[k] > 127 or bytes[k] == 0:
            raise Error("ValueError: BatchSource: index.txt holds a byte that is not printable ASCII")
        out += chr(Int(bytes[k]))
    return out^


def _read_index(path: String) raises -> List[String]:
    """The shard stems of `<path>/index.txt`, after its format word."""
    var fd = _open_read(path + "/index.txt")
    var size = _file_size(fd)
    if size < 0 or size > _BS_INDEX_LIMIT:
        _close(fd)
        raise Error("ValueError: BatchSource: index.txt is too large")
    var bytes = List[UInt8](length=max(size, 1), fill=0)
    var got = _pread(fd, Int(bytes.unsafe_ptr()), size, 0)
    _close(fd)
    if got != size:
        raise Error("ValueError: BatchSource: cannot read index.txt")
    # each line as its byte range (a trailing CR dropped); the text is ASCII (stems that are not are refused)
    var starts = List[Int]()
    var stops = List[Int]()
    var start = 0
    for i in range(size + 1):  # driver-loop: over the index's bytes (a short text file)
        if i == size or bytes[i] == 10:
            var stop = i
            if stop > start and bytes[stop - 1] == 13:
                stop -= 1
            starts.append(start)
            stops.append(stop)
            start = i + 1
    if len(starts) == 0 or _ascii(bytes, starts[0], stops[0]) != BATCH_SOURCE_FORMAT:
        raise Error("ValueError: BatchSource: index.txt does not start with the format words '"
                    + BATCH_SOURCE_FORMAT + "'")
    var stems = List[String]()
    for i in range(1, len(starts)):  # driver-loop: over the index's shard lines
        if stops[i] == starts[i]:
            continue
        if stops[i] - starts[i] < 7 or _ascii(bytes, starts[i], starts[i] + 6) != "shard ":
            raise Error("ValueError: BatchSource: index.txt line " + String(i + 1) + " is not 'shard <stem>'")
        var stem = _ascii(bytes, starts[i] + 6, stops[i])
        if stem.find("/") >= 0 or stem.find("\0") >= 0 or stem == "." or stem == "..":
            raise Error("ValueError: BatchSource: a shard stem names a file outside the directory")
        stems.append(stem)
    if len(stems) == 0:
        raise Error("ValueError: BatchSource: index.txt lists no shard")
    return stems^


struct BatchSource(Movable):
    """An opened source: the shard descriptors, every record's (shard, first step, length), the current epoch's
    plan, the host buffers of the read-ahead batch and the consumed position. Opened by `batch_source_open`,
    released by `close` (the caller closes it on every path)."""

    var spec: BatchSourceSpec
    var width: Int
    var twidth: Int
    var max_len: Int
    var n: Int
    var n_batches: Int
    var fingerprint: UInt64
    var steps_fd: List[Int]
    var steps_off: List[Int]
    var tgt_fd: List[Int]
    var tgt_off: List[Int]
    var rec_shard: List[Int]
    var rec_start: List[Int]
    var rec_len: List[Int]
    var rec_row: List[Int]
    var plan_epoch: Int
    var order: List[Int]
    var bounds: List[Int]
    var h_steps: List[Float32]
    var h_lens: List[Float32]
    var h_mask: List[Float32]
    var h_tgt: List[Float32]
    var held_epoch: Int
    var held_batch: Int
    var held_rows: Int
    var held_len: Int
    var epoch: Int
    var consumed: Int

    def __init__(out self, var spec: BatchSourceSpec):
        self.spec = spec^
        self.width = 0
        self.twidth = 0
        self.max_len = 0
        self.n = 0
        self.n_batches = 0
        self.fingerprint = 0
        self.steps_fd = List[Int]()
        self.steps_off = List[Int]()
        self.tgt_fd = List[Int]()
        self.tgt_off = List[Int]()
        self.rec_shard = List[Int]()
        self.rec_start = List[Int]()
        self.rec_len = List[Int]()
        self.rec_row = List[Int]()
        self.plan_epoch = -1
        self.order = List[Int]()
        self.bounds = List[Int]()
        self.h_steps = List[Float32]()
        self.h_lens = List[Float32]()
        self.h_mask = List[Float32]()
        self.h_tgt = List[Float32]()
        self.held_epoch = -1
        self.held_batch = -1
        self.held_rows = 0
        self.held_len = 0
        self.epoch = 0
        self.consumed = 0

    def close(mut self):
        """Close every shard descriptor (idempotent)."""
        for k in range(len(self.steps_fd)):  # driver-loop: over shard descriptors
            _close(self.steps_fd[k])
            self.steps_fd[k] = -1
        for k in range(len(self.tgt_fd)):  # driver-loop: over shard descriptors
            _close(self.tgt_fd[k])
            self.tgt_fd[k] = -1

    def window(self) -> Int:
        return self.spec.batch * self.spec.bucket

    def next_position(self, epoch: Int, batch: Int) -> Tuple[Int, Int]:
        """The batch after (epoch, batch)."""
        if batch + 1 < self.n_batches:
            return (epoch, batch + 1)
        return (epoch + 1, 0)

    def pending(self) -> Tuple[Int, Int]:
        """The next batch to train: the consumed position, normalised past an epoch's end."""
        if self.consumed >= self.n_batches:
            return (self.epoch + 1, 0)
        return (self.epoch, self.consumed)

    def plan(mut self, epoch: Int):
        """The record order and the batch bounds of `epoch` (module note THE PLAN). Kept for the epoch; a
        different epoch replans."""
        if self.plan_epoch == epoch:
            return
        var order = List[Int](capacity=self.n)
        for i in range(self.n):  # driver-loop: over record ids (host integers, once an epoch)
            order.append(i)
        if self.spec.shuffle:
            _fisher_yates(order, _stream_key(self.spec.seed, epoch, 0))
        var w = self.window()
        if self.spec.bucket > 1:
            var windows = (self.n + w - 1) // w
            var optr = _ip(Int(order.unsafe_ptr()))
            var lens = _ip(Int(self.rec_len.unsafe_ptr()))
            var n = self.n
            # key = length * w + position in the window: sorting the keys sorts by length, ties in window order
            def sort_window(k: Int) {imm optr, imm lens, imm n, imm w}:
                var lo = k * w
                var hi = min(n, lo + w)
                var keys = List[Int](capacity=hi - lo)
                for i in range(lo, hi):  # driver-loop: over one window's records
                    keys.append(lens[optr[i]] * w + (i - lo))
                sort(keys)
                var ids = List[Int](capacity=hi - lo)
                for i in range(hi - lo):  # driver-loop: over one window's records
                    ids.append(optr[lo + keys[i] % w])
                for i in range(hi - lo):  # driver-loop: over one window's records
                    optr[lo + i] = ids[i]
            host_parallelize(sort_window, windows)
        # cut into batches: whole windows give `bucket` batches each, the last window ceil(rest / batch)
        var starts = List[Int]()
        var at = 0
        while at < self.n:  # driver-loop: over the epoch's batches (host integers)
            starts.append(at)
            var window_end = min(self.n, (at // w + 1) * w)
            at = min(at + self.spec.batch, window_end)
        var batches = len(starts)
        var pick = List[Int](capacity=batches)
        for b in range(batches):  # driver-loop: over the epoch's batches
            pick.append(b)
        if self.spec.shuffle:
            _fisher_yates(pick, _stream_key(self.spec.seed, epoch, 1))
        var planned = List[Int](capacity=self.n)
        var bounds = List[Int](capacity=batches + 1)
        bounds.append(0)
        for b in range(batches):  # driver-loop: over the epoch's batches in their drawn order
            var s = starts[pick[b]]
            var e = starts[pick[b] + 1] if pick[b] + 1 < batches else self.n
            for i in range(s, e):  # driver-loop: over one batch's record ids
                planned.append(order[i])
            bounds.append(len(planned))
        self.order = planned^
        self.bounds = bounds^
        self.plan_epoch = epoch

    def records(mut self, epoch: Int, batch: Int) -> List[Int]:
        """The record ids of batch `batch` of `epoch`, in row order."""
        self.plan(epoch)
        var out = List[Int]()
        for i in range(self.bounds[batch], self.bounds[batch + 1]):  # driver-loop: over one batch's rows
            out.append(self.order[i])
        return out^

    def read(mut self, epoch: Int, batch: Int) raises:
        """Read batch `batch` of `epoch` into the host buffers (module note THE READ): one host task a record."""
        if batch < 0 or batch >= self.n_batches:
            raise Error("ValueError: BatchSource: batch " + String(batch) + " is outside the epoch")
        self.plan(epoch)
        var lo = self.bounds[batch]
        var rows = self.bounds[batch + 1] - lo
        var length = 0
        for r in range(rows):  # driver-loop: over the batch's rows (its padded length)
            length = max(length, self.rec_len[self.order[lo + r]])
        var F = self.width
        var W = self.twidth
        var status = List[Int](length=rows, fill=0)
        var sp = _ip(Int(status.unsafe_ptr()))
        var ids = _ip(Int(self.order.unsafe_ptr())) + lo
        var shard = _ip(Int(self.rec_shard.unsafe_ptr()))
        var first = _ip(Int(self.rec_start.unsafe_ptr()))
        var lens = _ip(Int(self.rec_len.unsafe_ptr()))
        var rowin = _ip(Int(self.rec_row.unsafe_ptr()))
        var sfd = _ip(Int(self.steps_fd.unsafe_ptr()))
        var soff = _ip(Int(self.steps_off.unsafe_ptr()))
        var tfd = _ip(Int(self.tgt_fd.unsafe_ptr()))
        var toff = _ip(Int(self.tgt_off.unsafe_ptr()))
        var xs = FP(unsafe_from_address=Int(self.h_steps.unsafe_ptr()))
        var xl = FP(unsafe_from_address=Int(self.h_lens.unsafe_ptr()))
        var xm = FP(unsafe_from_address=Int(self.h_mask.unsafe_ptr()))
        var xt = FP(unsafe_from_address=Int(self.h_tgt.unsafe_ptr()))
        var has_t = W > 0
        def read_row(r: Int) {imm sp, imm ids, imm shard, imm first, imm lens, imm rowin, imm sfd, imm soff, imm tfd,
                               imm toff, imm xs, imm xl, imm xm, imm xt, imm F, imm W, imm length, imm has_t}:
            var rec = ids[r]
            var sh = shard[rec]
            var ln = lens[rec]
            var row = xs + r * length * F
            var want = ln * F * 4
            var got = _pread(sfd[sh], Int(row), want, soff[sh] + first[rec] * F * 4)
            if got != want:
                sp[r] = got if got < 0 else -5
            for i in range(ln * F, length * F):  # driver-loop: over one row's padding words
                row[i] = Float32(0)
            xl[r] = Float32(ln)
            for t in range(length):  # driver-loop: over one row's steps (the mask)
                xm[r * length + t] = Float32(1) if t < ln else Float32(0)
            if has_t:
                var tw = W * 4
                var tg = _pread(tfd[sh], Int(xt + r * W), tw, toff[sh] + rowin[rec] * tw)
                if tg != tw and sp[r] == 0:
                    sp[r] = tg if tg < 0 else -5
        host_parallelize(read_row, rows)
        for r in range(rows):  # driver-loop: over the batch's rows (their read status)
            if status[r] != 0:
                raise Error("OSError: BatchSource: reading record " + String(self.order[lo + r])
                            + " failed (" + ("errno " + String(-status[r]) if status[r] != -5 else "short read") + ")")
        self.held_epoch = epoch
        self.held_batch = batch
        self.held_rows = rows
        self.held_len = length




def _fnv(h: UInt64, word: Int) -> UInt64:
    var x = h
    var v = UInt64(word)
    for k in range(8):  # small-loop(8: the bytes of one word)
        x = (x ^ ((v >> UInt64(8 * k)) & UInt64(255))) * UInt64(0x100000001B3)
    return x


def batch_source_open(var spec: BatchSourceSpec) raises -> BatchSource:
    """Open the shards of `spec.path`, admit every header and offset table, and return the source at the spec's
    consumed position. Refuses a bad format word, mismatched shards, a record of length 0, a fingerprint that
    differs from the one the handle was opened with (other data under the same path) and a position outside the
    stream. The descriptors stay open until `close`."""
    if spec.batch < 1:
        raise Error("ValueError: BatchSource: batch_size must be >= 1")
    if spec.bucket < 1:
        raise Error("ValueError: BatchSource: bucket_batches must be >= 1")
    var stems = _read_index(spec.path)
    var src = BatchSource(spec^)
    try:
        var with_targets = False
        for k in range(len(stems)):  # driver-loop: over shards (open and admit, once a fit)
            var base = src.spec.path + "/" + stems[k]
            var steps = _open_npy(base + ".steps.npy")
            src.steps_fd.append(steps.fd)
            src.steps_off.append(steps.offset)
            var offsets = _open_npy(base + ".offsets.npy")
            if offsets.dtype != "<i8" or len(offsets.shape) != 1 or offsets.shape[0] < 2:
                _close(offsets.fd)
                raise Error("ValueError: BatchSource: " + base + ".offsets.npy must be int64 of shape (n + 1,)")
            var count = offsets.shape[0]
            var table = List[Int](length=count, fill=0)
            var got = _pread(offsets.fd, Int(table.unsafe_ptr()), count * 8, offsets.offset)
            _close(offsets.fd)
            if got != count * 8:
                raise Error("ValueError: BatchSource: " + base + ".offsets.npy is truncated")
            if steps.dtype != "<f4" or len(steps.shape) != 2:
                raise Error("ValueError: BatchSource: " + base + ".steps.npy must be float32 of shape (steps, F)")
            if k == 0:
                src.width = steps.shape[1]
            elif steps.shape[1] != src.width:
                raise Error("ValueError: BatchSource: shard " + stems[k] + " has another step width F")
            if src.width < 1 or table[0] != 0 or table[count - 1] != steps.shape[0]:
                raise Error("ValueError: BatchSource: " + base + ".offsets.npy does not cover its steps file")
            var tfd = -1
            var toff = 0
            if k == 0:
                with_targets = _exists(base + ".targets.npy")
            if with_targets:
                var tg = _open_npy(base + ".targets.npy")
                tfd = tg.fd
                toff = tg.offset
                if tg.dtype != "<f4" or len(tg.shape) != 2 or tg.shape[0] != count - 1:
                    _close(tfd)
                    raise Error("ValueError: BatchSource: " + base + ".targets.npy must be float32 of shape (n, W)")
                if k == 0:
                    src.twidth = tg.shape[1]
                elif tg.shape[1] != src.twidth:
                    _close(tfd)
                    raise Error("ValueError: BatchSource: shard " + stems[k] + " has another target width W")
            elif _exists(base + ".targets.npy"):
                raise Error("ValueError: BatchSource: shard " + stems[k] + " has targets and the first shard has none")
            src.tgt_fd.append(tfd)
            src.tgt_off.append(toff)
            for r in range(count - 1):  # driver-loop: over the shard's records (the offset table, once a fit)
                var length = table[r + 1] - table[r]
                if length < 1:
                    raise Error("ValueError: BatchSource: shard " + stems[k] + " record " + String(r)
                                + " has no step (offsets must increase)")
                src.rec_shard.append(k)
                src.rec_start.append(table[r])
                src.rec_len.append(length)
                src.rec_row.append(r)
                src.max_len = max(src.max_len, length)
        src.n = len(src.rec_len)
        var w = src.window()
        var whole = src.n // w
        var rest = src.n - whole * w
        src.n_batches = whole * src.spec.bucket + (rest + src.spec.batch - 1) // src.spec.batch
        var h = UInt64(0xCBF29CE484222325)
        h = _fnv(h, src.n)
        h = _fnv(h, src.width)
        h = _fnv(h, src.twidth)
        for i in range(src.n):  # driver-loop: over record lengths (the fingerprint, once a fit)
            h = _fnv(h, src.rec_len[i])
        src.fingerprint = h
        if src.spec.fingerprint != 0 and src.spec.fingerprint != h:
            raise Error("ValueError: BatchSource: the shards under " + src.spec.path
                        + " are not the data this source (or its saved position) was opened on")
        for k in range(len(src.spec.inputs)):  # driver-loop: over the requested fields
            if src.spec.inputs[k] == BS_FIELD_TARGETS and src.twidth == 0:
                raise Error("ValueError: BatchSource: the field 'targets' needs <stem>.targets.npy files")
        if src.spec.target == BS_FIELD_TARGETS and src.twidth == 0:
            raise Error("ValueError: BatchSource: target='targets' needs <stem>.targets.npy files")
        if src.spec.epoch < 0 or src.spec.consumed < 0 or src.spec.consumed > src.n_batches:
            raise Error("ValueError: BatchSource: the position (epoch " + String(src.spec.epoch) + ", batch "
                        + String(src.spec.consumed) + ") is outside the stream of "
                        + String(src.n_batches) + " batches an epoch")
        src.epoch = src.spec.epoch
        src.consumed = src.spec.consumed
        var rows = min(src.spec.batch, src.n)
        src.h_steps = List[Float32](length=max(1, rows * src.max_len * src.width), fill=Float32(0))
        src.h_lens = List[Float32](length=max(1, rows), fill=Float32(0))
        src.h_mask = List[Float32](length=max(1, rows * src.max_len), fill=Float32(0))
        src.h_tgt = List[Float32](length=max(1, rows * src.twidth), fill=Float32(0))
    except error:
        src.close()
        raise error
    return src^


def _exists(path: String) -> Bool:
    try:
        var fd = _open_read(path)
        _close(fd)
        return True
    except:
        return False


def batch_source_saved[IS_DEVICE: Bool](src: BatchSource) -> Tuple[Int, Int]:
    """The position a checkpoint records after a fit: the CONSUMED one (module note THE POSITION). The sabotage
    arms hand back the read-ahead batch's position instead, on one side each."""
    comptime if (BATCH_SOURCE_SABOTAGE_READ_AHEAD and IS_DEVICE) or (
            BATCH_SOURCE_SABOTAGE_READ_AHEAD_HOST and not IS_DEVICE):
        if src.held_batch >= 0:
            var ahead = src.next_position(src.held_epoch, src.held_batch)  # SABOTAGE: past the read-ahead batch
            return ahead
    return (src.epoch, src.consumed)
