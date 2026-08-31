# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Stage hashes, so a cross-backend difference has an ADDRESS.

NOT A PORT. CatBoost has no equivalent and does not need one: it ships one
GPU backend and accepts a non-deterministic answer. We ship Metal, CUDA and
HIP from one source and claim a bit-identical model, and **a claim that can
only be checked at the END is a claim nobody can debug.** Two model files
that differ tell you nothing about where the first bit moved. This file makes
the fit emit an ordered list of named checkpoints, each an FNV-1a64 over a
buffer's raw bit patterns, so `tools/identity_trace_diff.py` can name the
FIRST stage on which two runs disagree.

    MOJOLEARN_IDENTITY_TRACE=/tmp/apple.trace   pixi run ...
    MOJOLEARN_IDENTITY_TRACE=/tmp/nvidia.trace  pixi run ...
    python tools/identity_trace_diff.py /tmp/apple.trace /tmp/nvidia.trace

Companion to `core/launch_log.mojo`, which answers "which kernel ran" for a
profile; this one answers "which stage moved" for an identity claim.

## The format

One record per line, TAB separated, five fields:

    <seq>\\t<tag>\\t<dtype>\\t<count>\\t<hash16hex>

`seq` is 0-based and increases by one. `hash` is FNV-1a64 (offset basis
`0xCBF29CE484222325`, prime `0x100000001B3`) over the LITTLE-ENDIAN BYTES of
the elements IN INDEX ORDER, which makes it a pure function of the buffer's
contents and of nothing else -- not of the block shape that filled it, not of
the thread that wrote each cell, not of the core count.

Set `MOJOLEARN_IDENTITY_TRACE_DUMP=<substring>` and every record whose tag
contains that substring also writes `<trace>.<seq>.<tag>.bin`, the raw
elements, so the differ can go from "this stage" to "these cells, this many
ULPs apart, and here is the one that is a denormal on one side and a zero on
the other".

## Four rules for anyone adding a checkpoint

1. **BIT PATTERNS, NEVER DECIMAL TEXT.** `String(Float32)` does not round
   trip in this toolchain -- see `[[mojo-string-float-roundtrip]]`, which is
   why `original/` writes `<decimal>/<hex bits>` pairs everywhere. A trace
   built on formatted floats would report agreement across a real
   difference, which is the worst failure an instrument of this kind can
   have.

2. **THE TAG MUST BE MACHINE-INDEPENDENT.** Two traces are compared by
   ALIGNING THEIR TAG SEQUENCES, so a tag carrying an SM count, a block
   count or a grid width produces two disjoint tag sets and the comparison
   degenerates into "everything differs". Tags name a POSITION IN THE
   ALGORITHM -- `tree03.level02.hist.after_scan` -- and never a property of
   the machine running it.

3. **HASH THE LOGICAL BUFFER, NOT A MACHINE-SIZED SCRATCH.** `stat_partials`
   is sized from the core count (`partition_chunks_sm_for`, IDENTITY_PATHS
   row 7) and two backends will legitimately have different amounts of it
   holding different partial sums that reduce to the same answer. Hashing
   the scratch reports a difference where there is none and buries the one
   that matters. Checkpoint the REDUCED result.

4. **A TRACED RUN IS NOT A MEASUREMENT.** Every record drains the queue and
   copies a buffer to the host. That is a control-plane change of exactly
   the kind `HOST_AND_DEVICE.md` is about, and `quiet_window` would rightly
   refuse to certify a timing taken under it. Trace runs are trace subjects.

## What a matching hash does NOT prove

That the computation was identical. It proves the two buffers agree AT THAT
CHECKPOINT. Two different orders of summation that happen to round the same
way on this fixture agree here and diverge on the next one, which is why
`IDENTITY_PATHS.md` enumerates pathways by MECHANISM and this file only
localizes what the enumeration missed. The instrument finds where; the
ledger says why.

## Cost when unset

One `getenv` per `IdentityTrace` construction, not per record: the enabled
flag is read once and every `record_*` returns on a single boolean test.
`core/launch_log.mojo` reads its env per call because it is stateless; this
one already carries a sequence counter, so it carries the flag too.
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.os import getenv
from std.sys.info import size_of


comptime FNV_OFFSET: UInt64 = 0xCBF29CE484222325
comptime FNV_PRIME: UInt64 = 0x100000001B3
"""FNV-1a64, the same constants `ensemble/original/fingerprint_probe.mojo`
folds a whole forest with. Reused rather than re-chosen so a hash from either
instrument means the same kind of thing, and because a second hash function
in one repository is a second thing to get wrong."""

comptime TRACE_ENV = "MOJOLEARN_IDENTITY_TRACE"
comptime TRACE_DUMP_ENV = "MOJOLEARN_IDENTITY_TRACE_DUMP"


@always_inline
def fnv1a64_bytes[
    o: MutOrigin, //
](seed: UInt64, ptr: UnsafePointer[UInt8, o], n: Int) -> UInt64:
    """FNV-1a64 over `n` bytes, in order. The whole hash, in five lines.

    Byte at a time on purpose. A word-at-a-time variant is faster and is a
    DIFFERENT FUNCTION, and the differ recomputes this from a `.bin` dump to
    verify the writer and the reader agree -- so the two implementations
    have to be the same one, spelled the same way.
    """
    var h = seed
    for i in range(n):
        h = (h ^ UInt64(ptr.unsafe_load(i))) * FNV_PRIME
    return h


def _hex16(v: UInt64) -> String:
    """Sixteen lowercase hex digits, zero padded, most significant first."""
    comptime DIGITS = "0123456789abcdef"
    var out = String("")
    for i in range(16):
        var nib = Int((v >> UInt64(60 - 4 * i)) & UInt64(0xF))
        out += String(DIGITS[byte=nib])
    return out


def _sanitize(tag: StringSlice) -> String:
    """A tag as a filename component. Anything outside `[A-Za-z0-9._-]`
    becomes `_`, so a dump path can never escape its directory or collide
    with a shell metacharacter."""
    var out = String("")
    var s = String(tag)
    for i in range(s.byte_length()):
        var c = s[byte=i]
        var b = ord(String(c))
        var ok = (
            (b >= 48 and b <= 57)
            or (b >= 65 and b <= 90)
            or (b >= 97 and b <= 122)
            or b == 46
            or b == 95
            or b == 45
        )
        out += String(c) if ok else String("_")
    return out


def _dtype_name[dt: DType]() -> String:
    """The differ's `dtype` field. Its parser accepts exactly these six."""
    comptime if dt == DType.float32:
        return String("f32")
    comptime if dt == DType.float64:
        return String("f64")
    comptime if dt == DType.uint32:
        return String("u32")
    comptime if dt == DType.int32:
        return String("i32")
    comptime if dt == DType.uint8:
        return String("u8")
    comptime if dt == DType.int64:
        return String("i64")
    return String("?")


struct IdentityTrace(Movable):
    """One fit's trace. Construct once, pass it down, record at each stage.

    Carried by value through the searcher rather than reached through a
    global, so that two fits in one process (the interleaved harness runs
    exactly that) cannot interleave their sequence numbers into one file and
    make both traces unreadable.
    """

    var enabled: Bool
    var path: String
    var dump_match: String
    var seq: Int
    var seen: List[String]
    """Every tag emitted so far, for the UNIQUENESS INVARIANT below."""

    def __init__(out self):
        """Reads the environment ONCE. Disabled is the shipping state."""
        self.path = String(getenv(TRACE_ENV))
        self.dump_match = String(getenv(TRACE_DUMP_ENV))
        self.enabled = self.path != ""
        self.seq = 0
        self.seen = List[String]()

    @staticmethod
    def to_path(
        path: StringSlice,
        dump_match: StringSlice = "",
        truncate: Bool = True,
    ) raises -> Self:
        """A trace pointed at an explicit file, ignoring the environment.

        For CHECKS. `original/identity_trace_check.mojo` has to produce two
        traces in one process and compare them, and a check whose behavior
        depends on whether the operator happens to have
        `MOJOLEARN_IDENTITY_TRACE` exported is a check that passes or fails
        for reasons outside itself.
        """
        var t = Self()
        t.path = String(path)
        t.dump_match = String(dump_match)
        t.enabled = t.path != ""
        t.seq = 0
        t.seen = List[String]()
        # TRUNCATE BY DEFAULT HERE AND NEVER IN THE ENV CONSTRUCTOR. Records
        # are appended, so a check re-run against a path it used last time
        # reads back its own previous run concatenated with this one -- which
        # is exactly the failure this default exists to prevent, found by
        # re-running `check-identity-trace` twice and watching T1 report a
        # line-count mismatch that had nothing to do with the instrument.
        # The env path must NOT truncate: two fits in one process (the
        # interleaved harness) share one file deliberately.
        if t.enabled and truncate:
            with open(t.path, "w") as fh:
                fh.write("")
        return t^

    @staticmethod
    def disabled() -> Self:
        """An explicitly-off trace, for call sites that do not want to read
        the environment at all (a check that must not change behavior when
        someone has a trace variable exported in their shell)."""
        var t = Self()
        t.enabled = False
        return t^

    def header(mut self, what: StringSlice) raises:
        """Write a `#` comment line. The differ skips comments, so this is
        free-form provenance -- dataset, parameters, backend -- and it is
        worth writing, because two traces that turn out to disagree are
        worthless if nobody recorded what produced them."""
        if not self.enabled:
            return
        with open(self.path, "a") as fh:
            fh.write(String("# ") + String(what) + "\n")

    def _emit[
        o: MutOrigin, //
    ](
        mut self,
        tag: StringSlice,
        dtype: String,
        count: Int,
        raw: UnsafePointer[UInt8, o],
        n_bytes: Int,
    ) raises:
        # ============ THE TAG UNIQUENESS INVARIANT ============
        # The differ aligns two traces by their TAG SEQUENCES, with
        # `difflib.SequenceMatcher` over the tag lists. Repeated tags are
        # what breaks that: if one backend skips a stage, the matcher can
        # align a tag from tree 3 against the same tag from tree 7 and
        # produce a pairing that is plausible and wrong, which is the worst
        # thing this instrument can do.
        #
        # A tag is therefore required to be UNIQUE WITHIN A TRACE, which in
        # practice means carrying its `treeNN.levelMM.` prefix. That was a
        # convention in the header; it is an invariant here, because a
        # convention is what a tired author breaks at midnight.
        for i in range(len(self.seen)):
            if self.seen[i] == String(tag):
                raise Error(
                    String("identity_trace: duplicate tag '")
                    + String(tag)
                    + "' at seq "
                    + String(self.seq)
                    + ". Tags must be unique within a trace or the differ"
                    + " can align two runs' records wrongly; add the"
                    + " tree/level prefix."
                )
        self.seen.append(String(tag))

        if self.seq == 0:
            # A FORMAT VERSION, so a trace recorded today cannot be silently
            # reinterpreted by a reader written after the writer changes.
            # Comment lines are skipped by the differ's parser.
            with open(self.path, "a") as vh:
                vh.write("# format: mojolearn-identity-trace v1\n")

        var h = fnv1a64_bytes(FNV_OFFSET, raw, n_bytes)
        var line = (
            String(self.seq)
            + "\t"
            + String(tag)
            + "\t"
            + dtype
            + "\t"
            + String(count)
            + "\t"
            + _hex16(h)
            + "\n"
        )
        with open(self.path, "a") as fh:
            fh.write(line)

        if self.dump_match != "" and String(tag).find(self.dump_match) >= 0:
            var dump_path = (
                self.path
                + "."
                + String(self.seq)
                + "."
                + _sanitize(tag)
                + ".bin"
            )
            var bytes = List[UInt8]()
            for i in range(n_bytes):
                bytes.append(raw.unsafe_load(i))
            with open(dump_path, "w") as fh:
                fh.write_bytes(Span(bytes))
        self.seq += 1

    def record_device[
        dt: DType
    ](
        mut self,
        ctx: DeviceContext,
        tag: StringSlice,
        buf: DeviceBuffer[dt],
        count: Int = -1,
    ) raises:
        """Hash a DEVICE buffer's first `count` elements (default: all).

        **THIS DRAINS.** `synchronize` before the read is not optional -- a
        copy issued behind un-drained work hashes whatever the queue had got
        to -- and it is why rule 4 in this file's header says a traced run is
        not a measurement.

        `count` exists because several buffers here are allocated at a
        capacity and used at a length: `partition_stats` is sized for
        `max_leaves` and holds `len(leaves)` of them. Hashing the tail would
        fold uninitialized memory into the record, which differs run to run
        on ONE machine and would make the instrument report divergence
        everywhere. When a buffer is used short, PASS THE LENGTH.
        """
        if not self.enabled:
            return
        var n = count if count >= 0 else len(buf)
        if n > len(buf):
            raise Error(
                String("identity_trace: tag '")
                + String(tag)
                + "' asked for "
                + String(n)
                + " elements of a buffer holding "
                + String(len(buf))
            )
        var host = ctx.enqueue_create_host_buffer[dt](n)
        if n == len(buf):
            ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=buf)
        else:
            var view = buf.create_sub_buffer[dt](0, n)
            ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=view)
        ctx.synchronize()
        var p = host.unsafe_ptr()
        self._emit(
            tag,
            _dtype_name[dt](),
            n,
            p.bitcast[UInt8](),
            n * size_of[Scalar[dt]](),
        )
        # `[[mojo-buffer-freed-at-last-use]]`: `host` is dead at
        # `.unsafe_ptr()` unless something uses it later, and a freed host
        # buffer under an in-flight read hashes garbage. Keep it alive past
        # the hash.
        _ = host^

    def record_host[
        dt: DType, o: MutOrigin, //
    ](
        mut self,
        tag: StringSlice,
        ptr: UnsafePointer[Scalar[dt], o],
        count: Int,
    ) raises:
        """Hash `count` elements already on the host."""
        if not self.enabled:
            return
        self._emit(
            tag,
            _dtype_name[dt](),
            count,
            ptr.bitcast[UInt8](),
            count * size_of[Scalar[dt]](),
        )

    def record_scalar_f32(mut self, tag: StringSlice, v: Float32) raises:
        """One host float, by its bits. For the scalars that steer a tree --
        a chosen scale, a score standard deviation, a magnitude bound --
        where a difference is invisible in any buffer but moves every cell
        downstream of it."""
        if not self.enabled:
            return
        var one = List[Float32]()
        one.append(v)
        self.record_host(tag, one.unsafe_ptr(), 1)

    def record_list_f32(
        mut self, tag: StringSlice, values: List[Float32]
    ) raises:
        # A MUTABLE LOCAL COPY, not origin gymnastics. `record_host` wants a
        # mutable-origin pointer and a borrowed `List` yields an immutable
        # one; on a debug path that drains the queue for every record, one
        # host copy of a leaf-sized list is not a cost worth a generic
        # signature nobody can read.
        if not self.enabled:
            return
        var tmp = values.copy()
        self.record_host(tag, tmp.unsafe_ptr(), len(tmp))
        _ = tmp^

    def record_list_i32(
        mut self, tag: StringSlice, values: List[Int32]
    ) raises:
        if not self.enabled:
            return
        var tmp = values.copy()
        self.record_host(tag, tmp.unsafe_ptr(), len(tmp))
        _ = tmp^


# =====================================================================
# READING A TRACE BACK, IN MOJO
#
# Added 2026-08-22 by the DEPTHWISE lane, at the foot, behind its own
# header. `tools/identity_trace_diff.py` is the real reader -- it aligns on
# tag SEQUENCES and classifies each differing cell, which is the diagnosis
# where a tag is only the location -- and nothing here competes with it.
#
# What these two functions are for is the case Python cannot serve: a MOJO
# CHECK that produces two traces in one process and has to RAISE on the
# difference. `original/identity_trace_check.mojo` already needed exactly
# this and wrote the loop inline (`:71` and `:262-283`); the depthwise
# probe needed it second. Two inline copies is how a third gets written, so
# it lives here where both callers can reach it. THE INLINE COPY IN THAT
# CHECK CAN NOW BE DELETED -- that is the lossguide lane's call and its
# file, so it is named rather than done.
# =====================================================================


def read_trace_lines(path: String) raises -> List[String]:
    """Every RECORD line of a trace, comments and blanks dropped.

    `header()` writes `#` comments and the differ skips them, so a reader
    that keeps them compares provenance -- the dataset, the backend, the
    parameters -- and reports a divergence for two runs that agree on every
    number. Provenance is what those lines are FOR; it is not a record.
    """
    var out = List[String]()
    with open(path, "r") as fh:
        var body = fh.read()
        var cur = String("")
        for i in range(body.byte_length()):
            var c = String(body[byte=i])
            if c == "\n":
                _keep_record(cur, out)
                cur = String("")
            else:
                cur += c
        _keep_record(cur, out)
    return out^


def _keep_record(line: String, mut out: List[String]):
    """Append `line` unless it is blank or a `#` comment."""
    if line.byte_length() == 0:
        return
    if line.startswith("#"):
        return
    out.append(line)


def first_divergence(path_a: String, path_b: String) raises -> String:
    """The first record on which two traces disagree, or "" if they agree.

    Returns the two lines joined by `   VS   `, which is enough to read the
    tag, the dtype, the count and both hashes at a glance.

    A LENGTH MISMATCH IS REPORTED AS SUCH rather than compared line by line.
    Different lengths mean the two runs took a different number of stages --
    a different level count, a different number of leaves split -- which is
    a STRUCTURAL divergence, and a tag-by-tag diff past that point pairs
    records that were never meant to correspond. That is the failure mode
    `tools/identity_trace_diff.py` exists to handle properly with sequence
    alignment; this function's job is to say "go run that one".
    """
    var a = read_trace_lines(path_a)
    var b = read_trace_lines(path_b)
    if len(a) != len(b):
        return (
            String("<record counts differ: ")
            + String(len(a))
            + " vs "
            + String(len(b))
            + " -- STRUCTURAL divergence, the two runs did not take the"
            " same stages. Run tools/identity_trace_diff.py, which aligns"
            " on tag sequences instead of by position.>"
        )
    for i in range(len(a)):
        if a[i] != b[i]:
            return String(a[i]) + "   VS   " + String(b[i])
    return String("")
