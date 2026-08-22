"""Name every stage of a fit and hash what it produced, so a bitwise
disagreement between two GPUs says WHERE.

NO CATBOOST COUNTERPART. CatBoost's GPU learner is `non_deterministic` by
declaration -- their histograms flush through float `atomicAdd` and they make
no cross-device claim at all -- so they never needed this. It belongs to the
`IDENTICAL` mode, which is ours, and therefore to `mojo_only/`.

THE PROBLEM IT SOLVES
---------------------
`IDENTITY_PATHS.md` enumerates every pathway that can move a bit and says
what `IDENTICAL` does about each. `depthwise_check` claim 6 then asks the
end-to-end question -- same fit, two core counts, same model? -- and answers
yes or no.

**When the answer is NO, neither artifact tells you where.** A model that
differs in one leaf value could have diverged in the histogram accumulation,
in the scan, in the sibling subtraction, in the score, in the argmax, in the
partition, or in the leaf-value reduce. That is seven candidate stages and
the only tool for choosing between them has been to comment code out and
re-run, which is what cost this lane an hour on 2026-08-22 (three probes
pinning `replication_for` and the split-chain grids, chasing a divergence
that turned out to be the CHECK reusing a mutated fixture).

So: every stage emits a line. Two columns' logs go through `diff`, and the
FIRST differing tag is the stage that broke. Everything after it is
downstream noise and can be ignored.

    #dw/d0.hist.built        3f8a1c0d5e2b7a94  n=9472
    #dw/d0.hist.scanned      c1d4e88f0a23b567  n=9472
    #dw/d0.partstats         77b0a1e3c9d2f405  n=32
    #dw/d0.scores            0e5f3a2b81c4d769  n=32
    ...

HOW TO USE IT ACROSS TWO MACHINES
---------------------------------
    # on the Mac
    pixi run check-depthwise-digest > /tmp/dw-metal.txt
    # on the CUDA box (tools/remote_gpu.sh)
    pixi run check-depthwise-digest > /tmp/dw-cuda.txt

    diff /tmp/dw-metal.txt /tmp/dw-cuda.txt | head -4

The first `<`/`>` pair names the stage. `n=` is the element count, so a
differing `n` is a STRUCTURAL divergence (a different number of leaves, a
different plan) and not a numeric one -- which is a different bug and worth
distinguishing at a glance.

WHY THE HASH IS COMPUTED ON THE HOST
------------------------------------
It is a debugging facility, and a device-side digest would be a second
reduction whose own order could differ between the machines being compared.
A host fold over a plain copy cannot: it walks indices 0..n-1 in order, one
thread, on the CPU. **The digest must never be the thing that disagrees.**

The cost is a drain and a copy per stage, which is why this is OFF by
default and why the flag is a RUNTIME parameter and not a `comptime`.
`IDENTITY_PATHS.md`'s first finding was a toggle that could not be selected
because five files each declared their own `comptime`; a debugging switch
that needs a rebuild is a debugging switch nobody uses.

WHAT THE HASH IS OVER
---------------------
RAW BITS, always. Floats go through `to_bits()`, never through a decimal
(`String(Float32)` does not round-trip in this Mojo). Two values that print
the same and differ in the last bit produce different digests, which is the
entire point.

FNV-1a, 64-bit, over 32-bit words in index order. Not a cryptographic
choice -- it is a choice about REPRODUCIBILITY: the constants are fixed, the
arithmetic is integer, and `UInt64` wraparound is defined, so the same bytes
give the same digest on every machine that will ever run this. A float
checksum would not.

READING A DIGEST OF A NONDETERMINISTIC STAGE
--------------------------------------------
Under `NUMERIC_FAST` on a backend with float threadgroup atomics, a
histogram digest is EXPECTED to vary run to run on the same machine. That is
not the tool failing; it is the tool reporting what FAST is. Run the same
column twice first: any tag that differs from itself is nondeterministic and
cannot be used to compare columns until the mode is `IDENTICAL`.
"""

from std.memory import bitcast
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer


#: FNV-1a 64-bit, the standard constants. Fixed forever: changing either one
#: invalidates every digest anyone has recorded in a results file.
comptime FNV_OFFSET_BASIS = UInt64(14695981039346656037)
comptime FNV_PRIME = UInt64(1099511628211)


def _fnv_word(acc: UInt64, word: UInt32) -> UInt64:
    """One 32-bit word into the accumulator, LOW BYTE FIRST.

    Byte order is spelled out rather than inherited from the platform. Every
    machine this port targets is little-endian today, and a digest whose
    value depends on that fact would silently disagree on the first big-endian
    column and look like a numeric divergence. Four explicit shifts cost
    nothing and mean the digest is a property of the VALUE.
    """
    var a = acc
    var w = word
    for _ in range(4):
        a = a ^ UInt64(w & UInt32(0xFF))
        a = a * FNV_PRIME
        w = w >> 8
    return a


struct TStageDigest(Movable):
    """One fit's digest log. Disabled costs one Bool test per stage.

    `scope` prefixes every tag so two lanes' logs can be concatenated and
    still diff cleanly -- `#dw/...` for depthwise, `#lg/...` for lossguide,
    `#sym/...` for the symmetric arm if it ever grows one.
    """

    var ctx: DeviceContext
    var enabled: Bool
    var scope: String
    var count: Int
    """How many tags have been emitted. Read by the probe, which asserts a
    LADDER WAS ACTUALLY WALKED -- a run that emits nothing and a run whose
    stages all agree are the same empty diff, and only this number tells
    them apart."""
    var quiet: Bool
    """Suppress the printed line and only accumulate `log`. Used when two
    ladders are compared IN ONE PROCESS and only the first difference is
    worth showing."""
    var log: List[String]
    """Every line emitted, in order. This is what makes the ladder a
    LOCALIZER rather than a log: `first_difference` walks two of these and
    names the first tag that disagrees, which is the stage that broke.
    Everything after it is downstream noise."""

    def __init__(
        out self,
        var ctx: DeviceContext,
        enabled: Bool,
        var scope: String,
        quiet: Bool = False,
    ):
        self.ctx = ctx^
        self.enabled = enabled
        self.scope = scope^
        self.count = 0
        self.quiet = quiet
        self.log = List[String]()

    def _print(mut self, tag: String, dtype: String, acc: UInt64, n: Int):
        """One record. THE FORMAT IS THE LOSSGUIDE LANE'S, deliberately.

            <seq>\t<tag>\t<dtype>\t<count>\t<fnv1a64 as 16 hex>

        ================= WHY IT IS THEIRS AND NOT MINE =================
        Both lanes built this instrument inside the same hour without
        knowing -- `core/identity_trace.mojo` there, this file here -- with
        the same three arguments arrived at independently: fold on the HOST
        so the digest can never be the thing that disagrees, a RUNTIME flag
        rather than a `comptime` because of `IDENTITY_PATHS.md`'s
        unreachable-toggle finding, and the first differing tag as the
        answer.

        Two implementations of one instrument is the exact drift surface
        both lanes have been writing rules about all day, so they are
        converged rather than left side by side. Theirs owns the FORMAT and
        the READER: `tools/identity_trace_diff.py` aligns two traces on tag
        SEQUENCES (a plain `diff` degrades badly when the two runs have
        different stage sets, which is the interesting case), and it
        classifies each differing cell -- DENORMAL-vs-ZERO / SIGN /
        NAN-payload / ULP<=n / LARGE. **That classification is the
        diagnosis, where a hash is only the location**: an all-denormal
        divergence is `IDENTITY_PATHS.md` row 10 and a mode difference,
        while a scattered 1-ULP divergence is a summation order, and those
        two send you to different files.

        This file keeps only what theirs cannot do: `first_difference` over
        two ladders IN ONE PROCESS, with no files at all, which is how
        `sm_count_override` earns its keep.

        `seq` is emitted because their aligner reads it. Tags must be
        UNIQUE within a trace -- their aligner pairs by tag sequence, so a
        repeated tag lets it match one level's record against another's --
        which is why every stage here carries a `dNN.` prefix and why a
        boosting caller will have to add a `treeNN.` one.
        ================================================================
        """
        self.count += 1
        var line = (
            String(self.count - 1)
            + "\t" + self.scope + "." + tag
            + "\t" + dtype
            + "\t" + String(n)
            + "\t" + _hex16(acc)
        )
        self.log.append(line)
        if not self.quiet:
            print(line)

    def emit_f32(
        mut self,
        tag: String,
        buf: DeviceBuffer[DType.float32],
        buf_len: Int,
        n: Int,
    ) raises:
        """Hash the first `n` floats of a `buf_len`-element buffer, BY BITS.

        ================= WHY BOTH LENGTHS =================
        `enqueue_copy(dst_ptr=..., src_buf=...)` copies the WHOLE device
        buffer -- there is no count -- so the staging buffer has to be the
        buffer's full size or the copy runs off the end of it. But the part
        worth hashing is usually the LIVE PREFIX: the histogram plane is
        allocated for `max_leaves` and a depthwise level has fewer, and the
        slots past the frontier hold whatever the last tree left there.

        Hashing the stale tail would make the digest depend on history
        instead of on this stage, and two columns that agree completely
        would still diff. So: copy `buf_len`, hash `n`.
        ====================================================

        `to_bits()` and not a decimal: the whole reason this file exists is
        to see a last-bit difference, and `String(Float32)` does not
        round-trip in this Mojo (`mojo-string-float-roundtrip`).
        """
        if not self.enabled:
            return
        var h = self.ctx.enqueue_create_host_buffer[DType.float32](buf_len)
        self.ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=buf)
        self.ctx.synchronize()
        var acc = FNV_OFFSET_BASIS
        for i in range(n):
            acc = _fnv_word(
                acc, UInt32(h.unsafe_ptr().unsafe_load(i).to_bits())
            )
        self._print(tag, String("f32"), acc, n)
        # keep-alive past the drain: `mojo-buffer-freed-at-last-use`
        _ = h^

    def emit_u32(
        mut self,
        tag: String,
        buf: DeviceBuffer[DType.uint32],
        buf_len: Int,
        n: Int,
    ) raises:
        if not self.enabled:
            return
        var h = self.ctx.enqueue_create_host_buffer[DType.uint32](buf_len)
        self.ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=buf)
        self.ctx.synchronize()
        var acc = FNV_OFFSET_BASIS
        for i in range(n):
            acc = _fnv_word(acc, h.unsafe_ptr().unsafe_load(i))
        self._print(tag, String("u32"), acc, n)
        _ = h^

    def emit_i32(
        mut self,
        tag: String,
        buf: DeviceBuffer[DType.int32],
        buf_len: Int,
        n: Int,
    ) raises:
        """The fixed-point accumulator plane.

        Cast through `UInt32` and NOT through a widening chain: Mojo
        ZERO-extends where C++ `(ui64)(int)` sign-extends
        (`mojo-int-widening-sign-extends`), so a negative accumulator cell
        hashed via `UInt64(Int(x))` would digest differently on a build
        where the chain took the other route. One 32-bit reinterpret, no
        widening.
        """
        if not self.enabled:
            return
        var h = self.ctx.enqueue_create_host_buffer[DType.int32](buf_len)
        self.ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=buf)
        self.ctx.synchronize()
        var acc = FNV_OFFSET_BASIS
        for i in range(n):
            acc = _fnv_word(
                acc, bitcast[DType.uint32](h.unsafe_ptr().unsafe_load(i))
            )
        self._print(tag, String("i32"), acc, n)
        _ = h^

    def emit_u8(
        mut self,
        tag: String,
        buf: DeviceBuffer[DType.uint8],
        buf_len: Int,
        n: Int,
    ) raises:
        """Per-row side flags and one-hot marks."""
        if not self.enabled:
            return
        var h = self.ctx.enqueue_create_host_buffer[DType.uint8](buf_len)
        self.ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=buf)
        self.ctx.synchronize()
        var acc = FNV_OFFSET_BASIS
        for i in range(n):
            acc = acc ^ UInt64(h.unsafe_ptr().unsafe_load(i))
            acc = acc * FNV_PRIME
        self._print(tag, String("u8"), acc, n)
        _ = h^

    def emit_host_u32(mut self, tag: String, values: List[UInt32]):
        """A HOST list -- the level plan, the per-leaf winners.

        Host state is worth digesting for the same reason device state is:
        a level plan that pairs different siblings on two machines is a
        divergence, and it happens BEFORE any kernel runs, so no device
        digest can see it.
        """
        if not self.enabled:
            return
        var acc = FNV_OFFSET_BASIS
        for i in range(len(values)):
            acc = _fnv_word(acc, values[i])
        self._print(tag, String("u32"), acc, len(values))

    def emit_host_i32(mut self, tag: String, values: List[Int32]):
        if not self.enabled:
            return
        var acc = FNV_OFFSET_BASIS
        for i in range(len(values)):
            acc = _fnv_word(acc, bitcast[DType.uint32](values[i]))
        self._print(tag, String("i32"), acc, len(values))

    def emit_host_f32(mut self, tag: String, values: List[Float32]):
        if not self.enabled:
            return
        var acc = FNV_OFFSET_BASIS
        for i in range(len(values)):
            acc = _fnv_word(acc, UInt32(values[i].to_bits()))
        self._print(tag, String("f32"), acc, len(values))

    def note(mut self, tag: String, value: Int):
        """A scalar the ladder should carry -- a leaf count, a plan size.

        Emitted through the same line format with `n=` holding the value, so
        a structural divergence shows up in the same diff as a numeric one
        rather than needing a second log.
        """
        if not self.enabled:
            return
        var acc = _fnv_word(FNV_OFFSET_BASIS, UInt32(value))
        self._print(tag, String("scalar"), acc, value)


def _hex16(v: UInt64) -> String:
    """Sixteen lowercase hex digits, fixed width.

    Fixed width so the digests line up in a terminal and so `diff` output is
    readable at a glance; lowercase so two logs produced by different tools
    cannot differ in case alone.
    """
    var digits = String("0123456789abcdef")
    var out = String("")
    for i in range(16):
        var shift = (15 - i) * 4
        var nib = Int((v >> UInt64(shift)) & UInt64(0xF))
        out += digits[byte=nib]
    return out^


def first_difference(a: TStageDigest, b: TStageDigest) raises -> String:
    """The first tag on which two ladders disagree, or "" if they agree.

    THIS IS THE POINT OF THE WHOLE FILE. `depthwise_check` claim 6 answers
    "do these two configurations build the same model"; when the answer is
    no, this answers "starting where".

    Read the result like this:

        d2.hist.scanned   the histogram diverged at level 2. Everything
                          downstream -- scores, splits, partitions, the
                          model -- is a CONSEQUENCE and tells you nothing.
        d2.best.gain      the device agreed and the HOST reduce did not:
                          look at `best_split_properties_less` and at the
                          sign conversion at its call site, not at a kernel.
        d2.rowindex       the histogram and the scores agreed and the rows
                          ended up ordered differently: the stable
                          partition, not the arithmetic.
        model.nodes       every device stage and every host reduce agreed
                          and the MODEL differs: the path fold or the
                          pre-order flatten, both pure host code.

    A LADDER OF A DIFFERENT LENGTH is reported as such rather than compared
    line by line: it means the two runs took a different number of levels or
    split a different number of leaves, which is a STRUCTURAL divergence and
    the tag-by-tag diff below it would be meaningless.
    """
    if len(a.log) != len(b.log):
        return (
            String("<ladder length ")
            + String(len(a.log))
            + " vs "
            + String(len(b.log))
            + " -- structural divergence, the runs did not take the same"
            " path; compare the `n=` fields of the shared prefix>"
        )
    for i in range(len(a.log)):
        if a.log[i] != b.log[i]:
            return String(a.log[i]) + "   VS   " + String(b.log[i])
    return String("")
