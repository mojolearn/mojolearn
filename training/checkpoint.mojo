# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The checkpoint file of `mojolearn.identical.train.ckpt.file.v1`.

**THIS FILE HAS NEVER BEEN COMPILED AND HAS NEVER BEEN EXECUTED.** No
compiler has read it, no byte has ever been written to disk by it, and no
file has ever been loaded back. Every "refuses" below is a PREDICTION.
Written 2026-09-03 by the checkpoint-file lane, DEVIATIONS 2050 through 2069,
PROPOSED and not yet in the orchestrator's ledger. The specification is
`training/CHECKPOINT_FORMAT.md` and the gate is
`training/checks/checkpoint_check.mojo`, which has also never run.

WHAT THIS IS
------------
`archive/plans/training/TRAINING_LOOP_PLAN.md:49` states the gap in one line: *"No
checkpoint file format, so clause (d) tests resume WITHIN one process."* The
training loop reaches `h_all = 463245ce6c97e68d` on an Apple M4 and an AMD
MI325X over eight steps, which is a result about a RUN. It is not a result
about a CHECKPOINT, because nothing the loop produces survives the process.

This file makes training state into bytes, so that "train here, resume there"
becomes something that can be tested at all. **A checkpoint written on Apple
and a checkpoint written on AMD from the same training state must be
byte-for-byte identical, and THAT EQUALITY IS THE TEST.** `cmp` is the
verdict.

WHY THERE IS NO `DeviceContext` IN THIS FILE
---------------------------------------------
`[[always-gpu-agnostic]]`. **This module does not import `max.gpu.host` and
has no `DeviceBuffer` or `DeviceContext` in any signature.** It cannot name a
device because it cannot see one, so no field of the format can be derived
from one -- not a pointer, not a vendor tag, not a driver version, not an SM
or CU count, not a block size, not a workspace size, not a buffer CAPACITY.
That is a structural guarantee rather than a convention, and it is the reason
the download from device to host lives in the CALLER. A serializer that took
a `DeviceContext` would be one refactor away from putting `ctx`'s properties
in the header, and a header field that differs between vendors turns the
central comparison into noise.

There is no arithmetic on floats here at all. This file moves bit patterns,
so there is no rounding decision in it that could diverge.

WHAT IS AND IS NOT HASHED
--------------------------
`h_all` is FNV-1a64 over the PAYLOAD ONLY -- param, then `m`, then `v`, one
continuous byte stream -- and it is deliberately the same number as
`train_loop.mojo::digest_of_lists(...).h_all`. It does not read `t`, the
seed, the learning rate, the arm, the tensor names, the offsets or the file
length, which is `TRAINING_LOOP_PLAN.md` section 3's exclusion list kept
intact. `h_file` covers every byte before itself, `h_all` included, and is
what a TRANSFER is checked with. Two hashes because they answer two different
questions, and conflating "corrupted in transit" with "different training
state" is how a corruption gets reported as a divergence.

REFUSE, DO NOT GUESS
---------------------
Every check raises. None warns, none falls back, none repairs. Each raise
begins with a stable uppercase tag (`CKPT_REFUSE_*` below) so a gate can
assert on the CATEGORY of a refusal -- **a refusal for the wrong reason is
not a pass.** The order is length, then magic, then version, then the fixed
sizes, then the file hash, then the layout, then the payload: **nothing
indexes into the buffer until the length check has passed.**

WHAT THIS FILE IS LEAST CONFIDENT COMPILES
-------------------------------------------
  1. `open(path, "w")` plus `fh.write_bytes(Span(bytes))` for BINARY, and
     `open(path, "r")` plus `fh.read_bytes()` to read it back. That pair is
     `core/identity_trace.mojo:311` and `checks/identity_trace_check.mojo:376`
     and is the only binary-file idiom this repository has; it is used here
     rather than a second one being invented. If a NUL byte does not survive
     the round trip, that is where.
  2. The narrowing constructor `UInt8((v >> UInt32(8)) & UInt32(0xFF))`.
     `train_loop.mojo::TrainDigest.words` narrows `UInt64` to `UInt32` the
     same way; this narrows one step further.
  3. `_fold_bytes` taking `mut` so that `List.unsafe_ptr()` yields the
     MUTABLE origin `fnv1a64_bytes` requires (DEVIATION 1578).
  4. `String(NAME_CHARS[byte=i])` for the tensor-name charset.
     `core/identity_trace.mojo::_hex16` uses exactly this spelling on a digit
     table, so it is borrowed rather than reinvented -- and `chr()` is NOT
     used, because it appears nowhere in this repository and this lane cannot
     compile it to find out.

THE MOJO TRAPS, EACH ONE CHECKED
---------------------------------
`[[mojo-string-not-indexable]]`: the magic number is a table of eight `UInt8`
VALUES, never a string literal that gets indexed.
`[[mojo-amp-plus-is-bitwise-and]]`: **there is not one `&+` in this file.**
Every byte assembly is `|` and `<<`; every arithmetic is a plain `+` or `*`.
`[[mojo-int-widening-sign-extends]]`: every field is UNSIGNED and every byte
read widens from `UInt8`, which zero-extends. If any of these were `Int8` the
top byte of a field with its high bit set would read back as `0xFFFFFF..`.
`[[mojo-string-float-roundtrip]]`: no float is ever formatted into this file,
the hyperparameters included.
`[[mojo-buffer-freed-at-last-use]]`: every `fnv1a64_bytes` call site holds
the owning list alive past the fold.
"""

from std.memory import bitcast

from core.identity_trace import FNV_OFFSET, fnv1a64_bytes


# ===========================================================================
# THE FORMAT CONSTANTS
# ===========================================================================
# **NOT ONE OF THESE IS A MODEL SHAPE.** `n_total = 13376` and `J = 11` are
# the v1 training shape and they appear NOWHERE in this file: the layout
# arrives from the caller and is written into the file. A second architecture
# (`TRAINING_LOOP_PLAN.md` section 12 item 9) should need no change here.
# That is a prediction and it has not been tried.

comptime CKPT_FORMAT_VERSION: UInt32 = 1
comptime CKPT_HEADER_BYTES = 64
comptime CKPT_DESCRIPTOR_BYTES = 64
comptime CKPT_LAYOUT_ENTRY_BYTES = 32
comptime CKPT_NAME_BYTES = 16
comptime CKPT_TRAILER_BYTES = 16
comptime CKPT_N_ARRAYS: UInt32 = 3
"""param, then `m`, then `v`. The ORDER is part of the specification: `h_all`
is one continuous fold over the payload and swapping two arrays changes it."""
comptime CKPT_ELEM_BITS: UInt32 = 32
"""FP32 only. No bf16, fp16, tf32 or float64, and no device float64 exists in
this repository's matrix anyway (`[[mojolearn-hardware-limits]]`)."""

comptime CKPT_MAGIC0: UInt8 = 0x4D  # 'M'
comptime CKPT_MAGIC1: UInt8 = 0x4C  # 'L'
comptime CKPT_MAGIC2: UInt8 = 0x43  # 'C'
comptime CKPT_MAGIC3: UInt8 = 0x4B  # 'K'
comptime CKPT_MAGIC4: UInt8 = 0x50  # 'P'
comptime CKPT_MAGIC5: UInt8 = 0x54  # 'T'
comptime CKPT_MAGIC6: UInt8 = 0x30  # '0'
comptime CKPT_MAGIC7: UInt8 = 0x31  # '1'
"""`MLCKPT01`, as eight BYTE VALUES and not as a string literal.
`[[mojo-string-not-indexable]]`: `s[i]` is refused in this toolchain, so a
literal would have to be walked with `[byte=i]` anyway and the byte values are
what the format actually specifies."""

comptime NAME_CHARS = "-.0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz"
"""The tensor-name charset, `core/identity_trace.mojo::_sanitize`'s set.

Deliberately narrow. A name is a token that has to survive a filename, a shell
argument and a `xxd` dump, and every one of `embed`, `norm1_w`, `w_q`, `w_k`,
`w_v`, `w_o`, `norm2_w`, `w_gate`, `w_up`, `w_down`, `lm_head` fits in fifteen
bytes of it with room to spare. The WRITER and the READER share this one
definition, so a name that can be written can be read."""


# ---- the refusal tags. A gate asserts on THESE and not on "something
# ---- raised", because a refusal for the wrong reason is not a pass.

comptime CKPT_REFUSE_MAGIC = "CKPT_REFUSE_MAGIC"
comptime CKPT_REFUSE_VERSION = "CKPT_REFUSE_VERSION"
comptime CKPT_REFUSE_HEADER = "CKPT_REFUSE_HEADER"
comptime CKPT_REFUSE_TRUNCATED = "CKPT_REFUSE_TRUNCATED"
comptime CKPT_REFUSE_TRAILING = "CKPT_REFUSE_TRAILING"
comptime CKPT_REFUSE_FILE_HASH = "CKPT_REFUSE_FILE_HASH"
comptime CKPT_REFUSE_LAYOUT = "CKPT_REFUSE_LAYOUT"
comptime CKPT_REFUSE_SHAPE = "CKPT_REFUSE_SHAPE"
comptime CKPT_REFUSE_CONTENT_HASH = "CKPT_REFUSE_CONTENT_HASH"
comptime CKPT_REFUSE_STATE = "CKPT_REFUSE_STATE"


def checkpoint_payload_begin(j_count: Int) -> Int:
    """Byte offset of the first parameter element."""
    return (
        CKPT_HEADER_BYTES
        + CKPT_DESCRIPTOR_BYTES
        + CKPT_LAYOUT_ENTRY_BYTES * j_count
    )


def checkpoint_file_bytes(j_count: Int, n_total: Int) -> Int:
    """The ONE formula. `64 + 64 + 32*J + 12*n + 16`.

    At `J = 11, n_total = 13376` this is 161008. A file of any other length
    for that shape is refused before a single float is read, and the four
    redundant size fields in the header exist so that this can be computed
    from the header alone.
    """
    return (
        checkpoint_payload_begin(j_count)
        + Int(CKPT_N_ARRAYS) * n_total * 4
        + CKPT_TRAILER_BYTES
    )


# ===========================================================================
# BYTES, BUILT AND READ BY HAND
# ===========================================================================
# **LITTLE-ENDIAN BY DEFINITION AND NOT BY THE HOST.** Every multi-byte field
# is assembled from an integer one byte at a time rather than copied out of
# host memory. A `memcpy` of a `List[Float32]` is little-endian only because
# every machine in this repository's matrix happens to be, and "happens to be"
# is not a specification. See CHECKPOINT_FORMAT.md section 2.


def _put_u8(mut out: List[UInt8], v: UInt8):
    out.append(v)


def _put_u32(mut out: List[UInt8], v: UInt32):
    """Four bytes, least significant first."""
    out.append(UInt8(v & UInt32(0xFF)))
    out.append(UInt8((v >> UInt32(8)) & UInt32(0xFF)))
    out.append(UInt8((v >> UInt32(16)) & UInt32(0xFF)))
    out.append(UInt8((v >> UInt32(24)) & UInt32(0xFF)))


def _put_u64(mut out: List[UInt8], v: UInt64):
    """Eight bytes, least significant first, through `_put_u32` twice so that
    there is ONE narrowing spelling in this file and not two."""
    _put_u32(out, UInt32(v & UInt64(0xFFFFFFFF)))
    _put_u32(out, UInt32((v >> UInt64(32)) & UInt64(0xFFFFFFFF)))


def _put_f32(mut out: List[UInt8], v: Float32):
    """A float BY ITS BITS. `[[mojo-string-float-roundtrip]]`: `String(Float32)`
    does not round trip in this toolchain, and a checkpoint that went through a
    decimal representation anywhere could report agreement across a real
    difference -- the worst failure an instrument of this kind can have."""
    _put_u32(out, bitcast[DType.uint32](v))


def _get_u32(b: List[UInt8], at: Int) -> UInt32:
    """Four bytes, least significant first.

    **`b[at]` is a `UInt8`, which is UNSIGNED, so widening to `UInt32`
    ZERO-extends.** `[[mojo-int-widening-sign-extends]]`: if these were `Int8`
    the top byte of any field with its high bit set would read back as
    `0xFFFFFF..`, `n_total` would be enormous, and the length check would
    refuse a perfectly good file. The unsignedness is load-bearing.
    """
    var v = UInt32(b[at])
    v = v | (UInt32(b[at + 1]) << UInt32(8))
    v = v | (UInt32(b[at + 2]) << UInt32(16))
    v = v | (UInt32(b[at + 3]) << UInt32(24))
    return v


def _get_u64(b: List[UInt8], at: Int) -> UInt64:
    var lo = UInt64(_get_u32(b, at))
    var hi = UInt64(_get_u32(b, at + 4))
    return lo | (hi << UInt64(32))


def _get_f32(b: List[UInt8], at: Int) -> Float32:
    return bitcast[DType.float32](_get_u32(b, at))


def _fold_bytes(
    mut buf: List[UInt8], begin: Int, count: Int, seed: UInt64
) raises -> UInt64:
    """FNV-1a64 over `buf[begin : begin + count]`.

    `core/identity_trace.mojo::fnv1a64_bytes`, CALLED and not restated. Byte
    at a time on purpose: a word-at-a-time variant is faster and is a
    DIFFERENT FUNCTION, and `tools/identity_trace_diff.py` recomputes this
    spelling from a `.bin` dump, so the writer and the reader have to be the
    same one.

    **`mut` is DEVIATION 1578's tax and NOT a claim that this writes.**
    `fnv1a64_bytes` takes `o: MutOrigin`, a borrowed `List` yields an
    immutable origin, and the two do not unify. `train_loop.mojo` pays for
    this with a `.copy()` of three `n_total` lists per digest; this file pays
    with a `mut` on a reader, which is cheaper and says something false. The
    honest fix -- an immutable-origin overload in `core/identity_trace.mojo`
    -- is OWED and would delete a copy at every call site in the tree.
    """
    if begin < 0 or count < 0 or begin + count > len(buf):
        raise Error(
            String("checkpoint: internal fold range [")
            + String(begin)
            + ", "
            + String(begin + count)
            + ") is outside a buffer of "
            + String(len(buf))
            + " bytes. This is a defect in checkpoint.mojo and not a"
            + " property of the file."
        )
    var h = fnv1a64_bytes(seed, buf.unsafe_ptr() + begin, count)
    # `[[mojo-buffer-freed-at-last-use]]`: hold the owner past the fold.
    _ = buf
    return h


def _name_char(c: Int) raises -> String:
    """The one printable character with byte value `c`, or a refusal.

    `String(NAME_CHARS[byte=i])` is `core/identity_trace.mojo::_hex16`'s
    proven spelling on a table. `chr()` is NOT used: it appears nowhere in
    this repository, and this lane cannot compile, so an unproven builtin is a
    risk with no upside. A linear scan of sixty-five characters, sixteen times
    per tensor, eleven tensors, is not a cost worth an index trick that
    `[[mojo-string-not-indexable]]` would make unreadable anyway.
    """
    var t = String(NAME_CHARS)
    for i in range(t.byte_length()):
        var ch = String(t[byte=i])
        if ord(ch) == c:
            return ch^
    raise Error(
        String(CKPT_REFUSE_LAYOUT)
        + ": a tensor name holds byte value "
        + String(c)
        + ", which is outside [A-Za-z0-9._-]. A name is a token that has to"
        + " survive a filename, a shell argument and a hex dump; anything"
        + " else is refused rather than sanitized, because a sanitized name"
        + " no longer matches the layout it is compared against."
    )


def _put_name(mut out: List[UInt8], name: String) raises:
    """Sixteen bytes: the name, then NUL padding to the full width."""
    # A LOCAL COPY before `as_bytes()`. `train_loop.mojo::env_int` takes the
    # same route (`var s = env_str(name); var b = s.as_bytes()`), and a Span
    # over a borrowed argument is one more origin question this lane cannot
    # answer without a compiler.
    var s = String(name)
    var b = s.as_bytes()
    var n = len(b)
    if n < 1 or n > CKPT_NAME_BYTES - 1:
        raise Error(
            String(CKPT_REFUSE_STATE)
            + ": tensor name '"
            + name
            + "' is "
            + String(n)
            + " bytes and the field holds 1 to "
            + String(CKPT_NAME_BYTES - 1)
            + " plus a NUL terminator. Refused at SAVE time on purpose:"
            + " discovering it at LOAD time on another continent is worse."
        )
    for i in range(n):
        # Validated through the SAME function the reader uses, so a name that
        # can be written can be read.
        _ = _name_char(Int(b[i]))
        out.append(b[i])
    for _ in range(CKPT_NAME_BYTES - n):
        out.append(UInt8(0))
    # `[[mojo-buffer-freed-at-last-use]]`: `b` views `s`, so `s` has to
    # outlive the loop that read through it.
    _ = s


def _get_name(b: List[UInt8], at: Int) raises -> String:
    """A NUL-padded name back out of sixteen bytes.

    Refuses a field that is not NUL TERMINATED and refuses one whose padding
    is not all zero. **Trailing junk after the NUL is a writer that disagrees
    with this format**, and accepting it is how two versions coexist silently.
    """
    var out = String("")
    var term = -1
    for i in range(CKPT_NAME_BYTES):
        if Int(b[at + i]) == 0:
            term = i
            break
        out += _name_char(Int(b[at + i]))
    if term < 0:
        raise Error(
            String(CKPT_REFUSE_LAYOUT)
            + ": a 16-byte tensor-name field has no NUL terminator, so the"
            + " name runs into the next field. The file was not written by"
            + " this format."
        )
    if term == 0:
        raise Error(
            String(CKPT_REFUSE_LAYOUT)
            + ": a tensor-name field is EMPTY. Every tensor is named; an"
            + " unnamed one cannot be reported by name in a shape refusal,"
            + " which is the whole reason the field exists."
        )
    for k in range(term, CKPT_NAME_BYTES):
        if Int(b[at + k]) != 0:
            raise Error(
                String(CKPT_REFUSE_LAYOUT)
                + ": the padding after a tensor name is not all zero (byte "
                + String(k)
                + " of the field). Junk after the NUL is a writer that"
                + " disagrees with this format."
            )
    return out^


def _hex16(v: UInt64) -> String:
    """Sixteen lowercase hex digits, most significant first.

    `core/identity_trace.mojo::_hex16`'s spelling and
    `train_loop.mojo::hex16`'s, restated because both are private to their
    modules. The three must agree; `checkpoint_check.mojo` asserts this one
    against `train_loop.hex16` on a fixed value, the same way
    `train_step_check.mojo` clause (e) does for the other pair.
    """
    comptime DIGITS = "0123456789abcdef"
    var out = String("")
    for i in range(16):
        var nib = Int((v >> UInt64(60 - 4 * i)) & UInt64(0xF))
        out += String(DIGITS[byte=nib])
    return out^


# ===========================================================================
# THE STATE, ON THE HOST
# ===========================================================================


struct Checkpoint(Movable):
    """One checkpoint's whole content, on the HOST and nowhere else.

    **THE FLAT BUFFER IS THE STATE** (DEVIATION 1553). `param`, `m_state` and
    `v_state` are the three flat `n_total` arrays the optimizer writes; the
    eleven per-tensor weight buffers are a VIEW refreshed at the top of every
    step, and a checkpoint of the view would be a checkpoint of something the
    optimizer never wrote -- `TRAINING_LOOP_PLAN.md` failure mode V3.

    Not carried, because it does not exist between steps: the gradient
    (recomputed every step), every activation, and the KV cache (DEVIATION
    1554 -- a fresh one at `pos0 = 0` every step). Not carried because it is
    not state: any RNG stream, since `train_batch_ids(seed, step_index)` is a
    pure function of two integers, so `(seed, t)` reproduces every future
    batch. **That stops being true the moment anything stochastic enters** and
    then this struct is INCOMPLETE rather than merely inconvenient.

    A default-constructed `Checkpoint` is EMPTY and `save_checkpoint` refuses
    it. Fill every field; `validate` names the one that is missing.
    """

    var t: Int
    """Optimizer steps COMPLETED. ONE-BASED, so the next step is `t + 1` and a
    file written before any step has `t = 0`. Off by one here moves `beta^t`,
    whose two spellings the optimizer contract separates agree exactly through
    `t = 6` and first differ at `t = 7` -- so the error is INVISIBLE in any run
    shorter than seven steps and then appears at step seven of eight."""

    var seed: UInt64
    var opt_kind: Int
    var lr: Float32
    var beta1: Float32
    var beta2: Float32
    var eps: Float32
    var weight_decay: Float32
    var momentum: Float32
    var dampening: Float32
    var nesterov: Bool
    var max_norm: Float32
    var steps_planned: Int
    var arm: Int
    """The descriptor. NOT carried state, NOT in `h_all`, and here for exactly
    one job: so a loader can REFUSE a resume under a different configuration.
    `TRAINING_LOOP_PLAN.md` section 7 item 4 states the hole -- *"comparing two
    digests produced under different `lr` is meaningless and the harness cannot
    detect it"* -- and a file can be checked by a machine where a card header
    has to be checked by eye."""

    var offsets: List[Int]
    """Length `J + 1`. `offsets[j] .. offsets[j+1]` is tensor `j`."""
    var names: List[String]
    """Length `J`, in ascending `param_id`."""
    var buf_initialized: List[Bool]
    """Length `J`. The optimizer's per-tensor momentum flags.

    **Under AdamW these are never read** -- `identical_optimizer_step`
    consults them only on the SGD branch -- so under the v1 configuration they
    are carried and bit-inert, and the gate's flag control is REPORTED and not
    asserted for that reason (`[[reached-but-inert]]`). They are written
    anyway because dropping the flags on the way to disk is
    `OPT_SAB_RESUME_REINIT`'s named failure, and a format with nowhere to put
    them makes that failure unfixable rather than untested."""

    var param: List[Float32]
    var m_state: List[Float32]
    var v_state: List[Float32]

    def __init__(out self):
        self.t = 0
        self.seed = UInt64(0)
        self.opt_kind = 0
        self.lr = Float32(0.0)
        self.beta1 = Float32(0.0)
        self.beta2 = Float32(0.0)
        self.eps = Float32(0.0)
        self.weight_decay = Float32(0.0)
        self.momentum = Float32(0.0)
        self.dampening = Float32(0.0)
        self.nesterov = False
        self.max_norm = Float32(0.0)
        self.steps_planned = 0
        self.arm = 0
        self.offsets = List[Int]()
        self.names = List[String]()
        self.buf_initialized = List[Bool]()
        self.param = List[Float32]()
        self.m_state = List[Float32]()
        self.v_state = List[Float32]()

    def n_total(self) -> Int:
        return len(self.param)

    def j_count(self) -> Int:
        return len(self.names)

    def validate(self) raises:
        """Everything a save would have to be true for, checked BEFORE any
        byte is written.

        Refusing at SAVE time is cheaper than discovering it at LOAD time on
        another continent, and a half-written file that loaded would be the
        worst outcome available.
        """
        var n = len(self.param)
        var j = len(self.names)
        if n < 1:
            raise Error(
                String(CKPT_REFUSE_STATE)
                + ": the parameter array is EMPTY. A checkpoint of zero"
                + " elements would hash to the FNV offset basis on every"
                + " machine, which is three vendors agreeing about nothing."
            )
        if j < 1:
            raise Error(
                String(CKPT_REFUSE_STATE)
                + ": there are no tensors in the layout."
            )
        if len(self.m_state) != n or len(self.v_state) != n:
            raise Error(
                String(CKPT_REFUSE_STATE)
                + ": param, m and v must be the same length, got "
                + String(n)
                + "/"
                + String(len(self.m_state))
                + "/"
                + String(len(self.v_state))
                + ". A short moment array would be padded with whatever the"
                + " allocator left there, which differs run to run on ONE"
                + " machine."
            )
        if len(self.offsets) != j + 1:
            raise Error(
                String(CKPT_REFUSE_STATE)
                + ": offsets has "
                + String(len(self.offsets))
                + " entries and J + 1 is "
                + String(j + 1)
            )
        if len(self.buf_initialized) != j:
            raise Error(
                String(CKPT_REFUSE_STATE)
                + ": buf_initialized has "
                + String(len(self.buf_initialized))
                + " entries and J is "
                + String(j)
            )
        if self.offsets[0] != 0:
            raise Error(
                String(CKPT_REFUSE_STATE)
                + ": offsets[0] is "
                + String(self.offsets[0])
                + " and must be 0."
            )
        for k in range(j):
            if self.offsets[k + 1] <= self.offsets[k]:
                raise Error(
                    String(CKPT_REFUSE_STATE)
                    + ": offsets is not strictly increasing at j = "
                    + String(k)
                    + " ("
                    + String(self.names[k])
                    + ")."
                )
        if self.offsets[j] != n:
            raise Error(
                String(CKPT_REFUSE_STATE)
                + ": offsets spans "
                + String(self.offsets[j])
                + " elements and param holds "
                + String(n)
                + ". **A wrong offset gives plausible, in-bounds, wrong"
                + " numbers that are IDENTICAL on all three vendors**, so no"
                + " cross-vendor comparison could see this"
                + " (TRAINING_LOOP_PLAN.md V7)."
            )
        if self.t < 0:
            raise Error(
                String(CKPT_REFUSE_STATE)
                + ": t is "
                + String(self.t)
                + " and the step counter is one-based and non-negative."
            )
        if self.steps_planned < 0 or self.arm < 0:
            raise Error(
                String(CKPT_REFUSE_STATE)
                + ": steps_planned and arm are non-negative."
            )
        if self.opt_kind < 0:
            raise Error(
                String(CKPT_REFUSE_STATE) + ": opt_kind is non-negative."
            )


@fieldwise_init
struct CheckpointHashes(Copyable, Movable):
    """What a save produced, so the caller can record it without re-reading
    the file it just wrote."""

    var h_all: UInt64
    """FNV-1a64 over the PAYLOAD ONLY. The same number as
    `train_loop.mojo::digest_of_lists(...).h_all`, deliberately."""
    var h_file: UInt64
    """FNV-1a64 over every byte before this field, `h_all` included."""
    var bytes: Int

    def h_all_hex(self) -> String:
        return _hex16(self.h_all)

    def h_file_hex(self) -> String:
        return _hex16(self.h_file)


# ===========================================================================
# SAVE
# ===========================================================================


def save_checkpoint(path: String, ck: Checkpoint) raises -> CheckpointHashes:
    """Write `ck` to `path` and return its two hashes.

    THE WHOLE FILE IS BUILT IN ONE `List[UInt8]` AND WRITTEN ONCE. There is no
    seek and no buffered writer in this toolchain's surface as this repository
    uses it, and `core/identity_trace.mojo:311`'s
    `open(path, "w")` + `write_bytes(Span(bytes))` is the only binary idiom
    the tree has. At 161 KB that is the right shape anyway; at a real model
    size it is the first thing that would have to change, and
    `CHECKPOINT_FORMAT.md` section 6 item 9 says so rather than hiding it.

    **NO ATOMICITY.** No temp file, no rename, no lock. A crash mid-write
    leaves a short file that `load_checkpoint` REFUSES as truncated, which is
    the intended behavior: a half-written checkpoint that loaded would be
    worse than one that did not.
    """
    if path == "":
        raise Error(
            String(CKPT_REFUSE_STATE)
            + ": save_checkpoint was given an empty path. Silently doing"
            + " nothing here is how a leg reports 'the checkpoint was"
            + " written' about a file that does not exist."
        )
    ck.validate()

    var j = ck.j_count()
    var n = ck.n_total()
    var body = List[UInt8]()

    # ---- header, 64 bytes ------------------------------------------------
    _put_u8(body, CKPT_MAGIC0)
    _put_u8(body, CKPT_MAGIC1)
    _put_u8(body, CKPT_MAGIC2)
    _put_u8(body, CKPT_MAGIC3)
    _put_u8(body, CKPT_MAGIC4)
    _put_u8(body, CKPT_MAGIC5)
    _put_u8(body, CKPT_MAGIC6)
    _put_u8(body, CKPT_MAGIC7)
    _put_u32(body, CKPT_FORMAT_VERSION)
    _put_u32(body, UInt32(CKPT_HEADER_BYTES))
    _put_u32(body, UInt32(CKPT_DESCRIPTOR_BYTES))
    _put_u32(body, UInt32(CKPT_LAYOUT_ENTRY_BYTES * j))
    _put_u32(body, UInt32(Int(CKPT_N_ARRAYS) * n * 4))
    _put_u32(body, UInt32(CKPT_TRAILER_BYTES))
    _put_u32(body, UInt32(j))
    _put_u32(body, UInt32(n))
    _put_u32(body, CKPT_N_ARRAYS)
    _put_u32(body, CKPT_ELEM_BITS)
    _put_u32(body, UInt32(ck.t))
    _put_u32(body, UInt32(0))  # reserved0
    _put_u32(body, UInt32(0))  # reserved1
    _put_u32(body, UInt32(0))  # reserved2

    # ---- descriptor, 64 bytes -------------------------------------------
    # EVERY FLOAT BY ITS BITS. `lr` is `0x3A83126F`-shaped and never "0.001".
    _put_u64(body, ck.seed)
    _put_u32(body, UInt32(ck.opt_kind))
    _put_f32(body, ck.lr)
    _put_f32(body, ck.beta1)
    _put_f32(body, ck.beta2)
    _put_f32(body, ck.eps)
    _put_f32(body, ck.weight_decay)
    _put_f32(body, ck.momentum)
    _put_f32(body, ck.dampening)
    _put_u32(body, UInt32(1) if ck.nesterov else UInt32(0))
    _put_f32(body, ck.max_norm)
    _put_u32(body, UInt32(ck.steps_planned))
    _put_u32(body, UInt32(ck.arm))
    _put_u32(body, UInt32(0))  # reserved0
    _put_u32(body, UInt32(0))  # reserved1

    # ---- layout table, 32 bytes per tensor -------------------------------
    # ASCENDING param_id, and `param_id` IS the index. DEVIATION 1550: this
    # order is part of the checkpoint hash specification and changing it
    # invalidates every digest this profile has ever produced.
    for k in range(j):
        _put_u32(body, UInt32(k))
        _put_u32(body, UInt32(ck.offsets[k]))
        _put_u32(body, UInt32(ck.offsets[k + 1] - ck.offsets[k]))
        _put_u8(body, UInt8(1) if ck.buf_initialized[k] else UInt8(0))
        _put_u8(body, UInt8(0))
        _put_u8(body, UInt8(0))
        _put_u8(body, UInt8(0))
        _put_name(body, ck.names[k])

    var payload_at = len(body)
    if payload_at != checkpoint_payload_begin(j):
        raise Error(
            String("checkpoint: the writer produced a header of ")
            + String(payload_at)
            + " bytes and the formula says "
            + String(checkpoint_payload_begin(j))
            + ". This is a defect in checkpoint.mojo, caught before the file"
            + " reached disk."
        )

    # ---- payload, param then m then v ------------------------------------
    # **INDEX ORDER, THREE ARRAYS, NO PADDING.** Every element is its raw
    # IEEE-754 binary32 bit pattern, INCLUDING every NaN payload and both
    # signed zeros: `-0.0` is `00 00 00 80` and `+0.0` is `00 00 00 00` and
    # the format keeps them apart, because `m` starting at exactly `+0.0` is
    # what makes `OPT_SAB_MOMENT_LERP` bit-inert on a first step and a format
    # that normalized the sign would erase a distinction the optimizer
    # contract depends on.
    for i in range(n):
        _put_f32(body, ck.param[i])
    for i in range(n):
        _put_f32(body, ck.m_state[i])
    for i in range(n):
        _put_f32(body, ck.v_state[i])

    # ---- trailer ---------------------------------------------------------
    # `h_all` is ONE continuous fold over the payload, which is the same
    # number as `h(h(h(FNV_OFFSET, param, n), m, n), v, n)` because the three
    # arrays are contiguous and in that order.
    var payload_bytes = Int(CKPT_N_ARRAYS) * n * 4
    var h_all = _fold_bytes(body, payload_at, payload_bytes, FNV_OFFSET)
    _put_u64(body, h_all)
    var h_file = _fold_bytes(body, 0, len(body), FNV_OFFSET)
    _put_u64(body, h_file)

    var want = checkpoint_file_bytes(j, n)
    if len(body) != want:
        raise Error(
            String("checkpoint: the writer produced ")
            + String(len(body))
            + " bytes and the formula says "
            + String(want)
            + ". This is a defect in checkpoint.mojo, caught before the file"
            + " reached disk."
        )

    with open(path, "w") as fh:
        fh.write_bytes(Span(body))
    # `[[mojo-buffer-freed-at-last-use]]`: hold the byte list past the write.
    _ = body
    return CheckpointHashes(h_all, h_file, want)


# ===========================================================================
# LOAD
# ===========================================================================


def _read_all_bytes(path: String) raises -> List[UInt8]:
    """The whole file, as bytes.

    `checks/identity_trace_check.mojo:376`'s spelling exactly, copy loop
    included: that is the one place in this repository where a binary file
    written with `write_bytes` is read back with `read_bytes` and re-folded to
    its recorded FNV, so it is the spelling with evidence behind it.

    There is no partial read, no streaming and no seek. See
    `CHECKPOINT_FORMAT.md` section 6 item 9.
    """
    var raw = List[UInt8]()
    with open(path, "r") as fh:
        var body = fh.read_bytes()
        for k in range(len(body)):
            raw.append(body[k])
    return raw^


def read_checkpoint_unchecked(path: String) raises -> Checkpoint:
    """Every refusal EXCEPT the caller's shape expectation.

    Magic, version, the fixed sizes, the reserved fields, the file length,
    BOTH hashes and the layout's internal consistency are all enforced here.
    What is not enforced is the comparison against a layout the caller
    expects, because a tool that wants to look at a file whose shape it does
    not know has no expectation to offer.

    **THIS IS NOT A "LOAD ANYWAY" ESCAPE HATCH AND MUST NOT BE USED TO GET
    PAST A `CKPT_REFUSE_SHAPE`.** Use `load_checkpoint`.
    """
    var raw = _read_all_bytes(path)
    var size = len(raw)

    # ---- 1. LENGTH FIRST. Nothing below may index into `raw` until the
    # ---- header is known to be there.
    if size < CKPT_HEADER_BYTES:
        raise Error(
            String(CKPT_REFUSE_TRUNCATED)
            + ": '"
            + path
            + "' is "
            + String(size)
            + " bytes and the header alone is "
            + String(CKPT_HEADER_BYTES)
            + ". There is no atomic write in this format, so a crash"
            + " mid-write leaves exactly this. REFUSED rather than padded."
        )

    # ---- 2. magic ---------------------------------------------------------
    # Compared as `Int` rather than as `UInt8`: a comparison on a SIMD scalar
    # yields `SIMD[DType.bool, 1]`, and eight of those chained with `and` is
    # more spelling risk than this lane can carry without a compiler.
    var magic_bad = 0
    if Int(raw[0]) != Int(CKPT_MAGIC0):
        magic_bad += 1
    if Int(raw[1]) != Int(CKPT_MAGIC1):
        magic_bad += 1
    if Int(raw[2]) != Int(CKPT_MAGIC2):
        magic_bad += 1
    if Int(raw[3]) != Int(CKPT_MAGIC3):
        magic_bad += 1
    if Int(raw[4]) != Int(CKPT_MAGIC4):
        magic_bad += 1
    if Int(raw[5]) != Int(CKPT_MAGIC5):
        magic_bad += 1
    if Int(raw[6]) != Int(CKPT_MAGIC6):
        magic_bad += 1
    if Int(raw[7]) != Int(CKPT_MAGIC7):
        magic_bad += 1
    if magic_bad != 0:
        raise Error(
            String(CKPT_REFUSE_MAGIC)
            + ": '"
            + path
            + "' does not begin with MLCKPT01 (4D 4C 43 4B 50 54 30 31), it"
            + " begins with "
            + _hex16(UInt64(_get_u32(raw, 0)))
            + " / "
            + _hex16(UInt64(_get_u32(raw, 4)))
            + " as two little-endian words. A .trace, a .bin dump, a text"
            + " sidecar or a truncated download handed here by mistake would"
            + " otherwise be parsed as a header and produce a shape out of"
            + " noise."
        )

    # ---- 3. version -------------------------------------------------------
    var version = Int(_get_u32(raw, 8))
    if version != Int(CKPT_FORMAT_VERSION):
        raise Error(
            String(CKPT_REFUSE_VERSION)
            + ": '"
            + path
            + "' is format version "
            + String(version)
            + " and this reader is version "
            + String(Int(CKPT_FORMAT_VERSION))
            + ". **There is no migration path and there is not meant to be"
            + " one.** A v2 file read by a v1 reader is field-shifted garbage"
            + " that still has plausible float values in it."
        )

    # ---- 4. the fixed fields and the size arithmetic ---------------------
    var hdr_b = Int(_get_u32(raw, 12))
    var desc_b = Int(_get_u32(raw, 16))
    var layout_b = Int(_get_u32(raw, 20))
    var payload_b = Int(_get_u32(raw, 24))
    var trailer_b = Int(_get_u32(raw, 28))
    var j = Int(_get_u32(raw, 32))
    var n = Int(_get_u32(raw, 36))
    # Every field is taken as `Int` from here down. The bytes were read
    # UNSIGNED (`_get_u32`'s docstring says why that matters), and `Int` is
    # what the comparisons and the arithmetic below are written in.
    var n_arrays = Int(_get_u32(raw, 40))
    var elem_bits = Int(_get_u32(raw, 44))
    var t = Int(_get_u32(raw, 48))
    var res0 = Int(_get_u32(raw, 52))
    var res1 = Int(_get_u32(raw, 56))
    var res2 = Int(_get_u32(raw, 60))

    if hdr_b != CKPT_HEADER_BYTES or desc_b != CKPT_DESCRIPTOR_BYTES:
        raise Error(
            String(CKPT_REFUSE_HEADER)
            + ": header_bytes/descriptor_bytes are "
            + String(hdr_b)
            + "/"
            + String(desc_b)
            + " and this format's are "
            + String(CKPT_HEADER_BYTES)
            + "/"
            + String(CKPT_DESCRIPTOR_BYTES)
        )
    if trailer_b != CKPT_TRAILER_BYTES:
        raise Error(
            String(CKPT_REFUSE_HEADER)
            + ": trailer_bytes is "
            + String(trailer_b)
            + " and this format's is "
            + String(CKPT_TRAILER_BYTES)
        )
    if n_arrays != Int(CKPT_N_ARRAYS):
        raise Error(
            String(CKPT_REFUSE_HEADER)
            + ": n_arrays is "
            + String(n_arrays)
            + " and this format carries exactly "
            + String(Int(CKPT_N_ARRAYS))
            + " (param, m, v). A file with a fourth array holds optimizer"
            + " state this loader would drop on the floor."
        )
    if elem_bits != Int(CKPT_ELEM_BITS):
        raise Error(
            String(CKPT_REFUSE_HEADER)
            + ": elem_bits is "
            + String(elem_bits)
            + " and this format is FP32 only. No bf16, fp16, tf32 or"
            + " float64."
        )
    if j < 1 or n < 1:
        raise Error(
            String(CKPT_REFUSE_HEADER)
            + ": j_count is "
            + String(j)
            + " and n_total is "
            + String(n)
            + "; both must be at least 1. A checkpoint of zero elements"
            + " hashes to the FNV offset basis on every machine, which is"
            + " three vendors agreeing about nothing."
        )
    if layout_b != CKPT_LAYOUT_ENTRY_BYTES * j:
        raise Error(
            String(CKPT_REFUSE_HEADER)
            + ": layout_bytes is "
            + String(layout_b)
            + " and "
            + String(CKPT_LAYOUT_ENTRY_BYTES)
            + " * j_count ("
            + String(j)
            + ") is "
            + String(CKPT_LAYOUT_ENTRY_BYTES * j)
        )
    if payload_b != Int(CKPT_N_ARRAYS) * n * 4:
        raise Error(
            String(CKPT_REFUSE_HEADER)
            + ": payload_bytes is "
            + String(payload_b)
            + " and n_arrays * n_total * 4 is "
            + String(Int(CKPT_N_ARRAYS) * n * 4)
        )
    if res0 != 0 or res1 != 0 or res2 != 0:
        raise Error(
            String(CKPT_REFUSE_HEADER)
            + ": a reserved header field is nonzero ("
            + String(res0)
            + "/"
            + String(res1)
            + "/"
            + String(res2)
            + "). **Reserved-must-be-zero is what makes a v1 reader safe"
            + " against a writer that quietly started using one**, so this"
            + " is refused rather than ignored."
        )
    var desc_res0 = Int(_get_u32(raw, 120))
    var desc_res1 = Int(_get_u32(raw, 124))
    if desc_res0 != 0 or desc_res1 != 0:
        raise Error(
            String(CKPT_REFUSE_HEADER)
            + ": a reserved DESCRIPTOR field is nonzero ("
            + String(desc_res0)
            + "/"
            + String(desc_res1)
            + "). Same rule as the header's."
        )

    # ---- 5. the length, now that the shape is known ----------------------
    var want = checkpoint_file_bytes(j, n)
    if size < want:
        raise Error(
            String(CKPT_REFUSE_TRUNCATED)
            + ": '"
            + path
            + "' is "
            + String(size)
            + " bytes and its own header says "
            + String(want)
            + " (J = "
            + String(j)
            + ", n_total = "
            + String(n)
            + "). REFUSED rather than reading the payload it has and zeroing"
            + " the rest."
        )
    if size > want:
        raise Error(
            String(CKPT_REFUSE_TRAILING)
            + ": '"
            + path
            + "' is "
            + String(size)
            + " bytes and its own header says "
            + String(want)
            + ". Extra bytes mean the writer and this reader disagree about"
            + " the format, and **ignoring a tail is how two versions coexist"
            + " silently.**"
        )

    # ---- 6. the FILE hash, before anything is parsed ---------------------
    var stored_file = _get_u64(raw, want - 8)
    var got_file = _fold_bytes(raw, 0, want - 8, FNV_OFFSET)
    if got_file != stored_file:
        raise Error(
            String(CKPT_REFUSE_FILE_HASH)
            + ": '"
            + path
            + "' carries h_file "
            + _hex16(stored_file)
            + " and its bytes fold to "
            + _hex16(got_file)
            + ". The transfer or the disk changed a byte. **This is NOT the"
            + " same finding as a content-hash mismatch**: that one means a"
            + " different training state, this one means a damaged file, and"
            + " conflating them is how a corruption gets reported as a"
            + " cross-vendor divergence."
        )

    # ---- 7. the layout table, internally --------------------------------
    var ck = Checkpoint()
    ck.t = t
    ck.seed = _get_u64(raw, 64)
    ck.opt_kind = Int(_get_u32(raw, 72))
    ck.lr = _get_f32(raw, 76)
    ck.beta1 = _get_f32(raw, 80)
    ck.beta2 = _get_f32(raw, 84)
    ck.eps = _get_f32(raw, 88)
    ck.weight_decay = _get_f32(raw, 92)
    ck.momentum = _get_f32(raw, 96)
    ck.dampening = _get_f32(raw, 100)
    var nest = Int(_get_u32(raw, 104))
    if nest > 1:
        raise Error(
            String(CKPT_REFUSE_HEADER)
            + ": nesterov is "
            + String(nest)
            + " and the only legal values are 0 and 1. A boolean that is"
            + " neither is a field the writer put something else in."
        )
    ck.nesterov = nest == 1
    ck.max_norm = _get_f32(raw, 108)
    ck.steps_planned = Int(_get_u32(raw, 112))
    ck.arm = Int(_get_u32(raw, 116))

    ck.offsets.append(0)
    var at = CKPT_HEADER_BYTES + CKPT_DESCRIPTOR_BYTES
    var acc = 0
    for k in range(j):
        var pid = Int(_get_u32(raw, at))
        var off = Int(_get_u32(raw, at + 4))
        var cnt = Int(_get_u32(raw, at + 8))
        var flag = Int(raw[at + 12])
        var r0 = Int(raw[at + 13])
        var r1 = Int(raw[at + 14])
        var r2 = Int(raw[at + 15])
        var nm = _get_name(raw, at + 16)
        if pid != k:
            raise Error(
                String(CKPT_REFUSE_LAYOUT)
                + ": layout entry "
                + String(k)
                + " calls itself param_id "
                + String(pid)
                + ". **`param_id` IS the index and its ascending order is the"
                + " optimizer's cross-tensor summation order** (DEVIATION"
                + " 1550); a table out of order is a table that would be read"
                + " into the wrong tensors."
            )
        if cnt < 1:
            raise Error(
                String(CKPT_REFUSE_LAYOUT)
                + ": tensor "
                + String(k)
                + " ('"
                + String(nm)
                + "') has count "
                + String(cnt)
                + ", which must be at least 1."
            )
        if off != acc:
            raise Error(
                String(CKPT_REFUSE_LAYOUT)
                + ": tensor "
                + String(k)
                + " ('"
                + String(nm)
                + "') starts at "
                + String(off)
                + " and the tensors before it end at "
                + String(acc)
                + ". The layout must be CONTIGUOUS and ascending from 0:"
                + " a gap folds uninitialized bytes into the payload and an"
                + " overlap makes two tensors share cells, and **both are"
                + " identical on all three vendors**"
                + " (TRAINING_LOOP_PLAN.md V7)."
            )
        if flag > 1:
            raise Error(
                String(CKPT_REFUSE_LAYOUT)
                + ": tensor "
                + String(k)
                + " ('"
                + String(nm)
                + "') has buf_initialized = "
                + String(flag)
                + " and the only legal values are 0 and 1."
            )
        if r0 != 0 or r1 != 0 or r2 != 0:
            raise Error(
                String(CKPT_REFUSE_LAYOUT)
                + ": tensor "
                + String(k)
                + " ('"
                + String(nm)
                + "') has a nonzero reserved byte. Same"
                + " reserved-must-be-zero rule as the header's."
            )
        acc += cnt
        ck.offsets.append(acc)
        ck.names.append(nm^)
        ck.buf_initialized.append(flag == 1)
        at += CKPT_LAYOUT_ENTRY_BYTES
    if acc != n:
        raise Error(
            String(CKPT_REFUSE_LAYOUT)
            + ": the layout's counts sum to "
            + String(acc)
            + " and the header says n_total is "
            + String(n)
            + "."
        )

    # ---- 8. the payload, and only now ------------------------------------
    var payload_at = checkpoint_payload_begin(j)
    var p = payload_at
    for _ in range(n):
        ck.param.append(_get_f32(raw, p))
        p += 4
    for _ in range(n):
        ck.m_state.append(_get_f32(raw, p))
        p += 4
    for _ in range(n):
        ck.v_state.append(_get_f32(raw, p))
        p += 4

    # ---- 9. the CONTENT hash ---------------------------------------------
    # Reached only by an edit that repaired `h_file`, which is exactly why
    # the gate's arm 5 repairs it: a content hash no arm can fire is
    # decoration (`[[reached-but-inert]]`).
    var stored_all = _get_u64(raw, want - 16)
    var got_all = _fold_bytes(raw, payload_at, payload_b, FNV_OFFSET)
    if got_all != stored_all:
        raise Error(
            String(CKPT_REFUSE_CONTENT_HASH)
            + ": '"
            + path
            + "' carries h_all "
            + _hex16(stored_all)
            + " and its payload folds to "
            + _hex16(got_all)
            + ", with the FILE hash intact. The state in this file is not the"
            + " state its digest claims."
        )

    # `[[mojo-buffer-freed-at-last-use]]`: hold the byte list past every fold
    # and every index above.
    _ = raw
    return ck^


def load_checkpoint(
    path: String, expect_offsets: List[Int], expect_names: List[String]
) raises -> Checkpoint:
    """The entry point everything should use.

    `read_checkpoint_unchecked` plus the comparison against the layout the
    CALLER expects, refused BY NAME. The expectation is a parameter and not a
    compiled-in constant, so `n_total = 13376` and `J = 11` appear nowhere in
    this module and a second architecture needs no change to it.
    """
    var ck = read_checkpoint_unchecked(path)
    var j = ck.j_count()

    if len(expect_names) != j:
        raise Error(
            String(CKPT_REFUSE_SHAPE)
            + ": '"
            + path
            + "' holds "
            + String(j)
            + " tensors and the caller expects "
            + String(len(expect_names))
            + "."
        )
    if len(expect_offsets) != j + 1:
        raise Error(
            String(CKPT_REFUSE_SHAPE)
            + ": the caller's offset table has "
            + String(len(expect_offsets))
            + " entries and J + 1 is "
            + String(j + 1)
            + "."
        )
    if expect_offsets[j] != ck.n_total():
        raise Error(
            String(CKPT_REFUSE_SHAPE)
            + ": '"
            + path
            + "' holds "
            + String(ck.n_total())
            + " elements and the caller's layout spans "
            + String(expect_offsets[j])
            + ". **A different shape is a different run and its digests are"
            + " not comparable to these** (TRAINING_LOOP_PLAN.md section 7"
            + " item 3)."
        )
    for k in range(j):
        if ck.names[k] != expect_names[k]:
            raise Error(
                String(CKPT_REFUSE_SHAPE)
                + ": param_id "
                + String(k)
                + " is named '"
                + String(ck.names[k])
                + "' in '"
                + path
                + "' and '"
                + expect_names[k]
                + "' in the caller's layout. The parameter ORDER is part of"
                + " the checkpoint hash specification (DEVIATION 1550) and"
                + " loading these bytes into those tensors would give"
                + " plausible, in-bounds, wrong numbers."
            )
        if ck.offsets[k] != expect_offsets[k]:
            raise Error(
                String(CKPT_REFUSE_SHAPE)
                + ": tensor "
                + String(k)
                + " ('"
                + String(ck.names[k])
                + "') starts at "
                + String(ck.offsets[k])
                + " in '"
                + path
                + "' and at "
                + String(expect_offsets[k])
                + " in the caller's layout."
            )
        var file_count = ck.offsets[k + 1] - ck.offsets[k]
        var want_count = expect_offsets[k + 1] - expect_offsets[k]
        if file_count != want_count:
            raise Error(
                String(CKPT_REFUSE_SHAPE)
                + ": tensor "
                + String(k)
                + " ('"
                + String(ck.names[k])
                + "') holds "
                + String(file_count)
                + " elements in '"
                + path
                + "' and the caller expects "
                + String(want_count)
                + "."
            )
    return ck^


# ===========================================================================
# THE TWO-FILE COMPARISON
# ===========================================================================
# This is the function the orchestrator points at two vendors' checkpoints.
# A bare `cmp -l a.bin b.bin` is the same verdict with less of an address.


def _section_of(offset: Int, j: Int, n: Int) -> String:
    """Which part of the format a byte offset lands in.

    A difference in the DESCRIPTOR (two legs launched with different
    configuration) and a difference in the PAYLOAD (a real divergence) are
    completely different findings, and telling them apart without squinting at
    hex is the whole reason this exists.
    """
    if offset < CKPT_HEADER_BYTES:
        return String("header (magic/version/sizes/t)")
    if offset < CKPT_HEADER_BYTES + CKPT_DESCRIPTOR_BYTES:
        return String(
            "DESCRIPTOR -- seed, optimizer configuration, arm. The two legs"
            " were NOT launched with the same configuration, which makes the"
            " comparison meaningless rather than divergent."
        )
    var payload_at = checkpoint_payload_begin(j)
    if offset < payload_at:
        return String("layout table (param_id/offset/count/flag/name)")
    if offset < payload_at + n * 4:
        return (
            String("PAYLOAD, param[")
            + String((offset - payload_at) // 4)
            + "] -- a real parameter divergence"
        )
    if offset < payload_at + 2 * n * 4:
        return (
            String("PAYLOAD, m_state[")
            + String((offset - payload_at - n * 4) // 4)
            + "] -- the Adam first moment. m and v integrate a one-ULP"
            + " gradient difference across steps, so a divergence can appear"
            + " here a step or two before it reaches param."
        )
    if offset < payload_at + 3 * n * 4:
        return (
            String("PAYLOAD, v_state[")
            + String((offset - payload_at - 2 * n * 4) // 4)
            + "] -- the Adam second moment"
        )
    if offset < payload_at + 3 * n * 4 + 8:
        return String("trailer h_all")
    return String("trailer h_file")


def compare_checkpoint_files(path_a: String, path_b: String) raises -> String:
    """`""` when the two files are byte-for-byte identical, else a report.

    **BYTE-FOR-BYTE IS THE CLAIM.** Not "equal after parsing", not "equal to
    within the digest". Two machines that ran the same configuration for the
    same number of steps produce the same bytes, and any field that could
    differ between them is a field `CHECKPOINT_FORMAT.md` section 3 says must
    not be in the file.

    The report names the FIRST differing offset and the section it falls in,
    so a mismatch has an ADDRESS rather than a verdict -- which is
    `core/identity_trace.mojo`'s whole argument ("a claim that can only be
    checked at the END is a claim nobody can debug") applied to a file.
    """
    var a = _read_all_bytes(path_a)
    var b = _read_all_bytes(path_b)
    var na = len(a)
    var nb = len(b)

    if na == 0 or nb == 0:
        return (
            String("EMPTY FILE: '")
            + path_a
            + "' is "
            + String(na)
            + " bytes and '"
            + path_b
            + "' is "
            + String(nb)
            + ". **Two empty files comparing equal is two machines agreeing"
            + " about nothing**, so an empty side is reported as a failure"
            + " and never as a match."
        )

    # The section report needs a shape. Take it from A's header when A has
    # one; a length mismatch is reported either way.
    var j = 0
    var n = 0
    if na >= CKPT_HEADER_BYTES:
        j = Int(_get_u32(a, 32))
        n = Int(_get_u32(a, 36))

    if na != nb:
        return (
            String("LENGTH MISMATCH: '")
            + path_a
            + "' is "
            + String(na)
            + " bytes and '"
            + path_b
            + "' is "
            + String(nb)
            + ". Different lengths mean a different SHAPE or a truncated"
            + " transfer, not a numeric divergence. Load each one and read"
            + " the refusal."
        )

    var first = -1
    for i in range(na):
        if a[i] != b[i]:
            first = i
            break

    if first < 0:
        _ = a
        _ = b
        return String("")

    var av = Int(a[first])
    var bv = Int(b[first])
    var report = (
        String("DIFFER at byte ")
        + String(first)
        + " of "
        + String(na)
        + ": "
        + String(av)
        + " vs "
        + String(bv)
        + "\n  section: "
        + _section_of(first, j, n)
    )
    if na >= CKPT_HEADER_BYTES + 16:
        report += (
            String("\n  A h_all=")
            + _hex16(_get_u64(a, na - 16))
            + " h_file="
            + _hex16(_get_u64(a, na - 8))
            + "\n  B h_all="
            + _hex16(_get_u64(b, nb - 16))
            + " h_file="
            + _hex16(_get_u64(b, nb - 8))
        )
    report += (
        String("\n  This is the ADDRESS and not the diagnosis. A matching")
        + " h_all with differing descriptor bytes is two legs launched"
        + " differently; a differing h_all is a real divergence and"
        + " tools/identity_trace_diff.py on the two runs' traces names the"
        + " first STEP it appeared at."
    )
    _ = a
    _ = b
    return report^
