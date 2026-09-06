# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The gate of `mojolearn.identical.train.ckpt.file.v1`.

**NOTHING IN THIS FILE, IN `training/checkpoint.mojo` OR IN
`training/CHECKPOINT_FORMAT.md` HAS EVER BEEN COMPILED OR EXECUTED.** No
compiler has read any of it, no device has run it, no byte produced by it has
been observed, and no file written by it has been loaded back. **Every
"passes", "refuses" and "equals" below is a PREDICTION.** Written 2026-09-03
by the checkpoint-file lane, DEVIATIONS 2050 through 2069, PROPOSED and not
yet in the orchestrator's ledger. The design is
`training/CHECKPOINT_FORMAT.md`; the commands that would falsify this are in
its section 10.

WHAT THIS GATE IS FOR
----------------------
`archive/plans/training/TRAINING_LOOP_PLAN.md:49`: *"No checkpoint file format, so clause
(d) tests resume WITHIN one process."* Section 4.2's N6 says the same at
length and section 7 item 6 turns it into a disclaimer: **"No claim about
resuming from a file."** Clause (iii) below is the first thing in this
repository that puts a filesystem in the middle of a resume.

A SERIALIZER IS AN EASY PLACE TO PRODUCE A MEANINGLESS GREEN
-------------------------------------------------------------
Four ways, all of them cheap to fall into and none visible to a naive round
trip:

  * **A uniform fixture.** Write the same float everywhere and the round trip
    passes with `m` and `v` swapped, with the three arrays written in the
    wrong order, and with every offset wrong.
    `[[uniform-test-data-hides-permutation]]` applied to a serializer. Clause
    (i)'s fill is distinct per CELL and distinct per ARRAY.
  * **A refusal that fired for the wrong reason.** "Something raised" is not
    a pass. Every arm in clause (ii) asserts the refusal TAG.
  * **An arm that could not fire.** If the uncorrupted file already refuses,
    every arm below "refuses" whatever it holds and the clause gates nothing.
    `optimizer_check.mojo`'s clause (f) learned this directly; the vacuity
    guard here is copied from it deliberately.
  * **A resume that ignored the file.** A resume clause that reinitialized
    from the seed and never read the checkpoint would pass. Clause (iii)'s
    R1 and R2 are what make it non-vacuous, and clause (iii)'s FIRST
    assertion is that this file's own step loop agrees with
    `train_loop.mojo::run_training` -- **without that, the resume compares
    two things this lane wrote and nothing the training-loop lane wrote.**

WHAT A GREEN RUN HERE WOULD AND WOULD NOT SHOW
-----------------------------------------------
It would show that the format round-trips and refuses correctly ON THE BOX
THAT RAN IT. `[[one-box-verdict-is-not-three]]`: **the cross-vendor claim
needs two files from two vendors and this gate cannot manufacture the second
one.** And `TRAINING_LOOP_PLAN.md` section 1.1 still stands underneath all of
it: two of the twelve stages of a step have never been certified against
their oracle, so **three machines writing byte-identical checkpoints of the
same wrong gradient agree perfectly.**

THE ENVIRONMENT
----------------
    MOJOLEARN_TRAIN_STEPS       N for clause (iii) (default 8)
    MOJOLEARN_TRAIN_SEED        the one integer the data comes from
    MOJOLEARN_TRAIN_CKPT_FILE   write this leg's checkpoint here, after the
                                clean N-step run, for a cross-vendor `cmp`
    MOJOLEARN_CKPT_COMPARE_A    compare mode: two files, no training at all
    MOJOLEARN_CKPT_COMPARE_B
    MOJOLEARN_CKPT_SCRATCH      directory for this gate's temporary files
                                (default /tmp)

WHAT THIS FILE IS LEAST CONFIDENT COMPILES
-------------------------------------------
  1. Driving `train_step` directly rather than through `run_training`, which
     is what a resume-from-file requires because `TrainBuffers.__init__`
     always initializes from the seed and `run_training` always starts at
     `t = 1`. That makes `train_tail` below a FOURTH spelling of the step
     loop, next to `run_training`, `resume_training` and clause (a)'s, and
     the plan's owed item 11 already says three spellings of one thing is
     three chances to get it wrong. A `run_training_from(...)` in
     `train_loop.mojo` is the real fix and it is that lane's call
     (CHECKPOINT_FORMAT.md section 10, owed item 5).
  2. Uploading a host `List[Float32]` into an EXISTING `DeviceBuffer` --
     `train_loop.mojo::_upload` allocates a new one and is private, so
     `_upload_into` below is written out here.
  3. `List[UInt8]` element assignment, for the corruption arms.
     `metrics/checks/regression_metrics_check.mojo:593` does the same on a
     `List[Float32]`.
  4. `try` / `except e` around a `load_checkpoint` that returns a `Movable`.
"""

from std.memory import bitcast
from std.os import getenv

from max.gpu.host import DeviceBuffer, DeviceContext

from core.identity_trace import FNV_OFFSET, IdentityTrace, fnv1a64_bytes

from transformer.impl.transformers.models.llama.modeling_llama import (
    LlamaDeviceStages,
    LlamaDeviceWeights,
    LlamaRopeTable,
)
from transformer.checks.transformer_backward import LlamaBackwardStages

from training.checkpoint import (
    CKPT_DESCRIPTOR_BYTES,
    CKPT_HEADER_BYTES,
    CKPT_LAYOUT_ENTRY_BYTES,
    CKPT_REFUSE_CONTENT_HASH,
    CKPT_REFUSE_FILE_HASH,
    CKPT_REFUSE_HEADER,
    CKPT_REFUSE_LAYOUT,
    CKPT_REFUSE_MAGIC,
    CKPT_REFUSE_SHAPE,
    CKPT_REFUSE_STATE,
    CKPT_REFUSE_TRAILING,
    CKPT_REFUSE_TRUNCATED,
    CKPT_REFUSE_VERSION,
    Checkpoint,
    CheckpointHashes,
    checkpoint_file_bytes,
    checkpoint_payload_begin,
    compare_checkpoint_files,
    load_checkpoint,
    save_checkpoint,
)
from training.checks.optimizer_oracle import OPT_ADAMW
from training.checks.train_loop import (
    ARM_NONE,
    PID_NORM1,
    PID_NORM2,
    PID_W_DOWN,
    PID_W_GATE,
    PID_W_K,
    PID_W_O,
    PID_W_Q,
    PID_W_UP,
    PID_W_V,
    SEED_BASE,
    TRAIN_B,
    TRAIN_J,
    TRAIN_L,
    TRAIN_RMS_EPS,
    TRAIN_ROPE_POSITIONS,
    TRAIN_ROPE_THETA,
    TrainBuffers,
    TrainConfig,
    TrainDigest,
    digest_of_lists,
    download_f32,
    effective_step,
    env_int,
    env_str,
    env_u64,
    hex16,
    mode_banner,
    param_id_count,
    param_id_name,
    run_training,
    train_batch_ids,
    train_dims,
    train_init_tensor,
    train_n_total,
    train_offsets,
    train_step,
    unpack_params,
)


# ===========================================================================
# BITWISE COMPARISON
# ===========================================================================
# **BY BITS AND NEVER BY A TOLERANCE**, and never by `==` on a float either:
# clause (i) plants a NaN on purpose and `NaN == NaN` is False, so an equality
# comparison would report a difference where the bits are identical. That is
# not a hypothetical here; it is one of the planted values.


def bits_of(v: Float32) -> UInt32:
    return bitcast[DType.uint32](v)


def f32_report(v: Float32) -> String:
    """`<decimal>/<hex bits>`. `[[mojo-string-float-roundtrip]]`: the decimal
    does not round trip in this toolchain, so THE HEX IS THE VALUE and the
    decimal is for the reader."""
    return String(v) + "/" + hex16(UInt64(bits_of(v)))


def compare_f32_lists(
    label: String, want: List[Float32], got: List[Float32]
) raises -> Int:
    """Cell for cell, by bit pattern. Returns the number of failures printed.

    A LENGTH MISMATCH IS A FAILURE and not a truncated comparison. Comparing
    the shorter prefix of two different shapes is how a gate reports agreement
    about an array that does not exist.
    """
    if len(want) != len(got):
        print(
            "  **FAILED**: "
            + label
            + " has "
            + String(len(want))
            + " cells before the round trip and "
            + String(len(got))
            + " after."
        )
        return 1
    var moved = 0
    var first = -1
    for i in range(len(want)):
        if bits_of(want[i]) != bits_of(got[i]):
            moved += 1
            if first < 0:
                first = i
    if moved == 0:
        print(
            "  ok   " + label + "  " + String(len(want)) + " cells IDENTICAL"
        )
        return 0
    print(
        "  **FAILED**: "
        + label
        + "  "
        + String(moved)
        + "/"
        + String(len(want))
        + " cells MOVED, first at "
        + String(first)
        + "  want "
        + f32_report(want[first])
        + "  got "
        + f32_report(got[first])
    )
    return 1


# ===========================================================================
# FILES, AND HOW THIS GATE CORRUPTS THEM
# ===========================================================================


def scratch_dir() -> String:
    var d = String(getenv("MOJOLEARN_CKPT_SCRATCH"))
    if d.byte_length() > 0:
        return d^
    return String("/tmp")


def scratch(name: String) -> String:
    return scratch_dir() + "/mojolearn_ckpt_" + name


def read_file_bytes(path: String) raises -> List[UInt8]:
    """`checks/identity_trace_check.mojo:376`'s spelling."""
    var raw = List[UInt8]()
    with open(path, "r") as fh:
        var body = fh.read_bytes()
        for k in range(len(body)):
            raw.append(body[k])
    return raw^


def write_file_bytes(path: String, mut bytes: List[UInt8]) raises:
    """`core/identity_trace.mojo:311`'s spelling."""
    with open(path, "w") as fh:
        fh.write_bytes(Span(bytes))
    # `[[mojo-buffer-freed-at-last-use]]`: hold the owner past the write.
    _ = bytes


def repair_file_hash(mut bytes: List[UInt8]) raises:
    """Recompute `h_file` over `[0, len-8)` and write it into the last eight
    bytes.

    **THIS EXISTS FOR ONE ARM AND IT IS THE POINT OF THAT ARM.** Flipping a
    payload bit and stopping there fires `CKPT_REFUSE_FILE_HASH`, which is a
    correct refusal for the wrong reason and would leave the CONTENT hash as
    code that has never once decided anything (`[[reached-but-inert]]`).
    Repairing the file hash is what makes `CKPT_REFUSE_CONTENT_HASH`
    reachable.
    """
    var n = len(bytes)
    if n < 16:
        raise Error(
            String("checkpoint_check: repair_file_hash on ")
            + String(n)
            + " bytes; there is no trailer to repair."
        )
    var h = fnv1a64_bytes(FNV_OFFSET, bytes.unsafe_ptr(), n - 8)
    for i in range(8):
        bytes[n - 8 + i] = UInt8(
            UInt32((h >> UInt64(8 * i)) & UInt64(0xFF))
        )


def copy_file_with(
    src: String,
    dst: String,
    at: Int,
    xor_mask: Int,
    drop_tail: Int,
    pad_tail: Int,
    repair: Bool,
) raises:
    """Copy `src` to `dst` with exactly one thing changed.

    Flip bits at byte `at` (`at < 0` for none), drop `drop_tail` bytes from
    the end, append `pad_tail` zero bytes, then optionally recompute `h_file`.

    One function rather than nine, so that every corruption arm goes through
    the same read and the same write and the only difference between two arms
    is the corruption.
    """
    var b = read_file_bytes(src)
    if at >= 0:
        if at >= len(b):
            raise Error(
                String("checkpoint_check: corruption offset ")
                + String(at)
                + " is past the end of a "
                + String(len(b))
                + "-byte file."
            )
        b[at] = UInt8(UInt32(Int(b[at]) ^ xor_mask) & UInt32(0xFF))
    if drop_tail > 0:
        var keep = len(b) - drop_tail
        if keep < 1:
            raise Error("checkpoint_check: drop_tail would empty the file.")
        var t = List[UInt8]()
        for i in range(keep):
            t.append(b[i])
        b = t^
    for _ in range(pad_tail):
        b.append(UInt8(0))
    if repair:
        repair_file_hash(b)
    write_file_bytes(dst, b)
    _ = b


def expect_refusal(
    label: String,
    path: String,
    offsets: List[Int],
    names: List[String],
    tag: String,
) raises -> Int:
    """Load `path` and require a refusal carrying `tag`.

    **A REFUSAL FOR THE WRONG REASON IS NOT A PASS.** Asserting only that
    something raised would let a magic arm be credited to a hash check and a
    shape arm to a truncation check, and then the day one of those stops
    working nothing notices.
    """
    var raised = False
    var msg = String("")
    try:
        var ck = load_checkpoint(path, offsets, names)
        _ = ck^
    except e:
        raised = True
        msg = String(e)
    if not raised:
        print(
            "  **FAILED**: "
            + label
            + " LOADED CLEAN. It must refuse with "
            + tag
            + ". A loader that guesses is worse than no loader."
        )
        return 1
    if msg.find(tag) < 0:
        print(
            "  **FAILED**: "
            + label
            + " refused for the WRONG REASON. Expected "
            + tag
            + ", got: "
            + msg
        )
        return 1
    print("  ok   " + label + " -> " + tag)
    return 0


# ===========================================================================
# THE LAYOUT, AND A PLANTED CHECKPOINT
# ===========================================================================


def train_names() -> List[String]:
    var out = List[String]()
    for j in range(TRAIN_J):
        out.append(param_id_name(j))
    return out^


def offsets_from_counts(counts: List[Int]) -> List[Int]:
    var out = List[Int]()
    var acc = 0
    out.append(0)
    for j in range(len(counts)):
        acc += counts[j]
        out.append(acc)
    return out^


def train_counts() -> List[Int]:
    var out = List[Int]()
    for j in range(TRAIN_J):
        out.append(param_id_count(j))
    return out^


def planted_checkpoint(
    counts: List[Int], names: List[String]
) raises -> Checkpoint:
    """A fully populated checkpoint whose every cell is DISTINCT.

    **A UNIFORM FILL WOULD PASS THE ROUND TRIP WITH `m` AND `v` SWAPPED, WITH
    THE THREE ARRAYS IN THE WRONG ORDER, AND WITH EVERY OFFSET WRONG.**
    `[[uniform-test-data-hides-permutation]]` applied to a serializer. The
    three fills are

        param[i] = f32_from_bits(0x3F800000 + i)   normals near 1.0
        m[i]     = f32_from_bits(0xBF800000 + i)   normals near -1.0
        v[i]     = f32_from_bits(0x40000000 + i)   normals near 2.0

    distinct per cell AND distinct per array, so a swap is visible and a
    permutation is visible.

    Two cells are then overwritten with values a format that went anywhere
    near a decimal representation would quietly destroy:

        param[0]  = -0.0            bits 0x80000000
        m[1]      = NaN, payload 1  bits 0x7FC00001

    `-0.0` matters because `m` starting at exactly `+0.0` is what makes
    `OPT_SAB_MOMENT_LERP` bit-inert on a first step, and a format that
    normalized the sign would erase a distinction the optimizer contract
    depends on. The NaN payload matters because `NaN == NaN` is False, so a
    comparison written with `==` instead of on the bits would report a
    difference that is not there -- which is why `compare_f32_lists` compares
    bit patterns.
    """
    var ck = Checkpoint()
    var offs = offsets_from_counts(counts)
    var n = offs[len(counts)]

    ck.t = 4
    ck.seed = SEED_BASE
    ck.opt_kind = OPT_ADAMW
    ck.lr = Float32(1e-3)
    ck.beta1 = Float32(0.9)
    ck.beta2 = Float32(0.999)
    ck.eps = Float32(1e-8)
    ck.weight_decay = Float32(0.01)
    ck.momentum = Float32(0.0)
    ck.dampening = Float32(0.0)
    ck.nesterov = False
    ck.max_norm = Float32(0.0)
    ck.steps_planned = 8
    ck.arm = ARM_NONE

    ck.offsets = offs^
    for j in range(len(names)):
        # `String(...)` rather than a bare append: a borrowed element yields a
        # reference and this repository's spelling for materializing one is an
        # explicit construction (`core/identity_trace.mojo:274`).
        ck.names.append(String(names[j]))
        # Alternating, so a loader that returned a constant flag passes
        # nothing.
        ck.buf_initialized.append(j % 2 == 0)

    for i in range(n):
        ck.param.append(bitcast[DType.float32](UInt32(0x3F800000) + UInt32(i)))
    for i in range(n):
        ck.m_state.append(
            bitcast[DType.float32](UInt32(0xBF800000) + UInt32(i))
        )
    for i in range(n):
        ck.v_state.append(
            bitcast[DType.float32](UInt32(0x40000000) + UInt32(i))
        )
    ck.param[0] = bitcast[DType.float32](UInt32(0x80000000))
    ck.m_state[1] = bitcast[DType.float32](UInt32(0x7FC00001))
    return ck^


# ===========================================================================
# CLAUSE (i): THE ROUND TRIP, BIT FOR BIT
# ===========================================================================


def clause_i() raises -> Int:
    """Save, load, compare every cell and every scalar by BITS.

    Also asserts the format's `h_all` against
    `train_loop.mojo::digest_of_lists(...).h_all`. **That coincidence is
    argued in CHECKPOINT_FORMAT.md section 4 and is CHECKED here rather than
    trusted**: `digest_of_lists` folds the raw host memory behind a
    `List[Float32]` and the format folds bytes it built explicitly from
    `bitcast`. Those are the same sequence on a little-endian host and are NOT
    the same on a big-endian one, and on such a host this assertion fails
    loudly instead of the file carrying a hash that is wrong.
    """
    print("clause (i): the round trip, bit for bit")
    var failures = 0
    var counts = train_counts()
    var names = train_names()
    var offs = offsets_from_counts(counts)
    var n = offs[TRAIN_J]

    # ---- the THIRD `_hex16` in this tree must agree with the other two ----
    # `core/identity_trace.mojo::_hex16`, `train_loop.mojo::hex16` and
    # `checkpoint.mojo::_hex16` are three spellings of one function, each
    # private to its module. `train_step_check.mojo` clause (e) already
    # asserts the first pair on a fixed value; this asserts the third against
    # it, so a digest printed by any of them means the same thing.
    var probe = CheckpointHashes(UInt64(0x0123456789ABCDEF), UInt64(0), 0)
    if probe.h_all_hex() != hex16(UInt64(0x0123456789ABCDEF)):
        failures += 1
        print(
            "  **FAILED**: checkpoint.mojo's hex spelling gives "
            + probe.h_all_hex()
            + " and train_loop.mojo's gives "
            + hex16(UInt64(0x0123456789ABCDEF))
            + ". Two digests printed by the two modules would not be"
            + " comparable by eye."
        )
    else:
        print("  ok   the hex spelling agrees with train_loop.mojo::hex16")

    var ck = planted_checkpoint(counts, names)
    var path = scratch("roundtrip.bin")
    var h = save_checkpoint(path, ck)

    var want_bytes = checkpoint_file_bytes(TRAIN_J, n)
    if h.bytes != want_bytes:
        failures += 1
        print(
            "  **FAILED**: the save reported "
            + String(h.bytes)
            + " bytes and the formula says "
            + String(want_bytes)
        )
    else:
        print(
            "  ok   file is "
            + String(h.bytes)
            + " bytes = 64 + 64 + 32*"
            + String(TRAIN_J)
            + " + 12*"
            + String(n)
            + " + 16"
        )

    # ---- the format's h_all IS the training loop's h_all -----------------
    var d = digest_of_lists(ck.param, ck.m_state, ck.v_state, ck.offsets)
    if d.h_all != h.h_all:
        failures += 1
        print(
            "  **FAILED**: the file's h_all is "
            + hex16(h.h_all)
            + " and digest_of_lists says "
            + hex16(d.h_all)
            + ". The two must be the same number by construction"
            + " (CHECKPOINT_FORMAT.md section 4). If this box is"
            + " big-endian, that is the reason and the format is not"
            + " defined for it."
        )
    else:
        print("  ok   h_all == digest_of_lists h_all = " + hex16(h.h_all))
    _ = d^

    # ---- load it back ----------------------------------------------------
    var back = load_checkpoint(path, offs, names)

    failures += compare_f32_lists(String("param"), ck.param, back.param)
    failures += compare_f32_lists(String("m_state"), ck.m_state, back.m_state)
    failures += compare_f32_lists(String("v_state"), ck.v_state, back.v_state)

    if back.t != ck.t:
        failures += 1
        print(
            "  **FAILED**: t is "
            + String(ck.t)
            + " before and "
            + String(back.t)
            + " after. An off-by-one here moves beta^t, whose two spellings"
            + " agree exactly through t = 6 -- so it is INVISIBLE in any run"
            + " shorter than seven steps."
        )
    if back.seed != ck.seed:
        failures += 1
        print("  **FAILED**: seed did not round trip")
    if back.opt_kind != ck.opt_kind or back.steps_planned != ck.steps_planned:
        failures += 1
        print("  **FAILED**: opt_kind or steps_planned did not round trip")
    if back.arm != ck.arm:
        failures += 1
        print("  **FAILED**: arm did not round trip")
    if back.nesterov != ck.nesterov:
        failures += 1
        print("  **FAILED**: nesterov did not round trip")

    # Hyperparameters BY BITS, never by `==` on the decimal.
    var hp_bad = 0
    if bits_of(back.lr) != bits_of(ck.lr):
        hp_bad += 1
    if bits_of(back.beta1) != bits_of(ck.beta1):
        hp_bad += 1
    if bits_of(back.beta2) != bits_of(ck.beta2):
        hp_bad += 1
    if bits_of(back.eps) != bits_of(ck.eps):
        hp_bad += 1
    if bits_of(back.weight_decay) != bits_of(ck.weight_decay):
        hp_bad += 1
    if bits_of(back.momentum) != bits_of(ck.momentum):
        hp_bad += 1
    if bits_of(back.dampening) != bits_of(ck.dampening):
        hp_bad += 1
    if bits_of(back.max_norm) != bits_of(ck.max_norm):
        hp_bad += 1
    if hp_bad != 0:
        failures += 1
        print(
            "  **FAILED**: "
            + String(hp_bad)
            + " of the eight hyperparameters did not round trip BY BITS."
            + " They are stored as raw u32 patterns precisely so that this"
            + " cannot happen; a decimal round trip would."
        )
    else:
        print("  ok   eight hyperparameters round trip by bits")

    var layout_bad = 0
    for j in range(TRAIN_J):
        if back.names[j] != names[j]:
            layout_bad += 1
        if back.offsets[j] != offs[j]:
            layout_bad += 1
        if back.buf_initialized[j] != (j % 2 == 0):
            layout_bad += 1
    if back.offsets[TRAIN_J] != offs[TRAIN_J]:
        layout_bad += 1
    if layout_bad != 0:
        failures += 1
        print(
            "  **FAILED**: "
            + String(layout_bad)
            + " layout fields did not round trip (names, offsets or the"
            + " momentum flags)."
        )
    else:
        print(
            "  ok   layout round trips: "
            + String(TRAIN_J)
            + " names, offsets and momentum flags"
        )

    # ---- save the LOADED copy and compare the two FILES ------------------
    # The strongest cheap assertion available: it catches a writer and a
    # reader that are wrong in exactly compensating ways, which a
    # value-by-value comparison cannot.
    var path2 = scratch("roundtrip2.bin")
    var h2 = save_checkpoint(path2, back)
    if h2.h_file != h.h_file:
        failures += 1
        print(
            "  **FAILED**: re-saving the loaded checkpoint gives h_file "
            + hex16(h2.h_file)
            + " and the original is "
            + hex16(h.h_file)
        )
    var rep = compare_checkpoint_files(path, path2)
    if rep != "":
        failures += 1
        print("  **FAILED**: save -> load -> save is not byte-stable")
        print("    " + rep)
    else:
        print("  ok   save -> load -> save is byte-for-byte identical")

    _ = ck^
    _ = back^
    if failures == 0:
        print("  clause (i) GREEN")
    return failures


# ===========================================================================
# CLAUSE (ii): THE REFUSALS
# ===========================================================================


def clause_ii() raises -> Int:
    """Six arms, each corrupting exactly one thing in a file that has just
    been shown to load CLEAN.

    **THE CLEAN-LOAD PRECONDITION IS NOT OPTIONAL.** If the uncorrupted file
    already refuses, every arm below "refuses" whatever it holds and this
    clause gates nothing. `optimizer_check.mojo` clause (f) found exactly this
    shape of hole in itself and raises rather than reporting a pass; this does
    the same.
    """
    print("clause (ii): the refusals")
    var failures = 0
    var counts = train_counts()
    var names = train_names()
    var offs = offsets_from_counts(counts)
    var n = offs[TRAIN_J]

    var good = scratch("good.bin")
    var ck = planted_checkpoint(counts, names)
    var h = save_checkpoint(good, ck)
    _ = ck^

    # ---- THE VACUITY GUARD ----------------------------------------------
    var ctrl_raised = False
    var ctrl_msg = String("")
    try:
        var probe = load_checkpoint(good, offs, names)
        _ = probe^
    except e:
        ctrl_raised = True
        ctrl_msg = String(e)
    if ctrl_raised:
        raise Error(
            String("checkpoint_check: CLAUSE (ii) IS VACUOUS. The")
            + " UNCORRUPTED file refused to load, so every arm below would"
            + " be 'refused' whatever it held and this clause gates"
            + " nothing. The refusal was: "
            + ctrl_msg
        )
    print("  the uncorrupted file loads clean; the arms below can fire")

    # ---- 1. MAGIC --------------------------------------------------------
    var p_magic = scratch("bad_magic.bin")
    copy_file_with(good, p_magic, 0, 0xFF, 0, 0, False)
    failures += expect_refusal(
        String("1 magic: byte 0 flipped"),
        p_magic,
        offs,
        names,
        String(CKPT_REFUSE_MAGIC),
    )

    # ---- 2. VERSION ------------------------------------------------------
    # `format_version` is at byte 8; XOR 3 turns 1 into 2.
    var p_ver = scratch("bad_version.bin")
    copy_file_with(good, p_ver, 8, 0x03, 0, 0, True)
    failures += expect_refusal(
        String("2 version: format_version := 2, file hash repaired"),
        p_ver,
        offs,
        names,
        String(CKPT_REFUSE_VERSION),
    )

    # ---- 3. SHAPE, two sub-arms -----------------------------------------
    # (a) a file written with one FEWER tensor. Internally consistent, its
    #     own hashes correct, and simply not this model.
    var short_counts = List[Int]()
    var short_names = List[String]()
    for j in range(TRAIN_J - 1):
        short_counts.append(counts[j])
        short_names.append(String(names[j]))
    var p_short = scratch("bad_shape_short.bin")
    var ck_short = planted_checkpoint(short_counts, short_names)
    _ = save_checkpoint(p_short, ck_short)
    _ = ck_short^
    failures += expect_refusal(
        String("3a shape: a valid file with J-1 tensors"),
        p_short,
        offs,
        names,
        String(CKPT_REFUSE_SHAPE),
    )

    # (b) two tensors' counts SWAPPED. `n_total` is unchanged, the file is
    #     internally consistent, and every offset after the swap is wrong.
    #     **This is the defect no cross-vendor comparison can see**: it is
    #     plausible, in-bounds and identical on all three vendors
    #     (TRAINING_LOOP_PLAN.md V7).
    var swapped = List[Int]()
    for j in range(TRAIN_J):
        swapped.append(counts[j])
    var a_id = PID_W_Q
    var b_id = PID_W_K
    var tmp = swapped[a_id]
    swapped[a_id] = swapped[b_id]
    swapped[b_id] = tmp
    var p_swap = scratch("bad_shape_swap.bin")
    var ck_swap = planted_checkpoint(swapped, names)
    _ = save_checkpoint(p_swap, ck_swap)
    _ = ck_swap^
    failures += expect_refusal(
        String("3b shape: w_q and w_k counts swapped, n_total unchanged"),
        p_swap,
        offs,
        names,
        String(CKPT_REFUSE_SHAPE),
    )

    # ---- 4. TRUNCATED ----------------------------------------------------
    # The last eight bytes dropped, which is what a crash mid-write leaves.
    var p_trunc = scratch("bad_trunc.bin")
    copy_file_with(good, p_trunc, -1, 0, 8, 0, False)
    failures += expect_refusal(
        String("4 truncated: last 8 bytes dropped"),
        p_trunc,
        offs,
        names,
        String(CKPT_REFUSE_TRUNCATED),
    )

    # ---- 5. CONTENT HASH -------------------------------------------------
    # One payload bit flipped AND `h_file` repaired, so the file hash passes
    # and only `h_all` disagrees. **Without the repair this arm would fire
    # CKPT_REFUSE_FILE_HASH and the content hash would be code that has never
    # once decided anything** (`[[reached-but-inert]]`).
    var payload_at = checkpoint_payload_begin(TRAIN_J)
    var mid = payload_at + (n // 2) * 4
    var p_content = scratch("bad_content.bin")
    copy_file_with(good, p_content, mid, 0x01, 0, 0, True)
    failures += expect_refusal(
        String("5 content hash: one payload bit flipped, h_file REPAIRED"),
        p_content,
        offs,
        names,
        String(CKPT_REFUSE_CONTENT_HASH),
    )

    # ---- 6. FILE HASH ----------------------------------------------------
    # The same bit flipped and NOTHING repaired. Both hashes are now
    # demonstrated live, and the two findings are kept apart on purpose: one
    # means a damaged file and the other means different training state.
    var p_file = scratch("bad_file_hash.bin")
    copy_file_with(good, p_file, mid, 0x01, 0, 0, False)
    failures += expect_refusal(
        String("6 file hash: the same bit flipped, nothing repaired"),
        p_file,
        offs,
        names,
        String(CKPT_REFUSE_FILE_HASH),
    )

    # ---- 7. HEADER: a reserved field used ---------------------------------
    # **RESERVED-MUST-BE-ZERO IS WHAT MAKES A V1 READER SAFE AGAINST A WRITER
    # THAT QUIETLY STARTED USING ONE**, so it is a refusal and not a shrug.
    var p_res = scratch("bad_reserved.bin")
    copy_file_with(good, p_res, 52, 0x01, 0, 0, True)
    failures += expect_refusal(
        String("7 header: a reserved header field set nonzero"),
        p_res,
        offs,
        names,
        String(CKPT_REFUSE_HEADER),
    )

    # ---- 8. TRAILING: eight extra bytes -----------------------------------
    var p_tail = scratch("bad_trailing.bin")
    copy_file_with(good, p_tail, -1, 0, 0, 8, False)
    failures += expect_refusal(
        String("8 trailing: 8 extra bytes appended"),
        p_tail,
        offs,
        names,
        String(CKPT_REFUSE_TRAILING),
    )

    # ---- 9. LAYOUT: a param_id that is not its own index ------------------
    # Entry 1's `param_id` field, XOR 1, turning 1 into 0. The table is now
    # out of order, and **the order IS the checkpoint hash specification**
    # (DEVIATION 1550).
    var pid_at = CKPT_HEADER_BYTES + CKPT_DESCRIPTOR_BYTES + (
        CKPT_LAYOUT_ENTRY_BYTES * 1
    )
    var p_layout = scratch("bad_layout.bin")
    copy_file_with(good, p_layout, pid_at, 0x01, 0, 0, True)
    failures += expect_refusal(
        String("9 layout: entry 1 calls itself param_id 0"),
        p_layout,
        offs,
        names,
        String(CKPT_REFUSE_LAYOUT),
    )

    # ---- 10. STATE: an inconsistent checkpoint handed to the WRITER -------
    # The one refusal that fires before any byte reaches disk. Refusing at
    # SAVE time is cheaper than discovering it at LOAD time on another
    # continent, and a half-written file that loaded would be worse than one
    # that did not.
    var bad_ck = planted_checkpoint(counts, names)
    var short_m = List[Float32]()
    for i in range(n - 1):
        short_m.append(bad_ck.m_state[i])
    bad_ck.m_state = short_m^
    var save_raised = False
    var save_msg = String("")
    try:
        _ = save_checkpoint(scratch("never_written.bin"), bad_ck)
    except e:
        save_raised = True
        save_msg = String(e)
    _ = bad_ck^
    if not save_raised:
        failures += 1
        print(
            "  **FAILED**: the writer accepted an m array one element short"
            + " of param. The tail would be whatever the allocator left"
            + " there, which differs run to run on ONE machine."
        )
    elif save_msg.find(String(CKPT_REFUSE_STATE)) < 0:
        failures += 1
        print(
            "  **FAILED**: the writer refused for the WRONG REASON. Expected "
            + String(CKPT_REFUSE_STATE)
            + ", got: "
            + save_msg
        )
    else:
        print(
            "  ok   10 state: m one element short at SAVE -> "
            + String(CKPT_REFUSE_STATE)
        )

    _ = h
    if failures == 0:
        print(
            "  clause (ii) GREEN: ten arms, each refused BY ITS OWN TAG."
            " **Every CKPT_REFUSE_* tag the format defines has now fired at"
            " least once**, so none of them is code that has never decided"
            " anything."
        )
    return failures


# ===========================================================================
# THE DEVICE SIDE: A STEP LOOP THAT CAN START FROM A FILE
# ===========================================================================
# `TrainBuffers.__init__` always initializes from the seed and `run_training`
# always starts at `t = 1`, so a resume-from-file has to drive `train_step`
# itself. That makes this a FOURTH spelling of the step loop and the plan's
# owed item 11 already says three spellings of one thing is three chances to
# get it wrong -- so `clause_iii` asserts this loop against `run_training`
# BEFORE it uses it for anything. A `run_training_from(...)` in
# `train_loop.mojo` is the real fix and it is that lane's call.


def _upload_into(
    ctx: DeviceContext,
    mut dst: DeviceBuffer[DType.float32],
    values: List[Float32],
) raises:
    """Host list -> an EXISTING device buffer.

    `train_loop.mojo::_upload` allocates a new buffer and is private, so this
    is written out. `n` is `len(values)` and is CHECKED against the
    destination rather than assumed: a short upload leaves the tail holding
    whatever the allocator left there, which differs run to run on ONE machine
    and would make the instrument report divergence everywhere
    (`core/identity_trace.mojo` rule 3).
    """
    var n = len(values)
    if n < 1:
        raise Error("checkpoint_check: _upload_into with an empty list")
    if n > len(dst):
        raise Error(
            String("checkpoint_check: _upload_into wants ")
            + String(n)
            + " elements and the destination holds "
            + String(len(dst))
        )
    var h = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.synchronize()
    for i in range(n):
        h.unsafe_ptr().unsafe_store(i, values[i])
    if n == len(dst):
        ctx.enqueue_copy(dst_buf=dst, src_ptr=h.unsafe_ptr())
    else:
        var view = dst.create_sub_buffer[DType.float32](0, n)
        ctx.enqueue_copy(dst_buf=view, src_ptr=h.unsafe_ptr())
        ctx.synchronize()
        _ = view
    ctx.synchronize()
    # `[[mojo-buffer-freed-at-last-use]]`: the host staging buffer must
    # outlive the copy that is reading from it.
    _ = h


def device_weights(ctx: DeviceContext, seed: UInt64) raises -> LlamaDeviceWeights:
    """`train_step_check.mojo::device_weights`, restated.

    The VALUES do not matter: `unpack_params` refreshes all eleven from the
    flat parameter buffer at the top of every step (DEVIATION 1553). What
    matters is that the ALLOCATIONS are the sizes `LlamaDims` says.
    """
    return LlamaDeviceWeights(
        ctx,
        train_dims(),
        TRAIN_RMS_EPS,
        train_init_tensor(seed, PID_NORM1),
        train_init_tensor(seed, PID_NORM2),
        train_init_tensor(seed, PID_W_Q),
        train_init_tensor(seed, PID_W_K),
        train_init_tensor(seed, PID_W_V),
        train_init_tensor(seed, PID_W_O),
        train_init_tensor(seed, PID_W_GATE),
        train_init_tensor(seed, PID_W_UP),
        train_init_tensor(seed, PID_W_DOWN),
    )


def checkpoint_of(
    ctx: DeviceContext, mut tb: TrainBuffers, t: Int, cfg: TrainConfig
) raises -> Checkpoint:
    """The live device state, as a host `Checkpoint`.

    **THE DOWNLOAD LIVES HERE AND NOT IN `training/checkpoint.mojo`**, which
    has no `DeviceContext` in any signature and therefore cannot put a
    property of the device into the file. See CHECKPOINT_FORMAT.md section 3.

    Counts are `tb.n_total`, NEVER `len(buf)`. A buffer allocated with slack
    and read to its capacity carries uninitialized memory into the file, which
    differs run to run on ONE machine.
    """
    var ck = Checkpoint()
    ck.param = download_f32(ctx, tb.param, tb.n_total)
    ck.m_state = download_f32(ctx, tb.m_state, tb.n_total)
    ck.v_state = download_f32(ctx, tb.v_state, tb.n_total)
    ck.offsets = tb.offsets.copy()
    ck.names = train_names()
    for j in range(TRAIN_J):
        ck.buf_initialized.append(tb.buf_initialized[j])
    ck.t = t
    ck.seed = cfg.seed
    var oc = cfg.optimizer()
    ck.opt_kind = oc.kind
    ck.lr = oc.lr
    ck.beta1 = oc.beta1
    ck.beta2 = oc.beta2
    ck.eps = oc.eps
    ck.weight_decay = oc.weight_decay
    ck.momentum = oc.momentum
    ck.dampening = oc.dampening
    ck.nesterov = oc.nesterov
    ck.max_norm = oc.max_norm
    ck.steps_planned = cfg.steps
    ck.arm = cfg.arm
    return ck^


def restore_into(
    ctx: DeviceContext, mut tb: TrainBuffers, ck: Checkpoint
) raises:
    """A host `Checkpoint` back into the live device state.

    **ALL THREE ARRAYS AND THE FLAGS.** Restoring `param` alone is
    `OPT_SAB_RESUME_REINIT`'s exact shape -- a checkpoint that drops the
    optimizer state on the way to disk -- and clause (iii)'s R1 is the control
    that would catch it.
    """
    if ck.n_total() != tb.n_total:
        raise Error(
            String("checkpoint_check: restore_into has ")
            + String(ck.n_total())
            + " elements and the buffers hold "
            + String(tb.n_total)
        )
    _upload_into(ctx, tb.param, ck.param)
    _upload_into(ctx, tb.m_state, ck.m_state)
    _upload_into(ctx, tb.v_state, ck.v_state)
    for j in range(TRAIN_J):
        tb.buf_initialized[j] = ck.buf_initialized[j]
    ctx.synchronize()


def train_head(
    ctx: DeviceContext, cfg: TrainConfig, upto: Int
) raises -> Checkpoint:
    """Steps `1 .. upto` from a fresh seed-initialized state; the state after.

    `t` is ONE-BASED and equals the step index, which is
    `train_loop.mojo::run_training`'s convention.
    """
    if upto < 1:
        raise Error("checkpoint_check: train_head needs upto >= 1")
    var tb = TrainBuffers(ctx, cfg.seed)
    var w = device_weights(ctx, cfg.seed)
    var rope = LlamaRopeTable(
        ctx, train_dims(), TRAIN_ROPE_THETA, TRAIN_ROPE_POSITIONS
    )
    var stages = LlamaDeviceStages(ctx, TRAIN_B, TRAIN_L, TRAIN_L, train_dims())
    var bst = LlamaBackwardStages(ctx, TRAIN_B, TRAIN_L, TRAIN_L, train_dims())
    var off = IdentityTrace.disabled()

    for step in range(1, upto + 1):
        var es = effective_step(step, cfg.steps, cfg.arm)
        var ids = train_batch_ids(cfg.seed, es)
        unpack_params(ctx, tb, w)
        var loss = train_step(
            ctx, tb, w, rope, stages, bst, off, cfg, ids, step
        )
        _ = loss
        _ = ids

    var ck = checkpoint_of(ctx, tb, upto, cfg)
    # `[[mojo-buffer-freed-at-last-use]]`: every owner stays alive past the
    # last kernel that was handed one of its pointers, and past the download.
    _ = tb^
    _ = w^
    _ = rope^
    _ = stages^
    _ = bst^
    return ck^


def train_tail(
    ctx: DeviceContext,
    cfg: TrainConfig,
    ck: Checkpoint,
    data_first: Int,
    data_last: Int,
    t_offset: Int,
) raises -> TrainDigest:
    """Restore `ck`, then run data steps `data_first .. data_last`.

    `t = step - t_offset`. **The DATA index and the OPTIMIZER'S `t` are
    separate parameters on purpose**: control R2 needs a resume whose batches
    are correct and whose `t` restarts at 1, and if the two were one number
    that control would be a data control wearing a `t` control's name.
    """
    var tb = TrainBuffers(ctx, cfg.seed)
    var w = device_weights(ctx, cfg.seed)
    var rope = LlamaRopeTable(
        ctx, train_dims(), TRAIN_ROPE_THETA, TRAIN_ROPE_POSITIONS
    )
    var stages = LlamaDeviceStages(ctx, TRAIN_B, TRAIN_L, TRAIN_L, train_dims())
    var bst = LlamaBackwardStages(ctx, TRAIN_B, TRAIN_L, TRAIN_L, train_dims())
    var off = IdentityTrace.disabled()

    restore_into(ctx, tb, ck)

    for step in range(data_first, data_last + 1):
        var es = effective_step(step, cfg.steps, cfg.arm)
        var ids = train_batch_ids(cfg.seed, es)
        unpack_params(ctx, tb, w)
        var t = step - t_offset
        if t < 1:
            raise Error(
                String("checkpoint_check: train_tail computed t = ")
                + String(t)
                + " and t is one-based. A t of zero or below would make"
                + " beta^t meaningless rather than merely wrong."
            )
        var loss = train_step(ctx, tb, w, rope, stages, bst, off, cfg, ids, t)
        _ = loss
        _ = ids

    var p = download_f32(ctx, tb.param, tb.n_total)
    var m = download_f32(ctx, tb.m_state, tb.n_total)
    var v = download_f32(ctx, tb.v_state, tb.n_total)
    var d = digest_of_lists(p, m, v, tb.offsets)
    _ = tb^
    _ = w^
    _ = rope^
    _ = stages^
    _ = bst^
    return d^


def zeroed_moments(mut ck: Checkpoint):
    """R1's corruption: `m` and `v` replaced by `+0.0` everywhere.

    Rebuilt rather than assigned in place, so the two lists are unambiguously
    the right length.
    """
    var n = ck.n_total()
    var zm = List[Float32]()
    var zv = List[Float32]()
    for _ in range(n):
        zm.append(Float32(0.0))
        zv.append(Float32(0.0))
    ck.m_state = zm^
    ck.v_state = zv^


# ===========================================================================
# CLAUSE (iii): TRAIN, SAVE, LOAD, RESUME
# ===========================================================================


def clause_iii(
    ctx: DeviceContext, seed: UInt64, steps: Int, leg_path: String
) raises -> Int:
    """**The clause this whole lane exists for.**

    `train_step_check.mojo` clause (d) with a filesystem in the middle of it.
    `TRAINING_LOOP_PLAN.md` section 4.2's N6 says in as many words that its
    resume is *"a resume WITHIN one process and not from a file"* and that
    `OPT_SAB_RESUME_REINIT`'s real form is therefore unreachable. This is
    where it becomes reachable.
    """
    print("clause (iii): train, save, load, resume")
    var failures = 0
    if steps < 2:
        print(
            "  SKIPPED and NOT PASSED: a split needs N >= 2. At N = 1 there"
            " is nothing to resume into and the clause cannot fire."
        )
        return 1
    var first = steps // 2
    if first < 1:
        first = 1

    var cfg = TrainConfig.for_arm(steps, seed, ARM_NONE)
    var offs = train_offsets()
    var names = train_names()

    # ---- 0. THIS FILE'S STEP LOOP AGAINST `run_training` -----------------
    # **WITHOUT THIS, EVERYTHING BELOW COMPARES TWO THINGS THIS LANE WROTE
    # AND NOTHING THE TRAINING-LOOP LANE WROTE.** `train_head` is a fourth
    # spelling of the step loop (plan owed item 11) and it has to be shown to
    # be the same loop before it is used as a reference.
    var off1 = IdentityTrace.disabled()
    var off2 = IdentityTrace.disabled()
    var whole = run_training(ctx, cfg, off1, off2)
    var full_ck = train_head(ctx, cfg, steps)
    var d_full = digest_of_lists(
        full_ck.param, full_ck.m_state, full_ck.v_state, full_ck.offsets
    )
    if d_full.h_all != whole.final.h_all:
        failures += 1
        print(
            "  **FAILED**: this gate's own step loop gives h_all "
            + hex16(d_full.h_all)
            + " and train_loop.mojo::run_training gives "
            + hex16(whole.final.h_all)
            + ". **EVERY COMPARISON BELOW IS THEREFORE MEANINGLESS**: the"
            + " resume would be checked against a loop this lane wrote"
            + " rather than against the one the claim is about. Fix this"
            + " before reading any other line of this clause."
        )
        _ = whole^
        _ = full_ck^
        _ = d_full^
        return failures
    print(
        "  ok   this gate's step loop == run_training, h_all "
        + hex16(d_full.h_all)
        + " over "
        + String(steps)
        + " steps"
    )
    _ = d_full^

    # ---- the leg's file, if one was asked for ---------------------------
    if leg_path != "":
        var lh = save_checkpoint(leg_path, full_ck)
        print(
            "  LEG FILE written: "
            + leg_path
            + "  "
            + String(lh.bytes)
            + " bytes  h_all="
            + lh.h_all_hex()
            + "  h_file="
            + lh.h_file_hex()
        )
        print(
            "  A leg that writes this file and never has it compared against"
            " another VENDOR'S is not a result. See CHECKPOINT_FORMAT.md"
            " section 9."
        )
    _ = full_ck^

    # ---- 1. the split, through a file ------------------------------------
    var head = train_head(ctx, cfg, first)
    var path = scratch("resume.bin")
    var hh = save_checkpoint(path, head)
    print(
        "  saved the state after "
        + String(first)
        + " steps: h_all="
        + hh.h_all_hex()
        + "  t="
        + String(head.t)
    )
    _ = head^

    var loaded = load_checkpoint(path, offs, names)
    if loaded.t != first:
        failures += 1
        print(
            "  **FAILED**: the file says t = "
            + String(loaded.t)
            + " and "
            + String(first)
            + " steps were run."
        )
    var d_split = train_tail(ctx, cfg, loaded, first + 1, steps, 0)
    print(
        "  whole("
        + String(steps)
        + ") h_all="
        + hex16(whole.final.h_all)
        + "   file-split("
        + String(first)
        + "+"
        + String(steps - first)
        + ") h_all="
        + hex16(d_split.h_all)
    )
    if d_split.h_all != whole.final.h_all:
        failures += 1
        print(
            "  **FAILED**: a resume FROM A FILE does not reproduce an"
            + " uninterrupted run. Either the file dropped state (m, v or"
            + " the momentum flags), or `t` did not continue across the"
            + " boundary, or the data generator restarted its stream"
            + " (TRAINING_LOOP_PLAN.md V6)."
        )
    else:
        print(
            "  ok   TRAIN, SAVE, LOAD, RESUME reproduces "
            + String(steps)
            + " continuous steps EXACTLY"
        )
    _ = d_split^
    _ = loaded^

    # ---- R1: resume with m and v ZEROED. MUST differ. --------------------
    # A resume that ignored the file and reinitialized from the seed would
    # pass the clause above. This is what makes it non-vacuous.
    var ck_r1 = load_checkpoint(path, offs, names)
    zeroed_moments(ck_r1)
    var d_r1 = train_tail(ctx, cfg, ck_r1, first + 1, steps, 0)
    if d_r1.h_all == whole.final.h_all:
        failures += 1
        print(
            "  **FAILED (R1)**: zeroing m and v before the resume did NOT"
            + " move h_all. **The optimizer state is not reaching the"
            + " resumed run at all**, and the clause above passed for a"
            + " reason that has nothing to do with the checkpoint. This is"
            + " OPT_SAB_RESUME_REINIT's exact shape."
        )
    else:
        print(
            "  ok   R1 m,v zeroed -> h_all "
            + hex16(d_r1.h_all)
            + " DIFFERS, required"
        )
    _ = d_r1^
    _ = ck_r1^

    # ---- R2: resume with `t` restarted at 1. MUST differ at N >= 7. ------
    var ck_r2 = load_checkpoint(path, offs, names)
    var d_r2 = train_tail(ctx, cfg, ck_r2, first + 1, steps, first)
    if steps < 7:
        print(
            "  R2 REPORTED, NOT ASSERTED at N = "
            + String(steps)
            + ": the optimizer contract's two beta^t spellings agree"
            + " exactly through t = 6, so a t restart is not guaranteed to"
            + " move a bit below N = 7. h_all="
            + hex16(d_r2.h_all)
            + ". Run at N = 8."
        )
    elif d_r2.h_all == whole.final.h_all:
        failures += 1
        print(
            "  **FAILED (R2)**: restarting `t` at 1 for the second half did"
            + " NOT move h_all at N = "
            + String(steps)
            + ". The step counter is not reaching the bias correction, so"
            + " the `t` carried in the file is decoration."
        )
    else:
        print(
            "  ok   R2 t restarted at 1 -> h_all "
            + hex16(d_r2.h_all)
            + " DIFFERS, required"
        )
    _ = d_r2^
    _ = ck_r2^

    # ---- R3: the momentum flags dropped. EXPECTED INERT UNDER ADAMW. ----
    # **LABELLED IN ADVANCE AND NOT AFTER IT FAILED TO FIRE.**
    # `identical_optimizer_step` reads `buf_initialized` only on the SGD
    # branch, so under the v1 AdamW configuration dropping the flags cannot
    # move a bit. An arm whose predicted answer is "no bits move" is worth
    # having only when it is labelled that way BEFORE it runs --
    # `optimizer_check`'s OPT_SAB_SCALARS_PER_ELEMENT made the same call.
    # This becomes a real control the day an SGD configuration is added, and
    # it is the one that would catch OPT_SAB_RESUME_REINIT for real.
    var ck_r3 = load_checkpoint(path, offs, names)
    var flags = List[Bool]()
    for _ in range(TRAIN_J):
        flags.append(False)
    ck_r3.buf_initialized = flags^
    var d_r3 = train_tail(ctx, cfg, ck_r3, first + 1, steps, 0)
    if d_r3.h_all == whole.final.h_all:
        print(
            "  R3 REPORTED, NOT ASSERTED: dropping the momentum flags moved"
            " NO bit, which is the PREDICTED result under AdamW (the flags"
            " are read only on the SGD branch). This is not a pass and not a"
            " failure; it is the arm confirming its own prediction."
        )
    else:
        failures += 1
        print(
            "  **FAILED (R3)**: dropping the momentum flags MOVED h_all to "
            + hex16(d_r3.h_all)
            + " under AdamW, where the flags are supposed to be unread."
            + " Either the optimizer's AdamW branch reads them after all or"
            + " something else in the resume is not deterministic. **The"
            + " surprise is the finding**; do not adjust the prediction to"
            + " match it."
        )
    _ = d_r3^
    _ = ck_r3^
    _ = whole^

    if failures == 0:
        print("  clause (iii) GREEN")
    return failures


# ===========================================================================
# CLAUSE (iv): THE TWO-FILE COMPARISON
# ===========================================================================


def clause_iv() raises -> Int:
    """The comparator the orchestrator points at two vendors' files.

    A comparator that reported a difference everywhere, or nowhere, would be
    useless in exactly the situation it is for, so both directions are
    asserted and the offset it names is checked against the one that was
    planted.
    """
    print("clause (iv): the two-file comparison")
    var failures = 0
    var counts = train_counts()
    var names = train_names()

    var a = scratch("cmp_a.bin")
    var b = scratch("cmp_b.bin")
    var ck1 = planted_checkpoint(counts, names)
    _ = save_checkpoint(a, ck1)
    _ = ck1^
    var ck2 = planted_checkpoint(counts, names)
    _ = save_checkpoint(b, ck2)
    _ = ck2^

    var same = compare_checkpoint_files(a, b)
    if same != "":
        failures += 1
        print(
            "  **FAILED**: two files written from EQUAL state are not"
            + " byte-for-byte identical. That equality IS the cross-vendor"
            + " test, so this failing on ONE box means the test cannot be"
            + " run at all."
        )
        print("    " + same)
    else:
        print("  ok   equal state -> byte-for-byte identical files")

    # One planted byte in the PAYLOAD, and the report must name it.
    var payload_at = checkpoint_payload_begin(TRAIN_J)
    var target = payload_at + 7 * 4
    var c = scratch("cmp_c.bin")
    copy_file_with(a, c, target, 0x01, 0, 0, True)
    var rep = compare_checkpoint_files(a, c)
    if rep == "":
        failures += 1
        print(
            "  **FAILED**: a file with one flipped byte compared EQUAL. The"
            + " comparator is not comparing."
        )
    elif rep.find(String("byte ") + String(target)) < 0:
        failures += 1
        print(
            "  **FAILED**: the report does not name byte "
            + String(target)
            + ", which is the one that was planted. It said: "
            + rep
        )
    elif rep.find(String("param[7]")) < 0:
        failures += 1
        print(
            "  **FAILED**: the report names the right offset and the wrong"
            + " SECTION. It should say param[7]; it said: "
            + rep
        )
    else:
        print("  ok   one flipped payload byte -> named, with its section")

    # A byte in the DESCRIPTOR, which is a different FINDING and not a
    # divergence: it means the two legs were launched differently.
    var d_target = CKPT_HEADER_BYTES + 12  # `lr` bits
    var e = scratch("cmp_d.bin")
    copy_file_with(a, e, d_target, 0x01, 0, 0, True)
    var rep2 = compare_checkpoint_files(a, e)
    if rep2 == "" or rep2.find(String("DESCRIPTOR")) < 0:
        failures += 1
        print(
            "  **FAILED**: a difference in the learning-rate bits was not"
            + " reported as a DESCRIPTOR finding. Two legs launched with"
            + " different configuration is a meaningless comparison, not a"
            + " cross-vendor divergence, and telling them apart is what the"
            + " section report is for. It said: "
            + rep2
        )
    else:
        print("  ok   a descriptor difference is named as one")

    if failures == 0:
        print("  clause (iv) GREEN")
    return failures


# ===========================================================================
# CLAUSE (v): DETERMINISM AND NEUTRALITY, AS FAR AS A PROGRAM CAN CHECK THEM
# ===========================================================================


def clause_v(ctx: DeviceContext) raises -> Int:
    """The same state, saved twice, with unrelated device work in between.

    **THE REAL NEUTRALITY ARGUMENT IS STRUCTURAL** and CHECKPOINT_FORMAT.md
    section 3 states it: `training/checkpoint.mojo` does not import
    `max.gpu.host` and has no `DeviceContext` in any signature, so it cannot
    name a device. This clause catches the residue -- a serializer that picked
    up a clock, an allocation address or a counter -- which **no cross-vendor
    comparison could distinguish from a real divergence**, because it would
    differ on all three.
    """
    print("clause (v): determinism of the writer")
    var failures = 0
    var counts = train_counts()
    var names = train_names()
    var n = train_n_total()

    var p1 = scratch("det_1.bin")
    var ck = planted_checkpoint(counts, names)
    var h1 = save_checkpoint(p1, ck)

    # Unrelated device work between the two saves: allocate the whole training
    # buffer set and drain the queue. If anything about the process's device
    # state reaches the file, this is where it shows up.
    var tb = TrainBuffers(ctx, SEED_BASE)
    ctx.synchronize()
    var scratch_read = download_f32(ctx, tb.param, tb.n_total)
    _ = scratch_read
    _ = tb^

    var p2 = scratch("det_2.bin")
    var h2 = save_checkpoint(p2, ck)
    _ = ck^

    if h1.h_file != h2.h_file or h1.h_all != h2.h_all:
        failures += 1
        print(
            "  **FAILED**: two saves of the SAME state gave different"
            + " hashes ("
            + h1.h_file_hex()
            + " vs "
            + h2.h_file_hex()
            + "). Something outside the state is reaching the file."
        )
    var rep = compare_checkpoint_files(p1, p2)
    if rep != "":
        failures += 1
        print(
            "  **FAILED**: two saves of the same state, with device work"
            + " between them, are not byte-identical."
        )
        print("    " + rep)
    else:
        print(
            "  ok   two saves across unrelated device work are"
            " byte-identical"
        )

    var want = checkpoint_file_bytes(TRAIN_J, n)
    var got = len(read_file_bytes(p1))
    if got != want:
        failures += 1
        print(
            "  **FAILED**: the file on disk is "
            + String(got)
            + " bytes and the formula says "
            + String(want)
        )
    else:
        print(
            "  ok   on-disk length is exactly 64 + 64 + 32*"
            + String(TRAIN_J)
            + " + 12*"
            + String(n)
            + " + 16 = "
            + String(want)
        )

    if failures == 0:
        print("  clause (v) GREEN")
    return failures


# ===========================================================================
# MAIN
# ===========================================================================


def compare_mode(path_a: String, path_b: String) raises:
    """Two files, no training, no device. The cross-vendor verdict.

    `cmp -l a.bin b.bin` is the same verdict; this adds the ADDRESS -- which
    section the first differing byte falls in, and both files' hashes -- so a
    descriptor difference (two legs launched differently) is distinguished
    from a payload difference (a real divergence) without anybody squinting at
    hex.
    """
    print("=== checkpoint compare mode")
    print("  A: " + path_a)
    print("  B: " + path_b)
    var rep = compare_checkpoint_files(path_a, path_b)
    if rep == "":
        print(
            "  IDENTICAL, byte for byte. **That equality is the whole"
            " cross-vendor claim about these two files, and it is NOT a"
            " claim that either one is correct: two of the twelve stages of"
            " a step have never been certified against their oracle"
            " (TRAINING_LOOP_PLAN.md 1.1).**"
        )
        return
    print("  " + rep)
    raise Error(
        "checkpoint_check: the two checkpoint files DIFFER. See the address"
        " above, then run tools/identity_trace_diff.py on the two runs'"
        " traces to name the first STEP at which they parted."
    )


def main() raises:
    var steps = env_int("MOJOLEARN_TRAIN_STEPS", 8)
    var seed = env_u64("MOJOLEARN_TRAIN_SEED", SEED_BASE)
    var leg_path = env_str("MOJOLEARN_TRAIN_CKPT_FILE")
    var cmp_a = env_str("MOJOLEARN_CKPT_COMPARE_A")
    var cmp_b = env_str("MOJOLEARN_CKPT_COMPARE_B")

    print(
        "=== checkpoint-file gate, profile"
        " mojolearn.identical.train.ckpt.file.v1"
    )
    print(
        "=== NOTHING IN THIS FILE, IN training/checkpoint.mojo OR IN"
        " training/CHECKPOINT_FORMAT.md HAD EVER BEEN COMPILED OR RUN"
        " BEFORE THIS PROCESS. Every prediction in their headers is"
        " unfalsified."
    )
    print(
        "=== A MATCHING FILE SHOWS AGREEMENT AND NEVER CORRECTNESS. Two of"
        " the twelve stages of a step have no certifying gate"
        " (transformer_backward and embedding_identical), and three machines"
        " writing byte-identical checkpoints of the same wrong gradient"
        " agree perfectly. TRAINING_LOOP_PLAN.md section 1.1."
    )

    if cmp_a != "" or cmp_b != "":
        if cmp_a == "" or cmp_b == "":
            raise Error(
                "checkpoint_check: compare mode needs BOTH"
                " MOJOLEARN_CKPT_COMPARE_A and MOJOLEARN_CKPT_COMPARE_B."
                " Comparing a file against nothing would print a verdict"
                " about one file, which is not a comparison."
            )
        compare_mode(cmp_a, cmp_b)
        return

    print(
        "mode "
        + mode_banner()
        + "   steps N = "
        + String(steps)
        + "   seed = "
        + String(seed)
        + "   n_total = "
        + String(train_n_total())
        + "   J = "
        + String(TRAIN_J)
    )
    print(
        "file bytes at this shape = "
        + String(checkpoint_file_bytes(TRAIN_J, train_n_total()))
        + "   scratch = "
        + scratch_dir()
    )
    if steps < 7:
        print(
            "WARNING: at N < 7 control R2 (a `t` that restarts) CANNOT be"
            " asserted -- the optimizer contract's two beta^t spellings"
            " agree exactly through t = 6 -- and it is REPORTED rather than"
            " passed. N = 8 is the number to run identity from."
        )

    # The host-only clauses first: they cost nothing and a failure in them
    # makes the device clauses uninterpretable.
    var f1 = clause_i()
    var f2 = clause_ii()
    var f4 = clause_iv()

    var ctx = DeviceContext()
    var f5 = clause_v(ctx)
    var f3 = clause_iii(ctx, seed, steps, leg_path)

    var total = f1 + f2 + f3 + f4 + f5
    print("")
    print("=== SUMMARY")
    print("  (i)   round trip    " + String(f1) + " failed")
    print("  (ii)  refusals      " + String(f2) + " failed")
    print("  (iii) file resume   " + String(f3) + " failed")
    print("  (iv)  comparison    " + String(f4) + " failed")
    print("  (v)   determinism   " + String(f5) + " failed")

    if total != 0:
        raise Error(
            String("checkpoint_check: ")
            + String(total)
            + " failures. See the clause reports above."
        )
    if steps < 2:
        print(
            "VACUOUS: every clause that ran is green, but at N = 1 the file"
            " resume could not fire. This is NOT a pass."
        )
        raise Error(
            "checkpoint_check: refusing to report a pass at N < 2"
        )
    print(
        "ALL CLAUSES GREEN on ONE device. **That is a serializer result and"
        " not a cross-vendor one.** One box writing a file it can read back"
        " proves the format round-trips; TWO BOXES WRITING THE SAME BYTES is"
        " the claim, and this gate cannot manufacture the second file."
        " CHECKPOINT_FORMAT.md section 9 is how that run is done and it is"
        " OWED."
    )
