# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""WHICH STAGE OF OLS MOVES BETWEEN TWO IDENTICAL FITS. Run this on an H100
or an MI325X; it is expected to be silent on Apple.

    pixi run mojo run -I . glm/original/ols_launch_localize.mojo
    sh tools/with_identical_mode.sh pixi run mojo run -I . \
        glm/original/ols_launch_localize.mojo

WHAT IS OPEN, AND WHY IT HAS OUTLIVED FOUR LEGS
================================================
`check_ols_is_launch_invariant` fails on the H100 and the MI325X in BOTH
numeric modes and has since E3 leg 10. Two IDENTICAL fits in one process
disagree at coefficient 0 -- `0xbbb60202` vs `0xbbb87825` on an H100 under
FAST at `a0a0eee`, and `0xbbc76fa8` vs `0xbbc6fa1b` on 2026-08-23. Those are
0.2% apart, which is not a last-bit story. It has never reproduced on Apple.
`CHANGELOG.md` records it as "an uninitialized read or a race, and a defect
in BOTH modes", and localization is owed.

THE INSTRUMENT THAT WAS SUPPOSED TO LOCALIZE IT CANNOT, AND THAT IS THE CLUE
============================================================================
`check_ols_is_launch_invariant` already tries: on a failure it emits two
`emit_ols_card` traces and runs `first_divergence` over them. Every leg has
come back with `<no traced repro: the card fixture's two fits agreed>`. The
traced path does not reproduce the bug.

**SOLVED 2026-08-30. THE PARAGRAPH THAT STOOD HERE WAS A GOOD ARGUMENT FOR
THE WRONG ANSWER, AND IT IS KEPT ONLY AS A WARNING.** It said that a traced
fit is not the same program as an untraced one, because `record_device`
copies a device buffer to the host and calls `ctx.synchronize()`,
`lstsq_eig_traced` carries eight of those, and so the traced fit runs the
same kernels with eight extra drains interleaved and every recorded buffer
held live across them. All of that is TRUE. It concluded that whatever moves
is sensitive to drain placement or buffer lifetime rather than to
arithmetic, which is ALSO true, and still it pointed at the wrong file.

The traced path does not reproduce the bug for a much simpler reason that no
amount of reasoning about drains would have found: **it has a different
fixture uploader.** `emit_ols_card` in `ols_trace.mojo` allocates TWO host
buffers, fills both, and only then enqueues both copies. `_fit_bits` in
`ols_check.mojo` allocated ONE, enqueued the copy into `A`, overwrote the
first 8,192 floats of it for `b`, and enqueued the second copy with no
synchronize between them. On a discrete GPU the `A` upload is an
asynchronous DMA of 98,304 floats and the host rewrite races it. The two
programs differ in their INPUT, not in their drains.

The lesson worth keeping: when an instrument and the thing it measures
disagree, compare their FIXTURES before theorising about their execution.
This file copied the broken uploader too, which is why it was silent on the
M4 and would have been silent on a GPU box for the wrong reason. Both are
fixed. `PORTING.md` item 12 stated the rule from the start.

Same shape as DEVIATION 1944 and as `ensemble/original/rf_ctx_probe.mojo`,
which passed two fits in one process because its `one_fit` takes `ctx` as a
borrowed argument, so `main` owned the context and every buffer died while it
was alive -- the probe accidentally fixed the ordering it was written to
reproduce.

WHAT THIS FILE DOES INSTEAD
===========================
It reads the stages from OUTSIDE the fit. Every buffer `lstsq_eig` writes
except `info` is the CALLER'S, allocated here and passed in, so hashing them
after `ols_fit` returns adds no drain inside the fit and does not extend one
buffer's life by one instruction. The fit that runs here is byte for byte the
fit `check_ols_is_launch_invariant` runs.

Six stages, in the order the data flows, so the FIRST one that moves names
the kernel:

    step1.covA    gemm_tn        vendor matmul under FAST, split-K pinned
                                 under IDENTICAL -- the ONLY stage whose
                                 implementation differs between modes
    step2.Ab      xty_kernel     ours in both modes
    step3.eigvals diagonal_to_vector after jacobi_eigh_kernel
    step3.eigvecs jacobi_eigh_kernel
    step4.QS      divide_columns_by_nonzero_kernel
    step5.inv     gemm_nt        vendor matmul under FAST, pinned under
                                 IDENTICAL
    step6.w       gemv_n         vendor gemv under FAST, pinned under
                                 IDENTICAL

READ THE RESULT LIKE THIS, and the reading is written down BEFORE the run so
a surprising answer cannot be re-narrated afterwards:

  * `step2.Ab` moves          -> `xty_kernel` or `pinned_block_sum`. Ours,
                                 both modes, and the most alarming outcome
                                 because that kernel has no cross-block
                                 communication at all.
  * `step1.covA` moves, Ab stable, under FAST ONLY
                              -> MAX's `linalg.matmul` split-K on this box.
                                 A vendor library's run-to-run wobble, which
                                 FAST does not promise against, and then the
                                 IDENTICAL failure is a DIFFERENT bug and
                                 must be chased separately.
  * `step1.covA` moves in BOTH modes
                              -> the split-K Gram kernel or its `xt`
                                 workspace, since the two modes share no
                                 matmul implementation here.
  * covA and Ab stable, `step3.*` moves
                              -> `jacobi_eigh_kernel`. Then read the SWEEP
                                 column: the file's own DEVIATION BLOCK 3
                                 says a last-bit move in the convergence
                                 fold changes the SWEEP COUNT, and one extra
                                 sweep is n(n-1)/2 more rotations -- "not in
                                 the last bit, in the fifth decimal", which
                                 is the size of the coefficient move
                                 actually observed.
  * everything stable through step5, `step6.w` moves
                              -> `gemv_n`.
  * NOTHING moves             -> report it as silent AT THIS SHAPE, on this
                                 box, at this commit, and do not read it as
                                 a fixed bug. The gate is still the
                                 authority; this is a finer instrument
                                 pointed at the same program.

IT RUNS THE GATE'S ALLOCATION SEQUENCE, NOT A REPEATED SHAPE
=============================================================
The first version of this file ran eight fits at one padding width and was
silent on Apple, which proves nothing on a box where the gate is silent too.
The gate does something this did not: it interleaves TWO padding widths,
`pad = 0` and `pad = 37`, inside one context. Those are different allocation
sizes, so the second fit's buffers land at different offsets and the third
fit -- the one that is compared against the first -- receives memory the
SECOND fit last wrote, not the first.

That churn is the gate's A/B/C, and it is the only structural difference
between a fit that reproduces and a fit that does not, so this file
reproduces it exactly: `pad` alternates 0, 37, 0, 37, ... and the comparison
is between fits at the SAME width. A `pad = 0` fit is compared only against
other `pad = 0` fits, which is the gate's A-vs-C control -- the assertion
that fails, in both modes, on both boxes.
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.memory import bitcast

from core.identity_trace import FNV_OFFSET, fnv1a64_bytes
from glm.derived.glm.ols import OLS_ALGO_EIG, ols_fit
from glm.derived.linalg.detail.lstsq import OLS_ELEM_TPB
from original.numerics import numeric_mode_name


comptime N_ROWS = 8192
comptime N_COLS = 12
comptime REPEATS = 8

#: The gate's two padding widths. Interleaving them is what makes fit `k`
#: and fit `k + 2` receive memory that fit `k + 1` last touched.
comptime PAD_A = 0
comptime PAD_B = 37

#: The stages, in data-flow order. The first that moves names the kernel.
comptime N_STAGES = 7


def _hash_f32(i: Int, salt: Int) -> Float32:
    """The card fixture's own generator, bit-assembled.

    NO HOST FLOATING-POINT OPERATION builds this fixture, deliberately: a
    host `target += v * w` chain is IDENTITY_PATHS row 18's contraction
    decision and would hand two runs different design matrices before the
    first kernel ran."""
    var z = (
        UInt64(i) * 0x9E3779B97F4A7C15 + UInt64(salt + 1) * 0xBF58476D1CE4E5B9
    )
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    z = z ^ (z >> 31)
    # 24 mantissa bits into [1, 2), which is normal, finite and never a
    # denormal, so this probe cannot be reading row 10's flush story.
    var bits = UInt32(0x3F800000) | UInt32((z >> 40) & 0x7FFFFF)
    return bitcast[DType.float32](bits)


def _hash_device(
    ctx: DeviceContext,
    buf: DeviceBuffer[DType.float32],
    count: Int,
) raises -> UInt64:
    """FNV-1a64 over the RAW BYTES of a device buffer, read after the fit.

    Bytes and not values: a tolerance cannot see a summation order, and the
    whole question here is whether two runs of one program produced the same
    bits."""
    var host = ctx.enqueue_create_host_buffer[DType.float32](count)
    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=buf)
    ctx.synchronize()
    var h = fnv1a64_bytes(
        FNV_OFFSET, host.unsafe_ptr().bitcast[UInt8](), count * 4
    )
    _ = host^
    return h


def _one_fit(
    ctx: DeviceContext, pad: Int, mut c0_out: Float32
) raises -> List[UInt64]:
    """One untraced `ols_fit` at the gate's shape, then hash seven stages.

    THE FIT IS THE GATE'S FIT. Same entry point, same algo, same
    `elem_tpb`, same bit-assembled fixture, no trace. Everything that
    differs from `check_ols_is_launch_invariant::_fit_bits` is AFTER
    `ols_fit` returns.

    Returns the seven stage hashes, and writes coefficient 0 through
    `c0_out` so the caller can print the value the gate actually
    compares."""
    var a = ctx.enqueue_create_buffer[DType.float32](N_ROWS * N_COLS + pad)
    var b = ctx.enqueue_create_buffer[DType.float32](N_ROWS + pad)
    var w = ctx.enqueue_create_buffer[DType.float32](N_COLS + pad)
    var cov_a = ctx.enqueue_create_buffer[DType.float32](N_COLS * N_COLS + pad)
    var q = ctx.enqueue_create_buffer[DType.float32](N_COLS * N_COLS + pad)
    var qs = ctx.enqueue_create_buffer[DType.float32](N_COLS * N_COLS + pad)
    var s_vec = ctx.enqueue_create_buffer[DType.float32](N_COLS + pad)
    var ab = ctx.enqueue_create_buffer[DType.float32](N_COLS + pad)
    var inv = ctx.enqueue_create_buffer[DType.float32](N_COLS * N_COLS + pad)
    var a_alias = ctx.enqueue_create_buffer[DType.float32](
        N_ROWS * N_COLS + pad
    )
    var a_alias2 = ctx.enqueue_create_buffer[DType.float32](
        N_ROWS * N_COLS + pad
    )
    ctx.synchronize()

    # Poison every buffer the fit writes, with the SAME poison every repeat,
    # so a survivor is visible and so two repeats cannot differ because their
    # scratch differed. `check_ols_is_launch_invariant` varies the poison
    # between arms A and B; this file deliberately does not, because the
    # question here is what moves when NOTHING is varied.
    var big = ctx.enqueue_create_host_buffer[DType.float32](
        N_ROWS * N_COLS + pad
    )
    ctx.synchronize()
    for i in range(N_ROWS * N_COLS + pad):
        big.unsafe_ptr().unsafe_store(i, Float32(-987654.0))
    ctx.enqueue_copy(dst_buf=w, src_ptr=big.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=cov_a, src_ptr=big.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=q, src_ptr=big.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=qs, src_ptr=big.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=s_vec, src_ptr=big.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=ab, src_ptr=big.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=inv, src_ptr=big.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=a_alias, src_ptr=big.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=a_alias2, src_ptr=big.unsafe_ptr())
    ctx.synchronize()

    # ONE STAGING BUFFER PER COPY. THIS INSTRUMENT CARRIED THE DEFECT IT WAS
    # WRITTEN TO LOCALIZE. It copied the fixture block out of `ols_check.mojo`
    # verbatim, including the host rewrite of `big` between the two
    # `enqueue_copy` calls, so on a discrete GPU it corrupted its own design
    # matrix exactly the way the gate did. That is why it was silent on the
    # M4, where unified memory leaves no DMA to race, and it would have sent
    # a reader of its own stage table to the split-K kernel on a `step1.covA`
    # move when the cause is upstream of step 1. PORTING.md item 12.
    var big_b = ctx.enqueue_create_host_buffer[DType.float32](N_ROWS)
    ctx.synchronize()
    for i in range(N_ROWS * N_COLS):
        big.unsafe_ptr().unsafe_store(i, _hash_f32(i, 527))
    for i in range(N_ROWS):
        big_b.unsafe_ptr().unsafe_store(i, _hash_f32(i, 528))
    ctx.enqueue_copy(dst_buf=a, src_ptr=big.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=b, src_ptr=big_b.unsafe_ptr())
    ctx.synchronize()
    _ = big_b^

    ols_fit(
        ctx, a, b, w, cov_a, q, qs, s_vec, ab, inv, a_alias, a_alias2,
        N_ROWS, N_COLS, OLS_ALGO_EIG, False, False, OLS_ELEM_TPB,
    )

    # --- everything below is OUTSIDE the fit ------------------------------
    var out = List[UInt64]()
    out.append(_hash_device(ctx, cov_a, N_COLS * N_COLS))
    out.append(_hash_device(ctx, ab, N_COLS))
    out.append(_hash_device(ctx, s_vec, N_COLS))
    out.append(_hash_device(ctx, q, N_COLS * N_COLS))
    out.append(_hash_device(ctx, qs, N_COLS * N_COLS))
    out.append(_hash_device(ctx, inv, N_COLS * N_COLS))
    out.append(_hash_device(ctx, w, N_COLS))

    var hw = ctx.enqueue_create_host_buffer[DType.float32](N_COLS)
    ctx.enqueue_copy(dst_ptr=hw.unsafe_ptr(), src_buf=w)
    ctx.synchronize()
    c0_out = hw.unsafe_ptr().unsafe_load(0)

    _ = a
    _ = b
    _ = w
    _ = cov_a
    _ = q
    _ = qs
    _ = s_vec
    _ = ab
    _ = inv
    _ = a_alias
    _ = a_alias2
    _ = big^
    _ = hw^
    return out^


def _stage_name(i: Int) -> String:
    if i == 0:
        return "step1.covA    (gemm_tn)"
    if i == 1:
        return "step2.Ab      (xty_kernel)"
    if i == 2:
        return "step3.eigvals (jacobi + diagonal_to_vector)"
    if i == 3:
        return "step3.eigvecs (jacobi_eigh_kernel)"
    if i == 4:
        return "step4.QS      (divide_columns_by_nonzero)"
    if i == 5:
        return "step5.inv     (gemm_nt)"
    return "step6.w       (gemv_n)"


def main() raises:
    print(
        "ols_launch_localize: "
        + String(REPEATS)
        + " untraced fits, one process, one context, "
        + String(N_ROWS)
        + " x "
        + String(N_COLS)
        + ", mode ["
        + numeric_mode_name()
        + "]"
    )
    print(
        "Every fit is byte for byte the gate's fit. All hashing happens"
        " AFTER ols_fit returns, so no drain is added inside it."
    )
    print("")

    var hashes = List[UInt64]()
    var c0s = List[Float32]()

    var pads = List[Int]()
    with DeviceContext() as ctx:
        for r in range(REPEATS):
            var pad = PAD_A if r % 2 == 0 else PAD_B
            pads.append(pad)
            var c0 = Float32(0.0)
            var one = _one_fit(ctx, pad, c0)
            for s in range(N_STAGES):
                hashes.append(one[s])
            c0s.append(c0)

    # COMPARE WITHIN A PADDING WIDTH, NEVER ACROSS ONE. Under FAST a vendor
    # kernel is entitled to tile by pointer alignment, so a pad-0 fit and a
    # pad-37 fit disagreeing is the measurement that prices the pins, not a
    # defect. The gate says the same thing: A vs B is a REPORT under FAST and
    # only A vs C is an assertion. Two fits at the SAME width have nothing
    # left to differ about in either mode.
    var first_moved = -1
    for s in range(N_STAGES):
        var moved_a = False
        var moved_b = False
        var base_a = -1
        var base_b = -1
        for r in range(REPEATS):
            var h = hashes[r * N_STAGES + s]
            if pads[r] == PAD_A:
                if base_a < 0:
                    base_a = r
                elif h != hashes[base_a * N_STAGES + s]:
                    moved_a = True
            else:
                if base_b < 0:
                    base_b = r
                elif h != hashes[base_b * N_STAGES + s]:
                    moved_b = True
        var moved = moved_a or moved_b
        var mark = String("stable")
        if moved_a and moved_b:
            mark = "MOVED (both widths)"
        elif moved_a:
            mark = "MOVED (pad=0)"
        elif moved_b:
            mark = "MOVED (pad=37)"
        if moved and first_moved < 0:
            first_moved = s
        var line = String("  ") + _stage_name(s) + "  " + mark + "  "
        for r in range(REPEATS):
            line += hex(hashes[r * N_STAGES + s]) + " "
        print(line)

    print("")
    var c_line = String("  coefficient 0 across repeats: ")
    for r in range(REPEATS):
        c_line += hex(bitcast[DType.uint32](c0s[r])) + " "
    print(c_line)
    print("")

    if first_moved < 0:
        print(
            "VERDICT: nothing moved across "
            + String(REPEATS)
            + " fits at two interleaved padding widths, in mode ["
            + numeric_mode_name()
            + "]. Report that as SILENT AT THIS SHAPE ON THIS BOX AT THIS"
            " COMMIT. It is not a fixed bug and it does not close the"
            " CHANGELOG item: `check_ols_is_launch_invariant` remains the"
            " authority, and if the gate fails on the same box in the same"
            " leg while this is silent, the difference between the two"
            " programs is the next thing to read."
        )
    else:
        print(
            "VERDICT: the FIRST stage to move is "
            + _stage_name(first_moved)
            + ". Read the module docstring's table against that stage before"
            " changing any kernel."
        )
        raise Error(
            "ols_launch_localize: "
            + String(N_STAGES - first_moved)
            + " stage(s) from "
            + _stage_name(first_moved)
            + " onward are not reproducible across two identical fits in one"
            " process, in mode ["
            + numeric_mode_name()
            + "]"
        )
