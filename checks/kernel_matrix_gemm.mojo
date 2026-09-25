# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The kernel matrix rows of the identical GEMM.

These rows lived in checks/kernel_matrix.mojo until 2026-09-25 and moved here
unchanged, so an edit to them no longer changes the source closure (and the
release reuse identity) of every binding that imports the kernel matrix: only
gemm/checks/gemm_identical.mojo imports this file."""

from std.sys.compile import is_defined

from checks.kernel_matrix import (
    COLUMN_AMD,
    COLUMN_APPLE,
    COLUMN_CPU,
    COLUMN_NVIDIA,
)


def lib_gemm_stage_ftz_for[column: Int]() -> Bool:
    """Flush input operands once before shared staging, preserving their bits.

    NVIDIA already uses this transport. AMD CDNA qualified 2026-09-18:
    22.4% lower fixed-shape GEMM sum; 700-step enwik8/Pile comparisons
    improve 13.9%/15.3% with identical loss and state witnesses. The FMA
    rounding seam and fold topology do not change. Other columns remain
    on their existing path. See LANE_STATUS_gemm-kernel-speed.md.
    """
    return column == COLUMN_NVIDIA or column == COLUMN_AMD


def lib_postround_class_flush_for[column: Int]() -> Bool:
    """AMD post-round class flush, measured 2026-09-18.

    Same 262,144-triple hash as shipped round-then-flush; 700-step loss and
    final-state witnesses match NVIDIA on both corpora. The product seam
    change reduces AMD's late step by 8.43% across those corpora. Fold
    topology is unchanged.
    """
    return column == COLUMN_AMD


def lib_gemm_detect_seam_for[column: Int]() -> Bool:
    """SPELLING row (lane/amd-step-time, 2026-09-24): the IDENTICAL GEMM's
    per-step seam as a BARE (packed) FMA plus a per-thread subnormal witness,
    with an exact recompute of every cell of a thread that saw one.

    The contract step is `ftz(fma_rn(a, b, acc))` with `a`, `b` flushed at
    staging and `acc` the previous flushed step. A bare-FMA chain equals it
    step for step until some rounded step result is subnormal (a normal, zero,
    infinite or NaN result is its own flush). Each thread tests every step
    result of its cells with one class compare OR-ed into a flag; a thread
    whose flag stayed clear stored the contract's bits, a thread whose flag is
    set recomputes all of its cells with the exact contract step in the
    contract's order and fold tree and overwrites them. Same operands, same
    order, same rounding: a spelling, not a numeric change. It replaces the
    select-and-mask flush (a compare, a mask and a select per step on AMD) by
    the compare alone, and lets the FMAs pack two to an instruction.

    AMD only. The hardware alternative (the wave's sticky TRAPSTS.EXCP bits)
    was measured on an MI300X on 2026-09-24 to record NOTHING, not even an FMA
    consuming a subnormal (`gemm/checks/amd_excp_probe.mojo`), so the witness
    is computed in software. `-D MOJOLEARN_GEMM_NO_DETECT_SEAM` is the revert
    arm (the shipped class-flush seam).

    MEASURED AND NOT TAKEN (MI300X, 2026-09-24): bit-identical to the shipped
    seam on all 28 T3-shape GEMM cases, but SLOWER once the kernels stopped
    spilling (proj_fwd 1.28 -> 1.63 ms, head_fwd 56.6 -> 66.2 ms), and the
    recompute makes operand words near 2^-110 cost 4x to 16x. So the row is
    off on every column; `-D MOJOLEARN_GEMM_DETECT_SEAM=1` turns it on for
    AMD as a trial arm."""
    comptime if is_defined["MOJOLEARN_GEMM_DETECT_SEAM"]():
        return column == COLUMN_AMD
    return False


def lib_gemm_leaf_split_for[column: Int]() -> Bool:
    """SCHEDULING row (lane/amd-step-time, 2026-09-24): every IDENTICAL GEMM
    call `choose_gemm_plan` sends to the TUNED 128x128 plan runs the
    `ksplit_leaf` geometry wherever its rule takes the call: the 128x128 group
    kernel over the FINEST power-of-two leaf groups whose node workspace fits
    the cap, then the group fold (DEVIATION 2591's arm, unchanged). A group size
    reaches no leaf boundary and no tree level (long-k brief section 5), so it
    is a schedule and never a result.

    AMD, MEASURED on a Hot Aisle MI300X with the launch bound in place (T3
    shard shapes, `bench/gemm_excp_ab_main.mojo`, every output hash equal to
    the shipped dispatch): the twelve step calls weighted by their per-shard
    counts 685 -> 606 ms (head_dA 79.9 -> 58.0, proj_fwd 1.26 -> 1.01,
    proj_dA 1.29 -> 1.00, gateup_dA 3.20 -> 2.73, down_fwd 3.26 -> 2.71 ms).
    Every other column False (unmeasured). `-D MOJOLEARN_GEMM_NO_LEAF_SPLIT=1`
    is the revert arm."""
    comptime if is_defined["MOJOLEARN_GEMM_NO_LEAF_SPLIT"]():
        return False
    return column == COLUMN_AMD


def lib_gemm_window_admit_for[column: Int]() -> Bool:
    """SPELLING row (lane/nvidia-step-time, 2026-09-25): the IDENTICAL kpack
    GEMM drops the per-step flush multiply on a WINDOW it has proven cannot
    produce a subnormal step result, and runs the contract's exact step
    everywhere else.

    The contract step is `ftz(fma_rn(a, b, acc))` with `a`, `b` flushed at
    staging and `acc` the previous flushed step (NVIDIA spells the flush as
    `mul.rn.ftz` by one). While a window is staged, the block takes the
    minimum biased exponent field `Ea` over the NONZERO words of its A tile
    and `Eb` over its B tile (zeros constrain nothing; Inf and NaN read 255).
    Every nonzero flushed operand is a multiple of 2^(E - 150), so every
    product the window forms is a multiple of G = 2^(Ea + Eb - 300). If
    `Ea + Eb >= 174`, G >= 2^-126; if in addition every accumulator entering
    the window is a multiple of 2^-126 (true at a leaf start, where it is
    +0.0, and after every admitted window), then every exact step result is a
    multiple of 2^-126, every rounded one is too (a multiple of G below 2^24 G
    is representable; above, it rounds to a multiple of its ulp, a power of
    two at least G), and a nonzero multiple of 2^-126 is not subnormal. So
    `ftz` is the identity on every step result of the window and the bare
    `fma.rn` IS the contract's step: the same instruction on the same
    operands in the same order, one rounding each. A window that fails the
    test, and every later window of the same leaf, runs the exact
    two-instruction step. The decision is block-uniform.

    NVIDIA only (the column whose step is `fma.rn` + `mul.rn.ftz`).
    MEASURED on a RunPod H100 80GB HBM3 (2026-09-25, leg 2): the T3-shape
    GEMM A/B hashes equal the shipped step's on all 36 cases (ordinary
    operands, the subnormal-forcing kind, and the mixed kind); the sabotage
    arm differs on all 12 subnormal-forcing cases; every call about 20
    percent faster; the lean B4 step 0.608 -> 0.532 s with equal witnesses;
    the T3 replays of steps 101..103 and 1999..2000 PASS against the H100
    chain at 33.80 and 34.04 s a step (38.83 before); GEMM 401.7 -> 323.6 ms
    a shard.
    `-D MOJOLEARN_GEMM_NO_WINDOW_ADMIT=1` is the revert arm (the shipped
    step on every window); `-D MOJOLEARN_GEMM_SABOTAGE_ADMIT_ALWAYS=1`
    admits every window and must FAIL the subnormal-forcing operands.

    Apple (lane/apple-identical-gemm, 2026-09-25): the same admission on the
    tuned kernel (`TUNED_WINDOW_ADMIT` in gemm/checks/gemm_identical.mojo),
    whose step there is `ftz(fma)` under the block admission. An admitted
    window drops the `ftz`: every exact step result is zero or at least
    2^-126 in magnitude, so Apple's flush before round FMA returns the same
    normal or zero a round then flush FMA does, and the flush is the identity.
    The block admission and its exact recompute are unchanged. PROVEN bit for
    bit on the M4 at small shapes (gemm_rtf_boundary_check, adversarial and
    mixed exponent operands, before and after hashes equal; the reach
    sabotage `-D MOJOLEARN_GEMM_SABOTAGE_WINDOW_ADMIT_STEP=1` fails). Speed
    NOT measured on Apple (owed). `-D MOJOLEARN_GEMM_NO_WINDOW_ADMIT=1` is
    the revert arm on both columns."""
    comptime if is_defined["MOJOLEARN_GEMM_NO_WINDOW_ADMIT"]():
        return False
    return column == COLUMN_NVIDIA or column == COLUMN_APPLE


def lib_gemm_kpack_narrow_for[column: Int]() -> Bool:
    """SCHEDULING row (lane/nvidia-step-time, 2026-09-25): the IDENTICAL kpack
    GEMM kernel runs a 128x64 output tile (a thread's register tile 8x4, 32
    cells instead of 64) under a 512-thread launch bound on its 256-thread
    launch, which budgets 128 registers a thread so two blocks share an SM
    and each hides the other's staging barrier, prefetch and fold. A thread's
    cells, each cell's ascending chain, its leaves and its fold tree are
    unchanged: a tile shape is a schedule and never a result.

    NVIDIA, MEASURED on a RunPod H100 80GB HBM3 (2026-09-25, leg 2): the
    T3-shape GEMM A/B hashes equal the 128x128 kernel's on all 36 cases
    (three operand kinds), every call 5 to 9 percent faster; the lean B4 step
    0.534 -> 0.508 s with equal witnesses; the T3 replay of steps 101..103
    PASS at 32.43 s a step. Every other column False (their kernels compile
    exactly as before). `-D MOJOLEARN_GEMM_NO_KPACK_NARROW=1` is the revert."""
    comptime if is_defined["MOJOLEARN_GEMM_NO_KPACK_NARROW"]():
        return False
    return column == COLUMN_NVIDIA


def lib_gemm_mfma_for[column: Int]() -> Bool:
    """SPELLING row (lane/amd-step-time, 2026-09-24): the IDENTICAL GEMM's
    TUNED 128x128 calls on the matrix cores (`identical_gemm_mfma_kernel`):
    each contract step is one `v_mfma_f32_32x32x1f32` (K = 1: one product plus
    its accumulator, `fma_rn`, measured equal to the VALU fma) followed by the
    flush as a product by one under the wave's MODE f32 output flush (measured
    equal to `ftz` on every element). Same operands, same order, same
    rounding, same fold tree. AMD only.

    MEASURED on a Hot Aisle MI300X (2026-09-24, leg 6): the T3-shape GEMM
    A/B, 28 of 28 output hashes identical to the VALU kernels (the kind that
    forces subnormal intermediates included), most calls about 1.9x faster;
    lean B4 step 0.763 -> 0.617 s with every step witness equal; the T3
    replays of steps 101..103 (ckpt 100) and 1999..2000 (ckpt 1998) PASS
    against the H100 chain at 39.5 s a step (49.1 s before).
    `-D MOJOLEARN_GEMM_NO_MFMA=1` is the revert arm."""
    comptime if is_defined["MOJOLEARN_GEMM_NO_MFMA"]():
        return False
    return column == COLUMN_AMD


def lib_gemm_block_parallelism_for[column: Int]() -> Int:
    """SCHEDULING row, SHIPPED since DEVIATION 2595 (2026-09-11; first added by DEVIATION 2591 as a trial-arm row): how many 256-thread GEMM blocks the column runs side by side. A value above 0 TURNS ON the `ksplit` default in `gemm/checks/gemm_identical.mojo::identical_gemm_shipped_into`: every call the long-k group rule takes (section 4, rules 1, 2 and 4, at `S` = this value) runs the 128x128 group kernel over power-of-two leaf groups plus one fold launch, and every other call runs the plan `choose_gemm_plan` picks, as before. 0 turns it off: the dispatch compiles to the old line and the TUNED 128x128 plan runs exactly as it did. NVIDIA 132, MEASURED: the H100's SM count, and the value the `ksplit` arm ran at on the H100 leg that flipped it (bench/results/e1g/2026-09-11_152822-nvidia-h100-80gb-hbm3-gemm-longk, lean step geomean 0.895 on enwik8 and Pile GitHub, every step witness equal). AMD 110, MEASURED 2026-09-11 on the Hot Aisle MI300X (bench/results/e1g/2026-09-11_164818-amd-mi300x-hotaisle-gemm-longk): lean step 1.953 -> 1.198 s on both corpora, `ksplit` verdict FLIP at geomean 0.6136 and `ksplit_leaf` 0.6090, every step witness equal to shipped. The classical callers were then checked to HOLD under this row (68be1c79: SVC, kmeans, PCA and KDE on the same MI300X). This sentence previously said "AMD 0, OFF UNTIL THE MI300X LEG DECIDES" and contradicted the body below it for two days. The trial arm still runs on AMD at the column's reading through `lib_gemm_block_parallelism_trial_for`. Every other column 0 (Apple included, so the Apple identity card compiles the old line). A wrong value costs time and can never move a bit, because the group size reaches no leaf boundary and no tree level (brief section 5.5)."""
    if column == COLUMN_NVIDIA:
        return 132
    if column == COLUMN_AMD:
        # Measured 2026-09-11 on the Hot Aisle MI300X with the trial arm at S=110
        # (e1g/2026-09-11_164818-amd-mi300x-hotaisle-gemm-longk): lean step
        # 1.953 -> 1.198 s on both corpora, geomean 0.614, witnesses equal.
        return 110
    if column == COLUMN_CPU:
        # 0: the host oracle (`gemm_oracle`) folds the contract tree without
        # groups, and the dispatch compiles to the old line as it did under
        # the Apple fallthrough.
        return 0
    return 0


def lib_gemm_kernel_body_for[column: Int]() -> Int:
    """SCHEDULING row, SHIPPED since DEVIATION 2707 (2026-09-13): which KERNEL BODY the IDENTICAL GEMM dispatch runs on every call the TUNED 128x128 plan serves. 0 is the 2595 dispatch as it stood (`identical_gemm_tuned_kernel` where the ksplit rule declines the call, `identical_gemm_ksplit_kernel` groups where it takes it). 1 is the `kpack_hg` body (DEVIATION 2706): `identical_gemm_kpack_kernel` at the shipped 128x128 geometry with the padded, 16-byte-aligned packed page (2700, 2703), ONE 8-wide conflict-free shared store per thread per window instead of sixteen scalar stores at a four-way bank conflict (gather staging), and the fold's flush spelled as the hardware `mul.rn.ftz` by one that the step seam already uses (one instruction for six); the same group rule at the same row, the same fold tree, the same words at the same addresses in the same order, so no bit moves and the M4 arms check and the H100 step check say so. NVIDIA 1, MEASURED 2026-09-13 on a RunPod H100 (bench/results/e1g/2026-09-13_175602-nvidia-h100-gemm-hfgs): lean step 0.232 -> 0.211 s on enwik8 and Pile GitHub (geomean 0.9085), GEMM sum 143 -> 122 ms (0.852), every step witness equal to shipped; the decomposition that named the two costs is bench/results/e1g/2026-09-13_174125-nvidia-h100-gemm-diag2 (staging phase a third of the window, fold a fifth). AMD 1 as well, MEASURED 2026-09-13 on a Hot Aisle MI300X (the body's comment names the leg): the gather staging is placement and applies, the fold flush is NVIDIA's instruction and compiles out there. Apple and every other column 0: they compile exactly the line they compiled before. A wrong value costs time and can never move a bit. This row is the switch: 0 here is the revert."""
    if column == COLUMN_NVIDIA:
        return 1
    if column == COLUMN_AMD:
        # AMD 1, MEASURED 2026-09-13 on a Hot Aisle MI300X (gfx942), 8core VM,
        # one heat window, commit bfde0442
        # (bench/results/e1g/2026-09-13_193949-amd-mi300x-hotaisle-gemm-amd-row):
        #   verdict kpack_gs FLIP geomean=0.9570 enwik8=0.9559 pilegithub=0.9580
        #   verdict kpack_hg FLIP geomean=0.9585 enwik8=0.9573 pilegithub=0.9597
        #   every step witness equal to shipped on both corpora; step check PASS
        # GEMM sum 592 -> 561 ms (kpack_gs 0.947) and 559 ms (kpack_hg 0.944).
        # On this column the body's hardware fold flush compiles out
        # (`comptime if HW and TUNED_HW_FTZ_FMA`, NVIDIA only), so 1 here IS
        # the gather staging alone and the two arms price the same to 0.15
        # percent; one row value keeps one code path. Shipped-build gate on
        # the MI300X: brief section 18.4.
        return 1
    if column == COLUMN_CPU:
        return 0  # the revert line; no GEMM kernel body runs on the host
    return 0


def lib_gemm_block_parallelism_trial_for[column: Int]() -> Int:
    """SCHEDULING row (DEVIATION 2595, 2026-09-11, trial arm only): the `S` the `ksplit` TRIAL arm reads, so a leg can still force the arm on a column whose shipped row is 0. The shipped row wherever it is above 0 (NVIDIA 132, so the arm and the default split identically there). AMD 110, from a READING, not a measurement (DEVIATION 2591): the attention brief section 11.1 records 110 CUs (pinned to the MI250X; the MI325X and MI300X counts are not in the repository) and resident blocks per CU as `min(2048 // 256, 65536 // page bytes)`; the shipped 128x128 GEMM block holds two 20,480 B pages (40,960 B), so one block per CU and 110 side by side. The MI300X leg's CONTROL pair `ctl_nt_1536x1408x768` / `ctl_nt_1664x1408x768` (`bench/gemm_step_price_main.mojo`) reads the real value. Every other column 0, meaning no reading: the arm then takes the finest split the workspace cap allows. The shipped build reads it nowhere."""
    comptime shipped = lib_gemm_block_parallelism_for[column]()
    if shipped > 0:
        return shipped
    if column == COLUMN_AMD:
        return 110
    if column == COLUMN_CPU:
        return 0  # no reading: the host has no blocks to run side by side
    return 0


def gemm_wide_split_for[column: Int]() -> Bool:
    """Execution-only wide split-K tiles on the measured NVIDIA column. The CPU column answers False.

    The 128x128/KS16 tile reduces operand reloads for complete output tiles.
    Other columns keep their previous dispatcher pending local timings;
    every column's all-plan correctness gate still exercises the new tile.
    """
    if column == COLUMN_CPU:
        return False
    return column == COLUMN_NVIDIA
