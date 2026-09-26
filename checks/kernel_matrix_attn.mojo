# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The kernel matrix rows of the fused attention.

These rows lived in checks/kernel_matrix.mojo until 2026-09-25 and moved here
unchanged, so an edit to them no longer changes the source closure (and the
release reuse identity) of every binding that imports the kernel matrix: only
transformer/impl/llama/fused_attention.mojo imports this file."""

from std.sys.compile import is_defined

from checks.kernel_matrix import (
    COLUMN_AMD,
    COLUMN_APPLE,
    COLUMN_CPU,
    COLUMN_NVIDIA,
)


def attn_masked_tail_replay_for[column: Int]() -> Bool:
    """Exact omitted-tail replay, measured H100 and MI300X 2026-09-18.

    NVIDIA: paired 700-step enwik8/Pile GitHub runs: 3697/1768 backward
    corner refusals become zero; every loss and state witness matches. Late
    medians 0.455439 -> 0.197157 and 0.377134 -> 0.196831 seconds (geomean
    2.1038x).

    AMD (Hot Aisle MI300X, same probe, seed and shape): all 700 losses, the
    six state hashes at steps 0/699 AND every per-step, per-layer status and
    replay-site vector equal NVIDIA's, in both arms. Refusals 3697/1768 -> 0
    (dQ replay 3449/1612 sites; zdot 0). Late medians 0.835384 -> 0.632183
    and 0.773939 -> 0.630843 seconds (1.3214x / 1.2268x, geomean 1.2733x).
    AMD's default arm has no estash zdot kernel, so its zdot keeps refusing
    on a corner (zero in training on either corpus).
    Merged with AMD GEMM operand staging, an MI325X enwik8 pair again equals
    NVIDIA everywhere: 0.762693 -> 0.550012 seconds (1.3867x).

    Apple: the Metal corner fixtures (zdot, 64 dQ cells, dk/dv) equal eager
    bit for bit, the replay-disabled arms differ (0x80000000 vs 0x00000000),
    and all 130 sites equal NVIDIA's. Apple's DEFAULT arm (`stash_tiled`)
    reaches none of the replay kernels, so on Apple this row is INERT in
    default training; a reduced HD64 training witness under the NVIDIA
    schedule define equals NVIDIA's.
    """
    return column == COLUMN_NVIDIA or column == COLUMN_AMD or column == COLUMN_APPLE


def attn_zdot_rows_per_block_for[column: Int]() -> Int:
    """SCHEDULING row (DEVIATION 2528, 2026-09-11, trial arm only): query rows per 256-thread block of the fused attention's register-blocked y/dy kernel (`fused_bwd_ydy_tiled_kernel`), 64 or 32. The kernel's shared page is `(2 * rows + 128) * 20` floats (20,480 B at 64, 15,360 B at 32), and on a column whose shared memory is partitioned per compute unit the page bounds the resident blocks. The rows are a schedule, never a numeric term: every chain keeps its terms and order at either value. UNMEASURED on every column. AMD reads 32 as the variant section 11.3 named to price; the page-only count (3 blocks x 64 rows vs 4 x 32 rows per CU) does not favor it, so the AMD leg prices both through the `_r32` / `_r64` arm names and this row follows that measurement. The shipped build reads it nowhere."""
    if column == COLUMN_AMD:
        return 32
    if column == COLUMN_CPU:
        return 64  # the non-vendor default; no attention kernel runs here
    return 64


def attn_fwd_rows_per_block_for[column: Int]() -> Int:
    """SCHEDULING row (DEVIATION 2531, 2026-09-11): query rows per 256-thread block of the fused attention's second-round forward kernel (`fused_attn_forward_r2_kernel`), 64 (the shipped hd-64 sstash geometry) or 32, read by the bare `_fgrid` arm token (`_fgrid_r32` / `_fgrid_r64` force it). The kernel's shared page is `(32 * 64 + rows * 35) * 4` bytes (17,152 B at 64 rows, 12,672 B at 32), and on a column whose shared memory is partitioned per compute unit the page bounds the resident blocks. The rows are a schedule, never a numeric term: the score, denominator and context chains keep their terms and order at either value, and the row maximum is an `identical_fmax` fold whose grouping is free. NVIDIA 32, MEASURED (DEVIATION 2534, H100 leg bench/results/e1g/2026-09-11_154257-nvidia-h100-80gb-hbm3-attention-round3, commit 5bcfa71d): the lean LM step under `stash_tiled_fgrid_r32` was 0.3718 / 0.3716 s against `stash_tiled` 0.3845 / 0.3819 s (enwik8 / Pile GitHub), every step witness equal, and `fgrid_r64` priced at 1.00x of stash_tiled on real activations while `fgrid_r32` priced 1.07x. AMD 32 is still the variant brief section 11.4 named to price (the page-only count, 3 blocks x 64 rows against 5 x 32 per CU, does not settle it); THE MI300X LEG DECIDES IT through the `_fgrid_r32` / `_fgrid_r64` arm names. Every other column 64, unmeasured. The shipped default arm (`attn_default_arm_for`) forces its rows with `_fgrid_r32`, so this row never moves a shipped path."""
    if column == COLUMN_NVIDIA:
        return 32
    if column == COLUMN_AMD:
        return 32
    if column == COLUMN_CPU:
        return 64  # the non-vendor default; no attention kernel runs here
    return 64


def attn_dkdv_keys_per_block_for[column: Int]() -> Int:
    """SCHEDULING row (DEVIATION 2597, 2026-09-11): keys per 256-thread block of the fused attention's trial dk/dv folds over the stash (`fused_bwd_dkdv_r2_kernel`, and `fused_bwd_kvfold_r2_kernel` under the `_kvsplit` token), 64 (the shipped `fused_bwd_dkdv_tiled_pf_kernel` geometry) or 32, read by the bare `_kvgrid` arm token (`_kvgrid_r32` / `_kvgrid_r64` force it). At 32 keys a thread holds 8 dk and 8 dv accumulators instead of 16 and 16, and the joint page is `(2 * 16 * 64 + 2 * 16 * keys) * 4` bytes (16,384 B at 64, 12,288 B at 32; a `_kvsplit` fold page is half that). The keys per block are a schedule, never a numeric term: every dk and dv chain keeps its terms and its order (heads of the kv group ascending, queries ascending over the key's visible range) at either value. AMD 32, MEASURED: DigitalOcean MI325X leg bench/results/e1g/2026-09-11_180903-amd-mi325x-do-attention-dkdv (commit 5cc3b8df), `stash_tiled_fgrid_r32_qres_pf_kvgrid_r32` against `baseline` lean step 1.623 / 1.633 -> 1.376 / 1.370 s (enwik8 / Pile GitHub, FLIP geomean 0.8436, every step witness equal), in-step dk/dv 169.8 ms (baseline) -> 21.2 ms; section 16 had read 32 from the tiled dk/dv thread state (32 accumulators and 10 operand registers per thread, twice the tiled dq fold's). The AMD shipped default (`attn_default_arm_for`) forces the same 32 with `_kvgrid_r32` (brief section 18), so this row and the default agree on AMD and a bare `_kvgrid` resolves to the default's instantiation there. Every other column 64, unmeasured. A shipped build reads this row only for a default carrying bare `_kvgrid`, which no column's default does."""
    if column == COLUMN_AMD:
        return 32
    if column == COLUMN_CPU:
        return 64  # the non-vendor default; no attention kernel runs here
    return 64


comptime ATTN_DEFAULT_WORD_BASELINE = 0
comptime ATTN_DEFAULT_WORD_STASH_TILED = 7
"""The attention arm word `stash_tiled`: bits 1 (fwd_sstash), 2 (bwd_stash) and 4 (bwd_tiled) of transformer/impl/llama/fused_attention.mojo (DEVIATIONS 2525 to 2527). The matrix cannot import that file (it imports this one), so the word is a literal here and fused_attention.mojo asserts at build time that it equals its own composition."""

comptime ATTN_DEFAULT_WORD_STASH_TILED_FGRID_R32_QRES_PF = 3175
"""The attention arm word `stash_tiled_fgrid_r32_qres_pf`: stash_tiled (7) | 64 (fwd_grid, DEVIATION 2531) | 2048 (forward rows 32) | 32 (fwd_qres, DEVIATION 2530) | 1024 (preflush, DEVIATION 2533) = 3175. fused_attention.mojo asserts at build time that it equals its own composition."""

comptime ATTN_DEFAULT_WORD_STASH_TILED_FGRID_R32_QRES_PF_KVGRID_R32 = 52327
"""The attention arm word `stash_tiled_fgrid_r32_qres_pf_kvgrid_r32`: stash_tiled_fgrid_r32_qres_pf (3175) | 16384 (bwd_kvgrid, DEVIATION 2597) | 32768 (dk/dv keys per block 32) = 52327. fused_attention.mojo asserts at build time that it equals its own composition (`ATTN_ARM_R3_KVGRID_R32_DEFAULT`)."""

comptime ATTN_DEFAULT_WORD_STASH_TILED_FGRID_R32_QRES_PF_ESTASH_DRES_KVGRID_R32 = 6343783
"""The attention arm word `stash_tiled_fgrid_r32_qres_pf_estash_dres_kvgrid_r32`: stash_tiled_fgrid_r32_qres_pf_kvgrid_r32 (52327) | 2097152 (bwd_estash, DEVIATION 2650) | 4194304 (estash_dres, DEVIATION 2651) = 6343783. fused_attention.mojo asserts at build time that it equals its own composition (`ATTN_ARM_R3_KVGRID_R32_ESTASH_DRES_DEFAULT`)."""
comptime ATTN_DEFAULT_WORD_STASH_TILED_FGRID_R32_QRES_PF_ESTASH_DRES_KVGRID_R32_BSWZ = 14732391
"""The word above plus DEVIATION 2900's `_bswz` bit (8388608): the causal
block-index map of the four kernels that arm runs. The matrix cannot import
transformer/impl/llama/fused_attention.mojo (that file imports this one), so
the word is a literal here and that file asserts at build time that it equals
its own composition (`ATTN_ARM_R3_KVGRID_R32_ESTASH_DRES_BSWZ_DEFAULT`)."""


def attn_fwd_launch_bound_for[column: Int]() -> Int:
    """SCHEDULING row (lane/nvidia-step-time, 2026-09-25): the launch bound
    `fused_attn_forward_r2_kernel` declares on its 256-thread launch. NVIDIA
    1024: 64 registers a thread, so four blocks share an SM and hide the
    kernel's stash round trips. MEASURED on a RunPod H100 80GB HBM3 (leg 3,
    T3 shape): 5.69 ms a launch at 256 (96 registers, two blocks an SM),
    4.18 at 768, 3.57 at 1024; lean B4 step 0.506 -> 0.481 s with equal
    witnesses (with the dq row below); replay of steps 101..103 PASS at 30.55
    s a step. Every other column 1024, the bound its backend assumes without
    a declaration (gfx942's default flat work-group size; Metal carries none),
    so they are expected to compile what they compiled before (the AMD
    re-proof is owed before a release). `-D MOJOLEARN_ATTN_NO_LAUNCH_BOUND=1`
    gives NVIDIA 256 (the 255-register budget it had with no declaration).
    Register allocation only: no operation, operand or order changes."""
    comptime if is_defined["MOJOLEARN_ATTN_NO_LAUNCH_BOUND"]():
        return 256 if column == COLUMN_NVIDIA else 1024
    return 1024


def attn_dq_launch_bound_for[column: Int]() -> Int:
    """SCHEDULING row (lane/nvidia-step-time, 2026-09-25): the same for
    `fused_bwd_dq_tiled_pf_kernel`. NVIDIA 768 (85 registers, three blocks an
    SM): 2.49 -> 2.43 ms a launch; 1024 (64 registers) was slower (2.62).
    Every other column 1024 (its backend's default, see above)."""
    comptime if is_defined["MOJOLEARN_ATTN_NO_LAUNCH_BOUND"]():
        return 256 if column == COLUMN_NVIDIA else 1024
    return 768 if column == COLUMN_NVIDIA else 1024


def attn_default_arm_for[column: Int]() -> Int:
    """ROUTING row (DEVIATION 2534, 2026-09-11): the attention arm word the SHIPPED build runs on this column (`ATTN_ARM_DEFAULT` in transformer/impl/llama/fused_attention.mojo; a `-D MOJOLEARN_ATTN_ARM_TRIAL=1` build runs it when MOJOLEARN_ATTN_ARM is unset and keeps every other arm selectable by name). Every arm is bit-equal to the eager oracle by the identity arguments of brief sections 4, 12, 14 and 16, so this row picks a schedule and never a result. NVIDIA `stash_tiled_fgrid_r32_qres_pf`, MEASURED: H100 leg bench/results/e1g/2026-09-11_154257-nvidia-h100-80gb-hbm3-attention-round3 (commit 5bcfa71d), lean LM step 0.3845 / 0.3819 s under stash_tiled against 0.3346 / 0.3340 s (enwik8 / Pile GitHub), every step witness equal, fwd+bwd on real activations 1.41x of stash_tiled; CONTRIBUTING.md (Performance claims) flips it. AMD `stash_tiled_fgrid_r32_qres_pf_kvgrid_r32` (ATTN_DEFAULT_WORD_STASH_TILED_FGRID_R32_QRES_PF_KVGRID_R32), MEASURED on the DigitalOcean MI325X against the previous AMD default `baseline`, every step witness equal (the comment in the body names the evidence and the verdict); a shipped build compiles its DEVIATION 2597 dk/dv kernel because the default carries it (brief section 18). Apple and every other column `stash_tiled` (unmeasured for the round 3 and 2597 arms as a price). `-D MOJOLEARN_ATTN_DEFAULT_R3_EVERY_COLUMN=1` returns the NVIDIA word on every column, so a no-trial build on a Mac reaches the shipped round 3 branch; `-D MOJOLEARN_ATTN_DEFAULT_KVGRID_EVERY_COLUMN=1` returns the previous NVIDIA word (now AMD's) on every column, so the same build reaches the shipped DEVIATION 2597 dk/dv branch; `-D MOJOLEARN_ATTN_DEFAULT_ESTASH_EVERY_COLUMN=1` returns the NVIDIA word as of the estash flip (DEVIATION 2657, without DEVIATION 2900's `_bswz` bit) on every column, so a no-trial build on a Mac reaches the shipped DEVIATION 2650 / 2651 estash branch (DEVIATION 2657, `ATTN_SHIPPED_BWD_ESTASH`; this is how the M4 gates that branch, since Apple's own default carries no estash bit). `-D MOJOLEARN_ATTN_DEFAULT_BSWZ_EVERY_COLUMN=1` returns the CURRENT NVIDIA word on every column, so a no-trial build on a Mac reaches the shipped DEVIATION 2900 branch (`ATTN_DEFAULT_BSWZ`; this is how the M4 gates a branch Apple's own default does not carry). Check knobs, the `MOJOLEARN_EXPERIMENTAL_SMALLK_IDENTICAL` pattern; never a shipped build; at most one of the four."""
    comptime assert not (is_defined["MOJOLEARN_ATTN_DEFAULT_R3_EVERY_COLUMN"]() and is_defined["MOJOLEARN_ATTN_DEFAULT_KVGRID_EVERY_COLUMN"]()), (
        "MOJOLEARN_ATTN_DEFAULT_R3_EVERY_COLUMN and"
        " MOJOLEARN_ATTN_DEFAULT_KVGRID_EVERY_COLUMN each name a different"
        " default for every column; define at most one"
    )
    comptime assert not (is_defined["MOJOLEARN_ATTN_DEFAULT_R3_EVERY_COLUMN"]() and is_defined["MOJOLEARN_ATTN_DEFAULT_ESTASH_EVERY_COLUMN"]()), (
        "MOJOLEARN_ATTN_DEFAULT_R3_EVERY_COLUMN and"
        " MOJOLEARN_ATTN_DEFAULT_ESTASH_EVERY_COLUMN each name a different"
        " default for every column; define at most one"
    )
    comptime assert not (is_defined["MOJOLEARN_ATTN_DEFAULT_KVGRID_EVERY_COLUMN"]() and is_defined["MOJOLEARN_ATTN_DEFAULT_ESTASH_EVERY_COLUMN"]()), (
        "MOJOLEARN_ATTN_DEFAULT_KVGRID_EVERY_COLUMN and"
        " MOJOLEARN_ATTN_DEFAULT_ESTASH_EVERY_COLUMN each name a different"
        " default for every column; define at most one"
    )
    comptime assert not (is_defined["MOJOLEARN_ATTN_DEFAULT_R3_EVERY_COLUMN"]() and is_defined["MOJOLEARN_ATTN_DEFAULT_BSWZ_EVERY_COLUMN"]()), (
        "MOJOLEARN_ATTN_DEFAULT_R3_EVERY_COLUMN and"
        " MOJOLEARN_ATTN_DEFAULT_BSWZ_EVERY_COLUMN each name a different"
        " default for every column; define at most one"
    )
    comptime assert not (is_defined["MOJOLEARN_ATTN_DEFAULT_KVGRID_EVERY_COLUMN"]() and is_defined["MOJOLEARN_ATTN_DEFAULT_BSWZ_EVERY_COLUMN"]()), (
        "MOJOLEARN_ATTN_DEFAULT_KVGRID_EVERY_COLUMN and"
        " MOJOLEARN_ATTN_DEFAULT_BSWZ_EVERY_COLUMN each name a different"
        " default for every column; define at most one"
    )
    comptime assert not (is_defined["MOJOLEARN_ATTN_DEFAULT_ESTASH_EVERY_COLUMN"]() and is_defined["MOJOLEARN_ATTN_DEFAULT_BSWZ_EVERY_COLUMN"]()), (
        "MOJOLEARN_ATTN_DEFAULT_ESTASH_EVERY_COLUMN and"
        " MOJOLEARN_ATTN_DEFAULT_BSWZ_EVERY_COLUMN each name a different"
        " default for every column; define at most one"
    )
    comptime if is_defined["MOJOLEARN_ATTN_DEFAULT_BSWZ_EVERY_COLUMN"]():
        return ATTN_DEFAULT_WORD_STASH_TILED_FGRID_R32_QRES_PF_ESTASH_DRES_KVGRID_R32_BSWZ
    comptime if is_defined["MOJOLEARN_ATTN_DEFAULT_ESTASH_EVERY_COLUMN"]():
        return ATTN_DEFAULT_WORD_STASH_TILED_FGRID_R32_QRES_PF_ESTASH_DRES_KVGRID_R32
    comptime if is_defined["MOJOLEARN_ATTN_DEFAULT_KVGRID_EVERY_COLUMN"]():
        return ATTN_DEFAULT_WORD_STASH_TILED_FGRID_R32_QRES_PF_KVGRID_R32
    comptime if is_defined["MOJOLEARN_ATTN_DEFAULT_R3_EVERY_COLUMN"]():
        return ATTN_DEFAULT_WORD_STASH_TILED_FGRID_R32_QRES_PF
    if column == COLUMN_NVIDIA:
        # DEVIATION 2657. Measured 2026-09-11 on a RunPod H100 80GB HBM3
        # against the previous NVIDIA default
        # stash_tiled_fgrid_r32_qres_pf_kvgrid_r32, commit 8dc33f00, every step
        # witness equal on both corpora, Apple vs NVIDIA identity trace
        # IDENTICAL over 60 stages
        # (bench/results/e1g/2026-09-11_215636-nvidia-h100-80gb-hbm3-attention-estash):
        #   verdict stash_tiled_fgrid_r32_qres_pf_estash_dres_kvgrid_r32 FLIP
        #   geomean=0.8207 enwik8=0.8226 pilegithub=0.8190 (same pod)
        # Lean step 0.2922 / 0.2916 -> 0.2403 / 0.2388 s (enwik8 / Pile GitHub).
        # In-step zdot 66.4 -> 14.9 ms; on real activations the backward is
        # 7.52 -> 3.23 ms (fwd+bwd 1.85x, the forward untouched at 1.00x).
        # The runner-up on the same leg, _estash without _dres, FLIP 0.8348:
        # CONTRIBUTING.md (Performance claims) takes the winner. The register lens (brief
        # section 20.2) reads the same: the shipped zdot kernel is 134 regs and
        # 1 block per SM, _estash 125 and 2, _estash_dres 64 and 4.
        # Before it: stash_tiled_fgrid_r32_qres_pf_kvgrid_r32 (e1g/...185833,
        # FLIP geomean 0.9908); before that stash_tiled_fgrid_r32_qres_pf
        # (round 3, e1g/...154257).
        #
        # DEVIATION 2900 (brief section 22). Measured 2026-09-17 on a RunPod
        # H100 80GB HBM3, pod d7piefs556qlqe, commit de7d2063e, one heat
        # window, against the estash word above
        # (bench/results/e1g/2026-09-17_201140-nvidia-h100-attention-bswz):
        #   verdict stash_tiled_fgrid_r32_qres_pf_estash_dres_kvgrid_r32_bswz
        #   FLIP geomean=0.9592 enwik8=0.9583 pilegithub=0.9601
        #   witnesses_equal_baseline=True on both corpora
        # Lean step 0.20648 / 0.20651 -> 0.19786 / 0.19827 s (enwik8 / Pile
        # GitHub). It changes NO arithmetic: `_blk_map` is a bijection over
        # block_idx.x that hands the same (tile, head, batch) triples out
        # heaviest first instead of lightest first, so the hardware, which
        # dispatches in increasing block_idx.x, stops ending each causal
        # kernel in a tail of long blocks running alone.
        # In-step, per the lmtiming probes of the same leg (enwik8):
        #   forward r2   20.85 -> 17.19 ms  (0.825), 100 -> 96 regs, 2 -> 2 blocks/SM
        #   dq tiled     12.49 ->  9.68 ms  (0.775), 118 -> 128 regs, 2 -> 2
        #   dk/dv kvgrid 10.07 ->  7.10 ms  (0.706),  63 ->  63 regs, 4 -> 4
        #   zdot estash  14.79 -> 15.61 ms  (1.056),  64 ->  70 regs, 4 -> 3
        # The zdot kernel REGRESSES and the readback says why: the map's
        # index arithmetic costs it 6 registers, which crosses its occupancy
        # cliff from 4 resident blocks per SM to 3. The other three kernels
        # keep their occupancy and win on schedule alone. The net is -8.6 ms
        # of a 206.5 ms step; the zdot regression is a measured 0.83 ms left
        # on the table and brief section 22.7 names the follow-on.
        return ATTN_DEFAULT_WORD_STASH_TILED_FGRID_R32_QRES_PF_ESTASH_DRES_KVGRID_R32_BSWZ
    if column == COLUMN_APPLE:
        # lane/apple-identical-neural, 2026-09-26: the round 3 word on Apple.
        # Measured on the M4 (10-core GPU), byte LM 1 x 2048, d768, 12 heads,
        # 2 layers, V 50,257, consecutive resident lean steps, alternating
        # builds, two rounds: stash_tiled 2.025 / 2.051 s, _r3 1.928 / 1.832,
        # _kvgrid 2.029 / 1.833, _estash 1.797 / 1.764, _bswz 1.698 / 1.739.
        # NOT an estash word: the estash words keep a [B, nh, L, S] exp stash
        # per layer from forward to backward (201 MB a layer at 1 x 2048 x 12
        # heads), and on this 16 GB Mac a 12-layer step then refused at the
        # first forward ("infinity in hidden_states at flat index 0"; 4, 6
        # and 8 layers trained, peak RSS 8.8 GB at 8), while _r3 trained at 12
        # layers (7.78 s a step). A Mac with the memory can take the estash
        # words by name. Per-step witnesses equal to stash_tiled's.
        # Schedules only: every arm is bit-equal to the eager oracle.
        return ATTN_DEFAULT_WORD_STASH_TILED_FGRID_R32_QRES_PF
    if column == COLUMN_AMD:
        # DEVIATION 2657 ON AMD TOO. Measured 2026-09-12 on a Hot Aisle MI300X
        # (gfx942) against the previous AMD default
        # stash_tiled_fgrid_r32_qres_pf_kvgrid_r32, commit bb679f19, one VM and
        # one heat window, every step witness equal on both corpora
        # (bench/results/e1g/2026-09-12_133010-amd-mi300x-hotaisle-attn-estash):
        #   verdict stash_tiled_fgrid_r32_qres_pf_estash_dres_kvgrid_r32 FLIP
        #   geomean=0.9716 enwik8=0.9724 pilegithub=0.9709
        #   witnesses_equal_baseline=True on both corpora
        # Lean step 0.7573 / 0.7593 -> 0.7363 / 0.7372 s (enwik8 / Pile GitHub).
        # The gain is a third of NVIDIA's (0.8207 there) and the register lens
        # on this same VM says why. The shipped zdot kernel `zdot_stash_pf` is
        # 118 regs at a 17,696 byte page and ALREADY fits 3 blocks per CU here,
        # where on the H100 the same kernel was 134 regs at 1 block per SM. The
        # flipped kernel `zdot_estash_dres_pf` is 60 regs at 12,480 bytes and 5
        # blocks per CU (the plain `_estash` variant is 116 regs, 10,432 bytes,
        # 4 blocks). So AMD goes 3 -> 5 blocks where NVIDIA went 1 -> 4, and it
        # was never as starved to begin with, which is the whole difference in
        # the two gains. CONTRIBUTING.md (Performance claims) takes the win anyway: below 1 on
        # both corpora with the bits unmoved, and the rule sets no magnitude bar.
        # Before it: stash_tiled_fgrid_r32_qres_pf_kvgrid_r32, measured
        # 2026-09-11 on the DigitalOcean MI325X against baseline, commit
        # 5cc3b8df (e1g/2026-09-11_180903-amd-mi325x-do-attention-dkdv, FLIP
        # geomean=0.8436, in-step dk/dv 169.8 -> 21.2 ms); before that baseline
        # (e1g/2026-09-11_171959-amd-mi300x-runpod-attention-three).
        #
        # DEVIATION 2900 ON AMD TOO (lane/amd-step-time, 2026-09-24): NVIDIA's
        # causal block-index map `_bswz`, measured on a Hot Aisle MI300X at the
        # T3 shard shape (batch 4, length 2048) with the GEMM launch bound, the
        # AMD leaf split and the class-spelled ftz in place, one VM and one
        # heat window, the trial build against itself
        # (bench/results/amd_step_time_2026-09-24/legs/*-leg5):
        #   lean B4 step 0.8179 (this word) -> 0.7632 s (+ _bswz), 0.9331x;
        #   every step witness (loss, gradients, parameters, m, v, flags)
        #   equal to the default's and to the shipped build's.
        # A bijection over block_idx.x: the same (tile, head, batch) triples
        # heaviest first. `_kvgrid_r64`, with or without `_bswz`, was 2.4 to
        # 2.7 s (not taken).
        return ATTN_DEFAULT_WORD_STASH_TILED_FGRID_R32_QRES_PF_ESTASH_DRES_KVGRID_R32_BSWZ
    if column == COLUMN_CPU:
        # The non-vendor default, the word the Apple fallthrough compiled into
        # the byte LM host binding. No fused attention kernel runs on the
        # host; the host step goes to the transformer oracles.
        return ATTN_DEFAULT_WORD_STASH_TILED
    return ATTN_DEFAULT_WORD_STASH_TILED
