# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of Mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The NVIDIA identical-column rounded FMA (`fma.rn.f32`) of the hot kernel
loops: the pinned distance tile, the identical GEMM and fused attention.

lane fix-hdbscan-nv (2026-11-13, owed row `wave2-hdbscan-nv`). 62a6577b39
(2026-11-12) moved these sites from the `llvm.nvvm.fma.rn.f` intrinsic
(0.8.55's form) to the inline-asm helper `checks.numerics.nv_fma_rn_f32`,
because Mojo 1.2 dropped `llvm.nvvm.mul.rn.*`; the fma intrinsic moved with it
for uniformity, not because it was dropped. An inline-asm block is opaque to
LLVM, which may no longer schedule, hoist or combine the loads around it in
an inner loop. On record (NVIDIA L40S, 0.8.55 -> candidate 4c2340386abc, AMD
flat): hdbscan taxi fit 435 -> 552 ms, knn-reg taxi 11.5 -> 18.9 ms, knn-clf
taxi 14.2 -> 21.6 ms; lm-train-step 35.6 -> 48.9 ms against 0.8.66.

A/B (owed row wave2-hdbscan-nv): A = the intrinsic, B = the asm helper
(`-D MOJOLEARN_IDN_NV_FMA_ASM_OLD`). Same PTX instruction, same bits, same
fold order. VERDICT 2026-11-18, the asm helper is the default again (lane
fx-verdicts; A/B ab2, NVIDIA L40S host c6dd9ebea3e3, driver 595.112.07,
wheel d652a4b0a1 and its B set, base and arm back to back in one job, one
run per arm): digests and quality equal on all 30 cells; asm/intrinsic
geomean 0.971 over the 29 cells scored alike (ols taxi fit 19.93 -> 16.02 ms
(0.80), ols taxi infer 1.85 -> 1.37 (0.74), ridge taxi 3.84 -> 3.16 (0.82),
the rest 0.93-1.06: hdbscan taxi 548 -> 562 (1.03), knn taxi 0.97, knn-clf
taxi 1.04, knn-reg taxi 1.01, lm-train-step 1.03, tsvd istella 1.06). The
pca taxi infer cell (90.28 -> 2.30 ms, 0.03) is not a speed result: its
base warmup took 118 ms, so the base was scored as one call, the arm's 8.7
ms, so the arm was the median of 20 calls (AGENTS.md small-row rule); the
arm's slowest call was 6.86 ms, digests equal. The intrinsic was not the
cause of the HDBSCAN / k-NN regression against 0.8.55 (portable_sqrtf's
candidate loop was: checks/numerics.mojo NUMERICS_SQRT_COMPTIME_STEP).
`-D MOJOLEARN_IDN_NV_FMA_ASM_OFF` restores the intrinsic. NVIDIA-only code
generation: the other columns do not compile this function. Limits: NVIDIA
only (AGENTS.md part 8). Call only in
code compiled for an NVIDIA GPU (every caller sits under an NVIDIA-only
comptime branch). The `mul.rn` / `mul.rn.ftz` intrinsics are gone; lane
fix-nv-mul-rn-via-fma (2026-11-14) spells the pinned product and the flush by
one as `fma.rn(a, b, -0.0)` / `fma.rn.ftz(x, 1.0, -0.0)` (same IEEE words,
schedulable) in `checks.numerics` (`nv_mul_rn_kernel_f32`,
`nv_mul_rn_kernel_f64`, `nv_flush_by_one_kernel`; B =
`-D MOJOLEARN_IDN_NV_MUL_RN_ASM_OLD`). They live there, not here, because
`checks.numerics.ftz` and `pinned_mul_f32` call them and this module imports
`checks.numerics`.
"""
from std.sys import llvm_intrinsic
from std.sys.compile import is_defined
from checks.numerics import nv_fma_rn_f32

comptime IDN_NV_FMA_ASM = not is_defined["MOJOLEARN_IDN_NV_FMA_ASM_OFF"]()


@always_inline
def nv_fma_rn_kernel(a: Float32, b: Float32, c: Float32) -> Float32:
    """NVIDIA `fma.rn.f32` (`a*b + c`, one rounding, never flushed); see above."""
    comptime if IDN_NV_FMA_ASM:
        return nv_fma_rn_f32(a, b, c)
    return llvm_intrinsic["llvm.nvvm.fma.rn.f", Float32, has_side_effect=False](a, b, c)
