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

A (default) = the intrinsic again. B = `-D MOJOLEARN_IDN_NV_FMA_ASM_OLD`, the
asm helper. Same PTX instruction, same bits, same fold order. Call only in
code compiled for an NVIDIA GPU (every caller sits under an NVIDIA-only
comptime branch). The `mul.rn` / `mul.rn.ftz` asm stays: those intrinsics are
gone.
"""
from std.sys import llvm_intrinsic
from std.sys.compile import is_defined
from checks.numerics import nv_fma_rn_f32

comptime IDN_NV_FMA_ASM_OLD = is_defined["MOJOLEARN_IDN_NV_FMA_ASM_OLD"]()


@always_inline
def nv_fma_rn_kernel(a: Float32, b: Float32, c: Float32) -> Float32:
    """NVIDIA `fma.rn.f32` (`a*b + c`, one rounding, never flushed); see above."""
    comptime if IDN_NV_FMA_ASM_OLD:
        return nv_fma_rn_f32(a, b, c)
    return llvm_intrinsic["llvm.nvvm.fma.rn.f", Float32, has_side_effect=False](a, b, c)
