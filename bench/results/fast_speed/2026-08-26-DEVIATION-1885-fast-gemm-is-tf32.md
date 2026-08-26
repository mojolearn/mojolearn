# DEVIATION 1885: the FAST path's vendor GEMM is a PRECISION CUT, and it was not declared

## What was claimed and what is true

DEVIATIONS 1873/1875/1876/1877 were measured on an H100 on 2026-08-26 and
the speedups are enormous and real:

| lane | shape | before | after | speedup |
|---|---|---|---|---|
| transformer | llama8b.prefill.t1 | 1009.778 ms | 1.885 ms | **535.70x** |
| transformer | llama8b.prefill.t8 | 980.440 | 2.885 | 339.79x |
| transformer | llama8b.decode.t1.ctx512 | 936.078 | 4.615 | 202.84x |
| transformer | llama8b.prefill.t128 | 950.180 | 5.614 | 169.26x |
| transformer | llama8b.prefill.t512 | 996.401 | 10.783 | **92.41x** |
| mlp | llama8b.prefill.t512 | 51.065 | 0.704 | 72.52x |
| attention | llama8b.prefill.t512 | 14.774 | 2.402 | 6.15x |

Against the best torch arm, the Llama decoder block went from **686x slower
to 7.12x slower** at t512.

**PART OF THAT IS A MANTISSA CUT WE DID NOT DECLARE.**

## The evidence, and why it is not a summation-order story

Every lane's agreement against the torch fp32 reference, before and after:

| lane | uses GEMM | max_abs_diff before | after | worse by |
|---|---|---|---|---|
| **rmsnorm** | **NO** | 4.77e-06 | **4.77e-06** | **UNCHANGED** |
| attention | yes | 0.0183 | 0.859 | 47x |
| transformer | yes | 0.0889 | 4.334 | 49x |
| mlp | yes | 0.0163 | 3.888 | **239x** |

`rmsnorm` is the control and it did not move by one bit. Every lane that
routes a GEMM moved by one to two orders of magnitude. A reordered
summation does not do that; **a shorter mantissa does.**

The corroborating measurement is already in this repository. On the same
H100, `cublas-fp32` runs at 44.4 TFLOP/s and `cublas-tf32` at 207.5, and
MAX's `linalg.matmul` measured **200** -- on the TF32 line, not the FP32
one. `_fast_vendor_gemm` (DEVIATION 1876) routes the FAST path into exactly
that call. TF32 carries **10 explicit mantissa bits against fp32's 23**.

## The second symptom, same cause

`verify.mamba_block.fast` FAILED on this leg (exit 1) while
`verify.mamba_block.identical` passed (exit 0):

    mamba_check: CLAUSE (a) FAILED, 15 stages differ from the oracle,
    first at norm.out on 10 of 32 cells

Fifteen stages, differing from one ULP upward, in the FAST build only. That
check was green before DEVIATION 1876 and it is the same precision cut
arriving at a bit-exactness assertion. **The IDENTICAL build is unaffected**
-- `_fast_vendor_gemm` is gated behind
`comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL`, and that gate held.

## Why this matters more than the number it costs

A FAST path is allowed to be non-deterministic and is allowed to take a
vendor's fastest kernel. **It is not allowed to take a precision cut
silently.** Every one of the speedups above sat in a table beside an
agreement column computed against an **fp32** reference, which reads as "we
got 535x faster and the answer is as accurate as before". It is not.

It also changes which opponent is fair. This repository already wrote the
TF32 trap down for the GEMM lane -- our FAST arm's fair opponent there is
`cublas-tf32`, not `cublas-fp32`, precisely because our arm took the same
precision cut. That reasoning was applied to `gemm` and then the same cut
was let into `transformer`, `attention` and `mlp` through the back door
without anybody re-deriving it. The ranked table happens to pick
`torch-gpu-tf32` as best opponent on most of those rows, so the SPEED
ratios are roughly fair; the AGREEMENT lines are not.

## What is owed

1. **Declare it at the source.** `_fast_vendor_gemm` must say in its own
   comment that on NVIDIA it is a TF32-class kernel, and every lane routing
   through it must emit a note saying so, so no future table can carry the
   speedup without the caveat.
2. **Give the seq lanes a TF32 reference for agreement.** Comparing our
   TF32 output against torch fp32 measures the precision cut, not a defect.
   `torch-gpu-tf32` already runs in these lanes; the AGREE line should be
   computed against it.
3. **Decide whether FAST should be able to opt out.** If MAX exposes an
   fp32-accumulate path, the honest shape is both arms measured and a
   KERNEL MATRIX ROW, the same discipline the kNN arms just got -- not a
   default chosen once and never re-derived.
4. **`mamba_check`'s FAST arm needs a stated tolerance**, or an explicit
   record that its clause (a) is an IDENTICAL-only assertion. Right now a
   green FAST run is impossible and the failure looks like a regression
   rather than a declared consequence.

Until 1 and 2 are done, quote these speedups **only** with the sentence
"part of this is a TF32 precision cut" attached.
