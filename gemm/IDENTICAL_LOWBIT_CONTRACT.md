# The low-bit profiles: `mojolearn.identical.gemm.bf16f32.v1` and `mojolearn.identical.gemm.int8i32.v1`

Lane lane/identical-lowbit-inference, 2026-09-17. DEVIATIONS 2900 to 2909.
Answers: `gemm/host/gemm_lowbit_oracle.mojo`. Device: `gemm/checks/gemm_lowbit.mojo`.
Gates: `gemm/checks/gemm_lowbit_check.mojo`. Seams: `checks/numerics.mojo`
(the block headed LOW-BIT STORAGE SEAMS).

## STATUS

Both profiles are built and gated on one Apple M4 (Metal) against their host
oracles, with the value-flip sabotage arm of each seen to fail the oracle
gates. Neither has a three-vendor card yet. Until one exists, every sentence
below is a construction argument and the M4 measurement, not a certificate.

`gemm/IDENTICAL_FP32_CONTRACT.md` (fp32.v1) forbids a flag inside itself that
switches the accumulator or the operand type. These two profiles are the
names that rule anticipated. Neither changes one character of fp32.v1's
arithmetic; the bf16 profile calls it and the int8 profile does not need it.

## 0. What each profile is

**bf16f32.v1.** A bf16 is the top sixteen bits of a float32. Widening one is
a shift and cannot round. The profile is therefore fp32.v1 applied to the
exactly widened operands, plus a named narrowing seam for a caller who wants
the output stored as bf16. The leaf rule, the fold tree, the FMA pin and the
seven flush seams are fp32.v1's, inherited, and every fp32.v1 gate that
passes on widened operands is a gate this profile passed.

**int8i32.v1.** Both operands are int8 codes with one power-of-two scale
per row. The dot product is an integer sum of integer products, exact, so
the fold rule, the FMA pin and the accumulator flush of fp32.v1 have nothing
to act on. What the profile pins is the scale rule, the rounding of the
codes, the integer-to-float conversion of the sum and the one multiply that
applies the scale. OP_NT only: an activation row and a weight row are each
scaled along `k`, and that is the only orientation where a per-row scale on
both sides describes a per-cell scale.

## 1. The seams

| clause | seam | spelling | deviation |
|---|---|---|---|
| L-1 | bf16 widening | `bf16_bits_to_f32`: bits shifted up 16; exact | 2900 |
| L-2 | bf16 narrowing | `f32_to_bf16_bits_rne`: `ftz` first, then round to nearest even on the low 16 bits; NaN kept quiet with its sign; overflow rounds to the infinity | 2901 |
| L-3 | int8 row scale | `int8_row_exponent`: `e = floor(log2 absmax) - 6`, read from the exponent field, so `absmax * 2^-e` lies in `[64, 128)`; an all-zero row takes `e = 0`; the absmax is a maximum, exact and order-free | 2905 |
| L-4 | int8 code | `quantize_int8_value`: `clamp(rne(ftz(x) * 2^-e), -127, 127)`; the multiply is by a power of two and exact unless it lands below the smallest normal, where the flush zeroes it; `rne` is the magic-constant rounding `f32_round_half_even`; a NaN codes to 0 | 2902, 2905 |
| L-5 | sum to float | `i32_to_f32_pinned`: the Int32 sum split into a 20-bit high part and a 12-bit low part, each converted exactly, the high part scaled by `2^12` exactly, one IEEE addition | 2904 |
| L-6 | dequantization | `dequant_int8_pinned`: `ftz(identical_mul(f, 2^(ea + eb)))`, one multiply by a power of two built from its exponent field (`pow2_f32`), then the flush; exponents above 127 give the infinity and below -126 give `+0.0` | 2903, 2904 |
| L-7 | int8 accumulation | Int32, `p` ascending, one product per step. Exact for `k <= 131072` (`INT8_MAX_K`); a larger `k` is refused by name | 2907 |
| L-8 | bf16 execution plans | FUSED (`identical_gemm_bf16w_flat_kernel`, the fp32 flat plan with the right operand widened at the load) below `BF16W_FUSED_MAX_CELLS` output cells, WIDEN (`bf16_widen_kernel` then `identical_gemm_into[False]`) above; both are the profile and `check_bf16_plans_agree` requires their bits to match on every shape, so the threshold is scheduling | 2906 |

The narrowing seam L-2 is ordered flush-then-round. They do not commute at
the subnormal boundary: a float32 subnormal rounds to a bf16 subnormal or to
the smallest bf16 normal, and the flush would then be a no-op on a value the
fp32 profile has already declared zero. Flushing first keeps the bf16 image
of a flushed float32 equal to the bf16 image the fp32 profile's own output
seam (5g) would have produced. Every bf16 subnormal therefore narrows to a
signed zero, and the gate counts all 254 of them.

## 2. What is promised

Given the same operand bits, `m`, `n`, `k` and `op`, the same output bits on
every certified backend, under the same scope rules as fp32.v1 (row-major,
contiguous, no epilogue, NaN cells compared as NaN). For int8i32.v1 the
promise covers the codes and exponents the quantizer produces from a float32
row as well as the product, so a row quantized on one vendor and a row
quantized on another carry the same codes.

## 3. What is not promised

- Agreement with the float32 product. A bf16 weight is a rounded weight and
  an int8 weight is a rounded weight with a coarser step; the profiles pin
  the rounded computation, not its distance from the unrounded one.
- Agreement between the two profiles, or between either and a vendor's
  bf16 or int8 matrix unit. A tensor core's internal rounding is the
  vendor's and undocumented; nothing here emulates one (the direction
  `gemm/IDENTICAL_FP32_CONTRACT.md` section 0 declines by name).
- Speed. The fused bf16 plan reads half the weight bytes of the fp32 plan;
  the int8 plan is one thread per cell with no tiling. Neither has been
  timed against the fp32 plans, and no number in this tree says either is
  faster.
- Any orientation but OP_NT for int8i32.v1.

## 4. The sabotage arms

`-D MOJOLEARN_LOWBIT_SABOTAGE=1` flips the value of every cell the two device
kernels store. `-D MOJOLEARN_HOST_SABOTAGE=1` flips the value of every leaf
partial in `gemm_oracle` (which the bf16 oracle reaches) and every
dequantized cell in `gemm_int8_oracle`. Both are value arms, for the reason
`gemm/host/gemm_oracle.mojo` records: an exact sum folds an order arm away,
and every int8 sum is exact. Measured on the M4, 2026-09-17: the device arm
fails `check_bf16_device_matches_oracle`, `check_bf16_plans_agree` and
`check_int8_device_matches_oracle`; the host arm fails the two oracle gates
and leaves the plan gate passing, which is correct, since the host arm does
not reach a device kernel.

## 5. Owed

A three-vendor card for both profiles at the nine shapes of the gate, the
same way `gemm/README.md` records fp32.v1's 62 shapes; a tiled int8 plan if
the flat one is ever the step's cost; and a bf16 activation seam (L-2 applied
between blocks), which no block uses today because the inference classes
keep activations in float32 and store only weights low-bit.
