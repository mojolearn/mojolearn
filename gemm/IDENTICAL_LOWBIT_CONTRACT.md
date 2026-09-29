# The low-bit profiles: `mojolearn.identical.gemm.bf16f32.v1` and `mojolearn.identical.gemm.int8i32.v1`

Lane lane/identical-lowbit-inference, 2026-09-17. DEVIATIONS 2900 to 2909.
Clause L-9 (matrix units): lane lane/int8-mma, 2026-09-17, DEVIATION 2910.
Answers: `gemm/host/gemm_lowbit_oracle.mojo`. Device: `gemm/checks/gemm_lowbit.mojo`
and, for the int8 matrix-unit plan, `gemm/checks/gemm_int8_mma.mojo`.
Gates: `gemm/checks/gemm_lowbit_check.mojo`. Seams: `checks/numerics.mojo`
(the block headed LOW-BIT STORAGE SEAMS).

## STATUS

Both profiles are built and gated on an Apple M4 (Metal, flat int8), an NVIDIA
H100 (`bench/results/lowbit/2026-09-17_h100-lowbit-mma/`, int8 on the IMMA
units) and an AMD MI325X (`bench/results/lowbit/2026-09-17_mi325x-lowbit-mma/`,
int8 on the MFMA units), each against the same host oracles, each with the
value-flip sabotage arm of each profile seen to fail the oracle gates. That is
three vendor columns on the nine gate shapes of both profiles. The fourteen
harness lanes that store block weights low-bit have the M4 columns only. Until one exists, every sentence
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
| L-9 | int8 execution plans, matrix units | FLAT (`identical_gemm_int8_flat_kernel`, one thread per cell) on every column; MMA (`identical_gemm_int8_mma_kernel`, the vendor's integer matrix unit, Int32 accumulation, zero-code padding) on a column whose kernel-matrix row `lib_int8_matrix_unit_for` says True; both are the profile and `check_int8_mma_matches_flat` requires their bits to match on every shape, so the choice is scheduling; section 1.1 | 2910 |

The narrowing seam L-2 is ordered flush-then-round. They do not commute at
the subnormal boundary: a float32 subnormal rounds to a bf16 subnormal or to
the smallest bf16 normal, and the flush would then be a no-op on a value the
fp32 profile has already declared zero. Flushing first keeps the bf16 image
of a flushed float32 equal to the bf16 image the fp32 profile's own output
seam (5g) would have produced. Every bf16 subnormal therefore narrows to a
signed zero, and the gate counts all 254 of them.

### 1.1 Clause L-9, the integer matrix units

**The construction.** An int8 product is an integer of magnitude at most
`127 * 127 = 16129` and is exact. A sum of exact integers in an Int32 that
cannot overflow (L-7 bounds `k` so that `16129 * k < 2^31`) is the same
integer under every order and every grouping of its terms. So the k-tile of
a matrix unit (32 on both vendors), the fragment a thread holds, and the
order in which the unit adds the products of one step are SCHEDULING: the
Int32 that leaves the unit's last step is the Int32 `int8_dot_cell` produces
with `p` ascending, and the profile's only floating steps, L-5 and L-6, run
on that Int32 through the same `dequant_int8_pinned` on either plan. This is
the direction section 3 declines for a vendor's bf16 or fp8 unit, and it is
admissible here for one reason only: an integer unit has no rounding to be
undocumented.

**The units.** NVIDIA: IMMA, `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`
(sm_80 and later), reached as the NVVM intrinsic
`llvm.nvvm.mma.m16n8k32.row.col.s8`. AMD CDNA: MFMA `v_mfma_i32_16x16x32_i8`
(gfx942, gfx950), reached as `llvm.amdgcn.mfma.i32.16x16x32.i8`. The public
`mma` of the shipped stdlib (`max.gpu.compute.mma.mma`, MAX 26.5) dispatches
float shapes only, so the int8 forms are reached by name through
`llvm_intrinsic`, as `llvm.nvvm.fma.rn.f` already is in this tree.

**The padding rule.** A `k` that is not a multiple of the unit's k-tile,
and a row or column of a warp tile that lies beyond `m` or `n`, are filled
with the ZERO CODE, never with a float: a zero code contributes exactly
nothing to an integer sum, so the padded product is the unpadded product.
Rows beyond `m` and columns beyond `n` are masked at the store.

**The capability row.** `checks/kernel_matrix.mojo::lib_int8_matrix_unit_for`:
True on NVIDIA and AMD, False on Apple and on the CPU column. Apple stays on
the flat kernel: Metal's simdgroup matrix takes half and float operands
only. `-D MOJOLEARN_INT8_FORCE_FLAT=1` keeps the flat plan on every column
without changing the row, so a box that has the unit can run every gate
through the flat plan and compare.

**What is promised under L-9.** The same output bits from either plan on
every shape, and the same bits as `gemm_int8_oracle`. **What is not.**
Speed: neither plan has been timed on any vendor. And this clause is a
construction argument until the gate runs on a box that has a unit:
`check_int8_mma_matches_flat` and `check_int8_device_matches_oracle` on an
H100 and on an MI300X or MI325X (`tools/lowbit_mma_leg.sh`), with the
sabotage arms seen to fail; the M4 cannot run the MMA plan at all.

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
  bf16 or fp8 matrix unit. A floating tensor core's internal rounding is
  the vendor's and undocumented; nothing here emulates one (the direction
  `gemm/IDENTICAL_FP32_CONTRACT.md` section 0 declines by name). The
  INTEGER matrix units are the one exception, and clause L-9 says why: an
  int8 by int8 product into an Int32 has no rounding to document, so the
  int8 profile may run on one and still be the profile.
- Speed. The fused bf16 plan reads half the weight bytes of the fp32 plan;
  the int8 flat plan is one thread per cell with no tiling, and the int8
  MMA plan (L-9) runs on the integer matrix unit of NVIDIA and AMD. None
  has been timed against the fp32 plans, and no number in this tree says
  any is faster.
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
same way `gemm/README.md` records fp32.v1's 62 shapes; the first run of the
int8 MMA plan (L-9) on an H100 and on an MI300X or MI325X, whose fragment
layouts are read from the two ISA documents and unverified until
`check_int8_mma_matches_flat` runs there; a CDNA2 (gfx90a) form of that
plan, which needs the k16 MFMA; and a bf16 activation seam (L-2 applied
between blocks), which no block uses today because the inference classes
keep activations in float32 and store only weights low-bit.

## 6. THE FIFTEEN-BIT PROFILE: `mojolearn.identical.gemm.int15i64.v1`

Lane lane/lowbit-int15, 2026-09-29. DEVIATIONS 2965 to 2973. Clauses W-1 to
W-8. Answers: `gemm/host/gemm_int15_oracle.mojo`. Device:
`gemm/checks/gemm_int15.mojo`. Gates: `gemm/checks/gemm_int15_check.mojo` and
`gemm/checks/gemm_int15_sim_check.mojo`. Seams: `checks/numerics_int15.mojo`,
a file of its own so that no binding which imports `checks/numerics.mojo` is
built from a changed source.

This section changes no character of `fp32.v1`, `bf16f32.v1` or
`int8i32.v1`. Sections 0 to 5 above are as they were.

### 6.0 What the profile is, and why it exists

`int8i32.v1` one step wider. Both operands are integer codes with one
power-of-two scale per row; the dot product is an exact integer sum; the
only floating steps are the conversion of that sum to float32, one multiply
by `2^(ea + eb)` and the flush. OP_NT only, for `int8i32.v1`'s reason.

The quality lane (`lane/lowbit-quality`, `bench/lowbit_quality/`) measured
fifteen-bit codes on both operands inside the noise of the fp32 baseline on
SmolLM2-360M, and `int8i32.v1` far outside it. THE CODES OF THIS PROFILE
ARE THAT LANE'S, `bench/lowbit_quality/arith.py`, kind `int15`. The number
belongs to the model path and is recorded there; this section pins the
arithmetic and claims no quality.

### 6.1 The clauses

| clause | what | spelling | deviation |
|---|---|---|---|
| W-1 | row scale | `int15_row_exponent`: `e = floor(log2 absmax) - 13`, read from the exponent field, so `absmax * 2^-e` lies in `[8192, 16384)`; an all-zero row takes `e = 0`; the absmax is `int8i32.v1`'s, a maximum after the flush, which a NaN never wins and an infinity does | 2965 |
| W-2 | code | `quantize_int15_value`: `clamp(rne(ftz(ftz(x) * 2^-e)), -16383, 16383)`, stored as Int16; `rne` is `f32_round_half_even`; a NaN codes to 0, BEFORE the scaling and AFTER it (a zero of a row whose scale is the infinity is `0 * inf`); an integer has one zero | 2966 |
| W-3 | pieces | `int15_piece_lo`: `c mod 128` as a floor modulus, a mask, in `[0, 127]`; `int15_piece_hi`: `floor(c / 128)`, an arithmetic shift, in `[-128, 127]`; `c = hi * 128 + lo`; both are int8 | 2967 |
| W-4 | piece accumulation | three Int32 accumulators: `HH`, `MID = HL + LH`, `LL`. Exact for `k <= 65536` (`INT15_MAX_K`); a larger `k` is refused by name, by every plan and by the oracle | 2968 |
| W-5 | recombination | `int15_recombine`: `HH * 2^14 + MID * 2^7 + LL` in Int64, two shifts and two additions | 2969 |
| W-6 | sum to float | `i64_to_f32_pinned`: the magnitude split into two 24-bit parts, each converted exactly from an Int32, the high part scaled by `2^24` exactly, ONE IEEE addition; at and above `2^48` the low sixteen bits fold into a sticky bit first and the result is scaled by `2^16` exactly | 2970 |
| W-7 | dequantization | `dequant_int15_pinned`: `ftz(identical_mul(f, 2^(ea + eb)))`, `pow2_f32` as in L-6 | 2971 |
| W-8 | execution plans | FLAT (the Int16 codes, one Int32 product per step, the sum in Int64, one thread per cell), PIECES (the int8 planes, the three accumulators of W-4, one thread per cell), MMA (the planes on the vendor's integer matrix unit through `gemm_int8_mma.mojo`'s own fragment loads and step, four products per k-tile of 32, zero-code padding). All three are the profile; `check_int15_plans_agree` requires their bits to match on every shape, so the choice is scheduling | 2972 |

### 6.2 THE BOUNDS

Every bound below is derived here and has a planted case AT it, which must
be exact, and where the bound is a limit on `k` a planted case ABOVE it,
which must not be (`check_int15_bounds_are_where_the_contract_says`).
`int8i32.v1`'s bound on `k` does not carry over: L-7 assumes a magnitude of
127, and a high piece reaches -128.

**(a) The range of each piece.** A code lies in `[-16383, 16383]`.
`lo = c mod 128` is in `[0, 127]` by the definition of a floor modulus.
`hi = floor(c / 128)`: the smallest is `floor(-16383 / 128) = -128`, the
largest `floor(16383 / 128) = 127`. `hi = -128` holds for the 127 codes
from -16383 to -16257 and for no other, and there `lo = c + 16384` is in
`[1, 127]`. So a piece takes the one int8 value whose negation is not an
int8. Nothing in the profile negates a piece, and the integer matrix units
take `-128` as the signed byte it is: the gate plants it on every lane of
both vendors' units (the code -16383 on both operands). Planted:
`check_int15_pieces_cover_every_code`, all 32767 codes.

**(b) One piece product, and one step of a unit.** The largest magnitude of
a piece product is `128 * 128 = 16384 = 2^14` (`HH`, both pieces -128). The
others: `|hi * lo| <= 128 * 127 = 16256`, `lo * lo <= 127 * 127 = 16129`.
One step of a unit adds 32 products to an Int32: at most
`32 * 16384 = 2^19` in magnitude, and the step is an integer addition with
no saturation asked for, so the accumulator after a step is the accumulator
before it plus the exact sum of the 32 products, provided that sum of the
whole `k` fits, which is (c) and (d).

**(c) Each piece sum over the whole `k`.** Every partial sum of `k` terms of
magnitude at most `B` has magnitude at most `B * k`, under any order and
any grouping. `HH`: `16384 * k <= 2^31 - 1` holds up to `k = 131071`, and
at `k = 131072` every term at its largest gives `2^31` exactly, one past
the largest Int32. `LL`, `HL` and `LH` alone are smaller. Planted: the code
-16383 on both operands, `(hi, lo) = (-128, 1)`, exact at `k = 131071` and
wrapped at `k = 131072`.

**(d) The cross term.** `HL` and `LH` each reach -16256 on the same term
(the code -16257 on both operands, `(hi, lo) = (-128, 127)`), so their sum
reaches -32512 per term. CHOSEN: `HL` and `LH` SHARE ONE INT32
ACCUMULATOR, and the bound on `k` is taken for the sum:
`32512 * k <= 2^31 - 1` holds up to `k = 66052` (`32512 * 66052 =
2147482624`) and fails at `k = 66053`. WHY: the alternative, two
accumulators kept apart until they are Int64, admits `k` up to 131071 by
(c), but the profile stops at a power of two, and the power of two below
131071 and the power of two below 66052 are the same number, 65536. So the
shared accumulator costs no admitted `k`, and it saves one Int32 register
per output cell of every thread of a matrix-unit kernel and one Int64
widening per cell. Planted: the code -16257 on both operands, exact at
`k = 66052` and wrapped at `k = 66053`.

**(e) The recombination.** `S = HH * 2^14 + MID * 2^7 + LL` in Int64.
At `k <= 65536`: `|HH * 2^14| <= 2^30 * 2^14 = 2^44`,
`|MID * 2^7| <= 2130706432 * 128 < 2^38`, `0 <= LL <= 16129 * 65536 <
2^30`, so every intermediate is below `2^45` in magnitude and `S` itself,
being the sum of `k` code products of magnitude at most `16383^2`, is at
most `16383^2 * 65536 = 2^44 - 2^31 + 2^16`. The shifts are shifts of a
two's complement Int64 and are the multiplications by the powers of two.
No 64-bit multiply and no 64-bit division appears in any plan. Planted: the
accumulators at their ends, on the host and in a kernel on every column
(`check_int15_device_integers_match_host`).

**(f) The scale exponent.** A row exponent is `floor(log2 absmax) - 13`
with the field's exponent in `[-126, 128]` (128 is the infinity), or 0 for
a row of zeros, so `e` lies in `[-139, 115]` and `ea + eb` in
`[-278, 230]`. `pow2_f32` gives the infinity above 127 and `+0.0` below
-126: the contract has no subnormal scale. So under `ea + eb >= 128` a
nonzero sum dequantizes to the infinity of its sign and a sum of zero to a
NaN (`0 * inf`), and under `ea + eb <= -127` every sum dequantizes to the
zero of its sign. Both are what the simulation computes. A finite scale can
still overflow: the largest sum converts to `2^44 - 2^31` and is finite
under `2^84` and the infinity under `2^85`. Planted: `ea + eb` at 127, 128,
230, -126, -127 and -278 on every plan, and the largest sum at 84 and 85.

**(g) The final rounding.** W-6. The exact sum of the two addends of the
one addition is the integer itself, so the addition's rounding is the
conversion's, round to nearest even. A code generator that fuses the
scaling of the high part into the addition changes nothing, because that
product is exact. Planted, on the host and in a kernel on every column,
each against an integer-only spelling of the correctly rounded word that
shares no step with the seam: exact values; the ties `2^24 + 1` (even
neighbor below) and `2^24 + 3` (even neighbor above), which separate round
to nearest even from truncation and from round half away; `2^40 + 2^16`
and `2^40 + 3 * 2^16` with the integer one above and one below each; the
largest admitted sum and its two neighbors; `2^44`; the reduced branch at
`2^48` and its ties; and the two ends of Int64. The column's OWN int64
conversion is run beside the seam and the count of words on which it
differs from the correctly rounded one is printed, never trusted.

**THE LARGEST ADMITTED `k`** is the smallest of the bounds, 66052 from (d),
rounded down to a power of two so the bound is a number a reader can check:
`INT15_MAX_K = 65536`. Above it every plan and the oracle refuse by name.

### 6.3 What is promised

Given the same operand bits, `m`, `n` and `k`, the same output bits on
every certified backend and from every plan, under section 2's scope rules
(NaN cells compared as NaN). The promise covers the codes, the exponents
and the planes the conversions produce from a float32 row as well as the
product.

### 6.4 What is not promised

- Agreement with the float32 product, with `int8i32.v1`, or with any
  vendor's own quantized product.
- Quality. It is measured on the model path by the quality lane and
  recorded in `python/mojolearn/_gemm_profile.py` only when a model class
  computes under the profile.
- Speed. Section 6.7 records what was measured.
- Any orientation but OP_NT.

### 6.5 The sabotage arms

`-D MOJOLEARN_LOWBIT_SABOTAGE=1`, the family's device arm, flips the value
of every cell the three kernels store. `-D MOJOLEARN_HOST_SABOTAGE=1`, the
family's host arm, flips every cell of `gemm_int15_oracle`; it does not
reach `gemm_int15_pieces_oracle`, so `check_int15_pieces_oracle_matches_oracle`
fails under it on a box with no GPU at all. `-D
MOJOLEARN_INT15_PIECE_SABOTAGE=1` is a DEFECT arm, not a value arm: the
device split writes -127 where the high piece is -128, the mistake a split
written for a symmetric int8 range makes. It changes no code above -16257
and it does not reach the FLAT plan. `-D MOJOLEARN_LOWBIT_CONVERT_SABOTAGE=1`
reaches the host conversions, as it does for `int8i32.v1`.

### 6.6 The cross-check against the simulation

No other profile has this gate. `bench/lowbit_quality/int15_export.py` runs
the quality lane's `arith.py` (its blob hash is written into the export) on
float32 operands that include the planted rows of W-2, and writes the
operands, the simulation's codes and exponents and its float32 product.
`gemm/checks/gemm_int15_sim_check.mojo` reads them and requires the host
quantizer's codes and exponents and `gemm_int15_oracle`'s product to be the
simulation's, bit for bit, NaN cells as NaN. So the arithmetic whose quality
was measured is the arithmetic that ships. Under the host arm the check
must fail.
