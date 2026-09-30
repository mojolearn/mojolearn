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
Speed: both plans were timed on 2026-09-29 (section 3) and the clause
promises nothing about the times. And this clause is a
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
  MMA plan (L-9) runs on the integer matrix unit of NVIDIA and AMD. They
  were timed against the fp32 plans on 2026-09-29 (lane/lowbit-units,
  `bench/gemm_lowbit_price_main.mojo`; an H100, an MI325X, an M3 Ultra and
  an M2 Pro, one run of record each):
  `bench/results/lowbit_units/2026-09-29/TABLE.md`. At the 512-token rows
  the complete operation (conversions and product) of every low-bit plan a
  dispatcher picks took MORE time than fp32.v1 on all four boxes. The int8
  product alone on the integer unit took more than fp32.v1 on the H100 and
  less on the MI325X; the quantization of the activations, one thread per
  row, took more time than that product on both.
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

## 4.1 The Apple exact-chunk probe (not a plan of the profile)

`gemm/checks/gemm_int8_apple_chunk.mojo` (lane/lowbit-units, 2026-09-29)
holds each int8 code as a float32 and runs the product on Metal's float
matrix unit, `k` cut into chunks of 1024 steps so that no partial sum leaves
the integers a float32 holds (`16129 * 1040 < 2^24`), the chunk sums carried
in Int32, the epilogue `dequant_int8_pinned`. It is NOT in the dispatcher and
`lib_int8_matrix_unit_for` still answers False on Apple. Its gate,
`gemm/checks/gemm_int8_apple_chunk_check.mojo`, passed on an M2 Pro and on
an M3 Ultra (bits equal to the flat kernel and to `gemm_int8_oracle` on 21
shapes of quantized fixtures and 110 planted worst cases, both tile
geometries), and on both the arm that removes the chunk boundary failed 54
of the 110 planted cases and none of the quantized fixtures. At the twelve
transformer rows its digests on the two Apple boxes equal the H100's and the
MI325X's integer-unit digests. It assumes the unit computes in IEEE float32
at every internal step, which is a measurement per Apple generation: M2 and
M3 are measured for this kernel, the M4 is not.

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

Lane lane/lowbit-int15, 2026-09-29. DEVIATIONS 2965 to 2979. Clauses W-1 to
W-14. Answers: `gemm/host/gemm_int15_oracle.mojo`. Device:
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

| W-9 | which values a product quantizes | EVERY PRODUCT QUANTIZES ITS OWN OPERANDS FROM THEIR FLOAT32 VALUES, along that product's own contracted extent. Codes are never carried from one product to another and never transposed; section 6.8 | 2976 |
| W-10 | conversion schedules | ROWS (`quantize_rows_int15_kernel`, and `quantize_cols_int15_kernel` for an operand stored the other way: one thread per row of the matrix being quantized) and PARALLEL (the row's absmax in chunks of 256 values, the chunk maxima reduced to one exponent per row, then one thread per value, to codes or straight to planes). Both are the quantizer: the absmax is a maximum of magnitudes that a NaN never enters, the same float under every grouping, and a code is a function of its own value and its row's exponent. `check_int15_device_conversions_match_host` requires both schedules' codes, exponents and planes to be the host's | 2974 |
| W-11 | fragment loads of the matrix units | a load of four or eight codes states its alignment only where it is a fact: the offset inside the buffer is a multiple of the word (tested in `_pack4` and `_pack8`) and the base of the buffer is a multiple of 8 (read by the launch off the pointers it passes). Elsewhere the load is unstated, and beyond the row the byte path with its zero codes. The same bytes reach the unit whichever load fetched them; `-D MOJOLEARN_INT8_MMA_UNSTATED_LOADS=1` keeps the unstated loads and must print the same digests. It applies to `int8i32.v1`'s unit plan as well, whose kernel shares the step | 2975 |

| W-12 | Apple's float matrix unit | `gemm/checks/gemm_int15_apple.mojo`, two forms, both the profile; section 6.9. NOT a plan of clause W-8 by another name: it has bounds of its own, because a float unit is exact only while every value it holds is an integer below `2^24` | 2977 |
| W-13 | the tuned unit plan | `gemm/checks/gemm_int15_tuned.mojo`: lane/lowbit-mma-speed's four products with one staging (three exact Int32 sums per cell, `HH`, `HL + LH`, `LL`, no float), then the epilogue of every plan as a launch of its own. It admits `k <= 65535` (the sums kernel allows a low piece of -128, which this profile never makes) and takes the reference unit plan at `k = 65536` | 2978 |
| W-14 | a launch on Apple is bounded in work | the two one-thread-per-cell plans launch their cells in slices of at most `2^30` multiply-accumulates on the Apple column and wait between slices; the Apple dispatchers take the float-unit plan above `2^28`; section 6.10 | 2979 |

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

### 6.7 What was measured

Filled from runs of record only. Until a run is recorded, no sentence of
this contract states a time.

### 6.8 Clause W-9: the products of a training step

The scale of an operand is one power of two per ROW ALONG THE CONTRACTED
EXTENT. A training step's three products contract three different extents,
so one tensor has DIFFERENT codes in each product it enters, and "quantize
the transposed tensor" and "transpose the quantized tensor" are different
arithmetic. The profile is the first, always:

| product | contracted extent | left operand, quantized along it | right operand, quantized along it |
|---|---|---|---|
| forward, `Y = X W^T` | the input features | `X`, one scale per TOKEN (each row of `X` over the features) | `W`, one scale per OUTPUT FEATURE (each row of `W` over the input features) |
| weight gradient, `dW = dY^T X` | the tokens | `dY`, one scale per OUTPUT FEATURE (each column of `dY` over the tokens) | `X`, one scale per INPUT FEATURE (each column of `X` over the tokens) |
| input gradient, `dX = dY W` | the output features | `dY`, one scale per TOKEN (each row of `dY` over the output features) | `W`, one scale per INPUT FEATURE (each column of `W` over the output features) |

**Why this rule and no other.** It is the only one under which a per-row
scale on both operands is a per-cell scale of the output, which is what
makes the integer sum the product (section 0): cell `(i, j)` is
`2^(ea[i] + eb[j])` times the sum over the contracted extent of two codes,
and that needs `ea` to be constant along the contraction for row `i` of the
left operand and `eb` constant along it for row `j` of the right one.

**What the transposing quantizer computes.** Every product is run as
OP_NT, `C = A B^T` with `A` of `m` rows and `B` of `n` rows, each row
running along the contracted extent `k`. An operand that is STORED the
other way (`k` rows of `m`, or `k` rows of `n`: the columns of the stored
matrix run along the contraction) is read by
`quantize_cols_int15_kernel`, or by the parallel schedule with its two
strides exchanged: for each COLUMN `c` of the stored `rows x cols` matrix
`x`, the absmax of `x[0, c], x[1, c], ...` after the flush, its exponent by
W-1, and the codes of those values by W-2, written as row `c` of the
`cols x rows` result. Those are the codes of the rows of the transpose of
the float32 matrix. No code of the stored orientation is read, because
none is ever made: the quantizer's input is float32. So it is W-9, and
`check_int15_device_conversions_match_host` holds the device's codes and
exponents to `quantize_rows_int15` applied to the transposed float32
values, spelled in the check and not taken from the function under test.

**What is owed under W-9.** Vectors of the two backward products exported
from the quality lane's training simulation, with the host oracle and the
devices held to them bit for bit, as section 6.6 does for the forward
product.

### 6.9 Clause W-12: the product on Apple's float matrix unit

Metal has no integer matrix unit. Its float unit multiplies 8 by 8 by 8 in
float32, and it is exact on integers while every value it holds stays below
`2^24` in magnitude. lane/lowbit-units showed that for int8 codes (its
exact-chunk probe); this clause is the fifteen-bit product on the same
unit, in two forms.

**Form TWO: the left operand whole.** `sum(a * b) = sum(a * bh) * 2^7 +
sum(a * bl)`, two unit products per step.

| what | bound | why |
|---|---|---|
| an operand | `abs(a) <= 16383`, `abs(bh) <= 128`, `0 <= bl <= 127` | integers below `2^24`, so each is a float32; the left operand is staged as `ah * 128 + al`, exact |
| one product | `abs(a * bh) <= 16383 * 128 = 2097024` | below `2^24`, exact |
| one step of the unit | `8 * 2097024 = 16776192 < 2^24 = 16777216` | the accumulator enters every step at zero, so every value the unit can hold is a sum of at most 8 products, under any order and with or without fusing; NINE products would reach 18873216 |
| the carry | `abs(sum(a * bh)) <= 2097024 * 65536 < 2^38` | each accumulator converts to Int32 after every step (exact) and is added to an Int64 running sum |
| the result | `HI * 2^7 + LO` in Int64 | a shift and an addition |

**Form FOUR: both operands in pieces.** The construction of clauses W-3 to
W-5 with float accumulators: the largest magnitude one step of `k` adds to
an accumulator is the cross term's, 32512, so a chunk of 512 steps holds at
most `32512 * 512 = 16646144 < 2^24` (517 steps would pass it). At a chunk
end each accumulator converts to Int32 and is added to an Int32 running sum,
which W-4 bounds.

**What the argument assumes.** That the unit computes in IEEE float32 with
a 24-bit significand at every internal step. That is a measurement per
Apple generation, not a documented property, so the gate plants the cases
that separate it: every product odd and at its largest (`a = 16383`,
`bh = 127`); an operand whose sums over sixteen consecutive steps are ODD
and above `2^24` while its sums over eight are below; the largest magnitude
(`bh = -128`); halves that cancel; `k = 65536`. The arm
`-D MOJOLEARN_INT15_APPLE_CHUNK_SABOTAGE=1` removes the chunk boundary (form
TWO carries across two steps, form FOUR across the whole of `k`) and must
fail the planted cases.

### 6.10 Clause W-14: a launch on Apple is bounded in work

macOS aborts a Metal command buffer that holds the GPU for seconds
(`kIOGPUCommandBufferCallbackErrorImpactingInteractivity`). The cells
written before the abort stay, the rest keep what the buffer held, and the
wait returns as if the launch had finished. Measured on an M2 Pro,
2026-09-29: the pieces kernel at 512 x 4096 x 14336, one launch of `3e10`
multiply-accumulates, left cells unwritten in two runs of three, at a
different cell each time. On an M3 Ultra the same launch takes a quarter
of a second and was never cut.

So a cell that was never written is a failure this profile can have on
Apple whatever its arithmetic, and the clause is about the LAUNCH:

- the FLAT and PIECES plans launch their cells in slices of at most `2^30`
  multiply-accumulates and wait between slices (a PIECES step is four
  products, so its slice is a quarter of FLAT's in cells);
- the dispatchers send a product above `2^28` multiply-accumulates to the
  float-unit plan, whose launches take milliseconds;
- which cells a launch covers cannot move a bit: a cell is computed by one
  thread from its own two rows whatever the slice.

`check_int15_large_product_is_written_whole` runs the row that failed on
every plan of every column: the output poisoned first and read back whole,
one digest for every plan, and the host oracle on cells sampled across the
whole output. What this clause cannot do is make the runtime report an
aborted command buffer; until it does, a launch that is never long is what
keeps a launch from being cut.

