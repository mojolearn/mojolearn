# fp32 GEMM on the integer matrix units by exact integer slicing (`fp32.v2-int`, research prototype)

Lane `lane/gemm-int-slices`, 2026-10-28. Idea 13 of
`~/mojolearn-evidence/neural-perf-ideas-20261028/REPORT.md` (diagnosis 2c: the
identical fp32 and bf16 paths use no matrix units; only `int8i32.v1` does).
Switch `-D MOJOLEARN_IDN_GEMM_INT_SLICES=1`, default OFF.

**STATUS: PROTOTYPE, PARTIAL.** Compiled (CPU column, Apple, NVIDIA sm_89, AMD gfx942;
receipts in `experiments/gemm_int_slices_20261028/compile.json`). Not run: no gate, no
identity check, no timing and no quality number exists yet. Everything below is a
construction argument until the gate and the A/B in section 7 have run.

## 1. What it is

A new numerical profile, not a change to `fp32.v1`. Each fp32 operand row is turned into
an exact integer code with one power-of-two scale per row; each code is cut into int8
planes; every plane product and its k-sum is computed exactly in integers on the vendor's
int8 matrix unit (IMMA on NVIDIA, MFMA-i8 on AMD); the integer products are recombined
exactly and the result is rounded ONCE to fp32. The fold order is not a hardware fp
accumulation order: there is no fp accumulation. The only floating steps are the operand
coding (exact power-of-two scaling and one round-to-integer per value) and the final
rounding, so the result is the same bits on every column by construction.

Code: seams `gemm/int_slices_seams.mojo`; device plans `gemm/int_slices.mojo`; host
column `gemm/host/gemm_int_slices_oracle.mojo`; gate `gemm/checks/gemm_int_slices_check.mojo`;
dispatch hooks `gemm/host_entry.mojo` (GPU columns) and `gemm/host/linalg_native.mojo`
(CPU column), both `comptime if` on the switch; off, both compile to nothing.

## 2. The clauses

| clause | what | spelling |
|---|---|---|
| S1 | row exponent | for every row of op(A) and column of op(B) along `k`: `e = floor(log2 absmax) - 27` from the exponent field of the flushed absmax (a maximum, order-free; NaN never wins); all-zero row `e = 0` |
| S2 | code | `q = clamp(rne(ftz(x) * 2^-e), +-(2^28 - 1))`; the scaling on the exponent field (exact; below normal range a signed zero, which rounds to 0 as the exact value would); `rne` = ties-to-even via the `2^23` magic constant below `2^23`, identity above; NaN -> 0; inf -> clamp |
| S3 | halves, planes | `hi = q >> 14` in `[-16384, 16383]`, `lo = q & 0x3FFF` in `[0, 16383]`; each half -> two int8 planes `h = half >> 7`, `l = half & 127` |
| S4 | products | `P_hh, P_hl, P_lh, P_ll`, the exact integer sums of half products. UNIT plan: four launches of the tuned four-plane kernel (`identical_gemm_int8_pieces_tuned_with_plan`), three exact Int32 sums each, `int15_recombine` to Int64. FLAT plan and host oracle: Int64 sums of Int32 products, `p` ascending |
| S5 | recombination | `T = P_hh 2^28 + (P_hl + P_lh) 2^14 + P_ll` held as `(H, L)`, `T = H 2^28 + L`, `L in [0, 2^28)`, all exact Int64 |
| S6 | conversion | one round-to-nearest-even of `T` to a 24-bit significand (`i64_to_f32_pinned` with a sticky bit above `2^62`) |
| S7 | scale | times `2^(ea + eb)` on the exponent field: exact while normal; below the normal range a signed zero; above, the signed infinity |

Admission: `0 < m, n` and `0 < k <= 65536` (`INT_SLICES_MAX_K`). No shape rule.

### 2.1 Why every column returns the same bits (the exactness argument)

1. S1 and S2 are functions of a value and its row maximum. A maximum is the same float
   under every grouping; the scaling is an exponent-field edit (no multiply, so no FMA,
   contraction or subnormal mode applies); the rounding is one IEEE addition with an
   exact result range; the conversion to Int32 is of an integer below `2^28`.
2. S3 is integer shifts and masks.
3. S4: an int8 x int8 product is exact. With the plane domain of S3 (the high plane of
   `hi` any int8, every other plane in `[0, 127]`), the tuned kernel's three Int32 sums per
   launch are exact for `k <= 65536` (its own stated bound `INT8_PIECES_MAX_K`:
   `HH <= 16384 k`, `HL + LH <= 32512 k`, `LL <= 16129 k`). An exact integer sum is the
   same integer under every order, tile, staging, k-window and vendor. The Int64 sums of
   the flat plan and the oracle are exact by a wide margin (`|hi_a hi_b| <= 2^28`,
   `k 2^28 <= 2^44`).
4. S5: `|P_hh| <= 2^44`, `|P_hl|, |P_lh| < 2^44`, `0 <= P_ll < 2^44`, so
   `(P_hl + P_lh) 2^14 + P_ll` is below `2^60` and the carry below `2^32`: exact Int64.
5. S6 and S7 are integer edits plus `i64_to_f32_pinned`, whose two exact conversions and
   one IEEE addition round exactly once (DEVIATION 2970).

So the cell is a pure function of the input bits, `m, n, k, op`; the plan (UNIT, FLAT,
HOST) and the vendor are scheduling. `gemm_int_slices_check.mojo` G1 requires it on a box.

## 3. The precision contract (bits change AND precision change)

**What the cell is.** `C_ij = RN_f32( sum_p q_a(i,p) q_b(j,p) * 2^(ea_i + eb_j) )` with ONE
rounding: the correctly rounded value of the exact dot product of the CODED operands
(below the normal range, a signed zero).

**Against fp32.v1** (`gemm/IDENTICAL_FP32_CONTRACT.md`: one `ftz(fma_rn)` chain per
128-leaf, a balanced tree over leaves):

- It is a BITS change on every cell in general (different fold; one rounding instead of
  about `k`).
- It is a PRECISION change on the operand side. The coding error per value is at most
  half a unit of `2^e`, and `2^e <= 2^-27 max|row|`, so `|x - q 2^e| <= 2^-28 max|row|`.
  A value within `2^4` of its row maximum is carried exactly (its 24-bit significand fits
  in the 28-bit code); a value `2^(4+t)` below the maximum keeps `24 - t` significant bits;
  a value more than `2^28` below the maximum codes to 0. Absolute error is bounded by the
  row maximum, not by the value: the Ozaki trade-off.
- Error bound per cell, from the two:
  `|C_ij - (A B)_ij| <= 2^-24 |(A B)_ij| + k (2^-27 + 2^-56) max_p|a_ip| max_p|b_pj|`
  (the gate checks the looser `2^-23 |C| + k 2^-26 max max`). fp32.v1's worst case is
  about `(128 + log2(k/128)) 2^-24 sum_p |a_ip||b_pj|`. For rows of comparable magnitudes
  (gaussian, uniform: the board's data) the v2-int bound is the smaller one; for rows
  with a wide dynamic range, small entries carry fewer bits than in fp32.v1.
- Therefore **the quality metric must be checked**, not assumed: the board's
  `max_rel_err_vs_fp64` for `neural:gemm:gaussian` (fp32.v1: 2.40e-07 on the Oct 27
  NVIDIA board) is reported by the A/B race, and the gate prints the measured ratio of
  both profiles' errors to `k max|a| max|b|` on its shapes (G2).

## 4. What is not promised

- `fp32.v1` bits. A caller who has stored fp32.v1 outputs sees different bits.
- Per-element relative precision for rows with a wide dynamic range (section 3).
- Speed. Section 6 is a model, not a measurement.
- Non-finite inputs: defined (NaN codes 0, inf clamps) and the same on every column, but
  not IEEE propagation; outside what the prototype admits as tested.
- Apple at large shapes: the FLAT plan has no bounded-launch slicing yet (W-14's rule).
- gemm-bf16 (`matmul_bf16`, the bf16f32.v1 route) is not hooked. A bf16 operand has an
  8-bit significand, so a row-scaled bf16 code needs fewer planes (two halves would carry
  every value within `2^20` of its row max exactly); that is the next step, not this one.

## 5. Prior art in this tree, and the difference

Two Ozaki routes were measured and deleted (`docs/TOMBSTONES.md`):
`MOJOLEARN_IDN_GEMM_OZAKI_LINALG` (2026-10-08, gemm gaussian 4096^3: S=4 NV 69.6 -> 276.6
ms, AMD 28.5 -> 58.8 ms; S=5 slower still) and `MOJOLEARN_IDN_NEURAL_GEMM_OZAKI_SLICES`
(2026-10-11, slower on all 10 neural rows). Both used the REFERENCE int8 unit kernel
(`gemm_int8_mma.mojo`, one warp one tile, every fragment from device memory each k step,
measured 1.38-1.66x fp32.v1's time on an H100), Int32 atomics across a k-split, balanced
digits and the high triangle only. This prototype uses the TUNED four-plane kernel (planes
staged once per window in threadgroup memory, four unit steps per tile; 67
TFLOPS-equivalent on the L40S at qkv, `IDENTICAL_NEURAL_ROADMAP.md:161`), no atomics, and
all sixteen int8 products (a correctly rounded result instead of a truncated one). The
work is 16 int8 products per fp32 product; whether the tuned kernel makes that pay is
exactly what the A/B measures, and the deleted results say it may not.

## 6. Cost model (not a measurement)

Per 4096^3 call: 16 int8 GEMM-equivalents of work in 4 launches, two quantizer launches
reading A and B once each (the op(B) read of OP_NN is strided, not coalesced), 4 (m+n) k
bytes of planes, 48 m n bytes of Int32 sums written and read once by the epilogue
(805 MB at 4096^2), and per-call workspace allocation. Dense peaks for scale: L40S int8
362 TOPS vs fp32 91.6 TFLOPS (16 int8 products cost about 4x an fp32 product at peak; the
case for the plan is that fp32.v1 runs far below fp32 peak); MI325X int8 MFMA 2615 TOPS
vs fp32 matrix 163 TFLOPS (about parity at peak). Obvious next levers if the A/B is close:
a fused epilogue (the four-plane kernel's FUSED form with this profile's cell), pooled
workspace, a coalesced transposing quantizer, and dropping lo.lo (12 products; a
truncation, which breaks section 3's correctly rounded statement and needs its own bound).

## 7. Runs owed (the orchestrator queues them; lanes never run them)

Gate and cross-vendor identity (same binary, digests compared NV vs AMD):

    lq add nv  CMD lane/gemm-int-slices gis-gate  BUILDS=none bash -c 'pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_COLUMN_NVIDIA -I . -I bindings gemm/checks/gemm_int_slices_check.mojo -o /tmp/gis && /tmp/gis'
    lq add amd CMD lane/gemm-int-slices gis-gate  BUILDS=none bash -c 'pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_COLUMN_AMD -I . -I bindings gemm/checks/gemm_int_slices_check.mojo -o /tmp/gis && /tmp/gis'

plus the same two with `-D MOJOLEARN_INT_SLICES_FORCE_FLAT=1` (UNIT vs FLAT) and with
`-D MOJOLEARN_INT_SLICES_SABOTAGE=1` (must FAIL). A/B (one run per arm):

    lq add nv  RACE lane/gemm-int-slices gemm gaussian ARMS=A:,B:-DMOJOLEARN_IDN_GEMM_INT_SLICES=1
    lq add amd RACE lane/gemm-int-slices gemm gaussian ARMS=A:,B:-DMOJOLEARN_IDN_GEMM_INT_SLICES=1
    lq add nv  ID   lane/gemm-int-slices gemm gaussian
    lq add amd ID   lane/gemm-int-slices gemm gaussian

(lq RACE has been algos-only; if it refuses, the same two arms through
`lq add <box> CMD ... tools/six_lane_grid_bb.py --race neural:gemm:gaussian` with
`MOJOLEARN_BUILD_DEFINES=MOJOLEARN_IDN_GEMM_INT_SLICES=1` on the B tag.) The ID line needs
the B define in its build; if `lq ID` cannot take a define, the gate's DIGEST lines on nv
and amd are the identity evidence.
