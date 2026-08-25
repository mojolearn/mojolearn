# The speed lane: what `mojolearn.identical.gemm.fp32.v1` costs, on three vendors

Measured 2026-08-25. Commits `ba35096` through `f5b905a`. This file is the
appendix source for the GEMM performance claim. **Read section 1 before
quoting any number in it.**

---

## 1. The three things that will make you quote this wrong

### 1.1 The FAST arm is not FP32 on NVIDIA, and neither is cuBLAS by default

`core/gemm.mojo` under FAST calls MAX's `linalg.matmul`. On an H100 80GB that
arm measured **200 TFLOP/s** at `llama8b.mlp_up.t512`. The H100's FP32
non-tensor peak is **67 TFLOP/s**. Three times over, which is arithmetically
impossible in strict FP32.

Confirmed independently, not inferred. `tools/vendor_gemm_price.py` times
cuBLAS through torch in both modes on the same shapes, same box, same run:

| `llama8b.qkv.t512` | TFLOP/s |
|---|---|
| cuBLAS, `allow_tf32=False` (true FP32) | 44.4 |
| cuBLAS, `allow_tf32=True` (TF32) | **207.5** |
| MAX `linalg.matmul` | **200** |

MAX matches the TF32 column and not the FP32 one. **TF32 keeps 10 explicit
mantissa bits; FP32 keeps 23.** Our kernel is strict FP32 by contract.

Comparing our strict FP32 against a TF32 baseline and calling the difference
"the cost of identity" charges this contract for someone else's precision cut.
On these shapes that mistake is worth about **5x**.

### 1.2 The cost of identity is 1.52x, and it is NOT the summation tree

**MEASURED 2026-08-25, Apple M4**, by `gemm/mojo_only/gemm_unpinned_price.mojo`:
three arms in ONE binary alternating call by call, twenty shapes, zero
refused, **zero bitwise inert**.

| llama8b t512 | pinned / unpinned | pinned / strict (NACC=1) |
|---|---|---|
| qkv | **1.522x** | 1.234x |
| mlp_up | **1.538x** | 1.226x |
| mlp_down | **1.546x** | 1.241x |
| lm_head | **1.542x** | 1.247x |

**The prediction on record was 1.10x to 1.45x. It was wrong, on the low side.**
`gemm/UNPINNED_CONTROL.md` wrote it down before the run, which is why the miss
is usable.

The decomposition is the part worth carrying into the paper:

| | cost | share |
|---|---|---|
| the seams (`ftz` at 7 places, refusing FMA contraction) | 1.23x | ~46% |
| refusing to sub-partition a leaf (contract 7.1) | 1.24x | ~54% |
| **the balanced fold tree itself** | **small** | the `P == 1` rows, which have NO tree to remove, still show 1.42x |

**The thing the contract is named after is not the expensive part.** What costs
is per-operation: flushing denormals at every seam, and forbidding the
compiler to contract a multiply-add into an FMA.

`UNPINNED_CONTROL.md`'s prediction 2 -- flagged there as the one most likely
to be wrong -- HELD: pinned-to-strict and strict-to-unpinned came out 1.23x
and 1.24x, within 1%.

**So of the 12x to 25x gap to a vendor library, about 1.5x is the constraint
and 8x to 16x is kernel engineering.** That is the sentence this whole lane
was built to be able to write.

FIXTURE NOTE, and it is why this section can be trusted. The first run of this
driver used a generator emitting integers scaled by `2^-4`. Every such product
is exactly representable, so FMA contraction was bit-neutral and `ftz` could
never fire, and SIXTEEN of twenty rows came back with the two arms producing
IDENTICAL BITS. The timing was unaffected (1.554x then, 1.522x now) but the
bit half of the experiment was vacuous. DEVIATION 1147 replaced it with a
full-mantissa generator and all twenty rows now differ. The driver reported
those rows as `INERT` rather than as agreement, which is the only reason it
was visible instead of reading as success.

STILL OWED: a subnormal-bearing fixture. Nothing here is small enough to make
`ftz` fire, so the denormal half of the seam cost is measured in TIME and
never in bits. And this is ONE box; the H100 and MI325X columns are owed.

### 1.2a AND IT IS NOT THE SAME PRICE ON EVERY VENDOR

**H100 80GB, same driver, same twenty shapes, zero refused, zero inert:**

| llama8b | t1 | t8 | t512 |
|---|---|---|---|
| qkv | 1.03x | 1.42x | **2.33x** |
| mlp_up | 1.65x | 1.61x | **2.32x** |
| mlp_down | 1.02x | 1.42x | **2.31x** |
| lm_head | 1.53x | 1.53x | **2.32x** |

**The pin costs 2.31x on an H100 against 1.55x on an M4.** Fifty percent more
on NVIDIA, from one source.

Had this lane only ever measured Apple, "identity costs 1.5x" would have gone
into a paper as a property of the CONTRACT. It is not. It is a property of the
contract AND the silicon, and Apple is the cheap end.

The decomposition moves too. On the M4 the seams and the no-sub-partition
clause split it evenly, 1.24x and 1.24x. On the H100 **the seams alone are
1.83x of the 2.31x**, so there the dominant cost is `ftz` plus refusing FMA
contraction rather than the partition rule. The plausible mechanism is that
the H100 has FMA throughput we are forbidding it to use.

**The cost tracks arithmetic intensity, which is a mechanism rather than a
number.** At `t1` the kernel is bandwidth bound and the extra instructions
hide behind memory latency (1.02x to 1.03x at two rows). At `t512` there is
nothing to hide behind and the pin shows at full price. That predicts where
the cost lands on a shape nobody has run.

### 1.2c THREE VENDORS, AND APPLE IS THE OUTLIER

| llama8b t512 | M4 | H100 | MI325X |
|---|---|---|---|
| qkv | 1.52x | 2.33x | **2.21x** |
| mlp_up | 1.54x | 2.32x | **2.20x** |
| mlp_down | 1.55x | 2.31x | **2.17x** |
| lm_head | 1.54x | 2.32x | **2.06x** |

**AMD sits with NVIDIA, not with Apple.** The honest headline for the fold pin
on datacenter silicon is **2.1x to 2.3x**, and the M4 figure was 35% LOW. A
paper quoting 1.5x from the box it was developed on would have understated the
cost of its own contract by a third.

**AND THE VENDORS DIFFER IN WHICH CLAUSE IS EXPENSIVE, NOT ONLY BY HOW MUCH.**
A mechanism proposed after the H100 leg -- that the cost is `ftz` plus
forbidding FMA contraction, so it should vanish where the kernel is bandwidth
bound -- covers NVIDIA and DOES NOT SURVIVE AMD:

| | M4 | H100 | MI325X |
|---|---|---|---|
| seams share at t512 (`pin/strict`) | 1.24x of 1.55x | 1.83x of 2.31x | 2.12x of 2.21x |
| `mlp_down.t1` `pin/strict` | -- | 0.93x | **1.00x** |
| `mlp_down.t1` `pin/unpinned` | -- | 1.02x | **2.30x** |

On the MI325X at `t1` the seams cost NOTHING (`pin/strict` = 1.00) while the
whole 2.30x comes from the no-sub-partition clause. On the H100 at the same
row both are ~1.0. On the M4 the two split evenly at `t512`. **There is no
single mechanism yet, and the file says so rather than keeping the tidy one.**

What this means for anyone quoting these numbers: the fold pin's price is a
function of the vendor AND the shape AND which clause that vendor happens to
find expensive. Report the range with the table, never a scalar.

### 1.2b The confounds this does not remove

DEVIATION 1092. Three arms exist and none of them is the experiment:

- **`device` vs `pinned`** is not it. `pinned_gemm_nt_kernel` is FLAT -- one
  thread per output cell, serial whole-`k` loop, no tiling, no shared memory.
  Our identical kernel picks among eight plans including tiled and split-K
  arms, and **beats `pinned` by 1.6x to 1.8x at every `t512` row on the M4**.
  That result says tiling beats not-tiling.
- **`device` vs `vendor`** is our kernel engineering plus the fold pin against
  Modular's kernel engineering, with both terms moving at once, and on NVIDIA
  with a precision difference on top.
- **`device` vs cuBLAS** is the same, against a mature vendor library.

The arm that would answer it is the identical kernel's OWN tiled plan with
ONLY the fold-order pin removed, everything else held. It is now WRITTEN
(`gemm/mojo_only/gemm_unpinned.mojo`, 945 lines; `gemm/UNPINNED_CONTROL.md`,
439) and is NOT YET WIRED INTO THE PRICE HARNESS OR RUN. **Until it runs, the
cost of the profile as distinct from the cost of writing our own GEMM is
UNMEASURED.** Do not take the nearest available ratio.

Its prediction is on record ahead of the measurement, which is this project's
habit because a wrong prediction is the most useful thing a gate produces:
**1.10x to 1.45x** at the tiled `t512` rows, 1.00x to 1.06x at the
bandwidth-bound `t1` rows, and **the fold itself contributing under 2%**
anywhere. If that holds, the pin is a small part of the 12x to 30x and the
rest is ours to fix.

### 1.3 The variance is large and the sample was one

DEVIATION 1094. Legs 15 and 16 both ran `ROUNDS=1` on an H100 an hour apart,
same commit, same shape. Our v1 arm measured **4.90 ms and 8.799 ms** at
`llama8b.qkv.t512`. A 1.8x spread.

Every H100 number below is therefore a **band across two single-round runs**,
not a median. The harness takes medians over rounds and was being told to take
the median of one; it now defaults to five. **The Apple and MI325X columns are
also single-round.** Re-run before publication.

---

## 2. Method

- Shapes: the twenty rows of `bench/gemm_shapes.mojo`, which are Llama-3-8B's
  QKV, MLP up/down and LM head at 1, 8 and 512 tokens, plus the classical
  Gram, OLS, PCA and k-means shapes. **Every row runs. Nothing is dropped for
  being slow or unflattering.**
- One MAC cap (`5e10`) applies to BOTH sides, so a row is measured on both
  arms or skipped on both. `llama8b.lm_head.t512` is above it and is skipped
  on both, and says so.
- The mojo arms alternate CALL BY CALL inside one timed loop, which defeats
  thermal drift at the arm level. The two MODES are two binaries
  (`GLOBAL_NUMERIC_MODE` is comptime) so they alternate round by round.
- Every device arm poisons its output and reads it back before timing, so a
  kernel that launches without writing cannot post the best time.
- The vendor library is reached through torch in a throwaway venv, never the
  pinned pixi environment every recorded timing in this repository was taken
  under. It **refuses on a CPU torch** rather than falling back.

---

## 3. Results

### 3.1 Apple M4, full shapes, all strict FP32

| `llama8b` | ours (IDENTICAL) | MAX `linalg.matmul` | MPS via torch |
|---|---|---|---|
| qkv.t1 | 2.21 ms | 1.21 | 1.41 |
| qkv.t8 | 4.50 | 1.41 | 2.72 |
| qkv.t512 | 218.8 | 11.6 | 8.7 |
| mlp_up.t512 | 771.5 | 39.8 | 25.8 |
| mlp_down.t512 | 770.4 | 68.9 | 29.0 |
| lm_head.t8 | 113.1 | 39.5 | 32.3 |

Peak achieved: ours **78 GF/s**, MPS **2332 GF/s**. About **2%** of what the
box gives torch.

### 3.2 H100 80GB, full shapes

| `llama8b` t512 | ours (strict FP32) | cuBLAS strict FP32 | cuBLAS TF32 | ours vs cuBLAS FP32 |
|---|---|---|---|---|
| qkv | 1.95 - 3.51 TF/s | **44.4** | 207.5 | 12.7x - 22.8x |
| mlp_up | 2.00 - 3.55 | **42.8** | 291.3 | 12.1x - 21.4x |
| mlp_down | 1.96 - 3.47 | **49.8** | 263.6 | 14.4x - 25.4x |

cuBLAS strict FP32 reaches **66%** of the H100's 67 TF/s vector peak. We reach
**3 to 5%**.

### 3.3 AMD Instinct MI325X, full shapes

torch 2.9.1+rocm6.4, ROCm 6.4.43484, hipBLASLt/rocBLAS, 10 repeats, median.

| `llama8b` | ours (strict FP32) | hipBLASLt strict FP32 | ratio |
|---|---|---|---|
| qkv.t512 | 4.31 TF/s | **77.1** | 17.9x |
| mlp_up.t512 | 4.52 | **70.9** | 15.7x |
| mlp_down.t512 | 4.37 | **50.4** | 11.5x |
| lm_head.t8 | 3.81 | **12.9** | 3.4x |

hipBLASLt reaches **62% to 94%** of the MI325X's 81.7 TF/s FP32 vector peak.
We reach about **5%**.

**One caveat that keeps this from being called confound-free.** CDNA3 does
have an XF32 matrix mode, AMD's analogue of TF32, and on ROCm torch the same
`allow_tf32` switch reaches it. This harness does not touch that flag on the
hip backend, so what is measured is torch's default, which is off. That makes
the number a strict-FP32 number by default rather than by assertion, and the
XF32 arm is OWED here the way the TF32 arm is measured on NVIDIA.

### 3.4 The three columns together

| | ours | vendor lib, strict FP32 | ratio | ours as share of FP32 peak |
|---|---|---|---|---|
| Apple M4 | 0.078 TF/s | 2.33 (MPS) | ~30x | ~2% |
| H100 80GB | 1.95 - 3.51 | 44.4 (cuBLAS) | 12.7x - 22.8x | 3 - 5% |
| MI325X | 4.31 | 77.1 (hipBLASLt) | 17.9x | ~5% |

The consistency is the result worth reporting. **We sit at 2 to 5 percent of
FP32 peak on all three vendors**, which says the gap is structural to the
kernel rather than an accident of one backend.

---

## 4. Two defects in MAX found by running this

- **`linalg.matmul` crashes on an H100** at `llama8b.lm_head.t1` (`m=1`,
  `n=128256`, `k=4096`), inside `max/kernels/src/linalg/gemv.mojo:1201`,
  `CUDA_ERROR_INVALID_VALUE`. `llama8b.qkv.t1` is also `m=1` and runs fine,
  and all twenty shapes complete cleanly on an MI325X, so the trigger is the
  wide `n` in the gemv path on NVIDIA. No formula is guessed from one point.
  DEVIATION 1093 turns it into a printed refusal instead of a lost run.
- **`linalg.matmul` does not write its output at `n == 1`.** Recorded earlier
  in `core/gemm.mojo`: at `m=64, n=1, k=32` with the output poisoned, 63 of
  the 64 rows still held the poison. That is why the vendor arm is skipped
  at `n == 1` here, and it is a defect rather than a shape.

---

## 5. What this file does NOT claim

- That the gap is caused by identity. See 1.2. That is unmeasured.
- That these are medians. See 1.3. They are single-round samples.
- That these shapes are where a served model spends its time. They are one
  model's projection shapes, unfused, with no epilogue and no batching.
- Anything about FP16, BF16, FP8 or tensor-core paths. This profile is FP32.
- Anything about MAX's kernels other than the two defects in section 4 and the
  measurement in 1.1. Modular's kernels are tuned and ours are not, and that
  is the expected result rather than a finding.
