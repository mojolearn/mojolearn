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

### 1.2 Nothing here isolates the cost of identity

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
ONLY the fold-order pin removed, everything else held. It is being built
(`gemm/mojo_only/gemm_unpinned.mojo`, `gemm/UNPINNED_CONTROL.md`). **Until it
runs, the cost of the profile as distinct from the cost of writing our own
GEMM is UNMEASURED.** Do not take the nearest available ratio.

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

### 3.3 AMD Instinct MI325X

*(hipBLASLt column pending; the mojo arms are measured and complete at all
twenty shapes.)*

Ours, full shapes: **4306 - 4587 GF/s** at the `t512` rows, which is about
**5.5%** of the MI325X's 81.7 TF/s FP32 vector peak.

AMD is the interesting column precisely because CDNA3's matrix cores do
**full-precision FP32** at 163 TF/s, unlike NVIDIA's TF32. If the precision
confound is absent there, the AMD ratio is the cleanest number in this file.

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
