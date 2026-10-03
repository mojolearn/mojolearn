# afn-lm A/B requests: the byte LM on Apple FAST

Binding `byte_lm`, shape `full` ([1, 2048, 384, 6, 6, 64, 1024, 8 layers, vocab 8192]).
Each line in `lm.txt` races arm A (FAST, no define) against arm B (FAST + one define).
Every define is compiled only when the build is FAST and the host is Apple
(`BYTE_LM_FAST_APPLE`, training/byte_lm_afn.mojo). IDENTICAL builds and FAST builds
on NVIDIA or AMD compile main's code.

**Baseline.** No FAST-vs-IDENTICAL line is written here. The FAST byte LM is new
(item 0: `MOJOLEARN_NUMERIC_MODE=fast bash bindings/build_byte_lm.sh`), and the tier
lane's `afn_ab.sh` baseline mode (arm A IDENTICAL, arm B FAST, no define) covers it.
Note that FAST with no define is not exactly IDENTICAL's schedule. The block backward
kept three host waits per RMSNorm backward in non-IDENTICAL tiers and routed FAST
around IDENTICAL's fused SiLU-gate and norm2+residual kernels, so FAST off can be
slower than IDENTICAL. BWD_NOSYNC and BWD_FUSE are the fixes.

**lm-forward** (`logits(ids)`) reaches only `_unpack_block` from this lane's
changes. Only PARAM_VIEWS and ALL have a forward line. NOSYNC, BWD_*, and
HEAD_FUSE change nothing on that path.

## MOJOLEARN_AFN_LM_NOSYNC (lm-train-step)
**Mechanism.** The resident step makes one host wait instead of about 13 on Apple.
- The ids go up straight from the caller's List: two raw-pointer copies per row,
  with no staging host buffers and none of the two waits.
- The loss is not downloaded after the forward.
- The gradient scan, the optimizer's four entry scans and wait, and the four
  validate-after scans all go. Their place is taken by one status kernel: a
  grid-stride pass over grad, param, m and v plus the loss cell, using integer
  atomic minimums.
- The status and the loss come back in one readback with one wait.
- The update stays the shipped shadow copy and the shipped `adam_update_kernel`
  with the shipped scalars.

**Expected.** About 12 fewer round trips, at roughly 0.2 ms or more each on the M3,
plus about 400 MB less scan traffic.

**Risk.**
- A failing check raises after the update instead of before it. The shadow is valid,
  so `_byte_recover` rolls back to the same state with the same message order.
- A nonfinite loss is reported as "nonfinite loss" only after a full backward and
  update that are then rolled back.
- Numerics: none. The same kernels run in the same order.

## MOJOLEARN_AFN_LM_BWD_NOSYNC (lm-train-step)
**Mechanism.** `_bwd_rms_norm_kernels` (transformer/checks/transformer_backward.mojo)
kept `ctx.synchronize()` three times per norm on every non-IDENTICAL tier. That is
6 per layer and 48 per step at 8 layers. This define drops them. Every launch is on
one in-order context and no host read sits between them.

**Expected.** 48 fewer waits per step.

**Risk.** Low. IDENTICAL never had these waits.

## MOJOLEARN_AFN_LM_BWD_FUSE (lm-train-step)
**Mechanism.** FAST takes the routes that IDENTICAL takes on Apple:
- the shipped norm-kernel arm, with no internal wait;
- the fused `bwd_mul2_silu_backward_kernel` for the SwiGLU gate VJP, one launch
  instead of the split mul2 and SiLU pair;
- the fused norm2 + residual backward.

**Expected.** About 2 to 3 fewer launches per layer, plus the one internal wait.

**Risk.** Low. These are IDENTICAL's own kernels.

## MOJOLEARN_AFN_LM_PARAM_VIEWS (lm-train-step, lm-forward)
**Mechanism.**
- Each block's nine weights become `create_sub_buffer` views of the flat `param`, and
  its nine weight gradients become views of the flat `grad`. They are re-bound every
  step (host only, like the shipped emb/head views).
- The per-layer unpack launch (forward) and pack launch (backward) go: 16 launches
  per train step and 8 per forward.
- The 144 separately allocated weight and weight-gradient buffers are released.
  Launch cost grows with live buffers on Metal.

**Expected.** A small launch saving and a cheaper launch everywhere.

**Risk.**
- The view correctness rests on each weight gradient being written whole, once, per
  backward. Each one is a single GEMM output (checked in transformer_backward.mojo).
- Rollback copies in place, so the views stay valid.

## MOJOLEARN_AFN_LM_HEAD_FUSE (lm-train-step)
**Mechanism.** After the head GEMM, one row kernel does all of the cross entropy:
- one threadgroup per token row computes the row max and the softmax denominator;
- it writes `dlogits = (softmax - onehot) / M` into `ce_dlogits`;
- it adds the row's NLL / M into the loss with one f32 atomic.

A one-cell reset launch zeroes the loss first. This replaces:
- the CE forward: the 64 MB logits nonfinite scan, a targets download with two waits,
  and its kernels;
- the CE backward's two kernels.

**Expected.** Two waits and several 64 MB passes fewer per step.

**Risk.**
- The loss mean and the denominator change fold order, an f32 reassociation. It uses
  the same identical_exp/log/div.
- Nonfinite logits are reported as a nonfinite loss instead of "logits nonfinite".
- Targets are validated on the host before upload, as before.

## MOJOLEARN_AFN_LM_ALL (lm-train-step, lm-forward)
All five together. They compose: NOSYNC resets the status cells and HEAD_FUSE resets
the loss cell in the same scratch.
