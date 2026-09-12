# Byte LM training on a CPU

DEVIATION 2680. This file tracks work in progress. Today the CPU can run the
byte LM's forward pass (docs/BYTE_LM_CPU_INFERENCE.md) and nothing else. There
is no CPU backward pass and no CPU optimizer, so **nothing here claims CPU
training yet.** What exists is the gate that will judge it.

## Why a gate came first

The retained three-vendor capture
(`bench/results/resume/2026-09-07-root-byte-lm-three-vendor`) holds, for every
one of the 128 training steps and on each of Apple Metal, NVIDIA CUDA and AMD
HIP, the full FP32 tensors of that step. The parameters, both Adam moments and
the state flags before the step, the token ids it consumed, the loss, the
gradient, and the parameters, moments and flags after the update. Nothing is
sampled and nothing is reduced to a digest; the digests in each `capture.json`
sit on top of the bytes.

So a CPU training step can be certified against recorded bytes from three
vendors without renting a GPU. Each step records its own starting state, so a
step is judged in isolation and a disagreement at step 87 needs no replay of
the 86 before it.

Until today nothing read those gradients. `tools/byte_lm_host_gate.py` reads
parameters and loss, and takes `comparison.json`'s `identity_admitted` flag on
trust for everything else. The one comparison that did cover gradients and
moments was root-only evidence code that ran once and compared the vendors to
each other.

## The gate

`tools/byte_lm_cpu_train_gate.py`, two modes.

`vendors` re-derives the three-vendor agreement from the raw bytes. For each
selected step, every array of every pair of vendor trees must be equal byte for
byte, and every array must match the SHA-256 its own `capture.json` records.
This needs no GPU, no binding and no build, and runs in about a second.

| | |
|---|---|
| steps | 128 of 128 |
| arrays per step | 11, including `grad`, `post_m`, `post_v` |
| comparisons | 4224 of 4224 equal |
| recorded digests verified | 4224 |
| vendors | Apple M4 Metal, NVIDIA CUDA, AMD MI325X HIP |

`cpu` is the gate proper and does not run yet. It replays selected steps
through a CPU training surface and compares `grad`, `loss`, `post_p`, `post_m`
and `post_v` against one vendor tree. It refuses with exit 2 while that surface
is absent, and names what has to appear, rather than reporting a pass over
nothing.

A difference is reported by tensor. The registry is a fixed order of 21
tensors, so a flat element index is localized to a name, an index inside that
tensor and the two IEEE-754 bit patterns.

## What a pass will and will not say

A pass is a statement about this model profile
(`mojolearn.byte-lm.b2-l32-d32-h4-kv2-ff64-v256-blocks2.fp32.v1`), this batch
shape and this optimizer configuration, which the byte LM trainer restricts to
plain positive-lr AdamW.

It will not say anything about other batch schedules. The weight gradient
contracts over the token count, and `gemm/checks/gemm_backward.mojo` states the
consequence directly: the gradient at 1024 tokens is not the same bits as the
gradient at 512 tokens accumulated twice, and cannot be under any fixed
partition, because the two are different sums over different partitions. The
microbatch schedule is part of a training run's numerical specification. So CPU
training identity is claimed per shape, exactly as the inference sweep claims
it per shape.

## What is still missing

That table was wrong when first written and is corrected here. Almost none of
it was missing. The decoder block backward exists in full as a host oracle,
`transformer/checks/transformer_backward_oracle.mojo`, 1598 lines, all 37
stages, alongside host RMS norm, SiLU, RoPE and softmax backward and the host
GEMM backward routing. `training/checks/train_step_check.mojo` already composes
an entire host step out of these and reports thirteen stages compared against
the device bitwise with four negative controls firing, at a one block fixture
with its own registry, on one device.

| Piece | Status |
|---|---|
| AdamW | normative host oracle, `training/checks/optimizer_oracle.mojo`, seams O1 to O14 |
| Cross-entropy backward | normative host oracle, `training/checks/loss_oracle.mojo` |
| Embedding gradient | normative host oracle, fixed ascending fold over sorted runs |
| GEMM backward | host routing over the reference GEMM, `_gemm_bwd_a` / `_gemm_bwd_b` |
| Decoder block backward, 37 stages | normative host oracle, already written |
| Byte LM shaped composition, two blocks | `training/byte_lm_host_backward.mojo`, **compiles on seven CPUs**, never yet run |
| Binding and Python surface | absent, this is the remaining work |
| The `cpu` mode of the gate | refuses until that surface exists |

## The import question, measured

`transformer_backward_oracle.mojo` computes entirely on the host but imports
three names from `gemm/checks/gemm_backward.mojo`, which imports
`max.gpu.host`, and `IdentityTrace` from `core/identity_trace.mojo`, which does
too. `bindings/_mojolearn_byte_lm_host.mojo` says "HOST ONLY. No DeviceContext,
no kernel, no GPU, and nothing imported from the GPU side." Whether that rule
is enforced decided whether a CPU trainer needed a refactor of shared certified
files, so it was measured rather than argued from the comment.

`training/checks/byte_lm_host_bwd_probe.mojo` imports the oracle and the host
step and compiles with no accelerator target. **Exit 0 on all seven runners,
x86-64, ARM64 and Apple.** So the rule is policy the toolchain does not enforce
here, no shared file has to move, and no GPU code path is touched.

The same probe compiles `byte_lm_host_backward.mojo`, which nothing else
imports and which therefore no build would otherwise check. Its first pass
produced three errors, all the same rule, that reading a `List` out of a tuple
is an explicit copy or a transfer. Nothing structural failed, no import was
rejected and no oracle was missing.

The GPU backward is a useful starting point rather than a translation problem.
It contains no float atomic anywhere. Every place that needs a fold or a
scatter is written so one thread owns one output cell and accumulates in a
fixed ascending order, or work is partitioned so ownership is exclusive, and
there is an "eager" attention backward whose kernels are one thread per output
cell with serial chains. That is a sequential algorithm that happens to run on
a GPU.

One thing to establish rather than assume: the GPU's default attention
backward is the fused path, and a host port would follow the eager one, so the
two must be shown to agree bitwise.
