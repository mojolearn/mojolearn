# Byte LM training on a CPU

DEVIATION 2680. The CPU runs one byte LM training step, forward, backward and
the AdamW update, and reproduces the recorded GPU bytes exactly.

## What has been measured

On all seven CI runners (five x86-64 Linux draws, ARM64 Linux on Azure Cobalt
100, and Apple M1 macOS), replaying steps of the retained capture from each
step's own parameters, Adam moments, token ids and recorded optimizer:

| | |
|---|---|
| CPU step against the recorded Metal bytes | 640 of 640 array comparisons equal, **all 128 steps** |
| arrays compared per step | gradient, loss bits, post-step parameters, post-step `m`, post-step `v` |
| negative control, a wrong-gradient build | caught on every runner, 2 of 10 equal |
| certified CPU inference, unmoved | loss gate 33 of 33, DEVIATION 2612 catch 24 of 33 |

Because `vendors` mode separately shows Apple, CUDA and HIP agreeing on all 128
steps and all 11 arrays, a step that equals the Apple bytes equals all three
vendors. So the CPU step agrees bitwise with AMD and NVIDIA as well, through the
capture rather than through a fresh rental.

**The control fires for the right reason, which is the part worth checking.**
The wrong-gradient build got 8 of 10 arrays wrong and 2 right, and the 2 right
are the loss: a backward-only corruption leaves the forward exact and then
propagates from the gradient into both moments and the updated parameters. The
gate named the tensor, `block0.w_q` element 0, with both bit patterns.

## What this does NOT say

- Every step is covered, so this limit is retired. CI runs all 128 on every
  push, on all seven runners. An earlier revision sampled `every:16` because a
  replay was estimated at ten seconds per step; measured, the full job takes
  2m23s against the sample's 2m6s, so the sample bought seventeen seconds and
  cost 119 steps.
- One model profile, one batch shape. Nine of the gradients contract over the
  token count, so the same tokens in a different batch or microbatch schedule
  are a different sum. Identity here is per shape, exactly as inference is.
- The reference path only, one thread, and that is a deliberate stop rather
  than an unfinished job. **Measured on 2026-09-12: a whole step is 37 to 39 ms
  on the Linux runners and 69 ms on Apple M1**, forward, backward and the AdamW
  update, so all 128 steps replay in under five seconds. Threading it would need
  two fast GEMM orientations that do not exist (the host fast kernel is NT only;
  the backward needs NN and TN) and leaf-parallel folding to keep the weight
  gradients' summation order exact, because they sum across every row and so
  cross every thread boundary. That is a large build whose payoff is making an
  already-fast thing faster, against a real risk to the bit identity that is the
  point. It should be justified by a workload that 37 ms a step makes painful,
  not by the limit's existence.
- Nothing about other algorithm families. Trees and the classical models have
  no backward pass and remain GPU-only.
- `LanguageModelHostTrainer` is deliberately not exported from
  `mojolearn/__init__.py`. An unexported class cannot be mistaken for a promise.

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

`cpu` is the gate proper and it runs. It replays selected steps through
`LanguageModelHostTrainer` and compares `grad`, `loss`, `post_p`, `post_m` and
`post_v` against one vendor tree, taking each step's optimizer configuration
from that step's own `capture.json` rather than from defaults. If the surface is
absent it still refuses with exit 2 and names what has to appear, rather than
reporting a pass over nothing.

`--expect-mismatch` inverts the verdict, which is how the wrong-gradient build
is required to be caught. CI runs both arms on every push: the clean binding
must agree, and a binding built with
`-D MOJOLEARN_GEMM_SABOTAGE_BWD_UNTRANSPOSED=1` must disagree.

That arm was chosen because **DEVIATION 2612's arm cannot reach this path.**
2612 reverses a fold inside `byte_host_logits`, and the training step calls
`gemm_oracle` directly, so a 2612 build computes a correct training step. Until
this control existed the training gate had never been shown capable of failing.
`byte_lm_host_sabotage` was widened in the same change, because it reported the
2612 flag alone and a binding carrying a GEMM backward arm would otherwise
compute wrong gradients while reading back as clean.

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
| Byte LM shaped composition, two blocks | `training/byte_lm_host_backward.mojo`, compiles and **runs correctly on seven CPUs** |
| Binding entry | `byte_lm_host_train_step`, eight addresses, five AdamW scalars, returns the loss bits |
| Python surface | `LanguageModelHostTrainer`, unexported on purpose |
| The `cpu` mode of the gate | runs, and its negative control fires |

Nothing in the list above is now missing. What remains is coverage rather than
capability: nine of 128 steps in CI, one profile, one batch shape, and the
reference path only. Widening any of those is measurement, not construction.

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
