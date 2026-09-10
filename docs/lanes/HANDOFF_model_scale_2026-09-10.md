# LANE: what language model to build next (opened 2026-09-10)

The current published training result is a two-block byte model,
**34,944 FP32 parameters**, batch 2, context 32, `d_model` 32, vocab 256,
128 steps, bitwise identical on Metal, CUDA and HIP. The result is real. The
problem is that "bitwise-identical AI across GPUs" invites a reader to think
of language models, and 34,944 parameters is five orders of magnitude away
from anything anyone calls one. That gap, not novelty, is the largest
credibility risk in the paper.

This lane closes it. It does not chase GPT-3.

## Runtime shape and backward status (corrected 2026-09-10)

The byte-LM wrapper originally fixed its dimensions to define the first
qualified 34,944-parameter profile, tensor registry and checkpoint witness.
That was a scope boundary for the initial training result, not an arithmetic
restriction in the underlying runtime-shaped Llama block.

`training/byte_lm_config.mojo` now supplies host-validated runtime batch,
length, model width, heads, KV heads, head dimension and intermediate width.
`ByteTrainer` and the Python `ByteLanguageModelConfig` surface use those
shapes. The default retains the exact original profile/registry/ABI; other
shapes carry an explicit v2 profile. Two blocks and the 256-byte alphabet
remain architectural constants. New shapes are not numerically qualified
by host checks or a successful build. The JSON checkpoint codec retains its
2 MiB bound; larger states can be exported as state_dict arrays.

`TransformerBlock.backward` was already implemented, registered in the Mojo
binding, and used by SambaStack before this handoff was drafted. The
zero-state prefill VJP is IDENTICAL-only. New host wiring checks cover the
21-address/8-scalar ABI and mixed attention/Mamba reverse gradient routing;
they do not replace numerical GPU tests. The stale claim that all attention
layers are refused has been removed.

## Target 1: enwik8 or text8 bits-per-byte, ~10M parameters

The compute, price and quality estimates below are planning hypotheses, not
measured training results or authorization to start these long runs.

Do this before anything larger.

- Byte-level, which is what we already have, so enwik8 and text8 are the
  natural benchmarks and no tokenizer work is needed.
- Roughly 5 to 20M parameters, context 256 to 512, about 1e9 training bytes.
- Compute is on the order of 1e16 to 1e17 FLOPs, single-digit hours per leg.
- Estimated **$50 to $200 total** across an NVIDIA and an AMD rental, with
  reruns. Apple is free.

Why this and not a bigger model: the credibility gap is not parameter count,
it is that we report "held-out cross-entropy fell 5.5413 to 2.8436" on pinned
bytes identified by hash. That is rigorous and tells a reviewer nothing about
whether the model is good. A BPC number on a named public benchmark is read
instantly and compared to published baselines. It is worth more than a 10x
parameter count with a private metric.

## Target 2: GPT-2 Small scale, conditional

124M parameters over roughly 2.5B tokens is about 1.9e18 FLOPs for forward
and backward.

| achieved FP32 throughput | hours per leg | approx cost per leg |
|---|---|---|
| 7 TFLOP/s (older planning assumption) | ~74 | ~$185 |
| 35 TFLOP/s (speed-lane target) | ~15 | ~$40 |

So this is already affordable on rented NVIDIA and AMD. The speed lane does
not unlock it, it makes it cheap.

Do not start it until Target 1 has run the pipeline end to end. That is not a
scale concession, it is de-risking: the shape has to be unfrozen, context has
to reach 256 or more, attention paths untested at that length have to hold, a
real corpus loader has to exist and a BPC evaluation has to be written.
Discovering all of that at $185 a leg on three-day turnarounds is worse than
discovering it for $100 total on hour-long ones.

## Read this before the Apple arithmetic below

Metal's throughput does **not** cap the model size. It caps how many
identical-across-three-vendors STEPS we run, which is a different and far
cheaper quantity. The section after next explains why, and it is the reason
Target 2 is affordable despite the numbers immediately below.

Do not read the Apple estimates as a scale ceiling. They are the cost of the
identity evidence, not the cost of the model.

## Apple's cost, and this is why

Every all-three-vendor identity result has to run on the M4. The recorded
Apple column is an M4 base 10-core, which is roughly 4 TFLOP/s of FP32 peak.
If achieved efficiency there resembles the H100's older ~10% estimate, the effective
figure is well under 1 TFLOP/s; even at a generous 30% it is about
1.3 TFLOP/s.

1.9e18 FLOPs at 1.3 TFLOP/s is about 17 days of continuous compute, and the
M4 throttles roughly 1.7x within twenty minutes of sustained load
(`the-m4-drifts-1.7x-in-20-min`), on a fanless laptop that is also the
development machine. Call it a month, with no interruption tolerated, for one
leg of a three-leg claim.

That is the cost of training GPT-2 Small ON Metal, which is not what we are
going to do. **The achieved M4 TFLOP/s is currently unmeasured and is item 3
of the speed lane.** A second M4 is available; if it is actively cooled
(mini or MacBook Pro) rather than fanless, most of the 1.7x drift goes away
and it can run unattended, which removes the operational blocker even though
it does not move peak.

## The experimental design, and why Metal does not cap scale

**Identity is a per-step property.** If step k returns identical bits on
three vendors given identical inputs, and step k+1's inputs are step k's
outputs, the trajectory is identical by induction. N identical steps
demonstrates the property; running the whole training three times is a linear
scaling of the same evidence, not stronger evidence. So the model size is set
by whatever hardware trains it, and Metal's job is to carry identity
evidence, not to carry the training.

Split the claim into three pieces that each say exactly one thing.

1. **Identity.** N steps, all three vendors, bit for bit, where N is what an
   M4 sustains inside one thermal window.
2. **Quality.** The full run to a reportable benchmark number, in identical
   mode, on the fastest single leg. This is what proves the model is real.
3. **Portability of the trained artifact.** Take the final checkpoint from
   the long run, resume it for N steps on the other two vendors, and match.
   The existing bidirectional NVIDIA/AMD checkpoint continuation is already
   this at the current scale, so the machinery exists.

**Spot-check at several points, not only at step zero.** The one legitimate
objection to this design is that early steps may not exercise a rare path: a
denormal, an unusual value range, a NaN branch that only appears deep in
training. The answer is to keep checkpoints from the long run at early,
middle and late points and repeat the N-step three-vendor comparison from
each. Three windows of N steps on two rented legs is a rounding error against
the long run, and it answers the objection directly in a way a single early
window does not.

Each piece states its own scope, which makes this more honest than an
undifferentiated three-way training run, and it costs perhaps a twentieth as
much.

## Order of work

1. Runtime byte-LM shapes are implemented; qualify alternate shapes numerically.
2. TransformerBlock backward is already bound; retain the mixed-stack wiring
   checks and run the existing numerical hybrid gate when hardware work resumes.
3. Context length to 256+, where the 32 KB Metal threadgroup ceiling and the
   fused-attention head-dim limits will bite
   (`fused_attention.mojo`, head-dim-128 backward already falls back to the
   eager path on a 32 KB column). Some GPU.
4. Target 1, enwik8 BPC at ~10M parameters, under the three-piece design.
5. Reassess Target 2 against whatever the speed lane has achieved by then.

## What not to do

Do not attempt GPT-3 175B. FP32 weights alone are roughly 700 GB.

Do not report a parameter count as the headline. Report the benchmark number
and the three-vendor identity, and let the parameter count be a row in a
table.

## Execution rules

Root runs every build, test and measurement; lanes author source and record
RUN OWED with the exact command (`subagents-no-local-tests`). No heavy local
compute (`no-heavy-local-compute`); the Apple leg is bounded, two cores,
`nice 19`, inside one thermal window. Rented GPUs self-expire on a one-hour
lease and an unleased GPU is an orphan.
