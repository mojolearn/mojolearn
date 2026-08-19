# Resume here

Written 2026-08-19 at the end of a long session. Read this, then `UNWIRED.md`,
then `PORTED_MAP.tsv`.

## What this repository is

A clean-room port of **CatBoost's GPU oblivious (symmetric) tree learner**
into Mojo, targeting Metal first. Nothing from mojotrees. `ported/` mirrors
CatBoost's own paths file for file (their constant `catboost/cuda/` prefix
dropped, and `cuda_util` renamed `gpu_util` because CUDA is not our
vocabulary). `mojo_only/` is what CatBoost never had to write.

**The one rule: COPY, DO NOT IMPROVE.** Every deviation is a confound when a
number finally arrives, and every deviation is recorded in `PORTING.md`.

## State: what works

- All three packing policies: binary 32 features per `UInt32`, half-byte 8,
  one-byte 4. Verified on device.
- Six histogram kernels: direct and gather for each policy. The gather
  variant is verified against the direct one under a REVERSED index.
- Scan, sibling subtraction, zero, copy, `WriteReducesHistograms`, score and
  argmax, split flags, segmented stable partition, gathers, partition update,
  per-leaf stats, leaf values. All verified against hand calculations.
- Uniform-binary trees grow correctly at depths 1 to 8, rows conserved, 46 of
  64 leaves populated at depth 6.
- The compressed-index layout builder, per-policy feature blocks, and
  bin-feature to (feature, bin) resolution: all verified on a mixed dataset.
- CatBoost's option names with `check()` refusing every unported option BY
  NAME, plus a three-level `determinism` ladder that has no CatBoost
  counterpart.
- Harness moved in from mojotrees: interleaved arms, machine lock, box
  record. Plus `tools/remote_gpu.sh` for the nvidia and amd columns.

## State: what is broken

**Mixed-width trees do not split.** They grow, conserve every row and produce
`2^depth` partitions, but leave 1 non-empty leaf. `UNWIRED.md` has the full
trail. The live target, sharply characterized:

- one-byte features ALONE fail, 3 of 4 slices wrong. Binary alone and
  half-byte alone are exact, so cross-block interference is excluded.
- the standalone `check_one_byte_bits[6](2)` PASSES at what look like
  identical parameters.
- two enumerable differences remain: the probe uses 2048 rows (4 accumulation
  iterations) against the standalone's 640 (2), and passes `leaf_count = 2`
  rather than 1.

**Row count RULED OUT.** `check_one_byte_bits[6](2, 32)` runs at 2048 rows
and passes, 0 wrong of 512. The kernel is correct at EVERY parameter the
probe uses: 4 features, 64 folds, 6 bits, 2 stat planes, 2048 rows.

**A sixth real bug was found and fixed here and was NOT the cause:** the
per-block scratch was never zeroed. CatBoost's writeback is guarded by
`if (abs(val) > 1e-20f)`, so a cell whose value is zero is never written and
keeps whatever the buffer held. This port zeroed the FLAT histogram and not
the per-block scratch the kernels actually write, which also means the
scratch cannot be shared between blocks without clearing.

**WHAT IS LEFT.** The kernel is correct in isolation; the probe path is not.
The remaining difference is that the probe goes through
`launch_histograms_for_blocks` and reads the POST-BRIDGE flat histogram,
where the standalone launches the kernel directly and reads its output.

**NEXT STEP:** read `block_hist` directly in the probe, BEFORE the bridge
runs, and compare it against the host tally. That splits the remaining space
exactly in half: if block_hist is right, the bridge is wrong; if it is wrong,
the launch parameters `launch_histograms_for_blocks` computes differ from
what the standalone passes. Everything else about the kernel is excluded.

## The method that has worked, and the one that has not

Five hypotheses have been killed on this bug. **Every one was killed by a
measurement in a single attempt. None were killed by reasoning.** Three
inferences each found a REAL bug without finding the one being chased:

1. compressed-index base conflated with the feature-block stride
2. `WriteReducesHistograms` never ported (CatBoost keeps TWO histogram
   layouts and this port only had one)
3. the block-to-flat bridge duplicated, advancing the offset twice

When stuck: **read the bytes.** `mojo_only/mixed_hist_probe.mojo` localized in
one run what three rounds of inference could not.

## Rules earned here, do not relearn them

- A kernel is not ported until it has been **enqueued**. Compiling is not
  evidence; `--emit=object` targets the host.
- Kernel params must be `Int32`, pointers `MutPointer[T, MutAnyOrigin]`.
- `enqueue_copy` is host<->device ONLY. Passing a device pointer as
  `src_ptr` silently does nothing.
- One host staging buffer per `enqueue_copy`; they are asynchronous.
- Only interleaved arms inside ONE process compare. This box drifts 2-3x.
  The harness overturned a committed claim the first time it ran.
- A configuration that cannot be varied inside one process cannot be
  measured here, so make it a parameter.

## Measurements that stand

- Replication pays **8.56x on a wide histogram** (1024 cells) and is
  INDISTINGUISHABLE on a narrow one (64 cells). `replicas_for` keys on output
  size. Fitted to two points; the interval between them is unmeasured.
- **32 binary features is the WORST shape for CatBoost's design.** Every
  timing conclusion in this repo comes from it and should be re-taken on a
  wide dataset once mixed trees work.
- Depth-8 uniform-binary tree, 500k x 32: roughly 45-70 ms depending on
  thermal window. Not comparable to anything without interleaving.
