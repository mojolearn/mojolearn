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

Nothing known. Mixed-width trees now grow, split and conserve every row:
depth 4 gives 16 of 16 non-empty leaves, depth 6 gives 44 of 64 against the
uniform binary tree's 46. `UNWIRED.md` has the full trail of nine exclusions
and seven real bugs.

The cause was `TPointHistOneByte::Reduce`, which is TWO stages in CatBoost
(`hist_one_byte.cu:177-230`) and had been collapsed into one strided fold.
The second defect was the test data: `check_mixed_tree` planted bins as
`x % folds[f]`, but `Folds` is the border count and a feature takes bins
`0..Folds`, so every binary feature was the constant 0.

**READ THIS BEFORE WRITING ANOTHER CHECK.** Both defects were invisible to
the checks written to catch them, for the same reason. A single strided fold
gives the RIGHT answer whenever every cell holds the same value, and
`check_one_byte_bits` assigned bins `(r + f) % n_folds`, which makes every
cell equal. Two separate exclusions therefore reported the kernel correct at
exactly the parameters that were failing. Same kernel, same parameters, only
the bin pattern changed:

    uniform   (r + f) % 64      0 wrong of 512
    scattered hashed          490 wrong of 512

A histogram check whose expected value is the same in every cell verifies the
total and nothing about placement. Plant SCATTERED bins and compare per cell
against a host tally. Likewise, conservation cannot see a tree that never
splits, so both tree checks now require `nonempty >= 2`.

## State: what is next

Every timing number in this port was measured on 32 uniform binary features,
which is one policy, one launch per level and a 64-cell histogram. That is
the shape CatBoost's design is LEAST suited to and the one it was least
written for. The mixed path is what needed fixing before those numbers meant
anything, and it now works, so the whole timing table should be re-measured
on a wide mixed dataset before any of it is quoted.

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
