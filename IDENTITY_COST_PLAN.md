# The price of the equal sign

Written 2026-08-23 night, on Andrew's question. Two separate ideas that sound
like one. First, what bit identity COSTS, and whether that cost can be
estimated without measuring it. Second, whether a certified equal sign lets you
name the cheapest GPU for a job. The second is the product idea; the first is
the part we can do tonight, and it is the part that looks novel.

STANDING CONSTRAINT ON THIS WHOLE FILE. Andrew's order of 2026-08-23 is that
identity comes first and MEASUREMENT WAITS. `tools/lanes_price.sh` and
`tools/gemm_price.sh` exist and stay blank. Nothing in this plan authorizes a
timing run. The reason Part 1 is worth doing now is precisely that it needs
none.

---

## Part 1. The identity tax is a STATIC property, not an empirical one

The literature treats the cost of determinism as a slowdown you report after
benchmarking. That framing is available to us too, and `lanes_price.sh` already
implements it honestly with alternated pre-built binaries and a mode witness.

But our profiles are FROZEN VERSIONED DOCUMENTS that enumerate every seam. That
makes the overhead countable by reading source. You cannot measure right now.
You can COUNT. Counting is free and it is not measurement.

The tax decomposes into five terms, each derivable per stage.

**1. Transcendental tax.** Every portable transcendental replaces one hardware
instruction with a polynomial. Count the ops in `identical_exp`,
`identical_log1p`, `identical_rsqrt`, `identical_div` in `mojo_only/numerics.mojo`.
Count the call sites per stage from the profile's stage list. Multiply. For the
Mamba scan this term is dominated by S6, one `exp` per (d, n, l).

**2. Flush tax.** The contracts say every seam RESULT and every operand LOADED
from a buffer passes `ftz`. `mamba/IDENTICAL_MAMBA_CONTRACT.md` section 4
enumerates 17 seams. That is a countable number of extra ops per cell, not an
estimate. Note the term is ZERO on hardware that already flushes, which is
itself a result worth stating.

**3. Fold depth tax.** The profile's tree depth against the vendor primitive's
depth, computed from the profile constants. For GEMM that is
`contract_leaf_size(k)` and the fixed balanced tree against a native warp
reduction. Arithmetic, not benchmarking.

**4. Refused hardware tax.** No tensor cores, no vendor BLAS dispatch, no
unordered atomics. A ratio of published peak numbers off a datasheet.

**5. Occupancy tax.** A pinned launch geometry (one thread per (b, d) in the
scan) against the vendor's chunking bounds occupancy analytically from the
shape.

Sum per stage and you have a PREDICTED identity tax before running anything.

**Why this is the right shape for this repo.** The prediction is falsifiable.
When measurement is eventually authorized, `lanes_price.sh` becomes the
FALSIFICATION of the static estimate rather than its source. If the two
disagree, that disagreement is a finding, because it means the toolchain is not
doing what the source says it does. That is the same discipline as every
sabotage here. See [[verify reach, not output]] in spirit: a predicted number
you cannot falsify is not a result.

**Honesty flag.** Not yet checked against prior art. The claim to check is
narrow: has anyone derived the cost of a determinism contract from the contract
itself, rather than reporting a measured slowdown?

---

## Part 2. The cheapest GPU, and what is actually new about it

Under a certified identity contract the step count is shared BY CONSTRUCTION,
so cost for candidate GPU g is

    C_g = N_steps * t_g * p_g

with t_g a measured per-step time and p_g the effective price per second.

**What is NOT novel, recorded here so nobody re-claims it.** Predicting a
network's runtime on another GPU from measurements on one you have, explicitly
for cost-efficient selection, is Habitat (arXiv 2102.00527, reported 11.8%
average error across six architectures). Recommending cost-efficient cloud GPU
configurations is Srifty (MLSys 2022). The cost formula is not a contribution.
Deterministic training on ONE fixed hardware and software configuration is
ordinary engineering. NVIDIA states cuDNN does not guarantee bitwise
reproducibility across architectures and PyTorch does not guarantee it across
platforms or releases, so the ABSENCE of the guarantee is well documented and
claiming we noticed it is not a contribution either.

**What looks defensible, in descending order of strength.**

1. **The identity tax as a computable static property of a versioned contract,
   validated against a measurement harness.** Part 1. This is the strongest and
   the cheapest to produce.

2. **Exactness on the algorithmic axis.** A large part of any cross-GPU
   prediction error is modeling whether the other device reaches the same
   outcome. Under a certified contract that term is not modeled, it is
   eliminated, and cross-GPU cost comparison stops being a prediction and
   becomes arithmetic over a measured per-step time. The claim to check is
   whether anyone has framed hardware selection as a controlled counterfactual
   rather than a regression.

3. **Mid-run migration.** If the bits are identical, checkpoint on one vendor
   and resume on another and the trajectory is unchanged. That makes spot-price
   arbitrage possible DURING a run. Today resuming on different hardware
   perturbs the trajectory and nobody can prove it did not matter.

4. **The certificate as an audit artifact.** A cheaper run carrying
   stage-by-stage evidence that it performed the identical computation is a
   compliance object, not only an optimization.

---

## Part 3. What it would actually take, stated so nobody underestimates it

Claims 2 and 3 need identity closed for TRAINING, not inference. That means
backward, optimizer state, RNG, data ordering, checkpoint serialization and
collectives, each closed the way the forward seams are closed here.

What exists today is the classical lanes plus ONE Mamba-1 block, forward only,
and `mamba/IDENTICAL_MAMBA_CONTRACT.md` section 9 says plainly that no
training, no backward and no multi-block model is claimed.

The narrowest honest first target is a dense MLP in float32 with fixed batch
shapes, ReLU, softmax cross entropy, the landed GEMM, plain SGD, one GPU. That
is a few dozen operators and a closed loop, not a framework.

This competes for calendar with the IAAI system paper (September 8) and the
ICLR methods paper (September 18). It is a THIRD thread. Naming it here does
not schedule it.

---

## Part 4. Why "fast AND identical" matters to this argument

A cost story about identity is much weaker if identity is understood to mean
serialization. It does not. Bit identity does not require SERIAL, it requires
FIXED. A parallel tree is identical as long as its shape is a function of the
dimensions and the declared profile and not of warp width, block size,
occupancy or sequence length.

GEMM v1 already proves this in this repository and is certified on three
vendors. Its topology is serial ascending inside a leaf of at least 128, then a
FIXED balanced tree across leaves with adjacent pairing and a bit-for-bit
odd-tail carry. That is a parallel reduction.

What upstream's fast scan does wrong for our purposes is not that it uses a
tree. It is that it lets the MACHINE pick the tree.
`csrc/selective_scan/selective_scan_fwd_kernel.cuh:62` selects
`cub::BlockScan<scan_t, kNThreads, cub::BLOCK_SCAN_WARP_SCANS>`, whose shape
follows warp width, and lines 354 to 372 pick `kNThreads` FROM THE SEQUENCE
LENGTH. Their fast path is therefore not bit-identical to itself across
sequence lengths, let alone across vendors, because `SSMScanOp`
(`selective_scan_common.h:141-145`) is associative in exact arithmetic and not
in float32.

So the fast identical scan is a SECOND PROFILE with a known design. Run the
recurrence serially inside a chunk of C tokens, have each chunk emit its
composed linear map, then combine chunk maps with GEMM v1's frozen balanced
tree, adjacent pairing, odd-tail carry. Depth falls from L to L/C plus
log2(L/C), the same asymptotics as the cub scan. The cost against cub is the
constant factor of emulating one fixed tree on three warp widths, not a loss of
parallelism. Two profiles coexisting is already the expected end state here,
per the GEMM charter's ruling on `core/gram_splitk.mojo`.

Sequenced AFTER the v1 serial scan lands, never beside it, because two profiles
of the same math written at once is the file convergence that has killed
rounds.
