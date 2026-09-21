# Multi-GPU language model training across device counts and vendors, small shape

Evidence only. `mojolearn verify` does not run this.

## The claim

`ParallelByteLanguageModelTrainer` cuts every step into K logical shards. K is
part of the recipe and is independent of the device count. Each shard's
gradient comes from the same IDENTICAL kernels a one-GPU step uses, and the
shards are combined by a fixed left fold
(`total = g0`, then `total = ftz(fma(1, ftz(total), ftz(g_k)))`, in shard order,
`training/byte_lm_parallel.mojo`), never by a collective that picks its own
order. So the bits must not depend on how many GPUs share the shards, or on
which vendor computed which shard.

## The recipe

`tools/par_lm_xvendor.py run`, defaults: 2 blocks, d_model 32, 4 heads, 2 KV
heads, FF 64, vocabulary 512, batch 2, length 32, K = 4 shards, 6 steps, seed
20260921. Weights and tokens come from SHAKE-256, not from an RNG library, so
every box builds the same problem from the recipe alone. Each run happens once.

Per step, on the box:

1. the trainer under test equals a replicated one-device trainer (losses,
   parameters, m, v, flags, summed gradient);
2. each shard trained alone from the pre-step state gives the same loss it has
   inside the K-shard step;
3. the host's ordered fold of the shard gradients equals the device's sum, bit
   for bit.

`compare` then pools runs from any boxes and also folds every MIXED assignment
(shard k's gradient from run V(k)) and holds it to every run's device sum. It
exits 1 on any disagreement and exits 1 when nothing was compared. Verified to
fail: a corrupted state hash, a single record, and a one-ulp change to one
shard gradient (caught in exactly the 7 of 8 assignments that use that shard).

## Columns

Every run once. State hash after each step (parameters, m, v, flags):

| step | Apple M4, 1 GPU | H100, 1 GPU | 2x H100 | MI300X, 1 GPU | 2x H100 resumed from the 1-GPU step 3 | M4 resumed from the 2x H100 step 3 | M4 resumed from the MI300X step 3 |
|---|---|---|---|---|---|---|---|
| 1 | 9932e700be7baf9c | 9932e700be7baf9c | 9932e700be7baf9c | 9932e700be7baf9c | - | - | - |
| 2 | ddb92404d62a16e1 | ddb92404d62a16e1 | ddb92404d62a16e1 | ddb92404d62a16e1 | - | - | - |
| 3 | c4e8c600747c8d47 | c4e8c600747c8d47 | c4e8c600747c8d47 | c4e8c600747c8d47 | - | - | - |
| 4 | 52c78ce6847ae49c | 52c78ce6847ae49c | 52c78ce6847ae49c | 52c78ce6847ae49c | 52c78ce6847ae49c | 52c78ce6847ae49c | 52c78ce6847ae49c |
| 5 | c3546a777d6e628f | c3546a777d6e628f | c3546a777d6e628f | c3546a777d6e628f | c3546a777d6e628f | c3546a777d6e628f | c3546a777d6e628f |
| 6 | 0abc34974e913d92 | 0abc34974e913d92 | 0abc34974e913d92 | 0abc34974e913d92 | 0abc34974e913d92 | 0abc34974e913d92 | 0abc34974e913d92 |

`compare` over all seven, with the shard gradients of the Apple, H100 1-GPU,
2x H100 and MI300X runs: **1536 comparisons, 0 disagreements**, including
**1512 mixed assignments** of shards to Apple, NVIDIA and AMD runs, each
folded on the host and equal to every device sum. Every on-box check passed
on every run.

Where each came from:

- Apple M4: `apple-m4-1gpu.json`, commit 42bea8726, this Mac, Metal.
- 2x H100 80GB HBM3: RunPod pod mobzs40xr7gejg, commit fab190e5a, sm_90a,
  `nvidia-sm_90a-*.json`, `nvidia-h100x2-gate.txt`. Pod verified gone (404).
- AMD Instinct MI300X: Hot Aisle VM, commit c2367f3e7, gfx942,
  `amd-gfx942-one.json`, `amd-mi300x-gate.txt`. One GPU (Hot Aisle's 2x VM pins
  each body to one GPU); AMD two-GPU equality is the Sep 14 `par-byte-lm`
  record. VM verified gone.
- M4 resumed from the MI300X step-3 handoff: `apple-m4-from-amd.json`.
- M4 resumed from `nvidia-sm_90a-two.handoff.npz`: `apple-m4-from-nvidia-2gpu.json`,
  run on this Mac after the leg came home. The vendor and the device count
  change at the same handoff.

The H100 leg did not resume from the Apple handoff because `.gitattributes`
left `bench/results/*` out of `git archive`; `bench/results/par_lm_xvendor` is
now exported. The handoff was run in the other direction instead.

Shard gradient files (about 4 MB each) are kept outside the repository.

A live step whose shards ran on the three vendors at the same moment is in
`bench/results/live_xvendor/2026-09-21/`.

## Found on the way

The Apple-only residual2/RMSNorm fusion (7365fd0ac) read the next block's
buffers one past the end of the list after a `pop`, aborting every Apple LM
step with two or more blocks at M <= 2048
(`training/byte_lm.mojo:1161: index 1 is out of bounds`). Fixed in ea9c1dd3a.
After the fix, steps 1 to 4 equal the Sep 19 binding built before the fusion
existed, bit for bit.
