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

| column | commit | result |
|---|---|---|
| Apple M4, 1 GPU (`apple-m4-1gpu.json`) | 42bea8726 | PASS, 6 steps; step-3 handoff committed beside it |

(NVIDIA and AMD two-GPU legs: see below once they land.)

## Found on the way

The Apple-only residual2/RMSNorm fusion (7365fd0ac) read the next block's
buffers one past the end of the list after a `pop`, aborting every Apple LM
step with two or more blocks at M <= 2048
(`training/byte_lm.mojo:1161: index 1 is out of bounds`). Fixed in ea9c1dd3a.
After the fix, steps 1 to 4 equal the Sep 19 binding built before the fusion
existed, bit for bit.
