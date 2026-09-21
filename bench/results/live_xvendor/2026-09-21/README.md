# One model trained live on Apple, NVIDIA and AMD at once

Evidence only. `mojolearn verify` does not run this, and no release depends on it.

## What ran

Commit 82d9ff0f4 on the boxes, 2026-09-21 14:32Z. The Mac's coordinator and
worker ran from the working tree, which differed from 82d9ff0f4 only by the
coordinator's `bound_port`/`ready` attributes (no change on the wire). `tools/live_xvendor_leg.sh OUT nvidia amd`:

| worker | device | shards |
|---|---|---|
| apple-m4 | this Mac, Metal | 0, 1 |
| nvidia-sm_90a | RunPod H100 80GB HBM3, pod bn3ytspmdtkeof | 2 |
| amd-gfx942 | Hot Aisle MI300X VM | 3 |

The coordinator (`mojolearn.cross_vendor.Coordinator`, CPU only) ran on the Mac.
Each remote worker reached it through a reverse ssh tunnel. Every step the
coordinator sent `step`; each worker computed its shards' gradients with
`ParallelByteLanguageModelTrainer.shard_gradient` and sent them; the
coordinator summed all four in shard order with `ordered_fold`; every worker
applied that total with `apply_gradient` and reported the sha256 of its
complete state (parameters, m, v, flags).

The model and data are the `tools/par_lm_xvendor.py` recipe: 2 blocks,
d_model 32, vocabulary 512, batch 2, length 32, K = 4 shards, 6 steps, weights
and tokens from SHAKE-256.

## Result

All three workers reported the same state hash after every step, and every
step equals the one-process column recorded on the M4
(`bench/results/par_lm_xvendor/2026-09-21/apple-m4-1gpu.json`), which the H100
and MI300X one-process columns also equal:

| step | state (all three workers) |
|---|---|
| 1 | 9932e700be7baf9c |
| 2 | ddb92404d62a16e1 |
| 3 | c4e8c600747c8d47 |
| 4 | 52c78ce6847ae49c |
| 5 | c3546a777d6e628f |
| 6 | 0abc34974e913d92 |

`rows.json` has the full 64-hex hashes per worker and every shard's loss.
`coordinator.log`: `AGREED on 6 steps across 3 workers`, then
`6 steps compared, 0 differ` against the recorded column.

## What would have stopped it

The coordinator refuses, by name, on any state disagreement after a step, on
workers starting from different states, and on shards not owned exactly once.
Seen to fire on the M4 (`tools/live_xvendor.py local` with a sabotaged worker):
a one-ulp change to one element of one worker's summed gradient at step 4 was
refused after step 4; overlapping and missing shard assignments were refused
before step 1. The host tests in `python/mojolearn/tests/test_cross_vendor.py`
hold the same refusals and the fold with fake trainers.

## Boxes

Both remote runners fetched their logs (`nvidia-gate.txt`, `amd-gate.txt`) and
deleted their boxes; see the teardown lines in those runners' logs.
