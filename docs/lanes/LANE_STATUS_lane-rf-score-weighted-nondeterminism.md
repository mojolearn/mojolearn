# lane/rf-score-weighted-nondeterminism

**Status 2026-09-16: REPRODUCED, localized to ONE NODE. NOT root-caused. NO FIX LANDED.**
The 0.8.6 blocker stands. No shipped code is changed on this branch.

Three MI300X legs, $0.30 total, each box verified deleted.

## What the defect is, corrected

`rf-score-weighted` moves between two fits on one MI300X. Two claims in the original
framing are measurably wrong:

1. **The clf/reg asymmetry is NOT an output-shape artifact.** It was argued the classifier's
   argmax swallows a last-bit difference. Measured over 80 fits: the classifier's **model**
   is bit-stable in all five exported arrays. Its trees never moved. The defect is
   regressor-specific.
2. **The weighted metric path is exonerated.** `reg_unweighted` passes no weights and moves
   with the other three reg parts.

## Leg 1 - reproduction and stage

80 fits from the installed 0.8.6 release wheel:

| fixture | stable cell | stable | moved |
|---|---|---|---|
| wide | `49be8ea935a47640` | 39/40 | 1 (`5f7a3bd31486fb53`) |
| base | `d744878e7c0e31ee` | 38/40 | 2 (`6debf9a1b42c0d43`, `1b5a80605dd90a43`) |

~4% of fits. Every odd value is **distinct**, and none equals leg 3's `50ce4a9f62cddf8e` or
leg 4's `eb475cefa0f32408` from the release record, so this is a race, not two deterministic
paths. Stable values match NVIDIA and the CPU column.

Per-array and per-tree hashing: `_offsets` and `_left_child` never move, node counts are
identical every run (wide 7592, base 6718) so **tree SHAPE is invariant**; `_colid` and
`_quesval` move, usually dragging `_leaves`; exactly ONE tree differs per occurrence.

## Leg 2 - the n_streams discriminator (NEGATIVE)

| arm | moves / fits | rate |
|---|---|---|
| K=4 (shipped default) | 4 / 80 | 5.0% |
| K=1 (serialized) | 3 / 80 | 3.8% |

Serializing the pipeline does not fix it. `n_streams` creates **slots, not streams** (no
`create_stream` or second `DeviceContext` on the fit path; `randomforest.mojo:2352` says K
trees pipelined over "the one" queue). With one slot there is no shared staging to race and
no cross-tree cache reuse, so the pipelined `SplitStaging` is excluded.

## Leg 3 - the diverging NODE

3 divergences in 59 wide repeats; 0 in 59 base repeats. **Every divergence is one node at
depth 6**, with `shape_same=True`, and the depth-7 nodes that also differ are that node's own
children (node 106 -> children 211, 212; node 95 -> children 189, 190) - a downstream cascade,
not scattered corruption.

    tree=2   node_local=106  depth=6  colid  15 ->  1   quesval 4801.08   -> -0.000123
    tree=0   node_local=76   depth=6  colid  10 ->  7   quesval 19.3959   ->  0.822187
    tree=12  node_local=95   depth=6  colid  14 -> 15   quesval -5725.53  ->  3855.31

The competing splits are on **different features** (the `wide` fixture scales column j by
`logspace(-4,4)`, so the thresholds differ by scale, not by a near-tie in value).

**Why this is not a tie-break bug.** `Split::update` awards an equal-gain tie to the HIGHER
`colid`. Flips run both ways (15->1 and 10->7, but also 14->15), so equal gain plus the
documented tie-break cannot produce this. Either a **gain differs between runs**, or a
block's candidate is **lost** from the cross-block merge. A lost publish explains both
directions, since the reference is just the first fit and may itself be the run that lost one.

**Depth 6 is probably VISIBILITY, not location.** At ~31 bootstrap rows per node candidates
are close enough that losing one flips the winner; at shallow depths the best split wins
decisively, and at depth 7-8 (~8-16 rows) nodes go pure and never select. The race may occur
at every depth. Depth 6 is not reported as the bug's location.

## Ruled out, each with a reason

- **Weighted metrics** - `reg_unweighted` moves.
- **Label scale** (`bindings/_mojolearn_rf.mojo:594-600`) - the `mag` loop reads `yp[i]`, the
  caller's own numpy buffer, after `ctx.synchronize()`: a serial Float64 fold over stable host
  memory. `choose_scale` snaps to a power of two and every fixture's Sigma|y| is far from a
  boundary (fractional log2 0.14 to 0.92), so a perturbed `mag` could not flip the scale.
- **Float reduction ordering** - `RegressionBin` is `label_sum: Int32` fixed point plus
  `count: UInt32`, relaxed **integer** atomics. No dither on this path (`split.mojo:37`).
- **Histogram zeroing extent** - global zero is `size_of[O.BinT]() * len_histograms`, bytes
  scaled by the real bin type, and `enqueue_zero_bytes` covers exactly `nbytes` (16-byte body
  plus a `grid=1` tail); the shared zero uses typed slot indexing `histogram[i] = O.BinT()`.
  Correct at both 4 and 8 bytes. The 16 KiB reservation differs (4096 vs 2048 slots) but only
  `histogram_len` cells are ever zeroed *and* only `histogram_len` are ever read.
- **Compare-then-skip H2D caches** - `DeviceArgs.upload` guards by **content** (byte compare
  against last-sent), so `_invalidate_args_cache` not resetting `staged` is safe; and the
  queue is in-order. Also excluded empirically by leg 2's K=1 arm.
- **Wavefront-64 grouping** - DEVIATION 404 pins the reduce to 32 lanes under IDENTICAL, which
  is what the wheel ships. `eval_best_split_pinned` read end to end: barriers in uniform
  control flow, correct write-after-read ordering, `N_SPLIT_SCRATCH = TPB`, `comptime assert
  TPB % 32 == 0`, single-thread publish. Sound.
- **Device-property launch shapes** - no `get_attribute`, occupancy or CU query on the fit
  path; the one SM-shaped number (`4 * 108`, `core/philox.mojo`) is a frozen stride, and the
  bootstrap and feature draws are pure functions of `(seed, tree_id, node index, index, n)`.
- **Cross-device peer copy** (Sep 15 hazard) - single-device box, no peer copy on this path.

**Every mechanism reachable by reading is now either excluded or verified sound, and there is
no named cause.** That is weaker than naming a survivor, and is stated rather than papered over.

## NEXT: leg 4, the cross-block merge

`find_best_splits_kernel` launches one block per (node, sampled column) and merges per node
through a spin mutex (`_publish_to_global`). `max_features` is public, and
`n_sampled_cols = max(1, int(n_cols * max_features))`, so on this 16-feature fixture
`max_features = 1/16` leaves **one block per node and no cross-block merge**.

Arms: 16, 10, 2 and 1 columns, 100 repeats each, `wide`.
**1 column stable while 16 moves implicates the merge; still moving at 1 column puts it
within-block, upstream.** CONFOUND, stated: fewer columns also means fewer near-tied
candidates, so a null at 1 column is suggestive, not conclusive; the 16-column arm is the
in-session positive control.

## Underpowered results, flagged

- Leg 1's n-sweep (n = 500..8000, 3 repeats) read `distinct=1` everywhere. At ~4% per fit,
  3 repeats expects 0.12 moves. **Not evidence of n-independence.**
- Leg 2's K=1 vs K=4 (3 vs 4 of 80) excludes staging as the *sole* mechanism, because full
  serialization should have removed the effect; it could not detect a 30% reduction.
- Leg 3's base fixture moved 0/59. At the wide rate that is an unlucky draw, not a fix.

## Evidence (outside the repo)

    ~/mojolearn-evidence/rf-score-weighted-blocker/leg-1/remote/identity/rf_probe.json
    ~/mojolearn-evidence/rf-score-weighted-blocker/leg-2/remote/identity/rf_probe2.json
    ~/mojolearn-evidence/rf-score-weighted-blocker/leg-3/remote/identity/rf_probe3.json

## Resume

    WT=$(mktemp -d)/wt && git worktree add --detach $WT origin/main
    S=$WT/tools/rf_nondeterminism
    bash $S/make_rf_probe_body.sh /tmp/body.sh 40 1500        # mints presigned GET+PUT
    cd $WT && MOJOLEARN_HOTAISLE_SPEC=8core MOJOLEARN_GPU_ARCHS=gfx942 \
      MOJOLEARN_GEMM_LEG_EXTRA=/tmp/body.sh \
      MOJOLEARN_GEMM_LEG_OUT=$HOME/mojolearn-evidence/rf-score-weighted-blocker/leg-N \
      MOJOLEARN_HOTAISLE_LANE=rf-probe \
      bash tools/hotaisle_leg.sh amd --rent --minutes 40 --skip-gates   # drop --rent to dry run

## Boxes and cost

| leg | VM | deleted | cost |
|---|---|---|---|
| 1 | `8796a71c` | 204 then GET 404 | $0.15 |
| 2 | `68ef24e1` | 204 then GET 404 | $0.05 |
| 3 | `f03ec215` | 204 then GET 404 | $0.05 |

Balance $32.78 to $32.48, **$0.30 total**. One box at a time, partials uploaded to R2 every
60 s so a dead box never costs a leg.
