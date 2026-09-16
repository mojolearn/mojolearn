# lane/rf-score-weighted-nondeterminism

**Status 2026-09-16: REPRODUCED, localized to ONE NODE, and narrowed to a TRIGGER.
NOT root-caused. NO FIX LANDED.** The 0.8.6 blocker stands. No shipped code is changed here.

Four MI300X legs, $0.45 total, every box verified deleted.

## What the defect is, corrected

Two claims in the original framing are measurably wrong:

1. **The clf/reg asymmetry is NOT an output-shape artifact.** It was argued the classifier's
   argmax swallows a last-bit difference. Measured over 80 fits: the classifier's **model** is
   bit-stable in all five exported arrays. Its trees never moved. The defect is
   regressor-specific.
2. **The weighted metric path is exonerated.** `reg_unweighted` passes no weights and moves
   with the other three reg parts.

## Leg 1 - reproduction and stage

80 fits from the installed 0.8.6 release wheel. wide 39/40 stable at `49be8ea935a47640`,
base 38/40 stable at `d744878e7c0e31ee`; ~4% of fits move. Every odd value is **distinct**,
so this is a race, not two deterministic paths. Stable values match NVIDIA and the CPU column.

`_offsets` and `_left_child` never move and node counts are identical every run (wide 7592,
base 6718), so **tree SHAPE is invariant**; `_colid` and `_quesval` move, usually dragging
`_leaves`; exactly ONE tree differs per occurrence.

## Leg 2 - n_streams (NEGATIVE: not the pipeline)

K=4 moves 4/80, K=1 moves 3/80. Serializing does not fix it. `n_streams` creates **slots, not
streams** (no `create_stream` or second `DeviceContext` on the fit path; `randomforest.mojo:2352`
pipelines K trees over "the one" queue). With one slot there is no shared `SplitStaging` to race
and no cross-tree cache reuse, so the staging hypothesis is excluded.

## Leg 3 - the diverging NODE

3 divergences in 59 wide repeats, 0 in 59 base. **Every divergence is one node at depth 6**
with shape intact, and the depth-7 nodes that also differ are that node's own children
(106 -> 211/212, 95 -> 189/190): a downstream cascade, not scattered corruption.

    tree=2   node 106  depth 6  colid 15 ->  1
    tree=0   node  76  depth 6  colid 10 ->  7
    tree=12  node  95  depth 6  colid 14 -> 15

**Not a tie-break bug.** `Split::update` awards an equal-gain tie to the HIGHER `colid`, so
flips in both directions cannot come from equal gain plus that rule. Either a gain differs
between runs, or a block's candidate is LOST.

**Depth 6 is VISIBILITY, not location.** At ~31 bootstrap rows per node candidates are close
enough that losing one flips the winner; shallower nodes decide decisively and depth 7-8 nodes
go pure and never select.

## Leg 4 - the TRIGGER is the second launch, not the column count

| arm | sampled cols | launches | moves / 100 | distinct |
|---|---|---|---|---|
| cols16 | 16 | 2 | **3** | 4 |
| cols10 | 10 | 1 | 0 | 1 |
| cols2 | 2 | 1 | 0 | 1 |
| cols1 | 1 | 1 | 0 | 1 |

**Tie density cannot explain this.** 10 of 16 features has nearly the same candidate density
and still merges 10 blocks per node through the mutex, yet was stable over 100 repeats. What
differs is the **launch count**: `N_BLKS_FOR_COLS = 10` (`builder.mojo:78`) caps
`n_blocks_dimy = min(10, n_sampled_cols - col)` (`:1827`) and `enqueue_best_splits` strides
`c += N_BLKS_FOR_COLS` (`:2061`), so 16 columns runs TWO `find_best_splits` launches and 10
runs one.

Both launches accumulate into the SAME `split[node]` slot, which `initSplit` initializes only
ONCE per round: DEVIATION 1916 fuses initSplit and the mutex re-zero into the setup launch
(`:2011`), which runs BEFORE the column loop. So the suspect is **cross-launch accumulation
into an already-populated global split slot** - which is where DEVIATION 2502 records that
"a merge and the kernel's direct store of the flag can land in any order."

## NEXT: leg 5, straddle the boundary

10 columns is 1 launch; **11 columns is 2** (10 + 1). They differ by a single feature, so tie
density is nearly identical while launch count differs. Arms 10, 11, 12, 16 at 100 repeats.

- **11 moves, 10 stable** -> the second launch is the trigger; the tie-density confound dies.
- **11 stable, only 16 moves** -> it is something about 16 specifically, not the launch count,
  and this reading is wrong.

## Ruled out, each with a reason

- **Weighted metrics** - `reg_unweighted` moves.
- **Label scale** - the `mag` loop reads `yp[i]`, the caller's own numpy buffer, after
  `ctx.synchronize()`: a serial Float64 fold over stable host memory. `choose_scale` snaps to a
  power of two and every fixture's Sigma|y| is far from a boundary (fractional log2 0.14-0.92).
- **Float reduction ordering** - `RegressionBin` is `label_sum: Int32` fixed point plus
  `count: UInt32`, relaxed **integer** atomics. No dither on this path (`split.mojo:37`).
- **Histogram zeroing extent** - the global zero is `size_of[O.BinT]() * len_histograms`, bytes
  scaled by the real bin type, and `enqueue_zero_bytes` covers exactly `nbytes`; the shared zero
  uses typed slot indexing `histogram[i] = O.BinT()`. Correct at both 4 and 8 bytes.
- **Compare-then-skip H2D caches** - guard by CONTENT (byte compare vs last-sent) on an in-order
  queue; also excluded empirically by leg 2's K=1 arm.
- **Wavefront-64 grouping** - DEVIATION 404 pins the reduce to 32 lanes under IDENTICAL, which
  the wheel ships. `eval_best_split_pinned` read end to end: barriers in uniform control flow,
  correct write-after-read ordering, `N_SPLIT_SCRATCH = TPB`, `comptime assert TPB % 32 == 0`,
  single-thread publish. Sound.
- **Device-property launch shapes** - no `get_attribute`, occupancy or CU query on the fit path;
  the draws are pure functions of `(seed, tree_id, node index, index, n)`.
- **Cross-device peer copy** (Sep 15 hazard) - single-device box, no peer copy on this path.

## Underpowered results, flagged

- Leg 1's n-sweep (3 repeats per n) read `distinct=1` everywhere. At ~4% per fit that expects
  0.12 moves. **Not evidence of n-independence.**
- Leg 2's K=1 vs K=4 (3 vs 4 of 80) excludes staging as the *sole* mechanism; it could not
  detect a 30% reduction.
- Leg 3's base fixture moved 0/59 - an unlucky draw at the wide rate, not a fix.
- Leg 4's null arms are 0/100 each, which at the cols16 rate (~3%) would expect ~3. That is
  meaningful for cols10 but weak for cols1 and cols2, whose tie density is genuinely lower.

## Evidence (outside the repo)

    ~/mojolearn-evidence/rf-score-weighted-blocker/leg-{1,2,3,4}/remote/identity/rf_probe*.json

## Resume

    WT=$(mktemp -d)/wt && git worktree add --detach $WT origin/main
    S=$WT/tools/rf_nondeterminism
    bash $S/make_rf_probe5_body.sh /tmp/body.sh 100 1500       # mints presigned GET+PUT
    cd $WT && MOJOLEARN_HOTAISLE_SPEC=8core MOJOLEARN_GPU_ARCHS=gfx942 \
      MOJOLEARN_GEMM_LEG_EXTRA=/tmp/body.sh \
      MOJOLEARN_GEMM_LEG_OUT=$HOME/mojolearn-evidence/rf-score-weighted-blocker/leg-N \
      MOJOLEARN_HOTAISLE_LANE=rf-probe \
      bash tools/hotaisle_leg.sh amd --rent --minutes 40 --skip-gates   # drop --rent to dry run

The tree must be CLEAN or a real leg is refused; commit before renting.

## Boxes and cost

| leg | VM | deleted | cost |
|---|---|---|---|
| 1 | `8796a71c` | 204 then GET 404 | $0.15 |
| 2 | `68ef24e1` | 204 then GET 404 | $0.05 |
| 3 | `f03ec215` | 204 then GET 404 | $0.05 |
| 4 | `bcce7c94` | 204 then GET 404 | $0.05 |

Balance $32.78 to $32.33, **$0.45 total**. One box at a time, partials uploaded to R2 every
60 s so a dead box never costs a leg.
