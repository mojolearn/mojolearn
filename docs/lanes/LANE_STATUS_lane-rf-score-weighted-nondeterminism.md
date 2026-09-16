# lane/rf-score-weighted-nondeterminism

**Status 2026-09-16: REPRODUCED, localized to ONE NODE, and the TRIGGER is isolated with
p = 9.2e-4. NOT root-caused to a line. NO FIX LANDED.** The 0.8.6 blocker stands. No shipped
code is changed here.

Five MI300X legs, $0.60 total, every box verified deleted.

## THE HEADLINE

The defect requires a **SECOND `find_best_splits` launch** accumulating into a `split[node]`
slot that `initSplit` initialized only once.

| launches per round | fits | moves |
|---|---|---|
| **1** | 400 | **0** |
| **2** | 400 | **10** |

Fisher one-sided **p = 9.2e-4**. Arms: 1 launch = cols1, cols2, cols10 (leg 4), cols10 (leg 5);
2 launches = cols16 (leg 4), cols11, cols12, cols16 (leg 5).

10 columns and 11 columns differ by **one feature** - nearly identical tie density and
candidate count - yet 11 moves (3/100) and 10 never does (0/200 across two legs). At 11
columns the second launch carries a **single** column, so it is one block per node: the race
is NOT between blocks inside the second launch, but between the second launch and state the
first launch left behind.

## What the defect is, corrected

Two claims in the original framing are measurably wrong:

1. **The clf/reg asymmetry is NOT an output-shape artifact.** Over 80 fits the classifier's
   **model** is bit-stable in all five exported arrays; its trees never moved. The defect is
   regressor-specific.
2. **The weighted metric path is exonerated.** `reg_unweighted` passes no weights and moves
   with the other three reg parts.

## Leg 1 - reproduction and stage

80 fits from the installed 0.8.6 release wheel. wide 39/40 stable at `49be8ea935a47640`,
base 38/40 stable at `d744878e7c0e31ee`; ~4% move. Every odd value is **distinct**, so this is
a race. Stable values match NVIDIA and the CPU column.

`_offsets` and `_left_child` never move and node counts are identical every run, so **tree
SHAPE is invariant**; `_colid` and `_quesval` move, usually dragging `_leaves`; exactly ONE
tree differs per occurrence.

## Leg 2 - n_streams (NEGATIVE: not the pipeline)

K=4 moves 4/80, K=1 moves 3/80. `n_streams` creates **slots, not streams** (no `create_stream`
or second `DeviceContext` on the fit path; `randomforest.mojo:2352` pipelines K trees over "the
one" queue). With one slot there is no shared `SplitStaging` to race, so staging is excluded.

## Leg 3 - the diverging NODE

3 of 59 wide repeats, 0 of 59 base. **Every divergence is one node at depth 6** with shape
intact; the depth-7 nodes that also differ are that node's own children (106 -> 211/212,
95 -> 189/190): a downstream cascade.

    tree=2   node 106  depth 6  colid 15 ->  1
    tree=0   node  76  depth 6  colid 10 ->  7
    tree=12  node  95  depth 6  colid 14 -> 15

**Not a tie-break bug.** `Split::update` awards an equal-gain tie to the HIGHER `colid`, so
flips in both directions cannot come from equal gain plus that rule.

**DIRECTION, worth keeping.** At 16 columns launch 1 covers cols 0-9 and launch 2 covers 10-15.
Two of the three flips lose the SECOND launch's winner (15 -> 1 and 10 -> 7, falling back to a
launch-1 column); the third (14 -> 15) is within launch 2's own range.

**Depth 6 is VISIBILITY, not location.** At ~31 bootstrap rows per node candidates are close
enough that losing one flips the winner; shallower nodes decide decisively and depth 7-8 nodes
go pure and never select.

## Legs 4 and 5 - the launch-count discriminator

| leg | arm | cols | launches | moves / 100 |
|---|---|---|---|---|
| 4 | cols16 | 16 | 2 | 3 |
| 4 | cols10 | 10 | 1 | 0 |
| 4 | cols2 | 2 | 1 | 0 |
| 4 | cols1 | 1 | 1 | 0 |
| 5 | cols10 | 10 | 1 | 0 |
| 5 | cols11 | 11 | 2 | 3 |
| 5 | cols12 | 12 | 2 | 3 |
| 5 | cols16 | 16 | 2 | 1 |

`N_BLKS_FOR_COLS = 10` (`builder.mojo:78`) caps `n_blocks_dimy = min(10, n_sampled_cols - col)`
(`:1827`) and `enqueue_best_splits` strides `c += N_BLKS_FOR_COLS` (`:2061`). DEVIATION 1916
fuses `initSplit` and the mutex re-zero into the setup launch (`:2011`), which runs **before**
the column loop - so every launch after the first merges into an already-populated slot.

Leg 4 alone could have been "something about 16"; leg 5 kills that by straddling the boundary
at 10 vs 11. Tie density cannot explain a one-feature difference.

## Ruled out, each with a reason

- **Weighted metrics** - `reg_unweighted` moves.
- **Label scale** - the `mag` loop reads `yp[i]`, the caller's own numpy buffer, after
  `ctx.synchronize()`: a serial Float64 fold over stable host memory. `choose_scale` snaps to a
  power of two and every fixture's Sigma|y| is far from a boundary (fractional log2 0.14-0.92).
- **Float reduction ordering** - `RegressionBin` is `label_sum: Int32` fixed point plus
  `count: UInt32`, relaxed **integer** atomics. No dither on this path (`split.mojo:37`).
- **Histogram zeroing extent** - the global zero is `size_of[O.BinT]() * len_histograms`, bytes
  scaled by the real bin type, and `enqueue_zero_bytes` covers exactly `nbytes`; the shared zero
  uses typed slot indexing. Correct at both 4 and 8 bytes.
- **Compare-then-skip H2D caches** - guard by CONTENT on an in-order queue; also excluded by
  leg 2's K=1 arm.
- **Wavefront-64 grouping** - DEVIATION 404 pins the reduce to 32 lanes under IDENTICAL, which
  the wheel ships. `eval_best_split_pinned` read end to end: barriers in uniform control flow,
  correct write-after-read ordering, `N_SPLIT_SCRATCH = TPB`, `comptime assert TPB % 32 == 0`,
  single-thread publish. Sound.
- **The cross-block merge WIDTH** - equally wide at 10 columns (10 blocks per node) and stable
  there. Width is not the variable; launch count is.
- **Device-property launch shapes** - no `get_attribute`, occupancy or CU query on the fit path.
- **Cross-device peer copy** (Sep 15 hazard) - single-device box.

## Candidate mechanisms, NOT yet distinguished

Both live in the cross-launch handoff of `split[node]`:

1. **Stale read.** Launch 2's `_publish_to_global` reads `split[0]` before launch 1's non-atomic
   payload write (`split[unsafe_offset=0] = split_reg.copy()`) is visible. The mutex's
   RELEASE/ACQUIRE pair orders this within a launch; across launches visibility rests on the
   kernel boundary.
2. **Lost write-back.** Launch 2 merges correctly but its write-back is dropped, leaving
   launch 1's winner - which matches the observed direction in 2 of 3 flips.

DEVIATION 2502 already records that "a merge and the kernel's direct store of the flag can land
in any order", which is the same seam.

## NEXT (leg 6, about $0.05)

Run `cols11` with leg 3's node dumping. At 11 columns the second launch is exactly column 10,
so every divergence must involve col 10 against a col 0-9 winner. **If the diverging node always
flips between col 10 and a col 0-9 column, the cross-launch handoff is confirmed and the two
mechanisms above can be told apart by which side wins.** Naming the line still needs an
instrumented build (dump per-node (launch, colid, gain, update_result)), which is a larger leg.

## Underpowered results, flagged

- Leg 1's n-sweep (3 repeats per n) read `distinct=1` everywhere. At ~4% that expects 0.12
  moves. **Not evidence of n-independence.**
- Leg 2's K=1 vs K=4 (3 vs 4 of 80) excludes staging as the *sole* mechanism; it could not
  detect a 30% reduction.
- Leg 3's base fixture moved 0/59 - an unlucky draw, not a fix.
- A single 0/100 arm is ~5% likely by luck at a 3% rate. The strength here is the POOLED
  0/400 vs 10/400 and the one-feature 10-vs-11 contrast, not any single zero.

## Evidence (outside the repo)

    ~/mojolearn-evidence/rf-score-weighted-blocker/leg-{1,2,3,4,5}/remote/identity/rf_probe*.json

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
| 5 | `3df6a953` | 204 then GET 404 | $0.05 |

Balance $32.78 to $32.18, **$0.60 total**. One box at a time, partials to R2 every 60 s.
