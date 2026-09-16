# lane/rf-score-weighted-nondeterminism

**Status 2026-09-16: REPRODUCED and localized to one node. The LAUNCH-COUNT reading I
committed earlier is now COMPROMISED and may be wrong. NOT root-caused. NO FIX LANDED.**
The 0.8.6 blocker stands. No shipped behaviour is changed (leg 6+ adds a diagnostic define
whose default is the reference value).

Seven MI300X legs, $0.95 total, every box verified deleted.

## READ THIS FIRST: a correction to my own earlier headline

Commits `ce9e4ffd9` and `061f9162b` say the trigger is a SECOND `find_best_splits` launch,
with ONE launch 0/400 and TWO launches 10/400, Fisher p = 9.2e-4. **That statistic is real
but the variable was probably misattributed.**

At the stock cap of 10, "<=10 sampled columns" and "1 launch" are **perfectly correlated**,
and so are ">=11 columns" and "2 launches". Legs 4 and 5 varied column count and I read the
effect as launch count. Leg 7 broke the correlation for the first time by raising the cap to
16, so that 11 columns runs ONE launch -- and it **still moved, 1/77**.

I am NOT claiming the launch reading is falsified, because that arm is compromised (below).
I am claiming it is **unproven and probably wrong**, and that the variable which survives
every stock measurement is the **sampled column count**, not the launch count.

## The defect

Two claims in the original framing are measurably wrong:

1. **The clf/reg asymmetry is NOT an output-shape artifact.** Over 80 fits the classifier's
   **model** is bit-stable in all five exported arrays. The defect is regressor-specific.
2. **The weighted metric path is exonerated.** `reg_unweighted` passes no weights and moves.

~2-4% of regressor fits move. Every odd value is distinct, so this is a race. Stable values
match NVIDIA and the CPU column.

## What the stock binary says (legs 1-5, 7 arm A)

| sampled columns | fits | moves |
|---|---|---|
| <= 10 | 400 | **0** |
| >= 11 | 600 | 13 |

Fisher one-sided p is ~1e-3 either way you label the variable, because at the stock cap the
two labels are the same partition.

**The diverging node** (leg 3, and leg 7 arm A agrees): exactly ONE node per occurrence, at
depth 6, with the depth-7 nodes that also differ being that node's own children -- a
downstream cascade. The competing splits are on different FEATURES.

    tree=2   node 106  depth 6  colid 15 ->  1
    tree=0   node  76  depth 6  colid 10 ->  7
    tree=12  node  95  depth 6  colid 14 -> 15

**Not a tie-break bug**: `Split::update` gives an equal-gain tie to the HIGHER `colid`, so
flips in both directions cannot come from equal gain plus that rule.

**Depth 6 is VISIBILITY, not location.** At ~31 bootstrap rows per node candidates are close
enough that losing one flips the winner; shallower nodes decide decisively and depth 7-8
nodes go pure and never select.

## Leg 7, and why arm B is compromised

Arm A (stock, cap 10) reproduced, so a source build is a valid vehicle:

    stock  wide/11   2/100
    stock  wide/16   1/100
    stock  odd/17    0/100     <-- the intended control, INSENSITIVE

Arm B (cap 16, `-D MOJOLEARN_RF_BLKS_COLS16=1`, .so digest confirmed different) ran only
`wide/11` before crashing:

    cols16cap wide/11  1/77     <-- ONE launch, and it still moved

Two problems, both stated rather than smoothed over:

1. **The intended control was insensitive.** `odd/17` is two launches at the stock cap and
   should have moved under the launch reading. It read 0/100. So it can prove nothing in
   either arm, and it also argues against launch count on its own. (`odd` is plain standard
   normal like `base`, which also ran at a much lower rate; `wide`'s logspace column scaling
   is probably what manufactures the near-tied gains.)
2. **Arm B crashed on a SHAPE CHANGE**, which is a finding in itself:

       ValueError: operands could not be broadcast together with shapes (6812,) (6814,)

   The node count differed between fits. Legs 1-3 and arm A never saw that: `_offsets` and
   `_left_child` were invariant across hundreds of fits. Either
   (a) **my define is unsound** -- raising a constant the reference calls "a plain member
       initialised to 10 and never reassigned" is untested territory, and then arm B's move
       is MY bug and the launch reading is untested; or
   (b) **it is the same defect with a bigger blast radius** -- a flip at a shallower node
       changes which children go pure, so the node count changes.

   By reading, the define looks sound: all five shipped uses are the definition, two SIZING
   sites for `max_len_histograms`, `n_blocks_dimy = min(cap, cols - col)` and the loop
   stride; the histogram INDEX uses runtime `grid_dim.y <= cap` while the SIZE uses the
   comptime cap, so index <= size at any cap and nothing assumes 10. But "sound by reading"
   is exactly the class of claim this lane has falsified by experiment three times.

## NEXT: leg 8 settles (a) vs (b) for about $0.10

Run both binaries over `wide/10`, `wide/11`, `wide/16`, 100 repeats each, with shape changes
now RECORDED instead of crashing the arm.

**`cap16 @ wide/10` is the control that matters.** 10 columns is ONE launch under both caps,
and the stock binary is stable there across 400 fits.

- **cap16 wide/10 stable** -> the define is sound, arm B's `wide/11` move is real, and the
  launch-count reading is dead: column count is the variable.
- **cap16 wide/10 MOVES** -> raising the cap is itself unsound, every cap16 arm is void, and
  the launch reading remains untested.

`odd/17` is dropped as measured-insensitive.

## Ruled out, each with a reason

- **Weighted metrics** - `reg_unweighted` moves.
- **Label scale** - a serial Float64 host fold over the caller's own buffer after
  `ctx.synchronize()`; `choose_scale` snaps to a power of two and every fixture's Sigma|y| is
  far from a boundary.
- **Float reduction ordering** - `RegressionBin` is Int32 fixed point plus UInt32 counts under
  relaxed **integer** atomics. No dither on this path.
- **Histogram zeroing extent** - global zero is `size_of[O.BinT]() * len_histograms` (bytes,
  scaled by the real bin type); shared zero uses typed slot indexing. Correct at 4 and 8 bytes.
- **Compare-then-skip H2D caches** - guard by CONTENT on an in-order queue; also excluded by
  leg 2's K=1 arm.
- **The pipeline** - K=1 moves 3/80 against K=4's 4/80; `n_streams` makes slots, not streams.
- **Wavefront-64 grouping** - DEVIATION 404 pins the reduce to 32 lanes under IDENTICAL;
  `eval_best_split_pinned` read end to end and is sound.
- **Device-property launch shapes** - no `get_attribute`, occupancy or CU query on the path.
- **Cross-device peer copy** - single-device box.

## Underpowered and failed results, flagged

- Leg 1's n-sweep (3 repeats per n) read `distinct=1` everywhere; at ~4% that expects 0.12
  moves. **Not evidence of n-independence.**
- Leg 3's base fixture 0/59 - an unlucky draw, not a fix.
- A single 0/100 arm is ~5% likely by luck at a 3% rate. Strength comes from the pooled
  contrast, never a lone zero.
- **Leg 6 was a total loss ($0.05, my error):** the body imported `mojolearn._identity_break`,
  a WHEEL packaging artifact, and died in a source checkout. Fixed by loading
  `tools/identity_break.py` by path.
- **`define_mentions_in_build_log=0` was a weak check** that failed for an uninteresting
  reason (`build_rf.sh` does not echo its command line). The `.so` digest comparison is the
  authoritative guard that a define took effect, and it worked.

## Evidence (outside the repo)

    ~/mojolearn-evidence/rf-score-weighted-blocker/leg-{1,2,3,4,5,6,7}/remote/identity/

## Resume

    WT=$(mktemp -d)/wt && git worktree add --detach $WT origin/main
    S=$WT/tools/rf_nondeterminism
    bash $S/make_rf_probe6_body.sh /tmp/body.sh 100 900     # builds from source, 28 s per build
    cd $WT && MOJOLEARN_HOTAISLE_SPEC=8core MOJOLEARN_GPU_ARCHS=gfx942 \
      MOJOLEARN_GEMM_LEG_EXTRA=/tmp/body.sh \
      MOJOLEARN_GEMM_LEG_OUT=$HOME/mojolearn-evidence/rf-score-weighted-blocker/leg-N \
      MOJOLEARN_HOTAISLE_LANE=rf-probe \
      bash tools/hotaisle_leg.sh amd --rent --minutes 40 --skip-gates

The tree must be CLEAN or a real leg is refused; commit before renting.

## Boxes and cost

| leg | VM | deleted | cost | outcome |
|---|---|---|---|---|
| 1 | `8796a71c` | 204 / 404 | $0.15 | reproduced, stage localized |
| 2 | `68ef24e1` | 204 / 404 | $0.05 | pipeline excluded |
| 3 | `f03ec215` | 204 / 404 | $0.05 | node named |
| 4 | `bcce7c94` | 204 / 404 | $0.05 | column/launch threshold |
| 5 | `3df6a953` | 204 / 404 | $0.05 | boundary straddled |
| 6 | `aa2f3ccb` | 204 / 404 | $0.05 | LOST to a wheel-only import |
| 7 | `4b164b21` | 204 / 404 | $0.15 | arm A reproduced; arm B compromised |

Balance $32.78 to $31.83, **$0.95 total**. One box at a time, partials to R2 every 60 s.
