# lane/rf-score-weighted-nondeterminism

**Status 2026-09-16: REPRODUCED, localized to one node. TWO successive hypotheses of mine
have been falsified by my own experiments. A third now fits every measurement and has an
unrun falsifier. NOT root-caused. NO FIX LANDED.** The 0.8.6 blocker stands; no shipped
behaviour changes (the diagnostic define's default is the reference value).

Eight MI300X legs, $1.05 total, every box verified deleted.

## The defect

1. **The clf/reg asymmetry is NOT an output-shape artifact.** Over 80 fits the classifier's
   **model** is bit-stable in all five exported arrays. The defect is regressor-specific.
2. **The weighted metric path is exonerated.** `reg_unweighted` passes no weights and moves.

~2-4% of regressor fits move; every odd value is distinct, so it is a race. Stable values
match NVIDIA and the CPU column. Exactly ONE node per occurrence, at depth 6, with its
depth-7 children differing as a downstream cascade; the competing splits are on different
FEATURES. `Split::update` gives an equal-gain tie to the HIGHER `colid`, so flips in both
directions are not a tie-break bug. Depth 6 is VISIBILITY (~31 bootstrap rows per node), not
location.

## TWO RETRACTIONS, both mine

**1. "The trigger is a SECOND launch" (`ce9e4ffd9`, `061f9162b`) is FALSE.** At the stock cap
of 10, "<=10 columns" and "1 launch" are the same partition, so legs 4-5 could not tell them
apart. Raising the cap to 16 makes 11 columns run ONE launch, and it still moves: 5/177
across legs 7-8. Launch count is dead.

**2. "The variable is column count >= 11" is ALSO false.** At cap 16, 16 columns is 0/100
while 11 columns is 4/100. Column count alone does not predict.

## The hypothesis that fits everything: a PARTIAL final column-block group

`n_blocks_dimy = min(N_BLKS_FOR_COLS, n_sampled_cols - col)` per launch.

| cap | cols | dimy sequence | final group | moves |
|---|---|---|---|---|
| 10 | 1, 2, 10 | [n] | full | 0 / 600 |
| 16 | 10 | [10] | partial | 0 / 100 |
| 16 | **16** | [16] | **FULL** | **0 / 100** |
| 10 | 11 | [10, **1**] | partial | 7 / 300 |
| 10 | 12 | [10, **2**] | partial | 3 / 100 |
| 10 | 16 | [10, **6**] | partial | 8 / 300 |
| 16 | **11** | [**11**] | **partial** | **5 / 177** |

Every mover has **>=11 columns AND a partial final group**. Every stable arm lacks one or
both. This is the first reading that explains `cap16/16` stable and `cap16/11` moving.

**It is a hypothesis fitted to existing data, not a tested one.** It was built after the fact
to fit seven arms, which is exactly the kind of story that looks compelling and is wrong.

## THE FALSIFIER (leg 9, about $0.10)

At **cap 8, 16 columns is [8, 8]: two launches, both groups FULL.**

- "partial final group" predicts **STABLE**
- "column count >= 11" predicts **MOVES**
- "launch count" (already dead) would predict moves

300 repeats per arm so a null means something: at ~2.7%, 0/300 has probability ~3e-4, while
leg 8's 0/100 arms were only ~7% unlikely, which is why nothing is built on them alone.

Arms: stock/16 `[10,6]` positive control; **cap8/16 `[8,8]` the discriminator**; cap16/11
`[11]` and cap16/16 `[16]` to replicate leg 8's contrast at higher N.

## Ruled out, each with a reason

- **Weighted metrics** - `reg_unweighted` moves.
- **Label scale** - serial Float64 host fold over the caller's own buffer after
  `ctx.synchronize()`; `choose_scale` snaps to a power of two, every fixture far from a
  boundary.
- **Float reduction ordering** - Int32 fixed point plus UInt32 counts, relaxed **integer**
  atomics; no dither on this path.
- **Histogram zeroing extent** - global zero is bytes scaled by the real bin type; shared zero
  uses typed slot indexing. Correct at 4 and 8 bytes.
- **Compare-then-skip H2D caches** - guard by CONTENT on an in-order queue; also excluded by
  leg 2's K=1 arm.
- **The pipeline** - K=1 moves 3/80 vs K=4's 4/80; `n_streams` makes slots, not streams.
- **Launch count** - falsified above.
- **Wavefront-64 grouping** - DEVIATION 404 pins the reduce to 32 lanes; `eval_best_split_pinned`
  read end to end and is sound.
- **Device-property launch shapes** - no `get_attribute`, occupancy or CU query on the path.
- **Cross-device peer copy** - single-device box.
- **My diagnostic define** - `cap16 @ 10 cols` reads 0/100, matching stock, so raising the cap
  introduces no instability of its own and the cap arms are valid evidence.

## Underpowered and failed results, flagged

- Leg 1's n-sweep (3 repeats per n): at ~4% that expects 0.12 moves. **Not evidence.**
- Leg 3's base fixture 0/59, and leg 8's 0/100 arms: single zeros, ~5-7% likely by luck.
- **`odd/17` is measured INSENSITIVE** (0/100 in the stock arm where it should have moved), so
  it proves nothing in any arm and was dropped.
- **Leg 6 was a total loss ($0.05, my error):** the body imported `mojolearn._identity_break`,
  a wheel packaging artifact, and died in a source checkout.
- **Leg 7 arm B crashed** on a shape change (`(6812,) vs (6814,)`) because the probe assumed
  shape invariance. Node counts CAN differ between fits; now recorded rather than fatal. Seen
  once in 10 divergences, only where divergences occurred, so not define-specific.
- **`define_mentions_in_build_log=0` was a weak check** (`build_rf.sh` does not echo its
  command line). The `.so` digest comparison is the authoritative guard and it worked.
- **A stale R2 object at the same key served leg 7's numbers to leg 8's monitor.** Detected
  only because the arm labels carry the fixture and column count; with `moved/runs` alone I
  would have read it as fresh.

## Evidence (outside the repo)

    ~/mojolearn-evidence/rf-score-weighted-blocker/leg-{1..8}/remote/identity/

## Resume

    WT=$(mktemp -d)/wt && git worktree add --detach $WT origin/main
    S=$WT/tools/rf_nondeterminism
    bash $S/make_rf_probe6_body.sh /tmp/body.sh 300 900     # builds 3 binaries, ~28 s each
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
| 4 | `bcce7c94` | 204 / 404 | $0.05 | threshold found |
| 5 | `3df6a953` | 204 / 404 | $0.05 | boundary straddled |
| 6 | `aa2f3ccb` | 204 / 404 | $0.05 | LOST, wheel-only import |
| 7 | `4b164b21` | 204 / 404 | $0.15 | launch reading broken |
| 8 | `33266e44` | 204 / 404 | $0.05 | define sound; partial-group reading |

Balance $32.78 to $31.73, **$1.05 total**. One box at a time, partials to R2 every 60 s.
