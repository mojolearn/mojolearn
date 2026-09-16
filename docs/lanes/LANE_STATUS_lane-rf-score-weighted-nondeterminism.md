# lane/rf-score-weighted-nondeterminism

**Status 2026-09-16: REPRODUCED and localized to one node. THREE of my hypotheses have been
falsified by my own experiments, and the honest conclusion is that everything I measured is
EXPOSURE, not MECHANISM. NO named cause. NO fix.** The 0.8.6 blocker stands. Nothing shipped
changes (the diagnostic define's default is the reference value).

Nine MI300X legs, $1.30 total, every box verified deleted.

## The defect

1. **The clf/reg asymmetry is NOT an output-shape artifact.** Over 80 fits the classifier's
   **model** is bit-stable in all five exported arrays. The defect is regressor-specific.
2. **The weighted metric path is exonerated.** `reg_unweighted` passes no weights and moves.

1-5% of regressor fits move; every odd value is distinct, so it is a race. Stable values match
NVIDIA and the CPU column. Exactly ONE node per occurrence, at depth 6, its depth-7 children
differing as a downstream cascade; the competing splits are on different FEATURES.
`Split::update` gives an equal-gain tie to the HIGHER `colid`, so flips in both directions are
not a tie-break bug. Depth 6 is VISIBILITY (~31 bootstrap rows per node), not location.

Tree SHAPE can also move: node counts differ between fits (`6812` vs `6814`), seen in the
STOCK binary as well as the modified ones, so it belongs to the defect and not to my define.

## WHAT I GOT WRONG, three times

| hypothesis | committed in | killed by |
|---|---|---|
| The trigger is a SECOND `find_best_splits` launch | `ce9e4ffd9`, `061f9162b` | leg 7/8: `cap16/11` is ONE launch and moves |
| The variable is column count >= 11 | `241fa7a59` | leg 8 `cap16/16` 0/100 ... which leg 9 showed was a FLUKE |
| A PARTIAL final column-block group | `2e70de849` | leg 9: `cap8/16` is `[8,8]`, both groups FULL, moves 10/300 |

The second one is the instructive failure. I flagged leg 8's `cap16/16` 0/100 as only ~7%
unlikely **and retracted a hypothesis on it in the same message**. Leg 9 ran it at 300 and got
4/300. The retraction was wrong and is itself retracted. **Do not conclude from a single
underpowered zero, including when you labelled it underpowered.**

## Leg 9: the numbers that survive (300 per arm, three provably distinct binaries)

    digests distinct: stock=24c92e17 cap8=f3eb543b cap16=50fcb884   (all runs exit 0)

| arm | cols | dimy | launches | moves | rate |
|---|---|---|---|---|---|
| stock (cap 10) | 16 | [10,6] | 2 | 16/300 | 5.3% |
| cap8 | 16 | [8,8] | 2 | 10/300 | 3.3% |
| cap16 | 11 | [11] | 1 | 5/300 | 1.7% |
| cap16 | 16 | [16] | 1 | 4/300 | 1.3% |
| any cap | <=10 | - | 1 | **0/600** | - |

- **Launch count MODULATES the rate, it does not gate it.** 1 launch 9/600 (1.50%) against
  2 launches 26/600 (4.33%), one-sided p = 0.003, a ~2.9x effect. Real, but not a switch.
- **<=10 columns is clean across 600 fits.** At the lowest observed rate that absence has
  P = 3.4e-4. Solid -- but it is an ABSENCE of events, and the rate GROWS with column count.

## THE CONCLUSION THAT MATTERS

Column count and launch count both change **how many near-tied split candidates get resolved**
per fit. They are two **exposure proxies for the same thing**, which is exactly why each one
looked like the answer until it was tested against the other. Block-group shape was a third.

**I can reliably expose this race and dial its rate. I cannot say why it happens.** Those are
different things and the gap is not closable by more black-box parameter sweeps -- every such
sweep can only ever move exposure.

## NEXT, and it is not another sweep

Instrument the merge itself: record per (node, launch, block) the candidate `colid`, its gain,
the value read from `split[node]`, `update_result`, and the value written back; then diff a
stable run against a moved one and find the first disagreeing publish. That distinguishes the
two live mechanisms, which remain exactly where leg 3 left them:

1. **Stale read** -- a publisher reads `split[0]` before a prior write is visible.
2. **Dropped write-back** -- a merge lands and is lost.

`DEVIATION 2502` already records that seam ("a merge and the kernel's direct store of the flag
can land in any order"). This is a real code change (threading a debug buffer through
`_publish_to_global` and its callers), not a parameter, which is why it is flagged for a
decision rather than done unilaterally.

## Ruled out, each with a reason

- **Weighted metrics** - `reg_unweighted` moves.
- **Label scale** - serial Float64 host fold over the caller's own buffer after
  `ctx.synchronize()`; `choose_scale` snaps to a power of two, every fixture far from a boundary.
- **Float reduction ordering** - Int32 fixed point plus UInt32 counts, relaxed **integer**
  atomics; no dither on this path.
- **Histogram zeroing extent** - global zero is bytes scaled by the real bin type; shared zero
  uses typed slot indexing. Correct at 4 and 8 bytes.
- **Compare-then-skip H2D caches** - guard by CONTENT on an in-order queue; also excluded by
  leg 2's K=1 arm.
- **The pipeline** - K=1 moves 3/80 vs K=4's 4/80; `n_streams` makes slots, not streams.
- **Wavefront-64 grouping** - DEVIATION 404 pins the reduce to 32 lanes; `eval_best_split_pinned`
  read end to end and is sound.
- **Device-property launch shapes** - no `get_attribute`, occupancy or CU query on the path.
- **Cross-device peer copy** - single-device box.
- **My diagnostic define** - `cap16 @ 10 cols` reads 0/100 matching stock, and leg 9's three
  digests are distinct, so the cap arms are valid evidence.

## Method notes worth keeping

- **A lone zero proves nothing.** Leg 8's 0/100 and leg 3's 0/59 both misled; leg 9's 300-arm
  replication is what corrected them.
- **`odd/17` is measured INSENSITIVE** (0/100 where the launch reading predicted movement), so
  it was dropped rather than reported as a null.
- **The `.so` digest comparison is the authoritative guard** that a build define took effect.
  Grepping the build log is NOT: `build_rf.sh` does not echo its command line, so that check
  returned 0 for an uninteresting reason and was removed.
- **A stale R2 object at a reused key served an old leg's numbers to a new leg's monitor.**
  Detected only because arm labels carry the fixture and column count; the monitor now prints
  `[STALE]` when it sees a retired arm name.
- **Leg 6 was a total loss ($0.05, my error):** the body imported `mojolearn._identity_break`,
  a wheel packaging artifact, and died in a source checkout.
- **Verify a fix in the GENERATED artifact, not the template.** Legs 6 and 7 both died from
  changes that never reached what actually ran.

## Evidence (outside the repo)

    ~/mojolearn-evidence/rf-score-weighted-blocker/leg-{1..9}/remote/identity/

## Resume

    WT=$(mktemp -d)/wt && git worktree add --detach $WT origin/main
    S=$WT/tools/rf_nondeterminism
    bash $S/make_rf_probe6_body.sh /tmp/body.sh 300 900     # 3 builds, ~28 s each
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
| 8 | `33266e44` | 204 / 404 | $0.05 | define proven sound |
| 9 | `c1640753` | 204 / 404 | $0.10 | partial-group falsified; exposure, not mechanism |

Balance $32.78 to $31.48, **$1.30 total** of the $5.00 authorized. One box at a time, partials
to R2 every 60 s.
