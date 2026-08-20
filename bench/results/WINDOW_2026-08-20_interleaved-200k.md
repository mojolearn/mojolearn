# 200k x 100 interleaved vs CatBoost CPU, 2026-08-20 19:51-19:55 EDT

    pixi run -e bench python tools/interleaved_prep.py /tmp/ilrun synth 200000 100
    pixi run -e bench mojo run -I . \
        bench/interleaved/catboost_interleaved.mojo /tmp/ilrun synth 200000

Harness UNCHANGED -- no timing code, metric, dataset, or CatBoost parameter
was touched for this window. Every run held `tools/with_build_lock.sh`,
including the CatBoost-side prep, so no peer measurement could overlap.

**REDUCED SIZE, ON REQUEST.** The shipped size for this harness is 800k and
that is the number the scoreboard quotes. Andrew authorized 200k for this one
window. The speed rows below are therefore NOT the standing result, and the
reason is quantified in "Scale is the whole story" further down.

**A CONCURRENT SESSION WAS EDITING `gbdt/` THROUGHOUT.** Commit `14c1948`
("Fuse the fixed-to-float bridge into the histogram writeback") landed at
19:54:28, i.e. *during* run 2. Each row below names exactly which tree it
measured; this is not a single-configuration window and must not be read as
one.

## The rows

    run  time      tree state                       border  theirs      ours        speedup  our mse         their mse
    ---  --------  -------------------------------  ------  ----------  ----------  -------  --------------  --------------
    1    19:51:15  1d994d4 + UNCOMMITTED mid-edit      254  8.82-8.94   11.4-12.7   0.69-0.78  14.97115375   0.14726951974
    1    19:51:15  1d994d4 + UNCOMMITTED mid-edit      128  8.11-8.44    8.9-9.8    0.83-0.95  14.75660250   0.15289802058
    H    19:52:00  1d994d4 clean (cold worktree)       254  8.82-8.95   24.9-27.8   0.32-0.36   0.14726953125 0.14726951974
    H    19:52:00  1d994d4 clean (cold worktree)       128  8.13-8.37   19.6-21.3   0.39-0.43   0.15289801758 0.15289802058
    2    19:54:06  fused, uncommitted -> 14c1948       254  8.88-9.61   19.8-21.2   0.42-0.49   0.14726953125 0.14726951974
    2    19:54:06  fused, uncommitted -> 14c1948       128  8.25-8.34   16.3-17.3   0.48-0.51   0.15289801758 0.15289802058
    3    19:54:55  14c1948 clean, warm                 254  8.95-9.85   18.6-18.7   0.48-0.53   0.14726953125 0.14726951974
    3    19:54:55  14c1948 clean, warm                 128  8.26-8.67   15.3-17.7   0.48-0.56   0.15289801758 0.15289802058

ms/tree, 3 reps each, arms alternating inside one process. `speedup` > 1 means
ours is faster. **Run 3 is the only row that measures a committed tree with
warm caches, and it is the one to quote: ~0.50x at both border counts.**

## Accuracy is exact, and that is the result worth having

Every correct row agrees with CatBoost to eight significant figures at both
border counts:

    254:  ours 0.14726953125    theirs 0.14726951974
    128:  ours 0.15289801758    theirs 0.15289802058

Identical across runs 2, 3 and the clean-HEAD run -- so the fusion in
`14c1948` is bit-stable against the tree it replaced on this dataset, which is
what its deviation block claims.

## The 19:51:15 row is a broken intermediate, and it is instructive

Run 1's arm did not learn. `var(y) = 15.27885`, and the mse of a
constant-mean predictor is **15.27885**; run 1 finished at **14.971**, having
moved 2% off the mean in twenty trees where the correct build reaches 0.147.
It was also **~2.2x faster than correct code** (11.4 vs 25.2 ms/tree at 254).

That state existed on disk for under a minute -- the peer rewrote both files
at 19:51:50 and committed the working version at 19:54:28 -- so **this is not
a live defect and nothing needs fixing.** It is recorded for one reason: a
histogram fusion that is wrong is *faster*, not slower, and the speedup column
alone would have read as a win. What separated them here was the end-to-end
mse against CatBoost in the same process.

HONEST LIMIT ON THIS CLAIM: `check-hist2` was run at 19:53:47 and passed, but
by then the tree already held the rewritten version. **I never ran a
cell-level check against the 19:51:15 state**, so I cannot say the unit gate
would have missed it, and this note is not evidence that it would.

## Scale is the whole story

Against `WINDOW_2026-08-20_interleaved-254.md` (800k, same box, same day,
earlier commit):

    border  200k theirs  800k theirs  ratio    200k ours  800k ours  ratio
    ------  -----------  -----------  -------  ---------  ---------  -------
    254     8.9 ms/t     30.1 ms/t    3.4x     18.6 ms/t  35.4 ms/t  1.9x
    128     8.4          30.2         3.6x     15.3       29.4       1.9x

Their CPU arm scales with rows. **Ours does not**, because a large part of our
per-tree cost is row-independent control plane. So we read ~0.50x at 200k and
0.85x-to-1.03x at 800k, from the same code. Neither number is wrong; the small
one is simply not the regime the scoreboard quotes.

CAVEAT, and it matters: the 800k column is from an earlier commit in an
earlier window, so the two columns are not interleaved against each other and
the ratio is INDICATIVE ONLY. Pinning the fixed component honestly needs one
800k run in this same window at `14c1948`. It was not run.

## What this is NOT

- Our GPU against their CPU. `task_type="GPU"` raises on Apple, so CatBoost
  has no GPU arm on this box at all. That is the thesis, not a handicap we
  chose, but it belongs beside every row.
- Stable. Our arm read 25.2 -> 20.4 -> 18.6 ms/tree at 254 across three runs
  while their CPU arm stayed inside 8.8-9.9. The CPU canary being flat does
  NOT certify the GPU window on this box; warm-up and tree state both moved
  here, and they are confounded.
- covtype. Synthetic only. The covtype standing is untouched and stale.
