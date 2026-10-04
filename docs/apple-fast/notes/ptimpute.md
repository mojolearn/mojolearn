# ptimpute: PowerTransformer and SimpleImputer on the Apple GPU (FAST), profile notes

Lane af-ptimpute, 2026-10-03, read from main 8897404da (no measurement in this lane; the M3 manager
measures). Board lanes: power-transformer (yeo-johnson, standardize=True) and simple-imputer
(strategy=median), datasets taxi (1,000,000 x 11) and istella (1,000,000 x 220). The board clocks
`fit()` (tools/bench_board_algos.py `fit`), transform is a separate inference clock, so every
number below is a fit.

M3 Ultra, docs/apple-fast/m3-gaps-0834.tsv, ms:

| lane | dataset | FAST | IDENTICAL | sklearn |
|---|---|---|---|---|
| power-transformer | taxi | 477 | 2,732 | 13,228 |
| power-transformer | istella | 2,957 | 433 | 276,767 |
| simple-imputer | taxi | 23.6 | 181 | 370 |
| simple-imputer | istella | 317 | 1,438 | 12,578 |

## 1. Why FAST is 7x slower than IDENTICAL on istella (power-transformer)

Both tiers run the SAME staged golden-section search on istella. `PowerTransformer.fit`
(python/mojolearn/_expansion_prep.py:2843) speculates only in IDENTICAL (`spec = _pt_spec_depth(n, d)
if mode == "identical" else 0`), and on istella `_pt_spec_depth` collapses to 0 anyway: the
candidates' transforms are capped at 2^28 words and 2 * n * d = 440M words exceed it
(_expansion_prep.py:2790-2803). So on istella both tiers stage `pt_init`, `pt_log`, then 50 x
(`pt_map` over n*d elements, `pt_fold` over d columns). The only difference is how `pt_fold` runs:

- IDENTICAL: `prep_kernel[pt_fold]`, one thread per column, 220 threads in two 128-thread blocks
  (x_prep/transform.mojo:430 `pt_fold_unit`). At K = 0 it reads X and T; at every later K it folds T
  ALONE: "a row is NaN exactly where its T word is" (transform.mojo:476-483), and when K = 0 counted
  n rows it does not even test (transform.mojo:466-473). T is column major (`pt_map_unit`,
  transform.mojo:380: T[c*n + i]), so each thread streams its own contiguous column, 16 rows a load
  (`run_block`), and the 220 lockstep threads read X at K = 0 as one row-major stream (consecutive
  threads = consecutive words of one row). Bandwidth bound.
- FAST: `pt_fold_fast_kernel` (x_prep/fastred.mojo:103), one 256-thread block per column, 220 blocks.
  Its first pass reads `f[X + i*dd + c]` for EVERY row at EVERY one of the 50 evaluations, only to test
  NaN (fastred.mojo:118-122), then reads T. Within a block the 256 threads read rows tid, tid+256, ...
  of ONE column: addresses d words apart, 880 bytes at d = 220, one cache line per word. The 220
  blocks are not in lockstep, so the lines are not shared through the cache: up to 220M line fetches
  (28 GB at 128 B lines) per evaluation, 50 times, for a NaN test the IDENTICAL unit does once. The
  second pass (squared deviations) reads T only, coalesced.

Scaling with d: the strided X read costs one line per element, so it is n*d line fetches per
evaluation. At taxi (d = 11, 44-byte rows) three rows share a line and the 11 blocks are tiny, so
the stream is small (11M lines) and the fold is latency bound instead (11 blocks x 256 threads, ~4k
dependent strided loads per thread, the GPU mostly idle); the whole fit is still 5.7x faster than
IDENTICAL's speculated search, whose `pt_sfold` runs 77 threads each walking 1M rows. At istella the
same kernel does 220M strided line fetches 50 times: that is the regression. Nothing else in the
staged program differs between the tiers except `col_stats` (FAST `col_stats_fast_kernel`, the same
one-block-per-column strided pattern, two passes, run twice per fit: on X and on the transformed
block).

## 2. PowerTransformer.fit, FAST on Apple, per fit (istella shape: n = 1M, d = 220, 880 MB per n*d block)

Program (one `x_prep_run_ranges` call, one arena, _expansion_prep.py:2831-2889):

| stage | launch | reads / writes | notes |
|---|---|---|---|
| upload X | ranges upload | 880 MB up | once |
| col_stats | 220 blocks x 256 | X twice, strided (2 x 220M lines) | fastred.mojo:28 |
| pt_init | 2 blocks (d threads) | state | trivial |
| pt_log | n*d threads | X read (coalesced), LG write column major (scattered, cache-merged) | once; the log kept for every evaluation |
| 50 x pt_map | n*d threads | X read, LG read (c*n+i: scattered), T write (c*n+i: scattered) | ~2.6 GB of useful words per evaluation |
| 50 x pt_fold | 220 blocks x 256 | X strided (220M lines), T twice (coalesced per block) | THE regression, section 1 |
| pt_apply | n*d threads | X read, LAM, TX write | the standardize tail |
| col_stats | 220 blocks x 256 | TX twice, strided | second strided col_stats |
| std_params | d threads | | trivial |
| download | ranges | lam, mean, scale (and every non-input arena word that is not an input) | |

Arena: X (n*d) + ST (6d) + LAM + STATE (10d) + LEVAL + TV (n*d) + LG (n*d) + MEAN + SCALE + TX (n*d)
+ ST2: four n*d blocks, ~3.5 GB of device words for a 880 MB input (the three working blocks are
device-only in effect but still part of `dev_len`, device.mojo:218). Launches: ~106 per fit, each
~0.2 ms of launch + wait cost on Apple on top of the work.

Live buffers per program (device.mojo:213-221): df (arena), dw (sort scratch, 1 word here), dq, dcg,
dre, dmw, dmu: seven. Not the problem here.

## 3. SimpleImputer.fit (strategy=median), FAST on Apple, per fit

Program (_expansion_prep.py:1593-1616): `sort_cols` (radix, x_prep/dradix.mojo: 4 passes of count +
scatter over the n*d block into the `work` scratch, 6 launch kinds), `col_stats` (the strided
two-pass `col_stats_fast_kernel`, 2 x 220M line fetches at istella), `quantile` (d threads, two
words each). Download: the stats rows. The sort dominates; the strided col_stats is the second cost
and the one this lane changes. lane/apple-fast-prep2 (unmerged, 364547564) already replaces the sort
for the median with a device radix select (`MOJOLEARN_X_PREP_FAST_QSELECT=1`, x_prep/fastprep2.mojo
`qselect_device`); this lane does not duplicate it. The imputer's `transform` is a separate program
(fill over n*dout, output region) and a separate board clock.

## 4. What this lane changes (every change FAST + Apple + its own define; docs/apple-fast/ab/ptimpute.md)

1. `MOJOLEARN_PT_FOLD_NOX`: `pt_fold_fast_kernel` reads X only at K = 0 (the unit's rule), later
   evaluations fold T alone. The minimal fix of section 1.
2. `MOJOLEARN_PT_COLBATCH`: the evaluation as ONE row-tiled grid: block = (row chunk, 32-column group),
   a thread owns one column over the chunk's rows (a simdgroup reads consecutive words of one row),
   computes the lambda-free log and the transform in registers, folds count / mean / M2 by Welford
   into per-(chunk, column) partials; a finish kernel (a block a column) merges the partials (Chan)
   and takes the golden step (`pt_finish`). No T block, no LG block, no `pt_log` / `pt_map` stage:
   one coalesced read of X per evaluation. Tile set {64, 256} rows picked by d at run time.
3. `MOJOLEARN_PT_SPEC`: COLBATCH's kernel over the 2^S - 1 speculated points of the IDENTICAL search
   (S = 3, 7 candidates a thread in registers): 18 passes over X instead of 50, `pt_spts` / `pt_sres`
   unchanged (the same points and decisions), no candidate buffer.
4. `MOJOLEARN_PT_FUSED_TRANSFORM`: the standardize tail's `pt_apply` + `col_stats` as one row-tiled
   pass (the transform in registers, Welford + min / max / maxabs partials, a finish kernel writes
   the six stats rows): no TX block, no strided col_stats.
5. `MOJOLEARN_SI_ONEPASS`: every `col_stats` stage as the row-tiled one-pass grid of (4): the
   imputer's statistics and the power transformer's opening col_stats. Replaces two strided passes
   with one coalesced pass.
6. `MOJOLEARN_PTIMPUTE_ALL`: 1 to 5 together (NOX is moot under COLBATCH, harmless).
