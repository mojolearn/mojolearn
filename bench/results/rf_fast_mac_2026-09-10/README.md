# RandomForest on the Apple M4, 2026-09-10 night: DEVIATION 2500, 2501, 2502

HIGGS first 1,000,000 rows x 28 features, 100 trees, depth 16, sqrt
features, 128 bins, bootstrap, seed 7; `bench/speed/forest_speed_arm.py
--lane rf --ours-only`, one warm-up dropped, MAX 26.5 / Mojo 1.0.0, macOS
Metal. Wall time of `RandomForestClassifier.fit`. Runs were not interleaved
with each other and the M4 drifts with heat, so the numbers below are
attributions and directions, not a certifiable A/B; the H100 leg
(`bench/results/trees_identical/h100_2026-09-10b/`) carries the IDENTICAL
timing that counts against the opponent rows.

## Where a 17.5 s IDENTICAL fit went before tonight (`logs/stage.identical.before.log`)

| stage | s |
|---|---:|
| fit_total | 17.5 |
| device_wait (the one drain per pipeline cycle) | 4.3 |
| other (host, unattributed) | 12.8 |
| leaf_values | 0.23 |

On the H100 the same `other` was 0.49 s. The M4's host side was the fit.

## Attribution (`logs/stage.fast.*`, FAST tier, same forest hash as IDENTICAL)

- `logs/launch_counts.before.txt`: 99,730 enqueues per fit; 14,640
  histogram rounds for 3,640 node batches (setup, zero, histogram,
  best-split per round), 8 partition launches per batch.
- MAX Metal enqueue floor, `launch_floor.mojo`: 19.4 us per
  `enqueue_function`, 155 us per launch+synchronize, 126 us per small
  D2H copy+synchronize. `memset_floor.mojo`: a lone `enqueue_memset` is
  12 us (4 KB) to 41 us (4 MB), but memset+launch is 126 us and
  memset+2 launches 168 us against 58 us for three launches: a memset
  between kernel launches costs about 110 us extra; an H2D copy or a
  zero-fill kernel in the same position costs nothing extra.
- `logs/stage.fast.hist_memset_split.log`: the per-round histogram
  `enqueue_memset` was 9.48 s of a 14.4 s fit (650 us each, the region is
  up to 21 MB); the histogram launch 0.15 s and the best-split launch
  0.12 s beside it.

## DEVIATION 2501: histogram zero as a kernel (`core/device_zero.mojo`)

`logs/stage.fast.zero_kernel.log`: host_hist_zero 0.14 s (was 9.48),
fit 12.6 s, hash 3ffa2951595422d4 unchanged; device_wait rose 4.4 -> 11.0 s
because the GPU was no longer hidden behind the host stall. Three-round
confirmation with 2502 opted out, `logs/speed.fast.2501_only_optout2502.log`:
12,376 / 12,354 / 12,397 ms, hash 3ffa2951595422d4 all three, logloss
0.538850, AUC 0.809906 (the Sep 10 values). `core/device_zero_check.mojo`
holds the fill to exactly its span over 79 offset/length cases
(`pixi run check-device-zero`).

## GPU time per kernel (Metal System Trace of one FAST fit, 8,337 rounds captured)

| position in a histogram round | mean us | total s |
|---|---:|---:|
| phase_setup | 227 | 1.89 |
| zero | 59 | 0.49 |
| build_histograms | 458 | 3.82 |
| find_best_splits | 416 | 3.47 |

Compute 11.7 s over 42,579 encoders (MAX commits one command buffer per
launch); blits 0.09 s; GPU idle gaps 1.21 s. Not committed: the .trace is
426 MB.

## DEVIATION 2502: a pure node is a leaf (changes the forest)

`logs/rounds.fast.retry_counters.log` (scratch counters, one fit):

| round | items | rows |
|---|---:|---:|
| 0 | 2,536,425 | 1,594,189,711 |
| 1 | 530,696 | 4,912,937 |
| 2 | 527,498 | 4,881,293 |
| 3 | 526,816 | 4,874,599 |
| 4 | 526,614 | 4,872,578 |
| 5 | 526,532 | 4,871,699 |

Of the 530,696 nodes retried after round 0, 526,514 (99.2%) end the batch
with no split after every column was tried; 248,385 have four rows or
fewer. `logs/purity_audit.fast.log` reads the leaf vectors of every node
the kernel marked pure: 55k per tree, `impure 0` on every tree. The other
4,182 are pure nodes whose Gini gain is not exactly zero in the
objective's arithmetic and that "split" into two pure children on a
retry.

Result (`logs/speed.fast.2501_2502.log`, `logs/speed.identical.2501_2502.log`):

| tier | fit ms (7 or 3 rounds) | hash | logloss | AUC | histogram rounds |
|---|---:|---|---:|---:|---:|
| FAST, before tonight | 16,100 to 20,800 | 3ffa2951595422d4 | 0.538850 | 0.809906 | 14,640 |
| FAST, 2501 only | 12,354 to 12,397 | 3ffa2951595422d4 | 0.538850 | 0.809906 | 14,640 |
| FAST, 2501 + 2502 | 8,988 to 9,234 | efd14ab2c09ff57c | 0.538817 | 0.809830 | 3,627 |
| IDENTICAL, 2501 + 2502 | 8,867 to 8,904 | efd14ab2c09ff57c | 0.538817 | 0.809830 | 3,627 |

The forest changed: skipped pure-node splits shift every later node's
tree index, and the column sampler seeds on that index. Identity holds:
8 of 8 FAST fits and 3 of 3 IDENTICAL fits hash equal, FAST equals
IDENTICAL, `identity_break --lanes rf-clf,rf-reg` is stable x2 on 18 of
18 cells (`logs/identity_break.rf.txt`,
`bench/results/identity_break/apple-m4.identical.rf-2502-2026-09-10.json`):
rf-reg equals the retained Apple set 9 of 9 (regression never marks a
node); rf-clf moved on 9 of 9 fixtures, `predict` and `proba`, as this
deviation predicts. `-D MOJOLEARN_2502_RETRY_PURE=1` restores the previous
forests exactly (hash 3ffa2951595422d4). `rf_perf_candidates_check` ALL
ARMS GREEN under the IDENTICAL define (`logs/rf_perf_candidates_check.identical.log`).

A first cut stored the flag through `splits[nid].pure = x`, which lowers
to a struct-level read-modify-write and lost a concurrent mutex-guarded
publish about one fit in five (`logs/speed.fast.2502_struct_store_race*.log`:
hashes efd1.., d4a8.., 1b27..). The flag is now the struct's first word and
is stored through an Int32 pointer.

## DEVIATION 2500 in this fit (`logs/speed.identical.dev2500.log`, `before_dev2500`)

16,102 to 16,139 ms with the native label encoder against 19,417 to
20,848 ms without it, later in a hotter window; hash 3ffa2951595422d4 both.
