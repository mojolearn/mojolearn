# The 2M rung, on a different H100: the crossover is box-dependent and our 5M arm went superlinear

H100 80GB HBM3 (RunPod, a DIFFERENT host than the 2026-08-26 morning leg),
commit 374875a (pre-fix baseline: the speed-round-1 merge 6fd4621 is NOT in
this leg), FAST both sides, GPU arms only, HIGGS nested front prefixes
scored against the same fixed 500,000-row tail, three timed rounds after
one warm-up, alternated in one process. Artifacts:
`bench/results/e1g/2026-08-26_162556-nvidia-speed-forest/`, table
`2026-08-26_162556-nvidia-forest.md`.

## gbdt-symmetric vs catboost-gpu

| rows | ours | catboost-gpu | |
|---|---|---|---|
| 1,000,000 | 1073.1 ms (min 787) | 903.7 | 1.19x slower |
| 2,000,000 | 1346.7 ms | 1324.1 | 1.02x slower (tie) |
| 5,000,000 | 5515.4 ms | 2570.1 | 2.15x slower |

## rf vs cuml-rf-gpu

| rows | ours | cuml-rf-gpu | |
|---|---|---|---|
| 1,000,000 | 7077.1 | 3329.6 | 2.13x |
| 2,000,000 | 9788.9 | 4596.7 | 2.13x |
| 5,000,000 | 16572.0 | 7126.5 | 2.33x |

et ran ours-only (no legal opponent on NVIDIA): 4517.7 / 7306.0 / 15086.5 ms.
Accuracy is at parity in every cell that has an opponent (logloss matches
CatBoost to the third decimal at every rung, ours slightly better at 1M/2M).

## Finding 1: the morning leg's crossover number does not transfer across boxes

Same source (374875a differs from e07ddd5 only in the leg script), same GPU
SKU, and the 1M gbdt verdict flipped from **1.86x FASTER** (636.6 vs 1182.3)
to **1.19x slower** (1073.1 vs 903.7). Both arms moved: ours +69%, theirs
-24%. RunPod hosts differ in CPU allocation and our arm is host-prep-heavy
(the four-serial-copies bill, DEVIATION 1887's motivation) while CatBoost's
pool build is multithreaded C++ — a slower/contended host CPU punishes our
arm specifically. The rf ratios (device-bound, host-light) barely moved
(2.01/2.08 → 2.13/2.13), which corroborates the host-side attribution.
Consequence: **"crossover at ~3.55M rows" is a property of one box, not of
the code.** Cross-leg comparisons are for direction only; every number that
matters must be intra-leg.

## Finding 2: OUR arm is superlinear between 2M and 5M; CatBoost's is linear

Per-1M-row marginal from adjacent rungs, this box:

| | 1M→2M | 2M→5M |
|---|---|---|
| ours | 273.6 ms | **1389.6 ms** |
| catboost-gpu | 420.4 ms | 415.3 ms |

Theirs is a straight line; ours bends 5x between 2M and 5M. A linear
fixed+marginal model no longer describes our arm at this scale, and the
bend, not the ratio, is now the primary 1M+ finding. Candidate mechanisms,
in the order the recon ranked them: (a) the un-ported CUB radix sort —
CatBoost sorts leaves above 500,000 rows with `cub::DeviceRadixSort` where
we run the 3-launch block scan at every size, and the number of levels
holding a >500k leaf grows with rows (gbdt/gpu_util/kernel/
reorder_one_bit.mojo:9-16 records the refusal); (b) the density cliff
(indexed cindex loads at 140 MB working set vs 50 MB L2); (c) host memory
behavior of the prep chain at 5M. The level-resolved `sym.split` timers
already in the code are the probe.

## What this leg sets up

The speed-round-1 merge (6fd4621: prep copies cut, estimation pool + drain
cuts, relaxed atomics, FAST-dead memset gated) is the A arm; this leg is
the B baseline. The follow-up leg at 6fd4621 on the same ladder answers:
does the 1M gap close (prep fix), and does the 2M→5M bend survive (if yes,
the radix-sort port is the next campaign, DEVIATION number reserved).

## POST-FIX ADDENDUM (leg 2026-08-26_170628, commit 132d754, same day)

CatBoost's own arm moved <2% between the two boxes (995/1312/2544 vs
904/1324/2570 ms), so this pair of legs is comparable. Ours:

| rows | pre-fix | post-fix | verdict moved |
|---|---|---|---|
| 1M | 1073.1 (1.19x slower) | **668.1 (1.49x FASTER)** | -38% |
| 2M | 1346.7 (1.02x slower) | **1185.9 (1.11x FASTER)** | -12% |
| 5M | 5515.4 (2.15x slower) | **2760.2 (1.09x slower)** | -50% |

The 2M->5M superlinear bend is GONE: our marginal is 518/525 ms per 1M
rows across both intervals (linear), vs 274->1390 pre-fix. Marginal ratio
vs CatBoost: 1.57x -> ~1.26x. The crossover moved from ~1M on this box
class to past 4M. gbdt logloss 0.542247 at 1M -- unchanged to the sixth
decimal, as the bit-identity gates promised. rf 2.04/1.98/2.19x barely
moved: this leg predates the rf-port merge (0a8ba2f); the rf H100 A/B is
the owed next leg. Deviations credited: 1887 (host prep), 1890-1892
(estimation pool + drains + dead memset), 1898 (relaxed atomics).
