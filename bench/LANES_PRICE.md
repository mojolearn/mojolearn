# What IDENTICAL costs on the six frozen lanes: the price harness

Status: **CONSTRUCTION plus one Apple device's gates; no second vendor has
run this.** The harness is built and smoke-tested (both modes build, the mode
witness reads back, the hash is stable); **no clean-window measurement has
been taken yet and no timing number appears in this file.**

Files:

- `bench/lanes_price_main.mojo`, the driver: one lane per process, the lane's
  own public entry on the lane's own shipped fixture, ROUNDS timed rounds
  after one untimed warm-up, one `LPRICE` line per round.
- `tools/lanes_price.sh`, the window: builds the driver twice (FAST bare,
  IDENTICAL through `tools/with_identical_mode.sh`'s `mojo build` form),
  then per lane alternates the two binaries F I F I F I ... inside one
  window, reads the mode back from every leg, and writes the table.
- `bench/results/lanes_price/<timestamp>_<sha>[_SMOKE]/`, the evidence.

## What the harness measures

Each round is ONE call of the lane's public entry, the same call the lane's
`*_main.mojo` makes, on a fixture built once before the loop from the lane's
own fixture builder (imported, not re-spelled; the two transcriptions are
named in the driver's docstring). The clock is the host `perf_counter_ns`
around the call with a `ctx.synchronize()` on both sides, so a number is
wall seconds for the whole entry including its host work and its launches.
It is not device time and it is not a per-kernel number.

| lane | entry (what `*_main.mojo` calls) | shipped size (default) | SMOKE size | output hashed |
|---|---|---|---|---|
| cd | `solver.ported.solver.cd.cd_fit_traced` (Lasso, alpha 0.01, intercept, 1000 epochs, tol 1e-3) on `fixture_planted_sparse(n, d, 610)` | 2048 x 16 | 256 x 4 | coef, intercept, n_iter |
| kde | `kde.ported.kde.kde.score_samples` (gaussian, euclidean, weighted, h 2.75) on `train_fixture / query_fixture / weight_fixture` salt 1 | 1024 train x 256 query x 8 | 128 x 32 x 8 | the n_query log-densities |
| linkage | `hierarchy.ported.hierarchy.linkage.single_linkage` (pairwise, L2SqrtExpanded) on `FIX_BLOBS_DUPS` | 102 x 5, 3 clusters | `FIX_DUPS` 48 x 2 | children, labels, Boruvka rounds |
| svm | `svm.mojo_only.svc_check._run_device` (svc_fit + svc_predict decision + classes) on `F2.xor`, the card fixture | 240 x 2 rbf | same (no smaller fixture exists) | decision function, classes, n_support, b |
| metrics | every ported metric in `metrics_main.mojo`'s order on its hashed fixtures, timed as one pass | labels 2053, floats 2053, silhouette 521 x 4, trust 301 x 6 | 257 / 257 / 67 x 4 / 61 x 6 | every returned value by its bits plus the silhouette samples |
| gemm | `gemm.mojo_only.gemm_identical.identical_gemm` on one `bench/gemm_shapes.mojo` row | row 6 `kmeans.dist.4096x64x64` NT | row 4 `pca.transform.8192x4x4` | C |

Under FAST the same entry runs with its pins (`identical_mul_add`, `ftz`,
`identical_*`) compiled away; under IDENTICAL they are live. The two modes
are two binaries because `GLOBAL_NUMERIC_MODE` is comptime.

The hash is FNV-1a64 over the output bytes, the same function
`core/identity_trace.mojo` uses for the stage cards (byte at a time, little
endian), so a single-buffer lane's hash here equals its card's final-stage
hash where the card records that buffer whole. A price run is therefore also
a hash run: the driver prints `HASH-STABLE` or `HASH-MOVED` across its
in-process rounds and raises under IDENTICAL if the bytes moved; the script
repeats the comparison across legs and across the two modes.

## The mode witness

The driver prints the mode it COMPILED in, read from the comptime constant
(`_mode_name()`), in its header and on every `LPRICE` line. The script
asserts both against the mode the leg asked for and ABORTS the whole run
(exit 3) on a mismatch. Verified by sabotage: `MOJOLEARN_LANES_PRICE_SABOTAGE_SWAP=1`
hands the FAST binary to the IDENTICAL legs, and the run aborts with

    !! MODE MISMATCH (lane=kde round=1): asked for IDENTICAL,
    !! the binary's header says 'FAST'. ABORT. ...

(exit 3, observed 2026-08-23). The refusals were exercised too: with the
build lock held by another process the script exits 2 before building.

## How to run a CLEAN-WINDOW measurement

Nothing else on the GPU. The M4's GPU governor pins at MINIMUM clock under
heat (96% of an 11-second trace while 93.8% busy; up to 1.7x drift across
twenty minutes), so a FAST block followed by an IDENTICAL block measures
the drift, not the modes. The script alternates the two binaries per round
inside one window; that is the closest available thing to interleaving two
comptime modes, and the result is a BAND.

1. Confirm no other lane is running a leg (the script refuses if a `mojo`
   or `pixi` executable is running, or if another process holds
   `/tmp/cbsym-build.lock`; do not override with `ALLOW_BUSY` for a number).
2. Let the box idle a few minutes after any build.
3. Run, with the default five rounds and the shipped sizes:

       tools/lanes_price.sh

   (`MOJOLEARN_LANES_PRICE_ROUNDS=5` is the default; `MOJOLEARN_LANES_PRICE_LANES`
   narrows the lane list; the script holds the build lock for the whole
   run and writes `bench/results/lanes_price/<timestamp>_<sha>/`.)
4. Read `ratio.tsv`: per lane, the median IDENTICAL seconds over the median
   FAST seconds, with `ratio_min`/`ratio_max` the per-round paired ratios
   (IDENTICAL_r / FAST_r). Report the median WITH the min/max band, the
   date, the sha, the machine (`env.txt` records host, CPU, load average at
   both ends, and whether a concurrent process appeared), and this
   sentence: measured on one M4 laptop whose governor drifts under heat,
   alternated per round inside one window, a band and not a figure.
5. Check the hash columns: `fast!=ident bits` is expected wherever a pin
   changes a rounding; `IDENTICAL HASH MOVED across legs` is a FINDING to
   report, not a row to drop.

A run by hand of one lane, one mode, for a look at the lines:

    MOJOLEARN_LANES_PRICE_LANE=kde tools/with_build_lock.sh \
        pixi run mojo run -I . bench/lanes_price_main.mojo
    MOJOLEARN_LANES_PRICE_LANE=kde tools/with_identical_mode.sh \
        pixi run mojo run -I . bench/lanes_price_main.mojo

No pixi task is registered for any of this.

## The price table

Median IDENTICAL / median FAST per lane at the shipped size, 5 alternated
rounds, one clean window, Apple M4. **No number here yet; the first
clean-window run fills it.**

| lane | size | FAST median s | IDENTICAL median s | ratio (median) | ratio band (min .. max) | date / sha / host | note |
|---|---|---|---|---|---|---|---|
| cd | 2048 x 16 | | | | | | |
| kde | 1024 x 256 x 8 | | | | | | |
| linkage | blobs_dups 102 x 5 | | | | | | |
| svm | F2.xor 240 x 2 | | | | | | |
| metrics | 2053 / 2053 / 521x4 / 301x6 | | | | | | |
| gemm | kmeans.dist.4096x64x64 | | | | | | |

Thermal caveat to print beside every filled row: one M4 laptop, governor
drifts up to 1.7x under heat, FAST and IDENTICAL alternated per round inside
one window, band not figure. Nothing in this table ranks vendors or
certifies a timing.

## Hashes from the smoke run (tiny sizes; hashes are not timing)

`bench/results/lanes_price/2026-08-23_125704_e69db89_SMOKE/`, sha e69db89,
ROUNDS=1 (one warm-up plus one timed round per leg), `MOJOLEARN_LANES_PRICE_SMOKE=1`.
Every lane built in both modes, every leg's header and every `LPRICE` line
read back the mode it was asked for, and every lane's hash was stable across
its two in-process rounds in both modes.

| lane | SMOKE size | FAST hash | IDENTICAL hash | same bits? |
|---|---|---|---|---|
| cd | 256 x 4 | 3be34440e79b3131 | 99d86c334a5c05cc | no |
| kde | 128 x 32 x 8 | 61bb0e73dc92f3b3 | 61bb0e73dc92f3b3 | yes |
| linkage | dups_lattice 48 x 2 | af1408d790e1a446 | af1408d790e1a446 | yes |
| svm | F2.xor 240 x 2 | 2def386e7266a86a | 9bec403e03e1a2cc | no |
| metrics | lab257 flt257 sil67x4 tru61x6 | 826b40eee60c57ee | 3bc0cdcca4a035cd | no |
| gemm | pca.transform.8192x4x4 | 5e18d6d88ce1cafd | 5e18d6d88ce1cafd | yes |

"Same bits" at the smoke size says only that on this Apple device, at this
size, the pins were bit-inert for that lane (Apple's FAST codegen contracts
and flushes too; IDENTITY_PATHS row 9). It is not a statement about another
vendor and not a statement about the shipped size.

The smoke output's head (from the run's stdout):

    == lanes_price: rounds 1, lanes [cd kde linkage svm metrics gemm], smoke 1 -> bench/results/lanes_price/2026-08-23_125704_e69db89_SMOKE ==
    == build FAST -> .../bin/lanes_price.FAST ==
    == build IDENTICAL (tools/with_identical_mode.sh, mojo build form) -> .../bin/lanes_price.IDENTICAL ==
    == lane cd: 1 rounds, FAST then IDENTICAL per round ==
       round 1: HASH-STABLE cd FAST 2 rounds all 3be34440e79b3131
       round 1: HASH-STABLE cd IDENTICAL 2 rounds all 99d86c334a5c05cc
    ...
    SMOKE RUN: tiny sizes, 1 round(s). The seconds above are launch
    overhead and prove only that both binaries run, the mode reads back,
    and the hash is stable. NO NUMBER HERE IS A PRICE.

One thing the smoke found and the driver now handles: `cdFit` MUTATES `x`
and `labels` in place under `fit_intercept` and `postProcessData` does not
restore the bits exactly (documented in `solver/ported/solver/cd.mojo`), so
the first smoke printed a warm-up hash for cd that differed from every later
round. The driver re-uploads `x` and `y` before every round; that is a
property of the entry and of the harness, not a deviation.

## Deviations

None spent. The block 700-704 is reserved for this harness and stays unused
unless a harness change moves bits.

## Hand-off

None required. The harness reads and imports from the six frozen lanes and
edits none of them.
