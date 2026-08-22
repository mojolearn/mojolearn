# Random Forest: the forest loop learns to overlap, and the round finally gets its number

2026-08-21, near midnight, M4 base. Commit `617ee6b` (DEVIATION 117
ported) on top of `af91bae` (DEVIATION 313 + the memset fix + the
sampler syncs). cuML pin v26.08.00.

## The finding that drove it

cuML's SHIPPED forest loop is not serial: `#pragma omp parallel for
num_threads(n_streams)` over trees, each thread on its own CUDA stream
with its own workspace and its own `selected_rows_[stream_id]`
(`randomforest.cuh:336-367`), and the Python default is **n_streams=4**
(`randomforestclassifier.py:94`). The serial port had mirrored their
crippled non-OpenMP `#else` build. LightGBM was checked for anything
better and has nothing to offer here: its RF mode is bagging at the
boosting layer, one tree per sequential iteration.

The port: K-way PIPELINING over Metal's one queue. `doSplit` is cut at
its two sync points (`builder.cuh:479-481`, `:501-502`) into
`BatchState`/`TreeState` machines; K trees enqueue their next phase and
ONE synchronize per cycle serves all of them; `train`/`do_split` remain
the serial K=1 drive, operation for operation. Free on OUTPUT because
their RNG is a pure hash of `(seed, treeid[, nodeid])`, each in-flight
tree owns its Builder workspace (pool-of-K) and row-buffer slot, and the
only shared device objects are read-only.

**Gates.** `fingerprint_probe.mojo` fits five configs (Philox bootstrap,
OOB, no-bootstrap, regression, depth-14) at K=1 AND K=4: all ten hashes
identical. The sabotage that aliases every slot to row buffer 0 moves
every K4 line while the K1 lines stand -- the gate watches the
concurrency itself, not just totals. All 17 ensemble checks green;
`predict_check`'s three serial-contract assertions were falsified by the
port and now assert the new contract (accept 4, pass 4 through, clamp
only to n_trees per their `:585`).

## THE MEASUREMENT -- a CLEAN gated window, pre-round vs post-round

`quiet_window.py run` over an A/B/A/B of the two binaries
(`rf_bench_pre` built from a detached worktree at `f28b134`, before any
of this round; `rf_bench` at head). Process scan clean across 49
samples; canary floor **1.18x** (per-binary groups -- see below); both
rows outside it, ranges disjoint:

| arm | pre-round ms (median) | post-round ms | speedup |
|---|---|---|---|
| rf@100000 (100k x 50, 20 trees, depth 12) | 1967.31 [1938-2058] | 878.14 [860-902] | **2.24x** |
| rf@500000 (500k x 50) | 3954.62 [3912-4112] | 2789.01 [2783-2801] | **1.42x** |
| the fixed 20k x 8 canary fit | ~80-94 | ~49-58 | **~1.54x** |

Identical forests on both sides (same SUM digests, same ACC, same node
counts -- and the fingerprint gate held bit-exact through every step),
so this is the same computation arriving sooner. Consistent across four
windows this night (2.24-2.26x / 1.40-1.43x, disjoint ranges every
time); the first three were VOIDed by the gate for peer load, my own
probe, and Spotlight respectively, and are not quoted.

Where it came from, per the census (deterministic counts, same fixture):
dispatches unchanged at ~9.4k (the algorithm's own launches), trace span
2,595.6 -> 1,240.9 ms, GPU busy fraction 31.3% -> 47.0%. The memset fix
dominates the small-fit gain (the canary's 1.54x); the pipeline
dominates at scale, where the host's between-batch thinking now overlaps
three other trees' device work.

## The gate itself got two fixes, both earned by this round

- **Per-binary canary groups.** An A/B across two BUILDS changes the
  canary's own code; pooling both builds' ticks read the post build's
  1.5x-faster canary as machine noise and voided a window whose
  per-binary spreads were ~1.25x. `canary_verdict` now groups ticks by a
  `binary:` tag prefix, takes the WORST group's spread as the floor, and
  prints the cross-group ratio as a CODE finding to record. (First
  reading of that cross-group ratio was backwards -- called a K=4
  regression -- and the tagged rerun corrected it to the memset fix's
  small-fit WIN. Recorded so the wrong version does not resurface.)
- **Stdout wins over the canary log** when the driver passes canary
  lines through; reading both double-counted every tick.

## Ours vs scikit-learn, same night, NOT yet quotable

FIVE interleaved windows ran between 22:39 and 23:16 and every one was
VOIDed by the gate -- all on macOS nightly-maintenance daemons
(Spotlight, the Docker VM, duetexpertd, mediaanalysisd), never on a
peer lane. Across the five, ours-faster ratios of **2.40-3.15x at 100k
and 4.13-4.64x at 500k**, each individually outside its window's floor;
NONE is quoted, per the gate. What the five DO establish without a
clean window: our GPU arm's absolute times are stable to ~1%
(874.1-884.9 ms at 100k in all five) while sklearn's CPU arm swung
2102-2785 ms under the same daemons -- background CPU load is what
keeps voiding the windows, and it lands almost entirely on the CPU
arm. The clean run belongs to a daytime window, after the nightly
maintenance is done. For scale: the pre-round binary measured
INDISTINGUISHABLE from sklearn at 100k in the first (voided) window of
the night -- the round turned parity into a multiple whatever the
final digit turns out to be.

Morning of Aug 22, two more attempts, VOIDs six and seven. 07:19:
corespotlightd + an Apple Virtualization VM + a VS Code renderer
spiked past 130%, and the canary's FIRST tick (100.8 ms vs a steady
~55) blew the spread to 2.19x -- the window opened under interactive
Chrome/WindowServer load. 07:29: the PEER LANES woke up --
`oracle_main.mojo` at 367% peak across 27/150 samples, plus
`plan_fusion_check` and `sibling_tiebreak_*`, plus mediaanalysisd at
201%. Ratios printed 2.53x/3.86x and 2.65x/4.81x, our arm 881/877 ms
(the same ~1% band as all five nightly voids); none quoted. Next
attempt goes through a quiet-box watcher that refuses to open a
window until peers and daemons have been silent for two sustained
minutes, instead of burning attempts blind.

On eps500 (real data, 400k x 500): held-out accuracy **ours 0.6728 vs
sklearn 0.673275** -- identical to three decimals, so the 0.67 is what
forests do on epsilon, not a port defect. sklearn's single fit took
~12.9 MINUTES against our ~40-63 s on the same (contended) box;
calibration only, not a result -- the gated eps window needs ~45 quiet
minutes because sklearn's arm is that slow, and is deferred to a by-
arrangement window.

## Still owed

- A clean ours-vs-sklearn window (seven voids on record; watcher-gated
  retries in progress Aug 22 morning).
- The eps500 interleaved row, by arrangement (sklearn's arm alone is
  ~13 min/fit).
- NVIDIA parity: unchanged, nothing here has ever been compared against
  cuML itself, and it remains the lane's largest gap.
