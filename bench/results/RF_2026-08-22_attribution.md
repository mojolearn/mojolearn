# Random Forest: every nanosecond gets a name, and the large-data answer is one kernel

2026-08-22 morning, M4 base. cuML pin v26.08.00. The directive for the
round: optimize for LARGE datasets, decisions grounded in cuML.

## The instrument this round built

The stock 'Metal System Trace' template records the Shader Timeline
instrument Disabled, `xctrace` exposes no way to enable it
(`--instrument 'GPU'` records nothing into `gpu-shader-profiler-interval`
-- measured), and MAX has no Metal debug-label API. So no trace on this
box could say WHICH kernel the time was in -- until the workload started
naming its own dispatches: every enqueue site in the forest path now
calls `core/launch_log.mojo`'s `log_launch` first, and with
`RF_LAUNCH_LOG=<path>` set, the names land one per line in enqueue
order. Metal's compute channel is a single in-order queue, so enqueue
order IS device order, and `profile_metal.py attr` joins line `i` to
merged Compute interval `i`. Disabled it is one `getenv` per launch, on
the path of every number already certified.

Two facts the join depends on, measured by
`ensemble/mojo_only/dispatch_census_probe.mojo` (prime counts
7 memsets / 11 H2D / 13 D2D / 5 D2H / 17 kernels, so the observed
Compute-row total decomposes uniquely):

* `enqueue_memset` emits NO Compute interval on this stack -- a host
  write on unified memory. Memsets are therefore NOT logged.
* Every `enqueue_copy` direction IS a Compute dispatch. Every copy site
  is therefore logged (`xfer_*`).

And one more, found by the join refusing an inexact match: MAX sometimes
expands one enqueue into several dispatches inside ONE command buffer
(28 of 9405 rows at the 500k shape); consecutive same-cmdbuffer-id rows
merge, and after the merge the counts matched the log EXACTLY, 9377 =
9377. A join that does not match exactly is refused outright -- one slip
mislabels every row after it.

## THE ATTRIBUTION, 500k x 50, one fit, n_streams=4

| site | ms | share | n |
|---|---|---|---|
| histogram_shared | 2668.1 | **85.3%** | 1200 |
| find_best_splits | 251.0 | 8.0% | 1200 |
| scan_by_key_block | 56.7 | 1.8% | 240 |
| sample_features | 42.1 | 1.3% | 240 |
| everything else (31 sites) | ~111 | ~3.6% | 7497 |

GPU busy fraction at this shape is 78.6% (vs 47.0% at 100k after the
pipelining round), so at large data the device, not the control plane,
holds the time -- and **the device time is one kernel**. cuML's
`computeSplitKernel` port (`histogram_shared`) is 85.3% of everything.
The quantile phase this lane once suspected is 0.2% (computed once per
forest, as theirs is). Optimizing anything but the histogram kernel at
large n is optimizing the margin of the page.

Caveat: shares, counts, and the join identity are the finding here.
Absolute times in the trace ran under a partially contended box and are
not certified numbers.

## The first lead, and what the box did to it

The kernel's ~780 lane-cycles per histogram increment pointed at
occupancy: DEVIATION 103a reserves a static 16 KiB of threadgroup memory
per block where cuML passes the EXACT dynamic size (~2.6 KiB at this
shape, `builder.cuh:526-533`, their comment: "stays small enough for
good occupancy"). The tier machinery now in
`launch_build_histograms_kernel` can launch a right-sized blob (2/4/8 KiB
tiers, tier visible in the launch log by name), and the fingerprint gate
held bit-identical through the tiered build -- all ten hashes, K4 twins
included.

The MEASUREMENT is still owed. Informal single-build reads suggested the
smaller blob is SLOWER on M4 -- the opposite of the CUDA intuition, and
interesting if true (a latency-bound gather kernel preferring FEWER
resident threadgroups) -- but those reads violated the alternate-inside-
one-window rule, and the one proper gated ABAB
(`rf_bench_smem16k` vs `rf_bench_smem4k`) was VOIDed at canary spread
**10.5x**: a peer lane's GBM benchmark (`gbm-bench`, higgs,
`mojolearn-gbdt-gpu`) held the GPU through the window, alongside
extratrees builds and mediaanalysisd. The tier list therefore SHIPS
DISABLED -- rule 11 flips defaults on measured wins, not on theories --
and the A/B pair binaries stay in `build/`. TPB was checked against
theirs while the box burned: 128 both sides.

## Ours vs scikit-learn: voids eight through ten, and what they hold

Three more gated attempts this morning, all voided: transient daemon
spikes + a cold first canary tick (fixed -- see below); the extratrees
lane's check suite waking mid-window; and a near-miss where the canary
PASSED at floor 1.15x but the VM and a VS Code renderer tripped the
process scan. Ten voids on record now, none quoted. The warmup fix that
came out of it: the window's first GPU work pays the idle->active clock
ramp (100.8/80.7/84.9 ms first ticks vs steady ~52-60, while rounds 2-3
-- equally fresh processes -- ticked steady), so `rf_bench` now runs a
discarded `warmup` tick before `pre`; the gate excludes `warmup` tags
from the spread only (still printed, still logged, still in the
node-count self-check).

## Still owed

- The 16k-vs-4k histogram smem ABAB, gated, on a quiet box. Then flip
  or close the tier list per the verdict.
- A clean ours-vs-sklearn window (ten voids; the box hosted a VM,
  Spotlight bursts, peer suites, and a peer GPU benchmark today).
- The eps500 timing row, by arrangement.
- NVIDIA parity: unchanged, the lane's largest gap.
- The full 17-check suite on the final source (fingerprint gate is
  green; the rest deferred so as not to contend the peer's GPU
  benchmark mid-run).
