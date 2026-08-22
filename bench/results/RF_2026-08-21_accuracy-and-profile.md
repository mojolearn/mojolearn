# Random Forest: where the accuracy goes, where the time goes, and why there is still no timing

2026-08-21 evening, M4 base (10 GPU cores, 10 CPU cores, 16GB). cuML pin
v26.08.00. This file records two results that are NOT timings, and one
timing window that was VOIDED.

**Read the last section first if you are here for a speed number. There
isn't one.**

## 1. The accuracy gap against scikit-learn is the bin count, and the bin count is cuML's

Held-out accuracy, 100,000 x 50 synthetic, 4 classes, 20 trees, depth 12.
The label is `(x0>0.5) + (x1>0.5) + (x2>0.5)`, so the true boundaries are
axis-aligned at exactly 0.5 and a depth-3 tree can represent it perfectly.
Only `max_n_bins` varies. Deterministic, so this is load-independent and
was measured while the box was busy.

| `max_n_bins` | held-out acc | error | error x bins | nodes in tree 0 |
|---|---|---|---|---|
| 32 | 0.98066 | 0.01934 | 0.619 | 1053 |
| 64 | 0.99150 | 0.00850 | 0.544 | 531 |
| **128** (cuML's default) | **0.99534** | **0.00466** | 0.596 | 391 |
| 256 | 0.99749 | 0.00251 | 0.643 | 225 |
| 512 | 0.99738 | 0.00262 | 1.341 | 207 |

scikit-learn on the same bits: 0.99998 at 100k, 1.0 at 500k.

**`error x bins` is flat at ~0.6 across an 8x range of bin counts.** The
error is inversely proportional to the number of quantile candidates and
nothing else, and it stops improving past 256.

The mechanism is `quantiles.cuh`. cuML does not sort the data to find
thresholds. It draws `min(n_rows, max_n_bins * oversampling_factor)` rows --
`oversampling_factor` is hard-coded 4 at their call site, so **512 rows** at
the default -- sorts those, and takes `max_n_bins` order statistics. Those
are the only thresholds any tree may ever use. With the true boundary at
0.5, the nearest candidate is about `1/(4 * bins)` away, three signal
features compound, and the predicted floor `3/(4*128) = 0.0059` sits beside
a measured 0.0047.

`cfg_max_n_bins = 128` is their default (`decisiontree.hpp:85`) and it is
ours. **This is not a defect and must not be "fixed".** scikit-learn
searches exact thresholds; cuML trades that for the histogram. A CUDA box
running cuML would lose the same 0.47pp to scikit-learn. The knob is
exposed, it behaves monotonically, and a user who wants the accuracy back
pays for it in bins -- though note the node count FALLS as bins rise
(1053 -> 207), because better thresholds need fewer splits.

## 2. The GPU is idle 69% of a fit, and that is the finding

Apple Instruments, Metal System Trace, one fit at 100,000 x 50, 20 trees,
depth 12 (`ensemble/bench/profile_metal.py`). Device-side, unperturbed.

| | |
|---|---|
| compute dispatches | 9,377 |
| GPU busy (sum of intervals) | 813.5 ms |
| span, first to last | 2,595.6 ms |
| **GPU busy fraction** | **31.3%** |
| command buffer submissions | 11,609 |
| encoders per command buffer | mean 0.81, **max 1** |
| submissions carrying no encoder | 2,232 |
| dispatch duration | median 5.9 us, mean 86.8 us, max 3,014 us |
| the 200 longest dispatches | 38.1% of GPU busy time |
| the 1000 longest dispatches | 78.1% of GPU busy time |

**The cost is submission, not arithmetic.** One encoder per command buffer
means nothing is batched, and on Metal every commit is a trip through the
driver. cuML's design is written against CUDA, where a stream launch is
nearly free; that assumption does not survive the port, and it is a
platform difference rather than a porting error -- a kernel-matrix row.

This is what says DO NOT START OPTIMIZING KERNELS. Two thirds of the fit is
the GPU waiting for work, and cuML's kernels are deliberately plain -- not
one `__ldg`, `__restrict__` or `__launch_bounds__` in the whole
batched-levelalgo directory, and a single `#pragma unroll`. Rewriting a
kernel body cannot recover idle time, and it would stop being their
algorithm. Batching encoders does neither.

WHAT THIS PROFILE CANNOT SAY: which kernel is hot. Attribution needs the
Shader Timeline instrument and the stock template records it Disabled; our
encoders carry no debug labels, so the 21 loaded pipelines cannot be joined
to the durations. Naming the hot kernel needs a custom template or
`setLabel` on the Metal objects. Section 1's numbers point away from the
kernels regardless.

## 3. The timing window was VOID, and this is the second one

Attempted 21:14-21:20 through `ensemble/bench/quiet_window.py`, which took
the bench lock for the life of the window, sampled the box 154 times, and
refused the result. It printed:

    rf@100000   3119.30   3342.88   1.07x  INDISTINGUISHABLE
    rf@500000   7986.12  15844.16   1.98x  ours faster
    VOID: 17 competing process(es); the canary moved

**Those two ratios are not results.** Recorded here only so that nobody
re-derives them later and believes them.

What was actually on the box, from the gate's own sampling:

  * the **extratrees lane**, `/tmp/wiremat.py`, at ~100% in 139 of 154
    samples -- a source-rewriting script from another session;
  * the **gbdt lane**, `mojo run mojo_only/fit_pointwise_check` and
    `pointwise_pool_*`, at 205-221%;
  * macOS `XProtectRemediator` malware scans at 67-95%, `corespotlightd`
    at 160%, the Docker VM at 281%.

**The bench lock did not prevent any of this and could not have.** It
governs TIMING windows. Both peers were running CHECKS, which no rule says
to serialize, and they were entitled to. That is a coordination gap for
Andrew, not a bug: either checks take the lock too, or timing on this box
only ever happens by arrangement.

### The canary, and the resolution floor it sets

A fixed fit -- 20,000 x 8, 4 trees, depth 6 -- brackets every arm. Its node
count was 79 at all six ticks, which proves it did constant work. Its
duration was not constant:

    pre 110.1  mid 100.7  post 100.5  |  pre 96.6  mid 115.3  post 123.2 ms

**1.276x on identical work.** That is this machine, and it is consistent
with the standing note that it drifts 1.7x in twenty minutes.

The gate originally voided above a hand-picked 1.15x. That threshold was
wrong in a way worth recording: it was chosen before any measurement, and
the first honest window exceeded it. Raising it to fit the result would be
the exact move this project forbids. Instead the constant is gone. **The
canary's measured spread IS the resolution floor for its own window**, so a
window that drifted 1.28x cannot certify any ratio between 0.78x and 1.28x,
and a quieter window earns a sharper floor automatically. There is no
longer a setting that could be tuned until a result passes. A separate
bound at 1.5x remains for gross contamination only.

Under that floor, `rf@100000`'s 1.07x would have been indistinguishable
even on a clean box -- which run_bench.py's own overlapping-ranges test
also said, independently.

## 4. Still true, and unchanged by any of the above

**Nothing here has ever been compared against cuML.** Their RF is CUDA-only
and this is an M4; scikit-learn is in the harness because it is the
strongest thing that RUNS here, not because it is the reference. cuML
remains the algorithm source and the correctness oracle. A real head-to-head
needs a rented NVIDIA box and `bench/results/` still holds no such file.
