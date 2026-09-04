# The FAST speed lane: what an ordinary user's arm costs against the vendor

**Status as of 2026-08-25: the harness is BUILT and NOTHING HAS BEEN
MEASURED.** No number appears anywhere in this directory. Every file here is
a specification until a leg runs, and the first leg's job is as much to find
out what compiles for CUDA as it is to produce a table.

## The question this lane exists to answer

Every other measurement in this repository is about the IDENTICAL path: what
the pin costs, whether the bits agree across vendors, which stage diverges.
This lane asks the opposite question and it is the one a user asks first.

**How fast is the arm you actually get, against the library you would
actually reach for, on the vendor's own silicon?**

So: FAST mode, which is the default build with no `-D
MOJOLEARN_NUMERIC_IDENTICAL=1`; non-deterministic by construction; measured
on a rented NVIDIA box; against cuBLAS, cuML, cuVS, PyTorch, CatBoost,
XGBoost and LightGBM running on that same box in that same hour.

**This lane makes no identity claim and quotes no ratio as a cost of
identity.** It records one bit-level fact and only one: whether the FAST
output moved between rounds. That is the direct evidence for what the
IDENTICAL mode buys, taken on the same silicon in the same run, and it is
reported rather than judged.

## What is here

| file | what it is |
|---|---|
| `gemm_speed_main.mojo` | ours, FAST, over all twenty rows of `bench/gemm_shapes.mojo` |
| `seq_speed_main.mojo` | ours, FAST, the Llama decoder layer and the Mamba block |
| `classical_speed_main.mojo` | ours, FAST, the classical lanes with a RAPIDS opponent |
| `forest_speed_arm.py` | ours, through the Python bindings, for boosting and forests |
| `../../tools/speed_gemm_arm.py` | cuBLAS, both `allow_tf32` settings |
| `../../tools/speed_torch_seq.py` | PyTorch and `mamba-ssm` |
| `../../tools/speed_cuml_arm.py` | cuML and cuVS |
| `../../tools/speed_gbdt_arm.py` | CatBoost, XGBoost, LightGBM, cuML forests, both devices each |
| `../../tools/fast_speed_table.py` | the parser and the ratio table, run at home over the fetched logs |

## The output contract

Every arm, in either language, prints these and nothing else that the parser
reads:

    FSPEED-HEADER  family= lane= arm= mode= device= rounds= size=
    FSPEED         lane= arm= shape= round= ms= hash=
    FSPEED-WARMUP  lane= arm= shape= ms=
    FSPEED-ACC     lane= arm= metric= value=
    FSPEED-AGREE   lane= max_abs_diff= max_rel_diff= n=
    FSPEED-WEIGHTS lane= arm= tensor= hash=
    FSPEED-NOTE    lane= arm= <free text>
    FSPEED-REFUSED lane= arm= reason=
    FSPEED-DONE    lane= arm=

`FSPEED-AGREE` and `FSPEED-WEIGHTS` are the two lines that decide whether a
row means anything at all. A speed number for an arm that computed something
different from its opponent is worthless, and a pair of arms that ran on
DIFFERENT WEIGHTS is not a comparison. Both are reported and neither is
gated: a large difference does not fail the run, it disqualifies the row, and
the row has to be readable in order to be disqualified. The table builder
gives `FSPEED-AGREE` its own section for that reason.

Any other `FSPEED-*` kind falls through to the notes section rather than
being dropped, so an arm may add one without silently losing its output.

`arm` is `ours` or the real library and device: `cublas-tf32`, `cuml-gpu`,
`sklearn-cpu`, `catboost-gpu`, `torch-gpu-fp32`, `mamba-ssm-cuda`. An arm
labelled `cuml-gpu` must be cuML on the GPU; the label is what makes the
table readable and a mislabelled arm is worse than a missing one.

`mode` is read from the COMPTIME constant, never from the environment and
never from the flag that was passed. The leg refuses any run in which an
`ours` header reports IDENTICAL, because that is a correctly-labelled
measurement of the wrong arm, and this repository was bitten by exactly that
three times on 2026-08-23.

## Three rules that decide whether the table means anything

**1. The warm-up is timed, printed, and never averaged in.** Torch pays an
enormous first call for autotuning and lazy module init; a reader who cannot
see that number cannot tell it from a cost.

**2. The statistic is the median and the min is printed beside it.** A rented
box is shared and throttled, and this repository has measured a box drifting
1.7x inside twenty minutes. When the median and the min are far apart the box
was busy and the row deserves less weight, which is a judgment the table
should let a reader make rather than making it for them.

**3. TF32 is the trap that ruins this whole lane if it is missed.** On
Ampere and later, cuBLAS may satisfy an FP32 matmul with TF32 tensor cores:
10 explicit mantissa bits instead of 23. Measured here on an H100 at
`llama8b.qkv.t512`, cuBLAS was 44.4 TFLOP/s with `allow_tf32=False` and 207.5
with it on. That is a factor of five and it is a precision cut, not an
optimization.

Which column is the fair opponent depends on which of our arms is being
compared:

* Our IDENTICAL kernel is strict FP32 by contract, so `cublas-fp32` is its
  opponent and comparing it to `cublas-tf32` would charge our contract for
  someone else's precision cut.
* **Our FAST path calls MAX's `linalg.matmul`, which measured 200 TFLOP/s on
  that same shape and box** -- matching the TF32 column, not the FP32 one.
  So the FAST arm's honest opponent is `cublas-tf32`.

This lane is the FAST arm. Both columns are measured, both are labelled, and
the table never picks one for the reader.

## Why it is three legs and not one

A one-hour lease is capped in code (`tools/runpod_guard.sh`, and
`--minutes` refuses above the cap by name). A cold box has to install pixi,
compile these drivers for CUDA for the first time in this repository's
history, and pip-install the vendor libraries. Measured elsewhere in this
repo: a full bootstrap took fifty minutes on an RTX 4090 and died on its own
work bound. That does not fit.

So the lane is three sequential legs, one family each, each independently
useful, and a failure loses a third rather than everything:

    tools/gemm_remote_leg.sh nvidia --payload speed --family gemmseq   --rent
    tools/gemm_remote_leg.sh nvidia --payload speed --family classical --rent
    tools/gemm_remote_leg.sh nvidia --payload speed --family forest    --rent

`gemmseq` is deliberately first and is the cheapest: torch is already in the
RunPod pytorch image, so it installs almost nothing and spends the whole
lease compiling and measuring. `classical` pays a RAPIDS install. `forest`
pays a LightGBM-from-source build for its CUDA arm, measured elsewhere at
fifteen to thirty minutes, which is why that build is bounded and attempted
last of the installs.

**The vendor arms run on the image's system Python and not under pixi**, and
that is a resolution rather than a preference: the prebuilt
`python/mojolearn/*.so` are built against Python 3.14 and RAPIDS ships for
3.10 through 3.13. Those two cannot share an interpreter, so cuML in the pixi
environment would fail to resolve and take the Mojo half of the leg down with
it.

## What this lane is expected to say, written down before it runs

Recorded here so the result is usable either way, in the spirit of
`archive/evidence/gemm/UNPINNED_CONTROL.md` -- whose own prediction was wrong on the low side,
which is exactly why writing it down first was worth doing.

* **GEMM: roughly parity with `cublas-tf32`**, because under FAST we are
  calling MAX's tuned matmul and it already matched that column on an H100.
* **The classical lanes: we lose to cuML, probably by a lot.** cuML is a
  mature CUDA-specific library and these are one-source ports whose only
  measured platform is an M4. The interesting number is HOW MUCH, per lane.
* **Boosting: we lose to CatBoost GPU.** The 94-launch-per-tree tax is
  already recorded against the Apple column and nothing about NVIDIA silicon
  removes it.
* **Transformer and Mamba: we lose badly**, and against flash attention or a
  fused `selective_scan_cuda` it will not be close. Our port is eager and
  unfused by construction.

**If any of these comes back a win on the first attempt, be suspicious rather
than pleased.** No line of most of these lanes has ever been compiled for
CUDA. The first run is a BUILD.

## What no leg here can say

* Nothing about Apple. A wall-clock number is a property of one box, and an
  M4 millisecond cannot enter a table of H100 milliseconds.
* Nothing about identity. That is the other lane and it uses a different
  build.
* Nothing about inference, only about the entry each driver times. Where a
  lane times `fit` and not `predict`, its own file says so.

## Findings this lane produced before it measured anything

**DEVIATION 1810 (fixed, `bench/bench_sklearn.py`).** Building the cuML arm
meant reading the existing scikit-learn arm against `bench_main.mojo` line by
line, and the k-means initial centroids did not match. `u01(rows, cols,
salt)` returns a `rows x cols` block indexed `0..rows-1`, and the centroids
were built as `u01(c * 7919 + 1, km_cols, 5)[0]` -- row ZERO for every `c`.
The `+ 1` is the tell: it exists to make row `c * 7919` the last row of the
block, and the subscript then asked for the first one. The Mojo side uses row
`c * 7919`.

So our k-means started from 64 distinct centroids and scikit-learn's started
from one centroid repeated 64 times. **Every k-means ratio in
`bench/results/` taken against that file has to be re-read rather than
believed**, and which direction it moved is not something that can be
asserted here: a degenerate initialization can converge instantly and flatter
them, or thrash on empty-cluster relocation and flatter us. It was not one
variable, so it was not a comparison.

Nothing was red. No check covered it. It was found by reading two files
against each other, which is the only thing that finds this class.

The lane also carries the deviations its own harness recorded: 1810-1829
(classical), 1830-1849 (forest), 1850-1869 (sequence), 1870 (the leg
payload). Each is written up in the markdown beside the file that records it.

## Owed investigations, from the first H100 numbers

Written down so they are not rediscovered. None of these is fixed.

**`rmsnorm` is 4.8x slower than torch at ONE token** (0.282 ms against
0.058 ms) and 18x at t512. `llama_rms_norm` is a single asynchronous kernel
with `grid_dim = (_grid(m), 1, 1)`, so at `m == 1` it launches ONE block onto
a 132-SM H100 and 131 of them idle. That explains a bad number; it does not
explain 280 microseconds for 4096 floats, which is roughly ten times what a
launch plus a synchronize costs. Something else is in that path and it has
not been profiled. Do not guess at it -- measure it on the box.

**The four TN rows are still 1.8x to 3.8x cuBLAS after DEVIATION 1873.** The
redundant transpose is gone, but ONE transpose remains, and cuBLAS does the
same product with none: it is column-major native, so a TN is an op flag
rather than a data movement. The remaining routes are a `transpose_a` that
MAX does not expose, or letting `core/gram_splitk.mojo` serve NVIDIA under
FAST. That second one is a KERNEL MATRIX row
(`gram_splitk_is_target_arm`), which is the sanctioned place to make a
vendor decision, and it is currently comptime-False off Apple on the
argument that MAX has its own split-K there. **That argument never accounted
for the transpose the vendor route has to pay first**, so it deserves to be
re-decided against a measurement rather than left as reasoning.

**`mamba-ssm` has never been the opponent.** The bounded install hit its
240s timeout, so every Mamba number so far is against the pure-PyTorch
reference scan, which is sequential in sequence length and is not what
anybody deploys. The fix is a prebuilt wheel matching the image's
(torch, CUDA, abi, python) tuple, or baking it into a pod image once,
outside a lease.

**`llama8b.lm_head.t1` cannot run at all** (DEVIATION 1872): MAX's gemv
kernel raises `CUDA_ERROR_INVALID_VALUE` at n = 128,256, which is past the
65,535 cap on a CUDA grid's y and z dimensions. That is the lm_head at batch
one -- the single most common operation in LLM decode -- and it is a limit in
the arm a FAST user gets. It is refused by row now rather than killing the
process, but it is not FIXED, and it is worth reporting upstream.
