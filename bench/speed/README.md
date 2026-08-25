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

    FSPEED-HEADER family= lane= arm= mode= device= rounds= size=
    FSPEED         lane= arm= shape= round= ms= hash=
    FSPEED-WARMUP  lane= arm= shape= ms=
    FSPEED-ACC     lane= arm= metric= value=
    FSPEED-NOTE    lane= arm= <free text>
    FSPEED-REFUSED lane= arm= reason=

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
`gemm/UNPINNED_CONTROL.md` -- whose own prediction was wrong on the low side,
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
