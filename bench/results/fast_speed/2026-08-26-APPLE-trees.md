# Trees on the M4: symmetric, rf and et against the only opponents Apple has

MacBook Pro M4, 10 cores (4 performance). FAST on both sides. Shipped sizes,
three timed rounds after one untimed warm-up, ours and the opponent
ALTERNATING INSIDE ONE PROCESS -- which matters more here than anywhere
else, because this box has been measured drifting 1.7x inside twenty minutes
when heat pins the GPU governor. A box that throttles throttles BOTH arms;
the ratio survives what an absolute number does not. Run at `nice 19`, one
lane at a time.

**OUR ARM IS ON THE M4 GPU. THEIR ARMS ARE ON THE CPU, AND THAT IS THE ONLY
COMPARISON APPLE HAS.** CatBoost's `task_type="GPU"` raises on Apple silicon
and LightGBM ships no Metal learner, so there is no vendor GPU arm to run.
This is the case the standing GPU-path-only rule carves out: the vendors'
CPU path is legal HERE, on the MacBook, because it is the only path they
have. It is never legal on NVIDIA or AMD.

`cuml-rf-gpu`, `lightgbm-cpu` and `lightgbm-cuda` refused by name (not
installed / no Metal build), which is why the rf and et lanes show three
refusals each.

## The three lanes

| lane | dataset | ours | their CPU arm | | our accuracy | theirs |
|---|---|---|---|---|---|---|
| gbdt-symmetric | year 463,715 x 90 | **2061.0 ms** | catboost-cpu 2273.4 | **1.10x FASTER** | **RMSE 9.2159** | 9.2301 |
| et | covtype 522,911 x 54 | 11745.1 ms | sklearn-et-cpu 11242.9 | 1.04x slower | acc 0.6701 | **0.6768** |
| rf | covtype 522,911 x 54 | 19744.9 ms | sklearn-rf-cpu 9148.0 | **2.16x slower** | acc 0.7078 | **0.7095** |

Medians of three. Our hash was IDENTICAL across all three rounds in every
lane (`2868341a9401e697`, `d3b3fb332d0cd2a8`, `d17ccce2fde38b00`) -- the FAST
path happened to be run-to-run stable here, which it does not promise.

## The forest is ~2x off on BOTH vendors, against DIFFERENT opponents

This is the finding worth keeping, and it only appears when the two boxes are
put side by side:

| box | our rf | opponent | ratio |
|---|---|---|---|
| H100 | 6188.4 ms @ 1M higgs | cuml-rf-gpu 3079.5 | 2.01x slower |
| H100 | 14698.9 ms @ 5M higgs | cuml-rf-gpu 7057.5 | 2.08x slower |
| **M4** | **19744.9 ms @ covtype** | **sklearn-rf-cpu 9148.0** | **2.16x slower** |

Three different sizes, two different vendors, two different opponents --
NVIDIA's own GPU forest and scikit-learn's CPU forest -- and `ensemble/` is
between 2.0x and 2.2x behind every one of them. A gap that is invariant to
the box and to the opponent is not a hardware story and not a dispatch
story. It is our kernels doing about twice the work, and it is the clearest
optimization target this project has.

## Our own arm, M4 against H100

Same source, same dataset, same size, so this is a clean device ratio for
our code rather than a comparison of two libraries:

| lane | dataset | M4 | H100 | H100 is |
|---|---|---|---|---|
| gbdt-symmetric | year 463,715 | 2061.0 ms | 895.4 ms | 2.30x faster |
| rf | covtype 522,911 | 19744.9 ms | 5231.4 ms | 3.77x faster |

**THE H100 NUMBERS IN THIS TABLE ARE THE TRANSCRIBED ONES** from
`2026-08-25-nvidia-trees-PARTIAL.md`, read off a live box after the leg
script died before its fetch. They are not card-backed and this ratio
inherits that. The M4 column is card-backed.

## Accuracy, and where it is not fine

Symmetric boosting produces the BEST RMSE in its lane, as it did on NVIDIA.
The forests do not:

* `rf` 0.7078 against scikit-learn's 0.7095 -- 0.0018, small.
* `et` 0.6701 against scikit-learn's 0.6768 -- **0.0067, and that is the
  largest accuracy gap in any tree lane on any box.** It is not a timing
  question and it is not covered by anything above. `extratrees/`'s own
  parity work is the place to take it, and it is recorded here because a
  speed table that ignores a 0.0067 accuracy deficit is the exact failure
  this file's accuracy column exists to prevent.

## What this run does NOT say

* No HIGGS ladder on the Mac. `HIGGS.csv.gz` is present locally (2.6 GB) but
  a five-million-row fit on a laptop is the kind of load this project
  deliberately sends to a rented box.
* No `depthwise` or `lossguide` lane: XGBoost is installed here but those
  lanes were not part of this leg.
* No H100 `et` at covtype to compare against -- the leg that measured it is
  the one whose logs were destroyed.
