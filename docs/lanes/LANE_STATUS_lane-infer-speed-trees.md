# LANE STATUS: faster IDENTICAL inference for forests and GBDT (2026-09-17)

Branch `lane/infer-speed-trees`, cut from `main` at `e3213a59a`. Written for a
session with no memory. Numbers below are filled from the pod evidence under
`bench/results/infer_speed_trees_2026-09-17/` (small summaries) and
`~/mojolearn-evidence/infer-speed-trees/` (logs and JSON, outside the repo).

## What changed, and why no bit can move

Three deviations, each in the file it names. Every change alters WHEN and
WHERE the same arithmetic runs, never WHICH arithmetic.

| deviation | where | what | the invariant that pins the bits |
|---|---|---|---|
| 2900 | `core/forest_host_predict.mojo` | `rf_host_predict` and `et_host_predict` fan rows out to host threads (`MOJOLEARN_CPU_THREADS`, absent means one per physical core, `1` means the calling thread). The GPU `rf` and `trees` bindings' `sequential` entries now call these functions instead of carrying their own copies of the loop. Trees are rebuilt with the host lane's bounds checks; the input is one memcpy. | Each row's vote is zeroed, receives `predict_one` (RF) or `predict_one_accumulate` (ET) for every tree in increasing tree order, and is divided by `Float32(n_trees)`. `DecisionTree.predict` with one row IS `predict_one` at that row's offset. A thread owns whole rows, so no cell sees a different sequence of adds. |
| 2901 | `core/gbdt_host_predict.mojo` | One serial pass validates the model and packs per-feature, per-record and per-tree tables (every check and message of the old per-tree loops). Rows fan out to threads and are quantized and walked in 2048-row blocks with a block-local cursor. Non-decreasing border lists are bisected. | Per cell `(d, row)`: one float32 add per tree in tree order into a cursor seeded at `Float32(bias)`; the block-local cursor cell is the whole-row-set cursor cell. The bin is the count of borders the value EXCEEDS with the same `>` on the same float32 values; for a non-decreasing list that count is the length of the true prefix, which the bisection returns; a list that is not non-decreasing keeps the linear count. The compressed-index words are built by the same `build_layout`, mask and shift. |
| 2902 | `bindings/_mojolearn_gbdt.mojo`, `bindings/_mojolearn_forest_host.mojo`, `python/mojolearn/ensemble.py`, `python/mojolearn/_gbdt_host.py` | `gbdt_sigmoid_pair` / `forest_host_gbdt_sigmoid_pair` write both Logloss and CrossEntropy `predict_proba` columns in the binding; the Python layer uses them when the binary carries them and keeps the DEVIATION 2333 comprehension otherwise. | `p` is `gbdt_sigmoid`'s value (`1 / (1 + identical_exp64(-raw))` in double). `1.0 - p` is one IEEE double subtraction, the same operation Python performed per row; a lone subtraction has no fusion partner and no association. |

Not changed: the `parallel_groves` GPU engine, its fixed 32-group reduction,
the RF input FTZ rule, equality routes left, the GBDT GPU quantize and
per-tree launches (`gbdt/train.mojo`, `gbdt/methods/doc_parallel_boosting.mojo`
are not this lane's files), the model text format and archives.

Refusal behavior that moved (documented in the module docstring): a GBDT
model that is malformed AND fed a NaN on an `AsIs` column is refused for the
model before the NaN; the old loops refused the NaN first. Both are refusals.

## The sabotage control

The existing define `MOJOLEARN_FOREST_HOST_SABOTAGE=1` is the negative
control for every path this lane touched: the forest vote divides by
`n_trees + 1`, the GBDT cursor seeds at `bias + 1`, and (new) the two
probability columns are written swapped. It is passed to the host families
through `MOJOLEARN_BUILD_EXTRA_DEFINES` and to the GPU `rf`, `trees` and
`gbdt` bindings through `MOJOLEARN_EXTRA_DEFINES`. The read-back flags of the
`rf`, `trees` and `gbdt` CPU TRAINING families answer to
`MOJOLEARN_HOST_SABOTAGE`, not to this define, so a CPU column built this way
records `sabotage: false` for those three families and `true` for `forest`;
the column is a control, never a production column, and its JSON is kept
only under the evidence tree.

## Evidence

Box: RunPod `hhb4bs9mhcv6lt`, NVIDIA GeForce RTX 4090 (driver 580.159.04),
AMD Ryzen 9 7950X (16 cores), Mojo 1.0.0, $0.74 per hour, 13:42Z to 16:54Z,
about $2.37 (plus about $0.04 for a first pod whose 570 driver Mojo
refuses); reaped and verified gone (HTTP 404). BEFORE is `main` at
`e3213a59a`, AFTER is `411a69a96`; the later commits on the branch change
the harness, the CPU training GBDT family's exports and the docs, not the
timed paths. Summaries are committed under
`bench/results/infer_speed_trees_2026-09-17/`; every per-process JSON, log
and `.so` digest is under `~/mojolearn-evidence/infer-speed-trees/leg_out/`.

### Speed, taxi, 1,000,000 prediction rows, 16 threads

Qualified means every quoted process passed the 1.10 spread gate and every
process of both arms hashed the same output bytes.

| model | path | before ms | after ms | ratio |
|---|---|---|---|---|
| RF regressor, 100 trees, depth 16 | GPU class `predict` (sequential engine) | 38864 | 7570 | 5.13, qualified |
| RF regressor, 100 trees, depth 16 | `host_model().predict` | 35116 | 7393 | 4.75, qualified |
| ET regressor, 100 trees, depth 16 | GPU class `predict` (sequential engine) | 15257 | 4572 | 3.34, qualified on the stable processes; the other AFTER processes read 4518 to 5322 ms at spreads 1.12 to 1.27 |
| ET regressor, 100 trees, depth 16 | `host_model().predict` | 14827 | 4477 | 3.31, qualified |
| GBDT Logloss, 1000 iterations, depth 6 | `host_model().predict_proba` | 22883 | 1383 | 16.5, qualified |
| GBDT Logloss, 1000 iterations, depth 6 | `host_model().predict` | 24134, 24110 | 1339, 1372 | 17.8 from four stable processes in the log; the JSONs were overwritten (harness naming defect, fixed in `e08ce80cb`) |
| GBDT Logloss, 100 iterations, depth 6 | `host_model().predict` | 2733 | 199 | not qualified (AFTER spreads 1.11 to 1.40 at 200 ms); every AFTER round is below every BEFORE round |
| GBDT Logloss, 1000 iterations, depth 6 | GPU class `predict_proba` | 828 to 877 | 129 to 137 | not qualified (AFTER spreads 1.35 to 1.40); every AFTER round is below every BEFORE round |
| GBDT Logloss, 100 iterations, depth 6 | GPU class `predict_proba` | 784 | 86 | not qualified (AFTER spreads 1.48 to 1.54); every AFTER round is below every BEFORE round |
| GBDT Logloss, 1000 iterations, depth 6 | GPU class `predict` | 99 | 109 | not qualified, both arms spread 1.4 to 1.7 at 100 ms; unchanged path |
| GBDT Logloss, 1000 iterations | model text parse alone | 34 | 34 | unchanged; about a third of the 100 ms GPU `predict` call |

The 16-thread pod result is not the one-core Mac result; on one core the
forest gain is the removed per-row allocation and per-tree checks only, and
the GBDT gain is the loop interchange and the bisection. Neither was timed
on one core.

### Identity, six columns, 33 lanes, five fixtures, two repeats

| diff | result |
|---|---|
| before-cuda vs after-cuda | 280 infer and model cells IDENTICAL, 145 batch cells IDENTICAL, nothing moved |
| before-cuda-sw vs after-cuda-sw (the two score-weighted lanes, run after the metrics binding was built) | 10 IDENTICAL, nothing moved |
| before-cpu vs after-cpu | 210 infer and model cells IDENTICAL, 135 batch cells IDENTICAL, nothing moved; 10 cells ONE-COLUMN (below) |
| after-cuda vs after-cpu | 210 infer and model cells IDENTICAL, 125 batch cells IDENTICAL, nothing moved |
| after-cuda vs sabotage-cuda | 55 DIVERGENT: every rf and et lane (predict, proba, batch) and the two Logloss-proba gbdt lanes; the GPU-path gbdt lanes IDENTICAL, as the sabotage does not reach the device path |
| after-cuda-sw vs sabotage-cuda-sw | `rf-score-weighted` DIVERGENT on its four regression parts |
| after-cpu vs sabotage-cpu | 105 infer and model cells DIVERGENT and 125 batch cells DIVERGENT: every rf, et and gbdt lane the CPU column runs |

The CUDA columns skip `rf-clf-balanced-parallel` and
`et-reg-bootstrap-parallel`: their batch part hung the UNMODIFIED main tree
(90 minutes in `futex_wait`, 0 percent CPU and GPU) on the one-GPU box, so
it is the two-device batch protocol on this box and not this lane. The CPU
columns carry them (REFUSED by name on both, as on main).

The ten ONE-COLUMN cells: the AFTER CPU column REFUSED `gbdt-symmetric` and
`gbdt-pointwise-l2-bayesian-eval` because the CPU training GBDT family of
that build (`_mojolearn_gbdt_host`) did not export `gbdt_sigmoid_pair`, and
a CPU-only install's binding proxy raises ImportError by name. `e08ce80cb`
adds the entry point to that family and makes the Python layer treat the
ImportError as absence. Those ten cells are OWED a rerun of the CPU
column; on the CUDA column and through `host_model` (the AFTER CPU column's
`infer` cells of every other Logloss lane, and the Mac gate) the pair reads
IDENTICAL.

Mac, one core: `tools/forest_host_gate.py check` on the 24 recorded
fixtures under `bench/results/forest_host/2026-09-13-*` reads IDENTICAL
with the new `_mojolearn_forest_host.so` (`predict` and `predict_proba` of
every RF, ET and GBDT fixture, Apple and NVIDIA recordings alike).

## Commands

Pod (`tools/trees_leg.sh`, RTX 4090, `TREES_LEG_CUDA_VERSIONS=13.0` because
Mojo 1.0.0's GPU runtime refuses drivers below 580):

```sh
export MOJOLEARN_RUNPOD_KEY_FILE=$HOME/.mojolearn_runpod_key
export TREES_LEG_STATE=$HOME/mojolearn-evidence/infer-speed-trees/pod
TREES_LEG_NAME=mojolearn-infer-trees TREES_LEG_CUDA_VERSIONS=13.0 \
  MOJOLEARN_STAGE_KEYS="gbm-bench/taxi/taxi_speed.npz gbm-bench/istella/istella_speed.npz" \
  sh tools/trees_leg.sh rent --gpu "NVIDIA GeForce RTX 4090" --minutes 120
# ship main at the base commit to /root/mojolearn-before with a COMMIT file, then
sh tools/trees_leg.sh ssh 'cd /root/mojolearn && bash tools/infer_speed_trees_body.sh setup'
sh tools/trees_leg.sh ssh 'cd /root/mojolearn && bash tools/infer_speed_trees_body.sh identity'
sh tools/trees_leg.sh ssh 'cd /root/mojolearn && bash tools/infer_speed_trees_body.sh speed'
sh tools/trees_leg.sh pull /root/leg_out/ $HOME/mojolearn-evidence/infer-speed-trees/leg_out/
sh tools/trees_leg.sh reap
```

Mac, one core (the host inference door only, no Metal):

```sh
nice -n 19 env OMP_NUM_THREADS=1 MOJOLEARN_CPU_THREADS=1 MOJOLEARN_COMPILE_JOBS=1 MOJOLEARN_BUILD_JOBS=1 \
  sh bindings/build_forest_host.sh
nice -n 19 env OMP_NUM_THREADS=1 MOJOLEARN_CPU_THREADS=1 PYTHONPATH=python MOJOLEARN_NUMERIC_MODE=identical \
  python3 tools/forest_host_gate.py check bench/results/forest_host/2026-09-13-*
```

## Owed

- The Apple and AMD columns of the next release record. This lane rented
  NVIDIA only and ran no Metal.
- The `rf-clf-balanced-parallel` and `et-reg-bootstrap-parallel` lanes need
  two devices and read REFUSED on the one-GPU pod, on both columns.
- The GBDT GPU `predict` still parses the model text and creates a
  `DeviceContext` per call (`gbdt/estimator.mojo` header, policy 1). The
  parse alone is 34 ms of a 100 ms call for a 1000-tree model on the 4090;
  a parsed-model cache keyed on the text is the fix the header names and was
  not built here.
- A rerun of the CPU identity column at `e08ce80cb` or later for the ten
  `gbdt-symmetric` and `gbdt-pointwise-l2-bayesian-eval` cells above.
- One-core timings of the host paths (the Mac rule allowed no more than one
  core and the pod ran 16 threads).
- The GPU `predict_proba` and the 100-iteration host `predict` AFTER cells
  are faster on every round but too jittery at 85 to 200 ms for the 1.10
  gate; a larger batch or more rounds would qualify them.
