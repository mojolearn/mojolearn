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

FILLED BELOW FROM THE POD RUN.

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
  `DeviceContext` per call (`gbdt/estimator.mojo` header, policy 1). See the
  parse measurement below for its share; a parsed-model cache keyed on the
  text is the fix the header names and was not built here.
