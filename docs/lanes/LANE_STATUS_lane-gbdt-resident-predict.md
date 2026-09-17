# LANE STATUS: device-resident GBDT predict (2026-09-17)

Branch `lane/gbdt-resident-predict`, cut from `main` at `c85657041`. Written
for a session with no memory. Numbers come from the pod evidence under
`bench/results/gbdt_resident_2026-09-17/` (small summaries, committed) and
`~/mojolearn-evidence/gbdt-resident/` (logs and JSON, outside the repo).

## What changed, and why no bit can move

One deviation, DEVIATION 2980, in `gbdt/resident_model.mojo`,
`bindings/_mojolearn_gbdt.mojo` and `python/mojolearn/ensemble.py`.

`gbdt/estimator.mojo`'s header (policy choice 1) said the model crosses the
CPython boundary as its own text and that this costs a parse per `predict`
call, and named the fix, a cached parse keyed on the text. lane/infer-speed-
trees measured the cost on an RTX 4090: 34 ms of a 100 ms `predict` over
1,000,000 taxi rows for a 1000-tree Logloss model, the rest a `DeviceContext`
per call, three host copies of the input, a pinned staging ring per feature
and the pack and upload of every tree's split records and leaf values.

The first `predict` or `predict_proba` after a fit or a load now hands the
text to `gbdt_resident_prepare`, which parses it once, packs the oblivious
ensemble once, uploads the pack and every feature's border list once, and
keeps them in a process-wide `_Global` registry (the FOREST-RESIDENT-1 and
DEVIATION 2921 shape) under an integer handle the Python instance holds as
`_resident = (binding, text, sha256, handle)`. Every later call goes through
`gbdt_resident_predict` with the handle.

| what | statement |
|---|---|
| the compressed index | the same `binarize_float_feature_kernel` launch per bordered feature with the same offset, mask, shift and border list over the same float32 values after the same NaN substitution; features without borders are skipped as `_build_cindex_from_floats` skips them; the refusal for a NaN on an `AsIs` column names the lowest such feature, as the serial column-order scan did |
| the apply | the cursor filled with `Float32(bias)`, one `compute_bins_and_add_kernel` launch per oblivious tree in tree order, back to back on one stream, the same grid; the split records and leaf values are the bytes `doc_parallel_boosting.mojo::predict` packs, packed once. A non-symmetric ensemble (Depthwise, Lossguide) is applied by that `predict` function itself over the resident compressed index |
| the transforms | `multiclass_probabilities` and `one_vs_all_probabilities` from `gbdt/train.mojo`; the Logloss and CrossEntropy pair is `1 / (1 + identical_exp64(-r))` and `1 - p` in double over the exact widening of the float32 raw value, `gbdt_sigmoid_pair`'s two statements (DEVIATION 2902), now written by the binding as float64 straight from the readback |
| what moves out of the call | the parse, the `DeviceContext`, the six pack uploads, the border uploads, the per-feature staging ring and its drain per revolution; the whole input goes up in one copy, the whole cursor comes back in one, and the call waits on the stream once |
| what is retained | an exact-size per-row set (pinned input staging, device input, compressed index, cursor, pinned readback) while the row count repeats, FOREST-IO-REUSE-1's rule; a new row count releases the set first |
| the input layout | a 2-D float32 C-order block is a zero-copy borrow (`_buffer.as_f32_forest_layout`, DEVIATION 2637's shape) and the staging pass transposes it; every other input takes `as_f32_colmajor` as before. The staging pass and the Logloss pair fan out to host threads (`MOJOLEARN_CPU_THREADS`, the forests' `host_worker_count`); staging moves bytes and substitutes NaNs, the pair is per row, so no bit depends on the thread count |

Lifetime rules, `neighbors.py`'s for the k-NN index: the copy is keyed on
the text's sha256 (the fast check is `str` identity, a different object is
hashed and compared), released at the top of `fit`, released when the
instance is collected, never carried through pickle or deepcopy
(`__getstate__`), and a handle is meaningful only for the binding (tier,
vendor) that minted it. `load` builds a fresh instance and so prepares its
own copy at its first call. The adapters' `set_params` rebuilds the adapter
and drops its learner, whose `__del__` releases.

The old per-call entries `gbdt_predict` and `gbdt_predict_multi` stay for
older bindings and for the verifier. `MOJOLEARN_GBDT_RESIDENT=0` in the
environment at import, or `mojolearn.ensemble.GBDT_RESIDENT = False` at run
time, takes the per-call parse in the same binary; the speed harness flips
it to interleave the two arms in one process. A CPU-only install's binding
proxy raises ImportError for the absent entry point, which the wrapper reads
as "no residency here".

Not changed: `gbdt/train.mojo`, `gbdt/methods/`, the kernels, the model
text format and archives, `core/gbdt_host_predict.mojo` and the CPU
inference door.

## The sabotage control

`-D MOJOLEARN_GBDT_RESIDENT_SABOTAGE=1` (default off, passed to
`bindings/build_gbdt.sh` through `MOJOLEARN_EXTRA_DEFINES`) adds 1.0 to row 0
of plane 0 of the readback before any transform, so a build with it moves
every infer and proba cell that goes through the resident door and nothing
that does not. The resident pair path also honors
`MOJOLEARN_FOREST_HOST_SABOTAGE` (the swapped columns of DEVIATION 2902's
control) so that column keeps its reach over the GPU proba cells.

## Evidence

EVIDENCE_PLACEHOLDER

## Commands

Pod (`tools/trees_leg.sh`, RTX 4090, `TREES_LEG_CUDA_VERSIONS=13.0` because
Mojo 1.0.0's GPU runtime refuses drivers below 580):

```sh
export MOJOLEARN_RUNPOD_KEY_FILE=$HOME/.mojolearn_runpod_key
export TREES_LEG_STATE=$HOME/mojolearn-evidence/gbdt-resident/pod
TREES_LEG_NAME=mojolearn-gbdt-resident TREES_LEG_CUDA_VERSIONS=13.0 \
  MOJOLEARN_STAGE_KEYS="gbm-bench/taxi/taxi_speed.npz gbm-bench/higgs/higgs_speed.npz gbm-bench/covtype/covtype_speed.npz" \
  sh tools/trees_leg.sh rent --gpu "NVIDIA GeForce RTX 4090" --minutes 150
# ship main at the base commit to /root/mojolearn-before with a COMMIT file, then
sh tools/trees_leg.sh ssh 'cd /root/mojolearn && bash tools/gbdt_resident_body.sh setup'
sh tools/trees_leg.sh ssh 'cd /root/mojolearn && bash tools/gbdt_resident_body.sh identity'
sh tools/trees_leg.sh ssh 'cd /root/mojolearn && bash tools/gbdt_resident_body.sh speed'
sh tools/trees_leg.sh ssh 'cd /root/mojolearn && bash tools/gbdt_resident_body.sh catboost'
sh tools/trees_leg.sh ssh 'cd /root/mojolearn && bash tools/gbdt_resident_body.sh summarize'
sh tools/trees_leg.sh pull /root/leg_out/ $HOME/mojolearn-evidence/gbdt-resident/leg_out/
sh tools/trees_leg.sh reap
```

The HIGGS and Covtype keys were staged from a checkout whose
`bench/results/dataset_store/manifest.tsv` lists them (`main` at
`f3bff5be4`); this branch's base predates that row, so `rent` staged taxi
only and `tools/dataset_store.sh stage` added the other two.

## Owed

OWED_PLACEHOLDER
