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

Box: RunPod `u3g00x1o4wy6fo`, NVIDIA GeForce RTX 4090 (driver 580.159.04),
AMD EPYC 7K62, $0.74 per hour. BEFORE is `main` at `c85657041`; AFTER is this
branch at `5f94cd14a`. Small summaries are in
`bench/results/gbdt_resident_2026-09-17/`; the full pod output is in
`~/mojolearn-evidence/gbdt-resident/leg_out_tip/`.

THE FIRST RUN DID NOT COVER THE TIP. The session that built this lane was cut
off after pushing `5f94cd14a` (the threaded MultiClass and OneVsAll
transforms) to the pod: `gbdt/resident_model.mojo` on the pod was dated
18:14:22Z and the AFTER binding 17:54:53Z, so the identity and speed columns
taken at 18:03Z to 18:34Z ran the commit before it. They are kept as
`leg_out_pre_tip_1855Z/` and `speed_pre_tip_table.md`. Everything below is
the rerun after `build-after` rebuilt AFTER (`_mojolearn_gbdt.so` f90665ac,
was bff8559a) and the sabotage arm at 18:56Z to 19:00Z.

### Identity (tools/identity_break.py, 22 GBDT lanes, five fixtures, two repeats)

| diff | train | infer/model | batch |
|---|---|---|---|
| before-cuda vs after-cuda | IDENTICAL=110 | IDENTICAL=200, N/A=20 | IDENTICAL=105, N/A=5 |
| after-cuda vs sabotage-cuda | DIVERGENT=110 | DIVERGENT=105, IDENTICAL=95, N/A=20 | BATCH_MOVED=105, N/A=5 |
| before-cpu vs after-cpu | IDENTICAL=100, REFUSED=10 | IDENTICAL=150, N/A=10, NOT-COMPARED=20, REFUSED=40 | IDENTICAL=95, N/A=5, NOT-COMPARED=10 |
| after-cuda vs after-cpu | IDENTICAL=100, ONE-COLUMN=10 | IDENTICAL=150, N/A=20, ONE-COLUMN=50 | IDENTICAL=95, N/A=5, ONE-COLUMN=10 |
| before-cuda vs before-cpu | IDENTICAL=100, ONE-COLUMN=10 | IDENTICAL=150, N/A=20, ONE-COLUMN=50 | IDENTICAL=95, N/A=5, ONE-COLUMN=10 |

No cell moved. The CPU column refuses `gbdt-multiclass`, `gbdt-onevsall`,
`gbdt-categorical-ctr-tables` and `gbdt-tensor-ctr-tables` by name in BEFORE
and AFTER alike (CPU training does not cover them), so the evidence for the
threaded MultiClass and OneVsAll transforms is the CUDA pair: BEFORE runs
`multiclass_probabilities` and `one_vs_all_probabilities` themselves, AFTER
the per-row restatement, and all ten cells of each lane read IDENTICAL x2 on
train, infer, model and batch. The sabotage arm moves `predict` and `proba`
on every fixture of both lanes, so the resident door is what those cells
went through.

### Speed (bench/speed/gbdt_resident_ab.py, ms per call, seven rounds)

Resident against the per-call parse in ONE binary, interleaved, two
processes per cell; `u` marks a side outside the 1.10 spread gate. CatBoost
(the version is in `pip_catboost.log`) on the same box and rows, its GPU
evaluator and its CPU at all cores.

| model | path | calls | rows | per-call ms | resident ms | ratio | qualified | hashes equal | CatBoost GPU ms | CatBoost CPU ms |
|---|---|---|---|---|---|---|---|---|---|---|
| covtype-multiclass-100 | predict | 1 | 581012 | 304.8u | 25.5u | 11.96 | False | True | refused | 191.5u |
| covtype-multiclass-100 | predict | 8 | 581012 | 275.8u | 24.7 | 11.16 | False | True | refused | 174.8u |
| covtype-multiclass-100 | proba | 1 | 581012 | 353.1u | 28.6u | 12.35 | False | True | refused | 149.8u |
| covtype-multiclass-100 | proba | 8 | 581012 | 352.1u | 31.5u | 11.18 | False | True | refused | 159.5u |
| covtype-multiclass-1000 | predict | 1 | 581012 | 765.4u | 63.5u | 12.05 | False | True | refused | 352.7u |
| covtype-multiclass-1000 | predict | 8 | 581012 | 764.7 | 61.5 | 12.43 | True | True | refused | 352.1u |
| covtype-multiclass-1000 | proba | 1 | 581012 | 846.9u | 68.5 | 12.37 | False | True | refused | 345.9u |
| covtype-multiclass-1000 | proba | 8 | 581012 | 845.2 | 64.7 | 13.07 | True | True | refused | 352.1u |
| higgs-logloss-100 | predict | 1 | 1000000 | 225.8u | 17.0u | 13.30 | False | True | 183.7 | 78.4u |
| higgs-logloss-100 | predict | 8 | 1000000 | 181.1u | 16.1 | 11.22 | False | True | 186.4 | 81.1u |
| higgs-logloss-100 | proba | 1 | 1000000 | 262.7u | 18.0u | 14.63 | False | True | 190.5 | 91.6u |
| higgs-logloss-100 | proba | 8 | 1000000 | 241.2 | 17.4 | 13.88 | True | True | 192.9 | 88.2 |
| higgs-logloss-1000 | predict | 1 | 1000000 | 286.3u | 28.6u | 10.01 | False | True | 265.6 | 174.3u |
| higgs-logloss-1000 | predict | 8 | 1000000 | 274.9 | 27.5 | 9.99 | True | True | 267.9 | 191.8u |
| higgs-logloss-1000 | proba | 1 | 1000000 | 347.6u | 30.2u | 11.52 | False | True | 273.7 | 193.3u |
| higgs-logloss-1000 | proba | 8 | 1000000 | 340.0u | 29.6 | 11.50 | False | True | 273.9 | 201.7u |
| taxi-logloss-100 | predict | 1 | 1000000 | 124.0u | 10.5u | 11.75 | False | True | 168.0 | 67.6u |
| taxi-logloss-100 | predict | 8 | 1000000 | 110.1 | 10.0u | 11.03 | False | True | 164.1 | 64.9u |
| taxi-logloss-100 | proba | 1 | 1000000 | 172.5u | 13.1u | 13.18 | False | True | 171.9 | 83.9u |
| taxi-logloss-100 | proba | 8 | 1000000 | 181.2u | 11.7 | 15.46 | False | True | 173.4u | 73.9u |
| taxi-logloss-1000 | predict | 1 | 1000000 | 203.3u | 24.2u | 8.40 | False | True | 205.2 | 165.9u |
| taxi-logloss-1000 | predict | 8 | 1000000 | 204.8u | 21.6 | 9.46 | False | True | 225.9 | 160.2u |
| taxi-logloss-1000 | proba | 1 | 1000000 | 267.6u | 25.7u | 10.43 | False | True | 232.8u | 190.1u |
| taxi-logloss-1000 | proba | 8 | 1000000 | 263.3u | 24.2 | 10.86 | False | True | 212.6 | 166.6u |
| taxireg-rmse-100 | predict | 1 | 1000000 | 121.8u | 10.3u | 11.86 | False | True | 121.3 | 64.1u |
| taxireg-rmse-100 | predict | 8 | 1000000 | 108.0u | 9.6u | 11.30 | False | True | 121.8 | 62.5u |
| taxireg-rmse-1000 | predict | 1 | 1000000 | 205.8u | 22.7u | 9.07 | False | True | 207.7 | 151.1u |
| taxireg-rmse-1000 | predict | 8 | 1000000 | 201.0 | 21.0u | 9.55 | False | True | 206.7 | 159.5u |

Every one of the 28 cells hashed equal between the two arms. Four cells pass
the 1.10 gate on both sides (`covtype-multiclass-1000` predict and proba at
eight calls, `higgs-logloss-100` proba and `higgs-logloss-1000` predict at
eight calls); the per-call arm is the jittery side on this pod, and the ratios (9
to 14) sit far outside any spread in the table. The tip commit is what took
Covtype `predict_proba` from 119.7 to 28.6 ms (100 trees) and from 143.2 to
68.5 ms (1000 trees): before it the MultiClass transform ran on one thread
through two List copies. CatBoost's GPU evaluator refuses a
multi-dimensional model, so those rows have its CPU only.

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

- The Apple and AMD columns at the next release record. The resident door is
  compiled on every vendor and was run on NVIDIA only.
- The CPU column has no `gbdt-multiclass` or `gbdt-onevsall` cell (CPU
  training refuses both), so the two transforms' identity rests on the CUDA
  BEFORE and AFTER pair above.
- Most cells miss the 1.10 spread gate on the per-call side; a quieter box
  or more rounds would qualify them. No ratio is quoted as a claim.
- A non-symmetric ensemble (Depthwise, Lossguide) gains the cached parse and
  the resident compressed index but still applies per tree through
  `doc_parallel_boosting.mojo::predict`.
