# Native entry points after 0.8.66: the owed performance work as a program

Andrew, 2026-11-06: the performance work that the fix lanes recorded as OWED is done AFTER release 0.8.66, as a
planned program. "About 300 rows are recorded as owed. Most of them need new native entry points, for example
keeping more estimators' fitted state on the device, native training sessions, and cross-validation folds staying
on the device."

This document is the plan. The row-level detail is in `native_entry_points_after_0866/owed.tsv`, produced by
`native_entry_points_after_0866/collect_owed.py` from the lane tables; `dropped.tsv` lists the rows removed as
already closed, `overrides.tsv` the hand-assigned workstreams, `counts.txt` the totals. Row ids below (`N001`)
are the ids in `owed.tsv`; `file:line` are on `integ/perf-20261105` as the lane tables recorded them (line
numbers drift; the symbol names do not).

Standing rules this plan is written under (CLAUDE.md, AGENTS.md): lanes write code and never compile; one
compile per integration round; ONE verification round per release on the built wheels; no same-bits checks
during development; no CPU-route timing; no benchmark-shape rules; bits may change between versions but must
match across NVIDIA, AMD and Apple within one version, with the host column changed together.

## 1. The inventory

How it was collected. `collect_owed.py` reads every table the fix lanes left in `~/mojolearn-evidence/`
(`perf2-*.md`, `perf3-*.md`, `pass2-*.md`, `p3-*.md`; 36 tables, four rounds in that order) and keeps every
table row whose status cell says `OWED`, `OWED-SHARED`, `partly` / `partial`, or `LEFT` with wording that names a
new entry, signature, ABI or kernel change. Rows of the third-pass tables carry the rank of
`ideas-pass3-from-scratch.tsv` (964 rows), so kind, frequency, public methods and the original proposal are
joined by rank; earlier-round rows are joined by file and line overlap where one exists, else kind and frequency
are inferred from the row's wording (`kind_source` column says which). Rows reached from two tables for the same
site are merged (`also_in` column). Then every row that named another lane's file (OWED-SHARED) and every
earlier-round row was checked against the code on `integ/perf-20261105`; the ones already done are in
`dropped.tsv` with the file:line that closes them.

Counts (from `counts.txt`):

| | count |
|---|---:|
| rows with an open status in the 36 lane tables | 523 |
| merged as the same site reached from two tables | 8 |
| dropped as already closed in the code (`dropped.tsv`) | 55 |
| **owed rows kept (`owed.tsv`)** | **460** |
| by status: OWED / OWED-SHARED / partly / LEFT needing a new entry | 338 / 90 / 22 / 10 |
| by kind: 1 per-item Python / 2 per-call Python / 3 redundant native work / 4 training loop / 5 cache hazard / 6 compile | 29 / 120 / 244 / 60 / 3 / 4 |
| by frequency: per-item / per-call / per-fit / rare | 126 / 292 / 36 / 6 |
| by round of the table: p3 / perf2 / pass2 / perf3 | 426 / 22 / 7 / 5 |
| by workstream: A / B / C / D / E / F / G / H / J / K / P | 46 / 41 / 87 / 30 / 85 / 12 / 111 / 37 / 4 / 3 / 4 |

The "about 300" estimate counted the `p3-*` rows marked OWED; the real figure with OWED-SHARED, partial and
new-entry LEFT rows from all four rounds, after de-duplication and after removing what later lanes closed, is 460.

Dropped as already closed: 55 rows (`dropped.tsv`, each with the closing file:line): 34 from `perf2-*`, 11 from
`pass2-*`, 3 from `perf3-*`, 7 from `p3-*`. The groups: the eight `perf2-labels.md` label-door rows (closed by
`perf3-label-doors` and the `p3-*` owners, e.g. `gp_public_lifecycle_pyapi.mojo:111-141`); seven
`pass2-pass2-shared-table.md` keep-alive and table-form rows (e.g. `sequence_array_native.mojo:89-96`); the
dlsym-per-call rows of `perf2-cluster`, `perf2-neural`, `perf2-pca-decomp`, `perf2-shared` (closed by the shared
tables, e.g. `native_class_admission.mojo:131-137`); `classical_pandas` `_cached_value_types` (`:169`); the
`perf2-perf2-prep` rows closed by `p3-prep`; `perf2-trees` admitted-X and meta-record rows; the sequence
resident-params row (`sequence/mlp_fit.mojo:986-1003`); the `qn_proba` and `ols_predict_wide` registrations; two
`mode_metadata_native` overloads; the k-NN registry lock; the ridge-gram shard recipe (now tsvd-gram); the
light-checks workflow line. 22 further rows came back PARTLY done; they stay in `owed.tsv` with the remaining part
in the `code_check` column.

What the inventory says. Three quarters of the open rows are kind 2 and 3 at per-call frequency: the fix lanes
removed the per-item Python work they could reach from inside one file, and what is left is work that crosses an
ownership line: a kernel that must take a pointer instead of a List, a device entry that must leave its result on
the device, a session that must hold admitted words between steps, a fold that must be a device handle rather
than a host Array. That is why the workstreams below are organised by the entry point that is missing, not by
file.

## 2. Workstreams

Each workstream is one design with one owner lane (or two with disjoint files), so that the entries it adds are
coherent across vendors and the host column. "Proof" names the identity lanes that already exercise the path (the
ONE verification round per release covers them) and the small checks a lane adds to `tools/` for the release
gates; nothing is run during development.

### A. Device-resident fitted state per estimator family (A-resident-state, 46 rows)

Goal. An estimator that was fitted on a device keeps its fitted arrays on that device for the life of the model,
so `predict`, `transform`, `score_samples` and `decision_function` upload only X. The scheme exists for KMeans
(`cluster/resident_centers.mojo`), k-NN / KDE (`neighbors/resident_index.mojo`), KRR (`kernel_methods/krr_resident.mojo`),
sequence models (`sequence/perf3_seq_resident.mojo`), PCA / PLS operands and Conv2d weights; this workstream gives
every remaining family the same handle and the same rules.

Rows N001-N046, plus the resident rows filed under G by their file. Forest and tree export handles N007, N036,
N038, N044 (host rf rebuilds trees per call), N012; DART flat forest and stacked ensemble predict N003, N011, N039,
N041; GBDT predict reading X twice N005, N006, N042, N046; SpectralClustering predict N326 and its one-program
affinity N397; random projection N350; IPCA running state N310; ALS and TruncatedSVD components N325, N351; SVC and
GP predict pointers N376, N363, N365, N384; ARIMA, ExponentialSmoothing, forecast and Prophet fitted state N028-N032;
MiniBatchKMeans partial state N417; k-NN / KDE / ANN / IVF / cluster registries N014, N020-N026; the sequence store's
lock and free-at-death N172; IsolationForest refit on every `fast` score N385, N390, N392, N411; host (CPU column)
twins N012, N044, N441, N442.

Design.
- One side table per family, keyed by the model's identity (the object address plus the per-model generation
  word of workstream E), holding a `ResidentHandle`: device id, the uploaded fitted buffers, their shapes and item
  codes, the address and shape of the public array each one mirrors, and a byte count. The handle is a Mojo
  struct in a `_Global` registry inside the family's own binding (never in the model's `__dict__`).
- Read-only rule (decided 2026-11-06, `bindings/p3nc_fitted_freeze.mojo`): when the resident copy is taken, every
  public fitted array it mirrors that is an exact Mojolearn `Array` is made read-only at the buffer level, so an
  in-place write through NumPy raises NumPy's read-only error. Replacing the attribute re-uploads (the key changes).
  Storage that cannot be frozen (a caller-assigned ndarray) keeps the per-call upload. No content hash anywhere.
- Invalidation: `fit`, `set_params`, `__setstate__`, `load`, and every attribute setter the family owns bump the
  generation word; a handle whose generation differs is released on the next call. Model death releases the handle
  (weakref callback already used by `native_prepared_owner`).
- Memory: a per-device byte cap (default a fixed fraction of the device's memory, read once at connection; the
  sequence store's 64 blocks / 1 GiB is the model) with least-recently-used eviction; an evicted model re-uploads
  on its next call. A `mojolearn.resident_stats()` style readout is for tests only.
- Numerics: no bit change. The device copy is the same bytes the host array holds; the kernels read it where
  they read the per-call upload today. The host column keeps its own in-process copies (the pointer forms in
  N012, N044, N441 remove a parse or rebuild per call, not a fold order).

Files and bindings. `bindings/trees_public_{forest,boost}.mojo`, `_mojolearn_{rf,trees,x_trees,forest_host}`,
`cfb_spectral`, `rp_native`, `ipca_native` with `x_decomp_ipca_owner_pyapi`, `als_native`, `tsvd_native`,
`_mojolearn_{gp,svm}`, `tsa_{arima,hw}_native`, `xseq_{forecast,prophet}_native`, `cluster_fit_a_native`,
`sequence/perf3_seq_resident.mojo`. Bindings to build: trees, rf, forest host, svm, gp, x_decomp, x_cluster,
x_sequence and their host twins.

Depends on E (the generation word and the one record per model). Risks: a fitted array replaced through a path the
family does not own (a bridge `.py` file assigning a NumPy array) stays on the per-call upload; the cap must be
measured once on a box (profile item 14 asks how many watcher slots and how much memory the stores take).

Proof. Identity lanes: `rf-*` (7), `et-*` (6), `trees-*` (25), `gbdt-*` (36), `gp-*` (7), `svc*`, `kmeans-*` (6),
`x-decomp-als`, `tsvd`, `x-decomp-ipca`, `arima-*` (3), `sequence-*` (3), `spectral-*` (3), `iforest-*`. New small
checks: one `tools/` test per family that fits, predicts, writes a fitted array in place (expects the read-only
error), replaces it (expects a new upload), and deletes the model (expects the handle count to fall).

### B. Folds, members and rounds stay on the device (B-device-rows, 41 rows)

Goal. Cross-validation, search, permutation tests, Bagging / AdaBoost / Stacking / Voting members, RFE and
IterativeImputer rounds, and the multi-output score never gather a fold or a member's input on the host.
`_arena_io.DeviceRows` (base Array plus int64 row indices; `python/mojolearn/_arena_io.py:147`) is already what
model selection hands to an estimator whose class says `_mojolearn_device_rows = True`; today one class does
(`_expansion_prep.py:1527`), so every other estimator makes `msel_cv.mojo:303-353` gather and download the fold
(N058, N087).

Rows N047-N087. Fold rows N058, N087, N051, N057; search and permutation N048, N055, N056, N086, N070; members
N049, N060, N071-N076, N081-N085; the DART and AdaBoost native fit loops N061, N062, N063, N064; RFE and
IterativeImputer N059 (with N010, N034, N035 under A); resample kernels N053, N078, N052, N054, N047, N065; HDBSCAN
pointer forms N077; scorer and splitter records N066-N069.

Design.
- A `DeviceRows` entry on every native estimator's `fit`, `predict`, `predict_proba`, `transform` and `score`:
  the door accepts either a host array or a DeviceRows object (exact class check), and the native side resolves a
  DeviceRows to (device handle of `base`, device handle of `indices`, row count) with no host words. The kernel
  gathers rows on the device where it reads X today (the gather kernel exists in model selection's resident
  take). One native split plan per CV call keeps every fold's indices on the device (N051).
- Member calls: a meta-estimator door passes its own admitted X (the `AdmittedCall.inputs` view of E) and, for
  classifiers, asks the member for codes (`trees_forest_predict_codes`, N075/N076) instead of decoded labels; the
  parent decodes once against its own `classes_`.
- Rounds: RFE and IterativeImputer keep X resident across rounds and hand the round's column subset as a
  DeviceRows-like column view (N059); the DART and AdaBoost fit loops become one native loop with X, y and weights
  resident (N061-N064).
- Parallel CV workers receive X and y once and fold indices per request (N057).
- Numerics: no change in `identical`: the same rows reach the same kernel in the same order; the fold order of a
  member's reduction is unchanged. A DeviceRows path and a host-array path must produce the same bits on every
  vendor, which the existing `cross-val`, `cross-val-folds` and `x-metrics-search` lanes check.

Files. `bindings/msel_cv.mojo`, `msel_search.mojo`, `msel_split.mojo`, `parallel_cv_native.mojo`,
`hotpath_device.mojo` (fold helpers), `trees_public_meta*.mojo`, `trees_bagging_members_native.mojo`,
`trees_public_boost.mojo`, `prep_select_native.mojo`, `prep_iterative_*`, `_mojolearn_resample.mojo`, the
`resample_*_pyapi.mojo` providers, plus one door change per estimator family (the DeviceRows branch). Bindings:
model_selection, trees, prep, resample, and every family whose door gains the branch.

Depends on A (resident members) and E (the admitted input view). Risk: the DeviceRows branch touches every
family's door, so it is the one workstream that must compile with everything; land the entry in the shared door
helper (`classical_check_x` / `family_public_lifecycle`) once, not per family.

Proof. `cross-val`, `cross-val-folds`, `x-metrics-search`, `x-metrics-splitters`, `x-metrics-permutation-test-score`,
`x-prep-rfe`, `x-prep-iterative-imputer`, `x-prep-iterative-options`, `trees-stacking-clf`, `trees-voting-clf`,
`trees-calibrated`, `bootstrap`, `resample-*`. New check: a tools test that runs `cross_val_score` on one
estimator per family with a DeviceRows-aware door and with the host path and compares the digests.

### C. Native training sessions (C-training-sessions, 87 rows)

Goal. A training loop admits its model, optimizer and plan once, keeps weights, gradients and optimizer state on
the device between steps, and crosses into Python once per step (or once per epoch) with scalars. Today the byte
LM step re-reads shape and parameter lists per step (N107, N109), Samba re-walks its admission (N121, N126), the
small MLP and Sequential models upload parameters per step (N131, N132), the CNN epoch loop is driven from Python
(N099), and ParallelNeuralTrainer workers re-send state per update (N134).

Rows N088-N174: byte LM N091, N092, N100-N109; Samba N094, N121-N126, N129, N130, N133, N166-N169; SmallMLP /
Sequential N123, N127, N128, N131, N132, N172; CNN N090, N095-N099, N135-N138, N145-N151; neural blocks, transformer,
embedding N088, N112-N117, N158-N165; ParallelNeuralTrainer and the cross-vendor protocol N093, N110, N111, N120,
N134; autograd N089, N141-N143, N167, N170, N171.

Design.
- A `TrainingSession` handle per trainer kind (`byte_lm_session_open(model, optimizer, plan) -> handle`,
  `session_step(handle, batch_handle, n_steps) -> loss words`, `session_state(handle)`, `session_close`). The
  session owns the admitted configuration words, the device weights, gradients and optimizer state, the loss
  pin and the flags; `train_step` by default returns loss and scalars only, gradients on explicit request
  (decision 2026-11-06). A multi-step entry runs `n_steps` steps on device-resident batches and returns once
  (N109, N107, N101).
- Batches: the corpus iterator's `TokenBatches` handle gathers straight into the step buffer (N105); the caller's
  own buffer is pinned for the step lifetime (N102); ragged lengths go into the kernel (N091, N092).
- Public state: `parameters()`, `state_dict()` and checkpoints download on request and mark the session dirty on
  `load_state` (N133); block weights written in place through `parameters()` cannot be frozen (N159), so the
  session's generation word is bumped by the Python callers that expose them (N120).
- Vendor and host column: the session holds the same fold order on NVIDIA, AMD and Apple; the host twin
  (`byte_lm_host_trainer_native`, `_mojolearn_x_cnn_host`) gets the same session struct with host buffers, so
  the identity lanes compare the device column against the host column as today. No bit change by design; the
  per-step digest stays as the cross-vendor witness in the parallel trainer (N110).

Files. `bindings/byte_lm_{trainer,host_trainer}_native.mojo`, `_mojolearn_byte_lm`, `training/` (new provider
op), `samba_*_native`, `samba_execution_pyapi`, `small_mlp_native`, `sequence/opt_public_device.mojo`,
`sequence_optimizer_native`, `_mojolearn_x_cnn(_host)`, `xcnn_*_native`, `neural_block_public`,
`_mojolearn_transformer`, `_mojolearn_embedding(_host)`, `parallel_neural_native`, `_mojolearn_training` with
`python/mojolearn/_training_impl.py`. Bindings: byte_lm, training, samba, x_sequence, x_cnn, transformer,
embedding, parallel neural, and their host twins.

Depends on E (one record per trainer) and D (the generation side of the same sessions). Risks: the session ABI
replaces a stateless contract (N101); the worker protocol change (N134, N057) is the largest uncompiled change in
the program and goes in its own lane; the autograd frontend's 60-branch chain is a compile-memory suspect
(compile-fix-2 item 3) and gets integer op codes first (N089).

Proof. `byte-lm*` (5), `samba*` (3), `mamba*`, `transformer*`, `mlp*`, `batchgrad`, `batchscale`, `stepfull`,
`steps*`, `optim*`, `hf-causal-lm`, the CNN and sequence lanes. New check: one tools test per session kind that
runs two steps through the session entry and two through the old per-step entry and compares the loss words.

### D. Generation and decode sessions (D-generation-sessions, 30 rows)

Goal. A kept decode session owns a stream's KV state and its admitted block, state and configuration words; the
engine writes its output into the caller's buffer; the greedy pick runs on the device for ragged lengths; no
per-token download and no per-step re-admission.

Rows N175-N204. Session state and KV N198, N177, N186, N190, N188, N181, N178; engine output into the caller's
address N199, N201; device greedy pick N185 (with N091 under C); per-layer loop and worker spans N184, N183, N175,
N196; weight records N200, N179, N180, N187, N189; admission per call N193, N195, N202, N203, N191, N182, N176,
N194, N197; checkpoints N204; first-use cost N192 (profile 11 first, with N457).

Design. `decode_session_open(model, config) -> handle` admits once and allocates the KV buffers; `decode_step(handle,
tokens_in, out_address, lengths)` runs embedding, all layers and the head in one native loop with reused scratch
(ping-pong ownership of the activation, N184), writes logits or the picked token into `out_address`, and keeps the
KV state in the handle with a dirty flag per stream; `decode_close`. The parallel causal path sends one request per
contiguous worker span with the activation resident on the worker (N183). Host twins mirror the handle with host
buffers. No numeric change: the same kernels in the same order; the device greedy pick must equal the host pick
on every vendor (tie rule: lowest index), which `transformer*`, `mamba*` and `hf-causal-lm` check.

Files. `bindings/causal_lm_public.mojo`, `transformer_decode_pyapi.mojo`, `transformer/native_generation_engine.mojo`,
`generation_buffers.mojo`, `generation_metadata_pyapi.mojo`, `parallel_causal_native.mojo`,
`mamba_public_native.mojo`, `_mojolearn_mamba(.mojo/_host)`, `_mojolearn_neural_host.mojo`, `neural_prefill_native.mojo`,
`python/mojolearn/models/causal_lm.py`, `_mamba_impl.py`. Bindings: transformer, mamba, neural host, parallel causal.

Depends on C (the same session struct and generation word) and E. Risk: route change N181 (provider route as the
default) needs its own A/B on the box after the compile.

Proof. `transformer`, `transformer-window`, `mamba1`, `mamba2*`, `mamba3`, `hf-causal-lm`, `hf-checkpoint`,
`byte-lm-host-infer*`, `saved-model-host-infer`. New check: a tools test that decodes 8 tokens through a kept
session and through 8 single-step calls and compares the token ids.

### E. One admission record per call: `AdmittedCall`, one record helper, one CPython table (E-admitted-call, 85 rows)

Goal. A public call admits its modules, binding, model fields and inputs once, in the door, into one Mojo struct
passed by reference to every helper (`~/mojolearn-evidence/ideas-cpu-per-call-cost.md`, Part 2 (b)); the several
per-family record helpers become users of one native side-table module with one generation word per model; the
four CPython function tables become one.

Rows N205-N289 (and N050, N071, N320 for the trees and GP generation stamps): record helpers and generation stamps
N225, N230, N238, N241-N245, N253, N255, N278, N283; per-call module and binding resolution N219, N232, N236, N240,
N256, N257, N262, N265, N267; CPython table folds N066, N067, N144, N197, N260, N264, N287; the second admission of
X at family doors N237, N254, N266, N269, N272, N275, N279-N281; Python-side descriptors and operation codes N205,
N206, N217, N221, N222, N228, N251; the 26 calls in an unproven form (plan item A on main).

Design.
- `AdmittedCall` (stack struct, never stored on a Python object): `modules` (borrowed builtins, sys, NumPy, the
  buffer module and the type objects, filled by one stamp), `binding` (the provider from F with mode and capability
  words), `fields` (the model's instance dict admitted once plus a lazily filled fixed array of fitted fields by
  index), `inputs` (per argument: address, shape, strides, item code, read-only flag, sparse code, owner),
  `layout_checked`. Built in `classical_c` and the family doors; a helper that needs something not in it adds a
  field; a source guard refuses `native_loaded_module(` outside the doors and the record constructor.
- One record module `bindings/native_model_records.mojo`: a weak side table keyed by model identity with a
  generation word, one slot per family (trees, cluster, gp/svm, msel plans, prepared registry, arrays, guards)
  replacing `trees_side_records`, `cluster_side_records`, `gp_svm_records`, `msel_plan_records`,
  `p3_array_arena_records`, `p3_guard_records`; fixes the `msel_plan_records` one-entry-per-model leak. The
  generation word is bumped by `fit`, `set_params`, `__setstate__` and the family's setters (the `__setattr__` hook
  ownership question of N050 is settled here once).
- One CPython table `bindings/cpython_table.mojo` (global `MojolearnCPythonTableV3` layout) absorbing
  `perf2_shared_cpython_table.mojo`, `p3_lm_tokenizer_cpython.mojo` and the 18 entries of `p3_labels_metrics_native`;
  the 26 unproven forms (`cpy.PyModule_GetDict`, `PyObject_GetIter` / `PyIter_Next`) settled at the compile.
- Numerics: none touched.

Files. `bindings/classical_native.mojo`, `family_public_lifecycle_pyapi`, `estimator_binding_pyapi`,
`native_loaded_module`, `native_class_admission`, `native_prepared_owner`, `public_attributes_native`,
`{params,mode}_metadata_native`, `classical_validate`, `classical_attributes`, the record modules above, the three
extra tables. Bindings: every group (this is the shared layer), which is why it compiles first, alone.

Depends on nothing; A, B, C, D, G, H depend on it. Risks: the shared admission layer is the compile-memory
suspect of compile-fix-2 (+418 defs, +44 structs); folding the tables and records reduces instantiations and must
be measured at the compile (peak RSS per group).

Proof. Every identity lane (the door is on every path). New check: a tools source guard (`tools/check_admitted_call.py`)
that fails on `native_loaded_module(`, `Python.import_module(` and `.cpython()` outside the allowed files.

### F. Native kernel call table and the manifold driver (F-kernel-call-table, 12 rows)

Goal. Kits that drive a device algorithm as a sequence of primitive operations (`mani_kit.mojo` 78 Python-level
calls per manifold fit, `decomp_kit_native.mojo`, `classical_tsqr.mojo`, `classical_ridge.mojo`, `linalg_ops_kit.mojo`)
call the kernel through one `abi("C")` dispatch function resolved once at connection
(`ideas-cpu-per-call-cost.md`, Part 2 (a)), with handles travelling as integers in the int span.

Rows N290-N301: N290, N291, N292, N295, N296, N301 (manifold kit and driver); N297-N300 (metrics entries taking
caller pointers, about 29 device entries plus the host twins); N293, N294 (LinKit buffer borrow and shared
constants). The DecompKit / TSQR / Ridge call sites are counted under G where the row asks for a whole-step entry.

Design. A `descriptor_schema.mojo` per kit (x_decomp, linalg, estimators) in the shape of
`x_neighbors/descriptor_schema.mojo`; `<family>_native_api_v1` factories in each vendor binding and host twin;
the provider built once at import and kept in the context keyed by mode; `kit_call(provider, OP, addresses, ints,
floats)` with stack spans and the GIL released; `MKit` keeps host and device copies with an `on_device` flag as
`DecompKit` does. No numeric change; same op order.

Files. `bindings/mani_kit.mojo`, `manifold_native.mojo`, `mani_*_native.mojo`, `decomp_kit_native.mojo`,
`linalg_ops_kit.mojo`, `_mojolearn_x_decomp(.mojo/_host)`, `_mojolearn_linalg*`, `_mojolearn_metrics(.mojo/_host)`.
Bindings: x_decomp, linalg, metrics and host twins. Depends on E (the provider slot in `AdmittedCall.binding`).
Proof: `x-decomp-manifold`, `x-decomp-umap-options`, `umap`, `spectral*`, `x-decomp-*` (16), `linalg-*`,
`x-metrics-*` (6).

### G. Fused whole-step entries (G-fused-entries, 111 rows)

Goal. Where a public method today makes several native crossings with host round-trips between them, one entry
does the whole step on the device. Rows N302-N412: TSQR solve (R_aug -> coef, N404), RidgeCV and weighted OLS
(N403, N406, N408), PLS fit / predict / score (N348, N349), DecompKit steps and randomized SVD (N311, N332, N333,
N338, N340-N345, N396), SVC / GPC predict-all (N306, N366, N369, N372, N373, N380-N383), LOF / KNNImputer / label
propagation / NearestCentroid fused predicts (N312, N322-N324, N327), logistic `qn_proba` on device (N318, N357,
N360, N361), Huber outlier flags (N302), score = predict + metric (N416; N439 under H), HDBSCAN prediction data in
the fit session (N395), auto-ARIMA grouped search (N305, N398), mixture queries without block copies (N314, N331,
N335, N346, N354, N399, N401), kernel PCA and MCD (N313, N334, N344, N345, N400).

Design. Each entry is one native function per algorithm step that takes the admitted inputs (from E) and the
resident fitted state (from A) and returns its public outputs in the caller's buffers; the per-op entries stay for
the Python callers that remain. Numerics: inside one version the fold order of every fused entry is fixed on all
three vendors and the host column together (the standing rule); where a fused entry changes a reduction order
(PLS score, R^2 on device) the host column is changed in the same lane and the lane says so in its reply, because
the identity table re-records those lanes at the next release.

Files. `bindings/classical_{tsqr,ridge,ols}.mojo`, `pls_native`, `decomp_kit_native`, `linalg_decomp_native`,
`als_native`, `svc_kernel_native`, `host_svc_native`, `gp_classifier_native`, `_mojolearn_{gp,svm}(_host)`,
`neighbors_{local,semi}_native`, `logistic_native`, `x_linear_native`, `mixture_native`, `kpca_native`, `mcd_native`,
`_mojolearn_hdbscan`, plus the kernels under `decomposition/`, `glm/`, `svm/`, `gaussian_process/`, `x_neighbors/`,
`mixture/`. Bindings: estimators, x_decomp, linalg, svm, gp, x_neighbors, x_linear, mixture, hdbscan and host twins.

Depends on A, E, F. Risk: these are the rows the lanes called "too large without a compile"; each fused entry is
one lane with its kernel and its host twin, never a sweep.

Proof. `ols*`, `ridge*`, `x-ridge`, `linear-*`, `logistic-*`, `x-logistic-*`, `x-huber-*`, `x-decomp-pls`,
`x-decomp-lstsq-rsvd`, `x-decomp-pca-randomized`, `svc*`, `svr*`, `gpc*`, `gp*`, `knn-*` (11), `x-neighbors-*` (22),
`hdbscan*`, `x-prep-score-edges`, `metrics-*`.

### H. Labels and metrics still handled wrapper-side (H-labels-metrics, 37 rows)

Goal. Label encoding, class tables, class weights and the metric a `score` computes are native entries with integer
operation codes; the Python wrappers route and never touch values.

Rows N413-N449. Native encode / union encode N413, N414, N437, N438; class weights N446; operation codes instead
of string dispatch N424, N425, N427, N429, N426, N443, N428, N415; score from codes N435, N436, N422, N423, N439,
N416; pandas and Series label sources N430, N434, N421; GBDT text parse per predict N441, N442; connected components
as one device op N432; lazy attribute descriptors N418-N420; `prepare_targets` composition N444, N445; LabelBinarizer
N447; `MultiOutputClassifier.score` with string labels (N081, N084 under B).

Design. `labels_encode(y, kind) -> codes, classes` and `labels_union_encode(y_true, y_pred)` as binding entries
with the storage sources `label_input_sources.mojo` already admits (ndarray, Array, list, Series); every family's
`score` calls one `score_from_codes` entry (truth encoded against `classes_`, metric by code) so the predict output
never comes back to Python before the metric; `_expansion_*.py` string dispatch becomes integer codes read from one
table. Numerics: the metric kernels are the existing ones; the only change is where the encode runs.

Files. `bindings/classical_labels.mojo`, `label_input_sources.mojo`, `label_integer_storage.mojo`,
`p3_labels_metrics_native.mojo`, `classical_score.mojo`, `trees_public_score.mojo`, `family_native.mojo`,
`python/mojolearn/_labels.py`, `_metrics_impl.py`, `_class_weights.py`, `_expansion_{metrics,neighbors,trees}.py`,
`_forest_host.py`, `_gbdt_host.py`. Bindings: core, metrics, trees, family and host twins. Depends on E.
Proof: `gbdt-class-weights`, `gbdt-multiclass*`, `x-metrics-classification`, `x-metrics-cluster`,
`x-metrics-ranking`, `metrics-*`, `trees-*`, `knn-*`.

### I. Python 3.10 / 3.11 without watchers (I-py310-311, no inventory rows; plan item C on main)

Goal. The "check once" records key on CPython 3.12 type and dict watchers; wheels declare `>=3.10`, so 3.10 and
3.11 take the slow path on every call (`ideas-every-slow-lane-pass2.md`). Design: a per-class version stamp
(`tp_version_tag` read through the table) plus a per-model generation word bumped by the family's own setters (E)
replaces the dict watcher on those versions; the instance-dict walk is kept only for fields the family does not
own. Rows: none in the tables (the item is C of the plan on main); the per-class epoch rows N236, N259 and N258 are
its native half. Files: `native_class_admission.mojo`, `native_loaded_module.mojo`, the
record module of E. Proof: the CPU identity gate runs 3.12; add one 3.11 job to the tools tests (import, fit,
predict on one estimator per family) so the path is exercised; no timing.

### J. Multi-device and product gaps (J-multi-device-and-gaps, 4 rows)

- `LogisticRegression` and every GLM with several devices runs on one device in `identical` (the tiled gradient
  never reaches `glm/impl/qn/multi_gpu.mojo`): shard the tiled gradient (N453). Proof: the multi-GPU logical-shard
  gate (12 lanes, 1 / 2 / 3 logical devices) gains a GLM lane; `tools/parallel_gram_check.py` gains Ridge /
  LinearRegression (N455).
- Parallel shard workers carry int32 codes plus a class table instead of `y.tolist()` (N450), return the CSR triple
  (N451), store X once per worker (N452).
- `MultiOutputClassifier.score` with string labels (N081, N084): the native multi-output score takes codes (H).
- `x-decomp-lda` and `x-decomp-ipca` (Apple / CPU small envelopes): after profile 6.
- Unit tests of the multi-device split logic (no binding exposes it yet): one binding entry that returns the
  plan for a given device count and shape, and a tools test over it.

### K. Tools and gate items carried from the tables (K-tools-gate, 3 rows)

Not native work; listed so nothing is lost: `tools/identity_wheel_record.py:155` venv NumPy (N454),
`tools/parallel_gram_check.py` Ridge / LinearRegression cases (N455), the remaining `build_*.sh` `-j` defaults
(N456). These go into the release-process lane of the first release after 0.8.66.

### P. Rows the profile pass must settle before a design (P-profile-first, 4 rows)

`x-prep-nb-partial` (N458), the sequence `other` +3.5 ms per call (N459), `x-decomp-pca-randomized` fit (N460)
and the decode-session first-use rows (N457, N192 under D): no lane is written for them until the profile pass has
run.

## 3. Order and releases

Ranking rule: benefit per unit of risk. Benefit = per-item rows first, then per-call, then per-fit; weighted by
how many public methods and identity lanes pay the row (`public_methods` column; lane counts above) and by the
regressed seconds in `ideas-every-slow-lane-pass2.tsv` (open and partial groups: KMeans lifecycle 11.0 s,
tree / forest predict per call 6.9 s under profile, causal LM owner 6.3 s, search + GBDT adapter 5.3 s, PCA query
envelope 3.1 s, mixture 1.5 s, iterative imputer 1.4 s; everything else under 1 s). Risk = how many groups must
compile together and whether a protocol or ABI changes.

| rank | workstream | why here |
|---|---|---|
| 1 | E | every other workstream needs the generation word, the record module and the one table; compiles alone; no numerics |
| 2 | A | most per-call rows paid by the most lanes (trees, GP, SVC, decomposition, TSA); scheme already proven in five families; no numerics |
| 3 | B | the per-item rows (folds, members, rounds) with the largest regressed seconds left (search + GBDT adapter, iterative imputer); one shared door change |
| 4 | H | per-item label rows on the most-called methods (`score`, `fit` with string labels); small kernels |
| 5 | D | causal LM owner 6.3 s, per-token downloads; contained in the generation files |
| 6 | C | the largest ABI change (sessions, worker protocol); high value for training users, few identity lanes pay it per call |
| 7 | G | one lane per fused entry; some change reduction order (host column changed with them) |
| 8 | F | mechanism change under G's files; after G's entries exist, or in the same lanes |
| 9 | I, J, K | correctness and gaps; small |

Release split (each release: one integration branch, one compile of every group, one verification round):

- **First release after 0.8.66 (highest value, lowest risk):** E, A, H, K, and the B fold entry on the shared door
  only (the DeviceRows branch in `classical_check_x` / `family_public_lifecycle` plus `msel_cv` using it for the
  estimators whose door has it). About 9 Opus lanes with disjoint files: E-records (record module + generation
  word), E-table (one CPython table + the 26 forms), E-admitted-call (the struct and the doors), A-trees,
  A-gp-svm, A-decomp-tsa (IPCA, ALS, TSVD, RP, ARIMA, HW, forecast), A-cluster-spectral, H-labels, B-door. Must
  compile together: everything (E is the shared layer), so this release's compile is the full set of groups and
  the two large CPU groups need the swap already in place (lesson 11).
- **Second release:** B (members, rounds, search, parallel CV workers), D, G part 1 (TSQR / RidgeCV / OLS, PLS,
  randomized SVD, logistic `qn_proba`), I, J. About 8 lanes: B-members, B-rounds, B-search-workers, D-session,
  D-engine-output, G-linear, G-decomp, I-J. Compile: msel, trees, prep, transformer / mamba, estimators, x_decomp,
  linalg, glm and host twins.
- **Third release:** C (one lane per session kind: byte LM, Samba, SmallMLP / Sequential, CNN, neural blocks and
  embedding, ParallelNeuralTrainer), G part 2 (SVC / GPC predict-all, neighbours fused predicts, HDBSCAN), F.
  About 9 lanes. Compile: byte_lm, training, samba, x_sequence, x_cnn, transformer, embedding, svm, gp,
  x_neighbors, x_decomp, metrics and host twins.

Each release's lanes write code and never compile; the orchestrator compiles the integration branch once (every
switch on and off, NVIDIA and AMD targets, on the cheap fast-CPU boxes, through `tools/compile_slot.sh`), fixes
what breaks, merges, freezes, runs the ONE verification round on the built wheels, publishes.

What is NOT known until the post-compile profile pass of 0.8.66 (the 16 profiles of
`ideas-every-slow-lane-pass2.md` "Post-compile profile list" and the forest predict profile of
`forest-regressor-other-stage.md`), and how each answer re-ranks this plan:

1. Forest / tree predict per call (profile 1 and the forest profile: ms per call door / `_resident_with` / kernel
   / `_public_result` against 0.8.55). If the door is the cost, A-trees moves up and gets the forest export handle
   (N007) first; if the kernel sync is, A-trees stays and the fix is in the kernel launch, not an entry.
2. `x-metrics-search` GBDT adapter fit and predict per fold (profile 2): decides whether B-members or the GBDT
   adapter's own door (N253, N266) is the first B lane.
3. `x-prep-nb-partial` (profile 3) and `x-prep-iterative-options` (profile 4): whether the program build or the
   member calls dominate; decides N458 and the order of B-rounds against H.
4. PCA transform and decomposition fits (profiles 5, 6): kit operations and seconds per operation decide whether F
   (the call table) or G (whole-step entries) comes first for x_decomp; today G is ranked first by row count.
5. Mixture `score_samples` and BGMM fit (profiles 8, 9): the block-copy share decides whether the mixture rows stay
   in A or need a fused query entry (G).
6. First-lane effect and decode first use (profile 11): decides whether D starts with the binding-resolution
   cache (N192) or the session.
7. Watcher slots and cross-extension snapshot acceptance (profile 14): decides the shape of E's record module
   (one shared watcher or per-extension) and whether I is needed for 3.12 as well.
8. Sequence predict after the resident store (profile 15) and KMeans CPU-column predict (profile 16): confirm A's
   scheme and the cap; a miss there changes A's eviction design before it is copied to other families.
9. Normalizer and GP `return_std` (profiles 12, 13): per-call CPython share; decides whether E's `AdmittedCall`
   alone recovers the stateless prep and GP lanes or G entries are needed there too.

Until those numbers exist, the ranking above is by row counts and by reading, and the first release's set is
chosen so that no item in it depends on a profile answer.

## 4. Lessons from this round (apply in every lane brief)

1. Argument and local names that are Mojo keywords (`out`, `alias`, `owned`, `where`) fail to parse; rename before
   the compile (`compile-fix-1.md`: `classical_labels.mojo:737`, `xtrees_api_pyapi.mojo:907`, `metric_input_snapshot_pyapi.mojo:40`).
2. `py_call(obj.<name>)` where `<name>` is also a Mojo method of `PythonObject` (`copy`, `__eq__`, `__mul__`) binds the
   Mojo overload: use `py_call(py_attr(obj, "copy"))`, `Bool(py=a == b)`, the operator (`compile-fix-1.md`, 6 sites).
3. A default argument that constructs a `PythonObject` is evaluated at compile time and cannot be: make it required
   and have every caller pass it (`compile-fix-2.md` item 2, `mixture_native.mojo:413`).
4. Moving a field out of the middle of a struct value (`x.field^`) is refused: take it into a local and refill the
   field empty (`_mojolearn_ivf.mojo:165`, `compile-fix-2.md` item 1).
5. An import of a name a later lane renamed or removed (`DictWatch`, `_dict_watcher`) only shows at the compile:
   resolve every `from <module> import` against the integration branch before pushing.
6. A raising unlock inside a `DictKeyError`-typed `try` is refused: lock helpers are non-raising (`ivf_host_search.mojo:286`).
7. The compiler stopped without a diagnostic on the model_selection host group when estimator cloning recursed through
   a Tuple-returning function; cause not isolated (a standalone three-function reproducer compiles); fixed by keeping
   the recursion on `PythonObject` results and making the Tuple-returning clone a wrapper (`e289f32355`). Avoid
   recursion through `Tuple`-returning functions across `def` / `fn` boundaries.
8. One shared helper per concern, never one copy per lane: four CPython tables and five record helpers cost +57 %
   compile memory for +6 % lines (`compile-fix-2.md` item 4). A lane adds to the shared module and says so in its reply.
9. Records live in native side tables keyed by model identity with a generation word, never in a model's `__dict__`
   and never as a content hash per call (`p3nc_fitted_freeze.mojo`; decision 2026-11-06).
10. A resident device copy needs the read-only rule or a per-call upload; there is no third option.
11. The two largest CPU groups (tokenizer host 12.1 GiB, model selection host) need swap to compile on 16 GiB
    machines: swap is configured on the compile boxes before the first compile, and group memory is recorded per compile.
12. Large `elif` chains in one function (the 60-branch autograd frontend) are a compile-memory suspect: dispatch on
    integer codes through a table.
13. The 26 CPython calls in an unproven form (`cpy.PyModule_GetDict`, `PyObject_GetIter` / `PyIter_Next`) are settled
    once at the first compile of E; every lane aligns to that form and none invents a new one.
14. Line numbers drift as soon as a sibling lane touches the file: rows name the symbol as well as the line.
15. An OWED-SHARED row is closed by the owner of the named file in the same round when the owner is told; run
    `collect_owed.py` once per round before the compile so the next round's tables do not carry rows another lane closed.
