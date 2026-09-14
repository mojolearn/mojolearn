# BRIEF, forest inference on a CPU with no GPU (2026-09-13)

Branch `lane/forest-host-inference`. Andrew's goal, Sep 13. Prove that forest
INFERENCE is bitwise identical on CPUs that have no GPU we can target (Intel
Xeon, AMD EPYC, Qualcomm Snapdragon, Azure Cobalt) by training on a GPU, saving
the model, loading it on the CPU box, predicting, and comparing hashes to the
GPU's own predictions. The byte LM already has this shape (DEVIATION 2610,
`bindings/build_byte_lm_host.sh`, `.github/workflows/byte-lm-cpu-gate.yml`).
Forests did not. This brief records what was found, what was built, and what
is still owed.

## 1. Feasibility

### 1.1 Where the sequential forest predict lives

`inference_engine="sequential"` (the default, docs/FOREST_INFERENCE_ENGINES.md)
dispatches from `python/mojolearn/_forest_protocol.py:164-180`
(`_predict_forest`) to one native entry per family, with the five model arrays,
X row-major and the output handed over as addresses plus
`[n_rows, n_cols, n_trees, num_outputs]`.

RandomForest, `bindings/_mojolearn_rf.mojo`

| step | function | file:lines |
| --- | --- | --- |
| entry, classifier | `rf_predict_proba_binding` | `bindings/_mojolearn_rf.mojo:694-755` |
| entry, regressor | `rf_predict_reg_binding` | `bindings/_mojolearn_rf.mojo:758-815` |
| flat arrays to trees | `_rebuild_trees` | `bindings/_mojolearn_rf.mojo:635-668` |
| forest loop, classifier | `RandomForest.predict_proba` | `ensemble/randomforest.mojo:1076-1161` |
| forest loop, regressor | `RandomForest.predict` | `ensemble/randomforest.mojo:972-1074` |
| per-tree walk | `DecisionTree.predict` to `predict_one` | `ensemble/decisiontree/decisiontree.mojo:529-654` |
| the feature flush | `_ftz_feature` (DEVIATION 1942) | `ensemble/decisiontree/decisiontree.mojo:152-160` |
| node | `SparseTreeNode` | `ensemble/flatnode.mojo:94` |

ExtraTrees, `bindings/_mojolearn_trees.mojo`

| step | function | file:lines |
| --- | --- | --- |
| entry, both | `et_predict_binding` | `bindings/_mojolearn_trees.mojo:451-533` |
| forest loop | `forest_vote` | `extratrees/impl/randomforest/randomforest.mojo:516-547` |
| per-tree walk | `predict_one_accumulate` to `predict_leaf` | `extratrees/impl/decisiontree/flatnode.mojo:376-483` |
| node, tree | `SparseTreeNode`, `TreeMetaDataNode` | `extratrees/impl/decisiontree/flatnode.mojo:113, 312` |

The arithmetic of both forest loops is the same three statements. Zero a
float32 row vector, add every tree's leaf vector into it in increasing tree
order, divide each element by `Float32(n_trees)`. RF's classifier entry
returns that vector and Python takes the argmax (`_labels.argmax_rows`, first
max wins) and maps it through `classes_`; RF's regressor entry returns element
0 of the same vector. ET returns the vector for both and the Python surface
widens it to float64 (`extratrees.py:461-471, 554-558`). RF flushes the
feature it compares to a subnormal-free value (`_ftz_feature`, a comptime
no-op under FAST); ET does not.

### 1.2 GPU entanglement, measured

The census the lane asked for, run on `bfde04428`, matches printed and not
counted.

```
$ git grep -n -E "from gpu|max\.gpu|DeviceContext|std\.gpu" -- \
    bindings/_mojolearn_rf.mojo bindings/_mojolearn_trees.mojo \
    ensemble/randomforest.mojo ensemble/decisiontree/decisiontree.mojo ensemble/flatnode.mojo \
    extratrees/impl/randomforest/randomforest.mojo extratrees/impl/decisiontree/flatnode.mojo \
    checks/numerics.mojo bindings/hostptr.mojo
bindings/_mojolearn_rf.mojo:48:from max.gpu.host import DeviceBuffer, DeviceContext
bindings/_mojolearn_rf.mojo:388:        var ctx = DeviceContext()
bindings/_mojolearn_rf.mojo:556:        var ctx = DeviceContext()
bindings/_mojolearn_rf.mojo:866:        var ctx = DeviceContext()
bindings/_mojolearn_trees.mojo:45:from max.gpu.host import DeviceContext
bindings/_mojolearn_trees.mojo:353:        var ctx = DeviceContext()
bindings/_mojolearn_trees.mojo:430:        var ctx = DeviceContext()
bindings/_mojolearn_trees.mojo:584:        var ctx = DeviceContext()
ensemble/randomforest.mojo:5:from std.gpu import global_idx
ensemble/randomforest.mojo:8:from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
ensemble/randomforest.mojo:930:        ctx: DeviceContext,
(... nine more DeviceContext parameters in ensemble/randomforest.mojo, all fit-side ...)
extratrees/impl/randomforest/randomforest.mojo:100:from max.gpu.host import DeviceContext
extratrees/impl/randomforest/randomforest.mojo:328:    ctx: DeviceContext,
extratrees/impl/randomforest/randomforest.mojo:466:    ctx: DeviceContext,
```

And the same grep restricted to the per-tree layer returns nothing
(`git grep` exit 1) for `ensemble/decisiontree/decisiontree.mojo`,
`ensemble/flatnode.mojo`, `extratrees/impl/decisiontree/flatnode.mojo`,
`checks/numerics.mojo` and `bindings/hostptr.mojo`. Their complete import
lists are `ensemble.flatnode.SparseTreeNode` plus `checks.numerics.ftz`
(decisiontree.mojo:148-149), nothing (ensemble/flatnode.mojo), nothing
(extratrees flatnode.mojo), `std.sys.compile`, `std.memory`, `std.math`
(numerics.mojo), and `std.memory` (hostptr.mojo).

Verdict. The two per-tree walks that decide every bit of a prediction are
GPU-free and can be imported as they are. The two forest loops are not,
because they live in `ensemble/randomforest.mojo` and
`extratrees/impl/randomforest/randomforest.mojo`, both of which import
`max.gpu.host` at module level and hold the device fits. Importing either
module into a build with no accelerator target was not attempted, because
this Mac was allowed one small host-only compile and a probe that pulls in the
whole RF trainer is not small. The lane's fallback applies. The forest loops
are restated, three statements each, in a new import-only module
`core/forest_host_predict.mojo`, which the CPU binding imports and which the
GPU bindings do not, so no file the GPU bindings compile changes at all. The
host module says which lines it MIRRORS, and the smoke in section 5 plus the
gate in section 4 are what check that the restatement and the original agree
by bits.

### 1.3 The saved model format

`RandomForest*.save` (`randomforest.py:510-548`) and `ExtraTrees*.save`
(`extratrees.py:303-344`) write an npz through `_serialize.write_npz`
(`python/mojolearn/_serialize.py`). Members and dtypes, every one exact and
refused on a dtype mismatch by `_serialize.exact`.

| member | dtype, shape | note |
| --- | --- | --- |
| `format` | `<U` scalar | `mojolearn-randomforest-1` or `mojolearn-extratrees-1`; a `parallel_groves` archive appends `-parallel-groves-1` and adds `numeric_mode` |
| `estimator` | `<U` scalar | the class name, checked by `load` |
| `device` | `<U` scalar | the fit's `device` argument |
| `offsets` | `<i4` `(n_trees + 1,)` | prefix scan of node counts |
| `colid` | `<i4` `(nodes,)` | split column |
| `quesval` | `<f4` `(nodes,)` | split threshold, `<=` goes left |
| `left_child` | `<i4` `(nodes,)` | tree-local index, `-1` is a leaf, right child is `left + 1` |
| `leaves` | `<f4` `(nodes * num_outputs,)` | leaf vector per node, internal nodes carry dead slots |
| `meta` | `<i8` | RF `[n_features, n_trees, num_outputs]`; ET adds `[depth_cap_bound, max_depth_resolved, max_features]` |
| `classes` | `<i8`, `<f8` or `<U` list | classifiers only, `_labels.classes_member` |

The file bytes are a pure function of the arrays (member order sorted,
timestamps pinned to the zip epoch), so equal models give equal file hashes
across machines. Floats never pass through decimal text. Nothing in the file
depends on the vendor that fitted it except the `device` string, which is
informational.

### 1.4 What a CPU-only binding must expose

Load the five arrays plus X, return the divided vote. Concretely, three
entries with the same address contract as the GPU bindings, extended by the
node count so the binding can refuse a malformed file instead of reading past
it.

| entry | refusal | output |
| --- | --- | --- |
| `forest_host_rf_predict_proba` | `num_outputs < 2` | `n_rows * num_outputs` float32, the divided vote (`RandomForestClassifier.predict_proba`; `predict` is Python's argmax through `classes_`) |
| `forest_host_rf_predict_reg` | `num_outputs != 1` | `n_rows` float32 (`RandomForestRegressor.predict`) |
| `forest_host_et_predict` | `num_outputs < 1` | `n_rows * num_outputs` float32, the divided vote; the Python surface widens to float64 and takes the classifier's argmax |

Plus the read-backs every binding has (`forest_host_vendor` answering `cpu`,
`forest_host_numeric_mode` answering 1, `forest_host_sabotage`), and the seven
host helpers the Python layer resolves natively with no Python fallback
(`all_finite_f32`, `all_finite_f64`, `cast_f64_to_f32`, `argmax_rows_f32`,
`argmax_rows_f64`, `gather_i64`, `gather_f64`), because on a CPU-only install
the base GPU binding that normally exports them is a stub and
`_labels.argmax_rows` and `_labels.decode_labels` would otherwise raise. The
byte LM host binding already carries the first three under DEVIATION 2614.

The Python side then needs a loader that reads the npz, refuses the
`-parallel-groves-1` archives by name (their recorded GPU predictions are the
other algorithm's bits, and the host engine is the sequential one), and
returns exactly the dtypes the GPU classes return. RF probabilities float32,
RF regression float32, ET probabilities and regression float64, class labels
through `decode_labels`.

## 2. What was built

- `core/forest_host_predict.mojo`, the import-only host module. Rebuilds the
  trees the way `_rebuild_trees` and `et_predict_binding` do, refuses a
  child index that does not point forward inside its tree and a column past
  `n_features`, and runs the two forest loops over the GPU bindings' own
  per-tree walks. `MOJOLEARN_FOREST_HOST_SABOTAGE=1` divides by `n_trees + 1`
  instead, the gate's negative control.
- `bindings/host_helpers.mojo`, the seven helper bindings copied from
  `bindings/_mojolearn.mojo` so a CPU-only install can convert inputs and
  decode labels without a Python copy of native arithmetic.
- `bindings/_mojolearn_forest_host.mojo` and `bindings/build_forest_host.sh`,
  flag for flag with the byte LM host build. Output
  `python/mojolearn/host/_mojolearn_forest_host.so`, IDENTICAL only, no
  accelerator target, `--target-cpu apple-m1` on Darwin and `x86-64-v3` on
  Linux x86. `MOJOLEARN_BUILD_JOBS` sets the compile jobs (default 2).
- `python/mojolearn/_forest_host.py` with `HostForest.from_file(path)`,
  `predict`, `predict_proba`, and the module-level `host_predict(path, X)` and
  `host_predict_proba(path, X)`, exported from `mojolearn`. `_backend` now
  treats a built forest host binding like the byte LM one for the CPU-only
  install (DEVIATION 2615), and `_buffer._host_native` resolves the seven
  helpers from it.
- `tools/forest_host_gate.py` with `record` (a GPU box, through the normal
  package) and `check` (the CPU box, through `HostForest`), both over a
  fixture spec that regenerates X from a seed and verifies its SHA-256, and
  `.github/workflows/forest-host-gate.yml` on the byte LM gate's seven
  runners.
- `bench/results/forest_host/`, the fixtures the workflow reads.

## 3. Verdict and owed items

Feasible, built, and shown on one machine and one vendor (section 5). The
host binding compiles with no accelerator target from the GPU bindings' own
per-tree walks, and on this Mac it reproduced every prediction hash the Metal
IDENTICAL build recorded, for all four estimators.

Owed, in order.

1. The NVIDIA and AMD recordings. `bench/results/forest_host/OWED-nvidia/`
   and `OWED-amd/` hold a `fixture.json` and an `expected.json` whose
   `status` is `OWED` and no `model.npz`. On each box, fit a small sequential
   forest under `MOJOLEARN_NUMERIC_MODE=identical`, save it there, run
   `tools/forest_host_gate.py record <dir>`, commit. The gate exits 2 on the
   placeholder and the workflow fails until then, on purpose.
2. The Linux x86 and ARM64 runs of `.github/workflows/forest-host-gate.yml`.
   Nothing on a CPU other than this Mac's has run the host binding. The
   workflow triggers on push to `main` and to this lane; this branch is not
   pushed.
3. The sabotage build. The workflow builds it and requires the gate to catch
   it. It was not built on this Mac (one host build was the budget). The
   gate's own ability to fail was shown instead by tampering a recorded hash
   (section 5).
4. Cross-vendor. A CPU reproducing Apple's bits and the same CPU reproducing
   NVIDIA's bits is the claim Andrew wants; today only the first half has a
   recording, and only on Apple's own host CPU.

## 4. The gate

`tools/forest_host_gate.py record <dir>` on a GPU box loads `model.npz` with
its own estimator class through the normal package, regenerates the rows
from `fixture.json` (SplitMix64, `(bits >> 40) / 2^24` as float32, verified
against `x_sha256`), predicts, and writes `expected.json` with the SHA-256,
dtype and shape of `predict` and, for classifiers, `predict_proba`, plus the
vendor, numeric mode and model file hash. It refuses a CPU-only install, so
the host binding can never record its own answer as the reference.

`tools/forest_host_gate.py check <dir>...` on the CPU box does the same
through `HostForest` and exits 0 only when every hash, dtype and shape is
equal. 1 on a mismatch, 2 when it cannot run, which includes an OWED
placeholder, a model file whose hash is not the recorded one, and fixture
rows that do not regenerate to `x_sha256`. `--expect-mismatch` inverts the
verdict for the sabotage build. `--package-root` says where `mojolearn` is
imported from, which is how the Mac recording below used the GPU build in the
main checkout while running the tool from this worktree.

`.github/workflows/forest-host-gate.yml` runs on the byte LM CPU gate's seven
runners. Per runner it builds the host binding, checks every RECORDED fixture,
builds the sabotage binding and requires the gate to catch it, then runs the
gate over the OWED directories and fails.

## 5. The Apple data point, 2026-09-13

Machine, Apple M4, macOS 26.5.2, Python 3.14.7, Mojo 1.0.0 (ed45d567). GPU
side, the main checkout's Metal IDENTICAL build of `bfde04428` (its
`python/mojolearn/identical/*.so`, built the same afternoon). Host side, this
worktree at `b0da9af0f` with no GPU binary at all, so the package imported as
a CPU-only install (`mojolearn.vendor()` answered `cpu`,
`gpu_arch_how()` answered "no GPU binary set loaded (CPU-only install,
DEVIATION 2615)") and the host binding built once on this Mac with
`MOJOLEARN_BUILD_JOBS=1 nice -n 19 sh bindings/build_forest_host.sh` (exit 0,
266,264 bytes, the same deprecation warnings the GPU bindings print).

The smoke. Four forests fitted on Metal through the normal package on 200
rows of 8 features (SplitMix64 seed 1), labels `int((x0 * 7 + x1 * 13) * 3) % 3`
for the classifiers and `x1 * 2 + x2` for the regressors, `n_estimators=8,
max_depth=6, random_state=7, numeric_mode='identical',
inference_engine='sequential'`, saved with `save`, then predicted on 200
held-out rows (seed 7, `x_sha256`
`6f4b874d9d5a82d52b6754df37b8f1f0b56c29ed311ff311b4413c2ac40dcc0b`) by the
GPU class (`record`) and by `HostForest` (`check`).

| fixture | estimator | nodes | output | GPU sha256 | host sha256 | equal |
| --- | --- | --- | --- | --- | --- | --- |
| apple-m4-rf_classifier | RandomForestClassifier | 426 | predict `<i8` [200] | `4202c25dfdabe4cda3373ccd9b3564b00bdf29910edeca82b302f2a07897a502` | same | yes |
| apple-m4-rf_classifier | RandomForestClassifier | 426 | predict_proba `<f4` [200, 3] | `49d496cff13daff8540b266415f33ee39c3b4b48a8c27f61d38af509c6b128f4` | same | yes |
| apple-m4-rf_regressor | RandomForestRegressor | 904 | predict `<f4` [200] | `6b1e214356a98d98d8b8e83f3236f60c4a751a56c0c58ed45c85d2783da2781f` | same | yes |
| apple-m4-et_classifier | ExtraTreesClassifier | 652 | predict `<i8` [200] | `1febdff20a95525b8716892b70950e35405aa67b1516597a3898601003d455c3` | same | yes |
| apple-m4-et_classifier | ExtraTreesClassifier | 652 | predict_proba `<f8` [200, 3] | `178c5a299a623a67144eda04d9120387c5bec3260b21d8dad0368258c9e1de10` | same | yes |
| apple-m4-et_regressor | ExtraTreesRegressor | 792 | predict `<f8` [200] | `32f5d5961e9aa131a700cee9b6255ad7ebba09d26e74f32eb1d7b692ce88bd5c` | same | yes |

"same" means the `check` line printed the host hash equal to the GPU hash
character for character; the verbatim lines are

```
check 2026-09-13-apple-m4-rf_classifier RandomForestClassifier predict EQUAL gpu 4202c25d... host 4202c25d...
check 2026-09-13-apple-m4-rf_classifier RandomForestClassifier predict_proba EQUAL gpu 49d496cf... host 49d496cf...
check 2026-09-13-apple-m4-rf_regressor RandomForestRegressor predict EQUAL gpu 6b1e2143... host 6b1e2143...
check 2026-09-13-apple-m4-et_classifier ExtraTreesClassifier predict EQUAL gpu 1febdff2... host 1febdff2...
check 2026-09-13-apple-m4-et_classifier ExtraTreesClassifier predict_proba EQUAL gpu 178c5a29... host 178c5a29...
check 2026-09-13-apple-m4-et_regressor ExtraTreesRegressor predict EQUAL gpu 32f5d596... host 32f5d596...
gate verdict IDENTICAL (4 fixtures, exit 0)
```

and the full hashes are in each fixture's `expected.json` under
`bench/results/forest_host/2026-09-13-apple-m4-*/`.

The gate can fail. A copy of the RF classifier fixture with the first
character of its recorded `predict_proba` hash changed (`49d496cf` to
`09d496cf`) gave `predict EQUAL`, `predict_proba DIFFER`, verdict MISMATCH,
exit 1; the same copy under `--expect-mismatch` gave EXPECTED MISMATCH SEEN,
exit 0; `check bench/results/forest_host/OWED-nvidia` gave exit 2 naming the
owed recording.

What this is and is not. It is one machine whose GPU (Metal) and whose CPU
(the M4's own cores) agree on 1,200 predictions of six kinds through two
different binaries, the GPU binding's sequential engine and the host
binding. It is not a second CPU, not a second vendor, and not a large model.
The certified table below grows only from gate reports, one row per report
read.

| CPU | recorded by | fixtures | verdict | report |
| --- | --- | --- | --- | --- |
| Apple M4 (this Mac, host cores) | Apple M4 Metal, IDENTICAL, `bfde04428` build | 4 (RF and ET, classifier and regressor) | IDENTICAL | the check lines above, 2026-09-13 |

## Seven CPUs reproduce the Metal predictions (2026-09-13, run 34780078300)

The workflow ran on the branch's first push. Every runner built the host binding and
reported `gate verdict IDENTICAL (4 fixtures, exit 0)` on the four Apple M4 recordings,
and `EXPECTED MISMATCH SEEN (4 fixtures, exit 0)` on the sabotage build, so the gate
reads the predictions it claims to. The jobs are red only because the NVIDIA and AMD
recordings are still the OWED placeholders, which is what the placeholders are for.

| runner | CPU | verdict |
|---|---|---|
| x86-64 Linux draw a, b, c (24.04) | AMD EPYC 7763 | IDENTICAL |
| x86-64 Linux draw d (22.04) | AMD EPYC 9V74 | IDENTICAL |
| x86-64 Linux draw e (22.04) | GenuineIntel (Xeon, hosted) | IDENTICAL |
| ARM64 Linux (Azure Cobalt 100) | Neoverse-N2 | IDENTICAL |
| Apple M1 macOS (hosted) | Apple M1 (Virtual) | IDENTICAL |

So a forest trained on an Apple GPU predicts the same bits on an Intel CPU, on AMD CPUs
and on an ARM server CPU with no GPU present. This is the CPU half of the question
"is inference bitwise identical on vendors whose GPUs Mojo cannot target". Their GPUs
remain out of reach until a Mojo backend exists for them.

## GBDT host inference (2026-09-13, the same lane, second deliverable)

Andrew's Sep 13 goal widened the question from the two forests to every
lane with an `infer` column in `tools/identity_break.py`. This section is
the GBDT half of that, the four lanes `gbdt-symmetric`, `gbdt-depthwise`,
`gbdt-lossguide` and `gbdt-rmse` (`tools/identity_break.py:190-214`), all
of which save a model through `GradientBoosting.save`
(`python/mojolearn/ensemble.py:1515`). The classical half is the section
after this one.

### What the GPU path does, measured

Every step of a GBDT prediction on the GPU path is a kernel or a device
buffer, unlike the forests, whose per-tree walks were host code.

| step | function | file:lines | device? |
| --- | --- | --- | --- |
| parse the model text | `load_model_text` | `gbdt/models/model_text.mojo:706-1449` | returns a `TrainedModel` from `gbdt/train.mojo`, which imports `max.gpu.host` at `:21` |
| quantize the rows | `_build_cindex_from_floats` | `gbdt/train.mojo:261-370` | launches `binarize_float_feature_kernel` (`gbdt/gpu_data/kernel/binarize.mojo:83-160`) |
| pack and apply, oblivious | `predict` | `gbdt/methods/doc_parallel_boosting.mojo:2319-2495` | `enqueue_fill(cursor, Float32(bias))` then `compute_bins_and_add_kernel` per tree (`gbdt/models/kernel/add_bin_values.mojo:33-124`) |
| apply, non-symmetric | `add_non_symmetric_tree_to_cursor` | `gbdt/models/add_non_symmetric_tree_doc_parallel.mojo:189-249` | `compute_non_symmetric_decision_tree_bins_kernel` (`add_bin_values.mojo:222-320`) then `add_bin_model_value_kernel` (`gbdt/methods/kernel_add_model_value.mojo:114-184`) |
| read back | `predict_floats` / `predict_multi_floats` | `gbdt/train.mojo:2322-2474` | plane-major cursor to row-major output |
| the Logloss link | `gbdt_sigmoid_binding` | `bindings/_mojolearn_gbdt.mojo:133-149` | HOST already, `identical_exp64` in float64 |

The tree structures cannot be imported either. `gbdt/models/oblivious_model.mojo:37`
imports `non_symmetric_tree.mojo`, which imports `gbdt/data/leaf_path.mojo`
(`:74` the tensor builder) and `gbdt/methods/helpers.mojo` (`:89` the
searcher). What IS GPU-free, by `grep -n -E "^from |^import "` on each file,
is `gbdt/gpu_data/grid_policy.mojo` and `gbdt/gpu_data/gpu_structures.mojo`
(no imports at all), `gbdt/gpu_data/compressed_index_builder.mojo` (`:25`
grid_policy, `:34` gpu_structures), and `gbdt/data/quantization.mojo`
(`:63` std.math, `:65` `gbdt/grid_creator/binarization.mojo`, itself
`std.math`, `std.sys.compile`, `checks.numerics`, and `:66`
`gbdt/options/data_processing_options.mojo`, no imports).

So the design differs from the forests in two ways. The model text is
parsed in Python (`python/mojolearn/_gbdt_host.py`, `parse_model_text`),
which reads the same records the Mojo loader reads, from the hex bits half
of every float token, and refuses by name what the host walk does not carry
(CTR columns and tables, tensor CTR registries, a feature of type `ctr` or
`tensor_ctr`). And the host module `core/gbdt_host_predict.mojo` REUSES
`build_layout` and the NaN treatment helpers as they are, so the packed
compressed index is built by the same function the GPU path builds it
with, and RESTATES the four kernels as row loops over those same words,
shift and mask exactly as each kernel writes them (the oblivious kernel
masks a pre-shifted value, the non-symmetric one shifts then masks; both
are kept as written). The arithmetic a prediction's bits depend on is the
quantizer's float32 `value > border` comparisons, the integer bin walk, and
one float32 add per tree into a cursor seeded with `Float32(bias)`, in tree
order. The binding grows two entries, `forest_host_gbdt_predict` (the
address contract is in its docstring, `bindings/_mojolearn_forest_host.mojo`)
and `forest_host_gbdt_sigmoid` (the body of `gbdt_sigmoid`). No fourth
binary. `mojolearn.host_model(path)` returns a `HostForest` or a `HostGBDT`
by the file's `format` member, and `host_predict` / `host_predict_proba`
route through it.

One seam on the Python side. `GradientBoosting._check_fitted`
(`ensemble.py:1306`) stages X column-major through `as_f32_colmajor`, whose
2-D flip is the base binding's `transpose_f32`, a `_NoGpuBinding` stub on a
CPU-only install (the first `check` failed there, `ImportError: the base
binding has no transpose_f32`). `HostGBDT.predict` takes the float32 cast
through the same `cast_f64_to_f32` helper (the host binding carries it) and
the flip through `Array._as_order('F')`, the package's own permutation.
`transpose_f32` documents itself as "a pure move, no arithmetic: every
float32 bit pattern, NaN payloads included, arrives unchanged"
(`bindings/_mojolearn.mojo:765-768`), so the bytes are the same by that
contract, and the hashes below are what checked it.

### The Apple data point, GBDT

Machine, Apple M4, macOS 26.5.2, Python 3.14.7, Mojo 1.0.0 (ed45d567). GPU
side, the main checkout's Metal IDENTICAL build (`python/mojolearn/identical/
_mojolearn_gbdt.so`, built 2026-09-13 15:45, package at `00a63d7cc`). Host
side, this worktree at `1f73788eb` plus the lane's changes, imported as a
CPU-only install (`mojolearn.vendor()` answered `cpu` in the check process),
the host binding built once with `MOJOLEARN_BUILD_JOBS=1 nice -n 19 sh
bindings/build_forest_host.sh` (exit 0, 313,928 bytes, the deprecation
warnings the GPU bindings print and nothing else).

The fixtures. `tools/forest_host_gate.py make <dir> --kind <kind>` fitted
each on Metal through the normal package on 200 rows of 8 features
(SplitMix64 seed 1), labels `int((x0 * 7 + x1 * 13) * 3) % 2` for the three
Logloss models and `x1 * 2 + x2` for RMSE, `n_estimators=8, max_depth=4,
random_state=7, numeric_mode='identical'`, `grow_policy` SymmetricTree,
Depthwise or Lossguide (`max_leaves=16`), saved with `save`, then predicted
on the same 200 held-out rows as the forests (seed 7, `x_sha256`
`6f4b874d9d5a82d52b6754df37b8f1f0b56c29ed311ff311b4413c2ac40dcc0b`) by
`GradientBoosting.load(...).predict` / `predict_proba` (`record`) and by
`HostGBDT` (`check`). The models are not trivial. The symmetric and RMSE
files hold 8 `tree` records of depth 4 (32 `split`, 128 `leaf`); the
depthwise file 8 `ntree` records with 79 `node` and 87 `leaf`; the
lossguide file 8 `ntree` with 109 `node` and 117 `leaf`; the RMSE file
carries a `bias` record (`boost_from_average`), so the cursor seed is
exercised. The host predictions take 68, 88, 119 and 24 distinct values
over the 200 rows.

| fixture | loss, policy | model sha256 | output | GPU sha256 | host sha256 | equal |
| --- | --- | --- | --- | --- | --- | --- |
| apple-m4-gbdt_symmetric | Logloss, SymmetricTree | `e92aeb34b0c5b3fd4e3b022d351a70b8973de0d5aa3f863921898a4a6fd9591c` | predict `<f4` [200] | `2bb7dc22a0c1e1706b93aa3148504232e041530abf404b4f9f39de0a78601499` | same | yes |
| apple-m4-gbdt_symmetric | | | predict_proba `<f8` [200, 2] | `29d34cc881ed9647e6b5de129a5dc48c0b94c268ef9af808dc7d9d0e1203ef24` | same | yes |
| apple-m4-gbdt_depthwise | Logloss, Depthwise | `1ce58bb8aebc42b79fb65e7e65156d17c4105889ac3d002a56c3594e7fbaf6f3` | predict `<f4` [200] | `0972e2daf11e7fe89cf32dd57c408ee9aa79b06171c21f25e05f72e2b83029e1` | same | yes |
| apple-m4-gbdt_depthwise | | | predict_proba `<f8` [200, 2] | `7060e6061689856fe964c9f4ef1a934ba46161505e1b8370ca640e608f6beb3a` | same | yes |
| apple-m4-gbdt_lossguide | Logloss, Lossguide | `ce045c6ad8de4e51f6210fbede267ad28dac797d931c98799a2e55cb1fa523bc` | predict `<f4` [200] | `2ca7a92587a85556077560c42e1bb9039b7bec23039f2de9b233607ec5fbbd23` | same | yes |
| apple-m4-gbdt_lossguide | | | predict_proba `<f8` [200, 2] | `c32f234d45f7043ba5a1310f7e4f5db3e8aa7dd5dd19cd15970ac9afc46b2262` | same | yes |
| apple-m4-gbdt_rmse | RMSE, SymmetricTree | `9039f13ad5bb32afce674bf1c4d225ea42c392e5d6e37e6ab72881515a7be9ed` | predict `<f4` [200] | `98318916ca1e5eceb099c51e7be4b7efdd1eb350dbe1184dbad1b8d642c41fb5` | same | yes |

"same" means the `check` line printed the host hash equal to the GPU hash
character for character. The verbatim lines, with the four forest fixtures
checked in the same run through the same binary:

```
check 2026-09-13-apple-m4-gbdt_symmetric GradientBoosting predict EQUAL gpu 2bb7dc22... host 2bb7dc22...
check 2026-09-13-apple-m4-gbdt_symmetric GradientBoosting predict_proba EQUAL gpu 29d34cc8... host 29d34cc8...
check 2026-09-13-apple-m4-gbdt_depthwise GradientBoosting predict EQUAL gpu 0972e2da... host 0972e2da...
check 2026-09-13-apple-m4-gbdt_depthwise GradientBoosting predict_proba EQUAL gpu 7060e606... host 7060e606...
check 2026-09-13-apple-m4-gbdt_lossguide GradientBoosting predict EQUAL gpu 2ca7a925... host 2ca7a925...
check 2026-09-13-apple-m4-gbdt_lossguide GradientBoosting predict_proba EQUAL gpu c32f234d... host c32f234d...
check 2026-09-13-apple-m4-gbdt_rmse GradientBoosting predict EQUAL gpu 98318916... host 98318916...
check 2026-09-13-apple-m4-rf_classifier RandomForestClassifier predict EQUAL gpu 4202c25d... host 4202c25d...
check 2026-09-13-apple-m4-rf_classifier RandomForestClassifier predict_proba EQUAL gpu 49d496cf... host 49d496cf...
check 2026-09-13-apple-m4-rf_regressor RandomForestRegressor predict EQUAL gpu 6b1e2143... host 6b1e2143...
check 2026-09-13-apple-m4-et_classifier ExtraTreesClassifier predict EQUAL gpu 1febdff2... host 1febdff2...
check 2026-09-13-apple-m4-et_classifier ExtraTreesClassifier predict_proba EQUAL gpu 178c5a29... host 178c5a29...
check 2026-09-13-apple-m4-et_regressor ExtraTreesRegressor predict EQUAL gpu 32f5d596... host 32f5d596...
gate verdict IDENTICAL (8 fixtures, exit 0)
```

The gate can fail on GBDT. The sabotage build (the second and last host
build this Mac was allowed, `-D MOJOLEARN_FOREST_HOST_SABOTAGE=1`, which
seeds the GBDT cursor at `bias + 1` and divides the forest vote by
`n_trees + 1`) over the same eight fixtures gave `DIFFER` on all seven
GBDT cases and on every forest case except the two classifiers' `predict`
(an argmax is invariant to the divisor, as section 5 already noted),
verdict `EXPECTED MISMATCH SEEN (8 fixtures, exit 0)`. Section 3's owed item
3, the sabotage build on this Mac, is therefore closed as well.

### What this is and is not, and what is owed

It is one machine whose GPU (Metal) and whose CPU agree on 1,400 GBDT
predictions of seven kinds through two binaries, the GPU binding's kernels
and the host binding, for all three tree shapes and for the two links the
identity lanes read (raw scores, the Logloss sigmoid). It is not a second
CPU, not a second vendor, not a CTR model, not a multi-output loss.

Owed, in order.

1. NVIDIA and AMD GBDT recordings, the same shape as the forests' (`make`
   then `record` on each box, then commit). The workflow will run the GBDT
   fixtures on its seven CPUs the first time this branch is pushed; nothing
   but this Mac has run the GBDT host walk.
2. `predict_proba` for `MultiClass` and `MultiClassOneVsAll`. The softmax
   and elementwise sigmoid run on the host inside `gbdt_predict_multi`
   (`gbdt/train.mojo:2481-2560`, `identical_exp64` in float64), so they are
   restatable, but no fixture measures them and `HostGBDT.predict_proba`
   refuses those losses by name rather than ship an unmeasured transform.
   The raw `predict` for `dim > 1` is built and unmeasured for the same
   reason.
3. The decimal half of the float tokens. `load_model_text` checks that the
   decimal agrees with the bits within one ULP (`model_text.mojo:315-343`);
   the Python parser reads the bits only, as `_tree_metadata` does, so a
   hand-edited decimal is caught by the GPU loader and not by the host one.
4. Subnormal features. The quantizer's `value > border` runs on the device
   for the GPU path and on the host here. The fixture rows are in `[2^-24, 1)`
   and never subnormal; whether a Metal, CUDA or HIP comparison of a
   subnormal feature against a border agrees with the host's has not been
   measured (the forests answered this with `_ftz_feature`, DEVIATION 1942;
   GBDT has no such flush and may need none, but that is a prediction).

## Classical lanes: what CPU inference needs

The remaining `infer` lanes of `tools/identity_break.py` are knn, knn-clf,
knn-reg, pca, tsvd, ols, ridge, lasso, elasticnet, logistic, svc, kde and
iforest. None is implemented here. This section states, per lane, what a
host predict entry would need, from the code and not from prose, and ranks
the estimates. Every file path is relative to the repository root; the
census was `grep -n -E "^from |^import "` on each file named.

### Three facts that apply to every classical lane

**No classical estimator had a save or load when this was written
(2026-09-13 morning).** SUPERSEDED for eight of them: the classical lane
(2026-09-13 evening) gave LinearRegression, Ridge, TruncatedSVD,
LogisticRegression and PCA a `save`/`load`, and the knn lane (2026-09-14)
gave NearestNeighbors, KNeighborsClassifier and KNeighborsRegressor one
(`mojolearn-knn-1`, `python/mojolearn/neighbors.py`); the grep below now
returns those. It still returns nothing for `_solver_impl.py`,
`_svm_impl.py`, `density.py` and `_iforest_impl.py`. The grep
`grep -n -E "def save|def load|write_npz|read_npz|__getstate__|__setstate__|__reduce__|pickle"`
over `python/mojolearn/neighbors.py`, `decomposition.py`, `linear_model.py`,
`_solver_impl.py`, `_svm_impl.py`, `density.py` and `_iforest_impl.py`
returned nothing. The only `save`/`load` pairs in the package are the tree
families (`ensemble.py:1515/1581`, `extratrees.py:303/346`,
`randomforest.py:510/550`). The codec exists (`_serialize.write_npz`,
`python/mojolearn/_serialize.py:267`, and `read_npz`, `:282`) and is what
each lane would add a `save` on top of, one to two hours per estimator
including the exact-dtype refusals the forest loader has. `Array.__reduce__`
(`python/mojolearn/_array.py:425`) makes default pickling round-trip the
fitted Arrays, but no class opts into it and pickle is not a format the gate
should read.

**Every arithmetic module is GPU-entangled, and so are the host oracles the
checks use.** The per-estimator `estimator.mojo` files all import
`max.gpu.host` (`neighbors/estimator.mojo:122`, `decomposition/estimator.mojo:34`,
`glm/estimator.mojo:37-38`, `solver/estimator.mojo:73`, `svm/estimator.mojo:69`,
`isolation_forest/estimator.mojo:36`, `kde/estimator.mojo:43`). The host
oracles under `*/checks/` mostly avoid `max.gpu` on their own lines but
import it transitively (`neighbors/checks/metric_oracle.mojo:40-41` through
`core/row_norms.mojo:56-57`; `kde/checks/kde_oracle.mojo:46,55` through the
same and `kde/impl/neighbors/kernel_density.mojo:103`;
`svm/checks/smo_oracle.mojo:72` through `svm/impl/smosolver.mojo:69,72`;
`solver/checks/cd_oracle.mojo:85` through `solver/checks/profile_dot.mojo:43`;
`isolation_forest/checks/if_oracle.mojo:42,48` through the device
implementation it checks). Two are fully GPU-free and importable as they
are. `gemm/checks/gemm_oracle.mojo` (its only import is `:75 from
checks.numerics import ftz, identical_mul_add`), the IDENTICAL GEMM contract
reference (`gemm_oracle_cell` at `:454`, `fold_balanced_tree` at `:397`), and
`checks/numerics.mojo` itself (`:5` std.sys.compile, `:70` std.memory), which
holds `ftz`, `identical_mul_add`, `identical_exp64` and the rest. So the
forest pattern applies again. Reuse what is GPU-free, restate the kernel
loop bodies in an import-only module, and let the gate measure the
restatement.

**The CPU-only install has three helpers the classical lanes lean on.**
`_buffer._native` (`python/mojolearn/_buffer.py:734`) resolves
`all_finite_f32`, `all_finite_f64`, `cast_f64_to_f32`, the two `argmax_rows`
and the two `gather` helpers from the host bindings (`_host_native`,
`_buffer.py:770-800`) and nothing else; `transpose_f32` and
`cast_colmajor_f64_to_f32` are not among them, which is the seam the GBDT
section hit. Adding those two to `bindings/host_helpers.mojo` (copied from
`bindings/_mojolearn.mojo:759` as the other seven were) and to
`_FOREST_HOST_NATIVE_KEYS` is half an hour plus one host build, and every
column-major lane below assumes it.

### Per lane

**ols, ridge (`LinearRegression.predict`, `Ridge.predict`).**
(a) No save/load. (b) `coef_` `<f4` `(n_features,)` (`linear_model.py:543`
and `:677`, `empty((cols,), "<f4")`), `intercept_` a Python float
(`:554/:556` and `:688/:690`, computed with `math.fsum` in float64 at fit
time), `n_features_in_` an int; a save would store the intercept as `<f8`
and pass `float(intercept_)` exactly as `predict` does at `:570` and `:706`.
(c) Both call `_mojolearn_estimators.ols_predict` (`linear_model.py:567`,
`:703`) to `bindings/_mojolearn_estimators.mojo:391`, then `ols_predict_host`
(`glm/estimator.mojo:192`, `DeviceContext` at `:406` of the binding), then
`gemv_n` (`core/gemm.mojo:322`), which under IDENTICAL launches
`pinned_gemv_n_kernel` (`core/gemm.mojo:93-113`), then a conditional
`_add_scalar_kernel` for the intercept (`glm/estimator.mojo:227-231`). The
pinned kernel is one serial loop per row, `acc = ftz(identical_mul_add(ftz(x),
ftz(y), acc))` over `k` in feature order, then `ftz(0 + ftz(acc))`. That is
five lines of `checks.numerics` calls, GPU-free, and the only thing to
restate; `core/gemm.mojo` itself imports `max.gpu.host` at `:8` and
`std.gpu` at `:9`. (d) None in `predict` beyond `float(self.intercept_)`.
Estimate 4 to 6 hours for both lanes together (one entry serves both, as
`ols_predict` does), of which half is the save/load and the fixture.

**lasso, elasticnet (`ElasticNet.predict`, `Lasso.predict`).**
(a) No save/load. (b) `coef_` `<f4` (`_solver_impl.py:340`), `intercept_`
a Python float from a `<f4` slot (`:354`), `n_features_in_` (`:356`); X is
staged COLUMN-major (`_solver_impl.py:363`, `_as_fortran` at `:290`).
(c) `_mojolearn_solver.cd_predict` (`_solver_impl.py:369`) to
`bindings/_mojolearn_solver.mojo:118`, then `cd_predict_host`
(`solver/estimator.mojo:166`), then `cd_predict` (`solver/impl/cd.mojo:544`,
imports `max.gpu.host` at `:106`, `std.gpu` at `:107`, and `gemv_n` at
`:110`), then `linear_reg_h` (`solver/impl/functions/linear_reg.mojo:51`).
CORRECTED 2026-09-13 evening: the sentence that stood here, "the same
pinned gemv over a column-major X plus the intercept", was FALSE. Under
IDENTICAL `linear_reg_h` calls `identical_gemm(ctx, pred, x, coef, n_rows,
1, n_cols, OP_TN)`, the `mojolearn.identical.gemm.fp32.v1` profile whose
GPU-free definition is `gemm_oracle`; `gemv_n` is its FAST arm only. The
phase 1 host binding (`bindings/_mojolearn_solver_host.mojo::cd_predict`,
1b319230) already restates exactly that, `gemm_oracle` at OP_TN then
`ftz(v + intercept)`, and the seven-runner CPU gate (run 34793118831)
reads its infer cells IDENTICAL x4, so this lane was DONE by phase 1 and
the classical host inference lane skipped it. (d) None.

**logistic (`LogisticRegression.predict_proba`).**
(a) No save/load. (b) `_w` `<f4` `(n_features + fit_intercept,)`
(`linear_model.py:912-923`) is what inference reads (`:943`); `coef_` and
`intercept_` (`:926-928`) are copies not read by inference; `classes_` a
Python list (`:897`); `fit_intercept`; output `<f8` `(n, 2)` (`:959`).
(c) Two binding calls. `qn_decision_function` (`linear_model.py:942` to
`bindings/_mojolearn_estimators.mojo:475`, `DeviceContext` at `:491`, then
`qn_decision_function_host`, `glm/estimator.mojo:353`, then
`qn_decision_function`, `glm/impl/qn/qn.mojo:230`, imports `max.gpu.host` at
`:34`), a device dot product per row whose fold order must be read off that
kernel; and `qn_sigmoid` (`linear_model.py:960` to
`bindings/_mojolearn_estimators.mojo:497`, then `qn_sigmoid_host`,
`glm/estimator.mojo:384-410`), which creates NO context and is already a
host loop, `p = 1 / (1 + identical_exp64(-Float64(score)))`, `1 - p` and `p`
in float64 (DEVIATION 549). Its body is six lines and needs only relocating
to an import-only module, since `glm/estimator.mojo` imports `std.gpu` at
`:37`. (d) `predict` thresholds in Python (`linear_model.py:955`, `s > 0.0`
over `scores.tolist()`), `predict_log_proba` is a Python `math.log` loop
(`:965-981`). Estimate 4 to 6 hours, the same shape as ols plus the
relocated sigmoid; if `qn_decision_function`'s kernel is the pinned gemv's
fold order it shares the ols restatement.

**tsvd (`TruncatedSVD.transform`).**
(a) No save/load. (b) `components_` `<f4` `(n_components, n_features)`
(`decomposition.py:448`), `n_components_`, `n_features_in_` (`:454-455`);
`singular_values_` is not read by `transform`; output `<f4` (`:464`).
(c) `tsvd_transform` (`decomposition.py:465` to
`bindings/_mojolearn_estimators.mojo:325`, `DeviceContext` at `:340`, then
`tsvd_transform_host`, `decomposition/estimator.mojo:265-281`), whose body
is buffer copies and ONE `gemm_nt(ctx, out, x, components, n_rows,
n_components, n_features)` at `:280`. `gemm/checks/gemm_oracle.mojo` is the
GPU-free host reference of the IDENTICAL GEMM the identity gates check
bitwise against the device, so the host entry is the oracle called on the
saved components; whether `gemm_nt`'s IDENTICAL path (`core/gemm.mojo:150`,
with the `GEMM_IDENT_SWAP_537` arm at `:151`) matches `gemm_oracle` for
this shape is what the fixture would measure. (d) None. Estimate 3 to 4
hours.

**pca (`PCA.transform`).**
(a) No save/load. (b) `components_` `<f4` (`decomposition.py:295`),
`mean_` `<f4` (`:296`), `singular_values_` `<f4` (`:299`, read only on the
whiten arm at `:332`), `n_components_`, `n_features_in_` (`:306-307`);
output `<f4` (`:322`). (c) `pca_transform` (`decomposition.py:337` to
`bindings/_mojolearn_estimators.mojo:181`, `DeviceContext` at `:197`, then
`pca_transform_host`, `decomposition/estimator.mojo:196-215`, then
`pca_transform`, `decomposition/impl/linalg/detail/pca.mojo:319`, imports
`std.gpu` at `:5` and `max.gpu.host` at `:7`), a kernel that subtracts
`mean_` and multiplies by `components_`; the whiten arm is a second entry
(`pca_whiten_transform`, `decomposition.py:329`). No host oracle for the
transform exists under `decomposition/checks/` (`jacobi_eigh.mojo` is the
only GPU-free file there, `:42 from std.math import sqrt`, and it is the
fit's). (d) The whiten arm calls `all_finite` twice (`:326`, `:334`), which
the host binding resolves. Estimate 4 to 6 hours, the centering kernel's
fold order read off `pca.mojo:319` and restated beside the GEMM oracle.

**kde (`KernelDensity.score_samples`).**
(a) No save/load. (b) `_x` `<f4` `(n_train, n_features)` (`density.py:435`),
`_w` `<f4` or None (`:456/:458`), `n_features_in_`, `n_samples_fit_`
(`:436-437`), `bandwidth`, `kernel`, `metric` (`:293-296`); output `<f4`
(`:470`). (c) `kde_score_samples` (`density.py:472` to
`bindings/_mojolearn_estimators.mojo:514`, no context in the binding, then
`kde_score_samples_host_ptr`, `kde/estimator.mojo:167`, `DeviceContext` at
`:202`, then `score_samples`, `kde/impl/kde.mojo:53`, imports `max.gpu.host`
at `:29`; the per-element kernel arithmetic is
`kde/impl/neighbors/kernel_density.mojo`, `std.gpu` at `:101`). The host
oracle `kde/checks/kde_oracle.mojo` (`oracle_score_samples` at `:256`,
`reference_score_samples_f64` at `:463`, `oracle_logsumexp_row` at `:224`,
the halving-tree row norms at `:94-115`) is a complete host restatement
already, GPU-entangled only through constant imports (`:46 NORM_TPB` from
`core/row_norms.mojo`, `:55` from `kernel_density.mojo`); relocating those
constants makes it importable. (d) None in `score_samples`. Estimate 8 to
12 hours, most of it the fixture and proving the oracle's reductions are the
device's for a saved model rather than the check's synthetic one.

**knn, knn-clf, knn-reg (`kneighbors`, `predict`, `predict_proba`).**
(a) No save/load. (b) `_index` `<f4` `(n_index, n_features)`
(`neighbors.py:470`), `n_samples_fit_`, `n_features_in_` (`:471-472`); the
classifier adds `_y_cols` `<i4` `(n_outputs, n_index)` (`:692`) and
`_classes_list` (`:697`); the regressor `_y_cols` `<f4` (`:859-867`);
outputs `dist` `<f4`, `ind` `<u4` (`:541-542`), labels `<i4`, proba `<f4`
(`:725-726`), regression `<f4` (`:891`). (c) Brute arm `knn_search`
(`neighbors.py:546` to `bindings/_mojolearn.mojo:183`, `DeviceContext` at
`:229`, then `neighbors/estimator.mojo:316`, then `brute_force_knn_impl` at
`:573`, then `neighbors/impl/detail/knn_brute_force.mojo`, imports
`max.gpu.host` at `:69`, `core.gemm.gemm_nt` at `:76`, `core.row_norms` at
`:74`, the fused L2 kernel at `:191` and the radix and warpsort selects at
`:183-187`); the rbc arm (`neighbors.py:515`, `rbc_knn_search`) is a second
algorithm with its own kernels (`neighbors/impl/ball_cover/knn.mojo:306-310`);
the classifier and regressor votes are `selection_knn_classify` and
`selection_knn_regress` (`neighbors/impl/selection/knn.mojo`, `std.gpu` at
`:75`). No GPU-free host oracle. `neighbors/checks/metric_oracle.mojo` is
distances only and imports `core/row_norms.mojo` (`:40`), and every
brute-force reference lives in files that import `max.gpu.host`
(`knn_check.mojo:46`, `ball_cover_knn_check.mojo:120`). The identity risk is
not the distances (row norms plus a GEMM, the oracle again) but the
selection. A top-k with the device's tie order and the fused kernel's
distance rounding must be restated statement for statement, and the
classifier's vote and argmax on top of it. (d) `rind.min() < 0`
(`neighbors.py:534`), `astype("<i8")` on the indices (`:537-563`), the
class-set cross-check (`:749-760`) and a proba slice copy (`:778`), all
Python, none float arithmetic. Estimate 12 to 20 hours for the brute arm
with the three surfaces, the rbc arm excluded.

**svc (`SVC.decision_function`, `SVC.predict`).**
(a) No save/load. (b) `dual_coef_` `<f4` `(1, n_SV)` (`_svm_impl.py:540`),
`support_vectors_` `<f4` `(n_SV, n_features)` (`:538-539`), `intercept_`
`<f4` `(1,)` (`:533`), `_label0`, `_label1`, `_gamma` Python floats
(`:541-543`), `n_support_`, `n_features_in_`, `classes_` (`:529-531`);
output `<f4` (`:568`). (c) `svc_predict` (`_svm_impl.py:573` to
`bindings/_mojolearn_svm.mojo:183`, which reads the buffers into host Lists
at `:236-243`, then `svc_predict_host`, `svm/estimator.mojo:283`,
`DeviceContext` at `:349`, then `svc_predict`, `svm/impl/svc_impl.mojo:475`,
imports `std.gpu` at `:32` and `max.gpu.host` at `:34`), the kernel matrix
in batches sized by `buffer_size_mib` and their `applyPrediction` epilogue.
The host oracle `svm/checks/smo_oracle.mojo:814 smo_oracle_decision` is the
sum `sum_j alpha_j K(x, sv_j) + b` with `_kernel_cell` at `:200` and
`identical_exp` from `checks.numerics`, GPU-entangled through
`svm/impl/smosolver.mojo` (`fold_order_for`, `hash_f32_list`) at `:72`. The
question a fixture has to answer is whether the batched device sum's order
is independent of the batch size, because a host entry with no batches
reproduces one order only. (d) `predict` maps the returned float label in
Python (`_svm_impl.py:596-598`); `decision_function` returns the Array as
is. Estimate 10 to 16 hours.

**iforest (`IsolationForest.score_samples`, `predict`).**
(a) No save/load, and nothing to save. `_iforest_impl.py:338-340` and
`:380`, "the fit happens here, every time, on the training matrix `fit`
kept" (DEVIATION 874): the fitted state is `_x` `<f4`, the whole training
matrix, plus the constructor's seed and sizes; the trees exist only inside
`iforest_run_host` (`isolation_forest/estimator.mojo:336`, `DeviceContext`
at `:397`) for the length of one call, and `offset_` and `max_samples_` are
rewritten by every scoring call (`:364-365`). (b) `_x` `<f4`, `_seed`,
`_max_samples_*`, `_max_features_*`, `bootstrap`, `_contamination*`
(`:304-311`); outputs `values` `<f4`, `labels` `<i4`, `info` `<f8` `(3,)`
(`:353-355`). (c) `iforest_run` (`_iforest_impl.py:356` to
`bindings/_mojolearn_svm.mojo:423`, then `iforest_run_host`, then
`isolation_forest/impl/isolation_forest.mojo`, `std.gpu` at `:63`,
`max.gpu.host` at `:64`, the device tree builder
`isolation_tree_builder.mojo`, `std.gpu` at `:122`). The host oracle
`isolation_forest/checks/if_oracle.mojo` is a complete host fit and score
(`oracle_fit` at `:239`, `oracle_path_lengths` at `:337`, `oracle_scores` at
`:356`, over the GPU-free `isolation_forest/impl/rng/xorwow.mojo`),
entangled through its imports of the device implementation (`:42`, `:48`).
A host predict therefore needs either a model export the GPU binding does
not have (the trees, the sample sizes and `offset_`, materialized once
after the device fit and written by a new `save`), or a host refit through
the oracle, which is a second fit and not an inference entry. (d) `float(info[0])`
and `int(info[1])` read-backs (`:364-365`). Estimate 20 to 30 hours, and the
first day of it changes the GPU side before any host code exists.

### Ranked, cheapest first

| rank | lanes | hours | why |
| --- | --- | --- | --- |
| 1 | ols, ridge | 4 to 6 | one serial `ftz(identical_mul_add)` loop per row plus a scalar; `checks.numerics` is GPU-free |
| 2 | tsvd | 3 to 4 (after 1) | one `gemm_nt`; the GPU-free GEMM oracle is the contract reference |
| 3 | lasso, elasticnet | 3 to 4 (after 1) | the same gemv over column-major X plus the intercept |
| 4 | logistic | 4 to 6 (after 1) | one dot product per row plus a six-line host sigmoid that only needs relocating |
| 5 | pca | 4 to 6 (after 2) | centering kernel plus the GEMM; whiten arm a second entry |
| 6 | kde | 8 to 12 | a complete host oracle exists, entangled by constants; reductions are halving trees to reproduce |
| 7 | svc | 10 to 16 | the batched kernel-matrix sum's order is the open question |
| 8 | knn, knn-clf, knn-reg | 12 to 20 | distances are a GEMM; the top-k selection and its tie order are the work |
| 9 | iforest | 20 to 30 | no fitted model exists to save; a GPU-side export comes first |

Lanes 1 through 5 share one restated gemv and the GEMM oracle and together
are about 20 hours plus the save/load each needs; they would take every
`infer` column but knn, svc, kde and iforest onto the CPU. Every estimate
above is for reproducing the GPU bits and measuring it through
`tools/forest_host_gate.py`; a host entry that computes the right answer
in a different order is not what any of these hours buy.

## Classical host inference, lanes 1, 2, 4 and 5 (2026-09-13 evening, branch `lane/classical-host-inference`)

Lane 3 (lasso, elasticnet) was already done by phase 1 (see the corrected
paragraph above) and was skipped. The other four landed as one host binding
extension, one restatement module, one gate and one loader:

- `core/classical_host_predict.mojo`: `host_pinned_cell` MIRRORS
  `pinned_gemm_nt_kernel` (`core/gemm.mojo:33-62`; `pinned_gemv_n_kernel`,
  `:93-113`, is the same fold over one row, and `gemm_nt` routes `n == 1`
  there), and on top of it `host_ols_predict` (`ols_predict_host` plus the
  `_add_scalar_kernel` epilogue, `glm/estimator.mojo:53-74, 192-231`),
  `host_qn_decision` (`linear_fwd` at C == 1 plus `add_bias_kernel`,
  `glm/impl/qn/glm_base.mojo:121-134, 198-244`; the bias read is NOT
  flushed, only the sum), `host_qn_sigmoid` (`qn_sigmoid_host`,
  `glm/estimator.mojo:384-410`, relocated), `host_pca_transform`
  (`shift_columns_kernel` at sign -1.0, `core/column_stats.mojo:153-204`,
  then the gemm) and `host_tsvd_transform` (one gemm). Sabotage define
  MOJOLEARN_HOST_SABOTAGE, the phase 1 spelling: every k loop walked
  descending.
- `bindings/_mojolearn_estimators_host.mojo` exports `ols_predict`,
  `tsvd_transform`, `pca_transform`, `qn_decision_function`, `qn_sigmoid`
  under the GPU binding's names and params lists; the whiten pair, every
  fit and `inverse_transform` stay absent and refuse by name.
- `save`/`load` on LinearRegression and Ridge (`mojolearn-linear-1`:
  `coef` `<f4`, `intercept` `<f8`, `meta` [n_features_in_, fit_intercept],
  Ridge adds `alpha` `<f8`), LogisticRegression (`mojolearn-logistic-1`:
  `w` `<f4`, `classes`, `meta`), TruncatedSVD (`mojolearn-tsvd-1`) and PCA
  (`mojolearn-pca-1`, whiten flag stored; the host transform refuses a
  whitened model by name). `_serialize.write_npz`, exact dtypes, no cast on
  load; `numeric_mode` persisted as GradientBoosting persists it.
- `python/mojolearn/_classical_host.py`: `host_model(path)` returns a HOST
  SUBCLASS of the saved class whose `_bind` answers the CPU binding, so the
  Python predict is the GPU class's own code and only the binding differs;
  `mojolearn.host_model` dispatches the four formats there.
- `tools/classical_host_gate.py`, over `tools/identity_break.py`'s own nine
  fixtures and held-out rows, so its `identity_hash` IS the identity
  tool's `infer` cell and one Mac run is judged against every committed
  GPU column (`--gpu-column`).

Measured on this Mac (Apple M4, Metal record, host check, 45 fixtures =
5 lanes x 9), `bench/results/classical_host/2026-09-13-apple-m4/`:

| check | verdict | evidence |
| --- | --- | --- |
| host vs the Metal recording, every surface (predict, predict_proba, decision_function, transform: sha256 + dtype + shape) | IDENTICAL, 45/45, 243 EQUAL lines, 0 DIFFER | `check_apple-m4_host.json` |
| host `identity_hash` vs the 2026-09-13_46-lanes `infer` cells of apple-m4, nvidia-h100-sm_90a AND amd-mi325x-gfx942 | EQUAL on all 135 (45 x 3) | same file, `columns` |
| sabotage set (`-D MOJOLEARN_HOST_SABOTAGE=1`, `--expect-mismatch`) | EXPECTED MISMATCH SEEN: 234 DIFFER; the only 9 EQUAL cells are logistic `predict` labels, whose sign survives the reordered fold while every proba, score and transform cell differs | `check_apple-m4_sabotage.json` |
| GPU-path reload (`type(est).load` then the same probe) | equal on every fixture, or `record` would have exited 1 | each `expected.json`, `reload_equal` |
| `tools/identity_break.py` over the five lanes with the new `model` column | 45 cells, train/infer/model stable=45, RELOAD-MOVED 0 | `identity_break_apple-m4_five-lanes.{json,txt}` |

Example cells: ols/base infer `2546a13c03838433`, ridge/base
`a5a404bb201b0eeb`, tsvd/base `36793a22ecfc11d9`, logistic/base
`103f76e4a2b23c03`, pca/base `5ec98c317c9314a6`, each the value the three
GPU columns already carried.

What this is and is not. The three-vendor comparison is on the
`identity_hash` (the identity tool's 16-hex digest of the probe output)
against JSONs recorded with models FITTED on those GPUs; the byte-level
sha256 comparison of every surface is against a Metal recording only. The
host binding was built and checked on ONE CPU (Apple M4); the seven-runner
CPU gate does not run this gate yet. OWED when this was written: a
`record` on an NVIDIA box and an AMD box, checked on a CPU box, and a
workflow step that runs `check` on the seven runners. The first two were
DONE 2026-09-14 (6796ceff9): `bench/results/classical_host/2026-09-14-nvidia-h100/`
and `2026-09-14-amd-mi300x/`, recorded at 5e7acf79 and each checked on the
Mac's CPU path IDENTICAL against the three 2026-09-14 GPU columns, 45
fixtures each. The workflow step is still owed.

## k-NN host inference, lane 8 (2026-09-14, branch `lane/knn-host-inference`)

The three k-nearest-neighbor lanes of `tools/identity_break.py` (knn,
knn-clf, knn-reg) predict on a CPU from a saved model, through the same
host pattern as the classical lane. What landed:

- `core/knn_host_predict.mojo`: the GPU-free restatement. Under IDENTICAL
  the GPU search is `knn_search_traced` with AUTO pinned to the TILED arm
  on every column (DEVIATION 509), and every distance spelling and every
  selector of that arm is written to one contract, which the file states
  once and names by file and line: `host_row_norm` MIRRORS
  `row_norm_kernel` (`core/row_norms.mojo:66-113`, NORM_TPB = 128 strided
  partials of `ftz(identical_mul_add(v, v, acc))`, then
  `pinned_block_sum`'s halving tree `red[t] = red[t] + red[t + step]`,
  `core/pinned_reduce.mojo:95-125`, then `ftz`); `host_l2_expanded_cell`
  MIRRORS `pinned_distance_tile_kernel` (`neighbors/checks/
  pinned_distance_tile.mojo:66-107`: the ascending feature chain, the
  `-2 acc + (qn + yn)` fma epilogue, the clamp at zero, `identical_sqrt`
  when the metric roots); `host_composite_key` MIRRORS `composite_key`
  over `twiddle_in` (`select_radix_identical.mojo:102-119`,
  `select_radix.mojo:155-170`); `host_select_k` returns the k smallest
  keys (what the small-k selector, the radix rank pass and the partial
  merge all return) and then runs the estimator's own insertion sort by
  `(distance, index)` (`neighbors/estimator.mojo:664-682`);
  `host_unique_labels`, `host_monotonic`, `host_class_probs`,
  `host_class_vote` and `host_regress_avg` MIRROR `getUniquelabels`,
  `make_monotonic` plus the subtract-one, `class_probs_kernel`,
  `class_vote_kernel` and `regress_avg_kernel`
  (`neighbors/impl/label/classlabels.mojo`, `neighbors/impl/selection/
  knn.mojo`); `host_distance_weights` and the two weighted kernels are
  `distance_weights.mojo`'s, relocated. The host computes the L2 expanded
  pair (euclidean/l2, sqeuclidean) and both weightings; cosine, L1, Linf,
  L2 unexpanded, Lp and the ball cover arm refuse BY NAME.
- `bindings/_mojolearn_core_host.mojo` exports `knn_search`, `knn_classify`
  and `knn_regress` under the GPU binding's names, `params` lists and
  `dist_params` triple, so `neighbors.py` runs unchanged; the returned
  "query tile" is 1. `kmeans_fit`, `rbc_knn_search` and
  `radius_neighbors_*` stay absent.
- `save`/`load` on NearestNeighbors, KNeighborsClassifier and
  KNeighborsRegressor (`mojolearn-knn-1`: `index` `<f4`, `meta` `<i8`
  [n_features_in_, n_samples_fit_, n_neighbors, query_tile, outputs_2d,
  n_outputs], `p` `<f8`, `metric`, `algorithm`, `weights` as text; the
  classifier adds `y_cols` `<i4` (n_outputs, n_samples_fit_), `classes`
  and `class_counts` `<i8`; the regressor `y_cols` `<f4`). `load` rebuilds
  `classes_` from the label columns exactly as `fit` does and refuses a
  file whose `classes` member disagrees.
- `python/mojolearn/_classical_host.py`: `HostNearestNeighbors`,
  `HostKNeighborsClassifier`, `HostKNeighborsRegressor`, whose `_bind`
  answers `_mojolearn_core_host` (the `_HostBound` base now binds the
  class's own family); `algorithm='rbc'` is refused at load by name.
- `tools/classical_host_gate.py` LANES gains knn (identity probe
  `kneighbors(Xh[:64])`, surfaces `kneighbors_distances` and
  `kneighbors_indices`), knn-clf (`predict`, `predict_proba`) and knn-reg
  (`predict`).

Measured on this Mac (Apple M4, Metal record, host check, 27 fixtures =
3 lanes x 9), `bench/results/classical_host/2026-09-14-apple-m4-knn/`:

| check | verdict | evidence |
| --- | --- | --- |
| host vs the Metal recording, every surface (distances, indices, predict, predict_proba: sha256 + dtype + shape) | IDENTICAL, 27/27, 45 surface cases EQUAL, 0 DIFFER | `check_apple-m4_host.json` |
| host `identity_hash` vs the 2026-09-14_46-lanes `infer` cells of apple-m4, nvidia-h100-sm_90a AND amd-mi300x-gfx942 | EQUAL on all 81 (27 x 3), 0 ABSENT | same file, `columns` |
| GPU-path reload (`type(est).load` then the same probe) | equal on every fixture, or `record` would have exited 1 | each `expected.json`, `reload_equal` |
| sabotage set (`-D MOJOLEARN_HOST_SABOTAGE=1`: every distance chain descending, the selection key's tie toward the HIGHER index, the vote's tie to the LAST class, the mean's slots descending; `--expect-mismatch`) | EXPECTED MISMATCH SEEN: 136 DIFFER, 17 EQUAL. Every EQUAL cell is one the sabotage cannot reach: `ties` distances (integer-grid dot products are exact in any order, and on that fixture the INDICES differ, the reversed tie order caught), the indices of the eight fixtures with no tie at the k-th boundary, and `predict_proba` on those eight (a uniform tally is order-invariant). knn-clf `predict` differs on all nine, knn-reg `predict` on all nine | `check_apple-m4_sabotage.json` |
| the sabotage set without MOJOLEARN_HOST_ALLOW_SABOTAGE=1 | refused by name, exit 2 | `load_host_module` |

Example cells: knn/base infer `da53642fe53493ad`, knn-clf/base
`8381d6badb2ee31b`, knn-reg/base `12b3c8b2763ae645`, each the value the
three GPU columns already carried.

What this is and is not. The three-vendor comparison is on the
`identity_hash` against JSONs recorded with models FITTED on those GPUs;
the byte-level comparison of every surface is against a Metal recording
only. What the nine fixtures exercise is euclidean, uniform weights, one
output, k = 8 over a 4096-row index; the sqeuclidean metric, the
`weights='distance'` arm and multi-output `y` are restated from the same
kernels but NOT measured by this gate. The host binding was built and
checked on ONE CPU (Apple M4). OWED: a `record` on an NVIDIA box and an AMD
box, the classical lane's leg body with the lanes changed:

    MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=/root/mojolearn/python \
      pixi run python tools/classical_host_gate.py record \
      "$OUT/2026-09-14-$LABEL" --lanes knn,knn-clf,knn-reg

then, on the Mac, `check` of each directory against the three
2026-09-14_46-lanes columns into
`bench/results/classical_host/2026-09-14-<vendor>-knn/`.
