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

See section 5 for the Apple data point recorded on this Mac. The NVIDIA and
AMD recordings are OWED. Until they exist their fixture directories hold an
`expected.json` whose `status` is `OWED`, which the gate reports as exit 2
and the workflow fails on.
