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
