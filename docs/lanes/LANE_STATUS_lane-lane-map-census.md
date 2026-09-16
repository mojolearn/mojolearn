# lane/lane-map-census: the two things `lane/lane-selector` owed to a person

Branch `lane/lane-map-census`, from `main` at bd5324742.
Worktree `/Users/andrewhendel/mojolearn-wt/lane-map-census`.
Evidence in `/Users/andrewhendel/mojolearn-evidence/lane-map-census-2026-09-16/`.

`lane/lane-selector` mechanised two inversions over the whole tree and then
stopped, saying it had not mechanised JUDGEMENT. It left two items "owed to a
person, not to another rule". This is that pass. Both were real, and between
them they cost five under-attributions, each of which reads as a narrow PASS
rather than as a sweep.

`tools/lane_select.py` was checked before editing: `lane/lane-selector` is
merged, `git diff main lane/lane-selector -- tools/lane_select.py` is empty and
no process held it.

## ITEM 1. `core/forest_inference.mojo`: a forward-walk gap, not correct as is

That file carries the forest prediction kernels and was in NO lane's map. The
question was whether any lane's cells can move when it changes. They can, and
the answer is measured rather than read off the import graph.

**The shipped artifact carries its kernels.** `strings` over
`python/mojolearn/_mojolearn_rf.so` and `_mojolearn_trees.so`, and over the
IDENTICAL build of the same:

    core_forest_inference_forest_g6A6A6A6A_8899ba0f8d34d002
    core_forest_inference_forest_v6A6A6A6A_367803409b38b2da
    core_forest_inference_forest_v6A6A6A6A_3849e0e6a65e26ad
    core_forest_inference_forest_v6A6A6A6A_86589f5c810d858b
    forest_inference / forest_inference_binding / forest_inference_model

THE FIRST CONTROL WAS INVALID and said so by finding nothing everywhere. It
named `python/mojolearn/_mojolearn.so`, `_mojolearn_estimators.so` and
`_mojolearn_forest_host.so`, which do not exist in that directory: `strings`
failed and its empty output read exactly like a clean probe. Redone over
`python/mojolearn/identical/`, where all 17 bindings exist, the blobs are in
`_mojolearn_rf.so` and `_mojolearn_trees.so` and in NEITHER of the other 15.
The probe can fail. The full listing is in
`forest_inference_shipped_probe.txt`.

**A lane's own code reaches it at run time.** On Metal, through
`mac_slot.sh metal`, with the IDENTICAL bindings, a
`RandomForestClassifier(n_estimators=16, class_weight="balanced",
inference_engine="parallel_groves")`, which is exactly what
`@lane("rf-clf-balanced-parallel")` builds (`tools/identity_break.py:1766`):

| arm | exports called |
|---|---|
| `inference_engine="parallel_groves"` | `forest_prepare_gpu` 2, `forest_predict_resident_reuse_gpu` 2, `forest_release_gpu` 1 |
| default (sequential), the control | `rf_predict_proba` 2, and none of the three |

Those three exports are registered by `bindings/forest_inference_binding.mojo`,
which imports `core/forest_inference_model.mojo`, whose `_predict_into_buffers`
calls `launch_forest_inference` at `core/forest_inference.mojo:203`, and whose
`resident_prepare` calls `validate_flat_forest` at `:258`. Two other lanes,
`et-reg-bootstrap-parallel` and `par-forest-pool`, take the same path.

Three rules each hid it, and each is a rule rather than an exception.

1. **A Mojo import resolves against the importing file's own directory too.**
   Every binding is built `-I . -I bindings` (`bindings/build_rf.sh:123`,
   `build_trees.sh:134`, `build_gbdt.sh:268`), so
   `from forest_inference_binding import ...` inside
   `bindings/_mojolearn_rf.mojo` means `bindings/forest_inference_binding.mojo`.
   Root-only resolution found no such file and dropped the import as one of the
   toolchain's own. `tools/bincache.py:198`, written for the binding cache
   against the same compiler, already searched `list(roots) + [importer.parent]`.
   Six files and eleven targets were invisible this way, among them the whole
   `gbdt/gpu_lib/` tree.
2. **A parametrized `def_function` is still an export.**
   `def_function[forest_prepare_gpu_binding[True]]("forest_prepare_gpu")` is the
   ordinary spelling here. Requiring a bare identifier dropped 35 exports across
   five bindings, among them the FIT entry points `rf_classifier_fit` and
   `et_classifier_fit`. A dropped export is not a wide answer: the lane still
   hits other exports, so the per-export branch runs and the dropped export's
   tree is simply absent.
3. **An export reaches what its impl reaches.** `forest_prepare_gpu`'s impl is
   IMPORTED, so `blocks.get` returned `""` and the export contributed nothing.
   `rf_predict_proba_gpu_parallel_binding` calls the file-local
   `_rf_predict_gpu_parallel`, and only that helper names `forest_predict_gpu`.

## ITEM 2. The census at 3, read

67 files at the start, 59 now. Most are genuinely narrow and narrow for a
reason that is in the tree, not in a list. Three were not.

### Confident: a missing edge (all four fixed)

| file | was | now | why it was wrong |
|---|---|---|---|
| `core/forest_inference.mojo` and its tree | not in the map | 23 | item 1 |
| `python/mojolearn/UMAP.py`, `HDBSCAN.py` | 3 and 24 | gone | neither path is tracked. `_python_imports` asked `os.path.exists` whether an imported NAME is also a module, and this checkout's filesystem is case-insensitive. The map carried two files this repository does not have, and on the Linux boxes that run the CPU column the same map was a different map |
| `python/mojolearn/umap.py`, `neural_network.py`, `language_model.py` | not in the map | 2, 2, 11 | a lane writes `ml.UMAP`, and `__init__.py:147` binds that attribute with `from .umap import UMAP`. Seeding only the file that DEFINES the class walks past the public door that rebinds the name |
| `umap/graph.mojo`, `sparse_graph.mojo`, `estimator.mojo` | 1 | 4 | `bindings/_mojolearn_metrics.mojo:56` is `from umap.estimator import fit_transform as umap_fit_transform` and the export calls the ALIAS. The map recorded the original name, and `\bfit_transform\b` does not match inside `umap_fit_transform`. The whole umap tree was invisible to the per-export scan and reached the `umap` lane only because the metrics HOST family happens to list `umap/graph.mojo` among its host modules; `par-graph-umap`, which runs the same fit across devices, was credited with none of it |

### Confident: correct as is

* **The tokenizer, embedding and IVF families** (`tokenizer/impl/*`, `embedding/host/*`, `ivf/**`, and their doors) sit at 2 or 3 because the registry has exactly 2 tokenizer lanes, 2 embedding lanes and 3 IVF lanes. The count is the family, not an accident.
* **`core/gbdt_host_ctr.mojo` and `bindings/build_forest_host.sh` at 2.** They are reached only from `bindings/_mojolearn_forest_host.mojo`, and `host_surface.py`'s `forest` family declares exactly `gbdt-categorical-ctr-tables` and `gbdt-tensor-ctr-tables`. The third CTR-shaped lane, `gbdt-categorical-ctr`, says in its own docstring that NO CTR IS BUILT HERE.
* **The host oracles that omit their `par-*` siblings** (`hdbscan_host_oracle.mojo` 2, `km_host_oracle.mojo` 3, `cd_oracle.mojo` 3, `resample_host.mojo` 3, `spectral_oracle.mojo` 3, `if_oracle.mojo` 2, `umap/host/umap_oracle.mojo` 1). Each is linked only into `*_host` bindings (measured, by taking each binding's import closure), and only 13 of the 50 `par-*` lanes are named by any host family. `host_surface.py` is the declaration of which lanes have a CPU route (`_verify_all.py:126`, "every CPU training lane is declared there"), and the map reads it live, so the day a `par-*` lane is added to a family the map follows without an edit here. Widening the map to guess at it would be a hand-kept exception.
* **`python/mojolearn/linalg.py` at 2.** It re-exports `Cholesky`, but `__init__.py:178` binds `ml.Cholesky` straight from `._cholesky_impl`, so `linalg.py` is not on that lane's path. It holds `gemm-pinned` and `gemm-transposed`, which are the lanes that call `ml.linalg.matmul`.
* **None of the 37 census `.mojo` files is a standalone program**, so none of them is out of scope for the reason a `checks/` program would be.

### Not confident

* **`python/mojolearn/model_selection.py` at 3**: `cross-val` and `cross-val-folds` are right and `ivf-extend` looks like a symbol collision rather than a road. It over-fires by one lane, which is the harmless direction, so it is left alone and written down rather than patched.
* **The `par-*` host gap is correct AS DECLARED, and the declaration may be what is wrong.** If a `par-*` lane should have a CPU route, that belongs to whoever owns `host_surface.py`, not here. Several lanes are touching that file today.
* **No perturbation of `core/forest_inference.mojo` was built.** The AIR blobs and the run-time export trace are measured; a build with a deliberate arithmetic break in that file, and the cell hash moving, was not run. It would cost a Metal binding build and would confirm rather than change the verdict.

## Measured

| check | result |
|---|---|
| `--selfcheck` | OK, 0 lanes with an empty side |
| `tools/test_lane_select.py` | 50 tests, 0 failures (42 inherited, 8 added) |
| map, before -> after | 723 -> 792 files, median 33 -> 29 lanes, 41 files selecting every lane, unchanged |
| files that LOST a lane | 0 |
| `cluster/host/kmeans_oracle.mojo` | 20, unmoved |
| `core/gbdt_host_predict.mojo` | 23, unmoved |
| `python/mojolearn/neural_inference.py` | 21, unmoved |
| `core/forest_host_predict.mojo` | 7 -> 15, on purpose: `rf_predict_proba` routes to it and the rf lanes were missing |
| files that answered "212 of 212, falling back" and now answer narrowly | 68 |

Every rule added was run on its unfixed side first and watched to fail: root-only
Mojo resolution must lose `bindings/forest_inference_binding.mojo` while keeping
the ordinary root import; the old `def_function` pattern must miss
`rf_classifier_fit`; `_public_rebindings` blanked must lose all three public
doors; and the ORIGINAL import name must not match the export body that uses the
alias. The case that must STAY narrow is asserted beside each.

## Owed / not done

* No pod rented, no GPU column run. The only GPU work was the one Metal probe
  above, through the slot helper.
* `tools/verify_lanes.py` was not touched.
* The `par-*` CPU-route question is handed on, not answered.
