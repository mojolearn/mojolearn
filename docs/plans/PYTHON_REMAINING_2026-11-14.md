# Python remaining (2026-11-14)

Read-only audit of `origin/preinteg/wave2-20261113` at 8ab734d820 by lane `python-package-audit`. Nothing compiled,
imported or run; every count is a static scan (`git grep`, an AST walk of `python/mojolearn`, a function-level regex
walk of the Mojo boundary files). Scripts, logs and per-function tables:
`~/mojolearn-evidence/briefs-20261114/python-package-audit/` (`py_scan.py`, `py_post.py`, `mojo_door_scan.py`,
`py_functions.tsv` 3,820 rows, `py_modules.tsv` 307 rows, `mojo_pystructs.tsv` 325 rows,
`mojo_entangled_run_in_door.tsv`, `mojo_entanglement_scan.log`, `mojo_import_check.log`, `python_free_dirs.log`).

Scope of the Mojo side: 5,107 product `.mojo` files (1,527,706 lines; `experiments/`, `bench/`, `tests/` excluded).
Boundary files by the fence's own patterns (`tools/python_free_dirs.sh:67-69` plus the one `allow:` line in
`tools/python_free_doors.txt`): 1,031 files, 267,475 lines; 1,029 of them import Python. Non-boundary: 4,076 files,
of which 91 under `bindings/` and `providers/python/` (11,416 lines; `*_run.mojo`, entry tables, typed helpers).

| boundary pattern | files | lines |
|---|---:|---:|
| `*_door.mojo` | 360 | 174,199 |
| `*_unpack.mojo` | 259 | 23,618 |
| `*_pack.mojo` | 213 | 19,698 |
| `bindings/_mojolearn*.mojo` (PUBLISH) | 90 | 46,269 |
| `bindings/entry/*.mojo` | 92 | 1,163 |
| `bindings/py_*.mojo` | 16 | 2,473 |
| `allow:` (`bindings/spectral_affinity_boundary.mojo`) | 1 | 55 |

## 1. The three entanglements

### 1.1 Kits and program builders with Python fields

| struct (RUN_SPLIT_INVENTORY §Entanglements 1) | today | Python field passed into RUN under another name? |
|---|---|---|
| `DecompKit` (was `bindings/decomp_kit_native.mojo`, 1,203 lines) | deleted; typed kit in `bindings/decomp_kit_run.mojo:3` ("was the Python-holding `DecompKit`"); no `struct DecompKit` anywhere; the name survives only in 4 comments (`x_decomp/descriptor_schema.mojo:71`, `x_decomp/randomized_svd_owner.mojo:7`, `x_decomp/resident.mojo:310`, `bindings/decomp_kit_run.mojo:1067` TOMBSTONE) | no |
| `MKit` (was `bindings/mani_kit.mojo`, 1,117 lines) | deleted; typed in `bindings/mani_kit_run.mojo:3`, `bindings/mani_decomp_run.mojo:6`, `bindings/mani_umap_init_run.mojo:4`; comments only | no |
| `LinKit` (611 lines) | typed: `bindings/linalg_ops_kit_run.mojo:84` `struct LinKit(Movable)` holds `entry: EntryRef`, `List[KitMatrix]`, `List[KitStore]`, ints and Bools; the door's UNPACK passes `entry` and the capability word (`:95-96`) | no |
| `EncProg` | `bindings/prep_encode_native_door.mojo:164`: `prog: EncProgram` (typed, `bindings/prep_encode_prog_run.mojo`) + `owners: List[PythonObject]`; RUN receives `EncProgram` and `XprepInput` records (`prep_encode_prog_run.mojo:15`) | no: the owner list stays in the door |
| `ClsProg` | `bindings/prep_classify_prog_door.mojo:77`: `program: ClsProgram` (typed, `bindings/prep_classify_prog_run.mojo`) + `owners: List[PythonObject]` + `store: PythonObject` + `base: Int`; RUN takes `ClsProgram` | no |
| `MxProg`, `_Prog` | no struct of either name in product Mojo; `_Prog` appears in 8 non-boundary files as comments naming the Python-side `_expansion_prep._Prog` layout (`bindings/prep_encode_prog_run.mojo:5,17,20`) | n/a |

Done at the type level. What remains is RUN-shaped code still sitting inside door files and taking the Python-holding
wrapper instead of its typed program (from `mojo_entangled_run_in_door.tsv`: a function in a boundary file that
touches no Python object, has at least two typed-work lines, and takes a Python-holding struct):

| file:line | function | lines | takes |
|---|---|---:|---|
| `bindings/prep_encode_native_door.mojo:352` | `_sel` | 15 | `EncProg` |
| `bindings/prep_encode_native_door.mojo:367` | `_bad_rows_stages` | 9 | `EncProg` |
| `bindings/prep_iterimp_native_door.mojo:386` | `_ii_rows` | 16 | `EncProg` |
| `bindings/prep_labels_native_door.mojo:231` | `_label_load` | 17 | `EncProg`, `LabelBuf` |
| `bindings/prep_classify_common_door.mojo:487` | `p2m_class_counts` | 14 | `ClsProg` |
| `bindings/prep_classify_native_door.mojo:1456` | `_class_counts` | 14 | `ClsProg` |
| `bindings/metrics_exp_curve_door.mojo:224` | `_fold_stages` | 27 | `MetricDoorProg` |

The one lane that finishes it: **`door-run-split`** (Mojo, mechanical, ~112 lines): each of the seven takes
`EncProgram` / `ClsProgram` / the typed program field instead of the wrapper, moves to the family's `*_run.mojo`
(`prep_encode_prog_run.mojo`, `prep_classify_prog_run.mojo`, `metrics_exp_curve_run.mojo` new), and the door calls it
with `prog.prog` / `prog.program`. Also the four in 1.2 below (same lane, same shape): 11 functions, 227 lines total.

### 1.2 Owner-pinning admitted views used inside RUN

| view | today | Python field into RUN? |
|---|---|---|
| `AdmittedX` | door-only: `bindings/admitted_input_unpack.mojo:155` keeps `array`, `source`, `rows_base`, `rows_indices: PythonObject`; RUN takes `AdmittedXView` (`bindings/admitted_input_run.mojo:3-16`, "the door names the owner again after the last call that reads `address`"); `AdmittedX` is named in 21 boundary files and 1 non-boundary file, `bindings/admitted_input_run.mojo:10-14`, a docstring | no |
| `ArrayMetadata` | `core/array_metadata.mojo:11`, no Python field; used by 42 boundary files, 0 non-boundary | no |
| `AdmittedBorrowedBuffer` | `providers/python/borrowed_buffer_pyapi_unpack.mojo:113`, no Python field in the struct scan; 35 boundary users, 1 non-boundary mention (`core/boundary_layout.mojo:16`, a comment) | no |
| `AdmittedTyped` | one boundary mention; no struct of that name survives | n/a |
| `BufferView` | `bindings/boundary_read_door.mojo:283`, `owner: PythonObject`; used by that file only | no |
| `_RPArray` | door-only: `providers/python/resample_pandas_pyapi_door.mojo:115` holds `owner`, `dtype`, `api_owner`, `na`, `nat`, `time_from: PythonObject`; RUN takes `RPLayout` (`core/resample_pandas_layout.mojo:11`, "No Python object") | no |
| `AdmittedBuf` | `bindings/admitted_input_unpack.mojo:899`, `keep`, `owner: PythonObject`; 7 boundary users, 0 non-boundary | no |

Done at the type level. RUN-shaped functions still taking the Python-holding view inside a door:

| file:line | function | lines | takes |
|---|---|---:|---|
| `providers/python/resample_pandas_pyapi_door.mojo:396` | `_rp_copy` | 46 | `_RPArray` (has `RPLayout` to take instead) |
| `providers/python/resample_pandas_pyapi_door.mojo:1305` | `_rp_sparse_fill_copy` | 29 | `_RPArray` |
| `bindings/prep_encode_native_door.mojo:2759` | `_check_train` | 20 | `IndexI64` |
| `providers/python/autograd_owned_program_factory_pyapi_unpack.mojo:38` | `owned_program_create_devices` | 20 | `NativeOwnedProgramCreateRequest` |

Finished by the same **`door-run-split`** lane.

### 1.3 The Python module cache and resident Python slots

- `NativeModules` (`bindings/native_loaded_module_door.mojo:483`, `_modules: Dict[String, PythonObject]`): named in
  141 boundary files, 0 non-boundary files, 0 `*_run.mojo` files, 0 `mods: NativeModules` parameters in any
  `*_run.mojo` (`git grep -c 'mods?\s*:\s*NativeModules' -- '*_run.mojo'` is empty). Done: no RUN signature carries it.
- Resident `SharedSlot`s whose element type holds a `PythonObject`: 73 sites, all in boundary files (`core/` has
  none). By what the Python fields hold: 31 cache Python types, classes, modules or builtins (e.g.
  `bindings/native_class_admission_door.mojo:213` `StaticTypes`, `bindings/p3_numpy_types_unpack.mojo:39`);
  18 pin providers, capsules, entry dicts or descriptors (e.g. `bindings/kmeans_descriptor_boundary_door.mojo:21`
  `ProviderPins`); 24 hold user data, records or owners across calls. Those 24 are the open part of entanglement 3:

| slot site | struct | Python fields |
|---|---|---|
| `bindings/gp_svm_gp_state_door.mojo:308` | `GpAdmittedInputs` | `items: List[PythonObject]` |
| `bindings/py_trees_resident.mojo:34` | `TreesResidentPins` | `rows: Dict[Int, List[PythonObject]]` |
| `bindings/py_admitted_registry.mojo:54` | `AdmittedPins` | `items`, `parents: List[PythonObject]` |
| `bindings/trees_public_inputs_door.mojo:228` | `_DecodeTables` | `owners`, `items`, `tables: List[PythonObject]` |
| `bindings/trees_side_records_unpack.mojo:254` | `TreesFiniteSources` | `items: List[PythonObject]` |
| `bindings/parallel_shard_common_door.mojo:99` | `ShardTrace` | `records: PythonObject` |
| `bindings/parallel_shard_native_door.mojo:110` | `ParallelShardBoundary` | `tables: PythonObject` |
| `bindings/parallel_neighbor_worker_door.mojo:38` | `NeighborSnapshots` | `models: PythonObject` |
| `bindings/neighbors_knn_native_door.mojo:111` | `_AdmittedToken` | `token: PythonObject` |
| `bindings/numeric_profile_native_door.mojo:69` | `ProfileAdmission` | `context`, `dicts: List[PythonObject]` |
| `bindings/ocsvm_scalar_boundary_door.mojo:47` | `OcScalarTypes` | `constructor`, `arrays`, `scalars` |
| `bindings/samba_admission_door.mojo:208` | `SambaAdmissionShadows` | `table: PythonObject` |
| `bindings/sequential_admission_door.mojo:102`, `:245` | `_SequentialValueTypesCache`, `_SequentialFieldNames` | `contexts`, `sources`, `names` |
| `bindings/tokenizer_public_native_door.mojo:74` | `TokenizerPublicConnection` | `vocab`, `bpe`, `error_handlers`, `error_lookup`, `list_iter` |
| `bindings/causal_operation_owner_door.mojo:40` | `CausalFieldNames` | `model`, `layer_scalars`, `layer_buffers`, `layer_all`, ... |
| `bindings/causal_lm_public_door.mojo:88`, `causal_model_config_door.mojo:27`, `lm_corpus_metadata_door.mojo:23`, `model_catalog_native_door.mojo:32` | `*Connection` | `context: PythonObject` |
| `bindings/byte_checkpoint_native_door.mojo:31` | `ByteCheckpointBoundary` | `bi`, `os`, `magic` |
| `bindings/transformer_public_native_door.mojo:31` | `_GenerationStamp` | `key: PythonObject` |
| `providers/python/transformer_decode_pyapi_door.mojo:361` | `_TransformerDecodeBindings` | `selected`, `binding` |

The `ParallelMlpBoundary` (`bindings/parallel_mlp_native_door.mojo:35`), `GENERATION_BOUNDARY`
(`bindings/generation_buffers_door.mojo:55`), `OptimizerInputOwners` (`bindings/optimizer_container_boundary_unpack.mojo:22`)
and `ContentRecord` (`bindings/py_model_records.mojo:576`) named in the inventory all still hold Python objects, all
inside boundary files, none reachable from a `*_run.mojo`.

Status: done for RUN (nothing typed receives them). Not done as a design: 24 slots keep user data as Python objects
between public calls, which is where a lifetime bug or a hidden Python call survives the fence. The one lane that
finishes it: **`resident-slots`** (Mojo, judgment then mechanical, 24 slots): per slot, either the data becomes a typed
handle in a native registry (`TreesResidentPins.rows`, `_DecodeTables`, `GpAdmittedInputs`, `TreesFiniteSources`,
`NeighborSnapshots`, `SambaAdmissionShadows`) or the slot is shown to be a pure pin that the door names again after the
RUN call and is left with a one-line comment saying so (`*Connection.context`, `_GenerationStamp`, `_AdmittedToken`).

Should they be cleaned up: 1.1 and 1.2 yes, one small lane, because the fence already proves nothing typed imports
them and finishing removes the last RUN work from doors; 1.3 slot by slot, the six data slots first.

## 2. The Mojo side

### 2.1 Non-boundary files that name a boundary type

By import (a non-boundary file importing a name from a boundary module): **0** of 4,076 files for the 325
Python-holding structs; **1** file imports any boundary module at all:
`checks/native_tests/neural_entry_suite.mojo:38-39` (`bindings._mojolearn_embedding_host`,
`bindings._mojolearn_embedding_infer_host`, the `publish_*` entries; a native test suite reaching the PUBLISH module).
By plain name (text match, no import): 64 files, all collisions with unrelated generic names (`Session` in
`autograd/`, `Counts` in `gbdt/`, `metrics/`, `neighbors/`) or comments naming the old design
(`bindings/admitted_input_run.mojo:10-14` `AdmittedX`, `core/resample_pandas_layout.mojo:3-5` `_RPArray`,
`bindings/prep_encode_prog_run.mojo:11` `EncProg`, `bindings/prep_classify_prog_run.mojo:9` `ClsProg`,
`providers/python/neural_descriptor_run.mojo` `NeuralDescriptorProvider`). None is a type use.

### 2.2 Python-holding structs in boundary files, and the RUN code that receives them

325 structs in 206 boundary files hold a `PythonObject` field (`mojo_pystructs.tsv`); 0 are defined outside a boundary
file. Boundary files hold 8,367 functions; 39 of them (733 lines) are RUN-shaped (no Python touch, typed work), and 11
of those (227 lines, the tables in 1.1 and 1.2) take a Python-holding struct. That is the whole of "RUN code receiving
a Python-holding struct" today. The other 28 RUN-shaped door functions take typed values and are candidates for the
same move but violate nothing.

### 2.3 The fence rule that would catch a regression

`tools/python_free_dirs.sh` today: kernel directories import no Python (`:33-35`), `*_run.mojo` and
`core/*_entry_table*.mojo` import no Python (`tools/python_free_doors.txt`), and a Python-importing file under
`bindings/` or `providers/python/` must match a boundary pattern (`:67-69`, `:124-140`). It passes on this base
(`python_free_dirs.log`: "no kernel-directory Mojo file, listed door, *_run.mojo or non-boundary binding imports Python").

Extension (fourth check, same grep style, no baseline): a non-boundary Mojo file may not import a boundary module.
Proposed body, after `boundary_hits`:

```bash
# import_hits: non-boundary Mojo files that import a boundary module (a *_door,
# *_unpack, *_pack, py_*, _mojolearn* or entry/* module). RUN reaches the
# boundary only through typed values and entry tables, never by import.
import_hits() {
    local f mod src
    while IFS= read -r f; do
        [ -n "$f" ] || continue
        is_boundary "$f" && continue
        git grep -h -E '^[[:space:]]*(from|import)[[:space:]]+(bindings|providers\.python)\.[A-Za-z0-9_.]+' -- "$f" \
        | sed -E 's/^[[:space:]]*(from|import)[[:space:]]+([A-Za-z0-9_.]+).*/\2/' \
        | while IFS= read -r mod; do
            src=$(printf '%s' "$mod" | tr . /).mojo
            is_boundary "$src" && printf '%s imports %s\n' "$f" "$mod"
        done
    done < <(git ls-files '*.mojo' ':(exclude)experiments/' ':(exclude)bench/' ':(exclude)tests/')
}
```

with `is_boundary` the `boundary_allow` + `allow:` match already written inline in `boundary_hits` (`:128-137`),
lifted into a function. Violations on this base: **0** for `*_door`/`*_unpack`/`*_pack` imports; **1 file, 2 lines**
when `_mojolearn*` is included (`checks/native_tests/neural_entry_suite.mojo:38-39`). The lane that adds the check
moves that suite's two imports to the entry table the PUBLISH module fills, or lists `checks/` as a test directory.

### 2.4 Toward "not even importing Python except in the interconnect files"

The interconnect files are the 1,031 boundary files. Every other product file (4,076) already imports no Python, and
no typed file imports a boundary file. What is left is inside the boundary set itself: 360 door files hold 174,199
lines, and the largest (`bindings/causal_lm_public_door.mojo` 3,503, `bindings/prep_encode_native_door.mojo` 3,411,
`bindings/host_archive_native_door.mojo` 2,951, `bindings/x_linear_native_door.mojo` 2,646,
`bindings/trees_public_meta_door.mojo` 2,398) are families' whole public surfaces with UNPACK, typed call and PACK in
one file. Shrinking the Python-importing set further means splitting each door into `<stem>_unpack.mojo` +
`<stem>_pack.mojo` + a typed `<stem>_run.mojo`; the inventory priced that at 307 splits + 226 renames on 2026-11-12
and the orchestrator chose `*_door.mojo` instead. Nothing in this audit changes that choice: the measurable
regression risk (RUN code taking Python-holding structs) is 11 functions, section 2.2.

## 3. The Python side (`python/mojolearn`, 307 modules, 113,292 lines, tests excluded)

Method: AST walk, one row per function or method (nested functions counted in the parent), module-level lines
(imports, docstrings, constants, class bodies) assigned to the module's class. Classes:

- (a) connection glue: a function with no validation, no type or fitted check, no data arithmetic, no dispatch on a
  string parameter; plus import-time wiring (modules `_backend`, `gpu_plugins`, `_merged`, `_native_loaded_modules`,
  `_surface_*`, `_version`, `_host_import_aliases`, `_sklearn_modules`, `_runtime_fallbacks`, `host_surface`,
  `_rocm_runtime`, `_native_backend_public`, and every `connect*` / `install*` / `register*` function that runs once
  at import) and module-level lines of product modules.
- (b) Python work the rule forbids: a product-reachable function that raises `ValueError`/`TypeError`/`NotFittedError`
  itself, calls `isinstance`/`hasattr(self, ...)`/`_check_*`/`_validate_*`, reads `.shape`/`.ndim`/`.dtype` to decide,
  converts with `np.asarray`-class calls, dispatches on a string parameter, or does arithmetic or a loop with
  arithmetic on values.
- (c) metadata protocols: `__sklearn_tags__`, `_more_tags`, `get_params`, `set_params`, `__repr__`, `_repr_html_`,
  `__sklearn_is_fitted__`, `__getattr__`/`__dir__`.
- (d) verify (protected): the 46 modules reachable from `mojolearn.__main__` and from no public module
  (`__main__`, `_conformance`, `_families`, `_identity`, `_verify*` x 32, `_verification_*` x 10).
- (e) tooling and test controllers: `_tokenizer_synthetic`, `_gpu_witness`, `_causal_lm_fixtures`, `_bufcheck`,
  `_arena_io`, `_parallel_cv_witness`, `_distributed_program` (part), `_cpu_reference` (part).
- (f) dead or duplicate: a function whose AST equals another's (27 groups, 62 bodies, 462 extra lines) or a private
  name referenced nowhere in `python/`, `tools/`, `bindings/`, `providers/`, `core/`, `checks/`, `docs-site/` (1).

| class | lines | share | functions |
|---|---:|---:|---:|
| (a) connection glue and wiring | 81,364 | 71.8 % | 3,380 |
| (b) Python work the rule forbids | 5,148 | 4.5 % | 308 |
| (c) metadata protocols | 578 | 0.5 % | 65 |
| (d) verify | 24,363 | 21.5 % | 46 modules |
| (e) tooling / test controllers | 1,375 | 1.2 % | 8 modules |
| (f) dead or duplicate | 464 | 0.4 % | 63 |

(a) decomposes into: 25,342 lines in glue functions; 6,295 in import-time `connect`/`install` functions; 3,084 in
functions of wiring modules; 46,643 module-level (docstrings are 37,760 lines of the (a)-class modules by
`ast.get_docstring`; `host_surface.py` alone is 5,234 lines of constant tables; the 21 wiring modules total 10,477).

(b) by reason (lines): validation `raise` 2,721; type or fitted-state check 675; data arithmetic or loop with
arithmetic 1,283; dispatch on a string parameter 373; array preparation 53; shape check 43.

### Top 20 modules by (b) lines, and the Mojo entry that should own each

| module | lines | (b) | (b) fns | largest (b) functions | Mojo owner |
|---|---:|---:|---:|---|---|
| `_buffer.py` | 1,109 | 610 | 22 | `_materialize:414` 116, `_materialize_buffer:532` 58, `Buf.__init__:124` 47, `_native:986` 47 | `bindings/admitted_input_unpack.mojo` (`admit_x`), `bindings/array_native_door.mojo` |
| `_sk_validate.py` | 634 | 320 | 18 | `_check_X:358` 39, `_check_y:428` 36, `describe:259` 32, `after_fit:560` 32 | `bindings/admitted_input_unpack.mojo` (the refusal wording moves beside `admitted_finite_wording`) |
| `neighbors.py` | 1,498 | 295 | 8 | `_resolve_rbc_metric:292` 113, `NearestNeighbors._check_refusals:583` 67, `_resolve_metric:235` 55 | `bindings/neighbors_knn_native_door.mojo` UNPACK (metric enum) |
| `_serialize.py` | 408 | 267 | 14 | `_describe:131` 50, `decode_npy:233` 50, `_parse_header:89` 29 | `bindings/archive_schema_native_door.mojo`, `bindings/classical_archive_read_door.mojo` |
| `_metrics_impl.py` | 1,672 | 251 | 14 | `_native_classification_labels:1151` 42, `_prepare_cluster_labels:266` 29, `_sample_weight_f32:235` 27 | `bindings/metrics_exp_cls_door.mojo`, `metrics_reg_door.mojo` |
| `_labels.py` | 457 | 239 | 11 | `prepare_targets:183` 42, `_encode_labels_native:242` 36, `unique_inverse:423` 35 | `bindings/prep_labels_native_door.mojo`, `bindings/classical_labels_door.mojo` |
| `randomforest.py` | 888 | 172 | 6 | `_RandomForestBase.__init__:339` 82, `_max_features_fraction:270` 32, `_max_leaves_slot:304` 29 | `bindings/trees_public_forest_door.mojo` UNPACK |
| `model_selection.py` | 2,152 | 163 | 11 | `_native_default_folds:233` 41, `_clone:168` 24, `_folds:294` 21 | `bindings/msel_cv_door.mojo`, `msel_split_door.mojo` |
| `extratrees.py` | 656 | 152 | 6 | `ExtraTreesRegressor.__init__:553` 58, `ExtraTreesClassifier.__init__:345` 54 | `bindings/trees_public_forest_door.mojo` UNPACK |
| `models/config.py` | 512 | 148 | 6 | `_transformer_common:254` 57, `_plan_mamba2:442` 31, `_plan_mamba:416` 24 | `bindings/causal_lm_construct_door.mojo` |
| `_forest_protocol.py` | 292 | 136 | 10 | `forest_estimator:48` 40, `ForestProtocol._prediction_engine:178` 21, `_validate_params:107` 18 | `bindings/forest_inference_binding_door.mojo` |
| `_native_classical_bridge.py` | 413 | 121 | 3 | `_state_context:150` 59, `_numpy_free_context:91` 33, `_layout:283` 29 | `bindings/classical_pandas_door.mojo`, `bindings/x_linear_native_door.mojo` |
| `_native_public_attributes.py` | 314 | 114 | 7 | `connect_classes` is wiring; the rest are attribute refusals | `bindings/public_attributes_native_door.mojo` |
| `_linalg_impl.py` | 842 | 101 | 9 | `_operand:223` 63 | `bindings/linalg_native_door.mojo` |
| `_numeric_profile.py` | 507 | 98 | 5 | `default_profile:336` 37 | `bindings/numeric_profile_native_door.mojo` |
| `_classical_host.py` | 1,552 | 95 | 4 | `_connect_host_inference:1413` 91 (wiring-shaped but raises per call) | `bindings/host_inference_native_unpack.mojo` |
| `_expansion_cnn.py` | 990 | 94 | 6 | layer argument checks | `bindings/xcnn_common_door.mojo` |
| `_byte_lm_impl.py` | 947 | 93 | 5 | `_load:168` 38 | `bindings/byte_lm_trainer_native_door.mojo` |
| `_expansion_prep.py` | 3,038 | 92 | 7 | encoder parameter checks | `bindings/prep_encode_native_door.mojo` |
| `_transformer_impl.py` | 858 | 91 | 6 | `TransformerBlock._extension:543` 41 | `bindings/transformer_public_native_door.mojo` |

The full 307-module table is the appendix. 219 modules have zero (b), (e) and (f) lines and are untouched by the
lanes below.

### Lanes that remove (b) and (f), sized by (b) lines

| lane | modules (family) | (b) lines | what moves to Mojo |
|---|---|---:|---|
| `py-admission` | `_buffer`, `_sk_validate`, `_labels`, `_sample_weight`, `_class_weights`, `_array*` | 1,295 | buffer materialization and refusal, X/y shape and dtype refusals, label and weight preparation: into `admitted_input_unpack.mojo`'s `admit_x` / `prepare_targets` as the one refusal path; unblocks every estimator lane |
| `py-trees` | `randomforest`, `extratrees`, `_forest_protocol`, `_expansion_trees`, `_gbdt_*`, `_iforest_*`, `_native_trees_bridge` | 529 | constructor parameter validation (`max_features`, `max_leaves`, bootstrap sizes) into the forest/GBDT door UNPACK |
| `py-neighbors` | `neighbors`, `_expansion_neighbors`, `_native_neighbor_queries` | 397 | metric and weights resolution (string to enum, p) into `neighbors_knn_native_door.mojo` |
| `py-archive` | `_serialize`, `_native_archive_bridge`, `_native_host_archive_*` | 292 | npy/npz header parsing and dtype description into `archive_schema_native_door.mojo` |
| `py-classical-linalg` | `_native_classical_bridge`, `_classical_host`, `linear_model`, `_expansion_linear`, `_linalg_impl`, `linalg_fast`, `_portable_math` | 514 | state-context refusals, operand shape checks, `matmul_fused` argument checks |
| `py-metrics` | `_metrics_impl`, `_expansion_metrics` | 259 | label preparation and sample-weight checks into `metrics_exp_cls_door.mojo` |
| `py-params-mode` | `_native_public_attributes`, `_native_params_metadata`, `_native_full_reset_params`, `_numeric_profile`, `_mode`, `_native_mode_*` | 356 | attribute and mode refusals into `public_attributes_native_door.mojo`, `numeric_profile_native_door.mojo` |
| `py-lm-neural` | `models/config`, `models/causal_lm`, `_byte_lm_*`, `_transformer_impl`, `_mamba_impl`, `_mlp_impl`, `embedding` | 533 | model-plan arithmetic (`_plan_*`) and block argument checks into `causal_lm_construct_door.mojo` |
| `py-msel-prep-cnn` | `model_selection`, `_native_msel_*`, `_expansion_prep`, `preprocessing`, `_expansion_cnn`, `_native_xcnn_bridge` | 380 | fold defaults, encoder and layer argument checks |
| `py-tail` | the remaining 40 modules (`ann`, `gp`, `manifold`, `decomp`, `ops`, `resample`, `tokenizer`, `_ragged`, `_user_objects`, `torch`, `kernel_methods`, `exceptions`, `svm`, `cluster`, `training`, `importers`, `arena`, `sequence`) | 593 | one small refusal or dispatch each |
| `py-dedupe` | 27 duplicate groups (`_expansion_trees` score x3 x2, `_expansion_prep` fit_transform x4 x2, `_mamba_impl` backward x3, `linear_model` Ridge/OLS get_params/set_params/score/vendor_used/numeric_mode_used, `extratrees`/`randomforest` save/load/predict, `_native_*_bridge._refusal` x3) | 464 (f) | one shared method per group; the `_refusal` triple becomes one helper |

Total (b): 5,148 lines in 308 functions; after the lanes each becomes one native call (about 3 lines), so the Python
side loses about 4,200 lines and the Mojo UNPACK side gains the refusals as typed comparisons. (e) is not a removal
lane: every (e) module is a testing tool the verify command or the tests import (AGENTS.md "The verify command is
protected"); it is classified so that it is never counted as product work.

## 4. The end state

Mojo side: the 1,031 interconnect files (the only ones importing Python) are already the whole Python-importing set;
after `door-run-split` no boundary function does typed work on a Python-holding struct, and after `resident-slots`
no `SharedSlot` keeps user data as Python objects. The fence extension (2.3) holds it there with one grep.

Python side, what stays: (a) + (c) + (d) = 81,364 + 578 + 24,363 = **106,305 lines** of today's text, plus about
900 lines of one-call replacements for the 308 (b) functions: about **107,000 lines in 307 modules** (the (e)
modules stay as test tools, 1,375 lines, outside the product count). Of the (a) part, 37,760 lines are public
docstrings (user documentation, kept) and 10,477 are wiring modules; the product-glue functions proper are 25,342
lines. Two optional moves, each its own lane and neither required by the rule: the 6,295 lines of per-family
`connect*` functions can become table rows the PUBLISH modules fill (saves about 5,000 lines); `host_surface.py`'s
5,234-line constant table can be generated from the Mojo entry tables (saves about 5,000). With both, the package is
about **97,000 lines**, 71 % of it docstrings and verify.

Order of lanes and what each unblocks:

1. `door-run-split` (Mojo, 11 functions) and the fence extension in `tools/python_free_dirs.sh` (2.3): closes
   entanglements 1 and 2 and makes every later lane's regression visible at push.
2. `py-admission` (1,295): the shared `admit_x` / `prepare_targets` refusal path; every estimator lane after it
   deletes its own checks instead of re-homing them.
3. `py-trees`, `py-neighbors`, `py-classical-linalg`, `py-lm-neural` in parallel (disjoint files, 1,973 lines).
4. `py-archive`, `py-metrics`, `py-params-mode`, `py-msel-prep-cnn` in parallel (1,287 lines).
5. `py-tail` and `py-dedupe` (1,057 lines), then `resident-slots` (Mojo, 24 slots), which needs no Python change
   and can run beside 3 and 4.

Twelve lanes; the Python lanes are mechanical (each (b) function has its owner door named above), the two Mojo lanes
are the only judgment work.

## Appendix: every module by class (lines)

| module | lines | (a) glue | (b) Python work | (c) metadata | (d) verify | (e) tooling | (f) dead/dup |
|---|---:|---:|---:|---:|---:|---:|---:|
| `host_surface.py` | 5234 | 5234 | 0 | 0 | 0 | 0 | 0 |
| `_verify_all.py` | 4495 | 0 | 0 | 0 | 4495 | 0 | 0 |
| `_expansion_decomp.py` | 3478 | 3394 | 23 | 34 | 0 | 0 | 27 |
| `_verification_evidence_data.py` | 3375 | 0 | 0 | 0 | 3375 | 0 | 0 |
| `_expansion_prep.py` | 3038 | 2809 | 92 | 83 | 0 | 0 | 54 |
| `_backend.py` | 2867 | 2867 | 0 | 0 | 0 | 0 | 0 |
| `_expansion_trees.py` | 2842 | 2733 | 10 | 0 | 0 | 0 | 99 |
| `_expansion_linear.py` | 2544 | 2521 | 0 | 7 | 0 | 0 | 16 |
| `_expansion_neighbors.py` | 2313 | 2248 | 65 | 0 | 0 | 0 | 0 |
| `model_selection.py` | 2152 | 1971 | 163 | 18 | 0 | 0 | 0 |
| `_verify_graph.py` | 1841 | 0 | 0 | 0 | 1841 | 0 | 0 |
| `_verify.py` | 1686 | 0 | 0 | 0 | 1686 | 0 | 0 |
| `_metrics_impl.py` | 1672 | 1414 | 251 | 7 | 0 | 0 | 0 |
| `ops.py` | 1656 | 1624 | 32 | 0 | 0 | 0 | 0 |
| `_verification_api_reference.py` | 1625 | 0 | 0 | 0 | 1625 | 0 | 0 |
| `_classical_host.py` | 1552 | 1457 | 95 | 0 | 0 | 0 | 0 |
| `_training_impl.py` | 1549 | 1534 | 9 | 6 | 0 | 0 | 0 |
| `neighbors.py` | 1498 | 1188 | 295 | 0 | 0 | 0 | 15 |
| `linear_model.py` | 1455 | 1300 | 32 | 55 | 0 | 0 | 68 |
| `_verification_catalog.py` | 1386 | 0 | 0 | 0 | 1386 | 0 | 0 |
| `_mamba_impl.py` | 1296 | 1166 | 20 | 27 | 0 | 0 | 83 |
| `ensemble.py` | 1295 | 1290 | 0 | 5 | 0 | 0 | 0 |
| `_svm_impl.py` | 1266 | 1260 | 6 | 0 | 0 | 0 | 0 |
| `_verification_endpoint_data.py` | 1152 | 0 | 0 | 0 | 1152 | 0 | 0 |
| `_expansion_cluster.py` | 1142 | 1139 | 3 | 0 | 0 | 0 | 0 |
| `_buffer.py` | 1109 | 495 | 610 | 4 | 0 | 0 | 0 |
| `_expansion_ann.py` | 1083 | 1046 | 14 | 0 | 0 | 0 | 23 |
| `_conformance.py` | 1055 | 0 | 0 | 0 | 1055 | 0 | 0 |
| `_verify_reference.py` | 1035 | 0 | 0 | 0 | 1035 | 0 | 0 |
| `_expansion_cnn.py` | 990 | 896 | 94 | 0 | 0 | 0 | 0 |
| `_byte_lm_impl.py` | 947 | 854 | 93 | 0 | 0 | 0 | 0 |
| `_native_msel_metric_plans.py` | 941 | 941 | 0 | 0 | 0 | 0 | 0 |
| `randomforest.py` | 888 | 682 | 172 | 0 | 0 | 0 | 34 |
| `_gp_impl.py` | 885 | 842 | 41 | 2 | 0 | 0 | 0 |
| `_transformer_impl.py` | 858 | 746 | 91 | 3 | 0 | 0 | 18 |
| `_linalg_impl.py` | 842 | 741 | 101 | 0 | 0 | 0 | 0 |
| `_tsa_impl.py` | 744 | 738 | 0 | 6 | 0 | 0 | 0 |
| `_spectral_impl.py` | 740 | 725 | 15 | 0 | 0 | 0 | 0 |
| `_mode.py` | 734 | 587 | 60 | 87 | 0 | 0 | 0 |
| `_x_sequence_rnn.py` | 732 | 718 | 14 | 0 | 0 | 0 | 0 |
| `_arima_impl.py` | 719 | 717 | 0 | 2 | 0 | 0 | 0 |
| `kernel_methods.py` | 707 | 691 | 16 | 0 | 0 | 0 | 0 |
| `extratrees.py` | 656 | 504 | 152 | 0 | 0 | 0 | 0 |
| `tokenizer.py` | 640 | 584 | 38 | 18 | 0 | 0 | 0 |
| `__main__.py` | 639 | 0 | 0 | 0 | 639 | 0 | 0 |
| `_sk_validate.py` | 634 | 314 | 320 | 0 | 0 | 0 | 0 |
| `_x_sequence_model.py` | 627 | 625 | 0 | 0 | 0 | 0 | 2 |
| `_expansion_metrics.py` | 617 | 609 | 8 | 0 | 0 | 0 | 0 |
| `preprocessing.py` | 606 | 541 | 31 | 34 | 0 | 0 | 0 |
| `density.py` | 592 | 592 | 0 | 0 | 0 | 0 | 0 |
| `_samba_impl.py` | 578 | 575 | 0 | 3 | 0 | 0 | 0 |
| `decomposition.py` | 575 | 575 | 0 | 0 | 0 | 0 | 0 |
| `_array.py` | 561 | 498 | 63 | 0 | 0 | 0 | 0 |
| `__init__.py` | 543 | 517 | 0 | 26 | 0 | 0 | 0 |
| `parallel_classical.py` | 537 | 537 | 0 | 0 | 0 | 0 | 0 |
| `_ivf_impl.py` | 529 | 483 | 46 | 0 | 0 | 0 | 0 |
| `models/config.py` | 512 | 360 | 148 | 4 | 0 | 0 | 0 |
| `_numeric_profile.py` | 507 | 409 | 98 | 0 | 0 | 0 | 0 |
| `_gpc_impl.py` | 501 | 486 | 15 | 0 | 0 | 0 | 0 |
| `resample.py` | 470 | 444 | 26 | 0 | 0 | 0 | 0 |
| `_labels.py` | 457 | 218 | 239 | 0 | 0 | 0 | 0 |
| `hdbscan.py` | 454 | 454 | 0 | 0 | 0 | 0 | 0 |
| `gpu_plugins.py` | 448 | 448 | 0 | 0 | 0 | 0 | 0 |
| `cluster.py` | 444 | 444 | 0 | 0 | 0 | 0 | 0 |
| `neural_inference.py` | 443 | 433 | 5 | 5 | 0 | 0 | 0 |
| `_solver_impl.py` | 440 | 440 | 0 | 0 | 0 | 0 | 0 |
| `_native_full_reset_params.py` | 434 | 409 | 0 | 14 | 0 | 0 | 11 |
| `_native_msel_prep.py` | 428 | 428 | 0 | 0 | 0 | 0 | 0 |
| `mixture.py` | 423 | 423 | 0 | 0 | 0 | 0 | 0 |
| `_hierarchy_impl.py` | 421 | 421 | 0 | 0 | 0 | 0 | 0 |
| `_native_classical_bridge.py` | 413 | 292 | 121 | 0 | 0 | 0 | 0 |
| `_identity.py` | 410 | 0 | 0 | 0 | 410 | 0 | 0 |
| `_native_trees_bridge.py` | 410 | 362 | 40 | 0 | 0 | 0 | 8 |
| `_x_sequence_optim.py` | 409 | 409 | 0 | 0 | 0 | 0 | 0 |
| `_serialize.py` | 408 | 141 | 267 | 0 | 0 | 0 | 0 |
| `_verify_worker.py` | 405 | 0 | 0 | 0 | 405 | 0 | 0 |
| `svm.py` | 403 | 403 | 0 | 0 | 0 | 0 | 0 |
| `_families.py` | 393 | 0 | 0 | 0 | 393 | 0 | 0 |
| `_byte_lm_host.py` | 386 | 348 | 38 | 0 | 0 | 0 | 0 |
| `_verify_distributed.py` | 385 | 0 | 0 | 0 | 385 | 0 | 0 |
| `lm_corpus.py` | 385 | 383 | 0 | 2 | 0 | 0 | 0 |
| `models/causal_lm.py` | 385 | 358 | 23 | 4 | 0 | 0 | 0 |
| `_x_sequence_theta.py` | 375 | 375 | 0 | 0 | 0 | 0 | 0 |
| `_tokenizer_synthetic.py` | 354 | 0 | 0 | 0 | 0 | 354 | 0 |
| `_umap_impl.py` | 353 | 353 | 0 | 0 | 0 | 0 | 0 |
| `_gbdt_adapters.py` | 339 | 303 | 19 | 17 | 0 | 0 | 0 |
| `_x_sequence_mlp.py` | 333 | 333 | 0 | 0 | 0 | 0 | 0 |
| `_mlp_impl.py` | 329 | 273 | 56 | 0 | 0 | 0 | 0 |
| `_verify_model.py` | 328 | 0 | 0 | 0 | 328 | 0 | 0 |
| `_native_public_attributes.py` | 314 | 200 | 114 | 0 | 0 | 0 | 0 |
| `_native_optimizer_bridge.py` | 306 | 306 | 0 | 0 | 0 | 0 | 0 |
| `_forest_host.py` | 301 | 301 | 0 | 0 | 0 | 0 | 0 |
| `_forest_protocol.py` | 292 | 144 | 136 | 12 | 0 | 0 | 0 |
| `_verify_light.py` | 288 | 0 | 0 | 0 | 288 | 0 | 0 |
| `_iforest_impl.py` | 287 | 287 | 0 | 0 | 0 | 0 | 0 |
| `_verify_parallel_cv.py` | 285 | 0 | 0 | 0 | 285 | 0 | 0 |
| `embedding.py` | 284 | 223 | 58 | 3 | 0 | 0 | 0 |
| `_verification_api_targets.py` | 276 | 0 | 0 | 0 | 276 | 0 | 0 |
| `parallel_training.py` | 274 | 274 | 0 | 0 | 0 | 0 | 0 |
| `parallel_ensemble.py` | 267 | 267 | 0 | 0 | 0 | 0 | 0 |
| `_verification_coverage.py` | 262 | 0 | 0 | 0 | 262 | 0 | 0 |
| `_x_sequence_norm.py` | 252 | 252 | 0 | 0 | 0 | 0 | 0 |
| `_verify_native_runtime.py` | 248 | 0 | 0 | 0 | 248 | 0 | 0 |
| `_cholesky_impl.py` | 237 | 237 | 0 | 0 | 0 | 0 | 0 |
| `_gpu_witness.py` | 237 | 0 | 0 | 0 | 0 | 237 | 0 |
| `_native_xcnn_bridge.py` | 231 | 231 | 0 | 0 | 0 | 0 | 0 |
| `_x_sequence_sched.py` | 228 | 214 | 14 | 0 | 0 | 0 | 0 |
| `_verify_causal_lm.py` | 221 | 0 | 0 | 0 | 221 | 0 | 0 |
| `_x_sequence_prophet.py` | 220 | 220 | 0 | 0 | 0 | 0 | 0 |
| `_x_sequence_autoarima.py` | 218 | 218 | 0 | 0 | 0 | 0 | 0 |
| `_verify_api_runner.py` | 215 | 0 | 0 | 0 | 215 | 0 | 0 |
| `_x_sequence_croston.py` | 215 | 215 | 0 | 0 | 0 | 0 | 0 |
| `_native_declared_constructors.py` | 213 | 202 | 11 | 0 | 0 | 0 | 0 |
| `_x_sequence_moe.py` | 213 | 213 | 0 | 0 | 0 | 0 | 0 |
| `_x_sequence_ets.py` | 208 | 208 | 0 | 0 | 0 | 0 | 0 |
| `_gbdt_host.py` | 203 | 203 | 0 | 0 | 0 | 0 | 0 |
| `_causal_lm_fixtures.py` | 202 | 0 | 0 | 0 | 0 | 202 | 0 |
| `_verify_api_estimators.py` | 202 | 0 | 0 | 0 | 202 | 0 | 0 |
| `_verify_api_metadata.py` | 200 | 0 | 0 | 0 | 200 | 0 | 0 |
| `_native_parallel_public.py` | 199 | 187 | 12 | 0 | 0 | 0 | 0 |
| `_native_msel_dispatch.py` | 190 | 190 | 0 | 0 | 0 | 0 | 0 |
| `_verify_generation.py` | 190 | 0 | 0 | 0 | 190 | 0 | 0 |
| `_bufcheck.py` | 188 | 0 | 0 | 0 | 0 | 188 | 0 |
| `_rocm_runtime.py` | 187 | 187 | 0 | 0 | 0 | 0 | 0 |
| `lowbit.py` | 186 | 182 | 0 | 4 | 0 | 0 | 0 |
| `_verify_api_neural.py` | 183 | 0 | 0 | 0 | 183 | 0 | 0 |
| `_verify_partitioned_program.py` | 180 | 0 | 0 | 0 | 180 | 0 | 0 |
| `_native_samba_bridge.py` | 179 | 179 | 0 | 0 | 0 | 0 | 0 |
| `_ragged.py` | 177 | 129 | 48 | 0 | 0 | 0 | 0 |
| `_portable_math.py` | 174 | 105 | 69 | 0 | 0 | 0 | 0 |
| `cross_vendor.py` | 174 | 174 | 0 | 0 | 0 | 0 | 0 |
| `_surface_sequence.py` | 168 | 168 | 0 | 0 | 0 | 0 | 0 |
| `_byte_lm_config.py` | 167 | 165 | 0 | 2 | 0 | 0 | 0 |
| `importers/_graph.py` | 167 | 167 | 0 | 0 | 0 | 0 | 0 |
| `_native_params_metadata.py` | 165 | 111 | 54 | 0 | 0 | 0 | 0 |
| `_surface_prep.py` | 164 | 164 | 0 | 0 | 0 | 0 | 0 |
| `models/safetensors.py` | 164 | 158 | 0 | 6 | 0 | 0 | 0 |
| `_surface_metrics.py` | 163 | 163 | 0 | 0 | 0 | 0 | 0 |
| `_arena_io.py` | 162 | 0 | 0 | 0 | 0 | 162 | 0 |
| `_surface_neighbors.py` | 162 | 162 | 0 | 0 | 0 | 0 | 0 |
| `_native_host_archive_predict.py` | 157 | 157 | 0 | 0 | 0 | 0 | 0 |
| `_surface_trees.py` | 156 | 156 | 0 | 0 | 0 | 0 | 0 |
| `_verify_api_training.py` | 156 | 0 | 0 | 0 | 156 | 0 | 0 |
| `_verify_distributed_program.py` | 155 | 0 | 0 | 0 | 155 | 0 | 0 |
| `_native_arena_rows.py` | 150 | 150 | 0 | 0 | 0 | 0 | 0 |
| `models/parallel_causal_lm.py` | 148 | 148 | 0 | 0 | 0 | 0 | 0 |
| `_native_mode_reconstruction.py` | 143 | 124 | 19 | 0 | 0 | 0 | 0 |
| `_surface_linear.py` | 140 | 140 | 0 | 0 | 0 | 0 | 0 |
| `_native_logistic_bridge.py` | 139 | 120 | 13 | 0 | 0 | 0 | 6 |
| `models/tokenizer.py` | 139 | 131 | 6 | 2 | 0 | 0 | 0 |
| `_parallel_cv_witness.py` | 136 | 0 | 0 | 0 | 0 | 136 | 0 |
| `_x_sequence_var.py` | 134 | 134 | 0 | 0 | 0 | 0 | 0 |
| `_native_xseq_bridge.py` | 133 | 133 | 0 | 0 | 0 | 0 | 0 |
| `_x_sequence_garch.py` | 133 | 133 | 0 | 0 | 0 | 0 | 0 |
| `_native_loaded_modules.py` | 131 | 131 | 0 | 0 | 0 | 0 | 0 |
| `_verify_api_utilities.py` | 131 | 0 | 0 | 0 | 131 | 0 | 0 |
| `_native_archive_bridge.py` | 128 | 103 | 25 | 0 | 0 | 0 | 0 |
| `importers/_program.py` | 126 | 126 | 0 | 0 | 0 | 0 | 0 |
| `parallel_neighbors.py` | 126 | 126 | 0 | 0 | 0 | 0 | 0 |
| `linalg_fast.py` | 125 | 42 | 83 | 0 | 0 | 0 | 0 |
| `_merged.py` | 124 | 124 | 0 | 0 | 0 | 0 | 0 |
| `_native_host_archive_bridge.py` | 124 | 124 | 0 | 0 | 0 | 0 | 0 |
| `_native_x_linear_bridge.py` | 124 | 124 | 0 | 0 | 0 | 0 | 0 |
| `_x_sequence_stl.py` | 122 | 122 | 0 | 0 | 0 | 0 | 0 |
| `_native_labels_bridge.py` | 120 | 74 | 46 | 0 | 0 | 0 | 0 |
| `_native_family_bridge.py` | 117 | 117 | 0 | 0 | 0 | 0 | 0 |
| `_expansion_sequence.py` | 116 | 97 | 0 | 19 | 0 | 0 | 0 |
| `adapters/max.py` | 115 | 115 | 0 | 0 | 0 | 0 | 0 |
| `_native_backend_public.py` | 113 | 113 | 0 | 0 | 0 | 0 | 0 |
| `parallel_neighbors_reference.py` | 113 | 113 | 0 | 0 | 0 | 0 | 0 |
| `_native_mode_metadata.py` | 110 | 110 | 0 | 0 | 0 | 0 | 0 |
| `_verify_api_misc.py` | 110 | 0 | 0 | 0 | 110 | 0 | 0 |
| `_verify_small.py` | 108 | 0 | 0 | 0 | 108 | 0 | 0 |
| `adapters/torch.py` | 108 | 88 | 20 | 0 | 0 | 0 | 0 |
| `_native_decomp_kit_bridge.py` | 105 | 88 | 17 | 0 | 0 | 0 | 0 |
| `_x_sequence_device.py` | 105 | 105 | 0 | 0 | 0 | 0 | 0 |
| `_native_scaler_bridge.py` | 104 | 104 | 0 | 0 | 0 | 0 | 0 |
| `_native_spectral_umap_bridge.py` | 104 | 91 | 13 | 0 | 0 | 0 | 0 |
| `linalg.py` | 104 | 104 | 0 | 0 | 0 | 0 | 0 |
| `_native_neighbor_queries.py` | 100 | 63 | 37 | 0 | 0 | 0 | 0 |
| `importers/_table.py` | 98 | 98 | 0 | 0 | 0 | 0 | 0 |
| `_native_tsa_bridge.py` | 97 | 97 | 0 | 0 | 0 | 0 | 0 |
| `_native_parallel_shard.py` | 96 | 96 | 0 | 0 | 0 | 0 | 0 |
| `_native_parallel_cv.py` | 91 | 91 | 0 | 0 | 0 | 0 | 0 |
| `_verify_light_command.py` | 91 | 0 | 0 | 0 | 91 | 0 | 0 |
| `parallel_preprocessing.py` | 91 | 91 | 0 | 0 | 0 | 0 | 0 |
| `_native_pca_bridge.py` | 90 | 90 | 0 | 0 | 0 | 0 | 0 |
| `_native_decomp_bridge.py` | 88 | 88 | 0 | 0 | 0 | 0 | 0 |
| `_native_linear_constructors.py` | 88 | 79 | 9 | 0 | 0 | 0 | 0 |
| `_native_neural_bridge.py` | 88 | 71 | 0 | 17 | 0 | 0 | 0 |
| `_surface_cnn.py` | 87 | 87 | 0 | 0 | 0 | 0 | 0 |
| `_native_qn_bridge.py` | 86 | 86 | 0 | 0 | 0 | 0 | 0 |
| `training.py` | 85 | 85 | 0 | 0 | 0 | 0 | 0 |
| `_native_msel_more.py` | 84 | 84 | 0 | 0 | 0 | 0 | 0 |
| `_native_classical_warnings.py` | 83 | 83 | 0 | 0 | 0 | 0 | 0 |
| `_native_arena_prog.py` | 82 | 76 | 6 | 0 | 0 | 0 | 0 |
| `_verify_api_faults.py` | 82 | 0 | 0 | 0 | 82 | 0 | 0 |
| `importers/_native_reader_bridge.py` | 82 | 82 | 0 | 0 | 0 | 0 | 0 |
| `_native_prep_encode_bridge.py` | 81 | 81 | 0 | 0 | 0 | 0 | 0 |
| `_native_msel_extra.py` | 80 | 80 | 0 | 0 | 0 | 0 | 0 |
| `_native_prep_stateless_bridge.py` | 80 | 80 | 0 | 0 | 0 | 0 | 0 |
| `_native_minmax_bridge.py` | 79 | 79 | 0 | 0 | 0 | 0 | 0 |
| `_native_params_setter.py` | 79 | 65 | 0 | 14 | 0 | 0 | 0 |
| `_verify_api.py` | 78 | 0 | 0 | 0 | 78 | 0 | 0 |
| `_verify_api_models.py` | 78 | 0 | 0 | 0 | 78 | 0 | 0 |
| `importers/_native_import_bridge.py` | 78 | 67 | 11 | 0 | 0 | 0 | 0 |
| `_native_declared_params.py` | 77 | 77 | 0 | 0 | 0 | 0 | 0 |
| `_surface_decomp.py` | 77 | 77 | 0 | 0 | 0 | 0 | 0 |
| `adapters/_shipped.py` | 76 | 76 | 0 | 0 | 0 | 0 | 0 |
| `_user_objects.py` | 75 | 53 | 22 | 0 | 0 | 0 | 0 |
| `_sample_weight.py` | 72 | 21 | 51 | 0 | 0 | 0 | 0 |
| `parallel_graph.py` | 72 | 72 | 0 | 0 | 0 | 0 | 0 |
| `_surface_cluster.py` | 71 | 71 | 0 | 0 | 0 | 0 | 0 |
| `_distributed_program.py` | 68 | 0 | 0 | 0 | 0 | 68 | 0 |
| `_native_robust_bridge.py` | 68 | 68 | 0 | 0 | 0 | 0 | 0 |
| `_native_msel_inputs.py` | 66 | 66 | 0 | 0 | 0 | 0 | 0 |
| `_native_prep_classify_bridge.py` | 61 | 61 | 0 | 0 | 0 | 0 | 0 |
| `language_model.py` | 61 | 52 | 0 | 9 | 0 | 0 | 0 |
| `_native_tsvd_bridge.py` | 60 | 60 | 0 | 0 | 0 | 0 | 0 |
| `_verify_resources.py` | 60 | 0 | 0 | 0 | 60 | 0 | 0 |
| `_native_runtime_manifold_bridge.py` | 59 | 40 | 19 | 0 | 0 | 0 | 0 |
| `_verify_light_faults.py` | 59 | 0 | 0 | 0 | 59 | 0 | 0 |
| `models/__init__.py` | 59 | 50 | 0 | 9 | 0 | 0 | 0 |
| `parallel_model_selection.py` | 59 | 59 | 0 | 0 | 0 | 0 | 0 |
| `_verify_api_optimizers.py` | 58 | 0 | 0 | 0 | 58 | 0 | 0 |
| `_verify_api_programs.py` | 58 | 0 | 0 | 0 | 58 | 0 | 0 |
| `_native_table_key.py` | 57 | 57 | 0 | 0 | 0 | 0 | 0 |
| `_native_maxabs_bridge.py` | 56 | 56 | 0 | 0 | 0 | 0 | 0 |
| `_sklearn_modules.py` | 56 | 56 | 0 | 0 | 0 | 0 | 0 |
| `mamba.py` | 54 | 54 | 0 | 0 | 0 | 0 | 0 |
| `_surface_ann.py` | 53 | 53 | 0 | 0 | 0 | 0 | 0 |
| `parallel_forecasting.py` | 53 | 53 | 0 | 0 | 0 | 0 | 0 |
| `_native_family_layouts.py` | 50 | 46 | 4 | 0 | 0 | 0 | 0 |
| `_partitioned_program.py` | 50 | 50 | 0 | 0 | 0 | 0 | 0 |
| `_verification_profiles.py` | 50 | 0 | 0 | 0 | 50 | 0 | 0 |
| `parallel_ivf.py` | 50 | 50 | 0 | 0 | 0 | 0 | 0 |
| `_native_iforest_depth.py` | 49 | 49 | 0 | 0 | 0 | 0 | 0 |
| `_native_linear_score_bridge.py` | 49 | 39 | 10 | 0 | 0 | 0 | 0 |
| `exceptions.py` | 49 | 35 | 14 | 0 | 0 | 0 | 0 |
| `models/_catalog.py` | 49 | 49 | 0 | 0 | 0 | 0 | 0 |
| `_host_import_aliases.py` | 48 | 48 | 0 | 0 | 0 | 0 | 0 |
| `_native_numeric_profile.py` | 48 | 48 | 0 | 0 | 0 | 0 | 0 |
| `_arrays.py` | 47 | 47 | 0 | 0 | 0 | 0 | 0 |
| `_native_ipca_bridge.py` | 47 | 47 | 0 | 0 | 0 | 0 | 0 |
| `parallel_gaussian_process.py` | 47 | 47 | 0 | 0 | 0 | 0 | 0 |
| `_native_owner_lifetime.py` | 46 | 46 | 0 | 0 | 0 | 0 | 0 |
| `model_pool_training.py` | 46 | 46 | 0 | 0 | 0 | 0 | 0 |
| `_native_msel_scorers.py` | 44 | 44 | 0 | 0 | 0 | 0 | 0 |
| `_native_program_save_bridge.py` | 42 | 34 | 8 | 0 | 0 | 0 | 0 |
| `importers/onnx.py` | 42 | 42 | 0 | 0 | 0 | 0 | 0 |
| `_parallel_neural_admission.py` | 40 | 40 | 0 | 0 | 0 | 0 | 0 |
| `importers/__init__.py` | 40 | 34 | 0 | 6 | 0 | 0 | 0 |
| `_verification_ctr_models.py` | 39 | 0 | 0 | 0 | 39 | 0 | 0 |
| `offload_training.py` | 39 | 39 | 0 | 0 | 0 | 0 | 0 |
| `_native_msel_search.py` | 38 | 38 | 0 | 0 | 0 | 0 | 0 |
| `_native_array_storage.py` | 37 | 37 | 0 | 0 | 0 | 0 | 0 |
| `_native_operation_lock.py` | 37 | 37 | 0 | 0 | 0 | 0 | 0 |
| `_lazy_out.py` | 36 | 24 | 12 | 0 | 0 | 0 | 0 |
| `transformer.py` | 35 | 35 | 0 | 0 | 0 | 0 | 0 |
| `_verification_sources.py` | 34 | 0 | 0 | 0 | 34 | 0 | 0 |
| `_native_ops.py` | 33 | 31 | 0 | 2 | 0 | 0 | 0 |
| `_native_parallel_shard_worker.py` | 33 | 33 | 0 | 0 | 0 | 0 | 0 |
| `_verify_api_availability.py` | 32 | 0 | 0 | 0 | 32 | 0 | 0 |
| `_byte_lm_trainer_host.py` | 30 | 30 | 0 | 0 | 0 | 0 | 0 |
| `_cpu_reference.py` | 28 | 0 | 0 | 0 | 0 | 28 | 0 |
| `_native_array_protocol_bridge.py` | 28 | 28 | 0 | 0 | 0 | 0 | 0 |
| `_native_model_selection_bridge.py` | 28 | 28 | 0 | 0 | 0 | 0 | 0 |
| `_residual_dropout_impl.py` | 27 | 27 | 0 | 0 | 0 | 0 | 0 |
| `_native_mode_properties.py` | 26 | 26 | 0 | 0 | 0 | 0 | 0 |
| `_native_jl_bridge.py` | 25 | 25 | 0 | 0 | 0 | 0 | 0 |
| `adapters/__init__.py` | 24 | 24 | 0 | 0 | 0 | 0 | 0 |
| `_runtime_fallbacks.py` | 23 | 23 | 0 | 0 | 0 | 0 | 0 |
| `_byte_lm_checkpoint.py` | 22 | 22 | 0 | 0 | 0 | 0 | 0 |
| `_native_parallel_forecast_worker.py` | 22 | 22 | 0 | 0 | 0 | 0 | 0 |
| `_native_parallel_neural_worker.py` | 22 | 22 | 0 | 0 | 0 | 0 | 0 |
| `_parallel_mlp.py` | 22 | 22 | 0 | 0 | 0 | 0 | 0 |
| `_verify_public.py` | 21 | 0 | 0 | 0 | 21 | 0 | 0 |
| `_native_parallel_cv_worker.py` | 20 | 20 | 0 | 0 | 0 | 0 | 0 |
| `_optional_numpy.py` | 20 | 3 | 17 | 0 | 0 | 0 | 0 |
| `neural_network.py` | 17 | 17 | 0 | 0 | 0 | 0 | 0 |
| `_parallel_neural.py` | 16 | 16 | 0 | 0 | 0 | 0 | 0 |
| `_native_runtime_sequential_bridge.py` | 15 | 15 | 0 | 0 | 0 | 0 | 0 |
| `manifold.py` | 15 | 15 | 0 | 0 | 0 | 0 | 0 |
| `_native_process.py` | 13 | 13 | 0 | 0 | 0 | 0 | 0 |
| `metrics.py` | 13 | 13 | 0 | 0 | 0 | 0 | 0 |
| `_native_parallel_causal_worker.py` | 12 | 12 | 0 | 0 | 0 | 0 | 0 |
| `adapters/_torch_compile.py` | 12 | 12 | 0 | 0 | 0 | 0 | 0 |
| `umap.py` | 10 | 10 | 0 | 0 | 0 | 0 | 0 |
| `_class_weights.py` | 8 | 8 | 0 | 0 | 0 | 0 | 0 |
| `gaussian_process.py` | 8 | 8 | 0 | 0 | 0 | 0 | 0 |
| `impute.py` | 8 | 8 | 0 | 0 | 0 | 0 | 0 |
| `kernel_approximation.py` | 8 | 8 | 0 | 0 | 0 | 0 | 0 |
| `calibration.py` | 7 | 7 | 0 | 0 | 0 | 0 | 0 |
| `covariance.py` | 7 | 7 | 0 | 0 | 0 | 0 | 0 |
| `cross_decomposition.py` | 7 | 7 | 0 | 0 | 0 | 0 | 0 |
| `discriminant_analysis.py` | 7 | 7 | 0 | 0 | 0 | 0 | 0 |
| `feature_selection.py` | 7 | 7 | 0 | 0 | 0 | 0 | 0 |
| `isotonic.py` | 7 | 7 | 0 | 0 | 0 | 0 | 0 |
| `kernel_ridge.py` | 7 | 7 | 0 | 0 | 0 | 0 | 0 |
| `multiclass.py` | 7 | 7 | 0 | 0 | 0 | 0 | 0 |
| `multioutput.py` | 7 | 7 | 0 | 0 | 0 | 0 | 0 |
| `naive_bayes.py` | 7 | 7 | 0 | 0 | 0 | 0 | 0 |
| `random_projection.py` | 7 | 7 | 0 | 0 | 0 | 0 | 0 |
| `semi_supervised.py` | 7 | 7 | 0 | 0 | 0 | 0 | 0 |
| `tree.py` | 7 | 7 | 0 | 0 | 0 | 0 | 0 |
| `_version.py` | 5 | 5 | 0 | 0 | 0 | 0 | 0 |
| `_verification_light_reference.py` | 3 | 0 | 0 | 0 | 3 | 0 | 0 |
