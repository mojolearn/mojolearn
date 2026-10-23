# `# cpu-route:` tags in python/mojolearn (tests excluded), 2026-10-22

Lane purge-cpu-routes, from integ/20261022. Rule (Andrew, 2026-10-22): no Python in the runtime;
"host" means CPU Mojo. A tag stays only on a verification fixture, on the str/object label input
step, on a native Mojo host call, or with a reason naming a Python object that cannot cross into
Mojo. Counts: `git grep -c '# cpu-route:' <rev> -- python/mojolearn ':!python/mojolearn/tests'`.

Before 402, after 310.

## Moved or removed

| tag site (before) | tags | now |
|---|---|---|
| importers/_graph.py `resolve` (Python branch), `SHAPE_RULES` and every `_r_*` rule, `_bcast`, `_strides`, `_bcast_values`, `_slice_range`, `_numel`, `int_values` ("model graph description on the host ...") | 74 | Mojo for every graph: a read graph on its handle (autograd/import_resolve.mojo `onnx_resolve`), a Python GraphIR as words first (`_graph.resolve` -> `onnx_from_words` + `onnx_resolve`, autograd/import_lower.mojo `onnx_from_words`). The Python rules are the test oracle python/mojolearn/tests/_resolve_oracle.py. |
| importers/_graph.py `_message` (2) | 2 | `# glue:` (the lines of the UnsupportedModel error message, one per refusal entry) |
| importers/_program.py `_read` ONNX path, the `onnx_import` / `onnx_from_words` calls, `_graph_words` (22) | 22 | an ONNX path is one native call: autograd/import_lower.mojo `onnx_import_file` (read + resolve + refuse + lower + capture on the reader's own bytes), door autograd/import_lower_pyapi.mojo `onnx_import_file_py`, bindings `onnx_import_file`. `_graph_words` moved to importers/_graph.py `graph_words` (kept, below). |
| importers/onnx.py file mapping (`mmap`, `open`, `onnx_copy_blob`, the size check) | - | autograd/import_reader.mojo: the handle keeps the model and external data bytes (`_Reader.take_graph`, `OnnxGraph.data`), `onnx_buffers` gives their addresses (replaces `onnx_copy_blob`); CANNOT_OPEN reports whether the file exists (`OnnxGraph.missing`, replaces `os.path.exists`). Python opens no file. |
| models/_catalog.py `_sha256_file` (hashlib over 1 MB reads, 2) | 2 | autograd/model_verify.mojo `verify_file_sha256` (x_sequence binding); check-input and output digests through core/sha256.mojo (`sha256_hex`, tokenizer host binding). |
| models/config.py field checks (6) | 6 | `# glue:` (a fixed handful of named config fields, argument checks of the config dict) |
| _weight_import.py (3) | 3 | the two are native safetensors reader calls the gate does not flag (tag dropped); the offset is `# glue:` |
| _scale_gamma.py `scale_gamma` reference (1) | 1 | python/mojolearn/tests/_scale_gamma_oracle.py (tests only; the runtime is `scale_gamma_x`) |
| lm_corpus.py `_split_points` (3) | 3 | `# glue:` (at most three manifest ranges, eight points) |

## Kept

| tag site | tags | reason |
|---|---|---|
| importers/torch_export.py | 58 + header | torch.export ExportedProgram objects: torch.fx.Node, SymInt sizes, FakeTensor meta, torch.Tensor weights are torch Python objects and cannot cross into Mojo. Everything after the read (resolve, refusal, lowering) is Mojo. |
| importers/_graph.py `graph_words` | 18 | the GraphIR the torch.export reader built (Python Node/TensorInfo objects) to Mojo words; tensor bytes pass by address. |
| importers/onnx.py | 5 | native Mojo host reader calls (onnx_read, onnx_info, onnx_buffers, onnx_resolve); the gate asks for the note on a host-binding call. No Python step. |
| models/_catalog.py | 5 | `json.loads` (catalog() returns the shipped JSON as Python dicts), urllib download (an HTTP response object; Mojo has no HTTP client), the numpy seeded check input the catalog digests record, two native hash calls. Verification command. |
| models/safetensors.py `write_safetensors` | 2 | verification fixture writer (_causal_lm_fixtures.py, tests). |
| _verify_model.py | 8 | verification command: native digest/fill/hash calls and its report files. |
| _bpe_trainer.py, _tokenizer_synthetic.py and other "verification oracle or fixture text" | 52 | verification fixtures, not runtime. |
| _array.py reference arms | 8 | differential test reference arms. |
| label input steps (_labels.py, _expansion_prep.py, _expansion_trees.py, model_selection.py, ensemble.py, linear_model.py, resample.py) | ~75 | str/object labels, user cv splitters, user index pairs, user estimators and callables: Python objects. |
| _expansion_decomp.py, _spectral_impl.py, _class_weights.py, _portable_math.py, _native_*_bridge.py, model_selection.py native plans, tokenizer.py bindings, _byte_lm_impl.py, _x_sequence_prophet.py | ~35 | native Mojo host binding calls (CPU-only installs or host-scalar plans); no Python compute. |
| tokenizer.py text packing / decoding | 4 | Python str documents in and out (text input/output step). |

## Findings not fixed in this lane

- The 8 "str or object labels, the explicit label input step" sites (`_expansion_prep.py` 5441-5445,
  5630-5631, 5842; `_labels.py` 309, 633) do a per-label Python loop (`{c: i ...}`, `[index[v] for v
  in values]`, `sorted_classes` over the label list); they do not hand the objects to the native
  `prepare_targets` path. A native str-label encoder (UTF-8 blob + offsets -> sorted classes and
  codes, the ORDER RULE for str) is owed; it changes the label module of every classifier, so it is
  its own lane.
- ensemble.py `_tree_metadata` (2662, 2717) parses the model text in Python for
  `get_tree_leaf_counts` / `get_leaf_values`; a native metadata entry in the trees binding is owed.

## 2026-10-22, lane purge-user-objects (from purge-cpu-routes 1e3e2c5c2)

Counts: before 310, after 186 (`git grep -c '# cpu-route:' HEAD -- python/mojolearn ':!python/mojolearn/tests'`).
Public runtime baseline: python 360 -> 265, orchestrated 473 -> 568, native 200.

### Removed

| site | tags | now |
|---|---|---|
| importers/torch_export.py (the torch.export reader) | 59 | deleted. `import_model` takes an ONNX file; a torch object raises `ValueError` naming `torch.onnx.export(model, args, "model.onnx")`. |
| importers/_graph.py `graph_words`, the Python branch of `resolve` | 18 | deleted, with `onnx_from_words` (autograd/import_lower.mojo, import_lower_pyapi.mojo, both x_sequence bindings) that only it used. |

Docs: docs-site/user_guide/autograd.md "Import a pretrained model" is the two-step path. Tools retargeted to
export ONNX first: models/_catalog.py (verify_catalog_entry), tools/real_model_import.py, tools/import_compare.py
`--torch`, tools/autograd_smoke.py import-torch-smoke and real-model-smoke. No identity lane or sabotage arm named
the reader; tools/verify_coverage_audit.py drops its `onnx_from_words` n/a row.

### Moved into the native user-object reader

bindings/user_objects_native.mojo (`user_objects_c`, tokenizer host entry `user_objects_native_call`; typed kernel
core/user_objects.mojo `group_run_sizes`; door python/mojolearn/_user_objects.py) reads the object once:

| site (before) | object | op |
|---|---|---|
| model_selection.py `_take_rows`, `_take_any`; resample.py `_take` | Python list of label objects | take |
| model_selection.py `_encode_first_seen` | label objects (str/mixed) | first_seen |
| model_selection.py `check_cv` stratify test | label objects | kinds |
| model_selection.py scorer `predict_proba[:, 1]` of a Python list | external estimator's probability rows | column_f32 |
| model_selection.py label-object prediction placement | per-fold prediction lists | scatter, flatten, floats, kinds |
| model_selection.py `permutation_test_score` definition route | label objects, group labels | take, first_seen, group_rows, group_permute |
| ensemble.py `_group_sizes_rowwise`, `_group_sizes_by_runs`, `_group_id_key` | group_id objects (int/str/bytes) | group_runs (the row walk is the tests' oracle) |
| ensemble.py `_pairs_arrays` | user index pairs, pairs_weight | pairs |
| ensemble.py permutation type check | user permutation list | kinds |
| _umap_impl.py categorical target | label objects ordered by (type, value) | kinds, sorted_codes |
| _metrics_impl.py `_classification_labels`, `_label_map` | str / int label objects | kinds, strs, ints, first_seen + gather_i32 |
| linear_model.py `_flatten`, `_is_integer_labels` | nested user y | flatten, kinds |
| _array.py `_flatten` (ragged walk) | ragged user list, Arrays inside it | nested |
| _expansion_trees.py multi-output fit / predict | label rows and decoded columns | column, transpose |
| _expansion_prep.py `_partial_walk` | partial_fit labels against classes_ | lookup |
| _expansion_prep.py `_numeric_labels`, `_label_classes`, `_label_codes`, `_multilabel_indicator` | label lists, list-of-lists y | f32_exact, kinds, floats, indicator |
| _expansion_prep.py LabelEncoder.inverse_transform, LabelBinarizer decode | code lists, class objects | ints + check_indices_i64, take |
| _expansion_prep.py user covariance_, IterativeImputer predictions, `_pred_f64` | user estimator outputs | flatten, floats |
| _portable_math.py `fsum` twin | Python scalars | floats (then nsum_f64) |

### Kept, the object named (`# cpu-route: <object> read once by <entry>`)

- Native host-binding calls: _expansion_decomp.py (17, `_M` host buffers on CPU-only installs), _spectral_impl.py (4),
  _class_weights.py, _x_sequence_prophet.py, _byte_lm_impl.py, _cast_in.py (numpy.asarray), importers/onnx.py (5),
  tokenizer.py binding loads (2), model_selection.py plan entries (5), _expansion_linear.py (3),
  _expansion_trees.py (4), and the per-fold cv splitter reads (_expansion_prep.py 4, model_selection.py, _expansion_trees.py).
- User code: the strategy callable, the covariance estimator per class, IterativeImputer's estimator (4).
- tokenizer.py `_byte_to_char` is a verification fixture (tests and identity tools only).

### Left to the lanes that own them

- _labels.py (10) and _expansion_prep.py LabelEncoder / LabelBinarizer / MultiLabelBinarizer transform and fit
  (`index` maps, MLB flatten and offsets, 13): lane/migrate-classical-a (str-label encoder) and lane/migrate-shared-labels.
- ensemble.py `_tree_metadata` (2): lane/migrate-trees.
- _training_impl.py (4), _native_optimizer_bridge.py (2), _native_samba_bridge.py (2), _x_sequence_rnn.py (1): lane/migrate-neural.
- _native_pca_bridge.py (2): removed on lane/migrate-classical-a.
- tokenizer.py text packing / decoding (4): Python str documents in and out (text step).
