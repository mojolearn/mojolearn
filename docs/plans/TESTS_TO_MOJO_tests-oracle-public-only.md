# TESTS_TO_MOJO: lane tests-oracle-public-only (2026-11-15)

Scope: the 25 files whose lane is `L2-oracle-decouple` in `TESTS_TO_MOJO_2026-11-15.md`. They are test helpers with no
test functions of their own, so nothing moves to Mojo and nothing is deleted: every file is KEPT, as a testing tool
in Python, with its arithmetic unchanged. Python test lines removed: 0 (the change is +781 -229 across 22 helpers plus
the new 412-line `python/mojolearn/tests/_oracle_support.py`). Mojo checks added: none (no test function is owned
here). No product code changed.

What changed: the product-internal Python utilities the helpers imported (`_buffer`, `_array`, `_lazy_out`,
`_optional_numpy`, `_portable_math`, `_bufcheck`, `_labels.is_bool/argmax_rows/decode_labels/prepare_targets`,
`_user_objects`, family constant tables, `_backend.default_mode`, private `Array` methods) are now local copies in
`tests/_oracle_support.py` or in the file itself (stdlib, NumPy, the public `mojolearn.Array` /
`mojolearn.numeric_mode` / public estimators). That is the class of break behind 26 of the 69 smoke-2 failures (the
deleted `_arrays` shim, `_expansion_neighbors._f32/_f32_1d/_i32`, `_expansion_prep.full`).

What remains: an import marked `# oracle subject: ...` is the NATIVE PRIMITIVE the oracle drives, or the product
class or IR it is the reference of. None of them has a public route (25 marked sites in 14 files; see "native
entries missing" below).

| file | result | lines | remaining product-internal access (oracle subject) |
|---|---|---|---|
| `_array_reference.py` | kept | +169 -21 | `_array._EMPTY_ARRAY_STORAGE`, `_user_objects` "nested" reader, `_buffer._native_optional` (the seam test_hotpath_native plants) |
| `_cast_in.py` | kept, unchanged | +0 -0 | none (NumPy only) |
| `_former_prep_labels.py` | kept | +68 -27 | `_labels.flatten_labels/sorted_classes` (native ORDER RULE), `tokenizer._binding` |
| `_legacy_density_archive.py` | kept | +60 -8 | none |
| `_legacy_forest_auxiliary.py` | kept | +2 -2 | none (binding via `model._bind()`) |
| `_legacy_host_density_constructors.py` | kept, unchanged | +0 -0 | none |
| `_legacy_host_iforest_fit.py` | kept | +7 -3 | `_sk_validate` (native fit admission) |
| `_legacy_host_inference.py` | kept | +54 -30 | `_sk_validate`; binding via `model._bind()` |
| `_legacy_host_trees.py` | kept | +35 -9 | `_forest_host._load/binary_path` (native forest host binding) |
| `_legacy_neighbors_ann.py` | kept | +23 -7 | `_backend.binding` (native x_ann binding for module-level `refine`) |
| `_legacy_neighbors_graph.py` | kept | +32 -7 | none (binding via public `PageRank()`) |
| `_legacy_neighbors_kde.py` | kept | +1 -1 | none |
| `_legacy_neighbors_knn.py` | kept | +44 -19 | `_labels.encode_labels`, `_expansion_metrics.accuracy_fraction` |
| `_legacy_neighbors_loc.py` | kept | +24 -15 | `_labels.encode_labels`, `_expansion_metrics.accuracy_fraction` |
| `_legacy_neighbors_semi.py` | kept | +24 -11 | `_labels.encode_labels` |
| `_legacy_sample_weight.py` | kept | +16 -3 | none |
| `_lower_oracle.py` | kept | +92 -10 | `importers._graph.GraphIR`, `importers._program.ImportedProgram` |
| `_mamba_block_reference.py` | kept | +17 -10 | none (blocks from public `mojolearn`, binding via the block) |
| `_onnx_proto_oracle.py` | kept, unchanged | +0 -0 | none |
| `_onnx_reader_oracle.py` | kept | +6 -2 | `importers._graph.GraphIR/Node/TensorInfo` (the IR the test compares) |
| `_prep_prog_oracle.py` | kept | +46 -21 | `_expansion_prep` runner (`_PROG_CALL`, `_PROG_CONTEXT`, `_OPS`, `_Scratch`, `_Prog`, `_prep_binding`) |
| `_resolve_oracle.py` | kept | +28 -2 | `importers._graph.Resolved/TensorInfo` |
| `_svm_reference.py` | kept | +4 -4 | none |
| `_transformer_block_reference.py` | kept | +1 -1 | none (binding via `block._extension()`) |
| `_typed_input.py` | kept | +28 -16 | none |

## Native entries missing (public routes that would remove the remaining sites)

- Label ORDER RULE encode / flatten / sorted classes (`_labels.encode_labels`, `flatten_labels`, `sorted_classes`) and
  label-agnostic accuracy (`_expansion_metrics.accuracy_fraction`).
- The scikit-learn fit admission (`_sk_validate.before_fit/after_fit/reraise`).
- Host bindings by name: forest host (`_forest_host._load`), x_ann (`_backend.binding`), x_prep runner
  (`_expansion_prep._PROG_CALL`), tokenizer user-object entries (`tokenizer._binding`).
- Importer IR constructors (`GraphIR`, `Node`, `TensorInfo`, `Resolved`, `ImportedProgram`).

## Behavior notes for the reviewer

- `_lower_oracle.resolve` is now the Python oracle `_resolve_oracle.resolve`, not the product's native `resolve`
  (test_import_onnx_native holds the two equal).
- `numeric_mode()` (public) replaces `_backend.default_mode()` in three oracles; it also checks the binary's mode.
- `decode_labels` / `argmax_rows` used by the tree and neighbors oracles are NumPy/stdlib copies in
  `_legacy_host_inference.py` (first max on strict `>`, a gather); `_legacy_sample_weight` checks weights in NumPy
  with the native wording; `_typed_input.same_dtype_store` quiets float32 signaling NaNs by the stdlib round trip.
