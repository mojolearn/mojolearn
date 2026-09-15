# Lane status: lane/inference-gbdt-ctr-tables (2026-09-15)

Public, bitwise-identical CPU `predict` and `predict_proba` for saved
gradient boosting models that carry real CTR tables and tensor (combination)
CTRs, fitted on a GPU, through `mojolearn.host_model` and HostGBDT on the
shipped forest host binding. The follow-up the gbdt modes lane named in
`LANE_STATUS_lane-inference-gbdt-modes.md` ("Not public, and why").

## Saved-model format

No record was missing, and no record changed. A model with CTR tables
already carries `ctr_columns`, one `ctr_table` per CTR column (source input,
type, prior numerator and denominator, shift, scale, counter denominator,
target-class count, target border, entry count, every float with its IEEE
bits) and one `ctr_entry` per category (the counts, one per target class on a
Borders table); the CTR column's quantization borders are its `feature`
record's. A tensor CTR model carries `tensor_ctr_registry` (first model
column, table count) and one `feature_freq_tensor 2` record per tensor
column (the 64-bit tensor hash in two unsigned halves, canonical sources,
cardinalities, split history, classes, target border, prior bits,
denominator, counts). Categorical inputs are dense codes in this format, not
hashes of raw values (deviation 56), so the apply key is the code the fit
validated with `dense_category_code`. Nothing in `model_text.mojo` was
edited, so no saved-model byte of any existing lane can move from this lane.

## Design

- `gbdt/models/tensor_ctr_apply.mojo` (new, no device import): the apply half
  of a tensor CTR (the mixed-radix key, the FeatureFreq and Borders value,
  the split-history bit, the sequential expand over the model's borders).
  `TFeatureFreqTensorTable.key_for_row`, `value_for_key`, `_split_bit` and
  `TTensorCtrRegistry.expand_for_model_apply` now call it, so the GPU predict
  and the host run one body. `tensor_ctr_value_table.mojo` itself imports
  `max.gpu.host` and the compressed-index kernel and cannot be imported by a
  no-accelerator build.
- `core/gbdt_host_ctr.mojo` (new): rebuilds `TCtrValueTable`s and tensor apply
  tables from flat arrays and calls `expand_raw_columns` or
  `expand_tensor_ctr_columns`, exactly what `predict_floats` calls before it
  quantizes. `-D MOJOLEARN_GBDT_CTR_HOST_SABOTAGE=1` rotates every table's
  counts by one category (the CTR-path negative control).
- `bindings/_mojolearn_forest_host.mojo`: `forest_host_gbdt_expand_ctr` and
  `forest_host_gbdt_ctr_sabotage`.
- `python/mojolearn/_gbdt_host.py`: parses the CTR and tensor CTR records with
  the Mojo reader's checks and refuses at load what `predict_floats` refuses
  (both kinds in one model, fewer tables than declared CTR columns, a CTR type
  with no apply-time arithmetic: Buckets, BinarizedTargetMeanValue,
  FloatTargetMeanValue, by name). The tensor hash is not recomputed in Python.
- `tools/identity_break.py`: lanes `gbdt-categorical-ctr-tables` and
  `gbdt-tensor-ctr-tables`. A GPU column fits and, under
  `MOJOLEARN_IDENTITY_GBDT_CTR_MODELS`, saves `<lane>.<fixture>.npz`; a CPU
  column (whose training refuses CTR tables) loads that file through
  `host_model`, so its train, infer and batch cells are HostGBDT predictions
  from the GPU's own saved model and its model cell is that file's hash.

## Evidence

Pending.
