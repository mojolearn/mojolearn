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

Fixture probe, Apple M4 Metal, bindings built from 1386833b4, base fixture:

- gbdt-categorical-ctr-tables: the saved model carries `ctr_columns`, 8
  `ctr_table` records (Borders at priors {0,1}, {0.5,1}, {1,1} with two target
  classes, and FeatureFreq with denominator 20000, for each of the two
  categorical inputs) and 52 `ctr_entry` records; 17 model columns for 11 raw
  inputs; the trees split on CTR columns. GPU and HostGBDT hash equal on the
  held-out rows (unseen and seen-once categories included) and on the
  training rows; a NaN categorical value is refused by both with
  "categorical feature 0 row 3 is not finite".
- gbdt-tensor-ctr-tables: the saved model carries `tensor_ctr_registry 6 2`
  and two `feature_freq_tensor` records (the second with one split in its
  history, on the first tensor column); the tree splits on both tensor
  columns. GPU and HostGBDT hash equal on held-out and training rows; a NaN
  source value is refused by both with "tensor CTR split history cannot
  quantize NaN".

RunPod CPU pod guqti0t3eychjo (verified deleted, $0.009), commit e8cdd0aed:
the eight existing HostGBDT lanes on the base fixture with
`MOJOLEARN_IDENTITY_HOST_INFER=1`, diffed against the three 166-lane GPU
columns, read train IDENTICAL=8 (x4), infer/model IDENTICAL=16, batch
IDENTICAL=8. `docs_facts --check`, `wheel_ci pins` and `inventory` exit 0.

Identity columns, sabotage and the installed wheel: pending.
