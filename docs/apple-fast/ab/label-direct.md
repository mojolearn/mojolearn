# Label preprocessing: measured bundle prepared as default

Promotion branch: lane/apple-fast-label-default, based on main61ea51757.
Measured source: dbe2ab85a, original base35a72c5ac. Current-main source drift
in the affected routes is comments only. Only the presence/scatter bundle
was measured; it is prepared as the Apple FAST default with
`MOJOLEARN_LABEL_DIRECT_OFF` restoring both original paths. The original
opt-in define names below describe the measured experiment, not the new
rollback interface. IDENTICAL and other vendors remain on their old paths.

M3 taxi348.254458 ->327.625833ms, istella33.176500 ->27.578792ms;
one run per arm, board digests unchanged. Quality tag gap26-label-quality
passed both independent public-path oracles(41,697,776 indicator cells each).
See EXPERIMENTS.md for full evidence and coverage limitations. Default/OFF
compile and manager merge remain outstanding; no new timing is required.

## Original experiment

Base: origin/main 35a72c5ac. Branch: lane/apple-fast-label-direct.
No local builds or tests; manager owns remote compilation and M3 timing.

Bundle define `MOJOLEARN_LABEL_DIRECT` enables both. Two independent opt-in FAST + Apple build defines:

* `MOJOLEARN_LABEL_PRESENT`: reuse the existing exact GPU category presence
  stages for LabelEncoder, LabelBinarizer and MultiLabelBinarizer fitting.
  Eliminates the label sort, n-word distinct-output allocation and download
  when all labels are integers in [0,4096). GPU validation rejects every
  other input to the unchanged general GPU sort path. This was absent from
  the old batch label path. Category-encoder presence is already main's
  default, but label fitting does not currently use it.
* `MOJOLEARN_LABEL_SCATTER`: for zero-negative indicators only, device memset
  then one positive store per row, instead of a full dense output kernel
  repeating lookup/division for every class. Explicit clear preserves reused
  output regions, arbitrary positive labels, unknown labels and binary mode.
  Negative labels other than zero keep the existing kernel.

Bindings: `bindings/build_x_prep.sh` only. Build FAST with
`MOJOLEARN_MOJO_BUILD_FLAGS='-D MOJOLEARN_LABEL_PRESENT -D MOJOLEARN_LABEL_SCATTER'`.
A/B against fresh main with the same standard FAST dependencies. Suggested
first arms: label-binarizer taxi and istella (both defines), then
multilabel-binarizer taxi and istella (PRESENT only). One run per arm; never
retime opponents. If label improves, isolate PRESENT vs SCATTER only as needed.

Quality command in each built arm:

```
PYTHONPATH=python MOJOLEARN_NUMERIC_MODE=fast MOJOLEARN_VENDOR=apple \
  ~/board-0834/cache/venv/bin/python tools/label_fast_quality.py
```

This uses public estimators and compares ALL output cells against independently
constructed mathematical indicators, fitted class values, LabelEncoder codes,
and inverse-transform values. Tests board-like label width, non-default positive
and negative outputs, binary/single classes, unseen queries, duplicate/empty
multilabel rows, range and fractional fallbacks, and signed zeros. Optional
`--dump FILE.npz` records outputs for exact main/candidate comparison. It never
fits or times a scored opponent. Expected bits: unchanged.
