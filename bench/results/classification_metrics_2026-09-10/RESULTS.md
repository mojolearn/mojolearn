# GPU classification metrics, local M4 qualification

PASS on Apple M4, 2026-09-10. Unweighted confusion matrix, precision, recall and
F1 are implemented in FAST, DETERMINISTIC and IDENTICAL. This is local
source-checkout correctness evidence, not CUDA/HIP, installed-wheel,
end-to-end pipeline identity or throughput qualification.

| Gate | Result |
| --- | --- |
| Combined Python API/protocol/tree regression suite | 317 passed, including 85 new classification tests |
| Native classification, all three modes | PASS; independent integer and Float64 formulas, repeats, ragged rows, selected labels, undefined flags, synthetic large counters and bounds |
| Public classification, independent hand-count oracle | 2,760 checks passed; all three modes interleaved and revisited |
| Public classification, additional sklearn 1.8 oracle | Same 2,760 checks passed, including warning presence |
| Existing public GPU MSE/MAE/RMSE | All 261 checks passed against rebuilt artifacts |
| RF/ET prediction through sklearn make_scorer and new GPU metrics | Six fits passed, both classifiers in all three modes |

The classification public gate records 1,308 distinct mode/case fingerprints
and compares repeated calls and reverse/permuted row-pair results exactly.
The forest witness uses deliberately perturbed reference labels to produce
nontrivial scores; it is a scoring integration check, not a generalization
benchmark. All metric arithmetic runs on GPU. Host loops in checks are
independent test oracles, not fallback implementations.

See [native details](NATIVE_EVIDENCE.md), `native-toolchain.txt`,
`environment.json`, `native-binaries.sha256`, and `provenance.json` for
compiler/device/Python metadata, source hashes and the nine metrics/RF/ET
artifacts used. Production source was stable after the first successful
native FAST check. RF/ET artifacts were reused from the preceding qualified
lane work; all three metrics artifacts were rebuilt in this batch.

## Reproduction

All builds and GPU executions used `tools/with_build_lock.sh` to serialize
access to the shared local device. No remote endpoint or job was touched.
The new task combines the individual native/build/public commands used here:

```sh
MOJOLEARN_PYTHON="$PWD/.pixi/envs/default/bin/python" pixi run check-classification-metrics
```

Actual native executable builds used `pixi run mojo build -I .`, adding
`-D MOJOLEARN_NUMERIC_DETERMINISTIC=1` or
`-D MOJOLEARN_NUMERIC_IDENTICAL=1` for the corresponding mode, on
`metrics/checks/classification_metrics_check.mojo`. Executables are retained
under `build/classification_metrics/`; they are not committed.
Public builds ran `MOJOLEARN_NUMERIC_MODE=<mode> sh bindings/build_metrics.sh`
with `MOJOLEARN_PYTHON=$PWD/checks/pipeline_python.sh` and
`MOJOLEARN_PIPELINE_PYTHON=$PWD/.pixi/envs/default/bin/python`.
The FAST builder smoke passed, including every new kernel family and the
existing spectral smoke; it reported 73 AIR blobs and minos 11.0. The existing
builder skips its own DET/IDENT smoke; these are qualified here by the separate
public gates, not counted as builder smoke passes.

The hand-count gate and regression gate used that same Python wrapper:
`PYTHONPATH=python checks/pipeline_python.sh checks/classification_metrics_binding.py`
and `checks/regression_errors_binding.py`. Additional sklearn integration used
the isolated environment directly, without a broad DYLD_LIBRARY_PATH override:

```sh
tools/with_build_lock.sh env PYTHONPATH=python \
  /tmp/mojolearn-b1-sklearn/bin/python checks/classification_metrics_binding.py --require-sklearn
tools/with_build_lock.sh env PYTHONPATH=python \
  /tmp/mojolearn-b1-sklearn/bin/python checks/classification_forest_scoring.py
PYTHONPATH=python /tmp/mojolearn-b1-sklearn/bin/python -m pytest -q \
  python/mojolearn/tests/test_classification_metrics.py \
  python/mojolearn/tests/test_regression_metrics.py \
  python/mojolearn/tests/test_forest_protocol.py \
  python/mojolearn/tests/test_min_split_gain.py \
  python/mojolearn/tests/test_min_child_hessian.py \
  python/mojolearn/tests/test_tree_input_layout.py
```

## Retained failures and limits

Two initial native compile errors are retained: Int32/Int fixture comparisons,
and generic output-store casts. Both were corrected before the successful
native runs and public builds. The initial forest scorer fixture omitted
`pos_label` for string classes; sklearn validated its default integer 1 before
calling the metric. Supplying the valid string label fixed the fixture.
`forest-scoring.initial-pos-label-failure.log` preserves that finding.

Successful native runs emit `Context leak detected, CoreAnalytics returned
false`. Each public classification oracle run emits 54 instances and the
regression gate emits five. All processes still exited successfully with PASS
markers. The diagnostic also appeared in preceding lane work; its origin and
memory implications remain unresolved. No memory-stability claim is made.

The new API rejects weights, empty inputs, floating labels, multilabel and
multioutput targets. Confusion output is capped at 4,096 classes and internal
Int32 counts require at most INT32_MAX rows. Ratios/averages are Float32 with
explicit count conversion and fixed class order; they do not claim Float64
accuracy or exact sklearn bits. PRF uses O(k) count storage but a serial GPU
class fold. Performance work, log loss and ranking curves remain on the plan.
