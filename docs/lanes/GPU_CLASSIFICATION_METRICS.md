# GPU classification metrics: bounded A2 slice

The public surface adds `mojolearn.metrics.confusion_matrix`,
`precision_score`, `recall_score` and `f1_score`. [GPU log loss](GPU_LOG_LOSS.md)
is implemented separately with build/smoke validation only; ranking curves
remain planned. Functions accept explicit `numeric_mode=` and otherwise
resolve the process default at call time through the shared metrics loader.

```python
from mojolearn import metrics

matrix = metrics.confusion_matrix(
    ["no", "yes", "yes"], ["yes", "yes", "no"],
    labels=["no", "yes"], numeric_mode="identical",
)
f1 = metrics.f1_score(
    ["no", "yes", "yes"], ["yes", "yes", "no"],
    pos_label="yes", zero_division=0, numeric_mode="identical",
)
```

## Input and output contract

Inputs are nonempty, equally sized, one-dimensional single-label arrays.
A pair must contain either strings or integers; booleans count as integers.
Host encoding preserves large integers without narrowing their values.
Float labels, missing values, mixed string/integer labels, sample weights,
multilabel and multioutput targets are explicitly refused. At most
2,147,483,647 rows are supported by exact internal Int32 counters.

Default label order is the sorted observed union. Explicit labels must be
unique and nonempty and retain caller order. Confusion rows are true labels,
columns are predictions; rows involving an excluded label are dropped.
At least one selected label must occur in the true targets. Counts return
Int64; `normalize='true'`, `'pred'` or `'all'` returns Float32. Zero-mass
normalization denominators return zero. Confusion output is capped at 4096
labels, checked before allocating its quadratic output.

Precision/recall/F1 support `average='binary'`, `'micro'`, `'macro'`,
`'weighted'` or `None` (a Float32 vector in selected-label order).
Other averages return a Python float representing the GPU Float32 result.
Binary averaging uses `pos_label`, ignores `labels`, and rejects more than
two observed classes. An absent positive label is allowed for singleton
observations. String targets require a string positive label.

For nonbinary averaging, classes outside the selected output set remain in
native marginal counts so their false positives and false negatives are not
lost. Weighted averages use true support; when selected support sums to zero,
the result falls back to the unweighted mean of selected per-class values.
PRF uses linear class storage, with a native class-index bound of 715,827,882;
actual available host/device memory may impose a smaller practical limit.

`zero_division=0` and `1` substitute the chosen value where a metric is
undefined. The default `'warn'` substitutes zero and reports the requested
metric's native undefined flag. With sklearn installed, warnings use its
`UndefinedMetricWarning`; otherwise they use `RuntimeWarning`. Exact warning
text/count and sklearn's optional NaN zero-division behavior are not promised.

Label order, averaging and normalization follow the bounded portions of the
[sklearn 1.8 confusion matrix contract](https://scikit-learn.org/1.8/modules/generated/sklearn.metrics.confusion_matrix.html)
and [precision/recall/F-score contract](https://scikit-learn.org/1.8/modules/generated/sklearn.metrics.precision_recall_fscore_support.html).
The new ratios are Float32, so this is not a claim of sklearn Float64 bit
identity. The API deliberately refuses some inputs accepted by sklearn.

## Ownership, ABI and qualification

Python validates and encodes labels. Device kernels compute counts,
normalization, ratios and all averaging. Each call currently uploads host
arrays; this is a GPU metric API, not a fully resident pipeline.

`confusion_matrix(true_ptr, pred_ptr, out_ptr, [n, k, normalize])` uses Int32
encoded inputs, permits -1 for excluded labels, and writes Int64 counts or
Float32 normalized results. `precision_recall_fscore` takes
`[n, k, average, pos_idx, zero_division, n_selected]`, with selected labels
encoded first and the remaining observed labels appended. Its Float32
output has three metric rows followed by three undefined flags. No host
reduction reconstructs a metric from device counts.

Counts are exact Int32 integers under the row bound. Combined F1 denominators
and micro totals use Int64 before converting numerator and denominator
separately to Float32. Per-class ratios use the mode's division policy;
IDENTICAL selects portable division. Macro and weighted results fold classes
in the selected order with explicit Float32 operations. This fixes the
arithmetic schedule, but does not provide a Float64 or compensated sum;
rounding can accumulate with many classes. F1 uses `2 * TP / (true + predicted)`
directly, including when both precision and recall are zero.

Host checks in `python/mojolearn/tests/test_classification_metrics.py` exercise
actual public dispatch with a mocked native boundary and sklearn 1.8 oracles:
label subsets/order, absent classes, zero-support weighting, warning flags,
all averages, normalization, mode selection, ABI dtypes and input refusals.
Native and source-checkout public gates independently check device arithmetic;
passing the host tests alone does not qualify the GPU kernels. Three-vendor
qualification of these new kernels is pending; historical metric evidence
does not extend automatically to A2. No throughput claim is made here.

## Reproduce the bounded gate

```sh
MOJOLEARN_PYTHON="$PWD/.pixi/envs/default/bin/python" pixi run check-classification-metrics
```

The task serializes native checks and public extension builds for FAST,
DETERMINISTIC and IDENTICAL, then runs independent public hand-count oracles
with interleaved mode selection and repeated output fingerprints. The public
script can also use sklearn 1.8 as an additional oracle via
`checks/classification_metrics_binding.py --require-sklearn` in an environment
containing that version. Source-checkout artifacts are exercised here;
installed-wheel and CUDA/HIP qualification remain separate gates.

Local M4 qualification passed in all three modes: 317 combined Python tests,
2,760 public checks against an independent hand-count oracle and again with
sklearn 1.8, and six actual RF/ET fits scored through sklearn `make_scorer`.
The existing 261 public regression-metric checks also passed after rebuilding
the shared extension. See [recorded evidence](../../bench/results/classification_metrics_2026-09-10/RESULTS.md)
for provenance, commands and retained runtime diagnostics.

When using sklearn `make_scorer` with string class labels, supply a valid
`pos_label` explicitly, including with macro averaging: sklearn validates its
response metadata before invoking the scoring function. For example:

```python
from sklearn.metrics import make_scorer
scorer = make_scorer(metrics.f1_score, average="macro", pos_label="yes",
                     zero_division=0, numeric_mode="identical")
# scorer(fitted_classifier, X, y)
```

`checks/classification_forest_scoring.py` demonstrates this integration on
both forest classifiers; it requires their existing three-mode artifacts
and sklearn 1.8. It checks scoring, not generalization or pipeline identity.
