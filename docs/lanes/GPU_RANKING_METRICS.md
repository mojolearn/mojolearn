# Binary GPU ROC-AUC and precision-recall curves

`mojolearn.metrics.roc_auc_score` and `precision_recall_curve` score existing
binary classifier predictions. They do not add learning-to-rank objectives,
query groups, pairwise losses, LambdaRank or ranking-tree training.

```python
import numpy as np
from mojolearn import metrics

labels = np.array([0, 0, 1, 1], dtype=np.int32)
scores = np.array([0.1, 0.4, 0.35, 0.8], dtype=np.float32)
auc = metrics.roc_auc_score(labels, scores, numeric_mode="identical")
precision, recall, thresholds = metrics.precision_recall_curve(
    labels, scores, numeric_mode="identical",
)
```

## Bounded scoring contract

Targets are nonempty, one-dimensional strings or integers (including bools),
with at most two observed classes and one finite, strictly Float32 score per
row. Other score dtypes require an explicit cast. At most Int32.max rows
are accepted; available host/device memory may impose a lower limit. Scores can be probabilities or unrestricted decision scores;
there is no probability clipping or row normalization. Larger scores indicate
the positive class. Sample weights, multiclass and multilabel scoring remain
outside this slice.

ROC-AUC uses the greater of the two sorted labels as the positive class.
Both classes must occur; singleton-class ROC-AUC is refused. `max_fpr=None`
or `1` selects the full area; partial AUC is unsupported. Only the default
`average="macro"`, `multi_class="raise"` and `labels=None` are accepted.
It returns a Python float containing the GPU Float32 result. This bounded
surface follows the binary interpretation in the
[sklearn 1.8 ROC-AUC contract](https://scikit-learn.org/1.8/modules/generated/sklearn.metrics.roc_auc_score.html),
not its multiclass, multilabel or partial-AUC extensions.

Precision-recall accepts an explicit `pos_label` of the same type as the
targets. Without it, only the conventional integer label sets `{0}`, `{1}`,
`{-1}`, `{0, 1}` and `{-1, 1}` infer positive label 1. String labels and
other integers require an explicit positive label. An absent positive label
is allowed: the function warns and returns recall 1 at every threshold,
followed by the terminal zero. `drop_intermediate=True` is refused.

Precision-recall returns three Float32 arrays with increasing unique score
thresholds. Precision
and recall arrays have one extra endpoint, `(precision=1, recall=0)`, without
a corresponding threshold. A threshold predicts positive for scores greater
than or equal to it. These endpoint and threshold conventions follow
[sklearn 1.8 precision-recall curves](https://scikit-learn.org/1.8/modules/generated/sklearn.metrics.precision_recall_curve.html).

All calls resolve `numeric_mode=None` from the current process default, or
select the requested FAST, DETERMINISTIC or IDENTICAL metrics artifact.
Float32 outputs are not a claim of sklearn Float64 bit identity.

## GPU implementation and qualification

Python validates and encodes targets. An existing stable 32-pass UInt32
radix sort orders Float32 score keys ascending on the GPU, carrying encoded
binary labels. An integer exclusive scan supplies prefix positive counts.
Parallel group-start marking, an integer scan and compaction identify complete
tie groups, reusing sort scratch buffers. Separate groups compute PR suffix
ratios and thresholds, or exact Int64 AUC contributions, in parallel. Only
AUC uses a final single-thread fold in ascending group order. Equal scores
form complete threshold groups, so their row order cannot change which
samples a threshold includes.

AUC accumulates the exact Int64 numerator
`sum(group_positive * (2 * negatives_before + group_negative))` and divides
by `2 * total_positive * total_negative`. Both integers convert separately
to Float32 before the mode-aware division. PR ratios also use mode-aware
Float32 division; IDENTICAL selects portable division. The Int32 row bound
keeps the integer count products within Int64. This fixes the count arithmetic
but does not qualify final Float32 values across devices.

The 32 radix passes, serial block-total scans and serial AUC-only final fold
remain potential scaling limits. The AUC fold takes O(number of unique
scores) work; PR output computation is parallel across groups. Scaling and
throughput have not been measured; no high-throughput claim is made.
Signed zeros share a canonical positive-zero threshold; distinct subnormal
score bit patterns remain ordered rather than being flushed into zero ties.
Calls still upload host arrays and return host results; this does not certify
a fully resident training/scoring pipeline.

This turn prioritizes implementation and limits validation to local builds
and small smoke checks at the user's request. Full numerical qualification,
large-input scaling, cross-device comparisons and throughput measurements
remain pending. Neither existing metric evidence nor the IDENTICAL mode name
qualifies these new kernels across devices. No full sklearn-parity or
cross-device bit-identity claim is made.

See the [local build/smoke record](../../bench/results/binary_ranking_2026-09-10/RESULTS.md)
for completed modes, commands and results. That record is separate from the
broader qualification work listed above.
