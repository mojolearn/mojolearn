# GPU log loss: bounded probability metric

`mojolearn.metrics.log_loss` computes unweighted binary or multiclass
single-label negative log likelihood on the GPU. This extends A2 beyond
[confusion counts and precision/recall/F1](GPU_CLASSIFICATION_METRICS.md).
It does not implement ROC-AUC or precision-recall curves.

```python
import numpy as np
from mojolearn import metrics

probabilities = np.array([[0.8, 0.2], [0.1, 0.9]], dtype=np.float32)
loss = metrics.log_loss(
    ["no", "yes"], probabilities, labels=["no", "yes"],
    numeric_mode="identical",
)
```

## Contract

Targets are nonempty, one-dimensional strings or integers (including bools),
with one label per probability row. Probabilities must be finite Float32
values in `[0, 1]`; other dtypes require an explicit cast. Sample weights,
multilabel and multioutput targets are refused. Explicit labels define the
probability-column order and must be unique; every target must belong to that
class list. Inferred labels use sorted observed classes. At least two labels
are required; provide explicit labels when only one class occurs in the
observed targets.

Binary probability shapes `(n,)` and `(n, 1)` contain probabilities of the
second label. Python expands these to `[1 - p, p]` in Float32. Matrix inputs
have shape `(n, k)` with at least two columns. Both `n` and `n * k` must fit
in Int32. `normalize` accepts only bool values.

Probability matrix rows are checked with a Float64 host sum and must sum
to one within `sqrt(eps)`, where
`eps = np.finfo(np.float32).eps` (about `1.19209e-7`). The tolerance is about
`3.45267e-4`. Accepted rows are **not renormalized**. The GPU clips selected
class probabilities to `[eps, 1 - eps]` before taking their negative log.
Thus a probability of zero produces a finite clipped penalty, and even a
probability of one has a small positive clipped loss.

`normalize=True` returns the mean loss; `False` returns the sum. Both are
Python floats containing a GPU Float32 scalar. `numeric_mode=None` resolves
the current process default; explicit `fast`, `deterministic` or `identical`
selects that compiled metrics artifact without changing the default.

The mathematical reference is
[sklearn 1.8 log loss](https://scikit-learn.org/1.8/modules/generated/sklearn.metrics.log_loss.html).
This is an independent GPU implementation with a narrower input contract,
explicit caller label ordering and Float32 arithmetic; it does not claim
sklearn bit identity. Invalid probability sums are refused rather than
silently repaired.

## Arithmetic and qualification

Python validates probabilities, expands binary columns and encodes targets.
Clipping, selected-class negative log, reduction and mean/sum finalization
execute on the GPU. The implementation uses the mode-aware `identical_log`
path and 256-value slabs followed by an ordered GPU final fold. IDENTICAL
uses a fixed logical 256-slot reduction tree and portable logarithm; FAST
and DETERMINISTIC use the existing mode-specific block fold. Native host
entry points validate encoded labels and probability ranges; row-sum
validation belongs to the public Python boundary. Calls currently upload
host arrays; this is not a fully resident training/scoring pipeline.

Validation for this implementation turn is intentionally limited to build
and smoke checks, as requested by the user. Neither the fixed schedule nor
existing qualification of other metrics certifies log-loss bit identity.
Full adversarial numerical oracles, repeated/interleaved-mode fingerprint
qualification, cross-vendor comparisons and throughput measurements remain
pending. No cross-vendor, bitwise-qualification or performance claim is made.

The scoped run builds the shared metrics extension in all three modes and
runs its build smoke plus `checks/log_loss_smoke.py`. See the
[build/smoke evidence](../../bench/results/log_loss_2026-09-10/RESULTS.md)
for completed modes and results; this is not the deferred qualification gate.
