# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# SHIPS: compiled into a CPU host binding (python/mojolearn/host_surface.py names which); product, not only a check.
"""Out-of-sample labels for the transductive clusterings, on the host
(lane/inference-transductive-predict, 2026-09-15).

NEW CAPABILITY, NOT A REFERENCE FEATURE. Neither cuML nor scikit-learn
labels a new row under a fitted DBSCAN or AgglomerativeClustering
(DEVIATION 2740). This file is the CPU spelling of
`core/labeled_reference_predict.mojo`, the device pass; the two are held
to the same bytes by `tools/identity_break.py`'s infer and batch parts.

THE RULE, ONE FUNCTION FOR BOTH ESTIMATORS. Given reference rows `R`
(`n_refs x n_features`), one Int32 `key` and one Int32 `label` per
reference row, a query row `q` gets the label of the reference row that is
FIRST in the total order `(distance, key, reference index)`, among the
reference rows whose distance passes the radius test when there is one; a
query no reference row passes gets `-1`.

    distance   the DBSCAN eps predicate's own accumulator
               (`dbscan/impl/neighbors/epsilon_neighborhood.mojo::_eps_acc`,
               restated here as `dbscan/host/dbscan_oracle.mojo::
               host_brute_eps_row` restates it): features ascending,
               `diff = ftz(q_k - ftz(r_k))`, then on the L2 arm
               `acc = ftz(identical_mul_add(diff, diff, acc))` (a SQUARED
               distance, never rooted) and on the L1 arm
               `acc = ftz(acc + abs(diff))`.
    radius     `acc <= thresh`, `thresh = Float32(eps * eps)` on L2 and
               `Float32(eps)` on L1 (`dbscan_metric_threshold`).
    NaN        a reference row whose accumulator is NaN is never chosen.

DBSCAN passes its core samples as the references, each core sample's
training index as its key and the fit's eps: the label of the nearest core
sample within eps, ties to the lowest (distance, index), else noise.
AgglomerativeClustering (single linkage) passes every training row, the
fit's label as the key and no radius: the cluster at the smallest
single-linkage distance (the smallest distance to any member), ties to the
lowest cluster label, then the lowest training index.

THE NEGATIVE CONTROL. `-D MOJOLEARN_HOST_SABOTAGE=1` (the estimators
family's define) or `-D MOJOLEARN_TRANSDUCTIVE_PREDICT_SABOTAGE=1` (this
pass alone) XORs 1 into every non-noise predicted label, so any held-out
call with one clustered row moves.
"""
from std.sys.compile import is_defined

from checks.numerics import ftz, identical_mul_add


comptime LABELED_PREDICT_HOST_SABOTAGE = (
    is_defined["MOJOLEARN_HOST_SABOTAGE"]()
    or is_defined["MOJOLEARN_TRANSDUCTIVE_PREDICT_SABOTAGE"]()
)

#: The metric ids of `dbscan/impl/neighbors/epsilon_neighborhood.mojo`,
#: restated because that file imports the GPU.
comptime LABELED_METRIC_L2 = 0
comptime LABELED_METRIC_L1 = 1


def labeled_reference_threshold(metric: Int, eps: Float64) -> Float32:
    """`dbscan_metric_threshold`: `Float32(eps * eps)` on L2, `Float32(eps)`
    on L1."""
    if metric == LABELED_METRIC_L1:
        return Float32(eps)
    return Float32(eps * eps)


def labeled_reference_validate(
    n_refs: Int, n_queries: Int, n_features: Int, metric: Int
) raises:
    """The refusals both bindings raise, in one order and one wording."""
    if n_refs < 1:
        raise Error(
            "labeled_reference_predict: the fitted model holds no reference"
            " rows (a DBSCAN fit with no core sample predicts every row as"
            " noise in Python and never calls this); refused by name"
        )
    if n_queries < 1:
        raise Error(
            "labeled_reference_predict: X has no rows; refused by name"
        )
    if n_features < 1:
        raise Error(
            "labeled_reference_predict: X has no features; refused by name"
        )
    if metric != LABELED_METRIC_L2 and metric != LABELED_METRIC_L1:
        raise Error(
            "labeled_reference_predict: metric must be 0 (L2) or 1 (L1), got "
            + String(metric)
        )


@fieldwise_init
struct LabeledReferencePrediction(Movable):
    var labels: List[Int32]
    var refs: List[Int32]


def host_labeled_reference_predict(
    refs: List[Float32],
    n_refs: Int,
    keys: List[Int32],
    ref_labels: List[Int32],
    queries: List[Float32],
    n_queries: Int,
    n_features: Int,
    metric: Int,
    thresh: Float32,
    has_thresh: Bool,
) raises -> LabeledReferencePrediction:
    """The rule in the module docstring, one query at a time. `refs` is the
    index of the chosen reference row per query, -1 where none passed."""
    labeled_reference_validate(n_refs, n_queries, n_features, metric)
    if (
        len(refs) < n_refs * n_features
        or len(keys) < n_refs
        or len(ref_labels) < n_refs
        or len(queries) < n_queries * n_features
    ):
        raise Error(
            "labeled_reference_predict: a buffer is shorter than its shape;"
            " refused by name"
        )
    var out_labels = List[Int32](capacity=n_queries)
    var out_refs = List[Int32](capacity=n_queries)
    var d = n_features
    for q in range(n_queries):
        var best = -1
        var best_acc = Float32(0.0)
        var best_key = Int32(0)
        for r in range(n_refs):
            var acc = Float32(0.0)
            for k in range(d):
                var diff = ftz(queries[q * d + k] - ftz(refs[r * d + k]))
                if metric == LABELED_METRIC_L1:
                    acc = ftz(acc + abs(diff))
                else:
                    acc = ftz(identical_mul_add(diff, diff, acc))
            if has_thresh and not (acc <= thresh):
                continue
            var key = keys[r]
            if (best < 0 and acc == acc) or acc < best_acc or (
                acc == best_acc and key < best_key
            ):
                best = r
                best_acc = acc
                best_key = key
        var label = Int32(-1)
        if best >= 0:
            label = ref_labels[best]
            comptime if LABELED_PREDICT_HOST_SABOTAGE:
                if label >= Int32(0):
                    label = label ^ Int32(1)
        out_labels.append(label)
        out_refs.append(Int32(best))
    return LabeledReferencePrediction(out_labels^, out_refs^)
