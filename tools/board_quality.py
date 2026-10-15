# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn
"""Versioned, symmetric quality gates for board comparisons.

These are engineering admission margins, not statistical equivalence tests.
See docs/RACE_FAIRNESS_DECISIONS.md. Iterations are observations, not quality.
Unknown metrics require review; finite output alone is not equal quality.
"""
import math

POLICY = "mojolearn.quality-parity.v7"
# Absolute score units (one percentage point for bounded scores).
ABSOLUTE = dict.fromkeys(("accuracy", "auc", "roc_auc", "r2", "recall_at_k",
    "recall_at_10", "trustworthiness_k15", "explained_variance_ratio_sum",
    "explained_variance_fraction", "silhouette", "modularity"), 0.01)
# SCORER GAPS (lane quality-scorers, 2026-10-15): metrics the board already recorded on both arms of a
# race but that had no declared rule, so every such race read INSUFFICIENT
# (~/mojolearn-evidence/quality-vs-opponent-20261015.md "Scorer gaps"). Declared before any rerun.
#   mean_canonical_corr (cca, pls-canonical): a correlation, bounded like r2.
#   mean_r2 (cross-val-score): r2 averaged over the same folds.
#   pvalue (permutation-test): the Monte Carlo error of a permutation p-value with 9,999
#   resamples is at most sqrt(.25 / 9999) = 0.005, so 0.01 is two standard errors.
#   max_mean_shift_over_std (resample): the resample mean's shift in units of the column's
#   standard deviation; 0.01 std is the sampling noise of a mean over 10^4 rows.
ABSOLUTE.update(mean_canonical_corr=0.01, mean_r2=0.01, pvalue=0.01, max_mean_shift_over_std=0.01)
# Log density is already logarithmic: a relative difference in its numeric value
# is meaningless. 0.05 nats corresponds to exp(0.05), about 1.051, in density.
ABSOLUTE.update(mean_log_likelihood=0.05, mean_nll=0.05,
                mean_log_predictive_density=0.05)
RELATIVE = dict.fromkeys(("rmse", "masked_rmse", "forecast_rmse", "logloss",
    "inertia", "perplexity", "loss_last_step", "relative_reconstruction_error",
    "relative_reconstruction_error_100k_rows"), 0.05)
# SCORER GAPS (2026-10-15), declared before any rerun: errors and estimates both arms record.
#   kernel_rel_error (additive-chi2, skewed-chi2, poly-count-sketch, nystroem, rbf-sampler): the
#   approximation's relative Frobenius error against the exact kernel, lower is better.
#   mean_abs_excess_kurtosis (fastica): the non-Gaussianity of the recovered sources.
#   standard_error (bootstrap): the bootstrap estimate; its Monte Carlo error with 9,999 resamples
#   is about 1 / sqrt(2 * 9999) = 0.7%, far inside 5%.
#   residual_std (stl): the decomposition's residual spread on the same series.
RELATIVE.update(kernel_rel_error=0.05, mean_abs_excess_kurtosis=0.05, standard_error=0.05,
                residual_std=0.05)
# Exact contracts: random-trees-embedding activates exactly one leaf per tree in every row.
RELATIVE.update(nonzeros_per_row=0.0)
# mean_llf (garch): the series' total log likelihood at the fitted parameters. A likelihood-ratio
# test reads differences of total log likelihood directly; 0.05 nats is negligible at any length.
ABSOLUTE.update(mean_llf=0.05)
# A common reference bound, not a demand that fp32 and bf16 have equal errors.
# This gate deliberately allows the opponent's native fast precision.
REFERENCE = {"max_rel_err_vs_fp64": 0.01, "relative_residual": 1e-4,
             "relative_gram_difference": 1e-4,
             # FAIRNESS FIX (2026-10-13 audit): validate the retained Q
             # factor against its defining contract before new QR races.
             "relative_orthogonality_error": 1e-4,
             # FAIRNESS FIX (2026-10-13 audit): classical PCA inference
             # already records its projection error against FP64 evaluation
             # of the same fitted model. Apply the existing linear-algebra
             # reference bound instead of treating the metric as unknown.
             "transform_max_rel_err_own_fp64": 1e-4,
             # FAIRNESS FIX (2026-10-13 audit): these lanes had recorded
             # reference distances but no declared admission rule. Limits
             # precede the repaired run and do not depend on its timing.
             "rel_diff_vs_sklearn": 1e-4,
             "rel_fro_vs_torch_eager_fp32": 0.01,
             "optimizer_update_error_vs_torch_fp32": 1e-4,
             # Declared before the repaired dropout measurements. Each arm
             # must implement channel-wise inverted dropout at the requested
             # probability; their random masks need not be identical.
             "dropout_contract_relative_error": 1e-6,
             "dropout_probability_z": 6.0,
             # SCORER GAPS (2026-10-15), declared before any rerun. Distances to ONE named arm's
             # output on the same inputs (and, for the layers, the same weights): that arm's own
             # value is 0 by definition (SELF_REFERENCE below).
             # forward lanes (tools/bench_board_neural.py) and the schedules: max |o - ours| / max
             # |ours|; 0.01 is the same common bound as rel_fro_vs_torch_eager_fp32, which admits
             # an opponent's native bf16/tf32.
             "max_rel_diff_vs_ours": 0.01,
             # fp32 reductions and elementwise gradients of the same inputs (clip-grad-norm, cross-entropy)
             "norm_rel_diff_vs_ours": 1e-4,
             "grad_max_rel_diff_vs_ours": 1e-4,
             "loss_rel_err_vs_fp64": 1e-4,
             # kpss: the test statistic against statsmodels' on the same series
             "stat_max_rel_diff_vs_statsmodels": 1e-4,
             # pagerank: the L1 distance to networkx's scores over the bound the shared stopping rule
             # allows (bench_board_algos.pagerank_l1_bound: two arms each within alpha / (1 - alpha)
             # times the final L1 change n x tol of the fixed point); 1 is that bound.
             "l1_vs_networkx_over_stop_bound": 1.0,
             # connected components are exact: the same partition up to relabeling, or 1 - ARI.
             "partition_disagreement_vs_networkx": 0.0}
# SCORER GAPS (2026-10-15): agreement fractions in [0, 1] with one named arm; 1 is that arm's own
# output. Both arms' disagreement (1 - value) must be at most the margin (bounded-score policy).
AGREEMENT = {"documents_equal_to_ours": 0.01, "jaccard_vs_ours": 0.01, "jaccard_vs_sklearn": 0.01,
             "flag_agreement_vs_statsmodels": 0.01, "d_agreement_vs_statsmodels": 0.01}
# The arm a reference or agreement metric is measured against, and that arm's own value. A race
# that scored the arm itself (the stored torch-eager-fp32, networkx-cpu or statsmodels-cpu cell)
# records no distance to itself; the comparison reads this value instead of INSUFFICIENT.
SELF_REFERENCE = {
    "rel_fro_vs_torch_eager_fp32": ("torch-eager-fp32", 0.0),
    "optimizer_update_error_vs_torch_fp32": ("torch-eager-fp32", 0.0),
    "rel_diff_vs_sklearn": ("sklearn-cpu", 0.0),
    "max_rel_diff_vs_ours": ("ours", 0.0),
    "norm_rel_diff_vs_ours": ("ours", 0.0),
    "grad_max_rel_diff_vs_ours": ("ours", 0.0),
    "stat_max_rel_diff_vs_statsmodels": ("statsmodels-cpu", 0.0),
    "l1_vs_networkx_over_stop_bound": ("networkx-cpu", 0.0),
    "partition_disagreement_vs_networkx": ("networkx-cpu", 0.0),
    "documents_equal_to_ours": ("ours", 1.0),
    "jaccard_vs_ours": ("ours", 1.0),
    "jaccard_vs_sklearn": ("sklearn-cpu", 1.0),
    "flag_agreement_vs_statsmodels": ("statsmodels-cpu", 1.0),
    "d_agreement_vs_statsmodels": ("statsmodels-cpu", 1.0),
}


def metric_value(cell, quality, metric):
    """The arm's recorded value, or the self-reference value when the arm IS the metric's reference."""
    value = quality.get(metric)
    if value is None and metric in SELF_REFERENCE and cell.get("arm") == SELF_REFERENCE[metric][0]:
        return SELF_REFERENCE[metric][1]
    return value


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


# A scored outcome that is itself a quality failure: the scorer ran and the arm's output broke its contract
# (nonfinite, a host-reference mismatch, wrong output fields or shape). Every other recorded `error` means the
# scorer did not produce a metric (it crashed, its reference data or bindings were missing, it never ran): that
# is "quality not scored", never a quality failure of the arm (lane harness-evidence, 2026-10-14).
SCORED_FAILURE_ERRORS = ("host reference mismatch", "nonfinite host reference output",
                         "host reference output shape differs", "host reference output fields differ",
                         "host reference output is empty", "non-numeric host reference output",
                         # tools/bench_board_algos.py quality(): the scorer ran and rejected the arm's output
                         "nonfinite", "invalid optimizer output")
NOT_SCORED = "quality not scored (scorer failed)"


def scorer_failure(quality):
    """The reason this arm's quality was NOT scored, or None when a scorer produced a result.

    `scorer: "failed"` / `"not-run"` (set by the race drivers since 2026-10-14) names it outright; older cells
    carry the scorer's exception in `error`. A scored failure (SCORED_FAILURE_ERRORS, `finite: False`,
    `host_reference_passed: False`) is a result, not a scorer failure."""
    q = quality or {}
    if q.get("scorer") in ("failed", "not-run"):
        return str(q.get("scorer_error") or q.get("error") or "scorer %s" % q["scorer"])
    if q.get("scorer") == "scored":
        return None
    err = q.get("error")
    if err and not any(k in str(err).lower() for k in SCORED_FAILURE_ERRORS):
        return str(err)
    if q.get("quality_error"):              # tools/bench_board_extra.py: the per-arm scorer raised
        return str(q["quality_error"])
    return None


def scored_failure(quality):
    """True when a scorer ran and recorded a quality failure for this arm."""
    q = quality or {}
    if scorer_failure(q) is not None:
        return False
    return bool(q.get("error") or q.get("finite") is False or q.get("host_reference_passed") is False)


def compare(ours, theirs):
    """Return every check and its declared margin, with no timing input."""
    qo, qt = ours.get("quality") or {}, theirs.get("quality") or {}
    checks, failures, not_scored = [], [], []
    for arm, quality, cell in ((ours.get("arm"), qo, ours), (theirs.get("arm"), qt, theirs)):
        why = scorer_failure(quality)
        if why is None and cell.get("status") == "not-scored":
            why = str(cell.get("scorer_error") or "scorer did not run")
        if why is not None:
            not_scored.append("%s: %s: %s" % (NOT_SCORED, arm, why[:300]))
        elif scored_failure(quality):
            failures.append("recorded quality failure: " + str(arm))
    for metric in sorted(set(qo) | set(qt)):
        if metric not in ABSOLUTE and metric not in RELATIVE and metric not in REFERENCE \
                and metric not in AGREEMENT:
            continue
        a, b = metric_value(ours, qo, metric), metric_value(theirs, qt, metric)
        if not number(a) or not number(b):
            checks.append(dict(metric=metric, ours=a, opponent=b, verdict="INSUFFICIENT",
                               reason="metric missing or nonfinite on one arm"))
            continue
        difference = abs(a - b)
        if metric in ABSOLUTE:
            kind, margin, observed = "absolute", ABSOLUTE[metric], difference
        elif metric in RELATIVE:
            kind, margin = "symmetric-relative", RELATIVE[metric]
            scale = max(abs(a), abs(b))
            observed = difference / scale if scale else 0.0
        elif metric in AGREEMENT:
            kind, margin = "both-disagreements-at-most", AGREEMENT[metric]
            observed = max(1.0 - a, 1.0 - b)
        else:
            kind, margin, observed = "both-reference-errors-at-most", REFERENCE[metric], max(abs(a), abs(b))
        checks.append(dict(metric=metric, ours=a, opponent=b, comparison=kind,
                           margin=margin, observed=observed,
                           verdict="PASS" if observed <= margin else "FAIL"))
    # FAIRNESS FIX (2026-10-13 audit): a one-cluster density result has no
    # silhouette. Exact partition equivalence with the same noise mask still
    # proves output-quality parity. Partial label agreement is neither an
    # admission nor a quality failure; it requires an intrinsic metric.
    links = [q for q in (qo, qt) if "ari_vs_ours" in q]
    if (not checks and ours.get("lane") in ("dbscan", "hdbscan")
            and ours.get("lane") == theirs.get("lane")
            and type(qo.get("rows")) is int and type(qt.get("rows")) is int
            and qo["rows"] > 0 and qo["rows"] == qt["rows"]
            and links and all(number(q.get("ari_vs_ours")) and number(q.get("noise_agreement_vs_ours"))
                              and q["ari_vs_ours"] == 1.0 and q["noise_agreement_vs_ours"] == 1.0 for q in links)):
        checks.append(dict(metric="partition_equivalence", ours=1.0, opponent=1.0,
            comparison="same partition and noise mask on every consumed row", margin=0.0,
            observed=0.0, rows=qo["rows"], verdict="PASS"))
    if failures or any(c["verdict"] == "FAIL" for c in checks):
        verdict = "FAIL"
    elif not_scored or not checks or any(c["verdict"] == "INSUFFICIENT" for c in checks):
        # A metric the scorer never produced is insufficient evidence, never a pass or a failure.
        verdict = "INSUFFICIENT"
    else:
        verdict = "PASS"
    out = dict(policy=POLICY, verdict=verdict, checks=checks, failures=failures,
               interpretation="engineering margin; not a statistical equivalence claim")
    if not_scored:
        out["not_scored"] = not_scored
    return out
