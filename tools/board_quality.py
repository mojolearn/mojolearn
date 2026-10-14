# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn
"""Versioned, symmetric quality gates for board comparisons.

These are engineering admission margins, not statistical equivalence tests.
See docs/RACE_FAIRNESS_DECISIONS.md. Iterations are observations, not quality.
Unknown metrics require review; finite output alone is not equal quality.
"""
import math

POLICY = "mojolearn.quality-parity.v2"
# Absolute score units (one percentage point for bounded scores).
ABSOLUTE = dict.fromkeys(("accuracy", "auc", "roc_auc", "r2", "recall_at_k",
    "recall_at_10", "trustworthiness_k15", "explained_variance_ratio_sum",
    "explained_variance_fraction", "silhouette", "modularity"), 0.01)
# Log density is already logarithmic: a relative difference in its numeric value
# is meaningless. 0.05 nats corresponds to exp(0.05), about 1.051, in density.
ABSOLUTE.update(mean_log_likelihood=0.05, mean_nll=0.05,
                mean_log_predictive_density=0.05)
RELATIVE = dict.fromkeys(("rmse", "masked_rmse", "forecast_rmse", "logloss",
    "inertia", "perplexity", "loss_last_step", "relative_reconstruction_error",
    "relative_reconstruction_error_100k_rows"), 0.05)
# A common reference bound, not a demand that fp32 and bf16 have equal errors.
# This gate deliberately allows the opponent's native fast precision.
REFERENCE = {"max_rel_err_vs_fp64": 0.01, "relative_residual": 1e-4,
             "relative_gram_difference": 1e-4,
             # FAIRNESS FIX (2026-10-13 audit): these lanes had recorded
             # reference distances but no declared admission rule. Limits
             # precede the repaired run and do not depend on its timing.
             "rel_diff_vs_sklearn": 1e-4,
             "rel_fro_vs_torch_eager_fp32": 0.01,
             "optimizer_update_error_vs_torch_fp32": 1e-4}


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def compare(ours, theirs):
    """Return every check and its declared margin, with no timing input."""
    qo, qt = ours.get("quality") or {}, theirs.get("quality") or {}
    checks, failures = [], []
    for arm, quality in ((ours.get("arm"), qo), (theirs.get("arm"), qt)):
        if quality.get("error") or quality.get("finite") is False or quality.get("host_reference_passed") is False:
            failures.append("recorded quality failure: " + str(arm))
    for metric in sorted(set(qo) | set(qt)):
        if metric not in ABSOLUTE and metric not in RELATIVE and metric not in REFERENCE:
            continue
        a, b = qo.get(metric), qt.get(metric)
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
        else:
            kind, margin, observed = "both-reference-errors-at-most", REFERENCE[metric], max(abs(a), abs(b))
        checks.append(dict(metric=metric, ours=a, opponent=b, comparison=kind,
                           margin=margin, observed=observed,
                           verdict="PASS" if observed <= margin else "FAIL"))
    if failures or any(c["verdict"] == "FAIL" for c in checks):
        verdict = "FAIL"
    elif not checks or any(c["verdict"] == "INSUFFICIENT" for c in checks):
        verdict = "INSUFFICIENT"
    else:
        verdict = "PASS"
    return dict(policy=POLICY, verdict=verdict, checks=checks, failures=failures,
                interpretation="engineering margin; not a statistical equivalence claim")
