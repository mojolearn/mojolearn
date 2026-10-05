#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Board quality strings: parse them, know each metric's direction, compare two of them.

    from af_quality import parse, direction, compare
    compare("auc=0.632, logloss=0.528", "auc=0.631")   # candidate first, reference second

    python3 tools/af_quality.py "auc=0.93, logloss=0.43" "auc=0.931"   # prints the comparison

FAST passes only when quality does not go down (CLAUDE.md): no material drop against FAST
main, and at least as good as the best opponent. This module is the one place that knows
which way each metric points.

parse() takes a board cell ("auc=0.93, logloss=0.43", "r2=0.31; rmse=0.69", free text with
"name=value" pairs) or a JSON object / dict, and returns {name: value}. A value keeps its
first number ("0.264 (seed 7; 40-seed mean 0.233)" -> 0.264); a value with no number stays a
string (PASS, 100000x259). Free text without any "name=value" parses to {}.

direction(name) is "higher", "lower", "info" (reported, never judged) or "unknown".
Unknown metrics are reported as UNKNOWN by compare(), never silently passed.

compare(cand, ref, rel_tol=1e-3, abs_tol=1e-6) judges every metric both sides share:
SAME when |cand - ref| <= max(abs_tol, rel_tol * max(|cand|, |ref|)), else BETTER or WORSE by
direction. The overall verdict is WORSE if any judged metric is WORSE, else UNKNOWN when no
shared metric could be judged, else BETTER if any is BETTER, else SAME.
"""
import json, math, re, sys

BETTER, SAME, WORSE, UNKNOWN, INFO = "BETTER", "SAME", "WORSE", "UNKNOWN", "INFO"

# Reported, never judged. Checked first. "*_vs_ours" / "*_over_ours" measure agreement with
# our own identical arm, not quality, so they are informational too.
INFO_EXACT = {
    "n_iter", "rows", "n_clusters", "output_shape", "n_features", "pvalue", "statistic",
    "fraction_flagged", "n_selected", "n_components", "n_support", "n_tokens", "n_communities",
    "tokens", "steps", "neighbors_total", "nonzeros_per_row", "norm", "sum", "reference",
    "output_columns", "stationary_fraction", "component_sparsity", "noise_fraction",
    "exact_arrays", "smoothing_oracle", "loss_first_step", "loss_last_step",
    # bootstrap / resampling estimates: the estimate itself, not an error of the fit
    "standard_error", "ci_low", "ci_high",
}
INFO_RE = re.compile(r"(_vs_ours|_over_ours|_to_ours)$|^n_")

HIGHER_EXACT = {
    "auc", "accuracy", "r2", "mean_r2", "map", "silhouette", "f1", "precision", "recall",
    "mean_log_likelihood", "mean_log_predictive_density", "mean_llf", "modularity",
    "mean_canonical_corr", "equal_fraction_vs_sklearn", "finite",
}
HIGHER_RE = re.compile(r"^(recall_at_|trustworthiness|explained_variance|ndcg|jaccard|subspace_cos|f1_|precision_)"
                       r"|_agreement_vs_|_agreement$")

LOWER_EXACT = {
    "rmse", "mae", "mse", "logloss", "mlogloss", "mean_nll", "distortion", "inertia", "residual",
    "perplexity", "bic", "mean_aic", "forecast_rmse", "masked_rmse", "insample_rmse",
    "relative_gram_difference", "max_mean_shift_over_std", "nonfinite_proba_rows",
    "rows_without_density", "rows_with_repeated_ids",
}
LOWER_RE = re.compile(r"^error|_error|error_|^max_abs|rmse|^mae_|_mae$|^mse_|_mse$|distortion|inertia"
                      r"|residual|_rel_diff_vs_|^rel_fro_vs_|_err_vs_|^rel_err")


def direction(name):
    n = name.strip().lower()
    if n in INFO_EXACT or INFO_RE.search(n):
        return "info"
    if n in HIGHER_EXACT or HIGHER_RE.search(n):
        return "higher"
    if n in LOWER_EXACT or LOWER_RE.search(n):
        return "lower"
    return "unknown"


_KEY_RE = re.compile(r"(?:(?<=^)|(?<=[\s,;(]))([A-Za-z_][A-Za-z0-9_@.]*)\s*=")
_NUM_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|[-+]?inf|nan", re.I)


def _value(v):
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().rstrip(",;").strip()
    m = _NUM_RE.match(s)
    if m and not re.match(r"^\d+x\d+", s):
        try:
            return float(m.group(0))
        except ValueError:
            pass
    return s


def parse(q):
    """Board quality cell, JSON text or dict -> {metric: float | str}."""
    if q is None:
        return {}
    if isinstance(q, dict):
        return {str(k): _value(v) for k, v in q.items() if not isinstance(v, (dict, list)) and v is not None}
    s = str(q).strip()
    if s in ("", "-"):
        return {}
    if s.startswith("{"):
        try:
            return parse(json.loads(s))
        except ValueError:
            pass
    keys = list(_KEY_RE.finditer(s))
    out = {}
    for i, m in enumerate(keys):
        end = keys[i + 1].start() if i + 1 < len(keys) else len(s)
        out.setdefault(m.group(1), _value(s[m.end():end]))
    return out


def judge(name, a, b, rel_tol=1e-3, abs_tol=1e-6):
    """One metric: candidate a vs reference b -> BETTER / SAME / WORSE / UNKNOWN / INFO."""
    d = direction(name)
    if d == "info":
        return INFO
    if isinstance(a, str) or isinstance(b, str):
        return SAME if a == b and d != "unknown" else UNKNOWN
    if d == "unknown":
        return UNKNOWN
    if math.isnan(a) and math.isnan(b):
        return SAME
    if math.isnan(a):
        return WORSE
    if math.isnan(b):
        return BETTER
    if a == b or abs(a - b) <= max(abs_tol, rel_tol * max(abs(a), abs(b))):
        return SAME
    return BETTER if (a > b) == (d == "higher") else WORSE


def rel_change(name, a, b):
    """Signed relative quality change of a vs b (positive = better), for ranking; None if not numeric."""
    d = direction(name)
    if d not in ("higher", "lower") or isinstance(a, str) or isinstance(b, str):
        return None
    if math.isnan(a) or math.isnan(b) or math.isinf(a) or math.isinf(b):
        return None
    den = max(abs(a), abs(b), 1e-12)
    return ((a - b) if d == "higher" else (b - a)) / den


def compare(cand, ref, rel_tol=1e-3, abs_tol=1e-6):
    """Candidate vs reference (strings or dicts). Returns a dict:
    verdict: overall BETTER / SAME / WORSE / UNKNOWN
    metrics: {name: (verdict, cand, ref, rel_change)} for every shared metric
    unknown: shared metrics whose direction is unknown (or not comparable)
    worst:   most negative rel_change among WORSE metrics (0.0 if none)
    """
    a, b = parse(cand), parse(ref)
    res, unknown, worst = {}, [], 0.0
    for k in a:
        if k not in b:
            continue
        v = judge(k, a[k], b[k], rel_tol, abs_tol)
        rc = rel_change(k, a[k], b[k])
        res[k] = (v, a[k], b[k], rc)
        if v == UNKNOWN:
            unknown.append(k)
        if v == WORSE:
            worst = min(worst, rc if rc is not None else -float("inf"))
    vs = [r[0] for r in res.values()]
    if WORSE in vs:
        verdict = WORSE
    elif not any(v in (BETTER, SAME) for v in vs):
        verdict = UNKNOWN
    elif BETTER in vs:
        verdict = BETTER
    else:
        verdict = SAME
    return {"verdict": verdict, "metrics": res, "unknown": unknown, "worst": worst}


def describe(c):
    """Short text of a compare() result: 'WORSE (auc 0.62 < 0.63; logloss SAME)'."""
    parts = []
    for k, (v, a, b, _) in c["metrics"].items():
        if v == INFO:
            continue
        if v in (BETTER, WORSE):
            parts.append("%s %s %s %s" % (k, _g(a), ">" if a > b else "<", _g(b)))
        else:
            parts.append("%s %s" % (k, v))
    return "%s (%s)" % (c["verdict"], "; ".join(parts) or "no shared metric")


def _g(x):
    return ("%.6g" % x) if isinstance(x, float) else str(x)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    print(describe(compare(sys.argv[1], sys.argv[2])))
