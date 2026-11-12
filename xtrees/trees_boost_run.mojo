# SPDX-License-Identifier: Apache-2.0
"""RUN of the boosting door (bindings/trees_public_boost.mojo): AdaBoost,
Bagging and DART checks and choices that read only typed values. The door's
UNPACK reads the estimator's Python fields into plain values (in the order
the checks read them, so the first refusal is unchanged); these functions
decide. No Python import: `tools/python_free_dirs.sh` covers this directory.

Lane doorsplit-trees (2026-11-11): moved from the door. Each function is a
per-call scalar decision; there is no independent work to split across
cores.
"""
from xtrees.trees_forest_run import trees_refuse

comptime BOOST_DART_VALUES_CAP = 1 << 26


def boost_ada_session_choice(mode: String, switches: Int) -> String:
    """The AdaBoost members' forest data session: 'share' (later members
    reuse the first member's quantile tables) in a FAST + Apple build with
    MOJOLEARN_TE_ADA_SESSION_SHARE (fast-switch bit 4), else 'exact'."""
    if mode == "fast" and (switches & 4) != 0:
        return "share"
    return "exact"


def boost_bagging_validate(oob_score: Bool, bootstrap: Bool, warm_start: Bool, n_jobs_set: Bool,
                           verbose: Bool) raises:
    """`_BaggingBase._validate_params`' flag refusals."""
    if oob_score and not bootstrap:
        raise Error("Out of bag estimation only available if bootstrap=True")
    if warm_start:
        trees_refuse("warm_start=True", "there is no incremental fit here.")
    if n_jobs_set:
        trees_refuse("n_jobs", "the sub-fits run one after another on the device.")
    if verbose:
        trees_refuse("verbose", "nothing logs per estimator.")


def boost_check_n_estimators(n_estimators: Int) raises:
    if n_estimators < 1:
        raise Error("n_estimators must be >= 1")


def boost_check_learning_rate(learning_rate: Float64) raises:
    if not learning_rate > 0:
        raise Error("learning_rate must be > 0")


def boost_count_fraction_ok(f: Float64) -> Bool:
    """`_BaggingBase._count`'s fraction domain (0, 1]."""
    return f > 0.0 and f <= 1.0


def boost_count_check(name: String, k: Int, total: Int) raises -> Int:
    """`_BaggingBase._count`'s resolved-count refusal."""
    if not (1 <= k and k <= total):
        raise Error(name + " resolves to " + String(k) + ", outside [1, " + String(total) + "]")
    return k


def boost_weight_flags_check(bad: Bool, any_nonzero: Bool) raises:
    """`_trees_normalized_weights_native`'s refusals from the exact-sum
    flag words (any of the first four set: a non-finite or negative weight;
    the fifth: some weight is nonzero)."""
    if bad:
        raise Error("sample_weight must be finite and nonnegative")
    if not any_nonzero:
        raise Error("Sample weights must contain at least one non-zero number.")


def boost_dart_cap_check(n_iters: Int, classes: Int, node_cap: Int) raises:
    """The DART round's leaf-value store bound."""
    if not (1 <= node_cap and node_cap <= 65535 and n_iters * classes * node_cap <= BOOST_DART_VALUES_CAP):
        raise Error("NotImplementedError: mojolearn DART: n_estimators * classes * (2 * num_leaves - 1) must be at most "
                    + String(BOOST_DART_VALUES_CAP) + " and num_leaves at most 32768 (the round's leaf-value store); refused by name")
