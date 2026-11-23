# SPDX-License-Identifier: Apache-2.0
"""RUN of the boosting door (bindings/trees_public_boost_door.mojo): AdaBoost,
Bagging and DART checks and choices that read only typed values. The door's
UNPACK reads the estimator's Python fields into plain values (in the order
the checks read them, so the first refusal is unchanged); these functions
decide. No Python import: `tools/python_free_dirs.sh` covers this directory.

Lane doorsplit-trees (2026-11-11): moved from the door. Each function is a
per-call scalar decision; there is no independent work to split across
cores.
"""
from std.math import ceil
from core.entry_table import EntryRef
from core.host_parallel import host_parallelize
from xtrees.glue_calls import xt_apply, xt_leaf_numbering, xt_onehot_leaves, xt_uniform
from xtrees.trees_forest_run import trees_refuse, ForestFit, forest_fit_core
from bindings.forest_config_run import ForestConfig, forest_request, forest_target_need, FOREST_TARGET_ANY

comptime BOOST_DART_VALUES_CAP = 1 << 26


def boost_ada_session_choice(mode: String, switches: Int) -> String:
    """The AdaBoost members' forest data session: 'share' (later members
    reuse the first member's quantile tables) in a FAST + Apple build with
    MOJOLEARN_TE_ADA_SESSION_SHARE (fast-switch bit 4), else 'exact'."""
    if mode == "fast" and (switches & 4) != 0:
        return "share"
    return "exact"


def boost_bagging_validate(oob_score: Bool, bootstrap: Bool, warm_start: Bool, verbose: Bool) raises:
    """`_BaggingBase._validate_params`' flag refusals."""
    if oob_score and not bootstrap:
        raise Error("Out of bag estimation only available if bootstrap=True")
    if warm_start:
        trees_refuse("warm_start=True", "there is no incremental fit here.")
    # TOMBSTONE: the n_jobs refusal ("the sub-fits run one after another on the device") deleted 2026-11-21 by
    # lane ma-callers round 2: published members fit in parallel waves and n_jobs bounds their workers.
    # Restore: git revert of that commit (a refusal, not an experiment arm).
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


def _sub_seed(seed: Int, k: Int) -> Int:
    """`_trees_sub_seed`: the k-th member's random_state."""
    return (seed * 1000003 + 7919 * (k + 1)) % 2147483647


def _dart_thr(v: Float64) -> Int:
    """`_DARTBase._dart_thr`: skip_drop for the binding's 53-bit draw."""
    if not v > 0.0:
        return 0
    if v >= 1.0:
        return 1 << 53
    return Int(ceil(v * 9007199254740992.0))


# ---------------------------------------------------- RandomTreesEmbedding
# RUN of the embedding's operations: typed glue calls over addresses the
# door's UNPACK admitted and its PACK publishes. Each is one synchronous
# launch (or two) of the x_trees image; the parallelism is inside the image.
# Lane bm-rest-rte (2026-11-23): the fit (`rte_fit_run`) is typed too: the
# targets, their float32 words and the forest's XF_FIT, with no Python
# ExtraTreesRegressor request built over Python objects. A Mojo check calls
# `rte_fit_run`, exports the handle through the image (XF_EXPORT_*), then
# `rte_numbering_run` and `rte_transform_run`, as the door does.

comptime _RTE_CAST_CHUNK = 65536


def rte_targets_run(glue: EntryRef, targets: Int, n: Int, seed: Int) raises:
    """The embedding's uniform targets (float64, n words at `targets`):
    stream 0 of the estimator's seed."""
    _ = xt_uniform(glue, targets, n, seed, 0)


def _rte_targets_f32(src: Int, dst: Int, n: Int):
    """The float64 targets at `src` as float32 at `dst` (round to nearest
    even: the cast the door's `admit_f32` of the float64 vector made), split
    across the host pool in chunks of `_RTE_CAST_CHUNK` words (a
    memory-bandwidth grain, not a shape rule; one chunk runs on the caller)."""
    var units = (n + _RTE_CAST_CHUNK - 1) // _RTE_CAST_CHUNK

    def task(unit: Int) {imm src, imm dst, imm n}:
        var s = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=src)
        var t = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=dst)
        var stop = min(n, (unit + 1) * _RTE_CAST_CHUNK)
        for i in range(unit * _RTE_CAST_CHUNK, stop):  # native data loop (one host-pool chunk of targets, not a driver loop)
            t[i] = Float32(s[i])

    host_parallelize(task, units)


def rte_fit_run(glue: EntryRef, forest: EntryRef, config: ForestConfig, x: Int, n: Int, d: Int, row_major: Bool,
                device_scan: Bool, seed: Int) raises -> Int:
    """`RandomTreesEmbedding.fit`'s forest fit on typed values: the uniform
    targets (`rte_targets_run`, stream 0 of `seed`), their float32 words,
    the ExtraTreesRegressor request (`forest_request` over the forest's
    configuration `config`, X at `x`, n x d float32, read row-major when
    `row_major`) and the fit in the image `forest` names (`forest_fit_core`,
    XF_FIT). Returns the export handle; the caller exports and releases it
    (the door's PACK: `forest_export_handle`). The forest is the embedding's
    squared-error regressor, so its targets need no deviance check; a
    configuration with another criterion is refused."""
    if forest_target_need(config) != FOREST_TARGET_ANY:
        raise Error("an embedding's forest fits squared_error")
    var u = List[Float64](length=max(n, 1), fill=0.0)
    rte_targets_run(glue, Int(u.unsafe_ptr()), n, seed)
    var y = List[Float32](length=max(n, 1), fill=Float32(0))
    _rte_targets_f32(Int(u.unsafe_ptr()), Int(y.unsafe_ptr()), n)
    _ = len(u)  # the cast above read these words through their address
    var request = forest_request(config, forest, n, d, 0, x, Int(y.unsafe_ptr()), ForestFit(False, rowmajor_ok=True),
                                 False)
    var handle = forest_fit_core(forest, False, row_major, request, device_scan)
    _ = len(y)  # the fit above read the targets through their address
    _ = len(request)
    return handle


def rte_numbering_run(glue: EntryRef, left_child: Int, node_col: Int, nodes: Int) raises -> Int:
    """The leaf numbering of the fitted forest: the output column of every
    leaf node (int32, `nodes` words at `node_col`) from the forest's
    left-child array. Returns the embedding's width."""
    return xt_leaf_numbering(glue, left_child, node_col, nodes)


def rte_transform_run(glue: EntryRef, model: List[Int], x: Int, leaves: Int, tree_base: Int, node_col: Int,
                      dst: Int, n: Int, d: Int, trees: Int, width: Int) raises:
    """`apply`, then the one-hot: every row's leaf node per tree (int32,
    n x trees at `leaves`) from the forest's offsets, column ids, thresholds
    and left children (`model[0..3]`), then the dense one-hot (float64,
    n x width at `dst`)."""
    _ = xt_apply(glue, model[0], model[1], model[2], model[3], x, leaves, n, d, 0, trees)
    _ = xt_onehot_leaves(glue, leaves, tree_base, node_col, dst, n, trees, width)
