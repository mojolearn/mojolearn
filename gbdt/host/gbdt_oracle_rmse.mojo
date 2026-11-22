# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The GradientBoosting fit on the host for the gbdt-rmse lane, the RMSE arm
of `gbdt/host/gbdt_oracle.mojo` (workstream E batch 3, 2026-09-14).

SINCE 2026-11-22 (lane pv-gbdt-cpu-cat) THE LOOP DESCRIBED BELOW RUNS IN
`gbdt_oracle.mojo::gbdt_host_boost` under `objective=GBDT_BOOST_RMSE`, the
one Plain SymmetricTree loop, which also carries one-hot and CTR columns,
the permutations and the per-permutation estimation the device runs when
more than one permutation exists (DEVIATION 64 holds at one permutation
only). This module keeps the starting approx, the fit's pair and the text.
What follows is the mirror record of the one-permutation arm.

HOST ONLY. Nothing here imports `max.gpu`, `std.gpu`, a `DeviceContext` or a
module that defines a kernel. The imports are the `checks/numerics` seams,
the GPU-free host modules the symmetric oracle already reuses (the layout and
the policy blocks), and the symmetric oracle itself, whose restatements of
the device kernels this fit reaches unchanged (the grid and the binarize,
both histogram families, the pinned partition stats, the Cosine score, the
lane folds, `choose_scale_kernel`, the model text).

THE CONFIGURATION THIS COVERS, by name (tools/identity_break.py `gbdt-rmse`:
20 trees, depth 6, `loss="RMSE"`, every other option at its default). It is
the gbdt-symmetric configuration with the loss swapped, and the binding
refuses by name what it refuses there, plus `leaf_estimation_iterations`
other than 1 under RMSE (bindings/_mojolearn_gbdt_host.mojo, `_refuse`).
`boost_from_average` unset or True resolves to the seeded cursor and False
to the zero cursor; both are carried.

WHY THE LEAVES COME FROM THE SEARCHER. `fit_with_test` clears
`need_estimation` for RMSE with Newton leaves at one iteration and one
permutation (DEVIATION 64, `gbdt/methods/doc_parallel_boosting.mojo:
1294-1305`), and `train` resolves every lane option into that arm: RMSE's
default method is Newton at one iteration
(`gbdt/options/catboost_options.mojo:1270-1274`), `use_exact_leaves` does
not list RMSE, and a fit with no CTR feature has one permutation
(`gbdt/train.mojo:1126-1136`). So the greedy arm runs
`run_tree_layout_traced` with `export_offsets=False` and
`apply_to_cursor=True` (`doc_parallel_boosting.mojo:1997-2028`), and the
walker never runs.

WHAT IS MIRRORED, IN THE ORDER THE FIT REACHES IT (IDENTICAL build, the
default flags the symmetric oracle names)

  1. `AdjustBoostFromAverageDefaultValue` (`gbdt/train.mojo:1580-1605`):
     unset resolves TRUE for RMSE. `_rmse_starting_approx` below.
  2. The grid, the layout and the binarize (one-border columns through the
     symmetric oracle's `_binary_block` since 2026-09-22): the
     symmetric oracle's `gbdt_host_grid` and `_binarize_columns`, called.
  3. `calc_one_dimensional_optimum_const_approx`'s RMSE arm
     (`gbdt/metrics/optimal_const_for_loss.mojo:52-111`, unweighted branch:
     the Float64 row-order sum over `Float64(n)`, narrowed to Float32 at
     the return, widened back), the bias it sets and the Float32 cursor
     fill (`doc_parallel_boosting.mojo:1131-1164`). Restated, not imported:
     that module imports a kernel module.
  4. Per tree, the search pass `launch_approximate[False]` on RMSE
     (`doc_parallel_boosting.mojo:1544-1552`), which is
     `pointwise_target_kernel[OBJECTIVE_RMSE, False, False]`
     (`gbdt/targets/kernel/pointwise_targets.mojo:560-802`): plane 0 the
     unit weight, plane 1 `ftz(weight * (t - p))`, the score partial
     `-weight * ((t - p) * (t - p))` and the magnitude partials `|plane 0|`,
     `|der|` per 256-thread block through the halving tree
     (`pinned_block_sum`, `:147-179`), folded by
     `deterministic_sum_lanes_kernel` (`:808-863`), then
     `choose_scale_kernel`. `_rmse_search_pass` below.
  5. `run_tree_layout_traced`, every level, the gates and the rollback
     (`greedy_search_helper.mojo:4677-5682`). The level loop below is a
     TWIN of the one in `gbdt_host_fit` (`gbdt_oracle.mojo`); an edit to
     either must visit the other.
  6. The tail under `apply_to_cursor` (`greedy_search_helper.mojo:
     5692-5739`): `compute_partition_stats` over the final (rolled back)
     partitions at the pinned 32 chunks, `compute_leaf_values_kernel`
     (`gbdt/methods/leaves_estimation/leaves_estimation.mojo:75-181`: the
     Hessian `w + l2`, its `<= 0` guard, `g / (hessian + 1e-20)`, and
     `RegularizeImpl`'s `w < 1e-20` zero), and `add_model_value_kernel`
     (`gbdt/methods/kernel_add_model_value.mojo:45-104`, the cursor update
     `cursor + round(leaf * rate)`, gbdt/methods/cursor_rescale.mojo). `_rmse_leaf_value` below.
  7. The rescale on append (`doc_parallel_boosting.mojo:2131-2133`), the
     learn losses (`:2158-2169`, the final pass `:2180-2259`) and the model
     text with its `bias` record, written after `losses` when the bias is
     not zero (`gbdt/models/model_text.mojo:405-410`).

THE NEGATIVE CONTROL. `-D MOJOLEARN_HOST_SABOTAGE=1`, the define the
symmetric oracle reads (`GBDT_ORACLE_HOST_SABOTAGE`), adds 1.0 to the l2
regularizer of `_rmse_leaf_value`, so every leaf with a nonzero gradient sum
of every tree moves, the cursor moves with it, and every fixture's model and
predictions move. The symmetric arm's walker sabotage never runs here.

The restatement is a prediction until measured. The four-column diff of
tools/identity_break.py on the gbdt-rmse lane is the measurement.
"""
from std.memory import bitcast
from gbdt.metrics.sample_quantile import calculate_weighted_target_average
from gbdt.host.gbdt_oracle import (
    GbdtHostModel,
    gbdt_f64_token,
    gbdt_host_model_text,
)


@fieldwise_init
struct GbdtRmseHostFit(Movable):
    """The fitted ensemble and its bias (`TAdditiveModel.bias`, the starting
    approx `boost_from_average` sets; 0.0 without it)."""

    var model: GbdtHostModel
    var bias: Float64


# ===========================================================================
# THE STARTING APPROX
# ===========================================================================


def _rmse_starting_approx(targets: List[Float32], n_rows: Int) raises -> Float64:
    """`calculate_weighted_target_average` on the unweighted branch, through
    the RMSE arm of `calc_one_dimensional_optimum_const_approx`
    (`optimal_const_for_loss.mojo`): `summary_weight` is the exact row
    count, the target sum is `bfa_tree_sum`'s fixed tree in Float64, the
    quotient narrows to Float32 at the return and widens back."""
    if n_rows == 0:
        raise Error("optimal const approx: empty target")
    if len(targets) != n_rows:
        raise Error("optimal const approx: " + String(len(targets)) + " targets for " + String(n_rows) + " rows")
    # the shared unweighted arm: `bfa_tree_sum`'s fold order, the device's
    return Float64(
        calculate_weighted_target_average(targets, List[Float32](), False)
    )


# ===========================================================================
# THE FIT: `gbdt/host/gbdt_oracle.mojo::gbdt_host_fit_eval` with
# `objective=GBDT_BOOST_RMSE` (lane pv-gbdt-cpu-cat, 2026-11-22: the twin
# boosting loop that stood here, `gbdt_rmse_host_fit`, is deleted; the RMSE
# target kernel and the searcher's leaf moved into gbdt_oracle.mojo, whose
# one loop now trains RMSE on numeric, one-hot and CTR columns alike)
# ===========================================================================


def gbdt_text_with_bias(base: String, n_losses: Int, bias: Float64) raises -> String:
    """`model_text`'s `bias` record (`model_text.mojo:405-410`), written
    after the `losses` header record when the bias is not zero BY BITS (a
    -0.0 bias is written). Shared by the symmetric and non-symmetric text."""
    if bitcast[DType.uint64](bias) == UInt64(0):
        return base.copy()
    var header = String("losses ") + String(n_losses)
    var out = String("")
    var inserted = False
    var pieces = base.split("\n")
    for p in range(len(pieces)):
        var line = String(pieces[p])
        if p + 1 == len(pieces):
            # the text ends in a newline, so the last piece is the empty tail
            out += line
            break
        out += line + "\n"
        if not inserted and line == header:
            out += String("bias ") + gbdt_f64_token(bias) + "\n"
            inserted = True
    if not inserted:
        raise Error("gbdt host: the model text has no `losses` header record")
    return out^


def gbdt_rmse_host_model_text(fit: GbdtRmseHostFit) raises -> String:
    """The symmetric oracle's text with the `bias` record a seeded fit adds."""
    return gbdt_text_with_bias(
        gbdt_host_model_text(fit.model), len(fit.model.losses), fit.bias
    )
