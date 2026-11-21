# SPDX-License-Identifier: Apache-2.0
# The `family_msel_*` operation names of the classical owner's code table
# (bindings/native_operation_codes.mojo reads this file once per process, at
# table fill; lane bw2-core, 2026-11-20). A leaf: no Python, no imports of an
# owner.
#
# NOT YET CODED: the owner still receives the operation NAME (the table's own
# String, never built per call) and matches it. The lane that converts this
# owner gives each name a leaf code here (`comptime <FAMILY>_OPN_<NAME upper> =
# n`, numbered by sub-owner so a group is one range test) and appends it beside
# the name (`leaves.append(...)`, the form of
# bindings/operation_names_gp.mojo); the owner then takes `operation: Int`.
# Add every name a caller can pass: a name that is missing here still runs,
# through the door's String path (one failed lookup, then the old route).


def operation_names_msel(mut names: List[String], mut leaves: List[Int]):
    """Straight-line appends, once per process."""
    names.append("family_msel_check_cv")
    names.append("family_msel_cross_val_predict")
    names.append("family_msel_cross_val_score")
    names.append("family_msel_cross_validate")
    names.append("family_msel_default_folds")
    names.append("family_msel_delegate")
    names.append("family_msel_estimator_call")
    names.append("family_msel_estimator_clone")
    names.append("family_msel_get_scorer")
    names.append("family_msel_grid_init")
    names.append("family_msel_grid_item")
    names.append("family_msel_grid_iter")
    names.append("family_msel_grid_len")
    names.append("family_msel_learning_curve")
    names.append("family_msel_make_scorer")
    names.append("family_msel_n_splits")
    names.append("family_msel_parallel_capture")
    names.append("family_msel_parallel_initialize")
    names.append("family_msel_parallel_inventory")
    names.append("family_msel_parallel_last_run")
    names.append("family_msel_parallel_model_pack")
    names.append("family_msel_parallel_require_batch")
    names.append("family_msel_parallel_require_inventory")
    names.append("family_msel_parallel_score")
    names.append("family_msel_parallel_wire")
    names.append("family_msel_parallel_worker")
    names.append("family_msel_permutation_test_score")
    names.append("family_msel_sampler_init")
    names.append("family_msel_sampler_iter")
    names.append("family_msel_sampler_len")
    names.append("family_msel_scorer_call")
    names.append("family_msel_scorer_names")
    names.append("family_msel_scorer_repr")
    names.append("family_msel_search_classes")
    names.append("family_msel_search_estimator_type")
    names.append("family_msel_search_fit")
    names.append("family_msel_search_get_params")
    names.append("family_msel_search_init")
    names.append("family_msel_search_score")
    names.append("family_msel_search_set_params")
    names.append("family_msel_search_vendor_used")
    names.append("family_msel_split")
    names.append("family_msel_split_descriptor")
    names.append("family_msel_split_logo_at")
    names.append("family_msel_split_loo_at")
    names.append("family_msel_split_lpgo_at")
    names.append("family_msel_split_lpo_at")
    names.append("family_msel_splitter_init")
    names.append("family_msel_splitter_repr")
    names.append("family_msel_train_test_split")
    names.append("family_msel_validation_curve")
