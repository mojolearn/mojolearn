# SPDX-License-Identifier: Apache-2.0
"""Every native check, registered: the one list `run_cpu.mojo` and `run_gpu.mojo` run.

A NEW check is a TestSuite file instead (`# native-check: <columns>` on its
second line, `test_*` functions, its own `main()`; docs/plans/TESTSUITE_CONVERSION.md):
it has no row here. This list holds the checks not yet converted; it goes when
the last one is (the recipe's batches). The checks under
`checks/native_tests/` import as `checks.native_tests.<name>`; the older
standalone checks at the top of `checks/` stay where other checks import them
from and are registered as `checks.<name>`. The CI enumerates the
names from this file (`tools/ops/native_checks_report.py`), so a check that is
not appended here never runs. No `main()`: imported by the two mains.

Not here: `et_device_loop_early_exit` and `rf_device_loop_early_exit` build
one binary per compile-time route define (their own pixi tasks), and
`entry_table_image_provider` / `entry_table_image_consumer`, two shared
images a controller loads separately (they cannot share one binary).
"""
from checks.native_tests.ann_cagra_filtered import register as register_ann_cagra_filtered
from checks.native_tests.ann_ivf_surface import register as register_ann_ivf_surface
from checks.native_tests.ann_ivfpq_host_filter import register as register_ann_ivfpq_host_filter
from checks.native_tests.ann_ops import register as register_ann_ops
from checks.native_tests.byte_lm_host_column import register as register_byte_lm_host_column
from checks.native_tests.byte_lm_models_loader_f16 import register as register_byte_lm_models_loader_f16
from checks.native_tests.byte_lm_parallel_byte_native import register as register_byte_lm_parallel_byte_native
from checks.native_tests.byte_lm_trainer_session import register as register_byte_lm_trainer_session
from checks.native_tests.core_all_finite import register as register_core_all_finite
from checks.native_tests.core_chunked_lm_head_denormals import register as register_core_chunked_lm_head_denormals
from checks.native_tests.core_convert import register as register_core_convert
from checks.native_tests.core_device_units import register as register_core_device_units
from checks.native_tests.core_helpers import register as register_core_helpers
from checks.native_tests.core_import_lower import register as register_core_import_lower
from checks.native_tests.core_ne2_rounds import register as register_core_ne2_rounds
from checks.native_tests.core_ne3_fused_entries import register as register_core_ne3_fused_entries
from checks.native_tests.core_ops_import_kinds import register as register_core_ops_import_kinds
from checks.native_tests.core_ops_layernorm_view import register as register_core_ops_layernorm_view
from checks.native_tests.core_ops_program import register as register_core_ops_program
from checks.native_tests.core_ops_session import register as register_core_ops_session
from checks.native_tests.core_ops_session_blocks import register as register_core_ops_session_blocks
from checks.native_tests.core_partitioned_program import register as register_core_partitioned_program
from checks.native_tests.core_program_fit_stop import register as register_core_program_fit_stop
from checks.native_tests.core_program_loop import register as register_core_program_loop
from checks.native_tests.core_program_loop_append import register as register_core_program_loop_append
from checks.native_tests.core_ragged_lengths import register as register_core_ragged_lengths
from checks.native_tests.core_safetensors_f16 import register as register_core_safetensors_f16
from checks.native_tests.core_stratified_group_fp64 import register as register_core_stratified_group_fp64
from checks.native_tests.core_verifier_gap_methods import register as register_core_verifier_gap_methods
from checks.native_tests.core_vjp_rank import register as register_core_vjp_rank
from checks.native_tests.decomp_holtwinters_inference import register as register_decomp_holtwinters_inference
from checks.native_tests.decomp_ne2_g_decomp_entries import register as register_decomp_ne2_g_decomp_entries
from checks.native_tests.gbdt_ova_link import register as register_gbdt_ova_link
from checks.native_tests.gbdt_public_fits import register as register_gbdt_public_fits
from checks.native_tests.gbdt_resident_ns_predict import register as register_gbdt_resident_ns_predict
from checks.native_tests.gbdt_shape_sweep import register as register_gbdt_shape_sweep
from checks.native_tests.gbdt_tree_metadata import register as register_gbdt_tree_metadata
# TOMBSTONE: MOJOLEARN_DECOMP_FAST_MATRIX_UNITS, MOJOLEARN_NEIGHBORS_FAST_MATRIX_UNITS (NOISE, A/B abf2) deleted 2026-11-20 by
# lane/fx-verdicts6 with their check family_matrix_units; code recoverable at 06d1e8ff34.
# Restore: git apply experiments/removed/MOJOLEARN_DECOMP_FAST_MATRIX_UNITS.patch (the check is in that patch)
from checks.native_tests.gp_entry_table import register as register_gp_entry_table
from checks.native_tests.gp_native import register as register_gp_native
from checks.native_tests.gp_normalization import register as register_gp_normalization
from checks.native_tests.gp_theta import register as register_gp_theta
from checks.native_tests.gpc_score import register as register_gpc_score
from checks.native_tests.gpr_score import register as register_gpr_score
from checks.native_tests.hdbscan_native import register as register_hdbscan_native
from checks.native_tests.hdbscan_surface import register as register_hdbscan_surface
from checks.native_tests.iforest_forest_export import register as register_iforest_forest_export
from checks.native_tests.kit_roundtrip import register as register_kit_roundtrip
from checks.native_tests.logistic_label_weights import register as register_logistic_label_weights
from checks.native_tests.logistic_numeric import register as register_logistic_numeric
from checks.native_tests.mamba3_output_coverage import register as register_mamba3_output_coverage
from checks.native_tests.mamba_decode_n_tokens import register as register_mamba_decode_n_tokens
from checks.native_tests.mamba_decode_sessions import register as register_mamba_decode_sessions
from checks.native_tests.mamba3_prefill_session import register as register_mamba3_prefill_session
from checks.native_tests.mamba3_prefix_step import register as register_mamba3_prefix_step
from checks.native_tests.mani_decomp_repeat import register as register_mani_decomp_repeat
from checks.native_tests.metrics_selection_host_p2 import register as register_metrics_selection_host_p2
from checks.native_tests.metrics_smallfam_public import register as register_metrics_smallfam_public
from checks.native_tests.metrics_weighted_score import register as register_metrics_weighted_score
from checks.native_tests.mixture_gmm_sample import register as register_mixture_gmm_sample
from checks.native_tests.mixture_native import register as register_mixture_native
from checks.native_tests.ocsvm_native import register as register_ocsvm_native
from checks.native_tests.oob_exact_sum import register as register_oob_exact_sum
from checks.native_tests.ops_grad_identity import register as register_ops_grad_identity
from checks.native_tests.ops_higher import register as register_ops_higher
from checks.native_tests.ops_lm_head_v2 import register as register_ops_lm_head_v2
from checks.native_tests.ops_optimizer_requests import register as register_ops_optimizer_requests
from checks.native_tests.optics_native import register as register_optics_native
from checks.native_tests.optimizer_descriptor import register as register_optimizer_descriptor
from checks.native_tests.optimizer_device import register as register_optimizer_device
from checks.native_tests.optimizer_maximize import register as register_optimizer_maximize
from checks.native_tests.optimizer_owner import register as register_optimizer_owner
from checks.native_tests.prophet_state import register as register_prophet_state
from checks.native_tests.query_x_held import register as register_query_x_held
from checks.native_tests.resample_cross_vendor_fold import register as register_resample_cross_vendor_fold
from checks.native_tests.resample_cv_row_order import register as register_resample_cv_row_order
from checks.native_tests.resident_second_call import register as register_resident_second_call
from checks.native_tests.runner import Check
from checks.native_tests.sequence_croston import register as register_sequence_croston
from checks.native_tests.sequence_moe import register as register_sequence_moe
from checks.native_tests.sequence_norm import register as register_sequence_norm
from checks.native_tests.sequence_optimizer import register as register_sequence_optimizer
from checks.native_tests.sequence_rnn import register as register_sequence_rnn
from checks.native_tests.sequence_samba_surface import register as register_sequence_samba_surface
from checks.native_tests.sequence_stl import register as register_sequence_stl
from checks.native_tests.sequence_theta import register as register_sequence_theta
from checks.native_tests.sequence_theta_variants import register as register_sequence_theta_variants
from checks.native_tests.sequence_var import register as register_sequence_var
from checks.native_tests.solver_estimators_ops import register as register_solver_estimators_ops
from checks.native_tests.solver_qn_objectives import register as register_solver_qn_objectives
from checks.native_tests.spectral_assignment import register as register_spectral_assignment
from checks.native_tests.spectral_dense import register as register_spectral_dense
from checks.native_tests.spectral_native import register as register_spectral_native
from checks.native_tests.spectral_prediction import register as register_spectral_prediction
from checks.native_tests.spectral_rbf import register as register_spectral_rbf
from checks.native_tests.spectral_resident import register as register_spectral_resident
from checks.native_tests.svr_native import register as register_svr_native
from checks.native_tests.theta_state import register as register_theta_state
from checks.native_tests.training_autograd_model import register as register_training_autograd_model
from checks.native_tests.training_autograd_save_resume import register as register_training_autograd_save_resume
from checks.native_tests.training_autograd_usability import register as register_training_autograd_usability
from checks.native_tests.training_chunked_lm_head_door import register as register_training_chunked_lm_head_door
from checks.native_tests.training_distributed_program import register as register_training_distributed_program
from checks.native_tests.training_moe_backward_ownership import register as register_training_moe_backward_ownership
from checks.native_tests.training_neural_inference import register as register_training_neural_inference
from checks.native_tests.training_neural_rng_slices import register as register_training_neural_rng_slices
from checks.native_tests.training_primitives import register as register_training_primitives
from checks.native_tests.training_residual_dropout_mask import register as register_training_residual_dropout_mask
from checks.native_tests.training_small_mlp import register as register_training_small_mlp
from checks.native_tests.transformer_block_session import register as register_transformer_block_session
from checks.native_tests.transformer_host_block import register as register_transformer_host_block
from checks.native_tests.transformer_ops_program import register as register_transformer_ops_program
from checks.native_tests.tsa_holtwinters_initialization import register as register_tsa_holtwinters_initialization
from checks.native_tests.tsa_sequence_ets import register as register_tsa_sequence_ets
from checks.native_tests.tsa_training_arima import register as register_tsa_training_arima
from checks.native_tests.tsvd_native import register as register_tsvd_native
from checks.native_tests.umap_ops import register as register_umap_ops
# lane tests-to-mojo-models (2026-11-16)
from checks.native_tests.models_checkpoint_widen import register as register_models_checkpoint_widen
# lane tests-to-mojo-spatial (2026-11-16)
# lane tests-to-mojo-trees (2026-11-16)
from checks.native_tests.gbdt_feature_freq_experimental import register as register_gbdt_feature_freq_experimental
from checks.native_tests.gbdt_host_archive_text import register as register_gbdt_host_archive_text
# lane tests-to-mojo-neural (2026-11-15)
from checks.native_tests.lm_corpus_prepare import register as register_lm_corpus_prepare
from checks.native_tests.training_small_mlp_relations import register as register_training_small_mlp_relations
# lane checks-register (2026-11-16): the standalone main() checks no CI job ran, now registered
from checks.native_tests.boundary_layout import register as register_boundary_layout
from checks.native_tests.byte_lm_entry import register as register_byte_lm_entry
from checks.native_tests.class_weight_plan import register as register_class_weight_plan
from checks.native_tests.decomp_entry_table import register as register_decomp_entry_table
from checks.native_tests.gp_contract import register as register_gp_contract
from checks.native_tests.gp_label_owner import register as register_gp_label_owner
from checks.native_tests.gp_owner import register as register_gp_owner
from checks.native_tests.gp_state import register as register_gp_state
from checks.native_tests.hdbscan_plan import register as register_hdbscan_plan
from checks.native_tests.kde_contract import register as register_kde_contract
from checks.native_tests.kde_state import register as register_kde_state
from checks.native_tests.label_score_keys_stress import register as register_label_score_keys_stress
from checks.native_tests.lof_plan_contract import register as register_lof_plan_contract
from checks.native_tests.logistic_config import register as register_logistic_config
from checks.native_tests.logistic_contracts import register as register_logistic_contracts
from checks.native_tests.logistic_model_state import register as register_logistic_model_state
from checks.native_tests.logistic_query_width import register as register_logistic_query_width
from checks.native_tests.mani_tsne_run import register as register_mani_tsne_run
from checks.native_tests.mani_umap_run import register as register_mani_umap_run
from checks.native_tests.nmf_contract import register as register_nmf_contract
from checks.native_tests.nmf_state import register as register_nmf_state
from checks.native_tests.nonzero_scan import register as register_nonzero_scan
from checks.native_tests.portable_math import register as register_portable_math
from checks.native_tests.samba_entry_table import register as register_samba_entry_table
from checks.native_tests.seq_opt_entry import register as register_seq_opt_entry
from checks.native_tests.sequence_input_plan_check import register as register_sequence_input_plan_check
from checks.native_tests.spectral_contract import register as register_spectral_contract
from checks.native_tests.spectral_state import register as register_spectral_state
from checks.native_tests.tsqr_host_ownership import register as register_tsqr_host_ownership
from checks.native_tests.tsvd_contract import register as register_tsvd_contract
from checks.native_tests.tsvd_lease_state import register as register_tsvd_lease_state

# lane checks-toplevel (2026-11-16): the standalone main() checks at the top of checks/ no CI job ran, now registered
from checks.batch_invariance_main import register as register_batch_invariance_main
from checks.bin_order_check import register as register_bin_order_check
from checks.binarization_check import register as register_binarization_check
from checks.boosting_hist_check import register as register_boosting_hist_check
from checks.bootstrap_check import register as register_bootstrap_check
from checks.border_types_check import register as register_border_types_check
from checks.catboost_apply_check import register as register_catboost_apply_check
from checks.cindex_packing_check import register as register_cindex_packing_check
from checks.cityhash_check import register as register_cityhash_check
from checks.copy_histograms_check import register as register_copy_histograms_check
from checks.ctr_apply_check import register as register_ctr_apply_check
from checks.ctr_check import register as register_ctr_check
from checks.ctr_device_check import register as register_ctr_device_check
from checks.ctr_kernels_check import register as register_ctr_kernels_check
from checks.ctr_permutation_check import register as register_ctr_permutation_check
from checks.ctr_train_check import register as register_ctr_train_check
from checks.depthwise_check import register as register_depthwise_check
from checks.division_check import register as register_division_check
from checks.dynamic_boosting_folds_check import register as register_dynamic_boosting_folds_check
from checks.early_stop_check import register as register_early_stop_check
from checks.exact_estimation_check import register as register_exact_estimation_check
from checks.feature_parallel_identity_check import register as register_feature_parallel_identity_check
from checks.feature_tensor_check import register as register_feature_tensor_check
from checks.fit_pointwise_check import register as register_fit_pointwise_check
from checks.fixture_rng_gate import register as register_fixture_rng_gate
from checks.fold_tasks_check import register as register_fold_tasks_check
from checks.forest_inference_gpu import register as register_forest_inference_gpu
from checks.forest_inference_model import register as register_forest_inference_model
from checks.forest_ordered_resident_policy import register as register_forest_ordered_resident_policy
from checks.freq_ctr_device_check import register as register_freq_ctr_device_check
from checks.gbdt_cindex_staging_check import register as register_gbdt_cindex_staging_check
from checks.gbdt_fused_move_check import register as register_gbdt_fused_move_check
from checks.gbdt_large_partition_check import register as register_gbdt_large_partition_check
from checks.gbdt_partition_cache_check import register as register_gbdt_partition_cache_check
from checks.gbdt_partition_route_check import register as register_gbdt_partition_route_check
from checks.gbdt_per_round_check import register as register_gbdt_per_round_check
from checks.gbdt_prepared_check import register as register_gbdt_prepared_check
from checks.gbdt_step_cost_check import register as register_gbdt_step_cost_check
from checks.gpu_lib_check import register as register_gpu_lib_check
from checks.gpu_lib_worker_check import register as register_gpu_lib_worker_check
from checks.gram_splitk_check import register as register_gram_splitk_check
from checks.greedy_log_sum_check import register as register_greedy_log_sum_check
from checks.grow_policy_check import register as register_grow_policy_check
from checks.hardware_matrix_check import register as register_hardware_matrix_check
from checks.hist2_check import register as register_hist2_check
from checks.hist_check import register as register_hist_check
from checks.hist_dump_check import register as register_hist_dump_check
from checks.histograms_helper_check import register as register_histograms_helper_check
from checks.hostptr_check import register as register_hostptr_check
from checks.identity_trace_check import register as register_identity_trace_check
from checks.ieee_arith_check import register as register_ieee_arith_check
from checks.kpca_dense_centering_regression import register as register_kpca_dense_centering_regression
from checks.layout_check import register as register_layout_check
from checks.leaf_partition_check import register as register_leaf_partition_check
from checks.leafwise_scores_check import register as register_leafwise_scores_check
from checks.level_check import register as register_level_check
from checks.logloss_estimator_check import register as register_logloss_estimator_check
from checks.logloss_leaf_oracle_check import register as register_logloss_leaf_oracle_check
from checks.logloss_target_check import register as register_logloss_target_check
from checks.logloss_train_check import register as register_logloss_train_check
from checks.loss_oracle_check import register as register_loss_oracle_check
from checks.lossguide_check import register as register_lossguide_check
from checks.lossguide_policy_check import register as register_lossguide_policy_check
from checks.mcd_step_cost_check import register as register_mcd_step_cost_check
from checks.min_child_hessian_check import register as register_min_child_hessian_check
from checks.min_entropy_check import register as register_min_entropy_check
from checks.min_split_gain_check import register as register_min_split_gain_check
from checks.model_io_check import register as register_model_io_check
from checks.multiclass_oracle_check import register as register_multiclass_oracle_check
from checks.multiclass_score_check import register as register_multiclass_score_check
from checks.multiclass_train_check import register as register_multiclass_train_check
from checks.multilogit_check import register as register_multilogit_check
from checks.nan_forbidden_host_check import register as register_nan_forbidden_host_check
from checks.nan_mode_check import register as register_nan_mode_check
from checks.ndtri_check import register as register_ndtri_check
from checks.newton_walker_check import register as register_newton_walker_check
from checks.one_hot_cardinality_check import register as register_one_hot_cardinality_check
from checks.one_hot_check import register as register_one_hot_check
from checks.one_hot_flags_check import register as register_one_hot_flags_check
from checks.onebyte_reach_check import register as register_onebyte_reach_check
from checks.options_check import register as register_options_check
from checks.oracle_main import register as register_oracle_main
from checks.ordered_boosting_check import register as register_ordered_boosting_check
from checks.ordered_rmse_check import register as register_ordered_rmse_check
from checks.overfitting_detector_check import register as register_overfitting_detector_check
from checks.pack_check import register as register_pack_check
from checks.partitions_reduce_check import register as register_partitions_reduce_check
from checks.permutation_count_check import register as register_permutation_count_check
from checks.permuted_ids_check import register as register_permuted_ids_check
from checks.plan_fusion_check import register as register_plan_fusion_check
from checks.pointwise_dispatch_check import register as register_pointwise_dispatch_check
from checks.pointwise_driver_check import register as register_pointwise_driver_check
from checks.pointwise_hist2_8bit_check import register as register_pointwise_hist2_8bit_check
from checks.pointwise_hist2_check import register as register_pointwise_hist2_check
from checks.pointwise_hist2_half_byte_check import register as register_pointwise_hist2_half_byte_check
from checks.pointwise_identical_multiplier_check import register as register_pointwise_identical_multiplier_check
from checks.pointwise_loop_check import register as register_pointwise_loop_check
from checks.pointwise_offsets_check import register as register_pointwise_offsets_check
from checks.pointwise_pool_check import register as register_pointwise_pool_check
from checks.pointwise_resolve_check import register as register_pointwise_resolve_check
from checks.pointwise_scores_check import register as register_pointwise_scores_check
from checks.pointwise_small_bin_driver_check import register as register_pointwise_small_bin_driver_check
from checks.pointwise_subsets_check import register as register_pointwise_subsets_check
from checks.pointwise_target_check import register as register_pointwise_target_check
from checks.pointwise_vs_greedy_check import register as register_pointwise_vs_greedy_check
from checks.portable_fmax_check import register as register_portable_fmax_check
from checks.portable_gelu_check import register as register_portable_gelu_check
from checks.portable_log2_64_check import register as register_portable_log2_64_check
from checks.portable_log64_check import register as register_portable_log64_check
from checks.portable_nn_check import register as register_portable_nn_check
from checks.portable_pow64_check import register as register_portable_pow64_check
from checks.portable_sqrtcos_check import register as register_portable_sqrtcos_check
from checks.portable_translog_check import register as register_portable_translog_check
from checks.portable_trig_check import register as register_portable_trig_check
from checks.radix_sort_check import register as register_radix_sort_check
from checks.random_strength_check import register as register_random_strength_check
from checks.reorder_check import register as register_reorder_check
from checks.replicated_half_byte_check import register as register_replicated_half_byte_check
from checks.samba_target_count_check import register as register_samba_target_count_check
from checks.sample_indices_check import register as register_sample_indices_check
from checks.sample_quantile_unweighted_check import register as register_sample_quantile_unweighted_check
from checks.scaffold_check import register as register_scaffold_check
from checks.second_der_weights_check import register as register_second_der_weights_check
from checks.segmented_scan_check import register as register_segmented_scan_check
from checks.sibling_tiebreak_check import register as register_sibling_tiebreak_check
from checks.soft_f64_check import register as register_soft_f64_check
from checks.sym_arms_check import register as register_sym_arms_check
from checks.tensor_sync_state_check import register as register_tensor_sync_state_check
from checks.train_api_check import register as register_train_api_check
from checks.combination_code_device_check import register as register_combination_code_device_check
from checks.tree_ctr_device_check import register as register_tree_ctr_device_check
from checks.tree_ctr_median_device_check import register as register_tree_ctr_median_device_check
from checks.tree_ctr_fit_device_check import register as register_tree_ctr_fit_device_check
from checks.tree_ctr_slice_check import register as register_tree_ctr_slice_check
from checks.ordered_ctr_device_check import register as register_ordered_ctr_device_check
# lane fx-checks-orphans (2026-11-18): the family standalone main() checks nothing ran; each suite stays in
# <family>/checks/ (its main() is now run_checks()) and its registered wrapper is checks/native_tests/<family>_<stem>.mojo
from checks.native_tests.autograd_data_cpu import register as register_autograd_data_cpu
from checks.native_tests.autograd_data_gpu import register as register_autograd_data_gpu
from checks.native_tests.autograd_distributed_setup_failure import register as register_autograd_distributed_setup_failure
from checks.native_tests.autograd_embedding_cpu import register as register_autograd_embedding_cpu
from checks.native_tests.autograd_embedding_gpu import register as register_autograd_embedding_gpu
from checks.native_tests.autograd_fold_cpu import register as register_autograd_fold_cpu
from checks.native_tests.autograd_fold_gpu import register as register_autograd_fold_gpu
from checks.native_tests.autograd_fused_cpu import register as register_autograd_fused_cpu
from checks.native_tests.autograd_fused_gpu import register as register_autograd_fused_gpu
from checks.native_tests.autograd_gemm_gpu import register as register_autograd_gemm_gpu
from checks.native_tests.autograd_gpu_arguments import register as register_autograd_gpu_arguments
from checks.native_tests.autograd_loss_cpu import register as register_autograd_loss_cpu
from checks.native_tests.autograd_loss_gpu import register as register_autograd_loss_gpu
from checks.native_tests.autograd_lossblock_cpu import register as register_autograd_lossblock_cpu
from checks.native_tests.autograd_lossblock_gpu import register as register_autograd_lossblock_gpu
from checks.native_tests.autograd_optim_cpu import register as register_autograd_optim_cpu
from checks.native_tests.autograd_order_cpu import register as register_autograd_order_cpu
from checks.native_tests.autograd_order_gpu import register as register_autograd_order_gpu
from checks.native_tests.autograd_pass1_cpu import register as register_autograd_pass1_cpu
from checks.native_tests.autograd_pass1_gpu import register as register_autograd_pass1_gpu
from checks.native_tests.autograd_program_setup_failure import register as register_autograd_program_setup_failure
from checks.native_tests.autograd_view_cpu import register as register_autograd_view_cpu
from checks.native_tests.autograd_view_gpu import register as register_autograd_view_gpu
from checks.native_tests.ensemble_builder_check import register as register_ensemble_builder_check
from checks.native_tests.ensemble_builder_kernels_check import register as register_ensemble_builder_kernels_check
from checks.native_tests.ensemble_criteria_check import register as register_ensemble_criteria_check
from checks.native_tests.ensemble_cuml_oracle_check import register as register_ensemble_cuml_oracle_check
from checks.native_tests.ensemble_forest_check import register as register_ensemble_forest_check
from checks.native_tests.ensemble_objectives_check import register as register_ensemble_objectives_check
from checks.native_tests.ensemble_oob_check import register as register_ensemble_oob_check
from checks.native_tests.ensemble_predict_check import register as register_ensemble_predict_check
from checks.native_tests.ensemble_quantiles_check import register as register_ensemble_quantiles_check
from checks.native_tests.ensemble_regression_check import register as register_ensemble_regression_check
from checks.native_tests.ensemble_sample_weight_check import register as register_ensemble_sample_weight_check
from checks.native_tests.ensemble_sampled_cols_check import register as register_ensemble_sampled_cols_check
from checks.native_tests.ensemble_shuffle_check import register as register_ensemble_shuffle_check
from checks.native_tests.ensemble_sibling_column_overlap_check import register as register_ensemble_sibling_column_overlap_check
from checks.native_tests.ensemble_split_check import register as register_ensemble_split_check
from checks.native_tests.ensemble_train_check import register as register_ensemble_train_check
from checks.native_tests.gbdt_host_binarize_identical import register as register_gbdt_host_binarize_identical
from checks.native_tests.gbdt_tree_ctr_host_check import register as register_gbdt_tree_ctr_host_check
from checks.native_tests.gbdt_ordered_ctr_host_check import register as register_gbdt_ordered_ctr_host_check
from checks.native_tests.gbdt_host_device_cat import register as register_gbdt_host_device_cat
from checks.native_tests.gbdt_ordered_doc_id_check import register as register_gbdt_ordered_doc_id_check
from checks.native_tests.hdbscan_sparse_mr_check import register as register_hdbscan_sparse_mr_check
from checks.native_tests.ivf_ivf_large_k_check import register as register_ivf_ivf_large_k_check
from checks.native_tests.ivf_ops_table import register as register_ivf_ops_table
from checks.native_tests.kde_reused_workspace_check import register as register_kde_reused_workspace_check
from checks.native_tests.mamba_mamba3_increment_tile_check import register as register_mamba_mamba3_increment_tile_check
from checks.native_tests.mamba_mamba3_refusal_check import register as register_mamba_mamba3_refusal_check
from checks.native_tests.mamba_mamba3_s17_tail_tree_check import register as register_mamba_mamba3_s17_tail_tree_check
from checks.native_tests.mamba_mamba3_yintra_tile_check import register as register_mamba_mamba3_yintra_tile_check
from checks.native_tests.metrics_host_binary_ranking_check import register as register_metrics_host_binary_ranking_check
from checks.native_tests.metrics_host_rand_score_check import register as register_metrics_host_rand_score_check
from checks.native_tests.training_byte_lm_block_copy_check import register as register_training_byte_lm_block_copy_check
from checks.native_tests.training_chunked_lm_head_check import register as register_training_chunked_lm_head_check
from checks.native_tests.training_chunked_lm_head_device_check import register as register_training_chunked_lm_head_device_check
from checks.native_tests.training_dbscan_parallel_check import register as register_training_dbscan_parallel_check
from checks.native_tests.training_maximize_check import register as register_training_maximize_check
from checks.native_tests.transformer_attention_zdot_routing_check import register as register_transformer_attention_zdot_routing_check
from checks.native_tests.xtrees_glue_check import register as register_xtrees_glue_check
from checks.native_tests.xtrees_zero_cover_check import register as register_xtrees_zero_cover_check
from checks.native_tests.tsa_hw_route import register as register_tsa_hw_route
from checks.native_tests.tsa_arima_route import register as register_tsa_arima_route

def all_checks() -> List[Check]:
    var checks = List[Check]()
    checks.append(register_ann_cagra_filtered())
    checks.append(register_ann_ivf_surface())
    checks.append(register_ann_ivfpq_host_filter())
    checks.append(register_ann_ops())
    checks.append(register_byte_lm_host_column())
    checks.append(register_byte_lm_models_loader_f16())
    checks.append(register_byte_lm_parallel_byte_native())
    checks.append(register_byte_lm_trainer_session())
    checks.append(register_core_all_finite())
    checks.append(register_core_chunked_lm_head_denormals())
    checks.append(register_core_convert())
    checks.append(register_core_device_units())
    checks.append(register_core_helpers())
    checks.append(register_core_import_lower())
    checks.append(register_core_ne2_rounds())
    checks.append(register_core_ne3_fused_entries())
    checks.append(register_core_ops_import_kinds())
    checks.append(register_core_ops_layernorm_view())
    checks.append(register_core_ops_program())
    checks.append(register_core_ops_session())
    checks.append(register_core_ops_session_blocks())
    checks.append(register_core_partitioned_program())
    checks.append(register_core_program_fit_stop())
    checks.append(register_core_program_loop())
    checks.append(register_core_program_loop_append())
    checks.append(register_core_ragged_lengths())
    checks.append(register_core_safetensors_f16())
    checks.append(register_core_stratified_group_fp64())
    checks.append(register_core_verifier_gap_methods())
    checks.append(register_core_vjp_rank())
    checks.append(register_decomp_holtwinters_inference())
    checks.append(register_decomp_ne2_g_decomp_entries())
    checks.append(register_gbdt_ova_link())
    checks.append(register_gbdt_public_fits())
    checks.append(register_gbdt_resident_ns_predict())
    checks.append(register_gbdt_shape_sweep())
    checks.append(register_gbdt_tree_metadata())
    checks.append(register_gp_entry_table())
    checks.append(register_gp_native())
    checks.append(register_gp_normalization())
    checks.append(register_gp_theta())
    checks.append(register_gpc_score())
    checks.append(register_gpr_score())
    checks.append(register_hdbscan_native())
    checks.append(register_hdbscan_surface())
    checks.append(register_iforest_forest_export())
    checks.append(register_kit_roundtrip())
    checks.append(register_logistic_label_weights())
    checks.append(register_logistic_numeric())
    checks.append(register_mamba3_output_coverage())
    checks.append(register_mamba_decode_n_tokens())
    checks.append(register_mamba_decode_sessions())
    checks.append(register_mamba3_prefill_session())
    checks.append(register_mamba3_prefix_step())
    checks.append(register_mani_decomp_repeat())
    checks.append(register_metrics_selection_host_p2())
    checks.append(register_metrics_smallfam_public())
    checks.append(register_metrics_weighted_score())
    checks.append(register_mixture_gmm_sample())
    checks.append(register_mixture_native())
    checks.append(register_ocsvm_native())
    checks.append(register_oob_exact_sum())
    checks.append(register_ops_grad_identity())
    checks.append(register_ops_higher())
    checks.append(register_ops_lm_head_v2())
    checks.append(register_ops_optimizer_requests())
    checks.append(register_optics_native())
    checks.append(register_optimizer_descriptor())
    checks.append(register_optimizer_device())
    checks.append(register_optimizer_maximize())
    checks.append(register_optimizer_owner())
    checks.append(register_prophet_state())
    checks.append(register_query_x_held())
    checks.append(register_resample_cross_vendor_fold())
    checks.append(register_resample_cv_row_order())
    checks.append(register_resident_second_call())
    checks.append(register_sequence_croston())
    checks.append(register_sequence_moe())
    checks.append(register_sequence_norm())
    checks.append(register_sequence_optimizer())
    checks.append(register_sequence_rnn())
    checks.append(register_sequence_samba_surface())
    checks.append(register_sequence_stl())
    checks.append(register_sequence_theta())
    checks.append(register_sequence_theta_variants())
    checks.append(register_sequence_var())
    checks.append(register_solver_estimators_ops())
    checks.append(register_solver_qn_objectives())
    checks.append(register_spectral_assignment())
    checks.append(register_spectral_dense())
    checks.append(register_spectral_native())
    checks.append(register_spectral_prediction())
    checks.append(register_spectral_rbf())
    checks.append(register_spectral_resident())
    checks.append(register_svr_native())
    checks.append(register_theta_state())
    checks.append(register_training_autograd_model())
    checks.append(register_training_autograd_save_resume())
    checks.append(register_training_autograd_usability())
    checks.append(register_training_chunked_lm_head_door())
    checks.append(register_training_distributed_program())
    checks.append(register_training_moe_backward_ownership())
    checks.append(register_training_neural_inference())
    checks.append(register_training_neural_rng_slices())
    checks.append(register_training_primitives())
    checks.append(register_training_residual_dropout_mask())
    checks.append(register_training_small_mlp())
    checks.append(register_transformer_block_session())
    checks.append(register_transformer_host_block())
    checks.append(register_transformer_ops_program())
    checks.append(register_tsa_holtwinters_initialization())
    checks.append(register_tsa_sequence_ets())
    checks.append(register_tsa_training_arima())
    checks.append(register_tsvd_native())
    checks.append(register_umap_ops())
    # lane tests-to-mojo-models (2026-11-16)
    checks.append(register_models_checkpoint_widen())
    # lane tests-to-mojo-spatial (2026-11-16)
    # lane tests-to-mojo-trees (2026-11-16)
    checks.append(register_gbdt_feature_freq_experimental())
    checks.append(register_gbdt_host_archive_text())
    # lane tests-to-mojo-neural (2026-11-15)
    checks.append(register_lm_corpus_prepare())
    checks.append(register_training_small_mlp_relations())
    # lane checks-register (2026-11-16)
    checks.append(register_boundary_layout())
    checks.append(register_byte_lm_entry())
    checks.append(register_class_weight_plan())
    checks.append(register_decomp_entry_table())
    checks.append(register_gp_contract())
    checks.append(register_gp_label_owner())
    checks.append(register_gp_owner())
    checks.append(register_gp_state())
    checks.append(register_hdbscan_plan())
    checks.append(register_kde_contract())
    checks.append(register_kde_state())
    checks.append(register_label_score_keys_stress())
    checks.append(register_lof_plan_contract())
    checks.append(register_logistic_config())
    checks.append(register_logistic_contracts())
    checks.append(register_logistic_model_state())
    checks.append(register_logistic_query_width())
    checks.append(register_mani_tsne_run())
    checks.append(register_mani_umap_run())
    checks.append(register_ivf_ops_table())
    checks.append(register_nmf_contract())
    checks.append(register_nmf_state())
    checks.append(register_nonzero_scan())
    checks.append(register_portable_math())
    checks.append(register_samba_entry_table())
    checks.append(register_seq_opt_entry())
    checks.append(register_sequence_input_plan_check())
    checks.append(register_spectral_contract())
    checks.append(register_spectral_state())
    checks.append(register_tsqr_host_ownership())
    checks.append(register_tsvd_contract())
    checks.append(register_tsvd_lease_state())
    # lane checks-toplevel (2026-11-16)
    checks.append(register_batch_invariance_main())
    checks.append(register_bin_order_check())
    checks.append(register_binarization_check())
    checks.append(register_boosting_hist_check())
    checks.append(register_bootstrap_check())
    checks.append(register_border_types_check())
    checks.append(register_catboost_apply_check())
    checks.append(register_cindex_packing_check())
    checks.append(register_cityhash_check())
    checks.append(register_copy_histograms_check())
    checks.append(register_ctr_apply_check())
    checks.append(register_ctr_check())
    checks.append(register_ctr_device_check())
    checks.append(register_ctr_kernels_check())
    checks.append(register_ctr_permutation_check())
    checks.append(register_ctr_train_check())
    checks.append(register_depthwise_check())
    checks.append(register_division_check())
    checks.append(register_dynamic_boosting_folds_check())
    checks.append(register_early_stop_check())
    checks.append(register_exact_estimation_check())
    checks.append(register_feature_parallel_identity_check())
    checks.append(register_feature_tensor_check())
    checks.append(register_fit_pointwise_check())
    checks.append(register_fixture_rng_gate())
    checks.append(register_fold_tasks_check())
    checks.append(register_forest_inference_gpu())
    checks.append(register_forest_inference_model())
    checks.append(register_forest_ordered_resident_policy())
    checks.append(register_freq_ctr_device_check())
    checks.append(register_gbdt_cindex_staging_check())
    checks.append(register_gbdt_fused_move_check())
    checks.append(register_gbdt_large_partition_check())
    checks.append(register_gbdt_partition_cache_check())
    checks.append(register_gbdt_partition_route_check())
    checks.append(register_gbdt_per_round_check())
    checks.append(register_gbdt_prepared_check())
    checks.append(register_gbdt_step_cost_check())
    checks.append(register_gpu_lib_check())
    checks.append(register_gpu_lib_worker_check())
    checks.append(register_gram_splitk_check())
    checks.append(register_greedy_log_sum_check())
    checks.append(register_grow_policy_check())
    checks.append(register_hardware_matrix_check())
    checks.append(register_hist2_check())
    checks.append(register_hist_check())
    checks.append(register_hist_dump_check())
    checks.append(register_histograms_helper_check())
    checks.append(register_hostptr_check())
    checks.append(register_identity_trace_check())
    checks.append(register_ieee_arith_check())
    checks.append(register_kpca_dense_centering_regression())
    checks.append(register_layout_check())
    checks.append(register_leaf_partition_check())
    checks.append(register_leafwise_scores_check())
    checks.append(register_level_check())
    checks.append(register_logloss_estimator_check())
    checks.append(register_logloss_leaf_oracle_check())
    checks.append(register_logloss_target_check())
    checks.append(register_logloss_train_check())
    checks.append(register_loss_oracle_check())
    checks.append(register_lossguide_check())
    checks.append(register_lossguide_policy_check())
    checks.append(register_mcd_step_cost_check())
    checks.append(register_min_child_hessian_check())
    checks.append(register_min_entropy_check())
    checks.append(register_min_split_gain_check())
    checks.append(register_model_io_check())
    checks.append(register_multiclass_oracle_check())
    checks.append(register_multiclass_score_check())
    checks.append(register_multiclass_train_check())
    checks.append(register_multilogit_check())
    checks.append(register_nan_forbidden_host_check())
    checks.append(register_nan_mode_check())
    checks.append(register_ndtri_check())
    checks.append(register_newton_walker_check())
    checks.append(register_one_hot_cardinality_check())
    checks.append(register_one_hot_check())
    checks.append(register_one_hot_flags_check())
    checks.append(register_onebyte_reach_check())
    checks.append(register_options_check())
    checks.append(register_oracle_main())
    checks.append(register_ordered_boosting_check())
    checks.append(register_ordered_rmse_check())
    checks.append(register_overfitting_detector_check())
    checks.append(register_pack_check())
    checks.append(register_partitions_reduce_check())
    checks.append(register_permutation_count_check())
    checks.append(register_permuted_ids_check())
    checks.append(register_plan_fusion_check())
    checks.append(register_pointwise_dispatch_check())
    checks.append(register_pointwise_driver_check())
    checks.append(register_pointwise_hist2_8bit_check())
    checks.append(register_pointwise_hist2_check())
    checks.append(register_pointwise_hist2_half_byte_check())
    checks.append(register_pointwise_identical_multiplier_check())
    checks.append(register_pointwise_loop_check())
    checks.append(register_pointwise_offsets_check())
    checks.append(register_pointwise_pool_check())
    checks.append(register_pointwise_resolve_check())
    checks.append(register_pointwise_scores_check())
    checks.append(register_pointwise_small_bin_driver_check())
    checks.append(register_pointwise_subsets_check())
    checks.append(register_pointwise_target_check())
    checks.append(register_pointwise_vs_greedy_check())
    checks.append(register_portable_fmax_check())
    checks.append(register_portable_gelu_check())
    checks.append(register_portable_log2_64_check())
    checks.append(register_portable_log64_check())
    checks.append(register_portable_nn_check())
    checks.append(register_portable_pow64_check())
    checks.append(register_portable_sqrtcos_check())
    checks.append(register_portable_translog_check())
    checks.append(register_portable_trig_check())
    checks.append(register_radix_sort_check())
    checks.append(register_random_strength_check())
    checks.append(register_reorder_check())
    checks.append(register_replicated_half_byte_check())
    checks.append(register_samba_target_count_check())
    checks.append(register_sample_indices_check())
    checks.append(register_sample_quantile_unweighted_check())
    checks.append(register_scaffold_check())
    checks.append(register_second_der_weights_check())
    checks.append(register_segmented_scan_check())
    checks.append(register_sibling_tiebreak_check())
    checks.append(register_soft_f64_check())
    checks.append(register_sym_arms_check())
    checks.append(register_tensor_sync_state_check())
    checks.append(register_train_api_check())
    checks.append(register_combination_code_device_check())
    checks.append(register_tree_ctr_device_check())
    checks.append(register_tree_ctr_median_device_check())
    checks.append(register_tree_ctr_fit_device_check())
    checks.append(register_tree_ctr_slice_check())
    checks.append(register_ordered_ctr_device_check())
    checks.append(register_autograd_data_cpu())
    checks.append(register_autograd_data_gpu())
    checks.append(register_autograd_distributed_setup_failure())
    checks.append(register_autograd_embedding_cpu())
    checks.append(register_autograd_embedding_gpu())
    checks.append(register_autograd_fold_cpu())
    checks.append(register_autograd_fold_gpu())
    checks.append(register_autograd_fused_cpu())
    checks.append(register_autograd_fused_gpu())
    checks.append(register_autograd_gemm_gpu())
    checks.append(register_autograd_gpu_arguments())
    checks.append(register_autograd_loss_cpu())
    checks.append(register_autograd_loss_gpu())
    checks.append(register_autograd_lossblock_cpu())
    checks.append(register_autograd_lossblock_gpu())
    checks.append(register_autograd_optim_cpu())
    checks.append(register_autograd_order_cpu())
    checks.append(register_autograd_order_gpu())
    checks.append(register_autograd_pass1_cpu())
    checks.append(register_autograd_pass1_gpu())
    checks.append(register_autograd_program_setup_failure())
    checks.append(register_autograd_view_cpu())
    checks.append(register_autograd_view_gpu())
    checks.append(register_ensemble_builder_check())
    checks.append(register_ensemble_builder_kernels_check())
    checks.append(register_ensemble_criteria_check())
    checks.append(register_ensemble_cuml_oracle_check())
    checks.append(register_ensemble_forest_check())
    checks.append(register_ensemble_objectives_check())
    checks.append(register_ensemble_oob_check())
    checks.append(register_ensemble_predict_check())
    checks.append(register_ensemble_quantiles_check())
    checks.append(register_ensemble_regression_check())
    checks.append(register_ensemble_sample_weight_check())
    checks.append(register_ensemble_sampled_cols_check())
    checks.append(register_ensemble_shuffle_check())
    checks.append(register_ensemble_sibling_column_overlap_check())
    checks.append(register_ensemble_split_check())
    checks.append(register_ensemble_train_check())
    checks.append(register_gbdt_host_binarize_identical())
    checks.append(register_gbdt_tree_ctr_host_check())
    checks.append(register_gbdt_ordered_ctr_host_check())
    checks.append(register_gbdt_host_device_cat())
    checks.append(register_gbdt_ordered_doc_id_check())
    checks.append(register_hdbscan_sparse_mr_check())
    checks.append(register_ivf_ivf_large_k_check())
    checks.append(register_kde_reused_workspace_check())
    checks.append(register_mamba_mamba3_increment_tile_check())
    checks.append(register_mamba_mamba3_refusal_check())
    checks.append(register_mamba_mamba3_s17_tail_tree_check())
    checks.append(register_mamba_mamba3_yintra_tile_check())
    checks.append(register_metrics_host_binary_ranking_check())
    checks.append(register_metrics_host_rand_score_check())
    checks.append(register_training_byte_lm_block_copy_check())
    checks.append(register_training_chunked_lm_head_check())
    checks.append(register_training_chunked_lm_head_device_check())
    checks.append(register_training_dbscan_parallel_check())
    checks.append(register_training_maximize_check())
    checks.append(register_transformer_attention_zdot_routing_check())
    checks.append(register_xtrees_glue_check())
    checks.append(register_xtrees_zero_cover_check())
    checks.append(register_tsa_hw_route())
    checks.append(register_tsa_arima_route())
    return checks^
