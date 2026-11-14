# SPDX-License-Identifier: Apache-2.0
"""Every native check, registered: the one list `run_cpu.mojo` and `run_gpu.mojo` run.

A new check is one file `<name>.mojo` with `run_cases` and `register()` (see
`runner.mojo`) plus one import and one append here. The CI enumerates the
names from this file (`tools/ops/native_checks_report.py`), so a check that is
not appended here never runs. No `main()`: imported by the two mains.

Not here: `et_device_loop_early_exit` and `rf_device_loop_early_exit` build
one binary per compile-time route define (their own pixi tasks).
"""
from checks.native_tests.adafactor import register as register_adafactor
from checks.native_tests.agglo_native import register as register_agglo_native
from checks.native_tests.ann_ivf_surface import register as register_ann_ivf_surface
from checks.native_tests.ann_ops import register as register_ann_ops
from checks.native_tests.arima_state import register as register_arima_state
from checks.native_tests.bf16_conversion import register as register_bf16_conversion
from checks.native_tests.bf16_linalg import register as register_bf16_linalg
from checks.native_tests.byte_lm_host_column import register as register_byte_lm_host_column
from checks.native_tests.byte_lm_models_loader_f16 import register as register_byte_lm_models_loader_f16
from checks.native_tests.byte_lm_parallel_byte_native import register as register_byte_lm_parallel_byte_native
from checks.native_tests.byte_lm_trainer_session import register as register_byte_lm_trainer_session
from checks.native_tests.causal_lm_session import register as register_causal_lm_session
from checks.native_tests.centroid_fit_native import register as register_centroid_fit_native
from checks.native_tests.cluster_kmeans_query import register as register_cluster_kmeans_query
from checks.native_tests.cnn_batchnorm import register as register_cnn_batchnorm
from checks.native_tests.cnn_blocks import register as register_cnn_blocks
from checks.native_tests.cnn_conv import register as register_cnn_conv
from checks.native_tests.cnn_pool import register as register_cnn_pool
from checks.native_tests.cnn_trainer import register as register_cnn_trainer
from checks.native_tests.core_chunked_lm_head_denormals import register as register_core_chunked_lm_head_denormals
from checks.native_tests.core_convert import register as register_core_convert
from checks.native_tests.core_helpers import register as register_core_helpers
from checks.native_tests.core_ne2_rounds import register as register_core_ne2_rounds
from checks.native_tests.core_ops_import_kinds import register as register_core_ops_import_kinds
from checks.native_tests.core_ops_layernorm_view import register as register_core_ops_layernorm_view
from checks.native_tests.core_ops_program import register as register_core_ops_program
from checks.native_tests.core_ops_session import register as register_core_ops_session
from checks.native_tests.core_partitioned_program import register as register_core_partitioned_program
from checks.native_tests.core_program_fit_stop import register as register_core_program_fit_stop
from checks.native_tests.core_program_loop import register as register_core_program_loop
from checks.native_tests.core_ragged_lengths import register as register_core_ragged_lengths
from checks.native_tests.core_safetensors_f16 import register as register_core_safetensors_f16
from checks.native_tests.core_stratified_group_fp64 import register as register_core_stratified_group_fp64
from checks.native_tests.core_verifier_gap_methods import register as register_core_verifier_gap_methods
from checks.native_tests.core_vjp_rank import register as register_core_vjp_rank
from checks.native_tests.dbscan_native import register as register_dbscan_native
from checks.native_tests.dbscan_ownership import register as register_dbscan_ownership
from checks.native_tests.decomp_classical_infer_one_call import register as register_decomp_classical_infer_one_call
from checks.native_tests.decomp_holtwinters_inference import register as register_decomp_holtwinters_inference
from checks.native_tests.decomp_ne2_g_decomp_entries import register as register_decomp_ne2_g_decomp_entries
from checks.native_tests.eigh_route import register as register_eigh_route
from checks.native_tests.embedding_ne3_entries import register as register_embedding_ne3_entries
from checks.native_tests.embedding_surface import register as register_embedding_surface
from checks.native_tests.gbdt_public_fits import register as register_gbdt_public_fits
from checks.native_tests.gbdt_resident_ns_predict import register as register_gbdt_resident_ns_predict
from checks.native_tests.gp_entry_table import register as register_gp_entry_table
from checks.native_tests.gp_native import register as register_gp_native
from checks.native_tests.gp_normalization import register as register_gp_normalization
from checks.native_tests.gp_theta import register as register_gp_theta
from checks.native_tests.gpc_score import register as register_gpc_score
from checks.native_tests.gpr_score import register as register_gpr_score
from checks.native_tests.hdbscan_native import register as register_hdbscan_native
from checks.native_tests.holtwinters_state import register as register_holtwinters_state
from checks.native_tests.iforest_forest_export import register as register_iforest_forest_export
from checks.native_tests.int8_linalg import register as register_int8_linalg
from checks.native_tests.ipca_native import register as register_ipca_native
from checks.native_tests.kde_native import register as register_kde_native
from checks.native_tests.kde_ownership import register as register_kde_ownership
from checks.native_tests.kernel_methods_cells import register as register_kernel_methods_cells
from checks.native_tests.kernel_methods_krr_options import register as register_kernel_methods_krr_options
from checks.native_tests.kmeans_native import register as register_kmeans_native
from checks.native_tests.kmeans_variants_native import register as register_kmeans_variants_native
from checks.native_tests.kpca_native import register as register_kpca_native
from checks.native_tests.labels_threshold import register as register_labels_threshold
from checks.native_tests.linalg_decompositions import register as register_linalg_decompositions
from checks.native_tests.linalg_numerical import register as register_linalg_numerical
from checks.native_tests.linear_owner import register as register_linear_owner
from checks.native_tests.linear_svc_native import register as register_linear_svc_native
from checks.native_tests.linear_svr_native import register as register_linear_svr_native
from checks.native_tests.lof_native import register as register_lof_native
from checks.native_tests.logistic_label_weights import register as register_logistic_label_weights
from checks.native_tests.logistic_numeric import register as register_logistic_numeric
from checks.native_tests.lowbit_weights import register as register_lowbit_weights
from checks.native_tests.mamba12_forward_exec import register as register_mamba12_forward_exec
from checks.native_tests.mamba3_output_coverage import register as register_mamba3_output_coverage
from checks.native_tests.mamba3_prefill_session import register as register_mamba3_prefill_session
from checks.native_tests.mamba3_prefix_step import register as register_mamba3_prefix_step
from checks.native_tests.metrics_selection_host_p2 import register as register_metrics_selection_host_p2
from checks.native_tests.metrics_smallfam_public import register as register_metrics_smallfam_public
from checks.native_tests.metrics_weighted_score import register as register_metrics_weighted_score
from checks.native_tests.mixture_gmm_sample import register as register_mixture_gmm_sample
from checks.native_tests.mixture_native import register as register_mixture_native
from checks.native_tests.nearest_centroid_native import register as register_nearest_centroid_native
from checks.native_tests.nearest_centroid_numeric import register as register_nearest_centroid_numeric
from checks.native_tests.neighbors_knn_metrics import register as register_neighbors_knn_metrics
from checks.native_tests.neighbors_nearest_centroid_native_capability import register as register_neighbors_nearest_centroid_native_capability
from checks.native_tests.neighbors_ops import register as register_neighbors_ops
from checks.native_tests.neighbors_umap_surface import register as register_neighbors_umap_surface
from checks.native_tests.neighbors_weighted_largek import register as register_neighbors_weighted_largek
from checks.native_tests.nmf_native import register as register_nmf_native
from checks.native_tests.ocsvm_native import register as register_ocsvm_native
from checks.native_tests.oob_exact_sum import register as register_oob_exact_sum
from checks.native_tests.ops_higher import register as register_ops_higher
from checks.native_tests.ops_optimizer import register as register_ops_optimizer
from checks.native_tests.ops_optimizer_requests import register as register_ops_optimizer_requests
from checks.native_tests.ops_session import register as register_ops_session
from checks.native_tests.optics_native import register as register_optics_native
from checks.native_tests.optimizer_descriptor import register as register_optimizer_descriptor
from checks.native_tests.optimizer_device import register as register_optimizer_device
from checks.native_tests.optimizer_maximize import register as register_optimizer_maximize
from checks.native_tests.optimizer_owner import register as register_optimizer_owner
from checks.native_tests.pca_native import register as register_pca_native
from checks.native_tests.pca_owned_ops import register as register_pca_owned_ops
from checks.native_tests.pca_whitening import register as register_pca_whitening
from checks.native_tests.pcs_native import register as register_pcs_native
from checks.native_tests.prep_label_binarizer_inverse_window import register as register_prep_label_binarizer_inverse_window
from checks.native_tests.prep_robust_scaler_native_lifecycle import register as register_prep_robust_scaler_native_lifecycle
from checks.native_tests.prophet_state import register as register_prophet_state
from checks.native_tests.query_x_held import register as register_query_x_held
from checks.native_tests.rankn_conv import register as register_rankn_conv
from checks.native_tests.rankn_graph import register as register_rankn_graph
from checks.native_tests.rankn_math import register as register_rankn_math
from checks.native_tests.rankn_views import register as register_rankn_views
from checks.native_tests.resample_cross_vendor_fold import register as register_resample_cross_vendor_fold
from checks.native_tests.resample_cv_row_order import register as register_resample_cv_row_order
from checks.native_tests.sequence_croston import register as register_sequence_croston
from checks.native_tests.sequence_garch import register as register_sequence_garch
from checks.native_tests.sequence_moe import register as register_sequence_moe
from checks.native_tests.sequence_norm import register as register_sequence_norm
from checks.native_tests.sequence_optimizer import register as register_sequence_optimizer
from checks.native_tests.sequence_rnn import register as register_sequence_rnn
from checks.native_tests.sequence_samba_surface import register as register_sequence_samba_surface
from checks.native_tests.sequence_stl import register as register_sequence_stl
from checks.native_tests.sequence_theta import register as register_sequence_theta
from checks.native_tests.sequence_var import register as register_sequence_var
from checks.native_tests.solver_estimators_ops import register as register_solver_estimators_ops
from checks.native_tests.solver_qn_objectives import register as register_solver_qn_objectives
from checks.native_tests.spectral_assignment import register as register_spectral_assignment
from checks.native_tests.spectral_dense import register as register_spectral_dense
from checks.native_tests.spectral_native import register as register_spectral_native
from checks.native_tests.spectral_prediction import register as register_spectral_prediction
from checks.native_tests.spectral_rbf import register as register_spectral_rbf
from checks.native_tests.spectral_resident import register as register_spectral_resident
from checks.native_tests.svc_accuracy_native import register as register_svc_accuracy_native
from checks.native_tests.svc_probability import register as register_svc_probability
from checks.native_tests.svr_native import register as register_svr_native
from checks.native_tests.theta_state import register as register_theta_state
from checks.native_tests.training_chunked_lm_head_door import register as register_training_chunked_lm_head_door
from checks.native_tests.training_neural_rng_slices import register as register_training_neural_rng_slices
from checks.native_tests.training_primitives import register as register_training_primitives
from checks.native_tests.training_small_mlp import register as register_training_small_mlp
from checks.native_tests.transformer_block_session import register as register_transformer_block_session
from checks.native_tests.transformer_decode_door import register as register_transformer_decode_door
from checks.native_tests.transformer_host_block import register as register_transformer_host_block
from checks.native_tests.transformer_ops_program import register as register_transformer_ops_program
from checks.native_tests.tsa_holtwinters_initialization import register as register_tsa_holtwinters_initialization
from checks.native_tests.tsa_sequence_ets import register as register_tsa_sequence_ets
from checks.native_tests.tsa_training_arima import register as register_tsa_training_arima
from checks.native_tests.tsvd_fit import register as register_tsvd_fit
from checks.native_tests.tsvd_native import register as register_tsvd_native
from checks.native_tests.umap_ops import register as register_umap_ops
from checks.native_tests.x_linear_ops import register as register_x_linear_ops
from checks.native_tests.runner import Check


def all_checks() -> List[Check]:
    var checks = List[Check]()
    checks.append(register_adafactor())
    checks.append(register_agglo_native())
    checks.append(register_ann_ivf_surface())
    checks.append(register_ann_ops())
    checks.append(register_arima_state())
    checks.append(register_bf16_conversion())
    checks.append(register_bf16_linalg())
    checks.append(register_byte_lm_host_column())
    checks.append(register_byte_lm_models_loader_f16())
    checks.append(register_byte_lm_parallel_byte_native())
    checks.append(register_byte_lm_trainer_session())
    checks.append(register_causal_lm_session())
    checks.append(register_centroid_fit_native())
    checks.append(register_cluster_kmeans_query())
    checks.append(register_cnn_batchnorm())
    checks.append(register_cnn_blocks())
    checks.append(register_cnn_conv())
    checks.append(register_cnn_pool())
    checks.append(register_cnn_trainer())
    checks.append(register_core_chunked_lm_head_denormals())
    checks.append(register_core_convert())
    checks.append(register_core_helpers())
    checks.append(register_core_ne2_rounds())
    checks.append(register_core_ops_import_kinds())
    checks.append(register_core_ops_layernorm_view())
    checks.append(register_core_ops_program())
    checks.append(register_core_ops_session())
    checks.append(register_core_partitioned_program())
    checks.append(register_core_program_fit_stop())
    checks.append(register_core_program_loop())
    checks.append(register_core_ragged_lengths())
    checks.append(register_core_safetensors_f16())
    checks.append(register_core_stratified_group_fp64())
    checks.append(register_core_verifier_gap_methods())
    checks.append(register_core_vjp_rank())
    checks.append(register_dbscan_native())
    checks.append(register_dbscan_ownership())
    checks.append(register_decomp_classical_infer_one_call())
    checks.append(register_decomp_holtwinters_inference())
    checks.append(register_decomp_ne2_g_decomp_entries())
    checks.append(register_eigh_route())
    checks.append(register_embedding_ne3_entries())
    checks.append(register_embedding_surface())
    checks.append(register_gbdt_public_fits())
    checks.append(register_gbdt_resident_ns_predict())
    checks.append(register_gp_entry_table())
    checks.append(register_gp_native())
    checks.append(register_gp_normalization())
    checks.append(register_gp_theta())
    checks.append(register_gpc_score())
    checks.append(register_gpr_score())
    checks.append(register_hdbscan_native())
    checks.append(register_holtwinters_state())
    checks.append(register_iforest_forest_export())
    checks.append(register_int8_linalg())
    checks.append(register_ipca_native())
    checks.append(register_kde_native())
    checks.append(register_kde_ownership())
    checks.append(register_kernel_methods_cells())
    checks.append(register_kernel_methods_krr_options())
    checks.append(register_kmeans_native())
    checks.append(register_kmeans_variants_native())
    checks.append(register_kpca_native())
    checks.append(register_labels_threshold())
    checks.append(register_linalg_decompositions())
    checks.append(register_linalg_numerical())
    checks.append(register_linear_owner())
    checks.append(register_linear_svc_native())
    checks.append(register_linear_svr_native())
    checks.append(register_lof_native())
    checks.append(register_logistic_label_weights())
    checks.append(register_logistic_numeric())
    checks.append(register_lowbit_weights())
    checks.append(register_mamba12_forward_exec())
    checks.append(register_mamba3_output_coverage())
    checks.append(register_mamba3_prefill_session())
    checks.append(register_mamba3_prefix_step())
    checks.append(register_metrics_selection_host_p2())
    checks.append(register_metrics_smallfam_public())
    checks.append(register_metrics_weighted_score())
    checks.append(register_mixture_gmm_sample())
    checks.append(register_mixture_native())
    checks.append(register_nearest_centroid_native())
    checks.append(register_nearest_centroid_numeric())
    checks.append(register_neighbors_knn_metrics())
    checks.append(register_neighbors_nearest_centroid_native_capability())
    checks.append(register_neighbors_ops())
    checks.append(register_neighbors_umap_surface())
    checks.append(register_neighbors_weighted_largek())
    checks.append(register_nmf_native())
    checks.append(register_ocsvm_native())
    checks.append(register_oob_exact_sum())
    checks.append(register_ops_higher())
    checks.append(register_ops_optimizer())
    checks.append(register_ops_optimizer_requests())
    checks.append(register_ops_session())
    checks.append(register_optics_native())
    checks.append(register_optimizer_descriptor())
    checks.append(register_optimizer_device())
    checks.append(register_optimizer_maximize())
    checks.append(register_optimizer_owner())
    checks.append(register_pca_native())
    checks.append(register_pca_owned_ops())
    checks.append(register_pca_whitening())
    checks.append(register_pcs_native())
    checks.append(register_prep_label_binarizer_inverse_window())
    checks.append(register_prep_robust_scaler_native_lifecycle())
    checks.append(register_prophet_state())
    checks.append(register_query_x_held())
    checks.append(register_rankn_conv())
    checks.append(register_rankn_graph())
    checks.append(register_rankn_math())
    checks.append(register_rankn_views())
    checks.append(register_resample_cross_vendor_fold())
    checks.append(register_resample_cv_row_order())
    checks.append(register_sequence_croston())
    checks.append(register_sequence_garch())
    checks.append(register_sequence_moe())
    checks.append(register_sequence_norm())
    checks.append(register_sequence_optimizer())
    checks.append(register_sequence_rnn())
    checks.append(register_sequence_samba_surface())
    checks.append(register_sequence_stl())
    checks.append(register_sequence_theta())
    checks.append(register_sequence_var())
    checks.append(register_solver_estimators_ops())
    checks.append(register_solver_qn_objectives())
    checks.append(register_spectral_assignment())
    checks.append(register_spectral_dense())
    checks.append(register_spectral_native())
    checks.append(register_spectral_prediction())
    checks.append(register_spectral_rbf())
    checks.append(register_spectral_resident())
    checks.append(register_svc_accuracy_native())
    checks.append(register_svc_probability())
    checks.append(register_svr_native())
    checks.append(register_theta_state())
    checks.append(register_training_chunked_lm_head_door())
    checks.append(register_training_neural_rng_slices())
    checks.append(register_training_primitives())
    checks.append(register_training_small_mlp())
    checks.append(register_transformer_block_session())
    checks.append(register_transformer_decode_door())
    checks.append(register_transformer_host_block())
    checks.append(register_transformer_ops_program())
    checks.append(register_tsa_holtwinters_initialization())
    checks.append(register_tsa_sequence_ets())
    checks.append(register_tsa_training_arima())
    checks.append(register_tsvd_fit())
    checks.append(register_tsvd_native())
    checks.append(register_umap_ops())
    checks.append(register_x_linear_ops())
    return checks^
