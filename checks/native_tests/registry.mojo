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
# TOMBSTONE: MOJOLEARN_DECOMP_FAST_MATRIX_UNITS, MOJOLEARN_NEIGHBORS_FAST_MATRIX_UNITS (NOISE, A/B abf2) deleted 2026-11-20 by
# lane/fx-verdicts6 with their check family_matrix_units; code recoverable at 06d1e8ff34.
# Restore: git apply experiments/removed/MOJOLEARN_DECOMP_FAST_MATRIX_UNITS.patch (the check is in that patch)
from checks.native_tests.hdbscan_native import register as register_hdbscan_native
from checks.native_tests.hdbscan_surface import register as register_hdbscan_surface
from checks.native_tests.iforest_forest_export import register as register_iforest_forest_export
from checks.native_tests.kit_roundtrip import register as register_kit_roundtrip
from checks.native_tests.ocsvm_native import register as register_ocsvm_native
from checks.native_tests.ops_grad_identity import register as register_ops_grad_identity
from checks.native_tests.ops_higher import register as register_ops_higher
from checks.native_tests.ops_lm_head_v2 import register as register_ops_lm_head_v2
from checks.native_tests.ops_optimizer_requests import register as register_ops_optimizer_requests
from checks.native_tests.optimizer_descriptor import register as register_optimizer_descriptor
from checks.native_tests.optimizer_device import register as register_optimizer_device
from checks.native_tests.optimizer_maximize import register as register_optimizer_maximize
from checks.native_tests.optimizer_owner import register as register_optimizer_owner
from checks.native_tests.prophet_state import register as register_prophet_state
from checks.native_tests.query_x_held import register as register_query_x_held
from checks.native_tests.resident_second_call import register as register_resident_second_call
from checks.native_tests.runner import Check
from checks.native_tests.sequence_croston import register as register_sequence_croston
from checks.native_tests.sequence_moe import register as register_sequence_moe
from checks.native_tests.sequence_norm import register as register_sequence_norm
from checks.native_tests.sequence_rnn import register as register_sequence_rnn
from checks.native_tests.sequence_samba_surface import register as register_sequence_samba_surface
from checks.native_tests.sequence_stl import register as register_sequence_stl
from checks.native_tests.sequence_theta import register as register_sequence_theta
from checks.native_tests.sequence_var import register as register_sequence_var
from checks.native_tests.spectral_assignment import register as register_spectral_assignment
from checks.native_tests.spectral_dense import register as register_spectral_dense
from checks.native_tests.spectral_native import register as register_spectral_native
from checks.native_tests.spectral_prediction import register as register_spectral_prediction
from checks.native_tests.spectral_rbf import register as register_spectral_rbf
from checks.native_tests.spectral_resident import register as register_spectral_resident
from checks.native_tests.training_autograd_model import register as register_training_autograd_model
from checks.native_tests.training_autograd_save_resume import register as register_training_autograd_save_resume
from checks.native_tests.training_autograd_usability import register as register_training_autograd_usability
from checks.native_tests.training_chunked_lm_head_door import register as register_training_chunked_lm_head_door
from checks.native_tests.training_distributed_program import register as register_training_distributed_program
from checks.native_tests.training_moe_backward_ownership import register as register_training_moe_backward_ownership
from checks.native_tests.training_neural_inference import register as register_training_neural_inference
from checks.native_tests.training_primitives import register as register_training_primitives
from checks.native_tests.training_small_mlp import register as register_training_small_mlp
from checks.native_tests.transformer_host_block import register as register_transformer_host_block
from checks.native_tests.transformer_ops_program import register as register_transformer_ops_program
# lane tests-to-mojo-models (2026-11-16)
# lane tests-to-mojo-spatial (2026-11-16)
# lane tests-to-mojo-trees (2026-11-16)
# lane tests-to-mojo-neural (2026-11-15)
from checks.native_tests.training_small_mlp_relations import register as register_training_small_mlp_relations
# lane checks-register (2026-11-16): the standalone main() checks no CI job ran, now registered
from checks.native_tests.boundary_layout import register as register_boundary_layout
from checks.native_tests.class_weight_plan import register as register_class_weight_plan
from checks.native_tests.decomp_entry_table import register as register_decomp_entry_table
from checks.native_tests.hdbscan_plan import register as register_hdbscan_plan
from checks.native_tests.nmf_contract import register as register_nmf_contract
from checks.native_tests.nmf_state import register as register_nmf_state
from checks.native_tests.nonzero_scan import register as register_nonzero_scan
from checks.native_tests.portable_math import register as register_portable_math
from checks.native_tests.spectral_contract import register as register_spectral_contract
from checks.native_tests.spectral_state import register as register_spectral_state
from checks.native_tests.tsqr_host_ownership import register as register_tsqr_host_ownership

# lane checks-toplevel (2026-11-16): the standalone main() checks at the top of checks/ no CI job ran, now registered
# lane fx-checks-orphans (2026-11-18): the family standalone main() checks nothing ran; each suite stays in
# <family>/checks/ (its main() is now run_checks()) and its registered wrapper is checks/native_tests/<family>_<stem>.mojo
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
from checks.native_tests.hdbscan_sparse_mr_check import register as register_hdbscan_sparse_mr_check

def all_checks() -> List[Check]:
    var checks = List[Check]()
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
    checks.append(register_hdbscan_native())
    checks.append(register_hdbscan_surface())
    checks.append(register_iforest_forest_export())
    checks.append(register_kit_roundtrip())
    checks.append(register_ocsvm_native())
    checks.append(register_ops_grad_identity())
    checks.append(register_ops_higher())
    checks.append(register_ops_lm_head_v2())
    checks.append(register_ops_optimizer_requests())
    checks.append(register_optimizer_descriptor())
    checks.append(register_optimizer_device())
    checks.append(register_optimizer_maximize())
    checks.append(register_optimizer_owner())
    checks.append(register_prophet_state())
    checks.append(register_query_x_held())
    checks.append(register_resident_second_call())
    checks.append(register_sequence_croston())
    checks.append(register_sequence_moe())
    checks.append(register_sequence_norm())
    checks.append(register_sequence_rnn())
    checks.append(register_sequence_samba_surface())
    checks.append(register_sequence_stl())
    checks.append(register_sequence_theta())
    checks.append(register_sequence_var())
    checks.append(register_spectral_assignment())
    checks.append(register_spectral_dense())
    checks.append(register_spectral_native())
    checks.append(register_spectral_prediction())
    checks.append(register_spectral_rbf())
    checks.append(register_spectral_resident())
    checks.append(register_training_autograd_model())
    checks.append(register_training_autograd_save_resume())
    checks.append(register_training_autograd_usability())
    checks.append(register_training_chunked_lm_head_door())
    checks.append(register_training_distributed_program())
    checks.append(register_training_moe_backward_ownership())
    checks.append(register_training_neural_inference())
    checks.append(register_training_primitives())
    checks.append(register_training_small_mlp())
    checks.append(register_transformer_host_block())
    checks.append(register_transformer_ops_program())
    # lane tests-to-mojo-models (2026-11-16)
    # lane tests-to-mojo-spatial (2026-11-16)
    # lane tests-to-mojo-trees (2026-11-16)
    # lane tests-to-mojo-neural (2026-11-15)
    checks.append(register_training_small_mlp_relations())
    # lane checks-register (2026-11-16)
    checks.append(register_boundary_layout())
    checks.append(register_class_weight_plan())
    checks.append(register_decomp_entry_table())
    checks.append(register_hdbscan_plan())
    checks.append(register_nmf_contract())
    checks.append(register_nmf_state())
    checks.append(register_nonzero_scan())
    checks.append(register_portable_math())
    checks.append(register_spectral_contract())
    checks.append(register_spectral_state())
    checks.append(register_tsqr_host_ownership())
    # lane checks-toplevel (2026-11-16)
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
    checks.append(register_hdbscan_sparse_mr_check())
    return checks^
