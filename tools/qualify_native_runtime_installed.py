#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Test selected public native-runtime operations from actual installed wheels.

Test/packaging controller only: never compiles, patches a loader, or changes a
wheel's source witness. Linux assembly delegates to qualification_wheels.py;
prebuilt Mac/Linux wheels can use the same fresh-install checks. This produces
partial installed-test evidence, never a release or native source-reuse proof.
"""
import argparse
import base64
import csv
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tempfile
import sys
import types
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_TESTS = {
    "test_classical_exceptions_native_owner.py": "Fitted-state exception compatibility through native estimator entries and connected reduction/restoration bridges, with caller-owned pickle",
    "test_not_fitted_native_lifetime.py": "Public NotFittedError reduction payload, caller-owned pickle compatibility and native singleton ownership",
    "test_cache_supports_native.py": "Public DeviceCache.supports native binding metadata, Unicode prefix names, CPU refusal compatibility and callback admission",
    "test_cache_lifecycle_native.py": "Public DeviceCache.live connected native store counts and identity-preserving context entry",
    "test_cache_upload_native.py": "Public DeviceCache.id_of native hit/miss preparation, upload publication, retained owners and failure cleanup",
    "test_cache_key_native.py": "Public DeviceCache.get and resident.stats native tuple-key admission, deep nesting and callback refusals",
    "test_cache_release_native.py": "Public DeviceCache constructor/reinitialization, release, close and context exit, and resident scope enter/exit, with first-failure error order and owners released after every free",
    "test_resident_lifecycle_native.py": "model_selection resident constructor/reinitialization, take, release, close and context exit for NumPy, pandas and Arrow-backed pandas owners, with first-failure error order",
    "test_archive_unicode_native.py": "Public archive strict Unicode errors, loaded state, partial save bytes and connected codec identity refusals",
    "test_classical_archive_reader_native.py": "Public linear archive loading, compressed state, native helper identities, file errors and input/output ownership",
    "test_host_density_save_native.py": "Public host KDE and IsolationForest save bytes, native configuration, partial state, loaded state and callback refusals",
    "test_host_density_hash_native.py": "Public host KDE and IsolationForest model digests, native field ordering, stored buffer bytes, ownership and hook refusals",
    "test_classical_archive_writer_native.py": "Public linear/scaler archive bytes, file errors, output ownership and native callback refusals",
    "test_standard_scaler_native_lifecycle.py": "Public StandardScaler lifecycle, checkpoints, consumed state and native refusals",
    "test_minmax_scaler_native_lifecycle.py": "Public MinMaxScaler lifecycle, checkpoints, consumed state and native refusals",
    "test_sequential_native_buffers.py": "Public Sequential fit/predict preparation and consumed owned buffers",
    "test_iforest_depth_native.py": "Public HostIsolationForest.fit native depth-state promotion, arithmetic, warning phases and consumed state",
    "test_cache_get_native.py": "Public DeviceCache.get native lookup, metadata conversion and callback refusals",
    "test_cache_stats_native.py": "Public DeviceCache length and resident.stats native metadata, counters, thread-local semantics and hook refusals",
    "test_optimizer_native_scalar_constructors.py": "Native scalar preparation across public optimizer constructors, updates, checkpoints and device controls",
    "test_late_numpy_native_metadata.py": "Public cold/late NumPy metadata connection, builtin identity preservation and pre-hook factory/key refusals",
    "test_optimizer_native_config_order.py": "Native public optimizer get/set parameter order independent of mutable sorting",
    "test_optimizer_native_container_admission.py": "Native optimizer and training container admission with replaced builtin names",
    "test_training_cross_entropy_public_native.py": "Public cross entropy preparation, options, native scalar configuration and consumed host/device outputs",
    "test_training_device_native_allocator.py": "Native training device owner allocation independent of mutable object constructors",
    "test_host_density_constructors_native.py": "Public host KDE and IsolationForest constructor options, inherited modes and safe reinitialization",
    "test_device_rows_materialize_native.py": "Public DeviceRows materialization, typed resident snapshots, cache aliases, conversion failures and native hook refusals",
    "test_native_owner_builtin_identity.py": "Public DeviceRows reinitialization preserves native retained-owner builtin identities",
    "test_training_embedding_public_native.py": "Public embedding forward/backward preparation, index conversion, vocabulary options and owned outputs",
    "test_optimizer_native_array_ownership.py": "Native output Array storage, retained checkpoint ownership and device downloads",
    "test_device_rows_metadata_native.py": "DeviceRows constructor and length metadata ownership and native hook refusals",
    "test_host_kde_fit_native.py": "Explicit CPU HostKernelDensity fit preparation, stored state, aliases and native refusals",
    "test_host_iforest_fit_native.py": "Explicit CPU HostIsolationForest fit preparation, seeded training, state publication and native refusals",
    "test_native_scalar_decode_public.py": "Native numeric scalar and 0-D preparation through HostIsolationForest.fit; cache.get and other NumPy numeric call sites remain separate work",
    "test_optimizer_native_step_errors.py": "Public optimizer step exception categories, unchanged refused state and native entry admission",
    "test_training_rms_public_native.py": "Public RMS forward/backward preparation, scalar options, consumed results and native refusals",
    "test_training_scalar_buffer_native.py": "Public RMS, embedding and cross-entropy primitive NumPy scalar/size-one array decoding, consumed outputs, warnings and native refusals",
    "test_training_linear_public_native.py": "Public linear forward/backward input preparation, output ownership and native failure paths",
    "test_maxabs_scaler_native_transport.py": "MaxAbsScaler scratch, mapped outputs, resident transport and operation-owned slot cleanup",
    "test_robust_scaler_native_lifecycle.py": "RobustScaler lifecycle, options, inputs, state and native boundary refusals",
    "test_scaler_complex_warning_native.py": "Native scaler warning categories, filter actions and Python hook refusals",
    "test_scaler_rows_snapshot_native.py": "RobustScaler and MaxAbsScaler retained typed DeviceRows snapshots",
    "test_robust_scaler_native_transport.py": "RobustScaler scratch, mapped outputs, resident transport and operation-owned slots",
    "test_training_device_native_rollback.py": "Training device allocation and optimizer parameter copy rollback with real native failures",
    "test_training_device_native_finalizer.py": "Automatic training device owner cleanup, views, rollback guards and shutdown",
    "test_samba_step_native.py": "SambaStack.train_step, optimizer owners, rollback and native schedule refusal",
    "test_samba_inference_native.py": "SambaStack.forward/loss/allocate_state/step and consumed full states",
    "test_samba_checkpoint_native.py": "SambaStack save/load and continuation/corruption",
    "test_samba_native_admission.py": "Public Samba argument admission before Python hooks",
    "test_sequential_native_execution.py": "Sequential.fit/predict/score and consumed training state",
    "test_sequential_native_checkpoint.py": "Sequential checkpoint state and continuation",
    "test_sequential_native_admission.py": "Public Sequential argument admission before Python hooks",
    "test_generator_draw_native.py": "Generator public draws and stream/error semantics",
    "test_generation_native.py": "CausalLM generation/stream/stopping/resume and consumed outputs",
    "test_native_owner_qualification.py": "Bare SGD/Adam/AdamW state replacement, checkpoint/resume, IO refusal and resident transfer cleanup",
    "test_sequence_shutdown_native.py": "Public SequenceDeviceTensor finalization, interpreter shutdown, release retry and callback refusal",
    "test_host_inference_native.py": "Explicit CPU HostElasticNet/HostLasso/HostKMeans public prediction, validation and attributes",
    "test_host_kernel_inference_native.py": "Explicit CPU HostKernelRidge/HostNystroem/HostRBFSampler public dense/sparse prediction and transform",
    "test_host_projection_native.py": "Explicit CPU HostPCA/HostTruncatedSVD public projection, inverse projection and whitening",
    "test_host_logistic_native.py": "Explicit CPU HostLogisticRegression public decisions, predictions and probabilities",
    "test_host_svr_native.py": "Explicit CPU HostSVR public predictions and computed coefficients",
    "test_host_svc_native.py": "Explicit CPU HostSVC public margins, predictions, probabilities and computed coefficients",
    "test_host_kde_native.py": "Explicit CPU HostKernelDensity public sample scores, total score and native state admission",
    "test_host_iforest_native.py": "Explicit CPU HostIsolationForest public scores, labels and saved control/state readback",
    # Release runtime-<vendor> coverage (tools/release_runtime_coverage.json):
    # native local workers and built-in estimator combinations.
    "test_native_process_pool.py": "Native local worker pool wire, process observations and callback-free pool operations",
    "test_native_parallel_cv_wire.py": "Native parallel CV constructor wire and authenticated worker inventory contracts",
    "test_native_meta_public_boundary.py": "Public meta-estimator calls and native lifecycle behavior",
    "test_meta_native.py": "Meta-estimators on the public path, refusing non-mojolearn members by name",
    "test_native_cv_lda.py": "LDA constructor, parameter metadata and native cross-validation entries",
    "test_native_cv_gbdt_nb.py": "Boosting and naive Bayes native CV, search and permutation paths",
    "test_native_search_dispatch.py": "Public search construction, configuration, sampling and callback refusal",
    "test_native_cv_scorers.py": "Native scorer ABI, option semantics and whole public CV audit coverage",
    "test_native_cv_outputs.py": "Public native CV label placement and canonical descriptor compatibility",
}
# Candidate array tests call _OPT_NATIVE directly. Keep them outside this public
# registry until tests actually exercise the public __array__ protocol.
TEST_DEPENDENCIES = {
    "test_classical_exceptions_native_owner.py": ("scikit-learn",),
    "test_cache_release_native.py": ("pyarrow",),
    "test_resident_lifecycle_native.py": ("pandas", "pyarrow"),
    "test_host_kde_fit_native.py": ("pandas", "scipy", "torch"),
    "test_host_iforest_fit_native.py": ("pandas", "scipy", "torch"),
    "test_host_inference_native.py": ("pandas",),
    "test_host_kernel_inference_native.py": ("pandas", "scipy"),
    "test_host_projection_native.py": ("pandas", "scipy"),
    "test_host_logistic_native.py": ("pandas",),
    "test_host_svr_native.py": ("pandas", "scipy"),
    "test_host_svc_native.py": ("pandas", "scipy"),
    "test_host_kde_native.py": ("pandas", "scipy", "torch"),
    "test_host_iforest_native.py": ("pandas", "scipy", "torch"),
    "test_native_cv_lda.py": ("pandas",),
}
CPU_RESIDENT_SKIP = "the CPU binding has no resident tensor pool"
CPU_NOT_APPLICABLE = {
    "test_cache_release_native.py": {
        'test_real_slots_are_freed_by_one_native_call[close-x_prep]': 'real word-store slots require a GPU package; explicitly excluded from CPU selection',
        'test_real_slots_are_freed_by_one_native_call[exit-x_prep]': 'real word-store slots require a GPU package; explicitly excluded from CPU selection',
        'test_real_slots_are_freed_by_one_native_call[reinit-x_prep]': 'real word-store slots require a GPU package; explicitly excluded from CPU selection',
        'test_real_slots_are_freed_by_one_native_call[release-x_prep]': 'real word-store slots require a GPU package; explicitly excluded from CPU selection',
        'test_error_order_first_failure_all_slots_attempted_failed_retained[close-x_prep]': 'real word-store slots require a GPU package; explicitly excluded from CPU selection',
        'test_error_order_first_failure_all_slots_attempted_failed_retained[exit-x_prep]': 'real word-store slots require a GPU package; explicitly excluded from CPU selection',
        'test_error_order_first_failure_all_slots_attempted_failed_retained[reinit-x_prep]': 'real word-store slots require a GPU package; explicitly excluded from CPU selection',
        'test_repeated_close_and_close_after_release[x_prep]': 'real word-store slots require a GPU package; explicitly excluded from CPU selection',
        'test_context_exit_on_error_frees_and_propagates[x_prep]': 'real word-store slots require a GPU package; explicitly excluded from CPU selection',
        'test_failed_release_retains_the_entry[x_prep]': 'real word-store slots require a GPU package; explicitly excluded from CPU selection',
        'test_arrow_owner_finalizer_runs_after_every_free[x_prep]': 'real word-store slots require a GPU package; explicitly excluded from CPU selection',
        'test_resident_scope_exit_frees_scope_caches[x_prep]': 'real word-store slots require a GPU package; explicitly excluded from CPU selection',
        'test_real_slots_are_freed_by_one_native_call[close-x_metrics]': 'real word-store slots require a GPU package; explicitly excluded from CPU selection',
        'test_real_slots_are_freed_by_one_native_call[exit-x_metrics]': 'real word-store slots require a GPU package; explicitly excluded from CPU selection',
        'test_real_slots_are_freed_by_one_native_call[reinit-x_metrics]': 'real word-store slots require a GPU package; explicitly excluded from CPU selection',
        'test_real_slots_are_freed_by_one_native_call[release-x_metrics]': 'real word-store slots require a GPU package; explicitly excluded from CPU selection',
        'test_error_order_first_failure_all_slots_attempted_failed_retained[close-x_metrics]': 'real word-store slots require a GPU package; explicitly excluded from CPU selection',
        'test_error_order_first_failure_all_slots_attempted_failed_retained[exit-x_metrics]': 'real word-store slots require a GPU package; explicitly excluded from CPU selection',
        'test_error_order_first_failure_all_slots_attempted_failed_retained[reinit-x_metrics]': 'real word-store slots require a GPU package; explicitly excluded from CPU selection',
        'test_repeated_close_and_close_after_release[x_metrics]': 'real word-store slots require a GPU package; explicitly excluded from CPU selection',
        'test_context_exit_on_error_frees_and_propagates[x_metrics]': 'real word-store slots require a GPU package; explicitly excluded from CPU selection',
        'test_failed_release_retains_the_entry[x_metrics]': 'real word-store slots require a GPU package; explicitly excluded from CPU selection',
        'test_arrow_owner_finalizer_runs_after_every_free[x_metrics]': 'real word-store slots require a GPU package; explicitly excluded from CPU selection',
        'test_resident_scope_exit_frees_scope_caches[x_metrics]': 'real word-store slots require a GPU package; explicitly excluded from CPU selection',
    },
    "test_cache_lifecycle_native.py": {
        'test_cache_live_matches_real_store_count_after_consumed_allocations[0-x_prep]': "real live-slot counter requires a GPU package; explicitly excluded from CPU selection",
        'test_cache_live_matches_real_store_count_after_consumed_allocations[0-x_metrics]': "real live-slot counter requires a GPU package; explicitly excluded from CPU selection",
        'test_cache_live_matches_real_store_count_after_consumed_allocations[1-x_prep]': "real live-slot counter requires a GPU package; explicitly excluded from CPU selection",
        'test_cache_live_matches_real_store_count_after_consumed_allocations[1-x_metrics]': "real live-slot counter requires a GPU package; explicitly excluded from CPU selection",
        'test_cache_live_matches_real_store_count_after_consumed_allocations[2-x_prep]': "real live-slot counter requires a GPU package; explicitly excluded from CPU selection",
        'test_cache_live_matches_real_store_count_after_consumed_allocations[2-x_metrics]': "real live-slot counter requires a GPU package; explicitly excluded from CPU selection",
        'test_cache_live_uses_captured_native_entry_and_ignores_mutable_int[x_prep]': "real live-slot counter requires a GPU package; explicitly excluded from CPU selection",
        'test_cache_live_uses_captured_native_entry_and_ignores_mutable_int[x_metrics]': "real live-slot counter requires a GPU package; explicitly excluded from CPU selection",
    },

    "test_cache_upload_native.py": {
        'test_cache_id_of_real_upload_hit_alias_and_owned_publication[x_prep]': 'real word-store upload requires a GPU package; explicitly excluded from CPU selection',
        'test_cache_id_of_real_upload_hit_alias_and_owned_publication[x_metrics]': 'real word-store upload requires a GPU package; explicitly excluded from CPU selection',
        'test_cache_id_of_real_native_upload_failure_preserves_empty_state[null-x_prep]': 'real word-store upload requires a GPU package; explicitly excluded from CPU selection',
        'test_cache_id_of_real_native_upload_failure_preserves_empty_state[null-x_metrics]': 'real word-store upload requires a GPU package; explicitly excluded from CPU selection',
        'test_cache_id_of_real_native_upload_failure_preserves_empty_state[address_overflow-x_prep]': 'real word-store upload requires a GPU package; explicitly excluded from CPU selection',
        'test_cache_id_of_real_native_upload_failure_preserves_empty_state[address_overflow-x_metrics]': 'real word-store upload requires a GPU package; explicitly excluded from CPU selection',
        'test_cache_id_of_real_native_upload_failure_preserves_empty_state[size_overflow-x_prep]': 'real word-store upload requires a GPU package; explicitly excluded from CPU selection',
        'test_cache_id_of_real_native_upload_failure_preserves_empty_state[size_overflow-x_metrics]': 'real word-store upload requires a GPU package; explicitly excluded from CPU selection',
        'test_cache_id_of_real_upload_uses_connected_store_entries[x_prep]': 'real word-store upload requires a GPU package; explicitly excluded from CPU selection',
        'test_cache_id_of_real_upload_uses_connected_store_entries[x_metrics]': 'real word-store upload requires a GPU package; explicitly excluded from CPU selection',
        'test_cache_id_of_real_concurrent_publication_cleanup[replace_dictionary-x_prep]': 'real word-store upload requires a GPU package; explicitly excluded from CPU selection',
        'test_cache_id_of_real_concurrent_publication_cleanup[replace_dictionary-x_metrics]': 'real word-store upload requires a GPU package; explicitly excluded from CPU selection',
        'test_cache_id_of_real_concurrent_publication_cleanup[publish_competitor-x_prep]': 'real word-store upload requires a GPU package; explicitly excluded from CPU selection',
        'test_cache_id_of_real_concurrent_publication_cleanup[publish_competitor-x_metrics]': 'real word-store upload requires a GPU package; explicitly excluded from CPU selection',
        'test_cache_id_of_real_concurrent_publication_cleanup[change_route-x_prep]': 'real word-store upload requires a GPU package; explicitly excluded from CPU selection',
        'test_cache_id_of_real_concurrent_publication_cleanup[change_route-x_metrics]': 'real word-store upload requires a GPU package; explicitly excluded from CPU selection',
        'test_cache_id_of_real_concurrent_publication_cleanup[delete_counter_slot-x_prep]': 'real word-store upload requires a GPU package; explicitly excluded from CPU selection',
        'test_cache_id_of_real_concurrent_publication_cleanup[delete_counter_slot-x_metrics]': 'real word-store upload requires a GPU package; explicitly excluded from CPU selection',
        'test_cache_id_of_real_concurrent_publication_cleanup[foreign_dictionary-x_prep]': 'real word-store upload requires a GPU package; explicitly excluded from CPU selection',
        'test_cache_id_of_real_concurrent_publication_cleanup[foreign_dictionary-x_metrics]': 'real word-store upload requires a GPU package; explicitly excluded from CPU selection',
    },
    "test_training_scalar_buffer_native.py": {
        "test_device_loss_scalar_conversion_and_failure_remain_native[label_smoothing-g-0.125-swapped]": "resident cross-entropy scalar options require the GPU training pool",
        "test_device_loss_scalar_conversion_and_failure_remain_native[ignore_index-c8-(-100+1j)-scalar]": "resident cross-entropy scalar options require the GPU training pool",
        "test_device_loss_scalar_conversion_and_failure_remain_native[num_items-g-3.75-swapped]": "resident cross-entropy scalar options require the GPU training pool",
        "test_device_loss_scalar_conversion_and_failure_remain_native[num_items-g-nan-scalar]": "resident cross-entropy scalar options require the GPU training pool",
    },
    "test_optimizer_native_scalar_constructors.py": {
        "test_device_scalar_controls_use_retained_native_integer[reshape]": "device scalar controls require a GPU training pool",
        "test_device_scalar_controls_use_retained_native_integer[param_device]": "device scalar controls require a GPU training pool",
    },
    'test_training_cross_entropy_public_native.py': {'test_public_cross_entropy_device_failure_rollback_has_no_python_finalizer': 'training device pool requires GPU', 'test_public_cross_entropy_device_outputs_are_native_and_consumed[mean-False-0.0]': 'training device pool requires GPU', 'test_public_cross_entropy_device_outputs_are_native_and_consumed[mean-True-0.0]': 'training device pool requires GPU', 'test_public_cross_entropy_device_outputs_are_native_and_consumed[none-False-0.2]': 'training device pool requires GPU', 'test_public_cross_entropy_device_outputs_are_native_and_consumed[sum-True-0.2]': 'training device pool requires GPU', 'test_public_cross_entropy_device_preserves_refusal_categories[logits-rank]': 'training device pool requires GPU', 'test_public_cross_entropy_device_preserves_refusal_categories[none-backward]': 'training device pool requires GPU', 'test_public_cross_entropy_device_preserves_refusal_categories[targets-float]': 'training device pool requires GPU', 'test_public_cross_entropy_device_preserves_refusal_categories[targets-rank]': 'training device pool requires GPU'},
    'test_training_device_native_allocator.py': {'test_training_device_owner_refuses_replaced_object_constructor[upload]': 'training device allocation requires the GPU pool', 'test_training_device_owner_refuses_replaced_object_constructor[reshape]': 'training device allocation requires the GPU pool'},
    "test_optimizer_native_array_ownership.py": {
        "test_native_device_download_does_not_call_mutable_memoryview[False]": "device downloads require the GPU training pool",
        "test_native_device_download_does_not_call_mutable_memoryview[True]": "device downloads require the GPU training pool",
        "test_native_device_download_does_not_call_mutable_object[False]": "device downloads require the GPU training pool",
        "test_native_device_download_does_not_call_mutable_object[True]": "device downloads require the GPU training pool",
        "test_native_device_download_does_not_call_mutable_bytes[False]": "device downloads require the GPU training pool",
        "test_native_device_download_does_not_call_mutable_bytes[True]": "device downloads require the GPU training pool",
    },
    'test_training_linear_public_native.py': {
        'test_public_device_linear_consumed_outputs_are_native[linear_forward-matrix]': 'resident linear operations require the GPU training binding',
        'test_public_device_linear_consumed_outputs_are_native[linear_forward-batch]': 'resident linear operations require the GPU training binding',
        'test_public_device_linear_consumed_outputs_are_native[linear_forward-vector]': 'resident linear operations require the GPU training binding',
        'test_public_device_linear_consumed_outputs_are_native[linear_forward-strided]': 'resident linear operations require the GPU training binding',
        'test_public_device_linear_consumed_outputs_are_native[linear_backward-matrix]': 'resident linear operations require the GPU training binding',
        'test_public_device_linear_consumed_outputs_are_native[linear_backward-batch]': 'resident linear operations require the GPU training binding',
        'test_public_device_linear_consumed_outputs_are_native[linear_backward-vector]': 'resident linear operations require the GPU training binding',
        'test_public_device_linear_consumed_outputs_are_native[linear_backward-strided]': 'resident linear operations require the GPU training binding',
        'test_public_device_linear_refuses_mixed_owners[linear_forward]': 'resident linear operations require the GPU training binding',
        'test_public_device_linear_refuses_mixed_owners[linear_backward]': 'resident linear operations require the GPU training binding',
        'test_public_device_linear_nonfinite_refusal_releases_outputs[linear_forward-input]': 'resident linear operations require the GPU training binding',
        'test_public_device_linear_nonfinite_refusal_releases_outputs[linear_forward-weight]': 'resident linear operations require the GPU training binding',
        'test_public_device_linear_nonfinite_refusal_releases_outputs[linear_backward-input]': 'resident linear operations require the GPU training binding',
        'test_public_device_linear_nonfinite_refusal_releases_outputs[linear_backward-weight]': 'resident linear operations require the GPU training binding',
        'test_public_device_linear_nonfinite_refusal_releases_outputs[linear_backward-gradient]': 'resident linear operations require the GPU training binding',
    },
    'test_maxabs_scaler_native_transport.py': {
        'test_resident_snapshot_counters_and_slot_lifetime_are_preserved[False]': 'GPU ranges/store ABI is not applicable to the CPU facade',
        'test_resident_snapshot_counters_and_slot_lifetime_are_preserved[True]': 'GPU ranges/store ABI is not applicable to the CPU facade',
        'test_device_row_failure_releases_operation_slots_and_retains_scope_owner[False]': 'GPU ranges/store ABI is not applicable to the CPU facade',
        'test_device_row_failure_releases_operation_slots_and_retains_scope_owner[True]': 'GPU ranges/store ABI is not applicable to the CPU facade',
        'test_every_owned_release_is_attempted_and_primary_error_is_retained[False]': 'GPU ranges/store ABI is not applicable to the CPU facade',
        'test_every_owned_release_is_attempted_and_primary_error_is_retained[True]': 'GPU ranges/store ABI is not applicable to the CPU facade',
        'test_resident_class_hooks_are_refused_before_cache_allocation[dict]': 'GPU ranges/store ABI is not applicable to the CPU facade',
        'test_resident_class_hooks_are_refused_before_cache_allocation[finalizer]': 'GPU ranges/store ABI is not applicable to the CPU facade',
        'test_resident_class_hooks_are_refused_before_cache_allocation[base]': 'GPU ranges/store ABI is not applicable to the CPU facade',
    },
    "test_robust_scaler_native_transport.py": {
        'test_every_owned_release_is_attempted_and_primary_error_is_retained[False]': "GPU ranges/store ABI is not applicable to the CPU facade",
        'test_every_owned_release_is_attempted_and_primary_error_is_retained[True]': "GPU ranges/store ABI is not applicable to the CPU facade",
        'test_resident_snapshot_counters_and_slot_lifetime_are_preserved[False]': "GPU ranges/store ABI is not applicable to the CPU facade",
        'test_resident_snapshot_counters_and_slot_lifetime_are_preserved[True]': "GPU ranges/store ABI is not applicable to the CPU facade",
        'test_device_row_failure_releases_operation_slots_and_retains_scope_owner[False]': "GPU ranges/store ABI is not applicable to the CPU facade",
        'test_device_row_failure_releases_operation_slots_and_retains_scope_owner[True]': "GPU ranges/store ABI is not applicable to the CPU facade",
        'test_resident_class_hooks_are_refused_before_cache_allocation[dict]': "GPU ranges/store ABI is not applicable to the CPU facade",
        'test_resident_class_hooks_are_refused_before_cache_allocation[finalizer]': "GPU ranges/store ABI is not applicable to the CPU facade",
        'test_resident_class_hooks_are_refused_before_cache_allocation[base]': "GPU ranges/store ABI is not applicable to the CPU facade",
    },
    "test_training_device_native_rollback.py": {
        'test_native_device_allocation_refusal_destroys_prepared_owner_without_python': "training device rollback requires the GPU binding",
        'test_native_upload_failure_releases_real_allocation': "training device rollback requires the GPU binding",
        'test_native_parameter_copy_failure_releases_real_allocation': "training device rollback requires the GPU binding",
        'test_native_allocation_refuses_python_ownership_entries[train_dev_alloc]': "training device rollback requires the GPU binding",
        'test_native_allocation_refuses_python_ownership_entries[train_dev_put]': "training device rollback requires the GPU binding",
        'test_native_allocation_refuses_python_ownership_entries[train_dev_free]': "training device rollback requires the GPU binding"
    },
    "test_training_device_native_finalizer.py": {
        'test_native_training_automatic_cleanup_releases_last_device_owner[False]': "training device arrays require the GPU binding",
        'test_native_training_automatic_cleanup_releases_last_device_owner[True]': "training device arrays require the GPU binding",
        'test_native_training_release_failure_retains_owner_for_retry': "training device arrays require the GPU binding",
        'test_native_training_cleanup_refuses_python_conversion_and_release[handle]': "training device arrays require the GPU binding",
        'test_native_training_cleanup_refuses_python_conversion_and_release[release]': "training device arrays require the GPU binding",
        'test_native_training_cleanup_survives_import_shutdown[explicit]': "training device arrays require the GPU binding",
        'test_native_training_cleanup_survives_import_shutdown[shutdown]': "training device arrays require the GPU binding"
    },
    "test_native_owner_qualification.py": {
        "test_failed_native_sequence_transfer_releases_allocated_handle[seq_tensor_upload]": CPU_RESIDENT_SKIP,
        "test_failed_native_sequence_transfer_releases_allocated_handle[seq_tensor_copy]": CPU_RESIDENT_SKIP,
    },
    "test_sequence_shutdown_native.py": {
        "test_sequence_cleanup_without_imports[shutdown]": CPU_RESIDENT_SKIP,
        "test_sequence_cleanup_without_imports[explicit]": CPU_RESIDENT_SKIP,
        "test_sequence_cleanup_without_imports[gc]": CPU_RESIDENT_SKIP,
        "test_sequence_cleanup_retries_failed_native_release": CPU_RESIDENT_SKIP,
        "test_sequence_cleanup_refuses_conversion_and_release_callbacks": CPU_RESIDENT_SKIP,
    },
}


def test_applicability(selected, vendor):
    result = {}
    for name in selected:
        scope = dict(package_vendor=vendor, execution_route="selected package backend",
                     operation_mode="test-specific configuration", cpu_not_applicable_cases={})
        if name == "test_not_fitted_native_lifetime.py":
            scope.update(execution_route="native fitted-error reduction on every package backend; no GPU kernel",
                         operation_mode="identical public unfitted refusal and bound exception reduction",
                         coverage_limit="Only public NotFittedError.__reduce__ and its payload; pickle is caller-owned. CPython 3.11 is required to expose borrowed-None lifetime loss; immortal-None interpreters cover payload and native calls only. The ordinary optional-package fixture does not qualify an installed sklearn release.")
        if name == "test_classical_exceptions_native_owner.py":
            scope.update(execution_route="native unfitted estimator entries and connected exception bridges; no GPU kernel",
                         operation_mode="test-specific unfitted refusals and exception reduction/restoration",
                         coverage_limit="Selected exception identity, sklearn relationship and bridge cases only; pickle is caller-owned. Direct native entry checks do not qualify every public estimator wrapper or exception path. Borrowed-None lifetime needs separate CPython 3.11 execution.")
        if name in CPU_NOT_APPLICABLE:
            scope["operation_mode"] = "identical (explicit fixture configuration)"
            scope["execution_route"] = ("resident GPU tensor pool" if name == "test_sequence_shutdown_native.py"
                                        else "bare optimizer owner; resident cleanup only on GPU")
            scope["coverage_limit"] = "Owner lifecycle checks do not qualify the public __array__ conversion protocol"
            if vendor == "cpu":
                scope["cpu_not_applicable_cases"] = dict(CPU_NOT_APPLICABLE[name])
        if name in ("test_standard_scaler_native_lifecycle.py", "test_minmax_scaler_native_lifecycle.py", "test_robust_scaler_native_lifecycle.py", "test_scaler_rows_snapshot_native.py", "test_robust_scaler_native_transport.py", "test_maxabs_scaler_native_transport.py"):
            scope.update(execution_route="selected package scaler backend",
                         operation_mode="identical (explicit fixture configuration)",
                         coverage_limit="Public scaler operations only; resident context-manager entry and exit remain separate runtime migration work")
        if name == "test_classical_archive_writer_native.py":
            scope.update(execution_route="native public linear/scaler archive writer on every package backend",
                         operation_mode="identical (explicit fixture configuration)",
                         coverage_limit="Selected public save bytes, native file errors and callback refusals only; private PCA archive entry and other model formats remain separate")
        if name == "test_classical_archive_reader_native.py":
            scope.update(execution_route="native host archive owner, including in GPU packages",
                         operation_mode="saved identical state; parsing does not select a GPU kernel",
                         coverage_limit="Selected public linear load paths, compressed state, native file errors and helper identities only; arbitrary name metadata, codec-hook edge cases and other archive formats remain separate")
        if name == "test_archive_unicode_native.py":
            scope.update(execution_route="native host archive reader/writer, including in GPU packages",
                         operation_mode="saved identical state and test-specific malformed Unicode; no GPU kernel",
                         coverage_limit="Selected HostLinearRegression.load strict ZIP UTF-8, NPY UTF-32 and literal errors, valid loaded state, StandardScaler.save surrogate errors/partial bytes and connected codec refusals only; Unicode-escape invalid/octal warning delivery may still reach Python warning display, and complete archive callback closure remains unqualified")
        if name == "test_host_density_save_native.py":
            scope.update(execution_route="native host density archive owner, including in GPU packages",
                         operation_mode="saved mode and configuration are test-specific; save does not train or select a GPU kernel",
                         coverage_limit="Selected public host KDE and IsolationForest save/load compatibility and native refusals only; fit/inference, other archive formats and cross-vendor reference/fault evidence remain separate")
        if name == "test_host_density_hash_native.py":
            scope.update(execution_route="native explicit CPU host model digest owner, including in GPU packages",
                         operation_mode="stored model-field bytes; hashing does not train or select a GPU kernel",
                         coverage_limit="Selected public HostKernelDensity.model_sha256 and HostIsolationForest.model_sha256 only; stored field ordering, Array physical bytes and native buffer logical C-order bytes are covered. Other estimators' hash methods, archive-file digests, fit/inference and cross-vendor reference/fault evidence remain separate")
        if name == "test_sequential_native_buffers.py":
            scope.update(execution_route="selected package Sequential fit/predict backend",
                         operation_mode="identical (explicit fixture configuration)",
                         coverage_limit="Selected public buffer preparation and consumed training/output state only; raw ABI calls are independent test oracles, not public qualification")
        if name == "test_training_linear_public_native.py":
            scope.update(execution_route="host buffers on the selected training backend; resident device owners on GPU",
                         operation_mode="identical (explicit fixture configuration)",
                         coverage_limit="Selected public linear operations and retained legacy contracts only; installed reference/fault and other training primitives remain separate")
        if name == "test_training_device_native_rollback.py":
            scope.update(execution_route="GPU device allocation, upload and resident optimizer parameter copy",
                         coverage_limit="Retained native failure paths only; native heap exhaustion is not injected")
        if name == "test_training_device_native_finalizer.py":
            scope.update(execution_route="training GPU owner cleanup; partial-owner and guard checks also apply on CPU",
                         coverage_limit="Explicit cleanup checks verify release; process shutdown only checks exit status and stderr")
        if name in ("test_host_inference_native.py", "test_host_kernel_inference_native.py", "test_host_projection_native.py", "test_host_logistic_native.py", "test_host_svr_native.py", "test_host_svc_native.py", "test_host_kde_native.py", "test_host_iforest_native.py", "test_host_kde_fit_native.py", "test_host_iforest_fit_native.py"):
            scope.update(execution_route="explicit CPU host binding, including in GPU packages",
                         operation_mode="identical host operation; fast-mode refusal tested separately",
                         coverage_limit="Native qualification covers audited prediction/transform, preparation and attributes; inherited fit/score cases check compatibility only")
        if name == "test_host_kde_native.py":
            scope["coverage_limit"] = "Native qualification covers score_samples/score and their state/input admission; inherited construction/fit/save/hash remain outside this migration slice"
        if name == "test_host_kde_fit_native.py":
            scope["coverage_limit"] = "Native qualification covers explicit host fit preparation and state publication; ordinary KernelDensity fit, constructors, save and hash remain separate"
        if name == "test_host_iforest_fit_native.py":
            scope["coverage_limit"] = "Native qualification covers explicit host fit preparation, seeded training and state publication; GPU IsolationForest fit, construction, save and hash remain separate"
        if name == "test_iforest_depth_native.py":
            scope.update(execution_route="explicit CPU host binding, including in GPU packages",
                         operation_mode="identical host operation",
                         coverage_limit="Public HostIsolationForest.fit depth-state arithmetic and publication only; Linux extended formats, GPU IsolationForest and other operations require separate execution evidence")
        if name == "test_host_density_constructors_native.py":
            scope.update(execution_route="native explicit CPU host constructor owner, including in GPU packages",
                         coverage_limit="Public KDE and IsolationForest construction/reinitialization only; fit/inference, save/hash and subclass registration remain separate")
        if name == "test_cache_key_native.py":
            scope.update(execution_route="native cache metadata only; no device slots",
                         coverage_limit="Selected DeviceCache.get and resident.stats tuple-key admission only; constructor, upload, release and context lifecycle remain separate")
        if name == "test_cache_supports_native.py":
            scope.update(execution_route="native DeviceCache.supports metadata on CPU and GPU; no device allocation or computation",
                         operation_mode="identical (explicit fixture configuration)",
                         coverage_limit="Only public capability lookup and its CPU refusal/prefix contracts; constructor, release, close and resident scope cleanup remain separate")
        if name == "test_cache_release_native.py":
            scope.update(execution_route="native DeviceCache constructor/release/close/exit and resident scope enter/exit on CPU/GPU; real x_prep/x_metrics store slots on GPU",
                         operation_mode="identical (explicit fixture configuration)",
                         coverage_limit="Only the DeviceCache and resident scope lifecycle; id_of, get, live and stats have their own scopes")
        if name == "test_resident_lifecycle_native.py":
            scope.update(execution_route="native model_selection resident lifecycle and take on every package backend, established identical base-binding store",
                         operation_mode="identical base-binding transport, independent of estimator numeric mode",
                         coverage_limit="Only the model-selection resident lifecycle and take; foreign owners (pandas, Arrow-backed pandas) keep the established input-preparation step")
        if name == "test_cache_lifecycle_native.py":
            scope.update(execution_route="native DeviceCache.live metadata and __enter__ on CPU/GPU; real x_prep/x_metrics store counters on GPU",
                         operation_mode="identical (explicit fixture configuration)",
                         coverage_limit="Only public live-slot queries and context entry; constructor, supports, release, close, context exit and resident scope cleanup remain separate migration work")
        if name == "test_cache_upload_native.py":
            scope.update(execution_route="native DeviceCache.id_of metadata on CPU and GPU; real x_prep/x_metrics word-store uploads on GPU",
                         operation_mode="identical (explicit fixture configuration)",
                         coverage_limit="Public id_of hit/miss preparation, publication and cleanup only; constructor, release, close, resident contexts and compound cleanup failure injection remain separate qualification work")
        if name == "test_cache_get_native.py":
            scope.update(execution_route="native host metadata on CPU and GPU packages; no device slots acquired",
                         coverage_limit="Only DeviceCache.get; construction/upload/release and resident context lifecycle remain separate migration work")
        if name == "test_cache_stats_native.py":
            scope.update(execution_route="native host metadata on CPU and GPU packages; no device slots acquired",
                         coverage_limit="Only DeviceCache.__len__ and resident.stats; get, construction/upload/release and resident context lifecycle remain separate migration work")
        if name == "test_device_rows_materialize_native.py":
            scope.update(execution_route="selected package backend, established identical base-binding resident store",
                         operation_mode="identical base-binding transport, independent of estimator numeric mode",
                         coverage_limit="Public DeviceRows.materialize only; _Resident construction/close and resident scope/cache orchestration remain separate migration work")
        if name == "test_optimizer_native_scalar_constructors.py":
            scope.update(execution_route="native optimizer preparation; device controls require a GPU training pool",
                         coverage_limit="Selected scalar constructor routes only; other public callbacks, package/reference/fault qualification and cross-vendor identity remain separate")
        if name == "test_late_numpy_native_metadata.py":
            scope.update(execution_route="native public DeviceRows constructor and StandardScaler fit/attribute admission on every package backend",
                         coverage_limit="Cold/late NumPy connection only; successful constructor output and refusal paths are checked in fresh subprocesses")
        if name == "test_device_rows_metadata_native.py":
            scope.update(execution_route="native DeviceRows metadata owner on every package backend",
                         coverage_limit="Constructor and length only; materialize and resident context lifecycle remain separate")
        if name == "test_native_owner_builtin_identity.py":
            scope.update(execution_route="native DeviceRows metadata owner on every package backend",
                         coverage_limit="Retained-owner release admission during public reinitialization only; cache allocation and resident close remain separate")
        if name == "test_optimizer_native_array_ownership.py":
            scope.update(execution_route="native training output allocation; device downloads only on GPU",
                         coverage_limit="Published Array buffer ownership only; public __array__ protocol and allocation failure injection remain separate")
        if name == "test_host_iforest_native.py":
            scope["coverage_limit"] = "Native qualification covers score_samples/decision_function/predict and saved state admission/readback; inherited construction/fit/save/hash remain outside this migration slice"
        if name == "test_training_rms_public_native.py":
            scope.update(execution_route="public training RMS operations through the selected CPU or GPU training binding",
                         coverage_limit="Exact preparation/output checks use retained numerical entries; independent primitive checks are separate. NumPy scalar-option coverage follows the installed NumPy version.")
        if name == "test_training_scalar_buffer_native.py":
            scope.update(execution_route="public RMS, embedding and cross entropy on selected host buffers; resident loss cases require GPU device owners",
                         operation_mode="identical (explicit fixture configuration)",
                         coverage_limit="Selected primitive NumPy numeric scalar and size-one array conversion only; consumed outputs reuse unchanged numerical kernels and rank/warning/error contracts follow the installed NumPy version. Text, void, object and StringDType branches, other numeric conversion sites, full-model training and cross-vendor identity/fault qualification remain separate")
        if name == "test_training_embedding_public_native.py":
            scope.update(execution_route="public embedding operations through the selected CPU or GPU training binding",
                         coverage_limit="Native preparation and retained kernel outputs only; NumPy scalar-option coverage follows the installed version, and full-model training remains separate")
        if name == "test_training_cross_entropy_public_native.py":
            scope.update(execution_route="public cross entropy on selected host buffers or GPU device owners",
                         coverage_limit="Native preparation and consumed outputs use retained numerical kernels; scalar warning contracts follow the installed NumPy version, and complete model training remains separate")
        if name == "test_training_device_native_allocator.py":
            scope.update(execution_route="GPU training owner allocation; layout admission also applies on CPU",
                         coverage_limit="Constructor callback and layout admission only; numerical behavior, rollback and allocation exhaustion are separate checks")
        if name == "test_optimizer_native_container_admission.py":
            scope.update(execution_route="native optimizer and training host-buffer preparation on the selected backend",
                         operation_mode="identical (explicit fixture configuration)",
                         coverage_limit="Selected container and scalar hook refusals plus consumed checkpoint state; other public operations and option combinations remain separate")
        if name == "test_optimizer_native_config_order.py":
            scope.update(execution_route="native optimizer configuration owner on the selected backend",
                         operation_mode="identical (explicit fixture configuration)",
                         coverage_limit="Public parameter ordering and selected option/reset behavior; unrelated training operations remain separate")
        result[name] = scope
    return result


def test_dependencies(selected, extra=()):
    # Independent test oracles/data preparation, never product dependencies.
    return list(dict.fromkeys(["numpy", "pytest"] + [dependency for name in selected
                          for dependency in TEST_DEPENDENCIES.get(name, ())] + list(extra)))


def require(condition, why):
    if not condition:
        raise ValueError(why)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def wheel_inventory(wheels):
    """Verify every wheel RECORD row independently before installing anything."""
    payload, records = {}, []
    for path in wheels:
        path = Path(path).resolve()
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            require(len(names) == len(set(names)), "duplicate ZIP member")
            files = {name for name in names if not name.endswith("/")}
            for name in names:
                parts = PurePosixPath(name).parts
                require(not name.startswith("/") and "\\" not in name
                        and ".." not in parts, "unsafe ZIP path")
            record_names = [name for name in files if name.endswith(".dist-info/RECORD")]
            require(len(record_names) == 1, "wheel must have one RECORD")
            record_name = record_names[0]
            rows = list(csv.reader(io.StringIO(archive.read(record_name).decode())))
            require(all(len(row) == 3 for row in rows), "invalid RECORD row")
            require(len(rows) == len({row[0] for row in rows}), "duplicate RECORD row")
            require({row[0] for row in rows} == files, "RECORD does not cover every wheel file")
            for name, recorded, size in rows:
                raw = archive.read(name)
                sha = hashlib.sha256(raw).hexdigest()
                if name == record_name:
                    require(not recorded and not size, "RECORD self hash must be blank")
                else:
                    encoded = base64.urlsafe_b64encode(bytes.fromhex(sha)).decode().rstrip("=")
                    require(recorded == "sha256=" + encoded and size == str(len(raw)),
                            "wheel RECORD mismatch: " + name)
                require(name not in payload, "two wheels own " + name)
                payload[name] = dict(sha256=sha, bytes=len(raw), wheel=str(path))
            records.append(dict(path=str(path), sha256=digest(path), bytes=path.stat().st_size))
    require("mojolearn/__init__.py" in payload, "core wheel missing")
    return payload, records


def source_match(source, commit, wheels, payload, git_root=None):
    git_root = Path(git_root or source)
    subprocess.run(["git", "-C", str(git_root), "cat-file", "-e", commit + "^{commit}"],
                   check=True, capture_output=True)
    witness = "mojolearn/identity_columns/COMMIT"
    require(witness in payload, "wheel source witness missing")
    with zipfile.ZipFile(payload[witness]["wheel"]) as archive:
        require(archive.read(witness).decode().strip() == commit, "wheel source witness differs")
    matched = {}
    for member, row in payload.items():
        path = source / "python" / member
        if member.startswith("mojolearn/") and member.endswith(".py") and path.is_file():
            require(digest(path) == row["sha256"], "wheel/source Python mismatch: " + member)
            blob = subprocess.check_output(["git", "-C", str(git_root), "show", commit + ":python/" + member])
            require(hashlib.sha256(blob).hexdigest() == row["sha256"], "uncommitted package source: " + member)
            matched[member] = row["sha256"]
    return matched


def _export_ignore_attributes(git_root, commit, paths):
    """`git check-attr -z export-ignore` of `paths` as `commit`'s own
    .gitattributes decide them (raw NUL-separated response). `--source` needs
    git 2.40; an older git refuses the option (status 129), so the same
    question is asked of a temporary index read from `commit`, which
    `--cached` consults and nothing else (the checkout's index and worktree
    are never read or written)."""
    stdin = b"\0".join(path.encode() for path in paths) + b"\0"
    try:
        return subprocess.check_output(
            ["git", "-C", str(git_root), "check-attr", "--source=" + commit,
             "-z", "--stdin", "export-ignore"], input=stdin, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as error:
        if error.returncode != 129:
            raise
    with tempfile.TemporaryDirectory() as scratch:
        env = dict(os.environ, GIT_INDEX_FILE=os.path.join(scratch, "index"))
        subprocess.check_output(["git", "-C", str(git_root), "read-tree", commit], env=env)
        return subprocess.check_output(
            ["git", "-C", str(git_root), "check-attr", "--cached", "-z", "--stdin", "export-ignore"],
            input=stdin, env=env)


def native_source_inventory(source, commit, committed, git_root):
    """Check frozen native bytes, recording archive-omitted historical evidence.

    A git archive intentionally excludes some old benchmark source snapshots.
    Only bench/results evidence can be omitted here, and only when the pinned
    commit's export-ignore attribute excludes that path or a parent directory.
    Missing product sources still refuse, including export-ignored products.
    """
    selected = {rel: blob for rel, blob in committed.items()
                if rel.endswith((".mojo", ".mojopkg")) or rel in ("pixi.toml", "pixi.lock")}
    missing = [rel for rel in selected if not (source / rel).exists()]
    for rel in missing:
        require(rel.startswith("bench/results/") and not (source / rel).is_symlink(),
                "missing requested native source: " + rel)
    ancestors = {str(parent) for rel in missing
                 for parent in (PurePosixPath(rel), *PurePosixPath(rel).parents)
                 if str(parent) != "."}
    attributes = {}
    if ancestors:
        # Check ancestors too: export-ignore on a directory excludes descendants
        # from git archive although check-attr on the child can say unspecified.
        raw = _export_ignore_attributes(git_root, commit, sorted(ancestors))
        fields = raw.split(b"\0")[:-1]
        require(len(fields) % 3 == 0, "invalid git export attribute response")
        attributes = {fields[i].decode(): fields[i + 2].decode()
                      for i in range(0, len(fields), 3)}
    hashes, omitted = {}, {}
    for rel, blob in selected.items():
        path = source / rel
        if rel in missing:
            excluded_by = next((str(parent) for parent in
                                (PurePosixPath(rel), *PurePosixPath(rel).parents)
                                if attributes.get(str(parent)) == "set"), None)
            require(excluded_by is not None, "missing non-export-ignored native source: " + rel)
            omitted[rel] = dict(git_blob=blob, export_ignore_path=excluded_by,
                                reason="historical benchmark evidence omitted by committed git archive policy")
            continue
        raw = path.read_bytes()
        require(hashlib.sha1(b"blob %d\0" % len(raw) + raw).hexdigest() == blob,
                "uncommitted requested native source: " + rel)
        hashes[rel] = hashlib.sha256(raw).hexdigest()
    return hashes, omitted


def installed_module_inventory(modules, prefix, payload, tests, native_artifacts=()):
    """Tests may live externally; every product Python/extension byte must ship."""
    prefix, tests = Path(prefix).resolve(), Path(tests).resolve()
    found = {}
    candidates = dict(modules)
    observed = {}

    def fingerprint(path):
        stat = path.stat()
        return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)

    def observed_digest(path):
        # A native extension may appear through many module aliases. Hash its
        # actual bytes once in this observation, preserving every alias/path/
        # wheel/witness check. No hash survives into the next test teardown.
        # Require a stable filesystem snapshot, including ctime and inode,
        # rather than trusting a restored size or mtime after a mutation.
        path = Path(path).resolve()
        before = fingerprint(path)
        if path in observed:
            prior, value = observed[path]
            require(before == prior, "installed product changed during inventory: " + str(path))
            return value
        value = digest(path)
        require(fingerprint(path) == before, "installed product changed while hashing: " + str(path))
        observed[path] = (before, value)
        return value
    backend = modules.get("mojolearn._backend")
    placeholders = tuple(value for key in ("_HostBinding", "_NoGpuBinding", "_MissingUpperTier", "_IdenticalOnlyTier")
                         if isinstance(value := vars(backend).get(key) if backend else None, type))

    def admit(path, *, native=False, module=None, path_source=None):
        path = Path(path).resolve()
        require(path.is_relative_to(prefix) and "site-packages" in path.parts,
                "product imported outside installed environment: " + str(path))
        at = path.parts.index("site-packages")
        member = PurePosixPath(*path.parts[at + 1:]).as_posix()
        require(member in payload, "loaded product file absent from wheel: " + member)
        require(observed_digest(path) == payload[member]["sha256"], "installed product bytes differ: " + member)
        row = dict(path=str(path), sha256=payload[member]["sha256"], native=native)
        if module:
            row.update(module=module, path_source=path_source)
        found[member] = row

    # The worker supplies the installed verifier's shared loaded-native-v2
    # inventory. Apply the additional installed-prefix and wheel constraints
    # here; a source checkout or an unshipped extension never qualifies.
    for row in native_artifacts:
        require(row.get("module") and row.get("file") and row.get("sha256") and row.get("size", 0) > 0,
                "incomplete loaded native witness")
        path = Path(row["file"])
        require(path.suffix in (".so", ".dylib", ".pyd"), "native witness is not an extension")
        require(path.stat().st_size == row["size"] and observed_digest(path) == row["sha256"],
                "loaded native witness bytes changed")
        admit(path, native=True, module=row["module"], path_source=row.get("path_source"))
    for name, module in list(modules.items()):
        if name.startswith("mojolearn") and isinstance(module, types.ModuleType):
            for key, value in vars(module).items():
                if isinstance(value, types.ModuleType):
                    candidates.setdefault(name + ":" + key, value)
    for name, module in candidates.items():
        if not isinstance(module, types.ModuleType):
            continue
        fields = vars(module)
        actual_name = fields.get("__name__", name)
        file_path, origin = fields.get("__file__"), getattr(fields.get("__spec__"), "origin", None)
        paths = {Path(value).resolve() for value in (file_path, origin)
                 if value and value not in ("built-in", "frozen")}
        product = actual_name.startswith(("mojolearn", "_mojolearn"))
        if not paths:
            # Native extensions cached only on a package attribute also need
            # a resolvable origin. Namespace modules and exact backend routing
            # placeholders carry no executed native artifact.
            require(type(module) in placeholders or not actual_name.rsplit(".", 1)[-1].startswith("_mojolearn"),
                    "loaded native module has no artifact path: " + actual_name)
            continue
        if not product and not any(p.name.startswith("_mojolearn") for p in paths):
            continue
        require(len(paths) == 1, "conflicting loaded product paths: " + actual_name)
        path = paths.pop()
        if actual_name.startswith("mojolearn.tests"):
            require(path.is_relative_to(tests), "test namespace escaped frozen test copy")
            continue
        admit(path, native=path.suffix in (".so", ".dylib", ".pyd"), module=actual_name,
              path_source="__file__" if file_path else "__spec__.origin")
    require("mojolearn/__init__.py" in found, "installed package was not observed")
    for path, (before, _) in observed.items():
        require(fingerprint(path) == before, "installed product changed during inventory: " + str(path))
    return found


def junit_summary(path):
    root = ET.parse(path).getroot()
    cases = list(root.iter("testcase"))
    failed = [dict(name=c.get("name"), classname=c.get("classname"),
                   reason="\n".join((x.text or x.get("message", "")) for x in list(c)
                                    if x.tag in ("failure", "error")))
              for c in cases if c.find("failure") is not None or c.find("error") is not None]
    skipped = [dict(name=c.get("name"), classname=c.get("classname"), reason=c.find("skipped").get("message", ""))
               for c in cases if c.find("skipped") is not None]
    return dict(total=len(cases), passed=len(cases)-len(failed)-len(skipped),
                failures=failed, skipped=skipped,
                cases=[dict(name=c.get("name"), classname=c.get("classname"),
                            exact_node=next((p.get("value", "") for p in c.findall("properties/property")
                                             if p.get("name") == "installed_exact_node"), ""),
                            outcome="failed" if c.find("failure") is not None or c.find("error") is not None
                            else "skipped" if c.find("skipped") is not None else "passed",
                            skip_reason=c.find("skipped").get("message", "") if c.find("skipped") is not None else "")
                       for c in cases])


def selected_files(test_files=None, test_nodes=None):
    """Exact nodes supplement retained runs without claiming whole-file coverage."""
    require(not (test_files and test_nodes), "select whole files OR exact nodes")
    if test_nodes is None:
        return test_files or list(PUBLIC_TESTS)
    require(bool(test_nodes), "exact node selection must not be empty")
    require(len(set(test_nodes)) == len(test_nodes), "duplicate exact test node")
    names = []
    for node in test_nodes:
        name, separator, case = node.partition("::")
        require(name in PUBLIC_TESTS and separator and case and
                not any(ord(c) < 32 for c in node), "invalid exact test node: " + node)
        if name not in names:
            names.append(name)
    return names


def exact_node_summary(expected, observed):
    return dict(expected_count=len(expected), observed_count=len(observed),
                missing=sorted(set(expected) - set(observed)),
                unexpected=sorted(set(observed) - set(expected)),
                duplicates=sorted({node for node in observed if observed.count(node) > 1}),
                complete=len(expected) == len(observed) and set(expected) == set(observed)
                and len(set(observed)) == len(observed))


class ExactNodeGuard:
    def __init__(self, tests, expected):
        self.tests, self.expected = tests, expected
        self.collected = []

    def pytest_collection_finish(self, session):
        # Check before any product call. A function-wide selector that expands
        # to several parameters is not an exact node, even if all cases pass.
        self.collected = [Path(item.path).resolve().relative_to(self.tests.resolve()).as_posix()
                          + "::" + item.nodeid.partition("::")[2] for item in session.items]
        require(exact_node_summary(self.expected, self.collected)["complete"],
                "collected tests differ from requested exact nodes")
        for item, node in zip(session.items, self.collected):
            item.user_properties.append(("installed_exact_node", node))


def test_scope_summary(summary, selected, vendor, exact_nodes=None):
    """Only exact CPU resident cases are N/A; all other skips remain incomplete."""
    scopes = {name: dict(total=0, passed=0, failures=0, not_applicable=[], unexpected_skips=[])
              for name in selected}
    unmatched = []
    for case in summary.get("cases", []):
        module_words = (case.get("classname") or "").split(".")
        matches = [name for name in selected if Path(name).stem in module_words]
        if len(matches) != 1:
            unmatched.append(case)
            continue
        name = matches[0]
        scope = scopes[name]
        scope["total"] += 1
        if case["outcome"] == "passed":
            scope["passed"] += 1
        elif case["outcome"] == "failed":
            scope["failures"] += 1
        else:
            declared = CPU_NOT_APPLICABLE.get(name, {}).get(case["name"]) if vendor == "cpu" else None
            key = "not_applicable" if declared is not None and case["skip_reason"] == declared else "unexpected_skips"
            scope[key].append(case)
    for scope in scopes.values():
        scope["status"] = ("failed-or-incomplete" if not scope["total"] or scope["failures"] or scope["unexpected_skips"]
                           else "passed" if scope["passed"] else "not-applicable")
    result = dict(scopes=scopes, unmatched_cases=unmatched,
                applicable_passed=sum(row["passed"] for row in scopes.values()),
                not_applicable_count=sum(len(row["not_applicable"]) for row in scopes.values()),
                complete=bool(scopes) and not unmatched and all(row["status"] != "failed-or-incomplete"
                                                               for row in scopes.values()))
    if exact_nodes is not None:
        result["exact_nodes"] = exact_node_summary(exact_nodes, [c.get("exact_node", "")
                                                               for c in summary.get("cases", [])])
        result["complete"] = result["complete"] and result["exact_nodes"]["complete"]
    return result


def selected_test_status(pytest_exit, summary, scope, receipt_exists):
    if pytest_exit != 0 or not scope["complete"] or summary.get("failures") or not receipt_exists:
        return "failed-or-incomplete"
    if scope["applicable_passed"] > 0:
        return "selected-tests-passed"
    if scope["not_applicable_count"] > 0:
        return "selected-tests-not-applicable"
    return "failed-or-incomplete"


def test_environment(parent, vendor, numeric_mode):
    env = {k: v for k, v in parent.items() if k not in ("PYTHONPATH", "PYTHONHOME")
           and not k.startswith("MOJOLEARN_")}
    # These test fixtures pass an explicit device to CausalLM.load. Their CPU
    # default overrides MOJOLEARN_VENDOR unless this testing knob is also set.
    env.update(PYTHONNOUSERSITE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
               MOJOLEARN_NUMERIC_MODE=numeric_mode, MOJOLEARN_VENDOR=vendor,
               MOJOLEARN_GENERATION_DEVICE="cpu" if vendor == "cpu" else "gpu")
    return env


def generation_test_devices(modules, expected):
    found = {}
    for name, module in list(modules.items()):
        if not isinstance(module, types.ModuleType):
            continue
        path = vars(module).get("__file__", "")
        if path and Path(path).name in ("test_generation_native.py", "test_causal_model_native_loading.py"):
            actual = vars(module).get("DEVICE")
            require(actual == expected, "generation fixture selected a different device: " + name)
            found[name] = actual
    return found


def worker(request_path):
    request = json.loads(Path(request_path).read_text())
    out, tests = Path(request["out"]), Path(request["tests"])
    require(Path.cwd() == out / "work", "test worker must run outside the source checkout")
    import mojolearn
    import pytest
    from mojolearn._verify import binding_artifacts
    require(mojolearn.vendor() == request["vendor"], "installed backend silently changed")
    from mojolearn import _backend
    require(_backend.default_mode() == request["numeric_mode"], "installed numeric mode differs")
    require(os.environ.get("MOJOLEARN_GENERATION_DEVICE") == request["generation_device"],
            "generation test device override differs")
    namespace = types.ModuleType("mojolearn.tests")
    namespace.__path__ = [str(tests)]
    namespace.__package__ = "mojolearn.tests"
    sys.modules["mojolearn.tests"] = namespace
    mojolearn.tests = namespace
    loaded, test_devices, native, inventory_errors = {}, {}, {}, []

    def observe(phase, *, require_loaded=False):
        try:
            # A lazy package can have no native modules until a test runs.
            # Final qualification always requires actual loaded native bytes.
            artifacts = binding_artifacts(require_loaded=require_loaded)
            require(not require_loaded or artifacts, "no loaded native binding witnesses")
            loaded.update(installed_module_inventory(sys.modules, sys.prefix, request["payload"], tests, artifacts))
            native.update({row["module"]: row for row in artifacts})
            test_devices.update(generation_test_devices(sys.modules, request["generation_device"]))
            return True
        except Exception as error:
            inventory_errors.append(dict(phase=phase, error=type(error).__name__ + ": " + str(error)))
            return False

    class InstalledGuard:
        def pytest_runtest_teardown(self, item, nextitem):
            require(observe("test-teardown"), "installed product inventory refused during test teardown")

    nodes = request.get("test_nodes")
    selected_files(request["test_files"] if nodes is None else None, nodes)
    exact_guard = ExactNodeGuard(tests, nodes) if nodes is not None else None
    code, worker_error = None, None
    try:
        require(observe("initial"), "initial installed product inventory refused")
        code = int(pytest.main(["--import-mode=importlib", "-q", "-ra", "--confcutdir=" + str(tests),
                               "--junitxml=" + str(out / "tests.xml"),
                               "--basetemp=" + str(out / "test-tmp")]
                              + [str(tests / name) for name in (nodes or request["test_files"])],
                              plugins=[InstalledGuard()] + ([exact_guard] if exact_guard else [])))
    except Exception as error:
        worker_error = type(error).__name__ + ": " + str(error)
    finally:
        observe("final", require_loaded=True)
        (out / "installed-modules.json").write_text(json.dumps(dict(
            package=str(Path(mojolearn.__file__).resolve()), prefix=sys.prefix,
            vendor=mojolearn.vendor(), numeric_mode=_backend.default_mode(), modules=loaded,
            binding_provenance_protocol="loaded-native-v2", bindings=list(native.values()),
            generation_test_device=request["generation_device"],
            observed_generation_test_devices=test_devices, pytest_exit=code,
            inventory_errors=inventory_errors, worker_error=worker_error,
            exact_node_collection=exact_node_summary(nodes, exact_guard.collected)
            if exact_guard else None), indent=2) + "\n")
    return code if code is not None and not inventory_errors and worker_error is None else 1


def run_command(name, command, out, *, cwd, env=None, timeout=900):
    with (out / (name + ".log")).open("wb") as log:
        result = subprocess.run(command, cwd=cwd, env=env, stdout=log,
                                stderr=subprocess.STDOUT, timeout=timeout)
    require(result.returncode == 0, name + " failed; see retained log")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--git-root", type=Path, help="authoritative object database for an archived frozen source tree")
    parser.add_argument("--wheel", type=Path, action="append", default=[])
    parser.add_argument("--installed-site", type=Path)
    parser.add_argument("--native-overlay", type=Path)
    parser.add_argument("--native-proof", type=Path, action="append", default=[])
    parser.add_argument("--vendor", choices=("cpu", "metal", "cuda", "hip"), required=True)
    parser.add_argument("--numeric-mode", choices=("identical", "fast"), default="identical")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--test-file", choices=tuple(PUBLIC_TESTS), action="append")
    selection.add_argument("--test-node", action="append",
                           help="Exact registered filename::test[parameter] node; repeat for each case")
    parser.add_argument("--dependency", action="append", default=[])
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    selected = selected_files(args.test_file, args.test_node)
    source, out = args.source_root.resolve(), args.out.resolve()
    git_root = (args.git_root or source).resolve()
    require(not out.exists() and not out.is_relative_to(source), "output must be new and outside source")
    require(bool(args.wheel) != bool(args.installed_site), "supply wheels OR installed-site for Linux assembly")
    require(args.numeric_mode != "fast" or args.vendor == "metal", "fast is Apple only")
    out.mkdir(parents=True)
    report = dict(schema="mojolearn.installed-public-native-audit.v1", source_commit=args.source_commit,
                  vendor=args.vendor, numeric_mode=args.numeric_mode, release_qualified=False,
                  coverage_complete=False, native_source_reuse_qualified=False,
                  scope="Selected public operations only; no private-owner or whole-library qualification",
                  source_root=str(source), controller_sha256=digest(__file__), status="running")
    try:
        wheels = args.wheel
        if wheels:
            retained = out / "input-wheels"; retained.mkdir()
            require(len({p.name for p in wheels}) == len(wheels), "duplicate supplied wheel filename")
            for path in wheels:
                shutil.copyfile(path, retained / path.name)
            wheels = [retained / path.name for path in wheels]
        if args.installed_site:
            require(sys.platform.startswith("linux"), "qualification assembly is Linux only; supply existing Mac wheels")
            command = [args.python, str(source / "packaging/linux/qualification_wheels.py"),
                       "--source-root", str(source), "--source-commit", args.source_commit,
                       "--installed-site", str(args.installed_site.resolve()), "--out", str(out / "wheels")]
            if args.native_overlay:
                command += ["--native-overlay", str(args.native_overlay.resolve())]
            run_command("assemble", command, out, cwd=out)
            manifest = json.loads((out / "wheels/qualification-inputs.json").read_text())
            report["assembly_manifest"] = dict(path=str(out / "wheels/qualification-inputs.json"),
                                               sha256=digest(out / "wheels/qualification-inputs.json"))
            wheels = [Path(row["path"]) for row in manifest["wheels"]]
        payload, report["wheels"] = wheel_inventory(wheels)
        report["package_source_hashes"] = source_match(source, args.source_commit, wheels, payload, git_root)
        report["native_assets"] = {name:row for name,row in payload.items()
                                    if name.startswith("mojolearn/") and name.endswith((".so", ".dylib"))}
        report["supplied_native_proofs"] = [dict(path=str(p.resolve()), sha256=digest(p)) for p in args.native_proof]
        report["native_proof_limit"] = "Proof files are retained, not interpreted as successful build or source equivalence. Review them alongside loaded native hashes."
        proofdir = out / "native-proofs"; proofdir.mkdir()
        for i, path in enumerate(args.native_proof):
            shutil.copyfile(path, proofdir / (str(i) + "-" + path.name))
        tests = out / "frozen-tests"
        shutil.copytree(source / "python/mojolearn/tests", tests,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        tree = subprocess.check_output(["git", "-C", str(git_root), "ls-tree", "-r", "-z", args.source_commit])
        committed = {}
        for row in tree.split(b"\0"):
            if row:
                metadata, name = row.split(b"\t", 1)
                committed[name.decode()] = metadata.split()[2].decode()
        (report["requested_native_source_hashes"],
         report["archive_omitted_historical_native_sources"]) = native_source_inventory(
            source, args.source_commit, committed, git_root)
        report["test_hashes"] = {}
        for path in tests.rglob("*"):
            if path.is_file():
                rel = path.relative_to(tests).as_posix(); raw = path.read_bytes()
                blob = hashlib.sha1(b"blob %d\0" % len(raw) + raw).hexdigest()
                require(committed.get("python/mojolearn/tests/" + rel) == blob,
                        "uncommitted frozen test: " + rel)
                report["test_hashes"][rel] = hashlib.sha256(raw).hexdigest()
        report["test_selection_kind"] = "exact-nodes" if args.test_node else "whole-files"
        report["selected_test_nodes"] = args.test_node
        report["selected_public_scope"] = {name:PUBLIC_TESTS[name] for name in selected}
        report["selected_test_applicability"] = test_applicability(selected, args.vendor)
        report["test_dependencies"] = test_dependencies(selected, args.dependency)
        venv = out / "venv"
        run_command("venv", [args.python, "-m", "venv", str(venv)], out, cwd=out)
        python = venv / "bin/python"
        env = test_environment(os.environ, args.vendor, args.numeric_mode)
        report["test_device_overrides"] = {"MOJOLEARN_GENERATION_DEVICE": env["MOJOLEARN_GENERATION_DEVICE"]}
        run_command("install-wheels", [str(python), "-m", "pip", "install", "--no-deps", "--no-cache-dir"]
                    + [str(Path(p).resolve()) for p in wheels], out, cwd=out, env=env)
        run_command("install-test-dependencies", [str(python), "-m", "pip", "install"]
                    + report["test_dependencies"], out, cwd=out, env=env)
        run_command("pip-freeze", [str(python), "-m", "pip", "freeze", "--all"], out, cwd=out, env=env)
        shutil.copyfile(__file__, out / "controller.py")
        request = dict(out=str(out), tests=str(tests), payload=payload, test_files=selected,
                       test_nodes=args.test_node,
                       vendor=args.vendor, numeric_mode=args.numeric_mode,
                       generation_device=env["MOJOLEARN_GENERATION_DEVICE"])
        (out / "request.json").write_text(json.dumps(request, indent=2) + "\n")
        (out / "work").mkdir()
        with (out / "tests.log").open("wb") as log:
            result = subprocess.run([str(python), str(out / "controller.py"), "--worker", str(out / "request.json")],
                                    cwd=out / "work", env=env, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=args.timeout)
        report["worker_exit"] = result.returncode
        receipt_path = out / "installed-modules.json"
        receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else {}
        report["pytest_exit"] = receipt.get("pytest_exit")
        report["installed_guard_summary"] = dict(binding_count=len(receipt.get("bindings", [])),
                                                  inventory_errors=receipt.get("inventory_errors", []),
                                                  worker_error=receipt.get("worker_error"))
        if (out / "tests.xml").exists():
            report["tests"] = junit_summary(out / "tests.xml")
        report["installed_guard_receipt"] = str(out / "installed-modules.json")
        tests_summary = report.get("tests", {})
        report["test_scope_summary"] = test_scope_summary(tests_summary, selected, args.vendor, args.test_node)
        scope = report["test_scope_summary"]
        report["status"] = selected_test_status(result.returncode, tests_summary, scope,
                                               (out / "installed-modules.json").is_file())
    except Exception as error:
        report["status"] = "failed-or-incomplete"
        report["error"] = type(error).__name__ + ": " + str(error)
    finally:
        (out / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k:report.get(k) for k in ("status", "pytest_exit", "release_qualified")}))
    # An entirely inapplicable selection records honest N/A, never a green
    # qualification with zero applicable public operations.
    return 0 if report["status"] == "selected-tests-passed" else 1


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        raise SystemExit(worker(sys.argv[2]))
    raise SystemExit(main())
