# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""RUN pieces of `bindings/_mojolearn_estimators_host.mojo`: the
preprocessing address reads, the typed bodies of the host TruncatedSVD
transform and inverse transform, and the native entry-table publications at
module init. Typed Mojo only, no Python import (tools/python_free_dirs.sh).
The door keeps the argument reads and releases the GIL around each body
exactly where it did before; the bodies are moved unchanged, so the bits are
unchanged (lane rs-metrics-estimators, 2026-11-12)."""
from core.linear_entry_table import LinearEntryKey, linear_table_publish, LINEAR_TABLE_ESTIMATORS
from core.linear_fit_types import LINEAR_FIT_ABI, LINEAR_FIT_OPERATION_COUNT
from core.linear_boundary_types import LINEAR_PREDICT_ABI
from glm.host.linear_fit_native import linear_fit_host_v1, LINEAR_FIT_HOST_API
from glm.host.linear_predict_native import linear_predict_host_v1
from glm.host.logistic_native_host import LOGISTIC_HOST_API, qn_host_native_v1
from core.estimators_entry_table import EstimatorsEntryKey, estimators_table_publish, ESTIMATORS_TABLE_QN
from core.estimators_host_impl import _positive
from core.classical_host_predict import host_inverse_transform_into, host_tsvd_transform_into
from core.host_finite_scan import host_finite_status
from core.host_predict_threads import host_predict_task_count
from core.hostptr import f32_ptr, read_f32
from checks.numerics import GLOBAL_NUMERIC_MODE


def _pp_load(addr: Int, n: Int) raises -> List[Float32]:
    if addr == 0:
        raise Error("preprocessing: null Float32 pointer")
    return read_f32(addr, max(0, n))


def _pp_out(addr: Int) raises -> MutPointer[Float32, MutUntrackedOrigin]:
    if addr == 0:
        raise Error("preprocessing: null Float32 pointer")
    return f32_ptr(addr)


def tsvd_transform_host_words(
    x_address: Int, c_address: Int, op: MutPointer[Float32, MutUntrackedOrigin],
    nr: Int, nf: Int, nc: Int, test: Bool,
) raises -> Int:
    # Called with the GIL released by the door (tsvd_transform_binding).
    var status = 0
    _positive(nr, "n_rows")
    _positive(nf, "n_features")
    _positive(nc, "n_components")
    if test:
        # parallel: the deferred finite test of X (SIMD scan across the host's cores)
        status = Int(host_finite_status(x_address, nr * nf))
    if status == 0:
        host_tsvd_transform_into(
            f32_ptr(x_address), f32_ptr(c_address), op, nr, nf, nc,
            host_predict_task_count(nr),
        )
    return status


def inverse_transform_host_words(
    z_address: Int, c_address: Int, m_address: Int, op: MutPointer[Float32, MutUntrackedOrigin],
    nr: Int, nf: Int, nc: Int, add_mean: Bool,
) raises:
    # Called with the GIL released by the door (inverse_transform_binding).
    _positive(nr, "n_rows")
    _positive(nf, "n_features")
    _positive(nc, "n_components")
    var components = read_f32(c_address, nc * nf)
    var mu = read_f32(m_address, nf) if add_mean else List[Float32]()
    host_inverse_transform_into(
        f32_ptr(z_address), components, mu, op, nr, nf, nc, add_mean,
        host_predict_task_count(nr),
    )


def _publish_linear_entry_table() raises:
    # Native publication only; the existing capsule ABI remains for clients
    # not yet converted. The original descriptor owner has image lifetime.
    var api = LINEAR_FIT_HOST_API.get_or_create_ptr()
    var key = LinearEntryKey(GLOBAL_NUMERIC_MODE, 0)
    linear_table_publish[LINEAR_TABLE_ESTIMATORS](key,
        Int(api[].entries.unsafe_ptr()), [String(LINEAR_FIT_ABI), String(LINEAR_PREDICT_ABI)], [LINEAR_FIT_OPERATION_COUNT, 1], [linear_fit_host_v1, linear_predict_host_v1])


def _publish_estimators_qn_table() raises:
    # The CPU column of the estimators QN table
    # (core/estimators_entry_table.mojo): QNRegressor's owner calls it
    # instead of the `qn_fit` / `qn_decision_function` Python exports, which
    # remain for the other doors and the Python tests.
    var api = LOGISTIC_HOST_API.get_or_create_ptr()
    estimators_table_publish[ESTIMATORS_TABLE_QN](EstimatorsEntryKey(Int(GLOBAL_NUMERIC_MODE), 0),
        Int(api[].entries.unsafe_ptr()), qn_host_native_v1)
