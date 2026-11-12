# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU binding for `_mojolearn_x_decomp` (lane/algos-decomp, 2026-09-27).
HOST ONLY: the entry points of x_decomp/api.mojo on the host executor
(x_decomp/host.mojo), under the GPU binding's names and address contract,
plus the family read-backs."""
from x_decomp.publish_host_run import _publish_linear_entry_table
from core.experiment_guards_decomposition import DECOMPOSITION_GUARDS_OK
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder

from checks.kernel_matrix import COLUMN_CPU, TARGET_COLUMN, column_name
from checks.numerics import GLOBAL_NUMERIC_MODE
from providers.python.x_decomp_api_pyapi import (
    colsum_py, eigh_py, ew_py, gemm_py, lu_py, lu_solve_py, numeric_mode_py, orth_py, orth_diag_py, rand_py, svd_py, lasso_rows_py, lars_rows_py, lu_aux_py, omp_rows_py, absmax_sign_py, geqrf_py, orgqr_py, rank_above_py, move_py, select_smallest_py, argmin_all_py, pca_mle_rank_host_py, rowsum_py, sqdist_py, vendor_py, idn_flags_py, ols_tsqr_r_py,
)
from x_decomp.host import HostExec, X_DECOMP_HOST_SABOTAGE
from x_decomp.pca_randomized import NoRDev
from providers.python.x_decomp_tsvd_owner_pyapi import tsvd_randomized_py
from x_decomp.ipca_owner import NoIpcaDev
from providers.python.x_decomp_ipca_owner_pyapi import ipca_fit_py, ipca_partial_py, ipca_project_py
from x_decomp.decomp_owner_host import HostDecomp
from providers.python.x_decomp_decomp_owner_pyapi import ica_fit_py, ica_project_py, nmf_fit_py, nmf_inverse_py, fa_fit_py, fa_query_py, lda_py, rp_fit_py, rp_project_py
from providers.python.x_decomp_select_ops_pyapi import order_small_py, reduce_py


def x_decomp_host_numeric_mode_binding() raises -> PythonObject:
    return PythonObject(GLOBAL_NUMERIC_MODE)


def x_decomp_host_vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


def x_decomp_host_column_binding() raises -> PythonObject:
    comptime assert TARGET_COLUMN == COLUMN_CPU, "x_decomp host: pass -D MOJOLEARN_COLUMN_CPU"
    return PythonObject(column_name(TARGET_COLUMN))


def x_decomp_host_sabotage_binding() raises -> PythonObject:
    return PythonObject(X_DECOMP_HOST_SABOTAGE)


from x_decomp.descriptor_host import X_DECOMP_HOST_API
from providers.python.neural_descriptor_pyapi import neural_descriptor_capsule


from core.shared_slot import SharedSlot
from core.tsvd_native_decomp import TSVDDecompAPI
from core.nmf_native_provider import NMFProviderAPI
from core.ipca_native_provider import IPCAProviderAPI
from core.kpca_native_api import KPCAProviderAPI
from core.kpca_native_provider import KPCAEigenKernels
from checks.numerics import NUMERIC_IDENTICAL

comptime _TSVD_DECOMP_API_NAME = "MojolearnTSVDDecompHostIdenticalV1" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnTSVDDecompHostFastV1"
comptime TSVD_DECOMP_API = SharedSlot[TSVDDecompAPI[HostExec, NoRDev, False], _TSVD_DECOMP_API_NAME]


comptime _NMF_API_NAME = "MojolearnNMFHostIdenticalV1" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnNMFHostFastV1"
comptime NMF_API = SharedSlot[NMFProviderAPI[HostExec, NoRDev, False, HostDecomp], _NMF_API_NAME]


comptime _IPCA_API_NAME = "MojolearnIPCAHostIdenticalV1" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnIPCAHostFastV1"
comptime IPCA_API = SharedSlot[IPCAProviderAPI[HostExec, NoRDev, False, NoIpcaDev], _IPCA_API_NAME]


comptime _KPCA_DECOMP_API_NAME = "MojolearnKPCADecompHostIdenticalV1" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnKPCADecompHostFastV1"
comptime KPCA_DECOMP_API = SharedSlot[KPCAProviderAPI[KPCAEigenKernels[HostExec, NoRDev, False, HostDecomp]], _KPCA_DECOMP_API_NAME]


def kpca_decomp_native_api_v1_binding() raises -> PythonObject:
    var api = KPCA_DECOMP_API.get_or_create_ptr()
    return neural_descriptor_capsule(api[].name, api[].entries)


def ipca_native_api_v1_binding() raises -> PythonObject:
    var api = IPCA_API.get_or_create_ptr()
    return neural_descriptor_capsule(api[].name, api[].entries)


def nmf_native_api_v1_binding() raises -> PythonObject:
    var api = NMF_API.get_or_create_ptr()
    return neural_descriptor_capsule(api[].name, api[].entries)


def tsvd_decomp_native_api_v1_binding() raises -> PythonObject:
    var api = TSVD_DECOMP_API.get_or_create_ptr()
    return neural_descriptor_capsule(api[].name, api[].entries)


def x_decomp_native_api_v1_binding() raises -> PythonObject:
    """The x_decomp kernel-call table of the host image (host-address
    operations only; x_decomp/descriptor_host.mojo)."""
    var api = X_DECOMP_HOST_API.get_or_create_ptr()
    return neural_descriptor_capsule(api[].name, api[].entries)


# The module body. Its @export PyInit lives in bindings/entry/entry__mojolearn_x_decomp_host.mojo
# (the per-binding build) and in the merged roots tools/merged_bindings.py
# writes from bindings/merged_groups.json (one shared file per directory set).

def build__mojolearn_x_decomp_host() -> PythonObject:
    comptime assert DECOMPOSITION_GUARDS_OK, "invalid integrated experiment configuration (decomposition guards)"
    try:
        var m = PythonModuleBuilder("_mojolearn_x_decomp_host")
        m.def_function[x_decomp_native_api_v1_binding]("x_decomp_native_api_v1")
        m.def_function[tsvd_decomp_native_api_v1_binding]("tsvd_decomp_native_api_v1")
        m.def_function[nmf_native_api_v1_binding]("nmf_native_api_v1")
        m.def_function[ipca_native_api_v1_binding]("ipca_native_api_v1")
        m.def_function[kpca_decomp_native_api_v1_binding]("kpca_decomp_native_api_v1")
        m.def_function[x_decomp_host_numeric_mode_binding]("x_decomp_host_numeric_mode")
        m.def_function[x_decomp_host_vendor_binding]("x_decomp_host_vendor")
        m.def_function[x_decomp_host_column_binding]("x_decomp_host_column")
        m.def_function[x_decomp_host_sabotage_binding]("x_decomp_host_sabotage")
        m.def_function[gemm_py[HostExec]]("x_decomp_gemm")
        m.def_function[ew_py[HostExec]]("x_decomp_ew")
        m.def_function[colsum_py[HostExec]]("x_decomp_colsum")
        m.def_function[rowsum_py[HostExec]]("x_decomp_rowsum")
        m.def_function[sqdist_py[HostExec]]("x_decomp_sqdist")
        m.def_function[rand_py[HostExec]]("x_decomp_rand")
        m.def_function[lu_py[HostExec]]("x_decomp_lu")
        m.def_function[lu_solve_py[HostExec]]("x_decomp_lu_solve")
        # TOMBSTONE: the manifold kit's named x_decomp entries (graph_*, dev_graph_*, barycenter_rows, lle_apply, lle_local, lle_iterate, lle_*_ordered, mds_fit, lanczos, dev_lanczos, eigh_tridiag, dev_eigh_tridiag, trisolve, dev_trisolve, dsum_sq, dev_dsum_sq, order_f, dev_order_f; now x_decomp table operations, x_decomp/descriptor_mani_schema.mojo) deleted 2026-11-12 by lane entangled-mkit; code recoverable at e561a66d6b.
        # Restore: git apply experiments/removed/x_decomp_mani_named_entries.patch
        # TOMBSTONE: the dead x_decomp exports (registrations only where the body stays live in the decomp owners; see docs/TOMBSTONES.md) deleted 2026-10-26 by 25b5daca2; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_x_decomp.patch
        m.def_function[eigh_py[HostExec]]("x_decomp_eigh")
        # TOMBSTONE: the dead x_decomp exports (registrations only where the body stays live in the decomp owners; see docs/TOMBSTONES.md) deleted 2026-10-26 by 25b5daca2; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_x_decomp.patch
        # TOMBSTONE: the dead x_decomp exports (registrations only where the body stays live in the decomp owners; see docs/TOMBSTONES.md) deleted 2026-10-26 by 25b5daca2; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_x_decomp.patch
        m.def_function[orth_py[HostExec]]("x_decomp_orth")
        m.def_function[orth_diag_py[HostExec]]("x_decomp_orth_diag")
        m.def_function[svd_py[HostExec]]("x_decomp_svd")
        m.def_function[lasso_rows_py[HostExec]]("x_decomp_lasso_rows")
        m.def_function[omp_rows_py[HostExec]]("x_decomp_omp_rows")
        m.def_function[lars_rows_py[HostExec]]("x_decomp_lars_rows")
        m.def_function[lu_aux_py[HostExec]]("x_decomp_lu_aux")
        # TOMBSTONE: the dead x_decomp exports (registrations only where the body stays live in the decomp owners; see docs/TOMBSTONES.md) deleted 2026-10-26 by 25b5daca2; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_x_decomp.patch
        # TOMBSTONE: the dead x_decomp exports (registrations only where the body stays live in the decomp owners; see docs/TOMBSTONES.md) deleted 2026-10-26 by 25b5daca2; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_x_decomp.patch
        m.def_function[absmax_sign_py[HostExec]]("x_decomp_absmax_sign")
        # TOMBSTONE: the dead x_decomp exports (registrations only where the body stays live in the decomp owners; see docs/TOMBSTONES.md) deleted 2026-10-26 by 25b5daca2; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_x_decomp.patch
        m.def_function[geqrf_py[HostExec]]("x_decomp_geqrf")
        m.def_function[orgqr_py[HostExec]]("x_decomp_orgqr")
        # TruncatedSVD(algorithm='randomized').fit, one driver (x_decomp/tsvd_owner.mojo);
        # the TSVD_QFIX route is FAST GPU only (qfix_flags is 0 here)
        m.def_function[tsvd_randomized_py[HostExec, NoRDev, False]]("x_decomp_tsvd_randomized")
        # lane pyout-decomp-linalg (2026-11-11): x_decomp_als_fit, _topn_desc, _gather,
        # _dict_learning, _dict_minibatch, _mcd, _consistency_factor, _chi2_quantile,
        # _pls_fit, _lu_gesv and _randomized_svd are x_decomp table operations
        # (XD_ALS_FIT .. XD_RANDOMIZED_SVD, x_decomp/descriptor_owner_*.mojo); their
        # Python entries and `*_py` doors are deleted.
        # IncrementalPCA, one driver per operation (x_decomp/ipca_owner.mojo)
        m.def_function[ipca_fit_py[HostExec, NoRDev, False, NoIpcaDev]]("x_decomp_ipca_fit")
        m.def_function[ipca_partial_py[HostExec, NoRDev, False, NoIpcaDev]]("x_decomp_ipca_partial")
        m.def_function[ipca_project_py[HostExec, NoRDev, False]]("x_decomp_ipca_project")
        # FastICA (x_decomp/ica_owner.mojo)
        m.def_function[ica_fit_py[HostExec, NoRDev, False, HostDecomp]]("x_decomp_ica_fit")
        m.def_function[ica_project_py[HostExec, NoRDev, False]]("x_decomp_ica_project")
        m.def_function[nmf_fit_py[HostExec, NoRDev, False, HostDecomp]]("x_decomp_nmf_fit")
        m.def_function[nmf_inverse_py[HostExec, NoRDev, False]]("x_decomp_nmf_inverse")
        m.def_function[fa_fit_py[HostExec, NoRDev, False, HostDecomp]]("x_decomp_fa_fit")
        m.def_function[fa_query_py[HostExec, NoRDev, False, HostDecomp]]("x_decomp_fa_query")
        m.def_function[lda_py[HostExec, NoRDev, False, HostDecomp]]("x_decomp_lda_owner")
        m.def_function[rp_fit_py[HostExec, NoRDev, False, HostDecomp]]("x_decomp_rp_fit")
        m.def_function[rp_project_py[HostExec, NoRDev, False, HostDecomp]]("x_decomp_rp_project")
        m.def_function[rank_above_py]("x_decomp_rank_above")
        # lane idn-dense-linalg: the one-entry OLS and solve routes
        m.def_function[idn_flags_py]("x_decomp_idn_flags")
        # x_decomp_tsqr_r, x_decomp_tsqr_q and x_decomp_r_signs removed (lane entangled-decompkit,
        # 2026-11-12): their callers use the table ops XD_TSQR_R, XD_TSQR_Q and XD_R_SIGNS.
        m.def_function[ols_tsqr_r_py[HostExec]]("x_decomp_ols_tsqr_r")
        # TOMBSTONE: MOJOLEARN_IDN_OLS_R_TRISOLVE (loser: noise, ols taxi/istella avg ~0.98x, bits changed) deleted 2026-10-11 by c14beeda5; code recoverable at c14beeda5^.
        # Restore: git apply experiments/removed/MOJOLEARN_IDN_OLS_R_TRISOLVE.patch
        # TOMBSTONE: the dead x_decomp exports (registrations only where the body stays live in the decomp owners; see docs/TOMBSTONES.md) deleted 2026-10-26 by 25b5daca2; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_x_decomp.patch
        # TOMBSTONE: the dead x_decomp exports (registrations only where the body stays live in the decomp owners; see docs/TOMBSTONES.md) deleted 2026-10-26 by 25b5daca2; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_x_decomp.patch
        # TOMBSTONE: the dead x_decomp exports (registrations only where the body stays live in the decomp owners; see docs/TOMBSTONES.md) deleted 2026-10-26 by 25b5daca2; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_x_decomp.patch
        # TOMBSTONE: the dead x_decomp exports (registrations only where the body stays live in the decomp owners; see docs/TOMBSTONES.md) deleted 2026-10-26 by 25b5daca2; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_x_decomp.patch
        # TOMBSTONE: the dead x_decomp exports (registrations only where the body stays live in the decomp owners; see docs/TOMBSTONES.md) deleted 2026-10-26 by 25b5daca2; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_x_decomp.patch
        m.def_function[move_py]("x_decomp_move")
        m.def_function[reduce_py]("x_decomp_reduce")
        m.def_function[order_small_py]("x_decomp_order_small")
        m.def_function[select_smallest_py]("x_decomp_select_smallest")
        m.def_function[argmin_all_py]("x_decomp_argmin_all")
        # TOMBSTONE: the dead x_decomp exports (registrations only where the body stays live in the decomp owners; see docs/TOMBSTONES.md) deleted 2026-10-26 by 25b5daca2; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_x_decomp.patch
        m.def_function[pca_mle_rank_host_py]("x_decomp_pca_mle_rank")
        m.def_function[numeric_mode_py]("x_decomp_numeric_mode")
        m.def_function[vendor_py[HostExec]]("x_decomp_vendor")
        var published_module = m.finalize()
        _publish_linear_entry_table()
        return published_module
    except e:
        abort(String("failed to create _mojolearn_x_decomp_host: ", e))
