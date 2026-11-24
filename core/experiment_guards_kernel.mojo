# SPDX-License-Identifier: Apache-2.0
"""Opt-in strategy admission for the kernel bindings (SVM, kernel methods, Gaussian processes).

Each binding of this family imports KERNEL_GUARDS_OK and asserts it in its build function, so every compile of one of
those bindings (per-binding or merged) evaluates these guards. A guard whose define is read in code by bindings
of more than one family belongs in core/six_lane_experiment_guards.mojo, which every binding reaches through
checks/numerics.mojo.
"""
from std.sys.compile import is_defined, get_defined_int
# The entries below moved verbatim, in their original order, from core/six_lane_experiment_guards.mojo on
# 2026-11-04 (lane split-hot-groups; the placement of each entry is docs/build/guard_split.json). A header comment
# that headed lines of several families is carried with each of them.


def _check_configuration() -> Bool:
    # Deleted 2026-10-07 (lane serial-cleanup): forbidden serial shape; incumbent route is parallel. Recoverable at origin/integration/switches-20261007 608a7cf4a.
    comptime assert not is_defined["MOJOLEARN_CLASSICAL_C20_ROW_CACHE"](), "removed: C20_ROW_CACHE was a host loop of 3 launches per working-set row (one 1x1); forbidden serial shape, incumbent square tile is parallel"
    # Deleted 2026-10-08 (lane grid-losers-1): IDENTICAL grid ge123e6f9 losers (docs/apple-fast/EXPERIMENTS.md). Recoverable at main ad7ed2370.
    # C25_PROJECTION_REUSE: deleted 2026-10-08 above, restored 2026-10-10 (lane c25-restore-promote) from the tombstone patch after grid pass 2.
    comptime assert not is_defined["MOJOLEARN_CLASSICAL_C25_PROJECTION_REUSE"](), "promoted 2026-10-10 (grid ge123e6f9 + ge123e6f9r2, lane/c25-restore-promote): four-row projection reuse is the IDENTICAL default for rbf-sampler (geo-mean 0.903x, worst 0.983x); drop -D MOJOLEARN_CLASSICAL_C25_PROJECTION_REUSE, use -D MOJOLEARN_CLASSICAL_C25_PROJECTION_REUSE_OFF for the old path"
    # Lane grid-act-16 (2026-10-11), removed: grid g50ebe26a5 pass-1 losers against today's defaults, one run per arm (docs/TOMBSTONES.md,
    # docs/apple-fast/EXPERIMENTS.md, experiments/removed/*.patch). Deleted code recoverable at main 885c23dc7^ / the lane shas in TOMBSTONES.md.
    comptime assert not is_defined["MOJOLEARN_IDN_CHOL_NB128"](), "removed 2026-10-11 (lane/chol-nb128-delete, dd9a3d31b): IDN_CHOL_NB128 (128-wide blocked Cholesky) was SLOWER: cholesky synthetic NV 0.92x / AMD 2.29x (avg 1.6x, vendor split), v0005/a1614; code at b2657cc89, patch experiments/removed/MOJOLEARN_IDN_CHOL_NB128.patch"
    # wave-2 w2-decide-algos (2026-10-11): promoted (absent = new path; _OFF the old path) and removed switches.
    comptime assert not is_defined["MOJOLEARN_IDN_SVC_ONE_BLOCK"](), "promoted 2026-10-11 (lane/w2-decide-algos, same-box A/B on main c8ef6c66d): one-block SMO solve is the IDENTICAL default (svc 0.720/0.663); drop -D MOJOLEARN_IDN_SVC_ONE_BLOCK, use -D MOJOLEARN_IDN_SVC_ONE_BLOCK_OFF for the old path"
    # lane n8-gp-krr-smo (2026-11-09): the GPC Newton loop is one device loop (gaussian_process/gpc_newton.mojo).
    comptime assert not is_defined["MOJOLEARN_IDN_GPC_NEWTON_LOGDET_OFF"](), "removed 2026-11-09 (lane/n8-gp-krr-smo): IDN_GPC_NEWTON_LOGDET_OFF read log|B| home per Newton iteration; the loop is now one bounded device loop over the executor's Cholesky ops and log|B| never leaves the device. Code at f59d913b0b, patch experiments/removed/N8_GPC_NEWTON_ONE_LOOP.patch (docs/TOMBSTONES.md)"
    # Lane expdel-06-base-cluster-kernel (2026-11-15): owed deletions of docs/plans/EXPERIMENT_DELETIONS_2026-11-15.md (docs/TOMBSTONES.md).
    # Restore: git apply experiments/removed/MOJOLEARN_KDE_FAST_SLICES_OFF.patch
    comptime assert not is_defined["MOJOLEARN_KDE_FAST_SLICES_OFF"](), "removed 2026-11-15 (lane/expdel-06-base-cluster-kernel): KDE_FAST_SLICES_OFF (FAST+Apple rollback to one thread per query over every train row) is gone; the train-row slices are the FAST+Apple path (kde istella 1,519 -> 141 ms, core-kde-slices-istella); restore: experiments/removed/MOJOLEARN_KDE_FAST_SLICES_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_KERNEL_FAST_NYS_RR_EIGH_OFF.patch
    comptime assert not is_defined["MOJOLEARN_KERNEL_FAST_NYS_RR_EIGH_OFF"](), "removed 2026-11-15 (lane/expdel-06-base-cluster-kernel): KERNEL_FAST_NYS_RR_EIGH_OFF (FAST+Apple rollback of the Nystroem basis eigh to the one-block cyclic Jacobi) is gone; the round-robin Jacobi is the FAST+Apple path (nystroem taxi 533 -> 190 ms, kap2-km-nysrr-taxi); restore: experiments/removed/MOJOLEARN_KERNEL_FAST_NYS_RR_EIGH_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_KM_FAST_PTR_IN_OFF.patch
    comptime assert not is_defined["MOJOLEARN_KM_FAST_PTR_IN_OFF"](), "removed 2026-11-15 (lane/expdel-06-base-cluster-kernel): KM_FAST_PTR_IN_OFF (FAST+Apple rollback of the transforms' X to an owned host copy and host finiteness walk) is gone; the caller-pointer upload is the FAST+Apple path (rbf-sampler istella 121 -> 94 ms, kap2-km-ptrin-rbf-istella); IDENTICAL keeps MOJOLEARN_IDN_KM_PTR_IN_OFF; restore: experiments/removed/MOJOLEARN_KM_FAST_PTR_IN_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_KM_FAST_RBF_PIPE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_KM_FAST_RBF_PIPE_OFF"](), "removed 2026-11-15 (lane/expdel-06-base-cluster-kernel): KM_FAST_RBF_PIPE_OFF (FAST+Apple rollback of RBFSampler fit_transform to the unpiped STAGED sequence) is gone; the row-block pipeline is the FAST+Apple path (rbf-sampler istella 56.7 -> 43.0 ms, rab12-rbfpipe); restore: experiments/removed/MOJOLEARN_KM_FAST_RBF_PIPE_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_KM_FAST_RBF_STAGED_OFF.patch
    comptime assert not is_defined["MOJOLEARN_KM_FAST_RBF_STAGED_OFF"](), "removed 2026-11-15 (lane/expdel-06-base-cluster-kernel): KM_FAST_RBF_STAGED_OFF (FAST+Apple rollback of the RBFSampler projection to one raw host-pointer copy) is gone; the staged download is the path wherever KM_FAST_RBF_RESIDENT is (rbf-sampler istella 77.6 -> 57.1 ms, w2-w3kf-rbf-q PASS); restore: experiments/removed/MOJOLEARN_KM_FAST_RBF_STAGED_OFF.patch"
    # Lane expdel2-06-base (2026-11-15): the C13-C28 classical arms with no recorded A/B and the _OFF rollbacks of measured
    # IDENTICAL defaults read by experiments/classical_identical_ideas/linear_controls.mojo (plan EXPERIMENT_DELETIONS_V2_2026-11-15 (git history)).
    # Restore: git apply experiments/removed/MOJOLEARN_CLASSICAL_C20_PAIR_LOAD.patch
    comptime assert not is_defined["MOJOLEARN_CLASSICAL_C20_PAIR_LOAD"](), "removed 2026-11-15 (lane/expdel2-06-base): CLASSICAL_C20_PAIR_LOAD (Kernel (SVM working-set row + norm gather, IDENTICAL)) was unmeasured opt-in arm; the opt-in arm is gone and the default path is unchanged; code recoverable at 911bf993d5; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_CLASSICAL_C21_EXTREMA.patch
    comptime assert not is_defined["MOJOLEARN_CLASSICAL_C21_EXTREMA"](), "removed 2026-11-15 (lane/expdel2-06-base): CLASSICAL_C21_EXTREMA (Kernel (SMO four-way extrema tree, IDENTICAL)) was unmeasured opt-in arm; the opt-in arm is gone and the default path is unchanged; code recoverable at 9aca9aa819; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_CLASSICAL_C22_TRIANGLE.patch
    comptime assert not is_defined["MOJOLEARN_CLASSICAL_C22_TRIANGLE"](), "removed 2026-11-15 (lane/expdel2-06-base): CLASSICAL_C22_TRIANGLE (Kernel (SMO square tile upper triangle mirrored, IDENTICAL)) was unmeasured opt-in arm; the opt-in arm is gone and the default path is unchanged; code recoverable at 45379ff055; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_CLASSICAL_C25_PROJECTION_REUSE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_CLASSICAL_C25_PROJECTION_REUSE_OFF"](), "removed 2026-11-15 (lane/expdel2-06-base): CLASSICAL_C25_PROJECTION_REUSE_OFF (Kernel (RBFSampler projection old path, IDENTICAL old path)) was old path, superseded by the measured IDENTICAL default; the measured IDENTICAL default is the only path; code recoverable at 935328056e; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_KDE_DIRECT_DISTANCE.patch
    comptime assert not is_defined["MOJOLEARN_KDE_DIRECT_DISTANCE"](), "removed 2026-11-15 (lane/expdel2-05-base): KDE_DIRECT_DISTANCE (opt-in IDENTICAL arm, never measured) is gone; the incumbent path is the only one"
    # Restore: git apply experiments/removed/MOJOLEARN_C52_PAIR_ROWS.patch
    comptime assert not is_defined["MOJOLEARN_C52_PAIR_ROWS"](), "removed 2026-11-15 (lane/expdel2-07-base): C52_PAIR_ROWS (Kernel density (KDE pair-combine route with fixed 128/512 leaf, IDENTICAL)) was unmeasured opt-in arm; the opt-in arm is gone and the default path is unchanged; code recoverable at dd94b7ad88; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_C54_PREDICT_TILES.patch
    comptime assert not is_defined["MOJOLEARN_C54_PREDICT_TILES"](), "removed 2026-11-15 (lane/expdel2-07-base): C54_PREDICT_TILES (Kernel / GP (four kernel columns share one scaled query load, IDENTICAL)) was unmeasured opt-in arm; the opt-in arm is gone and the default path is unchanged; code recoverable at 843366c4fd; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_GPC_OVR_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_GPC_OVR_OFF"](), "removed 2026-11-15 (lane/expdel2-07-base): IDN_GPC_OVR_OFF (Kernel / GP (GaussianProcessClassifier one-vs-rest rollback to per-class binding calls, IDENTICAL old path)) was old path, no A/B recorded; the batched one-vs-rest default kept; the default is the only path; code recoverable at abe1b14d9d; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_GPR_PTR_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_GPR_PTR_OFF"](), "removed 2026-11-15 (lane/expdel2-07-base): IDN_GPR_PTR_OFF (Kernel / GP (GaussianProcessRegressor rollback to host-list gpr_fit / gpr_predict, IDENTICAL old path)) was old path, no A/B recorded; the resident pointer default kept; the default is the only path; code recoverable at b2a4adf292; see docs/TOMBSTONES.md"
    # Lane expdel2-16-decomposition (2026-11-15): unmeasured arms and superseded old paths deleted, plan EXPERIMENT_DELETIONS_V2_2026-11-15 (git history) (docs/TOMBSTONES.md).
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_CHOL_OPS_FIT.fb12-loser-arms.patch
    comptime assert not is_defined["MOJOLEARN_IDN_CHOL_OPS_FIT"](), "removed 2026-11-17 (lane/fb12-loser-arms): IDN_CHOL_OPS_FIT (decided B arm, the executor Cholesky ops for KernelRidge / GPR / GPC / GP optimizer / GMM / MinCovDet fits; L40S ba08dac006 14 cells B 0.91-1.10x, same speed) is gone; the default path is the only one; code recoverable at 2203db7094; restore: experiments/removed/MOJOLEARN_IDN_CHOL_OPS_FIT.fb12-loser-arms.patch"
    # Lane expdel2-19-kernel (2026-11-15): unmeasured arms and old paths deleted, plan EXPERIMENT_DELETIONS_V2_2026-11-15 (git history) (docs/TOMBSTONES.md).
    # MOJOLEARN_AFCL_L11: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-classical: Apple fast paused, not abandoned; opt-in, default off; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFCL_G13: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-classical: Apple fast paused, not abandoned; opt-in, default off; see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_FAMILY_CTX_PER_CALL.patch
    comptime assert not is_defined["MOJOLEARN_FAMILY_CTX_PER_CALL"](), "removed 2026-11-15 (lane/expdel2-19-kernel): FAMILY_CTX_PER_CALL (opt-in arm) is gone; the default path is the only one; code at 26b4b3147a; restore: experiments/removed/MOJOLEARN_FAMILY_CTX_PER_CALL.patch"
    # MOJOLEARN_GP_BULK_DOWNLOAD: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-classical: Apple fast paused, not abandoned; opt-in, default off; see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_GP_BULK_DOWNLOAD_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_GP_BULK_DOWNLOAD_OFF"](), "removed 2026-11-15 (lane/expdel2-19-kernel): IDN_GP_BULK_DOWNLOAD_OFF (old path) is gone; the default is kept; code at 2fd4a6c6c1; restore: experiments/removed/MOJOLEARN_IDN_GP_BULK_DOWNLOAD_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_GP_PREDICT_LAZY_L_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_GP_PREDICT_LAZY_L_OFF"](), "removed 2026-11-15 (lane/expdel2-19-kernel): IDN_GP_PREDICT_LAZY_L_OFF (old path) is gone; the default is kept; code at 6969d83ccb; restore: experiments/removed/MOJOLEARN_IDN_GP_PREDICT_LAZY_L_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_GP_GRAD_ROWS4.patch
    comptime assert not is_defined["MOJOLEARN_IDN_GP_GRAD_ROWS4"](), "removed 2026-11-15 (lane/expdel2-19-kernel): IDN_GP_GRAD_ROWS4 (opt-in arm) is gone; the default path is the only one; code at 0deab16a32; restore: experiments/removed/MOJOLEARN_IDN_GP_GRAD_ROWS4.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_GP_GRAD_ROWFOLD_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_GP_GRAD_ROWFOLD_OFF"](), "removed 2026-11-15 (lane/expdel2-19-kernel): IDN_GP_GRAD_ROWFOLD_OFF (old path) is gone; the default is kept; code at e53df78bab; restore: experiments/removed/MOJOLEARN_IDN_GP_GRAD_ROWFOLD_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_GP_VAR_SEG_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_GP_VAR_SEG_OFF"](), "removed 2026-11-15 (lane/expdel2-19-kernel): IDN_GP_VAR_SEG_OFF (old path) is gone; the default is kept; code at 87c68144e3; restore: experiments/removed/MOJOLEARN_IDN_GP_VAR_SEG_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_KDE_FAST_DIRECT_PREP_OFF.patch
    comptime assert not is_defined["MOJOLEARN_KDE_FAST_DIRECT_PREP_OFF"](), "removed 2026-11-15 (lane/expdel2-19-kernel): KDE_FAST_DIRECT_PREP_OFF (old path) is gone; the default is kept; code at 9b758c74c1; restore: experiments/removed/MOJOLEARN_KDE_FAST_DIRECT_PREP_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_KDE_PARTIAL_POOL.patch
    comptime assert not is_defined["MOJOLEARN_IDN_KDE_PARTIAL_POOL"](), "removed 2026-11-15 (lane/expdel2-19-kernel): IDN_KDE_PARTIAL_POOL (opt-in arm) is gone; the default path is the only one; code at 3af082d6f4; restore: experiments/removed/MOJOLEARN_IDN_KDE_PARTIAL_POOL.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_KDE_BLOCK_SUM_OFF.patch
    comptime assert not is_defined["MOJOLEARN_KDE_BLOCK_SUM_OFF"](), "removed 2026-11-15 (lane/expdel2-19-kernel): KDE_BLOCK_SUM_OFF (old path) is gone; the default is kept; code at 8b543e49e0; restore: experiments/removed/MOJOLEARN_KDE_BLOCK_SUM_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_LEGACY_NARROW_KDE_DIMTILE.patch
    comptime assert not is_defined["MOJOLEARN_LEGACY_NARROW_KDE_DIMTILE"](), "removed 2026-11-15 (lane/expdel2-19-kernel): LEGACY_NARROW_KDE_DIMTILE (old path) is gone; the default is kept; code at 531563873b; restore: experiments/removed/MOJOLEARN_LEGACY_NARROW_KDE_DIMTILE.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_KDE_DIMTILE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_KDE_DIMTILE_OFF"](), "removed 2026-11-15 (lane/expdel2-19-kernel): KDE_DIMTILE_OFF (old path) is gone; the default is kept; code at 82d4e4de40; restore: experiments/removed/MOJOLEARN_KDE_DIMTILE_OFF.patch"
    # MOJOLEARN_IDN_KDE_QTILE_FORCE: refusal removed 2026-11-15 (restored; owed B arm, see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_KDE_QTILE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_KDE_QTILE_OFF"](), "removed 2026-11-15 (lane/expdel2-19-kernel): IDN_KDE_QTILE_OFF (old path) is gone; the default is kept; code at 4bf6c39024; restore: experiments/removed/MOJOLEARN_IDN_KDE_QTILE_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_KDE_CHUNK_LSE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_KDE_CHUNK_LSE_OFF"](), "removed 2026-11-15 (lane/expdel2-19-kernel): IDN_KDE_CHUNK_LSE_OFF (old path) is gone; the default is kept; code at 9dcc758292; restore: experiments/removed/MOJOLEARN_IDN_KDE_CHUNK_LSE_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_KM_RBF_CELL_D16.patch
    comptime assert not is_defined["MOJOLEARN_IDN_KM_RBF_CELL_D16"](), "removed 2026-11-15 (lane/expdel2-19-kernel): IDN_KM_RBF_CELL_D16 (opt-in arm) is gone; the default path is the only one; code at 0e42489093; restore: experiments/removed/MOJOLEARN_IDN_KM_RBF_CELL_D16.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_KM_DOT_CELL_D16.patch
    comptime assert not is_defined["MOJOLEARN_IDN_KM_DOT_CELL_D16"](), "removed 2026-11-15 (lane/expdel2-19-kernel): IDN_KM_DOT_CELL_D16 (opt-in arm) is gone; the default path is the only one; code at b1d14459cf; restore: experiments/removed/MOJOLEARN_IDN_KM_DOT_CELL_D16.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_KM_RBF_CELL.patch
    comptime assert not is_defined["MOJOLEARN_IDN_KM_RBF_CELL"](), "removed 2026-11-15 (lane/expdel2-19-kernel): IDN_KM_RBF_CELL (opt-in arm) is gone; the default path is the only one; code at 0ade8f67d9; restore: experiments/removed/MOJOLEARN_IDN_KM_RBF_CELL.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_KM_DOT_CELL_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_KM_DOT_CELL_OFF"](), "removed 2026-11-15 (lane/expdel2-19-kernel): IDN_KM_DOT_CELL_OFF (old path) is gone; the default is kept; code at 16e283c44f; restore: experiments/removed/MOJOLEARN_IDN_KM_DOT_CELL_OFF.patch"
    # MOJOLEARN_RBF_FUSED: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-classical: Apple fast paused, not abandoned; opt-in, default off; see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_RBF_FUSED_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_RBF_FUSED_OFF"](), "removed 2026-11-15 (lane/expdel2-19-kernel): IDN_RBF_FUSED_OFF (old path) is gone; the default is kept; code at f2f200fc1b; restore: experiments/removed/MOJOLEARN_IDN_RBF_FUSED_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_NYS_RR_EIGH_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_NYS_RR_EIGH_OFF"](), "removed 2026-11-15 (lane/expdel2-19-kernel): IDN_NYS_RR_EIGH_OFF (old path) is gone; the default is kept; code at 0e594fa5e2; restore: experiments/removed/MOJOLEARN_IDN_NYS_RR_EIGH_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_KRR_DEV_SCALE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_KRR_DEV_SCALE_OFF"](), "removed 2026-11-15 (lane/expdel2-19-kernel): IDN_KRR_DEV_SCALE_OFF (old path) is gone; the default is kept; code at 0765c7997d; restore: experiments/removed/MOJOLEARN_IDN_KRR_DEV_SCALE_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_KRR_PTR_IN_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_KRR_PTR_IN_OFF"](), "removed 2026-11-15 (lane/expdel2-19-kernel): IDN_KRR_PTR_IN_OFF (old path) is gone; the default is kept; code at 25dea25b3b; restore: experiments/removed/MOJOLEARN_IDN_KRR_PTR_IN_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_KM_BULK_DOWNLOAD_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_KM_BULK_DOWNLOAD_OFF"](), "removed 2026-11-15 (lane/expdel2-19-kernel): IDN_KM_BULK_DOWNLOAD_OFF (old path) is gone; the default is kept; code at 1a2458c687; restore: experiments/removed/MOJOLEARN_IDN_KM_BULK_DOWNLOAD_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_KM_PTR_IN_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_KM_PTR_IN_OFF"](), "removed 2026-11-15 (lane/expdel2-19-kernel): IDN_KM_PTR_IN_OFF (old path) is gone; the default is kept; code at d6e7cef9f2; restore: experiments/removed/MOJOLEARN_IDN_KM_PTR_IN_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_NYS_RR_DEV_STOP_B4.patch
    comptime assert not is_defined["MOJOLEARN_IDN_NYS_RR_DEV_STOP_B4"](), "removed 2026-11-15 (lane/expdel2-19-kernel): IDN_NYS_RR_DEV_STOP_B4 (opt-in arm) is gone; the default path is the only one; code at c8712ad631; restore: experiments/removed/MOJOLEARN_IDN_NYS_RR_DEV_STOP_B4.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_NYS_RR_DEV_STOP.patch
    comptime assert not is_defined["MOJOLEARN_IDN_NYS_RR_DEV_STOP"](), "removed 2026-11-15 (lane/expdel2-19-kernel): IDN_NYS_RR_DEV_STOP (opt-in arm) is gone; the default path is the only one; code at 27b1a08df9; restore: experiments/removed/MOJOLEARN_IDN_NYS_RR_DEV_STOP.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_NYS_DEV_ORDER_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_NYS_DEV_ORDER_OFF"](), "removed 2026-11-15 (lane/expdel2-19-kernel): IDN_NYS_DEV_ORDER_OFF (old path) is gone; the default is kept; code at a1459da190; restore: experiments/removed/MOJOLEARN_IDN_NYS_DEV_ORDER_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_NYS_FIT_PTR_IN_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_NYS_FIT_PTR_IN_OFF"](), "removed 2026-11-15 (lane/expdel2-19-kernel): IDN_NYS_FIT_PTR_IN_OFF (old path) is gone; the default is kept; code at 1c49bdcfcc; restore: experiments/removed/MOJOLEARN_IDN_NYS_FIT_PTR_IN_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_NYS_DEV_BASIS_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_NYS_DEV_BASIS_OFF"](), "removed 2026-11-15 (lane/expdel2-19-kernel): IDN_NYS_DEV_BASIS_OFF (old path) is gone; the default is kept; code at 91606051a7; restore: experiments/removed/MOJOLEARN_IDN_NYS_DEV_BASIS_OFF.patch"
    # Lane expdel2-20-linear (2026-11-15): unmeasured arms deleted, plan EXPERIMENT_DELETIONS_V2_2026-11-15 (git history) (docs/TOMBSTONES.md).
    # MOJOLEARN_AFCL_L06: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-classical: Apple fast paused, not abandoned; opt-in, default off; see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_SVC_ONE_BLOCK_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_SVC_ONE_BLOCK_OFF"](), "removed 2026-11-15 (lane/expdel2-20-linear): IDN_SVC_ONE_BLOCK_OFF (old path of the promoted default) is gone; the default is kept; code at ed4cab202e; restore: experiments/removed/MOJOLEARN_IDN_SVC_ONE_BLOCK_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_SVM_IDN_FUSED_TILE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_SVM_IDN_FUSED_TILE_OFF"](), "removed 2026-11-15 (lane/expdel2-20-linear): SVM_IDN_FUSED_TILE_OFF (old path, no A/B recorded) is gone; the default is kept; code at 8398142558; restore: experiments/removed/MOJOLEARN_SVM_IDN_FUSED_TILE_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_SVM_IDN_FUSED_UPDATE_F_OFF.patch
    comptime assert not is_defined["MOJOLEARN_SVM_IDN_FUSED_UPDATE_F_OFF"](), "removed 2026-11-15 (lane/expdel2-20-linear): SVM_IDN_FUSED_UPDATE_F_OFF (old path, no A/B recorded) is gone; the default is kept; code at 30ab49dcd7; restore: experiments/removed/MOJOLEARN_SVM_IDN_FUSED_UPDATE_F_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_SVM_IDN_NAN_SCAN_ONE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_SVM_IDN_NAN_SCAN_ONE_OFF"](), "removed 2026-11-15 (lane/expdel2-20-linear): SVM_IDN_NAN_SCAN_ONE_OFF (old path, no A/B recorded) is gone; the default is kept; code at 788f90f025; restore: experiments/removed/MOJOLEARN_SVM_IDN_NAN_SCAN_ONE_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_SVM_RBF_EPILOGUE_1D.patch
    comptime assert not is_defined["MOJOLEARN_SVM_RBF_EPILOGUE_1D"](), "removed 2026-11-15 (lane/expdel2-20-linear): SVM_RBF_EPILOGUE_1D (opt-in arm, no A/B recorded) is gone; the default is kept; code at 58cc0078e2; restore: experiments/removed/MOJOLEARN_SVM_RBF_EPILOGUE_1D.patch"
    # ---- lane ab-prep-classical (2026-11-23): owed arms closed without a run (plumbing, rule 2026-11-22)
    # MOJOLEARN_IDN_QUERY_X_HELD_OFF: its 2026-11-15 refusal was removed in core/six_lane_experiment_guards.mojo (restored owed arm); the arm is now deleted and refused here.
    # TOMBSTONE: MOJOLEARN_IDN_QUERY_X_HELD_OFF (removed without measurement: plumbing, rule 2026-11-22) deleted 2026-11-23 by lane/ab-prep-classical; code recoverable at 4d76a9e16e.
    # Restore: git apply ~/mojolearn-evidence/experiments-removed/MOJOLEARN_IDN_QUERY_X_HELD_OFF.ab-prep.patch; record in docs/TOMBSTONES.md.
    comptime assert not is_defined["MOJOLEARN_IDN_QUERY_X_HELD_OFF"](), "removed 2026-11-23 (lane/ab-prep-classical, removed without measurement: plumbing, rule 2026-11-22: the per-call query buffers fed the same bytes to the same kernels, core/query_x_held.mojo NUMERICS; only per-call allocation, pinned pair and status fill went); code at 4d76a9e16e; restore: ~/mojolearn-evidence/experiments-removed/MOJOLEARN_IDN_QUERY_X_HELD_OFF.ab-prep.patch"
    # TOMBSTONE: MOJOLEARN_CLASSICAL_NATIVE_DISPATCH (removed without measurement: plumbing, rule 2026-11-22) deleted 2026-11-23 by lane/ab-prep-classical; code recoverable at 4d76a9e16e.
    # Restore: git apply ~/mojolearn-evidence/experiments-removed/MOJOLEARN_CLASSICAL_NATIVE_DISPATCH.patch; record in docs/TOMBSTONES.md.
    comptime assert not is_defined["MOJOLEARN_CLASSICAL_NATIVE_DISPATCH"](), "removed 2026-11-23 (lane/ab-prep-classical, removed without measurement: plumbing, rule 2026-11-22: both arms ran the same numerical owners on the same words, svr_fit_host_borrowed / svc tables; the B arm only dropped the device-resident query and device-X fit and added a mode readback and a published diagnostic); code at 4d76a9e16e; restore: ~/mojolearn-evidence/experiments-removed/MOJOLEARN_CLASSICAL_NATIVE_DISPATCH.patch"
    # ---- end lane ab-prep-classical
    return True


comptime KERNEL_GUARDS_OK = _check_configuration()
