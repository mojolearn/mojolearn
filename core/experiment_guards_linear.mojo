# SPDX-License-Identifier: Apache-2.0
"""Opt-in strategy admission for the linear-model bindings (estimators, solver, linalg, x_linear, the core host, the classical and model-selection runtimes).

Each binding of this family imports LINEAR_GUARDS_OK and asserts it in its build function, so every compile of one of
those bindings (per-binding or merged) evaluates these guards. A guard whose define is read in code by bindings
of more than one family belongs in core/six_lane_experiment_guards.mojo, which every binding reaches through
checks/numerics.mojo.
"""
from std.sys.compile import is_defined, get_defined_int
# The entries below moved verbatim, in their original order, from core/six_lane_experiment_guards.mojo on
# 2026-11-04 (lane split-hot-groups; the placement of each entry is docs/build/guard_split.json). A header comment
# that headed lines of several families is carried with each of them.


def _check_configuration() -> Bool:
    # Deleted 2026-10-08 (lane grid-losers-1): IDENTICAL grid ge123e6f9 losers (docs/apple-fast/EXPERIMENTS.md). Recoverable at main ad7ed2370.
    # C25_PROJECTION_REUSE: deleted 2026-10-08 above, restored 2026-10-10 (lane c25-restore-promote) from the tombstone patch after grid pass 2.
    comptime assert not (is_defined["MOJOLEARN_QN_IDN_DCONV"]() or is_defined["MOJOLEARN_QN_IDN_DCONV_POLL_2"]() or is_defined["MOJOLEARN_QN_IDN_DCONV_POLL_8"]()), "removed: MOJOLEARN_QN_IDN_DCONV (and _POLL_2/_POLL_8) retired 2026-10-08: neutral/slower, NV 0.95-1.06x / AMD 0.98-1.08x on logreg, linearsvc, linearsvr (grid ge123e6f9); see EXPERIMENTS.md"
    comptime assert not is_defined["MOJOLEARN_CLASSICAL_LINEAR_GRAM_SOLVE"](), "promoted 2026-10-08 (grid ge123e6f9, lane/grid-flips-1): OLS/Ridge resident Gram is the IDENTICAL default; drop -D MOJOLEARN_CLASSICAL_LINEAR_GRAM_SOLVE, use -D MOJOLEARN_CLASSICAL_LINEAR_GRAM_SOLVE_OFF for the old path"
    comptime assert not is_defined["MOJOLEARN_CLASSICAL_C24_ROWS2048"](), "promoted 2026-10-08 (grid ge123e6f9, lane/grid-flips-1): TSQR 2048-row leaves is the IDENTICAL default; drop -D MOJOLEARN_CLASSICAL_C24_ROWS2048, use -D MOJOLEARN_CLASSICAL_C24_ROWS2048_OFF for the old path"
    # Deleted 2026-10-08 (lane grid-act-2): IDENTICAL grid ge123e6f9 losers (docs/apple-fast/EXPERIMENTS.md). Recoverable at main 42d1e42c6.
    comptime assert not is_defined["MOJOLEARN_IDN_GEMM_OZAKI_LINALG"](), "removed: MOJOLEARN_IDN_GEMM_OZAKI_LINALG retired 2026-10-08: slower, gemm S=4 NV 3.97x / AMD 2.06x, S=5 NV 7.99x / AMD 2.51x (grid ge123e6f9); the neural Ozaki profile (MOJOLEARN_IDN_NEURAL_GEMM_OZAKI_SLICES) is separate; see EXPERIMENTS.md"
    comptime assert not is_defined["MOJOLEARN_CLASSICAL_TSVD_FUSED_STATS"](), "removed 2026-10-10 (grid g50ebe26a5, lane/grid-act-6): TSVD_FUSED_STATS (one blocked [X | X V^T] variance kernel for TruncatedSVD's explained variance) was SLOWER: tsvd istella NV 2.34x / AMD 3.46x (2.844x combined), taxi NV 1.83x / AMD 1.38x (1.593x); code at main 328b0ae58; see docs/TOMBSTONES.md"
    comptime assert not is_defined["MOJOLEARN_IDN_OLS_ONE_ENTRY_OFF"](), "removed: MOJOLEARN_IDN_OLS_ONE_ENTRY_OFF retired 2026-10-08: the off arm is slower, ols NV 2.97x / AMD 1.74x istella, NV 6.17x / AMD 3.83x taxi, r2 SAME (grid ge123e6f9); the one-entry route is the only IDENTICAL OLS route; see EXPERIMENTS.md"
    # Lane grid-act-3 (2026-10-08): IDENTICAL grid ge123e6f9 promotions and losers (docs/apple-fast/EXPERIMENTS.md). Deleted code recoverable at main bc10b8b56.
    comptime assert not is_defined["MOJOLEARN_CLASSICAL_C08_UNIQUE_SCAN"](), "promoted 2026-10-08 (grid ge123e6f9, lane/grid-act-3): the unique_cols run scan is the IDENTICAL default (onehot/ordinal/target-encoder 0.525x combined, same bits); drop -D MOJOLEARN_CLASSICAL_C08_UNIQUE_SCAN, use -D MOJOLEARN_CLASSICAL_C08_UNIQUE_SCAN_OFF for the old path"
    comptime assert not is_defined["MOJOLEARN_CLASSICAL_C13_FOLD_STATS_OFF"](), "removed: MOJOLEARN_CLASSICAL_C13_FOLD_STATS_OFF retired 2026-10-08: the off arm is slower, ridge-cv NV 1.19x / AMD 1.11x istella, NV 19.5x / AMD 18.0x taxi, quality SAME (grid ge123e6f9); the RidgeCV fold cache is the only IDENTICAL route; see EXPERIMENTS.md"
    # Lane postmerge-act-2 (2026-10-09): post-merge A/B races on main, one run per arm (nv L40S, amd MI325X; docs/apple-fast/EXPERIMENTS.md). Deleted code recoverable at main 5c137b55e.
    comptime assert not is_defined["MOJOLEARN_IDN_SGD_EPOCH_KERNEL"](), "removed 2026-10-09 (post-merge A/B nv n0606->n0615, amd a1065->a1074, lane/postmerge-act-2): fg-linear S1 (large-batch SGD through the K-batch chunk kernel) was NOISE: sgd-clf istella NV 0.99x / AMD 1.00x, taxi NV 1.00x / AMD 1.00x, accuracy SAME, same digests; code at main 5c137b55e; see EXPERIMENTS.md"
    # Lane postmerge-act-5 (2026-10-09): fg2 board-bridge A/B on main 0a7b206f1, one run per arm (nv2 L40S v1021-v1050, amd MI325X a1161-a1190; ratio = arm / fg2.default-linear; neural nv2 v1010/v1012, amd a1141/a1143; docs/apple-fast/EXPERIMENTS.md). Two promoted (ridge bits change, equal on both vendors), two removed; deleted code recoverable at main 0a7b206f1.
    comptime assert not is_defined["MOJOLEARN_IDN_GRAM_FF_FALLBACK"](), "promoted 2026-10-09 (fg2 A/B nv2 v1021-v1050, amd a1161-a1190, lane/postmerge-act-5): the float-float Gram second chance for a gate-rejected Ridge fit is the IDENTICAL default (ridge istella NV 0.96x / AMD 0.51x, avg 0.74x, hash ff4ab841 -> 5a348e50 on both vendors, r2 0.328682 -> 0.328674; ridge taxi NV 0.91x / AMD 0.99x; ols 0.98x-1.03x, hashes unchanged); drop -D MOJOLEARN_IDN_GRAM_FF_FALLBACK, use -D MOJOLEARN_IDN_GRAM_FF_FALLBACK_OFF for the direct eig-route fallback"
    # Lane owed-deletions-D3 (2026-10-09): owed DROP deletions (docs/TOMBSTONES.md 'Owed deletions'; docs/apple-fast/EXPERIMENTS.md). Deleted code recoverable at main b639a2bd2.
    comptime assert not is_defined["MOJOLEARN_CHOL_FAST_BLOCKED"](), "removed 2026-10-09 (lane/owed-deletions-D3): CHOL_FAST_BLOCKED (blocked right-looking Cholesky, x_decomp/fast_chol.mojo) was SLOWER and less accurate: cholesky synthetic 259.4 -> 330.7 ms, relative_residual 1.66e-7 -> 1.98e-6 (M3, 1 run per arm); code at main b639a2bd2; see docs/TOMBSTONES.md"
    # Lane consol-k0-dead-code (2026-11-08): loser define whose gate was still in the code (docs/TOMBSTONES.md batch 3).
    # Restore: git apply experiments/removed/MOJOLEARN_CHOL_FAST_NB512.patch
    comptime assert not is_defined["MOJOLEARN_CHOL_FAST_NB512"](), "removed 2026-11-08 (lane consol-k0-dead-code): MOJOLEARN_CHOL_FAST_NB512, the 512-wide FAST Cholesky panel, was HOLD-speed: w2-cholnb512-synthetic 260.4 -> 265.7 ms (no benefit), quality PASS (lane/apple-fast-w4-linalg 19bb1c7a1); the 256 default stays; code at 41ffcec665; see docs/TOMBSTONES.md"
    comptime assert not is_defined["MOJOLEARN_KDE2_ALL"](), "removed 2026-10-09 (lane/owed-deletions-D3): KDE2_ALL (every kde2 FAST Apple define at once) was DROPPED: kde istella 139.5 -> 63.1 ms but -0.6% vs KDE_DIMTILE alone (the gain is DIMTILE, now the default); it includes SAMPLE_FUSED (taxi +354%); code at main b639a2bd2; see docs/TOMBSTONES.md"
    comptime assert not is_defined["MOJOLEARN_KDE_KERNEL_VARIANTS"](), "removed 2026-10-09 (lane/owed-deletions-D3): KDE_KERNEL_VARIANTS (kde2 tile kernel instantiated per metric at compile time) was DROPPED: with DIMTILE no istella gain, taxi jitter-dominated (~10 ms), quality unchanged; code at main b639a2bd2; see docs/TOMBSTONES.md"
    comptime assert not is_defined["MOJOLEARN_KDE_LSE_FUSED"](), "removed 2026-10-09 (lane/owed-deletions-D3): KDE_LSE_FUSED (kde2 chunk merge and score normalization in one launch) was DROPPED: with DIMTILE no istella gain, taxi jitter-dominated, quality unchanged; code at main b639a2bd2; see docs/TOMBSTONES.md"
    comptime assert not is_defined["MOJOLEARN_KDE_NORM_FUSED"](), "removed 2026-10-09 (lane/owed-deletions-D3): KDE_NORM_FUSED (euclidean kde2 cells as |q|^2 + |t|^2 - 2 q.t) was DROPPED: with DIMTILE no istella gain, taxi jitter-dominated, quality unchanged; code at main b639a2bd2; see docs/TOMBSTONES.md"
    comptime assert not is_defined["MOJOLEARN_KDE_SAMPLE_FUSED"](), "removed 2026-10-09 (lane/owed-deletions-D3): KDE_SAMPLE_FUSED (resident kde score with its download behind it, one drain per call) was DROPPED: kde taxi +354% (old base), with DIMTILE no istella gain; code at main b639a2bd2; see docs/TOMBSTONES.md"
    comptime assert not is_defined["MOJOLEARN_QN_FAST_COALESCED_OFF"](), "removed 2026-10-09 (lane/owed-deletions-D3): QN_FAST_COALESCED_OFF (rollback from the row-coalesced qn gradient X^T dZ to fast_xtdz) is SLOWER: logreg istella 3,889 -> 4,996 ms (+28.5%); the coalesced route is the only FAST Apple route (QN_FAST_XTDZ_OFF still turns both off); code at main b639a2bd2; see docs/TOMBSTONES.md"
    comptime assert not is_defined["MOJOLEARN_DBSCAN_FAST_DENSEBALL"](), "removed 2026-10-09 (lane/owed-deletions-D3): DBSCAN_FAST_DENSEBALL (no-edge-list dbscan: dense-ball cliques and union-find over landmark pairs, dbscan/impl/denseball.mojo) was SLOWER: dbscan istella arm A 247,185 ms, arm B timed out (rab3-denseball); code at main b639a2bd2; see docs/TOMBSTONES.md"
    # Lane postmerge-act-6 (2026-10-10), promoted: post-merge A/B on main, one run per arm (neural: nv2 L40S v1052-v1062 on 9f83ea479, amd MI325X a1148-a1159 on 0a7b206f1; classical fg2: nv2 v1021-v1050, amd a1161-a1190; docs/apple-fast/EXPERIMENTS.md). The old on-defines are refused; each _OFF restores the old path.
    comptime assert not is_defined["MOJOLEARN_IDN_COLVAR_FUSED"](), "promoted 2026-10-10 (fg2 A/B nv2 v1021-v1050, amd a1161-a1190, lane/postmerge-act-6): the fused column variance (centering and squaring in registers) is the IDENTICAL default (tsvd taxi NV 0.76x / AMD 0.97x, istella 0.97x / 0.97x, pca 1.00-1.03x, same bits); drop -D MOJOLEARN_IDN_COLVAR_FUSED, use -D MOJOLEARN_IDN_COLVAR_FUSED_OFF for the shift + square passes"
    # Lane postmerge-act-6 (2026-10-10), removed: losers of the same A/B (docs/TOMBSTONES.md, docs/apple-fast/EXPERIMENTS.md). Deleted code recoverable at main 9f83ea479.
    comptime assert not is_defined["MOJOLEARN_IDN_CD_GRAM_EPOCHS_64"](), "removed 2026-10-10 (lane/postmerge-act-6): IDN_CD_GRAM_EPOCHS_64 (C2: Lasso / ElasticNet IDENTICAL Gram sweep at 64 epochs a launch for n_cols <= 64) was SLOWER: lasso taxi NV 1.39x / AMD 1.01x, lasso istella and elasticnet 1.00-1.02x (fg2 nv2 v1021-v1050, amd a1161-a1190), hashes unchanged; code at main 9f83ea479; see docs/TOMBSTONES.md"
    # Lane postmerge-act-7 (2026-10-10), removed: losers of the same fg2 A/B (docs/TOMBSTONES.md, docs/apple-fast/EXPERIMENTS.md). Deleted code recoverable at main ca8ea1f8d.
    comptime assert not is_defined["MOJOLEARN_IDN_DBSCAN_BATCH_SAMPLE"](), "removed 2026-10-10 (lane/postmerge-act-7): IDN_DBSCAN_BATCH_SAMPLE (D4: DBSCAN first ranges sized from a 1,024-row sampled degree) was NOISE on top of EPS_TILE: dbscan istella NV 0.42x / AMD 0.35x against EPS_TILE alone 0.42x / 0.34x (fg2 nv2 v1044, amd a1184), hash unchanged; code at main ca8ea1f8d; see docs/TOMBSTONES.md"
    comptime assert not is_defined["MOJOLEARN_IDN_DBSCAN_CC_HOOK"](), "removed 2026-10-10 (lane/postmerge-act-7): IDN_DBSCAN_CC_HOOK (D5: DBSCAN weak_cc fixed point by lock-free hook and compress) was NOISE: dbscan istella NV 1.00x / AMD 0.96x (fg2 nv2 v1045, amd a1185), hash unchanged; code at main ca8ea1f8d; see docs/TOMBSTONES.md"
    # Lane grid-act-16 (2026-10-11), removed: grid g50ebe26a5 pass-1 losers against today's defaults, one run per arm (docs/TOMBSTONES.md,
    # docs/apple-fast/EXPERIMENTS.md, experiments/removed/*.patch). Deleted code recoverable at main 885c23dc7^ / the lane shas in TOMBSTONES.md.
    comptime assert not is_defined["MOJOLEARN_IDN_LU_NB128"](), "removed 2026-10-11 (lane/grid-act-16): IDN_LU_NB128 (128-wide blocked LU) was SLOWER: lu-solve synthetic NV 1.54x / AMD 2.10x (avg 1.80x), grid g50ebe26a5"
    # Lane i17-decide-linalg-seq (2026-10-11), removed: losers of the A/B on main b2657cc89 / 2afa942e3, one run per arm (docs/TOMBSTONES.md,
    # docs/apple-fast/EXPERIMENTS.md, experiments/removed/*.patch).
    comptime assert not is_defined["MOJOLEARN_IDN_LU_DELAYED_UPDATE"](), "removed 2026-10-11 (lane/i17-decide-linalg-seq, aaf94d8a9): IDN_LU_DELAYED_UPDATE (256-column delayed LU trailing update) was NOISE with bits changed: lu-factor NV 1.02x / AMD 1.01x, lu-solve 0.94x / 1.03x, v0034/a1617; code at aaf94d8a9^, patch experiments/removed/MOJOLEARN_IDN_LU_DELAYED_UPDATE.patch"
    comptime assert not is_defined["MOJOLEARN_IDN_LINEAR_PINNED_UPLOAD_OFF"](), "removed 2026-10-11 (lane/ridge-ff-tiled, 47a3af928): the linear pinned upload stage (IDN_LINEAR_PINNED_UPLOAD) was SLOWER than its OFF arm, grid ge123e6f9: ols istella AMD 0.66x, ols taxi NV 0.86x / AMD 0.26x, ridge taxi 0.64x / 0.68x; the pageable copy is the only path; patch experiments/removed/MOJOLEARN_IDN_LINEAR_PINNED_UPLOAD.patch"
    # Lane i17-decide-nbr-lin-prep (2026-10-11), removed: A/B on main b2657cc89, one run per arm, NV L40S nv2 / AMD MI325X (docs/TOMBSTONES.md,
    # docs/apple-fast/EXPERIMENTS.md, experiments/removed/*.patch).
    comptime assert not is_defined["MOJOLEARN_IDN_BM_GRAM_RB4"](), "removed 2026-10-11 (lane/i17-decide-nbr-lin-prep, bde9c2243): IDN_BM_GRAM_RB4 (register-tiled 64x64 leaf Gram) was SLOWER net ~1.0-1.1x: pca taxi 1.24, pca istella 1.14, ols taxi NV 1.18, ridge taxi 1.08 (v0049, a1632); code at bde9c2243^, patch experiments/removed/MOJOLEARN_IDN_BM_GRAM_RB4.patch"
    # wave-2 w2-decide-algos (2026-10-11): promoted (absent = new path; _OFF the old path) and removed switches.
    comptime assert not is_defined["MOJOLEARN_IDN_KDE_QTILE"](), "promoted 2026-10-11 (lane/w2-decide-algos, same-box A/B on main c8ef6c66d): 4-query KDE tile is the IDENTICAL default (kde taxi 0.343, istella 1.239 regression, avg 0.791); drop -D MOJOLEARN_IDN_KDE_QTILE, use -D MOJOLEARN_IDN_KDE_QTILE_OFF for the old path"
    # wave-2 w2-decide-linalg (2026-10-11): A/B on main c8ef6c66d, same box, one run per arm, NV==AMD digests (docs/TOMBSTONES.md,
    # docs/apple-fast/EXPERIMENTS.md, experiments/removed/*.patch). Promoted: MOJOLEARN_IDN_GFF_GRAM_TILED (ridge istella 0.917, taxi 0.973).
    comptime assert not is_defined["MOJOLEARN_IDN_GFF_GRAM_TILED"](), "promoted 2026-10-11 (lane/w2-decide-linalg, d9441e439): the tiled float-float leaf Gram is the IDENTICAL default (ridge istella 0.917 NV 0.857 / AMD 0.977, taxi 0.973; nv2 v0135/v0080, amd a1717/a1662); -D MOJOLEARN_IDN_GFF_GRAM_TILED_OFF restores gff_leaf_kernel"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_GS_SOLVE_ONE_LAUNCH.patch
    comptime assert not is_defined["MOJOLEARN_IDN_GS_SOLVE_ONE_LAUNCH"](), "removed 2026-10-11 (lane/w2-decide-linalg, fbf3d50a0): IDN_GS_SOLVE_ONE_LAUNCH one-launch Gram Cholesky/solve walks (C2) were NOISE: ols 0.997/1.036, ridge 1.008/0.961, ridge-cv/ard/bayesian-ridge 1.00 (nv2 v0136/v0081, amd a1718/a1663); see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_TSQR_RSTACK_TREE.patch
    comptime assert not is_defined["MOJOLEARN_IDN_TSQR_RSTACK_TREE"](), "removed 2026-10-11 (lane/w2-decide-linalg, 7d76f567e): IDN_TSQR_RSTACK_TREE stacked-R TSQR levels were SLOWER: 8 cells mean 1.087, every cell > 1 (lstsq 1.12/1.10, rsvd istella 1.196, svd taxi 1.128; nv2 v0093, amd a1675); see docs/TOMBSTONES.md"
    # Lane k1-delete-unverdicted-fast (2026-11-09): Apple FAST experiments with no verdict, never to be measured.
    # Restore: git apply experiments/removed/MOJOLEARN_SCOPED_GEMM_AUDIT.patch
    comptime assert not is_defined["MOJOLEARN_SCOPED_GEMM_AUDIT"](), "removed 2026-11-09 (lane k1-delete-unverdicted-fast): MOJOLEARN_SCOPED_GEMM_AUDIT, the reach audit of the Apple FAST scoped G1/G2 GEMM adapter (experiments/apple_fast/gemm/scoped_dispatch.mojo: scoped_gemm_flags / _count / _metadata and the _mojolearn_scoped_gemm_probe binding), never measured, deleted with the adapter; code at f59d913b0b; see docs/TOMBSTONES.md"
    comptime assert not is_defined["MOJOLEARN_SCOPED_GEMM_G1_TALL"](), "removed 2026-11-09 (lane k1-delete-unverdicted-fast): MOJOLEARN_SCOPED_GEMM_G1_TALL, the scoped G1 adapter for tall NN products (F01/F02), no verdict row (one-sample M3 B/A 0.48-1.21, mixed) and will not be measured; code at f59d913b0b; see docs/TOMBSTONES.md"
    comptime assert not is_defined["MOJOLEARN_SCOPED_GEMM_G1_DENSE"](), "removed 2026-11-09 (lane k1-delete-unverdicted-fast): MOJOLEARN_SCOPED_GEMM_G1_DENSE, the scoped G1 adapter for dense NN products (F01/F02), no verdict row and will not be measured; code at f59d913b0b; see docs/TOMBSTONES.md"
    comptime assert not is_defined["MOJOLEARN_SCOPED_GEMM_G1_GRAM"](), "removed 2026-11-09 (lane k1-delete-unverdicted-fast): MOJOLEARN_SCOPED_GEMM_G1_GRAM, the scoped G1 adapter for aliased self-Grams (F01/F02), no verdict row and will not be measured; code at f59d913b0b; see docs/TOMBSTONES.md"
    comptime assert not is_defined["MOJOLEARN_SCOPED_GEMM_G2_NARROW"](), "removed 2026-11-09 (lane k1-delete-unverdicted-fast): MOJOLEARN_SCOPED_GEMM_G2_NARROW, the scoped G2 32x32 adapter for narrow NT products (F01/F02), no verdict row and will not be measured; code at f59d913b0b; see docs/TOMBSTONES.md"
    comptime assert not is_defined["MOJOLEARN_SCOPED_GEMM_SPLIT"](), "removed 2026-11-09 (lane k1-delete-unverdicted-fast): MOJOLEARN_SCOPED_GEMM_SPLIT, the split-K form of the scoped adapter (F01/F02), no verdict row and will not be measured; code at f59d913b0b; see docs/TOMBSTONES.md"
    comptime assert not is_defined["MOJOLEARN_SCOPED_GEMM_PCA"](), "removed 2026-11-09 (lane k1-delete-unverdicted-fast): MOJOLEARN_SCOPED_GEMM_PCA, the scoped adapter on the PCA covariance (F01; HOLD-quality scoped-pca-fit-istella-q-v1, full PCA B/A 0.99 / 0.95 one sample), no verdict row and will not be measured; code at f59d913b0b; see docs/TOMBSTONES.md"
    comptime assert not is_defined["MOJOLEARN_LEGACY_NARROW_SCOPED_GEMM"](), "removed 2026-11-09 (lane k1-delete-unverdicted-fast): MOJOLEARN_LEGACY_NARROW_SCOPED_GEMM, the benchmark-tuned shape windows of the scoped adapter (removed as dimension targeting 2026-10-04), deleted with the adapter; code at f59d913b0b; see docs/TOMBSTONES.md"
    comptime assert not is_defined["MOJOLEARN_SOFTMAX_FAST_G2_NARROW"](), "removed 2026-11-09 (lane k1-delete-unverdicted-fast): MOJOLEARN_SOFTMAX_FAST_G2_NARROW, F05, the multiclass QN linear forward through the scoped G2 kernel (softmax_gemm_nt), no verdict row (one-sample M3 B/A 0.89-1.12, mixed) and will not be measured; gemm_nt is the one path; code at f59d913b0b; see docs/TOMBSTONES.md"
    comptime assert not is_defined["MOJOLEARN_SOFTMAX_G2_AUDIT"](), "removed 2026-11-09 (lane k1-delete-unverdicted-fast): MOJOLEARN_SOFTMAX_G2_AUDIT, the reach audit of MOJOLEARN_SOFTMAX_FAST_G2_NARROW (softmax_g2_state / _count / _last / _reset), deleted with it; code at f59d913b0b; see docs/TOMBSTONES.md"
    comptime assert not is_defined["MOJOLEARN_LEGACY_NARROW_SOFTMAX_G2"](), "removed 2026-11-09 (lane k1-delete-unverdicted-fast): MOJOLEARN_LEGACY_NARROW_SOFTMAX_G2, the benchmark-tuned window of MOJOLEARN_SOFTMAX_FAST_G2_NARROW, deleted with it; code at f59d913b0b; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_CLASSICAL_SGD_EPOCH_CONTIGUOUS.patch
    comptime assert not is_defined["MOJOLEARN_CLASSICAL_SGD_EPOCH_CONTIGUOUS"](), "removed 2026-11-09 (lane n3-sgd-minibatch-loops): CLASSICAL_SGD_EPOCH_CONTIGUOUS (a per-epoch n x d re-gather of X, y, sw into permuted copies) was default off and never measured on its own; the batch kernels read the rows through the epoch's order (docs/TOMBSTONES.md)"
    comptime assert not is_defined["MOJOLEARN_X_LINEAR_JACOBI_LEAD"](), "removed 2026-11-09 (lane n9b-device-svd-qr): MOJOLEARN_X_LINEAR_JACOBI_LEAD restored the lead-only cyclic Jacobi of the team eigh `t_jacobi_eig`, deleted with it; BayesianRidge takes the executor eigh op (x_decomp/eigh_ops.mojo eigh_op_dense); code at 5b5490960d; see docs/TOMBSTONES.md"
    # Lane expdel-03-linear (2026-11-15, docs/plans/EXPERIMENT_DELETIONS_2026-11-15.md): old paths of measured defaults.
    # Restore: git apply experiments/removed/MOJOLEARN_ARD_FAST_CLS1_PARTS_OFF.patch
    comptime assert not is_defined["MOJOLEARN_ARD_FAST_CLS1_PARTS_OFF"](), "removed 2026-11-15 (lane/expdel-03-linear): ARD_FAST_CLS1_PARTS_OFF (FAST+Apple rollback to the one-thread ARD partials) is gone; the block partials are the FAST+Apple path (ard taxi 14.3 -> 12.0 ms, KEPT 9decae29f); code recoverable at a656258c20"
    # Restore: git apply experiments/removed/MOJOLEARN_ARD_FAST_CLS1_BATCH_OFF.patch
    comptime assert not is_defined["MOJOLEARN_ARD_FAST_CLS1_BATCH_OFF"](), "removed 2026-11-15 (lane/expdel-03-linear): ARD_FAST_CLS1_BATCH_OFF (FAST+Apple rollback to one ARD witness check per iteration) is gone; the batched check is the FAST+Apple path (ard taxi 14.4 -> 8.3 ms with the bundle, KEPT 9decae29f); code recoverable at d30f8b724f"
    # Restore: git apply experiments/removed/MOJOLEARN_BAYES_FAST_CLS1_PARTS_OFF.patch
    comptime assert not is_defined["MOJOLEARN_BAYES_FAST_CLS1_PARTS_OFF"](), "removed 2026-11-15 (lane/expdel-03-linear): BAYES_FAST_CLS1_PARTS_OFF (FAST+Apple rollback to the one-thread BayesianRidge partials) is gone; the block partials are the FAST+Apple path (bayesian-ridge taxi -3%, KEPT 9decae29f); code recoverable at 3cadf0f84b"
    # Restore: git apply experiments/removed/MOJOLEARN_BAYES_FAST_CLS1_BATCH_OFF.patch
    comptime assert not is_defined["MOJOLEARN_BAYES_FAST_CLS1_BATCH_OFF"](), "removed 2026-11-15 (lane/expdel-03-linear): BAYES_FAST_CLS1_BATCH_OFF (FAST+Apple rollback to two synchronizations per BayesianRidge iteration) is gone; the batched loop is the FAST+Apple path (bayesian-ridge taxi 102 -> 15.3 ms with the bundle, KEPT 9decae29f); code recoverable at 9a688a11e2"
    # Restore: git apply experiments/removed/MOJOLEARN_RIDGE_FAST_CLS1_CODES_OFF.patch
    comptime assert not is_defined["MOJOLEARN_RIDGE_FAST_CLS1_CODES_OFF"](), "removed 2026-11-15 (lane/expdel-03-linear): RIDGE_FAST_CLS1_CODES_OFF (FAST+Apple rollback to host-built RidgeClassifier targets) is gone; the device targets are the FAST+Apple path (ridge-clf taxi 120 -> 19.0 ms, KEPT 9decae29f); code recoverable at cb5f47564f"
    # Restore: git apply experiments/removed/MOJOLEARN_X_LINEAR_LARS_FAST_GRAM_OFF.patch
    comptime assert not is_defined["MOJOLEARN_X_LINEAR_LARS_FAST_GRAM_OFF"](), "removed 2026-11-15 (lane/expdel-03-linear): X_LINEAR_LARS_FAST_GRAM_OFF (FAST+Apple rollback to the moments path for Lars / LassoLars) is gone; the grid Gram is the FAST+Apple path (lars taxi 59.1 -> 15.1 ms, KEPT c5e1bbeb6); code recoverable at 8d8b814530"
    # Restore: git apply experiments/removed/MOJOLEARN_X_LINEAR_RIDGE_FAST_GRAM_OFF.patch
    comptime assert not is_defined["MOJOLEARN_X_LINEAR_RIDGE_FAST_GRAM_OFF"](), "removed 2026-11-15 (lane/expdel-03-linear): X_LINEAR_RIDGE_FAST_GRAM_OFF (FAST+Apple rollback to main's Ridge moments and kf_cells_kernel) is gone; the grid Gram is the FAST+Apple path (ridge-cv taxi 3,614 -> 37 ms, KEPT c5e1bbeb6); code recoverable at 9eb96f91e6"
    # Restore: git apply experiments/removed/MOJOLEARN_SGD_PERC_AVG.patch
    comptime assert not is_defined["MOJOLEARN_SGD_PERC_AVG"](), "removed 2026-11-15 (lane/expdel-03-linear): SGD_PERC_AVG (minibatch Perceptron epoch-end averaging) is gone; it lost on quality (rab5-perc perceptron taxi accuracy 0.76219 -> 0.74097); the last iterate is the only form; code recoverable at 8b2e35c54e"
    # Restore: git apply experiments/removed/MOJOLEARN_SGDOC_FAST_TAIL_OFF.patch
    comptime assert not is_defined["MOJOLEARN_SGDOC_FAST_TAIL_OFF"](), "removed 2026-11-15 (lane/expdel-03-linear): SGDOC_FAST_TAIL_OFF (FAST+Apple rollback to the full SGDOneClassSVM per-sample run) is gone; the tail replay is the FAST+Apple path (taxi 42142.4 -> 153.7 ms, istella 75533.9 -> 407.0 ms, w2-sgdoc-q PASS); code recoverable at 43214c2e1e"
    # Restore: git apply experiments/removed/MOJOLEARN_HUBER_FAST_BLOCK512_OFF.patch
    comptime assert not is_defined["MOJOLEARN_HUBER_FAST_BLOCK512_OFF"](), "removed 2026-11-15 (lane/expdel-03-linear): HUBER_FAST_BLOCK512_OFF (FAST+Apple rollback to FOLD_BLOCK partials in the device L-BFGS Huber fit) is gone; 512-row partials are the only form (huber taxi 192.0 -> 161.6 ms, r2 equal, rab19-huber512); code recoverable at 246a53ea57"
    # Restore: git apply experiments/removed/MOJOLEARN_FAST_OLS_NORMAL_EQ_OFF.patch
    comptime assert not is_defined["MOJOLEARN_FAST_OLS_NORMAL_EQ_OFF"](), "removed 2026-11-15 (lane/expdel-03-linear): FAST_OLS_NORMAL_EQ_OFF (FAST+Apple rollback of LinearRegression to the blocked TSQR) is gone; the resident normal equations are the FAST+Apple path (ols taxi 286 -> 110 ms, istella 2,921 -> 842 ms, KEPT 94cb5ba59); code recoverable at ed9ef96a34"
    # Restore: git apply experiments/removed/MOJOLEARN_QN_FAST_BLOCKS_OFF.patch
    comptime assert not is_defined["MOJOLEARN_QN_FAST_BLOCKS_OFF"](), "removed 2026-11-15 (lane/expdel-03-linear): QN_FAST_BLOCKS_OFF (FAST+Apple rollback to the nine-launch QN evaluation) is gone; the two-launch block evaluation is the FAST+Apple path (logreg istella 3,893 -> 3,612 ms, linearsvr 913 -> 219, KEPT 0ae9c28cd); code recoverable at 1385d5e6ff"
    # Restore: git apply experiments/removed/MOJOLEARN_PREP_FAST_MINMAX_OFF.patch
    comptime assert not is_defined["MOJOLEARN_PREP_FAST_MINMAX_OFF"](), "removed 2026-11-15 (lane/expdel-03-linear): PREP_FAST_MINMAX_OFF (FAST+Apple rollback of MinMaxScaler to extrema_chunks_kernel) is gone; the row-tiled extrema are the FAST+Apple path (minmax-scaler istella 152 -> 106 ms, KEPT f419ea9f1); code recoverable at 6c544e9855"
    # Restore: git apply experiments/removed/MOJOLEARN_X_PREP_FAST_STAGED_OUT_OFF.patch
    comptime assert not is_defined["MOJOLEARN_X_PREP_FAST_STAGED_OUT_OFF"](), "removed 2026-11-15 (lane/expdel-03-linear): X_PREP_FAST_STAGED_OUT_OFF (FAST+Apple rollback to raw host-pointer copies of the x_prep output) is gone; the staged download is the FAST+Apple path (label-binarizer taxi 563 -> 345 ms, digests the same); code recoverable at 509aa1c3a5"
    # Restore: git apply experiments/removed/MOJOLEARN_X_PREP_FAST_TE_GLOBAL_OFF.patch
    comptime assert not is_defined["MOJOLEARN_X_PREP_FAST_TE_GLOBAL_OFF"](), "removed 2026-11-15 (lane/expdel-03-linear): X_PREP_FAST_TE_GLOBAL_OFF (FAST+Apple rollback of TargetEncoder te_global to one thread per unit) is gone; the threadgroup tree is the FAST+Apple path (target-encoder taxi 201.4 -> 90.2 ms, digest identical); code recoverable at d1ba1439d6"
    # Restore: git apply experiments/removed/MOJOLEARN_X_PREP_FAST_TE_ENC_OFF.patch
    comptime assert not is_defined["MOJOLEARN_X_PREP_FAST_TE_ENC_OFF"](), "removed 2026-11-15 (lane/expdel-03-linear): X_PREP_FAST_TE_ENC_OFF (FAST+Apple rollback of TargetEncoder te_enc to one thread per bucket) is gone; the threadgroup form is the FAST+Apple path (target-encoder taxi 203.1 -> 148.7 ms, digest identical); code recoverable at 3827e5294b"
    # Restore: git apply experiments/removed/MOJOLEARN_PREP2_FAST_EIGH_BLOCK_OFF.patch
    comptime assert not is_defined["MOJOLEARN_PREP2_FAST_EIGH_BLOCK_OFF"](), "removed 2026-11-15 (lane/expdel-03-linear): PREP2_FAST_EIGH_BLOCK_OFF (FAST+Apple rollback of the x_prep eigh stage to one thread per matrix) is gone; the threadgroup Jacobi is the FAST+Apple path (iterative-imputer taxi 218.6 -> 185.3 ms, digest identical); code recoverable at 4867560045"
    # Restore: git apply experiments/removed/MOJOLEARN_CD_FAST_GRID_GRAM_OFF.patch
    comptime assert not is_defined["MOJOLEARN_CD_FAST_GRID_GRAM_OFF"](), "removed 2026-11-15 (lane/expdel-03-linear): CD_FAST_GRID_GRAM_OFF (FAST+Apple rollback of the CD Gram to the staged copy + gemm_nt / cd_gram_partial_kernel) is gone; the grid Gram is the only CD_FAST_GRAM product (lasso istella 264 -> 149 ms, KEPT 0ae9c28cd); code recoverable at e592fb6e9f"
    # Restore: git apply experiments/removed/MOJOLEARN_CD_FAST_ROWMAJOR_OFF.patch
    comptime assert not is_defined["MOJOLEARN_CD_FAST_ROWMAJOR_OFF"](), "removed 2026-11-15 (lane/expdel-03-linear): CD_FAST_ROWMAJOR_OFF (FAST+Apple rollback of the CD grid Gram to the transposed, centered design) is gone; the row-major tiles are the FAST+Apple path (lasso istella 264 -> 149 ms with GRID_GRAM, KEPT 0ae9c28cd); code recoverable at 7d40f892a7"
    # Lane expdel2-06-base (2026-11-15): the C13-C28 classical arms with no recorded A/B and the _OFF rollbacks of measured
    # IDENTICAL defaults read by experiments/classical_identical_ideas/linear_controls.mojo (docs/plans/EXPERIMENT_DELETIONS_V2_2026-11-15.md).
    # Restore: git apply experiments/removed/MOJOLEARN_CLASSICAL_C13_LOGCV_WEIGHTS.patch
    comptime assert not is_defined["MOJOLEARN_CLASSICAL_C13_LOGCV_WEIGHTS"](), "removed 2026-11-15 (lane/expdel2-06-base): CLASSICAL_C13_LOGCV_WEIGHTS (Linear (weighted LogisticRegressionCV fold weight sums, IDENTICAL)) was unmeasured opt-in arm; the opt-in arm is gone and the default path is unchanged; code recoverable at 337c096a5b; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_CLASSICAL_C14_GROUP_RHS.patch
    comptime assert not is_defined["MOJOLEARN_CLASSICAL_C14_GROUP_RHS"](), "removed 2026-11-15 (lane/expdel2-06-base): CLASSICAL_C14_GROUP_RHS (Linear / decomposition (triangular solves in groups of four RHS, IDENTICAL)) was unmeasured opt-in arm; the opt-in arm is gone and the default path is unchanged; code recoverable at 9b13c93ee0; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_CLASSICAL_C15_FACTOR_SOLVE.patch
    comptime assert not is_defined["MOJOLEARN_CLASSICAL_C15_FACTOR_SOLVE"](), "removed 2026-11-15 (lane/expdel2-06-base): CLASSICAL_C15_FACTOR_SOLVE (Linear (ARD posterior mean from the Cholesky factor, IDENTICAL)) was unmeasured opt-in arm; the opt-in arm is gone and the default path is unchanged; code recoverable at 72270680f8; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_CLASSICAL_C16_GLM_FUSED.patch
    comptime assert not is_defined["MOJOLEARN_CLASSICAL_C16_GLM_FUSED"](), "removed 2026-11-15 (lane/expdel2-06-base): CLASSICAL_C16_GLM_FUSED (Linear (GLM loss, gradient and Hessian in one pass, IDENTICAL)) was unmeasured opt-in arm; the opt-in arm is gone and the default path is unchanged; code recoverable at 2a301a3aa3; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_CLASSICAL_C17_LS_TRIALS.patch
    comptime assert not is_defined["MOJOLEARN_CLASSICAL_C17_LS_TRIALS"](), "removed 2026-11-15 (lane/expdel2-06-base): CLASSICAL_C17_LS_TRIALS (Linear (quasi-Newton line search with two speculative trials, IDENTICAL)) was unmeasured opt-in arm; the opt-in arm is gone and the default path is unchanged; code recoverable at 10e3cad6a6; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_CLASSICAL_C17_OVR.patch
    comptime assert not is_defined["MOJOLEARN_CLASSICAL_C17_OVR"](), "removed 2026-11-15 (lane/expdel2-06-base): CLASSICAL_C17_OVR (Linear (SGD one-vs-rest in waves of four, IDENTICAL)) was unmeasured opt-in arm; the opt-in arm is gone and the default path is unchanged; code recoverable at 8c1b568a83; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_CLASSICAL_C18_RESIDUAL_NEXT.patch
    comptime assert not is_defined["MOJOLEARN_CLASSICAL_C18_RESIDUAL_NEXT"](), "removed 2026-11-15 (lane/expdel2-06-base): CLASSICAL_C18_RESIDUAL_NEXT (Linear (CD next-coordinate residual forwarding, IDENTICAL)) was unqualified opt-in arm, neutral; the opt-in arm is gone and the default path is unchanged; code recoverable at 7ca2a3400c; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_CLASSICAL_C19_SGD_CHUNK.patch
    comptime assert not is_defined["MOJOLEARN_CLASSICAL_C19_SGD_CHUNK"](), "removed 2026-11-15 (lane/expdel2-06-base): CLASSICAL_C19_SGD_CHUNK (Linear (SGD ordered samples per launch sweep, IDENTICAL)) was unmeasured opt-in arm; the opt-in arm is gone and the default path is unchanged; code recoverable at 6690c70fa8; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_CLASSICAL_ENETCV_FOLD_BLOCKS.patch
    comptime assert not is_defined["MOJOLEARN_CLASSICAL_ENETCV_FOLD_BLOCKS"](), "removed 2026-11-15 (lane/expdel2-06-base): CLASSICAL_ENETCV_FOLD_BLOCKS (Linear (LassoCV / ElasticNetCV fold-block chunk count arms, IDENTICAL)) was losing arms =16|32; the opt-in arm is gone and the default path is unchanged; code recoverable at 4eb80b4abf; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_CLASSICAL_ENETCV_FOLD_BLOCKS_OFF.patch
    comptime assert not is_defined["MOJOLEARN_CLASSICAL_ENETCV_FOLD_BLOCKS_OFF"](), "removed 2026-11-15 (lane/expdel2-06-base): CLASSICAL_ENETCV_FOLD_BLOCKS_OFF (Linear (LassoCV / ElasticNetCV staged means + Gram, IDENTICAL old path)) was old path, superseded by the measured IDENTICAL default; the measured IDENTICAL default is the only path; code recoverable at a483990d2f; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_CLASSICAL_ENETCV_SCORE_BLOCKS.patch
    comptime assert not is_defined["MOJOLEARN_CLASSICAL_ENETCV_SCORE_BLOCKS"](), "removed 2026-11-15 (lane/expdel2-06-base): CLASSICAL_ENETCV_SCORE_BLOCKS (Linear (LassoCV / ElasticNetCV score rows per block arm, IDENTICAL)) was losing arm =4096; the opt-in arm is gone and the default path is unchanged; code recoverable at 7d1eee8179; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_CLASSICAL_ENETCV_SCORE_BLOCKS_OFF.patch
    comptime assert not is_defined["MOJOLEARN_CLASSICAL_ENETCV_SCORE_BLOCKS_OFF"](), "removed 2026-11-15 (lane/expdel2-06-base): CLASSICAL_ENETCV_SCORE_BLOCKS_OFF (Linear (LassoCV / ElasticNetCV 4096-row score blocks, IDENTICAL old path)) was old path, superseded by the measured IDENTICAL default; the measured IDENTICAL default is the only path; code recoverable at 6ac3a7434d; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_CLASSICAL_LINEAR_GRAM_SOLVE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_CLASSICAL_LINEAR_GRAM_SOLVE_OFF"](), "removed 2026-11-15 (lane/expdel2-06-base): CLASSICAL_LINEAR_GRAM_SOLVE_OFF (Linear (OLS / Ridge incumbent routes beside the resident Gram, IDENTICAL old path)) was old path, superseded by the measured IDENTICAL default; the measured IDENTICAL default is the only path; code recoverable at 439c466840; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_CLASSICAL_RIDGECV_FF_BLOCKED.patch
    comptime assert not is_defined["MOJOLEARN_CLASSICAL_RIDGECV_FF_BLOCKED"](), "removed 2026-11-15 (lane/expdel2-06-base): CLASSICAL_RIDGECV_FF_BLOCKED (Linear (RidgeCV float-float fallback in block partials, IDENTICAL)) was unmeasured opt-in arm; the opt-in arm is gone and the default path is unchanged; code recoverable at 49c61a4093; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_RIDGE_RESIDENT_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_RIDGE_RESIDENT_OFF"](), "removed 2026-11-15 (lane/expdel2-05-base): IDN_RIDGE_RESIDENT_OFF (rollback of the Ridge resident eig fallback to status 1 and the Python glue) is gone; the resident fallback is the IDENTICAL path"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_GRAM_FF_FALLBACK_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_GRAM_FF_FALLBACK_OFF"](), "removed 2026-11-15 (lane/expdel2-05-base): IDN_GRAM_FF_FALLBACK_OFF (rollback of the Ridge float-float Gram second chance to the direct eig route) is gone; the float-float fallback is the IDENTICAL path"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_GFF_GRAM_TILED_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_GFF_GRAM_TILED_OFF"](), "removed 2026-11-15 (lane/expdel2-05-base): IDN_GFF_GRAM_TILED_OFF (rollback of the float-float leaf Gram to the per-cell gff_leaf_kernel) is gone; the register-tiled leaf Gram is the IDENTICAL path"
    # Restore: git apply experiments/removed/MOJOLEARN_CLASSICAL_C08_UNIQUE_SCAN_OFF.patch
    comptime assert not is_defined["MOJOLEARN_CLASSICAL_C08_UNIQUE_SCAN_OFF"](), "removed 2026-11-15 (lane/expdel2-07-base): CLASSICAL_C08_UNIQUE_SCAN_OFF (Prep (unique_cols one-thread-per-column walk, IDENTICAL old path)) was old path, superseded by the measured IDENTICAL default; the default is the only path; code recoverable at 9b76c98115; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_CLASSICAL_C02_LINEAR_PAIR.patch
    comptime assert not is_defined["MOJOLEARN_CLASSICAL_C02_LINEAR_PAIR"](), "removed 2026-11-15 (lane/expdel2-07-base): CLASSICAL_C02_LINEAR_PAIR (Linear (OLS / Ridge independent C02 linear sums, IDENTICAL)) was unmeasured opt-in arm; the opt-in arm is gone and the default path is unchanged; code recoverable at c95b22c391; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_CLASSICAL_C05_OLS_PHASE.patch
    comptime assert not is_defined["MOJOLEARN_CLASSICAL_C05_OLS_PHASE"](), "removed 2026-11-15 (lane/expdel2-07-base): CLASSICAL_C05_OLS_PHASE (Linear (invocation-owned OLS covariance / inverse phase storage, IDENTICAL)) was unmeasured opt-in arm; the opt-in arm is gone and the default path is unchanged; code recoverable at eb67cbe21f; see docs/TOMBSTONES.md"
    # Lane expdel2-08-base (2026-11-15): unmeasured arms and superseded old paths deleted, docs/plans/EXPERIMENT_DELETIONS_V2_2026-11-15.md (docs/TOMBSTONES.md).
    # Restore: git apply experiments/removed/MOJOLEARN_MULTIOUT_RIDGE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_MULTIOUT_RIDGE_OFF"](), "removed 2026-11-15 (lane/expdel2-08-base): MULTIOUT_RIDGE_OFF (old path of the FAST Apple default) is gone; the default is kept; code at 7866bd9303; restore: experiments/removed/MOJOLEARN_MULTIOUT_RIDGE_OFF.patch"
    # Lane expdel2-16-decomposition (2026-11-15): unmeasured arms and superseded old paths deleted, docs/plans/EXPERIMENT_DELETIONS_V2_2026-11-15.md (docs/TOMBSTONES.md).
    # Restore: git apply experiments/removed/MOJOLEARN_CHOL_FAST_DEVIO_OFF.patch
    comptime assert not is_defined["MOJOLEARN_CHOL_FAST_DEVIO_OFF"](), "removed 2026-11-15 (lane/expdel2-16-decomposition): CHOL_FAST_DEVIO_OFF (unmeasured old path beside the FAST Apple and IDENTICAL default) is gone; the default is kept; code at 70d6bad292; restore: experiments/removed/MOJOLEARN_CHOL_FAST_DEVIO_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_CHOL_HOST_SCALAR.patch
    comptime assert not is_defined["MOJOLEARN_CHOL_HOST_SCALAR"](), "removed 2026-11-15 (lane/expdel2-16-decomposition): CHOL_HOST_SCALAR (unmeasured opt-in before arm) is gone; the default is kept; code at 61bd06b986; restore: experiments/removed/MOJOLEARN_CHOL_HOST_SCALAR.patch"
    # Lane expdel2-20-linear (2026-11-15): unmeasured arms deleted, docs/plans/EXPERIMENT_DELETIONS_V2_2026-11-15.md (docs/TOMBSTONES.md).
    # MOJOLEARN_AFCL_L05: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_CD_FAST_GRAM_OFF.patch
    comptime assert not is_defined["MOJOLEARN_CD_FAST_GRAM_OFF"](), "removed 2026-11-15 (lane/expdel2-20-linear): CD_FAST_GRAM_OFF (old path, no A/B recorded) is gone; the default is kept; code at d75e2fbb7a; restore: experiments/removed/MOJOLEARN_CD_FAST_GRAM_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_CD_FUSED_OFF.patch
    comptime assert not is_defined["MOJOLEARN_CD_FUSED_OFF"](), "removed 2026-11-15 (lane/expdel2-20-linear): CD_FUSED_OFF (old path, no A/B recorded) is gone; the default is kept; code at 1048118370; restore: experiments/removed/MOJOLEARN_CD_FUSED_OFF.patch"
    # MOJOLEARN_CD_GRAM_BLOCKS: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-classical: Apple fast paused, not abandoned; opt-in, default off; see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_CD_IDN_GRAM_COLS_128.patch
    comptime assert not is_defined["MOJOLEARN_CD_IDN_GRAM_COLS_128"](), "removed 2026-11-15 (lane/expdel2-20-linear): CD_IDN_GRAM_COLS_128 (opt-in arm, no A/B recorded) is gone; the default is kept; code at 72da417819; restore: experiments/removed/MOJOLEARN_CD_IDN_GRAM_COLS_128.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_CD_IDN_GRAM_COLS_256.patch
    comptime assert not is_defined["MOJOLEARN_CD_IDN_GRAM_COLS_256"](), "removed 2026-11-15 (lane/expdel2-20-linear): CD_IDN_GRAM_COLS_256 (opt-in arm, no A/B recorded) is gone; the default is kept; code at 72da417819; restore: experiments/removed/MOJOLEARN_CD_IDN_GRAM_COLS_256.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_CD_IDN_GRAM_COST_RULE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_CD_IDN_GRAM_COST_RULE_OFF"](), "removed 2026-11-15 (lane/expdel2-20-linear): CD_IDN_GRAM_COST_RULE_OFF (old path, no A/B recorded) is gone; the default is kept; code at 72da417819; restore: experiments/removed/MOJOLEARN_CD_IDN_GRAM_COST_RULE_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_CD_IDN_GRAM_OFF.patch
    comptime assert not is_defined["MOJOLEARN_CD_IDN_GRAM_OFF"](), "removed 2026-11-15 (lane/expdel2-20-linear): CD_IDN_GRAM_OFF (old path, no A/B recorded) is gone; the default is kept; code at 72da417819; restore: experiments/removed/MOJOLEARN_CD_IDN_GRAM_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_CD_IDN_GRAM_WIDE.patch
    comptime assert not is_defined["MOJOLEARN_CD_IDN_GRAM_WIDE"](), "removed 2026-11-15 (lane/expdel2-20-linear): CD_IDN_GRAM_WIDE (opt-in arm, no A/B recorded) is gone; the default is kept; code at 72da417819; restore: experiments/removed/MOJOLEARN_CD_IDN_GRAM_WIDE.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_CD_THREE_LAUNCH_OFF.patch
    comptime assert not is_defined["MOJOLEARN_CD_THREE_LAUNCH_OFF"](), "removed 2026-11-15 (lane/expdel2-20-linear): CD_THREE_LAUNCH_OFF (old path, no A/B recorded) is gone; the default is kept; code at b7039f2e07; restore: experiments/removed/MOJOLEARN_CD_THREE_LAUNCH_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_CD_TWO_STEP_OFF.patch
    comptime assert not is_defined["MOJOLEARN_CD_TWO_STEP_OFF"](), "removed 2026-11-15 (lane/expdel2-20-linear): CD_TWO_STEP_OFF (old path, no A/B recorded) is gone; the default is kept; code at f2dad24e5a; restore: experiments/removed/MOJOLEARN_CD_TWO_STEP_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_QN_EXACT_TRIALS.patch
    comptime assert not is_defined["MOJOLEARN_IDN_QN_EXACT_TRIALS"](), "removed 2026-11-15 (lane/expdel2-20-linear): IDN_QN_EXACT_TRIALS (opt-in arm, no A/B recorded) is gone; the default is kept; code at fb9523bf09; restore: experiments/removed/MOJOLEARN_IDN_QN_EXACT_TRIALS.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_LEGACY_NARROW_FX_D.patch
    comptime assert not is_defined["MOJOLEARN_LEGACY_NARROW_FX_D"](), "removed 2026-11-15 (lane/expdel2-20-linear): LEGACY_NARROW_FX_D (old path, no A/B recorded) is gone; the default is kept; code at fd8a1ee12a; restore: experiments/removed/MOJOLEARN_LEGACY_NARROW_FX_D.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_LEGACY_NARROW_QN_ALL.patch
    comptime assert not is_defined["MOJOLEARN_LEGACY_NARROW_QN_ALL"](), "removed 2026-11-15 (lane/expdel2-20-linear): LEGACY_NARROW_QN_ALL (old path, no A/B recorded) is gone; the default is kept; code at e1833e6a98; restore: experiments/removed/MOJOLEARN_LEGACY_NARROW_QN_ALL.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_LSVR_ALL_OFF.patch
    comptime assert not is_defined["MOJOLEARN_LSVR_ALL_OFF"](), "removed 2026-11-15 (lane/expdel2-20-linear): LSVR_ALL_OFF (old path) is gone; the default is kept; code at ff4bf11f19; restore: experiments/removed/MOJOLEARN_LSVR_ALL_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_LSVR_EVAL_SLIM_OFF.patch
    comptime assert not is_defined["MOJOLEARN_LSVR_EVAL_SLIM_OFF"](), "removed 2026-11-15 (lane/expdel2-20-linear): LSVR_EVAL_SLIM_OFF (old path of a one-vendor promotion) is gone; the default is kept; code at e9b815205e; restore: experiments/removed/MOJOLEARN_LSVR_EVAL_SLIM_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_LSVR_LINESEARCH_BATCH_OFF.patch
    comptime assert not is_defined["MOJOLEARN_LSVR_LINESEARCH_BATCH_OFF"](), "removed 2026-11-15 (lane/expdel2-20-linear): LSVR_LINESEARCH_BATCH_OFF (old path of a one-vendor promotion) is gone; the default is kept; code at 07f25ee298; restore: experiments/removed/MOJOLEARN_LSVR_LINESEARCH_BATCH_OFF.patch"
    # MOJOLEARN_OLS_FAST_SMALL_NT: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-classical: Apple fast paused, not abandoned; opt-in, default off; see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_PURITY2_2_OFF.patch
    comptime assert not is_defined["MOJOLEARN_PURITY2_2_OFF"](), "removed 2026-11-15 (lane/expdel2-20-linear): PURITY2_2_OFF (old path, no A/B recorded) is gone; the default is kept; code at eb10315ddd; restore: experiments/removed/MOJOLEARN_PURITY2_2_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_PURITY_1_OFF.patch
    comptime assert not is_defined["MOJOLEARN_PURITY_1_OFF"](), "removed 2026-11-15 (lane/expdel2-20-linear): PURITY_1_OFF (old path, no A/B recorded) is gone; the default is kept; code at a3181ec16f; restore: experiments/removed/MOJOLEARN_PURITY_1_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_QN_DEV_ARGMAX_OFF.patch
    comptime assert not is_defined["MOJOLEARN_QN_DEV_ARGMAX_OFF"](), "removed 2026-11-15 (lane/expdel2-20-linear): QN_DEV_ARGMAX_OFF (old path, no A/B recorded) is gone; the default is kept; code at 4218560b53; restore: experiments/removed/MOJOLEARN_QN_DEV_ARGMAX_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_QN_FAST_XTDZ_OFF.patch
    comptime assert not is_defined["MOJOLEARN_QN_FAST_XTDZ_OFF"](), "removed 2026-11-15 (lane/expdel2-20-linear): QN_FAST_XTDZ_OFF (old path) is gone; the default is kept; code at 83e86f6d61; restore: experiments/removed/MOJOLEARN_QN_FAST_XTDZ_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_QN_IDN_FUSED_OFF.patch
    comptime assert not is_defined["MOJOLEARN_QN_IDN_FUSED_OFF"](), "removed 2026-11-15 (lane/expdel2-20-linear): QN_IDN_FUSED_OFF (old path, no A/B recorded) is gone; the default is kept; code at 42a2855cbe; restore: experiments/removed/MOJOLEARN_QN_IDN_FUSED_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_QN_IDN_SLIM_OFF.patch
    comptime assert not is_defined["MOJOLEARN_QN_IDN_SLIM_OFF"](), "removed 2026-11-15 (lane/expdel2-20-linear): QN_IDN_SLIM_OFF (old path, no A/B recorded) is gone; the default is kept; code at cf050feb2f; restore: experiments/removed/MOJOLEARN_QN_IDN_SLIM_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_QN_OVR_ONE_UPLOAD_OFF.patch
    comptime assert not is_defined["MOJOLEARN_QN_OVR_ONE_UPLOAD_OFF"](), "removed 2026-11-15 (lane/expdel2-20-linear): QN_OVR_ONE_UPLOAD_OFF (old path, no A/B recorded) is gone; the default is kept; code at 021105e119; restore: experiments/removed/MOJOLEARN_QN_OVR_ONE_UPLOAD_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_QN_TILED_OFF.patch
    comptime assert not is_defined["MOJOLEARN_QN_TILED_OFF"](), "removed 2026-11-15 (lane/expdel2-20-linear): QN_TILED_OFF (old path) is gone; the default is kept; code at da95043b31; restore: experiments/removed/MOJOLEARN_QN_TILED_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_RIDGE_NO_U_OFF.patch
    comptime assert not is_defined["MOJOLEARN_RIDGE_NO_U_OFF"](), "removed 2026-11-15 (lane/expdel2-20-linear): RIDGE_NO_U_OFF (old path of a one-vendor promotion) is gone; the default is kept; code at 620c4ce37c; restore: experiments/removed/MOJOLEARN_RIDGE_NO_U_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_RIDGE_RESIDENT_OFF.patch
    comptime assert not is_defined["MOJOLEARN_RIDGE_RESIDENT_OFF"](), "removed 2026-11-15 (lane/expdel2-20-linear): RIDGE_RESIDENT_OFF (old path of a one-vendor promotion) is gone; the default is kept; code at 3fcdcb78f6; restore: experiments/removed/MOJOLEARN_RIDGE_RESIDENT_OFF.patch"
    # Lane expdel2-22-linear (2026-11-15): unmeasured arms and superseded old paths deleted, docs/plans/EXPERIMENT_DELETIONS_V2_2026-11-15.md (docs/TOMBSTONES.md).
    # MOJOLEARN_AFCL_P01: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-classical: Apple fast paused, not abandoned; opt-in, default off; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFCL_P02: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-classical: Apple fast paused, not abandoned; opt-in, default off; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFCL_P03: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-classical: Apple fast paused, not abandoned; opt-in, default off; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFCL_P05: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-classical: Apple fast paused, not abandoned; opt-in, default off; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFCL_P06: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-classical: Apple fast paused, not abandoned; opt-in, default off; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFCL_P13: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-classical: Apple fast paused, not abandoned; opt-in, default off; see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_CLASSICAL_TE_BLOCKED_FOLD.patch
    comptime assert not is_defined["MOJOLEARN_CLASSICAL_TE_BLOCKED_FOLD"](), "removed 2026-11-15 (lane/expdel2-22-linear): CLASSICAL_TE_BLOCKED_FOLD (opt-in arm) is gone; the default path is the only one; code at 74754fe939; restore: experiments/removed/MOJOLEARN_CLASSICAL_TE_BLOCKED_FOLD.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_DRADIX_RANGE_PASSES_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_DRADIX_RANGE_PASSES_OFF"](), "removed 2026-11-15 (lane/expdel2-22-linear): IDN_DRADIX_RANGE_PASSES_OFF (old path) is gone; the default is kept; code at 26ab6baddf; restore: experiments/removed/MOJOLEARN_IDN_DRADIX_RANGE_PASSES_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_II_CONV_TREE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_II_CONV_TREE_OFF"](), "removed 2026-11-15 (lane/expdel2-22-linear): IDN_II_CONV_TREE_OFF (old path) is gone; the default is kept; code at ef6c79d70a; restore: experiments/removed/MOJOLEARN_IDN_II_CONV_TREE_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_II_GRAM_TILE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_II_GRAM_TILE_OFF"](), "removed 2026-11-15 (lane/expdel2-22-linear): IDN_II_GRAM_TILE_OFF (old path) is gone; the default is kept; code at 079ed8c3e7; restore: experiments/removed/MOJOLEARN_IDN_II_GRAM_TILE_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_LABEL_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_LABEL_OFF"](), "removed 2026-11-15 (lane/expdel2-22-linear): IDN_LABEL_OFF (old path) is gone; the default is kept; code at 886a739db7; restore: experiments/removed/MOJOLEARN_IDN_LABEL_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_NB_CAT_ATOMIC_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_NB_CAT_ATOMIC_OFF"](), "removed 2026-11-15 (lane/expdel2-22-linear): IDN_NB_CAT_ATOMIC_OFF (old path) is gone; the default is kept; code at 7cf272e256; restore: experiments/removed/MOJOLEARN_IDN_NB_CAT_ATOMIC_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_RR_EIGH_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_RR_EIGH_OFF"](), "removed 2026-11-15 (lane/expdel2-22-linear): IDN_RR_EIGH_OFF (old path) is gone; the default is kept; code at 43f37f93b7; restore: experiments/removed/MOJOLEARN_IDN_RR_EIGH_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_RR_MIN_16.patch
    comptime assert not is_defined["MOJOLEARN_IDN_RR_MIN_16"](), "removed 2026-11-15 (lane/expdel2-22-linear): IDN_RR_MIN_16 (opt-in arm) is gone; the default path is the only one; code at 52d1c6b87e; restore: experiments/removed/MOJOLEARN_IDN_RR_MIN_16.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_RR_MIN_48.patch
    comptime assert not is_defined["MOJOLEARN_IDN_RR_MIN_48"](), "removed 2026-11-15 (lane/expdel2-22-linear): IDN_RR_MIN_48 (opt-in arm) is gone; the default path is the only one; code at 9086d9b3d1; restore: experiments/removed/MOJOLEARN_IDN_RR_MIN_48.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_RR_MIN_8.patch
    comptime assert not is_defined["MOJOLEARN_IDN_RR_MIN_8"](), "removed 2026-11-15 (lane/expdel2-22-linear): IDN_RR_MIN_8 (opt-in arm) is gone; the default path is the only one; code at e71d35eb21; restore: experiments/removed/MOJOLEARN_IDN_RR_MIN_8.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_RR_MIN_96.patch
    comptime assert not is_defined["MOJOLEARN_IDN_RR_MIN_96"](), "removed 2026-11-15 (lane/expdel2-22-linear): IDN_RR_MIN_96 (opt-in arm) is gone; the default path is the only one; code at 688ea8608f; restore: experiments/removed/MOJOLEARN_IDN_RR_MIN_96.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_TE_ENC_TREE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_TE_ENC_TREE_OFF"](), "removed 2026-11-15 (lane/expdel2-22-linear): IDN_TE_ENC_TREE_OFF (old path) is gone; the default is kept; code at 955fcda5ec; restore: experiments/removed/MOJOLEARN_IDN_TE_ENC_TREE_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_TE_GLOBAL_TREE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_TE_GLOBAL_TREE_OFF"](), "removed 2026-11-15 (lane/expdel2-22-linear): IDN_TE_GLOBAL_TREE_OFF (old path) is gone; the default is kept; code at 44b08dfc1e; restore: experiments/removed/MOJOLEARN_IDN_TE_GLOBAL_TREE_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_XPREP_RADIX_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_XPREP_RADIX_OFF"](), "removed 2026-11-15 (lane/expdel2-22-linear): IDN_XPREP_RADIX_OFF (old path) is gone; the default is kept; code at 60adf729d6; restore: experiments/removed/MOJOLEARN_IDN_XPREP_RADIX_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_LABEL_DIRECT_OFF.patch
    comptime assert not is_defined["MOJOLEARN_LABEL_DIRECT_OFF"](), "removed 2026-11-15 (lane/expdel2-22-linear): LABEL_DIRECT_OFF (old path) is gone; the default is kept; code at 2ec4e389be; restore: experiments/removed/MOJOLEARN_LABEL_DIRECT_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_LDAQDA_DEC_TILE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_LDAQDA_DEC_TILE_OFF"](), "removed 2026-11-15 (lane/expdel2-22-linear): LDAQDA_DEC_TILE_OFF (old path) is gone; the default is kept; code at d64015f87d; restore: experiments/removed/MOJOLEARN_LDAQDA_DEC_TILE_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_LDAQDA_PAR_STAGES_OFF.patch
    comptime assert not is_defined["MOJOLEARN_LDAQDA_PAR_STAGES_OFF"](), "removed 2026-11-15 (lane/expdel2-22-linear): LDAQDA_PAR_STAGES_OFF (old path) is gone; the default is kept; code at 7decc17403; restore: experiments/removed/MOJOLEARN_LDAQDA_PAR_STAGES_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_LDAQDA_RR_EIGH_OFF.patch
    comptime assert not is_defined["MOJOLEARN_LDAQDA_RR_EIGH_OFF"](), "removed 2026-11-15 (lane/expdel2-22-linear): LDAQDA_RR_EIGH_OFF (old path) is gone; the default is kept; code at fc4f469cc4; restore: experiments/removed/MOJOLEARN_LDAQDA_RR_EIGH_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_LEGACY_NARROW_FASTPT_ROWS.patch
    comptime assert not is_defined["MOJOLEARN_LEGACY_NARROW_FASTPT_ROWS"](), "removed 2026-11-15 (lane/expdel2-22-linear): LEGACY_NARROW_FASTPT_ROWS (old path) is gone; the default is kept; code at b4c2d444d0; restore: experiments/removed/MOJOLEARN_LEGACY_NARROW_FASTPT_ROWS.patch"
    # MOJOLEARN_MI_CLF_RANKMAJOR: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-classical: Apple fast paused, not abandoned; opt-in, default off; see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_MI_CLF_RANKMAJOR_OFF.patch
    comptime assert not is_defined["MOJOLEARN_MI_CLF_RANKMAJOR_OFF"](), "removed 2026-11-15 (lane/expdel2-22-linear): MI_CLF_RANKMAJOR_OFF (old path) is gone; the default is kept; code at 31dc3269c3; restore: experiments/removed/MOJOLEARN_MI_CLF_RANKMAJOR_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_MI_REG_RANKMAJOR.patch
    comptime assert not is_defined["MOJOLEARN_MI_REG_RANKMAJOR"](), "removed 2026-11-15 (lane/expdel2-22-linear): MI_REG_RANKMAJOR (opt-in arm) is gone; the default path is the only one; code at 6a50f39dd3; restore: experiments/removed/MOJOLEARN_MI_REG_RANKMAJOR.patch"
    # MOJOLEARN_MI_REG_SORTCOUNT: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-classical: Apple fast paused, not abandoned; opt-in, default off; see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_MI_REG_TIES.patch
    comptime assert not is_defined["MOJOLEARN_MI_REG_TIES"](), "removed 2026-11-15 (lane/expdel2-22-linear): MI_REG_TIES (opt-in arm) is gone; the default path is the only one; code at 293c8f301c; restore: experiments/removed/MOJOLEARN_MI_REG_TIES.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_MI_REG_TIES_OFF.patch
    comptime assert not is_defined["MOJOLEARN_MI_REG_TIES_OFF"](), "removed 2026-11-15 (lane/expdel2-22-linear): MI_REG_TIES_OFF (old path) is gone; the default is kept; code at 8bf90f94c1; restore: experiments/removed/MOJOLEARN_MI_REG_TIES_OFF.patch"
    # MOJOLEARN_NB_CAT_ATOMIC: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-classical: Apple fast paused, not abandoned; opt-in, default off; see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_PT_SCORE.patch
    comptime assert not is_defined["MOJOLEARN_PT_SCORE"](), "removed 2026-11-15 (lane/expdel2-22-linear): PT_SCORE (opt-in arm) is gone; the default path is the only one; code at 1de4f202c5; restore: experiments/removed/MOJOLEARN_PT_SCORE.patch"
    # MOJOLEARN_SI_ONEPASS: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-classical: Apple fast paused, not abandoned; opt-in, default off; see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_SI_ONEPASS_OFF.patch
    comptime assert not is_defined["MOJOLEARN_SI_ONEPASS_OFF"](), "removed 2026-11-15 (lane/expdel2-22-linear): SI_ONEPASS_OFF (old path) is gone; the default is kept; code at 8193833d2e; restore: experiments/removed/MOJOLEARN_SI_ONEPASS_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_X_PREP_CLASS_COV_GRID_OFF.patch
    comptime assert not is_defined["MOJOLEARN_X_PREP_CLASS_COV_GRID_OFF"](), "removed 2026-11-15 (lane/expdel2-22-linear): X_PREP_CLASS_COV_GRID_OFF (old path) is gone; the default is kept; code at 518d83d2b0; restore: experiments/removed/MOJOLEARN_X_PREP_CLASS_COV_GRID_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_X_PREP_POOL_ARENA_OFF.patch
    comptime assert not is_defined["MOJOLEARN_X_PREP_POOL_ARENA_OFF"](), "removed 2026-11-15 (lane/expdel2-22-linear): X_PREP_POOL_ARENA_OFF (old path) is gone; the default is kept; code at cc8e7c0dfd; restore: experiments/removed/MOJOLEARN_X_PREP_POOL_ARENA_OFF.patch"
    # Lane expdel2-21-linear (2026-11-15): unmeasured arms and superseded old paths deleted, docs/plans/EXPERIMENT_DELETIONS_V2_2026-11-15.md (docs/TOMBSTONES.md).
    # MOJOLEARN_AFCL_L01: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFCL_L02: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFCL_L03: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFCL_L04: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFCL_L14: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_ARD_SIGMA_QOLD.patch
    comptime assert not is_defined["MOJOLEARN_ARD_SIGMA_QOLD"](), "removed 2026-11-15 (lane/expdel2-21-linear): ARD_SIGMA_QOLD (unmeasured old path) is gone; the default is kept; code at 085dcde095; restore: experiments/removed/MOJOLEARN_ARD_SIGMA_QOLD.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_BAYES_GRID_GUARD_OFF.patch
    comptime assert not is_defined["MOJOLEARN_BAYES_GRID_GUARD_OFF"](), "removed 2026-11-15 (lane/expdel2-21-linear): BAYES_GRID_GUARD_OFF (old path of the FAST Apple default) is gone; the default is kept; code at f7077f3ddc; restore: experiments/removed/MOJOLEARN_BAYES_GRID_GUARD_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_GLM_CELLS_MIXED.patch
    comptime assert not is_defined["MOJOLEARN_GLM_CELLS_MIXED"](), "removed 2026-11-15 (lane/expdel2-21-linear): GLM_CELLS_MIXED (unmeasured opt-in arm) is gone; the default is kept; code at a602b53e45; restore: experiments/removed/MOJOLEARN_GLM_CELLS_MIXED.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_HUBER_DEVICE_LBFGS_OFF.patch
    comptime assert not is_defined["MOJOLEARN_HUBER_DEVICE_LBFGS_OFF"](), "removed 2026-11-15 (lane/expdel2-21-linear): HUBER_DEVICE_LBFGS_OFF (old path of the FAST Apple default) is gone; the default is kept; code at a1003b0db5; restore: experiments/removed/MOJOLEARN_HUBER_DEVICE_LBFGS_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_HUBER_FAST_BLOCK512.patch
    comptime assert not is_defined["MOJOLEARN_HUBER_FAST_BLOCK512"](), "removed 2026-11-15 (lane/expdel2-21-linear): HUBER_FAST_BLOCK512 (force-on knob of the FAST Apple default) is gone; the default is kept; code at e7c79bbf05; restore: experiments/removed/MOJOLEARN_HUBER_FAST_BLOCK512.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_LEGACY_NARROW_SGD_PS_SIMD.patch
    comptime assert not is_defined["MOJOLEARN_LEGACY_NARROW_SGD_PS_SIMD"](), "removed 2026-11-15 (lane/expdel2-21-linear): LEGACY_NARROW_SGD_PS_SIMD (unmeasured old path) is gone; the default is kept; code at 51a85b3df6; restore: experiments/removed/MOJOLEARN_LEGACY_NARROW_SGD_PS_SIMD.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_LEGACY_NARROW_SVM_WS.patch
    comptime assert not is_defined["MOJOLEARN_LEGACY_NARROW_SVM_WS"](), "removed 2026-11-15 (lane/expdel2-21-linear): LEGACY_NARROW_SVM_WS (old path, EXP:1344 NEUTRAL) is gone; the default is kept; code at ab40e72f25; restore: experiments/removed/MOJOLEARN_LEGACY_NARROW_SVM_WS.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_RIDGE_FF_ALWAYS.patch
    comptime assert not is_defined["MOJOLEARN_RIDGE_FF_ALWAYS"](), "removed 2026-11-15 (lane/expdel2-21-linear): RIDGE_FF_ALWAYS (unmeasured opt-in arm) is gone; the default is kept; code at 67fe970eba; restore: experiments/removed/MOJOLEARN_RIDGE_FF_ALWAYS.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_SGDOC_FAST_TAIL_LONG.patch
    comptime assert not is_defined["MOJOLEARN_SGDOC_FAST_TAIL_LONG"](), "removed 2026-11-15 (lane/expdel2-21-linear): SGDOC_FAST_TAIL_LONG (unmeasured opt-in arm) is gone; the default is kept; code at ea2d15591a; restore: experiments/removed/MOJOLEARN_SGDOC_FAST_TAIL_LONG.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_SGD_ADAPTIVE_BATCH_OFF.patch
    comptime assert not is_defined["MOJOLEARN_SGD_ADAPTIVE_BATCH_OFF"](), "removed 2026-11-15 (lane/expdel2-21-linear): SGD_ADAPTIVE_BATCH_OFF (unmeasured old path) is gone; the default is kept; code at 5d8783f621; restore: experiments/removed/MOJOLEARN_SGD_ADAPTIVE_BATCH_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_SGD_AUTO_STEPS_256_OFF.patch
    comptime assert not is_defined["MOJOLEARN_SGD_AUTO_STEPS_256_OFF"](), "removed 2026-11-15 (lane/expdel2-21-linear): SGD_AUTO_STEPS_256_OFF (unmeasured old path) is gone; the default is kept; code at 735cd099cb; restore: experiments/removed/MOJOLEARN_SGD_AUTO_STEPS_256_OFF.patch"
    # MOJOLEARN_SGD_FAST_EPOCH_RESIDENT: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-classical: Apple fast paused, not abandoned; opt-in, default off; see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_SGD_IDN_CHUNK_WIDE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_SGD_IDN_CHUNK_WIDE_OFF"](), "removed 2026-11-15 (lane/expdel2-21-linear): SGD_IDN_CHUNK_WIDE_OFF (unmeasured old path) is gone; the default is kept; code at 2c034b8d40; restore: experiments/removed/MOJOLEARN_SGD_IDN_CHUNK_WIDE_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_SGD_IDN_DEV_FINITE_APPLE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_SGD_IDN_DEV_FINITE_APPLE_OFF"](), "removed 2026-11-15 (lane/expdel2-21-linear): SGD_IDN_DEV_FINITE_APPLE_OFF (unmeasured old path) is gone; the default is kept; code at 1d51928517; restore: experiments/removed/MOJOLEARN_SGD_IDN_DEV_FINITE_APPLE_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_SGD_IDN_DEV_FINITE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_SGD_IDN_DEV_FINITE_OFF"](), "removed 2026-11-15 (lane/expdel2-21-linear): SGD_IDN_DEV_FINITE_OFF (unmeasured old path) is gone; the default is kept; code at 2040a5c03e; restore: experiments/removed/MOJOLEARN_SGD_IDN_DEV_FINITE_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_SGD_IDN_MB_FUSE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_SGD_IDN_MB_FUSE_OFF"](), "removed 2026-11-15 (lane/expdel2-21-linear): SGD_IDN_MB_FUSE_OFF (unmeasured old path) is gone; the default is kept; code at f3550a03d5; restore: experiments/removed/MOJOLEARN_SGD_IDN_MB_FUSE_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_SGD_IDN_OVR_PAR_OFF.patch
    comptime assert not is_defined["MOJOLEARN_SGD_IDN_OVR_PAR_OFF"](), "removed 2026-11-15 (lane/expdel2-21-linear): SGD_IDN_OVR_PAR_OFF (unmeasured old path) is gone; the default is kept; code at 255a84c8ff; restore: experiments/removed/MOJOLEARN_SGD_IDN_OVR_PAR_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_SGD_IDN_PS_WARP_OFF.patch
    comptime assert not is_defined["MOJOLEARN_SGD_IDN_PS_WARP_OFF"](), "removed 2026-11-15 (lane/expdel2-21-linear): SGD_IDN_PS_WARP_OFF (unmeasured old path) is gone; the default is kept; code at 29555cd114; restore: experiments/removed/MOJOLEARN_SGD_IDN_PS_WARP_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_SGD_OC_IDN_MB_DEFAULT_OFF.patch
    comptime assert not is_defined["MOJOLEARN_SGD_OC_IDN_MB_DEFAULT_OFF"](), "removed 2026-11-15 (lane/expdel2-21-linear): SGD_OC_IDN_MB_DEFAULT_OFF (unmeasured old path) is gone; the default is kept; code at 51028ea5eb; restore: experiments/removed/MOJOLEARN_SGD_OC_IDN_MB_DEFAULT_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_SGD_OC_MB_IMPLICIT_OFF.patch
    comptime assert not is_defined["MOJOLEARN_SGD_OC_MB_IMPLICIT_OFF"](), "removed 2026-11-15 (lane/expdel2-21-linear): SGD_OC_MB_IMPLICIT_OFF (unmeasured old path) is gone; the default is kept; code at 561bfcf338; restore: experiments/removed/MOJOLEARN_SGD_OC_MB_IMPLICIT_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_SVM_FAST_SORT_OFF.patch
    comptime assert not is_defined["MOJOLEARN_SVM_FAST_SORT_OFF"](), "removed 2026-11-15 (lane/expdel2-21-linear): SVM_FAST_SORT_OFF (unmeasured old path) is gone; the default is kept; code at 7684b61797; restore: experiments/removed/MOJOLEARN_SVM_FAST_SORT_OFF.patch"
    # MOJOLEARN_SVM_WS1024: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-classical: Apple fast paused, not abandoned; opt-in, default off; see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_XLIN_IDN_DEV_FINITE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_XLIN_IDN_DEV_FINITE_OFF"](), "removed 2026-11-15 (lane/expdel2-21-linear): XLIN_IDN_DEV_FINITE_OFF (unmeasured old path) is gone; the default is kept; code at 784535aaec; restore: experiments/removed/MOJOLEARN_XLIN_IDN_DEV_FINITE_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_X_LINEAR_ARD_LEAD.patch
    comptime assert not is_defined["MOJOLEARN_X_LINEAR_ARD_LEAD"](), "removed 2026-11-15 (lane/expdel2-21-linear): X_LINEAR_ARD_LEAD (unmeasured opt-in arm) is gone; the default is kept; code at a33761f577; restore: experiments/removed/MOJOLEARN_X_LINEAR_ARD_LEAD.patch"
    # MOJOLEARN_X_LINEAR_FAST_FMA: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-classical: Apple fast paused, not abandoned; opt-in, default off; see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_X_LINEAR_NO_STAGED_FOLD.patch
    comptime assert not is_defined["MOJOLEARN_X_LINEAR_NO_STAGED_FOLD"](), "removed 2026-11-15 (lane/expdel2-21-linear): X_LINEAR_NO_STAGED_FOLD (unmeasured opt-in arm) is gone; the default is kept; code at 51eabe26ef; restore: experiments/removed/MOJOLEARN_X_LINEAR_NO_STAGED_FOLD.patch"
    # lane dbscan-brute-edgefree (2026-11-16): the MOJOLEARN_DBSCAN_FAST_BRUTE_EDGE_FREE fast-mode refusal moved
    # to the code that reads the define (dbscan/impl/runner.mojo, lane owed-deletions 2026-11-16), so the arm
    # rebuilds only the bindings that compile DBSCAN.
    # ---- lane fb15-ab-verdicts (2026-11-18): A/B verdicts of ab1 (NVIDIA L40S, wheel d652a4b0a1)
    # TOMBSTONE: MOJOLEARN_LINEAR_LBD_BATCH_OFF (DROPPED-noise: one flag read per trial 1.00-1.02x on all 8 cells, NV L40S 2026-11-18) deleted 2026-11-18 by lane/fb15-ab-verdicts; code recoverable at c989eb9222.
    # Restore: git apply experiments/removed/MOJOLEARN_LINEAR_LBD_BATCH_OFF.patch; record in docs/TOMBSTONES.md.
    comptime assert not is_defined["MOJOLEARN_LINEAR_LBD_BATCH_OFF"](), "removed 2026-11-18 (lane/fb15-ab-verdicts, A/B ab1 NVIDIA L40S, wheel d652a4b0a1, one run per arm): one flag read per trial (0.8.55's L-BFGS form) was 1.00-1.02x on all 8 huber / logreg-cv cells, digests equal; the gated queue (LBD_BATCH units per read) is the one identical form off Apple; recoverable at c989eb9222"
    # ---- end lane fb15-ab-verdicts
    # ---- lane fx-verdicts2 (2026-11-18): A/B verdicts of ab3 (NVIDIA L40S, wheel d652a4b0a1)
    # TOMBSTONE: MOJOLEARN_GEMM_POOL_OFF (LOSER, A/B ab3 NVIDIA L40S, wheel d652a4b0a1, job nvc-ab-20261118T182558Z-cb6e02, one run per arm, digests equal) deleted 2026-11-18 by lane/fx-verdicts2; code recoverable at 85c36e324f.
    # Restore: git apply experiments/removed/MOJOLEARN_GEMM_POOL_OFF-STAGE_DOWN_OFF.patch; record in docs/TOMBSTONES.md.
    comptime assert not is_defined["MOJOLEARN_GEMM_POOL_OFF"](), "removed 2026-11-18 (lane/fx-verdicts2, A/B ab3 NVIDIA L40S, wheel d652a4b0a1, job nvc-ab-20261118T182558Z-cb6e02, one run per arm, digests equal): fresh device buffers per matmul were arm/base 1.044 geomean over the 3 gemm cells; the pool is the route; recoverable at 85c36e324f"
    # TOMBSTONE: MOJOLEARN_GEMM_STAGE_DOWN_OFF (LOSER, A/B ab3 NVIDIA L40S, wheel d652a4b0a1, job nvc-ab-20261118T182558Z-cb6e02, one run per arm, digests equal) deleted 2026-11-18 by lane/fx-verdicts2; code recoverable at 85c36e324f.
    # Restore: git apply experiments/removed/MOJOLEARN_GEMM_POOL_OFF-STAGE_DOWN_OFF.patch; record in docs/TOMBSTONES.md.
    comptime assert not is_defined["MOJOLEARN_GEMM_STAGE_DOWN_OFF"](), "removed 2026-11-18 (lane/fx-verdicts2, A/B ab3 NVIDIA L40S, wheel d652a4b0a1, job nvc-ab-20261118T182558Z-cb6e02, one run per arm, digests equal): the raw GEMM download was arm/base 1.125 geomean over the 3 gemm cells; the staged download is the route; recoverable at 85c36e324f"
    # ---- end lane fx-verdicts2
    # ---- lane fx-verdicts4 (2026-11-19): verdicts of A/B round 2 (NVIDIA L40S, wheel 0c68d6cd27, job nvc-ab-20261119T154232Z-0642a5)
    # TOMBSTONE: MOJOLEARN_XLINEAR_FAST_WB_RESIDENT (NOISE, A/B round 2 NVIDIA L40S, wheel 0c68d6cd27, job nvc-ab-20261119T154232Z-0642a5, one run per arm, digests equal) deleted 2026-11-19 by lane/fx-verdicts4; code recoverable at 8f3f639b13.
    # Restore: git apply experiments/removed/MOJOLEARN_XLINEAR_FAST_WB_RESIDENT.patch; record in docs/TOMBSTONES.md.
    comptime assert not is_defined["MOJOLEARN_XLINEAR_FAST_WB_RESIDENT"](), "removed 2026-11-19 (lane/fx-verdicts4, A/B round 2 NVIDIA L40S, wheel 0c68d6cd27, job nvc-ab-20261119T154232Z-0642a5, one run per arm, digests equal): the resident weight block of the linear decision entries was arm/base 0.94-1.20 on predict with three rows slower and fits flat (noise); code at 8f3f639b13, experiments/removed/MOJOLEARN_XLINEAR_FAST_WB_RESIDENT.patch"
    # ---- end lane fx-verdicts4
    return True


comptime LINEAR_GUARDS_OK = _check_configuration()
