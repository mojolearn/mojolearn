# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Lane fg-linear (flagship gaps, 2026-10-09) IDENTICAL switches for the
linear models: OLS, Ridge, Lasso / ElasticNet (CD Gram sweep), logistic
regression (L-BFGS) and SGDClassifier. Read docs/plans/flagship-gaps-20261009/
read_linear.md for the stage tables each switch answers.

NOT COMPILED, NOT VERIFIED, NOT MEASURED by the lane (code-only lane); the
orchestrator compiles every arm, runs the ID checks and the A/Bs. No GPU
import here: the host bindings read the same flags, so the device columns
and the host column switch together. Every rule below is by width / work /
bytes, never by a board shape.
"""
from std.sys.compile import is_defined
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from experiments.classical_identical_ideas.linear_controls import LINEAR_GRAM_SOLVE

comptime _FGL_IDN = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL

# L2 (DEFAULT ON, -D MOJOLEARN_IDN_RIDGE_RESIDENT_OFF restores the old path).
# When the resident Gram solve (LINEAR_GRAM_SOLVE) rejects a Ridge fit at its
# trust gate, the entry no longer returns status 1 to Python (which then made
# four PCIe crossings of X: column sums, center, centered download, the fit's
# upload, and recomputed the sums / means). It runs Ridge's eig route right
# there on the X and y already resident and the means already formed: the
# `ridge_fit_resident_host` body (lane apple-fast-ridgespeed, FAST+Apple only
# until now) on device buffers: center_buf of X and y, then
# ridge_eig_scratch_traced with the dead raw X as gemm_tn scratch. Cost
# reasoning: removes 3 uploads + 1 download of n x d floats (880 MB at
# 1M x 220: 150-350 ms of PCIe) and two duplicated sum/center passes; the
# work left is the incumbent's Gram + Jacobi + U. Bits: none expected; the
# means come from the same col_sums_pair_buf / col_means_buf launches the
# Python glue reaches and the eig route is the same kernels and launch shapes
# (ridge_eig_scratch_traced's contract), so the host column (whose own gate
# reject runs the host eig route) is unchanged. ID check owed (nv == amd).
comptime IDN_RIDGE_RESIDENT = LINEAR_GRAM_SOLVE and not is_defined["MOJOLEARN_IDN_RIDGE_RESIDENT_OFF"]()

# L1 (DEFAULT ON since 2026-10-09, lane/postmerge-act-5; -D
# MOJOLEARN_IDN_JACOBI_ROUND_ROBIN_OFF restores the one-block cyclic Jacobi;
# the old on-define is refused in core/six_lane_experiment_guards.mojo).
# Post-merge A/B on main 0a7b206f1, one run per arm (nv2 L40S v1021-v1050,
# MI325X a1161-a1190; ratio = arm / fg2.default-linear): ridge istella NV
# 0.25x / AMD 0.14x (avg 0.20x; hash ff4ab841 -> cce6558a on BOTH vendors,
# r2 0.328682 -> 0.328636), ridge taxi NV 0.99x / AMD 1.15x (avg 1.07x,
# VENDOR SPLIT: taxi 1.15x slower on AMD; hash 520d0ae1 unchanged), ols
# istella NV 1.00x / AMD 0.99x, ols taxi NV 0.99x / AMD 1.02x. The small symmetric
# eigensolver of the linear models' Gram routes: Ridge's svdEig
# (glm/impl/linalg/detail/svd.mojo, the eig route the Gram gate falls back
# to), OLS's lstsqEig (glm/impl/linalg/detail/lstsq.mojo) and the
# minimum-norm Gram (lstsq_min_norm.mojo). Today each is ONE block running the
# cyclic Jacobi: d (d - 1) / 2 serial rotations a sweep, each a barrier pair
# (at d = 220: 24,090 rotations, ~48k barriers a sweep, 12-15 sweeps:
# 0.45-0.7 s on one SM). Here the round-robin (chess tournament) order: d - 1
# rounds a sweep, the d / 2 disjoint rotations of a round in parallel across
# the GPU, the rotation schedule a pure function of d (x_decomp/rr.mojo
# `pj_first` / `pj_second`), the same on every vendor and the host. The
# device driver is PCA's `_eig_rr_device` (decomposition/impl/linalg/detail/
# pca.mojo, called, not edited: its rounds, its device-decided convergence
# test `rr_converged`, `rr_fro_kept`, RR_EIGH_SWEEPS = 60) and the host column
# is `host_eigh_rr` (x_decomp/rr.mojo), the pair PCA_RR_EIGH already holds
# equal on nv, amd and host. Cost reasoning: 2 (d - 1) launches a sweep of
# O(d^2) parallel work each instead of d^2 / 2 serialized rotations on one
# SM: at d of a few hundred, launch-bound at ~8-15 us a launch (~20-40 ms for
# a 12-sweep solve) instead of 0.45-0.7 s. At d <= ~16 the cyclic block is a
# few hundred rotations and the round-robin chain is ~30 launches a sweep:
# roughly a wash, so the switch is one flag for all d (no width rule).
# BITS CHANGE (different rotation order and convergence test): every device
# column and the host column move together (glm/host/glm_oracle.mojo reads
# this flag through decomposition/host/jacobi_select_host.mojo). Lane fg-pca's
# PCA-only switches (MOJOLEARN_IDN_PCA_RR_*) stay theirs: if they change the
# inside of `_eig_rr_device` the linear models inherit it under both flags.
comptime IDN_JACOBI_ROUND_ROBIN = _FGL_IDN and not is_defined["MOJOLEARN_IDN_JACOBI_ROUND_ROBIN_OFF"]()

# L3 (DEFAULT ON since 2026-10-09, lane/postmerge-act-5; -D
# MOJOLEARN_IDN_GRAM_FF_FALLBACK_OFF restores the direct eig-route fallback;
# the old on-define is refused in core/six_lane_experiment_guards.mojo).
# Post-merge A/B on main 0a7b206f1, one run per arm (nv2 L40S v1021-v1050,
# MI325X a1161-a1190; ratio = arm / fg2.default-linear): ridge istella NV
# 0.96x / AMD 0.51x (avg 0.74x; hash ff4ab841 -> 5a348e50 on BOTH vendors,
# r2 0.328682 -> 0.328674), ridge taxi NV 0.91x / AMD 0.99x (hash
# unchanged), ols istella 1.00x / 1.00x, ols taxi NV 0.98x / AMD 1.03x.
# Measured alone, never with L1 on: both-on is owed. When the float32 Gram
# fails the trust gate on a RIDGE fit (alpha > 0), a second chance before the
# eig route: the centered Gram and cross re-formed in float-float from the
# resident X (glm/impl/gram_ff_cells.mojo: 64 row leaves folded ascending),
# the same power-of-two equilibration, a float-float Cholesky (one launch per
# column, rows in parallel) under a 2^-24 gate, float-float triangular
# solves, coef rounded once. Cost reasoning: one more pass over X at ~10x
# the fp32 flops per cell (n d^2 / 2 float-float multiply-adds: at
# 1M x 220 ~0.5 TFLOP of IDENTICAL float32, 50-100 ms on an L40S class GPU)
# plus 3 d small launches, against the eig route's Gram + Jacobi + U = A V
# (0.5-1.2 s today). Quality: float-float ~ binary64 normal equations, at
# least the fp32 eig route's accuracy for alpha > 0. OLS (alpha 0) keeps its
# TSQR + SVD-cutoff fallback (a rank-deficient design needs the cutoff).
# When the float-float gate fails too, the L2 / status-1 fallback runs.
# BITS CHANGE for the fits it takes (normal equations instead of eig); the
# host column (glm/host/gram_solve_host.mojo) runs the same cells.
comptime IDN_GRAM_FF_FALLBACK = LINEAR_GRAM_SOLVE and not is_defined["MOJOLEARN_IDN_GRAM_FF_FALLBACK_OFF"]()

# L4 (DEFAULT ON, -D MOJOLEARN_IDN_LINEAR_PINNED_UPLOAD_OFF restores the
# direct copy). The linear fits' large host uploads (the resident Gram fit's
# X and y, the incumbent Ridge entry's, logistic regression's X and y) go
# through a process-lifetime PINNED stage of two 32 MB halves, chunk by
# chunk, the host memcpy of chunk i overlapping the DMA of chunk i - 1
# (glm/impl/pinned_upload.mojo; gemm/host_transport.mojo's double buffer),
# instead of one enqueue_copy from the caller's pageable NumPy pointer.
# Cost reasoning: a pageable H2D is staged by the driver through a small
# bounce buffer (~10-12 GB/s at best on PCIe 4 x16, often far less), a pinned
# one runs at the link rate (~25 GB/s); the added memcpy runs at host memory
# bandwidth and overlaps the DMA. Expected: 176 MB 12-15 -> ~7-9 ms, 880 MB
# ~70 -> ~35-45 ms. Below 1M floats (4 MB) the direct copy stays (the
# per-chunk waits are not repaid). Not on Apple (unified memory: a second
# host copy for nothing) and IDENTICAL only (FAST is untouched by the lane).
# Transport only: NO bit moves. Lane fg-knn-nb's DeviceStore ring
# (core/device_store.mojo) is the same pattern for the x_prep store; the
# linear entries do not upload through a DeviceStore, so they take this one.
comptime IDN_LINEAR_PINNED_UPLOAD = _FGL_IDN and not is_defined["MOJOLEARN_IDN_LINEAR_PINNED_UPLOAD_OFF"]()

# TOMBSTONE: S1 (MOJOLEARN_IDN_SGD_EPOCH_KERNEL, large-batch SGD through the K-batch
# chunk kernel) deleted 2026-10-09 (lane postmerge-act-2): noise, sgd-clf istella
# NV 0.99x / AMD 1.00x, taxi 1.00x / 1.00x (nv n0606->n0615, amd a1065->a1074),
# accuracy SAME, same digests; refused in core/six_lane_experiment_guards.mojo; main 5c137b55e.

# C2 (DEFAULT OFF, -D MOJOLEARN_IDN_CD_GRAM_EPOCHS_64). Lasso / ElasticNet's
# IDENTICAL Gram sweep (solver/impl/cd.mojo `cd_idn_gram_sweep_kernel`) runs
# CD_IDN_GRAM_EPOCHS = 16 epochs a launch, then the host reads four state
# words (one wait, ~30-60 us). At narrow designs an epoch is d coordinates x
# ~3 barriers (~1-3 us a coordinate: an epoch at d = 16 is ~50 us), so the
# wait is about half the sweep time. Here a launch runs 64 epochs when
# n_cols <= CD_EK_SMALL_COLS (64): the epoch is then at most ~64 coordinates,
# a 64-epoch launch stays a few ms (far from any watchdog), and the epochs a
# launch runs after convergence (frozen: barriers only, no stores) cost at
# most 63 x d x 3 barriers. Wider designs keep 16 (their epochs already
# dominate the wait). Rule by the per-epoch cost (n_cols), not a board width.
# NO BIT MOVES: the convergence test and the freeze are decided on the device
# per epoch, so coef and n_iter_ are the same at any epochs-per-launch; the
# host column (solver/host/cd_oracle.mojo) is unchanged.
comptime IDN_CD_GRAM_EPOCHS_64 = _FGL_IDN and is_defined["MOJOLEARN_IDN_CD_GRAM_EPOCHS_64"]()
comptime CD_EK_SMALL_COLS = 64
