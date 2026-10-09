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
