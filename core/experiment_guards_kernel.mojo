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
    return True


comptime KERNEL_GUARDS_OK = _check_configuration()
