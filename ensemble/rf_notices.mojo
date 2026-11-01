# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The RandomForest fit notices: cuML's three user warnings
(`randomforest_common.pyx:529-536`, `randomforest.cuh:301-306`,
`randomforest_common.pyx:725-732`), one bit each.

A fit records the notices it raised as a bit mask on its result
(`RandomForestMetaData.notices`, `RfHostForest.notices`). The bindings set
`notices_to_stderr=False`, carry the mask out as the third fit metadata
word, and `bindings/trees_public_forest.mojo` raises each as a Python
UserWarning through the native warnings path
(`bindings/classical_warnings.mojo`). A caller with no Python layer (a Mojo
program calling `fit_forest`/`rf_host_fit` directly) keeps the default and
gets each notice on stderr. Never stdout: a notice there corrupts the
caller's own output (it broke `verify --self-test --json`).

No device or host-thread code: importing it pulls neither into the
importer."""

from std.sys import stderr

comptime RF_NOTICE_OOB_PARTIAL = 1
comptime RF_NOTICE_BINS_CLAMPED = 2
comptime RF_NOTICE_MAX_SAMPLES_IGNORED = 4
comptime RF_NOTICE_ALL = 7
"""Every defined notice bit; a metadata word outside it is refused."""

comptime RF_NOTICE_OOB_PARTIAL_TEXT = (
    "Some inputs do not have OOB scores. This probably means too few trees"
    " were used to compute any reliable OOB estimates."
)
comptime RF_NOTICE_BINS_CLAMPED_TEXT = (
    "The number of bins, `n_bins` is greater than the number of samples used"
    " for training. Changing `n_bins` to number of training samples."
)
comptime RF_NOTICE_MAX_SAMPLES_IGNORED_TEXT = (
    "If bootstrap sampling is disabled, max_samples value is ignored and"
    " whole dataset is used for building each tree"
)


def rf_notice_text(code: Int) -> String:
    """The message of one notice bit; empty for an unknown bit."""
    if code == RF_NOTICE_OOB_PARTIAL:
        return String(RF_NOTICE_OOB_PARTIAL_TEXT)
    if code == RF_NOTICE_BINS_CLAMPED:
        return String(RF_NOTICE_BINS_CLAMPED_TEXT)
    if code == RF_NOTICE_MAX_SAMPLES_IGNORED:
        return String(RF_NOTICE_MAX_SAMPLES_IGNORED_TEXT)
    return String("")


def rf_notice(mut notices: Int, code: Int, to_stderr: Bool):
    """Record one notice bit; with `to_stderr` (no Python layer), also
    write its message to stderr."""
    notices |= code
    if to_stderr:
        print("WARN: " + rf_notice_text(code), file=stderr)
