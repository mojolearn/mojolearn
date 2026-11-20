# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`_log_or_inf` of one binary64 word (`py2mojo_rows` ROWS_LOG), with no GPU
or host-thread import, so the device kernel (`core/py2mojo_rows.mojo`) and
the host column (`core/classical_host_predict.mojo`) share one statement
list: log(p) for p > 0 (`sf64_log`), -inf at +-0, NaN otherwise."""
# ne2-g-linear (N361): moved here unchanged from core/py2mojo_rows.mojo so the
# host qn_proba log runs the same words on host tasks.
from checks.soft_f64 import SF64_NAN, SF64_SIGN, SF64_ZERO, sf64_gt, sf64_is_nan, sf64_log

comptime _NEG_INF_WORD = UInt64(0xFFF0000000000000)


@always_inline
def log_or_inf(p: UInt64) -> UInt64:
    if sf64_gt(p, SF64_ZERO) and not sf64_is_nan(p):
        return sf64_log(p)
    if (p & ~SF64_SIGN) == 0:
        return _NEG_INF_WORD
    return SF64_NAN
