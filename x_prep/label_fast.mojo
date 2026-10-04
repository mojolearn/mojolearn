# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Exact Apple FAST label preprocessing default; LABEL_DIRECT_OFF restores baseline.

LABEL_PRESENT exposes existing parallel GPU presence stages to label fitting.
LABEL_SCATTER clears the dense indicator with a device memset and scatters
one positive word per row, avoiding one lookup/division per dense output cell.
Lane idn-int-prep (2026-10-04): IDENTICAL takes the same bundle on every vendor
and in the host column (IDN_LABEL below); FAST off Apple is unchanged.
"""
from std.gpu import block_idx, block_dim, thread_idx
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST, NUMERIC_IDENTICAL
from x_prep.common import FP, IP, p, ld, sti

comptime _FAST_APPLE = GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
# KEEP candidate, M3 FAST one run per arm, measured dbe2ab85a vs base35a72c5ac:
# gap26-label-taxi 348.254459 -> 327.625833 ms (-5.9%), digest0703e6f6396640c4;
# gap26-label-istella 33.176500 -> 27.578792 ms (-16.9%), digestde224838a841fa8c.
# gap26-label-quality: both arms PASS, 41,697,776 checked indicator cells,
# plus fitted classes, encoder codes and LabelBinarizer inverse values.
# Reviewed against main61ea51757: relevant source drift is comments only.
# MOJOLEARN_LABEL_DIRECT_OFF restores both old paths for rollback / A/B.
comptime LABEL_DIRECT = _FAST_APPLE and not is_defined["MOJOLEARN_LABEL_DIRECT_OFF"]()
#: lane idn-int-prep (2026-10-04): the same presence stages and scatter in
#: IDENTICAL on every vendor and in the host column (ON by default). The
#: presence flags, the ascending flag scan and the scattered indicator are
#: integer words, so the classes, codes and indicator are the sort route's
#: words. -D MOJOLEARN_IDN_LABEL_OFF restores the device sort + per-cell lookup.
comptime IDN_LABEL = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and not is_defined["MOJOLEARN_IDN_LABEL_OFF"]()
comptime LABEL_PRESENT = LABEL_DIRECT or IDN_LABEL
comptime LABEL_SCATTER = LABEL_DIRECT or IDN_LABEL


def label_scatter_kernel(f: FP, q: IP, total: Int32):
    """label_binarize q = [CODES, n, K, BINARY, NEG, POS, W, OUT].
    Dispatcher explicitly clears OUT and only calls this when NEG == 0.
    One thread owns each output row, so there are no conflicting writes.
    """
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(total):
        return
    var code = Int(ld(f, p(q, 0) + i))
    if p(q, 3) != 0:
        if code == 1:
            sti(f, p(q, 7) + i * p(q, 6), p(q, 5))
    elif code >= 0 and code < p(q, 6):
        sti(f, p(q, 7) + i * p(q, 6) + code, p(q, 5))
