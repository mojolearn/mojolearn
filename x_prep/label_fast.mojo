# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Exact Apple FAST label preprocessing default.

LABEL_PRESENT exposes existing parallel GPU presence stages to label fitting.
LABEL_SCATTER clears the dense indicator with a device memset and scatters
one positive word per row, avoiding one lookup/division per dense output cell.
Lane idn-int-prep (2026-10-04): IDENTICAL takes the same bundle on every vendor
and in the host column (IDN_LABEL below); FAST off Apple is unchanged.
"""
from max.gpu import block_idx, block_dim, thread_idx
from core.accelerator_target import TARGET_HAS_APPLE_GPU
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST, NUMERIC_IDENTICAL
from x_prep.common import FP, IP, p, ld, sti

comptime _FAST_APPLE = GLOBAL_NUMERIC_MODE == NUMERIC_FAST and TARGET_HAS_APPLE_GPU
# KEEP candidate, M3 FAST one run per arm, measured dbe2ab85a vs base35a72c5ac:
# gap26-label-taxi 348.254459 -> 327.625833 ms (-5.9%), digest0703e6f6396640c4;
# gap26-label-istella 33.176500 -> 27.578792 ms (-16.9%), digestde224838a841fa8c.
# gap26-label-quality: both arms PASS, 41,697,776 checked indicator cells,
# plus fitted classes, encoder codes and LabelBinarizer inverse values.
# Reviewed against main61ea51757: relevant source drift is comments only.
# TOMBSTONE: MOJOLEARN_LABEL_DIRECT_OFF (old path; the direct label paths are the FAST+Apple default, EXP:762) deleted 2026-11-15 by lane/expdel2-22-linear; code recoverable at 2ec4e389be.
# Restore: git apply experiments/removed/MOJOLEARN_LABEL_DIRECT_OFF.patch
comptime LABEL_DIRECT = _FAST_APPLE
#: lane idn-int-prep (2026-10-04): the same presence stages and scatter in
#: IDENTICAL on every vendor and in the host column (ON by default). The
#: presence flags, the ascending flag scan and the scattered indicator are
#: integer words, so the classes, codes and indicator are the sort route's
#: words.
# TOMBSTONE: MOJOLEARN_IDN_LABEL_OFF (old path, no A/B recorded beside the IDENTICAL default) deleted 2026-11-15 by lane/expdel2-22-linear; code recoverable at 886a739db7.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_LABEL_OFF.patch
# TOMBSTONE: MOJOLEARN_IDN_ALL_OFF (old path umbrella; EXP:2438 deleted, the resident defaults kept) deleted 2026-11-15 by lane/expdel2-22-linear; code recoverable at 51e7b6d7ec.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_ALL_OFF.expdel2-22-linear.patch
comptime IDN_LABEL = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
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
