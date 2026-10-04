# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Opt-in, exact Apple FAST label preprocessing experiments.

LABEL_PRESENT exposes existing parallel GPU presence stages to label fitting.
LABEL_SCATTER clears the dense indicator with a device memset and scatters
one positive word per row, avoiding one lookup/division per dense output cell.
The switches are independent; neither changes IDENTICAL or other vendors.
"""
from std.gpu import block_idx, block_dim, thread_idx
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST
from x_prep.common import FP, IP, p, ld, sti

comptime _FAST_APPLE = GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
comptime LABEL_PRESENT = _FAST_APPLE and is_defined["MOJOLEARN_LABEL_PRESENT"]()
comptime LABEL_SCATTER = _FAST_APPLE and is_defined["MOJOLEARN_LABEL_SCATTER"]()


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
