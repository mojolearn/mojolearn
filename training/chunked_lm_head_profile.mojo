# SPDX-License-Identifier: Apache-2.0
"""Shared storage choices for the NN53 ByteLM chunked GEMM head.

Panel width changes allocation/launch count, never logical vocabulary or row
fold order. Host and device import this CPU-only selector. No evidence run.
"""
from std.sys.compile import is_defined
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL

comptime NN53_CHUNK512 = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and is_defined["MOJOLEARN_NN53_HEAD_CHUNK512"]()
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)
comptime NN53_CHUNK2048 = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and is_defined["MOJOLEARN_NN53_HEAD_CHUNK2048"]()
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)
comptime assert not (NN53_CHUNK512 and NN53_CHUNK2048), "choose one NN53 panel arm"
comptime LM_HEAD_V2_CHUNK = 512 if NN53_CHUNK512 else (2048 if NN53_CHUNK2048 else 1024)
