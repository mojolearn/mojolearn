# SPDX-License-Identifier: Apache-2.0
"""Shared storage choices for the NN53 ByteLM chunked GEMM head.

Panel width changes allocation/launch count, never logical vocabulary or row
fold order. Host and device import this CPU-only selector.
"""

# TOMBSTONE: MOJOLEARN_NN53_HEAD_CHUNK512 and MOJOLEARN_NN53_HEAD_CHUNK2048 (opt-in panel widths, no A/B recorded)
# deleted 2026-11-15 by lane/expdel2-31-neural with the MOJOLEARN_IDN_ALL_OFF term beside them; code recoverable at
# cba6209bbd. The names stay as False for training/chunked_lm_head_v2.mojo.
# Restore: git apply experiments/removed/MOJOLEARN_NN53_HEAD_CHUNK512.patch (and MOJOLEARN_NN53_HEAD_CHUNK2048.patch)
comptime NN53_CHUNK512 = False
comptime NN53_CHUNK2048 = False
comptime LM_HEAD_V2_CHUNK = 1024
