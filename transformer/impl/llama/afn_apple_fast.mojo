# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""What remains of the Apple FAST transformer-forward candidates (lane
afn-attn, 2026-10-03): only the FLASH caller-audit counter the transformer
binding still exports (`afn_flash_call_count`). With no candidate left it
always reads zero.

Every candidate kernel, launcher and define that lived here was deleted on
2026-11-15 by lane/expdel2-33-neural (never measured end to end, or measured
on M3 with mixed/regressing component ratios and kept OFF); the code is
recoverable at c3d1c7865f.
"""

# TOMBSTONE: MOJOLEARN_AFN_ATTN_ALL, MOJOLEARN_AFN_ATTN_NORM_SG, MOJOLEARN_AFN_ATTN_ROPE_CACHE, MOJOLEARN_AFN_ATTN_GQA_TILE, MOJOLEARN_AFN_ATTN_FLASH, MOJOLEARN_AFN_ATTN_FUSE_PRE, MOJOLEARN_AFN_ATTN_FUSE_MLP (DELETE-ARM: neural MERGED-UNMEASURED opt-in, never measured; FLASH/GQA_TILE M3 component B/A mixed/regressing, kept OFF) deleted 2026-11-15 by lane/expdel2-33-neural; code recoverable at c3d1c7865f.
# Restore: git apply experiments/removed/MOJOLEARN_AFN_ATTN_ALL.patch (one patch per define, same code)
# TOMBSTONE: MOJOLEARN_AFN_ATTN_AUDIT, MOJOLEARN_AFN_FLASH_HD_GENERIC_OFF, MOJOLEARN_AFN26_ATTN_NORM_TPB128, MOJOLEARN_AFN26_ATTN_NORM_TPB512, MOJOLEARN_AFN26_ATTN_FLASH_BK16, MOJOLEARN_AFN26_ATTN_FLASH_TQ16, MOJOLEARN_AFN26_ATTN_PROJ_KB16, MOJOLEARN_AFN26_ATTN_PROJ_BM32 (DELETE-ARM / DELETE-OLD-PATH: no A/B recorded) deleted 2026-11-15 by lane/expdel2-33-neural; code recoverable at c3d1c7865f.
# Restore: git apply experiments/removed/<DEFINE>.patch

from core.shared_slot import SharedSlot


struct FlashCallerAudit(Defaultable, Movable):
    var calls: Int
    var grouped_calls: Int
    def __init__(out self):
        self.calls = 0
        self.grouped_calls = 0

comptime FLASH_AUDIT = SharedSlot[FlashCallerAudit, "MojolearnAppleFlashCallerAuditV1"]


def afn_flash_call_count(grouped: Bool) raises -> Int:
    var audit = FLASH_AUDIT.get_or_create_ptr()
    return audit[].grouped_calls if grouped else audit[].calls
