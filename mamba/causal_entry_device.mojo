# SPDX-License-Identifier: Apache-2.0
"""Publication of `_mojolearn_mamba`'s descriptor entry in the CausalLM entry table
(lane calltable-causal-lm, 2026-11-11), family `CAUSAL_TABLE_MAMBA`.

The image's module init publishes the entry it already carries
(`mamba.native_state_dispatch.NATIVE_MAMBA_STATE_API`) under the key (tier, COMPILED_VENDOR). A build with no GPU
vendor ("none") publishes nothing: its callers refuse with the table's
"provider is not published". A repeated init publishes the same pointer.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE
from checks.vendor import COMPILED_VENDOR
from core.causal_lm_entry_table import CausalEntryKey, causal_entry_vendor, causal_table_publish, CAUSAL_TABLE_MAMBA
from mamba.native_state_dispatch import NATIVE_MAMBA_STATE_API


def publish_causal_mamba_device() raises:
    var vendor = causal_entry_vendor(String(COMPILED_VENDOR))
    if vendor < 0:
        return
    var key = CausalEntryKey(Int(GLOBAL_NUMERIC_MODE), vendor)
    var api = NATIVE_MAMBA_STATE_API.get_or_create_ptr()
    causal_table_publish[CAUSAL_TABLE_MAMBA](key, Int(api), api[].entries[0])
