# SPDX-License-Identifier: Apache-2.0
"""Publication of `_mojolearn_transformer`'s descriptor entry in the CausalLM entry table
(lane calltable-causal-lm, 2026-11-11), family `CAUSAL_TABLE_BLOCK`.

The image's module init publishes the entry it already carries
(`transformer.native_block_dispatch.NATIVE_TRANSFORMER_BLOCK_API`) under the key (tier, COMPILED_VENDOR). A build with no GPU
vendor ("none") publishes nothing: its callers refuse with the table's
"provider is not published". A repeated init publishes the same pointer.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE
from checks.vendor import COMPILED_VENDOR
from core.causal_lm_entry_table import CausalEntryKey, causal_entry_vendor, causal_table_publish, CAUSAL_TABLE_BLOCK
from transformer.native_block_dispatch import NATIVE_TRANSFORMER_BLOCK_API


def publish_causal_transformer_device() raises:
    var vendor = causal_entry_vendor(String(COMPILED_VENDOR))
    if vendor < 0:
        return
    var key = CausalEntryKey(Int(GLOBAL_NUMERIC_MODE), vendor)
    var api = NATIVE_TRANSFORMER_BLOCK_API.get_or_create_ptr()
    causal_table_publish[CAUSAL_TABLE_BLOCK](key, Int(api), api[].entries[0])
