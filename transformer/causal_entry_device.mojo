# SPDX-License-Identifier: Apache-2.0
"""Publication of `_mojolearn_transformer`'s CausalLM resident sessions
(lane calltable-causal-lm, 2026-11-11). The block descriptor entry is NOT
published in the CausalLM table: `publish_neural_transformer_device`
publishes it once in the neural table's NEURAL_TABLE_BLOCK_FORWARD family and
`CAUSAL_TABLE_BLOCK` reads that slot (precompile-fixes, 2026-11-11).

The image's module init publishes the sessions entry under the key (tier,
COMPILED_VENDOR). A build with no GPU vendor ("none") publishes nothing: its
callers refuse with the table's "provider is not published". A repeated init
publishes the same pointer.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE
from checks.vendor import COMPILED_VENDOR
from core.causal_lm_entry_table import CausalEntryKey, causal_entry_vendor
from core.causal_lm_entry_table_sessions import causal_sessions_publish, CAUSAL_SESSIONS_TRANSFORMER
from transformer.causal_session_entry import NATIVE_CAUSAL_SESSIONS_API


def publish_causal_transformer_device() raises:
    var vendor = causal_entry_vendor(String(COMPILED_VENDOR))
    if vendor < 0:
        return
    var key = CausalEntryKey(Int(GLOBAL_NUMERIC_MODE), vendor)
    # The resident sessions and the decode program behind native handles
    # (core/causal_lm_entry_table_sessions.mojo, lane causal-lm-sessions-native).
    var sessions = NATIVE_CAUSAL_SESSIONS_API.get_or_create_ptr()
    causal_sessions_publish[CAUSAL_SESSIONS_TRANSFORMER](key, Int(sessions), sessions[].entries[0])
