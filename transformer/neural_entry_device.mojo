# SPDX-License-Identifier: Apache-2.0
"""Publication of `_mojolearn_transformer`'s block descriptor entry in the
neural-blocks entry table (core/neural_entry_table.mojo; lane
calltable-neural-blocks, 2026-11-11), families NEURAL_TABLE_BLOCK_FORWARD and
NEURAL_TABLE_BLOCK_TRAIN (the GPU image serves both a block's forward and
its backward).

The image's module init publishes the entry it already carries
(`transformer.native_block_dispatch.NATIVE_TRANSFORMER_BLOCK_API`) under the
key (tier, COMPILED_VENDOR). A build with no GPU vendor ("none") publishes
nothing: its callers refuse with the table's "provider is not published".
"""
from checks.numerics import GLOBAL_NUMERIC_MODE
from checks.vendor import COMPILED_VENDOR
from core.neural_entry_table import (
    NeuralEntryKey, neural_entry_vendor, neural_table_publish,
    NEURAL_TABLE_BLOCK_FORWARD, NEURAL_TABLE_BLOCK_TRAIN,
)
from transformer.native_block_dispatch import NATIVE_TRANSFORMER_BLOCK_API


def publish_neural_transformer_device() raises:
    var vendor = neural_entry_vendor(String(COMPILED_VENDOR))
    if vendor <= 0:
        return
    var key = NeuralEntryKey(Int(GLOBAL_NUMERIC_MODE), vendor)
    var api = NATIVE_TRANSFORMER_BLOCK_API.get_or_create_ptr()
    neural_table_publish[NEURAL_TABLE_BLOCK_FORWARD](key, Int(api), api[].entries[0])
    neural_table_publish[NEURAL_TABLE_BLOCK_TRAIN](key.copy(), Int(api), api[].entries[0])
