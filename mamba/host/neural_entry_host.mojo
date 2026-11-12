# SPDX-License-Identifier: Apache-2.0
"""Publication of `_mojolearn_mamba_host`'s block descriptor entry in the
neural-blocks entry table (core/neural_entry_table.mojo; lane
calltable-neural-blocks, 2026-11-11), family NEURAL_TABLE_MAMBA_TRAIN under
the key (tier, cpu): the CPU image a Mamba block's backward connects; and
its state entry in its own (tier, cpu) slot of core/mamba_state_entry_table.mojo
(lane transformer-session-native): a CPU-build training block's carried-state
forward and decode.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.neural_entry_table import NeuralEntryKey, neural_table_publish, NEURAL_TABLE_MAMBA_TRAIN
from mamba.host.native_block_dispatch import NATIVE_MAMBA_HOST_BLOCK_API, NATIVE_MAMBA_HOST_STATE_API
from core.mamba_state_entry_table import mamba_host_state_publish


def publish_neural_mamba_host() raises:
    var key = NeuralEntryKey(Int(GLOBAL_NUMERIC_MODE), 0)
    var api = NATIVE_MAMBA_HOST_BLOCK_API.get_or_create_ptr()
    neural_table_publish[NEURAL_TABLE_MAMBA_TRAIN](key, Int(api), api[].entries[0])
    var state = NATIVE_MAMBA_HOST_STATE_API.get_or_create_ptr()
    mamba_host_state_publish(Int(GLOBAL_NUMERIC_MODE), Int(state), state[].entries[0])
