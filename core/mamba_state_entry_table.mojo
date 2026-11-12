# SPDX-License-Identifier: Apache-2.0
"""Which image serves a Mamba block's carried-state forward and decode, and
the CPU training image's own state slot (lane transformer-session-native,
2026-11-12; the item the samba-sessions-native lane left open).

Three images serve MAMBA_STATE_ABI_NAME (ops 2 * family + step: 4 Mamba-3
forward, 5 Mamba-3 decode step) for a Mamba block:

  a GPU block                  `_mojolearn_mamba` at (tier, gpu vendor) in
                               CAUSAL_TABLE_MAMBA (mamba/causal_entry_device.mojo);
  the CPU inference model's    `_mojolearn_neural_host` at (tier, cpu) in
  block                        CAUSAL_TABLE_MAMBA (core/causal_entry_neural_host.mojo);
  a CPU-build training block   `_mojolearn_mamba_host` at (tier, cpu) in THIS
                               table (mamba/host/neural_entry_host.mojo): it
                               cannot take the (tier, cpu) CAUSAL_TABLE_MAMBA
                               slot the neural host holds.

`Mamba3StateRoute` is the block's key, decided once at the public call's
UNPACK by its caller (the block's class and its connected training
selection); `state_ref()` resolves the entry. This replaces the former reach
into the block's binding module by attribute (`mamba3_forward`,
`mamba3_decode_step`) and the per-block state provider capsule for these
two calls. Publication and lookup are small metadata operations with no
independent numerical work to split.
"""
from core.entry_table import EntryRef, entry_table_publish, entry_table_resolve
from core.native_descriptor_abi import NativeDescriptorCall, MAMBA_STATE_ABI_NAME
from core.causal_lm_entry_table import CausalEntryKey, causal_ref, CAUSAL_TABLE_MAMBA, CAUSAL_MAMBA_OPERATIONS
from core.neural_entry_table import NeuralEntryKey, neural_ref, NEURAL_TABLE_MAMBA_TRAIN

comptime MAMBA3_STATE_FORWARD = 4
"""a: the Mamba-3 carried-state buffers; words [B, L, d, buffered, pending] -> buffered tokens."""
comptime MAMBA3_STATE_DECODE = 5
"""As MAMBA3_STATE_FORWARD for one token; words [B, d, buffered, pending]."""


def mamba_host_state_publish(mode: Int, publisher: Int, entry: NativeDescriptorCall) raises:
    """Publish `_mojolearn_mamba_host`'s state entry under (mode, cpu) (module
    init). A repeated init publishes the same pointer."""
    if mode == 1:
        entry_table_publish["MojolearnEntriesMambaHostStateIdenticalCPUV1"](mode, 0, publisher,
            [String(MAMBA_STATE_ABI_NAME)], [CAUSAL_MAMBA_OPERATIONS], [entry])
    elif mode == 0:
        entry_table_publish["MojolearnEntriesMambaHostStateFastCPUV1"](mode, 0, publisher,
            [String(MAMBA_STATE_ABI_NAME)], [CAUSAL_MAMBA_OPERATIONS], [entry])
    else:
        raise Error("native Mamba host state: invalid tier")


def mamba_host_state_ref(mode: Int) raises -> EntryRef:
    """The CPU training image's state entry for `mode`, resolved once; a
    missing publication refuses with the table's error."""
    if mode == 1:
        return entry_table_resolve["MojolearnEntriesMambaHostStateIdenticalCPUV1"](mode, 0, 0,
            String(MAMBA_STATE_ABI_NAME), CAUSAL_MAMBA_OPERATIONS)
    if mode == 0:
        return entry_table_resolve["MojolearnEntriesMambaHostStateFastCPUV1"](mode, 0, 0,
            String(MAMBA_STATE_ABI_NAME), CAUSAL_MAMBA_OPERATIONS)
    raise Error("native Mamba host state: invalid tier")


@fieldwise_init
struct Mamba3StateRoute(Copyable, Movable):
    var mode: Int
    """1 identical, 0 fast."""
    var vendor: Int
    """0 cpu, 1 metal, 2 cuda, 3 hip."""
    var training_host: Bool
    """A CPU-build training block (`_mojolearn_mamba_host`); False for a GPU
    block and for the CPU inference model's block."""

    def state_ref(self) raises -> EntryRef:
        if self.vendor == 0 and self.training_host:
            return mamba_host_state_ref(self.mode)
        return causal_ref[CAUSAL_TABLE_MAMBA](CausalEntryKey(self.mode, self.vendor))

    def has_sessions(self) -> Bool:
        """Whether the block's image keeps Mamba-3 prefill sessions: a
        training block (GPU or CPU build). The CPU inference model's block
        (`_mojolearn_neural_host`) has none."""
        return self.vendor != 0 or self.training_host

    def session_ref(self) raises -> EntryRef:
        """The Mamba entry of the training image that holds the block's
        prefill session (core/mamba3_prefill_session.mojo; lane
        mamba3-prefill-handle): NEURAL_TABLE_MAMBA_TRAIN at the block's key."""
        if not self.has_sessions():
            raise Error("RuntimeError: Mamba3 prefill sessions require a training binding")
        return neural_ref[NEURAL_TABLE_MAMBA_TRAIN](NeuralEntryKey(self.mode, self.vendor))
