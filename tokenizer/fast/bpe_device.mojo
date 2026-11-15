# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Apple FAST device BPE trainer and encoder (lane apple-fast-bpe, 2026-10-03) are gone; what
stays is the vocabulary bound, the result types and the False switches and refusing entry points
that `tokenizer/fast/bpe_fast_impl.mojo` and `bindings/_mojolearn_tokenizer_fast.mojo` still import
(the integrator removes them with their importers' dead branches).
"""

# TOMBSTONE: MOJOLEARN_BPE_ALL (unmeasured bundle, EXP:660 OPEN) deleted 2026-11-15 by lane/expdel2-30-neural; code recoverable at e39daf79ad.
# Restore: git apply experiments/removed/MOJOLEARN_BPE_ALL.patch
# TOMBSTONE: MOJOLEARN_BPE_TRAIN_DEVICE (unmeasured opt-in arm, EXP:666 OPEN) deleted 2026-11-15 by lane/expdel2-30-neural; code recoverable at e39daf79ad.
# Restore: git apply experiments/removed/MOJOLEARN_BPE_TRAIN_DEVICE.patch
# TOMBSTONE: MOJOLEARN_BPE_ENCODE_DEVICE (unmeasured opt-in arm, EXP:662 OPEN) deleted 2026-11-15 by lane/expdel2-30-neural; code recoverable at e39daf79ad.
# Restore: git apply experiments/removed/MOJOLEARN_BPE_ENCODE_DEVICE.patch
# TOMBSTONE: MOJOLEARN_BPE_MERGE_BATCH (unmeasured opt-in arm, EXP:665 OPEN) deleted 2026-11-15 by lane/expdel2-30-neural; code recoverable at e39daf79ad.
# Restore: git apply experiments/removed/MOJOLEARN_BPE_MERGE_BATCH.patch
# TOMBSTONE: MOJOLEARN_BPE_GROUP_FILTER (unmeasured opt-in arm, EXP:663 OPEN) deleted 2026-11-15 by lane/expdel2-30-neural; code recoverable at e39daf79ad.
# Restore: git apply experiments/removed/MOJOLEARN_BPE_GROUP_FILTER.patch
# TOMBSTONE: MOJOLEARN_BPE_LIVEBUF (unmeasured opt-in arm, EXP:664 OPEN) deleted 2026-11-15 by lane/expdel2-30-neural; code recoverable at e39daf79ad.
# Restore: git apply experiments/removed/MOJOLEARN_BPE_LIVEBUF.patch
comptime BPE_LIVEBUF = False
comptime BPE_MERGE_BATCH = False
comptime BPE_GROUP_FILTER = False
comptime BPE_TRAIN_DEVICE = False
comptime BPE_ENCODE_DEVICE = False

# (Sweep 2026-11-15: BpeMerges and the always-refusing bpe_train_merges_device / bpe_encode_batch_device removed with
# the dead device arms of tokenizer/fast/bpe_fast_impl.mojo; the BPE_* flags stay for bpe_fast_flags.)
#: the largest vocabulary whose pair key `left * V + right` (+1) fits an int32
comptime BPE_MAX_VOCAB = 46340


struct BpeDeviceTable(Defaultable, Movable):
    """The rank table's device copy of the removed LIVEBUF arm; holds nothing."""

    def __init__(out self):
        pass


