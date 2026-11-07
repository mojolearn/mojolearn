# SPDX-License-Identifier: Apache-2.0
"""Device-resident fitted parameters of the recurrent and MLP models (lane
perf3-seq-neural, 2026-11-06): the descriptor words, shared by the host
binding that names a fitted block and the generic predict entries.

A predict names its fitted block with three extra address words after the
operation's own addresses: `owner` (the model's identity), `token` (a
process-unique number the binding gives each admitted block; a refit or a
replaced field gets a new one) and `n` (the block's float count). The device
dispatch (`sequence/perf3_seq_resident.mojo`) swaps the params address for
the block's resident device copy, uploading it on the first predict only,
and leaves one marker word `SEQ_RESIDENT_ON_EXEC` in place of the three. A
host executor ignores the three words. Host-only code: no GPU import here,
so the CPU dispatch and the bindings can import it.
"""
from std.ffi import _Global

#: the marker left by the device dispatch: params is already this
#: executor's device memory, read-only; no allocation, no upload.
comptime SEQ_RESIDENT_ON_EXEC = 0x5E97E51


struct _SeqResidentTokens(Defaultable, Movable):
    var next: Int

    def __init__(out self):
        self.next = 1


comptime _SEQ_RESIDENT_TOKENS = _Global[
    StorageType=_SeqResidentTokens, name="MojoXSequenceResidentTokens", init_fn=_SeqResidentTokens.__init__
]


def seq_resident_token() raises -> Int:
    """A new process-unique, nonzero token for one admitted fitted block.
    Called with the GIL held (the binding's record write)."""
    var tokens = _SEQ_RESIDENT_TOKENS.get_or_create_ptr()
    var token = tokens[].next
    tokens[].next += 1
    return token


def seq_resident_words(addresses: List[Int], base: Int, operation: String) raises -> Bool:
    """Admit a predict's address list: `base` words (parameters uploaded
    per call), `base + 3` (a named block on an executor that keeps none:
    the same upload), or `base + 1` ending in the device dispatch's marker
    (parameters resident). True when resident."""
    var n = len(addresses)
    if n == base or n == base + 3:
        return False
    if n == base + 1 and addresses[base] == SEQ_RESIDENT_ON_EXEC:
        return True
    raise Error(operation + ": invalid address descriptor")


# Free at model death (lane ne1-a-cluster-spectral, owed row N172). The
# device store lives in the GPU sequence binding and the record that names a
# block in the provider; neither sees the model die. The model's death
# callback (the records module's release, bindings/native_model_records.mojo,
# OWED-SHARED to ne1-e-records) posts the model's identity word here, and the
# device store frees that model's block at its next call, before any lookup
# (`sequence/perf3_seq_resident.mojo` `_drain_dead`): a new model at the
# reused address never sees the old block. Host-only: the provider and the
# records module can import it.
comptime SEQ_DEAD_OWNERS_MAX = 4096
"""Posted deaths kept before the oldest are dropped (a bound on host memory;
a dropped post leaves that block to the store's least-recently-used
eviction, which is where it went before)."""


struct _SeqDeadOwners(Defaultable, Movable):
    var owners: List[Int]

    def __init__(out self):
        self.owners = List[Int]()


comptime _SEQ_DEAD_OWNERS = _Global[
    StorageType=_SeqDeadOwners, name="MojoXSequenceResidentDeadOwnersV1", init_fn=_SeqDeadOwners.__init__
]


def seq_resident_owner_died(owner: Int):
    """Post the death of the model whose identity word is `owner` (called
    with the GIL held, from the death callback). Non-raising."""
    try:
        var dead = _SEQ_DEAD_OWNERS.get_or_create_ptr()
        if len(dead[].owners) >= SEQ_DEAD_OWNERS_MAX:
            _ = dead[].owners.pop(0)
        dead[].owners.append(owner)
    except:
        pass


def seq_resident_take_dead() raises -> List[Int]:
    """Every posted death since the last take (the device store's drain)."""
    var dead = _SEQ_DEAD_OWNERS.get_or_create_ptr()
    var taken = dead[].owners^
    dead[].owners = List[Int]()
    return taken^
