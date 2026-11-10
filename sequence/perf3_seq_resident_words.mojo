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
from std.atomic import Atomic, Ordering
from core.records_death_ring import records_dead_keys_since
from core.shared_slot import SharedSlot

#: the marker left by the device dispatch: params is already this
#: executor's device memory, read-only; no allocation, no upload.
comptime SEQ_RESIDENT_ON_EXEC = 0x5E97E51


struct _SeqResidentTokens(Defaultable, Movable):
    var next: Int

    def __init__(out self):
        self.next = 1


comptime _SEQ_RESIDENT_TOKENS = SharedSlot[_SeqResidentTokens, "MojolearnXSequenceResidentTokensV1"]


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
# block in the provider; neither sees the model die. The records module's
# release (bindings/native_model_records.mojo) posts the model's identity
# word on its dead-key ring (core/records_death_ring.mojo, Python-free);
# `seq_resident_take_dead` pulls the ring from
# the last sequence word it saw (no callback, no function pointer), and the
# device store frees those models' blocks at its next call, before any
# lookup (`sequence/perf3_seq_resident.mojo` `_drain_dead_locked`): a new
# model at the reused address never sees the old block. Host-only: the
# provider and the records module can import it.
comptime SEQ_DEAD_OWNERS_MAX = 4096
"""Posted deaths kept before the oldest are dropped (a bound on host memory;
a dropped post leaves that block to the store's least-recently-used
eviction, which is where it went before)."""


struct _SeqDeadOwners(Defaultable, Movable):
    var owners: List[Int]
    # One lock word: the post runs with the GIL held, the take from a device
    # entry that may run with it released.
    var lock: List[Int64]
    # The records module's dead-key sequence word this queue pulled to.
    var seen: Int

    def __init__(out self):
        self.owners = List[Int]()
        self.lock = List[Int64](length=1, fill=Int64(0))
        self.seen = 0


comptime _SEQ_DEAD_OWNERS = SharedSlot[_SeqDeadOwners, "MojolearnXSequenceResidentDeadOwnersV4"]


def _dead_lock(dead: UnsafePointer[_SeqDeadOwners, MutAnyOrigin]):
    var word = rebind[MutPointer[Int64, MutUntrackedOrigin]](dead[].lock.unsafe_ptr())
    while True:  # driver-loop: spin on the queue's lock word (held for one list step only)
        var expected = Int64(0)
        if Atomic.compare_exchange[
            success_ordering=Ordering.SEQUENTIAL,
            failure_ordering=Ordering.RELAXED,
        ](word, expected, Int64(1)):
            return


def _dead_unlock(dead: UnsafePointer[_SeqDeadOwners, MutAnyOrigin]):
    var word = rebind[MutPointer[Int64, MutUntrackedOrigin]](dead[].lock.unsafe_ptr())
    Atomic.store[ordering=Ordering.RELEASE](word, Int64(0))


def _dead() raises -> UnsafePointer[_SeqDeadOwners, MutAnyOrigin]:
    return _SEQ_DEAD_OWNERS.get_or_create_ptr().unsafe_origin_cast[MutAnyOrigin]()


def seq_resident_owner_died(owner: Int):
    """Post the death of the model whose identity word is `owner` again
    (the store's repost of a pinned slot). Non-raising."""
    try:
        var dead = _dead()
        _dead_lock(dead)
        if len(dead[].owners) >= SEQ_DEAD_OWNERS_MAX:
            _ = dead[].owners.pop(0)
        dead[].owners.append(owner)
        _dead_unlock(dead)
    except:
        pass


def seq_resident_take_dead(mut forget_all: Bool) raises -> List[Int]:
    """Every death since the last take (the device store's drain): the
    records module's dead-key ring since `seen`, after the reposts. Sets
    `forget_all` when the ring overflowed since the last take (which models
    died is not known: the store frees every block it can). Any thread: the
    queue and the ring each take their own lock word."""
    var dead = _dead()
    _dead_lock(dead)
    var released = List[Int]()
    var now = records_dead_keys_since(dead[].seen, released)
    if now < 0:
        forget_all = True
        dead[].seen = -now
    else:
        dead[].seen = now
    var taken = dead[].owners.copy()
    dead[].owners.clear()
    _dead_unlock(dead)
    for j in range(len(released)):  # driver-loop: over models released since the last take
        taken.append(released[j])
    return taken^
