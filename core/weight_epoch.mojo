# SPDX-License-Identifier: Apache-2.0
"""The process weight epoch (lane x45c-causal-generate-program-decode,
2026-11-09): the version key of every resident copy of a neural model's
weights, in place of comparing the caller's weight bytes on every call.

A resident copy (a block session's packed weights, a cached Mamba program,
a CausalLM decode program) is keyed by the host weight ADDRESSES it was
loaded from and the epoch at its load. The binding's one weight door
(bindings/generation_buffers.mojo `generation_weight_address`) makes every
weight it hands to a kernel read-only (an exact mojolearn Array's flag, an
exact ndarray's writeable flag; the p3 fitted-freeze scheme) and bumps the
epoch whenever it hands over a weight that was writeable until then (a new
or reassigned array) or a per-call copy of storage it cannot freeze. So
equal addresses at an unchanged epoch hold the same bytes: no pass over the
weights. A changed epoch costs one re-upload, never a compare.

Public loads (`CausalLM.load`, `load_state_dict`, attribute assignment) give
new arrays, which bump at their first use. In-place edits of a frozen weight
are refused by the array itself.
"""
from std.atomic import Atomic, Ordering
from std.sys.compile import is_defined
from core.shared_slot import SharedSlot, shared_slot_lock, shared_slot_unlock


#: integ-w4: the epoch is a cross-module slot (every extension module that
#: keys a resident weight copy reads it), so it follows lane s1's rule
#: (core/shared_slot.mojo): versioned name, `layout` first, checked on read.
comptime _EPOCH_LAYOUT = 2
comptime _EPOCH_NAME = "MojolearnWeightEpochV2"


struct _WeightEpoch(Defaultable, Movable):
    var word: List[Int64]

    def __init__(out self):
        self.word = List[Int64](length=1, fill=Int64(1))


comptime _EPOCH = SharedSlot[_WeightEpoch, _EPOCH_NAME, _EPOCH_LAYOUT]


def _ptr() raises -> MutPointer[Int64, MutUntrackedOrigin]:
    var g = _EPOCH.get_or_create_ptr()
    return MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(g[].word.unsafe_ptr()))


def weight_epoch() raises -> Int:
    """The current epoch (>= 1)."""
    return Int(Atomic.load[ordering=Ordering.ACQUIRE](_ptr()))


def weight_epoch_bump() raises:
    """A weight a kernel reads may hold new bytes at an address a resident
    copy is keyed by: every resident copy reloads at its next use."""
    _ = Atomic.fetch_add(_ptr(), Int64(1))


struct WeightKey(Copyable, Movable):
    """The version key of one resident weight copy: the host addresses it was
    loaded from and the epoch at that load.

        var key = WeightKey()                # empty: never current
        if not key.current(addrs):           # compare: no pass over weights
            upload(addrs)                    # (the caller's copy, once)
            key = WeightKey.take(addrs)      # read the epoch BEFORE the upload
                                             # (take first, then upload, is safe too:
                                             # a bump in between only re-uploads)
    """
    var addrs: List[Int]
    var epoch: Int

    def __init__(out self):
        self.addrs = List[Int]()
        self.epoch = 0

    def __init__(out self, var addrs: List[Int], epoch: Int):
        self.addrs = addrs^
        self.epoch = epoch

    @staticmethod
    def take(addrs: List[Int]) raises -> WeightKey:
        """The key of a copy loaded now from `addrs`."""
        return WeightKey(addrs.copy(), weight_epoch())

    def current(self, addrs: List[Int]) raises -> Bool:
        """Whether the copy still holds the bytes at `addrs`: same addresses
        and no bump since it was loaded."""
        return self.epoch != 0 and self.epoch == weight_epoch() and self.addrs == addrs


# ---- lane fix-samba-weight-epoch-per-model (2026-11-13): the per-registry
# epoch. The process epoch above moves on ANY weight write of ANY model, so a
# Samba stack's resident registry copy (training/samba_resident.mojo, the
# stateless forward; training/samba_program.mojo, the stack program) was
# re-uploaded whole (~22 MB at the board shape) whenever ANOTHER model wrote
# its weights: two stacks alternating (train one, predict another; the
# board's warmup model, then the scored one) paid one full upload per model
# per call. A Samba resident copy is now keyed by (registry address, that
# registry's stamp): each operation that writes a model's `flat` stamps it
# after the write from the address it holds (the train step's resident and
# host routes, publish, load_weights; the public door
# bindings/samba_public_native_pack.mojo stamps only a FAILED write), so a
# write to model A never invalidates model B's copy. The door still moves
# the process epoch on those writes: the per-block programs
# (mamba/impl/mamba_block_program.mojo, transformer/attn_step_session.mojo)
# are keyed by views INTO a Samba registry when a stack runs its blocks
# (bindings/samba_training_native_door.mojo, samba_inference_native_door.mojo)
# and read only the process epoch.
#
# -D MOJOLEARN_SAMBA_WEIGHT_EPOCH_GLOBAL (default off) keys the Samba copies by
# the process epoch again (every registry reads and bumps `weight_epoch`): the
# B arm of the owed A/B (row fix-samba-weight-epoch-per-model in
# bench/results/paper_compare/owed-pre0867.json). Same kernels, same bytes
# uploaded when an upload happens: only the upload COUNT differs.
comptime SAMBA_WEIGHT_EPOCH_GLOBAL = is_defined["MOJOLEARN_SAMBA_WEIGHT_EPOCH_GLOBAL"]()

comptime _REGISTRY_LAYOUT = 2
comptime _REGISTRY_NAME = "MojolearnRegistryEpochV2"
#: Registry slots in the process table (one per model registry a resident
#: copy has been keyed by; a process holds a handful). A full table reuses
#: the oldest slot round-robin and gives it a fresh stamp, so a key still
#: pointing at that slot reloads once: bounded memory, never a stale copy.
comptime _REGISTRY_CAP = 1024
#: `ctl` layout: [lock, last stamp handed out, slots in use, next slot to
#: reuse when full].
comptime _C_LOCK = 0
comptime _C_STAMP = 1
comptime _C_USED = 2
comptime _C_CURSOR = 3

# Hot-path audit 2026-11-13 (BLOCK core/weight_epoch.mojo:155): the READ is
# lock-free. Each registry owns one epoch word at a fixed slot of the table
# (`epochs[i]`, allocated once at the table's creation, never moved). A
# resident copy's `RegistryKey` resolves its registry's slot ONCE, when it
# loads (`take`, under the lock, beside an upload), and keeps the word's
# address; `current` is then one `Atomic.load` of that word and two integer
# compares, as the process epoch's `WeightKey.current` was. The lock is taken
# only by writers (`registry_epoch_bump`) and by `take`. Every stamp is
# unique (one process counter), so a reused slot (full table) and a bump both
# leave every key taken before them not current.


struct _RegistryEpochs(Defaultable, Movable):
    var ctl: List[Int64]
    var addrs: List[Int]
    var epochs: List[Int64]

    def __init__(out self):
        # side-effect free (core/shared_slot.mojo first creation); the three
        # lists are sized once here and never grow, so a slot's epoch word
        # keeps its address for the process
        self.ctl = List[Int64](length=4, fill=Int64(0))
        self.addrs = List[Int](length=_REGISTRY_CAP, fill=0)
        self.epochs = List[Int64](length=_REGISTRY_CAP, fill=Int64(0))


comptime _REGISTRY = SharedSlot[_RegistryEpochs, _REGISTRY_NAME, _REGISTRY_LAYOUT]
comptime _EpochWord = MutPointer[Int64, MutUntrackedOrigin]


def _lock_word(slot: UnsafePointer[_RegistryEpochs, MutUntrackedOrigin]) -> _EpochWord:
    return _EpochWord(unsafe_from_address=Int(slot[].ctl.unsafe_ptr()) + _C_LOCK * 8)


def _slot_word(slot: UnsafePointer[_RegistryEpochs, MutUntrackedOrigin], i: Int) -> _EpochWord:
    return _EpochWord(unsafe_from_address=Int(slot[].epochs.unsafe_ptr()) + i * 8)


def _find(slot: UnsafePointer[_RegistryEpochs, MutUntrackedOrigin], addr: Int) -> Int:
    """The slot of `addr`, -1 when it has none. Caller holds the lock."""
    var used = Int(slot[].ctl[_C_USED])
    for i in range(used):  # driver-loop: over the registries keyed in this process (lock held; load or write only)
        if slot[].addrs[i] == addr:
            return i
    return -1


def _next_stamp(slot: UnsafePointer[_RegistryEpochs, MutUntrackedOrigin]) -> Int64:
    """A stamp no word holds yet. Caller holds the lock."""
    var stamp = slot[].ctl[_C_STAMP] + 1
    slot[].ctl[_C_STAMP] = stamp
    return stamp


def registry_word(addr: Int, mut epoch: Int) raises -> Int:
    """The address of the epoch word of the registry at host address `addr`,
    giving it a slot (and a fresh stamp) when it has none; `epoch` receives
    the word's value read under the same lock (a slot reused by another
    thread right after cannot hand this key the new owner's stamp). Resolved
    once per load by `RegistryKey.take`, never per call."""
    var slot = _REGISTRY.get_or_create_ptr()
    var lock = _lock_word(slot)
    shared_slot_lock(lock)
    var i = _find(slot, addr)
    if i < 0:
        var used = Int(slot[].ctl[_C_USED])
        if used < _REGISTRY_CAP:
            i = used
            slot[].ctl[_C_USED] = Int64(used + 1)
        else:
            # Full: reuse the next slot round-robin. Its fresh stamp leaves
            # every key of the registry that held it not current.
            i = Int(slot[].ctl[_C_CURSOR])
            slot[].ctl[_C_CURSOR] = Int64((i + 1) % _REGISTRY_CAP)
        slot[].addrs[i] = addr
        Atomic.store[ordering=Ordering.RELEASE](_slot_word(slot, i), _next_stamp(slot))
    var word = Int(_slot_word(slot, i))
    epoch = Int(Atomic.load[ordering=Ordering.ACQUIRE](_slot_word(slot, i)))
    shared_slot_unlock(lock)
    return word


def registry_epoch(addr: Int) raises -> Int:
    """The epoch (>= 1) of the registry at host address `addr` (its slot's
    word; a load-time read, never on a call's path)."""
    comptime if SAMBA_WEIGHT_EPOCH_GLOBAL:
        return weight_epoch()
    else:
        var epoch = 0
        _ = registry_word(addr, epoch)
        return epoch


def registry_epoch_bump(addr: Int) raises:
    """The registry at `addr` may hold new bytes: every resident copy keyed by
    it reloads at its next use; copies of other registries are untouched.
    A zero address (no registry) stamps nothing; nor does a registry no copy
    was ever keyed by (it has no slot, so no key can hold it)."""
    comptime if SAMBA_WEIGHT_EPOCH_GLOBAL:
        weight_epoch_bump()
    else:
        if addr == 0:
            return
        var slot = _REGISTRY.get_or_create_ptr()
        var lock = _lock_word(slot)
        shared_slot_lock(lock)
        var i = _find(slot, addr)
        if i >= 0:
            Atomic.store[ordering=Ordering.RELEASE](_slot_word(slot, i), _next_stamp(slot))
        shared_slot_unlock(lock)


struct RegistryKey(Copyable, Movable):
    """The version key of one resident copy of ONE weight registry (a Samba
    stack's flat registry): its host address, that registry's epoch at the
    load and the address of the registry's epoch word (resolved once, at the
    load). A write to another registry leaves it current.

        var key = RegistryKey()                  # empty: never current
        if not key.current(addr):                # one atomic load, no lock
            var fresh = RegistryKey.take(addr)   # read the epoch BEFORE the upload
            upload(addr)
            key = fresh^
    """
    var addr: Int
    var epoch: Int
    var word: Int

    def __init__(out self):
        self.addr = 0
        self.epoch = 0
        self.word = 0

    def __init__(out self, addr: Int, epoch: Int, word: Int):
        self.addr = addr
        self.epoch = epoch
        self.word = word

    @staticmethod
    def take(addr: Int) raises -> RegistryKey:
        """The key of a copy loaded now from the registry at `addr`."""
        comptime if SAMBA_WEIGHT_EPOCH_GLOBAL:
            return RegistryKey(addr, weight_epoch(), 0)
        else:
            var epoch = 0
            var word = registry_word(addr, epoch)
            return RegistryKey(addr, epoch, word)

    def current(self, addr: Int) raises -> Bool:
        """Whether the copy still holds the bytes of the registry at `addr`:
        same address and no write to that registry since the load. One
        atomic load of the held word; no lock, no table scan."""
        if self.epoch == 0 or self.addr != addr:
            return False
        comptime if SAMBA_WEIGHT_EPOCH_GLOBAL:
            return self.epoch == weight_epoch()
        else:
            return self.word != 0 and self.epoch == Int(
                Atomic.load[ordering=Ordering.ACQUIRE](_EpochWord(unsafe_from_address=self.word)))
