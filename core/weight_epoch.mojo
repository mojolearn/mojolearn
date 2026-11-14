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
# registry's stamp): the public door stamps the written registry
# (bindings/samba_public_native_pack.mojo, the operations of
# training/samba_public_run.mojo `_samba_writes_registry` that write a model's
# `flat`), so a write to model A never invalidates model B's copy. The door
# still moves the process epoch on those writes: the per-block programs
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

comptime _REGISTRY_LAYOUT = 1
comptime _REGISTRY_NAME = "MojolearnRegistryEpochV1"
#: Distinct registries stamped before the table is forgotten (one entry per
#: model registry a write has ever touched; a process holds a handful). A full
#: table is cleared and the floor moves past every stamp handed out, so every
#: copy keyed before reloads once: bounded memory, never a stale copy.
comptime _REGISTRY_CAP = 1024
#: `words` layout: [lock, last stamp handed out, floor (an unstamped
#: registry's epoch)].
comptime _W_LOCK = 0
comptime _W_STAMP = 1
comptime _W_FLOOR = 2


struct _RegistryEpochs(Defaultable, Movable):
    var words: List[Int64]
    var addrs: List[Int]
    var stamps: List[Int]

    def __init__(out self):
        # side-effect free (core/shared_slot.mojo first creation)
        self.words = List[Int64](length=3, fill=Int64(1))
        self.words[_W_LOCK] = Int64(0)
        self.addrs = List[Int]()
        self.stamps = List[Int]()


comptime _REGISTRY = SharedSlot[_RegistryEpochs, _REGISTRY_NAME, _REGISTRY_LAYOUT]


def registry_epoch(addr: Int) raises -> Int:
    """The epoch (>= 1) of the registry at host address `addr`: the stamp of
    its last write through the door, or the floor when none is recorded."""
    comptime if SAMBA_WEIGHT_EPOCH_GLOBAL:
        return weight_epoch()
    else:
        var slot = _REGISTRY.get_or_create_ptr()
        var lock = rebind[MutPointer[Int64, MutUntrackedOrigin]](slot[].words.unsafe_ptr())
        shared_slot_lock(lock)
        var found = Int(slot[].words[_W_FLOOR])
        for i in range(len(slot[].addrs)):  # driver-loop: over the registries a write has stamped
            if slot[].addrs[i] == addr:
                found = slot[].stamps[i]
                break
        shared_slot_unlock(lock)
        return found


def registry_epoch_bump(addr: Int) raises:
    """The registry at `addr` may hold new bytes: every resident copy keyed by
    it reloads at its next use; copies of other registries are untouched.
    A zero address (no registry) stamps nothing."""
    comptime if SAMBA_WEIGHT_EPOCH_GLOBAL:
        weight_epoch_bump()
    else:
        if addr == 0:
            return
        var slot = _REGISTRY.get_or_create_ptr()
        var lock = rebind[MutPointer[Int64, MutUntrackedOrigin]](slot[].words.unsafe_ptr())
        shared_slot_lock(lock)
        var stamp = Int(slot[].words[_W_STAMP]) + 1
        slot[].words[_W_STAMP] = Int64(stamp)
        var at = -1
        for i in range(len(slot[].addrs)):  # driver-loop: over the registries a write has stamped
            if slot[].addrs[i] == addr:
                at = i
                break
        if at >= 0:
            slot[].stamps[at] = stamp
        elif len(slot[].addrs) >= _REGISTRY_CAP:
            # Full: forget every entry and raise the floor to this new stamp.
            # Every unstamped registry (this one included) now reads a value
            # no key was taken at, so each copy reloads once.
            slot[].addrs.clear()
            slot[].stamps.clear()
            slot[].words[_W_FLOOR] = Int64(stamp)
        else:
            slot[].addrs.append(addr)
            slot[].stamps.append(stamp)
        shared_slot_unlock(lock)


struct RegistryKey(Copyable, Movable):
    """The version key of one resident copy of ONE weight registry (a Samba
    stack's flat registry): its host address and that registry's epoch at the
    load. A write to another registry leaves it current.

        var key = RegistryKey()                  # empty: never current
        if not key.current(addr):
            var fresh = RegistryKey.take(addr)   # read the epoch BEFORE the upload
            upload(addr)
            key = fresh^
    """
    var addr: Int
    var epoch: Int

    def __init__(out self):
        self.addr = 0
        self.epoch = 0

    def __init__(out self, addr: Int, epoch: Int):
        self.addr = addr
        self.epoch = epoch

    @staticmethod
    def take(addr: Int) raises -> RegistryKey:
        """The key of a copy loaded now from the registry at `addr`."""
        return RegistryKey(addr, registry_epoch(addr))

    def current(self, addr: Int) raises -> Bool:
        """Whether the copy still holds the bytes of the registry at `addr`:
        same address and no write to that registry since the load."""
        return self.epoch != 0 and self.addr == addr and self.epoch == registry_epoch(addr)
