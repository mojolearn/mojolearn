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
from std.ffi import _Global
from std.atomic import Atomic, Ordering
from core.shared_slot import shared_slot_word, shared_slot_check


#: integ-w4: the epoch is a cross-module slot (every extension module that
#: keys a resident weight copy reads it), so it follows lane s1's rule
#: (core/shared_slot.mojo): versioned name, `layout` first, checked on read.
comptime _EPOCH_LAYOUT = 1
comptime _EPOCH_NAME = "MojolearnWeightEpochV1"


struct _WeightEpoch(Movable):
    var layout: Int
    var word: List[Int64]

    def __init__(out self):
        self.layout = shared_slot_word[_WeightEpoch](_EPOCH_LAYOUT)
        self.word = List[Int64](length=1, fill=Int64(1))


comptime _EPOCH = _Global[StorageType=_WeightEpoch, name=_EPOCH_NAME, init_fn=_WeightEpoch.__init__]


def _ptr() raises -> MutPointer[Int64, MutUntrackedOrigin]:
    var g = _EPOCH.get_or_create_ptr()
    shared_slot_check(g[].layout, shared_slot_word[_WeightEpoch](_EPOCH_LAYOUT), _EPOCH_NAME)
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
