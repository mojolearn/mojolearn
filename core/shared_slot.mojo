# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The layout word and the lock word of a process-wide named slot
(lane s1-shared-state-hardening-and-selector, 2026-11-09; design note
~/mojolearn-evidence/design-shared-process-state.md sections 1-3).

WHAT IS SHARED. A `std.ffi._Global` is one slot per process per NAME: every
Mojolearn extension module (any .so, any merged group, host or GPU, core or
vendor wheel) finds the runtime's one `KGEN_CompilerRT_GetOrCreateGlobal`
table and keys it by the name string only. The creating module runs its
`init_fn`; every other module reinterprets those bytes with ITS struct. So a
name whose struct changed layout without a rename corrupts memory the moment
two modules built from different sources meet in one process (a stale .so in
a dev tree, a core and a vendor wheel of different versions).

THE RULE for every cross-module slot of ours (lane s2, 2026-11-09: every
slot is a `SharedSlot`; no hand-written `_Global` and no hand-written layout
field or check remain):
  * declare it `comptime MY_SLOT = SharedSlot[T, "Mojolearn<What>V<n>", n]`
    (tier suffix in the name only when the contents differ by tier); bump the
    name's `V<n>` and the version together on any field change of `T`;
  * `MY_SLOT.get_or_create_ptr()` stores the layout word in front of the
    value (`_SlotBox`) and checks it on every call before the value is
    handed out. A mismatch stops the process with
    `mojolearn: mixed binary versions (<name>)` instead of reinterpreting
    another binary's bytes. The word folds in `size_of[T]()`, so a field
    added without a bump is caught too whenever the size moves.

FIRST CREATION (S0, read from the runtime on the mac build,
`libKGENCompilerRTShared.dylib`: `KGEN_CompilerRT_GetOrCreateGlobal` at
0x49a8 tail-calls the table method at 0x8200c). The table is a fixed
open-addressed hash (4096 entries, 12 probes, a name compare per probe). On a
miss the CALLER runs `init_fn` with no lock held (0x820dc `blr x24`), builds
the entry and publishes it with one `casal` into the probed cell (0x82174).
A loser of that compare-and-swap calls `destroy_fn` on its own value
(0x82188 `blr x23`), frees its entry and returns the winner's pointer when
the names match. So the runtime publishes exactly one value per name and
every caller gets the same pointer, but `init_fn` CAN RUN MORE THAN ONCE,
concurrently, and a loser's value is destroyed. Consequences:
  * `init_fn` must be side-effect free besides building its value (no
    DeviceContext, no registration elsewhere, no Python object);
  * anything a slot creates LAZILY after `get_or_create_ptr` (a context, a
    list's first entry) is ours to guard: take the slot's lock word
    (`shared_slot_lock` / `shared_slot_unlock`) around the test and the
    store, and publish it with a flag so later reads take no lock.
    core/neural_context.mojo `process_ctx` is the model.

THE LINUX RUNTIME AGREES (lane s2, 2026-11-09, read-only). The Linux wheels
bundle their own copy (mojolearn_core 0.8.66 manylinux_2_35_x86_64,
`mojolearn/.libs/libKGENCompilerRTShared.so`, sha256 d8759513cd99a717...;
excerpts in ~/mojolearn-evidence/s2/linux-rt/). `KGEN_CompilerRT_
GetOrCreateGlobal` at 0x3f050 builds the table once under a C++ static
guard (`__cxa_guard_acquire`, memset of 0x8009 bytes: 4096 pointer cells
and a flag) and tail-calls the table method at 0xd0f60, the same algorithm
as the mac build: a hash of the name, linear probing of cells (h + i) &
0xfff for i < 12 with a length test and a `bcmp` per probe; on an empty cell
the CALLER runs `init_fn` with no lock (0xd1010 `callq *-0x58(%rbp)`), builds
a 0x38-byte entry {value, destroy_fn, name} and publishes it with one
`lock cmpxchgq` into the probed cell (0xd10be / 0xd10f4); a loser calls
`destroy_fn` on its own value (0xd110d `callq *%rsi`), frees its entry and
re-reads the winner (same name: the winner's pointer; else probing goes
on). So both runtimes publish one value per name, may run `init_fn` more
than once concurrently, and destroy the losers: the rules above hold on
Linux (NVIDIA and AMD wheels) and macOS alike.
(The ask for Modular, recorded here: document `_Global` first-creation
semantics, and whether `init_fn` running twice is a guaranteed contract.)
"""

from std.atomic import Atomic, Ordering
from std.ffi import _Global
from std.os import abort
from std.sys.info import size_of

#: 'MJLN' in the high half: a word an older layout's first field (a pointer,
#: a count, an Optional) is very unlikely to hold.
comptime SHARED_SLOT_MAGIC: Int = 0x4D4A4C4E << 32


@always_inline
def shared_slot_word[T: AnyType](version: Int) -> Int:
    """The layout word of a slot struct `T` at layout version `version`:
    the magic, the version (bits 20-31) and the struct size (bits 0-19)."""
    return SHARED_SLOT_MAGIC | ((version & 0xFFF) << 20) | (size_of[T]() & 0xFFFFF)


# Lane s2 removed `shared_slot_check`, the hand-written accessor check, with
# the last hand-written layout field: the check lives in
# `SharedSlot.get_or_create_ptr`.


@always_inline
def shared_slot_lock(word: MutPointer[Int64, MutUntrackedOrigin]):
    """Spin until this thread holds the lock word (0 free, 1 held)."""
    while True:  # driver-loop: spin on a slot's lock word
        var expected = Int64(0)
        if Atomic.compare_exchange[
            success_ordering=Ordering.SEQUENTIAL,
            failure_ordering=Ordering.RELAXED,
        ](word, expected, Int64(1)):
            return


@always_inline
def shared_slot_unlock(word: MutPointer[Int64, MutUntrackedOrigin]):
    """Release a lock word taken by `shared_slot_lock`."""
    Atomic.store[ordering=Ordering.RELEASE](word, Int64(0))


struct _SlotBox[T: Defaultable & Movable & Deinitable, version: Int](Defaultable, Movable):
    """What a `SharedSlot` stores in the runtime table: the layout word first,
    then the slot's own value."""

    var layout: Int
    var value: Self.T

    def __init__(out self):
        # side-effect free besides building the value (first creation, above)
        self.layout = shared_slot_word[Self.T](Self.version)
        self.value = Self.T()


struct SharedSlot[T: Defaultable & Movable & Deinitable, name: StaticString, version: Int = 1]:
    """A process-wide named slot that carries the layout word for its users
    (lane s2, 2026-11-09). A drop-in for
    `_Global[StorageType=T, name=name, init_fn=T.__init__]`: the same
    `get_or_create_ptr()` call returns a pointer to the `T`, after the layout
    word was checked, so no accessor can forget the check. A mismatch (a
    module built from another layout met this name in the process) stops the
    process with `mojolearn: mixed binary versions (<name>)` (reading on would
    corrupt memory). The accessor raises exactly where `_Global`'s does (its
    first-creation path), so every former `_Global` caller keeps its form.

    Names follow the rule above (`Mojolearn<What>V<n>`); `version` is the
    layout version folded into the word. Bump the name's `V<n>` and
    `version` together on any field change of `T`."""

    comptime _BOX = _SlotBox[Self.T, Self.version]
    comptime _SLOT = _Global[StorageType=Self._BOX, name=Self.name, init_fn=Self._BOX.__init__]

    @staticmethod
    @always_inline
    def get_or_create_ptr() raises -> UnsafePointer[Self.T, MutUntrackedOrigin]:
        """The slot's value, created on first use and kept for the process."""
        var box = Self._SLOT.get_or_create_ptr()
        if box[].layout != shared_slot_word[Self.T](Self.version):
            abort(String("mojolearn: mixed binary versions (", Self.name, ")"))
        return UnsafePointer(to=box[].value).unsafe_origin_cast[MutUntrackedOrigin]()
