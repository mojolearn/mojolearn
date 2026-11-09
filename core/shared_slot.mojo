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

THE RULE for every cross-module slot of ours:
  * the name is `Mojolearn<What>V<n>` (tier suffix only when the contents
    differ by tier); bump `n` on any field change;
  * the struct's FIRST field is `layout: Int`, set by `init_fn` to
    `shared_slot_word[T](n)`, and every accessor calls
    `shared_slot_check(slot[].layout, shared_slot_word[T](n), name)` before
    any other field is read. A mismatch raises
    `mojolearn: mixed binary versions (<name>)` instead of reinterpreting.
    The word folds in `size_of[T]()`, so a field added without a bump is
    caught too whenever the size moves.

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
    store. core/neural_context.mojo `process_ctx` is the model.
(The ask for Modular, recorded here: document `_Global` first-creation
semantics, and whether `init_fn` running twice is a guaranteed contract.)
"""

from std.atomic import Atomic, Ordering
from std.sys.info import size_of

#: 'MJLN' in the high half: a word an older layout's first field (a pointer,
#: a count, an Optional) is very unlikely to hold.
comptime SHARED_SLOT_MAGIC: Int = 0x4D4A4C4E << 32


@always_inline
def shared_slot_word[T: AnyType](version: Int) -> Int:
    """The layout word of a slot struct `T` at layout version `version`:
    the magic, the version (bits 20-31) and the struct size (bits 0-19)."""
    return SHARED_SLOT_MAGIC | ((version & 0xFFF) << 20) | (size_of[T]() & 0xFFFFF)


@always_inline
def shared_slot_check(seen: Int, want: Int, name: StaticString) raises:
    """Refuse a slot whose creator was built with another layout."""
    if seen != want:
        raise Error("mojolearn: mixed binary versions (" + String(name) + ")")


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
