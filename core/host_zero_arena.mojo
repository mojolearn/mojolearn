# SPDX-License-Identifier: Apache-2.0
"""The host arena of one x_prep program run by a typed caller (the prep,
select and classify members): `max(words, 1)` zeroed float32 words from libc
`calloc`. A large block is fresh zero pages the system maps on first touch,
as the doors' anonymous mapping (`xprep_storage`, R3 "mapped"), so arena
words a run never touches on the host (an input put on the device from its
own buffer, device scratch) cost nothing, and no serial fill pass runs
before the program (lane fx-audit 25, audit-trees-2 S1; was a `fill=` List
at thirteen member sites). Same zero words as the fill: no bit changes.

Moved here from bindings/prep_classify_prog_run.mojo `ClsArena` (now an
alias of this type) so the members share the one arena.
"""
from std.ffi import external_call, c_size_t

comptime _ArenaBytes = MutPointer[UInt8, MutUntrackedOrigin]
comptime _ArenaF32 = MutPointer[Float32, MutUntrackedOrigin]


struct HostZeroArena(Movable):
    """Zeroed float32 words for one run, freed with the value."""
    var address: Int
    var words: Int

    def __init__(out self, words: Int) raises:
        self.words = max(words, 1)
        var raw = external_call["calloc", Optional[_ArenaBytes]](c_size_t(self.words), c_size_t(4))
        if not raw:
            raise Error("MemoryError: x_prep: the program arena could not be allocated")
        self.address = Int(raw.value())

    def __del__(deinit self):
        if self.address:
            external_call["free", NoneType](_ArenaBytes(unsafe_from_address=self.address))

    def unsafe_ptr(self) -> _ArenaF32:
        """The first word (the form a `List[Float32]` arena was read in)."""
        return _ArenaF32(unsafe_from_address=self.address)

    def f32(self, offset: Int) -> _ArenaF32:
        return _ArenaF32(unsafe_from_address=self.address + 4 * offset)

    def __len__(self) -> Int:
        return self.words
