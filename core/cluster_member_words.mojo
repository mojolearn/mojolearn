# SPDX-License-Identifier: Apache-2.0
"""Float32 data blocks carried inside a member's NEW parameter words (lane
mf-clu6).

A setting whose value is data (KMeans / MiniBatchKMeans starting centers,
MeanShift seeds, a per-sample AffinityPropagation preference, the
BayesianGaussianMixture priors and warm state) is read once at the caller's
UNPACK and travels in `MemberPlan.params` behind the family's scalar words:
its float32 words are packed two per Float64 word, as BYTES (one copy, no
conversion and no per-value loop), so the member reads back the exact words
the class's own door hands its kernel.

SERIAL BY NATURE: one block copy per member admission or state mint.
"""
from std.memory import unsafe_memcpy

comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]


def member_words_f32_slots(count: Int) -> Int:
    """The Float64 words `count` packed float32 words take."""
    return (count + 1) // 2 if count > 0 else 0


def member_words_append_f32(mut words: List[Float64], address: Int, count: Int):
    """Append the `count` float32 words at host `address` to `words`, packed
    two per Float64 word (their bytes). A trailing half word is zero."""
    if count <= 0:
        return
    var start = len(words)
    words.resize(start + member_words_f32_slots(count), 0.0)
    unsafe_memcpy(dest=_F32P(unsafe_from_address=Int(words.unsafe_ptr()) + 8 * start),
        src=_F32P(unsafe_from_address=address), count=count)


def member_words_take_f32(words: Int, at: Int, count: Int) -> List[Float32]:
    """The `count` float32 words packed from Float64 word `at` of the word
    array at host address `words`, as an owned list (one copy)."""
    if count <= 0:
        return List[Float32]()
    var values = List[Float32](unsafe_uninit_length=count)
    unsafe_memcpy(dest=values.unsafe_ptr(), src=_F32P(unsafe_from_address=words + 8 * at), count=count)
    return values^


def member_words_byte_slots(count: Int) -> Int:
    """The Float64 words `count` packed bytes take."""
    return (count + 7) // 8 if count > 0 else 0


def member_words_append_text(mut words: List[Float64], text: String):
    """Append the UTF-8 bytes of `text`, packed eight per Float64 word (a
    name the family hands its entry by name, as the class's door does)."""
    var count = text.byte_length()
    if count <= 0:
        return
    var start = len(words)
    words.resize(start + member_words_byte_slots(count), 0.0)
    unsafe_memcpy(dest=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(words.unsafe_ptr()) + 8 * start),
        src=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(text.unsafe_ptr())), count=count)


def member_words_take_text(words: Int, at: Int, count: Int) raises -> String:
    """The `count` UTF-8 bytes packed from Float64 word `at` of the word
    array at host address `words`, as a String (length-delimited)."""
    if count <= 0:
        return String("")
    var bytes = Span(unsafe_ptr=Pointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=words + 8 * at), length=count)
    return String(from_utf8=bytes)
