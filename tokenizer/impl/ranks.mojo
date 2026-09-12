# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The GPT-2 rank table: `tokenizer/data/gpt2_ranks.tsv` as a byte-keyed
lookup, plus the id -> bytes table `decode` reads.

THE FILE is `rank<TAB>hex_of_token_bytes`, one line per rank, ascending from
0, 50256 lines. Hex because a token's bytes are a slice of UTF-8 and need not
be valid UTF-8 alone -- id 188 is the single byte 0x00 and id 35496 is 128
bytes long. `<|endoftext|>` is NOT in the file; it is id 50256 and
`encoding.mojo` owns it.

WHY A HAND-ROLLED TABLE AND NOT `Dict[String, Int]`. The keys are RAW BYTE
SEQUENCES, including NUL and including lone continuation bytes. Making them
`String` keys would mean either putting invalid UTF-8 and NUL inside a
`String` (a question this lane does not want to answer) or spelling every
lookup through the byte-to-unicode bijection (`byte_unicode.mojo`), which
would allocate a String per candidate pair inside the merge loop. Open
addressing over a flat byte arena needs neither: the key is
`(buffer, start, count)` straight out of the caller's pre-token, so a rank
probe allocates nothing.

Load is one pass: append the bytes to the arena, record offset and length per
id, insert the id into the bucket array. Ranks ARE ids for GPT-2 -- the rank
of a merge and the id of the merged token are the same number -- so one table
serves the merge loop and the encoder.
"""

comptime FNV_OFFSET64: UInt64 = 14695981039346656037
comptime FNV_PRIME64: UInt64 = 1099511628211


struct RankTable(Copyable, Movable):
    """Raw-byte keyed rank lookup plus the id -> bytes table.

    `buckets` holds `id + 1`, so 0 means empty and no separate occupancy
    array is needed. Its length is a power of two at least twice the entry
    count, which keeps the load factor under 0.5 and the linear probe short.
    """

    var arena: List[UInt8]
    var offset: List[Int]
    var length: List[Int]
    var buckets: List[Int]
    var mask: Int

    def __init__(out self):
        self.arena = List[UInt8]()
        self.offset = List[Int]()
        self.length = List[Int]()
        self.buckets = List[Int]()
        self.mask = 0

    def n_tokens(self) -> Int:
        return len(self.offset)

    def _reserve(mut self, n_entries: Int):
        var size = 1
        while size < 2 * n_entries:
            size *= 2
        self.buckets = List[Int]()
        for _ in range(size):
            self.buckets.append(0)
        self.mask = size - 1

    def _hash(self, data: List[UInt8], start: Int, count: Int) -> UInt64:
        var h = FNV_OFFSET64
        for i in range(start, start + count):
            h = (h ^ UInt64(data[i])) * FNV_PRIME64
        return h

    def _equals(
        self, id: Int, data: List[UInt8], start: Int, count: Int
    ) -> Bool:
        if self.length[id] != count:
            return False
        var at = self.offset[id]
        for k in range(count):
            if self.arena[at + k] != data[start + k]:
                return False
        return True

    def _insert(mut self, id: Int) raises:
        var start = self.offset[id]
        var count = self.length[id]
        var slot = Int(self._hash(self.arena, start, count) & UInt64(self.mask))
        while self.buckets[slot] != 0:
            var other = self.buckets[slot] - 1
            if self._equals(other, self.arena, start, count):
                raise Error(
                    "gpt2_ranks.tsv: token bytes appear twice, at rank "
                    + String(other)
                    + " and rank "
                    + String(id)
                )
            slot = (slot + 1) & self.mask
        self.buckets[slot] = id + 1

    def rank(self, data: List[UInt8], start: Int, count: Int) -> Int:
        """The rank (= id) of `data[start : start + count]`, or -1.

        A zero-length key cannot be in the table (no token is empty) and is
        answered -1 without probing, so the merge loop never has to special
        case it."""
        if count <= 0:
            return -1
        var slot = Int(self._hash(data, start, count) & UInt64(self.mask))
        while self.buckets[slot] != 0:
            var id = self.buckets[slot] - 1
            if self._equals(id, data, start, count):
                return id
            slot = (slot + 1) & self.mask
        return -1

    def token_bytes(self, id: Int) raises -> List[UInt8]:
        if id < 0 or id >= len(self.offset):
            raise Error("token id out of range: " + String(id))
        var out = List[UInt8]()
        var at = self.offset[id]
        for k in range(self.length[id]):
            out.append(self.arena[at + k])
        return out^

    def append_token_bytes(self, mut out: List[UInt8], id: Int) raises:
        """`token_bytes` without the intermediate list, for `decode`."""
        if id < 0 or id >= len(self.offset):
            raise Error("token id out of range: " + String(id))
        var at = self.offset[id]
        for k in range(self.length[id]):
            out.append(self.arena[at + k])


def _hex_digit(b: UInt8) raises -> Int:
    var v = Int(b)
    if v >= 48 and v <= 57:  # '0'..'9'
        return v - 48
    if v >= 97 and v <= 102:  # 'a'..'f'
        return v - 97 + 10
    if v >= 65 and v <= 70:  # 'A'..'F'
        return v - 65 + 10
    raise Error("gpt2_ranks.tsv: bad hex digit: " + String(v))


def load_rank_table(path: String) raises -> RankTable:
    """One pass over the file. Every check here is a REFUSAL, not a repair:
    a rank out of order, an odd hex field, an empty token or a duplicate key
    means the table is not the table the fixture was produced against, and
    continuing would produce a mismatch report that blames the merge loop.
    """
    var text: String
    with open(path, "r") as f:
        text = f.read()

    # Two passes over the lines: the first counts entries so the bucket array
    # is sized once rather than rehashed, the second fills the tables.
    var n_lines = 0
    for line_slice in text.split("\n"):
        if String(line_slice).byte_length() > 0:
            n_lines += 1

    var table = RankTable()
    table._reserve(n_lines)

    var expect = 0
    for line_slice in text.split("\n"):
        var line = String(line_slice)
        if line.byte_length() == 0:
            continue
        var fields = line.split("\t")
        if len(fields) != 2:
            raise Error(
                path
                + ": line "
                + String(expect + 1)
                + " is not 'rank<TAB>hex', it has "
                + String(len(fields))
                + " fields"
            )
        var rank = Int(String(fields[0]))
        if rank != expect:
            raise Error(
                path
                + ": ranks must be 0.. ascending with no gaps; expected "
                + String(expect)
                + ", read "
                + String(rank)
            )
        var hexbytes = fields[1].as_bytes()
        var n_hex = len(hexbytes)
        if n_hex == 0 or n_hex % 2 != 0:
            raise Error(
                path
                + ": rank "
                + String(rank)
                + " has a "
                + String(n_hex)
                + "-digit hex field; a token is a non-empty whole number of"
                + " bytes"
            )
        table.offset.append(len(table.arena))
        table.length.append(n_hex // 2)
        for i in range(0, n_hex, 2):
            table.arena.append(
                UInt8(_hex_digit(hexbytes[i]) * 16 + _hex_digit(hexbytes[i + 1]))
            )
        table._insert(rank)
        expect += 1

    if table.n_tokens() == 0:
        raise Error(path + ": no tokens")
    return table^
