# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE BYTE-PAIR VOCABULARY TRAINER: count adjacent pairs, merge the winner,
repeat. Deterministic by construction.

`tokenizer/impl/bpe.mojo` APPLIES a vocabulary; this file BUILDS one. The
tokenizer has shipped without a trainer since mojolearn ships no vocabulary
(2026-09-15), so a user who wanted their own table had to train it somewhere
else. This is that trainer, and `python/mojolearn/_bpe_trainer.py` is an
independent second implementation of the same stated algorithm which
`tokenizer/checks/trainer_check.mojo` holds this one to, file byte for file
byte.

WHAT THE PROPERTY IS. Vocabulary training is HOST-ONLY everywhere -- Hugging
Face, SentencePiece and tiktoken all train on a CPU, because counting and
merging is not a matmul workload -- so there is no GPU path here and nothing
to be identical about across vendors. The property is:

    THE SAME CORPUS AND CONFIG PRODUCE THE SAME VOCABULARY BYTES ON ANY
    MACHINE AND ARCHITECTURE.

and it rests on four things, every one of which is visible in this file:

1.  A TOTAL ORDER ON THE TIE-BREAK. The winner is the pair of highest count;
    ties go to the SMALLEST `(left_id, right_id)`. The pair is held as the
    single integer `left_id * V + right_id`, so comparing keys ascending IS
    comparing `(left_id, right_id)` lexicographically, and distinct pairs
    have distinct keys. There is no pair the rule cannot separate.
2.  NO REDUCTION ORDER TO GET WRONG. Counting is single-threaded. A parallel
    count would have to merge per-shard counts in shard index order; this
    trainer does not take that risk.
3.  NO ITERATION ORDER REACHES THE RESULT. `touched` is walked in whatever
    order pairs were first seen, and that is SAFE HERE precisely because the
    selection is a total order over `(count, key)`: the same set of pairs
    yields the same winner however it is walked. The Python reference walks
    its pairs in sorted order instead, and the gate proves the two agree.
4.  NO FLOATS ANYWHERE. Counts, ids and the comparison are all integers.
    There is no score and no probability -- which is exactly where a unigram
    trainer's reproducibility goes.

ONE MORE RULE, written down because it is a real choice: when a pair's two
sides are EQUAL (`aa` inside `aaa`), counting sees two occurrences and
rewriting applies one merge. Counting overlapping and applying
non-overlapping is what the established trainers do; both halves are
deterministic, and the Python reference does the same thing.

THE SABOTAGE. `-D MOJOLEARN_BPE_TRAINER_SABOTAGE=1` reverses ONLY the
tie-break, taking the largest key among the pairs sharing the top count. It
must make the gate FAIL. Note what that arm is really testing: on a corpus
that never produces a tie it would be INERT and the gate would pass,
which is why `n_ties_broken` is carried out of the trainer and the fixture
carries a corpus engineered to tie.

COST. Each merge rescans every sequence, so training is O(merges x corpus
pre-tokens). A pre-token is a word, the fixtures are small, and this module
makes no speed claim; the incremental-bookkeeping form is the same output by
construction.
"""

from std.sys.compile import is_defined

from tokenizer.impl.pretokenize import pretokenize
from tokenizer.impl.unicode_class import UnicodeClasses

comptime BPE_TRAINER_SABOTAGE = is_defined["MOJOLEARN_BPE_TRAINER_SABOTAGE"]()
"""The negative control. Reverses the tie-break and nothing else."""

comptime FNV_OFFSET64: UInt64 = 14695981039346656037
comptime FNV_PRIME64: UInt64 = 1099511628211


struct PieceGroups(Copyable, Movable):
    """The corpus reduced to `(pre-token bytes, count)`, deduplicated.

    Open addressing over a flat byte arena, the same shape `impl/ranks.mojo`
    uses, because the keys are RAW BYTE SEQUENCES and may hold NUL or a lone
    continuation byte. Deduplication is a SPEED decision only: the merge loop
    sums integer counts over groups and rewrites each group independently, so
    the result does not depend on how the pre-tokens were grouped, nor on the
    order the groups are held in. That is why the corpus-order axis is
    identical by construction rather than by luck.
    """

    var arena: List[UInt8]
    var offset: List[Int]
    var length: List[Int]
    var count: List[Int]
    var buckets: List[Int]
    var mask: Int

    def __init__(out self):
        self.arena = List[UInt8]()
        self.offset = List[Int]()
        self.length = List[Int]()
        self.count = List[Int]()
        self.buckets = List[Int]()
        self.mask = 0
        self._grow(1024)

    def n(self) -> Int:
        return len(self.offset)

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

    def _grow(mut self, size: Int):
        self.buckets = List[Int]()
        for _ in range(size):
            self.buckets.append(0)
        self.mask = size - 1
        for id in range(len(self.offset)):
            var slot = Int(
                self._hash(self.arena, self.offset[id], self.length[id])
                & UInt64(self.mask)
            )
            while self.buckets[slot] != 0:
                slot = (slot + 1) & self.mask
            self.buckets[slot] = id + 1

    def add(mut self, data: List[UInt8], start: Int, end: Int):
        """Count one occurrence of `data[start:end]`."""
        var count = end - start
        if count <= 0:
            return
        var slot = Int(self._hash(data, start, count) & UInt64(self.mask))
        while self.buckets[slot] != 0:
            var id = self.buckets[slot] - 1
            if self._equals(id, data, start, count):
                self.count[id] += 1
                return
            slot = (slot + 1) & self.mask
        var id = len(self.offset)
        self.offset.append(len(self.arena))
        self.length.append(count)
        self.count.append(1)
        for i in range(start, end):
            self.arena.append(data[i])
        self.buckets[slot] = id + 1
        if 2 * len(self.offset) > len(self.buckets):
            self._grow(len(self.buckets) * 2)


struct PairCounts(Movable):
    """Adjacent-pair counts for ONE pass of the merge loop, keyed by
    `left * V + right`, in an open-addressing table with linear probing.

    THE MEMORY BOUND. Let N0 be the number of adjacent positions in the
    initial pre-token groups (the sum over DISTINCT groups of `len - 1`,
    at most the unique pre-token bytes). A pass sees P distinct pairs with
    P <= the adjacent positions of that pass <= N0 (merges only shorten a
    group), and P <= V * V. The table holds a power of two of slots, at most
    4 * max(P, 256) because it doubles only when more than half full, each
    slot two Ints, plus one Int per occupied slot in `touched`. Peak:
    8 * (2 * 4P + P) = 72 * P bytes <= 72 * N0 bytes, and it never shrinks.
    The dense table it replaced was 8 * V * V bytes whatever the corpus
    (20.2 GB at V = 50,256).

    ORDER. `touched` lists occupied slots in first-seen order. Nothing reads
    that order into the result: the selection is a total order over
    `(count, key)`, so any walk of the same set finds the same winner.
    """

    var keys: List[Int]
    var vals: List[Int]
    var touched: List[Int]
    var mask: Int

    def __init__(out self):
        self.keys = List[Int]()
        self.vals = List[Int]()
        self.touched = List[Int]()
        self.mask = 0
        self._alloc(1024)

    def _alloc(mut self, size: Int):
        self.keys = List[Int](length=size, fill=-1)
        self.vals = List[Int](length=size, fill=0)
        self.mask = size - 1

    @always_inline
    def _slot(self, key: Int) -> Int:
        # Fibonacci hashing: spreads the dense low keys (pairs of small ids)
        # across the table. The hash reaches only WHERE a count sits.
        var h = UInt64(key) * 11400714819323198485
        return Int((h >> 32) ^ h) & self.mask

    def n(self) -> Int:
        return len(self.touched)

    def clear(mut self):
        """Empty the table, touching only the occupied slots."""
        for t in range(len(self.touched)):
            var s = self.touched[t]
            self.keys[s] = -1
            self.vals[s] = 0
        self.touched.clear()

    def _grow(mut self):
        var size = (self.mask + 1) * 2
        var old_keys = self.keys^
        var old_vals = self.vals^
        var old_touched = self.touched^
        self.keys = List[Int](length=size, fill=-1)
        self.vals = List[Int](length=size, fill=0)
        self.mask = size - 1
        self.touched = List[Int](capacity=len(old_touched))
        for t in range(len(old_touched)):
            var s = old_touched[t]
            var key = old_keys[s]
            var slot = self._slot(key)
            while self.keys[slot] != -1:
                slot = (slot + 1) & self.mask
            self.keys[slot] = key
            self.vals[slot] = old_vals[s]
            self.touched.append(slot)

    @always_inline
    def add(mut self, key: Int, c: Int):
        var slot = self._slot(key)
        while True:
            var k = self.keys[slot]
            if k == key:
                self.vals[slot] += c
                return
            if k == -1:
                break
            slot = (slot + 1) & self.mask
        self.keys[slot] = key
        self.vals[slot] = c
        self.touched.append(slot)
        if 2 * len(self.touched) > self.mask + 1:
            self._grow()


struct TrainedVocabulary(Copyable, Movable):
    """The trained table: token bytes in rank order (rank = id), the merges
    in the order they were made, and the counters that let a caller see the
    tie rule was REACHED."""

    var arena: List[UInt8]
    var offset: List[Int]
    var length: List[Int]
    var merge_left: List[Int]
    var merge_right: List[Int]
    var n_ties_broken: Int
    var n_groups: Int

    def __init__(out self):
        self.arena = List[UInt8]()
        self.offset = List[Int]()
        self.length = List[Int]()
        self.merge_left = List[Int]()
        self.merge_right = List[Int]()
        self.n_ties_broken = 0
        self.n_groups = 0

    def n_tokens(self) -> Int:
        return len(self.offset)

    def n_merges(self) -> Int:
        return len(self.merge_left)

    def token_bytes(self, id: Int) raises -> List[UInt8]:
        if id < 0 or id >= len(self.offset):
            raise Error("trained vocabulary: id out of range: " + String(id))
        var out = List[UInt8]()
        var at = self.offset[id]
        for k in range(self.length[id]):
            out.append(self.arena[at + k])
        return out^


def train_bpe(
    documents: List[List[UInt8]],
    classes: UnicodeClasses,
    vocab_size: Int,
    min_frequency: Int,
    break_ties_high: Bool = False,
) raises -> TrainedVocabulary:
    """Train a byte-level BPE vocabulary.

    `documents` is the corpus. Each is pre-tokenized ALONE, so no pre-token
    spans a document join and the order the documents arrive in cannot reach
    the result. The vocabulary starts as the 256 single bytes (id = byte
    value) and grows to `vocab_size`, or stops early when no pair reaches
    `min_frequency`.

    `break_ties_high=True` is the RUNTIME spelling of the sabotage define:
    it reverses only the tie-break. It exists so the Python door's
    MOJOLEARN_BPE_TRAINER_SABOTAGE environment arm reaches this code through
    the host binding (lane/bpe-builder-native); nothing else passes it.
    """
    if vocab_size < 256:
        raise Error(
            "train_bpe: vocab_size "
            + String(vocab_size)
            + " is below the 256 single-byte tokens"
        )
    if min_frequency < 1:
        raise Error(
            "train_bpe: min_frequency "
            + String(min_frequency)
            + " must be at least 1"
        )

    # 1.  The corpus as pre-token groups.
    var groups = PieceGroups()
    for d in range(len(documents)):
        var bounds = pretokenize(documents[d], classes)
        for k in range(len(bounds) - 1):
            groups.add(documents[d], bounds[k], bounds[k + 1])

    # 2.  Each group as a sequence of ids, initially its raw bytes.
    var seqs = List[List[Int]]()
    for g in range(groups.n()):
        var ids = List[Int]()
        var at = groups.offset[g]
        for i in range(groups.length[g]):
            ids.append(Int(groups.arena[at + i]))
        seqs.append(ids^)

    # 3.  The 256 single bytes.
    var vocab = TrainedVocabulary()
    vocab.n_groups = groups.n()
    for b in range(256):
        vocab.offset.append(len(vocab.arena))
        vocab.length.append(1)
        vocab.arena.append(UInt8(b))

    # 4.  The merge loop. A pair is the single integer key
    #     `left * V + right`, so a key's ORDER is the pair's order and the
    #     tie-break needs no separate comparison. The counts live in
    #     `PairCounts`, an open-addressing table sized by the DISTINCT PAIRS
    #     THAT OCCUR in one pass (lane/bpe-builder-native, 2026-09-18); it
    #     replaced a dense `V * V` table that was 20.2 GB at V = 50,256.
    #     Where a count is stored cannot reach the result: the selection
    #     below reads only `(count, key)` over the set of keys seen.
    var V = vocab_size
    var counts = PairCounts()
    var reverse = break_ties_high
    comptime if BPE_TRAINER_SABOTAGE:
        reverse = True

    while vocab.n_tokens() < vocab_size:
        counts.clear()

        for g in range(len(seqs)):
            var c = groups.count[g]
            var n = len(seqs[g])
            for k in range(n - 1):
                counts.add(seqs[g][k] * V + seqs[g][k + 1], c)

        # THE SELECTION. Highest count, then the smallest key -- which is the
        # smallest `(left_id, right_id)`. Every comparison here is between
        # integers.
        var best_key = -1
        var best_count = 0
        for t in range(counts.n()):
            var key = counts.keys[counts.touched[t]]
            var c = counts.vals[counts.touched[t]]
            if c < min_frequency:
                continue
            if best_key < 0 or c > best_count:
                best_key = key
                best_count = c
            elif c == best_count:
                if reverse:
                    # THE SABOTAGE: the opposite end of the same total order.
                    if key > best_key:
                        best_key = key
                else:
                    if key < best_key:
                        best_key = key
        if best_key < 0:
            break

        var n_at_top = 0
        for t in range(counts.n()):
            if counts.vals[counts.touched[t]] == best_count:
                n_at_top += 1
        if n_at_top > 1:
            vocab.n_ties_broken += 1

        var a = best_key // V
        var b = best_key % V

        # The merged token's bytes are built into a temporary first: the
        # arena is about to be appended to, and reading it while growing it
        # is how a subtle corruption gets written.
        var joined = List[UInt8]()
        for i in range(vocab.length[a]):
            joined.append(vocab.arena[vocab.offset[a] + i])
        for i in range(vocab.length[b]):
            joined.append(vocab.arena[vocab.offset[b] + i])
        var new = vocab.n_tokens()
        vocab.offset.append(len(vocab.arena))
        vocab.length.append(len(joined))
        for i in range(len(joined)):
            vocab.arena.append(joined[i])
        vocab.merge_left.append(a)
        vocab.merge_right.append(b)

        # Rewrite every sequence LEFT TO RIGHT, NON-OVERLAPPING, in place:
        # the write index never passes the read index, so reading `k` and
        # writing `w <= k` in one buffer is the same rewrite as building a
        # fresh list, without an allocation per group per merge.
        for g in range(len(seqs)):
            var n = len(seqs[g])
            var w = 0
            var k = 0
            while k < n:
                if k + 1 < n and seqs[g][k] == a and seqs[g][k + 1] == b:
                    seqs[g][w] = new
                    k += 2
                else:
                    seqs[g][w] = seqs[g][k]
                    k += 1
                w += 1
            if w < n:
                seqs[g].shrink(w)

    return vocab^
