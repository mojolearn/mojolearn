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
3.  NO ITERATION ORDER REACHES THE RESULT. The winner comes off a heap
    ordered by the same total order over `(count, key)`, so which pair wins
    does not depend on the order pairs were first seen, the order groups are
    held in, or where a count sits in the table. The Python reference
    recounts every pair each merge and walks them in sorted order instead,
    and the gate proves the two agree.
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

COST. The pair counts are kept ACROSS merges and only the groups that hold
the merged pair are touched (2026-09-21). Before that every merge recounted
every group, O(merges x corpus pre-tokens): 731 s on an M4 for a 96 MB
sample. A count is a sum of integers over groups, so subtracting a touched
group's pairs and adding them back after the rewrite leaves exactly the
table a full recount would build; the vocabulary files are the same bytes
and `tools/bpe_trainer_determinism.py` and the gate hold that.
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


struct PairTable(Movable):
    """Adjacent-pair counts kept ACROSS the merge loop, keyed by
    `left * V + right`. A pair gets a dense entry id the first time it is
    seen and keeps it; an open-addressing table with linear probing maps the
    key to the id.

    `where[id]` lists the groups that have held the pair. It is only ever
    appended to, so it may name a group that no longer holds the pair, or
    name one twice: the merge loop scans the group and skips it when the
    pair is not there. It never MISSES a holder, and the loop checks that by
    requiring the merged pair's count to reach exactly zero.

    `pushed[id]` is the count last put on the heap for the pair and
    `stamp[id]` the last merge that changed it, so a merge pushes each
    changed pair once, with its final count.

    ORDER. Nothing here reaches the result: the heap orders `(count, key)`
    totally, and the counts are integer sums that do not depend on the order
    they were added in.
    """

    var keys: List[Int]
    var vals: List[Int]
    var pushed: List[Int]
    var stamp: List[Int]
    var where: List[List[Int]]
    var buckets: List[Int]
    var mask: Int

    def __init__(out self):
        self.keys = List[Int]()
        self.vals = List[Int]()
        self.pushed = List[Int]()
        self.stamp = List[Int]()
        self.where = List[List[Int]]()
        self.buckets = List[Int](length=1024, fill=0)
        self.mask = 1023

    @always_inline
    def _slot(self, key: Int) -> Int:
        # Fibonacci hashing: spreads the dense low keys (pairs of small ids)
        # across the table. The hash reaches only WHERE an id sits.
        var h = UInt64(key) * 11400714819323198485
        return Int((h >> 32) ^ h) & self.mask

    def n(self) -> Int:
        return len(self.keys)

    def _grow(mut self):
        var size = (self.mask + 1) * 2
        self.buckets = List[Int](length=size, fill=0)
        self.mask = size - 1
        for id in range(len(self.keys)):
            var slot = self._slot(self.keys[id])
            while self.buckets[slot] != 0:
                slot = (slot + 1) & self.mask
            self.buckets[slot] = id + 1

    @always_inline
    def find(self, key: Int) -> Int:
        """The pair's entry id, or -1."""
        var slot = self._slot(key)
        while self.buckets[slot] != 0:
            var id = self.buckets[slot] - 1
            if self.keys[id] == key:
                return id
            slot = (slot + 1) & self.mask
        return -1

    @always_inline
    def find_or_add(mut self, key: Int) -> Int:
        var slot = self._slot(key)
        while self.buckets[slot] != 0:
            var id = self.buckets[slot] - 1
            if self.keys[id] == key:
                return id
            slot = (slot + 1) & self.mask
        var id = len(self.keys)
        self.keys.append(key)
        self.vals.append(0)
        self.pushed.append(0)
        self.stamp.append(-1)
        self.where.append(List[Int]())
        self.buckets[slot] = id + 1
        if 2 * len(self.keys) > self.mask + 1:
            self._grow()
        return id

    @always_inline
    def note(mut self, id: Int, g: Int):
        """Record that group `g` holds the pair. Groups arrive in runs, so
        comparing with the last entry drops most repeats; the rest are
        harmless."""
        var n = len(self.where[id])
        if n == 0 or self.where[id][n - 1] != g:
            self.where[id].append(g)


struct PairHeap(Movable):
    """A binary heap of `(count, key, entry id)` whose top is the pair the
    selection rule picks: the highest count, then the smallest key (the
    LARGEST key under the sabotage). Entries are never updated in place. A
    pair whose count moved gets a new entry, and an entry whose count is no
    longer the pair's count is dropped when it reaches the top."""

    var count: List[Int]
    var key: List[Int]
    var id: List[Int]
    var reverse: Bool

    def __init__(out self, reverse: Bool):
        self.count = List[Int]()
        self.key = List[Int]()
        self.id = List[Int]()
        self.reverse = reverse

    def n(self) -> Int:
        return len(self.count)

    @always_inline
    def _before(self, i: Int, j: Int) -> Bool:
        if self.count[i] != self.count[j]:
            return self.count[i] > self.count[j]
        if self.reverse:
            # THE SABOTAGE: the opposite end of the same total order.
            return self.key[i] > self.key[j]
        return self.key[i] < self.key[j]

    @always_inline
    def _swap(mut self, i: Int, j: Int):
        var c = self.count[i]
        var k = self.key[i]
        var d = self.id[i]
        self.count[i] = self.count[j]
        self.key[i] = self.key[j]
        self.id[i] = self.id[j]
        self.count[j] = c
        self.key[j] = k
        self.id[j] = d

    def push(mut self, count: Int, key: Int, id: Int):
        self.count.append(count)
        self.key.append(key)
        self.id.append(id)
        var i = len(self.count) - 1
        while i > 0:
            var parent = (i - 1) // 2
            if not self._before(i, parent):
                break
            self._swap(i, parent)
            i = parent

    def pop(mut self):
        var last = len(self.count) - 1
        self._swap(0, last)
        _ = self.count.pop()
        _ = self.key.pop()
        _ = self.id.pop()
        var i = 0
        while True:
            var l = 2 * i + 1
            var r = l + 1
            var top = i
            if l < last and self._before(l, top):
                top = l
            if r < last and self._before(r, top):
                top = r
            if top == i:
                break
            self._swap(i, top)
            i = top


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
    #     tie-break needs no separate comparison. The counts are built ONCE
    #     and then maintained: a merge touches only the groups that hold the
    #     merged pair (see `PairTable`, and COST in the module docstring).
    var V = vocab_size
    var reverse = break_ties_high
    comptime if BPE_TRAINER_SABOTAGE:
        reverse = True
    var table = PairTable()
    var heap = PairHeap(reverse)

    for g in range(len(seqs)):
        var c = groups.count[g]
        for k in range(len(seqs[g]) - 1):
            var id = table.find_or_add(seqs[g][k] * V + seqs[g][k + 1])
            table.vals[id] += c
            table.note(id, g)
    for id in range(table.n()):
        table.pushed[id] = table.vals[id]
        heap.push(table.vals[id], table.keys[id], id)

    var changed = List[Int]()
    var merge_no = 0
    while vocab.n_tokens() < vocab_size:
        # THE SELECTION. Highest count, then the smallest key -- which is the
        # smallest `(left_id, right_id)`. Every comparison is between
        # integers. An entry whose count is no longer its pair's is stale.
        while heap.n() > 0 and heap.count[0] != table.vals[heap.id[0]]:
            heap.pop()
        if heap.n() == 0:
            break
        var best_id = heap.id[0]
        var best_key = heap.key[0]
        var best_count = heap.count[0]
        if best_count < min_frequency:
            break
        heap.pop()

        # A tie is another pair holding the same count.
        while heap.n() > 0 and (
            heap.count[0] != table.vals[heap.id[0]] or heap.id[0] == best_id
        ):
            heap.pop()
        if heap.n() > 0 and heap.count[0] == best_count:
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

        # Only the groups that hold the pair. `holders` is a copy because
        # the loop below appends to `table.where`.
        var holders = table.where[best_id].copy()
        table.where[best_id].clear()
        for h in range(len(holders)):
            var g = holders[h]
            var n = len(seqs[g])
            var found = False
            for k in range(n - 1):
                if seqs[g][k] == a and seqs[g][k + 1] == b:
                    found = True
                    break
            if not found:
                continue
            var c = groups.count[g]

            # Take this group's pairs out of the counts ...
            for k in range(n - 1):
                var id = table.find(seqs[g][k] * V + seqs[g][k + 1])
                if id < 0:
                    raise Error("train_bpe: a counted pair has no entry")
                table.vals[id] -= c
                if table.stamp[id] != merge_no:
                    table.stamp[id] = merge_no
                    changed.append(id)

            # ... rewrite it LEFT TO RIGHT, NON-OVERLAPPING, in place: the
            # write index never passes the read index, so reading `k` and
            # writing `w <= k` in one buffer is the same rewrite as building
            # a fresh list ...
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
            seqs[g].shrink(w)

            # ... and put its pairs back. A pair the group did not hold
            # before has the new token on one side.
            for k in range(w - 1):
                var left = seqs[g][k]
                var right = seqs[g][k + 1]
                var id = table.find_or_add(left * V + right)
                table.vals[id] += c
                if table.stamp[id] != merge_no:
                    table.stamp[id] = merge_no
                    changed.append(id)
                if left == new or right == new:
                    table.note(id, g)

        # `where` must have named EVERY holder: the pair is gone.
        if table.vals[best_id] != 0:
            raise Error(
                "train_bpe: the merged pair is still counted "
                + String(table.vals[best_id])
                + " times after its merge"
            )

        for t in range(len(changed)):
            var id = changed[t]
            if table.vals[id] != table.pushed[id]:
                table.pushed[id] = table.vals[id]
                if table.vals[id] > 0:
                    heap.push(table.vals[id], table.keys[id], id)
        changed.clear()
        merge_no += 1

    return vocab^
