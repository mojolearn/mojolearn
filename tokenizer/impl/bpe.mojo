# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The byte-pair merge loop, one pre-token at a time.

THE RULE, and there is only one: while some adjacent pair of pieces has a
rank in the table, merge the pair with the LOWEST rank. Rank order is merge
order, which is what makes the result independent of where you start
scanning. Ranks are unique per token in the table (`load_rank_table` refuses
a duplicate key), so "the lowest" is never ambiguous and no tie rule is
needed -- the `<` below is deliberate, and a `<=` would still be
deterministic but would be a different algorithm on a table that did have
ties.

A piece is always a CONTIGUOUS slice of the pre-token, because every merge
joins neighbours, so the state is just a list of boundary offsets into the
caller's byte buffer: merging is deleting one boundary and a rank probe is
`ranks.rank(data, bounds[k], bounds[k + 2] - bounds[k])`. Nothing is copied
and nothing is allocated per probe.

COST. The loop is O(pieces^2) probes: a scan for the minimum per merge. A
pre-token is a word, not a document -- the pattern cannot produce an
unbounded letter run in practice, and the longest pre-token in the fixture
is 20 bytes -- so the quadratic form is measured in tens of probes and buys
a loop with no incremental-rank bookkeeping to get wrong. The linked-list
form tiktoken uses is a speed decision and this module makes no speed claim.
"""

from tokenizer.impl.ranks import RankTable


def bpe_append(
    mut out: List[Int],
    ranks: RankTable,
    data: List[UInt8],
    start: Int,
    end: Int,
) raises:
    """Append the ids of the pre-token `data[start:end]` to `out`."""
    var count = end - start
    if count <= 0:
        return

    # The whole pre-token is usually already a token; probing once for it
    # skips the loop entirely and is what tiktoken does first as well.
    var whole = ranks.rank(data, start, count)
    if whole >= 0:
        out.append(whole)
        return

    var bounds = List[Int]()
    for i in range(start, end + 1):
        bounds.append(i)

    while len(bounds) > 2:
        var best_rank = -1
        var best_at = -1
        for k in range(len(bounds) - 2):
            var r = ranks.rank(data, bounds[k], bounds[k + 2] - bounds[k])
            if r >= 0 and (best_at < 0 or r < best_rank):
                best_rank = r
                best_at = k
        if best_at < 0:
            break
        # Merge: the boundary between the two pieces disappears.
        var merged = List[Int]()
        for k in range(len(bounds)):
            if k != best_at + 1:
                merged.append(bounds[k])
        bounds = merged^

    for k in range(len(bounds) - 1):
        var id = ranks.rank(data, bounds[k], bounds[k + 1] - bounds[k])
        if id < 0:
            # Unreachable on the GPT-2 table: all 256 single bytes are
            # tokens, and the loop only stops when no pair merges, so every
            # surviving piece is either a single byte or a piece that was
            # merged through a rank and therefore IS a token. Raised rather
            # than dropped because a silent gap would corrupt the id stream.
            raise Error(
                "bpe: piece at offset "
                + String(bounds[k])
                + " of length "
                + String(bounds[k + 1] - bounds[k])
                + " has no rank; the table is missing a single-byte token"
            )
        out.append(id)
