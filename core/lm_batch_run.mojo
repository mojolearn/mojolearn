# SPDX-License-Identifier: Apache-2.0
"""RUN of the token batch operations (`TokenBatches.ids`, its prefetch
iterator, the parallel byte step's shard gather and the tokenizer's
`tokens_gather_rows`): typed words only, no Python.

The door (`bindings/lm_corpus_public_native_door.mojo`) reads the batches
owner once into an `LmBatchRequest` at UNPACK (four words and the address of
the mapped token ids) and wraps the written int32 Array at PACK. The gather
itself is `tokenizer/impl/tokenizer_host_shell_impl.mojo`
`tokens_gather_rows_into`, unchanged.

`lm_batch_steps_run` walks the logical shards of ONE training step in order
(a handful, each one row copy per batch row). It is not split across cores:
a shard is a sub-millisecond copy bounded by memory bandwidth, and the
refusal of a bad step is defined by shard order.
"""
from tokenizer.impl.tokenizer_host_shell_impl import tokens_gather_rows_width, tokens_gather_rows_into

comptime _WORD_MAX = 9223372036854775807


struct LmBatchRequest(Copyable, Movable):
    """The admitted words of a batches owner: rows per batch, tokens per row
    (a row holds length + 1 ids), the train range start, the schedule
    modulus, and the mapped int32 ids (address, count)."""

    var batch: Int
    var length: Int
    var lo: Int
    var modulus: Int
    var source: Int
    var source_count: Int

    def __init__(out self, batch: Int, length: Int, lo: Int, modulus: Int):
        self.batch = batch
        self.length = length
        self.lo = lo
        self.modulus = modulus
        self.source = 0
        self.source_count = 0


def lm_batch_dims(request: LmBatchRequest) raises:
    if request.batch < 0 or request.length < 1:
        raise Error("ValueError: tokens_gather_rows: bad dims")


def lm_gather_rows(source_address: Int, source_count: Int, target_address: Int, target_count: Int,
                   batch: Int, length: Int, lo: Int, modulus: Int, base: Int) raises -> Int:
    """Row b of the target is ids[lo + (base + b*length) % modulus : + length + 1].
    The caller owns admission and keeps both stores alive; returns the cells
    written."""
    if (source_count < 0 or target_count < 0 or batch < 0 or length < 1 or length == _WORD_MAX or lo < 0
            or modulus < 1 or base < 0):
        raise Error("ValueError: tokens_gather_rows: bad dims")
    var requested_width = length + 1
    if (lo > source_count or modulus - 1 > source_count - lo
            or requested_width > source_count - lo - (modulus - 1)):
        raise Error("ValueError: tokens_gather_rows: a row runs past the id array")
    if batch > 0 and (batch - 1) > (_WORD_MAX - base) // length:
        raise Error("ValueError: token row position exceeds native indexing")
    var width = tokens_gather_rows_width(source_count, batch, length, lo, modulus, base)
    if batch > 0 and width > _WORD_MAX // batch:
        raise Error("ValueError: token row output size overflows native indexing")
    if target_count != batch * width:
        raise Error("ValueError: token row output size differs from batch times row width")
    if batch == 0:
        return 0
    if source_address == 0 or target_address == 0:
        raise Error("ValueError: token row gathering requires non-null retained storage")
    var src = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=source_address)
    var dst = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=target_address)
    return tokens_gather_rows_into(src, dst, batch, length, width, lo, modulus, base)


def lm_batch_run(request: LmBatchRequest, base: Int, target: Int, target_count: Int) raises -> Int:
    """One batch at row position `base` (step * batch * length) into `target`."""
    return lm_gather_rows(request.source, request.source_count, target, target_count,
                          request.batch, request.length, request.lo, request.modulus, base)


def lm_batch_steps_run(request: LmBatchRequest, steps: List[Int], target: Int) raises:
    """`ids(steps[k])` for every k, into len(steps) * batch * (length + 1)
    int32 cells at `target`: the same rows, in the same order, as one
    `lm_batch_run` per step."""
    var cells = request.batch * (request.length + 1)
    var row = request.batch * request.length
    for k in range(len(steps)):  # driver-loop: over the step's logical shards, one batch each
        var step = steps[k]
        if step < 0:
            raise Error("ValueError: tokens_gather_rows: bad dims")
        if row > 0 and step > _WORD_MAX // row:
            raise Error("ValueError: token row position exceeds native indexing")
        _ = lm_batch_run(request, step * row, target + k * cells * 4, cells)
