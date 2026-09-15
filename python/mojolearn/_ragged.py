# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Ragged (right-padded) sequence batches for the causal sequence models.

PRIVATE MODULE. The `lengths=` argument of `TransformerBlock.forward`,
`Mamba1Block.forward`, `Mamba2Block.forward`, `Mamba3Block.forward`,
`SambaStack.forward`, and the byte LM `logits` and `next_bytes` of
`SmallByteLanguageModelTrainer` and `LanguageModelInference` (2026-09-15).

WHAT A RAGGED BATCH IS HERE. `B` sequences padded on the RIGHT to one
length `L`; `lengths[i]` in `[1, L]` is how many leading positions of row
`i` are real. Positions `lengths[i]` and after are padding. The contract
of the argument is two sentences:

1. A real position's output is BYTE FOR BYTE the output of the same
   sequence run alone at its own length (`forward(x[i:i+1, :lengths[i]])`).
2. A padding position's output is exactly `+0.0` (`next_bytes` reads the
   last REAL position instead), whatever the caller put at the padding
   positions of the input.

Sentence 2 holds in every numeric mode. Sentence 1 is a claim of the
IDENTICAL tier, the tier whose contracts make a row independent of its batch
and its length; under FAST and DETERMINISTIC the two copies are the same and
the equality is not promised, as for any batch.

NO ARITHMETIC CHANGES, AND THIS IS WHY IT NEEDS NONE. Every model reached
here is CAUSAL: a position reads only itself and earlier positions. Right
padding therefore never enters any reduction that reaches a real token,
and sentence 1 is exactly the sequence-length invariance the block
contracts already state and gate: transformer contract 7.3 ("a row's bits
must be identical whether the sequence it belongs to has length 4 or
257", the masked tail exactly `+0.0` by 7.1) and batch composition 7.4;
the Mamba-1/2/3 contracts' clause (c) and their chunked scans' structural
zeros; and the loss-free logits path of the byte LM and the Samba stack,
which is embedding, those blocks, a per-token norm and a per-token head.
The one hole the transformer contract records against 7.1 (DEVIATION
1327, a `-0.0` value-sum accumulator reached through `ftz` of a negative
subnormal partial sum) is the same hole for a prefix of a longer call as
for padding, and the harness's prefix and ragged checks see it alike.

So the implementation is two COPIES and no kernel: the padding positions of
the input are replaced by `+0.0` (token id 0) before the ordinary batched
call, which keeps a caller's non-finite or out-of-vocabulary padding from
reaching the per-call refusals, and the padding positions of the output
are overwritten with `+0.0` after it. Numpy-free: both are flat byte copies
through `_buffer.memory_at`.

THE OUTPUT COPY IS INERT ON THE MAMBA BLOCKS, MEASURED. A Mamba-1, -2 or -3
block maps an all-`+0.0` token to exactly `+0.0` (the RMSNorm of zero is
zero and every path to the output is multiplied by it), so on those blocks
the padding outputs are already `+0.0` after the input copy and a
sabotage that skips the output copy passes their tests (M4, 2026-09-15).
The input copy is not inert there: skipping it hands the NaN padding to the
per-call refusal. On the transformer, the byte LM and Samba both copies
bite (test_ragged_lengths.py, both sabotages run).

WHAT IS REFUSED, BY NAME. `lengths` together with a carried state (a
decode continuation of a ragged batch would need per-row positions, which
no state here has), a length of 0 (a row with no real position has no
output a caller could want), a length above `L`, a count other than `B`,
and any value that is not an integer. BACKWARD WITH `lengths` IS NOT
OFFERED: a padded row's weight gradient contracts over all `B*L` tokens,
a different token count from the sequences alone, and clause 9.2 of the
optimizer contract says that count is part of the numerical specification;
a caller who wants per-sequence gradients runs the sequences as their own
calls and accumulates.
"""

from ._buffer import addr, addr_ro, empty, memory_at

__all__ = []


def lengths_for(lengths, batch, length, what):
    """`lengths` as a tuple of `batch` Python ints, each in [1, length],
    refused by name otherwise."""
    try:
        values = list(lengths)
    except TypeError:
        raise TypeError(f"mojolearn {what}: lengths must be a sequence of {batch} integers") from None
    if len(values) != batch:
        raise ValueError(f"mojolearn {what}: lengths has {len(values)} entries but the batch has B = {batch}")
    out = []
    for i, v in enumerate(values):
        if isinstance(v, bool) or type(v).__name__ in ("bool", "bool_"):
            raise TypeError(f"mojolearn {what}: lengths[{i}] is a bool, not an integer")
        try:
            n = v.__index__()
        except AttributeError:
            raise TypeError(f"mojolearn {what}: lengths[{i}] = {v!r} is not an integer") from None
        if n < 1 or n > length:
            raise ValueError(f"mojolearn {what}: lengths[{i}] = {n} is outside [1, L = {length}] "
                             "(right padding: a row has at least one real position and at most L)")
        out.append(int(n))
    return tuple(out)


def _row_bytes(arr):
    """(bytes per row, bytes per position) of a C-contiguous (B, L, ...) array."""
    b, l = int(arr.shape[0]), int(arr.shape[1])
    per_pos = arr.nbytes // (b * l) if b * l else 0
    return per_pos * l, per_pos


def padded_copy(src, lengths, dtype, what):
    """A new C-order Array with the bytes of `src` (B, L, ...) at every real
    position and zero bytes at every padding position."""
    shape = tuple(int(s) for s in src.shape)
    out = empty(shape, dtype)
    row, pos = _row_bytes(out)
    if row == 0:
        return out
    total = out.nbytes
    s = memory_at(addr_ro(src, name=what), total, writable=False)
    d = memory_at(addr(out, name=what), total, writable=True)
    for i, n in enumerate(lengths):
        base = i * row
        d[base:base + n * pos] = s[base:base + n * pos]
    return out


def zero_padding(out, lengths, what):
    """Overwrite every padding position of `out` (B, L, ...) with zero bytes,
    in place."""
    row, pos = _row_bytes(out)
    if row == 0:
        return out
    d = memory_at(addr(out, name=what), out.nbytes, writable=True)
    for i, n in enumerate(lengths):
        if n * pos < row:
            d[i * row + n * pos:(i + 1) * row] = bytes(row - n * pos)
    return out


def ragged_forward(run, x, state, lengths, dtype, what):
    """The one spelling every surface uses: refuse a carried state, check
    `lengths` against x's (B, L), zero the padding positions of a copy of
    x, run the ordinary batched call `run(copy)` and zero the padding
    positions of its (B, L, ...) output. `x` is already dtype- and
    shape-checked by the caller and C-contiguous."""
    if state is not None:
        raise ValueError(
            f"mojolearn {what}: lengths= (a ragged, right-padded batch) cannot carry a state; a decode "
            "continuation of a ragged batch would need a position per row, which no state here has. "
            "Run the ragged prefill with state=None, or continue each sequence as its own batch")
    b, l = int(x.shape[0]), int(x.shape[1])
    lens = lengths_for(lengths, b, l, what)
    return zero_padding(run(padded_copy(x, lens, dtype, what)), lens, what), lens


def last_real_rows(logits, lengths, what):
    """`(B, 1, V)` float32: row i's logits at position lengths[i] - 1, a
    copy, so a greedy pick reads each sequence's last REAL position."""
    b, l = int(logits.shape[0]), int(logits.shape[1])
    v = int(logits.shape[2])
    out = empty((b, 1, v), "<f4")
    s = memory_at(addr_ro(logits, name=what), logits.nbytes, writable=False)
    d = memory_at(addr(out, name=what), out.nbytes, writable=True)
    for i, n in enumerate(lengths):
        at = (i * l + n - 1) * v * 4
        d[i * v * 4:(i + 1) * v * 4] = s[at:at + v * 4]
    return out
