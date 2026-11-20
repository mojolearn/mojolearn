# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""RUN of `providers/python/sequence_seq_tensor_pyapi.mojo` (wave 8 run
split): output-slot allocation and release of one device layer call, typed
Mojo only. Imports no Python (tools/python_free_dirs.sh); bodies moved
unchanged (lane rs-sequence-tokenizer, 2026-11-12)."""
from sequence.seq_tensor import seq_tensor_alloc, seq_tensor_free


def seq_tensor_alloc_slots(mut handles: List[Int], sizes: List[Int]) raises -> List[Int]:
    """Allocate the output slots of one device layer call (lane
    ne3-c-blocks-embedding, rows N162 / N163): every slot `i` with
    `sizes[i] > 0` gets a new resident tensor of that many floats (contents
    unspecified, as `seq_tensor_alloc(n, 0)`), written into `handles[i]`.
    Returns the handles made, for `seq_tensor_free_made` when the layer call
    fails. One native step for every output instead of one Python crossing
    per output."""
    if len(sizes) != len(handles):
        raise Error("seq_tensor_alloc_slots: one size per handle slot is required")
    var made = List[Int]()
    try:
        for i in range(len(handles)):  # small-loop(output slots of one device layer call, at most six)
            if sizes[i] > 0:
                handles[i] = seq_tensor_alloc(sizes[i], False)
                made.append(handles[i])
    except error:
        seq_tensor_free_made(made)
        raise error^
    return made^


def seq_tensor_free_made(made: List[Int]):
    """Release the handles `seq_tensor_alloc_slots` made (a failed call)."""
    for i in range(len(made)):  # small-loop(handles made by one device layer call, at most six)
        try:
            _ = seq_tensor_free(made[i])
        except:
            pass
